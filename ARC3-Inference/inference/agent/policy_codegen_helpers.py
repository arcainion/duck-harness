"""Safe contract builders and transition classifiers for generated policies."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from operator import index
from types import MappingProxyType
from typing import Any, Iterable

import numpy as np

from inference.agent.policy_pathfinding import next_path_action


POLICY_CODEGEN_API_VERSION = 1
POLICY_ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "SPACE", "MOUSE", "ACTION7")
MAX_HELPER_MEMORY_BYTES = 32_768
MAX_HELPER_MEMORY_KEYS = 64
MAX_RECENT_TRANSITIONS = 64
_NO_PROGRESS_OUTCOMES = {
    "behavioral_noop",
    "exact_noop",
    "no_progress",
    "volatile_only",
}


def _point(value: Any) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        raise ValueError("point must be a (row, col) pair")
    try:
        if len(value) != 2:
            raise ValueError("point must be a (row, col) pair")
        row, col = value[0], value[1]
    except (TypeError, IndexError) as exc:
        raise ValueError("point must be a (row, col) pair") from exc
    if isinstance(row, bool) or isinstance(col, bool):
        raise ValueError("point coordinates must be integers")
    try:
        checked = index(row), index(col)
    except TypeError as exc:
        raise ValueError("point coordinates must be integers") from exc
    if not 0 <= checked[0] <= 63 or not 0 <= checked[1] <= 63:
        raise ValueError("point row and col must be between 0 and 63")
    return checked


def _grid(value: Any) -> np.ndarray:
    board = np.asarray(value)
    if board.ndim != 2 or not board.shape[0] or not board.shape[1]:
        raise ValueError("grid must be a non-empty two-dimensional array")
    if int(board.size) > 64 * 64 or int(board.nbytes) > 65_536:
        raise ValueError("grid exceeds the 4096-cell or 65536-byte limit")
    if board.dtype.hasobject:
        raise ValueError("grid may not contain object values")
    return board


def _matching_mask(board: np.ndarray, values: Any) -> np.ndarray:
    if np.isscalar(values):
        requested = (values,)
    else:
        if isinstance(values, (str, bytes)):
            requested = (values,)
        else:
            try:
                iterator = iter(values)
            except TypeError as exc:
                raise ValueError("values must be a scalar or iterable") from exc
            collected: list[Any] = []
            for position, value in enumerate(iterator):
                if position >= 256:
                    raise ValueError("values may contain at most 256 entries")
                collected.append(value)
            requested = tuple(collected)
    if not requested:
        return np.zeros(board.shape, dtype=bool)
    return np.isin(board, requested)


def _checked_grid_point(value: Any, shape: tuple[int, int]) -> tuple[int, int]:
    point = _point(value)
    if point[0] >= shape[0] or point[1] >= shape[1]:
        raise ValueError(f"point {point!r} is outside grid shape {shape!r}")
    return point


def _json_clone(value: Any, label: str) -> Any:
    try:
        encoded = json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain finite JSON data") from exc
    if len(encoded) > MAX_HELPER_MEMORY_BYTES:
        raise ValueError(
            f"{label} may encode to at most {MAX_HELPER_MEMORY_BYTES} bytes"
        )
    return json.loads(encoded.decode("utf-8"))


def _memory_mapping(memory: Any) -> dict[str, Any]:
    if memory is None:
        return {}
    if not isinstance(memory, Mapping):
        raise ValueError("memory must be a mapping or None")
    result = dict(memory)
    if len(result) > MAX_HELPER_MEMORY_KEYS:
        raise ValueError(f"memory may contain at most {MAX_HELPER_MEMORY_KEYS} keys")
    if any(not isinstance(key, str) or not key for key in result):
        raise ValueError("memory keys must be non-empty strings")
    return result


def _recent_items(transitions: Any) -> tuple[Mapping[str, Any], ...]:
    try:
        iterator = iter(() if transitions is None else transitions)
    except TypeError as exc:
        raise ValueError("recent transitions must be an iterable") from exc
    result: list[Mapping[str, Any]] = []
    for position, transition in enumerate(iterator):
        if position >= MAX_RECENT_TRANSITIONS:
            raise ValueError(
                f"recent transitions may contain at most {MAX_RECENT_TRANSITIONS} items"
            )
        if isinstance(transition, Mapping):
            result.append(transition)
    return tuple(result)


def _action_name(value: Any) -> str:
    action_name = str(value or "").strip().upper()
    if action_name not in POLICY_ACTIONS:
        raise ValueError(f"action must be one of {', '.join(POLICY_ACTIONS)}")
    return action_name


def action_payload(action: Any, point: Any | None = None) -> dict[str, Any]:
    """Build a canonical model-facing action mapping."""

    if isinstance(action, Mapping):
        raw = dict(action)
        action_name = str(raw.get("action") or "").strip().upper()
    else:
        raw = {}
        action_name = str(action or "").strip().upper()
    action_name = _action_name(action_name)
    coordinate_value = point
    if coordinate_value is None and (
        raw.get("row") is not None or raw.get("col") is not None
    ):
        coordinate_value = (raw.get("row"), raw.get("col"))
    if action_name == "MOUSE":
        if coordinate_value is None:
            raise ValueError("MOUSE requires a (row, col) point")
        row, col = _point(coordinate_value)
        return {"action": "MOUSE", "row": row, "col": col}
    if coordinate_value is not None:
        raise ValueError(f"{action_name} does not accept a coordinate point")
    return {"action": action_name}


def _prediction(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("prediction must be a mapping or None")
    return dict(value)


def continue_decision(
    action: Any,
    memory: Any,
    evidence: Any = "",
    prediction: Any | None = None,
    point: Any | None = None,
) -> dict[str, Any]:
    """Build a continue decision with exactly one canonical action."""

    return {
        "status": "continue",
        "action": action_payload(action, point),
        "memory": memory,
        "evidence": str(evidence or ""),
        "prediction": _prediction(prediction),
    }


def mouse_decision(
    point: Any,
    memory: Any,
    evidence: Any = "",
    prediction: Any | None = None,
) -> dict[str, Any]:
    """Build a continue decision containing a bounded MOUSE action."""

    return continue_decision("MOUSE", memory, evidence, prediction, point)


def subgoal_succeeded(memory: Any, evidence: Any = "") -> dict[str, Any]:
    """Build a terminal tactical-success decision."""

    return {
        "status": "subgoal_succeeded",
        "action": None,
        "memory": memory,
        "evidence": str(evidence or ""),
    }


def subgoal_failed(memory: Any, evidence: Any = "") -> dict[str, Any]:
    """Build a terminal tactical-failure decision."""

    return {
        "status": "subgoal_failed",
        "action": None,
        "memory": memory,
        "evidence": str(evidence or ""),
    }


def path_decision(
    path: Iterable[Any],
    valid_actions: Iterable[str],
    memory: Any,
    evidence: Any = "",
    prediction: Any | None = None,
) -> dict[str, Any]:
    """Continue with the first valid path action, or fail the tactical subgoal."""

    action_name = next_path_action(path, valid_actions)
    if action_name is None:
        detail = str(evidence or "path has no currently valid next action")
        return subgoal_failed(memory, detail)
    return continue_decision(action_name, memory, evidence, prediction)


def transition_outcome(transition: Any) -> str:
    """Classify a host transition as terminal/progress/guarded/failed/no_progress/unknown."""

    if not isinstance(transition, Mapping):
        return "unknown"
    if (
        bool(transition.get("level_completed"))
        or bool(transition.get("run_complete"))
        or bool(transition.get("game_over"))
    ):
        return "terminal"
    outcome = str(transition.get("outcome_class") or "").strip().lower()
    if transition.get("error") or transition.get("executed") is False:
        return "failed"
    if (
        outcome == "guarded"
        or bool(transition.get("cycle_risk"))
        or bool(transition.get("loop_detected"))
    ):
        return "guarded"
    if bool(transition.get("meaningful_progress")):
        return "progress"
    if outcome in _NO_PROGRESS_OUTCOMES:
        return "no_progress"
    if transition.get("meaningful_progress") is False or (
        transition.get("executed") is True and transition.get("board_changed") is False
    ):
        return "no_progress"
    return "unknown"


def transition_has_progress(transition: Any) -> bool:
    """Return whether the host classified the transition as meaningful progress."""

    if not isinstance(transition, Mapping):
        return False
    if bool(transition.get("game_over")) and not (
        bool(transition.get("level_completed")) or bool(transition.get("run_complete"))
    ):
        return False
    return transition_outcome(transition) in {"progress", "terminal"}


def transition_facts(transition: Any) -> dict[str, Any]:
    """Return a bounded JSON snapshot using the host's transition semantics."""

    if not isinstance(transition, Mapping):
        return {
            "outcome": "unknown",
            "action": "",
            "point": None,
            "executed": None,
            "post_action_observed": None,
            "board_changed": None,
            "meaningful_progress": False,
            "level_completed": False,
            "run_complete": False,
            "game_over": False,
            "error": "",
        }

    action = str(transition.get("action") or "").strip().upper()
    if action not in POLICY_ACTIONS:
        action = ""
    point: list[int] | None = None
    if action == "MOUSE" and "row" in transition and "col" in transition:
        try:
            point = list(_point((transition.get("row"), transition.get("col"))))
        except ValueError:
            point = None

    def optional_bool(key: str) -> bool | None:
        value = transition.get(key)
        return value if isinstance(value, bool) else None

    return {
        "outcome": transition_outcome(transition),
        "action": action,
        "point": point,
        "executed": optional_bool("executed"),
        "post_action_observed": optional_bool("post_action_observed"),
        "board_changed": optional_bool("board_changed"),
        "meaningful_progress": transition_has_progress(transition),
        "level_completed": bool(transition.get("level_completed")),
        "run_complete": bool(transition.get("run_complete")),
        "game_over": bool(transition.get("game_over")),
        "error": str(transition.get("error") or "")[:512],
    }


def objective_evidence_ready(objective: Any, transitions: Any) -> bool:
    """Return whether the objective has its requested post-action evidence count."""

    if not isinstance(objective, Mapping):
        raise ValueError("objective must be a mapping")
    required = objective.get("minimum_evidence_actions", 1)
    if isinstance(required, bool):
        raise ValueError("minimum_evidence_actions must be an integer from 0 through 32")
    try:
        required_count = index(required)
    except TypeError as exc:
        raise ValueError(
            "minimum_evidence_actions must be an integer from 0 through 32"
        ) from exc
    if not 0 <= required_count <= 32:
        raise ValueError("minimum_evidence_actions must be an integer from 0 through 32")
    if required_count == 0:
        return True
    objective_id = str(objective.get("objective_id") or objective.get("id") or "")
    evidence_count = 0
    for transition in _recent_items(transitions):
        transition_objective_id = str(transition.get("objective_id") or "")
        if objective_id and transition_objective_id and transition_objective_id != objective_id:
            continue
        if transition.get("executed") is not True:
            continue
        if transition.get("post_action_observed") is False:
            continue
        classified = transition.get("post_action_observed") is True or any(
            key in transition
            for key in ("outcome_class", "board_changed", "meaningful_progress")
        )
        if not classified:
            continue
        evidence_count += 1
        if evidence_count >= required_count:
            return True
    return False


def transition_requires_replan(
    transition: Any, replan_on_no_progress: bool = True
) -> bool:
    """Return whether a policy should stop reusing its current plan."""

    outcome = transition_outcome(transition)
    return outcome in {"failed", "guarded", "terminal"} or (
        bool(replan_on_no_progress) and outcome == "no_progress"
    )


def transition_repeats_nonprogress_action(
    transition: Any, action: Any, point: Any | None = None
) -> bool:
    """Detect an exact candidate repeat after a non-progress/guarded/failed action."""

    if not isinstance(transition, Mapping) or transition_outcome(transition) not in {
        "failed",
        "guarded",
        "no_progress",
    }:
        return False
    try:
        candidate = action_payload(action, point)
        previous = action_payload(
            {
                "action": transition.get("action"),
                "row": transition.get("row"),
                "col": transition.get("col"),
            }
        )
    except ValueError:
        return False
    return candidate == previous


def board_digest(grid: Any) -> str:
    """Return a compact deterministic digest without exposing or storing the board."""

    board = np.asarray(grid)
    if board.ndim != 2 or not board.shape[0] or not board.shape[1]:
        raise ValueError("grid must be a non-empty two-dimensional array")
    if int(board.size) > 64 * 64 or int(board.nbytes) > 65_536:
        raise ValueError("grid exceeds the 4096-cell or 65536-byte digest limit")
    if board.dtype.hasobject:
        raise ValueError("grid may not contain object values")
    canonical = np.ascontiguousarray(board)
    header = f"{canonical.shape!r}|{canonical.dtype.str}|".encode("ascii")
    digest = hashlib.blake2b(digest_size=8)
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def region_digest(grid: Any, bounds: Any) -> str:
    """Digest an inclusive ``(top, left, bottom, right)`` board rectangle."""

    board = _grid(grid)
    if isinstance(bounds, (str, bytes)):
        raise ValueError("bounds must be (top, left, bottom, right)")
    try:
        raw = tuple(bounds)
    except TypeError as exc:
        raise ValueError("bounds must be (top, left, bottom, right)") from exc
    if len(raw) != 4 or any(isinstance(value, bool) for value in raw):
        raise ValueError("bounds must be (top, left, bottom, right) integers")
    try:
        top, left, bottom, right = (index(value) for value in raw)
    except TypeError as exc:
        raise ValueError("bounds must be (top, left, bottom, right) integers") from exc
    if not (0 <= top <= bottom < board.shape[0]) or not (
        0 <= left <= right < board.shape[1]
    ):
        raise ValueError(f"bounds {raw!r} are outside grid shape {board.shape!r}")
    return board_digest(board[top : bottom + 1, left : right + 1])


def cells_digest(grid: Any, cells: Any) -> str:
    """Digest values at a bounded, order-independent set of board coordinates."""

    board = _grid(grid)
    try:
        iterator = iter(cells)
    except TypeError as exc:
        raise ValueError("cells must be an iterable of (row, col) pairs") from exc
    points: list[tuple[int, int]] = []
    for position, value in enumerate(iterator):
        if position >= 64 * 64:
            raise ValueError("cells may contain at most 4096 points")
        points.append(_checked_grid_point(value, board.shape))
    unique = tuple(sorted(set(points)))
    digest = hashlib.blake2b(digest_size=8)
    digest.update(f"{board.shape!r}|{board.dtype.str}|".encode("ascii"))
    if unique:
        digest.update(np.asarray(unique, dtype="<i2").tobytes(order="C"))
        values = np.ascontiguousarray(np.asarray([board[point] for point in unique]))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def first_matching_cell(grid: Any, values: Any) -> tuple[int, int] | None:
    """Return the first matching board coordinate in row-major order."""

    board = _grid(grid)
    matches = np.argwhere(_matching_mask(board, values))
    if not len(matches):
        return None
    return int(matches[0, 0]), int(matches[0, 1])


def nearest_matching_cell(
    grid: Any, values: Any, origin: Any
) -> tuple[int, int] | None:
    """Return the Manhattan-nearest matching cell with row-major tie breaking."""

    board = _grid(grid)
    start = _checked_grid_point(origin, board.shape)
    matches = np.argwhere(_matching_mask(board, values))
    if not len(matches):
        return None
    return min(
        ((int(row), int(col)) for row, col in matches),
        key=lambda point: (
            abs(point[0] - start[0]) + abs(point[1] - start[1]),
            point[0],
            point[1],
        ),
    )


def matching_region_center(grid: Any, values: Any) -> tuple[int, int] | None:
    """Return the matching cell nearest the matches' bounding-box center."""

    board = _grid(grid)
    matches = np.argwhere(_matching_mask(board, values))
    if not len(matches):
        return None
    top, left = np.min(matches, axis=0)
    bottom, right = np.max(matches, axis=0)
    center_row_twice = int(top) + int(bottom)
    center_col_twice = int(left) + int(right)
    return min(
        ((int(row), int(col)) for row, col in matches),
        key=lambda point: (
            abs(2 * point[0] - center_row_twice)
            + abs(2 * point[1] - center_col_twice),
            point[0],
            point[1],
        ),
    )


def _line(grid: Any, axis: Any, line_index: Any) -> np.ndarray:
    board = _grid(grid)
    normalized = str(axis).strip().lower()
    if normalized in {"0", "row", "rows"}:
        size = board.shape[0]
        row_axis = True
    elif normalized in {"1", "column", "columns", "col", "cols"}:
        size = board.shape[1]
        row_axis = False
    else:
        raise ValueError("axis must be row/0 or column/1")
    if isinstance(line_index, bool):
        raise ValueError("line index must be an integer")
    try:
        checked_index = index(line_index)
    except TypeError as exc:
        raise ValueError("line index must be an integer") from exc
    if not 0 <= checked_index < size:
        raise ValueError(f"line index must be between 0 and {size - 1}")
    return board[checked_index, :] if row_axis else board[:, checked_index]


def line_value_count(grid: Any, values: Any, axis: Any, line_index: Any) -> int:
    """Count matching cells in one row or column."""

    line = _line(grid, axis, line_index)
    return int(np.count_nonzero(_matching_mask(line.reshape(1, -1), values)))


def line_run_length(
    grid: Any,
    values: Any,
    axis: Any,
    line_index: Any,
    from_end: bool = False,
) -> int:
    """Count consecutive matching cells from either end of a row or column."""

    if not isinstance(from_end, bool):
        raise ValueError("from_end must be a boolean")
    line = _line(grid, axis, line_index)
    mask = _matching_mask(line.reshape(1, -1), values).reshape(-1)
    sequence = mask[::-1] if from_end else mask
    count = 0
    for matched in sequence:
        if not bool(matched):
            break
        count += 1
    return count


def edge_value_count(grid: Any, values: Any, edge: Any) -> int:
    """Count matching cells along the named top/bottom/left/right edge."""

    board = _grid(grid)
    normalized = str(edge).strip().lower()
    if normalized == "top":
        return line_value_count(board, values, "row", 0)
    if normalized == "bottom":
        return line_value_count(board, values, "row", board.shape[0] - 1)
    if normalized == "left":
        return line_value_count(board, values, "column", 0)
    if normalized == "right":
        return line_value_count(board, values, "column", board.shape[1] - 1)
    raise ValueError("edge must be top, bottom, left, or right")


def edge_run_length(grid: Any, values: Any, edge: Any, offset: Any) -> int:
    """Count matching cells inward from an edge at its row/column offset."""

    board = _grid(grid)
    normalized = str(edge).strip().lower()
    if normalized == "top":
        return line_run_length(board, values, "column", offset)
    if normalized == "bottom":
        return line_run_length(board, values, "column", offset, from_end=True)
    if normalized == "left":
        return line_run_length(board, values, "row", offset)
    if normalized == "right":
        return line_run_length(board, values, "row", offset, from_end=True)
    raise ValueError("edge must be top, bottom, left, or right")


def memory_update(memory: Any, updates: Any) -> dict[str, Any]:
    """Return new bounded JSON memory after applying a mapping of updates."""

    result = _memory_mapping(memory)
    if not isinstance(updates, Mapping):
        raise ValueError("memory updates must be a mapping")
    if any(not isinstance(key, str) or not key for key in updates):
        raise ValueError("memory update keys must be non-empty strings")
    result.update(dict(updates))
    if len(result) > MAX_HELPER_MEMORY_KEYS:
        raise ValueError(f"memory may contain at most {MAX_HELPER_MEMORY_KEYS} keys")
    return _json_clone(result, "memory")


def memory_with_defaults(memory: Any, defaults: Any) -> dict[str, Any]:
    """Fill missing memory fields from bounded JSON defaults without mutation."""

    if not isinstance(defaults, Mapping):
        raise ValueError("memory defaults must be a mapping")
    if any(not isinstance(key, str) or not key for key in defaults):
        raise ValueError("memory default keys must be non-empty strings")
    result = _json_clone(dict(defaults), "memory defaults")
    result.update(_memory_mapping(memory))
    if len(result) > MAX_HELPER_MEMORY_KEYS:
        raise ValueError(f"memory may contain at most {MAX_HELPER_MEMORY_KEYS} keys")
    return _json_clone(result, "memory")


def memory_push(memory: Any, key: Any, value: Any, limit: int = 16) -> dict[str, Any]:
    """Append one JSON value to a bounded rolling list in copied memory."""

    if not isinstance(key, str) or not key:
        raise ValueError("memory key must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64:
        raise ValueError("memory history limit must be an integer between 1 and 64")
    result = _memory_mapping(memory)
    existing = result.get(key, [])
    if not isinstance(existing, (list, tuple)):
        raise ValueError(f"memory value for {key!r} must be a list")
    history = [*existing, _json_clone(value, "memory history value")][-limit:]
    return memory_update(result, {key: history})


def accumulate_transition_evidence(
    memory: Any,
    transition: Any,
    key: Any = "transition_evidence",
    limit: int = 16,
) -> dict[str, Any]:
    """Append one normalized transition snapshot to bounded JSON memory."""

    return memory_push(memory, key, transition_facts(transition), limit=limit)


def history_push(history: Any, value: Any, limit: int = 16) -> list[Any]:
    """Append one JSON value to a copied bounded rolling list.

    This complements ``memory_push`` for policies that keep a standalone list
    before placing it into memory and avoids the easy-to-misread mapping API.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64:
        raise ValueError("history limit must be an integer between 1 and 64")
    if history is None:
        existing: list[Any] = []
    elif isinstance(history, (list, tuple)):
        existing = list(history)
    else:
        raise ValueError("history must be a list or tuple")
    result = [*existing, _json_clone(value, "history value")][-limit:]
    return _json_clone(result, "history")


def memory_increment(
    memory: Any,
    key: Any,
    amount: int | float = 1,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
) -> dict[str, Any]:
    """Increment and optionally clamp a finite numeric memory field."""

    if not isinstance(key, str) or not key:
        raise ValueError("memory key must be a non-empty string")
    numeric_values = (amount, minimum, maximum)
    if any(
        value is not None
        and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        )
        for value in numeric_values
    ):
        raise ValueError("memory increment bounds and amount must be finite numbers")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("memory increment minimum may not exceed maximum")
    result = _memory_mapping(memory)
    current = result.get(key, 0)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(f"memory value for {key!r} must be numeric")
    updated = current + amount
    if minimum is not None:
        updated = max(updated, minimum)
    if maximum is not None:
        updated = min(updated, maximum)
    return memory_update(result, {key: updated})


def recent_outcome_counts(transitions: Any) -> dict[str, int]:
    """Count host outcome classes across a bounded recent-transition window."""

    counts: dict[str, int] = {}
    for transition in _recent_items(transitions):
        outcome = transition_outcome(transition)
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def consecutive_outcome_count(transitions: Any, outcome: Any | None = None) -> int:
    """Count the trailing run of one outcome, defaulting to the most recent outcome."""

    items = _recent_items(transitions)
    if not items:
        return 0
    expected = (
        transition_outcome(items[-1])
        if outcome is None
        else str(outcome).strip().lower()
    )
    if expected not in {
        "failed",
        "guarded",
        "no_progress",
        "progress",
        "terminal",
        "unknown",
    }:
        raise ValueError("unknown transition outcome")
    count = 0
    for transition in reversed(items):
        if transition_outcome(transition) != expected:
            break
        count += 1
    return count


def recent_action_counts(
    transitions: Any, only_nonprogress: bool = False
) -> dict[str, int]:
    """Count model-facing action names, optionally only unsuccessful attempts."""

    counts: dict[str, int] = {}
    for transition in _recent_items(transitions):
        if only_nonprogress and transition_outcome(transition) not in {
            "failed",
            "guarded",
            "no_progress",
        }:
            continue
        action_name = str(transition.get("action") or "").strip().upper()
        if action_name not in POLICY_ACTIONS:
            continue
        counts[action_name] = counts.get(action_name, 0) + 1
    return counts


def recent_mouse_point_counts(
    transitions: Any, only_nonprogress: bool = False
) -> dict[str, int]:
    """Count valid MOUSE coordinates using JSON-safe ``"row,col"`` keys."""

    counts: dict[str, int] = {}
    for transition in _recent_items(transitions):
        if str(transition.get("action") or "").strip().upper() != "MOUSE":
            continue
        if only_nonprogress and transition_outcome(transition) not in {
            "failed",
            "guarded",
            "no_progress",
        }:
            continue
        try:
            row, col = _point((transition.get("row"), transition.get("col")))
        except ValueError:
            continue
        key = f"{row},{col}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def least_tried_mouse_point(
    candidates: Any,
    transitions: Any,
    exclude: Any = (),
    only_nonprogress: bool = False,
    allow_edge_hud: bool = False,
) -> tuple[int, int] | None:
    """Choose a least-attempted click, excluding the thin HUD edge by default."""

    if not isinstance(allow_edge_hud, bool):
        raise ValueError("allow_edge_hud must be a boolean")

    try:
        candidate_iterator = iter(candidates)
        exclude_iterator = iter(exclude)
    except TypeError as exc:
        raise ValueError("candidates and exclude must be point iterables") from exc
    checked_candidates: list[tuple[int, int]] = []
    for position, value in enumerate(candidate_iterator):
        if position >= 64 * 64:
            raise ValueError("candidates may contain at most 4096 points")
        point = _point(value)
        if point not in checked_candidates:
            checked_candidates.append(point)
    excluded: set[tuple[int, int]] = set()
    for position, value in enumerate(exclude_iterator):
        if position >= 64 * 64:
            raise ValueError("exclude may contain at most 4096 points")
        excluded.add(_point(value))
    edge_band = 2
    available = [
        point
        for point in checked_candidates
        if point not in excluded
        and (
            allow_edge_hud
            or (
                edge_band <= point[0] < 64 - edge_band
                and edge_band <= point[1] < 64 - edge_band
            )
        )
    ]
    if not available:
        return None
    counts = recent_mouse_point_counts(
        transitions, only_nonprogress=only_nonprogress
    )
    return min(
        available,
        key=lambda point: (
            counts.get(f"{point[0]},{point[1]}", 0),
            checked_candidates.index(point),
        ),
    )


def least_tried_action(
    valid_actions: Any,
    transitions: Any,
    exclude: Any = (),
    include_mouse: bool = False,
) -> str | None:
    """Choose the least attempted valid action, preserving valid-action tie order."""

    try:
        raw_valid = tuple(valid_actions)
        raw_excluded = tuple(exclude)
    except TypeError as exc:
        raise ValueError("valid_actions and exclude must be iterable") from exc
    if len(raw_valid) > len(POLICY_ACTIONS) or len(raw_excluded) > len(POLICY_ACTIONS):
        raise ValueError("action lists may contain at most seven entries")
    candidates: list[str] = []
    for value in raw_valid:
        action_name = _action_name(value)
        if action_name not in candidates:
            candidates.append(action_name)
    excluded = {_action_name(value) for value in raw_excluded}
    candidates = [
        action
        for action in candidates
        if action not in excluded and (bool(include_mouse) or action != "MOUSE")
    ]
    if not candidates:
        return None
    counts = recent_action_counts(transitions)
    return min(
        candidates, key=lambda action: (counts.get(action, 0), candidates.index(action))
    )


POLICY_CODEGEN_GLOBALS = MappingProxyType(
    {
        "POLICY_CODEGEN_API_VERSION": POLICY_CODEGEN_API_VERSION,
        "POLICY_ACTIONS": POLICY_ACTIONS,
        "accumulate_transition_evidence": accumulate_transition_evidence,
        "action_payload": action_payload,
        "board_digest": board_digest,
        "cells_digest": cells_digest,
        "consecutive_outcome_count": consecutive_outcome_count,
        "continue_decision": continue_decision,
        "edge_run_length": edge_run_length,
        "edge_value_count": edge_value_count,
        "first_matching_cell": first_matching_cell,
        "history_push": history_push,
        "least_tried_action": least_tried_action,
        "least_tried_mouse_point": least_tried_mouse_point,
        "line_run_length": line_run_length,
        "line_value_count": line_value_count,
        "matching_region_center": matching_region_center,
        "memory_increment": memory_increment,
        "memory_push": memory_push,
        "memory_update": memory_update,
        "memory_with_defaults": memory_with_defaults,
        "mouse_decision": mouse_decision,
        "nearest_matching_cell": nearest_matching_cell,
        "objective_evidence_ready": objective_evidence_ready,
        "path_decision": path_decision,
        "region_digest": region_digest,
        "subgoal_failed": subgoal_failed,
        "subgoal_succeeded": subgoal_succeeded,
        "recent_action_counts": recent_action_counts,
        "recent_mouse_point_counts": recent_mouse_point_counts,
        "recent_outcome_counts": recent_outcome_counts,
        "transition_facts": transition_facts,
        "transition_has_progress": transition_has_progress,
        "transition_outcome": transition_outcome,
        "transition_repeats_nonprogress_action": transition_repeats_nonprogress_action,
        "transition_requires_replan": transition_requires_replan,
    }
)
