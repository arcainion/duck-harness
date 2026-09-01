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
POLICY_ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "SPACE", "MOUSE")
POLICY_BOARD_HEX_SYMBOLS = "0123456789abcdef"
MAX_HELPER_MEMORY_BYTES = 32_768
MAX_HELPER_MEMORY_KEYS = 64
MAX_RECENT_TRANSITIONS = 64
_NO_PROGRESS_OUTCOMES = {
    "behavioral_noop",
    "exact_noop",
    "no_progress",
    "volatile_only",
}
_CHANGE_CLASSES = {
    "behavioral_noop",
    "exact_noop",
    "guarded",
    "level_progress",
    "negative_reward",
    "no_progress",
    "novel",
    "revisit",
    "terminal_failure",
    "transient_effect",
    "volatile_only",
}
_STABLE_CHANGE_CLASSES = {"changed", "level_progress", "novel", "revisit"}


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


def _mouse_point_exclusion(value: Any) -> tuple[int, int]:
    """Normalize a point pair or a canonical JSON-safe ``"row,col"`` key."""

    if isinstance(value, str):
        parts = value.split(",")
        if (
            len(parts) != 2
            or any(
                not part
                or not part.isascii()
                or not part.isdigit()
                or str(int(part)) != part
                for part in parts
            )
        ):
            raise ValueError(
                'excluded mouse point must be a (row, col) pair or canonical "row,col" key'
            )
        return _point((int(parts[0]), int(parts[1])))
    if isinstance(value, bytes):
        raise ValueError(
            'excluded mouse point must be a (row, col) pair or canonical "row,col" key'
        )
    return _point(value)


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


def transition_change_class(transition: Any) -> str:
    """Preserve the host's state-change class independently of progress semantics."""

    if not isinstance(transition, Mapping):
        return "unknown"
    outcome = str(transition.get("outcome_class") or "").strip().lower()
    if transition.get("error") or transition.get("executed") is False:
        return "failed"
    if bool(transition.get("game_over")):
        return "terminal_failure"
    if (
        outcome == "guarded"
        or bool(transition.get("cycle_risk"))
        or bool(transition.get("loop_detected"))
    ):
        return "guarded"
    if outcome in _CHANGE_CLASSES:
        return outcome
    if transition.get("executed") is True:
        return "changed" if transition.get("board_changed") is True else "exact_noop"
    return "unknown"


def transition_has_stable_change(transition: Any) -> bool:
    """Return whether an executed transition is stable learning evidence."""

    return (
        isinstance(transition, Mapping)
        and transition.get("executed") is True
        and transition.get("board_changed") is True
        and not transition.get("error")
        and not bool(transition.get("cycle_risk"))
        and not bool(transition.get("loop_detected"))
        and transition_change_class(transition) in _STABLE_CHANGE_CLASSES
    )


def transition_facts(transition: Any) -> dict[str, Any]:
    """Return a bounded JSON snapshot using the host's transition semantics."""

    if not isinstance(transition, Mapping):
        return {
            "outcome": "unknown",
            "change_class": "unknown",
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
        "change_class": transition_change_class(transition),
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


def stable_transition_evidence_status(
    objective: Any, transitions: Any
) -> tuple[bool, str]:
    """Validate the host contract for reproducible tactical learning evidence."""

    if not isinstance(objective, Mapping):
        raise ValueError("objective must be a mapping")
    evidence_mode = str(objective.get("evidence_mode") or "stable_transition")
    if evidence_mode != "stable_transition":
        return False, (
            "stable-transition evidence cannot resolve "
            f"{evidence_mode or 'unknown'} objectives"
        )
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

    objective_id = str(objective.get("objective_id") or objective.get("id") or "")
    relevant: list[Mapping[str, Any]] = []
    for transition in _recent_items(transitions):
        transition_objective_id = str(transition.get("objective_id") or "")
        if objective_id and transition_objective_id and transition_objective_id != objective_id:
            continue
        if transition.get("executed") is not True:
            continue
        if transition.get("post_action_observed") is False:
            continue
        relevant.append(transition)

    if len(relevant) < required_count:
        return False, (
            f"only {len(relevant)} of {required_count} required transition "
            "observations were found"
        )
    if not relevant:
        return False, "no executed post-action transition was found"
    latest = relevant[-1]
    if not transition_has_stable_change(latest):
        outcome = transition_change_class(latest)
        suffix = f" ({outcome})" if outcome != "unknown" else ""
        return False, (
            "latest transition is not an executed, stable, non-cyclic board change"
            + suffix
        )

    stable = [item for item in relevant if transition_has_stable_change(item)]
    minimum_stable = 1 if required_count <= 1 else 2
    if len(stable) < minimum_stable:
        return False, (
            f"only {len(stable)} of {minimum_stable} required stable transition "
            "observations were found"
        )
    if required_count > 1:
        signatures: dict[tuple[Any, Any, Any], int] = {}
        for item in stable:
            signature = (item.get("action"), item.get("row"), item.get("col"))
            signatures[signature] = signatures.get(signature, 0) + 1
        if max(signatures.values(), default=0) < 2:
            return False, (
                "stable changes were not reproduced by the same action or coordinate"
            )
    return True, "reproducible stable-transition requirements are met"


def stable_transition_evidence_ready(objective: Any, transitions: Any) -> bool:
    """Return whether host-verifiable stable tactical evidence is ready."""

    return stable_transition_evidence_status(objective, transitions)[0]


def contrastive_transition_evidence_status(
    objective: Any, transitions: Any
) -> tuple[bool, str]:
    """Validate causal evidence using repeated positive and distinct negative probes."""

    if not isinstance(objective, Mapping):
        raise ValueError("objective must be a mapping")
    evidence_mode = str(objective.get("evidence_mode") or "contrastive_transition")
    if evidence_mode != "contrastive_transition":
        return False, (
            "contrastive evidence cannot resolve "
            f"{evidence_mode or 'unknown'} objectives"
        )
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

    objective_id = str(objective.get("objective_id") or objective.get("id") or "")
    relevant: list[Mapping[str, Any]] = []
    for transition in _recent_items(transitions):
        transition_objective_id = str(transition.get("objective_id") or "")
        if objective_id and transition_objective_id and transition_objective_id != objective_id:
            continue
        if transition.get("executed") is not True:
            continue
        if transition.get("post_action_observed") is False:
            continue
        relevant.append(transition)
    if len(relevant) < required_count:
        return False, (
            f"only {len(relevant)} of {required_count} required transition "
            "observations were found"
        )

    groups: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = {}
    for item in relevant:
        signature = (item.get("action"), item.get("row"), item.get("col"))
        groups.setdefault(signature, []).append(item)
    positive_signatures = {
        signature
        for signature, items in groups.items()
        if sum(transition_has_stable_change(item) for item in items) >= 2
    }
    if not positive_signatures:
        return False, (
            "no exact action or coordinate produced stable change at least twice"
        )

    def action_family(signature: tuple[Any, Any, Any]) -> str:
        action = str(signature[0] or "").strip().upper()
        if action in {"UP", "RIGHT", "DOWN", "LEFT"}:
            return "directional"
        if action == "MOUSE":
            return "mouse"
        return f"action:{action}"

    def safe_negative(item: Mapping[str, Any]) -> bool:
        outcome_class = str(item.get("outcome_class") or "").strip().lower()
        return (
            not transition_has_stable_change(item)
            and not item.get("error")
            and (
                item.get("board_changed") is False
                or outcome_class
                in {
                    "exact_noop",
                    "behavioral_noop",
                    "volatile_only",
                    "transient_effect",
                }
            )
        )

    for positive_signature in positive_signatures:
        positive_family = action_family(positive_signature)
        for signature, items in groups.items():
            if signature == positive_signature:
                continue
            if action_family(signature) != positive_family:
                continue
            if items and all(safe_negative(item) for item in items):
                return True, (
                    "repeated positive and matched same-family negative-control "
                    "transition requirements are met"
                )
    return False, (
        "no matched same-family negative-control action or coordinate was observed "
        "without a corresponding stable change"
    )


def contrastive_transition_evidence_ready(objective: Any, transitions: Any) -> bool:
    """Return whether host-verifiable causal transition evidence is ready."""

    return contrastive_transition_evidence_status(objective, transitions)[0]


def palette_value(symbol: Any) -> int:
    """Convert one reducer-facing hexadecimal board symbol to a value from 0 to 15."""

    if not isinstance(symbol, str) or len(symbol) != 1:
        raise ValueError("palette symbol must be one hexadecimal character")
    normalized = symbol.lower()
    if normalized not in POLICY_BOARD_HEX_SYMBOLS:
        raise ValueError("palette symbol must be one hexadecimal character")
    return POLICY_BOARD_HEX_SYMBOLS.index(normalized)


def palette_values(symbols: Any) -> tuple[int, ...]:
    """Convert reducer-facing hexadecimal symbols to unique board values."""

    if isinstance(symbols, str):
        raw = list(symbols)
    elif isinstance(symbols, (list, tuple, set, frozenset)):
        raw = list(symbols)
    else:
        raise ValueError("palette symbols must be a string or bounded collection")
    if not raw or len(raw) > 16:
        raise ValueError("palette symbols must contain between 1 and 16 characters")
    values = [palette_value(symbol) for symbol in raw]
    return tuple(dict.fromkeys(values))


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


def memory_mapping_increment(
    memory: Any,
    field: Any,
    key: Any,
    amount: int | float = 1,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
) -> dict[str, Any]:
    """Increment one numeric entry inside a copied JSON memory mapping."""

    if not isinstance(field, str) or not field:
        raise ValueError("memory mapping field must be a non-empty string")
    if not isinstance(key, str) or not key:
        raise ValueError("memory mapping key must be a non-empty string")
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
    existing = result.get(field, {})
    if not isinstance(existing, Mapping):
        raise ValueError(f"memory value for {field!r} must be a mapping")
    nested = dict(existing)
    if len(nested) >= MAX_HELPER_MEMORY_KEYS and key not in nested:
        raise ValueError(
            f"memory mapping {field!r} may contain at most "
            f"{MAX_HELPER_MEMORY_KEYS} keys"
        )
    current = nested.get(key, 0)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(f"memory mapping value for {key!r} must be numeric")
    updated = current + amount
    if minimum is not None:
        updated = max(updated, minimum)
    if maximum is not None:
        updated = min(updated, maximum)
    nested[key] = updated
    return memory_update(result, {field: nested})


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
    """Choose a least-attempted click, excluding the thin HUD edge by default.

    Candidate values remain strict point pairs. Exclusions additionally accept the
    canonical ``"row,col"`` keys returned by :func:`recent_mouse_point_counts` so
    JSON policy memory can round-trip those coordinates without reparsing them.
    """

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
        excluded.add(_mouse_point_exclusion(value))
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


def _component_count(mask: np.ndarray, maximum: int = 64) -> tuple[int, bool]:
    """Count bounded four-connected components without exposing mask data."""

    remaining = {
        (int(row), int(col)) for row, col in np.argwhere(np.asarray(mask, dtype=bool))
    }
    count = 0
    while remaining and count < maximum:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            row, col = frontier.pop()
            for candidate in (
                (row - 1, col),
                (row, col + 1),
                (row + 1, col),
                (row, col - 1),
            ):
                if candidate in remaining:
                    remaining.remove(candidate)
                    frontier.append(candidate)
    return count, bool(remaining)


def _component_descriptors(
    board: np.ndarray, background: int, maximum: int = 32
) -> tuple[list[dict[str, Any]], bool]:
    """Describe bounded same-color objects for compact cross-turn matching."""

    descriptors: list[dict[str, Any]] = []
    for raw_value in sorted(int(value) for value in np.unique(board)):
        if raw_value == background:
            continue
        remaining = {
            (int(row), int(col)) for row, col in np.argwhere(np.equal(board, raw_value))
        }
        while remaining:
            component = [remaining.pop()]
            frontier = list(component)
            while frontier:
                row, col = frontier.pop()
                for candidate in (
                    (row - 1, col),
                    (row, col + 1),
                    (row + 1, col),
                    (row, col - 1),
                ):
                    if candidate in remaining:
                        remaining.remove(candidate)
                        component.append(candidate)
                        frontier.append(candidate)
            rows = [point[0] for point in component]
            cols = [point[1] for point in component]
            top, left, bottom, right = min(rows), min(cols), max(rows), max(cols)
            descriptors.append(
                {
                    "value": raw_value,
                    "size": len(component),
                    "bbox": [top, left, bottom, right],
                    "bbox_center_twice": [top + bottom, left + right],
                }
            )
    descriptors.sort(
        key=lambda item: (-int(item["size"]), int(item["value"]), item["bbox"])
    )
    return descriptors[:maximum], len(descriptors) > maximum


def _object_change_summary(
    current: list[dict[str, Any]], previous: Any, *, truncated: bool
) -> dict[str, Any]:
    """Greedily match compact objects by color, size, and nearest prior center."""

    unavailable = {
        "available": False,
        "truncated": truncated,
        "ambiguous_matches": 0,
        "stable_count": 0,
        "moved": [],
        "resized": [],
        "added": [],
        "removed": [],
    }
    if not isinstance(previous, list) or len(previous) > 32:
        return unavailable
    checked: list[dict[str, Any]] = []
    try:
        for item in previous:
            if not isinstance(item, Mapping):
                return unavailable
            center = item.get("bbox_center_twice")
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                return unavailable
            checked.append(
                {
                    "value": int(item["value"]),
                    "size": int(item["size"]),
                    "bbox_center_twice": [int(center[0]), int(center[1])],
                }
            )
    except (KeyError, TypeError, ValueError, OverflowError):
        return unavailable

    remaining_previous = set(range(len(checked)))
    unmatched_current: list[int] = []
    moved: list[dict[str, Any]] = []
    stable_count = 0
    ambiguous_matches = 0
    for current_index, item in enumerate(current):
        center = item["bbox_center_twice"]
        candidates = [
            previous_index
            for previous_index in remaining_previous
            if checked[previous_index]["value"] == item["value"]
            and checked[previous_index]["size"] == item["size"]
        ]
        if not candidates:
            unmatched_current.append(current_index)
            continue
        distances = {
            index_value: abs(
                checked[index_value]["bbox_center_twice"][0] - center[0]
            )
            + abs(checked[index_value]["bbox_center_twice"][1] - center[1])
            for index_value in candidates
        }
        nearest_distance = min(distances.values())
        ambiguous_matches += sum(
            1 for distance in distances.values() if distance == nearest_distance
        ) > 1
        previous_index = min(
            candidates,
            key=lambda index_value: (
                distances[index_value],
                index_value,
            ),
        )
        remaining_previous.remove(previous_index)
        old_center = checked[previous_index]["bbox_center_twice"]
        shift = [center[0] - old_center[0], center[1] - old_center[1]]
        if shift == [0, 0]:
            stable_count += 1
        else:
            moved.append(
                {
                    "value": item["value"],
                    "size": item["size"],
                    "from_center_twice": old_center,
                    "to_center_twice": center,
                    "shift_twice": shift,
                }
            )

    resized: list[dict[str, Any]] = []
    added_indices: list[int] = []
    for current_index in unmatched_current:
        item = current[current_index]
        center = item["bbox_center_twice"]
        candidates = [
            previous_index
            for previous_index in remaining_previous
            if checked[previous_index]["value"] == item["value"]
        ]
        if not candidates:
            added_indices.append(current_index)
            continue
        distances = {
            index_value: abs(
                checked[index_value]["bbox_center_twice"][0] - center[0]
            )
            + abs(checked[index_value]["bbox_center_twice"][1] - center[1])
            for index_value in candidates
        }
        nearest_distance = min(distances.values())
        ambiguous_matches += sum(
            1 for distance in distances.values() if distance == nearest_distance
        ) > 1
        previous_index = min(
            candidates,
            key=lambda index_value: (
                distances[index_value],
                index_value,
            ),
        )
        remaining_previous.remove(previous_index)
        resized.append(
            {
                "value": item["value"],
                "from_size": checked[previous_index]["size"],
                "to_size": item["size"],
                "from_center_twice": checked[previous_index]["bbox_center_twice"],
                "to_center_twice": center,
            }
        )

    def compact(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "value": int(item["value"]),
            "size": int(item["size"]),
            "center_twice": list(item["bbox_center_twice"]),
        }

    return {
        "available": True,
        "truncated": truncated,
        "ambiguous_matches": ambiguous_matches,
        "stable_count": stable_count,
        "moved": moved,
        "resized": resized,
        "added": [compact(current[index_value]) for index_value in added_indices],
        "removed": [compact(checked[index_value]) for index_value in sorted(remaining_previous)],
    }


def _object_relation_summary(
    objects: list[dict[str, Any]], maximum: int = 32
) -> dict[str, Any]:
    """Return bounded nearest-box relations and repeated-shape groups."""

    relations: list[dict[str, Any]] = []
    counts = {
        "same_value": 0,
        "same_size": 0,
        "row_overlap": 0,
        "column_overlap": 0,
        "bbox_contact_candidates": 0,
        "bbox_contains": 0,
    }
    shape_groups: dict[tuple[int, int, int], list[int]] = {}
    for object_index, item in enumerate(objects):
        box = item["bbox"]
        shape_groups.setdefault(
            (int(item["size"]), box[2] - box[0] + 1, box[3] - box[1] + 1),
            [],
        ).append(object_index)
    for first_index, first in enumerate(objects):
        for second_index in range(first_index + 1, len(objects)):
            second = objects[second_index]
            first_box, second_box = first["bbox"], second["bbox"]
            row_overlap = max(
                0,
                min(first_box[2], second_box[2])
                - max(first_box[0], second_box[0])
                + 1,
            )
            column_overlap = max(
                0,
                min(first_box[3], second_box[3])
                - max(first_box[1], second_box[1])
                + 1,
            )
            row_gap = max(
                0,
                second_box[0] - first_box[2] - 1,
                first_box[0] - second_box[2] - 1,
            )
            column_gap = max(
                0,
                second_box[1] - first_box[3] - 1,
                first_box[1] - second_box[3] - 1,
            )
            contains = (
                first_box[0] <= second_box[0]
                and first_box[1] <= second_box[1]
                and first_box[2] >= second_box[2]
                and first_box[3] >= second_box[3]
            ) or (
                second_box[0] <= first_box[0]
                and second_box[1] <= first_box[1]
                and second_box[2] >= first_box[2]
                and second_box[3] >= first_box[3]
            )
            same_value = first["value"] == second["value"]
            same_size = first["size"] == second["size"]
            bbox_gap = row_gap + column_gap
            counts["same_value"] += same_value
            counts["same_size"] += same_size
            counts["row_overlap"] += row_overlap > 0
            counts["column_overlap"] += column_overlap > 0
            counts["bbox_contact_candidates"] += bbox_gap == 0
            counts["bbox_contains"] += contains
            relations.append(
                {
                    "first_index": first_index,
                    "second_index": second_index,
                    "bbox_gap": bbox_gap,
                    "row_gap": row_gap,
                    "column_gap": column_gap,
                    "row_overlap": row_overlap,
                    "column_overlap": column_overlap,
                    "same_value": same_value,
                    "same_size": same_size,
                    "bbox_contains": contains,
                }
            )
    relations.sort(
        key=lambda item: (
            int(item["bbox_gap"]),
            int(item["first_index"]),
            int(item["second_index"]),
        )
    )
    repeated_shapes = [
        {
            "size": key[0],
            "height": key[1],
            "width": key[2],
            "object_indices": indices,
        }
        for key, indices in sorted(shape_groups.items())
        if len(indices) >= 2
    ][:16]
    return {
        "relation_count": len(relations),
        "relations": relations[:maximum],
        "relations_truncated": len(relations) > maximum,
        "counts": counts,
        "repeated_shapes": repeated_shapes,
    }


def _previous_state_token(previous_state: Any) -> Mapping[str, Any] | None:
    """Accept a prior full inference or its compact state token defensively."""

    if not isinstance(previous_state, Mapping):
        return None
    nested = previous_state.get("state_token")
    candidate = nested if isinstance(nested, Mapping) else previous_state
    if not isinstance(candidate.get("digest"), str):
        return None
    return candidate


def infer_game_state(
    observation: Any, previous_state: Any | None = None
) -> dict[str, Any]:
    """Return bounded JSON structure, evidence, and optional cross-turn state delta."""

    board = _grid(getattr(observation, "board", None))
    raw_actions = getattr(observation, "valid_actions", ())
    try:
        action_iterator = iter(raw_actions)
    except TypeError as exc:
        raise ValueError("observation.valid_actions must be iterable") from exc
    valid_actions: list[str] = []
    for position, raw_action in enumerate(action_iterator):
        if position >= len(POLICY_ACTIONS):
            raise ValueError("observation.valid_actions contains too many entries")
        action = str(raw_action or "").strip().upper()
        if action in POLICY_ACTIONS and action not in valid_actions:
            valid_actions.append(action)

    recent = _recent_items(getattr(observation, "recent_transitions", ()))
    last_transition = getattr(observation, "last_transition", None)
    if isinstance(last_transition, Mapping) and (
        not recent or recent[-1] != last_transition
    ):
        recent = (*recent, last_transition)[-MAX_RECENT_TRANSITIONS:]

    level = max(0, int(getattr(observation, "level", 0) or 0))
    raw_previous = _previous_state_token(previous_state)
    scope_matches = False
    if raw_previous is not None:
        try:
            previous_shape = raw_previous.get("shape")
            scope_matches = (
                int(raw_previous["level"]) == level
                and isinstance(previous_shape, (list, tuple))
                and [int(previous_shape[0]), int(previous_shape[1])]
                == [int(board.shape[0]), int(board.shape[1])]
            )
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            scope_matches = False
    previous = raw_previous if scope_matches else None
    values, counts = np.unique(board, return_counts=True)
    ranked_palette = sorted(
        ((int(count), int(value)) for value, count in zip(values, counts, strict=True)),
        key=lambda item: (-item[0], item[1]),
    )
    background = ranked_palette[0][1]
    background_source = "dominant_color"
    if previous is not None:
        try:
            previous_background = int(previous["background_value"])
            current_counts = {value: count for count, value in ranked_palette}
            if current_counts.get(previous_background, 0) >= max(1, int(board.size) // 8):
                background = previous_background
                background_source = "previous_state"
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    edge_mask = np.zeros(board.shape, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    palette: list[dict[str, Any]] = []
    for count, value in ranked_palette[:16]:
        mask = np.equal(board, value)
        points = np.argwhere(mask)
        components, components_truncated = _component_count(mask)
        palette.append(
            {
                "value": value,
                "count": count,
                "bbox": [
                    int(points[:, 0].min()),
                    int(points[:, 1].min()),
                    int(points[:, 0].max()),
                    int(points[:, 1].max()),
                ],
                "components": components,
                "components_truncated": components_truncated,
                "edge_cells": int(np.count_nonzero(mask & edge_mask)),
            }
        )
    foreground = np.not_equal(board, background)
    foreground_points = np.argwhere(foreground)
    foreground_bbox = (
        [
            int(foreground_points[:, 0].min()),
            int(foreground_points[:, 1].min()),
            int(foreground_points[:, 0].max()),
            int(foreground_points[:, 1].max()),
        ]
        if foreground_points.size
        else None
    )
    foreground_components, foreground_components_truncated = _component_count(foreground)
    objects, objects_truncated = _component_descriptors(board, background)
    object_layout = _object_relation_summary(objects)
    row_occupancy = np.count_nonzero(foreground, axis=1)
    col_occupancy = np.count_nonzero(foreground, axis=0)
    horizontal_symmetry = round(
        float(np.count_nonzero(np.equal(board, np.flip(board, axis=1))))
        / float(board.size),
        4,
    )
    vertical_symmetry = round(
        float(np.count_nonzero(np.equal(board, np.flip(board, axis=0))))
        / float(board.size),
        4,
    )

    action_effects: dict[str, dict[str, Any]] = {
        action: {
            "observed": 0,
            "executed": 0,
            "changed": 0,
            "stable_changed": 0,
            "progress": 0,
            "no_progress": 0,
            "guarded": 0,
            "distinct_points": 0,
        }
        for action in POLICY_ACTIONS
        if action in valid_actions
        or any(str(item.get("action") or "").strip().upper() == action for item in recent)
    }
    mouse_points: set[str] = set()
    for transition in recent:
        action = str(transition.get("action") or "").strip().upper()
        if action not in action_effects:
            continue
        effect = action_effects[action]
        effect["observed"] += 1
        if transition.get("executed") is True:
            effect["executed"] += 1
        if transition.get("board_changed") is True:
            effect["changed"] += 1
        if transition_has_stable_change(transition):
            effect["stable_changed"] += 1
        if transition_has_progress(transition):
            effect["progress"] += 1
        outcome = transition_outcome(transition)
        if outcome in {"failed", "guarded", "no_progress"}:
            effect["no_progress"] += 1
        if outcome == "guarded":
            effect["guarded"] += 1
        if action == "MOUSE":
            try:
                row, col = _point((transition.get("row"), transition.get("col")))
            except ValueError:
                continue
            mouse_points.add(f"{row},{col}")
    if "MOUSE" in action_effects:
        action_effects["MOUSE"]["distinct_points"] = len(mouse_points)
    for effect in action_effects.values():
        executed = int(effect["executed"])
        stable_changed = int(effect["stable_changed"])
        if executed == 0:
            classification = "untested"
        elif int(effect["progress"]) > 0:
            classification = "progressive"
        elif stable_changed >= 2:
            classification = "reliable_change"
        elif stable_changed > 0:
            classification = "responsive"
        elif int(effect["no_progress"]) >= 2:
            classification = "ineffective"
        else:
            classification = "inconclusive"
        effect["classification"] = classification
        effect["stable_rate"] = (
            round(float(stable_changed) / float(executed), 4) if executed else 0.0
        )

    motion_directions: dict[str, int] = {}
    expected_directions = {
        "UP": "up",
        "RIGHT": "right",
        "DOWN": "down",
        "LEFT": "left",
    }
    inverse_directions = {
        "up": "down",
        "right": "left",
        "down": "up",
        "left": "right",
    }
    directional_evidence: dict[str, dict[str, int]] = {}
    object_motion_evidence: dict[
        str, dict[str, dict[Any, int]]
    ] = {}
    for transition in recent:
        animation = transition.get("animation_summary")
        if not isinstance(animation, Mapping):
            continue
        action = str(transition.get("action") or "").strip().upper()
        direction = str(animation.get("motion_direction") or "").strip().lower()
        if direction in {"up", "right", "down", "left"}:
            motion_directions[direction] = motion_directions.get(direction, 0) + 1
            if action in expected_directions and transition_has_stable_change(transition):
                counts_for_action = directional_evidence.setdefault(action, {})
                counts_for_action[direction] = counts_for_action.get(direction, 0) + 1
        object_motion = animation.get("object_motion")
        if (
            action in expected_directions
            and transition_has_stable_change(transition)
            and isinstance(object_motion, Mapping)
            and object_motion.get("tracking_available") is True
        ):
            classification = str(object_motion.get("classification") or "").strip()
            if classification not in {
                "coherent",
                "opposing",
                "divergent",
                "stationary",
                "ambiguous",
                "edge_only",
            }:
                continue
            raw_shifts = object_motion.get("salient_distinct_shifts_twice")
            if not isinstance(raw_shifts, (list, tuple)):
                raw_shifts = object_motion.get("distinct_shifts_twice")
            shift_set: tuple[tuple[int, int], ...] = ()
            if isinstance(raw_shifts, (list, tuple)):
                try:
                    shift_set = tuple(
                        sorted(
                            (int(shift[0]), int(shift[1]))
                            for shift in raw_shifts[:8]
                            if isinstance(shift, (list, tuple)) and len(shift) == 2
                        )
                    )
                except (TypeError, ValueError, OverflowError):
                    shift_set = ()
            evidence = object_motion_evidence.setdefault(
                action, {"classifications": {}, "shift_sets": {}}
            )
            classifications = evidence["classifications"]
            classifications[classification] = classifications.get(classification, 0) + 1
            shift_sets = evidence["shift_sets"]
            shift_sets[shift_set] = shift_sets.get(shift_set, 0) + 1

    directional_by_action: dict[str, dict[str, Any]] = {}
    directional_samples = 0
    for action in expected_directions:
        counts_for_action = directional_evidence.get(action, {})
        samples = sum(counts_for_action.values())
        if not samples:
            continue
        directional_samples += samples
        dominant_direction, dominant_count = sorted(
            counts_for_action.items(), key=lambda item: (-item[1], item[0])
        )[0]
        expected_direction = expected_directions[action]
        if dominant_direction == expected_direction:
            mapping = "aligned"
        elif dominant_direction == inverse_directions[expected_direction]:
            mapping = "inverted"
        else:
            mapping = "rotated"
        directional_by_action[action] = {
            "samples": samples,
            "motion_directions": dict(sorted(counts_for_action.items())),
            "dominant_motion_direction": dominant_direction,
            "consistency": round(float(dominant_count) / float(samples), 4),
            "mapping": mapping,
        }
    observed_mappings = [
        str(item["mapping"]) for item in directional_by_action.values()
    ]
    mapping_is_consistent = all(
        float(item["consistency"]) >= 0.75 for item in directional_by_action.values()
    )
    if not observed_mappings:
        control_scheme = "unknown"
    elif mapping_is_consistent and all(item == "aligned" for item in observed_mappings):
        control_scheme = "standard"
    elif mapping_is_consistent and all(item == "inverted" for item in observed_mappings):
        control_scheme = "inverted"
    elif mapping_is_consistent and len(set(observed_mappings)) == 1:
        control_scheme = "rotated"
    elif mapping_is_consistent:
        control_scheme = "remapped"
    else:
        control_scheme = "mixed_or_state_dependent"
    control_scheme_confidence = (
        "high"
        if directional_samples >= 4 and len(directional_by_action) >= 2
        else "medium"
        if directional_samples >= 2
        else "low"
    )
    object_motion_by_action: dict[str, dict[str, Any]] = {}
    object_motion_classes: dict[str, int] = {}
    object_motion_samples = 0
    for action in expected_directions:
        evidence = object_motion_evidence.get(action)
        if evidence is None:
            continue
        classifications = evidence["classifications"]
        samples = sum(classifications.values())
        object_motion_samples += samples
        for classification, count in classifications.items():
            key = str(classification)
            object_motion_classes[key] = object_motion_classes.get(key, 0) + int(count)
        dominant_class, dominant_count = sorted(
            classifications.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )[0]
        shift_sets = sorted(
            evidence["shift_sets"].items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:8]
        object_motion_by_action[action] = {
            "samples": samples,
            "classifications": dict(sorted(classifications.items())),
            "dominant_classification": str(dominant_class),
            "consistency": round(float(dominant_count) / float(samples), 4),
            "observed_shift_sets_twice": [
                {
                    "shifts": [list(shift) for shift in shift_set],
                    "count": int(count),
                }
                for shift_set, count in shift_sets
            ],
        }
    if object_motion_classes.get("opposing", 0) and object_motion_classes.get(
        "coherent", 0
    ):
        object_motion_scheme = "linked_mixed"
    elif object_motion_classes.get("opposing", 0):
        object_motion_scheme = "linked_opposing"
    elif object_motion_classes.get("divergent", 0):
        object_motion_scheme = "divergent"
    elif object_motion_classes.get("coherent", 0):
        object_motion_scheme = "coherent"
    elif object_motion_samples:
        object_motion_scheme = "stationary_or_ambiguous"
    else:
        object_motion_scheme = "unknown"
    control_dynamics = {
        "scheme": control_scheme,
        "confidence": control_scheme_confidence,
        "directional_samples": directional_samples,
        "tested_actions": list(directional_by_action),
        "by_action": directional_by_action,
        "object_motion": {
            "scheme": object_motion_scheme,
            "samples": object_motion_samples,
            "classifications": dict(sorted(object_motion_classes.items())),
            "tested_actions": list(object_motion_by_action),
            "by_action": object_motion_by_action,
        },
        "advisory": True,
    }

    outcomes = [transition_outcome(item) for item in recent]
    latest = transition_facts(last_transition)
    if latest["run_complete"] or latest["game_over"] or latest["level_completed"]:
        phase = "terminal"
    elif latest["meaningful_progress"]:
        phase = "progress"
    elif not recent:
        phase = "initial"
    elif len(outcomes) >= 3 and all(
        outcome in {"failed", "guarded", "no_progress"} for outcome in outcomes[-4:]
    ):
        phase = "stalled"
    else:
        phase = "active"

    objective = getattr(observation, "objective", {})
    objective_summary: dict[str, Any] = {}
    if isinstance(objective, Mapping):
        for key in (
            "objective_id",
            "evidence_mode",
            "execution_mode",
            "solver_type",
        ):
            value = objective.get(key)
            if value is not None:
                objective_summary[key] = str(value)[:200]
        budget = objective.get("action_budget")
        used = objective.get("actions_used")
        if isinstance(budget, int) and not isinstance(budget, bool) and budget >= 0:
            objective_summary["action_budget"] = budget
        if isinstance(used, int) and not isinstance(used, bool) and used >= 0:
            objective_summary["actions_used"] = used
        if "action_budget" in objective_summary and "actions_used" in objective_summary:
            objective_summary["remaining_actions"] = max(
                0,
                objective_summary["action_budget"]
                - objective_summary["actions_used"],
            )

    palette_counts = {str(item["value"]): int(item["count"]) for item in palette}
    state_token = {
        "schema_version": 2,
        "level": level,
        "shape": [int(board.shape[0]), int(board.shape[1])],
        "digest": board_digest(board),
        "background_value": background,
        "foreground_count": int(np.count_nonzero(foreground)),
        "foreground_bbox": foreground_bbox,
        "foreground_components": foreground_components,
        "palette_counts": palette_counts,
        "horizontal_symmetry": horizontal_symmetry,
        "vertical_symmetry": vertical_symmetry,
        "objects": objects,
        "objects_truncated": objects_truncated,
    }
    state_delta: dict[str, Any] = {
        "comparable": False,
        "board_changed": None,
        "change_type": "scope_reset" if raw_previous is not None else "unavailable",
        "scope_reset_reason": (
            "level_or_shape_changed"
            if raw_previous is not None and not scope_matches
            else ""
        ),
        "foreground_count_delta": None,
        "component_count_delta": None,
        "bbox_center_shift_twice": None,
        "palette_count_delta": {},
        "object_changes": _object_change_summary(
            objects,
            previous.get("objects") if previous is not None else None,
            truncated=objects_truncated
            or bool(previous.get("objects_truncated"))
            if previous is not None
            else objects_truncated,
        ),
    }
    if previous is not None:
        try:
            previous_count = int(previous["foreground_count"])
            previous_components = int(previous["foreground_components"])
            previous_bbox = previous.get("foreground_bbox")
            previous_palette = previous.get("palette_counts")
            if not isinstance(previous_palette, Mapping):
                raise ValueError
            checked_previous_palette = {
                str(key): int(value) for key, value in previous_palette.items()
            }
            palette_keys = sorted(
                set(palette_counts) | set(checked_previous_palette),
                key=lambda value: (int(value) if value.lstrip("-").isdigit() else 256, value),
            )[:32]
            palette_delta = {
                key: palette_counts.get(key, 0) - checked_previous_palette.get(key, 0)
                for key in palette_keys
                if palette_counts.get(key, 0) != checked_previous_palette.get(key, 0)
            }
            center_shift: list[int] | None = None
            if (
                isinstance(previous_bbox, (list, tuple))
                and len(previous_bbox) == 4
                and foreground_bbox is not None
            ):
                center_shift = [
                    foreground_bbox[0]
                    + foreground_bbox[2]
                    - int(previous_bbox[0])
                    - int(previous_bbox[2]),
                    foreground_bbox[1]
                    + foreground_bbox[3]
                    - int(previous_bbox[1])
                    - int(previous_bbox[3]),
                ]
            count_delta = state_token["foreground_count"] - previous_count
            component_delta = foreground_components - previous_components
            changed = state_token["digest"] != str(previous["digest"])
            object_changes = _object_change_summary(
                objects,
                previous.get("objects"),
                truncated=objects_truncated or bool(previous.get("objects_truncated")),
            )
            pure_object_motion = (
                bool(object_changes["moved"])
                and not object_changes["truncated"]
                and not object_changes["ambiguous_matches"]
                and not object_changes["resized"]
                and not object_changes["added"]
                and not object_changes["removed"]
            )
            if not changed:
                change_type = "unchanged"
            elif pure_object_motion:
                change_type = "translation_candidate"
            elif count_delta > 0:
                change_type = "growth"
            elif count_delta < 0:
                change_type = "shrink"
            elif component_delta:
                change_type = "topology_change"
            elif not palette_delta and center_shift not in (None, [0, 0]):
                change_type = "translation_candidate"
            elif palette_delta:
                change_type = "recolor_or_transform"
            else:
                change_type = "structural_change"
            state_delta = {
                "comparable": True,
                "board_changed": changed,
                "change_type": change_type,
                "scope_reset_reason": "",
                "foreground_count_delta": count_delta,
                "component_count_delta": component_delta,
                "bbox_center_shift_twice": center_shift,
                "palette_count_delta": palette_delta,
                "object_changes": object_changes,
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            pass

    summary = {
        "schema_version": 5,
        "level": level,
        "step": max(0, int(getattr(observation, "step", 0) or 0)),
        "phase": phase,
        "controls": {
            "valid_actions": valid_actions,
            "movement_actions": [
                action
                for action in ("UP", "RIGHT", "DOWN", "LEFT")
                if action in valid_actions
            ],
            "has_space": "SPACE" in valid_actions,
            "has_mouse": "MOUSE" in valid_actions,
            "dynamics": control_dynamics,
        },
        "board": {
            "shape": [int(board.shape[0]), int(board.shape[1])],
            "digest": state_token["digest"],
            "background_value": background,
            "background_source": background_source,
            "foreground_count": int(np.count_nonzero(foreground)),
            "foreground_bbox": foreground_bbox,
            "foreground_components": foreground_components,
            "foreground_components_truncated": foreground_components_truncated,
            "edge_foreground_count": int(np.count_nonzero(foreground & edge_mask)),
            "horizontal_symmetry": horizontal_symmetry,
            "vertical_symmetry": vertical_symmetry,
            "peak_row_occupancy": int(row_occupancy.max()),
            "peak_column_occupancy": int(col_occupancy.max()),
            "objects": objects,
            "objects_truncated": objects_truncated,
            "object_layout": object_layout,
            "palette": palette,
            "palette_truncated": len(ranked_palette) > 16,
        },
        "objective": objective_summary,
        "state_token": state_token,
        "state_delta": state_delta,
        "latest_transition": latest,
        "recent": {
            "count": len(recent),
            "outcomes": recent_outcome_counts(recent),
            "motion_directions": motion_directions,
            "actions": action_effects,
        },
    }
    return _json_clone(summary, "game state inference")


def infer_game_type(
    observation: Any, previous_state: Any | None = None
) -> dict[str, Any]:
    """Rank advisory solver-family hypotheses from bounded host-observed evidence."""

    state = infer_game_state(observation, previous_state)
    effects = state["recent"]["actions"]
    movement = [effects.get(action, {}) for action in ("UP", "RIGHT", "DOWN", "LEFT")]
    movement_changed = sum(int(item.get("stable_changed", 0)) for item in movement)
    movement_no_progress = sum(int(item.get("no_progress", 0)) for item in movement)
    interaction_changed = sum(
        int(effects.get(action, {}).get("stable_changed", 0))
        for action in ("SPACE", "MOUSE")
    )
    interaction_no_progress = sum(
        int(effects.get(action, {}).get("no_progress", 0))
        for action in ("SPACE", "MOUSE")
    )
    changed_action_count = sum(
        1 for item in effects.values() if int(item.get("stable_changed", 0)) > 0
    )
    controls = state["controls"]
    control_dynamics = controls["dynamics"]
    transition_motion = control_dynamics["object_motion"]
    transition_motion_classes = transition_motion["classifications"]
    transition_opposing = int(transition_motion_classes.get("opposing", 0))
    transition_divergent = int(transition_motion_classes.get("divergent", 0))
    delta_type = str(state["state_delta"]["change_type"])
    moved_objects = state["state_delta"]["object_changes"]["moved"]
    distinct_shifts = sorted(
        {
            tuple(int(value) for value in item["shift_twice"])
            for item in moved_objects
        }
    )
    independent_motion = (
        len(moved_objects) >= 2 and len(distinct_shifts) >= 2
    ) or bool(transition_opposing or transition_divergent)
    transform_score = (
        4
        if delta_type in {"topology_change", "recolor_or_transform", "structural_change"}
        else 2
        if delta_type in {"growth", "shrink"}
        else 0
    )
    scores = {
        "hybrid": 6 if movement_changed and interaction_changed else 0,
        "routing": max(
            0,
            (1 if controls["movement_actions"] else 0)
            + min(4, movement_changed)
            + (2 if delta_type == "translation_candidate" else 0)
            - (1 if movement_no_progress >= 3 and not movement_changed else 0),
        ),
        "interaction": (1 if controls["has_mouse"] or controls["has_space"] else 0)
        + min(4, interaction_changed)
        - (1 if interaction_no_progress >= 3 and not interaction_changed else 0),
        "sequence": 3 if changed_action_count >= 2 else 0,
        "multi_agent": (
            6
            if transition_opposing
            else 5
            if transition_divergent
            else 4
            if independent_motion
            else 0
        ),
        "transform": transform_score,
        "observe": 2 if not movement_changed and not interaction_changed else 0,
    }
    evidence = {
        "hybrid": "stable changes were observed from movement and interaction controls",
        "routing": f"{movement_changed} stable movement transition(s) observed",
        "interaction": f"{interaction_changed} stable SPACE/MOUSE transition(s) observed",
        "sequence": f"{changed_action_count} distinct controls produced stable changes",
        "multi_agent": (
            f"{len(moved_objects)} cross-turn object(s), {transition_opposing} opposing "
            f"and {transition_divergent} divergent transition sample(s)"
        ),
        "transform": f"cross-turn state delta was classified as {delta_type}",
        "observe": "insufficient stable action-effect evidence; continue bounded probing",
    }
    ordered = sorted(scores, key=lambda family: (-scores[family], family))
    candidates = [
        {
            "family": family,
            "score": scores[family],
            "evidence": evidence[family],
        }
        for family in ordered
        if scores[family] > 0
    ]
    primary = candidates[0]["family"] if candidates else "observe"
    primary_score = int(candidates[0]["score"]) if candidates else 0
    executed_count = sum(int(item.get("executed", 0)) for item in effects.values())
    tested_valid_actions = [
        action
        for action in controls["valid_actions"]
        if int(effects.get(action, {}).get("executed", 0)) > 0
    ]
    stable_change_count = sum(
        int(item.get("stable_changed", 0)) for item in effects.values()
    )
    progress_count = sum(int(item.get("progress", 0)) for item in effects.values())
    confidence = (
        "high"
        if primary_score >= 5 and executed_count >= 4
        else "medium"
        if primary_score >= 3 and executed_count >= 2
        else "low"
    )
    recommendations = {
        "hybrid": ["hybrid"],
        "routing": ["navigation", "lattice-corridor", "guided-attraction"],
        "interaction": ["click-interaction", "relation-toggle", "signal"],
        "sequence": ["symbol-rule-sequence", "paired-sequence-arm"],
        "multi_agent": [
            "multi-agent",
            "linked-centroid",
            "paired-platform-alignment",
        ],
        "transform": ["pattern-transform", "transform-program", "mirror"],
        "observe": ["static", "cellular-automata"],
    }
    probe_priority = {"untested": 0, "inconclusive": 1, "responsive": 2, "ineffective": 3}
    probe_candidates = sorted(
        controls["valid_actions"],
        key=lambda action: (
            probe_priority.get(
                str(effects.get(action, {}).get("classification") or "untested"), 4
            ),
            int(effects.get(action, {}).get("executed", 0)),
            controls["valid_actions"].index(action),
        ),
    )
    recommended_probes = [
        {
            "action": action,
            "requires_point": action == "MOUSE",
            "reason": (
                "untested valid control"
                if effects.get(action, {}).get("classification") == "untested"
                else "least-observed unresolved control"
            ),
        }
        for action in probe_candidates
        if effects.get(action, {}).get("classification")
        not in {"progressive", "reliable_change"}
    ][:4]

    execution_mode = str(state["objective"].get("execution_mode") or "").strip().lower()
    compatible = {
        "navigate": {"routing", "multi_agent", "hybrid"},
        "interact": {"interaction", "hybrid"},
        "probe": {
            "observe",
            "routing",
            "interaction",
            "sequence",
            "multi_agent",
            "transform",
            "hybrid",
        },
    }
    if not execution_mode or execution_mode not in compatible:
        alignment_status = "unknown"
        alignment_reason = "objective execution mode is unavailable"
    elif primary in compatible[execution_mode]:
        alignment_status = "compatible"
        alignment_reason = "inferred family is compatible with the declared execution mode"
    elif confidence == "high":
        alignment_status = "conflict"
        alignment_reason = "high-confidence observed dynamics conflict with the execution mode"
    else:
        alignment_status = "uncertain"
        alignment_reason = "current evidence does not yet support the execution mode"

    return {
        "schema_version": 5,
        "primary_family": primary,
        "confidence": confidence,
        "candidates": candidates,
        "recommended_solver_types": recommendations[primary],
        "recommended_probes": recommended_probes,
        "evidence_coverage": {
            "recent_transitions": int(state["recent"]["count"]),
            "executed_actions": executed_count,
            "tested_valid_actions": tested_valid_actions,
            "stable_changes": stable_change_count,
            "engine_progress": progress_count,
        },
        "objective_alignment": {
            "status": alignment_status,
            "execution_mode": execution_mode,
            "reason": alignment_reason,
        },
        "object_motion": {
            "tracking_available": bool(
                state["state_delta"]["object_changes"]["available"]
            ),
            "moved_objects": len(
                moved_objects
            ),
            "distinct_shifts_twice": [list(shift) for shift in distinct_shifts[:8]],
            "coherent_motion": bool(moved_objects) and len(distinct_shifts) == 1,
            "independent_motion": independent_motion,
            "resized_objects": len(
                state["state_delta"]["object_changes"]["resized"]
            ),
            "added_objects": len(
                state["state_delta"]["object_changes"]["added"]
            ),
            "removed_objects": len(
                state["state_delta"]["object_changes"]["removed"]
            ),
            "truncated": bool(
                state["state_delta"]["object_changes"]["truncated"]
            ),
            "ambiguous_matches": int(
                state["state_delta"]["object_changes"]["ambiguous_matches"]
            ),
            "transition_scheme": transition_motion["scheme"],
            "transition_samples": int(transition_motion["samples"]),
            "transition_classifications": transition_motion_classes,
            "transition_by_action": transition_motion["by_action"],
        },
        "control_scheme": control_dynamics,
        "state_change_type": delta_type,
        "state_phase": state["phase"],
    }


POLICY_CODEGEN_GLOBALS = MappingProxyType(
    {
        "POLICY_CODEGEN_API_VERSION": POLICY_CODEGEN_API_VERSION,
        "POLICY_ACTIONS": POLICY_ACTIONS,
        "POLICY_BOARD_HEX_SYMBOLS": POLICY_BOARD_HEX_SYMBOLS,
        "accumulate_transition_evidence": accumulate_transition_evidence,
        "action_payload": action_payload,
        "board_digest": board_digest,
        "cells_digest": cells_digest,
        "consecutive_outcome_count": consecutive_outcome_count,
        "contrastive_transition_evidence_ready": contrastive_transition_evidence_ready,
        "continue_decision": continue_decision,
        "edge_run_length": edge_run_length,
        "edge_value_count": edge_value_count,
        "first_matching_cell": first_matching_cell,
        "history_push": history_push,
        "infer_game_state": infer_game_state,
        "infer_game_type": infer_game_type,
        "least_tried_action": least_tried_action,
        "least_tried_mouse_point": least_tried_mouse_point,
        "line_run_length": line_run_length,
        "line_value_count": line_value_count,
        "matching_region_center": matching_region_center,
        "memory_increment": memory_increment,
        "memory_mapping_increment": memory_mapping_increment,
        "memory_push": memory_push,
        "memory_update": memory_update,
        "memory_with_defaults": memory_with_defaults,
        "mouse_decision": mouse_decision,
        "nearest_matching_cell": nearest_matching_cell,
        "objective_evidence_ready": objective_evidence_ready,
        "palette_value": palette_value,
        "palette_values": palette_values,
        "path_decision": path_decision,
        "region_digest": region_digest,
        "subgoal_failed": subgoal_failed,
        "subgoal_succeeded": subgoal_succeeded,
        "recent_action_counts": recent_action_counts,
        "recent_mouse_point_counts": recent_mouse_point_counts,
        "recent_outcome_counts": recent_outcome_counts,
        "transition_facts": transition_facts,
        "transition_change_class": transition_change_class,
        "transition_has_progress": transition_has_progress,
        "transition_has_stable_change": transition_has_stable_change,
        "transition_outcome": transition_outcome,
        "transition_repeats_nonprogress_action": transition_repeats_nonprogress_action,
        "transition_requires_replan": transition_requires_replan,
        "stable_transition_evidence_ready": stable_transition_evidence_ready,
    }
)
