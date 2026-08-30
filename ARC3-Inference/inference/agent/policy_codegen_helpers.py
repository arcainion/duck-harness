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
        "action_payload": action_payload,
        "board_digest": board_digest,
        "consecutive_outcome_count": consecutive_outcome_count,
        "continue_decision": continue_decision,
        "history_push": history_push,
        "least_tried_action": least_tried_action,
        "memory_increment": memory_increment,
        "memory_push": memory_push,
        "memory_update": memory_update,
        "mouse_decision": mouse_decision,
        "path_decision": path_decision,
        "subgoal_failed": subgoal_failed,
        "subgoal_succeeded": subgoal_succeeded,
        "recent_action_counts": recent_action_counts,
        "recent_outcome_counts": recent_outcome_counts,
        "transition_has_progress": transition_has_progress,
        "transition_outcome": transition_outcome,
        "transition_repeats_nonprogress_action": transition_repeats_nonprogress_action,
        "transition_requires_replan": transition_requires_replan,
    }
)
