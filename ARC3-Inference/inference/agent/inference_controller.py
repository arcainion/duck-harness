"""Deterministic per-run experience summaries for the tool-using agent."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from inference.agent.runtime_state import Frame, HistoryEntry


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class InferenceControllerConfig:
    enabled: bool = False
    same_state_noop_limit: int = 2
    stagnation_window: int = 12
    cycle_window: int = 8
    recent_transition_limit: int = 8

    @classmethod
    def from_env(cls) -> "InferenceControllerConfig":
        return cls(
            enabled=_env_bool("LOCAL_ANALYZER_STRATEGY_ENABLED", False),
            same_state_noop_limit=max(
                1, _env_int("LOCAL_ANALYZER_SAME_STATE_NOOP_LIMIT", 2)
            ),
            stagnation_window=max(
                2, _env_int("LOCAL_ANALYZER_STAGNATION_WINDOW", 12)
            ),
            cycle_window=max(2, _env_int("LOCAL_ANALYZER_CYCLE_WINDOW", 8)),
            recent_transition_limit=8,
        )


@lru_cache(maxsize=2_048)
def _grid_fingerprint(level: int, grid: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.blake2b(digest_size=8)
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    digest.update(f"level={level};shape={rows}x{cols};".encode())
    for row in grid:
        digest.update(len(row).to_bytes(4, "big", signed=False))
        for cell in row:
            digest.update(int(cell).to_bytes(4, "big", signed=True))
    return digest.hexdigest()


def frame_fingerprint(frame: Frame | None) -> str:
    """Return a stable opaque identifier without exposing raw grid values."""
    if frame is None:
        return "none"
    return _grid_fingerprint(frame.level, frame.grid)


def normalize_action_key(action: str) -> str:
    return " ".join(str(action or "").strip().upper().split())


def _transitions(history: list[HistoryEntry]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for index in range(1, len(history)):
        before = history[index - 1].frame
        entry = history[index]
        action = normalize_action_key(entry.action)
        if not action:
            continue
        before_id = frame_fingerprint(before)
        after_id = frame_fingerprint(entry.frame)
        transitions.append(
            {
                "action": action,
                "before_state_id": before_id,
                "after_state_id": after_id,
                "board_changed": before_id != after_id,
                "level_before": before.level,
                "level_after": entry.frame.level,
            }
        )
    return transitions


def _cycle_period(state_ids: list[str], max_period: int) -> int | None:
    if len(state_ids) < 3:
        return None
    max_period = min(max_period, len(state_ids) // 2)
    for period in range(1, max_period + 1):
        if state_ids[-period:] == state_ids[-2 * period : -period]:
            return period
    return None


def _stagnation_count(transitions: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    stagnant = 0
    for transition in transitions:
        before_id = str(transition["before_state_id"])
        after_id = str(transition["after_state_id"])
        if not seen:
            seen.add(before_id)
        progressed = (
            int(transition["level_after"]) > int(transition["level_before"])
            or after_id not in seen
        )
        seen.add(after_id)
        stagnant = 0 if progressed else stagnant + 1
    return stagnant


def _no_op_streak(transitions: list[dict[str, Any]]) -> int:
    streak = 0
    for transition in reversed(transitions):
        if transition["board_changed"]:
            break
        streak += 1
    return streak


def action_noop_trials(
    history: list[HistoryEntry], current_frame: Frame | None, action: str
) -> int:
    state_id = frame_fingerprint(current_frame)
    action_key = normalize_action_key(action)
    return sum(
        1
        for transition in _transitions(history)
        if transition["before_state_id"] == state_id
        and transition["after_state_id"] == state_id
        and transition["action"] == action_key
    )


def action_guard_reason(
    history: list[HistoryEntry],
    current_frame: Frame | None,
    action: str,
    config: InferenceControllerConfig,
) -> str | None:
    if not config.enabled:
        return None
    trials = action_noop_trials(history, current_frame, action)
    if trials >= config.same_state_noop_limit:
        return (
            f"exact state/action pair already produced {trials} confirmed no-op trials"
        )
    return None


def build_experience_snapshot(
    history: list[HistoryEntry],
    current_frame: Frame | None,
    valid_actions: Iterable[str],
    config: InferenceControllerConfig,
) -> dict[str, Any]:
    transitions = _transitions(history)
    current_id = frame_fingerprint(current_frame)
    state_ids = [frame_fingerprint(history[0].frame)] if history else []
    state_ids.extend(str(item["after_state_id"]) for item in transitions)
    unique_states = set(state_ids)
    visits = sum(state_id == current_id for state_id in state_ids)
    no_op_streak = _no_op_streak(transitions)
    stagnation = _stagnation_count(transitions)
    cycle_period = _cycle_period(state_ids, config.cycle_window)

    tried: dict[str, dict[str, int]] = {}
    for item in transitions:
        if item["before_state_id"] != current_id:
            continue
        stats = tried.setdefault(
            str(item["action"]), {"trials": 0, "changes": 0, "no_ops": 0}
        )
        stats["trials"] += 1
        if item["board_changed"]:
            stats["changes"] += 1
        else:
            stats["no_ops"] += 1

    normalized_valid = [normalize_action_key(action) for action in valid_actions if str(action).strip()]
    untried = [action for action in normalized_valid if action not in tried]
    useful = [
        action
        for action in normalized_valid
        if tried.get(action, {}).get("changes", 0) > 0
    ]
    discouraged = [
        action
        for action in normalized_valid
        if tried.get(action, {}).get("no_ops", 0) >= config.same_state_noop_limit
    ]
    suggested = [*untried, *[a for a in useful if a not in untried]]
    suggested = [a for a in suggested if a not in discouraged][:6]

    current_level_transitions = [
        item
        for item in transitions
        if current_frame is not None and item["level_after"] == current_frame.level
    ]
    if not current_level_transitions:
        phase = "orient"
    elif (
        no_op_streak >= config.same_state_noop_limit
        or cycle_period is not None
        or stagnation >= config.stagnation_window
    ):
        phase = "recover"
    elif transitions[-1]["level_after"] > transitions[-1]["level_before"] or (
        transitions[-1]["after_state_id"]
        not in {item["before_state_id"] for item in transitions[:-1]}
    ):
        phase = "progress"
    else:
        phase = "explore"

    recent = transitions[-config.recent_transition_limit :]
    return {
        "enabled": config.enabled,
        "phase": phase,
        "state_id": current_id,
        "state_visits": visits,
        "unique_states": len(unique_states),
        "actions_observed": len(transitions),
        "no_op_actions": sum(not item["board_changed"] for item in transitions),
        "no_op_streak": no_op_streak,
        "stagnation_actions": stagnation,
        "cycle_period": cycle_period,
        "tried_here": tried,
        "suggested_actions": suggested,
        "discouraged_actions": discouraged,
        "recent_transitions": recent,
    }


def transition_metadata(
    before: Frame,
    after: Frame,
    prior_history: list[HistoryEntry],
    action: str,
    config: InferenceControllerConfig,
) -> dict[str, Any]:
    before_id = frame_fingerprint(before)
    after_id = frame_fingerprint(after)
    known_states = {frame_fingerprint(entry.frame) for entry in prior_history}
    provisional_history = [*prior_history, HistoryEntry(action=action, frame=after)]
    snapshot = build_experience_snapshot(provisional_history, after, (), config)
    return {
        "before_state_id": before_id,
        "after_state_id": after_id,
        "novel_state": after_id not in known_states,
        "loop_detected": snapshot["cycle_period"] is not None,
        "cycle_period": snapshot["cycle_period"],
        "controller_phase": snapshot["phase"],
        "no_op_streak": snapshot["no_op_streak"],
        "stagnation_actions": snapshot["stagnation_actions"],
    }
