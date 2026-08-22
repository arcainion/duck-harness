"""Deterministic per-run experience summaries for the tool-using agent."""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Any

from inference.agent.runtime_state import Frame, HistoryEntry

LEGACY_POLICY = "legacy"
OUTCOME_AWARE_POLICY = "outcome_aware"
VALID_POLICIES = frozenset({LEGACY_POLICY, OUTCOME_AWARE_POLICY})


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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _normalize_policy(value: Any) -> str:
    policy = str(value or "").strip().lower().replace("-", "_")
    return policy if policy in VALID_POLICIES else LEGACY_POLICY


@dataclass(frozen=True)
class InferenceControllerConfig:
    enabled: bool = True
    policy: str = OUTCOME_AWARE_POLICY
    same_state_noop_limit: int = 2
    stagnation_window: int = 12
    cycle_window: int = 8
    recent_transition_limit: int = 8
    volatile_window: int = 8
    volatile_min_samples: int = 4
    volatile_ratio: float = 0.75
    orient_action_budget: int = 1
    explore_action_budget: int = 1
    recover_action_budget: int = 1
    progress_action_budget: int = 4
    plan_min_support: int = 2
    plan_min_confidence: float = 0.75
    plan_max_depth: int = 6

    @property
    def outcome_aware(self) -> bool:
        return self.enabled and _normalize_policy(self.policy) == OUTCOME_AWARE_POLICY

    @classmethod
    def from_env(cls) -> InferenceControllerConfig:
        return cls(
            enabled=_env_bool("LOCAL_ANALYZER_STRATEGY_ENABLED", True),
            policy=_normalize_policy(os.environ.get("LOCAL_ANALYZER_STRATEGY_POLICY", OUTCOME_AWARE_POLICY)),
            same_state_noop_limit=max(1, _env_int("LOCAL_ANALYZER_SAME_STATE_NOOP_LIMIT", 2)),
            stagnation_window=max(2, _env_int("LOCAL_ANALYZER_STAGNATION_WINDOW", 12)),
            cycle_window=max(2, _env_int("LOCAL_ANALYZER_CYCLE_WINDOW", 8)),
            recent_transition_limit=8,
            volatile_window=max(2, _env_int("LOCAL_ANALYZER_VOLATILE_WINDOW", 8)),
            volatile_min_samples=max(2, _env_int("LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES", 4)),
            volatile_ratio=max(0.5, min(1.0, _env_float("LOCAL_ANALYZER_VOLATILE_RATIO", 0.75))),
            orient_action_budget=max(1, min(12, _env_int("LOCAL_ANALYZER_ORIENT_ACTION_BUDGET", 1))),
            explore_action_budget=max(1, min(12, _env_int("LOCAL_ANALYZER_EXPLORE_ACTION_BUDGET", 1))),
            recover_action_budget=max(1, min(12, _env_int("LOCAL_ANALYZER_RECOVER_ACTION_BUDGET", 1))),
            progress_action_budget=max(1, min(12, _env_int("LOCAL_ANALYZER_PROGRESS_ACTION_BUDGET", 4))),
            plan_min_support=max(1, _env_int("LOCAL_ANALYZER_PLAN_MIN_SUPPORT", 2)),
            plan_min_confidence=max(
                0.5, min(1.0, _env_float("LOCAL_ANALYZER_PLAN_MIN_CONFIDENCE", 0.75))
            ),
            plan_max_depth=max(1, min(12, _env_int("LOCAL_ANALYZER_PLAN_MAX_DEPTH", 6))),
        )


@lru_cache(maxsize=2_048)
def _grid_fingerprint(level: int, grid: tuple[tuple[int, ...], ...]) -> str:
    return _masked_grid_fingerprint(level, grid, frozenset())


def _masked_grid_fingerprint(
    level: int,
    grid: tuple[tuple[int, ...], ...],
    masked_cells: frozenset[tuple[int, int]],
) -> str:
    digest = hashlib.blake2b(digest_size=8)
    rows = len(grid)
    cols = max((len(row) for row in grid), default=0)
    digest.update(f"level={level};shape={rows}x{cols};".encode())
    for row_index, row in enumerate(grid):
        digest.update(len(row).to_bytes(4, "big", signed=False))
        for column_index, cell in enumerate(row):
            if (row_index, column_index) in masked_cells:
                digest.update(b"volatile;")
            else:
                digest.update(int(cell).to_bytes(4, "big", signed=True))
    return digest.hexdigest()


def frame_fingerprint(frame: Frame | None) -> str:
    """Return a stable opaque identifier without exposing raw grid values."""
    if frame is None:
        return "none"
    return _grid_fingerprint(frame.level, frame.grid)


def _behavioral_fingerprint(
    frame: Frame | None, masked_cells: frozenset[tuple[int, int]]
) -> str:
    if frame is None:
        return "none"
    return _masked_grid_fingerprint(frame.level, frame.grid, masked_cells)


def _volatile_cells(
    history: list[HistoryEntry],
    current_frame: Frame | None,
    config: InferenceControllerConfig,
) -> frozenset[tuple[int, int]]:
    if not config.outcome_aware or current_frame is None:
        return frozenset()
    level_frames = [entry.frame for entry in history if entry.frame.level == current_frame.level]
    if not level_frames or frame_fingerprint(level_frames[-1]) != frame_fingerprint(current_frame):
        level_frames.append(current_frame)
    level_frames = level_frames[-(config.volatile_window + 1) :]
    sample_count = len(level_frames) - 1
    if sample_count < config.volatile_min_samples:
        return frozenset()
    shape = current_frame.shape
    if any(frame.shape != shape for frame in level_frames):
        return frozenset()
    changes: Counter[tuple[int, int]] = Counter()
    for before, after in pairwise(level_frames):
        for row_index, (before_row, after_row) in enumerate(zip(before.grid, after.grid)):
            for column_index, (before_cell, after_cell) in enumerate(zip(before_row, after_row)):
                if before_cell != after_cell:
                    changes[(row_index, column_index)] += 1
    threshold = math.ceil(sample_count * config.volatile_ratio)
    return frozenset(cell for cell, count in changes.items() if count >= threshold)


def normalize_action_key(action: str) -> str:
    return " ".join(str(action or "").strip().upper().split())


def action_family(action: str) -> str:
    key = normalize_action_key(action)
    return "MOUSE" if key.startswith("MOUSE") else key


_MOUSE_COORDINATE_RE = re.compile(
    r"^MOUSE\s*\(\s*ROW\s*=\s*(-?\d+)\s*,\s*COL\s*=\s*(-?\d+)\s*\)$"
)


def action_coordinate(action: str) -> tuple[int, int] | None:
    """Return the exact model-facing mouse coordinate, when present."""
    match = _MOUSE_COORDINATE_RE.match(normalize_action_key(action))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _mouse_search_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    observations = []
    outcome_counts: Counter[str] = Counter()
    for item in transitions:
        coordinate = action_coordinate(str(item.get("action") or ""))
        if coordinate is None:
            continue
        outcome = str(item.get("outcome_class") or "unknown")
        outcome_counts[outcome] += 1
        observations.append(
            {
                "row": coordinate[0],
                "col": coordinate[1],
                "outcome": outcome,
                "changed": bool(item.get("behavioral_changed")),
            }
        )
    unique_coordinates = {(item["row"], item["col"]) for item in observations}
    return {
        "trials": len(observations),
        "unique_coordinates": len(unique_coordinates),
        "outcomes": dict(sorted(outcome_counts.items())),
        "recent": observations[-16:],
    }


def _plan_candidates(
    transitions: list[dict[str, Any]],
    current_behavioral_id: str,
    *,
    max_depth: int = 6,
    min_support: int = 2,
    min_confidence: float = 0.75,
    current_state_id: str = "",
) -> list[dict[str, Any]]:
    """Find short progress routes supported by repeatable transition evidence."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in transitions:
        action = normalize_action_key(raw.get("action") or raw.get("action_display") or "")
        pairs = {
            (
                str(raw.get("behavioral_before_state_id") or ""),
                str(raw.get("behavioral_after_state_id") or ""),
            ),
            (
                str(raw.get("before_state_id") or ""),
                str(raw.get("after_state_id") or ""),
            ),
        }
        for before, after in pairs:
            if before and after and action:
                sample = dict(raw)
                sample["planner_after_state_id"] = after
                grouped.setdefault((before, action), []).append(sample)

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for (before, action), samples in grouped.items():
        outcomes = Counter(
            (
                str(item.get("planner_after_state_id") or ""),
                str(item.get("outcome_class") or "unknown") == "level_progress",
            )
            for item in samples
        )
        (after, progresses), support = min(outcomes.items(), key=lambda item: (-item[1], item[0]))
        matching_outcomes = Counter(
            str(item.get("outcome_class") or "unknown")
            for item in samples
            if str(item.get("planner_after_state_id") or "") == after
        )
        outcome = "level_progress" if progresses else min(
            matching_outcomes.items(), key=lambda item: (-item[1], item[0])
        )[0]
        trials = len(samples)
        confidence = support / trials
        if (
            support < min_support
            or confidence < min_confidence
            or outcome in {"exact_noop", "volatile_only"}
        ):
            continue
        adjacency.setdefault(before, []).append({
            "action": action,
            "behavioral_after_state_id": after,
            "outcome_class": outcome,
            "support": support,
            "contradictions": trials - support,
            "confidence": confidence,
        })
    for edges in adjacency.values():
        edges.sort(key=lambda edge: (-float(edge["confidence"]), -int(edge["support"]), str(edge["action"])))

    start_ids = {current_behavioral_id, current_state_id} - {""}
    queue = deque((state_id, [], 1.0, 0) for state_id in sorted(start_ids))
    best_confidence = {state_id: 1.0 for state_id in start_ids}
    plans: list[dict[str, Any]] = []
    while queue and len(plans) < 4:
        state_id, path, path_confidence, contradictions = queue.popleft()
        if len(path) >= max_depth:
            continue
        for edge in adjacency.get(state_id, []):
            next_path = [*path, str(edge["action"])]
            next_confidence = path_confidence * float(edge["confidence"])
            next_contradictions = contradictions + int(edge["contradictions"])
            if edge.get("outcome_class") == "level_progress":
                plans.append(
                    {
                        "actions": next_path,
                        "target": "level_progress",
                        "verified_steps": len(next_path),
                        "confidence": round(next_confidence, 3),
                        "support": int(edge["support"]),
                        "contradictions": next_contradictions,
                    }
                )
                continue
            next_state = str(edge["behavioral_after_state_id"])
            if best_confidence.get(next_state, -1.0) >= next_confidence:
                continue
            best_confidence[next_state] = next_confidence
            queue.append((next_state, next_path, next_confidence, next_contradictions))
    plans.sort(key=lambda item: (-item["confidence"], len(item["actions"]), item["actions"]))
    return plans


def _transitions(
    history: list[HistoryEntry],
    masked_cells: frozenset[tuple[int, int]] = frozenset(),
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    known_behavioral = ({_behavioral_fingerprint(history[0].frame, masked_cells)} if history else set())
    for index in range(1, len(history)):
        before = history[index - 1].frame
        entry = history[index]
        action = normalize_action_key(entry.action)
        if not action:
            continue
        before_id = frame_fingerprint(before)
        after_id = frame_fingerprint(entry.frame)
        behavioral_before = _behavioral_fingerprint(before, masked_cells)
        behavioral_after = _behavioral_fingerprint(entry.frame, masked_cells)
        if entry.frame.level > before.level:
            outcome = "level_progress"
        elif before_id == after_id:
            outcome = "exact_noop"
        elif behavioral_before == behavioral_after:
            outcome = "volatile_only"
        elif behavioral_after not in known_behavioral:
            outcome = "novel"
        else:
            outcome = "revisit"
        transitions.append({
            "action": action,
            "action_family": action_family(action),
            "before_state_id": before_id,
            "after_state_id": after_id,
            "behavioral_before_state_id": behavioral_before,
            "behavioral_after_state_id": behavioral_after,
            "board_changed": before_id != after_id,
            "behavioral_changed": behavioral_before != behavioral_after,
            "outcome_class": outcome,
            "level_before": before.level,
            "level_after": entry.frame.level,
        })
        known_behavioral.update((behavioral_before, behavioral_after))
    return transitions


def _cycle_period(state_ids: list[str], max_period: int) -> int | None:
    if len(state_ids) < 3:
        return None
    max_period = min(max_period, len(state_ids) // 2)
    for period in range(1, max_period + 1):
        if state_ids[-period:] == state_ids[-2 * period : -period]:
            return period
    return None


def _stagnation_count(transitions: list[dict[str, Any]], *, behavioral: bool = False) -> int:
    before_key = "behavioral_before_state_id" if behavioral else "before_state_id"
    after_key = "behavioral_after_state_id" if behavioral else "after_state_id"
    seen: set[str] = set()
    stagnant = 0
    for transition in transitions:
        before_id = str(transition[before_key])
        after_id = str(transition[after_key])
        if not seen:
            seen.add(before_id)
        progressed = int(transition["level_after"]) > int(transition["level_before"]) or after_id not in seen
        seen.add(after_id)
        stagnant = 0 if progressed else stagnant + 1
    return stagnant


def _no_op_streak(transitions: list[dict[str, Any]], *, behavioral: bool = False) -> int:
    streak = 0
    for transition in reversed(transitions):
        changed = transition["behavioral_changed"] if behavioral else transition["board_changed"]
        if changed:
            break
        streak += 1
    return streak


def action_noop_trials(history: list[HistoryEntry], current_frame: Frame | None, action: str) -> int:
    state_id = frame_fingerprint(current_frame)
    action_key = normalize_action_key(action)
    return sum(
        1 for transition in _transitions(history)
        if transition["before_state_id"] == state_id
        and transition["after_state_id"] == state_id
        and transition["action"] == action_key
    )


def action_guard_reason_code(
    history: list[HistoryEntry], current_frame: Frame | None, action: str, config: InferenceControllerConfig
) -> str | None:
    if not config.enabled:
        return None
    if action_noop_trials(history, current_frame, action) >= config.same_state_noop_limit:
        return "repeated_exact_noop"
    return None


def action_guard_reason(
    history: list[HistoryEntry], current_frame: Frame | None, action: str, config: InferenceControllerConfig
) -> str | None:
    if action_guard_reason_code(history, current_frame, action, config) is None:
        return None
    trials = action_noop_trials(history, current_frame, action)
    return f"exact state/action pair already produced {trials} confirmed no-op trials"


def _rank_actions(
    transitions: list[dict[str, Any]], current_behavioral_id: str, valid_actions: list[str], config: InferenceControllerConfig
) -> list[dict[str, Any]]:
    local: dict[str, Counter[str]] = {}
    global_progress: Counter[str] = Counter()
    recent_behavioral = {
        str(item["behavioral_after_state_id"]) for item in transitions[-config.cycle_window :]
    }
    cycle_returns: Counter[str] = Counter()
    for item in transitions:
        family = str(item["action_family"])
        if item["outcome_class"] == "level_progress":
            global_progress[family] += 1
        if item["behavioral_before_state_id"] != current_behavioral_id:
            continue
        stats = local.setdefault(family, Counter())
        stats["trials"] += 1
        stats[str(item["outcome_class"])] += 1
        if item["behavioral_after_state_id"] in recent_behavioral:
            cycle_returns[family] += 1

    ranked: list[tuple[tuple[int, int, int, int, int], dict[str, Any]]] = []
    for valid_index, action in enumerate(valid_actions):
        family = action_family(action)
        stats = local.get(family, Counter())
        trials = int(stats["trials"])
        progress = int(stats["level_progress"])
        novel = int(stats["novel"])
        parameterized = family == "MOUSE"
        cycle_risk = int(cycle_returns[family])
        noops = int(stats["exact_noop"] + stats["volatile_only"])
        if progress:
            priority, reason = 0, "confirmed level progress from this state"
        elif trials == 0:
            priority, reason = 1, "untried action from this state"
        elif parameterized:
            priority, reason = 1, "parameterized search; choose an untried coordinate"
        elif novel and not cycle_risk:
            priority, reason = 2, "previously reached a novel behavioral state"
        elif progress + novel + int(stats["revisit"]):
            priority, reason = 3, "changes state but may revisit known territory"
        else:
            priority, reason = 4, "previous trials were no-op or cycle-prone"
        payload = {
            "action": family,
            "priority": priority,
            "trials": trials,
            "level_progress": progress,
            "novel": novel,
            "revisits": int(stats["revisit"]),
            "no_ops": noops,
            "cycle_risk": bool(cycle_risk),
            "parameterized": parameterized,
            "reason": reason,
        }
        ranked.append(((priority, trials, -int(global_progress[family]), -novel, valid_index), payload))
    ranked.sort(key=lambda item: item[0])
    return [payload for _, payload in ranked[:6]]


def _transition_models_here(
    transitions: list[dict[str, Any]], current_behavioral_id: str
) -> list[dict[str, Any]]:
    """Return compact empirical action models verified at the current state."""
    observations: dict[str, dict[str, Counter[str]]] = {}
    for item in transitions:
        if item["behavioral_before_state_id"] != current_behavioral_id:
            continue
        action = str(item["action"])
        stats = observations.setdefault(
            action, {"next_states": Counter(), "outcomes": Counter()}
        )
        stats["next_states"][str(item["behavioral_after_state_id"])] += 1
        stats["outcomes"][str(item["outcome_class"])] += 1

    models: list[dict[str, Any]] = []
    for action, stats in observations.items():
        next_states = stats["next_states"]
        outcomes = stats["outcomes"]
        trials = sum(next_states.values())
        next_state, support = min(
            next_states.items(),
            key=lambda item: (-item[1], item[0]),
        )
        outcome = min(
            outcomes.items(), key=lambda item: (-item[1], item[0])
        )[0]
        deterministic = len(next_states) == 1
        models.append(
            {
                "action": action,
                "trials": trials,
                "predicted_outcome": outcome,
                "predicted_behavioral_state_id": next_state if deterministic else None,
                "support": support,
                "contradictions": trials - support,
                "confidence": round(support / trials, 3) if trials else 0.0,
                "verified_deterministic": deterministic,
            }
        )
    models.sort(
        key=lambda model: (
            not bool(model["verified_deterministic"]),
            -int(model["trials"]),
            str(model["action"]),
        )
    )
    return models[:6]


def build_experience_snapshot(
    history: list[HistoryEntry], current_frame: Frame | None, valid_actions: Iterable[str], config: InferenceControllerConfig,
    external_transitions: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    masked_cells = _volatile_cells(history, current_frame, config)
    transitions = _transitions(history, masked_cells)
    current_id = frame_fingerprint(current_frame)
    behavioral_id = _behavioral_fingerprint(current_frame, masked_cells)
    state_ids = ([frame_fingerprint(history[0].frame)] if history else []) + [
        str(item["after_state_id"]) for item in transitions
    ]
    behavioral_ids = ([_behavioral_fingerprint(history[0].frame, masked_cells)] if history else []) + [
        str(item["behavioral_after_state_id"]) for item in transitions
    ]
    active_ids = behavioral_ids if config.outcome_aware else state_ids
    active_current_id = behavioral_id if config.outcome_aware else current_id
    no_op_streak = _no_op_streak(transitions)
    behavioral_no_op_streak = _no_op_streak(transitions, behavioral=True)
    stagnation = _stagnation_count(transitions, behavioral=config.outcome_aware)
    cycle_period = _cycle_period(active_ids, config.cycle_window)

    tried: dict[str, dict[str, int]] = {}
    for item in transitions:
        comparison_id = item["behavioral_before_state_id"] if config.outcome_aware else item["before_state_id"]
        if comparison_id != active_current_id:
            continue
        stats = tried.setdefault(str(item["action"]), {"trials": 0, "changes": 0, "no_ops": 0})
        stats["trials"] += 1
        stats["changes" if item["board_changed"] else "no_ops"] += 1

    normalized_valid = [normalize_action_key(action) for action in valid_actions if str(action).strip()]
    untried = [action for action in normalized_valid if action not in tried]
    useful = [action for action in normalized_valid if tried.get(action, {}).get("changes", 0) > 0]
    discouraged = [
        action for action in normalized_valid
        if tried.get(action, {}).get("no_ops", 0) >= config.same_state_noop_limit
    ]
    suggested = [action for action in [*untried, *[a for a in useful if a not in untried]] if action not in discouraged][:6]
    current_level_transitions = [
        item for item in transitions if current_frame is not None and item["level_after"] == current_frame.level
    ]
    recovery_reasons: list[str] = []
    active_noop_streak = behavioral_no_op_streak if config.outcome_aware else no_op_streak
    if active_noop_streak >= config.same_state_noop_limit:
        recovery_reasons.append("repeated_noop")
    if cycle_period is not None:
        recovery_reasons.append("short_cycle")
    if stagnation >= config.stagnation_window:
        recovery_reasons.append("stagnation")
    latest_outcome = transitions[-1]["outcome_class"] if transitions else None
    if not current_level_transitions:
        phase = "orient"
    elif recovery_reasons:
        phase = "recover"
    elif (
        config.outcome_aware and latest_outcome in {"level_progress", "novel"}
    ) or (
        not config.outcome_aware
        and (
            transitions[-1]["level_after"] > transitions[-1]["level_before"]
            or transitions[-1]["after_state_id"]
            not in {item["before_state_id"] for item in transitions[:-1]}
        )
    ):
        phase = "progress"
    else:
        phase = "explore"
    ranked_actions = _rank_actions(transitions, behavioral_id, normalized_valid, config) if config.outcome_aware else []
    transition_models = (
        _transition_models_here(transitions, behavioral_id)
        if config.outcome_aware
        else []
    )
    model_conflicts = sum(
        int(model["contradictions"] > 0) for model in transition_models
    )
    if model_conflicts and "transition_model_conflict" not in recovery_reasons:
        recovery_reasons.append("transition_model_conflict")
        phase = "recover"
    action_budget = {
        "orient": config.orient_action_budget,
        "explore": config.explore_action_budget,
        "recover": config.recover_action_budget,
        "progress": config.progress_action_budget,
    }.get(phase, 1)
    plan_candidates = (
        _plan_candidates(
            [*transitions, *(external_transitions or ())],
            behavioral_id,
            max_depth=config.plan_max_depth,
            min_support=config.plan_min_support,
            min_confidence=config.plan_min_confidence,
            current_state_id=current_id,
        ) if config.outcome_aware else []
    )
    return {
        "enabled": config.enabled,
        "policy": _normalize_policy(config.policy),
        "phase": phase,
        "action_budget": action_budget,
        "state_id": current_id,
        "behavioral_state_id": behavioral_id,
        "state_visits": sum(state_id == active_current_id for state_id in active_ids),
        "unique_states": len(set(state_ids)),
        "unique_behavioral_states": len(set(behavioral_ids)),
        "volatile_cells": len(masked_cells),
        "actions_observed": len(transitions),
        "no_op_actions": sum(not item["board_changed"] for item in transitions),
        "no_op_streak": no_op_streak,
        "behavioral_no_op_streak": behavioral_no_op_streak,
        "stagnation_actions": stagnation,
        "cycle_period": cycle_period,
        "latest_outcome": latest_outcome,
        "recovery_reasons": recovery_reasons,
        "tried_here": tried,
        "suggested_actions": suggested,
        "discouraged_actions": discouraged,
        "ranked_actions": ranked_actions,
        "mouse_search": _mouse_search_summary(transitions),
        "plan_candidates": plan_candidates,
        "recommended_plan": plan_candidates[0] if plan_candidates else None,
        "transition_models_here": transition_models,
        "model_conflicts_here": model_conflicts,
        "recent_transitions": transitions[-config.recent_transition_limit :],
    }


def transition_metadata(
    before: Frame,
    after: Frame,
    prior_history: list[HistoryEntry],
    action: str,
    config: InferenceControllerConfig,
    valid_actions: Iterable[str] | None = None,
) -> dict[str, Any]:
    before_id = frame_fingerprint(before)
    after_id = frame_fingerprint(after)
    known_states = {frame_fingerprint(entry.frame) for entry in prior_history}
    action_key = action_family(action)
    before_snapshot = build_experience_snapshot(
        prior_history,
        before,
        valid_actions if valid_actions is not None else [action_key],
        config,
    )
    provisional_history = [*prior_history, HistoryEntry(action=action, frame=after)]
    snapshot = build_experience_snapshot(provisional_history, after, [action_family(action)], config)
    transition = snapshot["recent_transitions"][-1]
    ranking = before_snapshot["ranked_actions"]
    ranked_action = next(
        (
            (index, item)
            for index, item in enumerate(ranking, start=1)
            if item["action"] == action_key
        ),
        None,
    )
    return {
        "before_state_id": before_id,
        "after_state_id": after_id,
        "behavioral_before_state_id": transition["behavioral_before_state_id"],
        "behavioral_after_state_id": transition["behavioral_after_state_id"],
        "novel_state": after_id not in known_states,
        "outcome_class": transition["outcome_class"],
        "loop_detected": snapshot["cycle_period"] is not None,
        "cycle_risk": snapshot["cycle_period"] is not None,
        "cycle_period": snapshot["cycle_period"],
        "controller_policy": snapshot["policy"],
        "controller_phase": snapshot["phase"],
        "controller_reason_codes": list(snapshot["recovery_reasons"]),
        "action_rank": ranked_action[0] if ranked_action is not None else None,
        "action_rank_reason": ranked_action[1]["reason"] if ranked_action is not None else None,
        "no_op_streak": snapshot["no_op_streak"],
        "behavioral_no_op_streak": snapshot["behavioral_no_op_streak"],
        "stagnation_actions": snapshot["stagnation_actions"],
    }
