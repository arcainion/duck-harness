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
        value = float(os.environ.get(name, "").strip() or default)
        return value if math.isfinite(value) else default
    except (ValueError, OverflowError):
        return default


def _transition_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _transition_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return False


def _transition_reward(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        reward = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return reward if math.isfinite(reward) else 0.0


def _transition_observation_count(value: Any) -> int:
    if isinstance(value, bool):
        return 1
    try:
        return max(1, min(1_000_000, int(value or 1)))
    except (TypeError, ValueError, OverflowError):
        return 1


def _transition_action_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    raw_values = [raw for raw in value if isinstance(raw, str)]
    if isinstance(value, (set, frozenset)):
        raw_values.sort()
    actions: list[str] = []
    for raw in raw_values[:16]:
        text = _transition_text(raw, 80)
        action = normalize_action_key(text) if text else ""
        if action and action not in actions:
            actions.append(action)
    return actions


def _normalize_external_transition(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    item = dict(raw)
    raw_action = _transition_text(
        item.get("action") or item.get("action_display"), 80
    )
    item["action"] = normalize_action_key(raw_action) if raw_action else ""
    if not item["action"]:
        return None
    item["action_family"] = action_family(item["action"])
    item["before_state_id"] = _transition_text(item.get("before_state_id"), 256)
    item["after_state_id"] = _transition_text(item.get("after_state_id"), 256)
    item["behavioral_before_state_id"] = _transition_text(
        item.get("behavioral_before_state_id"), 256
    ) or item["before_state_id"]
    item["behavioral_after_state_id"] = _transition_text(
        item.get("behavioral_after_state_id"), 256
    ) or item["after_state_id"]
    item["object_before_state_id"] = _transition_text(
        item.get("object_before_state_id"), 256
    )
    item["object_after_state_id"] = _transition_text(
        item.get("object_after_state_id"), 256
    )
    item["evidence_id"] = _transition_text(item.get("evidence_id"), 128)
    item["outcome_class"] = (
        _transition_text(item.get("outcome_class"), 40) or "unknown"
    )
    item["reward"] = _transition_reward(item.get("reward"))
    item["raw_observations"] = _transition_observation_count(
        item.get("raw_observations")
    )
    item["valid_actions_after"] = _transition_action_list(
        item.get("valid_actions_after")
    )
    item["board_changed"] = _transition_bool(item.get("board_changed"))
    item["decision_context_changed"] = _transition_bool(
        item.get("decision_context_changed")
    )
    item["game_over"] = _transition_bool(item.get("game_over"))
    item["behavioral_changed"] = (
        item["behavioral_before_state_id"] != item["behavioral_after_state_id"]
    )
    return item


def _normalize_policy(value: Any) -> str:
    policy = str(value or "").strip().lower().replace("-", "_")
    return policy if policy in VALID_POLICIES else LEGACY_POLICY


@dataclass(frozen=True)
class InferenceControllerConfig:
    enabled: bool = True
    policy: str = OUTCOME_AWARE_POLICY
    same_state_noop_limit: int = 2
    stagnation_window: int = 6
    cycle_window: int = 4
    cycle_stop_limit: int = 0
    repeat_action_limit: int = 0
    directional_no_progress_window: int = 0
    directional_no_progress_limit: int = 0
    directional_no_progress_strike_limit: int = 0
    directional_no_progress_stop_limit: int = 0
    ignore_edge_hud_changes: bool = False
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
    plan_max_terminal_risk: float = 0.25
    credit_horizon: int = 4
    credit_discount: float = 0.8
    progress_utility: float = 1.0
    novel_utility: float = 0.2
    revisit_utility: float = -0.05
    noop_utility: float = -0.4
    terminal_failure_utility: float = -2.0
    exploration_weight: float = 0.75
    level_action_limit_multiplier: float = 0.0
    level_action_limit_minimum: int = 0
    level_no_progress_token_limit: int = 0

    @property
    def outcome_aware(self) -> bool:
        return self.enabled and _normalize_policy(self.policy) == OUTCOME_AWARE_POLICY

    @classmethod
    def from_env(cls) -> InferenceControllerConfig:
        return cls(
            enabled=_env_bool("LOCAL_ANALYZER_STRATEGY_ENABLED", True),
            policy=_normalize_policy(
                os.environ.get("LOCAL_ANALYZER_STRATEGY_POLICY", OUTCOME_AWARE_POLICY)
            ),
            same_state_noop_limit=max(
                1, _env_int("LOCAL_ANALYZER_SAME_STATE_NOOP_LIMIT", 2)
            ),
            stagnation_window=max(2, _env_int("LOCAL_ANALYZER_STAGNATION_WINDOW", 12)),
            cycle_window=max(2, _env_int("LOCAL_ANALYZER_CYCLE_WINDOW", 8)),
            cycle_stop_limit=max(
                0, _env_int("LOCAL_ANALYZER_CYCLE_STOP_LIMIT", 0)
            ),
            repeat_action_limit=max(
                0, _env_int("LOCAL_ANALYZER_REPEAT_ACTION_LIMIT", 0)
            ),
            directional_no_progress_window=max(
                0, _env_int("LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_WINDOW", 0)
            ),
            directional_no_progress_limit=max(
                0, _env_int("LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_LIMIT", 0)
            ),
            directional_no_progress_strike_limit=max(
                0,
                _env_int("LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STRIKE_LIMIT", 0),
            ),
            directional_no_progress_stop_limit=max(
                0,
                _env_int("LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STOP_LIMIT", 0),
            ),
            ignore_edge_hud_changes=_env_bool(
                "LOCAL_ANALYZER_IGNORE_EDGE_HUD_CHANGES", False
            ),
            recent_transition_limit=8,
            volatile_window=max(2, _env_int("LOCAL_ANALYZER_VOLATILE_WINDOW", 8)),
            volatile_min_samples=max(
                2, _env_int("LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES", 4)
            ),
            volatile_ratio=max(
                0.5, min(1.0, _env_float("LOCAL_ANALYZER_VOLATILE_RATIO", 0.75))
            ),
            orient_action_budget=max(
                1, min(12, _env_int("LOCAL_ANALYZER_ORIENT_ACTION_BUDGET", 1))
            ),
            explore_action_budget=max(
                1, min(12, _env_int("LOCAL_ANALYZER_EXPLORE_ACTION_BUDGET", 1))
            ),
            recover_action_budget=max(
                1, min(12, _env_int("LOCAL_ANALYZER_RECOVER_ACTION_BUDGET", 1))
            ),
            progress_action_budget=max(
                1, min(12, _env_int("LOCAL_ANALYZER_PROGRESS_ACTION_BUDGET", 4))
            ),
            plan_min_support=max(1, _env_int("LOCAL_ANALYZER_PLAN_MIN_SUPPORT", 2)),
            plan_min_confidence=max(
                0.5, min(1.0, _env_float("LOCAL_ANALYZER_PLAN_MIN_CONFIDENCE", 0.75))
            ),
            plan_max_depth=max(
                1, min(12, _env_int("LOCAL_ANALYZER_PLAN_MAX_DEPTH", 6))
            ),
            plan_max_terminal_risk=max(
                0.0,
                min(
                    0.5,
                    _env_float("LOCAL_ANALYZER_PLAN_MAX_TERMINAL_RISK", 0.25),
                ),
            ),
            credit_horizon=max(
                1, min(12, _env_int("LOCAL_ANALYZER_CREDIT_HORIZON", 4))
            ),
            credit_discount=max(
                0.0,
                min(1.0, _env_float("LOCAL_ANALYZER_CREDIT_DISCOUNT", 0.8)),
            ),
            progress_utility=_env_float("LOCAL_ANALYZER_PROGRESS_UTILITY", 1.0),
            novel_utility=_env_float("LOCAL_ANALYZER_NOVEL_UTILITY", 0.2),
            revisit_utility=_env_float("LOCAL_ANALYZER_REVISIT_UTILITY", -0.05),
            noop_utility=_env_float("LOCAL_ANALYZER_NOOP_UTILITY", -0.4),
            terminal_failure_utility=_env_float(
                "LOCAL_ANALYZER_TERMINAL_FAILURE_UTILITY", -2.0
            ),
            exploration_weight=max(
                0.0, _env_float("LOCAL_ANALYZER_EXPLORATION_WEIGHT", 0.75)
            ),
            level_action_limit_multiplier=max(
                0.0,
                _env_float("LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MULTIPLIER", 0.0),
            ),
            level_action_limit_minimum=max(
                0, _env_int("LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM", 0)
            ),
            level_no_progress_token_limit=max(
                0, _env_int("LOCAL_ANALYZER_LEVEL_NO_PROGRESS_TOKEN_LIMIT", 0)
            ),
        )

    def level_action_limit(self, base_actions: int | None) -> int | None:
        """Return the baseline-relative action ceiling, or ``None`` when disabled."""
        if (
            self.level_action_limit_multiplier <= 0
            or base_actions is None
            or base_actions <= 0
        ):
            return None
        return max(
            self.level_action_limit_minimum,
            int(math.ceil(base_actions * self.level_action_limit_multiplier)),
        )

    def outcome_utility(self, outcome: str) -> float:
        return {
            "level_progress": self.progress_utility,
            "novel": self.novel_utility,
            "revisit": self.revisit_utility,
            "volatile_only": self.noop_utility * 0.5,
            "transient_effect": self.novel_utility * 0.5,
            "exact_noop": self.noop_utility,
            "terminal_failure": self.terminal_failure_utility,
            "negative_reward": self.terminal_failure_utility,
        }.get(str(outcome), 0.0)


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
    """Return a stable identifier for the complete observable decision state."""
    if frame is None:
        return "none"
    return _contextual_fingerprint(_grid_fingerprint(frame.level, frame.grid), frame)


def _contextual_fingerprint(grid_id: str, frame: Frame) -> str:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(grid_id.encode())
    digest.update(f";score={frame.score};state={frame.engine_state};".encode())
    for action in sorted(normalize_action_key(item) for item in frame.valid_actions):
        digest.update(f"action={action};".encode())
    return digest.hexdigest()


def _behavioral_fingerprint(
    frame: Frame | None, masked_cells: frozenset[tuple[int, int]]
) -> str:
    if frame is None:
        return "none"
    return _contextual_fingerprint(
        _masked_grid_fingerprint(frame.level, frame.grid, masked_cells), frame
    )


def _volatile_cells(
    history: list[HistoryEntry],
    current_frame: Frame | None,
    config: InferenceControllerConfig,
) -> frozenset[tuple[int, int]]:
    if not config.outcome_aware or current_frame is None:
        return frozenset()
    episode_history = history
    for index in range(len(history) - 1, -1, -1):
        if action_family(history[index].action) == "RESET":
            episode_history = history[index:]
            break
    level_frames = [
        entry.frame
        for entry in episode_history
        if entry.frame.level == current_frame.level
    ]
    if not level_frames or frame_fingerprint(level_frames[-1]) != frame_fingerprint(
        current_frame
    ):
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
        for row_index, (before_row, after_row) in enumerate(
            zip(before.grid, after.grid)
        ):
            for column_index, (before_cell, after_cell) in enumerate(
                zip(before_row, after_row)
            ):
                if before_cell != after_cell:
                    changes[(row_index, column_index)] += 1
    threshold = math.ceil(sample_count * config.volatile_ratio)
    return frozenset(cell for cell, count in changes.items() if count >= threshold)


def normalize_action_key(action: str) -> str:
    return " ".join(str(action or "").strip().upper().split())


def action_family(action: str) -> str:
    key = normalize_action_key(action)
    return "MOUSE" if key == "MOUSE" or action_coordinate(key) is not None else key


_MOUSE_COORDINATE_RE = re.compile(
    r"^MOUSE\s*\(\s*ROW\s*=\s*(\d{1,2})\s*,\s*COL\s*=\s*(\d{1,2})\s*\)$"
)


def action_coordinate(action: str) -> tuple[int, int] | None:
    """Return the exact model-facing mouse coordinate, when present."""
    match = _MOUSE_COORDINATE_RE.match(normalize_action_key(action))
    if match is None:
        return None
    coordinate = int(match.group(1)), int(match.group(2))
    return coordinate if all(0 <= value <= 63 for value in coordinate) else None


def _object_state_summary(frame: Frame | None) -> dict[str, Any]:
    """Extract bounded connected-component features from the visible grid."""
    if frame is None or not frame.grid or not frame.grid[0]:
        return {
            "background": None,
            "object_count": 0,
            "components": [],
            "shape_signature": "none",
            "colored_shape_signature": "none",
            "relational_signature": "none",
        }
    color_counts = Counter(cell for row in frame.grid for cell in row)
    background = min(color_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    rows, cols = frame.shape
    visited: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            color = frame.grid[row][col]
            if color == background or (row, col) in visited:
                continue
            frontier = [(row, col)]
            visited.add((row, col))
            cells: list[tuple[int, int]] = []
            while frontier:
                cell_row, cell_col = frontier.pop()
                cells.append((cell_row, cell_col))
                for next_row, next_col in (
                    (cell_row - 1, cell_col),
                    (cell_row + 1, cell_col),
                    (cell_row, cell_col - 1),
                    (cell_row, cell_col + 1),
                ):
                    if (
                        0 <= next_row < rows
                        and 0 <= next_col < cols
                        and (next_row, next_col) not in visited
                        and frame.grid[next_row][next_col] == color
                    ):
                        visited.add((next_row, next_col))
                        frontier.append((next_row, next_col))
            min_row = min(cell[0] for cell in cells)
            max_row = max(cell[0] for cell in cells)
            min_col = min(cell[1] for cell in cells)
            max_col = max(cell[1] for cell in cells)
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": [min_row, min_col, max_row, max_col],
                    "height": max_row - min_row + 1,
                    "width": max_col - min_col + 1,
                    "centroid": [
                        round(sum(cell[0] for cell in cells) / len(cells), 3),
                        round(sum(cell[1] for cell in cells) / len(cells), 3),
                    ],
                }
            )
    components.sort(
        key=lambda item: (-int(item["size"]), int(item["color"]), item["bbox"])
    )
    shapes = sorted(
        (int(item["size"]), int(item["height"]), int(item["width"]))
        for item in components
    )
    colored_shapes = sorted(
        (
            int(item["color"]),
            int(item["size"]),
            int(item["height"]),
            int(item["width"]),
        )
        for item in components
    )
    anchor_row = min((float(item["centroid"][0]) for item in components), default=0.0)
    anchor_col = min((float(item["centroid"][1]) for item in components), default=0.0)
    relational_layout = sorted(
        (
            int(item["color"]),
            int(item["size"]),
            int(item["height"]),
            int(item["width"]),
            round(float(item["centroid"][0]) - anchor_row, 3),
            round(float(item["centroid"][1]) - anchor_col, 3),
        )
        for item in components
    )

    def signature(value: Any) -> str:
        return hashlib.blake2b(repr(value).encode(), digest_size=8).hexdigest()

    return {
        "background": background,
        "object_count": len(components),
        "components": components[:24],
        "shape_signature": signature(shapes),
        "colored_shape_signature": signature(colored_shapes),
        "relational_signature": signature(relational_layout),
    }


def _object_state_id(frame: Frame | None) -> str:
    if frame is None:
        return "none"
    summary = _object_state_summary(frame)
    if not summary["object_count"]:
        # A background-only board has no object relation to generalize. Giving
        # every such board the same abstract identity would pool unrelated
        # states solely because their modal color differs.
        return ""
    return _contextual_fingerprint(
        f"object:{frame.level}:{summary['relational_signature']}", frame
    )


def _object_temporal_summary(
    history: list[HistoryEntry], current: Frame | None
) -> dict[str, Any]:
    frames = [entry.frame for entry in history]
    if current is not None and (not frames or frames[-1] is not current):
        frames.append(current)
    if len(frames) < 2:
        return {"tracked_pairs": 0, "motions": [], "structural_change": False}
    before = _object_state_summary(frames[-2])
    after = _object_state_summary(frames[-1])
    remaining = list(after["components"])
    motions = []
    for component in before["components"]:
        candidates = [
            item
            for item in remaining
            if item["color"] == component["color"]
            and item["size"] == component["size"]
            and item["height"] == component["height"]
            and item["width"] == component["width"]
        ]
        if not candidates:
            continue
        match = min(
            candidates,
            key=lambda item: (
                abs(item["centroid"][0] - component["centroid"][0])
                + abs(item["centroid"][1] - component["centroid"][1])
            ),
        )
        remaining.remove(match)
        motions.append(
            {
                "color": component["color"],
                "size": component["size"],
                "delta": [
                    round(match["centroid"][0] - component["centroid"][0], 3),
                    round(match["centroid"][1] - component["centroid"][1], 3),
                ],
            }
        )
    return {
        "tracked_pairs": len(motions),
        "motions": motions[:16],
        "structural_change": before["shape_signature"] != after["shape_signature"],
        "object_count_delta": after["object_count"] - before["object_count"],
    }


def _mouse_search_summary(
    transitions: list[dict[str, Any]], shape: tuple[int, int] = (0, 0)
) -> dict[str, Any]:
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
    positive = {
        (item["row"], item["col"])
        for item in observations
        if item["outcome"] in {"novel", "revisit", "level_progress"}
    }
    blocked = {
        (item["row"], item["col"])
        for item in observations
        if item["outcome"]
        in {"exact_noop", "volatile_only", "terminal_failure", "negative_reward"}
    }
    rows, cols = shape
    frontier: set[tuple[int, int]] = set()
    for row, col in positive:
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + delta_row, col + delta_col)
            if (
                0 <= candidate[0] < rows
                and 0 <= candidate[1] < cols
                and candidate not in unique_coordinates
            ):
                frontier.add(candidate)
    if not frontier and rows and cols:
        for candidate in (
            (rows // 2, cols // 2),
            (rows // 4, cols // 4),
            (rows // 4, 3 * cols // 4),
            (3 * rows // 4, cols // 4),
            (3 * rows // 4, 3 * cols // 4),
        ):
            if candidate not in unique_coordinates and candidate not in blocked:
                frontier.add(candidate)
    return {
        "trials": len(observations),
        "unique_coordinates": len(unique_coordinates),
        "outcomes": dict(sorted(outcome_counts.items())),
        "recent": observations[-16:],
        "positive_regions": [list(item) for item in sorted(positive)[:8]],
        "blocked_coordinates": [list(item) for item in sorted(blocked)[:16]],
        "recommended_coordinates": [
            {"row": row, "col": col, "reason": "spatial frontier"}
            for row, col in sorted(frontier)[:8]
        ],
    }


def _plan_candidates(
    transitions: list[dict[str, Any]],
    current_behavioral_id: str,
    *,
    max_depth: int = 6,
    min_support: int = 2,
    min_confidence: float = 0.75,
    current_state_id: str = "",
    current_object_state_id: str = "",
    valid_actions: Iterable[str] = (),
    config: InferenceControllerConfig | None = None,
) -> list[dict[str, Any]]:
    """Find short progress routes supported by repeatable transition evidence."""

    def planner_outcome(value: Any) -> str:
        outcome = str(value or "unknown")
        if outcome in {"novel", "revisit"}:
            return "state_change"
        if outcome in {"exact_noop", "volatile_only"}:
            return "no_effect"
        return outcome

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in transitions:
        action = normalize_action_key(
            raw.get("action") or raw.get("action_display") or ""
        )
        pairs = {
            (
                str(raw.get("behavioral_before_state_id") or ""),
                str(raw.get("behavioral_after_state_id") or ""),
            ),
            (
                str(raw.get("before_state_id") or ""),
                str(raw.get("after_state_id") or ""),
            ),
            (
                str(raw.get("object_before_state_id") or ""),
                str(raw.get("object_after_state_id") or ""),
            ),
        }
        for before, after in pairs:
            if before and after and action:
                sample = dict(raw)
                sample["planner_after_state_id"] = after
                grouped.setdefault((before, action), []).append(sample)

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for (before, action), samples in grouped.items():
        evidence_groups: dict[str, list[dict[str, Any]]] = {}
        for index, item in enumerate(samples):
            group_id = str(item.get("evidence_id") or f"legacy-observation:{index}")
            evidence_groups.setdefault(group_id, []).append(item)
        outcomes: Counter[tuple[str, str]] = Counter()
        for group_samples in evidence_groups.values():
            group_outcomes = Counter(
                (
                    str(item.get("planner_after_state_id") or ""),
                    planner_outcome(item.get("outcome_class")),
                )
                for item in group_samples
            )
            outcomes[
                min(group_outcomes.items(), key=lambda item: (-item[1], item[0]))[0]
            ] += 1
        (after, outcome), support = min(
            outcomes.items(), key=lambda item: (-item[1], item[0])
        )
        reported_outcome = {
            "state_change": "novel",
            "no_effect": "exact_noop",
        }.get(outcome, outcome)
        trials = len(evidence_groups)
        confidence = support / trials
        terminal_failures = sum(
            any(
                str(item.get("outcome_class") or "")
                in {"terminal_failure", "negative_reward"}
                or bool(item.get("game_over"))
                or float(item.get("reward") or 0.0) < 0.0
                for item in group_samples
            )
            for group_samples in evidence_groups.values()
        )
        terminal_risk = terminal_failures / trials
        branches = [
            {
                "after_state_id": branch_after,
                "outcome_class": {
                    "state_change": "novel",
                    "no_effect": "exact_noop",
                }.get(branch_outcome, branch_outcome),
                "support": branch_support,
                "probability": round(branch_support / trials, 3),
                "terminal": branch_outcome in {"terminal_failure", "negative_reward"},
            }
            for (branch_after, branch_outcome), branch_support in sorted(
                outcomes.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        mean_reward = (
            sum(
                sum(float(item.get("reward") or 0.0) for item in group_samples)
                / len(group_samples)
                for group_samples in evidence_groups.values()
            )
            / trials
        )
        supported_samples = [
            item
            for item in samples
            if str(item.get("planner_after_state_id") or "") == after
            and planner_outcome(item.get("outcome_class")) == outcome
        ]
        after_action_sets = [
            {
                normalize_action_key(value)
                for value in item.get("valid_actions_after") or []
            }
            for item in supported_samples
            if item.get("valid_actions_after")
        ]
        valid_after = (
            set.intersection(*after_action_sets) if after_action_sets else set()
        )
        if (
            support < min_support
            or confidence < min_confidence
            or outcome == "no_effect"
            or terminal_risk
            > (config or InferenceControllerConfig()).plan_max_terminal_risk
        ):
            continue
        adjacency.setdefault(before, []).append(
            {
                "action": action,
                "behavioral_after_state_id": after,
                "outcome_class": reported_outcome,
                "support": support,
                "independent_support": support,
                "raw_observations": len(samples),
                "contradictions": trials - support,
                "confidence": confidence,
                "terminal_risk": terminal_risk,
                "branches": branches,
                "mean_reward": mean_reward,
                "valid_actions_after": sorted(valid_after),
            }
        )
    for edges in adjacency.values():
        edges.sort(
            key=lambda edge: (
                -float(edge["confidence"]),
                -int(edge["support"]),
                str(edge["action"]),
            )
        )

    def known_continuation(start_state: str, depth_limit: int) -> list[str]:
        route_queue = deque([(start_state, [], frozenset({start_state}))])
        route_expansions = 0
        while route_queue and route_expansions < max(32, depth_limit * 32):
            route_expansions += 1
            state_id, route, visited = route_queue.popleft()
            if len(route) >= depth_limit:
                continue
            for edge in adjacency.get(state_id, []):
                next_route = [*route, str(edge["action"])]
                if edge.get("outcome_class") == "level_progress":
                    return next_route
                next_state = str(edge["behavioral_after_state_id"])
                if next_state not in visited:
                    route_queue.append((next_state, next_route, visited | {next_state}))
        return []

    start_ids = {
        current_behavioral_id,
        current_state_id,
        current_object_state_id,
    } - {""}
    allowed = {
        normalize_action_key(item) for item in valid_actions if str(item).strip()
    }
    queue = deque(
        (state_id, [], 1.0, 0, allowed, 0.0, 0.0, [], frozenset({state_id}))
        for state_id in sorted(start_ids)
    )
    best_frontiers = {state_id: [(1.0, 0.0)] for state_id in start_ids}
    plans: list[dict[str, Any]] = []
    expansions = 0
    max_expansions = max(64, max_depth * 64)
    while queue and expansions < max_expansions:
        expansions += 1
        (
            state_id,
            path,
            path_confidence,
            contradictions,
            valid_here,
            path_utility,
            path_terminal_risk,
            contingencies,
            visited_states,
        ) = queue.popleft()
        if len(path) >= max_depth:
            continue
        for edge in adjacency.get(state_id, []):
            allowed_here_families = {action_family(item) for item in valid_here}
            if (
                valid_here
                and action_family(edge["action"]) not in allowed_here_families
            ):
                continue
            next_path = [*path, str(edge["action"])]
            next_confidence = path_confidence * float(edge["confidence"])
            next_contradictions = contradictions + int(edge["contradictions"])
            edge_utility = (config or InferenceControllerConfig()).outcome_utility(
                str(edge.get("outcome_class") or "unknown")
            ) + float(edge.get("mean_reward") or 0.0)
            edge_utility -= float(edge.get("terminal_risk") or 0.0) * abs(
                (config or InferenceControllerConfig()).terminal_failure_utility
            )
            next_utility = path_utility + edge_utility
            next_terminal_risk = 1.0 - (1.0 - path_terminal_risk) * (
                1.0 - float(edge.get("terminal_risk") or 0.0)
            )
            next_contingencies = [
                *contingencies,
                {
                    "action": str(edge["action"]),
                    "expected_after_state_id": str(edge["behavioral_after_state_id"]),
                    "branches": list(edge.get("branches") or ()),
                },
            ]
            if edge.get("outcome_class") == "level_progress":
                plans.append(
                    {
                        "actions": next_path,
                        "target": "level_progress",
                        "verified_steps": len(next_path),
                        "confidence": round(next_confidence, 3),
                        "support": int(edge["support"]),
                        "contradictions": next_contradictions,
                        "expected_utility": round(next_utility, 3),
                        "terminal_risk": round(next_terminal_risk, 3),
                        "contingencies": next_contingencies,
                    }
                )
                continue
            next_state = str(edge["behavioral_after_state_id"])
            if next_state in visited_states:
                continue
            frontier = best_frontiers.setdefault(next_state, [])
            if any(
                confidence >= next_confidence and utility >= next_utility
                for confidence, utility in frontier
            ):
                continue
            frontier[:] = [
                (confidence, utility)
                for confidence, utility in frontier
                if not (next_confidence >= confidence and next_utility >= utility)
            ]
            frontier.append((next_confidence, next_utility))
            queue.append(
                (
                    next_state,
                    next_path,
                    next_confidence,
                    next_contradictions,
                    set(edge.get("valid_actions_after") or ()),
                    next_utility,
                    next_terminal_risk,
                    next_contingencies,
                    visited_states | {next_state},
                )
            )
    for plan in plans:
        actions = list(plan.get("actions") or ())
        observation_policy = []
        for step_index, contingency in enumerate(plan.get("contingencies") or ()):
            expected_after = str(contingency.get("expected_after_state_id") or "")
            policy_branches = []
            for branch in contingency.get("branches") or ():
                policy_branch = dict(branch)
                after_state = str(branch.get("after_state_id") or "")
                if branch.get("terminal"):
                    continuation: list[str] = []
                    status = "abort_terminal_branch"
                elif str(branch.get("outcome_class") or "") == "level_progress":
                    continuation = []
                    status = "target_reached"
                elif after_state == expected_after:
                    continuation = actions[step_index + 1 :]
                    status = "continue_verified_route" if continuation else "replan"
                else:
                    continuation = known_continuation(
                        after_state, max(0, max_depth - step_index - 1)
                    )
                    status = "alternate_verified_route" if continuation else "replan"
                policy_branch.update(
                    {
                        "status": status,
                        "continuation_actions": continuation,
                        "next_action": continuation[0] if continuation else None,
                    }
                )
                policy_branches.append(policy_branch)
            contingency["branches"] = policy_branches
            observation_policy.append(
                {
                    "step": step_index + 1,
                    "action": contingency.get("action"),
                    "branches": policy_branches,
                }
            )
        plan["observation_policy"] = observation_policy

    plans.sort(
        key=lambda item: (
            -float(item["expected_utility"]),
            -float(item["confidence"]),
            len(item["actions"]),
            item["actions"],
        )
    )
    return plans[:4]


def _edge_hud_only_change(before: Frame, after: Frame) -> bool:
    """Return whether every changed cell is confined to a thin edge HUD band."""
    if before.shape != after.shape or before.shape[0] <= 0 or before.shape[1] <= 0:
        return False
    rows, cols = before.shape
    edge_band = max(1, min(rows, cols) // 32)
    changed = [
        (row, col)
        for row, (before_row, after_row) in enumerate(zip(before.grid, after.grid))
        for col, (before_cell, after_cell) in enumerate(zip(before_row, after_row))
        if before_cell != after_cell
    ]
    return bool(changed) and all(
        row < edge_band
        or row >= rows - edge_band
        or col < edge_band
        or col >= cols - edge_band
        for row, col in changed
    )


def _transitions(
    history: list[HistoryEntry],
    masked_cells: frozenset[tuple[int, int]] = frozenset(),
    *,
    ignore_edge_hud_changes: bool = False,
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    known_behavioral = (
        {_behavioral_fingerprint(history[0].frame, masked_cells)} if history else set()
    )
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
        grid_changed = before.grid != entry.frame.grid
        object_before = _object_state_id(before)
        object_after = _object_state_id(entry.frame)
        animation = dict(entry.animation)
        transient_effect = bool(
            int(animation.get("transient_changed_cells") or 0) > 0
            or bool(animation.get("temporally_reversible"))
        )
        override = str(entry.outcome_class_override or "")
        if override in {"terminal_failure", "negative_reward"}:
            outcome = override
        elif entry.frame.level > before.level:
            outcome = "level_progress"
        elif transient_effect and before_id == after_id:
            outcome = "transient_effect"
        elif before_id == after_id:
            outcome = "exact_noop"
        elif ignore_edge_hud_changes and _edge_hud_only_change(before, entry.frame):
            outcome = "volatile_only"
        elif behavioral_before == behavioral_after:
            outcome = "volatile_only"
        elif object_before and object_before == object_after:
            # Preserve position-sensitive raw/behavioral state IDs, but do not
            # reward a pure translation of the same object layout as novelty.
            outcome = "revisit"
        elif behavioral_after not in known_behavioral:
            outcome = "novel"
        else:
            outcome = "revisit"
        transitions.append(
            {
                "action": action,
                "action_family": action_family(action),
                "before_state_id": before_id,
                "after_state_id": after_id,
                "behavioral_before_state_id": behavioral_before,
                "behavioral_after_state_id": behavioral_after,
                "object_before_state_id": object_before,
                "object_after_state_id": object_after,
                "valid_actions_before": [
                    normalize_action_key(value) for value in before.valid_actions
                ],
                "valid_actions_after": [
                    normalize_action_key(value) for value in entry.frame.valid_actions
                ],
                "board_changed": grid_changed,
                "decision_context_changed": before_id != after_id and not grid_changed,
                "behavioral_changed": behavioral_before != behavioral_after
                or transient_effect,
                "transient_effect": transient_effect,
                "animation": animation,
                "outcome_class": outcome,
                "reward": float(entry.reward),
                "level_before": before.level,
                "level_after": entry.frame.level,
            }
        )
        known_behavioral.update((behavioral_before, behavioral_after))
    exact_effects: dict[tuple[str, str], Counter[str]] = {}
    for transition in transitions:
        exact_effects.setdefault(
            (str(transition["before_state_id"]), str(transition["action"])),
            Counter(),
        )[str(transition["after_state_id"])] += 1
    verified_exact_effects = {
        pair
        for pair, after_states in exact_effects.items()
        if len(after_states) == 1 and sum(after_states.values()) >= 2
    }
    for transition in transitions:
        pair = (str(transition["before_state_id"]), str(transition["action"]))
        repeatable_exact_effect = (
            transition["outcome_class"] == "volatile_only"
            and pair in verified_exact_effects
        )
        transition["repeatable_exact_effect"] = repeatable_exact_effect
        if repeatable_exact_effect:
            transition["outcome_class"] = "revisit"
            transition["behavioral_changed"] = True
    return transitions


def _cycle_period(state_ids: list[str], max_period: int) -> int | None:
    if len(state_ids) < 3:
        return None
    max_period = min(max_period, len(state_ids) // 2)
    for period in range(1, max_period + 1):
        if state_ids[-period:] == state_ids[-2 * period : -period]:
            return period
    return None


def _behavioral_cycle_signal(
    transitions: list[dict[str, Any]], max_period: int
) -> dict[str, Any] | None:
    """Detect exact state cycles and noisy action/outcome loops."""
    state_ids = [
        str(item.get("behavioral_after_state_id") or "") for item in transitions
    ]
    exact = _cycle_period(state_ids, max_period)
    if exact is not None:
        return {"kind": "exact_state", "period": exact, "confidence": 1.0}
    signatures = []
    for item in transitions:
        normalized = normalize_action_key(str(item.get("action") or ""))
        family = str(item.get("action_family") or action_family(normalized))
        # Distinct click coordinates are distinct search arms. Collapsing all
        # MOUSE actions into one family made broad coordinate exploration look
        # like a period-one cycle.
        signature_action = normalized if family == "MOUSE" else family
        signatures.append(
            (signature_action, str(item.get("outcome_class") or "unknown"))
        )
    for period in range(1, min(max_period, len(signatures) // 2) + 1):
        first = signatures[-2 * period : -period]
        second = signatures[-period:]
        matches = sum(left == right for left, right in zip(first, second))
        confidence = matches / period
        if confidence < 0.75 or any(
            outcome == "level_progress" for _, outcome in second
        ):
            continue
        if any(outcome == "novel" for _, outcome in second):
            actions = [action for action, _ in second]
            inverse = {
                "UP": "DOWN",
                "DOWN": "UP",
                "LEFT": "RIGHT",
                "RIGHT": "LEFT",
            }
            repeated_click = period == 1 and action_family(actions[0]) == "MOUSE"
            inverse_cycle = (
                period == 2
                and len(actions) == 2
                and inverse.get(actions[0]) == actions[1]
            )
            if not repeated_click and not inverse_cycle:
                continue
        return {
            "kind": "action_outcome",
            "period": period,
            "confidence": round(confidence, 3),
        }
    return None


def _stagnation_count(
    transitions: list[dict[str, Any]], *, behavioral: bool = False
) -> int:
    before_key = "behavioral_before_state_id" if behavioral else "before_state_id"
    after_key = "behavioral_after_state_id" if behavioral else "after_state_id"
    seen: set[str] = set()
    stagnant = 0
    for transition in transitions:
        before_id = str(transition[before_key])
        after_id = str(transition[after_key])
        if not seen:
            seen.add(before_id)
        progressed = (
            int(transition["level_after"]) > int(transition["level_before"])
            or after_id not in seen
        )
        seen.add(after_id)
        stagnant = 0 if progressed else stagnant + 1
    return stagnant


def _no_op_streak(
    transitions: list[dict[str, Any]], *, behavioral: bool = False
) -> int:
    streak = 0
    for transition in reversed(transitions):
        changed = (
            transition["behavioral_changed"]
            if behavioral
            else transition["board_changed"]
        )
        if changed:
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
        and transition["outcome_class"] == "exact_noop"
    )


def _directional_no_progress_signal(
    history: list[HistoryEntry],
    current_frame: Frame | None,
    config: InferenceControllerConfig,
) -> dict[str, int | str] | None:
    """Detect one direction dominating a bounded window without real progress."""
    window = config.directional_no_progress_window
    limit = config.directional_no_progress_limit
    if current_frame is None or window <= 0 or limit <= 0 or limit > window:
        return None
    if len(history) < window + 1:
        return None
    baseline = history[-window - 1].frame
    recent = history[-window:]
    if baseline.level != current_frame.level or any(
        entry.frame.level != current_frame.level for entry in recent
    ):
        return None
    if current_frame.score > baseline.score or any(
        entry.reward > 0.0 or entry.outcome_class_override == "level_progress"
        for entry in recent
    ):
        return None
    directional = Counter(
        family
        for family in (action_family(entry.action) for entry in recent)
        if family in {"UP", "DOWN", "LEFT", "RIGHT"}
    )
    if not directional:
        return None
    action, count = min(
        directional.items(), key=lambda item: (-item[1], item[0])
    )
    if count < limit:
        return None
    return {"action": action, "count": count, "window": window}


def action_guard_reason_code(
    history: list[HistoryEntry],
    current_frame: Frame | None,
    action: str,
    config: InferenceControllerConfig,
) -> str | None:
    if not config.enabled:
        return None
    action_key = normalize_action_key(action)
    if config.repeat_action_limit > 0 and action_family(action_key) == "MOUSE":
        current_level = current_frame.level if current_frame is not None else None
        unsuccessful = [
            normalize_action_key(entry.action)
            for entry in history[1:]
            if entry.reward <= 0.0
            and (current_level is None or entry.frame.level == current_level)
        ]
        if unsuccessful.count(action_key) >= config.repeat_action_limit:
            return "repeated_parameterized_action"
    inverse = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    recent_families = [
        action_family(entry.action)
        for entry in history[1:]
        if entry.reward <= 0.0
        and (current_frame is None or entry.frame.level == current_frame.level)
    ]
    candidate_family = action_family(action_key)
    directional_signal = _directional_no_progress_signal(
        history, current_frame, config
    )
    if (
        directional_signal is not None
        and candidate_family == directional_signal["action"]
    ):
        return "directional_no_progress"
    if (
        config.repeat_action_limit > 0
        and len(recent_families) >= 3
        and inverse.get(candidate_family) == recent_families[-1]
        and recent_families[-3:]
        == [
            inverse[candidate_family],
            candidate_family,
            inverse[candidate_family],
        ]
    ):
        return "repeated_inverse_cycle"
    if (
        action_noop_trials(history, current_frame, action)
        >= config.same_state_noop_limit
    ):
        return "repeated_exact_noop"
    state_id = frame_fingerprint(current_frame)
    local_trials = [
        transition
        for transition in _transitions(history)
        if transition["before_state_id"] == state_id
        and transition["action"] == action_key
    ]
    failures = sum(
        transition["outcome_class"] in {"terminal_failure", "negative_reward"}
        for transition in local_trials
    )
    if harmful_evidence_is_decisive(failures, len(local_trials)):
        return "known_harmful_local"
    return None


def action_guard_reason(
    history: list[HistoryEntry],
    current_frame: Frame | None,
    action: str,
    config: InferenceControllerConfig,
) -> str | None:
    reason_code = action_guard_reason_code(history, current_frame, action, config)
    if reason_code is None:
        return None
    if reason_code == "known_harmful_local":
        return "exact state/action pair has decisive terminal-failure evidence"
    if reason_code == "repeated_parameterized_action":
        return (
            "the same parameterized action already failed to make progress "
            f"{config.repeat_action_limit} times"
        )
    if reason_code == "repeated_inverse_cycle":
        return "the requested action would repeat an unsuccessful inverse-action cycle"
    if reason_code == "directional_no_progress":
        return (
            "the requested direction dominates the recent action window without "
            "score or level progress; try a different direction"
        )
    trials = action_noop_trials(history, current_frame, action)
    return f"exact state/action pair already produced {trials} confirmed no-op trials"


def harmful_evidence_is_decisive(harmful_trials: int, total_trials: int) -> bool:
    """Return whether independent evidence supports a hard harm veto.

    A lone harmful observation remains decisive when it is the only evidence,
    but conflicting evidence must have a strict harmful majority. This keeps
    stochastic or subsequently recovered actions eligible for cautious
    revalidation instead of permanently suppressing them.
    """
    harmful = max(0, int(harmful_trials))
    total = max(harmful, int(total_trials))
    return harmful > 0 and harmful * 2 > total


def _independent_transition_samples(
    transitions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse repeated records from one trial into one state/action vote."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for index, item in enumerate(transitions):
        source = str(item.get("evidence_id") or f"legacy-observation:{index}")
        key = (
            source,
            str(item.get("behavioral_before_state_id") or ""),
            str(item.get("before_state_id") or ""),
            str(item.get("action") or ""),
        )
        groups.setdefault(key, []).append(item)

    independent: list[dict[str, Any]] = []
    for samples in groups.values():
        outcomes = Counter(
            (
                str(item.get("behavioral_after_state_id") or ""),
                str(item.get("after_state_id") or ""),
                str(item.get("outcome_class") or "unknown"),
            )
            for item in samples
        )
        modal = min(outcomes.items(), key=lambda item: (-item[1], item[0]))[0]
        matching = [
            item
            for item in samples
            if (
                str(item.get("behavioral_after_state_id") or ""),
                str(item.get("after_state_id") or ""),
                str(item.get("outcome_class") or "unknown"),
            )
            == modal
        ]
        representative = dict(matching[-1])
        representative["reward"] = sum(
            float(item.get("reward") or 0.0) for item in matching
        ) / len(matching)
        representative["raw_observations"] = len(samples)
        independent.append(representative)
    return independent


def _matches_current_state(
    item: dict[str, Any],
    current_behavioral_id: str,
    current_state_id: str,
    current_object_state_id: str = "",
) -> bool:
    return bool(
        item.get("behavioral_before_state_id") == current_behavioral_id
        or item.get("before_state_id") == current_state_id
        or (
            current_object_state_id
            and item.get("object_before_state_id") == current_object_state_id
        )
    )


def _transition_chain_is_continuous(
    earlier: dict[str, Any], later: dict[str, Any]
) -> bool:
    """Require continuity in at least one matching state representation."""
    return any(
        earlier.get(after_key) and earlier.get(after_key) == later.get(before_key)
        for after_key, before_key in (
            ("behavioral_after_state_id", "behavioral_before_state_id"),
            ("after_state_id", "before_state_id"),
            ("object_after_state_id", "object_before_state_id"),
        )
    )


def _nonstationary_actions(
    transitions: list[dict[str, Any]],
    current_behavioral_id: str,
    current_state_id: str,
    current_object_state_id: str = "",
) -> list[dict[str, Any]]:
    """Detect recent outcome regime changes using independent observations."""
    observations: dict[str, list[str]] = {}
    # Change points are temporal, so preserve chronological observations here.
    # Provenance collapsing is still used by ranking and calibrated models.
    for item in transitions:
        if not _matches_current_state(
            item,
            current_behavioral_id,
            current_state_id,
            current_object_state_id,
        ):
            continue
        observations.setdefault(str(item["action_family"]), []).append(
            str(item.get("outcome_class") or "unknown")
        )
    shifts = []
    for action, outcomes in observations.items():
        if len(outcomes) < 4:
            continue
        split = len(outcomes) // 2
        old_counts = Counter(outcomes[:split])
        recent_counts = Counter(outcomes[split:])
        old_outcome, old_support = min(
            old_counts.items(), key=lambda item: (-item[1], item[0])
        )
        recent_outcome, recent_support = min(
            recent_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if (
            old_outcome != recent_outcome
            and old_support / split >= 0.67
            and recent_support / (len(outcomes) - split) >= 0.67
        ):
            shifts.append(
                {
                    "action": action,
                    "previous_outcome": old_outcome,
                    "recent_outcome": recent_outcome,
                    "previous_support": old_support,
                    "recent_support": recent_support,
                    "observations": len(outcomes),
                }
            )
    return sorted(shifts, key=lambda item: str(item["action"]))


def _regime_adapted_transition_samples(
    transitions: list[dict[str, Any]],
    current_behavioral_id: str,
    current_state_id: str,
    current_object_state_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    independent = _independent_transition_samples(transitions)
    relevant = [
        item
        for item in independent
        if _matches_current_state(
            item,
            current_behavioral_id,
            current_state_id,
            current_object_state_id,
        )
    ]
    shifts = _nonstationary_actions(
        transitions,
        current_behavioral_id,
        current_state_id,
        current_object_state_id,
    )
    shifted_families = {str(item["action"]) for item in shifts}
    if not shifted_families:
        return relevant, shifts
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in relevant:
        grouped.setdefault(str(item["action_family"]), []).append(item)
    chronological: dict[str, list[dict[str, Any]]] = {}
    for item in transitions:
        if _matches_current_state(
            item,
            current_behavioral_id,
            current_state_id,
            current_object_state_id,
        ):
            chronological.setdefault(str(item["action_family"]), []).append(item)
    adapted = []
    for family, samples in grouped.items():
        if family in shifted_families:
            regime_samples = chronological.get(family, samples)
            samples = regime_samples[len(regime_samples) // 2 :]
        adapted.extend(samples)
    return adapted, shifts


def _rank_actions(
    transitions: list[dict[str, Any]],
    current_behavioral_id: str,
    current_state_id: str,
    valid_actions: list[str],
    config: InferenceControllerConfig,
    current_object_state_id: str = "",
) -> list[dict[str, Any]]:
    independent_transitions = _independent_transition_samples(transitions)
    regime_transitions, regime_shifts = _regime_adapted_transition_samples(
        transitions,
        current_behavioral_id,
        current_state_id,
        current_object_state_id,
    )
    shifted_families = {str(item["action"]) for item in regime_shifts}
    local: dict[str, Counter[str]] = {}
    raw_trials: Counter[str] = Counter()
    delayed_progress: Counter[str] = Counter()
    recent_behavioral = {
        str(item["behavioral_after_state_id"])
        for item in transitions[-config.cycle_window :]
    }
    cycle_returns: Counter[str] = Counter()
    for item in transitions:
        if _matches_current_state(
            item,
            current_behavioral_id,
            current_state_id,
            current_object_state_id,
        ):
            raw_trials[str(item["action_family"])] += 1
    source_sequences: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(independent_transitions):
        source_id = str(item.get("evidence_id") or f"observation:{index}")
        source_sequences.setdefault(source_id, []).append(item)

    for sequence in source_sequences.values():
        for index, item in enumerate(sequence):
            if not _matches_current_state(
                item,
                current_behavioral_id,
                current_state_id,
                current_object_state_id,
            ):
                continue
            family = str(item["action_family"])
            if str(item.get("outcome_class") or "") in {
                "exact_noop",
                "volatile_only",
                "terminal_failure",
                "negative_reward",
            }:
                continue
            for distance, later in enumerate(
                sequence[index + 1 : index + 1 + config.credit_horizon], start=1
            ):
                previous = sequence[index + distance - 1]
                if not _transition_chain_is_continuous(previous, later):
                    break
                outcome = str(later.get("outcome_class") or "unknown")
                if (
                    outcome in {"terminal_failure", "negative_reward"}
                    or str(later.get("action_family") or "") == "RESET"
                ):
                    break
                if outcome == "level_progress":
                    delayed_progress[family] += config.credit_discount**distance
                    break

    for item in regime_transitions:
        family = str(item["action_family"])
        stats = local.setdefault(family, Counter())
        stats["trials"] += 1
        stats[str(item["outcome_class"])] += 1
        stats["reward_sum"] += float(item.get("reward") or 0.0)
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
        failures = int(stats["terminal_failure"] + stats["negative_reward"])
        delayed_credit = float(delayed_progress[family])
        expected_value = (
            sum(
                config.outcome_utility(outcome) * count
                for outcome, count in stats.items()
                if outcome not in {"trials", "reward_sum"}
            )
            / trials
            if trials
            else 0.0
        )
        mean_reward = float(stats["reward_sum"]) / trials if trials else 0.0
        expected_value += mean_reward
        if trials:
            expected_value += (
                delayed_credit / trials * config.outcome_utility("level_progress")
            )
        if parameterized:
            # Coordinate outcomes are local evidence about individual search arms,
            # not evidence that the entire parameterized action family is unsafe.
            expected_value = max(0.0, expected_value)
        uncertainty = 1.0 / math.sqrt(trials + 1.0)
        outcome_counts = [
            int(stats[outcome])
            for outcome in (
                "level_progress",
                "novel",
                "revisit",
                "exact_noop",
                "volatile_only",
                "transient_effect",
                "terminal_failure",
                "negative_reward",
            )
            if stats[outcome]
        ]
        disagreement = (
            1.0 - max(outcome_counts) / sum(outcome_counts) if outcome_counts else 0.0
        )
        information_gain = uncertainty * (1.25 if parameterized else 1.0) + disagreement
        adaptive_exploration_weight = config.exploration_weight * (
            1.0 + disagreement + 0.5 * uncertainty
        )
        decision_value = expected_value + adaptive_exploration_weight * information_gain
        decisive_harm = not parameterized and harmful_evidence_is_decisive(
            failures, trials
        )
        if family == "RESET":
            decision_value = min(decision_value, config.noop_utility)
            priority, reason = (
                6,
                "reserved for explicit cycle or stagnation recovery",
            )
        elif parameterized and progress:
            priority, reason = (
                0,
                "confirmed progress at a coordinate; continue coordinate search",
            )
        elif parameterized:
            priority, reason = (
                1,
                "coordinate-specific evidence; choose an untried coordinate",
            )
        elif decisive_harm:
            priority, reason = 5, "observed terminal failure or negative reward"
        elif failures and progress:
            priority, reason = (
                3,
                "mixed progress and harmful outcomes; revalidate cautiously",
            )
        elif progress:
            priority, reason = 0, "confirmed level progress from this state"
        elif delayed_credit > 0.0 and not failures:
            priority, reason = 2, "observed delayed progress after this action"
        elif trials == 0:
            priority, reason = 1, "untried action from this state"
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
            "raw_observations": int(raw_trials[family]),
            "level_progress": progress,
            "delayed_progress_credit": round(delayed_credit, 3),
            "novel": novel,
            "revisits": int(stats["revisit"]),
            "no_ops": noops,
            "cycle_risk": bool(cycle_risk),
            "terminal_failures": failures,
            "harm_decisive": decisive_harm,
            "expected_value": round(expected_value, 3),
            "mean_reward": round(mean_reward, 3),
            "uncertainty": round(uncertainty, 3),
            "model_disagreement": round(disagreement, 3),
            "information_gain": round(information_gain, 3),
            "decision_value": round(decision_value, 3),
            "adaptive_exploration_weight": round(adaptive_exploration_weight, 3),
            "parameterized": parameterized,
            "regime_adapted": family in shifted_families,
            "reason": reason,
        }
        ranked.append(
            ((-decision_value, priority, failures, trials, valid_index), payload)
        )
    ranked.sort(key=lambda item: item[0])
    return [payload for _, payload in ranked[:6]]


def _transition_models_here(
    transitions: list[dict[str, Any]],
    current_behavioral_id: str,
    current_state_id: str = "",
    current_object_state_id: str = "",
) -> list[dict[str, Any]]:
    """Return compact empirical action models verified at the current state."""
    transitions, regime_shifts = _regime_adapted_transition_samples(
        transitions,
        current_behavioral_id,
        current_state_id,
        current_object_state_id,
    )
    exact_state_evidence = any(
        item.get("behavioral_before_state_id") == current_behavioral_id
        or item.get("before_state_id") == current_state_id
        for item in transitions
    )
    using_object_abstraction = bool(
        current_object_state_id and not exact_state_evidence
    )

    def model_matches(item: dict[str, Any]) -> bool:
        if exact_state_evidence:
            return bool(
                item.get("behavioral_before_state_id") == current_behavioral_id
                or item.get("before_state_id") == current_state_id
            )
        return bool(
            current_object_state_id
            and item.get("object_before_state_id") == current_object_state_id
        )

    shifted_families = {str(item["action"]) for item in regime_shifts}
    observations: dict[str, dict[str, Counter[Any]]] = {}
    for item in transitions:
        if not model_matches(item):
            continue
        action = str(item["action"])
        stats = observations.setdefault(
            action,
            {
                "joint_transitions": Counter(),
                "next_states": Counter(),
                "outcomes": Counter(),
            },
        )
        next_state = str(
            item.get("object_after_state_id")
            if using_object_abstraction
            else item["behavioral_after_state_id"]
        )
        outcome = str(item["outcome_class"])
        stats["joint_transitions"][(next_state, outcome)] += 1
        stats["next_states"][next_state] += 1
        stats["outcomes"][outcome] += 1

    models: list[dict[str, Any]] = []
    for action, stats in observations.items():
        next_states = stats["next_states"]
        outcomes = stats["outcomes"]
        joint_transitions = stats["joint_transitions"]
        trials = sum(joint_transitions.values())
        (next_state, outcome), support = min(
            joint_transitions.items(),
            key=lambda item: (-item[1], item[0]),
        )
        state_support = max(next_states.values(), default=0)
        state_deterministic = len(next_states) == 1
        deterministic = len(joint_transitions) == 1
        failures = int(outcomes["terminal_failure"] + outcomes["negative_reward"])
        calibrated_confidence = (support + 1.0) / (trials + 2.0)
        matching_rewards = [
            float(item.get("reward") or 0.0)
            for item in transitions
            if str(item.get("action") or "") == action and model_matches(item)
        ]
        models.append(
            {
                "action": action,
                "trials": trials,
                "raw_observations": sum(
                    int(item.get("raw_observations") or 1)
                    for item in transitions
                    if str(item.get("action") or "") == action and model_matches(item)
                ),
                "predicted_outcome": outcome,
                "predicted_behavioral_state_id": (
                    next_state
                    if state_deterministic and not using_object_abstraction
                    else None
                ),
                "predicted_object_state_id": (
                    next_state
                    if state_deterministic and using_object_abstraction
                    else None
                ),
                "state_abstraction": (
                    "object_relational" if using_object_abstraction else "behavioral"
                ),
                "support": support,
                "contradictions": trials - support,
                "confidence": round(support / trials, 3) if trials else 0.0,
                "verified_deterministic": deterministic,
                "verified_state_deterministic": state_deterministic,
                "state_confidence": (
                    round(state_support / trials, 3) if trials else 0.0
                ),
                "calibrated_confidence": round(calibrated_confidence, 3),
                "uncertainty": round(1.0 - calibrated_confidence, 3),
                "terminal_failures": failures,
                "mean_reward": round(
                    sum(matching_rewards) / max(1, len(matching_rewards)), 3
                ),
                "regime_adapted": action_family(action) in shifted_families,
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
    history: list[HistoryEntry],
    current_frame: Frame | None,
    valid_actions: Iterable[str],
    config: InferenceControllerConfig,
    external_transitions: Iterable[dict[str, Any]] | None = None,
    evidence_id: str = "",
) -> dict[str, Any]:
    masked_cells = _volatile_cells(history, current_frame, config)
    transitions = _transitions(
        history,
        masked_cells,
        ignore_edge_hud_changes=config.ignore_edge_hud_changes,
    )
    if evidence_id:
        for item in transitions:
            item["evidence_id"] = evidence_id
    external = []
    for raw in external_transitions or ():
        item = _normalize_external_transition(raw)
        if item is not None:
            external.append(item)
    # Persisted records carry observation timestamps and precede the live
    # in-memory trajectory. This preserves old-to-new ordering for change-point
    # detection while independent-evidence collapsing removes duplicates.
    evidence_transitions = [*external, *transitions]
    current_id = frame_fingerprint(current_frame)
    behavioral_id = _behavioral_fingerprint(current_frame, masked_cells)
    object_state_id = _object_state_id(current_frame)
    state_ids = ([frame_fingerprint(history[0].frame)] if history else []) + [
        str(item["after_state_id"]) for item in transitions
    ]
    behavioral_ids = (
        [_behavioral_fingerprint(history[0].frame, masked_cells)] if history else []
    ) + [str(item["behavioral_after_state_id"]) for item in transitions]
    active_ids = behavioral_ids if config.outcome_aware else state_ids
    active_current_id = behavioral_id if config.outcome_aware else current_id
    no_op_streak = _no_op_streak(transitions)
    behavioral_no_op_streak = _no_op_streak(transitions, behavioral=True)
    stagnation = _stagnation_count(transitions, behavioral=config.outcome_aware)
    cycle_signal = _behavioral_cycle_signal(transitions, config.cycle_window)
    cycle_period = int(cycle_signal["period"]) if cycle_signal is not None else None

    tried: dict[str, dict[str, int]] = {}
    for item in evidence_transitions:
        comparison_id = (
            item["behavioral_before_state_id"]
            if config.outcome_aware
            else item["before_state_id"]
        )
        if comparison_id != active_current_id:
            continue
        stats = tried.setdefault(
            str(item["action"]), {"trials": 0, "changes": 0, "no_ops": 0}
        )
        stats["trials"] += 1
        stats["changes" if item["board_changed"] else "no_ops"] += 1

    normalized_valid = [
        normalize_action_key(action) for action in valid_actions if str(action).strip()
    ]
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
    suggested = [
        action
        for action in [*untried, *[a for a in useful if a not in untried]]
        if action not in discouraged
    ][:6]
    current_level_transitions = [
        item
        for item in transitions
        if current_frame is not None and item["level_after"] == current_frame.level
    ]
    recovery_reasons: list[str] = []
    directional_no_progress = _directional_no_progress_signal(
        history, current_frame, config
    )
    active_noop_streak = (
        behavioral_no_op_streak if config.outcome_aware else no_op_streak
    )
    if active_noop_streak >= config.same_state_noop_limit:
        recovery_reasons.append("repeated_noop")
    if cycle_period is not None:
        recovery_reasons.append("short_cycle")
    if stagnation >= config.stagnation_window:
        recovery_reasons.append("stagnation")
    if directional_no_progress is not None:
        recovery_reasons.append("directional_no_progress")
    latest_outcome = transitions[-1]["outcome_class"] if transitions else None
    if latest_outcome in {"terminal_failure", "negative_reward"}:
        recovery_reasons.append("harmful_outcome")
    if not current_level_transitions:
        phase = "orient"
    elif recovery_reasons:
        phase = "recover"
    elif (config.outcome_aware and latest_outcome == "level_progress") or (
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
    ranked_actions = (
        _rank_actions(
            evidence_transitions,
            behavioral_id,
            current_id,
            normalized_valid,
            config,
            object_state_id,
        )
        if config.outcome_aware
        else []
    )
    transition_models = (
        _transition_models_here(
            evidence_transitions, behavioral_id, current_id, object_state_id
        )
        if config.outcome_aware
        else []
    )
    model_conflicts = sum(
        int(model["contradictions"] > 0) for model in transition_models
    )
    nonstationary_actions = _nonstationary_actions(
        evidence_transitions, behavioral_id, current_id, object_state_id
    )
    if model_conflicts and "transition_model_conflict" not in recovery_reasons:
        recovery_reasons.append("transition_model_conflict")
        phase = "recover"
    if nonstationary_actions:
        recovery_reasons.append("nonstationary_dynamics")
        phase = "recover"
    action_budget = {
        "orient": config.orient_action_budget,
        "explore": config.explore_action_budget,
        "recover": config.recover_action_budget,
        "progress": config.progress_action_budget,
    }.get(phase, 1)
    plan_candidates = (
        _plan_candidates(
            evidence_transitions,
            behavioral_id,
            max_depth=config.plan_max_depth,
            min_support=config.plan_min_support,
            min_confidence=config.plan_min_confidence,
            current_state_id=current_id,
            current_object_state_id=object_state_id,
            valid_actions=normalized_valid,
            config=config,
        )
        if config.outcome_aware
        else []
    )
    recommended_experiments = [
        {
            "action": item["action"],
            "information_gain": item["information_gain"],
            "hypothesis": (
                "resolve conflicting transition outcomes"
                if item["model_disagreement"] > 0.0
                else "reduce uncertainty about this transition"
            ),
            "observe": "outcome class and resulting behavioral state",
        }
        for item in sorted(
            ranked_actions,
            key=lambda item: (
                -float(item["information_gain"]),
                int(item["priority"]),
                str(item["action"]),
            ),
        )[:3]
    ]
    recovery_portfolio: list[dict[str, Any]] = []
    if recovery_reasons:
        shifting_actions = {item["action"] for item in nonstationary_actions}
        revalidation = next(
            (
                item
                for item in ranked_actions
                if item["action"] in shifting_actions
                or item["model_disagreement"] > 0.0
            ),
            None,
        )
        if revalidation is not None:
            recovery_portfolio.append(
                {
                    "strategy": "revalidate_changed_model",
                    "action": revalidation["action"],
                    "trigger": "nonstationary or conflicting transition evidence",
                    "success_signal": "repeatable outcome and behavioral successor",
                }
            )
        alternative = next(
            (
                item
                for item in ranked_actions
                if not item["harm_decisive"]
                and not item["cycle_risk"]
                and (revalidation is None or item["action"] != revalidation["action"])
            ),
            None,
        )
        if alternative is not None:
            recovery_portfolio.append(
                {
                    "strategy": "switch_safe_action_family",
                    "action": alternative["action"],
                    "trigger": ",".join(recovery_reasons),
                    "success_signal": "novel behavioral state or level progress",
                }
            )
        if "RESET" in normalized_valid and any(
            reason in recovery_reasons for reason in ("short_cycle", "stagnation")
        ):
            recovery_portfolio.append(
                {
                    "strategy": "reset_episode",
                    "action": "RESET",
                    "trigger": "persistent cycle or stagnation",
                    "success_signal": "return to a known controllable start state",
                }
            )
    return {
        "enabled": config.enabled,
        "policy": _normalize_policy(config.policy),
        "phase": phase,
        "action_budget": action_budget,
        "state_id": current_id,
        "behavioral_state_id": behavioral_id,
        "object_state_id": object_state_id,
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
        "cycle_signal": cycle_signal,
        "directional_no_progress": directional_no_progress,
        "latest_outcome": latest_outcome,
        "recovery_reasons": recovery_reasons,
        "recovery_portfolio": recovery_portfolio,
        "nonstationary_actions": nonstationary_actions,
        "tried_here": tried,
        "suggested_actions": suggested,
        "discouraged_actions": discouraged,
        "ranked_actions": ranked_actions,
        "recommended_experiments": recommended_experiments,
        "object_state": _object_state_summary(current_frame),
        "object_temporal": _object_temporal_summary(history, current_frame),
        "mouse_search": _mouse_search_summary(
            evidence_transitions, current_frame.shape if current_frame else (0, 0)
        ),
        "outcome_utilities": {
            outcome: config.outcome_utility(outcome)
            for outcome in (
                "level_progress",
                "novel",
                "revisit",
                "volatile_only",
                "transient_effect",
                "exact_noop",
                "terminal_failure",
                "negative_reward",
            )
        },
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
    *,
    reward: float = 0.0,
    game_over: bool = False,
    run_complete: bool = False,
    next_valid_actions: Iterable[str] | None = None,
    animation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_id = frame_fingerprint(before)
    after_id = frame_fingerprint(after)
    action_key = action_family(action)
    before_snapshot = build_experience_snapshot(
        prior_history,
        before,
        valid_actions if valid_actions is not None else [action_key],
        config,
    )
    provisional_history = [
        *prior_history,
        HistoryEntry(
            action=action,
            frame=after,
            reward=float(reward),
            animation=dict(animation or {}),
            outcome_class_override=(
                "terminal_failure"
                if game_over and not run_complete
                else "negative_reward"
                if reward < 0.0
                else ""
            ),
        ),
    ]
    snapshot = build_experience_snapshot(
        provisional_history, after, [action_family(action)], config
    )
    transition = snapshot["recent_transitions"][-1]
    outcome_class = str(transition["outcome_class"])
    if game_over and not run_complete:
        outcome_class = "terminal_failure"
    elif reward < 0.0:
        outcome_class = "negative_reward"
    ranking = before_snapshot["ranked_actions"]
    ranked_action = next(
        (
            (index, item)
            for index, item in enumerate(ranking, start=1)
            if item["action"] == action_key
        ),
        None,
    )
    recommended_plan = before_snapshot.get("recommended_plan") or {}
    planned_actions = list(recommended_plan.get("actions") or [])
    planned_action = normalize_action_key(planned_actions[0]) if planned_actions else ""
    observation_policy = list(recommended_plan.get("observation_policy") or [])
    policy_branches = [
        branch
        for step in observation_policy
        if isinstance(step, dict)
        for branch in step.get("branches") or ()
        if isinstance(branch, dict)
    ]
    return {
        "state_context_version": 2,
        "before_state_id": before_id,
        "after_state_id": after_id,
        "behavioral_before_state_id": transition["behavioral_before_state_id"],
        "behavioral_after_state_id": transition["behavioral_after_state_id"],
        "object_before_state_id": transition["object_before_state_id"],
        "object_after_state_id": transition["object_after_state_id"],
        "decision_context_changed": transition["decision_context_changed"],
        "novel_state": outcome_class == "novel",
        "outcome_class": outcome_class,
        "reward": float(reward),
        "game_over": bool(game_over),
        "run_complete": bool(run_complete),
        "valid_actions_before": [
            normalize_action_key(item) for item in (valid_actions or ())
        ],
        "valid_actions_after": [
            normalize_action_key(item) for item in (next_valid_actions or ())
        ],
        "loop_detected": snapshot["cycle_period"] is not None,
        "cycle_risk": snapshot["cycle_period"] is not None,
        "cycle_period": snapshot["cycle_period"],
        "controller_policy": snapshot["policy"],
        "controller_phase": snapshot["phase"],
        "controller_reason_codes": list(snapshot["recovery_reasons"]),
        "action_rank": ranked_action[0] if ranked_action is not None else None,
        "action_rank_reason": ranked_action[1]["reason"]
        if ranked_action is not None
        else None,
        "action_regime_adapted": bool(
            ranked_action is not None and ranked_action[1].get("regime_adapted")
        ),
        "recommended_plan_action": planned_action or None,
        "followed_recommended_plan": bool(
            planned_action and normalize_action_key(action) == planned_action
        ),
        "recommended_plan_confidence": recommended_plan.get("confidence"),
        "recommended_plan_expected_utility": recommended_plan.get("expected_utility"),
        "recommended_plan_branch_count": len(policy_branches),
        "recommended_plan_policy_ready": bool(
            policy_branches
            and all(str(branch.get("status") or "") for branch in policy_branches)
        ),
        "no_op_streak": snapshot["no_op_streak"],
        "behavioral_no_op_streak": snapshot["behavioral_no_op_streak"],
        "stagnation_actions": snapshot["stagnation_actions"],
    }
