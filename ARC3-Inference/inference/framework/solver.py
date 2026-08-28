"""TAAF solver adapter for the existing tool-using harness."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import html
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import arcengine
import taaf.game
from taaf.solver import Solver

from inference.agent.action_names import (
    MAX_ACTION_BATCH,
    to_engine_action,
    to_model_action,
    to_model_actions,
)
from inference.agent.inference_controller import (
    InferenceControllerConfig,
    action_family,
    action_guard_reason,
    action_guard_reason_code,
    build_experience_snapshot,
    frame_fingerprint,
    harmful_evidence_is_decisive,
    normalize_action_key,
    transition_metadata,
)
from inference.agent.runtime_state import (
    RUNTIME_STATE_FILENAME,
    Frame,
    HistoryEntry,
    write_runtime_state,
)
from inference.agent.tool_agent import ToolAgent
from inference.agent.trial_knowledge import TrialKnowledgeStore
from inference.framework.kaggle import (
    DEFAULT_EXPECTED_GPU_COUNT,
    DEFAULT_EXPECTED_GPU_TYPE,
    DEFAULT_QWEN_MODEL_SOURCE,
    DEFAULT_SERVED_MODEL_NAME,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_VLLM_MAX_NUM_SEQS,
    DEFAULT_VLLM_ENABLE_CHUNKED_PREFILL,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
    DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE,
    DEFAULT_WHEELHOUSE_STAMP_TEXT,
    DuckKaggleVllmConfig,
    duck_kaggle_dataset_sources,
    duck_kaggle_model_sources,
    duck_kaggle_setup_command,
    duck_kaggle_teardown_command,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.viewer_artifacts import (
    append_raw_events_sidecar,
    reset_raw_events_sidecar,
)

AnalyzerFactory = Callable[[taaf.game.Game, int], Any]

ANALYZER_RETRY_BACKOFF_SECONDS = 1.0
try:
    ANALYZER_MAX_CONSECUTIVE_FAILURES = max(
        1, int(os.environ.get("ANALYZER_MAX_CONSECUTIVE_FAILURES", "5"))
    )
except ValueError:
    ANALYZER_MAX_CONSECUTIVE_FAILURES = 5
try:
    ANALYZER_MAX_SAME_STATE_TIME_BUDGET_YIELDS = max(
        0,
        int(os.environ.get("ANALYZER_MAX_SAME_STATE_TIME_BUDGET_YIELDS", "2")),
    )
except ValueError:
    ANALYZER_MAX_SAME_STATE_TIME_BUDGET_YIELDS = 2
try:
    ANALYZER_RETRY_MAX_BACKOFF_SECONDS = max(
        ANALYZER_RETRY_BACKOFF_SECONDS,
        float(os.environ.get("ANALYZER_RETRY_MAX_BACKOFF_SECONDS", "16")),
    )
except ValueError:
    ANALYZER_RETRY_MAX_BACKOFF_SECONDS = 16.0
DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS = 120.0
_LOCAL_SERVER_PROCESS_ENV_KEYS = (
    "LOCAL_ANALYZER_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_ANALYZER_BASE_URL",
    "OPENAI_BASE_URL",
    "LOCAL_ANALYZER_PROVIDER",
    "OPENAI_PROVIDER",
)


@dataclass
class _LocalServerRuntime:
    index: int
    repo_dir: Path
    api_key_file: Path
    env_overrides: dict[str, str]
    base_url: str
    api_key: str = ""


def _analyzer_reported_tokens(analyzer: Any) -> int:
    value = (
        getattr(analyzer, "generated_tokens", None)
        if hasattr(analyzer, "generated_tokens")
        else getattr(analyzer, "total_tokens", 0)
    )
    return max(0, int(value or 0))


def artifact_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _grid_from_state(state: taaf.game.GameState | None) -> tuple[tuple[int, ...], ...]:
    if state is None:
        return ()
    data = state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _grid_from_data(data: Any) -> tuple[tuple[int, ...], ...]:
    rows = data.tolist() if hasattr(data, "tolist") else data
    if not isinstance(rows, (list, tuple)):
        return ()
    return tuple(
        tuple(int(cell) for cell in row)
        for row in rows
        if isinstance(row, (list, tuple))
    )


def _summarize_animation(
    before_grid: tuple[tuple[int, ...], ...],
    state: taaf.game.GameState,
) -> dict[str, Any]:
    """Compress action animation into bounded temporal evidence for the model."""
    raw_frames = getattr(getattr(state, "raw", None), "frame", None)
    frames = (
        [_grid_from_data(data) for data in raw_frames]
        if isinstance(raw_frames, (list, tuple))
        else []
    )
    frames = [frame for frame in frames if frame]
    if not frames:
        frames = [_grid_from_state(state)]

    changed_frame_count = 0
    total_changed_cells = 0
    peak_changed_cells = 0
    changed_coordinates: set[tuple[int, int]] = set()
    transition_counts: dict[tuple[int | None, int | None], int] = {}
    per_frame_changes: list[int] = []
    motion_centroids: list[tuple[float, float]] = []

    previous = before_grid
    for current in frames:
        rows = max(len(previous), len(current))
        cols = max(
            max((len(row) for row in previous), default=0),
            max((len(row) for row in current), default=0),
        )
        step_changes = 0
        step_coordinates: list[tuple[int, int]] = []
        for row in range(rows):
            for col in range(cols):
                old = (
                    previous[row][col]
                    if row < len(previous) and col < len(previous[row])
                    else None
                )
                new = (
                    current[row][col]
                    if row < len(current) and col < len(current[row])
                    else None
                )
                if old == new:
                    continue
                step_changes += 1
                step_coordinates.append((row, col))
                changed_coordinates.add((row, col))
                pair = (old, new)
                transition_counts[pair] = transition_counts.get(pair, 0) + 1
        if step_changes:
            changed_frame_count += 1
            total_changed_cells += step_changes
            peak_changed_cells = max(peak_changed_cells, step_changes)
            motion_centroids.append(
                (
                    sum(row for row, _ in step_coordinates) / step_changes,
                    sum(col for _, col in step_coordinates) / step_changes,
                )
            )
        per_frame_changes.append(step_changes)
        previous = current

    final_grid = frames[-1]
    final_changed_coordinates: set[tuple[int, int]] = set()
    rows = max(len(before_grid), len(final_grid))
    cols = max(
        max((len(row) for row in before_grid), default=0),
        max((len(row) for row in final_grid), default=0),
    )
    for row in range(rows):
        for col in range(cols):
            old = (
                before_grid[row][col]
                if row < len(before_grid) and col < len(before_grid[row])
                else None
            )
            new = (
                final_grid[row][col]
                if row < len(final_grid) and col < len(final_grid[row])
                else None
            )
            if old != new:
                final_changed_coordinates.add((row, col))

    def color_name(value: int | None) -> str:
        if value is None:
            return "outside"
        return ARC_COLOR_CHARS[max(0, min(15, int(value)))]

    dominant_transitions = [
        {"from": color_name(pair[0]), "to": color_name(pair[1]), "count": count}
        for pair, count in sorted(
            transition_counts.items(), key=lambda item: (-item[1], str(item[0]))
        )[:6]
    ]
    motion_bbox = None
    if changed_coordinates:
        changed_rows = [row for row, _ in changed_coordinates]
        changed_cols = [col for _, col in changed_coordinates]
        motion_bbox = [
            min(changed_rows),
            min(changed_cols),
            max(changed_rows),
            max(changed_cols),
        ]

    motion_vector = None
    motion_direction = None
    if len(motion_centroids) >= 2:
        delta_row = motion_centroids[-1][0] - motion_centroids[0][0]
        delta_col = motion_centroids[-1][1] - motion_centroids[0][1]
        motion_vector = [round(delta_row, 2), round(delta_col, 2)]
        vertical = "down" if delta_row > 0.25 else "up" if delta_row < -0.25 else ""
        horizontal = (
            "right" if delta_col > 0.25 else "left" if delta_col < -0.25 else ""
        )
        motion_direction = (
            "-".join(item for item in (vertical, horizontal) if item) or "stationary"
        )

    return {
        "frame_count": len(frames),
        "intermediate_frame_count": max(0, len(frames) - 1),
        "changed_frame_count": changed_frame_count,
        "total_changed_cells": total_changed_cells,
        "peak_changed_cells": peak_changed_cells,
        "final_changed_cells": len(final_changed_coordinates),
        "transient_changed_cells": len(changed_coordinates - final_changed_coordinates),
        "motion_bbox": motion_bbox,
        "dominant_color_transitions": dominant_transitions,
        "per_frame_changed_cells": per_frame_changes[:16],
        "motion_vector": motion_vector,
        "motion_direction": motion_direction,
        "temporally_reversible": bool(
            changed_coordinates and not final_changed_coordinates
        ),
    }


def _level_number(game: taaf.game.Game) -> int:
    state = game.current_state
    completed = int(state.levels_completed)
    if state.won:
        return max(1, int(game.number_of_levels))
    return max(1, min(int(game.number_of_levels), completed + 1))


def _decision_frame(
    game: taaf.game.Game, state: taaf.game.GameState, *, step: int
) -> Frame:
    """Build the canonical observable state used by history and live control."""
    return Frame(
        grid=_grid_from_state(state),
        step=step,
        level=_level_number(game),
        valid_actions=tuple(_engine_action_names(game)),
        engine_state=state.raw.state.name,
        score=int(state.levels_completed),
    )


def _engine_action_names(game: taaf.game.Game) -> list[str]:
    names: list[str] = []
    for action_id in game.current_state.available_actions:
        try:
            name = arcengine.GameAction.from_id(int(action_id)).name
        except Exception:
            continue
        if name == "RESET":
            continue
        if name not in names:
            names.append(name)
    return names


def _model_mouse_action_data(
    action_data: dict[str, Any] | None = None,
) -> dict[str, int]:
    data = action_data or {}
    return {"row": int(data.get("y", 0)), "col": int(data.get("x", 0))}


def _format_action_display(
    action_name: str, action_data: dict[str, Any] | None = None
) -> str:
    if action_name == "ACTION6":
        data = _model_mouse_action_data(action_data)
        return f"MOUSE(row={data['row']}, col={data['col']})"
    return to_model_action(action_name)


def _evaluate_strategy_prediction(
    prediction: dict[str, Any] | None, payload: dict[str, Any]
) -> dict[str, str] | None:
    if not isinstance(prediction, dict):
        return None
    test_action = normalize_action_key(prediction.get("test_action", ""))
    expected = str(prediction.get("expected_outcome") or "").strip().lower()
    actual_action = normalize_action_key(payload.get("action_display", ""))
    if (
        not test_action
        or not expected
        or not (
            actual_action == test_action
            or (test_action == "MOUSE" and action_family(actual_action) == "MOUSE")
        )
    ):
        return None
    if not payload.get("executed") or expected == "unknown":
        status = "inconclusive"
    else:
        matched = {
            "no_change": not bool(payload.get("board_changed")),
            "state_change": bool(payload.get("board_changed")),
            "new_state": bool(payload.get("novel_state")),
            "level_progress": bool(
                payload.get("level_completed")
                or payload.get("run_complete")
                or float(payload.get("reward") or 0.0) > 0.0
            ),
        }.get(expected)
        status = "supported" if matched else "contradicted"
    return {
        "status": status,
        "action": actual_action,
        "expected": expected,
        "actual": str(payload.get("outcome_class") or "unknown"),
    }


def _is_engine_game_over(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.GAME_OVER


def _is_run_complete(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.WIN


def _write_transcript_html(transcript_path: Path, html_path: Path, title: str) -> None:
    if not transcript_path.exists():
        return
    html_path.parent.mkdir(parents=True, exist_ok=True)
    text = transcript_path.read_text(encoding="utf-8")
    body = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{background:#1e1e1e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;"
        "padding:20px;max-width:1100px;margin:0 auto;line-height:1.4;}"
        "h1{color:#fff;}pre{white-space:pre-wrap;background:#111;padding:16px;border-radius:6px;"
        "border:1px solid #333;overflow:auto;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
        "</body></html>\n"
    )
    html_path.write_text(body, encoding="utf-8")


@dataclass
class _HarnessGameSession:
    solver: "HarnessSolver"
    game: taaf.game.Game
    analyzer: Any
    game_index: int
    pass_index: int
    state_path: Path
    transcript_path: Path
    analysis_html_relpath: str
    stop_event: threading.Event
    viewer_data_path: Path
    started_at: float = field(default_factory=time.monotonic)
    history_entries: list[HistoryEntry] = field(default_factory=list)
    viewer_events: list[dict[str, Any]] = field(default_factory=list)
    analysis_step: int = 0
    last_engine_action: str | None = None
    token_baseline: int = 0
    level_token_baseline: int = 0
    cycle_risk_streak: int = 0
    directional_guard_counts: dict[tuple[int, str], int] = field(default_factory=dict)
    directional_guard_totals: dict[int, int] = field(default_factory=dict)
    directional_blocked_actions: set[tuple[int, str]] = field(default_factory=set)
    controller_config: InferenceControllerConfig = field(
        default_factory=InferenceControllerConfig.from_env
    )
    _viewer_events_flushed: int = field(default=0, init=False, repr=False)

    def current_frame(self) -> Frame:
        return _decision_frame(
            self.game, self.game.current_state, step=self.action_count
        )

    def write_runtime_state(self) -> None:
        write_runtime_state(
            self.state_path,
            current_frame=self.current_frame(),
            history=self.history_entries,
        )

    def seed_initial_history(self) -> None:
        if not self.history_entries:
            self.history_entries.append(
                HistoryEntry(action="", frame=self.current_frame())
            )

    @property
    def action_count(self) -> int:
        run = getattr(self.game, "game_run", None)
        return len(run.history) if run is not None else 0

    def runtime_limit_reached(self) -> bool:
        if self.solver.max_runtime_s_per_game is None:
            return False
        return (
            time.monotonic() - self.started_at
        ) >= self.solver.max_runtime_s_per_game

    def level_action_limit_status(self) -> tuple[int, int, int] | None:
        """Return ``(level, used, limit)`` when the current level is capped."""
        run = self.game.game_run
        base_actions = getattr(run, "base_actions_per_level", None)
        actions = getattr(run, "actions_per_level", None)
        if run is None or not base_actions or not actions:
            return None
        level_index = max(
            0, min(int(getattr(run, "levels_completed", 0)), len(actions) - 1)
        )
        if level_index >= len(base_actions):
            return None
        limit = self.controller_config.level_action_limit(base_actions[level_index])
        if limit is None:
            return None
        return level_index + 1, int(actions[level_index]), limit

    def level_action_limit_reached(self) -> bool:
        status = self.level_action_limit_status()
        return status is not None and status[1] >= status[2]

    def level_no_progress_token_status(self) -> tuple[int, int, int] | None:
        limit = getattr(
            getattr(self, "controller_config", None),
            "level_no_progress_token_limit",
            0,
        )
        if limit <= 0:
            return None
        level = self.current_frame().level
        baseline = max(0, int(getattr(self, "level_token_baseline", 0) or 0))
        used = max(0, _analyzer_reported_tokens(self.analyzer) - baseline)
        return level, used, limit

    def level_no_progress_token_limit_reached(self) -> bool:
        status = self.level_no_progress_token_status()
        return status is not None and status[1] >= status[2]

    def cycle_stop_reached(self) -> bool:
        limit = getattr(getattr(self, "controller_config", None), "cycle_stop_limit", 0)
        return limit > 0 and getattr(self, "cycle_risk_streak", 0) >= limit

    def directional_action_blocked(self, level: int, action: str) -> bool:
        blocked = getattr(self, "directional_blocked_actions", set())
        return (int(level), action_family(action)) in blocked

    def register_directional_guard(
        self, level: int, action: str
    ) -> tuple[int, int, bool]:
        family = action_family(action)
        key = (int(level), family)
        counts = getattr(self, "directional_guard_counts", None)
        if counts is None:
            counts = self.directional_guard_counts = {}
        totals = getattr(self, "directional_guard_totals", None)
        if totals is None:
            totals = self.directional_guard_totals = {}
        blocked = getattr(self, "directional_blocked_actions", None)
        if blocked is None:
            blocked = self.directional_blocked_actions = set()
        counts[key] = counts.get(key, 0) + 1
        totals[key[0]] = totals.get(key[0], 0) + 1
        strike_limit = getattr(
            getattr(self, "controller_config", None),
            "directional_no_progress_strike_limit",
            0,
        )
        if strike_limit > 0 and counts[key] >= strike_limit:
            blocked.add(key)
        return counts[key], totals[key[0]], key in blocked

    def clear_directional_guards(self, level: int) -> None:
        level = int(level)
        counts = getattr(self, "directional_guard_counts", {})
        blocked = getattr(self, "directional_blocked_actions", set())
        for key in [key for key in counts if key[0] == level]:
            counts.pop(key, None)
        for key in [key for key in blocked if key[0] == level]:
            blocked.discard(key)
        getattr(self, "directional_guard_totals", {}).pop(level, None)

    def directional_no_progress_stop_status(self) -> tuple[int, int, int] | None:
        limit = getattr(
            getattr(self, "controller_config", None),
            "directional_no_progress_stop_limit",
            0,
        )
        if limit <= 0:
            return None
        level = self.current_frame().level
        total = getattr(self, "directional_guard_totals", {}).get(level, 0)
        return level, total, limit

    def directional_no_progress_stop_reached(self) -> bool:
        status = self.directional_no_progress_stop_status()
        return status is not None and status[1] >= status[2]

    def timing_payload(self) -> dict[str, float | None]:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        if self.solver.max_runtime_s_per_game is None:
            remaining = None
        else:
            remaining = max(0.0, self.solver.max_runtime_s_per_game - elapsed)
        return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": remaining}

    def request_timeout_seconds(self) -> float | None:
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        try:
            if configured is not None:
                candidates.append(float(configured))
        except (TypeError, ValueError):
            pass
        if self.solver.max_runtime_s_per_game is not None:
            remaining = self.timing_payload()["time_remaining_seconds"]
            if remaining is not None:
                candidates.append(float(remaining))
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is not None:
            candidates.append(soft_remaining)
        if not candidates:
            return None
        return max(0.1, min(candidates))

    def should_stop(self) -> bool:
        run = self.game.game_run
        if run is None or run.state != "playing":
            return True
        if self.stop_event.is_set():
            return True
        if _is_run_complete(self.game):
            return True
        if self.runtime_limit_reached():
            return True
        if self.level_action_limit_reached():
            return True
        if self.level_no_progress_token_limit_reached():
            return True
        if self.cycle_stop_reached():
            return True
        if self.directional_no_progress_stop_reached():
            return True
        if (
            self.solver.max_actions_per_game is not None
            and self.action_count >= self.solver.max_actions_per_game
        ):
            return True
        return False

    def play(self) -> None:
        run = self.game.game_run
        assert run is not None, "TAAF starts games before invoking the solver."
        run.solver_analysis_html = self.analysis_html_relpath
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.touch(exist_ok=True)
        self.token_baseline = _analyzer_reported_tokens(self.analyzer)
        self.level_token_baseline = self.token_baseline
        self.seed_initial_history()
        self.write_runtime_state()
        self._append_initial_viewer_event()
        self.write_viewer_payload()
        try:
            retry_analysis_step: int | None = None
            controller_only_reason: str | None = None
            consecutive_failures = 0
            no_action_state_id: str | None = None
            no_action_turns = 0
            time_budget_yield_key: tuple[str, int] | None = None
            time_budget_yields = 0

            def handle_no_action_turn() -> str:
                nonlocal no_action_state_id, no_action_turns
                state_id = frame_fingerprint(self.current_frame())
                if state_id == no_action_state_id:
                    no_action_turns += 1
                else:
                    no_action_state_id = state_id
                    no_action_turns = 1
                if no_action_turns < 2:
                    return "retry"
                if self._execute_controller_fallback("repeated_no_action_turn"):
                    no_action_state_id = None
                    no_action_turns = 0
                    return "executed"
                run.solver_note = (
                    "stopped: repeated analyzer turns produced no action at the same "
                    f"state; tokens={_analyzer_reported_tokens(self.analyzer)}"
                )
                return "stop"

            def handle_time_budget_yield(analysis_step: int) -> str:
                nonlocal time_budget_yield_key, time_budget_yields
                key = (frame_fingerprint(self.current_frame()), analysis_step)
                if key == time_budget_yield_key:
                    time_budget_yields += 1
                else:
                    time_budget_yield_key = key
                    time_budget_yields = 1
                if time_budget_yields <= ANALYZER_MAX_SAME_STATE_TIME_BUDGET_YIELDS:
                    return "retry"
                if self._execute_controller_fallback("repeated_turn_time_budget"):
                    time_budget_yield_key = None
                    time_budget_yields = 0
                    return "executed"
                # A controller rejection is not evidence that the game is
                # unsalvageable. Start a fresh analyzer turn so it receives a
                # new observe-plan-act prompt, while the existing no-progress
                # token ceiling remains the terminal safeguard.
                time_budget_yield_key = None
                time_budget_yields = 0
                return "rollover"

            while not self.should_stop():
                if (
                    _is_engine_game_over(self.game)
                    and self.last_engine_action != "RESET"
                ):
                    self._execute_auto_reset()
                    continue

                if controller_only_reason is not None:
                    if self._execute_controller_fallback(controller_only_reason):
                        continue
                    run.solver_note = (
                        f"stopped: {controller_only_reason}; "
                        f"tokens={_analyzer_reported_tokens(self.analyzer)}"
                    )
                    break

                if retry_analysis_step is None:
                    self.analysis_step += 1
                    analysis_step = self.analysis_step
                else:
                    analysis_step = retry_analysis_step

                self.write_runtime_state()
                transcript_before = self._read_transcript_bytes()
                try:
                    result = self.analyzer.analyze(
                        self.state_path,
                        self.action_count,
                        valid_actions=_engine_action_names(self.game),
                        step_env=self.step_env,
                        transcript_path=self.transcript_path,
                        analysis_step=analysis_step,
                        request_timeout_seconds=self.request_timeout_seconds(),
                        should_stop=self.should_stop,
                    )
                finally:
                    transcript_delta = self._transcript_delta_since(transcript_before)
                    if transcript_delta.strip():
                        self._append_analysis_viewer_event(
                            analysis_step, transcript_delta
                        )
                        self.write_viewer_payload()
                if result is None:
                    raise RuntimeError("Analyzer did not return a result.")
                if result.retryable_failure:
                    consecutive_failures += 1
                    if consecutive_failures >= ANALYZER_MAX_CONSECUTIVE_FAILURES:
                        run.solver_note = (
                            "stopped: analyzer circuit breaker opened after "
                            f"{consecutive_failures} consecutive failures: "
                            f"{getattr(result, 'failure_detail', '')}; "
                            f"tokens={_analyzer_reported_tokens(self.analyzer)}"
                        )
                        run.state = "gave_up"
                        break
                    retry_analysis_step = analysis_step
                    if self.should_stop():
                        break
                    time.sleep(
                        min(
                            ANALYZER_RETRY_MAX_BACKOFF_SECONDS,
                            ANALYZER_RETRY_BACKOFF_SECONDS
                            * (2 ** max(0, consecutive_failures - 1)),
                        )
                    )
                    continue

                retry_analysis_step = None
                failure_category = getattr(result, "failure_category", None)
                if failure_category in {"internal", "state_missing"}:
                    consecutive_failures += 1
                    if consecutive_failures >= ANALYZER_MAX_CONSECUTIVE_FAILURES:
                        run.solver_note = (
                            f"stopped: analyzer {failure_category} failure after "
                            f"{consecutive_failures} attempts: "
                            f"{getattr(result, 'failure_detail', '')}; "
                            f"tokens={_analyzer_reported_tokens(self.analyzer)}"
                        )
                        run.state = "gave_up"
                        break
                    retry_analysis_step = analysis_step
                    continue
                if failure_category == "tool_step_exhausted":
                    no_action_result = handle_no_action_turn()
                    if no_action_result == "executed":
                        retry_analysis_step = None
                        consecutive_failures = 0
                        continue
                    if no_action_result == "stop":
                        break
                    consecutive_failures += 1
                    if consecutive_failures >= ANALYZER_MAX_CONSECUTIVE_FAILURES:
                        run.solver_note = (
                            "stopped: analyzer circuit breaker opened after repeated "
                            "tool-step exhaustion; "
                            f"tokens={_analyzer_reported_tokens(self.analyzer)}"
                        )
                        run.state = "gave_up"
                        break
                    continue
                consecutive_failures = 0
                if getattr(result, "yielded_control", False):
                    yield_reason = getattr(result, "yield_reason", None)
                    if yield_reason == "game_token_budget":
                        controller_only_reason = "game_token_budget"
                        continue
                    if yield_reason == "turn_time_budget":
                        # Permit normal inspect-then-act continuation while
                        # bounding same-state reasoning loops. This counter is
                        # deliberately separate from genuine no-action turns.
                        time_yield_result = handle_time_budget_yield(analysis_step)
                        if time_yield_result == "executed":
                            no_action_state_id = None
                            no_action_turns = 0
                            retry_analysis_step = None
                            continue
                        if time_yield_result == "rollover":
                            retry_analysis_step = None
                            continue
                        retry_analysis_step = analysis_step
                        continue
                    no_action_result = handle_no_action_turn()
                    if no_action_result == "executed":
                        retry_analysis_step = None
                        continue
                    if no_action_result == "stop":
                        break
                    retry_analysis_step = analysis_step
                    continue
                if not result.step_executed:
                    no_action_result = handle_no_action_turn()
                    if no_action_result == "executed":
                        retry_analysis_step = None
                        continue
                    if no_action_result == "stop":
                        break
                    continue
                no_action_state_id = None
                no_action_turns = 0
                time_budget_yield_key = None
                time_budget_yields = 0
        except Exception as exc:
            if run.final_score is None:
                run.solver_note = f"error: {type(exc).__name__}: {exc}"
                if run.state == "playing":
                    run.state = "crashed"
                self._finish_if_needed()
        finally:
            total_tokens = _analyzer_reported_tokens(self.analyzer)
            if run.solver_note is None:
                level_limit = self.level_action_limit_status()
                if level_limit is not None and level_limit[1] >= level_limit[2]:
                    level, used, limit = level_limit
                    run.solver_note = (
                        f"stopped: level {level} action limit reached "
                        f"({used}/{limit}); tokens={total_tokens}"
                    )
                elif self.cycle_stop_reached():
                    run.solver_note = (
                        "stopped: consecutive cycle-risk limit reached "
                        f"({self.cycle_risk_streak}/"
                        f"{self.controller_config.cycle_stop_limit}); "
                        f"tokens={total_tokens}"
                    )
                elif self.directional_no_progress_stop_reached():
                    level, used, limit = self.directional_no_progress_stop_status()
                    run.solver_note = (
                        "stopped: directional no-progress guard limit reached "
                        f"on level {level} ({used}/{limit}); tokens={total_tokens}"
                    )
                elif self.level_no_progress_token_limit_reached():
                    level, used, limit = self.level_no_progress_token_status()
                    run.solver_note = (
                        "stopped: level no-progress analyzer token limit reached "
                        f"on level {level} ({used}/{limit}); tokens={total_tokens}"
                    )
                else:
                    run.solver_note = f"tokens={total_tokens}"
            self._finish_if_needed()
            self.state_path.unlink(missing_ok=True)
            self._write_analysis_html()
            self.write_viewer_payload()

    def _execute_controller_fallback(self, reason: str) -> bool:
        """Execute one empirically safe action when model control is unavailable."""
        if not self.controller_config.outcome_aware or self.should_stop():
            return False
        snapshot = self._controller_snapshot()
        candidates: list[dict[str, Any]] = []
        mouse_coordinates = list(
            (snapshot.get("mouse_search") or {}).get("recommended_coordinates") or ()
        )
        for ranked in snapshot.get("ranked_actions") or ():
            if not isinstance(ranked, dict) or ranked.get("harm_decisive"):
                continue
            action_name = str(ranked.get("action") or "")
            if action_name == "RESET":
                continue
            if action_name == "MOUSE":
                if not mouse_coordinates:
                    continue
                candidates.extend(
                    {
                        "action": "MOUSE",
                        "row": int(coordinate["row"]),
                        "col": int(coordinate["col"]),
                    }
                    for coordinate in mouse_coordinates
                )
                mouse_coordinates.clear()
            elif action_name:
                candidates.append({"action": action_name})
        reset_recommended = any(
            isinstance(item, dict) and item.get("strategy") == "reset_episode"
            for item in snapshot.get("recovery_portfolio") or ()
        )
        if reset_recommended:
            candidates.append({"action": "RESET"})
        recent_fallback = self._recent_no_progress_fallback_action()
        if recent_fallback:
            candidates.sort(
                key=lambda candidate: (
                    self._fallback_candidate_display(candidate) == recent_fallback
                )
            )
        rejections: list[dict[str, Any]] = []
        for candidate in candidates:
            payload = self.step_env(
                {
                    "actions": [candidate],
                    "controller_fallback_reason": str(reason),
                }
            )
            if payload.get("executed"):
                return True
            rejections.append(
                {
                    "action": self._fallback_candidate_display(candidate),
                    "error": str(payload.get("error") or ""),
                    "guarded": bool(payload.get("guarded")),
                    "guard_reason_code": str(payload.get("guard_reason_code") or ""),
                    "stop_reason": str(payload.get("stop_reason") or ""),
                    "stop_detail": str(payload.get("stop_detail") or "")[:500],
                }
            )
        self._record_controller_fallback_unavailable(reason, candidates, rejections)
        return False

    @staticmethod
    def _fallback_candidate_display(candidate: dict[str, Any]) -> str:
        action = str(candidate.get("action") or "")
        if action == "MOUSE":
            return f"MOUSE(row={candidate.get('row')}, col={candidate.get('col')})"
        return action

    def _recent_no_progress_fallback_action(self) -> str | None:
        for event in reversed(getattr(self, "viewer_events", ())):
            if event.get("type") != "action":
                continue
            if not event.get("controller_fallback_reason"):
                return None
            if event.get("level_completed") or float(event.get("reward") or 0.0) > 0:
                return None
            return str(event.get("action_display") or "") or None
        return None

    def _record_controller_fallback_unavailable(
        self,
        reason: str,
        candidates: list[dict[str, Any]],
        rejections: list[dict[str, Any]],
    ) -> None:
        current = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(current),
                "type": "controller",
                "title": "Fallback Unavailable",
                "action_num": self.action_count,
                "analysis_step": self.analysis_step,
                "controller_policy": self.controller_config.policy,
                "controller_phase": "recover",
                "controller_reason_codes": ["fallback_unavailable"],
                "controller_fallback_reason": str(reason),
                "fallback_candidate_count": len(candidates),
                "fallback_candidates": [
                    self._fallback_candidate_display(candidate)
                    for candidate in candidates
                ],
                "fallback_rejections": rejections,
            }
        )
        self.write_viewer_payload()

    def _controller_snapshot(self) -> dict[str, Any]:
        """Build host control state with independent prior-pass evidence."""
        external_transitions: list[dict[str, Any]] = []
        run = getattr(self.game, "game_run", None)
        solver = getattr(self, "solver", None)
        store = getattr(solver, "_knowledge_store", None)
        current_evidence_id = (
            f"{getattr(solver, '_knowledge_run_id', '')}:pass="
            f"{getattr(self, 'pass_index', 0)}"
        )
        if store is not None and run is not None:
            knowledge = store.snapshot(
                str(run.game_id),
                state_id=frame_fingerprint(self.current_frame()),
                exclude_evidence_id=current_evidence_id,
            )
            external_transitions = [
                dict(item)
                for item in knowledge.get("transition_records") or ()
                if isinstance(item, dict)
            ]
        return build_experience_snapshot(
            self.history_entries,
            self.current_frame(),
            _engine_action_names(self.game),
            self.controller_config,
            external_transitions=external_transitions,
            evidence_id=current_evidence_id,
        )

    def _finish_if_needed(self) -> None:
        run = self.game.game_run
        if run is not None and run.final_score is None:
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            current_tokens = _analyzer_reported_tokens(self.analyzer)
            final_generated_tokens = max(0, current_tokens - self.token_baseline)
            self.game.finish_game(
                generated_tokens=final_generated_tokens,
                uncached_input_tokens=0,
            )
            self.token_baseline = current_tokens

    def _write_analysis_html(self) -> None:
        if self.solver.job_dir is None:
            return
        _write_transcript_html(
            self.transcript_path,
            self.solver.job_dir / self.analysis_html_relpath,
            f"{self.game.game_run.game_id if self.game.game_run else self.game_index} analysis",
        )

    def _read_transcript_bytes(self) -> bytes:
        try:
            return self.transcript_path.read_bytes()
        except OSError:
            return b""

    def _transcript_delta_since(self, previous_transcript: bytes) -> str:
        try:
            current_size = self.transcript_path.stat().st_size
            previous_size = len(previous_transcript)
            with self.transcript_path.open("rb") as file:
                if current_size >= previous_size:
                    current_prefix = file.read(previous_size)
                    if current_prefix == previous_transcript:
                        return file.read().decode("utf-8", errors="replace").strip()
                    file.seek(0)
                return file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _base_viewer_event(self, frame: Frame) -> dict[str, Any]:
        run = self.game.game_run
        raw_state = self.game.current_state.raw.state
        return {
            "board": [list(row) for row in frame.grid],
            "board_ascii": frame.ascii,
            "state_id": frame_fingerprint(frame),
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "level": frame.level,
            "run_status": run.state if run is not None else "playing",
        }

    def _append_initial_viewer_event(self) -> None:
        if self.viewer_events:
            return
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "initial",
                "title": "Initial State",
                "action_num": self.action_count,
                "analysis_step": None,
                "action_display": "RESET",
                "reward": 0.0,
            }
        )

    def _append_analysis_viewer_event(
        self, analysis_step: int, transcript: str
    ) -> None:
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "analysis",
                "title": f"Analysis Step {analysis_step}",
                "action_num": self.action_count,
                "analysis_step": analysis_step,
                "transcript": transcript,
            }
        )

    def _append_action_viewer_event(
        self, payload: dict[str, Any], frame: Frame
    ) -> None:
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "action",
                "title": f"Action {int(payload.get('action_num') or self.action_count)}",
                "action_num": int(payload.get("action_num") or self.action_count),
                "analysis_step": self.analysis_step,
                "action_name": payload.get("action_name"),
                "action_display": payload.get("action_display"),
                "reward": payload.get("reward"),
                "board_changed": payload.get("board_changed"),
                "decision_context_changed": payload.get("decision_context_changed"),
                "done": payload.get("done"),
                "level_completed": payload.get("level_completed"),
                "game_over": payload.get("game_over"),
                "run_complete": payload.get("run_complete"),
                "batch_index": payload.get("batch_index"),
                "batch_size": payload.get("batch_size"),
                "before_state_id": payload.get("before_state_id"),
                "after_state_id": payload.get("after_state_id"),
                "state_context_version": payload.get("state_context_version"),
                "behavioral_before_state_id": payload.get("behavioral_before_state_id"),
                "behavioral_after_state_id": payload.get("behavioral_after_state_id"),
                "novel_state": payload.get("novel_state"),
                "outcome_class": payload.get("outcome_class"),
                "loop_detected": payload.get("loop_detected"),
                "cycle_risk": payload.get("cycle_risk"),
                "cycle_period": payload.get("cycle_period"),
                "controller_policy": payload.get("controller_policy"),
                "controller_phase": payload.get("controller_phase"),
                "controller_reason_codes": payload.get("controller_reason_codes"),
                "action_rank": payload.get("action_rank"),
                "action_rank_reason": payload.get("action_rank_reason"),
                "action_regime_adapted": payload.get("action_regime_adapted"),
                "prediction_result": payload.get("prediction_result"),
                "animation": payload.get("animation"),
                "recommended_plan_action": payload.get("recommended_plan_action"),
                "followed_recommended_plan": payload.get("followed_recommended_plan"),
                "recommended_plan_confidence": payload.get(
                    "recommended_plan_confidence"
                ),
                "recommended_plan_expected_utility": payload.get(
                    "recommended_plan_expected_utility"
                ),
                "recommended_plan_branch_count": payload.get(
                    "recommended_plan_branch_count"
                ),
                "recommended_plan_policy_ready": payload.get(
                    "recommended_plan_policy_ready"
                ),
                "controller_fallback_reason": payload.get("controller_fallback_reason"),
                "no_op_streak": payload.get("no_op_streak"),
                "behavioral_no_op_streak": payload.get("behavioral_no_op_streak"),
                "stagnation_actions": payload.get("stagnation_actions"),
            }
        )

    def write_viewer_payload(self) -> None:
        if self.solver.job_dir is None:
            return
        self.viewer_data_path.parent.mkdir(parents=True, exist_ok=True)
        run = self.game.game_run
        last_event_source = next(
            (
                event
                for event in reversed(self.viewer_events)
                if event.get("type") == "action"
            ),
            self.viewer_events[-1] if self.viewer_events else {},
        )
        last_event = dict(last_event_source)
        last_event.pop("board", None)
        last_event.pop("board_ascii", None)
        last_event.pop("transcript", None)
        payload = {
            "game_id": run.game_id if run is not None else str(self.game_index),
            "agent_name": self.solver.label,
            "status": run.state if run is not None else "playing",
            "pass_index": self.pass_index,
            "pass_label": str(self.pass_index),
            "eventCount": len(self.viewer_events),
            "lastEvent": last_event,
            "viewer_steps": [],
            "replay_url": self.analysis_html_relpath,
        }
        if run is not None:
            payload.update(
                {
                    "levels_completed": run.levels_completed,
                    "total_levels": run.number_of_levels,
                    "actions_per_level": list(run.actions_per_level),
                    "final_score": run.final_score,
                }
            )
        if self._viewer_events_flushed == 0:
            reset_raw_events_sidecar(self.viewer_data_path)
        append_raw_events_sidecar(
            self.viewer_data_path, self.viewer_events[self._viewer_events_flushed :]
        )
        self._viewer_events_flushed = len(self.viewer_events)
        tmp_path = self.viewer_data_path.with_suffix(
            f"{self.viewer_data_path.suffix}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.viewer_data_path)

    def _normalize_actions(
        self, arguments: dict[str, Any]
    ) -> tuple[list[arcengine.ActionInput] | None, str | None]:
        has_single = bool(str(arguments.get("action", "")).strip())
        has_batch = arguments.get("actions") is not None
        if has_single and has_batch:
            return None, "Use either `action` or `actions`, not both."

        if has_batch:
            raw_actions = arguments.get("actions")
            if not isinstance(raw_actions, list):
                return None, "`actions` must be a JSON array of action objects."
            if not raw_actions:
                return None, "`actions` must contain at least one action."
            if len(raw_actions) > MAX_ACTION_BATCH:
                return (
                    None,
                    f"`actions` may contain at most {MAX_ACTION_BATCH} actions.",
                )
        else:
            if not has_single:
                return None, "step_env requires `action` or `actions`."
            raw_actions = [
                {
                    "action": arguments.get("action"),
                    "row": arguments.get("row"),
                    "col": arguments.get("col"),
                }
            ]

        actions: list[arcengine.ActionInput] = []
        for index, raw_action in enumerate(raw_actions, start=1):
            if not isinstance(raw_action, dict):
                return None, f"Action {index} must be a JSON object."
            action_name = to_engine_action(raw_action.get("action"))
            if not action_name:
                return (
                    None,
                    f"Unknown action at index {index}: {raw_action.get('action')!r}",
                )
            action_id = arcengine.GameAction.from_name(action_name)
            data: dict[str, Any] = {}
            if action_id == arcengine.GameAction.ACTION6:
                try:
                    row = raw_action["row"]
                    column = raw_action["col"]
                    if any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in (row, column)
                    ):
                        raise TypeError
                    if not (0 <= row <= 63 and 0 <= column <= 63):
                        raise ValueError
                    data = {
                        "x": column,
                        "y": row,
                    }
                except (KeyError, TypeError, ValueError):
                    return (
                        None,
                        f"MOUSE action at index {index} requires integer row and col arguments between 0 and 63.",
                    )
            actions.append(arcengine.ActionInput(id=action_id, data=data))
        return actions, None

    def _error_payload(self, message: str) -> dict[str, Any]:
        return {
            "executed": False,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            **self.timing_payload(),
        }

    def _terminal_payload(
        self, requested_actions: list[arcengine.ActionInput]
    ) -> dict[str, Any]:
        raw_state = self.game.current_state.raw.state
        is_game_over = raw_state == arcengine.GameState.GAME_OVER
        is_win = raw_state == arcengine.GameState.WIN
        requested = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        stop_reason = (
            "run_complete" if is_win else "game_over" if is_game_over else "stopped"
        )
        return {
            "executed": False,
            "error": "No action was executed because the current game state is terminal or stopping.",
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "valid_actions": [],
            "board_changed": False,
            "done": is_win,
            "level_completed": False,
            "game_over": is_game_over,
            "run_complete": is_win,
            "batched": len(requested_actions) > 1,
            "requested_count": len(requested_actions),
            "executed_count": 0,
            "requested_actions": requested,
            "executed_actions": [],
            "stopped_early": True,
            "stop_reason": stop_reason,
            **self.timing_payload(),
        }

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        requested_actions, error = self._normalize_actions(arguments)
        if error is not None or requested_actions is None:
            return self._error_payload(error or "Could not parse action request.")
        if self.controller_config.outcome_aware:
            snapshot = self._controller_snapshot()
            action_budget = max(1, int(snapshot.get("action_budget") or 1))
            if len(requested_actions) > action_budget:
                phase = str(snapshot.get("phase") or "explore")
                payload = self._error_payload(
                    f"The {phase} phase permits at most {action_budget} action(s) "
                    "in one batch. Use a smaller falsifiable step."
                )
                payload.update(
                    {
                        "requested_count": len(requested_actions),
                        "executed_count": 0,
                        "controller_phase": phase,
                        "phase_action_budget": action_budget,
                    }
                )
                return payload
        if self.should_stop() or _is_engine_game_over(self.game):
            return self._terminal_payload(requested_actions)

        executed_payloads: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        stop_detail: str | None = None
        batch_size = len(requested_actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        strategy_prediction = arguments.get("strategy_prediction")
        if not isinstance(strategy_prediction, dict):
            strategy_prediction = None
        controller_fallback_reason = str(
            arguments.get("controller_fallback_reason") or ""
        )

        for batch_index, action in enumerate(requested_actions, start=1):
            if batch_index > 1 and self.controller_config.outcome_aware:
                current_snapshot = self._controller_snapshot()
                current_budget = max(1, int(current_snapshot.get("action_budget") or 1))
                if batch_index > current_budget:
                    stop_reason = "phase_action_budget"
                    stop_detail = (
                        "The controller phase changed and no longer supports the "
                        "remaining queued actions."
                    )
                    break
            if self.should_stop():
                stop_reason = "stopped"
                break
            if action.id.value not in self.game.current_state.available_actions:
                message = f"{_format_action_display(action.id.name, dict(action.data))} is not valid right now."
                if executed_payloads:
                    stop_reason = "invalid_action"
                    stop_detail = message
                    break
                return self._error_payload(message)

            action_display = _format_action_display(action.id.name, dict(action.data))
            current = self.current_frame()
            if self.directional_action_blocked(current.level, action_display):
                guard_reason_code = "directional_no_progress_persistent"
                guard_reason = (
                    "this direction is blocked for the current level after repeated "
                    "no-progress watchdog activations"
                )
            else:
                guard_reason = action_guard_reason(
                    self.history_entries,
                    current,
                    action_display,
                    self.controller_config,
                )
                guard_reason_code = action_guard_reason_code(
                    self.history_entries,
                    current,
                    action_display,
                    self.controller_config,
                )
            if guard_reason is None:
                cross_trial_harm = self._cross_trial_harm_reason(action_display)
                if cross_trial_harm is not None:
                    guard_reason = cross_trial_harm
                    guard_reason_code = "known_harmful_cross_trial"
            if guard_reason is not None:
                if guard_reason_code in {
                    "directional_no_progress",
                    "directional_no_progress_persistent",
                }:
                    strikes, total, persistently_blocked = (
                        self.register_directional_guard(current.level, action_display)
                    )
                    guard_reason = (
                        f"{guard_reason}; direction strikes={strikes}, "
                        f"level guard total={total}"
                    )
                    if persistently_blocked:
                        guard_reason += "; direction remains blocked until progress or reset"
                harm_guard = guard_reason_code in {
                    "known_harmful_local",
                    "known_harmful_cross_trial",
                }
                loop_guard = not harm_guard
                stop_reason = "loop_guard" if loop_guard else "harm_guard"
                stop_detail = guard_reason
                self.viewer_events.append(
                    {
                        **self._base_viewer_event(current),
                        "type": "controller",
                        "title": "Loop Guard" if loop_guard else "Harm Guard",
                        "action_num": self.action_count,
                        "analysis_step": self.analysis_step,
                        "action_display": action_display,
                        "guarded": True,
                        "guard_reason_code": guard_reason_code,
                        "loop_detected": loop_guard,
                        "controller_policy": self.controller_config.policy,
                        "controller_phase": "recover",
                        "controller_reason_codes": [guard_reason_code]
                        if guard_reason_code
                        else [],
                        "stop_reason": stop_reason,
                        "stop_detail": guard_reason,
                    }
                )
                self.write_viewer_payload()
                if executed_payloads:
                    break
                guarded_payload = {
                    "executed": False,
                    "action_num": self.action_count,
                    "level": current.level,
                    "score": int(self.game.current_state.levels_completed),
                    "reward": 0.0,
                    "state": self.game.current_state.raw.state.name,
                    "valid_actions": to_model_actions(_engine_action_names(self.game)),
                    "board_changed": False,
                    "done": False,
                    "level_completed": False,
                    "game_over": False,
                    "run_complete": False,
                    "guarded": True,
                    "guard_reason_code": guard_reason_code,
                    "loop_detected": loop_guard,
                    "controller_policy": self.controller_config.policy,
                    "controller_phase": "recover",
                    "controller_reason_codes": [guard_reason_code]
                    if guard_reason_code
                    else [],
                    "requested_count": batch_size,
                    "executed_count": 0,
                    "stopped_early": True,
                    "stop_reason": stop_reason,
                    "stop_detail": guard_reason,
                    "requested_actions": requested_displays,
                    "executed_actions": [],
                    "steps": [],
                    **self.timing_payload(),
                }
                return guarded_payload

            try:
                payload = self._execute_action(
                    action,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    strategy_prediction=strategy_prediction,
                    controller_fallback_reason=controller_fallback_reason,
                    flush_viewer_payload=False,
                )
            except Exception as exc:
                if executed_payloads:
                    stop_reason = "action_error"
                    stop_detail = f"{type(exc).__name__}: {exc}"
                    break
                return self._error_payload(f"{type(exc).__name__}: {exc}")
            executed_payloads.append(payload)
            total_reward += float(payload.get("reward", 0.0) or 0.0)
            if "prediction_result" in payload:
                strategy_prediction = None

            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                break

        if not executed_payloads:
            return self._error_payload("No action was executed.")

        final_payload = dict(executed_payloads[-1])
        final_payload["reward"] = total_reward
        final_payload["last_reward"] = executed_payloads[-1].get("reward", 0.0)
        final_payload["batched"] = batch_size > 1
        final_payload["requested_count"] = batch_size
        final_payload["executed_count"] = len(executed_payloads)
        final_payload["requested_actions"] = requested_displays
        final_payload["executed_actions"] = [
            str(item.get("action_display") or item.get("action_name") or "")
            for item in executed_payloads
        ]
        final_payload["steps"] = [
            {
                key: item.get(key)
                for key in (
                    "executed",
                    "action_num",
                    "level",
                    "score",
                    "state",
                    "action_display",
                    "before_state_id",
                    "after_state_id",
                    "behavioral_before_state_id",
                    "behavioral_after_state_id",
                    "object_before_state_id",
                    "object_after_state_id",
                    "state_context_version",
                    "valid_actions_before",
                    "valid_actions_after",
                    "board_changed",
                    "decision_context_changed",
                    "novel_state",
                    "outcome_class",
                    "reward",
                    "level_completed",
                    "game_over",
                    "run_complete",
                    "loop_detected",
                    "cycle_risk",
                    "cycle_risk_streak",
                    "cycle_period",
                    "controller_policy",
                    "controller_phase",
                    "controller_reason_codes",
                    "action_rank",
                    "action_rank_reason",
                    "prediction_result",
                    "animation",
                    "no_op_streak",
                    "behavioral_no_op_streak",
                    "stagnation_actions",
                )
                if key in item
            }
            for item in executed_payloads
        ]
        final_payload["board_changed"] = any(
            bool(item.get("board_changed")) for item in executed_payloads
        )
        final_payload["stopped_early"] = len(executed_payloads) < batch_size
        if stop_reason is not None:
            final_payload["stop_reason"] = stop_reason
            if stop_detail:
                final_payload["stop_detail"] = stop_detail
        self.write_viewer_payload()
        return final_payload

    def _cross_trial_harm_reason(self, action_display: str) -> str | None:
        if not self.controller_config.outcome_aware:
            return None
        store = getattr(self.solver, "_knowledge_store", None)
        run = self.game.game_run
        game_id = str(getattr(run, "game_id", "") or "")
        if store is None or not game_id:
            return None
        snapshot = store.snapshot(
            game_id, state_id=frame_fingerprint(self.current_frame())
        )
        action_key = normalize_action_key(action_display)
        for evidence in snapshot.get("state_action_evidence") or []:
            if not isinstance(evidence, dict):
                continue
            if normalize_action_key(evidence.get("action") or "") != action_key:
                continue
            outcomes = evidence.get("outcomes") or {}
            harmful_trials = int(outcomes.get("terminal_failure") or 0) + int(
                outcomes.get("negative_reward") or 0
            )
            total_trials = max(
                harmful_trials,
                int(evidence.get("trials") or 0),
                sum(int(count or 0) for count in outcomes.values()),
            )
            if harmful_evidence_is_decisive(harmful_trials, total_trials):
                return (
                    f"{action_key} has decisive harm evidence ({harmful_trials} of "
                    f"{total_trials} independent cross-trial observations): "
                    "terminal-failure or negative-reward observations in this state"
                )
        return None

    def _execute_auto_reset(self) -> None:
        action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
        self._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)

    def _execute_action(
        self,
        action: arcengine.ActionInput,
        *,
        batch_index: int,
        batch_size: int,
        strategy_prediction: dict[str, Any] | None = None,
        controller_fallback_reason: str = "",
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
    ) -> dict[str, Any]:
        previous_grid = _grid_from_state(self.game.current_state)
        previous_frame = self.current_frame()
        prior_history = list(self.history_entries)
        previous_valid_actions = to_model_actions(_engine_action_names(self.game))
        previous_completed = int(self.game.current_state.levels_completed)
        if generated_tokens is None:
            current_tokens = _analyzer_reported_tokens(self.analyzer)
            generated_tokens = max(0, current_tokens - self.token_baseline)
            self.token_baseline = current_tokens

        new_state = self.game.execute_action(
            action, generated_tokens=generated_tokens, uncached_input_tokens=0
        )
        self.last_engine_action = action.id.name
        action_display = _format_action_display(action.id.name, dict(action.data))
        completed = int(new_state.levels_completed)
        raw_state = new_state.raw.state
        animation = _summarize_animation(previous_grid, new_state)
        current_frame = _decision_frame(self.game, new_state, step=self.action_count)
        reward = float(completed - previous_completed) / max(
            1.0, float(self.game.number_of_levels)
        )
        if action.id == arcengine.GameAction.RESET or reward > 0.0:
            self.clear_directional_guards(previous_frame.level)
            self.level_token_baseline = _analyzer_reported_tokens(self.analyzer)
        self.history_entries.append(
            HistoryEntry(
                action=action_display,
                frame=current_frame,
                reward=reward,
                animation=animation,
                outcome_class_override=(
                    "terminal_failure"
                    if raw_state == arcengine.GameState.GAME_OVER
                    else ""
                ),
            )
        )
        self.write_runtime_state()

        board_changed = previous_grid != _grid_from_state(new_state)
        level_completed = bool(
            new_state.just_won_level and raw_state != arcengine.GameState.WIN
        )
        payload = {
            "executed": True,
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": completed,
            "reward": reward,
            "state": raw_state.name,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "board_changed": board_changed,
            "done": raw_state == arcengine.GameState.WIN,
            "level_completed": level_completed,
            "game_over": raw_state == arcengine.GameState.GAME_OVER,
            "run_complete": raw_state == arcengine.GameState.WIN,
            "action_name": action.id.name,
            "action_data": (
                _model_mouse_action_data(action.data)
                if action.id == arcengine.GameAction.ACTION6
                else dict(action.data)
            ),
            "action_display": action_display,
            "batch_index": batch_index,
            "batch_size": batch_size,
            "controller_fallback_reason": controller_fallback_reason or None,
            "animation": animation,
            **self.timing_payload(),
        }
        if self.controller_config.enabled:
            payload.update(
                transition_metadata(
                    previous_frame,
                    current_frame,
                    prior_history,
                    action_display,
                    self.controller_config,
                    previous_valid_actions,
                    reward=reward,
                    game_over=raw_state == arcengine.GameState.GAME_OVER,
                    run_complete=raw_state == arcengine.GameState.WIN,
                    next_valid_actions=to_model_actions(
                        _engine_action_names(self.game)
                    ),
                    animation=animation,
                )
            )
            if reward > 0.0 or level_completed or payload.get("run_complete"):
                self.cycle_risk_streak = 0
            elif payload.get("cycle_risk"):
                self.cycle_risk_streak += 1
            else:
                self.cycle_risk_streak = 0
            payload["cycle_risk_streak"] = self.cycle_risk_streak
        prediction_result = _evaluate_strategy_prediction(strategy_prediction, payload)
        if prediction_result is not None:
            payload["prediction_result"] = prediction_result
        self._append_action_viewer_event(payload, current_frame)
        if flush_viewer_payload:
            self.write_viewer_payload()
        return payload


@dataclass
class HarnessSolver(Solver):
    """Run the existing tool-using harness as a TAAF ``Solver``."""

    label: str = "HarnessSolver"
    model: str = ""
    analyzer_timeout: float | None = 120.0
    max_actions_per_game: int | None = None
    max_runtime_s_per_game: float | None = None
    concurrency: int = 16
    save_request_logs: bool = False
    start_local_server: bool = False
    local_server_config: str = ""
    local_server_api_key_file: str = ""
    local_server_repo_dir: str = ""
    local_server_port: int | None = None
    local_server_tensor_parallel_size: int | None = None
    local_server_count: int = 1
    kaggle_enable_vllm: bool = field(default=True, repr=False)
    kaggle_wheelhouse_dataset_source: str = field(
        default=DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE, repr=False
    )
    kaggle_model_source: str = field(
        default=DEFAULT_QWEN_MODEL_SOURCE, repr=False
    )
    kaggle_served_model_name: str = field(default=DEFAULT_SERVED_MODEL_NAME, repr=False)
    kaggle_vllm_port: int = field(default=DEFAULT_VLLM_PORT, repr=False)
    kaggle_vllm_max_model_len: int = field(
        default=DEFAULT_VLLM_MAX_MODEL_LEN, repr=False
    )
    kaggle_vllm_tensor_parallel_size: int = field(
        default=DEFAULT_VLLM_TENSOR_PARALLEL_SIZE, repr=False
    )
    kaggle_vllm_gpu_memory_utilization: float = field(
        default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION, repr=False
    )
    kaggle_vllm_max_num_seqs: int = field(default=DEFAULT_VLLM_MAX_NUM_SEQS, repr=False)
    kaggle_vllm_max_num_batched_tokens: int = field(
        default=DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS, repr=False
    )
    kaggle_vllm_enable_chunked_prefill: bool = field(
        default=DEFAULT_VLLM_ENABLE_CHUNKED_PREFILL, repr=False
    )
    kaggle_expected_gpu_type: str = field(default=DEFAULT_EXPECTED_GPU_TYPE, repr=False)
    kaggle_expected_gpu_count: int = field(
        default=DEFAULT_EXPECTED_GPU_COUNT, repr=False
    )
    kaggle_wheelhouse_stamp_text: str = field(
        default=DEFAULT_WHEELHOUSE_STAMP_TEXT, repr=False
    )
    cancel_drain_timeout_s: float = DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS
    analyzer_factory: AnalyzerFactory | None = field(
        default=None, repr=False, compare=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _local_server_started: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _local_server_env_overrides: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _local_server_cwd: str = field(default="", init=False, repr=False, compare=False)
    _local_server_api_key: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_server_base_url: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_servers: list[_LocalServerRuntime] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    _local_server_original_env: dict[str, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    # Custom pool sized to self.concurrency: asyncio.to_thread routes onto
    # Python's default executor, capped at min(32, cpu+4) — which would
    # silently cap real concurrency below self.concurrency.
    _worker_pool: ThreadPoolExecutor | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _knowledge_store: TrialKnowledgeStore = field(
        default_factory=TrialKnowledgeStore, init=False, repr=False, compare=False
    )
    _knowledge_run_id: str = field(default="", init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["analyzer_factory"] = None
        state.pop("_stop_event", None)
        state.pop("_local_server_started", None)
        state.pop("_local_server_env_overrides", None)
        state.pop("_local_server_cwd", None)
        state.pop("_local_server_api_key", None)
        state.pop("_local_server_base_url", None)
        state.pop("_local_servers", None)
        state.pop("_local_server_original_env", None)
        state.pop("_worker_pool", None)
        state.pop("_knowledge_store", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._local_server_started = False
        self._local_server_env_overrides = {}
        self._local_server_cwd = ""
        self._local_server_api_key = ""
        self._local_server_base_url = ""
        self._local_servers = []
        self._local_server_original_env = {}
        self._worker_pool = None
        self._knowledge_store = TrialKnowledgeStore()

    def __deepcopy__(self, memo: dict[int, Any]) -> "HarnessSolver":
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_stop_event":
                object.__setattr__(new, key, threading.Event())
            elif key == "analyzer_factory":
                object.__setattr__(new, key, value)
            elif key == "_local_servers":
                object.__setattr__(new, key, [])
            elif key == "_local_server_original_env":
                object.__setattr__(new, key, {})
            elif key == "_worker_pool":
                object.__setattr__(new, key, None)
            elif key == "_knowledge_store":
                object.__setattr__(new, key, TrialKnowledgeStore())
            else:
                object.__setattr__(new, key, copy.deepcopy(value, memo))
        return new

    @property
    def kaggle_dataset_sources(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return duck_kaggle_dataset_sources(self._kaggle_vllm_config())

    @property
    def kaggle_model_sources(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return duck_kaggle_model_sources(self._kaggle_vllm_config())

    @property
    def kaggle_setup_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_setup_command(self._kaggle_vllm_config())]

    @property
    def kaggle_teardown_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_teardown_command()]

    def _kaggle_vllm_config(self) -> DuckKaggleVllmConfig:
        return DuckKaggleVllmConfig(
            wheelhouse_dataset_source=self.kaggle_wheelhouse_dataset_source,
            model_source=self.kaggle_model_source,
            served_model_name=self.kaggle_served_model_name,
            vllm_port=self.kaggle_vllm_port,
            max_model_len=self.kaggle_vllm_max_model_len,
            tensor_parallel_size=self.kaggle_vllm_tensor_parallel_size,
            gpu_memory_utilization=self.kaggle_vllm_gpu_memory_utilization,
            max_num_seqs=self.kaggle_vllm_max_num_seqs,
            max_num_batched_tokens=self.kaggle_vllm_max_num_batched_tokens,
            enable_chunked_prefill=self.kaggle_vllm_enable_chunked_prefill,
            expected_gpu_type=self.kaggle_expected_gpu_type,
            expected_gpu_count=self.kaggle_expected_gpu_count,
            wheelhouse_stamp_text=self.kaggle_wheelhouse_stamp_text,
        )

    def _setup(self) -> None:
        self._knowledge_run_id = (
            Path(self.job_dir).name
            if self.job_dir is not None
            else f"session-{id(self)}"
        )
        knowledge_path = os.environ.get("LOCAL_ANALYZER_KNOWLEDGE_PATH", "").strip()
        if not knowledge_path and self.job_dir is not None:
            knowledge_path = str(Path(self.job_dir) / "cross_trial_knowledge.json")
        if knowledge_path:
            self._knowledge_store.configure_path(Path(knowledge_path))
        if self.start_local_server:
            self._start_local_servers()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.concurrency)),
            thread_name_prefix="harness-game",
        )

    def _teardown(self) -> None:
        if self._local_server_started:
            self._stop_local_servers()
        if self._worker_pool is not None:
            self._worker_pool.shutdown(wait=False)
            self._worker_pool = None

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        self._stop_event.clear()
        semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))
        pass_indices_by_game_id: dict[str, int] = {}
        game_locks: dict[str, asyncio.Lock] = {}
        loop = asyncio.get_running_loop()
        pool = self._worker_pool

        async def run_one(
            index: int, pass_index: int, game_id: str, game: taaf.game.Game
        ) -> None:
            # Passes of one game are intentionally ordered so later passes can
            # consume verified mechanics without reducing cross-game parallelism.
            async with game_locks.setdefault(game_id, asyncio.Lock()):
                async with semaphore:
                    args = (
                        game,
                        index,
                        pass_index,
                        self._local_server_for_game_index(index),
                    )
                    if pool is not None:
                        await loop.run_in_executor(
                            pool, functools.partial(self._play_one, *args)
                        )
                    else:
                        # _setup wasn't called (direct test invocation).
                        await asyncio.to_thread(self._play_one, *args)

        tasks: list[asyncio.Task[None]] = []
        for index, game in enumerate(games):
            game_id = game.game_run.game_id if game.game_run is not None else str(index)
            pass_index = pass_indices_by_game_id.get(game_id, 0)
            pass_indices_by_game_id[game_id] = pass_index + 1
            tasks.append(asyncio.create_task(run_one(index, pass_index, game_id, game)))
        try:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )
        except asyncio.CancelledError:
            self._stop_event.set()
            await self._drain_game_tasks(tasks)
            self._finish_remaining(games)
            raise

    async def _drain_game_tasks(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        timeout = max(0.0, float(self.cancel_drain_timeout_s))
        if timeout == 0.0:
            return
        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        if done:
            await asyncio.gather(*done, return_exceptions=True)

    def _start_local_servers(self) -> None:
        server_count = self._resolved_local_server_count()
        started: list[_LocalServerRuntime] = []
        self._capture_local_server_process_env()
        try:
            for server_index in range(server_count):
                runtime = self._local_server_settings(
                    server_index=server_index, server_count=server_count
                )
                print(
                    "Starting local inference server inside solver setup "
                    f"(server {server_index + 1}/{server_count})"
                )
                subprocess.run(
                    ["make", "server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=True,
                )
                if runtime.api_key_file.is_file():
                    runtime.api_key = runtime.api_key_file.read_text(
                        encoding="utf-8"
                    ).strip()
                started.append(runtime)
        except Exception:
            self._local_servers = started
            self._local_server_started = bool(started)
            with contextlib.suppress(Exception):
                self._stop_local_servers()
            raise

        self._local_servers = started
        self._local_server_started = bool(started)
        if started:
            first = started[0]
            self._local_server_cwd = str(first.repo_dir)
            self._local_server_env_overrides = first.env_overrides
            self._local_server_api_key = first.api_key
            self._local_server_base_url = first.base_url
            if first.api_key:
                os.environ["LOCAL_ANALYZER_API_KEY"] = first.api_key
                os.environ["OPENAI_API_KEY"] = first.api_key
            if first.base_url:
                os.environ["LOCAL_ANALYZER_BASE_URL"] = first.base_url
                os.environ["OPENAI_BASE_URL"] = first.base_url
                os.environ["LOCAL_ANALYZER_PROVIDER"] = "vllm"
                os.environ["OPENAI_PROVIDER"] = "vllm"

    def _stop_local_servers(self) -> None:
        runtimes = list(reversed(self._local_servers))
        if not runtimes and self._local_server_env_overrides:
            repo_dir = (
                Path(self._local_server_cwd)
                if self._local_server_cwd
                else self._local_server_repo_dir()
            )
            runtimes = [
                _LocalServerRuntime(
                    index=0,
                    repo_dir=repo_dir,
                    api_key_file=Path(
                        self._local_server_env_overrides.get("SERVER_API_KEY_FILE", "")
                    ),
                    env_overrides=self._local_server_env_overrides,
                    base_url=self._local_server_base_url,
                    api_key=self._local_server_api_key,
                )
            ]
        try:
            for runtime in runtimes:
                subprocess.run(
                    ["make", "stop-server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=False,
                )
        finally:
            self._local_servers = []
            self._local_server_started = False
            self._restore_local_server_process_env()

    def _capture_local_server_process_env(self) -> None:
        self._local_server_original_env = {
            key: os.environ.get(key) for key in _LOCAL_SERVER_PROCESS_ENV_KEYS
        }

    def _restore_local_server_process_env(self) -> None:
        if not self._local_server_original_env:
            return
        for key, value in self._local_server_original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._local_server_original_env = {}

    def _local_server_settings(
        self, *, server_index: int, server_count: int
    ) -> _LocalServerRuntime:
        config_path = self.local_server_config.strip()
        if not config_path:
            raise ValueError(
                "local_server_config is required when start_local_server is enabled."
            )

        repo_dir = self._local_server_repo_dir()
        run_dir = (self.job_dir or Path.cwd()).resolve()
        api_key_file = self._local_server_api_key_path(
            server_index=server_index,
            server_count=server_count,
            run_dir=run_dir,
        )
        pid_path = run_dir / (
            "server.pid" if server_count <= 1 else f"server-{server_index}.pid"
        )
        log_path = run_dir / (
            "server.log" if server_count <= 1 else f"server-{server_index}.log"
        )
        port = self._local_server_port(server_index=server_index)
        base_url = f"http://127.0.0.1:{port}/v1" if port is not None else ""
        env_overrides = {
            "CONFIG_PATH": config_path,
            "SERVER_API_KEY_FILE": str(api_key_file),
            "SERVER_PID": str(pid_path),
            "SERVER_LOG": str(log_path),
            "SERVER_TAIL_ON_WAIT": "true",
            "UV_PROJECT_ENVIRONMENT": str(repo_dir / ".venv"),
        }
        venv_python = self._local_server_venv_python(repo_dir)
        if venv_python is not None:
            env_overrides["SERVER_VENV_PYTHON"] = str(venv_python)
            env_overrides["PYTHON"] = str(venv_python)
        if port is not None:
            env_overrides.update(
                {
                    "SERVER_PORT": str(port),
                    "LOCAL_ANALYZER_BASE_URL": base_url,
                    "OPENAI_BASE_URL": base_url,
                    "LOCAL_ANALYZER_PROVIDER": "vllm",
                    "OPENAI_PROVIDER": "vllm",
                }
            )
        if self.local_server_tensor_parallel_size is not None:
            env_overrides["SERVER_TENSOR_PARALLEL_SIZE"] = str(
                int(self.local_server_tensor_parallel_size)
            )
        if server_count > 1:
            env_overrides["CUDA_VISIBLE_DEVICES"] = (
                self._cuda_visible_device_for_server(server_index)
            )
        return _LocalServerRuntime(
            index=server_index,
            repo_dir=repo_dir,
            api_key_file=api_key_file,
            env_overrides=env_overrides,
            base_url=base_url,
        )

    def _resolved_local_server_count(self) -> int:
        if not self.start_local_server:
            return 0
        return max(1, int(self.local_server_count or 1))

    def _local_server_port(self, *, server_index: int) -> int | None:
        if self.local_server_port is None:
            return None
        return int(self.local_server_port) + int(server_index)

    def _local_server_api_key_path(
        self, *, server_index: int, server_count: int, run_dir: Path
    ) -> Path:
        default_name = (
            "server-api-key" if server_count <= 1 else f"server-{server_index}-api-key"
        )
        base_path = self._resolve_local_server_path(
            self.local_server_api_key_file, default=run_dir / default_name
        )
        if server_count <= 1 or not str(self.local_server_api_key_file or "").strip():
            return base_path
        suffix = base_path.suffix
        stem = base_path.name[: -len(suffix)] if suffix else base_path.name
        return base_path.with_name(f"{stem}-{server_index}{suffix}")

    def _cuda_visible_device_for_server(self, server_index: int) -> str:
        visible_devices = [
            device.strip()
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        if server_index < len(visible_devices):
            return visible_devices[server_index]
        return str(server_index)

    def _local_server_repo_dir(self) -> Path:
        repo_dir = (
            Path(self.local_server_repo_dir).expanduser()
            if self.local_server_repo_dir
            else Path(__file__).parents[2]
        )
        repo_dir = repo_dir.resolve()
        if not repo_dir.is_dir():
            raise ValueError(f"local_server_repo_dir does not exist: {repo_dir}")
        return repo_dir

    def _local_server_venv_python(self, repo_dir: Path) -> Path | None:
        repo_venv_python = repo_dir / ".venv" / "bin" / "python"
        if repo_venv_python.is_file():
            return repo_venv_python
        return None

    def _resolve_local_server_path(self, raw_value: str, *, default: Path) -> Path:
        raw = str(raw_value or "").strip()
        if not raw:
            return default.resolve()
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        return (self._local_server_repo_dir() / path).resolve()

    def _local_server_env(
        self, overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(overrides or self._local_server_env_overrides)
        return env

    def soft_time_remaining_seconds(self) -> float | None:
        if self.soft_end_time is None:
            return None
        now = (
            datetime.now(self.soft_end_time.tzinfo)
            if self.soft_end_time.tzinfo
            else datetime.now()
        )
        return max(0.0, (self.soft_end_time - now).total_seconds())

    def _local_server_for_game_index(
        self, game_index: int
    ) -> _LocalServerRuntime | None:
        if not self._local_servers:
            return None
        return self._local_servers[int(game_index) % len(self._local_servers)]

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
        *,
        pass_index: int = 0,
    ) -> Any:
        if self.analyzer_factory is not None:
            return self.analyzer_factory(game, index)
        return ToolAgent(
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            api_key=(
                local_server.api_key
                if local_server is not None
                else self._local_server_api_key
            )
            or None,
            base_url=(
                local_server.base_url
                if local_server is not None
                else self._local_server_base_url
            )
            or None,
            provider="vllm" if local_server is not None else None,
            knowledge_store=self._knowledge_store,
            game_id=(
                game.game_run.game_id if game.game_run is not None else str(index)
            ),
            pass_index=pass_index,
            evidence_id=f"{self._knowledge_run_id}:pass={pass_index}",
        )

    def _play_one(
        self,
        game: taaf.game.Game,
        index: int,
        pass_index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> None:
        try:
            assert game.game_run is not None
            run = game.game_run
            run_stem = self._run_stem(run.game_id, pass_index)
            state_path = self._artifacts_dir() / f"{run_stem}_{RUNTIME_STATE_FILENAME}"
            viewer_data_path = self._artifacts_dir() / f"{run_stem}_viewer_data.json"
            transcript_path = self._transcripts_dir() / f"{run_stem}.txt"
            analysis_relpath = f"solver_analysis/{run_stem}.html"
            analyzer = self._make_analyzer(
                game, index, local_server, pass_index=pass_index
            )
            session = _HarnessGameSession(
                solver=self,
                game=game,
                analyzer=analyzer,
                game_index=index,
                pass_index=pass_index,
                state_path=state_path,
                transcript_path=transcript_path,
                analysis_html_relpath=analysis_relpath,
                stop_event=self._stop_event,
                viewer_data_path=viewer_data_path,
            )
            session.play()
        except Exception as exc:
            self._finish_after_error(game, exc)

    def _artifacts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _transcripts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_stem(self, game_id: str, index: int) -> str:
        return f"{artifact_stem(game_id)}_p{index}"

    def _finish_remaining(self, games: list[taaf.game.Game]) -> None:
        for game in games:
            run = game.game_run
            if run is not None and run.final_score is None:
                try:
                    if self._stop_event.is_set() and run.state == "playing":
                        run.state = "cancelled"
                    game.finish_game()
                except Exception:
                    pass

    def _finish_after_error(self, game: taaf.game.Game, exc: Exception) -> None:
        run = game.game_run
        if run is None or run.final_score is not None:
            return
        run.solver_note = f"error: {type(exc).__name__}: {exc}"
        if run.state == "playing":
            run.state = "crashed"
        with contextlib.suppress(Exception):
            game.finish_game()
        if run.final_score is None:
            with contextlib.suppress(Exception):
                run.final_score = run._compute_final_score()
