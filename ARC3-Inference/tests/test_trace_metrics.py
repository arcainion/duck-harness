from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from inference.tools.trace_metrics import summarize_run_traces


class TraceMetricsTests(TestCase):
    def test_streaming_summary_counts_noops_phases_and_guards(self) -> None:
        events = [
            {"type": "initial", "board": [[0]]},
            {
                "type": "action",
                "action_display": "LEFT",
                "board_changed": False,
                "reward": 0,
                "after_state_id": "a",
                "behavioral_after_state_id": "stable-a",
                "outcome_class": "exact_noop",
                "controller_phase": "explore",
            },
            {
                "type": "action",
                "action_display": "LEFT",
                "board_changed": False,
                "reward": 0,
                "after_state_id": "a",
                "behavioral_after_state_id": "stable-a",
                "outcome_class": "volatile_only",
                "controller_phase": "recover",
            },
            {
                "type": "controller",
                "guarded": True,
                "stop_reason": "loop_guard",
                "guard_reason_code": "repeated_exact_noop",
                "after_state_id": "a",
            },
            {
                "type": "action",
                "action_display": "RIGHT",
                "board_changed": True,
                "reward": 0.5,
                "after_state_id": "b",
                "behavioral_after_state_id": "stable-b",
                "outcome_class": "level_progress",
                "prediction_result": {"status": "supported"},
                "controller_phase": "progress",
                "animation": {
                    "intermediate_frame_count": 2,
                    "transient_changed_cells": 3,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir) / "artifacts"
            artifacts.mkdir()
            path = artifacts / "demo-game_p0_events.jsonl"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            summary = summarize_run_traces(Path(temp_dir))["overall"]

        self.assertEqual(summary["actions"], 3)
        self.assertEqual(summary["no_op_actions"], 2)
        self.assertEqual(summary["repeated_no_ops"], 1)
        self.assertEqual(summary["rewarding_actions"], 1)
        self.assertEqual(summary["multi_frame_actions"], 1)
        self.assertEqual(summary["transient_animation_actions"], 1)
        self.assertEqual(summary["unique_states_observed"], 3)
        self.assertEqual(summary["unique_behavioral_states_observed"], 2)
        self.assertEqual(summary["loop_interventions"], 1)
        self.assertEqual(
            summary["phase_counts"], {"explore": 1, "progress": 1, "recover": 1}
        )
        self.assertEqual(
            summary["outcome_counts"],
            {"exact_noop": 1, "level_progress": 1, "volatile_only": 1},
        )
        self.assertEqual(summary["prediction_counts"], {"supported": 1})
        self.assertEqual(
            summary["guard_reason_counts"], {"repeated_exact_noop": 1}
        )

    def test_empty_events_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir) / "artifacts"
            artifacts.mkdir()
            path = artifacts / "empty-game_p0_events.jsonl"
            path.write_text("", encoding="utf-8")
            summary = summarize_run_traces(Path(temp_dir))["overall"]
        self.assertEqual(summary["actions"], 0)
        self.assertEqual(summary["unique_states_observed"], 0)

    def test_initial_only_no_actions(self) -> None:
        events = [{"type": "initial", "board": [[0]]}]
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir) / "artifacts"
            artifacts.mkdir()
            path = artifacts / "init-game_p0_events.jsonl"
            path.write_text(
                "".join(json.dumps(e) + "\n" for e in events),
                encoding="utf-8",
            )
            summary = summarize_run_traces(Path(temp_dir))["overall"]
        self.assertEqual(summary["actions"], 0)

    def test_all_noop_actions(self) -> None:
        events = [
            {"type": "initial", "board": [[0]]},
        ] + [
            {
                "type": "action",
                "action_display": "LEFT",
                "board_changed": False,
                "reward": 0,
                "after_state_id": f"a{i}",
                "behavioral_after_state_id": "stable-a",
                "outcome_class": "exact_noop",
                "controller_phase": "explore",
            }
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir) / "artifacts"
            artifacts.mkdir()
            path = artifacts / "noop-game_p0_events.jsonl"
            path.write_text(
                "".join(json.dumps(e) + "\n" for e in events),
                encoding="utf-8",
            )
            summary = summarize_run_traces(Path(temp_dir))["overall"]
        self.assertEqual(summary["actions"], 5)
        self.assertEqual(summary["no_op_actions"], 5)

    def test_multiple_games_combined(self) -> None:
        events1 = [
            {"type": "initial", "board": [[0]]},
            {
                "type": "action",
                "action_display": "LEFT",
                "board_changed": True,
                "reward": 1,
                "after_state_id": "b",
                "behavioral_after_state_id": "stable-b",
                "outcome_class": "state_change",
                "controller_phase": "explore",
            },
        ]
        events2 = [
            {"type": "initial", "board": [[1]]},
            {
                "type": "action",
                "action_display": "RIGHT",
                "board_changed": False,
                "reward": 0,
                "after_state_id": "c",
                "behavioral_after_state_id": "stable-c",
                "outcome_class": "exact_noop",
                "controller_phase": "orient",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir) / "artifacts"
            artifacts.mkdir()
            (artifacts / "game1_p0_events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events1),
                encoding="utf-8",
            )
            (artifacts / "game2_p0_events.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events2),
                encoding="utf-8",
            )
            summary = summarize_run_traces(Path(temp_dir))["overall"]
        self.assertEqual(summary["actions"], 2)
        self.assertEqual(summary["no_op_actions"], 1)
        self.assertEqual(summary["rewarding_actions"], 1)

    def test_malformed_event_line_raises_error(self) -> None:
        events = [
            {"type": "initial", "board": [[0]]},
            "not a json line",
            {
                "type": "action",
                "action_display": "UP",
                "board_changed": True,
                "reward": 0.5,
                "after_state_id": "d",
                "behavioral_after_state_id": "stable-d",
                "outcome_class": "level_progress",
                "controller_phase": "progress",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir) / "artifacts"
            artifacts.mkdir()
            path = artifacts / "mixed-game_p0_events.jsonl"
            lines = []
            for e in events:
                if isinstance(e, dict):
                    lines.append(json.dumps(e))
                else:
                    lines.append(str(e))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize_run_traces(Path(temp_dir))

    def test_actions_without_outcome_class(self) -> None:
        events = [
            {"type": "initial", "board": [[0]]},
            {
                "type": "action",
                "action_display": "LEFT",
                "board_changed": True,
                "reward": 0,
                "after_state_id": "e",
                "behavioral_after_state_id": "stable-e",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir) / "artifacts"
            artifacts.mkdir()
            path = artifacts / "no-outcome-game_p0_events.jsonl"
            path.write_text(
                "".join(json.dumps(e) + "\n" for e in events),
                encoding="utf-8",
            )
            summary = summarize_run_traces(Path(temp_dir))["overall"]
        self.assertEqual(summary["actions"], 1)

    def test_controller_events_counted(self) -> None:
        events = [
            {"type": "initial", "board": [[0]]},
            {
                "type": "controller",
                "guarded": True,
                "stop_reason": "loop_guard",
                "guard_reason_code": "repeated_exact_noop",
                "after_state_id": "f",
            },
            {
                "type": "controller",
                "guarded": True,
                "stop_reason": "stagnation_guard",
                "guard_reason_code": "stagnation",
                "after_state_id": "f",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = Path(temp_dir) / "artifacts"
            artifacts.mkdir()
            path = artifacts / "ctrl-game_p0_events.jsonl"
            path.write_text(
                "".join(json.dumps(e) + "\n" for e in events),
                encoding="utf-8",
            )
            summary = summarize_run_traces(Path(temp_dir))["overall"]
        self.assertEqual(summary["loop_interventions"], 2)
        self.assertEqual(
            summary["guard_reason_counts"],
            {"repeated_exact_noop": 1, "stagnation": 1},
        )

    def test_no_artifacts_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = summarize_run_traces(Path(temp_dir))["overall"]
        self.assertEqual(summary["actions"], 0)
