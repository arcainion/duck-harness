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
