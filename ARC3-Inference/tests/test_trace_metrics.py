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
                "controller_phase": "explore",
            },
            {
                "type": "action",
                "action_display": "LEFT",
                "board_changed": False,
                "reward": 0,
                "after_state_id": "a",
                "controller_phase": "recover",
            },
            {
                "type": "controller",
                "guarded": True,
                "stop_reason": "loop_guard",
                "after_state_id": "a",
            },
            {
                "type": "action",
                "action_display": "RIGHT",
                "board_changed": True,
                "reward": 0.5,
                "after_state_id": "b",
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
        self.assertEqual(summary["loop_interventions"], 1)
        self.assertEqual(
            summary["phase_counts"], {"explore": 1, "progress": 1, "recover": 1}
        )
