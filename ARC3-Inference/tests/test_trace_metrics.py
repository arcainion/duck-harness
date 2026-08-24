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
                "before_state_id": "initial",
                "after_state_id": "a",
                "behavioral_after_state_id": "stable-a",
                "outcome_class": "exact_noop",
                "controller_phase": "explore",
                "controller_policy": "outcome_aware",
                "state_context_version": 2,
                "animation": {},
            },
            {
                "type": "action",
                "action_display": "LEFT",
                "board_changed": False,
                "reward": 0,
                "before_state_id": "a",
                "after_state_id": "a",
                "behavioral_after_state_id": "stable-a",
                "outcome_class": "volatile_only",
                "controller_phase": "recover",
                "controller_policy": "outcome_aware",
                "state_context_version": 2,
                "animation": {},
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
                "before_state_id": "a",
                "after_state_id": "b",
                "behavioral_after_state_id": "stable-b",
                "outcome_class": "level_progress",
                "prediction_result": {"status": "supported"},
                "controller_phase": "progress",
                "recommended_plan_action": "RIGHT",
                "followed_recommended_plan": True,
                "recommended_plan_policy_ready": True,
                "action_regime_adapted": True,
                "controller_fallback_reason": "game_token_budget",
                "animation": {
                    "intermediate_frame_count": 2,
                    "transient_changed_cells": 3,
                },
                "controller_policy": "outcome_aware",
                "state_context_version": 2,
            },
            {
                "type": "controller",
                "guarded": True,
                "stop_reason": "harm_guard",
                "guard_reason_code": "known_harmful_cross_trial",
                "after_state_id": "b",
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
        self.assertEqual(summary["harm_interventions"], 1)
        self.assertEqual(
            summary["phase_counts"], {"explore": 1, "progress": 1, "recover": 1}
        )
        self.assertEqual(
            summary["outcome_counts"],
            {"exact_noop": 1, "level_progress": 1, "volatile_only": 1},
        )
        self.assertEqual(summary["prediction_counts"], {"supported": 1})
        self.assertEqual(summary["plan_recommendations"], 1)
        self.assertEqual(summary["plan_follow_rate"], 1.0)
        self.assertEqual(summary["followed_plan_progress_rate"], 1.0)
        self.assertEqual(summary["inference_telemetry_rate"], 1.0)
        self.assertEqual(summary["prediction_evaluation_rate"], 1 / 3)
        self.assertEqual(summary["plan_recommendation_rate"], 1 / 3)
        self.assertEqual(summary["state_context_continuity_rate"], 1.0)
        self.assertEqual(summary["state_context_link_rate"], 1.0)
        self.assertEqual(summary["plan_policy_ready_rate"], 1.0)
        self.assertEqual(summary["controller_fallback_actions"], 1)
        self.assertEqual(summary["regime_adapted_actions"], 1)
        self.assertEqual(
            summary["guard_reason_counts"],
            {"known_harmful_cross_trial": 1, "repeated_exact_noop": 1},
        )
