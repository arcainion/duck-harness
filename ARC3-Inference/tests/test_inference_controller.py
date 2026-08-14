from __future__ import annotations

import os
from unittest import TestCase, mock

from inference.agent.inference_controller import (
    InferenceControllerConfig,
    action_guard_reason,
    build_experience_snapshot,
    frame_fingerprint,
    transition_metadata,
)
from inference.agent.runtime_state import Frame, HistoryEntry


def _frame(value: int, *, step: int, level: int = 1) -> Frame:
    return Frame(grid=((value, value), (value, value)), step=step, level=level)


class InferenceControllerTests(TestCase):
    def setUp(self) -> None:
        self.config = InferenceControllerConfig(
            enabled=True,
            same_state_noop_limit=2,
            stagnation_window=3,
            cycle_window=4,
        )

    def test_fingerprint_is_stable_and_level_sensitive(self) -> None:
        first = _frame(1, step=0)
        same_visible_state = _frame(1, step=99)
        next_level = _frame(1, step=0, level=2)

        self.assertEqual(frame_fingerprint(first), frame_fingerprint(same_visible_state))
        self.assertNotEqual(frame_fingerprint(first), frame_fingerprint(next_level))

    def test_absent_configuration_preserves_legacy_disabled_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = InferenceControllerConfig.from_env()

        self.assertFalse(config.enabled)
        self.assertEqual(config.same_state_noop_limit, 2)
        self.assertEqual(config.stagnation_window, 12)
        self.assertEqual(config.cycle_window, 8)

    def test_two_exact_noops_guard_the_third_trial(self) -> None:
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="LEFT", frame=_frame(1, step=1)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2)),
        ]

        reason = action_guard_reason(history, state, "LEFT", self.config)

        self.assertIn("2 confirmed no-op trials", reason or "")
        self.assertIsNone(action_guard_reason(history, state, "RIGHT", self.config))

    def test_cycle_and_stagnation_switch_to_recover(self) -> None:
        a = _frame(1, step=0)
        b = _frame(2, step=1)
        history = [
            HistoryEntry(action="", frame=a),
            HistoryEntry(action="RIGHT", frame=b),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2)),
            HistoryEntry(action="RIGHT", frame=_frame(2, step=3)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=4)),
        ]

        snapshot = build_experience_snapshot(history, history[-1].frame, ["LEFT", "RIGHT"], self.config)

        self.assertEqual(snapshot["phase"], "recover")
        self.assertEqual(snapshot["cycle_period"], 2)
        self.assertGreaterEqual(snapshot["stagnation_actions"], 2)

    def test_memory_is_per_history_but_retains_cross_level_evidence(self) -> None:
        first_level = _frame(1, step=0)
        second_level = _frame(3, step=1, level=2)
        run_history = [
            HistoryEntry(action="", frame=first_level),
            HistoryEntry(action="SPACE", frame=second_level),
        ]

        retained = build_experience_snapshot(run_history, second_level, ["SPACE"], self.config)
        fresh_pass = build_experience_snapshot(
            [HistoryEntry(action="", frame=first_level)], first_level, ["SPACE"], self.config
        )

        self.assertEqual(retained["actions_observed"], 1)
        self.assertEqual(fresh_pass["actions_observed"], 0)
        self.assertEqual(fresh_pass["phase"], "orient")

    def test_transition_metadata_attributes_one_action(self) -> None:
        before = _frame(1, step=0)
        after = _frame(2, step=1)
        history = [HistoryEntry(action="", frame=before)]

        metadata = transition_metadata(before, after, history, "RIGHT", self.config)

        self.assertTrue(metadata["novel_state"])
        self.assertNotEqual(metadata["before_state_id"], metadata["after_state_id"])
        self.assertEqual(metadata["controller_phase"], "progress")

