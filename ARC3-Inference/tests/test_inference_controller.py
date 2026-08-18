from __future__ import annotations

import os
from unittest import TestCase, mock

from inference.agent.inference_controller import (
    OUTCOME_AWARE_POLICY,
    InferenceControllerConfig,
    action_family,
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
        self.assertEqual(config.policy, "legacy")

    def test_environment_enables_and_bounds_outcome_aware_policy(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_STRATEGY_ENABLED": "true",
                "LOCAL_ANALYZER_STRATEGY_POLICY": "outcome-aware",
                "LOCAL_ANALYZER_VOLATILE_WINDOW": "1",
                "LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES": "1",
                "LOCAL_ANALYZER_VOLATILE_RATIO": "2",
            },
            clear=True,
        ):
            config = InferenceControllerConfig.from_env()

        self.assertTrue(config.outcome_aware)
        self.assertEqual(config.policy, OUTCOME_AWARE_POLICY)
        self.assertEqual(config.volatile_window, 2)
        self.assertEqual(config.volatile_min_samples, 2)
        self.assertEqual(config.volatile_ratio, 1.0)

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
        self.assertEqual(snapshot["action_budget"], 1)

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

    def test_outcome_aware_masks_repeatedly_volatile_cells(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            same_state_noop_limit=2,
            volatile_window=8,
            volatile_min_samples=4,
            volatile_ratio=0.75,
        )
        frames = [
            Frame(grid=((value, 9),), step=index, level=1)
            for index, value in enumerate((1, 2, 3, 4, 5))
        ]
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action="SPACE", frame=frame) for frame in frames[1:]
        ]

        snapshot = build_experience_snapshot(history, frames[-1], ["SPACE"], config)

        self.assertEqual(snapshot["volatile_cells"], 1)
        self.assertEqual(snapshot["latest_outcome"], "volatile_only")
        self.assertEqual(snapshot["unique_behavioral_states"], 1)
        self.assertEqual(snapshot["phase"], "recover")

    def test_outcome_aware_ranks_untried_before_noop_actions(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        a = _frame(1, step=0)
        b = _frame(2, step=1)
        history = [
            HistoryEntry(action="", frame=a),
            HistoryEntry(action="RIGHT", frame=b),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2)),
            HistoryEntry(action="UP", frame=_frame(1, step=3)),
        ]

        snapshot = build_experience_snapshot(
            history, history[-1].frame, ["RIGHT", "UP", "DOWN"], config
        )

        ranked = snapshot["ranked_actions"]
        self.assertEqual(ranked[0]["action"], "DOWN")
        self.assertEqual(ranked[0]["priority"], 1)
        self.assertEqual(ranked[-1]["action"], "UP")
        self.assertEqual(ranked[-1]["no_ops"], 1)

    def test_transition_model_verifies_repeated_state_action_effect(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            volatile_min_samples=10,
        )
        a = _frame(1, step=0)
        b = _frame(2, step=1)
        history = [
            HistoryEntry(action="", frame=a),
            HistoryEntry(action="RIGHT", frame=b),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2)),
        ]

        snapshot = build_experience_snapshot(history, history[-1].frame, ["RIGHT"], config)
        model = snapshot["transition_models_here"][0]

        self.assertEqual(model["action"], "RIGHT")
        self.assertEqual(model["trials"], 1)
        self.assertEqual(model["confidence"], 1.0)
        self.assertTrue(model["verified_deterministic"])
        self.assertEqual(model["contradictions"], 0)
        self.assertEqual(snapshot["model_conflicts_here"], 0)

    def test_transition_model_conflict_forces_hypothesis_recovery(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            volatile_min_samples=10,
        )
        history = [
            HistoryEntry(action="", frame=_frame(1, step=0)),
            HistoryEntry(action="RIGHT", frame=_frame(2, step=1)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2)),
            HistoryEntry(action="RIGHT", frame=_frame(3, step=3)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=4)),
        ]

        snapshot = build_experience_snapshot(history, history[-1].frame, ["RIGHT"], config)
        model = snapshot["transition_models_here"][0]

        self.assertFalse(model["verified_deterministic"])
        self.assertEqual(model["trials"], 2)
        self.assertEqual(model["confidence"], 0.5)
        self.assertEqual(model["contradictions"], 1)
        self.assertEqual(snapshot["model_conflicts_here"], 1)
        self.assertIn("transition_model_conflict", snapshot["recovery_reasons"])
        self.assertEqual(snapshot["phase"], "recover")

    def test_mouse_family_keeps_coordinate_search_open(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="MOUSE(row=1, col=1)", frame=_frame(1, step=1)),
        ]

        snapshot = build_experience_snapshot(history, history[-1].frame, ["MOUSE"], config)

        self.assertEqual(action_family("MOUSE(row=3, col=4)"), "MOUSE")
        self.assertTrue(snapshot["ranked_actions"][0]["parameterized"])
        self.assertEqual(snapshot["ranked_actions"][0]["priority"], 1)
        self.assertIsNone(
            action_guard_reason(history, history[-1].frame, "MOUSE(row=2, col=2)", config)
        )

    def test_outcome_metadata_reports_level_progress(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        before = _frame(1, step=0)
        after = _frame(2, step=1, level=2)

        metadata = transition_metadata(
            before, after, [HistoryEntry(action="", frame=before)], "SPACE", config
        )

        self.assertEqual(metadata["outcome_class"], "level_progress")
        self.assertEqual(metadata["controller_policy"], OUTCOME_AWARE_POLICY)
        self.assertEqual(metadata["action_rank"], 1)

    def test_outcome_metadata_reports_executed_action_rank(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            same_state_noop_limit=2,
        )
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="RIGHT", frame=_frame(1, step=1)),
            HistoryEntry(action="RIGHT", frame=_frame(1, step=2)),
        ]

        metadata = transition_metadata(
            state,
            _frame(1, step=3),
            history,
            "RIGHT",
            config,
            ["LEFT", "RIGHT"],
        )

        self.assertEqual(metadata["action_rank"], 2)
        self.assertEqual(
            metadata["action_rank_reason"],
            "previous trials were no-op or cycle-prone",
        )

    def test_progress_phase_allows_only_a_short_confirmed_batch(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            progress_action_budget=3,
        )
        before = _frame(1, step=0)
        after = _frame(2, step=1)
        history = [
            HistoryEntry(action="", frame=before),
            HistoryEntry(action="RIGHT", frame=after),
        ]

        snapshot = build_experience_snapshot(history, after, ["RIGHT"], config)

        self.assertEqual(snapshot["phase"], "progress")
        self.assertEqual(snapshot["action_budget"], 3)

    def test_snapshot_payloads_remain_bounded(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            recent_transition_limit=3,
        )
        history = [HistoryEntry(action="", frame=_frame(0, step=0))]
        for step in range(1, 20):
            history.append(HistoryEntry(action="RIGHT", frame=_frame(step, step=step)))

        snapshot = build_experience_snapshot(
            history,
            history[-1].frame,
            ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "RESET"],
            config,
        )

        self.assertLessEqual(len(snapshot["recent_transitions"]), 3)
        self.assertLessEqual(len(snapshot["ranked_actions"]), 6)
        self.assertLessEqual(len(snapshot["transition_models_here"]), 6)
