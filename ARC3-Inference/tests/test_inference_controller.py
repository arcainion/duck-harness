from __future__ import annotations

import os
from unittest import TestCase, mock

from inference.agent.inference_controller import (
    OUTCOME_AWARE_POLICY,
    InferenceControllerConfig,
    action_family,
    action_guard_decision,
    action_guard_reason,
    build_experience_snapshot,
    evaluate_outcome_match,
    _masked_grid_fingerprint,
    frame_fingerprint,
    normalize_action_key,
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

    def test_action_guard_decision_scans_noop_history_once(self) -> None:
        state = _frame(1, step=2)
        with mock.patch(
            "inference.agent.inference_controller.action_noop_trials",
            return_value=2,
        ) as noop_trials:
            code, reason = action_guard_decision([], state, "LEFT", self.config)

        self.assertEqual(code, "repeated_exact_noop")
        self.assertIn("2 confirmed no-op trials", reason)
        noop_trials.assert_called_once_with([], state, "LEFT")

    def test_masked_fingerprint_cache_serves_behavioral_rechecks(self) -> None:
        _masked_grid_fingerprint.cache_clear()
        grid = tuple(tuple((row + column) % 4 for column in range(20)) for row in range(20))
        masked = frozenset({(0, 0), (5, 5)})

        first = _masked_grid_fingerprint(1, grid, masked)
        second = _masked_grid_fingerprint(1, grid, masked)

        self.assertEqual(first, second)
        cache = _masked_grid_fingerprint.cache_info()
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)

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


class EvaluateOutcomeMatchTests(TestCase):
    def test_supported_no_change(self) -> None:
        result = evaluate_outcome_match("no_change", {"executed": True, "board_changed": False})
        self.assertEqual(result, "supported")

    def test_contradicted_no_change(self) -> None:
        result = evaluate_outcome_match("no_change", {"executed": True, "board_changed": True})
        self.assertEqual(result, "contradicted")

    def test_supported_state_change(self) -> None:
        result = evaluate_outcome_match("state_change", {"executed": True, "board_changed": True})
        self.assertEqual(result, "supported")

    def test_contradicted_state_change(self) -> None:
        result = evaluate_outcome_match("state_change", {"executed": True, "board_changed": False})
        self.assertEqual(result, "contradicted")

    def test_supported_new_state(self) -> None:
        result = evaluate_outcome_match("new_state", {"executed": True, "novel_state": True})
        self.assertEqual(result, "supported")

    def test_contradicted_new_state(self) -> None:
        result = evaluate_outcome_match("new_state", {"executed": True, "novel_state": False})
        self.assertEqual(result, "contradicted")

    def test_supported_level_progress_via_level_completed(self) -> None:
        result = evaluate_outcome_match("level_progress", {"executed": True, "level_completed": True})
        self.assertEqual(result, "supported")

    def test_supported_level_progress_via_run_complete(self) -> None:
        result = evaluate_outcome_match("level_progress", {"executed": True, "run_complete": True})
        self.assertEqual(result, "supported")

    def test_supported_level_progress_via_reward(self) -> None:
        result = evaluate_outcome_match("level_progress", {"executed": True, "reward": 0.5})
        self.assertEqual(result, "supported")

    def test_contradicted_level_progress(self) -> None:
        result = evaluate_outcome_match("level_progress", {"executed": True, "board_changed": False})
        self.assertEqual(result, "contradicted")

    def test_inconclusive_when_not_executed(self) -> None:
        result = evaluate_outcome_match("no_change", {"executed": False})
        self.assertEqual(result, "inconclusive")

    def test_inconclusive_when_expected_unknown(self) -> None:
        result = evaluate_outcome_match("unknown", {"executed": True, "board_changed": True})
        self.assertEqual(result, "inconclusive")


class NormalizeActionKeyTests(TestCase):
    def test_normalizes_uppercase(self) -> None:
        self.assertEqual(normalize_action_key("LEFT"), "LEFT")

    def test_strips_whitespace(self) -> None:
        self.assertEqual(normalize_action_key("  LEFT  "), "LEFT")

    def test_mouse_with_coords(self) -> None:
        result = normalize_action_key("MOUSE(row=3, col=4)")
        self.assertEqual(result, "MOUSE(ROW=3, COL=4)")

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_action_key(""), "")


class ActionFamilyTests(TestCase):
    def test_directional_actions(self) -> None:
        self.assertEqual(action_family("LEFT"), "LEFT")
        self.assertEqual(action_family("RIGHT"), "RIGHT")
        self.assertEqual(action_family("UP"), "UP")
        self.assertEqual(action_family("DOWN"), "DOWN")

    def test_space_action(self) -> None:
        self.assertEqual(action_family("SPACE"), "SPACE")

    def test_mouse_family(self) -> None:
        self.assertEqual(action_family("MOUSE(row=3, col=4)"), "MOUSE")
        self.assertEqual(action_family("MOUSE"), "MOUSE")

    def test_unknown_action(self) -> None:
        self.assertEqual(action_family("RESET"), "RESET")


class EmptyHistoryTests(TestCase):
    def setUp(self) -> None:
        self.config = InferenceControllerConfig()

    def test_empty_history_orient_phase(self) -> None:
        frame = _frame(1, step=0)
        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=frame)], frame, ["LEFT", "RIGHT"], self.config
        )
        self.assertEqual(snapshot["phase"], "orient")
        self.assertEqual(snapshot["actions_observed"], 0)

    def test_empty_history_suggested_actions(self) -> None:
        frame = _frame(1, step=0)
        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=frame)], frame, ["LEFT", "RIGHT"], self.config
        )
        self.assertIn("LEFT", snapshot["suggested_actions"])
        self.assertIn("RIGHT", snapshot["suggested_actions"])


class SingleFrameHistoryTests(TestCase):
    def setUp(self) -> None:
        self.config = InferenceControllerConfig()

    def test_single_frame_no_transitions(self) -> None:
        frame = _frame(1, step=0)
        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=frame)], frame, ["LEFT"], self.config
        )
        self.assertEqual(snapshot["actions_observed"], 0)
        self.assertEqual(snapshot["cycle_period"], None)

    def test_single_action_creates_transition(self) -> None:
        before = _frame(1, step=0)
        after = _frame(2, step=1)
        history = [
            HistoryEntry(action="", frame=before),
            HistoryEntry(action="LEFT", frame=after),
        ]
        snapshot = build_experience_snapshot(history, after, ["LEFT"], self.config)
        self.assertEqual(snapshot["actions_observed"], 1)
        self.assertIsNotNone(snapshot["latest_outcome"])


class FrameFingerprintTests(TestCase):
    def test_none_frame(self) -> None:
        result = frame_fingerprint(None)
        self.assertEqual(result, "none")

    def test_same_grid_same_level(self) -> None:
        f1 = _frame(5, step=0)
        f2 = _frame(5, step=99)
        self.assertEqual(frame_fingerprint(f1), frame_fingerprint(f2))

    def test_different_grids(self) -> None:
        f1 = _frame(1, step=0)
        f2 = _frame(2, step=0)
        self.assertNotEqual(frame_fingerprint(f1), frame_fingerprint(f2))

    def test_different_levels(self) -> None:
        f1 = _frame(1, step=0, level=1)
        f2 = _frame(1, step=0, level=2)
        self.assertNotEqual(frame_fingerprint(f1), frame_fingerprint(f2))

    def test_empty_grid(self) -> None:
        f1 = Frame(grid=(), step=0, level=1)
        f2 = Frame(grid=(), step=5, level=1)
        self.assertEqual(frame_fingerprint(f1), frame_fingerprint(f2))

    def test_single_cell_grid(self) -> None:
        f1 = Frame(grid=((5,),), step=0, level=1)
        f2 = Frame(grid=((5,),), step=10, level=1)
        self.assertEqual(frame_fingerprint(f1), frame_fingerprint(f2))

    def test_large_grid_fingerprint(self) -> None:
        grid = tuple(tuple(i % 16 for i in range(64)) for _ in range(64))
        f1 = Frame(grid=grid, step=0, level=1)
        f2 = Frame(grid=grid, step=100, level=1)
        self.assertEqual(frame_fingerprint(f1), frame_fingerprint(f2))


class InferenceControllerEdgeCaseTests(TestCase):
    def setUp(self) -> None:
        self.config = InferenceControllerConfig(
            enabled=True,
            same_state_noop_limit=2,
            stagnation_window=3,
            cycle_window=4,
        )

    def test_noop_guard_with_different_action(self) -> None:
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="LEFT", frame=_frame(1, step=1)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2)),
        ]
        reason = action_guard_reason(history, state, "RIGHT", self.config)
        self.assertIsNone(reason)

    def test_noop_guard_triggers_after_limit(self) -> None:
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="LEFT", frame=_frame(1, step=1)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=3)),
        ]
        reason = action_guard_reason(history, state, "LEFT", self.config)
        self.assertIsNotNone(reason)

    def test_cycle_detection_minimum_window(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            cycle_window=2,
        )
        a = _frame(1, step=0)
        b = _frame(2, step=1)
        history = [
            HistoryEntry(action="", frame=a),
            HistoryEntry(action="RIGHT", frame=b),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2)),
            HistoryEntry(action="RIGHT", frame=_frame(2, step=3)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=4)),
        ]
        snapshot = build_experience_snapshot(history, history[-1].frame, ["LEFT", "RIGHT"], config)
        self.assertIn(snapshot["phase"], ("recover", "explore"))

    def test_stagnation_window_edge_case(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            stagnation_window=1,
        )
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="LEFT", frame=_frame(1, step=1)),
        ]
        snapshot = build_experience_snapshot(history, state, ["LEFT", "RIGHT"], config)
        self.assertEqual(snapshot["phase"], "recover")

    def test_transition_metadata_no_history(self) -> None:
        before = _frame(1, step=0)
        after = _frame(2, step=1)
        history = [HistoryEntry(action="", frame=before)]
        metadata = transition_metadata(before, after, history, "RIGHT", self.config)
        self.assertTrue(metadata["novel_state"])
        self.assertEqual(metadata["controller_phase"], "progress")

    def test_transition_metadata_same_state(self) -> None:
        state = _frame(1, step=0)
        history = [HistoryEntry(action="", frame=state)]
        metadata = transition_metadata(state, state, history, "LEFT", self.config)
        self.assertFalse(metadata["novel_state"])
        self.assertEqual(metadata["before_state_id"], metadata["after_state_id"])

    def test_transition_metadata_invalid_action(self) -> None:
        before = _frame(1, step=0)
        after = _frame(2, step=1)
        history = [HistoryEntry(action="", frame=before)]
        metadata = transition_metadata(before, after, history, "INVALID", self.config)
        self.assertEqual(metadata["controller_phase"], "progress")

    def test_build_snapshot_with_many_actions(self) -> None:
        actions = [f"ACTION_{i}" for i in range(20)]
        frame = _frame(1, step=0)
        history = [HistoryEntry(action="", frame=frame)]
        snapshot = build_experience_snapshot(history, frame, actions, self.config)
        self.assertLessEqual(len(snapshot["ranked_actions"]), 6)

    def test_volatile_cells_calculation(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            volatile_window=4,
            volatile_min_samples=2,
            volatile_ratio=0.5,
        )
        frames = [
            Frame(grid=((value, 9),), step=index, level=1)
            for index, value in enumerate((1, 2, 1, 2))
        ]
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action="SPACE", frame=frame) for frame in frames[1:]
        ]
        snapshot = build_experience_snapshot(history, frames[-1], ["SPACE"], config)
        self.assertGreaterEqual(snapshot["volatile_cells"], 0)

    def test_outcome_classification_level_progress_reward_zero(self) -> None:
        result = evaluate_outcome_match("level_progress", {"executed": True, "reward": 0})
        self.assertEqual(result, "contradicted")

    def test_outcome_classification_level_progress_negative_reward(self) -> None:
        result = evaluate_outcome_match("level_progress", {"executed": True, "reward": -0.5})
        self.assertEqual(result, "contradicted")

    def test_outcome_classification_new_state_false(self) -> None:
        result = evaluate_outcome_match("new_state", {"executed": True, "novel_state": False})
        self.assertEqual(result, "contradicted")

    def test_normalize_action_key_mouse_variations(self) -> None:
        result = normalize_action_key("MOUSE(row=1, col=2)")
        self.assertIn("MOUSE", result)

    def test_action_family_edge_cases(self) -> None:
        self.assertEqual(action_family(""), "")
        self.assertEqual(action_family("MOUSE"), "MOUSE")
        self.assertEqual(action_family("RESET"), "RESET")

    def test_config_from_env_invalid_values(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_STRATEGY_ENABLED": "true",
                "LOCAL_ANALYZER_VOLATILE_WINDOW": "invalid",
                "LOCAL_ANALYZER_VOLATILE_RATIO": "not_a_float",
            },
            clear=True,
        ):
            config = InferenceControllerConfig.from_env()
            self.assertTrue(config.enabled)
            self.assertEqual(config.volatile_window, 8)
            self.assertEqual(config.volatile_ratio, 0.75)

    def test_config_bounds_volatile_params(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_STRATEGY_ENABLED": "true",
                "LOCAL_ANALYZER_VOLATILE_WINDOW": "0",
                "LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES": "100",
                "LOCAL_ANALYZER_VOLATILE_RATIO": "2.0",
            },
            clear=True,
        ):
            config = InferenceControllerConfig.from_env()
            self.assertGreaterEqual(config.volatile_window, 2)
            self.assertGreaterEqual(config.volatile_min_samples, 2)
            self.assertLessEqual(config.volatile_ratio, 1.0)

    def test_history_with_none_frames_skipped(self) -> None:
        history = [
            HistoryEntry(action="", frame=_frame(1, step=0)),
            HistoryEntry(action="LEFT", frame=_frame(1, step=1)),
        ]
        snapshot = build_experience_snapshot(history, _frame(1, step=1), ["LEFT"], self.config)
        self.assertEqual(snapshot["actions_observed"], 1)

    def test_transition_model_many_trials(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            volatile_min_samples=10,
        )
        a = _frame(1, step=0)
        b = _frame(2, step=1)
        history = [HistoryEntry(action="", frame=a)]
        for i in range(10):
            history.append(HistoryEntry(action="RIGHT", frame=b if i % 2 == 0 else a))
        snapshot = build_experience_snapshot(history, history[-1].frame, ["RIGHT"], config)
        model = snapshot["transition_models_here"][0]
        self.assertGreaterEqual(model["trials"], 1)
        self.assertLessEqual(model["confidence"], 1.0)
        self.assertGreaterEqual(model["confidence"], 0.0)
