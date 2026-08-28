from __future__ import annotations

import os
from unittest import TestCase, mock

from inference.agent.inference_controller import (
    OUTCOME_AWARE_POLICY,
    InferenceControllerConfig,
    _normalize_external_transition,
    action_coordinate,
    action_family,
    action_guard_reason,
    build_experience_snapshot,
    frame_fingerprint,
    harmful_evidence_is_decisive,
    transition_metadata,
)
from inference.agent.runtime_state import Frame, HistoryEntry


def _frame(value: int, *, step: int, level: int = 1) -> Frame:
    return Frame(grid=((value, value), (value, value)), step=step, level=level)


class InferenceControllerTests(TestCase):
    def test_action_family_does_not_accept_mouse_prefix_collisions(self) -> None:
        self.assertEqual(action_family("MOUSETRAP"), "MOUSETRAP")
        self.assertEqual(action_family("MOUSE"), "MOUSE")

    def test_action_coordinate_rejects_unbounded_numeric_text(self) -> None:
        coordinate = action_coordinate(
            f"MOUSE(row={'9' * 5000}, col={'8' * 5000})"
        )

        self.assertIsNone(coordinate)

    def test_action_coordinate_enforces_board_bounds(self) -> None:
        self.assertIsNone(action_coordinate("MOUSE(row=-1, col=0)"))
        self.assertIsNone(action_coordinate("MOUSE(row=64, col=0)"))
        self.assertEqual(
            action_coordinate(" mouse ( row = 3 , col = 4 ) "), (3, 4)
        )

    def test_nonfinite_environment_thresholds_use_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_VOLATILE_RATIO": "nan",
                "LOCAL_ANALYZER_PLAN_MIN_CONFIDENCE": "inf",
            },
        ):
            config = InferenceControllerConfig.from_env()

        self.assertEqual(config.volatile_ratio, 0.75)
        self.assertEqual(config.plan_min_confidence, 0.75)

    def test_nonfinite_environment_utilities_use_defaults(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_PROGRESS_UTILITY": "inf",
                "LOCAL_ANALYZER_TERMINAL_FAILURE_UTILITY": "nan",
            },
        ):
            config = InferenceControllerConfig.from_env()

        self.assertEqual(config.progress_utility, 1.0)
        self.assertEqual(config.terminal_failure_utility, -2.0)

    def test_level_action_limit_is_baseline_relative_with_a_floor(self) -> None:
        config = InferenceControllerConfig(
            level_action_limit_multiplier=2.0,
            level_action_limit_minimum=16,
        )

        self.assertEqual(config.level_action_limit(7), 16)
        self.assertEqual(config.level_action_limit(21), 42)
        self.assertIsNone(config.level_action_limit(None))
        self.assertIsNone(InferenceControllerConfig().level_action_limit(21))

    def test_edge_only_hud_change_is_not_novel_when_enabled(self) -> None:
        before = Frame(
            grid=tuple(tuple(0 for _ in range(5)) for _ in range(5)),
            step=0,
            level=1,
        )
        grid = [list(row) for row in before.grid]
        grid[0][2] = 1
        after = Frame(grid=tuple(tuple(row) for row in grid), step=1, level=1)
        config = InferenceControllerConfig(ignore_edge_hud_changes=True)

        metadata = transition_metadata(
            before,
            after,
            [HistoryEntry(action="", frame=before)],
            "SPACE",
            config,
        )

        self.assertEqual(metadata["outcome_class"], "volatile_only")
        self.assertFalse(metadata["novel_state"])

    def test_pure_object_translation_is_a_revisit_not_novelty(self) -> None:
        def translated_frame(col: int, step: int) -> Frame:
            grid = [[0 for _ in range(5)] for _ in range(5)]
            grid[2][col] = 1
            return Frame(
                grid=tuple(tuple(row) for row in grid),
                step=step,
                level=1,
            )

        before = translated_frame(1, 0)
        after = translated_frame(2, 1)
        metadata = transition_metadata(
            before,
            after,
            [HistoryEntry(action="", frame=before)],
            "RIGHT",
            InferenceControllerConfig(volatile_min_samples=99),
        )

        self.assertEqual(metadata["outcome_class"], "revisit")
        self.assertFalse(metadata["novel_state"])

    def test_repeat_action_guard_blocks_third_identical_mouse_coordinate(self) -> None:
        frames = [_frame(value, step=value) for value in range(3)]
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action="MOUSE(row=4, col=7)", frame=frame)
            for frame in frames[1:]
        ]
        config = InferenceControllerConfig(repeat_action_limit=2)

        reason = action_guard_reason(
            history, frames[-1], "MOUSE(row=4, col=7)", config
        )

        self.assertIn("same parameterized action", reason or "")

    def test_repeat_action_guard_blocks_second_inverse_cycle(self) -> None:
        frames = [_frame(value, step=value) for value in range(4)]
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action=action, frame=frame)
            for action, frame in zip(("UP", "DOWN", "UP"), frames[1:])
        ]
        config = InferenceControllerConfig(repeat_action_limit=2)

        reason = action_guard_reason(history, frames[-1], "DOWN", config)

        self.assertIn("inverse-action cycle", reason or "")

    def test_directional_watchdog_blocks_dominant_action_without_progress(self) -> None:
        frames = [_frame(value, step=value) for value in range(17)]
        actions = ["UP"] * 12 + ["RIGHT"] * 4
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action=action, frame=frame)
            for action, frame in zip(actions, frames[1:])
        ]
        config = InferenceControllerConfig(
            directional_no_progress_window=16,
            directional_no_progress_limit=12,
            volatile_min_samples=99,
        )

        reason = action_guard_reason(history, frames[-1], "UP", config)
        snapshot = build_experience_snapshot(
            history, frames[-1], ["UP", "RIGHT"], config
        )

        self.assertIn("dominates the recent action window", reason or "")
        self.assertIsNone(action_guard_reason(history, frames[-1], "RIGHT", config))
        self.assertIn("directional_no_progress", snapshot["recovery_reasons"])
        self.assertEqual(snapshot["directional_no_progress"]["action"], "UP")

    def test_directional_watchdog_allows_action_after_score_progress(self) -> None:
        frames = [
            Frame(
                grid=((value, value), (value, value)),
                step=value,
                level=1,
                score=int(value == 16),
            )
            for value in range(17)
        ]
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action="UP", frame=frame) for frame in frames[1:]
        ]
        config = InferenceControllerConfig(
            directional_no_progress_window=16,
            directional_no_progress_limit=12,
        )

        self.assertIsNone(action_guard_reason(history, frames[-1], "UP", config))

    def test_repeated_directional_progress_is_not_a_noisy_cycle(self) -> None:
        frames = [_frame(value, step=value) for value in range(5)]
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action="UP", frame=frame) for frame in frames[1:]
        ]

        snapshot = build_experience_snapshot(
            history,
            frames[-1],
            ["UP", "DOWN"],
            InferenceControllerConfig(volatile_min_samples=99),
        )

        self.assertIsNone(snapshot["cycle_period"])

    def test_inverse_directional_pattern_is_detected_as_cycle(self) -> None:
        frames = [_frame(value, step=value) for value in range(5)]
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action=action, frame=frame)
            for action, frame in zip(("UP", "DOWN", "UP", "DOWN"), frames[1:])
        ]

        snapshot = build_experience_snapshot(
            history,
            frames[-1],
            ["UP", "DOWN"],
            InferenceControllerConfig(volatile_min_samples=99),
        )

        self.assertEqual(snapshot["cycle_period"], 2)

    def test_external_transition_rejects_invalid_action(self) -> None:
        self.assertIsNone(_normalize_external_transition({}))
        self.assertIsNone(_normalize_external_transition({"action": {"bad": True}}))

    def test_external_transition_bounds_base_state_ids(self) -> None:
        normalized = _normalize_external_transition(
            {
                "action": "LEFT",
                "before_state_id": f"  {'a' * 300}  ",
                "after_state_id": "  after   state  ",
            }
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(len(normalized["before_state_id"]), 256)
        self.assertEqual(normalized["after_state_id"], "after state")

    def test_external_transition_bounds_context_identifiers(self) -> None:
        normalized = _normalize_external_transition(
            {
                "action": "LEFT",
                "behavioral_before_state_id": "b" * 300,
                "object_after_state_id": "o" * 300,
                "evidence_id": "e" * 300,
            }
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(len(normalized["behavioral_before_state_id"]), 256)
        self.assertEqual(len(normalized["object_after_state_id"]), 256)
        self.assertEqual(len(normalized["evidence_id"]), 128)

    def test_external_transition_normalizes_outcome_label(self) -> None:
        missing = _normalize_external_transition({"action": "LEFT"})
        oversized = _normalize_external_transition(
            {"action": "LEFT", "outcome_class": "x" * 100}
        )

        self.assertEqual(missing["outcome_class"], "unknown")
        self.assertEqual(len(oversized["outcome_class"]), 40)

    def test_external_transition_neutralizes_invalid_reward(self) -> None:
        invalid = _normalize_external_transition(
            {"action": "LEFT", "reward": "invalid"}
        )
        nonfinite = _normalize_external_transition(
            {"action": "LEFT", "reward": float("nan")}
        )

        self.assertEqual(invalid["reward"], 0.0)
        self.assertEqual(nonfinite["reward"], 0.0)

    def test_external_transition_bounds_raw_observations(self) -> None:
        invalid = _normalize_external_transition(
            {"action": "LEFT", "raw_observations": True}
        )
        oversized = _normalize_external_transition(
            {"action": "LEFT", "raw_observations": 10**12}
        )

        self.assertEqual(invalid["raw_observations"], 1)
        self.assertEqual(oversized["raw_observations"], 1_000_000)

    def test_external_transition_normalizes_valid_action_collection(self) -> None:
        scalar = _normalize_external_transition(
            {"action": "LEFT", "valid_actions_after": "RIGHT"}
        )
        collection = _normalize_external_transition(
            {
                "action": "LEFT",
                "valid_actions_after": [" right ", "RIGHT", {"bad": True}],
            }
        )

        self.assertEqual(scalar["valid_actions_after"], [])
        self.assertEqual(collection["valid_actions_after"], ["RIGHT"])

    def test_external_transition_parses_board_changed_boolean(self) -> None:
        unchanged = _normalize_external_transition(
            {"action": "LEFT", "board_changed": "false"}
        )
        changed = _normalize_external_transition(
            {"action": "LEFT", "board_changed": "yes"}
        )

        self.assertFalse(unchanged["board_changed"])
        self.assertTrue(changed["board_changed"])

    def test_external_transition_parses_context_changed_boolean(self) -> None:
        unchanged = _normalize_external_transition(
            {"action": "LEFT", "decision_context_changed": "off"}
        )
        changed = _normalize_external_transition(
            {"action": "LEFT", "decision_context_changed": 1}
        )

        self.assertFalse(unchanged["decision_context_changed"])
        self.assertTrue(changed["decision_context_changed"])

    def test_external_transition_parses_game_over_boolean(self) -> None:
        running = _normalize_external_transition(
            {"action": "LEFT", "game_over": "0"}
        )
        ended = _normalize_external_transition(
            {"action": "LEFT", "game_over": "true"}
        )

        self.assertFalse(running["game_over"])
        self.assertTrue(ended["game_over"])

    def test_state_identity_includes_observable_decision_context(self) -> None:
        base = Frame(
            grid=((1,),),
            step=1,
            level=1,
            valid_actions=("LEFT",),
            engine_state="NOT_FINISHED",
            score=0,
        )
        same_grid_different_actions = Frame(
            grid=((1,),),
            step=2,
            level=1,
            valid_actions=("RIGHT",),
            engine_state="NOT_FINISHED",
            score=0,
        )
        same_grid_different_state = Frame(
            grid=((1,),),
            step=2,
            level=1,
            valid_actions=("LEFT",),
            engine_state="GAME_OVER",
            score=0,
        )

        self.assertNotEqual(
            frame_fingerprint(base), frame_fingerprint(same_grid_different_actions)
        )
        self.assertNotEqual(
            frame_fingerprint(base), frame_fingerprint(same_grid_different_state)
        )

    def test_context_only_transition_is_not_reported_as_changed_pixels(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        before = Frame(grid=((1,),), step=0, level=1, valid_actions=("LEFT",))
        after = Frame(grid=((1,),), step=1, level=1, valid_actions=("RIGHT",))

        snapshot = build_experience_snapshot(
            [
                HistoryEntry(action="", frame=before),
                HistoryEntry(action="SPACE", frame=after),
            ],
            after,
            ["RIGHT"],
            config,
        )

        transition = snapshot["recent_transitions"][0]
        self.assertFalse(transition["board_changed"])
        self.assertTrue(transition["decision_context_changed"])
        self.assertEqual(transition["outcome_class"], "novel")
        self.assertEqual(snapshot["no_op_streak"], 1)
        self.assertEqual(snapshot["behavioral_no_op_streak"], 0)

    def test_object_state_tracks_translation_without_structural_change(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        before = Frame(grid=((1, 1, 0), (0, 0, 0), (0, 0, 0)), step=0, level=1)
        after = Frame(grid=((0, 0, 0), (0, 1, 1), (0, 0, 0)), step=1, level=1)
        before_snapshot = build_experience_snapshot([], before, ["RIGHT"], config)
        after_snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=before)], after, ["RIGHT"], config
        )

        self.assertEqual(
            before_snapshot["object_state"]["shape_signature"],
            after_snapshot["object_state"]["shape_signature"],
        )
        self.assertEqual(
            before_snapshot["object_state_id"], after_snapshot["object_state_id"]
        )
        self.assertFalse(after_snapshot["object_temporal"]["structural_change"])
        self.assertEqual(
            after_snapshot["object_temporal"]["motions"][0]["delta"], [1.0, 1.0]
        )

    def test_object_equivalent_evidence_drives_ranking_and_transition_model(
        self,
    ) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = Frame(grid=((1, 1, 0), (0, 0, 0), (0, 0, 0)), step=0, level=1)
        current_object_id = build_experience_snapshot([], current, ["RIGHT"], config)[
            "object_state_id"
        ]
        external = [
            {
                "before_state_id": "translated-visual-state",
                "after_state_id": "translated-result-state",
                "object_before_state_id": current_object_id,
                "object_after_state_id": "abstract-result-state",
                "action_display": "RIGHT",
                "outcome_class": "novel",
                "board_changed": True,
                "evidence_id": "prior-object-trial",
            }
        ]

        snapshot = build_experience_snapshot(
            [], current, ["RIGHT"], config, external_transitions=external
        )

        self.assertEqual(snapshot["ranked_actions"][0]["trials"], 1)
        model = snapshot["transition_models_here"][0]
        self.assertEqual(model["state_abstraction"], "object_relational")
        self.assertEqual(model["predicted_object_state_id"], "abstract-result-state")
        self.assertEqual(model["predicted_outcome"], "novel")

    def test_experiment_selection_prioritizes_conflicting_hypotheses(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)
        current_id = frame_fingerprint(current)
        external = [
            {
                "before_state_id": current_id,
                "after_state_id": f"outcome-{source}",
                "action_display": "RIGHT",
                "outcome_class": outcome,
                "evidence_id": source,
            }
            for source, outcome in (
                ("trial-a", "novel"),
                ("trial-b", "exact_noop"),
            )
        ]

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["RIGHT"],
            config,
            external_transitions=external,
        )

        ranking = snapshot["ranked_actions"][0]
        self.assertEqual(ranking["model_disagreement"], 0.5)
        self.assertEqual(
            snapshot["recommended_experiments"][0]["hypothesis"],
            "resolve conflicting transition outcomes",
        )
        self.assertGreater(
            ranking["adaptive_exploration_weight"], config.exploration_weight
        )

    def test_nonstationary_outcomes_trigger_a_recovery_portfolio(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)
        current_id = frame_fingerprint(current)
        external = [
            {
                "before_state_id": current_id,
                "after_state_id": f"state-{index}",
                "action_display": "RIGHT",
                "outcome_class": outcome,
                "evidence_id": f"trial-{index}",
            }
            for index, outcome in enumerate(
                ("exact_noop", "exact_noop", "novel", "novel")
            )
        ]

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["RIGHT", "LEFT"],
            config,
            external_transitions=external,
        )

        self.assertIn("nonstationary_dynamics", snapshot["recovery_reasons"])
        self.assertEqual(
            snapshot["nonstationary_actions"][0]["recent_outcome"], "novel"
        )
        self.assertEqual(
            snapshot["recovery_portfolio"][0]["strategy"],
            "revalidate_changed_model",
        )
        self.assertEqual(snapshot["recovery_portfolio"][0]["action"], "RIGHT")
        ranking = next(
            item for item in snapshot["ranked_actions"] if item["action"] == "RIGHT"
        )
        model = snapshot["transition_models_here"][0]
        self.assertTrue(ranking["regime_adapted"])
        self.assertEqual(ranking["trials"], 2)
        self.assertEqual(ranking["novel"], 2)
        self.assertTrue(model["regime_adapted"])
        self.assertEqual(model["predicted_outcome"], "novel")

    def test_within_pass_regime_change_uses_recent_chronological_samples(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)
        current_id = frame_fingerprint(current)
        external = [
            {
                "before_state_id": current_id,
                "after_state_id": f"state-{index}",
                "action_display": "RIGHT",
                "outcome_class": outcome,
                "evidence_id": "same-live-pass",
            }
            for index, outcome in enumerate(
                ("exact_noop", "exact_noop", "novel", "novel")
            )
        ]

        snapshot = build_experience_snapshot(
            [], current, ["RIGHT"], config, external_transitions=external
        )

        self.assertEqual(
            snapshot["nonstationary_actions"][0]["recent_outcome"], "novel"
        )
        ranking = snapshot["ranked_actions"][0]
        self.assertTrue(ranking["regime_adapted"])
        self.assertEqual(ranking["trials"], 2)
        self.assertEqual(ranking["novel"], 2)

    def test_harm_veto_requires_a_strict_majority_when_evidence_conflicts(self) -> None:
        self.assertTrue(harmful_evidence_is_decisive(1, 1))
        self.assertFalse(harmful_evidence_is_decisive(1, 2))
        self.assertTrue(harmful_evidence_is_decisive(2, 3))

    def test_reset_is_reserved_below_healthy_exploration_actions(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)

        snapshot = build_experience_snapshot([], current, ["RESET", "RIGHT"], config)

        self.assertEqual(snapshot["ranked_actions"][0]["action"], "RIGHT")
        reset = next(
            item for item in snapshot["ranked_actions"] if item["action"] == "RESET"
        )
        self.assertEqual(reset["priority"], 6)
        self.assertIn("reserved", reset["reason"])

    def test_local_history_reward_reaches_action_value_model(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        start = _frame(1, step=0)
        changed = _frame(2, step=1)
        history = [
            HistoryEntry(action="", frame=start),
            HistoryEntry(action="RIGHT", frame=changed, reward=0.5),
            HistoryEntry(action="LEFT", frame=start),
        ]

        snapshot = build_experience_snapshot(history, start, ["RIGHT"], config)

        ranking = snapshot["ranked_actions"][0]
        self.assertEqual(ranking["mean_reward"], 0.5)
        self.assertGreater(ranking["expected_value"], 0.5)

    def setUp(self) -> None:
        self.config = InferenceControllerConfig(
            enabled=True,
            same_state_noop_limit=2,
            stagnation_window=3,
            cycle_window=4,
            volatile_min_samples=10,
        )

    def test_fingerprint_is_stable_and_level_sensitive(self) -> None:
        first = _frame(1, step=0)
        same_visible_state = _frame(1, step=99)
        next_level = _frame(1, step=0, level=2)

        self.assertEqual(
            frame_fingerprint(first), frame_fingerprint(same_visible_state)
        )
        self.assertNotEqual(frame_fingerprint(first), frame_fingerprint(next_level))

    def test_absent_configuration_enables_outcome_aware_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = InferenceControllerConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertEqual(config.same_state_noop_limit, 2)
        self.assertEqual(config.stagnation_window, 12)
        self.assertEqual(config.cycle_window, 8)
        self.assertEqual(config.policy, OUTCOME_AWARE_POLICY)

    def test_environment_enables_and_bounds_outcome_aware_policy(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_STRATEGY_ENABLED": "true",
                "LOCAL_ANALYZER_STRATEGY_POLICY": "outcome-aware",
                "LOCAL_ANALYZER_VOLATILE_WINDOW": "1",
                "LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES": "1",
                "LOCAL_ANALYZER_VOLATILE_RATIO": "2",
                "LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STRIKE_LIMIT": "3",
                "LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STOP_LIMIT": "8",
                "LOCAL_ANALYZER_LEVEL_NO_PROGRESS_TOKEN_LIMIT": "75000",
            },
            clear=True,
        ):
            config = InferenceControllerConfig.from_env()

        self.assertTrue(config.outcome_aware)
        self.assertEqual(config.policy, OUTCOME_AWARE_POLICY)
        self.assertEqual(config.volatile_window, 2)
        self.assertEqual(config.volatile_min_samples, 2)
        self.assertEqual(config.volatile_ratio, 1.0)
        self.assertEqual(config.directional_no_progress_strike_limit, 3)
        self.assertEqual(config.directional_no_progress_stop_limit, 8)
        self.assertEqual(config.level_no_progress_token_limit, 75000)

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

    def test_transient_animation_is_informative_and_not_guarded_as_noop(self) -> None:
        state = _frame(1, step=0)
        animation = {
            "transient_changed_cells": 1,
            "temporally_reversible": True,
        }
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="LEFT", frame=_frame(1, step=1), animation=animation),
            HistoryEntry(action="LEFT", frame=_frame(1, step=2), animation=animation),
        ]

        snapshot = build_experience_snapshot(
            history, history[-1].frame, ["LEFT"], self.config
        )

        self.assertEqual(snapshot["latest_outcome"], "transient_effect")
        self.assertEqual(snapshot["behavioral_no_op_streak"], 0)
        self.assertIsNone(
            action_guard_reason(history, history[-1].frame, "LEFT", self.config)
        )

    def test_decisive_local_terminal_evidence_is_guarded(self) -> None:
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(
                action="RIGHT",
                frame=_frame(1, step=1),
                outcome_class_override="terminal_failure",
            ),
        ]

        reason = action_guard_reason(history, state, "RIGHT", self.config)

        self.assertIn("terminal-failure evidence", reason or "")

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

        snapshot = build_experience_snapshot(
            history, history[-1].frame, ["LEFT", "RIGHT"], self.config
        )

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

        retained = build_experience_snapshot(
            run_history, second_level, ["SPACE"], self.config
        )
        fresh_pass = build_experience_snapshot(
            [HistoryEntry(action="", frame=first_level)],
            first_level,
            ["SPACE"],
            self.config,
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
        self.assertEqual(metadata["controller_phase"], "explore")

    def test_transition_metadata_reports_executed_actions_actual_rank(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="RIGHT", frame=_frame(1, step=1)),
        ]

        metadata = transition_metadata(
            history[-1].frame,
            _frame(2, step=2),
            history,
            "RIGHT",
            config,
            ["LEFT", "RIGHT"],
        )

        self.assertEqual(metadata["action_rank"], 2)
        self.assertIn("no-op", metadata["action_rank_reason"])

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

    def test_repeatable_exact_action_effect_is_not_erased_by_volatility_mask(
        self,
    ) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            volatile_window=8,
            volatile_min_samples=4,
            volatile_ratio=0.75,
        )
        state_a = Frame(grid=((1, 9),), step=0, level=1)
        state_b = Frame(grid=((2, 9),), step=1, level=1)
        history = [
            HistoryEntry(action="", frame=state_a),
            HistoryEntry(action="RIGHT", frame=state_b),
            HistoryEntry(action="LEFT", frame=state_a),
            HistoryEntry(action="RIGHT", frame=state_b),
            HistoryEntry(action="LEFT", frame=state_a),
        ]

        snapshot = build_experience_snapshot(history, state_a, ["RIGHT"], config)

        self.assertEqual(snapshot["volatile_cells"], 1)
        self.assertEqual(snapshot["latest_outcome"], "revisit")
        self.assertTrue(snapshot["recent_transitions"][-1]["repeatable_exact_effect"])
        self.assertEqual(
            snapshot["transition_models_here"][0]["predicted_outcome"], "revisit"
        )

    def test_volatility_evidence_does_not_cross_reset_boundary(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            volatile_window=8,
            volatile_min_samples=4,
            volatile_ratio=0.75,
        )
        frames = [
            Frame(grid=((value, 9),), step=index, level=1)
            for index, value in enumerate((1, 2, 3, 4, 5))
        ]
        reset_frame = Frame(grid=((1, 9),), step=5, level=1)
        current = Frame(grid=((2, 9),), step=6, level=1)
        history = [HistoryEntry(action="", frame=frames[0])] + [
            HistoryEntry(action="SPACE", frame=frame) for frame in frames[1:]
        ]
        history.extend(
            [
                HistoryEntry(action="RESET", frame=reset_frame),
                HistoryEntry(action="RIGHT", frame=current),
            ]
        )

        snapshot = build_experience_snapshot(history, current, ["RIGHT"], config)

        self.assertEqual(snapshot["volatile_cells"], 0)
        self.assertNotEqual(snapshot["latest_outcome"], "volatile_only")

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

        snapshot = build_experience_snapshot(
            history, history[-1].frame, ["RIGHT"], config
        )
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

        snapshot = build_experience_snapshot(
            history, history[-1].frame, ["RIGHT"], config
        )
        model = snapshot["transition_models_here"][0]

        self.assertFalse(model["verified_deterministic"])
        self.assertEqual(model["trials"], 2)
        self.assertEqual(model["confidence"], 0.5)
        self.assertEqual(model["contradictions"], 1)
        self.assertEqual(snapshot["model_conflicts_here"], 1)
        self.assertIn("transition_model_conflict", snapshot["recovery_reasons"])
        self.assertEqual(snapshot["phase"], "recover")

    def test_transition_confidence_uses_joint_state_outcome_observations(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        state = _frame(1, step=0)
        state_id = frame_fingerprint(state)
        transitions = [
            {
                "before_state_id": state_id,
                "after_state_id": "state-b",
                "action_display": "RIGHT",
                "outcome_class": outcome,
                "board_changed": True,
            }
            for outcome in ("novel", "novel", "exact_noop")
        ]

        snapshot = build_experience_snapshot(
            [], state, ["RIGHT"], config, external_transitions=transitions
        )
        model = snapshot["transition_models_here"][0]

        self.assertEqual(model["predicted_behavioral_state_id"], "state-b")
        self.assertEqual(model["predicted_outcome"], "novel")
        self.assertTrue(model["verified_state_deterministic"])
        self.assertFalse(model["verified_deterministic"])
        self.assertEqual(model["state_confidence"], 1.0)
        self.assertEqual(model["confidence"], 0.667)
        self.assertEqual(model["support"], 2)
        self.assertEqual(model["contradictions"], 1)

    def test_action_models_weight_independent_trials_not_raw_repetitions(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        state = _frame(1, step=0)
        state_id = frame_fingerprint(state)
        failed = {
            "before_state_id": state_id,
            "after_state_id": "terminal",
            "action_display": "RIGHT",
            "outcome_class": "terminal_failure",
            "reward": -1,
            "evidence_id": "noisy-pass",
        }
        successful = [
            {
                "before_state_id": state_id,
                "after_state_id": "progress",
                "action_display": "RIGHT",
                "outcome_class": "level_progress",
                "evidence_id": source,
            }
            for source in ("successful-pass-a", "successful-pass-b")
        ]

        snapshot = build_experience_snapshot(
            [],
            state,
            ["RIGHT"],
            config,
            external_transitions=[*[failed] * 6, *successful],
        )
        ranking = snapshot["ranked_actions"][0]
        model = snapshot["transition_models_here"][0]

        self.assertEqual(ranking["trials"], 3)
        self.assertEqual(ranking["raw_observations"], 8)
        self.assertFalse(ranking["harm_decisive"])
        self.assertEqual(model["trials"], 3)
        self.assertEqual(model["raw_observations"], 8)
        self.assertEqual(model["predicted_outcome"], "level_progress")
        self.assertEqual(model["confidence"], 0.667)
        self.assertEqual(model["terminal_failures"], 1)

    def test_mouse_family_keeps_coordinate_search_open(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        state = _frame(1, step=0)
        history = [
            HistoryEntry(action="", frame=state),
            HistoryEntry(action="MOUSE(row=1, col=1)", frame=_frame(1, step=1)),
        ]

        snapshot = build_experience_snapshot(
            history, history[-1].frame, ["MOUSE"], config
        )

        self.assertEqual(action_family("MOUSE(row=3, col=4)"), "MOUSE")
        self.assertTrue(snapshot["ranked_actions"][0]["parameterized"])
        self.assertEqual(snapshot["ranked_actions"][0]["priority"], 1)
        self.assertIsNone(
            action_guard_reason(
                history, history[-1].frame, "MOUSE(row=2, col=2)", config
            )
        )
        self.assertEqual(snapshot["mouse_search"]["unique_coordinates"], 1)
        self.assertEqual(snapshot["mouse_search"]["recent"][0]["row"], 1)
        self.assertTrue(snapshot["mouse_search"]["recommended_coordinates"])
        self.assertNotIn(
            {"row": 1, "col": 1, "reason": "spatial frontier"},
            snapshot["mouse_search"]["recommended_coordinates"],
        )

    def test_failed_mouse_coordinate_does_not_poison_untried_coordinates(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        state = _frame(1, step=0)
        state_id = frame_fingerprint(state)

        snapshot = build_experience_snapshot(
            [],
            state,
            ["MOUSE", "LEFT"],
            config,
            external_transitions=[
                {
                    "before_state_id": state_id,
                    "after_state_id": "terminal",
                    "action_display": "MOUSE(row=0, col=0)",
                    "outcome_class": "negative_reward",
                    "reward": -1,
                }
            ],
        )

        mouse = next(
            item for item in snapshot["ranked_actions"] if item["action"] == "MOUSE"
        )
        self.assertEqual(mouse["priority"], 1)
        self.assertFalse(mouse["harm_decisive"])
        self.assertGreaterEqual(mouse["expected_value"], 0)
        self.assertIn([0, 0], snapshot["mouse_search"]["blocked_coordinates"])
        self.assertNotIn(
            {"row": 0, "col": 0, "reason": "spatial frontier"},
            snapshot["mouse_search"]["recommended_coordinates"],
        )

    def test_verified_transition_graph_proposes_short_progress_plan(self) -> None:
        config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY, volatile_min_samples=20
        )
        a = _frame(1, step=0)
        b = _frame(2, step=1)
        progressed = _frame(3, step=2, level=2)
        history = [
            HistoryEntry(action="", frame=a),
            HistoryEntry(action="RIGHT", frame=b),
            HistoryEntry(action="SPACE", frame=progressed),
            HistoryEntry(action="RESET", frame=_frame(1, step=3)),
            HistoryEntry(action="RIGHT", frame=_frame(2, step=4)),
            HistoryEntry(action="SPACE", frame=_frame(3, step=5, level=2)),
            HistoryEntry(action="RESET", frame=_frame(1, step=6)),
        ]

        snapshot = build_experience_snapshot(
            history, history[-1].frame, ["RIGHT"], config
        )

        self.assertEqual(snapshot["recommended_plan"]["actions"], ["RIGHT", "SPACE"])
        self.assertEqual(snapshot["recommended_plan"]["target"], "level_progress")
        self.assertEqual(snapshot["recommended_plan"]["confidence"], 1.0)

    def test_plan_exposes_observation_contingencies_with_bounded_risk(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            plan_max_terminal_risk=0.25,
        )
        current, middle, progressed = (
            _frame(1, step=0),
            _frame(2, step=1),
            _frame(3, step=2, level=2),
        )
        current_id, middle_id, progress_id = map(
            frame_fingerprint, (current, middle, progressed)
        )
        external = [
            {
                "before_state_id": current_id,
                "after_state_id": middle_id,
                "action_display": "RIGHT",
                "outcome_class": "novel",
                "evidence_id": f"safe-{source}",
            }
            for source in range(3)
        ]
        external.append(
            {
                "before_state_id": current_id,
                "after_state_id": "terminal",
                "action_display": "RIGHT",
                "outcome_class": "terminal_failure",
                "evidence_id": "risky-branch",
            }
        )
        external.extend(
            {
                "before_state_id": middle_id,
                "after_state_id": progress_id,
                "action_display": "SPACE",
                "outcome_class": "level_progress",
                "evidence_id": f"goal-{source}",
            }
            for source in range(2)
        )

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["RIGHT"],
            config,
            external_transitions=external,
        )

        plan = snapshot["recommended_plan"]
        self.assertEqual(plan["actions"], ["RIGHT", "SPACE"])
        self.assertEqual(plan["terminal_risk"], 0.25)
        first_branches = plan["contingencies"][0]["branches"]
        self.assertEqual(
            [branch["probability"] for branch in first_branches], [0.75, 0.25]
        )
        self.assertEqual(first_branches[0]["next_action"], "SPACE")
        self.assertEqual(first_branches[0]["status"], "continue_verified_route")
        self.assertTrue(first_branches[1]["terminal"])
        self.assertEqual(first_branches[1]["status"], "abort_terminal_branch")
        self.assertIsNone(first_branches[1]["next_action"])
        self.assertEqual(plan["observation_policy"][0]["action"], "RIGHT")

    def test_action_ranking_assigns_discounted_credit_for_later_progress(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            credit_horizon=3,
            credit_discount=0.8,
        )
        current, middle, progressed = (
            _frame(1, step=0),
            _frame(2, step=1),
            _frame(3, step=2, level=2),
        )
        current_id, middle_id, progress_id = map(
            frame_fingerprint, (current, middle, progressed)
        )
        external = [
            {
                "before_state_id": current_id,
                "after_state_id": middle_id,
                "action_display": "RIGHT",
                "outcome_class": "novel",
                "evidence_id": "trial-a",
            },
            {
                "before_state_id": middle_id,
                "after_state_id": progress_id,
                "action_display": "SPACE",
                "outcome_class": "level_progress",
                "evidence_id": "trial-a",
            },
        ]

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["RIGHT"],
            config,
            external_transitions=external,
        )

        ranking = snapshot["ranked_actions"][0]
        self.assertEqual(ranking["action"], "RIGHT")
        self.assertEqual(ranking["delayed_progress_credit"], 0.8)
        self.assertEqual(
            ranking["reason"], "observed delayed progress after this action"
        )

    def test_single_observation_is_not_mislabeled_as_verified_plan(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        history = [
            HistoryEntry(action="", frame=_frame(1, step=0)),
            HistoryEntry(action="RIGHT", frame=_frame(2, step=1)),
            HistoryEntry(action="SPACE", frame=_frame(3, step=2, level=2)),
            HistoryEntry(action="RESET", frame=_frame(1, step=3)),
        ]

        snapshot = build_experience_snapshot(
            history, history[-1].frame, ["RIGHT"], config
        )

        self.assertIsNone(snapshot["recommended_plan"])

    def test_repeated_observations_from_one_evidence_source_do_not_verify_plan(
        self,
    ) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)
        current_id = frame_fingerprint(current)
        progress_id = frame_fingerprint(_frame(2, step=1, level=2))
        external = [
            {
                "before_state_id": current_id,
                "after_state_id": progress_id,
                "action_display": "RIGHT",
                "outcome_class": "level_progress",
                "evidence_id": "run-a:pass=0",
            }
            for _ in range(5)
        ]

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["RIGHT"],
            config,
            external_transitions=external,
            evidence_id="run-a:pass=0",
        )

        self.assertIsNone(snapshot["recommended_plan"])

    def test_repeated_raw_observations_cannot_override_independent_outcome_consensus(
        self,
    ) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)
        current_id = frame_fingerprint(current)
        middle_id = frame_fingerprint(_frame(2, step=1))
        progress_id = frame_fingerprint(_frame(3, step=2, level=2))
        external = [
            *[
                {
                    "before_state_id": current_id,
                    "after_state_id": middle_id,
                    "action_display": "RIGHT",
                    "outcome_class": "exact_noop",
                    "evidence_id": "run-a:pass=0",
                }
                for _ in range(6)
            ],
            *[
                {
                    "before_state_id": current_id,
                    "after_state_id": middle_id,
                    "action_display": "RIGHT",
                    "outcome_class": "novel",
                    "evidence_id": f"run-a:pass={pass_index}",
                }
                for pass_index in (1, 2, 3)
            ],
            *[
                {
                    "before_state_id": middle_id,
                    "after_state_id": progress_id,
                    "action_display": "SPACE",
                    "outcome_class": "level_progress",
                    "evidence_id": f"run-a:pass={pass_index}",
                }
                for pass_index in (1, 2)
            ],
        ]

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["RIGHT"],
            config,
            external_transitions=external,
        )

        self.assertEqual(snapshot["recommended_plan"]["actions"], ["RIGHT", "SPACE"])
        self.assertEqual(snapshot["recommended_plan"]["expected_utility"], 1.2)

    def test_persisted_exact_state_evidence_can_supply_a_verified_plan(self) -> None:
        config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY, volatile_min_samples=20
        )
        a, b, progressed = (
            _frame(1, step=0),
            _frame(2, step=1),
            _frame(3, step=2, level=2),
        )
        a_id, b_id, progress_id = map(frame_fingerprint, (a, b, progressed))
        external = [
            {
                "before_state_id": before,
                "after_state_id": after,
                "action_display": action,
                "outcome_class": outcome,
            }
            for _ in range(2)
            for before, after, action, outcome in (
                (a_id, b_id, "RIGHT", "novel"),
                (b_id, progress_id, "SPACE", "level_progress"),
            )
        ]

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=a)],
            a,
            ["RIGHT"],
            config,
            external_transitions=external,
        )

        self.assertEqual(snapshot["recommended_plan"]["actions"], ["RIGHT", "SPACE"])

    def test_planner_keeps_lower_confidence_route_when_utility_is_higher(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)
        detour = _frame(2, step=1)
        shared = _frame(3, step=2)
        progressed = _frame(4, step=3, level=2)
        current_id, detour_id, shared_id, progress_id = map(
            frame_fingerprint, (current, detour, shared, progressed)
        )

        def evidence(
            before: str,
            after: str,
            action: str,
            outcome: str,
            source: str,
        ) -> dict[str, str]:
            return {
                "before_state_id": before,
                "after_state_id": after,
                "action_display": action,
                "outcome_class": outcome,
                "evidence_id": source,
            }

        external = [
            evidence(current_id, shared_id, "RIGHT", "revisit", f"direct-{index}")
            for index in range(2)
        ]
        external.extend(
            evidence(current_id, detour_id, "LEFT", "novel", f"detour-{index}")
            for index in range(3)
        )
        external.append(
            evidence(current_id, detour_id, "LEFT", "exact_noop", "detour-noise")
        )
        external.extend(
            evidence(detour_id, shared_id, "UP", "novel", f"join-{index}")
            for index in range(2)
        )
        external.extend(
            evidence(shared_id, progress_id, "SPACE", "level_progress", f"goal-{index}")
            for index in range(2)
        )

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["RIGHT", "LEFT"],
            config,
            external_transitions=external,
        )

        self.assertEqual(
            snapshot["recommended_plan"]["actions"], ["LEFT", "UP", "SPACE"]
        )
        self.assertEqual(snapshot["recommended_plan"]["expected_utility"], 1.4)

    def test_planner_ranks_after_bounded_search_instead_of_first_four_goals(
        self,
    ) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)
        detour = _frame(2, step=1)
        progressed = _frame(3, step=2, level=2)
        current_id, detour_id, progress_id = map(
            frame_fingerprint, (current, detour, progressed)
        )
        external: list[dict[str, str]] = []
        for action in ("UP", "DOWN", "LEFT", "RIGHT"):
            for source in range(2):
                external.append(
                    {
                        "before_state_id": current_id,
                        "after_state_id": progress_id,
                        "action_display": action,
                        "outcome_class": "level_progress",
                        "evidence_id": f"{action}-{source}",
                    }
                )
        for source in range(2):
            external.extend(
                [
                    {
                        "before_state_id": current_id,
                        "after_state_id": detour_id,
                        "action_display": "SPACE",
                        "outcome_class": "novel",
                        "evidence_id": f"detour-{source}",
                    },
                    {
                        "before_state_id": detour_id,
                        "after_state_id": progress_id,
                        "action_display": "ACTION6",
                        "outcome_class": "level_progress",
                        "evidence_id": f"finish-{source}",
                    },
                ]
            )

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"],
            config,
            external_transitions=external,
        )

        self.assertEqual(snapshot["recommended_plan"]["actions"], ["SPACE", "ACTION6"])
        self.assertEqual(len(snapshot["plan_candidates"]), 4)

    def test_planner_does_not_inflate_utility_by_revisiting_states(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        current = _frame(1, step=0)
        middle = _frame(2, step=1)
        progressed = _frame(3, step=2, level=2)
        current_id, middle_id, progress_id = map(
            frame_fingerprint, (current, middle, progressed)
        )
        external = [
            {
                "before_state_id": before,
                "after_state_id": after,
                "action_display": action,
                "outcome_class": outcome,
                "evidence_id": f"{action}-{source}",
            }
            for source in range(2)
            for before, after, action, outcome in (
                (current_id, middle_id, "RIGHT", "novel"),
                (middle_id, current_id, "LEFT", "revisit"),
                (middle_id, progress_id, "SPACE", "level_progress"),
            )
        ]

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=current)],
            current,
            ["RIGHT"],
            config,
            external_transitions=external,
        )

        self.assertEqual(snapshot["recommended_plan"]["actions"], ["RIGHT", "SPACE"])

    def test_plan_rejects_currently_invalid_or_contextually_invalid_actions(
        self,
    ) -> None:
        config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY, volatile_min_samples=20
        )
        a, b, progressed = (
            _frame(1, step=0),
            _frame(2, step=1),
            _frame(3, step=2, level=2),
        )
        a_id, b_id, progress_id = map(frame_fingerprint, (a, b, progressed))
        external = [
            {
                "before_state_id": before,
                "after_state_id": after,
                "action_display": action,
                "outcome_class": outcome,
                "valid_actions_after": valid_after,
            }
            for _ in range(2)
            for before, after, action, outcome, valid_after in (
                (a_id, b_id, "RIGHT", "novel", ["LEFT"]),
                (b_id, progress_id, "SPACE", "level_progress", []),
            )
        ]

        invalid_first = build_experience_snapshot(
            [HistoryEntry(action="", frame=a)],
            a,
            ["LEFT"],
            config,
            external_transitions=external,
        )
        invalid_second = build_experience_snapshot(
            [HistoryEntry(action="", frame=a)],
            a,
            ["RIGHT"],
            config,
            external_transitions=external,
        )

        self.assertIsNone(invalid_first["recommended_plan"])
        self.assertIsNone(invalid_second["recommended_plan"])

    def test_local_plan_respects_recorded_downstream_action_context(self) -> None:
        config = InferenceControllerConfig(
            enabled=True,
            policy=OUTCOME_AWARE_POLICY,
            plan_min_support=1,
            plan_min_confidence=0.5,
            volatile_min_samples=20,
        )
        start = Frame(grid=((1,),), step=0, level=1, valid_actions=("RIGHT",))
        middle = Frame(grid=((2,),), step=1, level=1, valid_actions=("LEFT",))
        progressed = Frame(grid=((3,),), step=2, level=2, valid_actions=("RIGHT",))
        history = [
            HistoryEntry(action="", frame=start),
            HistoryEntry(action="RIGHT", frame=middle),
            HistoryEntry(action="SPACE", frame=progressed),
        ]

        snapshot = build_experience_snapshot(history, start, ["RIGHT"], config)

        self.assertIsNone(snapshot["recommended_plan"])
        recent = snapshot["recent_transitions"]
        self.assertEqual(recent[0]["valid_actions_after"], ["LEFT"])

    def test_terminal_transition_is_classified_as_harm_and_not_planned(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        before, after = _frame(1, step=0), _frame(2, step=1)
        metadata = transition_metadata(
            before,
            after,
            [HistoryEntry(action="", frame=before)],
            "RIGHT",
            config,
            ["RIGHT"],
            game_over=True,
        )

        self.assertEqual(metadata["outcome_class"], "terminal_failure")
        self.assertTrue(metadata["game_over"])

    def test_rankings_expose_uncertainty_and_information_gain(self) -> None:
        config = InferenceControllerConfig(enabled=True, policy=OUTCOME_AWARE_POLICY)
        frame = _frame(1, step=0)

        snapshot = build_experience_snapshot(
            [HistoryEntry(action="", frame=frame)], frame, ["LEFT", "RIGHT"], config
        )

        self.assertTrue(
            all("information_gain" in item for item in snapshot["ranked_actions"])
        )
        self.assertEqual(snapshot["ranked_actions"][0]["trials"], 0)

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

        self.assertEqual(snapshot["phase"], "explore")
        self.assertEqual(snapshot["action_budget"], 1)

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
