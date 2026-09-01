from __future__ import annotations

import json
import unittest
from itertools import repeat

import numpy as np

from inference.agent.policy_codegen_helpers import (
    POLICY_ACTIONS,
    POLICY_BOARD_HEX_SYMBOLS,
    POLICY_CODEGEN_API_VERSION,
    POLICY_CODEGEN_GLOBALS,
    accumulate_transition_evidence,
    action_payload,
    board_digest,
    cells_digest,
    consecutive_outcome_count,
    contrastive_transition_evidence_ready,
    contrastive_transition_evidence_status,
    continue_decision,
    edge_run_length,
    edge_value_count,
    first_matching_cell,
    history_push,
    infer_game_state,
    infer_game_type,
    least_tried_action,
    least_tried_mouse_point,
    line_run_length,
    line_value_count,
    matching_region_center,
    memory_increment,
    memory_mapping_increment,
    memory_push,
    memory_update,
    memory_with_defaults,
    mouse_decision,
    nearest_matching_cell,
    objective_evidence_ready,
    palette_value,
    palette_values,
    path_decision,
    region_digest,
    subgoal_failed,
    subgoal_succeeded,
    recent_action_counts,
    recent_mouse_point_counts,
    recent_outcome_counts,
    transition_facts,
    transition_change_class,
    transition_has_progress,
    transition_has_stable_change,
    transition_outcome,
    transition_repeats_nonprogress_action,
    transition_requires_replan,
    stable_transition_evidence_ready,
    stable_transition_evidence_status,
)


class PolicyCodegenHelperTests(unittest.TestCase):
    def test_game_state_inference_returns_bounded_structural_and_action_evidence(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[5:7, 8:10] = 2
        board[63, 3:6] = 3
        board.setflags(write=False)
        transitions = (
            {
                "action": "RIGHT",
                "executed": True,
                "post_action_observed": True,
                "board_changed": True,
                "outcome_class": "novel",
                "meaningful_progress": False,
                "cycle_risk": False,
                "loop_detected": False,
            },
            {
                "action": "MOUSE",
                "row": 10,
                "col": 11,
                "executed": True,
                "post_action_observed": True,
                "board_changed": False,
                "outcome_class": "exact_noop",
                "meaningful_progress": False,
                "cycle_risk": False,
                "loop_detected": False,
            },
        )
        observation = type("Observation", (), {})()
        observation.board = board
        observation.level = 2
        observation.step = 7
        observation.valid_actions = POLICY_ACTIONS
        observation.last_transition = transitions[-1]
        observation.recent_transitions = transitions
        observation.objective = {
            "objective_id": "tactical:4",
            "evidence_mode": "engine_progress",
            "action_budget": 8,
            "actions_used": 3,
        }

        state = infer_game_state(observation)

        self.assertEqual("active", state["phase"])
        self.assertEqual(0, state["board"]["background_value"])
        self.assertEqual(7, state["board"]["foreground_count"])
        self.assertEqual([5, 3, 63, 9], state["board"]["foreground_bbox"])
        self.assertEqual(5, state["objective"]["remaining_actions"])
        self.assertEqual(1, state["recent"]["actions"]["RIGHT"]["stable_changed"])
        self.assertEqual("responsive", state["recent"]["actions"]["RIGHT"]["classification"])
        self.assertEqual(1, state["recent"]["actions"]["MOUSE"]["distinct_points"])
        self.assertEqual("inconclusive", state["recent"]["actions"]["MOUSE"]["classification"])
        self.assertEqual(5, state["schema_version"])
        self.assertIn("horizontal_symmetry", state["board"])
        self.assertLess(len(json.dumps(state)), 32_768)
        self.assertFalse(board.flags.writeable)

    def test_game_type_inference_ranks_observed_movement_without_overclaiming(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        transition = {
            "action": "RIGHT",
            "executed": True,
            "post_action_observed": True,
            "board_changed": True,
            "outcome_class": "novel",
            "meaningful_progress": False,
            "cycle_risk": False,
            "loop_detected": False,
        }
        observation = type("Observation", (), {})()
        observation.board = board
        observation.level = 1
        observation.step = 1
        observation.valid_actions = POLICY_ACTIONS
        observation.last_transition = transition
        observation.recent_transitions = (transition, transition)
        observation.objective = {}

        inferred = infer_game_type(observation)

        self.assertEqual("routing", inferred["primary_family"])
        self.assertIn("navigation", inferred["recommended_solver_types"])
        self.assertNotEqual("high", inferred["confidence"])
        self.assertNotIn("engine_progress", inferred)
        self.assertEqual(5, inferred["schema_version"])
        self.assertEqual(2, inferred["evidence_coverage"]["executed_actions"])
        self.assertEqual("UP", inferred["recommended_probes"][0]["action"])

    def test_game_inference_reports_inverted_directional_controls(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)

        def transition(action: str, direction: str) -> dict[str, object]:
            return {
                "action": action,
                "executed": True,
                "post_action_observed": True,
                "board_changed": True,
                "outcome_class": "novel",
                "meaningful_progress": False,
                "cycle_risk": False,
                "loop_detected": False,
                "animation_summary": {"motion_direction": direction},
            }

        transitions = (
            transition("UP", "down"),
            transition("UP", "down"),
            transition("LEFT", "right"),
            transition("LEFT", "right"),
        )
        observation = type("Observation", (), {})()
        observation.board = board
        observation.level = 1
        observation.step = 4
        observation.valid_actions = ("UP", "LEFT")
        observation.last_transition = transitions[-1]
        observation.recent_transitions = transitions
        observation.objective = {}

        state = infer_game_state(observation)
        inferred = infer_game_type(observation)

        dynamics = state["controls"]["dynamics"]
        self.assertEqual("inverted", dynamics["scheme"])
        self.assertEqual("high", dynamics["confidence"])
        self.assertEqual("down", dynamics["by_action"]["UP"]["dominant_motion_direction"])
        self.assertEqual(1.0, dynamics["by_action"]["LEFT"]["consistency"])
        self.assertEqual(dynamics, inferred["control_scheme"])

    def test_game_inference_ranks_opposing_transition_motion_as_multi_agent(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        transition = {
            "action": "RIGHT",
            "executed": True,
            "post_action_observed": True,
            "board_changed": True,
            "outcome_class": "novel",
            "meaningful_progress": False,
            "cycle_risk": False,
            "loop_detected": False,
            "animation_summary": {
                "motion_direction": None,
                "object_motion": {
                    "tracking_available": True,
                    "classification": "opposing",
                    "distinct_shifts_twice": [[0, -6], [0, 6]],
                },
            },
        }
        observation = type("Observation", (), {})()
        observation.board = board
        observation.level = 1
        observation.step = 2
        observation.valid_actions = ("LEFT", "RIGHT")
        observation.last_transition = transition
        observation.recent_transitions = (transition, transition)
        observation.objective = {}

        state = infer_game_state(observation)
        inferred = infer_game_type(observation)

        motion = state["controls"]["dynamics"]["object_motion"]
        self.assertEqual("linked_opposing", motion["scheme"])
        self.assertEqual(2, motion["classifications"]["opposing"])
        self.assertEqual(
            [[0, -6], [0, 6]],
            motion["by_action"]["RIGHT"]["observed_shift_sets_twice"][0][
                "shifts"
            ],
        )
        self.assertEqual("multi_agent", inferred["primary_family"])
        self.assertEqual("linked_opposing", inferred["object_motion"]["transition_scheme"])

    def test_game_type_inference_reports_high_confidence_execution_conflict(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        transitions = tuple(
            {
                "action": "MOUSE",
                "row": 10,
                "col": 10,
                "executed": True,
                "post_action_observed": True,
                "board_changed": True,
                "outcome_class": "novel",
                "meaningful_progress": False,
                "cycle_risk": False,
                "loop_detected": False,
            }
            for _ in range(4)
        )
        observation = type("Observation", (), {})()
        observation.board = board
        observation.level = 1
        observation.step = 4
        observation.valid_actions = POLICY_ACTIONS
        observation.last_transition = transitions[-1]
        observation.recent_transitions = transitions
        observation.objective = {"execution_mode": "navigate"}

        inferred = infer_game_type(observation)

        self.assertEqual("interaction", inferred["primary_family"])
        self.assertEqual("high", inferred["confidence"])
        self.assertEqual("conflict", inferred["objective_alignment"]["status"])

    def test_state_delta_distinguishes_translation_from_transform(self) -> None:
        def make_observation(board: np.ndarray, step: int) -> object:
            observation = type("Observation", (), {})()
            observation.board = board
            observation.level = 1
            observation.step = step
            observation.valid_actions = ("RIGHT", "SPACE")
            observation.last_transition = {
                "action": "RIGHT",
                "executed": True,
                "post_action_observed": True,
                "board_changed": True,
                "outcome_class": "novel",
                "meaningful_progress": False,
                "cycle_risk": False,
                "loop_detected": False,
            }
            observation.recent_transitions = (observation.last_transition,)
            observation.objective = {}
            return observation

        before = np.zeros((64, 64), dtype=np.uint8)
        before[10:12, 10:12] = 2
        after = np.zeros((64, 64), dtype=np.uint8)
        after[10:12, 13:15] = 2
        first = infer_game_state(make_observation(before, 1))

        translated = infer_game_state(
            make_observation(after, 2), first["state_token"]
        )
        inferred = infer_game_type(
            make_observation(after, 2), first["state_token"]
        )

        self.assertEqual(
            "translation_candidate", translated["state_delta"]["change_type"]
        )
        self.assertEqual([0, 6], translated["state_delta"]["bbox_center_shift_twice"])
        self.assertEqual("routing", inferred["primary_family"])
        self.assertEqual("translation_candidate", inferred["state_change_type"])

        transformed = after.copy()
        transformed[10, 13] = 3
        transform_state = infer_game_state(
            make_observation(transformed, 3), translated["state_token"]
        )
        transform_type = infer_game_type(
            make_observation(transformed, 3), translated["state_token"]
        )

        self.assertEqual(
            "recolor_or_transform", transform_state["state_delta"]["change_type"]
        )
        self.assertEqual("transform", transform_type["primary_family"])

    def test_state_delta_tracks_opposing_object_motion(self) -> None:
        def make_observation(board: np.ndarray) -> object:
            observation = type("Observation", (), {})()
            observation.board = board
            observation.level = 1
            observation.step = 1
            observation.valid_actions = ("LEFT", "RIGHT")
            observation.last_transition = None
            observation.recent_transitions = ()
            observation.objective = {}
            return observation

        before = np.zeros((64, 64), dtype=np.uint8)
        before[10:12, 10:12] = 2
        before[10:12, 30:32] = 2
        after = np.zeros((64, 64), dtype=np.uint8)
        after[10:12, 12:14] = 2
        after[10:12, 28:30] = 2
        first = infer_game_state(make_observation(before))

        second = infer_game_state(make_observation(after), first["state_token"])
        inferred = infer_game_type(make_observation(after), first["state_token"])

        changes = second["state_delta"]["object_changes"]
        self.assertEqual("translation_candidate", second["state_delta"]["change_type"])
        self.assertEqual(2, len(changes["moved"]))
        self.assertEqual(
            [[0, -4], [0, 4]],
            sorted(item["shift_twice"] for item in changes["moved"]),
        )
        self.assertEqual(2, inferred["object_motion"]["moved_objects"])
        self.assertTrue(inferred["object_motion"]["independent_motion"])
        self.assertEqual("multi_agent", inferred["primary_family"])

    def test_state_inference_summarizes_bounded_object_relations(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[10:12, 10:12] = 2
        board[10:12, 12:14] = 3
        observation = type("Observation", (), {})()
        observation.board = board
        observation.level = 1
        observation.step = 0
        observation.valid_actions = ("UP",)
        observation.last_transition = None
        observation.recent_transitions = ()
        observation.objective = {}

        state = infer_game_state(observation)

        layout = state["board"]["object_layout"]
        self.assertEqual(1, layout["relation_count"])
        self.assertEqual(1, layout["counts"]["bbox_contact_candidates"])
        self.assertEqual(2, layout["relations"][0]["row_overlap"])
        self.assertEqual(1, len(layout["repeated_shapes"]))

    def test_state_inference_bounds_dense_component_inventory(self) -> None:
        board = np.indices((64, 64)).sum(axis=0).astype(np.uint8) % 2
        observation = type("Observation", (), {})()
        observation.board = board
        observation.level = 1
        observation.step = 0
        observation.valid_actions = ("UP",)
        observation.last_transition = None
        observation.recent_transitions = ()
        observation.objective = {}

        state = infer_game_state(observation)

        self.assertTrue(state["board"]["objects_truncated"])
        self.assertEqual(32, len(state["state_token"]["objects"]))
        self.assertLess(len(json.dumps(state)), 32_768)

    def test_state_inference_resets_scope_and_preserves_background_continuity(self) -> None:
        def make_observation(board: np.ndarray, level: int) -> object:
            observation = type("Observation", (), {})()
            observation.board = board
            observation.level = level
            observation.step = 0
            observation.valid_actions = ("UP",)
            observation.last_transition = None
            observation.recent_transitions = ()
            observation.objective = {}
            return observation

        before = np.zeros((64, 64), dtype=np.uint8)
        before[:20, :] = 1
        after = np.zeros((64, 64), dtype=np.uint8)
        after[:40, :] = 1
        first = infer_game_state(make_observation(before, 1))

        continued = infer_game_state(
            make_observation(after, 1), first["state_token"]
        )
        reset = infer_game_state(make_observation(after, 2), first["state_token"])

        self.assertEqual(0, continued["board"]["background_value"])
        self.assertEqual("previous_state", continued["board"]["background_source"])
        self.assertTrue(continued["state_delta"]["comparable"])
        self.assertFalse(reset["state_delta"]["comparable"])
        self.assertEqual("scope_reset", reset["state_delta"]["change_type"])
        self.assertEqual(
            "level_or_shape_changed", reset["state_delta"]["scope_reset_reason"]
        )

    def test_action_payload_canonicalizes_names_and_mappings(self) -> None:
        self.assertEqual({"action": "UP"}, action_payload(" up "))
        self.assertEqual(
            {"action": "MOUSE", "row": 12, "col": 34},
            action_payload("mouse", (12, 34)),
        )
        self.assertEqual(
            {"action": "MOUSE", "row": 4, "col": 5},
            action_payload({"action": "MOUSE", "row": 4, "col": 5}),
        )

    def test_action_payload_rejects_invalid_names_and_coordinates(self) -> None:
        cases = (
            (("ACTION1", None), "one of"),
            (("ACTION7", None), "one of"),
            (("MOUSE", None), "requires"),
            (("MOUSE", (64, 0)), "between 0 and 63"),
            (("UP", (1, 2)), "does not accept"),
            (("MOUSE", (True, 2)), "integers"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    action_payload(*arguments)

    def test_decision_builders_emit_exact_wire_contracts(self) -> None:
        self.assertEqual(
            {
                "status": "continue",
                "action": {"action": "RIGHT"},
                "memory": {"step": 1},
                "evidence": "advance",
                "prediction": {"row": 2},
            },
            continue_decision("RIGHT", {"step": 1}, "advance", prediction={"row": 2}),
        )
        self.assertEqual(
            {"action": "MOUSE", "row": 2, "col": 3},
            mouse_decision((2, 3), {}, "click")["action"],
        )
        self.assertEqual(
            {
                "status": "subgoal_succeeded",
                "action": None,
                "memory": {},
                "evidence": "done",
            },
            subgoal_succeeded({}, "done"),
        )
        self.assertEqual("subgoal_failed", subgoal_failed({}, "blocked")["status"])
        with self.assertRaisesRegex(ValueError, "prediction"):
            continue_decision("UP", {}, prediction=[1])

    def test_path_decision_continues_or_fails_without_malformed_action(self) -> None:
        self.assertEqual(
            {"action": "RIGHT"},
            path_decision(((1, 1), (1, 2)), ("RIGHT",), {}, "route")["action"],
        )
        failed = path_decision(((1, 1), (1, 2)), ("UP",), {}, "route blocked")
        self.assertEqual("subgoal_failed", failed["status"])
        self.assertIsNone(failed["action"])
        self.assertEqual("route blocked", failed["evidence"])

    def test_transition_outcome_precedence_and_categories(self) -> None:
        cases = (
            (None, "unknown"),
            ({"game_over": True, "error": "loss"}, "terminal"),
            ({"executed": False}, "failed"),
            ({"error": "worker"}, "failed"),
            ({"cycle_risk": True, "meaningful_progress": True}, "guarded"),
            ({"loop_detected": True}, "guarded"),
            ({"outcome_class": "guarded"}, "guarded"),
            ({"meaningful_progress": True}, "progress"),
            ({"outcome_class": "exact_noop"}, "no_progress"),
            ({"executed": True, "board_changed": False}, "no_progress"),
            ({"executed": True, "board_changed": True}, "unknown"),
        )
        for transition, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(expected, transition_outcome(transition))

    def test_transition_progress_and_replan_helpers_preserve_terminal_semantics(
        self,
    ) -> None:
        self.assertTrue(transition_has_progress({"level_completed": True}))
        self.assertTrue(transition_has_progress({"run_complete": True}))
        self.assertFalse(transition_has_progress({"game_over": True}))
        self.assertTrue(transition_requires_replan({"game_over": True}))
        noop = {"outcome_class": "behavioral_noop"}
        self.assertTrue(transition_requires_replan(noop))
        self.assertFalse(transition_requires_replan(noop, False))
        self.assertFalse(transition_requires_replan({"meaningful_progress": True}))

    def test_change_helpers_preserve_novel_learning_evidence_without_progress(
        self,
    ) -> None:
        novel = {
            "executed": True,
            "board_changed": True,
            "outcome_class": "novel",
            "meaningful_progress": False,
        }
        self.assertEqual("no_progress", transition_outcome(novel))
        self.assertEqual("novel", transition_change_class(novel))
        self.assertTrue(transition_has_stable_change(novel))
        self.assertFalse(transition_has_progress(novel))
        for unstable in (
            {**novel, "outcome_class": "volatile_only"},
            {**novel, "loop_detected": True},
            {**novel, "cycle_risk": True},
            {**novel, "board_changed": False},
            {**novel, "executed": False},
        ):
            with self.subTest(unstable=unstable):
                self.assertFalse(transition_has_stable_change(unstable))

    def test_transition_facts_and_evidence_accumulation_are_json_bounded(self) -> None:
        transition = {
            "action": "mouse",
            "row": 4,
            "col": 5,
            "executed": True,
            "post_action_observed": True,
            "board_changed": False,
            "meaningful_progress": False,
            "outcome_class": "exact_noop",
            "error": "x" * 600,
        }
        facts = transition_facts(transition)
        self.assertEqual("MOUSE", facts["action"])
        self.assertEqual([4, 5], facts["point"])
        self.assertEqual("failed", facts["outcome"])
        self.assertEqual("failed", facts["change_class"])
        self.assertEqual(512, len(facts["error"]))
        memory = accumulate_transition_evidence({}, transition, limit=2)
        memory = accumulate_transition_evidence(memory, None, limit=2)
        self.assertEqual(2, len(memory["transition_evidence"]))
        self.assertEqual("unknown", memory["transition_evidence"][-1]["outcome"])

    def test_objective_evidence_ready_filters_objectives_and_unobserved_actions(
        self,
    ) -> None:
        objective = {"objective_id": "tactical:1", "minimum_evidence_actions": 2}
        transitions = (
            {
                "objective_id": "other",
                "executed": True,
                "post_action_observed": True,
            },
            {
                "objective_id": "tactical:1",
                "executed": True,
                "post_action_observed": False,
            },
            {
                "objective_id": "tactical:1",
                "executed": True,
                "outcome_class": "exact_noop",
            },
            {
                "objective_id": "tactical:1",
                "executed": True,
                "meaningful_progress": True,
            },
        )
        self.assertTrue(objective_evidence_ready(objective, transitions))
        self.assertFalse(
            objective_evidence_ready(
                {**objective, "minimum_evidence_actions": 3}, transitions
            )
        )
        self.assertTrue(objective_evidence_ready({"minimum_evidence_actions": 0}, ()))
        with self.assertRaisesRegex(ValueError, "0 through 32"):
            objective_evidence_ready({"minimum_evidence_actions": True}, ())

    def test_exact_nonprogress_repeat_includes_mouse_coordinates(self) -> None:
        movement = {"action": "LEFT", "executed": True, "board_changed": False}
        self.assertTrue(transition_repeats_nonprogress_action(movement, "left"))
        self.assertFalse(transition_repeats_nonprogress_action(movement, "right"))
        click = {
            "action": "MOUSE",
            "row": 4,
            "col": 5,
            "outcome_class": "exact_noop",
        }
        self.assertTrue(transition_repeats_nonprogress_action(click, "MOUSE", (4, 5)))
        self.assertFalse(transition_repeats_nonprogress_action(click, "MOUSE", (4, 6)))
        self.assertFalse(
            transition_repeats_nonprogress_action(
                {"action": "LEFT", "meaningful_progress": True}, "LEFT"
            )
        )

    def test_codegen_registry_is_versioned_immutable_and_complete(self) -> None:
        expected = {
            "accumulate_transition_evidence",
            "action_payload",
            "cells_digest",
            "continue_decision",
            "contrastive_transition_evidence_ready",
            "edge_run_length",
            "first_matching_cell",
            "least_tried_mouse_point",
            "line_value_count",
            "memory_with_defaults",
            "memory_mapping_increment",
            "mouse_decision",
            "nearest_matching_cell",
            "objective_evidence_ready",
            "palette_value",
            "palette_values",
            "path_decision",
            "region_digest",
            "subgoal_failed",
            "subgoal_succeeded",
            "transition_facts",
            "transition_change_class",
            "transition_has_stable_change",
            "transition_outcome",
            "transition_requires_replan",
            "stable_transition_evidence_ready",
        }
        self.assertEqual(1, POLICY_CODEGEN_API_VERSION)
        self.assertEqual(1, POLICY_CODEGEN_GLOBALS["POLICY_CODEGEN_API_VERSION"])
        self.assertEqual(POLICY_ACTIONS, POLICY_CODEGEN_GLOBALS["POLICY_ACTIONS"])
        self.assertEqual(
            POLICY_BOARD_HEX_SYMBOLS,
            POLICY_CODEGEN_GLOBALS["POLICY_BOARD_HEX_SYMBOLS"],
        )
        self.assertTrue(expected.issubset(POLICY_CODEGEN_GLOBALS))
        self.assertTrue(
            all(callable(POLICY_CODEGEN_GLOBALS[name]) for name in expected)
        )
        with self.assertRaises(TypeError):
            POLICY_CODEGEN_GLOBALS["unsafe"] = object()

    def test_board_digest_is_compact_deterministic_and_content_sensitive(self) -> None:
        board = np.arange(16, dtype=np.uint8).reshape(4, 4)
        first = board_digest(board)
        self.assertEqual(first, board_digest(board.copy()))
        self.assertEqual(16, len(first))
        changed = board.copy()
        changed[2, 2] += 1
        self.assertNotEqual(first, board_digest(changed))
        self.assertNotEqual(first, board_digest(board.astype(np.uint16)))
        with self.assertRaisesRegex(ValueError, "4096-cell"):
            board_digest(np.zeros((65, 64), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "object"):
            board_digest(np.full((2, 2), object(), dtype=object))

    def test_region_and_cell_digests_are_bounded_and_content_sensitive(self) -> None:
        board = np.arange(36, dtype=np.uint8).reshape(6, 6)
        self.assertEqual(
            board_digest(board[1:4, 2:5]), region_digest(board, (1, 2, 3, 4))
        )
        self.assertEqual(
            cells_digest(board, ((1, 1), (4, 4))),
            cells_digest(board, ((4, 4), (1, 1), (1, 1))),
        )
        changed = board.copy()
        changed[4, 4] += 1
        self.assertNotEqual(
            cells_digest(board, ((1, 1), (4, 4))),
            cells_digest(changed, ((1, 1), (4, 4))),
        )
        with self.assertRaisesRegex(ValueError, "outside grid"):
            region_digest(board, (0, 0, 6, 5))
        with self.assertRaisesRegex(ValueError, "outside grid"):
            cells_digest(board, ((6, 0),))

    def test_matching_cell_selectors_are_deterministic(self) -> None:
        board = np.zeros((5, 6), dtype=np.uint8)
        board[0, 5] = 7
        board[2, 2] = 7
        board[4, 0] = 7
        self.assertEqual((0, 5), first_matching_cell(board, 7))
        self.assertEqual((2, 2), nearest_matching_cell(board, 7, (2, 3)))
        self.assertEqual((2, 2), matching_region_center(board, 7))
        self.assertIsNone(first_matching_cell(board, 9))
        self.assertIsNone(nearest_matching_cell(board, (), (0, 0)))
        self.assertIsNone(matching_region_center(board, 9))

    def test_line_and_edge_statistics_generalize_generated_counters(self) -> None:
        board = np.array(
            [
                [7, 7, 0, 7],
                [7, 0, 0, 7],
                [7, 7, 7, 7],
                [0, 7, 7, 7],
            ],
            dtype=np.uint8,
        )
        self.assertEqual(3, line_value_count(board, 7, "row", 0))
        self.assertEqual(4, line_value_count(board, 7, 1, 3))
        self.assertEqual(2, line_run_length(board, 7, "row", 0))
        self.assertEqual(1, line_run_length(board, 7, "row", 0, from_end=True))
        self.assertEqual(3, edge_value_count(board, 7, "bottom"))
        self.assertEqual(2, edge_run_length(board, 7, "bottom", 1))
        self.assertEqual(4, edge_run_length(board, 7, "right", 2))
        with self.assertRaisesRegex(ValueError, "axis"):
            line_value_count(board, 7, "diagonal", 0)
        with self.assertRaisesRegex(ValueError, "edge"):
            edge_value_count(board, 7, "middle")

    def test_memory_update_copies_normalizes_and_bounds_json(self) -> None:
        original = {"count": 1, "items": []}
        updated = memory_update(original, {"items": (1, 2), "new": True})
        self.assertEqual(
            {"count": 1, "items": [1, 2], "new": True},
            updated,
        )
        self.assertEqual({"count": 1, "items": []}, original)
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            memory_update({}, {"bad": float("nan")})
        with self.assertRaisesRegex(ValueError, "at most 64 keys"):
            memory_update({}, {str(index): index for index in range(65)})
        with self.assertRaisesRegex(ValueError, "32768 bytes"):
            memory_update({}, {"large": "x" * 33_000})

    def test_memory_defaults_preserve_existing_values_without_aliasing(self) -> None:
        defaults = {"phase": "probe", "history": []}
        memory = {"phase": "execute"}
        result = memory_with_defaults(memory, defaults)
        self.assertEqual({"phase": "execute", "history": []}, result)
        result["history"].append(1)
        self.assertEqual([], defaults["history"])
        with self.assertRaisesRegex(ValueError, "defaults must be a mapping"):
            memory_with_defaults({}, [])

    def test_memory_push_rolls_history_without_mutation(self) -> None:
        original = {"history": [1, 2]}
        updated = memory_push(original, "history", {"step": 3}, limit=2)
        self.assertEqual({"history": [2, {"step": 3}]}, updated)
        self.assertEqual({"history": [1, 2]}, original)
        with self.assertRaisesRegex(ValueError, "must be a list"):
            memory_push({"history": 1}, "history", 2)
        with self.assertRaisesRegex(ValueError, "between 1 and 64"):
            memory_push({}, "history", 1, limit=0)

    def test_history_push_supports_standalone_bounded_lists(self) -> None:
        original = [{"step": 1}, {"step": 2}]
        updated = history_push(original, {"step": 3}, limit=2)
        self.assertEqual([{"step": 2}, {"step": 3}], updated)
        self.assertEqual([{"step": 1}, {"step": 2}], original)
        self.assertEqual([1], history_push(None, 1))
        with self.assertRaisesRegex(ValueError, "history must be"):
            history_push({"not": "a list"}, 1)
        with self.assertRaisesRegex(ValueError, "between 1 and 64"):
            history_push([], 1, limit=65)
        self.assertIs(POLICY_CODEGEN_GLOBALS["history_push"], history_push)

    def test_memory_increment_validates_and_clamps_numeric_fields(self) -> None:
        self.assertEqual({"attempts": 3}, memory_increment({"attempts": 2}, "attempts"))
        self.assertEqual(
            {"attempts": 3},
            memory_increment({"attempts": 2}, "attempts", 5, maximum=3),
        )
        self.assertEqual(
            {"attempts": 0},
            memory_increment({"attempts": 2}, "attempts", -5, minimum=0),
        )
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            memory_increment({"attempts": "two"}, "attempts")
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            memory_increment({}, "attempts", float("inf"))
        with self.assertRaisesRegex(ValueError, "may not exceed"):
            memory_increment({}, "attempts", minimum=2, maximum=1)

    def test_memory_mapping_increment_updates_nested_counter_without_dotted_keys(
        self,
    ) -> None:
        original = {"stable_keys": {"UP": 1}}
        updated = memory_mapping_increment(original, "stable_keys", "UP")
        updated = memory_mapping_increment(updated, "stable_keys", "SPACE", 2)
        self.assertEqual(
            {"stable_keys": {"UP": 2, "SPACE": 2}},
            updated,
        )
        self.assertEqual({"stable_keys": {"UP": 1}}, original)
        self.assertEqual(
            {"stable_keys.UP": 1},
            memory_increment({}, "stable_keys.UP"),
        )
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            memory_mapping_increment({"stable_keys": 1}, "stable_keys", "UP")
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            memory_mapping_increment(
                {"stable_keys": {"UP": "one"}}, "stable_keys", "UP"
            )

    def test_palette_helpers_translate_reducer_hex_symbols_not_ascii(self) -> None:
        self.assertEqual("0123456789abcdef", POLICY_BOARD_HEX_SYMBOLS)
        self.assertEqual(5, palette_value("5"))
        self.assertEqual(11, palette_value("b"))
        self.assertEqual(15, palette_value("F"))
        self.assertEqual((11, 5, 15), palette_values("b5f5"))
        self.assertNotEqual(ord("b"), palette_value("b"))
        with self.assertRaisesRegex(ValueError, "hexadecimal"):
            palette_value("W")
        with self.assertRaisesRegex(ValueError, "between 1 and 16"):
            palette_values("")

    def test_stable_evidence_helper_matches_exact_action_and_objective_contract(
        self,
    ) -> None:
        objective = {
            "objective_id": "tactical:1",
            "minimum_evidence_actions": 4,
        }

        def transition(action: str, *, objective_id: str = "tactical:1") -> dict:
            return {
                "objective_id": objective_id,
                "action": action,
                "executed": True,
                "post_action_observed": True,
                "board_changed": True,
                "outcome_class": "novel",
                "loop_detected": False,
                "cycle_risk": False,
            }

        unreproduced = [transition(action) for action in ("UP", "RIGHT", "DOWN", "LEFT")]
        ready, reason = stable_transition_evidence_status(objective, unreproduced)
        self.assertFalse(ready)
        self.assertIn("not reproduced", reason)
        reproduced = [transition(action) for action in ("UP", "RIGHT", "UP", "RIGHT")]
        self.assertTrue(stable_transition_evidence_ready(objective, reproduced))
        reproduced[-1]["outcome_class"] = "volatile_only"
        ready, reason = stable_transition_evidence_status(objective, reproduced)
        self.assertFalse(ready)
        self.assertIn("volatile_only", reason)
        reproduced[-1] = transition("RIGHT", objective_id="tactical:other")
        ready, reason = stable_transition_evidence_status(objective, reproduced)
        self.assertFalse(ready)
        self.assertIn("only 3 of 4", reason)

    def test_contrastive_transition_requires_positive_and_negative_controls(self) -> None:
        objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 4,
        }

        def transition(action: str, *, changed: bool) -> dict[str, object]:
            return {
                "objective_id": "tactical:1",
                "action": action,
                "executed": True,
                "post_action_observed": True,
                "board_changed": changed,
                "outcome_class": "novel" if changed else "exact_noop",
                "cycle_risk": not changed,
                "loop_detected": not changed,
            }

        only_positive = [transition("LEFT", changed=True) for _ in range(4)]
        ready, reason = contrastive_transition_evidence_status(
            objective, only_positive
        )
        self.assertFalse(ready)
        self.assertIn("negative-control", reason)
        self.assertFalse(stable_transition_evidence_ready(objective, only_positive))

        unmatched_control = [
            transition("LEFT", changed=True),
            transition("MOUSE", changed=False),
            transition("LEFT", changed=True),
            transition("MOUSE", changed=False),
        ]
        ready, reason = contrastive_transition_evidence_status(
            objective, unmatched_control
        )
        self.assertFalse(ready)
        self.assertIn("same-family", reason)

        contrasted = [
            transition("LEFT", changed=True),
            transition("RIGHT", changed=False),
            transition("LEFT", changed=True),
            transition("RIGHT", changed=False),
        ]
        self.assertTrue(
            contrastive_transition_evidence_ready(objective, contrasted)
        )
        stable_objective = {**objective, "evidence_mode": "stable_transition"}
        self.assertFalse(
            contrastive_transition_evidence_ready(stable_objective, contrasted)
        )

    def test_recent_transition_summaries_are_bounded_and_host_classified(self) -> None:
        transitions = (
            {"action": "UP", "meaningful_progress": True},
            {"action": "LEFT", "outcome_class": "exact_noop"},
            {"action": "LEFT", "executed": True, "board_changed": False},
            {"action": "RIGHT", "cycle_risk": True},
            "ignored",
        )
        self.assertEqual(
            {"progress": 1, "no_progress": 2, "guarded": 1},
            recent_outcome_counts(transitions),
        )
        self.assertEqual(
            {"UP": 1, "LEFT": 2, "RIGHT": 1}, recent_action_counts(transitions)
        )
        self.assertEqual(
            {"LEFT": 2, "RIGHT": 1},
            recent_action_counts(transitions, only_nonprogress=True),
        )
        self.assertEqual(1, consecutive_outcome_count(transitions))
        self.assertEqual(0, consecutive_outcome_count(transitions, "no_progress"))
        with self.assertRaisesRegex(ValueError, "at most 64"):
            recent_outcome_counts(repeat({}, 65))

    def test_least_tried_action_is_deterministic_and_skips_unlocated_mouse(
        self,
    ) -> None:
        transitions = (
            {"action": "UP"},
            {"action": "UP"},
            {"action": "RIGHT"},
        )
        valid = ("UP", "RIGHT", "DOWN", "MOUSE")
        self.assertEqual("DOWN", least_tried_action(valid, transitions))
        self.assertEqual(
            "RIGHT", least_tried_action(valid, transitions, exclude=("DOWN",))
        )
        self.assertEqual(
            "MOUSE",
            least_tried_action(("MOUSE",), transitions, include_mouse=True),
        )
        self.assertIsNone(least_tried_action(("MOUSE",), transitions))
        with self.assertRaisesRegex(ValueError, "at most seven"):
            least_tried_action(("UP",) * 8, ())

    def test_mouse_point_counts_and_selection_include_coordinates(self) -> None:
        transitions = (
            {"action": "MOUSE", "row": 1, "col": 1, "outcome_class": "exact_noop"},
            {"action": "MOUSE", "row": 1, "col": 1, "meaningful_progress": True},
            {"action": "MOUSE", "row": 2, "col": 2, "outcome_class": "guarded"},
            {"action": "MOUSE", "row": 99, "col": 2},
            {"action": "UP", "row": 3, "col": 3},
        )
        self.assertEqual(
            {"1,1": 2, "2,2": 1}, recent_mouse_point_counts(transitions)
        )
        self.assertEqual(
            {"1,1": 1, "2,2": 1},
            recent_mouse_point_counts(transitions, only_nonprogress=True),
        )
        candidates = ((1, 1), (2, 2), (3, 3))
        self.assertEqual((3, 3), least_tried_mouse_point(candidates, transitions))
        self.assertEqual(
            (2, 2), least_tried_mouse_point(candidates, transitions, exclude=((3, 3),))
        )
        counts = recent_mouse_point_counts(transitions)
        self.assertEqual(
            (3, 3),
            least_tried_mouse_point(
                ((1, 1), (2, 2), (3, 3)), (), exclude=counts.keys()
            ),
        )
        self.assertEqual(
            (3, 3),
            least_tried_mouse_point(
                ((2, 2), (3, 3)),
                (),
                exclude=("2,2", [2, 2], (2, 2)),
            ),
        )
        self.assertIsNone(least_tried_mouse_point(((1, 1),), (), exclude=((1, 1),)))
        self.assertIsNone(least_tried_mouse_point(((0, 10), (63, 10)), ()))
        self.assertEqual(
            (0, 10),
            least_tried_mouse_point(
                ((0, 10), (63, 10)), (), allow_edge_hud=True
            ),
        )
        with self.assertRaisesRegex(ValueError, "allow_edge_hud"):
            least_tried_mouse_point(((2, 2),), (), allow_edge_hud=1)
        with self.assertRaisesRegex(ValueError, "point must be"):
            least_tried_mouse_point(("2,2",), ())
        for malformed in ("2", "2,3,4", "x,3", " 2,3", "02,3", b"2,3"):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(ValueError, "excluded mouse point"):
                    least_tried_mouse_point(((2, 2),), (), exclude=(malformed,))


if __name__ == "__main__":
    unittest.main()
