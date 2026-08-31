from __future__ import annotations

import unittest
from itertools import repeat

import numpy as np

from inference.agent.policy_codegen_helpers import (
    POLICY_ACTIONS,
    POLICY_CODEGEN_API_VERSION,
    POLICY_CODEGEN_GLOBALS,
    accumulate_transition_evidence,
    action_payload,
    board_digest,
    cells_digest,
    consecutive_outcome_count,
    continue_decision,
    edge_run_length,
    edge_value_count,
    first_matching_cell,
    history_push,
    least_tried_action,
    least_tried_mouse_point,
    line_run_length,
    line_value_count,
    matching_region_center,
    memory_increment,
    memory_push,
    memory_update,
    memory_with_defaults,
    mouse_decision,
    nearest_matching_cell,
    objective_evidence_ready,
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
)


class PolicyCodegenHelperTests(unittest.TestCase):
    def test_action_payload_canonicalizes_names_and_mappings(self) -> None:
        self.assertEqual({"action": "UP"}, action_payload(" up "))
        self.assertEqual({"action": "ACTION7"}, action_payload({"action": "action7"}))
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
            "edge_run_length",
            "first_matching_cell",
            "least_tried_mouse_point",
            "line_value_count",
            "memory_with_defaults",
            "mouse_decision",
            "nearest_matching_cell",
            "objective_evidence_ready",
            "path_decision",
            "region_digest",
            "subgoal_failed",
            "subgoal_succeeded",
            "transition_facts",
            "transition_change_class",
            "transition_has_stable_change",
            "transition_outcome",
            "transition_requires_replan",
        }
        self.assertEqual(1, POLICY_CODEGEN_API_VERSION)
        self.assertEqual(1, POLICY_CODEGEN_GLOBALS["POLICY_CODEGEN_API_VERSION"])
        self.assertEqual(POLICY_ACTIONS, POLICY_CODEGEN_GLOBALS["POLICY_ACTIONS"])
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
