from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from inference.agent.gameplay_policy_runtime import (
    GameplayPolicyRuntime,
    PolicyObservation,
)
from inference.agent.objective_reduction import GameSolverType
from inference.agent.policy_solver_helpers import (
    POLICY_SOLVER_FAMILIES,
    POLICY_SOLVER_GLOBALS,
    POLICY_SOLVER_TYPES,
    solver_decide,
    solver_family,
    validate_solver_config,
)


EXPECTED_SOLVER_TYPES = {
    "beam",
    "beam-coverage",
    "carrier-placement",
    "cellular-automata",
    "click-interaction",
    "connector-align",
    "cycle-rotation",
    "flow-deflector",
    "glyph-transform-route",
    "gravity",
    "guided-attraction",
    "hybrid",
    "inertial-block",
    "inventory",
    "lattice-corridor",
    "linked-centroid",
    "marker-coverage",
    "mirror",
    "mirror-merge",
    "multi-agent",
    "navigation",
    "paired-platform-alignment",
    "paired-sequence-arm",
    "pattern-transform",
    "peg-jump",
    "puzzle",
    "push-pull",
    "relation-toggle",
    "signal",
    "sliding",
    "static",
    "switch-bridge",
    "symbol-rule-sequence",
    "template-paint",
    "trajectory-replay",
    "transform-program",
}


def observation(
    *, last_transition: dict | None = None, valid_actions: tuple[str, ...] = ()
) -> SimpleNamespace:
    board = np.zeros((64, 64), dtype=np.uint8)
    board[5, 5] = 1
    board[5, 9] = 2
    board[8, 8] = 3
    board[5, 2] = 4
    board.setflags(write=False)
    return SimpleNamespace(
        board=board,
        level=1,
        step=0,
        valid_actions=valid_actions
        or ("UP", "RIGHT", "DOWN", "LEFT", "SPACE", "MOUSE"),
        last_transition=last_transition,
        objective={},
        recent_transitions=(),
        backend="cpu",
    )


def config_for(solver_type: str) -> dict:
    family = solver_family(solver_type)
    common = {
        "actor_values": [1],
        "target_values": [2],
        "passable_values": [0, 1, 2, 4],
        "interactive_values": [3],
        "source_values": [4],
        "coverage_values": [],
        "interaction_actions": ["SPACE"],
        "approach_distance": 1,
    }
    if family == "sequence":
        common["action_sequences"] = [["RIGHT", "SPACE"], ["LEFT"]]
    if family == "observe":
        common["probe_actions"] = ["UP", "RIGHT"]
    if family == "hybrid":
        return {
            "fallback_types": ["static", "navigation"],
            "fallback_configs": {
                "static": {"probe_actions": ["SPACE"]},
                "navigation": {
                    "actor_values": [1],
                    "target_values": [2],
                    "passable_values": [0, 1, 2],
                    "approach_distance": 0,
                },
            },
        }
    return common


class PolicySolverHelperTests(unittest.TestCase):
    def test_catalog_matches_reference_solver_labels(self) -> None:
        self.assertEqual(EXPECTED_SOLVER_TYPES, set(POLICY_SOLVER_TYPES))
        self.assertEqual(EXPECTED_SOLVER_TYPES, {item.value for item in GameSolverType})
        self.assertEqual(EXPECTED_SOLVER_TYPES, set(POLICY_SOLVER_FAMILIES))
        self.assertTrue(all(solver_family(item) for item in POLICY_SOLVER_TYPES))
        self.assertIn("solver_decide", POLICY_SOLVER_GLOBALS)

    def test_every_solver_type_produces_a_bounded_json_decision(self) -> None:
        for solver_type in POLICY_SOLVER_TYPES:
            with self.subTest(solver_type=solver_type):
                current = observation()
                before = current.board.copy()
                result = solver_decide(solver_type, current, {}, config_for(solver_type))

                self.assertIn(
                    result["status"],
                    {"continue", "subgoal_succeeded", "subgoal_failed"},
                )
                json.dumps(result, allow_nan=False)
                np.testing.assert_array_equal(before, current.board)
                self.assertFalse(current.board.flags.writeable)

    def test_routing_replans_after_guarded_transition(self) -> None:
        cfg = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "approach_distance": 0,
        }
        result = solver_decide(
            "navigation",
            observation(last_transition={"outcome_class": "guarded"}),
            {"path": [[5, 5], [4, 5]], "stale": True},
            cfg,
        )

        self.assertEqual("continue", result["status"])
        self.assertEqual("RIGHT", result["action"]["action"])
        self.assertNotEqual([[5, 5], [4, 5]], result["memory"]["path"])

    def test_routing_avoids_failed_first_step_on_unchanged_board(self) -> None:
        cfg = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "approach_distance": 0,
        }
        first = solver_decide("navigation", observation(), {}, cfg)

        second = solver_decide(
            "navigation",
            observation(
                last_transition={
                    "action": first["action"]["action"],
                    "outcome_class": "exact_noop",
                }
            ),
            first["memory"],
            cfg,
        )

        self.assertEqual("continue", second["status"])
        self.assertNotEqual(first["action"], second["action"])
        self.assertTrue(second["memory"]["blocked_first_steps"])

    def test_routing_accumulates_distinct_failed_edges(self) -> None:
        cfg = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "approach_distance": 0,
        }
        result = solver_decide("navigation", observation(), {}, cfg)
        actions = [result["action"]["action"]]

        for _ in range(2):
            result = solver_decide(
                "navigation",
                observation(
                    last_transition={
                        "action": result["action"]["action"],
                        "outcome_class": "exact_noop",
                    }
                ),
                result["memory"],
                cfg,
            )
            actions.append(result["action"]["action"])

        self.assertEqual(3, len(set(actions)))
        self.assertEqual(2, len(result["memory"]["blocked_first_steps"]))

    def test_routing_clears_failed_edges_after_board_change(self) -> None:
        cfg = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "approach_distance": 0,
        }
        first = solver_decide("navigation", observation(), {}, cfg)
        failed = solver_decide(
            "navigation",
            observation(
                last_transition={
                    "action": first["action"]["action"],
                    "outcome_class": "exact_noop",
                }
            ),
            first["memory"],
            cfg,
        )
        changed_observation = observation(
            last_transition={
                "action": failed["action"]["action"],
                "outcome_class": "novel",
            }
        )
        changed_board = changed_observation.board.copy()
        changed_board[10, 10] = 8
        changed_board.setflags(write=False)
        changed_observation.board = changed_board

        recovered = solver_decide(
            "navigation", changed_observation, failed["memory"], cfg
        )

        self.assertEqual(first["action"], recovered["action"])
        self.assertEqual([], recovered["memory"]["blocked_first_steps"])

    def test_navigation_uses_probe_phase_for_contrastive_objective(self) -> None:
        config = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "probe_actions": ["LEFT", "RIGHT", "LEFT"],
        }
        first_observation = observation()
        first_observation.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 3,
        }
        first = solver_decide("navigation", first_observation, {}, config)
        second_observation = observation(
            last_transition={
                "objective_id": "tactical:1",
                "action": "LEFT",
                "executed": True,
                "post_action_observed": True,
                "board_changed": False,
                "outcome_class": "exact_noop",
            }
        )
        second_observation.objective = first_observation.objective
        second_observation.recent_transitions = (
            second_observation.last_transition,
        )
        second = solver_decide(
            "navigation", second_observation, first["memory"], config
        )

        self.assertEqual("LEFT", first["action"]["action"])
        self.assertEqual("RIGHT", second["action"]["action"])
        self.assertIn("transition evidence", second["evidence"])

    def test_navigation_completes_when_contrastive_evidence_is_ready(self) -> None:
        positive = {
            "objective_id": "tactical:1",
            "action": "RIGHT",
            "executed": True,
            "post_action_observed": True,
            "board_changed": True,
            "outcome_class": "novel",
            "cycle_risk": False,
            "loop_detected": False,
            "error": "",
        }
        negative = {
            "objective_id": "tactical:1",
            "action": "LEFT",
            "executed": True,
            "post_action_observed": True,
            "board_changed": False,
            "outcome_class": "exact_noop",
            "error": "",
        }
        current = observation(last_transition=negative)
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 3,
        }
        current.recent_transitions = (positive, positive, negative)

        result = solver_decide(
            "navigation", current, {}, config_for("navigation")
        )

        self.assertEqual("subgoal_succeeded", result["status"])
        self.assertIn("negative-control", result["evidence"])

    def test_engine_progress_navigation_stays_live_at_geometric_terminal(self) -> None:
        current = observation()
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "engine_progress",
            "minimum_evidence_actions": 4,
            "action_budget": 8,
            "actions_used": 0,
        }
        config = {
            **config_for("navigation"),
            "target_values": [1],
            "approach_distance": 0,
        }
        memory: dict = {}
        actions: list[str] = []
        recent: list[dict] = []

        for step in range(4):
            result = solver_decide("navigation", current, memory, config)
            self.assertEqual("continue", result["status"])
            actions.append(result["action"]["action"])
            memory = result["memory"]
            transition = {
                "objective_id": "tactical:1",
                "action": actions[-1],
                "executed": True,
                "post_action_observed": True,
                "board_changed": False,
                "meaningful_progress": False,
                "outcome_class": "exact_noop",
                "cycle_risk": False,
                "loop_detected": False,
                "error": "",
            }
            recent.append(transition)
            current.last_transition = transition
            current.recent_transitions = tuple(recent)
            current.step = step + 1

        self.assertEqual(4, len(set(actions)))

    def test_engine_progress_navigation_probes_when_route_is_unavailable(self) -> None:
        current = observation()
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "engine_progress",
            "minimum_evidence_actions": 4,
            "action_budget": 8,
            "actions_used": 0,
        }
        config = {
            **config_for("navigation"),
            "passable_values": [1, 2],
        }

        result = solver_decide("navigation", current, {}, config)

        self.assertEqual("continue", result["status"])
        self.assertIsNotNone(result["action"])
        self.assertIn("no traversable route", result["evidence"])

    def test_legacy_navigation_still_completes_at_geometric_target(self) -> None:
        current = observation()
        config = {
            **config_for("navigation"),
            "target_values": [1],
            "approach_distance": 0,
        }

        result = solver_decide("navigation", current, {}, config)

        self.assertEqual("subgoal_succeeded", result["status"])
        self.assertIsNone(result["action"])

    def test_routing_respects_actor_clearance_radius(self) -> None:
        narrow = np.full((64, 64), 9, dtype=np.uint8)
        narrow[5, 5:10] = 0
        narrow[5, 5] = 1
        narrow[5, 9] = 2
        narrow.setflags(write=False)
        current = observation()
        current.board = narrow
        config = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "approach_distance": 0,
        }

        point_actor = solver_decide("navigation", current, {}, config)
        blocked_wide_actor = solver_decide(
            "navigation", current, {}, {**config, "clearance_radius": 1}
        )

        self.assertEqual("continue", point_actor["status"])
        self.assertEqual("subgoal_failed", blocked_wide_actor["status"])

    def test_configuration_rejects_unknown_and_unbounded_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown.*key"):
            validate_solver_config("navigation", {"typo": [1]})
        with self.assertRaisesRegex(ValueError, "at most 16"):
            validate_solver_config("navigation", {"actor_values": list(range(17))})
        for invalid_color in (True, 1.5, "1"):
            with self.subTest(invalid_color=invalid_color):
                with self.assertRaisesRegex(ValueError, "JSON integers"):
                    validate_solver_config(
                        "navigation", {"actor_values": [invalid_color]}
                    )
        with self.assertRaisesRegex(ValueError, "unknown action"):
            validate_solver_config("static", {"probe_actions": ["ACTION99"]})
        with self.assertRaisesRegex(ValueError, "requires action_sequences"):
            validate_solver_config("peg-jump", {})
        with self.assertRaisesRegex(ValueError, "actor_values.*target_values"):
            validate_solver_config("navigation", {})
        with self.assertRaisesRegex(ValueError, "target_values"):
            validate_solver_config("marker-coverage", {})

    def test_interaction_avoids_edge_hud_and_exact_no_progress_repeat(self) -> None:
        edge_board = np.zeros((64, 64), dtype=np.uint8)
        edge_board[0, 0] = 3
        edge_board.setflags(write=False)
        edge_observation = observation(valid_actions=("MOUSE",))
        edge_observation.board = edge_board
        edge_result = solver_decide(
            "click-interaction",
            edge_observation,
            {},
            {"interactive_values": [3]},
        )
        repeated = solver_decide(
            "signal",
            observation(
                valid_actions=("SPACE",),
                last_transition={
                    "action": "SPACE",
                    "executed": True,
                    "board_changed": False,
                    "outcome_class": "exact_noop",
                },
            ),
            {},
            {"interaction_actions": ["SPACE"]},
        )

        self.assertEqual("subgoal_failed", edge_result["status"])
        self.assertEqual("subgoal_failed", repeated["status"])

    def test_interaction_uses_distinct_cells_within_one_connected_region(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[10:13, 20:23] = 3
        board.setflags(write=False)
        first_observation = observation(valid_actions=("MOUSE",))
        first_observation.board = board
        config = {
            "interactive_values": [3],
            "interaction_actions": ["MOUSE"],
        }

        first = solver_decide("click-interaction", first_observation, {}, config)
        first_action = first["action"]
        second_observation = observation(
            valid_actions=("MOUSE",),
            last_transition={
                **first_action,
                "executed": True,
                "board_changed": False,
                "outcome_class": "exact_noop",
            },
        )
        second_observation.board = board
        second = solver_decide(
            "click-interaction", second_observation, first["memory"], config
        )

        self.assertEqual("continue", first["status"])
        self.assertEqual("MOUSE", first_action["action"])
        self.assertEqual("continue", second["status"])
        self.assertEqual("MOUSE", second["action"]["action"])
        self.assertNotEqual(first_action, second["action"])

    def test_interaction_repeats_positive_coordinate_around_control(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[10, 10] = 3
        board[20, 20] = 3
        board.setflags(write=False)
        objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 3,
        }
        config = {"interactive_values": [3], "interaction_actions": ["MOUSE"]}
        current = observation(valid_actions=("MOUSE",))
        current.board = board
        current.objective = objective
        decisions = []
        memory = {}
        recent = []

        for _ in range(3):
            result = solver_decide("click-interaction", current, memory, config)
            decisions.append(result["action"])
            memory = result["memory"]
            transition = {
                **result["action"],
                "objective_id": "tactical:1",
                "executed": True,
                "post_action_observed": True,
                "board_changed": False,
                "outcome_class": "exact_noop",
            }
            recent.append(transition)
            current = observation(
                valid_actions=("MOUSE",), last_transition=transition
            )
            current.board = board
            current.objective = objective
            current.recent_transitions = tuple(recent)

        self.assertEqual(decisions[0], decisions[2])
        self.assertNotEqual(decisions[0], decisions[1])

    def test_interaction_completes_with_contrastive_coordinate_evidence(self) -> None:
        positive = {
            "objective_id": "tactical:1",
            "action": "MOUSE",
            "row": 10,
            "col": 10,
            "executed": True,
            "post_action_observed": True,
            "board_changed": True,
            "outcome_class": "novel",
            "cycle_risk": False,
            "loop_detected": False,
            "error": "",
        }
        negative = {
            "objective_id": "tactical:1",
            "action": "MOUSE",
            "row": 20,
            "col": 20,
            "executed": True,
            "post_action_observed": True,
            "board_changed": False,
            "outcome_class": "exact_noop",
            "error": "",
        }
        current = observation(valid_actions=("MOUSE",), last_transition=negative)
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 3,
        }
        current.recent_transitions = (positive, positive, negative)

        result = solver_decide(
            "click-interaction",
            current,
            {},
            {"interactive_values": [3], "interaction_actions": ["MOUSE"]},
        )

        self.assertEqual("subgoal_succeeded", result["status"])
        self.assertIn("negative-control", result["evidence"])

    def test_interaction_uses_modality_matched_scalar_evidence_schedule(self) -> None:
        objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 3,
        }
        config = {"interaction_actions": ["UP", "RIGHT", "DOWN"]}
        current = observation(valid_actions=("UP", "RIGHT", "DOWN"))
        current.objective = objective
        decisions = []
        memory = {}
        recent = []

        for _ in range(3):
            result = solver_decide("signal", current, memory, config)
            decisions.append(result["action"]["action"])
            memory = result["memory"]
            transition = {
                "objective_id": "tactical:1",
                "action": result["action"]["action"],
                "executed": True,
                "post_action_observed": True,
                "board_changed": False,
                "outcome_class": "exact_noop",
            }
            recent.append(transition)
            current = observation(
                valid_actions=("UP", "RIGHT", "DOWN"),
                last_transition=transition,
            )
            current.objective = objective
            current.recent_transitions = tuple(recent)

        self.assertEqual(["UP", "RIGHT", "UP"], decisions)

    def test_interaction_preserves_explicit_mixed_probe_order(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[10, 10] = 3
        board[20, 20] = 3
        board.setflags(write=False)
        objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 4,
        }
        config = {
            "interactive_values": [3],
            "interaction_actions": ["MOUSE", "SPACE"],
            "probe_actions": ["MOUSE", "SPACE", "MOUSE", "SPACE"],
        }
        current = observation(valid_actions=("MOUSE", "SPACE"))
        current.board = board
        current.objective = objective
        memory = {}
        decisions = []
        recent = []

        for _ in range(4):
            result = solver_decide("click-interaction", current, memory, config)
            decisions.append(result["action"])
            memory = result["memory"]
            transition = {
                "objective_id": "tactical:1",
                **result["action"],
                "executed": True,
                "post_action_observed": True,
                "board_changed": False,
                "outcome_class": "exact_noop",
            }
            recent.append(transition)
            current = observation(
                valid_actions=("MOUSE", "SPACE"),
                last_transition=transition,
            )
            current.board = board
            current.objective = objective
            current.recent_transitions = tuple(recent)

        self.assertEqual(
            ["MOUSE", "SPACE", "MOUSE", "SPACE"],
            [decision["action"] for decision in decisions],
        )
        self.assertNotEqual(decisions[0], decisions[2])

    def test_interaction_does_not_inject_mouse_into_explicit_scalar_probes(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[10, 10] = 3
        board[20, 20] = 3
        board.setflags(write=False)
        objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 2,
        }
        config = {
            "interactive_values": [3],
            "interaction_actions": ["SPACE", "UP"],
            "probe_actions": ["SPACE", "UP"],
        }
        current = observation(valid_actions=("MOUSE", "SPACE", "UP"))
        current.board = board
        current.objective = objective
        first = solver_decide("click-interaction", current, {}, config)
        transition = {
            "objective_id": "tactical:1",
            **first["action"],
            "executed": True,
            "post_action_observed": True,
            "board_changed": False,
            "outcome_class": "exact_noop",
        }
        second_observation = observation(
            valid_actions=("MOUSE", "SPACE", "UP"),
            last_transition=transition,
        )
        second_observation.board = board
        second_observation.objective = objective
        second_observation.recent_transitions = (transition,)
        second = solver_decide(
            "click-interaction", second_observation, first["memory"], config
        )

        self.assertEqual("SPACE", first["action"]["action"])
        self.assertEqual("UP", second["action"]["action"])

    def test_contrastive_scalar_evidence_requires_two_directional_actions(self) -> None:
        current = observation(valid_actions=("SPACE", "UP"))
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 3,
        }

        result = solver_decide(
            "signal",
            current,
            {},
            {"interaction_actions": ["SPACE", "UP"]},
        )

        self.assertEqual("subgoal_failed", result["status"])
        self.assertIsNone(result["action"])

    def test_interaction_never_emits_coordinate_less_mouse_fallback(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[8, 8] = 3
        board.setflags(write=False)
        first_observation = observation(valid_actions=("MOUSE",))
        first_observation.board = board
        config = {
            "interactive_values": [3],
            "interaction_actions": ["MOUSE"],
            "probe_actions": ["MOUSE"],
        }

        first = solver_decide("click-interaction", first_observation, {}, config)
        exhausted_observation = observation(
            valid_actions=("MOUSE",),
            last_transition={
                **first["action"],
                "executed": True,
                "board_changed": False,
                "outcome_class": "exact_noop",
            },
        )
        exhausted_observation.board = board
        exhausted = solver_decide(
            "click-interaction", exhausted_observation, first["memory"], config
        )

        self.assertEqual({"action": "MOUSE", "row": 8, "col": 8}, first["action"])
        self.assertEqual("subgoal_failed", exhausted["status"])
        self.assertIsNone(exhausted["action"])
        self.assertIn("no untried coordinate", exhausted["evidence"])

    def test_physics_normalizes_stride_and_rejects_blocked_gravity_repeat(self) -> None:
        physics_board = np.zeros((64, 64), dtype=np.uint8)
        physics_board[5, 5] = 1
        physics_board[5, 7] = 9
        physics_board[5, 9] = 2
        physics_board.setflags(write=False)
        current = observation()
        current.board = physics_board
        cfg = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
        }
        momentum = solver_decide(
            "sliding", current, {"actor": [5, 3]}, cfg
        )
        gravity = solver_decide(
            "gravity",
            observation(
                last_transition={
                    "action": "RIGHT",
                    "executed": True,
                    "board_changed": False,
                    "outcome_class": "exact_noop",
                }
            ),
            {},
            cfg,
        )

        self.assertEqual("LEFT", momentum["action"]["action"])
        self.assertEqual("subgoal_failed", gravity["status"])

    def test_gravity_mouse_interaction_is_coordinate_bearing_and_bounded(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[5, 5] = 1
        board[9, 5] = 2
        board.setflags(write=False)
        first_observation = observation(valid_actions=("MOUSE",))
        first_observation.board = board
        config = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "interaction_actions": ["MOUSE"],
        }
        first = solver_decide("gravity", first_observation, {}, config)
        repeated_observation = observation(
            valid_actions=("MOUSE",),
            last_transition={
                **first["action"],
                "outcome_class": "exact_noop",
            },
        )
        repeated_observation.board = board
        repeated = solver_decide(
            "gravity", repeated_observation, first["memory"], config
        )

        self.assertEqual(
            {"action": "MOUSE", "row": 9, "col": 5}, first["action"]
        )
        self.assertEqual("subgoal_failed", repeated["status"])
        self.assertIn("no alternate", repeated["evidence"])

    def test_clear_field_requires_coordinate_safe_interaction(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[5, 2] = 4
        board[5, 9] = 2
        board.setflags(write=False)
        current = observation(valid_actions=("MOUSE",))
        current.board = board
        config = {
            "source_values": [4],
            "target_values": [2],
            "passable_values": [0, 2, 4],
            "interaction_actions": ["MOUSE"],
        }

        result = solver_decide("beam", current, {}, config)
        unavailable_observation = observation(valid_actions=("SPACE",))
        unavailable_observation.board = board
        unavailable = solver_decide("beam", unavailable_observation, {}, config)
        evidence_observation = observation(valid_actions=("SPACE",))
        evidence_observation.board = board
        evidence_observation.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "stable_transition",
            "minimum_evidence_actions": 2,
        }
        premature = solver_decide(
            "beam",
            evidence_observation,
            {},
            {
                "source_values": [4],
                "target_values": [2],
                "passable_values": [0, 2, 4],
            },
        )

        self.assertEqual(
            {"action": "MOUSE", "row": 5, "col": 9}, result["action"]
        )
        self.assertEqual("subgoal_failed", unavailable["status"])
        self.assertIn("no alternate", unavailable["evidence"])
        self.assertEqual("subgoal_failed", premature["status"])
        self.assertIn("without required transition evidence", premature["evidence"])

    def test_transform_field_and_coverage_use_safe_family_specific_targets(self) -> None:
        transform = solver_decide(
            "transform-program",
            observation(valid_actions=("MOUSE",)),
            {},
            {"target_values": [2]},
        )

        field_board = np.zeros((64, 64), dtype=np.uint8)
        field_board[5, 2] = 4
        field_board[5, 5] = 9
        field_board[5, 9] = 2
        field_board[8, 8] = 3
        field_board.setflags(write=False)
        field_observation = observation(valid_actions=("MOUSE",))
        field_observation.board = field_board
        field = solver_decide(
            "flow-deflector",
            field_observation,
            {},
            {
                "source_values": [4],
                "target_values": [2],
                "passable_values": [0, 2, 4],
                "interactive_values": [3],
            },
        )

        edge_board = np.zeros((64, 64), dtype=np.uint8)
        edge_board[0, 0] = 2
        edge_board.setflags(write=False)
        coverage_observation = observation(valid_actions=("MOUSE",))
        coverage_observation.board = edge_board
        coverage = solver_decide(
            "marker-coverage",
            coverage_observation,
            {},
            {"target_values": [2]},
        )

        self.assertEqual({"action": "MOUSE", "row": 5, "col": 9}, transform["action"])
        self.assertEqual({"action": "MOUSE", "row": 8, "col": 8}, field["action"])
        self.assertEqual("subgoal_failed", coverage["status"])

    def test_direct_click_families_do_not_repeat_noop_coordinates(self) -> None:
        board = np.zeros((64, 64), dtype=np.uint8)
        board[10, 10] = 2
        board[20, 20] = 2
        board.setflags(write=False)

        for solver_type in ("marker-coverage", "transform-program"):
            with self.subTest(solver_type=solver_type):
                first_observation = observation(valid_actions=("MOUSE",))
                first_observation.board = board
                first = solver_decide(
                    solver_type,
                    first_observation,
                    {},
                    {"target_values": [2]},
                )
                second_observation = observation(
                    valid_actions=("MOUSE",),
                    last_transition={
                        **first["action"],
                        "executed": True,
                        "board_changed": False,
                        "outcome_class": "exact_noop",
                    },
                )
                second_observation.board = board
                second = solver_decide(
                    solver_type,
                    second_observation,
                    first["memory"],
                    {"target_values": [2]},
                )

                self.assertEqual("continue", second["status"])
                self.assertNotEqual(first["action"], second["action"])

    def test_coordinate_attempts_reset_when_candidate_set_changes(self) -> None:
        def target_observation(
            point: tuple[int, int], last_action: dict | None = None
        ) -> SimpleNamespace:
            board = np.zeros((64, 64), dtype=np.uint8)
            board[point] = 2
            board.setflags(write=False)
            current = observation(
                valid_actions=("MOUSE",),
                last_transition=(
                    {**last_action, "outcome_class": "exact_noop"}
                    if last_action is not None
                    else None
                ),
            )
            current.board = board
            return current

        first = solver_decide(
            "transform-program",
            target_observation((10, 10)),
            {},
            {"target_values": [2]},
        )
        second = solver_decide(
            "transform-program",
            target_observation((20, 20), first["action"]),
            first["memory"],
            {"target_values": [2]},
        )
        returned = solver_decide(
            "transform-program",
            target_observation((10, 10), second["action"]),
            second["memory"],
            {"target_values": [2]},
        )

        self.assertEqual(first["action"], returned["action"])

    def test_approach_interaction_uses_alternate_action_after_noop(self) -> None:
        config = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "interaction_actions": ["SPACE", "RIGHT"],
            "approach_distance": 4,
        }
        first = solver_decide("push-pull", observation(), {}, config)
        second = solver_decide(
            "push-pull",
            observation(
                last_transition={
                    "action": first["action"]["action"],
                    "outcome_class": "exact_noop",
                }
            ),
            first["memory"],
            config,
        )

        self.assertEqual("continue", first["status"])
        self.assertEqual("continue", second["status"])
        self.assertNotEqual(first["action"], second["action"])

    def test_approach_interaction_mouse_is_coordinate_bearing(self) -> None:
        result = solver_decide(
            "push-pull",
            observation(valid_actions=("MOUSE",)),
            {},
            {
                "actor_values": [1],
                "target_values": [2],
                "passable_values": [0, 1, 2],
                "interaction_actions": ["MOUSE"],
                "approach_distance": 4,
            },
        )

        self.assertEqual(
            {"action": "MOUSE", "row": 5, "col": 9}, result["action"]
        )

    def test_hybrid_rejects_recursion_and_unlisted_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "recursively"):
            validate_solver_config(
                "hybrid",
                {"fallback_types": ["hybrid"], "fallback_configs": {}},
            )
        with self.assertRaisesRegex(ValueError, "must appear"):
            validate_solver_config(
                "hybrid",
                {
                    "fallback_types": ["static"],
                    "fallback_configs": {
                        "static": {"probe_actions": ["SPACE"]},
                        "navigation": {},
                    },
                },
            )

    def test_sequence_exhaustion_is_visible(self) -> None:
        cfg = {"action_sequences": [["SPACE"]]}
        first = solver_decide("symbol-rule-sequence", observation(), {}, cfg)
        exhausted = solver_decide(
            "symbol-rule-sequence", observation(), first["memory"], cfg
        )

        self.assertEqual("continue", first["status"])
        self.assertEqual("subgoal_failed", exhausted["status"])

    def test_observation_solver_skips_repeated_noop_probe(self) -> None:
        config = {"probe_actions": ["UP", "UP", "RIGHT"]}
        first = solver_decide("static", observation(), {}, config)
        second = solver_decide(
            "static",
            observation(
                last_transition={
                    "action": first["action"]["action"],
                    "outcome_class": "exact_noop",
                }
            ),
            first["memory"],
            config,
        )

        self.assertEqual("UP", first["action"]["action"])
        self.assertEqual("RIGHT", second["action"]["action"])

    def test_observation_solver_honors_contrastive_objective_evidence(self) -> None:
        positive = {
            "objective_id": "tactical:1",
            "action": "UP",
            "executed": True,
            "post_action_observed": True,
            "board_changed": True,
            "outcome_class": "novel",
            "cycle_risk": False,
            "loop_detected": False,
            "error": "",
        }
        negative = {
            "objective_id": "tactical:1",
            "action": "LEFT",
            "executed": True,
            "post_action_observed": True,
            "board_changed": False,
            "outcome_class": "exact_noop",
            "error": "",
        }
        current = observation(last_transition=negative)
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "contrastive_transition",
            "minimum_evidence_actions": 3,
        }
        current.recent_transitions = (positive, positive, negative)

        result = solver_decide(
            "static", current, {}, {"probe_actions": ["DOWN"]}
        )

        self.assertEqual("subgoal_succeeded", result["status"])
        self.assertIn("negative-control", result["evidence"])

    def test_observation_solver_waits_for_complete_objective_evidence(self) -> None:
        stable = {
            "objective_id": "tactical:1",
            "action": "UP",
            "executed": True,
            "post_action_observed": True,
            "board_changed": True,
            "outcome_class": "novel",
            "cycle_risk": False,
            "loop_detected": False,
            "error": "",
        }
        current = observation(last_transition=stable)
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "stable_transition",
            "minimum_evidence_actions": 2,
        }
        current.recent_transitions = (stable,)

        incomplete = solver_decide(
            "static", current, {}, {"probe_actions": ["RIGHT"]}
        )
        current.recent_transitions = (stable, stable)
        complete = solver_decide(
            "static", current, incomplete["memory"], {"probe_actions": ["RIGHT"]}
        )

        self.assertEqual("continue", incomplete["status"])
        self.assertEqual("subgoal_succeeded", complete["status"])
        self.assertIn("reproducible", complete["evidence"])

    def test_solver_families_recover_from_malformed_json_memory(self) -> None:
        routing_seed = solver_decide(
            "navigation", observation(), {}, config_for("navigation")
        )
        cases = (
            (
                "navigation",
                {
                    **routing_seed["memory"],
                    "blocked_first_steps": [None, [True, 2], ["5", 6], [5]],
                },
                config_for("navigation"),
            ),
            ("sliding", {"actor": ["row", {}]}, config_for("sliding")),
            (
                "push-pull",
                {"action_counts": ["not", "a", "mapping"]},
                {**config_for("push-pull"), "approach_distance": 4},
            ),
            (
                "symbol-rule-sequence",
                {"sequence_index": "bad", "sequence_offset": []},
                config_for("symbol-rule-sequence"),
            ),
            ("static", {"probe_index": {"bad": 1}}, config_for("static")),
            (
                "hybrid",
                {"fallback_memory": ["not", "a", "mapping"]},
                config_for("hybrid"),
            ),
        )

        for solver_type, memory, config in cases:
            with self.subTest(solver_type=solver_type):
                result = solver_decide(solver_type, observation(), memory, config)
                self.assertIn(
                    result["status"],
                    {"continue", "subgoal_succeeded", "subgoal_failed"},
                )
                json.dumps(result, allow_nan=False)

    def test_dispatch_stops_before_action_after_objective_budget_exhaustion(self) -> None:
        current = observation()
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "engine_progress",
            "minimum_evidence_actions": 1,
            "action_budget": 3,
            "actions_used": 3,
        }

        result = solver_decide(
            "navigation", current, {}, config_for("navigation")
        )

        self.assertEqual("subgoal_failed", result["status"])
        self.assertIsNone(result["action"])
        self.assertIn("(3/3)", result["evidence"])
        self.assertIn("before required evidence", result["evidence"])

    def test_dispatch_accepts_ready_evidence_at_budget_boundary(self) -> None:
        transition = {
            "objective_id": "tactical:1",
            "action": "RIGHT",
            "executed": True,
            "post_action_observed": True,
            "board_changed": True,
            "meaningful_progress": True,
            "outcome_class": "progress",
            "cycle_risk": False,
            "loop_detected": False,
            "error": "",
        }
        current = observation(last_transition=transition)
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "engine_progress",
            "minimum_evidence_actions": 1,
            "action_budget": 3,
            "actions_used": 3,
        }
        current.recent_transitions = (transition,)

        result = solver_decide(
            "navigation", current, {}, config_for("navigation")
        )

        self.assertEqual("subgoal_succeeded", result["status"])
        self.assertIsNone(result["action"])

    def test_dispatch_ignores_malformed_objective_budget_values(self) -> None:
        current = observation()
        current.objective = {
            "objective_id": "tactical:1",
            "evidence_mode": "engine_progress",
            "minimum_evidence_actions": 1,
            "action_budget": "3",
            "actions_used": True,
        }

        result = solver_decide(
            "navigation", current, {}, config_for("navigation")
        )

        self.assertEqual("continue", result["status"])
        self.assertIsNotNone(result["action"])

    def test_solver_dispatch_is_available_inside_policy_worker(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
POLICY_SOLVER_TYPE = "navigation"
POLICY_SOLVER_CONFIG = {
    "actor_values": [1],
    "target_values": [2],
    "passable_values": [0, 1, 2],
    "approach_distance": 0,
}
def decide(observation, memory):
    return solver_decide(
        POLICY_SOLVER_TYPE, observation, memory, POLICY_SOLVER_CONFIG
    )
"""
        runtime = GameplayPolicyRuntime()
        try:
            activation = runtime.activate(source, context={})
            raw = observation(valid_actions=("UP", "RIGHT", "DOWN", "LEFT"))
            decision = runtime.decide(
                PolicyObservation(
                    board=raw.board,
                    level=1,
                    step=0,
                    valid_actions=raw.valid_actions,
                    last_transition=None,
                    objective={"solver_type": "navigation"},
                    recent_transitions=(),
                    backend=activation.backend,
                )
            )
        finally:
            runtime.close()

        self.assertEqual("continue", decision.status.value)
        self.assertEqual("RIGHT", decision.action["action"])


if __name__ == "__main__":
    unittest.main()
