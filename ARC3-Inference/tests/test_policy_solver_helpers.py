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
