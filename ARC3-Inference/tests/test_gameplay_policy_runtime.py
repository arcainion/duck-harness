from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from inference.agent.gameplay_policy_runtime import (
    GameplayPolicyRuntime,
    PolicyActivation,
    PolicyDecision,
    PolicyObservation,
    PolicyRuntimeError,
    PolicyStatus,
    choose_policy_backend,
    verify_policy_source,
)


GOOD_POLICY = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)

def initialize(context):
    return {"calls": 0}

def decide(observation, memory):
    memory = {"calls": int(memory.get("calls", 0)) + 1}
    return {
        "status": "continue",
        "action": {"action": observation.valid_actions[0]},
        "memory": memory,
        "evidence": "deterministic smoke decision",
    }
"""


def observation(
    *,
    valid_actions: tuple[str, ...] = ("ACTION1",),
    board: np.ndarray | None = None,
) -> PolicyObservation:
    return PolicyObservation(
        board=(np.zeros((64, 64), dtype=np.uint8) if board is None else board),
        level=1,
        step=0,
        valid_actions=valid_actions,
        last_transition=None,
        objective={"objective_id": "tactical:1"},
        recent_transitions=(),
        backend="cpu",
    )


class _FakeCuda:
    def __init__(self, *, available: bool, free_mb: int) -> None:
        self._available = available
        self._free_mb = free_mb

    def is_available(self) -> bool:
        return self._available

    def mem_get_info(self) -> tuple[int, int]:
        return self._free_mb * 1024 * 1024, 24 * 1024 * 1024 * 1024


class _FakeTorch:
    def __init__(self, *, available: bool, free_mb: int) -> None:
        self.cuda = _FakeCuda(available=available, free_mb=free_mb)


class _BrokenCuda:
    def is_available(self) -> bool:
        return True

    def mem_get_info(self) -> tuple[int, int]:
        raise RuntimeError("driver query failed")


class _BrokenTorch:
    cuda = _BrokenCuda()


class GameplayPolicyRuntimeTests(unittest.TestCase):
    def test_observation_copies_and_freezes_uint8_board(self) -> None:
        board = np.zeros((64, 64), dtype=np.int64)
        value = observation()
        value = PolicyObservation(
            board=board,
            level=value.level,
            step=value.step,
            valid_actions=value.valid_actions,
            last_transition=value.last_transition,
            objective=value.objective,
            recent_transitions=value.recent_transitions,
            backend=value.backend,
        )
        board[0, 0] = 9
        self.assertEqual(np.uint8, value.board.dtype)
        self.assertEqual(0, value.board[0, 0])
        self.assertFalse(value.board.flags.writeable)
        with self.assertRaises(ValueError):
            value.board[0, 0] = 1

    def test_observation_rejects_noncanonical_board_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, r"shape \(64, 64\)"):
            PolicyObservation(
                board=np.zeros((63, 64), dtype=np.uint8),
                level=1,
                step=0,
                valid_actions=("ACTION1",),
                last_transition=None,
                objective={},
                recent_transitions=(),
                backend="cpu",
            )

    def test_cpu_policy_persists_json_memory(self) -> None:
        with GameplayPolicyRuntime(
            requested_backend="cpu", decision_timeout_seconds=2
        ) as runtime:
            activation = runtime.activate(GOOD_POLICY, context={"game_id": "g"})
            self.assertEqual("cpu", activation.backend)
            decision = runtime.decide(observation())
            self.assertEqual(PolicyStatus.CONTINUE, decision.status)
            self.assertEqual({"action": "ACTION1"}, decision.action)
        self.assertEqual({"calls": 1}, runtime.memory)

    def test_preflight_requires_distinct_persistent_noop_probes(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    action = least_tried_action(
        observation.valid_actions, observation.recent_transitions
    )
    return continue_decision(action, memory, "bounded distinct probe")
"""
        initial = observation(valid_actions=("UP", "RIGHT", "DOWN", "LEFT"))
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            runtime.preflight(initial, minimum_actions=4)
            decision = runtime.decide(initial)
        self.assertEqual({"action": "UP"}, decision.action)
        self.assertEqual({}, decision.memory)

    def test_preflight_rejects_early_terminal_and_repeated_noop_action(self) -> None:
        early = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    if observation.last_transition is not None:
        return subgoal_failed(memory, "one noop is enough")
    return continue_decision("UP", memory)
"""
        repeated = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    return continue_decision("UP", memory)
"""
        initial = observation(valid_actions=("UP", "RIGHT", "DOWN", "LEFT"))
        for source, message in (
            (early, "terminated after only 1"),
            (repeated, "repeated the same action"),
        ):
            with (
                self.subTest(message=message),
                GameplayPolicyRuntime(requested_backend="cpu") as runtime,
            ):
                runtime.activate(source, context={})
                with self.assertRaisesRegex(PolicyRuntimeError, message) as raised:
                    runtime.preflight(initial, minimum_actions=4)
                self.assertEqual("policy_preflight", raised.exception.category)

    def test_verifier_reports_concrete_nonnull_board_rewrite(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    board = observation.board
    if board is None or not hasattr(board, "shape"):
        return subgoal_failed(memory, "missing board")
    return continue_decision(observation.valid_actions[0], memory)
"""
        with self.assertRaises(PolicyRuntimeError) as raised:
            verify_policy_source(source)
        detail = str(raised.exception)
        self.assertIn("always a non-None uint8[64,64]", detail)
        self.assertIn("call 'hasattr' is not permitted", detail)

    def test_policy_can_use_safe_ord_and_standalone_history_push(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    history = history_push(memory.get("values", []), ord("b"), limit=2)
    return continue_decision(
        observation.valid_actions[0], {"values": history}, "recorded cell value"
    )
"""
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            decision = runtime.decide(observation(valid_actions=("UP",)))
        self.assertEqual({"values": [98]}, decision.memory)

    def test_policy_can_use_host_pathfinding_helpers_without_imports(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    starts = find_cells(observation.board, 1)
    goals = find_cells(observation.board, 2)
    passable = np.isin(observation.board, (0, 1, 2))
    path = shortest_path(passable, starts[0], goals[0])
    distances = distance_map(passable, starts)
    action_name = next_path_action(path, observation.valid_actions)
    return {
        "status": "continue",
        "action": {"action": action_name},
        "memory": {
            "path_length": len(path),
            "goal_distance": int(distances[goals[0]]),
        },
        "evidence": "host BFS selected a valid cardinal step",
    }
"""
        board = np.zeros((64, 64), dtype=np.uint8)
        board[1, 1] = 1
        board[1, 3] = 2
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            decision = runtime.decide(
                observation(valid_actions=("RIGHT",), board=board)
            )
        self.assertEqual({"action": "RIGHT"}, decision.action)
        self.assertEqual({"path_length": 3, "goal_distance": 2}, decision.memory)

    def test_policy_can_use_weighted_clearance_and_projection_helpers(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    passable = np.ones((64, 64), dtype=bool)
    costs = np.ones((64, 64), dtype=float)
    costs[2, 2] = 20.0
    safe = clearance_mask(passable, 1)
    path = weighted_shortest_path(safe, costs, (2, 1), (2, 3))
    action_name = next_path_action(path, observation.valid_actions)
    projected = action_destination((2, 1), action_name)
    visible = line_of_sight(passable, (2, 1), (2, 3))
    return {
        "status": "continue",
        "action": {"action": action_name},
        "memory": {
            "projected": list(projected),
            "visible": bool(visible),
            "cost": float(path_cost(costs, path)),
        },
    }
"""
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            decision = runtime.decide(observation(valid_actions=("UP", "RIGHT")))
        self.assertEqual({"action": "UP"}, decision.action)
        self.assertEqual(
            {"projected": [1, 1], "visible": True, "cost": 4.0},
            decision.memory,
        )

    def test_policy_can_use_component_approach_and_route_reuse_helpers(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    targets = value_mask(observation.board, 2)
    passable = value_mask(observation.board, (0, 1))
    target_points = find_cells(observation.board, 2)
    path = shortest_approach_path(passable, (2, 1), target_points, 1)
    suffix = path_suffix(path, (2, 1))
    action_name = next_path_action(suffix, observation.valid_actions)
    return {
        "status": "continue",
        "action": {"action": action_name},
        "memory": {
            "api": PATHFINDING_API_VERSION,
            "box": list(component_boxes(targets)[0]),
            "center": list(component_centers(targets)[0]),
            "route_valid": bool(path_is_valid(passable, suffix)),
        },
    }
"""
        board = np.zeros((64, 64), dtype=np.uint8)
        board[2, 1] = 1
        board[2, 3:5] = 2
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            decision = runtime.decide(
                observation(valid_actions=("RIGHT",), board=board)
            )
        self.assertEqual({"action": "RIGHT"}, decision.action)
        self.assertEqual(
            {
                "api": 1,
                "box": [2, 3, 2, 4, 2],
                "center": [2, 3],
                "route_valid": True,
            },
            decision.memory,
        )

    def test_policy_can_use_decision_and_transition_contract_helpers(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    outcome = transition_outcome(observation.last_transition)
    if transition_repeats_nonprogress_action(observation.last_transition, "LEFT"):
        return subgoal_failed({"outcome": outcome}, "do not repeat blocked action")
    path = ((1, 1), (1, 2))
    return path_decision(
        path,
        observation.valid_actions,
        {"api": POLICY_CODEGEN_API_VERSION, "outcome": outcome},
        "follow safe route",
    )
"""
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            first = runtime.decide(observation(valid_actions=("RIGHT",)))
            blocked_observation = observation(valid_actions=("RIGHT",))
            blocked_observation = PolicyObservation(
                board=blocked_observation.board,
                level=1,
                step=1,
                valid_actions=("RIGHT",),
                last_transition={
                    "action": "LEFT",
                    "executed": True,
                    "board_changed": False,
                },
                objective=blocked_observation.objective,
                recent_transitions=(),
                backend="cpu",
            )
            second = runtime.decide(blocked_observation)
        self.assertEqual(PolicyStatus.CONTINUE, first.status)
        self.assertEqual({"action": "RIGHT"}, first.action)
        self.assertEqual({"api": 1, "outcome": "unknown"}, first.memory)
        self.assertEqual(PolicyStatus.SUBGOAL_FAILED, second.status)
        self.assertIsNone(second.action)
        self.assertEqual({"outcome": "no_progress"}, second.memory)

    def test_policy_can_use_digest_bounded_memory_and_exploration_helpers(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    digest = board_digest(observation.board)
    memory = memory_push(memory, "digests", digest, limit=2)
    memory = memory_increment(memory, "calls", maximum=10)
    memory = memory_update(
        memory,
        {"outcomes": recent_outcome_counts(observation.recent_transitions)},
    )
    action_name = least_tried_action(
        observation.valid_actions, observation.recent_transitions
    )
    return continue_decision(action_name, memory, "bounded exploration")
"""
        first_observation = observation(valid_actions=("UP", "RIGHT"))
        second_observation = PolicyObservation(
            board=first_observation.board,
            level=1,
            step=1,
            valid_actions=("UP", "RIGHT"),
            last_transition={
                "action": "UP",
                "executed": True,
                "board_changed": False,
            },
            objective=first_observation.objective,
            recent_transitions=(
                {
                    "action": "UP",
                    "executed": True,
                    "board_changed": False,
                },
            ),
            backend="cpu",
        )
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            first = runtime.decide(first_observation)
            second = runtime.decide(second_observation)
        self.assertEqual({"action": "UP"}, first.action)
        self.assertEqual({"action": "RIGHT"}, second.action)
        self.assertEqual(2, second.memory["calls"])
        self.assertEqual({"no_progress": 1}, second.memory["outcomes"])
        self.assertEqual(2, len(second.memory["digests"]))
        self.assertEqual(second.memory["digests"][0], second.memory["digests"][1])

    def test_invalid_policy_action_is_rejected(self) -> None:
        source = GOOD_POLICY.replace("observation.valid_actions[0]", '"ACTION4"')
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            with self.assertRaises(PolicyRuntimeError) as captured:
                runtime.decide(observation())
        self.assertEqual("invalid_action", captured.exception.category)

    def test_policydecision_accepts_string_status_from_generated_code(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    return PolicyDecision(
        status="continue",
        action={"action": observation.valid_actions[0]},
        memory=memory,
    )
"""
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            decision = runtime.decide(observation())
        self.assertEqual(PolicyStatus.CONTINUE, decision.status)
        self.assertEqual({"action": "ACTION1"}, decision.action)

    def test_policy_decision_enforces_status_action_contract(self) -> None:
        cases = (
            ({"status": "unknown", "memory": {}}, "invalid status"),
            ({"status": "continue", "memory": {}}, "requires exactly one"),
            (
                {
                    "status": "subgoal_succeeded",
                    "action": {"action": "ACTION1"},
                    "memory": {},
                },
                "may not include an action",
            ),
            (
                {"status": "continue", "action": ["ACTION1"], "memory": {}},
                "action must be a mapping",
            ),
            (
                {
                    "status": "continue",
                    "action": {"action": "ACTION1"},
                    "prediction": [1],
                    "memory": {},
                },
                "prediction must be a mapping",
            ),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PolicyRuntimeError, message) as captured:
                    PolicyDecision.from_payload(payload, valid_actions=("ACTION1",))
                self.assertEqual("invalid_decision", captured.exception.category)

    def test_policy_decision_rejects_non_json_and_non_finite_state(self) -> None:
        for memory in ({1, 2}, {"score": float("nan")}, object()):
            with self.subTest(memory=type(memory).__name__):
                with self.assertRaisesRegex(PolicyRuntimeError, "finite JSON"):
                    PolicyDecision.from_payload(
                        {
                            "status": "continue",
                            "action": {"action": "ACTION1"},
                            "memory": memory,
                        },
                        valid_actions=("ACTION1",),
                    )

    def test_mouse_action_requires_bounded_integer_coordinates(self) -> None:
        accepted = PolicyDecision.from_payload(
            {
                "status": "continue",
                "action": {"action": "mouse", "row": "4", "col": 63},
                "memory": {},
            },
            valid_actions=("MOUSE",),
        )
        self.assertEqual({"action": "MOUSE", "row": 4, "col": 63}, accepted.action)
        for action in (
            {"action": "MOUSE"},
            {"action": "MOUSE", "row": "north", "col": 3},
            {"action": "MOUSE", "row": -1, "col": 3},
            {"action": "MOUSE", "row": 4, "col": 64},
        ):
            with self.subTest(action=action):
                with self.assertRaises(PolicyRuntimeError) as captured:
                    PolicyDecision.from_payload(
                        {"status": "continue", "action": action, "memory": {}},
                        valid_actions=("MOUSE",),
                    )
                self.assertEqual("invalid_action", captured.exception.category)

    def test_mouse_alias_preserves_bounded_coordinates(self) -> None:
        accepted = PolicyDecision.from_payload(
            {
                "status": "continue",
                "action": {"action": "ACTION6", "row": "47", "col": 18},
                "memory": {},
            },
            valid_actions=("ACTION6",),
        )
        self.assertEqual({"action": "ACTION6", "row": 47, "col": 18}, accepted.action)
        with self.assertRaisesRegex(PolicyRuntimeError, "ACTION6 coordinates"):
            PolicyDecision.from_payload(
                {
                    "status": "continue",
                    "action": {"action": "ACTION6", "row": 47},
                    "memory": {},
                },
                valid_actions=("ACTION6",),
            )

    def test_model_facing_action_contract_rejects_engine_aliases(self) -> None:
        accepted = PolicyDecision.from_payload(
            {
                "status": "continue",
                "action": {"action": "MOUSE", "row": 11, "col": 29},
                "memory": {},
            },
            valid_actions=("UP", "MOUSE"),
        )
        self.assertEqual({"action": "MOUSE", "row": 11, "col": 29}, accepted.action)
        with self.assertRaisesRegex(PolicyRuntimeError, "not currently valid"):
            PolicyDecision.from_payload(
                {
                    "status": "continue",
                    "action": {"action": "ACTION6", "row": 11, "col": 29},
                    "memory": {},
                },
                valid_actions=("UP", "MOUSE"),
            )

    def test_forbidden_import_is_rejected_before_worker_start(self) -> None:
        with self.assertRaisesRegex(PolicyRuntimeError, "not permitted"):
            verify_policy_source(
                "import os\nPOLICY_API_VERSION=1\nSUPPORTED_BACKENDS=('cpu',)\ndef decide(o,m): return {}"
            )

    def test_source_verifier_rejects_empty_syntax_top_level_and_dangerous_ast(
        self,
    ) -> None:
        cases = (
            ("", "empty"),
            ("def decide(:", "syntax error"),
            (
                "while False:\n    pass\nPOLICY_API_VERSION=1\nSUPPORTED_BACKENDS=('cpu',)\n",
                "top-level While",
            ),
            (
                "POLICY_API_VERSION=1\nSUPPORTED_BACKENDS=('cpu',)\ndef decide(o,m): return eval('1')",
                "call 'eval'",
            ),
            (
                "POLICY_API_VERSION=1\nSUPPORTED_BACKENDS=('cpu',)\ndef decide(o,m): return o.__dict__",
                "attribute '__dict__'",
            ),
            (
                "POLICY_API_VERSION=1\nSUPPORTED_BACKENDS=('cpu',)\ndef decide(o,m):\n    with m:\n        pass",
                "With is not permitted",
            ),
        )
        for source, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PolicyRuntimeError, message) as captured:
                    verify_policy_source(source)
                self.assertEqual("policy_verification", captured.exception.category)

    def test_source_verifier_requires_exact_policy_entrypoint_signatures(self) -> None:
        cases = (
            (
                "POLICY_API_VERSION=1\nSUPPORTED_BACKENDS=('cpu',)\n",
                r"must define decide\(observation, memory\)",
            ),
            (
                "POLICY_API_VERSION=1\nSUPPORTED_BACKENDS=('cpu',)\n"
                "def wrapper():\n"
                "    def decide(observation, memory):\n"
                "        return {}\n",
                r"must define decide\(observation, memory\)",
            ),
            (
                GOOD_POLICY.replace(
                    "def decide(observation, memory):",
                    "def decide(observation):",
                ),
                "decide signature must be exactly",
            ),
            (
                GOOD_POLICY.replace(
                    "def decide(observation, memory):",
                    "def decide(observation, memory, extra=None):",
                ),
                "decide signature must be exactly",
            ),
            (
                GOOD_POLICY.replace(
                    "def initialize(context):",
                    "def initialize(context, extra):",
                ),
                "initialize signature must be exactly",
            ),
        )
        for source, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PolicyRuntimeError, message) as captured:
                    verify_policy_source(source)
                self.assertEqual("policy_verification", captured.exception.category)

    def test_source_verifier_rejects_invalid_observation_contract_usage(self) -> None:
        cases = (
            (
                "engine_state = observation.engine_state",
                "has no attribute 'engine_state'",
            ),
            (
                "rows = observation.board_hex_rows",
                "has no attribute 'board_hex_rows'",
            ),
            (
                "board = observation.board\n    if not board:\n        return {}",
                "cannot be used as a boolean",
            ),
            (
                "board = observation.board\n    changed = board != memory.get('board')",
                "cannot be compared directly",
            ),
            (
                "board = observation.board\n    memory = {'board': board}",
                "cannot be stored in a mapping",
            ),
        )
        for body, message in cases:
            source = (
                "POLICY_API_VERSION = 1\n"
                'SUPPORTED_BACKENDS = ("cpu",)\n'
                "def decide(observation, memory):\n"
                f"    {body}\n"
                "    return {'status': 'subgoal_failed', 'memory': {}}\n"
            )
            with self.subTest(message=message):
                with self.assertRaisesRegex(PolicyRuntimeError, message):
                    verify_policy_source(source)

    def test_source_verifier_reports_all_detectable_static_violations(self) -> None:
        source = """
import hashlib
import json
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    actions = getattr(observation, "valid_actions", ())
    state = observation.engine_state
    return {"status": "continue", "action": {"action": actions[0]}, "memory": memory}
"""
        with self.assertRaises(PolicyRuntimeError) as captured:
            verify_policy_source(source)

        detail = str(captured.exception)
        self.assertIn("import 'hashlib' is not permitted", detail)
        self.assertIn("import 'json' is not permitted", detail)
        self.assertIn("call 'getattr' is not permitted", detail)
        self.assertIn("PolicyObservation has no attribute 'engine_state'", detail)

    def test_source_fingerprint_is_semantic(self) -> None:
        formatted = "\n# harmless comment\n" + GOOD_POLICY.replace(
            'evidence": "deterministic smoke decision",',
            'evidence": "deterministic smoke decision",  # same AST',
        )
        self.assertEqual(
            verify_policy_source(GOOD_POLICY), verify_policy_source(formatted)
        )
        changed = GOOD_POLICY.replace("+ 1}", "+ 2}")
        self.assertNotEqual(
            verify_policy_source(GOOD_POLICY), verify_policy_source(changed)
        )

    def test_activation_rejects_invalid_api_and_entrypoint_contracts(self) -> None:
        cases = (
            (
                GOOD_POLICY.replace("POLICY_API_VERSION = 1", "POLICY_API_VERSION = 2"),
                "POLICY_API_VERSION",
            ),
            (
                GOOD_POLICY.replace(
                    'SUPPORTED_BACKENDS = ("cpu",)', 'SUPPORTED_BACKENDS = "cpu"'
                ),
                "list or tuple",
            ),
            (
                GOOD_POLICY.replace(
                    'SUPPORTED_BACKENDS = ("cpu",)',
                    'SUPPORTED_BACKENDS = ("cpu", "cpu")',
                ),
                "optional cuda once",
            ),
            (
                GOOD_POLICY.replace(
                    'SUPPORTED_BACKENDS = ("cpu",)',
                    'SUPPORTED_BACKENDS = ("cpu", "metal")',
                ),
                "optional cuda once",
            ),
            (
                "POLICY_API_VERSION=1\nSUPPORTED_BACKENDS=('cpu',)\n",
                "must define decide",
            ),
            (
                GOOD_POLICY.replace('return {"calls": 0}', "return {1, 2}"),
                "finite JSON",
            ),
            (
                GOOD_POLICY + "\ndef self_test():\n    return False\n",
                "self_test returned false",
            ),
        )
        for source, message in cases:
            with self.subTest(message=message):
                with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
                    with self.assertRaisesRegex(PolicyRuntimeError, message):
                        runtime.activate(source, context={})

    def test_activation_timeout_terminates_worker(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def initialize(context):
    while True:
        pass
def decide(observation, memory):
    return {"status": "subgoal_failed", "memory": {}}
"""
        with GameplayPolicyRuntime(
            requested_backend="cpu", activation_timeout_seconds=0.1
        ) as runtime:
            with self.assertRaises(PolicyRuntimeError) as captured:
                runtime.activate(source, context={})
            self.assertIsNone(runtime.activation)
        self.assertEqual("policy_timeout", captured.exception.category)

    def test_decision_timeout_terminates_worker(self) -> None:
        source = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    while True:
        pass
"""
        with GameplayPolicyRuntime(
            requested_backend="cpu", decision_timeout_seconds=0.1
        ) as runtime:
            runtime.activate(source, context={})
            with self.assertRaises(PolicyRuntimeError) as captured:
                runtime.decide(observation())
        self.assertEqual("policy_timeout", captured.exception.category)

    def test_runtime_exception_stops_worker_and_clears_activation(self) -> None:
        source = GOOD_POLICY.replace(
            'memory = {"calls": int(memory.get("calls", 0)) + 1}',
            "memory = 1 / 0",
        )
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            with self.assertRaises(PolicyRuntimeError) as captured:
                runtime.decide(observation())
            self.assertIsNone(runtime.activation)
            with self.assertRaisesRegex(PolicyRuntimeError, "no active policy"):
                runtime.decide(observation())
        self.assertEqual("policy_runtime", captured.exception.category)

    def test_policy_cannot_mutate_observation_board(self) -> None:
        source = GOOD_POLICY.replace(
            'memory = {"calls": int(memory.get("calls", 0)) + 1}',
            "observation.board[0, 0] = 7",
        )
        with GameplayPolicyRuntime(requested_backend="cpu") as runtime:
            runtime.activate(source, context={})
            with self.assertRaises(PolicyRuntimeError) as captured:
                runtime.decide(observation())
        self.assertEqual("policy_runtime", captured.exception.category)

    def test_set_memory_rejects_non_json_values(self) -> None:
        runtime = GameplayPolicyRuntime(requested_backend="cpu")
        for value in ({"x": float("inf")}, {1, 2}):
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(PolicyRuntimeError, "finite JSON"):
                    runtime.set_memory(value)
        runtime.close()

    def test_auto_cuda_requires_headroom(self) -> None:
        backend, reason = choose_policy_backend(
            "auto",
            ("cpu", "cuda"),
            min_free_mb=4096,
            torch_module=_FakeTorch(available=True, free_mb=2048),
        )
        self.assertEqual("cpu", backend)
        self.assertIn("cuda_headroom", reason)
        backend, reason = choose_policy_backend(
            "auto",
            ("cpu", "cuda"),
            min_free_mb=4096,
            torch_module=_FakeTorch(available=True, free_mb=8192),
        )
        self.assertEqual(("cuda", ""), (backend, reason))

    def test_strict_cuda_fails_when_unavailable(self) -> None:
        with self.assertRaises(PolicyRuntimeError) as captured:
            choose_policy_backend(
                "cuda",
                ("cpu", "cuda"),
                min_free_mb=1,
                torch_module=_FakeTorch(available=False, free_mb=0),
            )
        self.assertEqual("backend_unavailable", captured.exception.category)

    def test_backend_selection_rejects_unknown_or_cpu_less_policy(self) -> None:
        for requested, supported in (("metal", ("cpu",)), ("cpu", ("cuda",))):
            with self.subTest(requested=requested, supported=supported):
                with self.assertRaises(PolicyRuntimeError) as captured:
                    choose_policy_backend(
                        requested,
                        supported,
                        min_free_mb=1,
                        torch_module=_FakeTorch(available=True, free_mb=8192),
                    )
                self.assertEqual("backend_unavailable", captured.exception.category)

    def test_auto_backend_records_policy_and_probe_fallback_reasons(self) -> None:
        backend, reason = choose_policy_backend(
            "auto",
            ("cpu",),
            min_free_mb=1,
            torch_module=_FakeTorch(available=True, free_mb=8192),
        )
        self.assertEqual(("cpu", "policy_does_not_support_cuda"), (backend, reason))
        backend, reason = choose_policy_backend(
            "auto", ("cpu", "cuda"), min_free_mb=1, torch_module=_BrokenTorch()
        )
        self.assertEqual("cpu", backend)
        self.assertEqual("cuda_probe_failed:RuntimeError", reason)

    def test_auto_cuda_oom_reactivates_on_cpu_and_preserves_memory(self) -> None:
        source_hash = verify_policy_source(GOOD_POLICY)
        runtime = GameplayPolicyRuntime(requested_backend="auto")
        runtime._source = GOOD_POLICY
        runtime._context = {"game_id": "g"}
        runtime._memory = {"calls": 7}
        runtime.activation = PolicyActivation(
            source_hash=source_hash,
            backend="cuda",
            supported_backends=("cpu", "cuda"),
        )
        cpu_activation = PolicyActivation(
            source_hash=source_hash,
            backend="cpu",
            supported_backends=("cpu", "cuda"),
        )
        decision_response = {
            "ok": True,
            "decision": {
                "status": "continue",
                "action": {"action": "ACTION1"},
                "memory": {"calls": 8},
                "evidence": "cpu retry",
                "prediction": None,
            },
        }

        def activate_on_cpu(_source: str, *, context: dict) -> PolicyActivation:
            self.assertEqual({"game_id": "g"}, context)
            self.assertEqual("cpu", runtime.requested_backend)
            runtime.activation = cpu_activation
            runtime._memory = {"initialized": True}
            return cpu_activation

        with (
            mock.patch.object(
                runtime,
                "_exchange",
                side_effect=[
                    PolicyRuntimeError("CUDA out of memory", category="cuda_oom"),
                    decision_response,
                ],
            ),
            mock.patch.object(runtime, "activate", side_effect=activate_on_cpu),
        ):
            decision = runtime.decide(observation())
        self.assertEqual({"calls": 8}, decision.memory)
        self.assertEqual("cpu", runtime.activation.backend)
        self.assertEqual("cuda_oom", runtime.activation.backend_fallback_reason)
        self.assertEqual("auto", runtime.requested_backend)


if __name__ == "__main__":
    unittest.main()
