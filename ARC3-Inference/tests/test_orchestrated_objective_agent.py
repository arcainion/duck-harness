from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import requests

from inference.agent.gameplay_policy_runtime import (
    PolicyActivation,
    PolicyDecision,
    PolicyRuntimeError,
    PolicyStatus,
    verify_policy_source,
)
from inference.agent.orchestrated_objective_agent import (
    OrchestrationFailure,
    OrchestratedObjectiveAgent,
    _action_family_saturation_reason,
    _contract_requires_navigation,
    _contract_requests_mouse,
    _equivalent_attempted_tactical,
    _failed_engine_progress_since_recalibration,
    _meaningful_progress,
    _objective_contract_hash,
    _policy_source_from_message,
    _policy_reuse_scope_from_source,
    _policy_transition_payload,
    _reduction_from_message,
    _repeats_non_progress_action,
    _tactical_contract_similarity,
)
from inference.agent.objective_reduction import (
    ObjectiveStatus,
    ObjectiveTree,
    ReductionProposal,
    TacticalExecutionMode,
)
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import AnalyzerModelConfig
from inference.framework.solver import HarnessSolver


POLICY_SOURCE = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    return {"status": "continue", "action": {"action": "UP"}, "memory": memory}
"""

REUSABLE_POLICY_SOURCE = POLICY_SOURCE.replace(
    'SUPPORTED_BACKENDS = ("cpu",)',
    'SUPPORTED_BACKENDS = ("cpu",)\nPOLICY_REUSE_SCOPE = "tactical"',
)

NAVIGATION_POLICY_SOURCE = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
POLICY_REUSE_SCOPE = "none"
def decide(observation, memory):
    actor = first_matching_cell(observation.board, (1,))
    targets = find_cells(observation.board, (2,))
    passable = value_mask(observation.board, (0, 1))
    if transition_requires_replan(observation.last_transition):
        memory = {}
    if actor is None or not targets:
        return subgoal_failed(memory, "actor or target is unavailable")
    path = shortest_path(passable, actor, targets[0])
    return path_decision(path, observation.valid_actions, memory, "board-derived route")
"""


def frame(
    *,
    level: int = 1,
    step: int = 0,
    engine_state: str = "PLAYING",
    valid_actions: tuple[str, ...] = ("ACTION1",),
) -> Frame:
    grid = tuple(tuple(0 for _ in range(64)) for _ in range(64))
    return Frame(
        grid=grid,
        step=step,
        level=level,
        valid_actions=valid_actions,
        engine_state=engine_state,
        score=0,
    )


def reduction_for(
    objective_id: str,
    *,
    verdict: str = "decompose",
    title: str = "Probe ACTION1",
) -> dict[str, object]:
    subgoals: list[dict[str, object]] = []
    if verdict == "decompose":
        subgoals = [
            {
                "title": title,
                "success_criteria": "board changes",
                "failure_criteria": "action is guarded or unchanged",
                "expected_evidence": "one transition",
                "action_budget": 4,
                "minimum_evidence_actions": 4,
                "single_step": False,
            }
        ]
    return {
        "objective_id": objective_id,
        "verdict": verdict,
        "evidence": "scripted evidence",
        "rationale": "scripted rationale",
        "selected_index": 0,
        "subgoals": subgoals,
    }


def policy_for(
    objective_id: str,
    *,
    source: str = POLICY_SOURCE,
    capabilities: list[str] | None = None,
) -> dict[str, object]:
    return {
        "objective_id": objective_id,
        "source": source,
        "backend_capabilities": capabilities or ["cpu"],
        "self_test_notes": "deterministic",
    }


class _FakeModelClient:
    def __init__(self) -> None:
        self.calls = 0
        self.call_kwargs: list[dict[str, object]] = []
        self.tools: list[list[dict] | None] = []

    def complete(
        self,
        _messages: list[dict],
        *,
        tools: list[dict] | None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        self.calls += 1
        self.call_kwargs.append(dict(_kwargs))
        self.tools.append(tools)
        if "objective-reducer role" in _messages[0]["content"]:
            payload = {
                "objective_id": "level:1:1",
                "verdict": "decompose",
                "evidence": "initial observation",
                "rationale": "probe one legal movement",
                "selected_index": 0,
                "subgoals": [
                    {
                        "title": "Probe ACTION1",
                        "success_criteria": "board changes",
                        "failure_criteria": "action is guarded or unchanged",
                        "expected_evidence": "one transition",
                        "action_budget": 4,
                        "minimum_evidence_actions": 4,
                        "single_step": False,
                    }
                ],
            }
            message = {
                "role": "assistant",
                "content": (f"BEGIN_REDUCTION\n{json.dumps(payload)}\nEND_REDUCTION"),
            }
        else:
            message = {
                "role": "assistant",
                "content": f"BEGIN_POLICY\n{POLICY_SOURCE}\nEND_POLICY",
            }
        return SimpleNamespace(
            message=message,
            usage={"completion_tokens": 10, "total_tokens": 20},
            request_attempts=1,
        )


class _ScriptedModelClient:
    def __init__(
        self, responses: list[dict[str, object] | str | BaseException]
    ) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.messages: list[list[dict]] = []
        self.call_kwargs: list[dict[str, object]] = []
        self.tools: list[list[dict] | None] = []

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        self.calls += 1
        self.messages.append(messages)
        self.call_kwargs.append(dict(_kwargs))
        self.tools.append(tools)
        if not self.responses:
            raise AssertionError("scripted model response queue was exhausted")
        payload = self.responses.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        if not tools:
            if isinstance(payload, str):
                content = payload
            elif "objective-reducer role" in messages[0]["content"]:
                content = f"BEGIN_REDUCTION\n{json.dumps(payload)}\nEND_REDUCTION"
            else:
                content = f"BEGIN_POLICY\n{payload['source']}\nEND_POLICY"
            return SimpleNamespace(
                message={
                    "role": "assistant",
                    "content": content,
                },
                usage={"completion_tokens": 7, "total_tokens": 11},
                request_attempts=1,
            )
        if not isinstance(payload, dict):
            raise AssertionError("tool response payload must be an object")
        name = tools[0]["function"]["name"]
        return SimpleNamespace(
            message={
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"scripted-{self.calls}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(payload),
                        },
                    }
                ],
            },
            usage={"completion_tokens": 7, "total_tokens": 11},
            request_attempts=1,
        )


class _FakeRuntime:
    def __init__(self) -> None:
        self.activation: PolicyActivation | None = None
        self.memory: object = {}

    def activate(self, source: str, *, context: dict) -> PolicyActivation:
        del context
        self.activation = PolicyActivation(
            source_hash=verify_policy_source(source),
            backend="cpu",
            supported_backends=("cpu",),
        )
        return self.activation

    def decide(self, _observation: object) -> PolicyDecision:
        return PolicyDecision(
            status=PolicyStatus.CONTINUE,
            action={"action": "UP"},
            memory=self.memory,
            evidence="ordinary CPU policy step",
        )

    def preflight(self, _observation: object, *, minimum_actions: int = 4) -> None:
        del minimum_actions

    def set_memory(self, value: object) -> None:
        self.memory = value

    def close(self) -> None:
        self.activation = None


class _DecisionRuntime(_FakeRuntime):
    def __init__(self, decision: PolicyDecision) -> None:
        super().__init__()
        self._decision = decision

    def decide(self, _observation: object) -> PolicyDecision:
        self.memory = self._decision.memory
        return self._decision


class _SequentialDecisionRuntime(_FakeRuntime):
    def __init__(self, decisions: list[PolicyDecision]) -> None:
        super().__init__()
        self._decisions = list(decisions)
        self.decide_calls = 0

    def decide(self, _observation: object) -> PolicyDecision:
        self.decide_calls += 1
        if not self._decisions:
            raise AssertionError("scripted runtime decision queue was exhausted")
        decision = self._decisions.pop(0)
        self.memory = decision.memory
        return decision


class _RuntimeFactory:
    def __init__(self, decisions: list[PolicyDecision]) -> None:
        self.decisions = list(decisions)
        self.instances: list[_DecisionRuntime] = []

    def __call__(self) -> _DecisionRuntime:
        if not self.decisions:
            raise AssertionError("scripted runtime decision queue was exhausted")
        runtime = _DecisionRuntime(self.decisions.pop(0))
        self.instances.append(runtime)
        return runtime


class _ActivationFailureRuntime(_FakeRuntime):
    def activate(self, source: str, *, context: dict) -> PolicyActivation:
        del source, context
        raise PolicyRuntimeError(
            "generated module failed activation", category="policy_verification"
        )


class _ActivationRepairRuntimeFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> _FakeRuntime:
        self.calls += 1
        if self.calls == 1:
            return _ActivationFailureRuntime()
        return _FakeRuntime()


class OrchestratedObjectiveAgentTests(unittest.TestCase):
    def test_spatial_contract_classification_requires_navigation_mode(self) -> None:
        spatial = ReductionProposal.from_payload(
            reduction_for(
                "level:1:1", title="Move the actor toward the target boundary"
            )
        ).subgoals[0]
        probe = ReductionProposal.from_payload(
            reduction_for("level:1:1", title="Probe ACTION7 for a stable effect")
        ).subgoals[0]

        self.assertTrue(_contract_requires_navigation(spatial))
        self.assertFalse(_contract_requires_navigation(probe))

    def test_navigation_policy_requires_localization_planning_and_replanning(
        self,
    ) -> None:
        reduction = reduction_for("level:1:1", title="Reach the target component")
        subgoals = reduction["subgoals"]
        assert isinstance(subgoals, list)
        subgoals[0]["execution_mode"] = "navigate"
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction),
            remaining_level_actions=20,
        )
        agent._tree = tree

        with self.assertRaisesRegex(PolicyRuntimeError, "localization") as raised:
            agent._policy_validator({"source": POLICY_SOURCE})
        self.assertEqual("policy_navigation_contract", raised.exception.category)
        without_replan = NAVIGATION_POLICY_SOURCE.replace(
            "    if transition_requires_replan(observation.last_transition):\n"
            "        memory = {}\n",
            "",
        )
        with self.assertRaisesRegex(PolicyRuntimeError, "transition_requires_replan"):
            agent._policy_validator({"source": without_replan})
        without_passability = NAVIGATION_POLICY_SOURCE.replace(
            "    passable = value_mask(observation.board, (0, 1))\n",
            "    passable = observation.board\n",
        )
        with self.assertRaisesRegex(PolicyRuntimeError, "passability-mask"):
            agent._policy_validator({"source": without_passability})
        unreachable_helpers = (
            POLICY_SOURCE
            + "\ndef unused_navigation(observation):\n"
            "    cells = find_cells(observation.board, (1,))\n"
            "    transition_requires_replan(observation.last_transition)\n"
            "    return shortest_path(value_mask(observation.board, (0,)), "
            "cells[0], cells[-1])\n"
        )
        with self.assertRaisesRegex(PolicyRuntimeError, "localization"):
            agent._policy_validator({"source": unreachable_helpers})
        accepted = agent._policy_validator({"source": NAVIGATION_POLICY_SOURCE})

        self.assertEqual(
            verify_policy_source(NAVIGATION_POLICY_SOURCE), accepted["source_hash"]
        )
        self.assertEqual(TacticalExecutionMode.NAVIGATE, tree.active.execution_mode)
        agent.close()

    def test_reducer_rejects_spatial_contract_without_navigation_mode(self) -> None:
        spatial = reduction_for(
            "level:1:1", title="Drive the actor toward the goal boundary"
        )
        agent = self._agent()
        agent._tree = ObjectiveTree.start_game(
            "game-a", level=1, level_action_budget=20
        )
        agent.model_client = _ScriptedModelClient([spatial] * 3)

        with self.assertRaises(OrchestrationFailure) as raised:
            agent._reduce(frame(), [], request_deadline=None, should_stop=None)

        self.assertIn("execution_mode=navigate", str(raised.exception))
        agent.close()

    def test_policy_reuse_scope_requires_a_literal_opt_in(self) -> None:
        self.assertEqual(
            "tactical", _policy_reuse_scope_from_source(REUSABLE_POLICY_SOURCE)
        )
        self.assertEqual("none", _policy_reuse_scope_from_source(POLICY_SOURCE))
        self.assertEqual(
            "none",
            _policy_reuse_scope_from_source(
                'POLICY_REUSE_SCOPE = "tactical" if True else "none"'
            ),
        )

    def test_successful_same_level_policy_is_reused_with_fresh_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_path = root / "runtime_state.json"
            runtime_path.touch()
            policy_dir = root / "policies"
            policy_dir.mkdir()
            source_hash = verify_policy_source(REUSABLE_POLICY_SOURCE)
            artifact = policy_dir / f"tactical_1-{source_hash}.py"
            artifact.write_text(REUSABLE_POLICY_SOURCE, encoding="utf-8")

            agent = self._agent()
            agent._knowledge_game_id = "game-a"
            agent._session_runtime_dir = runtime_path
            agent._tree = ObjectiveTree.start_game(
                "game-a", level=1, level_action_budget=32
            )
            origin = agent._tree.apply_proposal(
                ReductionProposal.from_payload(reduction_for("level:1:1")),
                remaining_level_actions=32,
            )
            active_runtime = _FakeRuntime()
            active_runtime.activate(REUSABLE_POLICY_SOURCE, context={})
            agent._policy_runtime = active_runtime
            agent._policy_objective_id = origin.objective_id
            agent._policy_source_hash = source_hash
            agent._policy_artifact = str(artifact.relative_to(root))
            agent._active_policy_reuse_scope = "tactical"
            agent._policy_executed_action = True
            agent._policy_observed_host_progress = True
            agent._tree.record_action()
            agent._tree.complete_active_tactical("contract satisfied")
            agent._invalidate_policy(f"subgoal_succeeded:{origin.objective_id}")

            replacement = agent._tree.apply_proposal(
                ReductionProposal.from_payload(reduction_for("level:1:1")),
                remaining_level_actions=31,
            )
            agent.model_client.complete = mock.Mock(
                side_effect=AssertionError("coder must not run on successful reuse")
            )
            with mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ):
                agent._activate_policy(
                    frame(),
                    [],
                    request_deadline=None,
                    should_stop=None,
                    repair_reason=f"subgoal_succeeded:{origin.objective_id}",
                )

            self.assertEqual(replacement.objective_id, agent._policy_objective_id)
            self.assertEqual(1, agent._orchestration_metrics["policy_reuses"])
            agent.model_client.complete.assert_not_called()
            self.assertEqual({}, agent._policy_memory)
            self.assertEqual(
                origin.objective_id, agent._reusable_policies[0]["origin_objective_id"]
            )
            self.assertEqual("proven", agent._reusable_policies[0]["qualification"])
            agent.close()

    def test_provisional_policy_reuses_only_matching_contract_then_is_evicted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_path = root / "runtime_state.json"
            runtime_path.touch()
            policy_dir = root / "policies"
            policy_dir.mkdir()
            source_hash = verify_policy_source(REUSABLE_POLICY_SOURCE)
            artifact = policy_dir / f"tactical_1-{source_hash}.py"
            artifact.write_text(REUSABLE_POLICY_SOURCE, encoding="utf-8")

            agent = self._agent()
            agent._knowledge_game_id = "game-a"
            agent._session_runtime_dir = runtime_path
            agent._tree = ObjectiveTree.start_game(
                "game-a", level=1, level_action_budget=32
            )
            origin = agent._tree.apply_proposal(
                ReductionProposal.from_payload(reduction_for("level:1:1")),
                remaining_level_actions=32,
            )
            agent._policy_objective_id = origin.objective_id
            agent._policy_source_hash = source_hash
            agent._policy_artifact = str(artifact.relative_to(root))
            agent._active_policy_reuse_scope = "tactical"
            agent._policy_executed_action = True
            agent._tree.record_action()
            agent._tree.complete_active_tactical("mapping evidence collected")
            agent._invalidate_policy(f"subgoal_succeeded:{origin.objective_id}")

            cached = agent._reusable_policies[0]
            self.assertEqual("provisional", cached["qualification"])
            self.assertEqual(_objective_contract_hash(origin), cached["contract_hash"])

            replacement = agent._tree.apply_proposal(
                ReductionProposal.from_payload(reduction_for("level:1:1")),
                remaining_level_actions=31,
            )
            with mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ):
                self.assertTrue(agent._try_reuse_policy(frame()))
            self.assertEqual(replacement.objective_id, agent._policy_objective_id)
            self.assertTrue(agent._policy_was_reused)

            agent._policy_executed_action = True
            agent._tree.record_action()
            agent._tree.complete_active_tactical("same unproductive local evidence")
            agent._invalidate_policy(f"subgoal_succeeded:{replacement.objective_id}")
            self.assertEqual([], agent._reusable_policies)
            self.assertEqual(1, agent._orchestration_metrics["policy_reuse_evictions"])
            agent.close()

    def test_provisional_policy_does_not_cross_tactical_contracts(self) -> None:
        agent = self._agent()
        agent._knowledge_game_id = "game-a"
        agent._tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=32)
        origin = agent._tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=32,
        )
        agent._reusable_policies = [
            {
                "game_id": "game-a",
                "level": 1,
                "source_hash": "hash",
                "artifact": "policies/missing.py",
                "origin_objective_id": origin.objective_id,
                "contract_hash": _objective_contract_hash(origin),
                "qualification": "provisional",
            }
        ]
        agent._tree.complete_active_tactical("done")
        replacement = agent._tree.apply_proposal(
            ReductionProposal.from_payload(
                reduction_for("level:1:1", title="Inspect a different mechanism")
            ),
            remaining_level_actions=32,
        )
        self.assertNotEqual(
            agent._reusable_policies[0]["contract_hash"],
            _objective_contract_hash(replacement),
        )
        self.assertFalse(agent._try_reuse_policy(frame()))
        agent.close()

    def test_policy_transition_exposes_controller_progress_evidence(self) -> None:
        action = {"action": "MOUSE", "row": 30, "col": 61}
        volatile_result = {
            "executed": True,
            "board_changed": True,
            "reward": 0.0,
            "score": 0,
            "level": 1,
            "state": "NOT_FINISHED",
            "outcome_class": "volatile_only",
            "novel_state": False,
            "decision_context_changed": False,
            "animation": {
                "frame_count": 1,
                "total_changed_cells": 1,
                "dominant_color_transitions": [{"from": "P", "to": "c"}],
            },
        }

        transition = _policy_transition_payload(
            action,
            volatile_result,
            objective_id="tactical:1",
            policy_hash="policy-hash",
        )

        self.assertFalse(transition["meaningful_progress"])
        self.assertEqual("volatile_only", transition["outcome_class"])
        self.assertEqual("NOT_FINISHED", transition["engine_state"])
        self.assertEqual(1, transition["animation_summary"]["frame_count"])
        self.assertNotIn("dominant_color_transitions", transition["animation_summary"])
        self.assertFalse(
            _meaningful_progress(
                {
                    **volatile_result,
                    "outcome_class": "novel",
                    "novel_state": True,
                }
            )
        )
        self.assertFalse(
            _meaningful_progress(
                {
                    **volatile_result,
                    "outcome_class": "novel",
                    "novel_state": True,
                    "meaningful_progress": False,
                }
            )
        )
        self.assertTrue(
            _meaningful_progress(
                {**volatile_result, "outcome_class": "level_progress", "reward": 0.1}
            )
        )

    def test_repeated_non_progress_action_is_detected_per_objective(self) -> None:
        previous = {
            "objective_id": "tactical:1",
            "action": "MOUSE",
            "row": 30,
            "col": 61,
            "outcome_class": "volatile_only",
        }
        same = {"action": "MOUSE", "row": 30, "col": 61}

        self.assertTrue(
            _repeats_non_progress_action(previous, same, objective_id="tactical:1")
        )
        self.assertFalse(
            _repeats_non_progress_action(
                previous,
                {"action": "MOUSE", "row": 30, "col": 62},
                objective_id="tactical:1",
            )
        )
        self.assertFalse(
            _repeats_non_progress_action(
                {**previous, "outcome_class": "novel"},
                same,
                objective_id="tactical:1",
            )
        )

    def test_attempted_tactical_contract_paraphrase_is_detected(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        first = ReductionProposal.from_payload(
            {
                **reduction_for("level:1:1"),
                "subgoals": [
                    {
                        "title": "Calibrate UP against row 63 segment growth",
                        "success_criteria": (
                            "repeat UP and establish whether the row 63 segment grows "
                            "toward level completion"
                        ),
                        "failure_criteria": "UP produces no repeatable segment change",
                        "expected_evidence": (
                            "four transitions comparing UP with row 63 segment length"
                        ),
                        "action_budget": 8,
                        "minimum_evidence_actions": 4,
                        "single_step": False,
                    }
                ],
            }
        )
        tree.apply_proposal(first, remaining_level_actions=20)
        tree.record_action()
        tree.fail_active_tactical("UP did not advance the level")

        repeated = ReductionProposal.from_payload(
            {
                **reduction_for("level:1:1"),
                "subgoals": [
                    {
                        "title": "Repeat UP to grow the row 63 segment",
                        "success_criteria": (
                            "establish row 63 segment growth by repeating UP toward "
                            "level completion"
                        ),
                        "failure_criteria": "no repeatable growth is observed",
                        "expected_evidence": (
                            "four UP transitions measuring row 63 segment length"
                        ),
                        "action_budget": 8,
                        "minimum_evidence_actions": 4,
                        "single_step": False,
                    }
                ],
            }
        ).subgoals[0]
        original = tree.nodes["tactical:1"]

        self.assertGreaterEqual(
            _tactical_contract_similarity(original, repeated), 0.72
        )
        self.assertIs(original, _equivalent_attempted_tactical(tree, repeated))

        distinct = ReductionProposal.from_payload(
            reduction_for("level:1:1", title="Probe a central mouse target")
        ).subgoals[0]
        self.assertIsNone(_equivalent_attempted_tactical(tree, distinct))

    def test_attempted_contract_matching_preserves_distinct_spatial_targets(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)

        def targeted_subgoal(title: str) -> dict[str, object]:
            return {
                "title": title,
                "success_criteria": (
                    "an executed mouse probe produces a stable board response for "
                    "the candidate region"
                ),
                "failure_criteria": "the candidate produces no useful response",
                "expected_evidence": (
                    "four transitions comparing candidate coordinates and stable "
                    "board responses"
                ),
                "action_budget": 8,
                "minimum_evidence_actions": 4,
                "single_step": False,
            }

        first = ReductionProposal.from_payload(
            {
                **reduction_for("level:1:1"),
                "subgoals": [targeted_subgoal("Click west wall cell")],
            }
        )
        tree.apply_proposal(first, remaining_level_actions=20)
        tree.record_action()
        tree.fail_active_tactical("west wall did not respond")
        distinct = ReductionProposal.from_payload(
            {
                **reduction_for("level:1:1"),
                "subgoals": [targeted_subgoal("Probe north border coordinate")],
            }
        ).subgoals[0]
        original = tree.nodes["tactical:1"]

        self.assertGreaterEqual(
            _tactical_contract_similarity(original, distinct), 0.72
        )
        self.assertIsNone(_equivalent_attempted_tactical(tree, distinct))

    def test_mouse_family_saturation_requires_repeated_pure_no_progress(self) -> None:
        mouse_spec = ReductionProposal.from_payload(
            reduction_for("level:1:1", title="Click a candidate MOUSE coordinate")
        ).subgoals[0]
        movement_spec = ReductionProposal.from_payload(
            reduction_for("level:1:1", title="Move the object upward")
        ).subgoals[0]
        evidence = {
            "MOUSE": {
                "executed": 12,
                "no_progress": 12,
                "stable_changes": 0,
                "meaningful_progress": 0,
                "distinct_points": [f"{index},10" for index in range(12)],
            }
        }

        self.assertTrue(_contract_requests_mouse(mouse_spec))
        self.assertFalse(_contract_requests_mouse(movement_spec))
        self.assertIn(
            "12 distinct coordinates",
            _action_family_saturation_reason(evidence, "MOUSE"),
        )
        self.assertEqual("", _action_family_saturation_reason(evidence, "LEFT"))
        evidence["MOUSE"]["stable_changes"] = 1
        self.assertEqual("", _action_family_saturation_reason(evidence, "MOUSE"))

    def test_level_action_evidence_accumulates_across_objectives(self) -> None:
        agent = self._agent()
        for index in range(12):
            agent._record_level_action_evidence(
                {
                    "objective_id": f"tactical:{1 + index // 4}",
                    "action": "MOUSE",
                    "row": index,
                    "col": 10,
                    "executed": True,
                    "post_action_observed": True,
                    "board_changed": False,
                    "outcome_class": "exact_noop",
                    "loop_detected": True,
                    "cycle_risk": True,
                    "meaningful_progress": False,
                }
            )
        payload = agent._level_action_evidence_payload()

        self.assertEqual(12, payload["MOUSE"]["executed"])
        self.assertEqual(12, payload["MOUSE"]["no_progress"])
        self.assertEqual(12, len(payload["MOUSE"]["distinct_points"]))
        self.assertTrue(payload["MOUSE"]["saturated"])
        agent.close()

    def test_reducer_rejects_mouse_contract_after_family_saturation(self) -> None:
        mouse_reduction = reduction_for(
            "level:1:1", title="Click another MOUSE coordinate"
        )
        agent = self._agent()
        agent._tree = ObjectiveTree.start_game(
            "game-a", level=1, level_action_budget=20
        )
        agent._level_action_evidence = {
            "MOUSE": {
                "executed": 12,
                "no_progress": 12,
                "stable_changes": 0,
                "meaningful_progress": 0,
                "distinct_points": [f"{index},10" for index in range(12)],
            }
        }
        agent.model_client = _ScriptedModelClient([mouse_reduction] * 3)

        with self.assertRaises(OrchestrationFailure) as raised:
            agent._reduce(
                frame(), [], request_deadline=None, should_stop=None
            )

        self.assertEqual("orchestration_reducer_exhausted", raised.exception.category)
        self.assertIn("saturated action family", str(raised.exception))
        agent.close()

    def test_repeated_engine_failures_require_contrastive_recalibration(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=32)
        for index in range(3):
            proposal = ReductionProposal.from_payload(
                reduction_for("level:1:1", title=f"Execution hypothesis {index}")
            )
            tree.apply_proposal(proposal, remaining_level_actions=32 - index)
            tree.record_action()
            tree.fail_active_tactical("no controller-confirmed progress")
        self.assertEqual(3, _failed_engine_progress_since_recalibration(tree))

        engine_proposal = reduction_for(
            "level:1:1", title="Fourth execution hypothesis"
        )
        agent = self._agent()
        agent._tree = tree
        agent.model_client = _ScriptedModelClient([engine_proposal] * 3)
        with self.assertRaises(OrchestrationFailure) as raised:
            agent._reduce(frame(), [], request_deadline=None, should_stop=None)
        self.assertIn("contrastive_transition", str(raised.exception))

        recalibration = reduction_for(
            "level:1:1", title="Contrast ACTION7 against SPACE"
        )
        subgoals = recalibration["subgoals"]
        assert isinstance(subgoals, list)
        subgoals[0]["evidence_mode"] = "contrastive_transition"
        agent.model_client = _ScriptedModelClient([recalibration])
        agent._reduce(frame(), [], request_deadline=None, should_stop=None)
        self.assertEqual(
            "contrastive_transition", agent._tree.active.evidence_mode.value
        )
        agent._tree.fail_active_tactical("contrastive probe was inconclusive")
        self.assertEqual(
            3, _failed_engine_progress_since_recalibration(agent._tree)
        )
        agent._tree.apply_proposal(
            ReductionProposal.from_payload(recalibration),
            remaining_level_actions=agent._tree.remaining_level_actions,
        )
        agent._tree.complete_active_tactical("causal control was falsified")
        self.assertEqual(
            0, _failed_engine_progress_since_recalibration(agent._tree)
        )
        agent.close()

    def test_mouse_learning_contract_requires_contrastive_evidence(self) -> None:
        reduction = reduction_for(
            "level:1:1", title="Click MOUSE targets to learn interaction"
        )
        subgoals = reduction["subgoals"]
        assert isinstance(subgoals, list)
        subgoals[0]["evidence_mode"] = "stable_transition"
        agent = self._agent()
        agent._tree = ObjectiveTree.start_game(
            "game-a", level=1, level_action_budget=20
        )
        agent.model_client = _ScriptedModelClient([reduction] * 3)

        with self.assertRaises(OrchestrationFailure) as raised:
            agent._reduce(frame(), [], request_deadline=None, should_stop=None)

        self.assertIn("repeatability alone", str(raised.exception))
        agent.close()

    def test_raw_reduction_envelope_parses_json_object(self) -> None:
        payload = reduction_for("level:1:1")

        extracted = _reduction_from_message(
            {
                "role": "assistant",
                "content": (f"BEGIN_REDUCTION\n{json.dumps(payload)}\nEND_REDUCTION"),
            }
        )

        self.assertEqual(payload, extracted)

    def test_raw_reduction_accepts_one_bare_object_but_rejects_prose(self) -> None:
        payload = json.dumps(reduction_for("level:1:1"))
        self.assertEqual(
            json.loads(payload),
            _reduction_from_message({"role": "assistant", "content": payload}),
        )
        invalid_messages = [
            {"role": "assistant", "content": payload + " trailing prose"},
            {
                "role": "assistant",
                "content": f"Here it is\nBEGIN_REDUCTION\n{payload}\nEND_REDUCTION",
            },
            {
                "role": "assistant",
                "content": f"BEGIN_REDUCTION\n{payload}\nEND_REDUCTION",
                "tool_calls": [{"function": {"name": "submit_reduction"}}],
            },
        ]

        for message in invalid_messages:
            with self.subTest(message=message), self.assertRaises(ValueError):
                _reduction_from_message(message)

    def test_raw_policy_envelope_preserves_source_exactly(self) -> None:
        source = (
            "POLICY_API_VERSION = 1\n"
            'SUPPORTED_BACKENDS = ("cpu",)\n'
            "def decide(observation, memory):\n"
            '    return {"status": "continue", "action": '
            '{"action": "ACTION1"}, "memory": memory}'
        )

        extracted = _policy_source_from_message(
            {
                "role": "assistant",
                "content": f"BEGIN_POLICY\n{source}\nEND_POLICY",
            }
        )

        self.assertEqual(source, extracted)

    def test_raw_policy_envelope_rejects_ambiguous_responses(self) -> None:
        invalid_messages = [
            {"role": "assistant", "content": POLICY_SOURCE},
            {
                "role": "assistant",
                "content": f"Here is code\nBEGIN_POLICY\n{POLICY_SOURCE}\nEND_POLICY",
            },
            {
                "role": "assistant",
                "content": "BEGIN_POLICY\n\nEND_POLICY",
            },
            {
                "role": "assistant",
                "content": (
                    f"BEGIN_POLICY\n{POLICY_SOURCE}\nEND_POLICY\n"
                    f"BEGIN_POLICY\n{POLICY_SOURCE}\nEND_POLICY"
                ),
            },
            {
                "role": "assistant",
                "content": f"BEGIN_POLICY\n{POLICY_SOURCE}\nEND_POLICY",
                "tool_calls": [{"function": {"name": "submit_policy"}}],
            },
        ]

        for message in invalid_messages:
            with self.subTest(message=message), self.assertRaises(ValueError):
                _policy_source_from_message(message)

    def _agent(self) -> OrchestratedObjectiveAgent:
        config = AnalyzerModelConfig(
            provider="vllm", base_url="http://127.0.0.1:1/v1", model_id="test"
        )
        with (
            mock.patch(
                "inference.agent.tool_agent._resolve_analyzer_model",
                return_value=config,
            ),
            mock.patch("inference.agent.tool_agent.validate_sandbox_isolation"),
        ):
            return OrchestratedObjectiveAgent(model="test", game_id="game-a")

    def test_role_payloads_and_policy_observation_use_model_action_names(
        self,
    ) -> None:
        current_frame = frame(
            valid_actions=("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6")
        )
        history = [HistoryEntry(action="ACTION3", frame=current_frame)]
        agent = self._agent()
        agent._tree = ObjectiveTree.start_game(
            "game-a", level=1, level_action_budget=32
        )
        runtime = _FakeRuntime()
        runtime.activation = PolicyActivation(
            source_hash="test-policy",
            backend="cpu",
            supported_backends=("cpu",),
        )
        agent._policy_runtime = runtime

        reducer_payload = agent._reducer_payload(current_frame, history)
        policy_payload = agent._policy_payload(current_frame, history, "")
        observation = agent._observation(current_frame)
        agent.close()

        expected = ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"]
        self.assertEqual(expected, reducer_payload["observation"]["valid_actions"])
        self.assertEqual(expected, policy_payload["observation"]["valid_actions"])
        self.assertEqual(tuple(expected), observation.valid_actions)
        self.assertEqual("LEFT", reducer_payload["recent_transitions"][0]["action"])
        self.assertEqual("LEFT", policy_payload["recent_transitions"][0]["action"])
        self.assertIn("row and col", reducer_payload["action_contract"]["MOUSE"])
        self.assertEqual(
            {"used": 0, "limit": 32, "remaining": 32},
            reducer_payload["level_action_budget"],
        )
        self.assertEqual(
            reducer_payload["action_contract"], policy_payload["action_contract"]
        )

    def test_exhausted_host_level_budget_stops_without_llm_or_gameplay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = mock.Mock()
            step_env = mock.Mock()
            agent.set_level_action_status(1, 32, 32)
            result = agent.analyze(state_path, 32, step_env=step_env)
            tree = agent._tree
            agent.close()

        self.assertTrue(result.exhausted)
        self.assertEqual(
            "orchestration_level_action_budget_exhausted",
            result.failure_category,
        )
        self.assertIn("32/32", result.failure_detail)
        self.assertIsNotNone(tree)
        self.assertEqual(0, tree.remaining_level_actions)
        agent.model_client.complete.assert_not_called()
        step_env.assert_not_called()

    def test_final_allowed_action_is_capped_then_next_turn_is_llm_free(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            model_client = _FakeModelClient()
            agent.model_client = model_client
            step_env = mock.Mock(
                return_value={"executed": True, "board_changed": True}
            )
            agent.set_level_action_status(1, 31, 32)

            final_action = agent.analyze(state_path, 31, step_env=step_env)
            calls_after_final_action = model_client.calls
            self.assertEqual(1, agent._tree.active.action_budget)
            self.assertEqual(0, agent._tree.remaining_level_actions)

            agent.set_level_action_status(1, 32, 32)
            exhausted = agent.analyze(state_path, 32, step_env=step_env)
            agent.close()

        self.assertTrue(final_action.step_executed)
        self.assertEqual(2, calls_after_final_action)
        self.assertTrue(exhausted.exhausted)
        self.assertEqual(
            "orchestration_level_action_budget_exhausted",
            exhausted.failure_category,
        )
        self.assertEqual(calls_after_final_action, model_client.calls)
        step_env.assert_called_once()

    def test_ordinary_policy_turn_does_not_call_llm(self) -> None:
        current_frame = frame()
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=current_frame, history=[])
            agent = self._agent()
            model_client = _FakeModelClient()
            agent.model_client = model_client
            actions: list[dict] = []

            def step_env(payload: dict) -> dict:
                actions.append(payload)
                return {"executed": True, "board_changed": True}

            first = agent.analyze(
                state_path,
                0,
                valid_actions=["ACTION1"],
                step_env=step_env,
            )
            calls_after_activation = model_client.calls
            second = agent.analyze(
                state_path,
                1,
                valid_actions=["ACTION1"],
                step_env=step_env,
            )
            agent.close()

        self.assertTrue(first.step_executed)
        self.assertTrue(second.step_executed)
        self.assertEqual(2, calls_after_activation)
        self.assertEqual(calls_after_activation, model_client.calls)
        self.assertEqual(2, len(actions))
        self.assertEqual(["UP", "UP"], [item["action"] for item in actions])
        self.assertEqual(
            [4096, 8192],
            [item["max_output_tokens"] for item in model_client.call_kwargs],
        )
        self.assertEqual(
            [2048, 1024],
            [item["thinking_token_budget"] for item in model_client.call_kwargs],
        )
        self.assertEqual(
            [None, None],
            [item["tool_choice"] for item in model_client.call_kwargs],
        )
        self.assertIsNone(model_client.tools[0])
        self.assertIsNone(model_client.tools[1])
        self.assertEqual(
            [1, 1],
            [item["request_attempt_limit"] for item in model_client.call_kwargs],
        )

    def test_repeated_volatile_action_gets_one_same_leaf_repair_then_fails(
        self,
    ) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                policy_for("tactical:1"),
                policy_for("tactical:1"),
            ]
        )
        runtime = _SequentialDecisionRuntime(
            [
                PolicyDecision(
                    status=PolicyStatus.CONTINUE,
                    action={"action": "UP"},
                    memory={"phase": 1},
                    evidence="initial probe",
                ),
                PolicyDecision(
                    status=PolicyStatus.CONTINUE,
                    action={"action": "UP"},
                    memory={"phase": 2},
                    evidence="incorrectly repeat volatile probe",
                ),
                PolicyDecision(
                    status=PolicyStatus.CONTINUE,
                    action={"action": "RIGHT"},
                    memory={"phase": 3},
                    evidence="replacement chooses a distinct probe",
                ),
                PolicyDecision(
                    status=PolicyStatus.CONTINUE,
                    action={"action": "RIGHT"},
                    memory={"phase": 4},
                    evidence="replacement repeats too",
                ),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                return_value=runtime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            first_step = mock.Mock(
                return_value={
                    "executed": True,
                    "board_changed": True,
                    "outcome_class": "volatile_only",
                    "novel_state": False,
                    "level": 1,
                    "state": "NOT_FINISHED",
                }
            )
            first = agent.analyze(state_path, 0, step_env=first_step)
            write_runtime_state(
                state_path,
                current_frame=frame(step=1),
                history=[],
            )
            second_step = mock.Mock(
                return_value={
                    "executed": True,
                    "board_changed": True,
                    "outcome_class": "volatile_only",
                    "novel_state": False,
                    "level": 1,
                    "state": "NOT_FINISHED",
                }
            )
            second = agent.analyze(
                state_path,
                1,
                step_env=second_step,
            )
            write_runtime_state(
                state_path,
                current_frame=frame(step=2),
                history=[],
            )
            third_step = mock.Mock()
            third = agent.analyze(
                state_path,
                2,
                step_env=third_step,
                should_stop=lambda: True,
            )
            tree = agent._tree
            events = [
                json.loads(line)
                for line in (Path(temp_dir) / "orchestration_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            agent.close()

        self.assertTrue(first.step_executed)
        self.assertTrue(second.step_executed)
        self.assertTrue(third.yielded_control)
        first_step.assert_called_once()
        second_step.assert_called_once()
        third_step.assert_not_called()
        assert tree is not None
        self.assertEqual(ObjectiveStatus.FAILED, tree.nodes["tactical:1"].status)
        self.assertEqual(3, client.calls)
        self.assertEqual(
            1,
            len(
                [
                    event
                    for event in events
                    if event["type"] == "policy_non_progress_repeat_repair"
                ]
            ),
        )
        self.assertEqual(
            1,
            len(
                [
                    event
                    for event in events
                    if event["type"] == "policy_non_progress_repeat_rejected"
                ]
            ),
        )

    def test_model_request_timeout_is_independent_from_cooperative_yield(self) -> None:
        clock = {"now": 0.0}
        client = _ScriptedModelClient([reduction_for("level:1:1")])
        original_complete = client.complete

        def delayed_complete(
            messages: list[dict],
            *,
            tools: list[dict] | None,
            **kwargs: object,
        ) -> SimpleNamespace:
            result = original_complete(messages, tools=tools, **kwargs)
            clock["now"] = 61.0
            return result

        client.complete = delayed_complete  # type: ignore[method-assign]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.time.monotonic",
                side_effect=lambda: clock["now"],
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent._yield_seconds = 60
            agent.model_client = client
            result = agent.analyze(
                state_path,
                0,
                step_env=lambda _payload: {"executed": True},
                request_timeout_seconds=900,
            )
            tree = agent._tree
            metrics = dict(agent._orchestration_metrics)
            agent.close()

        self.assertTrue(result.yielded_control)
        self.assertEqual("turn_time_budget", result.yield_reason)
        self.assertEqual(1, client.calls)
        self.assertEqual(300.0, client.call_kwargs[0]["request_timeout_seconds"])
        self.assertEqual(4096, client.call_kwargs[0]["max_output_tokens"])
        self.assertEqual(2048, client.call_kwargs[0]["thinking_token_budget"])
        self.assertIsNone(client.call_kwargs[0]["tool_choice"])
        self.assertEqual(1, client.call_kwargs[0]["request_attempt_limit"])
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual("tactical:1", tree.active_id)
        self.assertEqual(1, metrics["reducer_attempts"])
        self.assertEqual(1, metrics["reducer_calls"])
        self.assertEqual(61.0, metrics["reducer_model_seconds"])

    def test_role_request_timeout_is_capped_by_analyzer_remaining_time(self) -> None:
        agent = self._agent()
        agent._orchestration_request_timeout_seconds = 300
        with mock.patch(
            "inference.agent.orchestrated_objective_agent.time.monotonic",
            return_value=100.0,
        ):
            timeout = agent._role_request_timeout(220.0)
        agent.close()
        self.assertEqual(120.0, timeout)

    def test_orchestration_timeout_and_caps_are_configurable(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS": "240",
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT": "3072",
                "LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT": "6144",
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET": "1536",
                "LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET": "2560",
            },
            clear=False,
        ):
            agent = self._agent()
        self.assertEqual(240.0, agent._orchestration_request_timeout_seconds)
        self.assertEqual(
            {"reducer": 3072, "coder": 6144},
            agent._orchestration_role_max_output,
        )
        self.assertEqual(
            {"reducer": 1536, "coder": 2560},
            agent._orchestration_role_thinking_budget,
        )
        agent.close()

    def test_thinking_budget_is_capped_below_total_output(self) -> None:
        agent = self._agent()
        agent._orchestration_role_thinking_budget["coder"] = 9000
        self.assertEqual(8191, agent._role_thinking_budget("coder", 8192))
        self.assertEqual(2250, agent._role_thinking_budget("coder", 8192, attempt=3))
        agent.close()

    def test_solver_factory_preserves_legacy_and_selects_flagged_agent(self) -> None:
        game = SimpleNamespace(
            game_run=SimpleNamespace(game_id="game-a"),
        )
        solver = HarnessSolver(model="test")
        with (
            mock.patch(
                "inference.framework.solver.objective_reduction_enabled",
                return_value=False,
            ),
            mock.patch("inference.framework.solver.ToolAgent") as legacy,
        ):
            solver._make_analyzer(game, 0)
            legacy.assert_called_once()
        with (
            mock.patch(
                "inference.framework.solver.objective_reduction_enabled",
                return_value=True,
            ),
            mock.patch(
                "inference.framework.solver.OrchestratedObjectiveAgent"
            ) as orchestrated,
        ):
            solver._make_analyzer(game, 0)
            orchestrated.assert_called_once()

    def test_orchestration_failure_is_marked_exhausted(self) -> None:
        current_frame = frame()
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=current_frame, history=[])
            agent = self._agent()
            agent.model_client = mock.Mock()
            agent.model_client.complete.side_effect = RuntimeError(
                "bad structured output"
            )
            result = agent.analyze(
                state_path,
                0,
                valid_actions=["ACTION1"],
                step_env=lambda _payload: {"executed": False},
            )
            agent.close()
        self.assertTrue(result.exhausted)
        self.assertEqual("orchestration_internal", result.failure_category)

    def test_reducer_and_coder_retry_rejected_contracts_then_execute(self) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:wrong:1"),
                reduction_for("level:1:1"),
                policy_for(
                    "tactical:wrong",
                    source='POLICY_API_VERSION = 1\nSUPPORTED_BACKENDS = ("cpu",)\n',
                ),
                policy_for("tactical:1"),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            result = agent.analyze(
                state_path,
                0,
                valid_actions=["ACTION1"],
                step_env=lambda _payload: {"executed": True, "board_changed": True},
            )
            reducer_history = list(agent._role_histories["reducer"])
            coder_history = list(agent._role_histories["coder"])
            metrics = dict(agent._orchestration_metrics)
            agent.close()
        self.assertTrue(result.step_executed)
        self.assertEqual(4, client.calls)
        self.assertIn("Previous validation errors", client.messages[1][-1]["content"])
        self.assertIn("Previous validation errors", client.messages[3][-1]["content"])
        self.assertEqual(2, len(reducer_history))
        self.assertEqual(2, len(coder_history))
        self.assertEqual(2, metrics["reducer_calls"])
        self.assertEqual(2, metrics["coder_calls"])
        self.assertEqual(2, metrics["reducer_attempts"])
        self.assertEqual(2, metrics["coder_attempts"])
        self.assertEqual(14, metrics["reducer_generated_tokens"])
        self.assertEqual(14, metrics["coder_generated_tokens"])

    def test_missing_decide_stays_in_coder_loop_and_saves_rejected_source(
        self,
    ) -> None:
        rejected_source = 'POLICY_API_VERSION = 1\nSUPPORTED_BACKENDS = ("cpu",)'
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                policy_for("tactical:1", source=rejected_source),
                policy_for("tactical:1"),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            result = agent.analyze(
                state_path,
                0,
                valid_actions=["ACTION1"],
                step_env=lambda _payload: {
                    "executed": True,
                    "board_changed": True,
                },
            )
            metrics = dict(agent._orchestration_metrics)
            rejected_artifacts = list(
                (Path(temp_dir) / "policies" / "rejected").glob("*.py")
            )
            rejected_artifact_text = rejected_artifacts[0].read_text(encoding="utf-8")
            events = [
                json.loads(line)
                for line in (Path(temp_dir) / "orchestration_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            agent.close()

        self.assertTrue(result.step_executed)
        self.assertEqual(3, client.calls)
        self.assertEqual(1, metrics["reducer_calls"])
        self.assertEqual(2, metrics["coder_calls"])
        self.assertIn("must define decide", client.messages[2][-1]["content"])
        self.assertIn("<REJECTED_POLICY_SOURCE>", client.messages[2][-1]["content"])
        self.assertIn(rejected_source, client.messages[2][-1]["content"])
        self.assertIn("every item must be fixed", client.messages[2][-1]["content"])
        coder_prompt = client.messages[1][0]["content"]
        self.assertIn("PolicyObservation is an immutable object", coder_prompt)
        self.assertIn("observation.valid_actions[0]", coder_prompt)
        self.assertIn("Never call observation.get()", coder_prompt)
        self.assertIn("shortest_path(passable, start, goal", coder_prompt)
        self.assertIn("next_path_action(path, observation.valid_actions)", coder_prompt)
        self.assertIn("find_cells(observation.board, VALUES", coder_prompt)
        self.assertIn("distance_map(passable, starts", coder_prompt)
        self.assertIn("weighted_shortest_path(passable, costs", coder_prompt)
        self.assertIn("clearance_mask(passable, radius=1)", coder_prompt)
        self.assertIn("line_of_sight(passable, start, goal)", coder_prompt)
        self.assertIn("component_boxes(mask, min_size=1)", coder_prompt)
        self.assertIn("shortest_approach_path(passable, start, targets", coder_prompt)
        self.assertIn("path_is_valid(passable, suffix)", coder_prompt)
        self.assertIn("continue_decision(", coder_prompt)
        self.assertIn("transition_outcome(last_transition)", coder_prompt)
        self.assertIn("transition_repeats_nonprogress_action", coder_prompt)
        self.assertIn("board_digest(observation.board)", coder_prompt)
        self.assertIn("region_digest(board, (top,left,bottom,right))", coder_prompt)
        self.assertIn("memory_with_defaults(memory, defaults)", coder_prompt)
        self.assertIn("accumulate_transition_evidence(memory, last_transition", coder_prompt)
        self.assertIn("memory_push(memory, key, value", coder_prompt)
        self.assertIn("least_tried_action(observation.valid_actions", coder_prompt)
        self.assertIn("least_tried_mouse_point(candidates", coder_prompt)
        self.assertIn("first_matching_cell(board, values)", coder_prompt)
        self.assertIn("line_value_count(board, values, axis, index)", coder_prompt)
        self.assertEqual(1, len(rejected_artifacts))
        self.assertEqual(rejected_source, rejected_artifact_text)
        rejected_events = [
            event for event in events if event["type"] == "policy_candidate_rejected"
        ]
        self.assertEqual(1, len(rejected_events))
        self.assertEqual("raw_content_validation", rejected_events[0]["phase"])
        self.assertEqual(1, rejected_events[0]["structured_attempt"])
        self.assertNotIn("source", rejected_events[0])

    def test_coder_repair_prompt_accumulates_validation_errors(self) -> None:
        missing_decide = 'POLICY_API_VERSION = 1\nSUPPORTED_BACKENDS = ("cpu",)'
        forbidden_getattr = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    valid = getattr(observation, "valid_actions", ())
    return {"status": "continue", "action": {"action": valid[0]}, "memory": memory}
"""
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                policy_for("tactical:1", source=missing_decide),
                policy_for("tactical:1", source=forbidden_getattr),
                policy_for("tactical:1"),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            result = agent.analyze(
                state_path,
                0,
                step_env=lambda _payload: {
                    "executed": True,
                    "board_changed": True,
                    "outcome_class": "novel",
                    "novel_state": True,
                },
            )
            agent.close()

        self.assertTrue(result.step_executed)
        self.assertEqual(4, client.calls)
        final_repair_prompt = client.messages[3][-1]["content"]
        self.assertIn("must define decide", final_repair_prompt)
        self.assertIn("call 'getattr' is not permitted", final_repair_prompt)
        self.assertIn(forbidden_getattr.strip(), final_repair_prompt)

    def test_activation_failure_repairs_with_coder_without_reducing_again(
        self,
    ) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                policy_for("tactical:1"),
                policy_for("tactical:1"),
            ]
        )
        runtime_factory = _ActivationRepairRuntimeFactory()
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                runtime_factory,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            result = agent.analyze(
                state_path,
                0,
                valid_actions=["ACTION1"],
                step_env=lambda _payload: {
                    "executed": True,
                    "board_changed": True,
                },
            )
            metrics = dict(agent._orchestration_metrics)
            events = [
                json.loads(line)
                for line in (Path(temp_dir) / "orchestration_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            rejected_artifacts = list(
                (Path(temp_dir) / "policies" / "rejected").glob("*.py")
            )
            rejected_artifact_text = rejected_artifacts[0].read_text(encoding="utf-8")
            agent.close()

        self.assertTrue(result.step_executed)
        self.assertEqual(3, client.calls)
        self.assertEqual(2, runtime_factory.calls)
        self.assertEqual(1, metrics["reducer_calls"])
        self.assertEqual(2, metrics["coder_calls"])
        self.assertEqual(1, metrics["policy_repairs"])
        policy_failures = [
            event for event in events if event["type"] == "policy_failed"
        ]
        self.assertEqual(1, len(policy_failures))
        self.assertEqual("coder", policy_failures[0]["repair_route"])
        rejected_events = [
            event for event in events if event["type"] == "policy_candidate_rejected"
        ]
        self.assertEqual(1, len(rejected_events))
        self.assertEqual("activation", rejected_events[0]["phase"])
        self.assertEqual(1, len(rejected_artifacts))
        self.assertEqual(
            POLICY_SOURCE.strip(),
            rejected_artifact_text.strip(),
        )

    def test_reducer_exhaustion_is_visible_after_three_attempts(self) -> None:
        client = _ScriptedModelClient(
            [reduction_for("level:wrong:1") for _ in range(3)]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            result = agent.analyze(
                state_path,
                0,
                step_env=lambda _payload: {"executed": False},
            )
            agent.close()
        self.assertTrue(result.exhausted)
        self.assertEqual("orchestration_reducer_exhausted", result.failure_category)
        self.assertEqual(3, client.calls)

    def test_host_objective_repair_explicitly_requires_decomposition(self) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1", verdict="continue"),
                reduction_for("level:1:1"),
                policy_for("tactical:1"),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            result = agent.analyze(
                state_path,
                0,
                step_env=lambda _payload: {"executed": True, "board_changed": True},
            )
            agent.close()
        self.assertTrue(result.step_executed)
        repair_prompt = client.messages[1][-1]["content"]
        self.assertIn('use verdict="decompose"', repair_prompt)
        self.assertIn("Do not return continue, complete, or fail", repair_prompt)

    def test_coder_exhaustion_is_visible_after_three_attempts(self) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                *["POLICY_API_VERSION = 1" for _ in range(3)],
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            result = agent.analyze(
                state_path,
                0,
                step_env=lambda _payload: {"executed": False},
            )
            agent.close()
        self.assertTrue(result.exhausted)
        self.assertEqual("orchestration_coder_exhausted", result.failure_category)
        self.assertEqual(4, client.calls)

    def test_stop_request_yields_without_calling_model_or_controller(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = mock.Mock()
            step_env = mock.Mock()
            result = agent.analyze(
                state_path,
                0,
                step_env=step_env,
                should_stop=lambda: True,
            )
            agent.close()
        self.assertTrue(result.yielded_control)
        self.assertEqual("turn_time_budget", result.yield_reason)
        agent.model_client.complete.assert_not_called()
        step_env.assert_not_called()

    def test_transport_failure_exhausts_after_three_structured_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = mock.Mock()
            agent.model_client.complete.side_effect = requests.ReadTimeout(
                "generation exceeded role deadline"
            )
            result = agent.analyze(
                state_path,
                0,
                step_env=lambda _payload: {"executed": False},
            )
            metrics = dict(agent._orchestration_metrics)
            agent.close()
        self.assertFalse(result.retryable_failure)
        self.assertTrue(result.exhausted)
        self.assertEqual(
            "orchestration_reducer_transport_exhausted", result.failure_category
        )
        self.assertEqual(3, agent.model_client.complete.call_count)
        self.assertEqual(3, metrics["reducer_attempts"])
        self.assertEqual(3, metrics["reducer_timeouts"])
        self.assertEqual(3, metrics["reducer_transport_failures"])
        self.assertEqual(0, metrics["reducer_calls"])

    def test_context_error_compacts_history_and_reduces_output_before_retry(self) -> None:
        context_error = requests.HTTPError(
            "400 Client Error: This model's maximum context length is 65536 tokens. "
            "However, you requested 4096 output tokens and your prompt contains at "
            "least 61441 input tokens, for a total of at least 65537 tokens. Please "
            "reduce the length of the input prompt. (parameter=input_tokens)"
        )
        client = _ScriptedModelClient(
            [
                context_error,
                reduction_for("level:1:1"),
                policy_for("tactical:1"),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            agent._ensure_session(state_path)
            agent._role_histories["reducer"] = [
                {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * 100}
                for index in range(6)
            ]
            result = agent.analyze(
                state_path,
                0,
                step_env=lambda _payload: {
                    "executed": True,
                    "board_changed": False,
                    "meaningful_progress": False,
                    "outcome_class": "exact_noop",
                    "state": "PLAYING",
                    "level": 1,
                },
            )
            metrics = dict(agent._orchestration_metrics)
            agent.close()

        self.assertTrue(result.step_executed)
        self.assertEqual(3, client.calls)
        self.assertEqual(8, len(client.messages[0]))
        self.assertEqual(4, len(client.messages[1]))
        self.assertEqual(4096, client.call_kwargs[0]["max_output_tokens"])
        self.assertEqual(3839, client.call_kwargs[1]["max_output_tokens"])
        self.assertEqual(1, metrics["role_context_adjustments"])
        self.assertEqual(1, metrics["reducer_transport_failures"])
        self.assertEqual(1, metrics["reducer_calls"])

    def test_engine_terminal_states_resolve_host_game_without_llm(self) -> None:
        for engine_state, expected_status in (
            ("WIN", ObjectiveStatus.COMPLETED),
            ("GAME_OVER", ObjectiveStatus.FAILED),
        ):
            with self.subTest(engine_state=engine_state):
                with tempfile.TemporaryDirectory() as temp_dir:
                    state_path = Path(temp_dir) / "runtime_state.json"
                    write_runtime_state(
                        state_path,
                        current_frame=frame(engine_state=engine_state),
                        history=[],
                    )
                    agent = self._agent()
                    agent.model_client = mock.Mock()
                    step_env = mock.Mock()
                    result = agent.analyze(state_path, 0, step_env=step_env)
                    tree = agent._tree
                    agent.close()
                self.assertFalse(result.step_executed)
                self.assertFalse(result.exhausted)
                self.assertIsNotNone(tree)
                assert tree is not None
                self.assertEqual(expected_status, tree.nodes[tree.root_id].status)
                agent.model_client.complete.assert_not_called()
                step_env.assert_not_called()

    def test_missing_state_and_step_callback_fail_without_fallback(self) -> None:
        agent = self._agent()
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = agent.analyze(
                Path(temp_dir) / "missing.json",
                0,
                step_env=lambda _payload: {},
            )
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            missing_callback = agent.analyze(state_path, 0, step_env=None)
        agent.close()
        self.assertEqual("state_missing", missing.failure_category)
        self.assertEqual(
            "orchestration_missing_step_env", missing_callback.failure_category
        )
        self.assertTrue(agent.disable_controller_fallback)

    def test_guarded_action_rejection_reduces_repairs_and_retries_in_turn(self) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                policy_for("tactical:1"),
                policy_for("tactical:1"),
            ]
        )
        results = iter(
            [
                {"executed": False, "stop_reason": "cycle guard"},
                {"executed": True, "board_changed": True},
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            step_env = mock.Mock(side_effect=lambda _payload: next(results))
            result = agent.analyze(state_path, 0, step_env=step_env)
            repairs = dict(agent._policy_repairs)
            metrics = dict(agent._orchestration_metrics)
            events = [
                json.loads(line)
                for line in (Path(temp_dir) / "orchestration_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            agent.close()
        self.assertTrue(result.step_executed)
        self.assertEqual(2, step_env.call_count)
        self.assertEqual(3, client.calls)
        self.assertEqual(1, repairs["tactical:1"])
        self.assertEqual(1, metrics["policy_repairs"])
        self.assertEqual(1, metrics["policy_steps"])
        policy_failures = [
            event for event in events if event["type"] == "policy_failed"
        ]
        self.assertEqual("coder", policy_failures[0]["repair_route"])

    def test_mature_loop_guard_falsifies_objective_without_coder_repair(self) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        for _ in range(4):
            tree.record_action()
        tactical = tree.active
        agent._tree = tree
        transition = {
            "objective_id": tactical.objective_id,
            "action": "RIGHT",
            "executed": False,
            "stop_reason": "loop_guard",
            "loop_detected": True,
        }
        agent._last_transition = transition

        resolved = agent._resolve_loop_guard_as_tactical_failure(transition)

        self.assertTrue(resolved)
        self.assertEqual(ObjectiveStatus.FAILED, tactical.status)
        self.assertEqual("level:1:1", tree.active_id)
        self.assertTrue(agent._reduction_required)
        self.assertEqual({}, agent._policy_repairs)
        self.assertEqual(1, agent._orchestration_metrics["objectives_failed"])
        self.assertEqual(
            1, agent._orchestration_metrics["guard_resolved_objectives"]
        )
        agent.close()

    def test_early_loop_guard_keeps_existing_coder_repair_route(self) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        tree.record_action()
        tactical = tree.active
        agent._tree = tree
        transition = {
            "objective_id": tactical.objective_id,
            "action": "RIGHT",
            "executed": False,
            "stop_reason": "loop_guard",
            "loop_detected": True,
        }

        resolved = agent._resolve_loop_guard_as_tactical_failure(transition)

        self.assertFalse(resolved)
        self.assertEqual(ObjectiveStatus.ACTIVE, tactical.status)
        self.assertEqual(0, agent._orchestration_metrics["guard_resolved_objectives"])
        agent.close()

    def test_mature_loop_guard_analyzer_path_makes_no_additional_model_call(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            client = _FakeModelClient()
            agent.model_client = client
            action_results = [
                {
                    "executed": True,
                    "board_changed": True,
                    "outcome_class": "novel",
                    "state": "PLAYING",
                    "level": 1,
                }
                for _ in range(4)
            ]
            action_results.append(
                {
                    "executed": False,
                    "board_changed": False,
                    "stop_reason": "loop_guard",
                    "loop_detected": True,
                    "state": "PLAYING",
                    "level": 1,
                }
            )
            step_env = mock.Mock(side_effect=action_results)
            for step in range(4):
                result = agent.analyze(
                    state_path,
                    step,
                    step_env=step_env,
                    should_stop=lambda: step_env.call_count >= 5,
                )
                self.assertTrue(result.step_executed)
                write_runtime_state(
                    state_path,
                    current_frame=frame(step=step + 1),
                    history=[],
                )
            guarded = agent.analyze(
                state_path,
                4,
                step_env=step_env,
                should_stop=lambda: step_env.call_count >= 5,
            )
            tactical = agent._tree.nodes["tactical:1"]
            metrics = dict(agent._orchestration_metrics)
            repairs = dict(agent._policy_repairs)
            agent.close()

        self.assertTrue(guarded.yielded_control)
        self.assertEqual(5, step_env.call_count)
        self.assertEqual(2, client.calls)
        self.assertEqual(ObjectiveStatus.FAILED, tactical.status)
        self.assertEqual({}, repairs)
        self.assertEqual(1, metrics["guard_resolved_objectives"])

    def test_premature_subgoal_completion_repairs_policy_without_reducing(self) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                policy_for("tactical:1"),
                policy_for("tactical:1"),
            ]
        )
        runtime_factory = _RuntimeFactory(
            [
                PolicyDecision(
                    status=PolicyStatus.SUBGOAL_SUCCEEDED,
                    action=None,
                    memory={"phase": 1},
                    evidence="first probe succeeded",
                ),
                PolicyDecision(
                    status=PolicyStatus.CONTINUE,
                    action={"action": "ACTION1"},
                    memory={"phase": 2},
                    evidence="second probe action",
                ),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                runtime_factory,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            step_env = mock.Mock(return_value={"executed": True, "board_changed": True})
            result = agent.analyze(state_path, 0, step_env=step_env)
            tree = agent._tree
            metrics = dict(agent._orchestration_metrics)
            agent.close()
        self.assertTrue(result.step_executed)
        self.assertEqual(1, step_env.call_count)
        self.assertEqual(3, client.calls)
        assert tree is not None
        self.assertEqual(ObjectiveStatus.ACTIVE, tree.nodes["tactical:1"].status)
        self.assertEqual("tactical:1", tree.active_id)
        self.assertEqual(0, metrics["objectives_completed"])
        self.assertEqual(1, metrics["objective_completion_rejections"])
        self.assertEqual(1, metrics["policy_repairs"])

    def test_exhausted_action_budget_allows_one_terminal_policy_evaluation(
        self,
    ) -> None:
        reduction = reduction_for("level:1:1")
        subgoals = reduction["subgoals"]
        assert isinstance(subgoals, list)
        subgoals[0]["action_budget"] = 1
        subgoals[0]["minimum_evidence_actions"] = 1
        subgoals[0]["single_step"] = True
        client = _ScriptedModelClient([reduction, policy_for("tactical:1")])
        runtime = _SequentialDecisionRuntime(
            [
                PolicyDecision(
                    status=PolicyStatus.CONTINUE,
                    action={"action": "ACTION1"},
                    memory={"phase": "acted"},
                    evidence="perform the single probe",
                ),
                PolicyDecision(
                    status=PolicyStatus.SUBGOAL_SUCCEEDED,
                    action=None,
                    memory={"phase": "observed"},
                    evidence="post-action observation resolved the probe",
                ),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                return_value=runtime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            agent = self._agent()
            agent.model_client = client
            first_step = mock.Mock(
                return_value={
                    "executed": True,
                    "board_changed": True,
                    "meaningful_progress": True,
                }
            )
            first = agent.analyze(state_path, 0, step_env=first_step)
            write_runtime_state(
                state_path,
                current_frame=frame(step=1),
                history=[],
            )
            second_step = mock.Mock()
            second = agent.analyze(
                state_path,
                1,
                step_env=second_step,
                should_stop=lambda: True,
            )
            tree = agent._tree
            metrics = dict(agent._orchestration_metrics)
            agent.close()

        self.assertTrue(first.step_executed)
        self.assertTrue(second.yielded_control)
        self.assertEqual(2, runtime.decide_calls)
        second_step.assert_not_called()
        self.assertEqual(2, client.calls)
        assert tree is not None
        self.assertEqual(ObjectiveStatus.COMPLETED, tree.nodes["tactical:1"].status)
        self.assertEqual(1, metrics["objectives_completed"])

    def test_completion_gate_requires_persistence_and_rejects_volatile_evidence(
        self,
    ) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        agent._tree = tree
        transition = {
            "objective_id": "tactical:1",
            "executed": True,
            "outcome_class": "novel",
            "level_completed": False,
            "run_complete": False,
            "meaningful_progress": False,
        }
        tree.record_action()
        agent._last_transition = dict(transition)
        agent._recent_transitions = [dict(transition)]
        allowed, reason = agent._tactical_completion_evidence()
        self.assertFalse(allowed)
        self.assertIn("1 of 4", reason)

        for _ in range(3):
            tree.record_action()
            agent._recent_transitions.append(dict(transition))
        allowed, reason = agent._tactical_completion_evidence()
        self.assertFalse(allowed)
        self.assertIn("novel or changed states alone", reason)

        agent._last_transition = {**transition, "meaningful_progress": True}
        agent._recent_transitions[-1] = dict(agent._last_transition)
        allowed, _ = agent._tactical_completion_evidence()
        self.assertTrue(allowed)

        agent._last_transition = {**transition, "outcome_class": "volatile_only"}
        agent._recent_transitions[-1] = dict(agent._last_transition)
        allowed, reason = agent._tactical_completion_evidence()
        self.assertFalse(allowed)
        self.assertIn("volatile_only", reason)

        agent._last_transition = {**transition, "outcome_class": "transient_effect"}
        agent._recent_transitions[-1] = dict(agent._last_transition)
        allowed, reason = agent._tactical_completion_evidence()
        self.assertFalse(allowed)
        self.assertIn("transient_effect", reason)
        agent.close()

    def test_stable_transition_mode_accepts_reproducible_novel_control_evidence(
        self,
    ) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        reduction = reduction_for("level:1:1")
        subgoals = reduction["subgoals"]
        assert isinstance(subgoals, list)
        subgoals[0]["evidence_mode"] = "stable_transition"
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction),
            remaining_level_actions=20,
        )
        agent._tree = tree
        transitions: list[dict[str, object]] = []
        for action in ("UP", "RIGHT", "UP", "RIGHT"):
            tree.record_action()
            transition: dict[str, object] = {
                "objective_id": "tactical:1",
                "action": action,
                "row": None,
                "col": None,
                "executed": True,
                "post_action_observed": True,
                "board_changed": True,
                "outcome_class": "novel",
                "novel_state": True,
                "meaningful_progress": False,
                "loop_detected": False,
                "cycle_risk": False,
                "level_completed": False,
                "run_complete": False,
            }
            transitions.append(transition)
        agent._last_transition = dict(transitions[-1])
        agent._recent_transitions = [dict(item) for item in transitions]

        allowed, reason = agent._tactical_completion_evidence()

        self.assertTrue(allowed, reason)
        self.assertIn("reproducible stable-transition", reason)

        agent._last_transition = {
            **agent._last_transition,
            "outcome_class": "volatile_only",
        }
        agent._recent_transitions[-1] = dict(agent._last_transition)
        allowed, reason = agent._tactical_completion_evidence()
        self.assertFalse(allowed)
        self.assertIn("volatile_only", reason)
        agent.close()

    def test_stable_transition_mode_rejects_unreproduced_novel_changes(self) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        reduction = reduction_for("level:1:1")
        subgoals = reduction["subgoals"]
        assert isinstance(subgoals, list)
        subgoals[0]["evidence_mode"] = "stable_transition"
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction),
            remaining_level_actions=20,
        )
        agent._tree = tree
        for action in ("UP", "RIGHT", "DOWN", "LEFT"):
            tree.record_action()
            transition = {
                "objective_id": "tactical:1",
                "action": action,
                "executed": True,
                "post_action_observed": True,
                "board_changed": True,
                "outcome_class": "novel",
                "meaningful_progress": False,
            }
            agent._recent_transitions.append(transition)
            agent._last_transition = transition

        allowed, reason = agent._tactical_completion_evidence()

        self.assertFalse(allowed)
        self.assertIn("not reproduced", reason)
        agent.close()

    def test_contrastive_mode_rejects_repeatability_without_negative_control(
        self,
    ) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        reduction = reduction_for("level:1:1")
        subgoals = reduction["subgoals"]
        assert isinstance(subgoals, list)
        subgoals[0]["evidence_mode"] = "contrastive_transition"
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction),
            remaining_level_actions=20,
        )
        agent._tree = tree

        def transition(action: str, *, changed: bool) -> dict[str, object]:
            return {
                "objective_id": "tactical:1",
                "action": action,
                "executed": True,
                "post_action_observed": True,
                "board_changed": changed,
                "outcome_class": "novel" if changed else "exact_noop",
                "meaningful_progress": False,
                "loop_detected": not changed,
                "cycle_risk": not changed,
            }

        for _ in range(4):
            tree.record_action()
            item = transition("LEFT", changed=True)
            agent._recent_transitions.append(item)
            agent._last_transition = item
        allowed, reason = agent._tactical_completion_evidence()
        self.assertFalse(allowed)
        self.assertIn("negative-control", reason)

        agent._recent_transitions = []
        for action, changed in (
            ("LEFT", True),
            ("RIGHT", False),
            ("LEFT", True),
            ("RIGHT", False),
        ):
            item = transition(action, changed=changed)
            agent._recent_transitions.append(item)
            agent._last_transition = item
        allowed, reason = agent._tactical_completion_evidence()
        self.assertTrue(allowed, reason)
        self.assertIn("same-family negative-control", reason)
        agent.close()

    def test_policy_observation_filters_transitions_from_previous_objective(
        self,
    ) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        tree.fail_active_tactical("replace first probe")
        tree.apply_proposal(
            ReductionProposal.from_payload(
                reduction_for("level:1:1", title="Second probe")
            ),
            remaining_level_actions=20,
        )
        runtime = _FakeRuntime()
        runtime.activate(POLICY_SOURCE, context={})
        agent._tree = tree
        agent._policy_runtime = runtime
        agent._policy_objective_id = "tactical:2"
        previous = {
            "objective_id": "tactical:1",
            "executed": True,
            "outcome_class": "novel",
        }
        current = {
            "objective_id": "tactical:2",
            "executed": True,
            "outcome_class": "novel",
        }
        agent._last_transition = previous
        agent._recent_transitions = [previous]
        scoped = agent._observation(frame())
        self.assertIsNone(scoped.last_transition)
        self.assertEqual((), scoped.recent_transitions)

        agent._last_transition = current
        agent._recent_transitions.append(current)
        scoped = agent._observation(frame(step=1))
        self.assertEqual(current, scoped.last_transition)
        self.assertEqual((current,), scoped.recent_transitions)
        agent.close()

    def test_nonprogress_success_is_failed_without_coder_repair(self) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        agent._tree = tree
        agent._last_transition = {
            "objective_id": "tactical:1",
            "executed": True,
            "outcome_class": "exact_noop",
        }
        agent._adjudicate_rejected_completion(
            policy_evidence="incorrectly claimed success",
            completion_reason="latest transition is non-progress outcome exact_noop",
        )
        self.assertEqual(ObjectiveStatus.FAILED, tree.nodes["tactical:1"].status)
        self.assertTrue(agent._reduction_required)
        self.assertEqual({}, agent._policy_repairs)
        self.assertEqual(0, agent._consecutive_activation_failures)
        self.assertEqual(
            1,
            agent._orchestration_metrics["objective_completion_reinterpretations"],
        )
        agent.close()

    def test_repeated_early_positive_success_fails_leaf_not_game(self) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        agent._tree = tree
        agent._last_transition = {
            "objective_id": "tactical:1",
            "executed": True,
            "outcome_class": "novel",
        }
        agent._adjudicate_rejected_completion(
            policy_evidence="one novel transition",
            completion_reason="only 1 of 4 required exploratory actions",
        )
        self.assertEqual(ObjectiveStatus.ACTIVE, tree.nodes["tactical:1"].status)
        self.assertEqual(1, agent._policy_repairs["tactical:1"])
        self.assertEqual(0, agent._consecutive_activation_failures)

        agent._adjudicate_rejected_completion(
            policy_evidence="same early claim",
            completion_reason="only 1 of 4 required exploratory actions",
        )
        self.assertEqual(ObjectiveStatus.FAILED, tree.nodes["tactical:1"].status)
        self.assertTrue(agent._reduction_required)
        self.assertEqual(0, agent._consecutive_activation_failures)
        agent.close()

    def test_premature_failure_gets_one_repair_then_fails_leaf(self) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        agent._tree = tree
        tree.record_action()
        agent._last_transition = {
            "objective_id": "tactical:1",
            "executed": True,
            "outcome_class": "exact_noop",
            "game_over": False,
        }
        allowed, reason = agent._tactical_failure_evidence()
        self.assertFalse(allowed)
        self.assertIn("1 of 4", reason)
        self.assertTrue(
            agent._repair_premature_failure(
                policy_evidence="one probe failed", failure_reason=reason
            )
        )
        self.assertEqual(0, agent._consecutive_activation_failures)
        self.assertFalse(
            agent._repair_premature_failure(
                policy_evidence="same early failure", failure_reason=reason
            )
        )
        agent.close()

    def test_policy_artifact_and_memory_resume_without_llm(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch(
                "inference.agent.orchestrated_objective_agent.GameplayPolicyRuntime",
                _FakeRuntime,
            ),
        ):
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            first_agent = self._agent()
            first_client = _FakeModelClient()
            first_agent.model_client = first_client
            first = first_agent.analyze(
                state_path,
                0,
                step_env=lambda _payload: {"executed": True, "board_changed": True},
            )
            first_agent.close()

            durable_path = Path(temp_dir) / "runtime_state_objective_state.json"
            durable_text = durable_path.read_text(encoding="utf-8")
            durable = json.loads(durable_text)
            artifact = Path(temp_dir) / durable["policy_artifact"]
            artifact_exists = artifact.is_file()
            artifact_text = artifact.read_text(encoding="utf-8")

            resumed_agent = self._agent()
            resumed_agent.model_client = mock.Mock()
            resumed = resumed_agent.analyze(
                state_path,
                1,
                step_env=lambda _payload: {"executed": True, "board_changed": True},
            )
            resumed_memory = resumed_agent._policy_memory
            resumed_agent.close()
        self.assertTrue(first.step_executed)
        self.assertTrue(resumed.step_executed)
        self.assertEqual(2, first_client.calls)
        resumed_agent.model_client.complete.assert_not_called()
        self.assertTrue(artifact_exists)
        self.assertEqual(POLICY_SOURCE.strip(), artifact_text.strip())
        self.assertNotIn(POLICY_SOURCE.strip(), durable_text)
        self.assertEqual({}, resumed_memory)

    def test_cross_game_state_and_policy_path_escape_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "runtime_state.json"
            write_runtime_state(state_path, current_frame=frame(), history=[])
            durable_path = Path(temp_dir) / "runtime_state_objective_state.json"
            durable_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "game_id": "another-game",
                        "objective_tree": ObjectiveTree.start_game(
                            "another-game", level=1, level_action_budget=20
                        ).to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            agent = self._agent()
            agent._ensure_session(state_path)
            self.assertIsNone(agent._tree)

            tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
            tree.apply_proposal(
                ReductionProposal.from_payload(reduction_for("level:1:1")),
                remaining_level_actions=20,
            )
            agent._tree = tree
            agent._policy_objective_id = tree.active_id
            agent._policy_artifact = "../escaped-policy.py"
            self.assertFalse(agent._restore_policy_if_possible())
            agent.close()

    def test_initial_policy_and_two_repairs_are_allowed_before_exhaustion(self) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(
                {
                    "objective_id": "level:1:1",
                    "verdict": "decompose",
                    "evidence": "initial",
                    "rationale": "probe",
                    "selected_index": 0,
                    "subgoals": [
                        {
                            "title": "Probe",
                            "success_criteria": "change",
                            "failure_criteria": "no change",
                            "expected_evidence": "transition",
                            "action_budget": 4,
                        }
                    ],
                }
            ),
            remaining_level_actions=20,
        )
        agent._tree = tree
        failure = PolicyRuntimeError("broken", category="policy_runtime")
        agent._policy_failure(failure)
        self.assertFalse(agent._reduction_required)
        agent._policy_failure(failure)
        self.assertFalse(agent._reduction_required)
        with self.assertRaisesRegex(
            RuntimeError, "initial policy and two replacements"
        ):
            agent._policy_failure(failure)
        self.assertTrue(agent._reduction_required)
        agent.close()

    def test_post_action_failures_exhaust_leaf_without_activation_exhaustion(
        self,
    ) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        agent._tree = tree
        failure = PolicyRuntimeError(
            "bad post-action decision", category="policy_runtime"
        )
        for _ in range(3):
            agent._policy_failure(failure, counts_activation_failure=False)
        self.assertEqual(ObjectiveStatus.FAILED, tree.nodes["tactical:1"].status)
        self.assertEqual(0, agent._consecutive_activation_failures)
        self.assertTrue(agent._reduction_required)
        agent.close()

    def test_policy_failure_streak_resets_for_a_new_tactical_leaf(self) -> None:
        agent = self._agent()
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_for("level:1:1")),
            remaining_level_actions=20,
        )
        agent._tree = tree
        failure = PolicyRuntimeError("broken", category="policy_runtime")
        agent._policy_failure(failure)
        agent._policy_failure(failure)
        tree.fail_active_tactical("replace the tactical leaf")
        tree.apply_proposal(
            ReductionProposal.from_payload(
                reduction_for("level:1:1", title="Replacement probe")
            ),
            remaining_level_actions=20,
        )

        agent._policy_failure(failure)

        self.assertEqual("tactical:2", tree.active_id)
        self.assertEqual("tactical:2", agent._failure_streak_objective_id)
        self.assertEqual(1, agent._consecutive_activation_failures)
        agent.close()


if __name__ == "__main__":
    unittest.main()
