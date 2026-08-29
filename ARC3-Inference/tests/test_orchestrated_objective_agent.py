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
from inference.agent.orchestrated_objective_agent import OrchestratedObjectiveAgent
from inference.agent.objective_reduction import (
    ObjectiveStatus,
    ObjectiveTree,
    ReductionProposal,
)
from inference.agent.runtime_state import Frame, write_runtime_state
from inference.agent.tool_agent import AnalyzerModelConfig
from inference.framework.solver import HarnessSolver


POLICY_SOURCE = """
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
def decide(observation, memory):
    return {"status": "continue", "action": {"action": "ACTION1"}, "memory": memory}
"""


def frame(*, level: int = 1, step: int = 0, engine_state: str = "PLAYING") -> Frame:
    grid = tuple(tuple(0 for _ in range(64)) for _ in range(64))
    return Frame(
        grid=grid,
        step=step,
        level=level,
        valid_actions=("ACTION1",),
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

    def complete(
        self, _messages: list[dict], *, tools: list[dict], **_kwargs: object
    ) -> SimpleNamespace:
        self.calls += 1
        self.call_kwargs.append(dict(_kwargs))
        name = tools[0]["function"]["name"]
        if name == "submit_reduction":
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
                    }
                ],
            }
        else:
            payload = {
                "objective_id": "tactical:1",
                "source": POLICY_SOURCE,
                "backend_capabilities": ["cpu"],
                "self_test_notes": "deterministic",
            }
        return SimpleNamespace(
            message={
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call-{self.calls}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(payload)},
                    }
                ],
            },
            usage={"completion_tokens": 10, "total_tokens": 20},
            request_attempts=1,
        )


class _ScriptedModelClient:
    def __init__(self, responses: list[dict[str, object] | BaseException]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.messages: list[list[dict]] = []
        self.call_kwargs: list[dict[str, object]] = []

    def complete(
        self, messages: list[dict], *, tools: list[dict], **_kwargs: object
    ) -> SimpleNamespace:
        self.calls += 1
        self.messages.append(messages)
        self.call_kwargs.append(dict(_kwargs))
        if not self.responses:
            raise AssertionError("scripted model response queue was exhausted")
        payload = self.responses.pop(0)
        if isinstance(payload, BaseException):
            raise payload
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
            action={"action": "ACTION1"},
            memory=self.memory,
            evidence="ordinary CPU policy step",
        )

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


class OrchestratedObjectiveAgentTests(unittest.TestCase):
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
        self.assertEqual(
            [4096, 8192],
            [item["max_output_tokens"] for item in model_client.call_kwargs],
        )
        self.assertEqual(
            [2048, 3072],
            [item["thinking_token_budget"] for item in model_client.call_kwargs],
        )
        self.assertEqual(
            ["required", "required"],
            [item["tool_choice"] for item in model_client.call_kwargs],
        )
        self.assertEqual(
            [1, 1],
            [item["request_attempt_limit"] for item in model_client.call_kwargs],
        )

    def test_model_request_timeout_is_independent_from_cooperative_yield(self) -> None:
        clock = {"now": 0.0}
        client = _ScriptedModelClient([reduction_for("level:1:1")])
        original_complete = client.complete

        def delayed_complete(
            messages: list[dict], *, tools: list[dict], **kwargs: object
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
        self.assertEqual("required", client.call_kwargs[0]["tool_choice"])
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
                policy_for("tactical:wrong"),
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
        self.assertIn(
            "Previous response was rejected", client.messages[1][-1]["content"]
        )
        self.assertIn(
            "Previous response was rejected", client.messages[3][-1]["content"]
        )
        self.assertEqual(2, len(reducer_history))
        self.assertEqual(2, len(coder_history))
        self.assertEqual(2, metrics["reducer_calls"])
        self.assertEqual(2, metrics["coder_calls"])
        self.assertEqual(2, metrics["reducer_attempts"])
        self.assertEqual(2, metrics["coder_attempts"])
        self.assertEqual(14, metrics["reducer_generated_tokens"])
        self.assertEqual(14, metrics["coder_generated_tokens"])

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

    def test_coder_exhaustion_is_visible_after_three_attempts(self) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                *[policy_for("tactical:wrong") for _ in range(3)],
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
                reduction_for("tactical:1", verdict="continue"),
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
            agent.close()
        self.assertTrue(result.step_executed)
        self.assertEqual(2, step_env.call_count)
        self.assertEqual(4, client.calls)
        self.assertEqual(1, repairs["tactical:1"])
        self.assertEqual(1, metrics["policy_repairs"])
        self.assertEqual(1, metrics["policy_steps"])

    def test_subgoal_completion_reduces_and_acts_in_same_turn(self) -> None:
        client = _ScriptedModelClient(
            [
                reduction_for("level:1:1"),
                policy_for("tactical:1"),
                reduction_for("level:1:1", title="Second probe"),
                policy_for("tactical:2"),
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
        self.assertEqual(4, client.calls)
        assert tree is not None
        self.assertEqual(ObjectiveStatus.COMPLETED, tree.nodes["tactical:1"].status)
        self.assertEqual("tactical:2", tree.active_id)
        self.assertEqual(1, metrics["objectives_completed"])

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
        agent._policy_failure(failure)
        with self.assertRaisesRegex(
            RuntimeError, "initial policy and two replacements"
        ):
            agent._policy_failure(failure)
        agent.close()


if __name__ == "__main__":
    unittest.main()
