from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from inference.agent.inference_controller import InferenceControllerConfig
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import ToolAgent


class ToolAgentStrategyTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._controller_config = InferenceControllerConfig(enabled=True)
        return agent

    def test_structured_strategy_is_bounded_and_updates_world_model(self) -> None:
        agent = self._agent()

        saved = agent._record_strategy(
            {
                "goal": "g" * 500,
                "hypothesis": "buttons move the matching object",
                "evidence": ["first", "second"],
                "confidence": 4,
                "open_question": "which color is selected?",
                "next_test": "press SPACE once",
            }
        )

        self.assertLessEqual(len(saved["goal"]), 280)
        self.assertEqual(saved["confidence"], 1.0)
        self.assertEqual(agent._summarized_knowledge["current_plan"], "press SPACE once")

    def test_prediction_is_bounded_and_checked_without_another_model_call(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy(
            {
                "test_action": "space",
                "expected_outcome": "level_progress",
                "fallback": "inspect a different object",
                "contradictions": ["old evidence was ambiguous"] * 10,
            }
        )

        result = agent._evaluate_strategy_prediction(
            {
                "executed": True,
                "action_display": "SPACE",
                "outcome_class": "level_progress",
                "level_completed": True,
                "reward": 0.1,
            }
        )

        self.assertEqual(saved["test_action"], "SPACE")
        self.assertEqual(saved["expected_outcome"], "level_progress")
        self.assertLessEqual(len(saved["contradictions"]), 3)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(agent._strategy_memory["prediction_result"], result)

    def test_prompt_contains_bounded_controller_summary_not_raw_grid(self) -> None:
        agent = self._agent()
        frame = Frame(grid=((1, 2), (3, 4)), step=0, level=1)
        history = [HistoryEntry(action="", frame=frame)]
        snapshot = {
            "enabled": True,
            "policy": "outcome_aware",
            "phase": "orient",
            "state_id": "opaque-id",
            "state_visits": 1,
            "unique_states": 1,
            "actions_observed": 0,
            "no_op_streak": 0,
            "stagnation_actions": 0,
            "cycle_period": None,
            "tried_here": {},
            "suggested_actions": ["LEFT"],
            "discouraged_actions": [],
            "ranked_actions": [
                {
                    "action": "LEFT",
                    "priority": 1,
                    "reason": "untried action from this state",
                }
            ],
            "recent_transitions": [{"large": "x" * 10000}],
        }

        prompt = agent._build_user_prompt(
            0,
            valid_actions=["LEFT"],
            current_frame=frame,
            history_entries=history,
            experience_snapshot=snapshot,
        )

        self.assertIn('"phase":"orient"', prompt)
        self.assertIn('"policy":"outcome_aware"', prompt)
        self.assertIn('"ranked_actions"', prompt)
        self.assertIn("opaque-id", prompt)
        self.assertNotIn("recent_transitions", prompt)
        self.assertNotIn("[[1, 2], [3, 4]]", prompt)
        self.assertLess(len(prompt), 10_000)

    def test_objective_reduction_prompt_is_opt_in_and_active_leaf_is_authoritative(self) -> None:
        with patch.dict("os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "true"}):
            agent = self._agent()
        created = agent._objective_reducer.apply(
            {
                "op": "initialize",
                "description": "Complete the level",
                "success_criterion": "level progress",
            }
        )
        frame = Frame(grid=((1,),), step=0, level=1)
        prompt = agent._build_user_prompt(0, valid_actions=["LEFT"], current_frame=frame)

        self.assertTrue(created["ok"])
        self.assertIn("Active objective (obj-1): Complete the level", prompt)
        self.assertIn("Active success criterion: level progress", prompt)
        self.assertIn("Orchestrated objective reduction", agent._system_prompt)

    def test_disabled_agent_keeps_objective_prompt_out(self) -> None:
        with patch.dict("os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "false"}):
            agent = self._agent()
        self.assertNotIn("Orchestrated objective reduction", agent._system_prompt)

    def test_objective_gate_blocks_actions_until_a_leaf_exists(self) -> None:
        with patch.dict("os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "true"}):
            agent = self._agent()
        calls: list[list[dict]] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            frame = Frame(grid=((1,),), step=0, level=1)
            write_runtime_state(
                state_path,
                current_frame=frame,
                history=[HistoryEntry(action="", frame=frame)],
            )
            agent._ensure_session(state_path)
            agent._current_valid_actions = ["LEFT"]
            agent._step_env_callback = lambda payload: calls.append(payload) or {}
            dispatch = agent._run_python_tool(state_path, {"code": "result = action(['LEFT'])"})

        payload = json.loads(dispatch.content)
        self.assertFalse(dispatch.step_executed)
        self.assertEqual(payload["result"]["error"]["code"], "objective_required")
        self.assertEqual(calls, [])

    def test_action_is_attributed_and_recorded_on_active_objective(self) -> None:
        with patch.dict("os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "true"}):
            agent = self._agent()
        received: list[dict] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            frame = Frame(grid=((1,),), step=0, level=1)
            write_runtime_state(
                state_path,
                current_frame=frame,
                history=[HistoryEntry(action="", frame=frame)],
            )
            agent._ensure_session(state_path)
            agent._current_valid_actions = ["LEFT"]
            agent._objective_reducer.apply(
                {
                    "op": "initialize",
                    "description": "Reach target",
                    "success_criterion": "board changes",
                }
            )

            def step(payload: dict) -> dict:
                received.append(payload)
                return {
                    "executed": True,
                    "action_num": 1,
                    "level": 1,
                    "score": 0,
                    "reward": 0,
                    "state": "NOT_FINISHED",
                    "valid_actions": ["LEFT"],
                    "board_changed": True,
                    "done": False,
                    "level_completed": False,
                    "game_over": False,
                    "run_complete": False,
                    "action_display": "LEFT",
                    "executed_count": 1,
                }

            agent._step_env_callback = step
            dispatch = agent._run_python_tool(state_path, {"code": "result = action(['LEFT'])"})

        node = agent.objective_snapshot["graph"]["nodes"][0]
        self.assertTrue(dispatch.step_executed)
        self.assertEqual(received[0]["objective_id"], "obj-1")
        self.assertEqual(received[0]["objective_path"], ["Reach target"])
        self.assertEqual(node["attempts"], 1)
