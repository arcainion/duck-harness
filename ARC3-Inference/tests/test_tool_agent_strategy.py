from __future__ import annotations

from unittest import TestCase

from inference.agent.inference_controller import InferenceControllerConfig
from inference.agent.runtime_state import Frame, HistoryEntry
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

    def test_prompt_contains_bounded_controller_summary_not_raw_grid(self) -> None:
        agent = self._agent()
        frame = Frame(grid=((1, 2), (3, 4)), step=0, level=1)
        history = [HistoryEntry(action="", frame=frame)]
        snapshot = {
            "enabled": True,
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
        self.assertIn("opaque-id", prompt)
        self.assertNotIn("recent_transitions", prompt)
        self.assertNotIn("[[1, 2], [3, 4]]", prompt)
        self.assertLess(len(prompt), 10_000)
