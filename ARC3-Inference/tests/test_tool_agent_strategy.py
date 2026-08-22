from __future__ import annotations

from unittest import TestCase

from inference.agent.action_names import MAX_ACTION_BATCH
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

    def test_python_action_batch_is_bounded_before_host_dispatch(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "at most 12 actions"):
            agent._normalize_python_actions(["LEFT"] * 13)

    def test_compact_action_result_tolerates_malformed_host_counts(self) -> None:
        agent = self._agent()

        compact = agent._compact_action_result(
            {
                "executed": True,
                "requested_count": "many",
                "executed_count": "several",
                "stopped_early": True,
                "executed_actions": ["LEFT"] * (MAX_ACTION_BATCH + 3),
            }
        )

        self.assertEqual(compact["requested_count"], 1)
        self.assertEqual(compact["executed_count"], 1)
        self.assertEqual(len(compact["executed_actions"]), MAX_ACTION_BATCH)

    def test_compact_action_result_keeps_bounded_animation_summary(self) -> None:
        agent = self._agent()
        animation = {
            "frame_count": 3,
            "transient_changed_cells": 4,
            "motion_bbox": [1, 2, 5, 8],
        }

        compact = agent._compact_action_result(
            {"executed": True, "action_display": "SPACE", "animation": animation}
        )

        self.assertEqual(compact["animation"], animation)

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

    def test_prediction_is_evaluated_once_then_consumed(self) -> None:
        agent = self._agent()
        agent._record_strategy(
            {
                "test_action": "space",
                "expected_outcome": "level_progress",
                "fallback": "inspect a different object",
                "contradictions": ["old evidence was ambiguous"] * 10,
            }
        )
        self.assertEqual(agent._strategy_memory["fallback"], "inspect a different object")
        self.assertLessEqual(len(agent._strategy_memory["contradictions"]), 3)

        action_result = {
            "executed": True,
            "action_display": "SPACE",
            "outcome_class": "level_progress",
            "level_completed": True,
            "reward": 0.1,
        }
        result = agent._evaluate_strategy_prediction(action_result)

        self.assertEqual(result["status"], "supported")
        self.assertNotIn("test_action", agent._strategy_memory)
        self.assertNotIn("expected_outcome", agent._strategy_memory)
        self.assertNotIn("prediction_result", agent._strategy_memory)
        consumed = agent._strategy_memory["last_evaluated_prediction"]
        self.assertEqual(consumed["test_action"], "SPACE")
        self.assertEqual(consumed["expected_outcome"], "level_progress")
        self.assertEqual(consumed["status"], "supported")

        repeat = agent._evaluate_strategy_prediction(dict(action_result))
        self.assertIsNone(repeat)

    def test_partial_prediction_update_keeps_declared_outcome(self) -> None:
        agent = self._agent()
        agent._record_strategy(
            {"test_action": "LEFT", "expected_outcome": "new_state"}
        )
        agent._record_strategy({"test_action": "RIGHT"})

        self.assertEqual(agent._strategy_memory["test_action"], "RIGHT")
        self.assertEqual(agent._strategy_memory["expected_outcome"], "new_state")

    def test_prompt_contains_bounded_controller_summary_not_raw_grid(self) -> None:
        agent = self._agent()
        frame = Frame(grid=((1, 2), (3, 4)), step=0, level=1)
        history = [HistoryEntry(action="", frame=frame)]
        snapshot = {
            "enabled": True,
            "policy": "outcome_aware",
            "phase": "orient",
            "action_budget": 1,
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
            "transition_models_here": [
                {
                    "action": "RIGHT",
                    "trials": 2,
                    "confidence": 1.0,
                    "verified_deterministic": True,
                }
            ],
            "model_conflicts_here": 0,
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
        self.assertIn('"action_budget":1', prompt)
        self.assertIn('"policy":"outcome_aware"', prompt)
        self.assertIn('"ranked_actions"', prompt)
        self.assertIn('"transition_models_here"', prompt)
        self.assertIn('"verified_deterministic":true', prompt)
        self.assertIn("opaque-id", prompt)
        self.assertNotIn("recent_transitions", prompt)
        self.assertNotIn("[[1, 2], [3, 4]]", prompt)
        self.assertIn("prefer batching it in one call", prompt)
        self.assertIn(
            "Before executing new actions you must always give the revised version",
            prompt,
        )
        self.assertLess(len(prompt), 10_000)

    def test_python_actions_are_canonicalized_and_checked_against_current_actions(self) -> None:
        agent = self._agent()
        agent._current_valid_actions = ["LEFT", "MOUSE"]

        self.assertEqual(
            agent._normalize_python_actions(["ACTION3"]),
            [{"action": "LEFT"}],
        )
        with self.assertRaisesRegex(ValueError, "RIGHT.*not currently valid"):
            agent._normalize_python_actions(["RIGHT"])

    def test_mouse_actions_require_integer_row_and_col(self) -> None:
        agent = self._agent()
        agent._current_valid_actions = ["MOUSE"]

        for action in (
            {"action": "MOUSE"},
            {"action": "MOUSE", "row": 1},
            {"action": "MOUSE", "row": True, "col": 2},
            {"action": "MOUSE", "row": 1, "col": 2.5},
        ):
            with self.subTest(action=action), self.assertRaisesRegex(
                ValueError, "requires integer|must be an integer"
            ):
                agent._normalize_python_actions([action])
