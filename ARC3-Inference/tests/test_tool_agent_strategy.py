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
        self.assertNotIn("test_action", agent._strategy_memory)
        self.assertNotIn("expected_outcome", agent._strategy_memory)
        self.assertIsNone(
            agent._evaluate_strategy_prediction(
                {
                    "executed": True,
                    "action_display": "SPACE",
                    "outcome_class": "level_progress",
                }
            )
        )

    def test_partial_prediction_update_cannot_reuse_a_stale_counterpart(self) -> None:
        agent = self._agent()
        agent._record_strategy(
            {"test_action": "LEFT", "expected_outcome": "no_change"}
        )

        updated = agent._record_strategy({"test_action": "RIGHT"})

        self.assertNotIn("test_action", updated)
        self.assertNotIn("expected_outcome", updated)
        self.assertNotIn("prediction_result", updated)

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
        self.assertIn("`replan`", agent._system_prompt)

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

    def test_repeated_controller_rejections_force_objective_review(self) -> None:
        with patch.dict("os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "true"}):
            agent = self._agent()
        calls: list[dict] = []
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
                    "success_criterion": "progress",
                }
            )

            def reject(payload: dict) -> dict:
                calls.append(payload)
                return {
                    "executed": False,
                    "requested_count": 1,
                    "executed_count": 0,
                    "valid_actions": ["LEFT"],
                    "error": {"code": "cycle_guard", "message": "cycle blocked"},
                }

            agent._step_env_callback = reject
            for _ in range(3):
                agent._run_python_tool(state_path, {"code": "result = action(['LEFT'])"})
            blocked_dispatch = agent._run_python_tool(
                state_path, {"code": "result = action(['LEFT'])"}
            )

        node = agent.objective_snapshot["graph"]["nodes"][0]
        payload = json.loads(blocked_dispatch.content)["result"]
        self.assertEqual(len(calls), 3)
        self.assertEqual(node["attempts"], 0)
        self.assertEqual(node["rejected_action_requests"], 3)
        self.assertEqual(node["rejection_streak"], 3)
        self.assertEqual(payload["error"]["code"], "objective_review_required")
        self.assertEqual(payload["blocking_objective_id"], "obj-1")

    def test_repeated_prediction_contradictions_force_objective_review(self) -> None:
        with patch.dict("os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "true"}):
            agent = self._agent()
        calls: list[dict] = []
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
                    "description": "Test movement model",
                    "success_criterion": "prediction supported",
                }
            )
            agent._record_strategy(
                {"test_action": "LEFT", "expected_outcome": "no_change"}
            )

            def changed(payload: dict) -> dict:
                calls.append(payload)
                return {
                    "executed": True,
                    "action_display": "LEFT",
                    "executed_count": 1,
                    "board_changed": True,
                    "novel_state": True,
                    "outcome_class": "novel",
                    "valid_actions": ["LEFT"],
                }

            agent._step_env_callback = changed
            agent._run_python_tool(state_path, {"code": "result = action(['LEFT'])"})
            agent._run_python_tool(state_path, {"code": "result = action(['LEFT'])"})
            agent._record_strategy(
                {"test_action": "LEFT", "expected_outcome": "no_change"}
            )
            agent._run_python_tool(state_path, {"code": "result = action(['LEFT'])"})
            blocked_dispatch = agent._run_python_tool(
                state_path, {"code": "result = action(['LEFT'])"}
            )

        node = agent.objective_snapshot["graph"]["nodes"][0]
        payload = json.loads(blocked_dispatch.content)["result"]
        self.assertEqual(len(calls), 3)
        self.assertEqual(node["attempts"], 3)
        self.assertEqual(node["prediction_contradictions"], 2)
        self.assertEqual(node["prediction_contradiction_streak"], 2)
        self.assertEqual(payload["error"]["code"], "objective_review_required")

    def test_no_progress_review_gate_identifies_leaf_to_revise(self) -> None:
        with patch.dict("os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "true"}):
            agent = self._agent()
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
                    "success_criterion": "progress",
                }
            )
            for _ in range(3):
                agent._objective_reducer.record_outcome(
                    "obj-1",
                    {"outcome_class": "exact_noop", "board_changed": False},
                )
            dispatch = agent._run_python_tool(
                state_path, {"code": "result = action(['LEFT'])"}
            )

        payload = json.loads(dispatch.content)["result"]
        self.assertEqual(payload["error"]["code"], "objective_review_required")
        self.assertEqual(payload["blocking_objective_id"], "obj-1")

    def test_run_completion_archives_objectives_as_successful(self) -> None:
        with patch.dict("os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "true"}):
            agent = self._agent()
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
                    "description": "Complete game",
                    "success_criterion": "run complete",
                }
            )
            agent._record_strategy(
                {
                    "hypothesis": "This level-specific rule will become stale",
                    "test_action": "LEFT",
                    "expected_outcome": "state_change",
                    "evidence": ["final level evidence"],
                }
            )
            agent._step_env_callback = lambda payload: {
                "executed": True,
                "action_num": 1,
                "level": 1,
                "score": 1,
                "reward": 1,
                "state": "WIN",
                "valid_actions": [],
                "board_changed": True,
                "done": True,
                "level_completed": False,
                "game_over": False,
                "run_complete": True,
                "action_display": "LEFT",
                "executed_count": 1,
            }
            dispatch = agent._run_python_tool(
                state_path, {"code": "result = action(['LEFT'])"}
            )

        snapshot = agent.objective_snapshot
        self.assertTrue(dispatch.step_executed)
        self.assertIsNone(snapshot["graph"]["root_id"])
        self.assertEqual(snapshot["archives"][-1]["reason"], "run_complete")
        self.assertTrue(snapshot["archives"][-1]["successful"])
        self.assertIn("Hypothesis: This level-specific rule", snapshot["archives"][-1]["lesson"])
        self.assertIn("Evidence: final level evidence", snapshot["archives"][-1]["lesson"])
        self.assertEqual(agent._strategy_memory, {})
        self.assertFalse(any(agent._summarized_knowledge.values()))

    def test_level_transition_and_game_reset_clear_transition_scoped_reasoning(self) -> None:
        for reason in ("level_transition", "game_reset"):
            with patch.dict(
                "os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "true"}
            ):
                agent = self._agent()
            agent._objective_reducer.apply(
                {
                    "op": "initialize",
                    "description": "Complete current level",
                    "success_criterion": "level transition",
                }
            )
            agent._record_strategy(
                {
                    "goal": "old level goal",
                    "hypothesis": "old level hypothesis",
                    "test_action": "LEFT",
                    "expected_outcome": "no_change",
                    "evidence": [f"lesson from {reason}"],
                }
            )

            archive = agent.archive_objectives(
                reason, successful=reason == "level_transition"
            )

            self.assertIsNotNone(archive)
            self.assertIn("Hypothesis: old level hypothesis", archive["lesson"])
            self.assertIn(f"Evidence: lesson from {reason}", archive["lesson"])
            self.assertEqual(agent._strategy_memory, {})
            self.assertFalse(any(agent._summarized_knowledge.values()))
            self.assertIsNone(agent.objective_snapshot["graph"]["root_id"])

    def test_transition_cleanup_does_not_require_an_objective_archive(self) -> None:
        with patch.dict(
            "os.environ", {"LOCAL_ANALYZER_OBJECTIVE_REDUCTION_ENABLED": "false"}
        ):
            agent = self._agent()
        agent._record_strategy(
            {
                "goal": "stale goal",
                "hypothesis": "stale hypothesis",
                "test_action": "LEFT",
                "expected_outcome": "no_change",
            }
        )

        archive = agent.archive_objectives("game_reset")

        self.assertIsNone(archive)
        self.assertEqual(agent._strategy_memory, {})
        self.assertFalse(any(agent._summarized_knowledge.values()))
