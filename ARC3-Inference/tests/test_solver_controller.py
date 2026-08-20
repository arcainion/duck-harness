from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import arcengine

from inference.agent.action_names import MAX_ACTION_BATCH
from inference.agent.inference_controller import (
    OUTCOME_AWARE_POLICY,
    InferenceControllerConfig,
)
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.framework.solver import (
    _HarnessGameSession,
    _evaluate_strategy_prediction,
    _summarize_animation,
)


class SolverControllerTests(TestCase):
    def test_outcome_aware_orient_phase_rejects_multi_action_probe(self) -> None:
        session = object.__new__(_HarnessGameSession)
        frame = Frame(grid=((1,),), step=0, level=1)
        action = SimpleNamespace(
            id=SimpleNamespace(value=arcengine.GameAction.ACTION1.value, name="ACTION1"),
            data={},
        )
        session.controller_config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY
        )
        session.history_entries = [HistoryEntry(action="", frame=frame)]
        session.game = SimpleNamespace(
            current_state=SimpleNamespace(
                available_actions={arcengine.GameAction.ACTION1.value}
            )
        )
        session._normalize_actions = lambda arguments: ([action, action], None)
        session.current_frame = lambda: frame
        session._error_payload = lambda message: {
            "executed": False,
            "error": {"message": message},
        }

        result = session.step_env({"actions": ["UP", "UP"]})

        self.assertFalse(result["executed"])
        self.assertEqual(result["controller_phase"], "orient")
        self.assertEqual(result["phase_action_budget"], 1)
        self.assertEqual(result["executed_count"], 0)

    def test_animation_summary_preserves_transient_visual_evidence(self) -> None:
        before = ((0, 0), (0, 0))
        state = SimpleNamespace(
            raw=SimpleNamespace(
                frame=[
                    [[0, 1], [0, 0]],
                    [[0, 0], [0, 0]],
                ]
            ),
            frame=SimpleNamespace(data=[[0, 0], [0, 0]]),
        )

        summary = _summarize_animation(before, state)

        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["intermediate_frame_count"], 1)
        self.assertEqual(summary["changed_frame_count"], 2)
        self.assertEqual(summary["final_changed_cells"], 0)
        self.assertEqual(summary["transient_changed_cells"], 1)
        self.assertEqual(summary["motion_bbox"], [0, 1, 0, 1])
        self.assertEqual(
            summary["dominant_color_transitions"],
            [
                {"from": "white (W)", "to": "light gray (w)", "count": 1},
                {"from": "light gray (w)", "to": "white (W)", "count": 1},
            ],
        )

    def test_solver_rejects_oversized_batch_before_action_parsing(self) -> None:
        session = object.__new__(_HarnessGameSession)
        actions, error = session._normalize_actions(
            {"actions": [{"action": "UP"}] * (MAX_ACTION_BATCH + 1)}
        )

        self.assertIsNone(actions)
        self.assertEqual(
            error, f"`actions` may contain at most {MAX_ACTION_BATCH} actions."
        )

    def test_terminal_action_stops_remaining_batch_and_keeps_step_details(self) -> None:
        state = SimpleNamespace(
            available_actions={arcengine.GameAction.ACTION1.value},
            raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED),
            levels_completed=0,
            won=False,
            frame=SimpleNamespace(data=[[0]]),
        )
        game = SimpleNamespace(
            current_state=state,
            game_run=SimpleNamespace(state="playing", history=[]),
            number_of_levels=2,
        )
        solver = SimpleNamespace(
            max_runtime_s_per_game=None,
            max_actions_per_game=None,
            soft_time_remaining_seconds=lambda: None,
            job_dir=None,
        )
        action = SimpleNamespace(
            id=SimpleNamespace(
                value=arcengine.GameAction.ACTION1.value,
                name="ACTION1",
            ),
            data={},
        )
        calls: list[int] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _HarnessGameSession(
                solver=solver,
                game=game,
                analyzer=SimpleNamespace(),
                game_index=0,
                pass_index=0,
                state_path=root / "state.json",
                transcript_path=root / "transcript.txt",
                analysis_html_relpath="analysis.html",
                stop_event=threading.Event(),
                viewer_data_path=root / "viewer.json",
                controller_config=InferenceControllerConfig(enabled=False),
            )
            session._normalize_actions = lambda arguments: ([action, action], None)

            def execute(*args, **kwargs):
                calls.append(1)
                return {
                    "executed": True,
                    "action_num": 1,
                    "level": 2,
                    "score": 1,
                    "reward": 0.5,
                    "state": "NOT_FINISHED",
                    "valid_actions": ["UP"],
                    "board_changed": True,
                    "done": False,
                    "level_completed": True,
                    "game_over": False,
                    "run_complete": False,
                    "action_display": "UP",
                    "before_state_id": "a",
                    "after_state_id": "b",
                    "novel_state": True,
                    "controller_phase": "progress",
                }

            session._execute_action = execute
            result = session.step_env({"actions": ["UP", "UP"]})

        self.assertEqual(len(calls), 1)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["stop_reason"], "level_completed")
        self.assertEqual(result["steps"][0]["before_state_id"], "a")
        self.assertEqual(result["steps"][0]["after_state_id"], "b")

    def test_prediction_result_is_structured(self) -> None:
        result = _evaluate_strategy_prediction(
            {"test_action": "SPACE", "expected_outcome": "new_state"},
            {
                "executed": True,
                "action_display": "SPACE",
                "novel_state": False,
                "outcome_class": "revisit",
            },
        )

        self.assertEqual(result["status"], "contradicted")
        self.assertEqual(result["actual"], "revisit")

    def test_outcome_aware_cycle_warning_does_not_stop_batch(self) -> None:
        state = SimpleNamespace(
            available_actions={arcengine.GameAction.ACTION1.value},
            raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED),
            levels_completed=0,
            won=False,
            frame=SimpleNamespace(data=[[0]]),
        )
        game = SimpleNamespace(
            current_state=state,
            game_run=SimpleNamespace(state="playing", history=[]),
            number_of_levels=2,
        )
        solver = SimpleNamespace(
            max_runtime_s_per_game=None,
            max_actions_per_game=None,
            soft_time_remaining_seconds=lambda: None,
            job_dir=None,
        )
        action = SimpleNamespace(
            id=SimpleNamespace(value=arcengine.GameAction.ACTION1.value, name="ACTION1"),
            data={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _HarnessGameSession(
                solver=solver,
                game=game,
                analyzer=SimpleNamespace(),
                game_index=0,
                pass_index=0,
                state_path=root / "state.json",
                transcript_path=root / "transcript.txt",
                analysis_html_relpath="analysis.html",
                stop_event=threading.Event(),
                viewer_data_path=root / "viewer.json",
                controller_config=InferenceControllerConfig(
                    enabled=True,
                    policy=OUTCOME_AWARE_POLICY,
                    orient_action_budget=2,
                ),
            )
            session._normalize_actions = lambda arguments: ([action, action], None)
            session._execute_action = lambda *args, **kwargs: {
                "executed": True,
                "action_num": 1,
                "level": 1,
                "score": 0,
                "reward": 0,
                "state": "NOT_FINISHED",
                "valid_actions": ["UP"],
                "board_changed": True,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "action_display": "UP",
                "loop_detected": True,
                "cycle_risk": True,
            }

            result = session.step_env({"actions": ["UP", "UP"]})

        self.assertEqual(result["executed_count"], 2)
        self.assertNotEqual(result.get("stop_reason"), "loop_detected")

    def test_empty_actions_list_returns_error(self) -> None:
        session = object.__new__(_HarnessGameSession)
        actions, error = session._normalize_actions({"actions": []})
        self.assertIsNone(actions)
        self.assertIn("at least one", error)

    def test_invalid_action_format_returns_error(self) -> None:
        session = object.__new__(_HarnessGameSession)
        actions, error = session._normalize_actions({"actions": [{"invalid": "format"}]})
        self.assertIsNone(actions)
        self.assertIn("missing", error)

    def test_mixed_string_and_dict_actions(self) -> None:
        session = object.__new__(_HarnessGameSession)
        actions, error = session._normalize_actions({"actions": ["LEFT", {"action": "RIGHT"}]})
        self.assertIsNone(actions)
        self.assertIn("empty", error)

    def test_action_budget_exhausted_in_orient_phase(self) -> None:
        session = object.__new__(_HarnessGameSession)
        frame = Frame(grid=((1,),), step=0, level=1)
        action = SimpleNamespace(
            id=SimpleNamespace(value=arcengine.GameAction.ACTION1.value, name="ACTION1"),
            data={},
        )
        session.controller_config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY, orient_action_budget=1
        )
        session.history_entries = [HistoryEntry(action="", frame=frame)]
        session.game = SimpleNamespace(
            current_state=SimpleNamespace(
                available_actions={arcengine.GameAction.ACTION1.value}
            )
        )
        session._normalize_actions = lambda arguments: ([action], None)
        session.current_frame = lambda: frame
        session._error_payload = lambda message: {
            "executed": False,
            "error": {"message": message},
        }

        result = session.step_env({"actions": ["UP"]})
        self.assertFalse(result["executed"])
        self.assertEqual(result["controller_phase"], "orient")
        self.assertEqual(result["phase_action_budget"], 1)

    def test_animation_summary_empty_frames(self) -> None:
        before = ()
        state = SimpleNamespace(
            raw=SimpleNamespace(frame=[]),
            frame=SimpleNamespace(data=[]),
        )
        summary = _summarize_animation(before, state)
        self.assertEqual(summary["frame_count"], 0)
        self.assertEqual(summary["intermediate_frame_count"], 0)
        self.assertEqual(summary["changed_frame_count"], 0)

    def test_animation_summary_single_frame_no_change(self) -> None:
        before = ((0, 0), (0, 0))
        state = SimpleNamespace(
            raw=SimpleNamespace(
                frame=[
                    [[0, 0], [0, 0]],
                ]
            ),
            frame=SimpleNamespace(data=[[0, 0], [0, 0]]),
        )
        summary = _summarize_animation(before, state)
        self.assertEqual(summary["frame_count"], 1)
        self.assertEqual(summary["intermediate_frame_count"], 0)
        self.assertEqual(summary["changed_frame_count"], 0)
        self.assertEqual(summary["transient_changed_cells"], 0)

    def test_animation_summary_mismatched_dimensions(self) -> None:
        before = ((0, 0), (0, 0))
        state = SimpleNamespace(
            raw=SimpleNamespace(
                frame=[
                    [[0, 0], [0, 0], [0, 0]],
                ]
            ),
            frame=SimpleNamespace(data=[[0, 0], [0, 0], [0, 0]]),
        )
        summary = _summarize_animation(before, state)
        self.assertEqual(summary["frame_count"], 1)

    def test_evaluate_strategy_prediction_no_execution(self) -> None:
        result = _evaluate_strategy_prediction(
            {"test_action": "LEFT", "expected_outcome": "state_change"},
            {"executed": False},
        )
        self.assertEqual(result["status"], "inconclusive")

    def test_evaluate_strategy_prediction_missing_keys(self) -> None:
        result = _evaluate_strategy_prediction(
            {"test_action": "LEFT", "expected_outcome": "state_change"},
            {},
        )
        self.assertEqual(result["status"], "inconclusive")

    def test_evaluate_strategy_prediction_invalid_outcome(self) -> None:
        result = _evaluate_strategy_prediction(
            {"test_action": "LEFT", "expected_outcome": "invalid"},
            {"executed": True, "board_changed": True},
        )
        self.assertEqual(result["status"], "inconclusive")

    def test_step_env_stops_on_game_over(self) -> None:
        state = SimpleNamespace(
            available_actions={arcengine.GameAction.ACTION1.value},
            raw=SimpleNamespace(state=arcengine.GameState.FINISHED),
            levels_completed=1,
            won=True,
            frame=SimpleNamespace(data=[[0]]),
        )
        game = SimpleNamespace(
            current_state=state,
            game_run=SimpleNamespace(state="finished", history=[]),
            number_of_levels=1,
        )
        solver = SimpleNamespace(
            max_runtime_s_per_game=None,
            max_actions_per_game=None,
            soft_time_remaining_seconds=lambda: None,
            job_dir=None,
        )
        action = SimpleNamespace(
            id=SimpleNamespace(value=arcengine.GameAction.ACTION1.value, name="ACTION1"),
            data={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _HarnessGameSession(
                solver=solver,
                game=game,
                analyzer=SimpleNamespace(),
                game_index=0,
                pass_index=0,
                state_path=root / "state.json",
                transcript_path=root / "transcript.txt",
                analysis_html_relpath="analysis.html",
                stop_event=threading.Event(),
                viewer_data_path=root / "viewer.json",
                controller_config=InferenceControllerConfig(enabled=False),
            )
            session._normalize_actions = lambda arguments: ([action], None)
            session._execute_action = lambda *args, **kwargs: {
                "executed": True,
                "action_num": 1,
                "level": 1,
                "score": 1,
                "reward": 1.0,
                "state": "FINISHED",
                "valid_actions": [],
                "board_changed": False,
                "done": True,
                "level_completed": True,
                "game_over": True,
                "run_complete": True,
                "action_display": "UP",
            }
            result = session.step_env({"actions": ["UP"]})

        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["stop_reason"], "game_over")

    def test_step_env_handles_execute_action_exception(self) -> None:
        state = SimpleNamespace(
            available_actions={arcengine.GameAction.ACTION1.value},
            raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED),
            levels_completed=0,
            won=False,
            frame=SimpleNamespace(data=[[0]]),
        )
        game = SimpleNamespace(
            current_state=state,
            game_run=SimpleNamespace(state="playing", history=[]),
            number_of_levels=2,
        )
        solver = SimpleNamespace(
            max_runtime_s_per_game=None,
            max_actions_per_game=None,
            soft_time_remaining_seconds=lambda: None,
            job_dir=None,
        )
        action = SimpleNamespace(
            id=SimpleNamespace(value=arcengine.GameAction.ACTION1.value, name="ACTION1"),
            data={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _HarnessGameSession(
                solver=solver,
                game=game,
                analyzer=SimpleNamespace(),
                game_index=0,
                pass_index=0,
                state_path=root / "state.json",
                transcript_path=root / "transcript.txt",
                analysis_html_relpath="analysis.html",
                stop_event=threading.Event(),
                viewer_data_path=root / "viewer.json",
                controller_config=InferenceControllerConfig(enabled=False),
            )
            session._normalize_actions = lambda arguments: ([action], None)
            session._execute_action = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("execution failed"))
            result = session.step_env({"actions": ["UP"]})

        self.assertFalse(result["executed"])
        self.assertIn("error", result)
