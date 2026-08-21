from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, patch

import arcengine

from inference.agent.action_names import MAX_ACTION_BATCH
from inference.agent.inference_controller import (
    OUTCOME_AWARE_POLICY,
    InferenceControllerConfig,
)
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.framework.solver import (
    HarnessSolver,
    _HarnessGameSession,
    _evaluate_strategy_prediction,
    _is_engine_game_over,
    _is_run_complete,
    _summarize_animation,
)


class SolverControllerTests(TestCase):
    @staticmethod
    def _deadline_session(soft_remaining: float | None) -> _HarnessGameSession:
        session = object.__new__(_HarnessGameSession)
        session.started_at = 0.0
        session.stop_event = threading.Event()
        session.game = SimpleNamespace(
            current_state=SimpleNamespace(
                raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED),
                levels_completed=0,
                won=False,
            ),
            game_run=SimpleNamespace(state="playing", history=[]),
        )
        session.solver = SimpleNamespace(
            max_runtime_s_per_game=None,
            max_actions_per_game=None,
            soft_time_remaining_seconds=lambda: soft_remaining,
        )
        session.analyzer = SimpleNamespace(_timeout=120.0)
        return session

    def test_soft_experiment_deadline_stops_game_session(self) -> None:
        session = self._deadline_session(0.0)

        self.assertTrue(session.should_stop())

    def test_retry_backoff_is_bounded_by_remaining_budget(self) -> None:
        session = self._deadline_session(0.25)

        self.assertEqual(session.retry_backoff_seconds(), 0.25)
        self.assertEqual(session.request_timeout_seconds(), 0.25)

    def test_play_one_closes_analyzer_if_session_initialization_fails(self) -> None:
        analyzer_close = MagicMock()
        analyzer = SimpleNamespace(close=analyzer_close)
        solver = object.__new__(HarnessSolver)
        solver._stop_event = threading.Event()
        solver._make_analyzer = lambda *args: analyzer
        solver._finish_after_error = MagicMock()
        solver._run_stem = lambda game_id, pass_index: "game-0"
        run = SimpleNamespace(game_id="game")
        game = SimpleNamespace(game_run=run)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            solver._artifacts_dir = lambda: root / "artifacts"
            solver._transcripts_dir = lambda: root / "transcripts"
            with patch.object(
                _HarnessGameSession,
                "play",
                side_effect=RuntimeError("initialization failed"),
            ):
                solver._play_one(game, 0, 0)

        analyzer_close.assert_called_once_with()
        solver._finish_after_error.assert_called_once()

    def test_transcript_delta_reads_only_bytes_after_saved_offset(self) -> None:
        session = object.__new__(_HarnessGameSession)
        with tempfile.TemporaryDirectory() as temp_dir:
            session.transcript_path = Path(temp_dir) / "analysis.txt"
            session.transcript_path.write_text("old transcript\n", encoding="utf-8")
            offset = session._transcript_size()
            with session.transcript_path.open("a", encoding="utf-8") as transcript:
                transcript.write("new delta\n")

            delta = session._transcript_delta_since(offset)

        self.assertEqual(delta, "new delta")

    def test_transcript_delta_recovers_if_file_was_truncated(self) -> None:
        session = object.__new__(_HarnessGameSession)
        with tempfile.TemporaryDirectory() as temp_dir:
            session.transcript_path = Path(temp_dir) / "analysis.txt"
            session.transcript_path.write_text("replacement", encoding="utf-8")

            delta = session._transcript_delta_since(10_000)

        self.assertEqual(delta, "replacement")

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

    def test_partial_batch_preserves_later_action_exception_detail(self) -> None:
        session = object.__new__(_HarnessGameSession)
        frame = Frame(grid=((0,),), step=0, level=1)
        action = SimpleNamespace(
            id=SimpleNamespace(value=arcengine.GameAction.ACTION1.value, name="ACTION1"),
            data={},
        )
        session.controller_config = InferenceControllerConfig(enabled=False)
        session.history_entries = [HistoryEntry(action="", frame=frame)]
        session.game = SimpleNamespace(
            current_state=SimpleNamespace(
                available_actions={arcengine.GameAction.ACTION1.value}
            )
        )
        session._normalize_actions = lambda arguments: ([action, action], None)
        session.current_frame = lambda: frame
        session.should_stop = lambda: False
        session.write_viewer_payload = lambda: None
        calls = 0

        def execute(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second action failed")
            return {
                "executed": True,
                "action_num": 1,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "NOT_FINISHED",
                "valid_actions": ["UP"],
                "board_changed": False,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "action_display": "UP",
            }

        session._execute_action = execute
        result = session.step_env({"actions": ["UP", "UP"]})

        self.assertEqual(result["executed_count"], 1)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["stop_reason"], "action_error")
        self.assertEqual(result["stop_detail"], "RuntimeError: second action failed")

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

    def test_null_single_action_does_not_conflict_with_valid_batch(self) -> None:
        session = object.__new__(_HarnessGameSession)
        actions, error = session._normalize_actions({
            "action": None,
            "actions": [{"action": "LEFT"}],
        })
        self.assertIsNone(error)
        self.assertEqual(len(actions or []), 1)

    def test_legacy_finished_state_distinguishes_win_from_game_over(self) -> None:
        legacy_state = SimpleNamespace(name="FINISHED")
        won_game = SimpleNamespace(
            current_state=SimpleNamespace(raw=SimpleNamespace(state=legacy_state), won=True)
        )
        lost_game = SimpleNamespace(
            current_state=SimpleNamespace(raw=SimpleNamespace(state=legacy_state), won=False)
        )
        self.assertTrue(_is_run_complete(won_game))
        self.assertFalse(_is_engine_game_over(won_game))
        self.assertFalse(_is_run_complete(lost_game))
        self.assertTrue(_is_engine_game_over(lost_game))

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

    def test_mouse_action_uses_row_col_and_clean_engine_payload(self) -> None:
        session = object.__new__(_HarnessGameSession)
        session.game = SimpleNamespace(grid_size=(12, 8))

        actions, error = session._normalize_actions(
            {"action": "MOUSE", "row": 7, "col": 11}
        )

        self.assertIsNone(error)
        self.assertEqual(len(actions or []), 1)
        self.assertEqual((actions or [])[0].data, {"x": 11, "y": 7})

    def test_mouse_action_rejects_coordinates_outside_actual_grid(self) -> None:
        session = object.__new__(_HarnessGameSession)
        session.game = SimpleNamespace(grid_size=(12, 8))

        actions, error = session._normalize_actions(
            {"action": "MOUSE", "row": 8, "col": 11}
        )

        self.assertIsNone(actions)
        self.assertIn("outside the current 8x12 board", error)
        self.assertIn("row must be 0..7", error)

    def test_mouse_action_rejects_non_integer_coordinates_without_coercion(self) -> None:
        session = object.__new__(_HarnessGameSession)
        session.game = SimpleNamespace(grid_size=(12, 8))

        for row in (1.9, True, "3"):
            with self.subTest(row=row):
                actions, error = session._normalize_actions(
                    {"action": "MOUSE", "row": row, "col": 2}
                )
                self.assertIsNone(actions)
                self.assertIn("requires integer row and col", error)

    def test_mouse_action_invalid_grid_size_uses_engine_default(self) -> None:
        session = object.__new__(_HarnessGameSession)
        session.game = SimpleNamespace(grid_size=(0, 0))

        actions, error = session._normalize_actions(
            {"action": "MOUSE", "row": 63, "col": 63}
        )

        self.assertIsNone(error)
        self.assertEqual((actions or [])[0].data, {"x": 63, "y": 63})

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

        with patch(
            "inference.framework.solver.build_experience_snapshot",
            return_value={"phase": "orient", "action_budget": 0},
        ):
            result = session.step_env({"actions": ["UP"]})
        self.assertFalse(result["executed"])
        self.assertEqual(result["controller_phase"], "orient")
        self.assertEqual(result["phase_action_budget"], 0)

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
            raw=SimpleNamespace(state=arcengine.GameState.GAME_OVER),
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


class SolverConcurrencyTests(IsolatedAsyncioTestCase):
    @staticmethod
    def _game(game_id: str) -> SimpleNamespace:
        return SimpleNamespace(game_run=SimpleNamespace(game_id=game_id))

    @staticmethod
    def _solver() -> HarnessSolver:
        solver = object.__new__(HarnessSolver)
        solver.concurrency = 1
        solver.cancel_drain_timeout_s = 1.0
        solver._stop_event = threading.Event()
        solver._worker_pool = None
        solver._local_server_for_game_index = lambda index: None
        solver._finish_remaining = MagicMock()
        solver._finish_after_error = MagicMock()
        return solver

    async def test_cancelled_run_does_not_start_queued_games(self) -> None:
        solver = self._solver()
        first_started = threading.Event()
        calls: list[int] = []

        def play_one(game, index, pass_index, local_server) -> None:
            calls.append(index)
            first_started.set()
            solver._stop_event.wait(timeout=2.0)

        solver._play_one = play_one
        task = asyncio.create_task(
            solver._run_games([self._game("first"), self._game("queued")])
        )
        self.assertTrue(await asyncio.to_thread(first_started.wait, 1.0))
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(calls, [0])
        solver._finish_remaining.assert_called_once()

    async def test_zero_timeout_drain_cancels_pending_tasks(self) -> None:
        solver = self._solver()
        solver.cancel_drain_timeout_s = 0.0
        pending = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)

        await solver._drain_game_tasks([pending])

        self.assertTrue(pending.cancelled())

    async def test_orchestration_failure_is_recorded_on_game(self) -> None:
        solver = self._solver()
        game = self._game("broken")

        def fail_server_assignment(index: int):
            raise RuntimeError("server assignment failed")

        solver._local_server_for_game_index = fail_server_assignment
        await solver._run_games([game])

        solver._finish_after_error.assert_called_once()
        recorded_game, recorded_error = solver._finish_after_error.call_args.args
        self.assertIs(recorded_game, game)
        self.assertIn("server assignment failed", str(recorded_error))
