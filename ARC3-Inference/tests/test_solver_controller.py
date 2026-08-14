from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import arcengine

from inference.agent.inference_controller import InferenceControllerConfig
from inference.framework.solver import _HarnessGameSession


class SolverControllerTests(TestCase):
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
