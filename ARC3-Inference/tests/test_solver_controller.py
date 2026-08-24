from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import arcengine

from inference.agent.tool_agent import AnalyzerTurnResult
from inference.framework import solver as solver_module
from inference.agent.action_names import (
    MAX_ACTION_BATCH,
    to_engine_action,
    to_model_action,
)
from inference.agent.inference_controller import (
    LEGACY_POLICY,
    OUTCOME_AWARE_POLICY,
    InferenceControllerConfig,
)
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.trial_knowledge import TrialKnowledgeStore
from inference.framework.solver import (
    _HarnessGameSession,
    _evaluate_strategy_prediction,
    _summarize_animation,
)


class SolverControllerTests(TestCase):
    def test_host_cross_trial_harm_check_uses_the_current_state(self) -> None:
        store = TrialKnowledgeStore()
        frame = Frame(grid=((2,),), step=1, level=1)
        state_id = solver_module.frame_fingerprint(frame)
        store.observe(
            "game-a",
            {
                "executed": True,
                "before_state_id": state_id,
                "after_state_id": "terminal",
                "action_display": "RIGHT",
                "outcome_class": "terminal_failure",
            },
            evidence_id="run-a:pass=0",
        )
        session = object.__new__(_HarnessGameSession)
        session.solver = SimpleNamespace(_knowledge_store=store)
        session.game = SimpleNamespace(game_run=SimpleNamespace(game_id="game-a"))
        session.controller_config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY
        )
        session.current_frame = lambda: frame

        reason = session._cross_trial_harm_reason("RIGHT")

        self.assertIn("1 of 1 independent cross-trial", reason)
        self.assertIsNone(session._cross_trial_harm_reason("LEFT"))

        store.observe(
            "game-a",
            {
                "executed": True,
                "before_state_id": state_id,
                "after_state_id": "progress",
                "action_display": "RIGHT",
                "outcome_class": "level_progress",
            },
            evidence_id="run-b:pass=0",
        )
        self.assertIsNone(session._cross_trial_harm_reason("RIGHT"))

    def test_host_controller_snapshot_uses_prior_pass_without_live_duplicates(
        self,
    ) -> None:
        store = TrialKnowledgeStore()
        for evidence_id, action in (
            ("run-a:pass=0", "LEFT"),
            ("run-a:pass=1", "RIGHT"),
        ):
            store.observe(
                "game-a",
                {
                    "executed": True,
                    "before_state_id": "state-a",
                    "after_state_id": f"after-{action}",
                    "action_display": action,
                    "outcome_class": "novel",
                },
                evidence_id=evidence_id,
            )
        session = object.__new__(_HarnessGameSession)
        session.solver = SimpleNamespace(
            _knowledge_store=store, _knowledge_run_id="run-a"
        )
        session.pass_index = 1
        session.game = SimpleNamespace(
            game_run=SimpleNamespace(game_id="game-a"), current_state=SimpleNamespace()
        )
        session.history_entries = []
        session.controller_config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY
        )
        session.current_frame = lambda: Frame(grid=((1,),), step=0, level=1)
        with (
            mock.patch.object(
                solver_module, "_engine_action_names", return_value=["ACTION1"]
            ),
            mock.patch.object(
                solver_module, "build_experience_snapshot", return_value={}
            ) as build,
        ):
            session._controller_snapshot()

        external = build.call_args.kwargs["external_transitions"]
        self.assertEqual([item["action_display"] for item in external], ["LEFT"])
        self.assertEqual(build.call_args.kwargs["evidence_id"], "run-a:pass=1")

    def test_game_token_budget_yield_finishes_instead_of_retrying_forever(self) -> None:
        run = SimpleNamespace(
            state="playing",
            history=[],
            final_score=None,
            solver_note=None,
            solver_analysis_html=None,
        )
        state = SimpleNamespace(
            raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED),
            available_actions={arcengine.GameAction.ACTION1.value},
        )
        calls = 0

        def analyze(*_args, **_kwargs) -> AnalyzerTurnResult:
            nonlocal calls
            calls += 1
            return AnalyzerTurnResult(
                step_executed=False,
                yielded_control=True,
                yield_reason="game_token_budget",
            )

        def finish_game() -> None:
            run.final_score = 0

        game = SimpleNamespace(
            current_state=state,
            game_run=run,
            finish_game=finish_game,
        )
        solver = SimpleNamespace(
            max_runtime_s_per_game=None,
            max_actions_per_game=None,
            soft_time_remaining_seconds=lambda: None,
        )
        analyzer = SimpleNamespace(total_tokens=100, analyze=analyze)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _HarnessGameSession(
                solver=solver,
                game=game,
                analyzer=analyzer,
                game_index=0,
                pass_index=0,
                state_path=root / "state.json",
                transcript_path=root / "transcript.txt",
                analysis_html_relpath="analysis.html",
                stop_event=threading.Event(),
                viewer_data_path=root / "viewer.json",
                controller_config=InferenceControllerConfig(enabled=False),
            )
            for method_name in (
                "seed_initial_history",
                "write_runtime_state",
                "_append_initial_viewer_event",
                "write_viewer_payload",
                "_write_analysis_html",
            ):
                setattr(session, method_name, lambda: None)
            session._read_transcript_bytes = lambda: b""
            session._transcript_delta_since = lambda _content: ""

            session.play()

        self.assertEqual(calls, 1)
        self.assertEqual(run.final_score, 0)
        self.assertIn("game_token_budget", run.solver_note)

    def test_game_token_budget_fallback_executes_safe_ranked_action(self) -> None:
        session = object.__new__(_HarnessGameSession)
        session.controller_config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY
        )
        session.history_entries = []
        session.should_stop = lambda: False
        session.current_frame = lambda: Frame(grid=((0,),), step=0, level=1)
        session.game = SimpleNamespace()
        captured: list[dict] = []

        def step_env(arguments: dict) -> dict:
            captured.append(arguments)
            return {"executed": True}

        session.step_env = step_env
        with (
            mock.patch.object(
                solver_module, "_engine_action_names", return_value=["ACTION4"]
            ),
            mock.patch.object(
                solver_module,
                "build_experience_snapshot",
                return_value={
                    "ranked_actions": [
                        {"action": "RIGHT", "harm_decisive": False},
                    ],
                    "mouse_search": {},
                },
            ),
        ):
            executed = session._execute_controller_fallback("game_token_budget")

        self.assertTrue(executed)
        self.assertEqual(captured[0]["actions"], [{"action": "RIGHT"}])
        self.assertEqual(captured[0]["controller_fallback_reason"], "game_token_budget")

    def test_controller_fallback_skips_reset_without_recovery_trigger(self) -> None:
        session = object.__new__(_HarnessGameSession)
        session.controller_config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY
        )
        session.should_stop = lambda: False
        captured: list[dict] = []
        session.step_env = lambda arguments: (
            captured.append(arguments) or {"executed": True}
        )
        session._controller_snapshot = lambda: {
            "ranked_actions": [
                {"action": "RESET", "harm_decisive": False},
                {"action": "RIGHT", "harm_decisive": False},
            ],
            "mouse_search": {},
            "recovery_portfolio": [],
        }

        executed = session._execute_controller_fallback("game_token_budget")

        self.assertTrue(executed)
        self.assertEqual(captured[0]["actions"], [{"action": "RIGHT"}])

    def test_game_token_budget_hands_off_without_reentering_analyzer(self) -> None:
        run = SimpleNamespace(
            state="playing",
            history=[],
            final_score=None,
            solver_note=None,
            solver_analysis_html=None,
        )
        analyze_calls = 0

        def analyze(*_args, **_kwargs) -> AnalyzerTurnResult:
            nonlocal analyze_calls
            analyze_calls += 1
            return AnalyzerTurnResult(
                step_executed=False,
                yielded_control=True,
                yield_reason="game_token_budget",
            )

        fallback_calls = 0

        def fallback(_reason: str) -> bool:
            nonlocal fallback_calls
            fallback_calls += 1
            return True

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = object.__new__(_HarnessGameSession)
            session.game = SimpleNamespace(
                game_run=run,
                current_state=SimpleNamespace(
                    available_actions={arcengine.GameAction.ACTION1.value},
                    raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED),
                ),
            )
            session.analyzer = SimpleNamespace(total_tokens=100, analyze=analyze)
            session.transcript_path = root / "transcript.txt"
            session.transcript_path.touch()
            session.analysis_html_relpath = "analysis.html"
            session.token_baseline = 0
            session.analysis_step = 0
            session.seed_initial_history = lambda: None
            session.write_runtime_state = lambda: None
            session._append_initial_viewer_event = lambda: None
            session.write_viewer_payload = lambda: None
            session._write_analysis_html = lambda: None
            session._finish_if_needed = lambda: None
            session._read_transcript_bytes = lambda: b""
            session._transcript_delta_since = lambda _content: ""
            session._execute_controller_fallback = fallback
            session.should_stop = lambda: fallback_calls >= 2
            session.request_timeout_seconds = lambda: None
            session.state_path = root / "state.json"

            session.play()

        self.assertEqual(analyze_calls, 1)
        self.assertEqual(fallback_calls, 2)

    def test_local_harm_guard_is_reported_as_harm_not_loop(self) -> None:
        session = object.__new__(_HarnessGameSession)
        session.controller_config = InferenceControllerConfig(
            enabled=True, policy=OUTCOME_AWARE_POLICY
        )
        session.history_entries = []
        session.viewer_events = []
        session.analysis_step = 1
        session.should_stop = lambda: False
        session.current_frame = lambda: Frame(grid=((0,),), step=0, level=1)
        session._base_viewer_event = lambda _frame: {}
        session.write_viewer_payload = lambda: None
        session.timing_payload = lambda: {
            "run_elapsed_seconds": 0.0,
            "time_remaining_seconds": None,
        }
        action = SimpleNamespace(id=SimpleNamespace(name="RIGHT", value=4), data={})
        session._normalize_actions = lambda _arguments: ([action], None)
        session.game = SimpleNamespace(
            game_run=SimpleNamespace(history=[]),
            current_state=SimpleNamespace(
                available_actions={4},
                levels_completed=0,
                raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED),
            ),
        )
        with (
            mock.patch.object(
                solver_module,
                "build_experience_snapshot",
                return_value={"action_budget": 1},
            ),
            mock.patch.object(
                solver_module, "_engine_action_names", return_value=["ACTION4"]
            ),
            mock.patch.object(
                solver_module,
                "action_guard_reason",
                return_value="decisive local terminal evidence",
            ),
            mock.patch.object(
                solver_module,
                "action_guard_reason_code",
                return_value="known_harmful_local",
            ),
        ):
            payload = session.step_env({"actions": [{"action": "RIGHT"}]})

        self.assertEqual(payload["stop_reason"], "harm_guard")
        self.assertFalse(payload["loop_detected"])
        self.assertEqual(session.viewer_events[-1]["title"], "Harm Guard")

    def test_action_viewer_event_preserves_inference_metric_fields(self) -> None:
        session = object.__new__(_HarnessGameSession)
        session.viewer_events = []
        session.analysis_step = 3
        session._base_viewer_event = lambda _frame: {}
        animation = {"intermediate_frame_count": 2, "transient_changed_cells": 1}

        session._append_action_viewer_event(
            {
                "action_num": 7,
                "action_display": "RIGHT",
                "decision_context_changed": True,
                "animation": animation,
                "recommended_plan_action": "RIGHT",
                "followed_recommended_plan": True,
                "recommended_plan_confidence": 0.8,
                "recommended_plan_expected_utility": 1.25,
            },
            Frame(grid=((1,),), step=7, level=1),
        )

        event = session.viewer_events[-1]
        self.assertEqual(event["animation"], animation)
        self.assertTrue(event["decision_context_changed"])
        self.assertEqual(event["recommended_plan_action"], "RIGHT")
        self.assertTrue(event["followed_recommended_plan"])
        self.assertEqual(event["recommended_plan_confidence"], 0.8)
        self.assertEqual(event["recommended_plan_expected_utility"], 1.25)

    def test_analyzer_retry_circuit_breaker_stops_repeated_transport_failures(
        self,
    ) -> None:
        run = SimpleNamespace(
            state="playing",
            history=[],
            final_score=None,
            solver_note=None,
            solver_analysis_html=None,
        )
        state = SimpleNamespace(
            raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED),
            available_actions={arcengine.GameAction.ACTION1.value},
        )

        def finish_game() -> None:
            run.final_score = 0

        game = SimpleNamespace(
            current_state=state,
            game_run=run,
            finish_game=finish_game,
        )
        solver = SimpleNamespace(
            max_runtime_s_per_game=None,
            max_actions_per_game=None,
            soft_time_remaining_seconds=lambda: None,
        )
        analyzer = SimpleNamespace(
            total_tokens=0,
            analyze=lambda *_args, **_kwargs: AnalyzerTurnResult(
                step_executed=False,
                retryable_failure=True,
                failure_category="transport",
                failure_detail="server unavailable",
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = _HarnessGameSession(
                solver=solver,
                game=game,
                analyzer=analyzer,
                game_index=0,
                pass_index=0,
                state_path=root / "state.json",
                transcript_path=root / "transcript.txt",
                analysis_html_relpath="analysis.html",
                stop_event=threading.Event(),
                viewer_data_path=root / "viewer.json",
                controller_config=InferenceControllerConfig(enabled=False),
            )
            for method_name in (
                "seed_initial_history",
                "write_runtime_state",
                "_append_initial_viewer_event",
                "write_viewer_payload",
                "_write_analysis_html",
            ):
                setattr(session, method_name, lambda: None)
            session._read_transcript_bytes = lambda: 0
            session._transcript_delta_since = lambda _offset: ""

            with (
                mock.patch.object(
                    solver_module, "ANALYZER_MAX_CONSECUTIVE_FAILURES", 2
                ),
                mock.patch.object(solver_module.time, "sleep"),
            ):
                session.play()

        self.assertEqual(run.state, "crashed")
        self.assertIn("circuit breaker opened", run.solver_note)

    def test_action7_round_trips_through_normalize_actions(self) -> None:
        self.assertEqual(to_engine_action("ACTION7"), "ACTION7")
        self.assertEqual(to_model_action("ACTION7"), "ACTION7")

        session = object.__new__(_HarnessGameSession)
        actions, error = session._normalize_actions(
            {"actions": [{"action": "ACTION7"}]}
        )

        self.assertIsNone(error)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].id, arcengine.GameAction.ACTION7)

    def test_outcome_aware_orient_phase_rejects_multi_action_probe(self) -> None:
        session = object.__new__(_HarnessGameSession)
        frame = Frame(grid=((1,),), step=0, level=1)
        action = SimpleNamespace(
            id=SimpleNamespace(
                value=arcengine.GameAction.ACTION1.value, name="ACTION1"
            ),
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
        self.assertEqual(summary["per_frame_changed_cells"], [1, 1])
        self.assertEqual(summary["motion_direction"], "stationary")
        self.assertTrue(summary["temporally_reversible"])
        self.assertEqual(
            summary["dominant_color_transitions"],
            [
                {"from": "W", "to": "w", "count": 1},
                {"from": "w", "to": "W", "count": 1},
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

    def test_partial_batch_failure_preserves_bounded_stop_detail(self) -> None:
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
        calls = 0

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
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("engine rejected transition")
                return {
                    "executed": True,
                    "action_num": 1,
                    "reward": 0,
                    "board_changed": True,
                    "level_completed": False,
                    "game_over": False,
                    "run_complete": False,
                    "action_display": "UP",
                }

            session._execute_action = execute
            result = session.step_env({"actions": ["UP", "UP"]})

        self.assertEqual(result["executed_count"], 1)
        self.assertEqual(result["stop_reason"], "action_error")
        self.assertEqual(
            result["stop_detail"], "RuntimeError: engine rejected transition"
        )

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
            id=SimpleNamespace(
                value=arcengine.GameAction.ACTION1.value, name="ACTION1"
            ),
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

    def test_legacy_loop_warning_does_not_stop_batch(self) -> None:
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
                value=arcengine.GameAction.ACTION1.value, name="ACTION1"
            ),
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
                    enabled=True, policy=LEGACY_POLICY
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

    def test_batch_consumes_strategy_prediction_after_first_match(self) -> None:
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
                value=arcengine.GameAction.ACTION1.value, name="ACTION1"
            ),
            data={},
        )

        def make_payload(strategy_prediction):
            payload = {
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
            }
            if strategy_prediction:
                payload["prediction_result"] = {"status": "supported"}
            return payload

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

            captured: list[dict | None] = []

            def execute(*args, **kwargs):
                prediction = kwargs.get("strategy_prediction")
                captured.append(prediction)
                return make_payload(prediction)

            session._execute_action = execute
            session.step_env(
                {
                    "actions": ["UP", "UP"],
                    "strategy_prediction": {
                        "test_action": "UP",
                        "expected_outcome": "new_state",
                    },
                }
            )

        self.assertEqual(
            captured[0],
            {"test_action": "UP", "expected_outcome": "new_state"},
        )
        self.assertIsNone(captured[1])
