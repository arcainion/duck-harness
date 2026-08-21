from __future__ import annotations

import json
import os
import requests
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from inference.agent.action_names import MAX_ACTION_BATCH
from inference.agent.inference_controller import InferenceControllerConfig
from inference.agent.python_tool_policy import (
    PROTECTED_RUNTIME_BINDINGS,
    RUNTIME_HELPER_SIGNATURES,
)
from inference.agent.program_ir import program_tool_parameters_schema
from inference.agent.prompts import TOOL_CALL_FORMAT_GUIDANCE
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from inference.agent.tool_agent import (
    ToolAgent,
    _ChatCompletionResult,
    _build_system_prompt,
    _compact_multimodal_history_message,
    _compact_no_tool_retry_message,
    _contains_tool_call_markup,
    _estimate_tokens,
    _extract_labeled_blocks,
    _extract_scientist_note,
    _format_action_span,
    _message_contains_image,
    _normalize_summary_text,
    _render_tool_call_markup,
    _render_tool_result_display,
    _strip_tool_call_markup,
)


class ToolAgentStrategyTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._controller_config = InferenceControllerConfig(enabled=True)
        return agent

    def test_usage_aliases_are_not_double_counted(self) -> None:
        agent = self._agent()
        agent._accumulate_usage_tokens({
            "prompt_tokens": 100,
            "input_tokens": 100,
            "completion_tokens": 20,
            "output_tokens": 20,
        })
        self.assertEqual(agent.total_tokens, 120)
        self.assertEqual(agent.generated_tokens, 20)

    def test_generated_tokens_alias_contributes_to_total(self) -> None:
        agent = self._agent()
        agent._accumulate_usage_tokens({
            "input_tokens": 5,
            "generated_tokens": 3,
        })
        self.assertEqual(agent.total_tokens, 8)
        self.assertEqual(agent.generated_tokens, 3)

    def test_level_start_multimodal_mode_attaches_one_image_per_level(self) -> None:
        with mock.patch.dict(os.environ, {"MULTIMODAL_CONTEXT": "level_start"}):
            agent = self._agent()
            level_one = Frame(grid=((1, 2),), step=0, level=1)
            later_level_one = Frame(grid=((2, 1),), step=1, level=1)
            level_two = Frame(grid=((3, 4),), step=0, level=2)

            first = agent._build_user_message("first", level_one)
            later = agent._build_user_message("later", later_level_one)
            next_level = agent._build_user_message("next", level_two)

        self.assertIsInstance(first["content"], list)
        self.assertEqual(later, {"role": "user", "content": "later"})
        self.assertIsInstance(next_level["content"], list)

    def test_current_grid_multimodal_mode_still_attaches_every_turn(self) -> None:
        with mock.patch.dict(os.environ, {"MULTIMODAL_CONTEXT": "current_grid"}):
            agent = self._agent()
            frame = Frame(grid=((1,),), step=0, level=1)

            first = agent._build_user_message("first", frame)
            second = agent._build_user_message("second", frame)

        self.assertIsInstance(first["content"], list)
        self.assertIsInstance(second["content"], list)

    def test_multimodal_history_omits_image_bytes_but_keeps_text(self) -> None:
        original = {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect this grid"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + ("A" * 100_000)},
                },
            ],
        }

        compact = _compact_multimodal_history_message(original)

        rendered = json.dumps(compact)
        self.assertIn("inspect this grid", rendered)
        self.assertIn("image omitted from retained history", rendered)
        self.assertNotIn("data:image/png;base64", rendered)
        self.assertLess(_estimate_tokens(compact), _estimate_tokens(original) // 100)

    def test_failed_request_does_not_consume_level_start_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            frame = Frame(grid=((1, 2),), step=0, level=1)
            write_runtime_state(
                state_path,
                current_frame=frame,
                history=[HistoryEntry(action="", frame=frame)],
            )
            with mock.patch.dict(os.environ, {"MULTIMODAL_CONTEXT": "level_start"}):
                agent = self._agent()
                agent._tool_steps = 1
                with mock.patch.object(
                    agent,
                    "_chat_completion",
                    side_effect=requests.ConnectionError("unavailable"),
                ):
                    result = agent.analyze(state_path, 0, valid_actions=["LEFT"])

        self.assertTrue(result.retryable_failure)
        self.assertIsNone(agent._last_image_level)

    def test_stop_before_first_request_does_not_commit_unsent_history(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            frame = Frame(grid=((1, 2),), step=0, level=1)
            write_runtime_state(
                state_path,
                current_frame=frame,
                history=[HistoryEntry(action="", frame=frame)],
            )
            with mock.patch.dict(os.environ, {"MULTIMODAL_CONTEXT": "level_start"}):
                agent = self._agent()
                agent._ensure_session(state_path)
                previous_history = [
                    {"role": "user", "content": "previous turn"},
                    {"role": "assistant", "content": "previous answer"},
                ]
                agent._history_messages = list(previous_history)
                with mock.patch.object(agent, "_chat_completion") as completion:
                    result = agent.analyze(
                        state_path,
                        0,
                        valid_actions=["LEFT"],
                        should_stop=lambda: True,
                    )

        completion.assert_not_called()
        self.assertTrue(result.yielded_control)
        self.assertEqual(agent._history_messages, previous_history)
        self.assertIsNone(agent._last_image_level)

    def test_context_trimmed_image_is_not_marked_as_delivered(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            frame = Frame(grid=((1, 2),), step=0, level=1)
            write_runtime_state(
                state_path,
                current_frame=frame,
                history=[HistoryEntry(action="", frame=frame)],
            )
            huge_image = {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + ("A" * 100_000)},
            }
            captured_messages: list[list[dict]] = []

            def complete(messages: list[dict], **_kwargs: object) -> _ChatCompletionResult:
                captured_messages.append(json.loads(json.dumps(messages)))
                return _ChatCompletionResult(message={"content": "no action"})

            with (
                mock.patch.dict(os.environ, {"MULTIMODAL_CONTEXT": "level_start"}),
                mock.patch(
                    "inference.agent.tool_agent.current_grid_image_part",
                    return_value=huge_image,
                ),
            ):
                agent = self._agent()
                agent._tool_steps = 1
                agent._context_budget_tokens = 6_000
                with mock.patch.object(agent, "_chat_completion", side_effect=complete):
                    agent.analyze(state_path, 0, valid_actions=["LEFT"])

        self.assertFalse(any(_message_contains_image(item) for item in captured_messages[0]))
        self.assertIsNone(agent._last_image_level)

    def test_tool_schema_and_token_estimate_are_cached_per_decode_mode(self) -> None:
        agent = self._agent()
        state_path = Path("state.json")
        fast_tools = agent._tools(state_path, strict=False)
        strict_tools = agent._tools(state_path, strict=True)

        self.assertIs(fast_tools, agent._tools(state_path, strict=False))
        self.assertIs(strict_tools, agent._tools(state_path, strict=True))
        self.assertIsNot(fast_tools, strict_tools)
        first = agent._estimate_request_input_tokens(
            [{"role": "user", "content": "inspect"}],
            tools=fast_tools,
        )
        second = agent._estimate_request_input_tokens(
            [{"role": "user", "content": "inspect"}],
            tools=fast_tools,
        )
        self.assertEqual(first, second)
        self.assertIn(False, agent._tool_token_estimates)

    def test_disabled_request_logging_skips_prompt_snapshot_serialization(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            frame = Frame(grid=((1,),), step=0, level=1)
            write_runtime_state(
                state_path,
                current_frame=frame,
                history=[HistoryEntry(action="", frame=frame)],
            )
            agent = self._agent()
            agent._tool_steps = 1
            response = _ChatCompletionResult(
                message={"content": "inspect next"},
                finish_reason="stop",
                usage={"completion_tokens": 2},
            )
            with (
                mock.patch.object(agent, "_chat_completion", return_value=response),
                mock.patch(
                    "inference.agent.tool_agent._write_prompt_log_snapshot"
                ) as snapshot,
            ):
                agent.analyze(state_path, 0, valid_actions=["LEFT"])

        snapshot.assert_not_called()

    def test_no_tool_retry_compacts_failed_attempt_in_request_and_history(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            frame = Frame(grid=((1,),), step=0, level=1)
            write_runtime_state(
                state_path,
                current_frame=frame,
                history=[HistoryEntry(action="", frame=frame)],
            )
            agent = self._agent()
            agent._tool_steps = 2
            long_content = "PLAN_START " + ("c" * 8_000) + " PLAN_TAIL"
            long_reasoning = "REASON_START " + ("r" * 8_000) + " REASON_TAIL"
            responses = iter(
                [
                    _ChatCompletionResult(
                        message={"content": long_content, "reasoning": long_reasoning},
                        finish_reason="stop",
                        usage={},
                    ),
                    _ChatCompletionResult(
                        message={"content": "still no tool"},
                        finish_reason="stop",
                        usage={},
                    ),
                ]
            )
            captured_requests: list[list[dict]] = []

            def complete(messages: list[dict], **_kwargs: object) -> _ChatCompletionResult:
                captured_requests.append(json.loads(json.dumps(messages)))
                return next(responses)

            with mock.patch.object(agent, "_chat_completion", side_effect=complete):
                agent.analyze(state_path, 0, valid_actions=["LEFT"])

        retry_messages = captured_requests[1]
        compact = next(message for message in retry_messages if message.get("role") == "assistant")
        self.assertNotIn("reasoning", compact)
        self.assertLess(len(compact["content"]), 400)
        self.assertIn("compacted for retry", compact["content"])
        retry_payload = json.dumps(retry_messages)
        history_payload = json.dumps(agent._history_messages)
        self.assertNotIn("PLAN_TAIL", retry_payload)
        self.assertNotIn("REASON_TAIL", retry_payload)
        self.assertNotIn("PLAN_TAIL", history_payload)
        self.assertNotIn("REASON_TAIL", history_payload)

    def test_no_tool_retry_compactor_uses_reasoning_when_content_is_empty(self) -> None:
        compact = _compact_no_tool_retry_message("", "reasoning-only plan")
        self.assertEqual(compact["role"], "assistant")
        self.assertIn("reasoning-only plan", compact["content"])
        self.assertNotIn("reasoning", compact.keys())

    def test_runtime_state_uses_compact_internal_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            frame = Frame(grid=((1, 2), (3, 4)), step=0, level=1)
            write_runtime_state(
                state_path,
                current_frame=frame,
                history=[HistoryEntry(action="", frame=frame)],
            )

            encoded = state_path.read_text(encoding="utf-8")

        self.assertNotIn("\n", encoded)
        self.assertNotIn(": ", encoded)
        self.assertIn('"current_frame":', encoded)

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
        self.assertIn("Phase: orient", prompt)
        self.assertIn("budget: 1 actions", prompt)
        self.assertIn("Experience controller:", prompt)
        self.assertNotIn("[[1, 2], [3, 4]]", prompt)
        self.assertLess(len(prompt), 10_000)

    def test_turn_prompt_does_not_repeat_cached_system_instructions(self) -> None:
        agent = self._agent()
        frame = Frame(grid=((1, 2), (3, 4)), step=0, level=1)

        prompt = agent._build_user_prompt(
            0,
            valid_actions=["LEFT", "RIGHT", "SPACE"],
            current_frame=frame,
            history_entries=[HistoryEntry(action="", frame=frame)],
            experience_snapshot=None,
        )

        self.assertLess(_estimate_tokens(prompt), 250)
        self.assertNotIn("Before writing code", prompt)
        self.assertNotIn(TOOL_CALL_FORMAT_GUIDANCE, prompt)
        self.assertNotIn("You may call `action(actions)` more than once", prompt)
        self.assertIn("Act now with minimal ProgramIR", prompt)

    def test_context_trim_drops_summary_if_summary_would_exceed_budget(self) -> None:
        agent = self._agent()
        system = {"role": "system", "content": "system"}
        current = {"role": "user", "content": "current"}
        messages = [
            system,
            {"role": "user", "content": "old" * 1000},
            {"role": "assistant", "content": "response"},
            current,
        ]
        agent._context_budget_tokens = agent._estimate_request_input_tokens(
            [system, current]
        )

        trimmed = agent._trim_messages_for_context(messages)

        self.assertEqual(trimmed, [system, current])
        self.assertLessEqual(
            agent._estimate_request_input_tokens(trimmed),
            agent._context_budget_tokens,
        )

    def test_context_trim_replaces_oversized_current_turn_with_regrounding(self) -> None:
        agent = self._agent()
        system = {"role": "system", "content": "system"}
        oversized = {"role": "user", "content": "state" * 5000}
        fallback = {
            "role": "user",
            "content": (
                "[Current turn content exceeded the input budget. Call python with a small "
                "ProgramIR inspection and re-ground from current_frame, experience, and "
                "valid_actions.]"
            ),
        }
        agent._context_budget_tokens = agent._estimate_request_input_tokens(
            [system, fallback]
        )

        trimmed = agent._trim_messages_for_context(
            [system, oversized],
            preserve_recent=1,
        )

        self.assertEqual(trimmed, [system, fallback])
        self.assertLessEqual(
            agent._estimate_request_input_tokens(trimmed),
            agent._context_budget_tokens,
        )

    def test_context_trim_fails_locally_when_base_request_cannot_fit(self) -> None:
        agent = self._agent()
        agent._context_budget_tokens = 1

        with self.assertRaisesRegex(ValueError, "System prompt and tool schema exceed"):
            agent._trim_messages_for_context([
                {"role": "system", "content": "system"},
                {"role": "user", "content": "current"},
            ])


class StripToolCallMarkupTests(TestCase):
    def test_strips_block(self) -> None:
        text = "analysis\n<tool_call>\n<function=python>\n\ncode\n</function>\n</tool_call>"
        stripped = _strip_tool_call_markup(text)
        self.assertIn("analysis", stripped)
        self.assertNotIn("<tool_call>", stripped)

    def test_no_markup_unchanged(self) -> None:
        text = "just plain text"
        self.assertEqual(_strip_tool_call_markup(text), text)

    def test_empty_text(self) -> None:
        self.assertEqual(_strip_tool_call_markup(""), "")

    def test_whitespace_only(self) -> None:
        self.assertEqual(_strip_tool_call_markup("   \n  "), "")

    def test_multiple_blocks(self) -> None:
        open_tag = "<tool_call>"
        close_tag = "</tool_call>"
        fn_open = "<function=python>"
        fn_close = "</function>"
        text = (
            "a"
            + open_tag + fn_open + "\nblock1\n" + fn_close + "\n" + close_tag
            + "b"
            + open_tag + fn_open + "\nblock2\n" + fn_close + "\n" + close_tag
            + "c"
        )
        stripped = _strip_tool_call_markup(text)
        self.assertNotIn("<tool_call>", stripped)
        self.assertIn("a", stripped)
        self.assertIn("c", stripped)


class ContainsToolCallMarkupTests(TestCase):
    def test_detects_tool_call(self) -> None:
        self.assertTrue(_contains_tool_call_markup("<tool_call>"))

    def test_detects_function_tag(self) -> None:
        self.assertTrue(_contains_tool_call_markup("<function=python>"))

    def test_no_markup(self) -> None:
        self.assertFalse(_contains_tool_call_markup("plain text"))

    def test_empty_string(self) -> None:
        self.assertFalse(_contains_tool_call_markup(""))

    def test_case_insensitive(self) -> None:
        self.assertTrue(_contains_tool_call_markup("<tool_call>"))

    def test_multiple_chunks(self) -> None:
        self.assertTrue(_contains_tool_call_markup("nope", "still no", "<tool_call>here"))
        self.assertFalse(_contains_tool_call_markup("a", "b", "c"))


class NormalizeSummaryTextTests(TestCase):
    def test_short_text_unchanged(self) -> None:
        result = _normalize_summary_text("hello", max_chars=100)
        self.assertEqual(result, "hello")

    def test_long_text_truncated(self) -> None:
        result = _normalize_summary_text("a" * 200, max_chars=50)
        self.assertIn("omitted", result)
        self.assertTrue(len(result) > 50)

    def test_none_becomes_empty(self) -> None:
        result = _normalize_summary_text(None)
        self.assertEqual(result, "")

    def test_whitespace_normalized(self) -> None:
        result = _normalize_summary_text("  hello   world  ")
        self.assertEqual(result, "hello world")

    def test_no_limit(self) -> None:
        result = _normalize_summary_text("a" * 1000, max_chars=None)
        self.assertEqual(len(result), 1000)

    def test_zero_max_chars(self) -> None:
        result = _normalize_summary_text("hello", max_chars=0)
        self.assertEqual(result, "hello")

    def test_numeric_value(self) -> None:
        result = _normalize_summary_text(42)
        self.assertEqual(result, "42")

    def test_list_value(self) -> None:
        result = _normalize_summary_text([1, 2, 3])
        self.assertIn("1", result)


class FormatActionSpanTests(TestCase):
    def test_same_start_end(self) -> None:
        result = _format_action_span(3, 3)
        self.assertEqual(result, "3")

    def test_different_start_end(self) -> None:
        result = _format_action_span(2, 5)
        self.assertEqual(result, "2-5")

    def test_none_values(self) -> None:
        self.assertIsNone(_format_action_span(None, 5))
        self.assertIsNone(_format_action_span(3, None))

    def test_zero_values(self) -> None:
        self.assertIsNone(_format_action_span(0, 5))
        self.assertIsNone(_format_action_span(3, 0))

    def test_negative_values(self) -> None:
        self.assertIsNone(_format_action_span(-1, 5))
        self.assertIsNone(_format_action_span(3, -1))

    def test_both_none(self) -> None:
        self.assertIsNone(_format_action_span(None, None))


class BuildSystemPromptTests(TestCase):
    def test_prompt_contains_key_sections(self) -> None:
        prompt = _build_system_prompt(tool_output_tokens=1024)
        self.assertIn("coding agent", prompt)
        self.assertIn("Game overview", prompt)
        self.assertIn("Runtime variables", prompt)
        self.assertIn("Python tool guidance", prompt)
        self.assertIn("Tool session rules", prompt)

    def test_prompt_contains_tool_output_tokens(self) -> None:
        prompt = _build_system_prompt(tool_output_tokens=2048)
        self.assertIn("2048", prompt)

    def test_prompt_contains_color_legend(self) -> None:
        prompt = _build_system_prompt(tool_output_tokens=1024)
        self.assertIn("Color legend", prompt)

    def test_prompt_documents_every_protected_runtime_binding(self) -> None:
        prompt = _build_system_prompt(tool_output_tokens=1024)
        binding_line = next(
            line for line in prompt.splitlines()
            if "Available protected runtime bindings are:" in line
        )
        for binding in PROTECTED_RUNTIME_BINDINGS:
            with self.subTest(binding=binding):
                self.assertIn(binding, binding_line)

    def test_prompt_documents_every_grid_helper_signature(self) -> None:
        prompt = _build_system_prompt(tool_output_tokens=1024)
        for signature in RUNTIME_HELPER_SIGNATURES:
            with self.subTest(signature=signature):
                self.assertIn(signature, prompt)

    def test_prompt_allows_locally_defined_program_names(self) -> None:
        prompt = _build_system_prompt(tool_output_tokens=1024)
        self.assertIn("names defined earlier in the same program", prompt)
        self.assertNotIn("Only access variables that are listed", prompt)

    def test_small_context_prompt_preserves_core_codegen_contract(self) -> None:
        compact = _build_system_prompt(
            tool_output_tokens=512,
            context_window_tokens=8192,
        )
        full = _build_system_prompt(tool_output_tokens=512)

        self.assertLess(len(compact), len(full) // 2)
        for required in (
            "ProgramIR",
            "version 1",
            "kind discriminator",
            "current_frame.segmentation",
            "previous_frame",
            "action([",
            "MOUSE",
            "recovery_hint",
            "run_complete",
        ):
            with self.subTest(required=required):
                self.assertIn(required, compact)

    def test_32k_kaggle_context_uses_compact_system_prompt(self) -> None:
        compact = _build_system_prompt(
            tool_output_tokens=512,
            context_window_tokens=32768,
        )
        full = _build_system_prompt(tool_output_tokens=512)

        self.assertLess(len(compact), len(full) // 2)
        self.assertIn("ProgramIR", compact)

    def test_persistent_history_has_fixed_prefill_budget(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "system"}
        ]
        for index in range(12):
            messages.extend(
                [
                    {"role": "user", "content": f"turn {index}"},
                    {"role": "assistant", "content": "p" * 2000},
                    {"role": "tool", "content": "r" * 2000},
                ]
            )

        history = agent._persistent_history_messages(messages)

        self.assertLessEqual(_estimate_tokens(history), 4096)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[-1]["role"], "tool")
        self.assertIn("turn 11", str(history))

    def test_program_ir_request_has_usable_headroom_in_8k_context(self) -> None:
        compact = _build_system_prompt(
            tool_output_tokens=512,
            context_window_tokens=8192,
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "python",
                "description": "Compile and execute structured ProgramIR.",
                "parameters": program_tool_parameters_schema(),
                "strict": True,
            },
        }]
        base_request = {
            "messages": [
                {"role": "system", "content": compact},
                {"role": "user", "content": ""},
            ],
            "tools": tools,
            "tool_choice": "auto",
        }

        input_budget = 8192 - 512 - 512
        self.assertLess(_estimate_tokens(base_request), input_budget - 1500)


class EstimateTokensTests(TestCase):
    def test_empty_value(self) -> None:
        self.assertEqual(_estimate_tokens(""), 1)

    def test_ascii_heavy_string(self) -> None:
        tokens = _estimate_tokens("hello world")
        self.assertGreater(tokens, 0)

    def test_json_dict(self) -> None:
        tokens = _estimate_tokens({"key": "value", "count": 42})
        self.assertGreater(tokens, 0)

    def test_nested_structure(self) -> None:
        tokens = _estimate_tokens({"a": [1, 2, 3], "b": {"c": "d"}})
        self.assertGreater(tokens, 0)

    def test_non_ascii_content(self) -> None:
        tokens = _estimate_tokens("日本語テスト")
        self.assertGreater(tokens, 0)

    def test_none_value(self) -> None:
        tokens = _estimate_tokens(None)
        self.assertGreaterEqual(tokens, 0)

    def test_numeric_value(self) -> None:
        tokens = _estimate_tokens(42)
        self.assertGreater(tokens, 0)

    def test_list_value(self) -> None:
        tokens = _estimate_tokens([1, 2, 3])
        self.assertGreater(tokens, 0)

    def test_empty_dict(self) -> None:
        tokens = _estimate_tokens({})
        self.assertEqual(tokens, 1)


class RenderToolCallMarkupTests(TestCase):
    def test_basic_python_call(self) -> None:
        markup = _render_tool_call_markup("python", {"code": "print(42)"})
        self.assertIn("<tool_call>", markup)
        self.assertIn("<function=python>", markup)
        self.assertIn("print(42)", markup)
        self.assertIn("</tool_call>", markup)

    def test_empty_tool_name_returns_empty(self) -> None:
        markup = _render_tool_call_markup("", {"code": "test"})
        self.assertEqual(markup, "")

    def test_none_arguments_returns_empty(self) -> None:
        markup = _render_tool_call_markup("python", None)
        self.assertEqual(markup, "")

    def test_dict_arguments_formatted(self) -> None:
        markup = _render_tool_call_markup("python", {"code": "x = 1"})
        self.assertIn("x = 1", markup)

    def test_multiple_parameters(self) -> None:
        markup = _render_tool_call_markup("python", {"code": "x = 1", "timeout": 5})
        self.assertIn("code", markup)
        self.assertIn("timeout", markup)

    def test_invalid_arguments_returns_empty(self) -> None:
        markup = _render_tool_call_markup("python", object())
        self.assertEqual(markup, "")


class RenderToolResultDisplayTests(TestCase):
    def test_string_content(self) -> None:
        display = _render_tool_result_display("hello world")
        self.assertEqual(display, "hello world")

    def test_dict_with_stdout(self) -> None:
        display = _render_tool_result_display({"stdout": "output", "error": ""})
        self.assertEqual(display, "output")

    def test_dict_with_error(self) -> None:
        display = _render_tool_result_display({"error": "something failed"})
        self.assertIn("something failed", display)

    def test_dict_with_result(self) -> None:
        display = _render_tool_result_display({"result": {"key": "value"}})
        self.assertIn("key", display)

    def test_json_string_content(self) -> None:
        display = _render_tool_result_display('{"key": "value"}')
        self.assertIn("key", display)

    def test_none_content(self) -> None:
        display = _render_tool_result_display(None)
        self.assertIsNotNone(display)

    def test_dict_with_stdout_and_error(self) -> None:
        display = _render_tool_result_display({"stdout": "out", "error": "err"})
        self.assertIn("out", display)
        self.assertIn("err", display)

    def test_empty_dict(self) -> None:
        display = _render_tool_result_display({})
        self.assertIsNotNone(display)


class ExtractScientistNoteTests(TestCase):
    def test_empty_content(self) -> None:
        result = _extract_scientist_note("")
        self.assertEqual(result, {})

    def test_whitespace_only(self) -> None:
        result = _extract_scientist_note("   \n  ")
        self.assertEqual(result, {})

    def test_extracts_world_model(self) -> None:
        result = _extract_scientist_note("World model: objects are red")
        self.assertIn("red", result.get("world_model", ""))

    def test_extracts_goal_model(self) -> None:
        result = _extract_scientist_note("Goal model: reach the target")
        self.assertIn("target", result.get("goal_model", ""))

    def test_extracts_plan(self) -> None:
        result = _extract_scientist_note("Plan: press SPACE")
        self.assertIn("SPACE", result.get("current_plan", ""))

    def test_falls_back_to_hypothesis(self) -> None:
        result = _extract_scientist_note("Hypothesis: objects move right")
        self.assertIn("right", result.get("world_model", ""))

    def test_falls_back_to_next_test(self) -> None:
        result = _extract_scientist_note("Next test: try LEFT")
        self.assertIn("LEFT", result.get("current_plan", ""))

    def test_multiple_labels(self) -> None:
        content = "World model: red objects\nGoal model: clear board"
        result = _extract_scientist_note(content)
        self.assertIn("red", result.get("world_model", ""))
        self.assertIn("clear", result.get("goal_model", ""))


class ExtractLabeledBlocksTests(TestCase):
    def test_empty_content(self) -> None:
        result = _extract_labeled_blocks("", ["Label"])
        self.assertEqual(result, {})

    def test_extracts_label(self) -> None:
        result = _extract_labeled_blocks("Label: value", ["Label"])
        self.assertEqual(result.get("Label"), "value")

    def test_multiline_value(self) -> None:
        content = "Label: line1\nline2\nline3"
        result = _extract_labeled_blocks(content, ["Label"])
        self.assertIn("line1", result.get("Label", ""))

    def test_bullet_points(self) -> None:
        content = "- Label: value1\n* Label: value2"
        result = _extract_labeled_blocks(content, ["Label"])
        self.assertIn("value1", result.get("Label", ""))


class SummarizedKnowledgeLinesTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_empty_knowledge_returns_empty_list(self) -> None:
        agent = self._agent()
        lines = agent._summarized_knowledge_lines()
        self.assertEqual(lines, [])

    def test_populated_knowledge_returns_lines(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "objects move right"
        agent._summarized_knowledge["goal_model"] = "clear the board"
        lines = agent._summarized_knowledge_lines()
        self.assertGreater(len(lines), 0)
        self.assertTrue(any("world model" in line.lower() for line in lines))

    def test_long_values_are_truncated(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "x" * 500
        lines = agent._summarized_knowledge_lines()
        full_text = "\n".join(lines)
        self.assertLess(len(full_text), 600)

    def test_single_knowledge_field(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["action_model"] = "LEFT moves left"
        lines = agent._summarized_knowledge_lines()
        self.assertGreater(len(lines), 0)


class UpdateKnowledgeFromAssistantTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_extracts_world_model(self) -> None:
        agent = self._agent()
        agent._update_summarized_knowledge_from_assistant(
            "World model: objects are red squares that move right"
        )
        self.assertIn("red squares", agent._summarized_knowledge["world_model"])

    def test_extracts_goal_model(self) -> None:
        agent = self._agent()
        agent._update_summarized_knowledge_from_assistant(
            "Goal model: reach the blue target"
        )
        self.assertIn("blue target", agent._summarized_knowledge["goal_model"])

    def test_empty_content_no_update(self) -> None:
        agent = self._agent()
        agent._update_summarized_knowledge_from_assistant("")
        self.assertEqual(agent._summarized_knowledge["world_model"], "")

    def test_no_labels_no_update(self) -> None:
        agent = self._agent()
        agent._update_summarized_knowledge_from_assistant("Just some random text")
        self.assertEqual(agent._summarized_knowledge["world_model"], "")

    def test_extracts_plan(self) -> None:
        agent = self._agent()
        agent._update_summarized_knowledge_from_assistant("Plan: try RIGHT next")
        self.assertIn("RIGHT", agent._summarized_knowledge["current_plan"])

    def test_extracts_cross_level_notes(self) -> None:
        agent = self._agent()
        agent._update_summarized_knowledge_from_assistant(
            "Cross-level notes: color mapping is consistent"
        )
        self.assertIn("color mapping", agent._summarized_knowledge["cross_level_notes"])


class UpdateKnowledgeFromStepSummaryTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_board_change_updates_recent_findings(self) -> None:
        agent = self._agent()
        agent._last_step_summary = {"board_changed": True, "level": 1}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertIn("Board changed", agent._summarized_knowledge["recent_findings"])

    def test_level_transition_clears_knowledge(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "old model"
        agent._summarized_knowledge["goal_model"] = "old goal"
        agent._last_step_summary = {"level_transition": True, "level": 2}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")
        self.assertEqual(agent._summarized_knowledge["goal_model"], "")

    def test_no_summary_no_op(self) -> None:
        agent = self._agent()
        agent._last_step_summary = None
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")

    def test_run_complete_clears_knowledge(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "model"
        agent._last_step_summary = {"run_complete": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["world_model"], "")

    def test_game_over_clears_knowledge(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["goal_model"] = "goal"
        agent._last_step_summary = {"game_over": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["goal_model"], "")

    def test_level_transition_preserves_cross_level_notes(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["cross_level_notes"] = "important note"
        agent._summarized_knowledge["world_model"] = "old"
        agent._last_step_summary = {"level_transition": True, "level": 2}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["cross_level_notes"], "important note")
        self.assertEqual(agent._summarized_knowledge["world_model"], "")

    def test_game_over_clears_cross_level_notes(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["cross_level_notes"] = "note"
        agent._last_step_summary = {"game_over": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertEqual(agent._summarized_knowledge["cross_level_notes"], "")


class RecordStrategyEdgeCasesTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_empty_update_preserves_existing(self) -> None:
        agent = self._agent()
        agent._strategy_memory["goal"] = "existing goal"
        saved = agent._record_strategy({})
        self.assertEqual(saved["goal"], "existing goal")

    def test_confidence_clamped_to_zero(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"confidence": -5})
        self.assertEqual(saved["confidence"], 0.0)

    def test_confidence_clamped_to_one(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"confidence": 10})
        self.assertEqual(saved["confidence"], 1.0)

    def test_invalid_confidence_ignored(self) -> None:
        agent = self._agent()
        agent._strategy_memory["confidence"] = 0.5
        saved = agent._record_strategy({"confidence": "not a number"})
        self.assertEqual(saved["confidence"], 0.5)

    def test_evidence_capped_at_five(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({
            "evidence": ["a", "b", "c", "d", "e", "f", "g"]
        })
        self.assertLessEqual(len(saved["evidence"]), 5)

    def test_contradictions_capped_at_three(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({
            "contradictions": ["a", "b", "c", "d", "e"]
        })
        self.assertLessEqual(len(saved["contradictions"]), 3)

    def test_test_action_normalizes_key(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"test_action": "  left  "})
        self.assertEqual(saved["test_action"], "LEFT")

    def test_expected_outcome_valid(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"expected_outcome": "state_change"})
        self.assertEqual(saved["expected_outcome"], "state_change")

    def test_expected_outcome_invalid_ignored(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"expected_outcome": "invalid_value"})
        self.assertNotIn("expected_outcome", saved)

    def test_clears_prediction_on_new_test_action(self) -> None:
        agent = self._agent()
        agent._strategy_memory["prediction_result"] = {"status": "supported"}
        agent._record_strategy({"test_action": "RIGHT"})
        self.assertNotIn("prediction_result", agent._strategy_memory)

    def test_long_goal_truncated(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"goal": "x" * 500})
        self.assertLessEqual(len(saved["goal"]), 280)

    def test_single_string_evidence(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"evidence": "single item"})
        self.assertEqual(len(saved["evidence"]), 1)

    def test_single_string_contradiction(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"contradictions": "single item"})
        self.assertEqual(len(saved["contradictions"]), 1)

    def test_updates_world_model_from_hypothesis(self) -> None:
        agent = self._agent()
        agent._record_strategy({"hypothesis": "objects move right"})
        self.assertIn("objects move right", agent._summarized_knowledge["world_model"])

    def test_updates_goal_model(self) -> None:
        agent = self._agent()
        agent._record_strategy({"goal": "clear the board"})
        self.assertIn("clear the board", agent._summarized_knowledge["goal_model"])


class EvaluateStrategyPredictionEdgeCasesTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_no_test_action_returns_none(self) -> None:
        agent = self._agent()
        result = agent._evaluate_strategy_prediction({"executed": True})
        self.assertIsNone(result)

    def test_no_expected_outcome_returns_none(self) -> None:
        agent = self._agent()
        agent._strategy_memory["test_action"] = "LEFT"
        result = agent._evaluate_strategy_prediction({"executed": True})
        self.assertIsNone(result)

    def test_existing_prediction_result_returned(self) -> None:
        agent = self._agent()
        existing = {"status": "supported", "action": "LEFT"}
        result = agent._evaluate_strategy_prediction({
            "executed": True,
            "prediction_result": existing,
        })
        self.assertEqual(result["status"], "supported")

    def test_matching_outcome_supported(self) -> None:
        agent = self._agent()
        agent._record_strategy({
            "test_action": "LEFT",
            "expected_outcome": "state_change",
        })
        result = agent._evaluate_strategy_prediction({
            "executed": True,
            "action_display": "LEFT",
            "outcome_class": "state_change",
            "board_changed": True,
        })
        self.assertEqual(result["status"], "supported")

    def test_non_matching_outcome_contradicted(self) -> None:
        agent = self._agent()
        agent._record_strategy({
            "test_action": "LEFT",
            "expected_outcome": "state_change",
        })
        result = agent._evaluate_strategy_prediction({
            "executed": True,
            "action_display": "LEFT",
            "outcome_class": "no_change",
            "board_changed": False,
        })
        self.assertEqual(result["status"], "contradicted")

    def test_action_not_found_in_steps(self) -> None:
        agent = self._agent()
        agent._record_strategy({
            "test_action": "LEFT",
            "expected_outcome": "state_change",
        })
        result = agent._evaluate_strategy_prediction({
            "executed": True,
            "steps": [{"action_display": "RIGHT", "outcome_class": "state_change"}],
        })
        self.assertIsNone(result)

    def test_mouse_family_matching(self) -> None:
        agent = self._agent()
        agent._record_strategy({
            "test_action": "MOUSE",
            "expected_outcome": "state_change",
        })
        result = agent._evaluate_strategy_prediction({
            "executed": True,
            "action_display": "MOUSE(row=3, col=4)",
            "outcome_class": "state_change",
            "board_changed": True,
        })
        self.assertEqual(result["status"], "supported")


class CompactActionResultEdgeCasesTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_none_payload_handled(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({})
        self.assertFalse(compact["executed"])

    def test_all_terminal_flags(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "run_complete": True,
            "level_completed": True,
            "game_over": True,
        })
        self.assertTrue(compact["run_complete"])
        self.assertTrue(compact["level_completed"])
        self.assertTrue(compact["game_over"])

    def test_executed_actions_preserved(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "executed_actions": ["LEFT", "RIGHT", "UP"],
        })
        self.assertEqual(compact["executed_actions"], ["LEFT", "RIGHT", "UP"])

    def test_requested_actions_preserved_separately_and_bounded(self) -> None:
        agent = self._agent()
        requested = [f"ACTION_{index}" for index in range(MAX_ACTION_BATCH + 5)]
        compact = agent._compact_action_result({
            "executed": True,
            "requested_actions": requested,
            "executed_actions": ["ACTION_0"],
        })
        self.assertEqual(compact["requested_actions"], requested[:MAX_ACTION_BATCH])
        self.assertEqual(compact["executed_actions"], ["ACTION_0"])

    def test_controller_keys_preserved(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "controller_phase": "progress",
            "outcome_class": "novel",
            "action_rank": 1,
            "prediction_result": {"status": "supported"},
        })
        self.assertEqual(compact["controller_phase"], "progress")
        self.assertEqual(compact["outcome_class"], "novel")
        self.assertEqual(compact["action_rank"], 1)
        self.assertEqual(compact["prediction_result"]["status"], "supported")

    def test_steps_list_truncated(self) -> None:
        agent = self._agent()
        steps = [{"executed": True, "step": i} for i in range(20)]
        compact = agent._compact_action_result({"executed": True, "steps": steps})
        self.assertLessEqual(len(compact["steps"]), 12)

    def test_stop_reason_and_detail(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "stop_reason": "max_actions",
            "stop_detail": "reached limit",
        })
        self.assertEqual(compact["stop_reason"], "max_actions")
        self.assertEqual(compact["stop_detail"], "reached limit")

    def test_timing_keys(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "run_elapsed_seconds": 1.5,
            "time_remaining_seconds": 10.0,
        })
        self.assertEqual(compact["run_elapsed_seconds"], 1.5)
        self.assertEqual(compact["time_remaining_seconds"], 10.0)

    def test_error_preserved(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": False,
            "error": "something failed",
        })
        self.assertEqual(compact["error"], "something failed")

    def test_batch_size_from_executed_count(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "executed_count": 5,
        })
        self.assertEqual(compact["requested_count"], 5)

    def test_non_executed_with_action_display(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": False,
            "action_display": "LEFT",
            "requested_actions": ["LEFT"],
        })
        self.assertEqual(compact["requested_actions"], ["LEFT"])
        self.assertEqual(compact["executed_actions"], [])


class NormalizePythonActionsEdgeTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_string_action(self) -> None:
        agent = self._agent()
        result = agent._normalize_python_actions("LEFT")
        self.assertEqual(result, [{"action": "LEFT"}])

    def test_dict_action(self) -> None:
        agent = self._agent()
        result = agent._normalize_python_actions({"action": "MOUSE", "row": 3, "col": 7})
        self.assertEqual(result, [{"action": "MOUSE", "row": 3, "col": 7}])

    def test_list_of_strings(self) -> None:
        agent = self._agent()
        result = agent._normalize_python_actions(["LEFT", "RIGHT"])
        self.assertEqual(result, [{"action": "LEFT"}, {"action": "RIGHT"}])

    def test_list_of_dicts(self) -> None:
        agent = self._agent()
        result = agent._normalize_python_actions([
            {"action": "LEFT"},
            {"action": "MOUSE", "row": 1, "col": 2},
        ])
        self.assertEqual(len(result), 2)

    def test_empty_list_raises_error(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "at least one"):
            agent._normalize_python_actions([])

    def test_legacy_xy_fields_rejected(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "legacy"):
            agent._normalize_python_actions({"action": "MOUSE", "x": 3, "y": 7})

    def test_mouse_requires_both_strict_integer_coordinates(self) -> None:
        agent = self._agent()
        invalid_actions = (
            {"action": "MOUSE", "row": 3},
            {"action": "MOUSE", "row": "3", "col": 7},
            {"action": "MOUSE", "row": True, "col": 7},
        )
        for action in invalid_actions:
            with self.subTest(action=action), self.assertRaisesRegex(
                ValueError,
                "requires integer row and col",
            ):
                agent._normalize_python_actions(action)

    def test_non_mouse_coordinates_are_rejected_instead_of_ignored(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "only valid for MOUSE"):
            agent._normalize_python_actions({"action": "LEFT", "row": 1, "col": 2})

    def test_unknown_action_fields_are_rejected_instead_of_ignored(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "unsupported field.*note"):
            agent._normalize_python_actions({"action": "LEFT", "note": "probe"})

    def test_non_action_dict_rejected(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "missing"):
            agent._normalize_python_actions({"row": 3})

    def test_batch_limit_enforced(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "at most"):
            agent._normalize_python_actions(["LEFT"] * (MAX_ACTION_BATCH + 1))

    def test_whitespace_trimmed(self) -> None:
        agent = self._agent()
        result = agent._normalize_python_actions("  LEFT  ")
        self.assertEqual(result, [{"action": "LEFT"}])

    def test_empty_string_action_raises(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "empty"):
            agent._normalize_python_actions("")

    def test_numeric_action_becomes_string(self) -> None:
        agent = self._agent()
        with self.assertRaises(TypeError):
            agent._normalize_python_actions(123)

    def test_invalid_type_raises(self) -> None:
        agent = self._agent()
        with self.assertRaises(TypeError):
            agent._normalize_python_actions(3.14)

    def test_tuple_input(self) -> None:
        agent = self._agent()
        result = agent._normalize_python_actions(("LEFT", "RIGHT"))
        self.assertEqual(len(result), 2)


class CompressExperienceSnapshotEdgeTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_empty_snapshot_returns_empty(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({})
        self.assertEqual(result, "")

    def test_disabled_snapshot_returns_empty(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({"enabled": False})
        self.assertEqual(result, "")

    def test_basic_snapshot_contains_phase(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "explore",
            "action_budget": 2,
        })
        self.assertIn("Phase: explore", result)
        self.assertIn("budget: 2", result)

    def test_no_op_streak_shown(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "recover",
            "action_budget": 1,
            "behavioral_no_op_streak": 5,
        })
        self.assertIn("No-op streak: 5", result)

    def test_suggested_actions_shown(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "orient",
            "action_budget": 1,
            "suggested_actions": ["LEFT", "RIGHT"],
        })
        self.assertIn("Suggested: LEFT, RIGHT", result)

    def test_discouraged_actions_shown(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "explore",
            "action_budget": 1,
            "discouraged_actions": ["UP"],
        })
        self.assertIn("Discouraged: UP", result)

    def test_outcome_shown(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "explore",
            "action_budget": 1,
            "latest_outcome": "novel",
        })
        self.assertIn("Last outcome: novel", result)

    def test_state_counts_shown(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "orient",
            "action_budget": 1,
            "state_visits": 5,
            "unique_states": 3,
        })
        self.assertIn("States: 3 unique, 5 visits", result)

    def test_tried_here_with_change_counts(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "explore",
            "action_budget": 1,
            "tried_here": {
                "LEFT": {"trials": 3, "changes": 2, "no_ops": 1},
                "RIGHT": {"trials": 1, "changes": 0, "no_ops": 1},
            },
        })
        self.assertIn("LEFT(3x,2change)", result)
        self.assertIn("RIGHT(1x,1noop)", result)

    def test_ranked_actions_in_snapshot(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "orient",
            "action_budget": 1,
            "ranked_actions": [
                {"action": "UP", "reason": "untried"},
                {"action": "DOWN", "reason": "change-producing"},
            ],
        })
        self.assertIn("Ranked: UP(untried), DOWN(change-producing)", result)

    def test_transition_models_in_snapshot(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "explore",
            "action_budget": 1,
            "transition_models_here": [
                {
                    "action": "LEFT",
                    "predicted_outcome": "state_change",
                    "confidence": 0.8,
                    "verified_deterministic": False,
                },
            ],
        })
        self.assertIn("LEFT", result)
        self.assertIn("state_change", result)
        self.assertIn("conf=80%", result)

    def test_deterministic_model(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "explore",
            "action_budget": 1,
            "transition_models_here": [
                {"action": "LEFT", "predicted_outcome": "change", "verified_deterministic": True}
            ],
        })
        self.assertIn("det", result)

    def test_chain_predictions_in_snapshot(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "progress",
            "action_budget": 4,
            "chain_predictions": [
                {
                    "first_action": "LEFT",
                    "second_action": "UP",
                    "chain_confidence": 0.75,
                },
            ],
        })
        self.assertIn("Chains: LEFT\u2192UP(conf=75%)", result)

    def test_cycle_info_in_snapshot(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "recover",
            "action_budget": 1,
            "cycle_period": 2,
            "cycle_confidence": 0.9,
        })
        self.assertIn("Cycle detected: period=2, confidence=90%", result)

    def test_recovery_reasons_in_snapshot(self) -> None:
        agent = self._agent()
        result = agent._compress_experience_snapshot({
            "enabled": True,
            "phase": "recover",
            "action_budget": 1,
            "recovery_reasons": ["repeated_noop", "short_cycle"],
        })
        self.assertIn("Recovery: repeated_noop, short_cycle", result)


class ValidateCodeCommonMistakesTests(TestCase):
    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        return agent

    def test_grid_access_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("print(current_frame._grid)")
        self.assertIn("_grid", warning)

    def test_too_many_actions_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes(
            "action(['LEFT'])\naction(['RIGHT'])\naction(['UP'])\naction(['DOWN'])"
        )
        self.assertIn("Many action()", warning)

    def test_infinite_loop_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("while True:\n    pass")
        self.assertIn("Infinite loop", warning)

    def test_os_import_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("import os")
        self.assertIn("not available", warning)

    def test_numpy_import_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("import numpy")
        self.assertIn("numpy", warning)

    def test_dict_access_on_history_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("x = history[0]['action']")
        self.assertIn("dict keys", warning)

    def test_action_without_list_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("action('LEFT')")
        self.assertIn("expects a list", warning)

    def test_print_full_board_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("print(current_frame.ascii)")
        self.assertIn("wastes output", warning)

    def test_clean_code_no_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes(
            "seg = current_frame.segmentation\n"
            "nodes = seg['nodes']\n"
            "action(['LEFT'])"
        )
        self.assertIsNone(warning)

    def test_numeric_while_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("while 1 < 1000000:\n    i += 1")
        self.assertIn("may timeout", warning)

    def test_sys_import_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("import sys")
        self.assertIn("not available", warning)

    def test_subprocess_import_warning(self) -> None:
        agent = self._agent()
        warning = agent._validate_code_common_mistakes("import subprocess")
        self.assertIn("not available", warning)

    def test_complex_nested_code_warning(self) -> None:
        agent = self._agent()
        code = "for i in range(10):\n    for j in range(10):\n        for k in range(10):\n            for l in range(10):\n                def f(): pass"
        warning = agent._validate_code_common_mistakes(code)
        self.assertIn("timeout", warning)

    def test_empty_code_no_warning(self) -> None:
        agent = self._agent()
        self.assertIsNone(agent._validate_code_common_mistakes(""))

    def test_single_action_no_warning(self) -> None:
        agent = self._agent()
        self.assertIsNone(agent._validate_code_common_mistakes("action(['LEFT'])"))

    def test_try_except_no_warning(self) -> None:
        agent = self._agent()
        self.assertIsNone(agent._validate_code_common_mistakes("try:\n    x = 1\nexcept:\n    pass"))


class StripToolCallMarkupEdgeTests(TestCase):
    def test_only_open_tag_no_close(self) -> None:
        text = _strip_tool_call_markup("<tool_call>some partial")
        self.assertIn("some partial", text)

    def test_close_tag_before_open(self) -> None:
        stripped = _strip_tool_call_markup("</tool_call><tool_call>block</tool_call>")
        self.assertIn("block", stripped)

    def test_adjacent_blocks(self) -> None:
        stripped = _strip_tool_call_markup("<tool_call>a</tool_call><tool_call>b</tool_call>")
        self.assertIn("a", stripped)
        self.assertIn("b", stripped)

    def test_only_close_tag(self) -> None:
        stripped = _strip_tool_call_markup("</tool_call>rest")
        self.assertIn("rest", stripped)


class ContainsToolCallMarkupEdgeTests(TestCase):
    def test_partial_tag_no_match(self) -> None:
        self.assertFalse(_contains_tool_call_markup("tool"))

    def test_similar_tag_not_match(self) -> None:
        self.assertFalse(_contains_tool_call_markup("<tool=python>"))


class NormalizeSummaryTextEdgeTests(TestCase):
    def test_non_string_numeric(self) -> None:
        self.assertEqual(_normalize_summary_text(3.14), "3.14")

    def test_empty_string(self) -> None:
        self.assertEqual(_normalize_summary_text(""), "")

    def test_string_with_only_newlines(self) -> None:
        self.assertEqual(_normalize_summary_text("\n\n\n"), "")

    def test_tabs_normalized(self) -> None:
        result = _normalize_summary_text("hello\tworld")
        self.assertIn("hello", result)
        self.assertIn("world", result)


class EstimateTokensEdgeTests(TestCase):
    def test_large_string(self) -> None:
        tokens = _estimate_tokens("x" * 100000)
        self.assertGreater(tokens, 0)

    def test_boolean_value(self) -> None:
        tokens = _estimate_tokens(True)
        self.assertGreater(tokens, 0)

    def test_nested_list_of_dicts(self) -> None:
        data = [{"key": [1, 2, {"nested": True}]} for _ in range(10)]
        self.assertGreater(_estimate_tokens(data), 0)


class RenderToolCallMarkupEdgeTests(TestCase):
    def test_multiline_code(self) -> None:
        code = "def foo():\n    return 42\nprint(foo())"
        markup = _render_tool_call_markup("python", {"code": code})
        self.assertIn(code, markup)

    def test_empty_code(self) -> None:
        markup = _render_tool_call_markup("python", {"code": ""})
        self.assertIn("<tool_call>", markup)


class RenderToolResultDisplayEdgeTests(TestCase):
    def test_empty_dict_content(self) -> None:
        display = _render_tool_result_display({})
        self.assertIsNotNone(display)

    def test_string_content(self) -> None:
        display = _render_tool_result_display("hello world")
        self.assertEqual(display, "hello world")


class ExtractScientistNoteEdgeTests(TestCase):
    def test_label_with_colons_in_value(self) -> None:
        result = _extract_scientist_note("Plan: press SPACE, then LEFT")
        self.assertIn("press SPACE, then LEFT", result.get("current_plan", ""))

    def test_multiple_plans_first_wins(self) -> None:
        result = _extract_scientist_note("Plan: first step\nPlan: second step")
        self.assertIn("first step", result.get("current_plan", ""))


class ExtractLabeledBlocksEdgeTests(TestCase):
    def test_label_at_end_of_content(self) -> None:
        result = _extract_labeled_blocks("other text\nLabel: value", ["Label"])
        self.assertEqual(result.get("Label"), "value")


class BuildSystemPromptEdgeTests(TestCase):
    def test_large_tool_output_tokens(self) -> None:
        prompt = _build_system_prompt(tool_output_tokens=1000000)
        self.assertIn("1000000", prompt)

    def test_zero_tool_output_tokens(self) -> None:
        prompt = _build_system_prompt(tool_output_tokens=0)
        self.assertIn("0", prompt)


class RecordStrategyAdditionalEdgeTests(TestCase):
    def _agent(self) -> ToolAgent:
        return ToolAgent(model="m", provider="vllm", base_url="http://127.0.0.1:1/v1")

    def test_float_confidence_within_bounds(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"confidence": 0.75})
        self.assertEqual(saved["confidence"], 0.75)

    def test_next_test_stored(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"next_test": "press RIGHT"})
        self.assertEqual(saved.get("next_test"), "press RIGHT")

    def test_fallback_stored(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"fallback": "try SPACE"})
        self.assertEqual(saved.get("fallback"), "try SPACE")

    def test_long_open_question_truncated(self) -> None:
        agent = self._agent()
        saved = agent._record_strategy({"open_question": "q" * 500})
        self.assertLessEqual(len(saved["open_question"]), 280)


class EvaluateStrategyPredictionAdditionalTests(TestCase):
    def _agent(self) -> ToolAgent:
        return ToolAgent(model="m", provider="vllm", base_url="http://127.0.0.1:1/v1")

    def test_not_executed_returns_none(self) -> None:
        agent = self._agent()
        agent._record_strategy({"test_action": "LEFT", "expected_outcome": "state_change"})
        self.assertIsNone(agent._evaluate_strategy_prediction({"executed": False}))

    def test_level_progress_match(self) -> None:
        agent = self._agent()
        agent._record_strategy({"test_action": "SPACE", "expected_outcome": "level_progress"})
        result = agent._evaluate_strategy_prediction({
            "executed": True,
            "action_display": "SPACE",
            "outcome_class": "level_progress",
            "level_completed": True,
        })
        self.assertEqual(result["status"], "supported")

    def test_exact_noop_match(self) -> None:
        agent = self._agent()
        agent._record_strategy({"test_action": "UP", "expected_outcome": "no_change"})
        result = agent._evaluate_strategy_prediction({
            "executed": True,
            "action_display": "UP",
            "outcome_class": "exact_noop",
            "board_changed": False,
        })
        self.assertEqual(result["status"], "supported")

    def test_mismatched_mouse_actions(self) -> None:
        agent = self._agent()
        agent._record_strategy({"test_action": "MOUSE", "expected_outcome": "state_change"})
        result = agent._evaluate_strategy_prediction({
            "executed": True,
            "action_display": "LEFT",
            "outcome_class": "state_change",
            "board_changed": True,
        })
        self.assertIsNone(result)


class CompactActionResultAdditionalEdgeTests(TestCase):
    def _agent(self) -> ToolAgent:
        return ToolAgent(model="m", provider="vllm", base_url="http://127.0.0.1:1/v1")

    def test_stopped_early_with_reason(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "stopped_early": True,
            "stop_reason": "level_completed",
            "stop_detail": "completed level 1",
        })
        self.assertTrue(compact["stopped_early"])
        self.assertEqual(compact["stop_reason"], "level_completed")

    def test_single_step_preserved(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "steps": [{"action_display": "LEFT", "board_changed": True}],
        })
        self.assertEqual(len(compact["steps"]), 1)

    def test_run_complete_flag(self) -> None:
        agent = self._agent()
        compact = agent._compact_action_result({
            "executed": True,
            "run_complete": True,
        })
        self.assertTrue(compact["run_complete"])


class SummarizedKnowledgeAdditionalTests(TestCase):
    def _agent(self) -> ToolAgent:
        return ToolAgent(model="m", provider="vllm", base_url="http://127.0.0.1:1/v1")

    def test_all_knowledge_fields(self) -> None:
        agent = self._agent()
        for key in ["world_model", "goal_model", "action_model", "current_plan",
                     "cross_level_notes", "recent_findings"]:
            agent._summarized_knowledge[key] = f"value for {key}"
        lines = agent._summarized_knowledge_lines()
        self.assertGreaterEqual(len(lines), 6)

    def test_single_char_value(self) -> None:
        agent = self._agent()
        agent._summarized_knowledge["world_model"] = "x"
        lines = agent._summarized_knowledge_lines()
        self.assertGreater(len(lines), 0)


class UpdateKnowledgeFromAssistantAdditionalTests(TestCase):
    def _agent(self) -> ToolAgent:
        return ToolAgent(model="m", provider="vllm", base_url="http://127.0.0.1:1/v1")

    def test_action_model_extracted(self) -> None:
        agent = self._agent()
        agent._update_summarized_knowledge_from_assistant(
            "Action model: LEFT always produces change"
        )
        self.assertIn("LEFT", agent._summarized_knowledge["action_model"])

    def test_long_world_model_stored(self) -> None:
        agent = self._agent()
        long_text = "World model: " + "detailed " * 100
        agent._update_summarized_knowledge_from_assistant(long_text)
        self.assertGreater(len(agent._summarized_knowledge["world_model"]), 0)

    def test_whitespace_only_no_update(self) -> None:
        agent = self._agent()
        agent._update_summarized_knowledge_from_assistant("   ")
        self.assertEqual(agent._summarized_knowledge["world_model"], "")


class UpdateKnowledgeFromStepSummaryAdditionalTests(TestCase):
    def _agent(self) -> ToolAgent:
        return ToolAgent(model="m", provider="vllm", base_url="http://127.0.0.1:1/v1")

    def test_action_completed_updates_findings(self) -> None:
        agent = self._agent()
        agent._last_step_summary = {"board_changed": True}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertGreater(len(agent._summarized_knowledge["recent_findings"]), 0)

    def test_no_valid_actions_updates_findings(self) -> None:
        agent = self._agent()
        agent._last_step_summary = {"valid_actions": []}
        agent._update_summarized_knowledge_from_step_summary()
        self.assertIsInstance(agent._summarized_knowledge, dict)
