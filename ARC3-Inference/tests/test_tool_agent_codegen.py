from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from inference.agent import tool_agent as tool_agent_module
from inference.agent.python_tool_sandbox import _SANDBOX_BOOTSTRAP
from inference.agent.tool_agent import (
    ToolAgent,
    _ChatCompletionResult,
    _normalize_generated_python_code,
    _normalize_tool_call_arguments,
    _python_tool_payload,
    _render_tool_result_display,
)


class ToolAgentCodeGenerationTests(TestCase):
    @staticmethod
    def _execute_prepared_code(code: str) -> dict:
        bootstrap_namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), bootstrap_namespace)
        tree = bootstrap_namespace["_prepare_user_code"](code)
        runtime: dict = {}
        exec(compile(tree, "<python_tool>", "exec"), runtime, runtime)
        return runtime

    def test_normalizes_whole_python_code_fence(self) -> None:
        code = _normalize_generated_python_code(
            "```python\nprint('components', 3)\nresult = {'count': 3}\n```"
        )

        self.assertEqual(
            code,
            "print('components', 3)\nresult = {'count': 3}",
        )

    def test_recovers_python_fence_with_explanatory_markdown(self) -> None:
        value = "Run this:\n```python\nresult = 3\n```"

        self.assertEqual(_normalize_generated_python_code(value), "result = 3")

    def test_recovers_python3_fence(self) -> None:
        value = "```python3\nresult = 3\n```"

        self.assertEqual(_normalize_generated_python_code(value), "result = 3")

    def test_recovers_exact_nested_code_mapping(self) -> None:
        value = {"language": "python", "code": "result = 5"}

        self.assertEqual(_normalize_generated_python_code(value), "result = 5")

    def test_recovers_json_encoded_code_mapping(self) -> None:
        value = json.dumps({"code": "result = 8"})

        self.assertEqual(_normalize_generated_python_code(value), "result = 8")

    def test_recovers_exact_xml_python_wrapper(self) -> None:
        value = "<python>\nresult = 13\n</python>"

        self.assertEqual(_normalize_generated_python_code(value), "result = 13")

    def test_recovers_standalone_python_label(self) -> None:
        value = "Python:\nresult = 21"

        self.assertEqual(_normalize_generated_python_code(value), "result = 21")

    def test_does_not_guess_between_multiple_code_fences(self) -> None:
        value = (
            "```python\nresult = 1\n```\n"
            "```python\nresult = 2\n```"
        )

        self.assertEqual(_normalize_generated_python_code(value), value)

    def test_recovers_unique_valid_python_from_multiple_code_fences(self) -> None:
        value = (
            "An invalid first attempt:\n```python\nfor\n```\n"
            "Use this corrected program:\n```python\nresult = 2\n```"
        )

        self.assertEqual(_normalize_generated_python_code(value), "result = 2")

    def test_ignores_non_python_fences_when_python_candidate_is_unique(self) -> None:
        value = (
            "Input shape:\n```json\n{\"rows\": 3}\n```\n"
            "Program:\n```python\nresult = current_frame.shape\n```"
        )

        self.assertEqual(
            _normalize_generated_python_code(value),
            "result = current_frame.shape",
        )

    def test_recovers_tilde_fence_without_closing_newline(self) -> None:
        value = "Use this:\n~~~python\nresult = 34~~~"

        self.assertEqual(_normalize_generated_python_code(value), "result = 34")

    def test_does_not_extract_invalid_fenced_python(self) -> None:
        value = "Try this:\n```python\nfor\n```"

        self.assertEqual(_normalize_generated_python_code(value), value)

    def test_does_not_unwrap_invalid_or_ambiguous_nested_payload(self) -> None:
        invalid = json.dumps({"code": "for"})
        ambiguous = json.dumps({"code": "result = 1", "alternative": "result = 2"})

        self.assertEqual(_normalize_generated_python_code(invalid), invalid)
        self.assertEqual(_normalize_generated_python_code(ambiguous), ambiguous)

    def test_does_not_extract_non_python_xml_wrapper(self) -> None:
        value = '<code language="javascript">result = 1</code>'

        self.assertEqual(_normalize_generated_python_code(value), value)

    def test_rejects_non_object_tool_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON object"):
            _normalize_tool_call_arguments('["not", "an", "object"]')
        with self.assertRaisesRegex(ValueError, "JSON object"):
            _normalize_tool_call_arguments(["not", "an", "object"])

    def test_recovers_direct_python_tool_arguments(self) -> None:
        arguments = _normalize_tool_call_arguments(
            "counts = current_frame.color_counts\nresult = max(counts, key=counts.get)"
        )

        self.assertEqual(
            arguments,
            {
                "code": (
                    "counts = current_frame.color_counts\n"
                    "result = max(counts, key=counts.get)"
                )
            },
        )

    def test_recovers_wrapped_python_tool_arguments(self) -> None:
        arguments = _normalize_tool_call_arguments(
            "Here is the program:\n```python\nresult = current_frame.shape\n```"
        )

        self.assertEqual(arguments, {"code": "result = current_frame.shape"})

    def test_rejects_invalid_direct_python_tool_arguments(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            _normalize_tool_call_arguments("for item in")

    def test_preserves_valid_raw_python_containing_backticks(self) -> None:
        value = 'result = "```python\\nnot executable\\n```"'

        self.assertEqual(_normalize_generated_python_code(value), value)

    def test_preflight_syntax_error_returns_structured_repair_context(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        response = agent._run_python_tool(
            Path("unused/tool_runtime_state.json"),
            {"code": "if True print('x')"},
        )
        payload = json.loads(response.content)

        self.assertEqual(payload["tool"], "python")
        self.assertTrue(payload["retryable"])
        self.assertIn("syntax error", payload["error"].lower())
        self.assertEqual(payload["diagnostic"]["type"], "SyntaxError")
        self.assertEqual(payload["diagnostic"]["line"], 1)
        self.assertEqual(payload["diagnostic"]["source"], "if True print('x')")
        self.assertIn("retry", payload["diagnostic"]["hint"].lower())

    def test_captures_notebook_style_final_expression(self) -> None:
        runtime = self._execute_prepared_code("values = [2, 3, 5]\nsum(values)")

        self.assertEqual(runtime["__tool_expression_result"], 10)

    def test_final_expression_executes_only_once(self) -> None:
        runtime = self._execute_prepared_code(
            "calls = []\n"
            "def produce():\n"
            "    calls.append('called')\n"
            "    return 7\n"
            "produce()"
        )

        self.assertEqual(runtime["calls"], ["called"])
        self.assertEqual(runtime["__tool_expression_result"], 7)

    def test_explicit_result_remains_authoritative(self) -> None:
        runtime = self._execute_prepared_code("result = {'answer': 42}\n'ignored expression'")

        selected = runtime.get("result", runtime.get("__tool_expression_result"))
        self.assertEqual(selected, {"answer": 42})

    def test_multi_action_snippet_keeps_latest_valid_actions(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        responses = iter(
            [
                {
                    "executed": True,
                    "action_num": 1,
                    "action_display": "LEFT",
                    "valid_actions": ["RIGHT"],
                },
                {
                    "executed": True,
                    "action_num": 2,
                    "action_display": "RIGHT",
                },
            ]
        )

        def step_env(_request: dict) -> dict:
            return next(responses)

        agent._step_env_callback = step_env
        agent._current_valid_actions = ["LEFT"]
        response = agent._run_python_tool(
            Path("unused/tool_runtime_state.json"),
            {
                "code": (
                    "action('LEFT')\n"
                    "after_first = list(valid_actions)\n"
                    "action('RIGHT')\n"
                    "{'after_first': after_first, 'after_second': list(valid_actions)}"
                )
            },
        )
        payload = json.loads(response.content)

        self.assertEqual(payload["result"]["after_first"], ["RIGHT"])
        self.assertEqual(payload["result"]["after_second"], ["RIGHT"])

    def test_analyze_returns_malformed_arguments_as_retryable_tool_result(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._tool_steps = 1
        agent._chat_completion = lambda *_args, **_kwargs: _ChatCompletionResult(
            message={
                "tool_calls": [
                    {
                        "id": "bad-arguments",
                        "type": "function",
                        "function": {"name": "python", "arguments": "[]"},
                    }
                ]
            },
            finish_reason="tool_calls",
        )

        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}),
                encoding="utf-8",
            )
            result = agent.analyze(state_path, action_num=0, valid_actions=["LEFT"])
            transcript = state_path.with_name("tool_runtime_state_analyzer.txt").read_text(
                encoding="utf-8"
            )

        self.assertIsNotNone(result)
        self.assertFalse(result.step_executed)
        self.assertIn("Provide one JSON object and retry", transcript)

    def test_analyze_executes_direct_python_tool_arguments(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._tool_steps = 1
        agent._chat_completion = lambda *_args, **_kwargs: _ChatCompletionResult(
            message={
                "tool_calls": [
                    {
                        "id": "direct-python",
                        "type": "function",
                        "function": {"name": "python", "arguments": "result = 42"},
                    }
                ]
            },
            finish_reason="tool_calls",
        )

        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}),
                encoding="utf-8",
            )
            with (
                patch(
                    "inference.agent.tool_agent._normalize_tool_call_arguments",
                    wraps=_normalize_tool_call_arguments,
                ) as normalize_arguments,
                patch.object(agent, "_tools", wraps=agent._tools) as build_tools,
            ):
                result = agent.analyze(
                    state_path,
                    action_num=0,
                    valid_actions=["LEFT"],
                )
            transcript = state_path.with_name("tool_runtime_state_analyzer.txt").read_text(
                encoding="utf-8"
            )

        self.assertIsNotNone(result)
        self.assertFalse(result.step_executed)
        self.assertIn("[TOOL RESULT: python]\n42", transcript)
        self.assertNotIn("Provide one JSON object and retry", transcript)
        self.assertEqual(normalize_arguments.call_count, 1)
        self.assertEqual(build_tools.call_count, 1)

    def test_analyze_streams_monotonic_transcript_snapshots_and_writes_prompt_log_once(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._tool_steps = 1
        agent._chat_completion = lambda *_args, **_kwargs: _ChatCompletionResult(
            message={
                "tool_calls": [
                    {
                        "id": "direct-python",
                        "type": "function",
                        "function": {"name": "python", "arguments": "result = 42"},
                    }
                ]
            },
            finish_reason="tool_calls",
            latency_seconds=0.125,
        )
        transcript_updates: list[str] = []

        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}),
                encoding="utf-8",
            )
            with patch.object(
                tool_agent_module,
                "_write_prompt_log_snapshot",
                wraps=tool_agent_module._write_prompt_log_snapshot,
            ) as write_prompt_log:
                result = agent.analyze(
                    state_path,
                    action_num=0,
                    valid_actions=["LEFT"],
                    transcript_updated=transcript_updates.append,
                )

            prompt_log = tool_agent_module._resolve_prompt_log_path(state_path)
            prompt_snapshot = prompt_log.read_text(encoding="utf-8")

        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(transcript_updates), 4)
        for previous, current in zip(transcript_updates, transcript_updates[1:]):
            self.assertTrue(current.startswith(previous))
        self.assertIn("[ANALYZER STATUS]", transcript_updates[-1])
        self.assertIn("efficiency_metrics", transcript_updates[-1])
        self.assertEqual(result.efficiency_metrics["model_calls"], 1)
        self.assertEqual(result.efficiency_metrics["model_seconds"], 0.125)
        self.assertEqual(write_prompt_log.call_count, 1)
        self.assertIn("LATEST MODEL CALL SNAPSHOT", prompt_snapshot)

    def test_usage_aliases_are_not_double_counted(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        agent._accumulate_usage_tokens(
            {
                "prompt_tokens": 11,
                "input_tokens": 11,
                "completion_tokens": 7,
                "output_tokens": 7,
            }
        )

        self.assertEqual(agent.generated_tokens, 7)
        self.assertEqual(agent.total_tokens, 18)

    def test_generated_only_usage_contributes_to_total(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        agent._accumulate_usage_tokens(
            {"input_tokens": 5, "generated_tokens": 3}
        )

        self.assertEqual(agent.generated_tokens, 3)
        self.assertEqual(agent.total_tokens, 8)

    def test_response_tool_calls_receive_unique_nonempty_ids(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        calls = agent._normalize_response_tool_calls(
            [
                None,
                {"id": "duplicate", "function": {"name": "python"}},
                {"id": "duplicate", "function": None},
            ]
        )

        ids = [call["id"] for call in calls]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(ids))
        self.assertTrue(all(isinstance(call["function"], dict) for call in calls))

    def test_payload_preserves_stdout_and_structured_result(self) -> None:
        payload = _python_tool_payload(
            {
                "stdout": "components: 3\n",
                "result": {"count": 3, "colors": [2, 4]},
                "error": "",
                "action_results": [],
            }
        )

        self.assertEqual(payload["stdout"], "components: 3\n")
        self.assertEqual(payload["result"], {"count": 3, "colors": [2, 4]})
        rendered = _render_tool_result_display(payload)
        self.assertIn("components: 3", rendered)
        self.assertIn("result:", rendered)
        self.assertIn("count: 3", rendered)

    def test_payload_keeps_action_summary_fallback_without_user_output(self) -> None:
        payload = _python_tool_payload(
            {
                "stdout": "",
                "result": None,
                "error": "",
                "action_results": [
                    {"executed": True, "action_display": "LEFT"},
                    {"executed": True, "action_display": "UP"},
                ],
            }
        )

        self.assertEqual(payload["result"]["action_calls"], 2)
        self.assertEqual(
            payload["result"]["last_action_result"]["action_display"],
            "UP",
        )

    def test_generated_python_limits_block_source_before_sandbox_execution(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        for limit_name, limit_value, code in (
            ("_LOCAL_ANALYZER_MAX_CODE_BYTES", 16, "result = '" + "x" * 32 + "'"),
            ("_LOCAL_ANALYZER_MAX_AST_NODES", 5, "values = [1, 2, 3]\nresult = sum(values)"),
        ):
            with (
                self.subTest(limit=limit_name),
                patch.object(tool_agent_module, limit_name, limit_value),
                patch.object(tool_agent_module, "run_sandboxed_python") as sandbox,
            ):
                response = agent._run_python_tool(
                    Path("unused/tool_runtime_state.json"), {"code": code}
                )

            payload = json.loads(response.content)
            self.assertEqual(payload["diagnostic"]["type"], "GeneratedCodeLimitError")
            self.assertIn("smaller bounded program", payload["diagnostic"]["hint"])
            sandbox.assert_not_called()

    def test_mouse_action_outside_current_frame_is_rejected_before_dispatch(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._current_valid_actions = ["MOUSE"]

        def unexpected_step(_request: dict) -> dict:
            raise AssertionError("out-of-bounds action reached the environment")

        agent._step_env_callback = unexpected_step
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "current_frame": {
                            "grid": [[0, 0], [0, 0]],
                            "step": 0,
                            "level": 1,
                        },
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            response = agent._run_python_tool(
                state_path,
                {"code": "action({'action': 'MOUSE', 'row': 2, 'col': 0})"},
            )

        payload = json.loads(response.content)
        self.assertIn("outside the current frame shape 2x2", payload["error"])
        self.assertFalse(response.step_executed)

    def test_analyze_returns_structured_missing_state_failure(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        result = agent.analyze(Path("missing/tool_runtime_state.json"), action_num=0)

        self.assertIsNotNone(result)
        self.assertEqual(result.failure_category, "state_missing")
        self.assertIn("does not exist", result.failure_detail)

    def test_diagnostic_io_failures_do_not_abort_analyzer_turn(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._tool_steps = 1
        agent._chat_completion = lambda *_args, **_kwargs: _ChatCompletionResult(
            message={
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": '{"code":"result=42"}',
                        },
                    }
                ]
            },
            finish_reason="tool_calls",
        )

        def failed_callback(_transcript: str) -> None:
            raise RuntimeError("viewer disconnected")

        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}), encoding="utf-8"
            )
            with patch.object(
                tool_agent_module,
                "_write_prompt_log_snapshot",
                side_effect=OSError("disk unavailable"),
            ), patch.object(tool_agent_module.log, "warning"):
                result = agent.analyze(
                    state_path,
                    action_num=0,
                    transcript_updated=failed_callback,
                )

        self.assertIsNotNone(result)
        self.assertEqual(result.failure_category, "tool_step_exhausted")
        self.assertTrue(result.exhausted)
