from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from inference.agent import tool_agent as tool_agent_module
from inference.agent.python_tool_sandbox import _SANDBOX_BOOTSTRAP
from inference.agent.runtime_state import load_runtime_state
from inference.agent.tool_agent import (
    ToolAgent,
    _ChatCompletionResult,
    _choice_has_executable_tool,
    _deterministic_generated_python_repair,
    _generated_python_preflight_issues,
    _normalize_generated_python_code,
    _normalize_tool_call_arguments,
    _parse_bounded_generated_python,
    _recover_tool_calls_from_markup,
    _generated_python_semantic_fingerprint,
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

    def test_recovers_six_bounded_structured_wrapper_levels(self) -> None:
        value: object = "result = 6"
        for _ in range(6):
            value = {"code": value}

        self.assertEqual(_normalize_generated_python_code(value), "result = 6")

    def test_recovers_json_encoded_code_mapping(self) -> None:
        value = json.dumps({"code": "result = 8"})

        self.assertEqual(_normalize_generated_python_code(value), "result = 8")

    def test_recovers_code_line_array_from_structured_payload(self) -> None:
        value = {
            "language": "python",
            "code": [
                "values = [2, 3, 5]",
                "result = sum(values)",
            ],
        }

        self.assertEqual(
            _normalize_generated_python_code(value),
            "values = [2, 3, 5]\nresult = sum(values)",
        )

    def test_recovers_long_code_line_array_within_parser_limits(self) -> None:
        lines = [f"value_{index} = {index}" for index in range(130)]
        lines.append("result = value_129")

        self.assertEqual(
            _normalize_generated_python_code({"code": lines}),
            "\n".join(lines),
        )

    def test_recovers_structured_line_array_beyond_legacy_cap(self) -> None:
        lines = [f"# planning note {index}" for index in range(1500)]
        lines.append("result = 7")

        self.assertEqual(
            _normalize_generated_python_code({"code": lines}),
            "\n".join(lines),
        )

    def test_recovers_exact_nested_code_content_wrapper(self) -> None:
        value = json.dumps(
            {
                "code": {
                    "language": "python",
                    "content": [
                        {"type": "output_text", "text": "result = (current_frame."},
                        {"type": "output_text", "text": "shape)"},
                    ],
                }
            }
        )

        self.assertEqual(
            _normalize_generated_python_code(value),
            "result = (current_frame.shape)",
        )

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
        invalid_lines = json.dumps({"code": ["for", "while"]})
        non_python_nested = json.dumps(
            {"code": {"language": "javascript", "code": "result = 1"}}
        )

        self.assertEqual(_normalize_generated_python_code(invalid), invalid)
        self.assertEqual(_normalize_generated_python_code(ambiguous), ambiguous)
        self.assertEqual(
            _normalize_generated_python_code(invalid_lines), invalid_lines
        )
        self.assertEqual(
            _normalize_generated_python_code(non_python_nested), non_python_nested
        )

    def test_recovers_content_parts_and_metadata_wrappers(self) -> None:
        wrapped = {
            "content": [
                {
                    "type": "output_text",
                    "text": "result = current_frame.shape",
                    "description": "generated program",
                }
            ],
            "mime_type": "application/json",
        }
        metadata = {"code": "result = 42", "language": "python", "explanation": "bounded"}

        self.assertEqual(_normalize_generated_python_code(wrapped), "result = current_frame.shape")
        self.assertEqual(_normalize_generated_python_code(metadata), "result = 42")

    def test_recovers_provider_parts_with_benign_transport_metadata(self) -> None:
        wrapped = {
            "content": [
                {
                    "type": "output_text",
                    "text": "result = 17",
                    "id": "part-1",
                    "index": 0,
                    "role": "assistant",
                    "status": "completed",
                }
            ]
        }

        self.assertEqual(_normalize_generated_python_code(wrapped), "result = 17")

    def test_does_not_guess_between_competing_content_parts(self) -> None:
        wrapped = {"content": [{"type": "text", "text": "result = 1"}, {"type": "text", "text": "result = 2"}]}

        self.assertEqual(_normalize_generated_python_code(wrapped), str(wrapped))

    def test_recovers_program_split_across_provider_content_parts(self) -> None:
        token_split = {
            "content": [
                {"type": "output_text", "text": "result = (current_frame."},
                {"type": "output_text", "text": "shape)"},
            ]
        }
        line_split = {
            "content": [
                {"type": "text", "text": "if True:"},
                {"type": "text", "text": "    result = 7"},
            ]
        }

        self.assertEqual(
            _normalize_generated_python_code(token_split),
            "result = (current_frame.shape)",
        )
        self.assertIn(
            _normalize_generated_python_code(line_split),
            {"if True:    result = 7", "if True:\n    result = 7"},
        )

    def test_recovers_program_split_across_more_than_sixteen_parts(self) -> None:
        fragments = ["result = ("] + ["1 +"] * 18 + ["1)"]
        wrapped = {
            "content": [
                {"type": "output_text", "text": fragment}
                for fragment in fragments
            ]
        }

        self.assertEqual(
            _normalize_generated_python_code(wrapped),
            "".join(fragments),
        )

    def test_reassembles_compiling_parts_with_cross_fragment_dependency(self) -> None:
        wrapped = {
            "content": [
                {"type": "text", "text": "values = [2, 3, 5]"},
                {"type": "text", "text": "result = sum(values)"},
            ]
        }

        self.assertEqual(
            _normalize_generated_python_code(wrapped),
            "values = [2, 3, 5]\nresult = sum(values)",
        )

    def test_comprehension_target_does_not_create_fragment_dependency(self) -> None:
        wrapped = {
            "content": [
                {"type": "text", "text": "result = [x for x in [1]]"},
                {"type": "text", "text": "result = x"},
            ]
        }

        self.assertEqual(_normalize_generated_python_code(wrapped), str(wrapped))

    def test_match_pattern_binding_creates_fragment_dependency(self) -> None:
        first = "match {'score': 7}:\n    case {'score': score}:\n        pass\n"
        second = "result = score"
        wrapped = {
            "content": [
                {"type": "text", "text": first},
                {"type": "text", "text": second},
            ]
        }

        self.assertEqual(
            _normalize_generated_python_code(wrapped),
            f"{first}{second}",
        )

    def test_walrus_binding_creates_fragment_dependency(self) -> None:
        first = "(score := 7)\n"
        second = "result = score"
        wrapped = {
            "content": [
                {"type": "text", "text": first},
                {"type": "text", "text": second},
            ]
        }

        self.assertEqual(
            _normalize_generated_python_code(wrapped), f"{first}{second}"
        )

    def test_try_star_body_binding_creates_fragment_dependency(self) -> None:
        first = (
            "try:\n"
            "    score = 7\n"
            "except* ValueError:\n"
            "    score = 0\n"
        )
        second = "result = score"
        wrapped = {
            "content": [
                {"type": "text", "text": first},
                {"type": "text", "text": second},
            ]
        }

        self.assertEqual(
            _normalize_generated_python_code(wrapped), f"{first}{second}"
        )

    def test_reassembles_provider_split_helper_definition(self) -> None:
        wrapped = {
            "content": [
                {"type": "text", "text": "def compute():\n    return 7"},
                {"type": "text", "text": "result = compute()"},
            ]
        }

        self.assertEqual(
            _normalize_generated_python_code(wrapped),
            "def compute():\n    return 7\nresult = compute()",
        )

    def test_deduplicates_semantically_identical_wrapped_programs(self) -> None:
        wrapped = {
            "content": [
                {"type": "text", "text": "result=1"},
                {"type": "text", "text": "result = 1  # same program"},
            ]
        }

        self.assertEqual(_normalize_generated_python_code(wrapped), "result=1")

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

    def test_repairs_complete_unambiguous_json_arguments(self) -> None:
        self.assertEqual(
            _normalize_tool_call_arguments('{"code":"result = 1",}'),
            {"code": "result = 1"},
        )
        self.assertEqual(
            _normalize_tool_call_arguments("Arguments: {'code': 'result = 2'}"),
            {"code": "result = 2"},
        )
        self.assertEqual(
            _normalize_tool_call_arguments('{"code":"line1\nline2"}'),
            {"code": "line1\nline2"},
        )

    def test_rejects_ambiguous_or_truncated_json_recovery(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            _normalize_tool_call_arguments('first {"code":"result=1"} second {"code":"result=2"}')
        with self.assertRaises(json.JSONDecodeError):
            _normalize_tool_call_arguments('{"code":"result=1"')

    def test_xml_tool_call_cdata_preserves_parameter_like_code(self) -> None:
        calls = _recover_tool_calls_from_markup(
            '<tool_call><function name="python"><parameter name="code"><![CDATA[result = "</parameter>"]]></parameter></function></tool_call>'
        )

        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(arguments["code"], 'result = "</parameter>"')

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

    def test_missing_colon_is_repaired_locally_before_execution(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        response = agent._run_python_tool(
            Path("unused/tool_runtime_state.json"),
            {"code": "if True\n    result = 7"},
        )
        payload = json.loads(response.content)

        self.assertEqual(payload["result"], 7)
        self.assertEqual(payload["auto_repair"]["repair"], "insert_missing_colon")

    def test_missing_colon_repair_preserves_inline_comment(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        response = agent._run_python_tool(
            Path("unused/tool_runtime_state.json"),
            {"code": "if True  # bounded branch\n    result = 7"},
        )
        payload = json.loads(response.content)

        self.assertEqual(payload["result"], 7)
        self.assertEqual(payload["auto_repair"]["repair"], "insert_missing_colon")

    def test_missing_colon_repair_supports_async_function_headers(self) -> None:
        code = "async def compute()\n    return 7\nresult = 1"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as exc:
            repaired = tool_agent_module._deterministic_syntax_repair(code, exc)
        else:
            self.fail("expected the async function header to require a colon")

        self.assertEqual(
            repaired,
            "async def compute():\n    return 7\nresult = 1",
        )

    def test_missing_colon_repair_supports_multiline_headers(self) -> None:
        code = "def compute(\n    value\n)\n    return value\nresult = compute(7)"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as exc:
            repaired = tool_agent_module._deterministic_syntax_repair(code, exc)
        else:
            self.fail("expected the multiline function header to require a colon")

        self.assertEqual(
            repaired,
            "def compute(\n    value\n):\n    return value\nresult = compute(7)",
        )

    def test_final_top_level_return_is_repaired_as_result(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": {"count": 3},
            "stdout": "",
            "action_results": [],
        }
        code = "values = [1, 2, 3]\nreturn {'count': len(values)}"

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], {"count": 3})
        self.assertEqual(
            payload["auto_repair"]["repair"], "replace_top_level_return"
        )
        repaired = sandbox.call_args.kwargs["code"]
        self.assertIn("result = {'count': len(values)}", repaired)
        self.assertNotIn("return {'count'", repaired)

    def test_unclosed_delimiters_are_completed_before_execution(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": 3,
            "stdout": "",
            "action_results": [],
        }
        code = "result = len([1, 2, 3"

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 3)
        self.assertEqual(
            payload["auto_repair"]["repair"], "close_unmatched_delimiters"
        )
        self.assertEqual(sandbox.call_args.kwargs["code"], f"{code}\n])")

    def test_deeper_bounded_unclosed_delimiters_are_completed(self) -> None:
        code = "result = ((((([1, 2, 3]"
        error = SyntaxError("'(' was never closed")

        repaired = tool_agent_module._deterministic_unclosed_delimiter_repair(
            code, error
        )

        self.assertEqual(repaired, f"{code}\n)))))")

    def test_uniformly_indented_snippet_is_dedented_before_execution(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": 6,
            "stdout": "",
            "action_results": [],
        }
        code = "    values = [1, 2, 3]\n    result = sum(values)"

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 6)
        self.assertEqual(
            payload["auto_repair"]["repair"], "dedent_uniform_block"
        )
        self.assertEqual(
            sandbox.call_args.kwargs["code"],
            "values = [1, 2, 3]\nresult = sum(values)",
        )

    def test_uniform_dedent_ignores_unindented_comment_lines(self) -> None:
        code = "# generated analysis\n    result = sum([1, 2, 3])"
        try:
            compile(code, "<test>", "exec")
        except SyntaxError as exc:
            repaired = tool_agent_module._deterministic_syntax_repair(code, exc)
        else:
            self.fail("expected the uniformly indented snippet to be invalid")

        self.assertEqual(
            repaired,
            "# generated analysis\nresult = sum([1, 2, 3])",
        )

    def test_recursive_repairs_preserve_ordered_metadata(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": [],
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "    result = current_frame.objcts()"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["auto_repair"]["repair"], "repair_chain")
        self.assertEqual(payload["auto_repair"]["repair_count"], 2)
        self.assertEqual(
            payload["auto_repair"]["repairs"],
            ["dedent_uniform_block", "correct_documented_attribute"],
        )
        self.assertIn("current_frame.objects()", sandbox.call_args.kwargs["code"])

    def test_boolean_operator_repair_preserves_strings_and_comments(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": True,
            "stdout": "",
            "action_results": [],
        }
        code = (
            'label = "literal && text"  # comment || text\n'
            "result = (True && False) || True"
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertTrue(payload["result"])
        self.assertEqual(
            payload["auto_repair"]["repair"], "normalize_boolean_operators"
        )
        repaired_code = sandbox.call_args.kwargs["code"]
        self.assertIn('"literal && text"', repaired_code)
        self.assertIn("# comment || text", repaired_code)
        self.assertIn("True  and  False", repaired_code)
        self.assertIn(" or  True", repaired_code)

    def test_equality_operator_repair_preserves_strings_and_comments(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": True,
            "stdout": "",
            "action_results": [],
        }
        code = (
            'label = "literal === text"  # comment !== text\n'
            "result = (1 === 1) and (2 !== 3)"
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertTrue(payload["result"])
        self.assertEqual(
            payload["auto_repair"]["repair"], "normalize_equality_operators"
        )
        repaired_code = sandbox.call_args.kwargs["code"]
        self.assertIn('"literal === text"', repaired_code)
        self.assertIn("# comment !== text", repaired_code)
        self.assertIn("1 == 1", repaired_code)
        self.assertIn("2 != 3", repaired_code)
        self.assertNotIn("1 === 1", repaired_code)
        self.assertNotIn("2 !== 3", repaired_code)

    def test_negation_operator_repair_preserves_strings_and_comments(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": True,
            "stdout": "",
            "action_results": [],
        }
        code = (
            'label = "literal ! text"  # comment ! text\n'
            "result = !False and !!True"
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertTrue(payload["result"])
        self.assertEqual(
            payload["auto_repair"]["repair"], "normalize_negation_operators"
        )
        repaired_code = sandbox.call_args.kwargs["code"]
        self.assertIn('"literal ! text"', repaired_code)
        self.assertIn("# comment ! text", repaired_code)
        self.assertIn("not False", repaired_code)
        self.assertIn("not  not True", repaired_code)

    def test_variable_declaration_repair_preserves_strings_and_comments(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": 3,
            "stdout": "",
            "action_results": [],
        }
        code = (
            'label = "const untouched = true"  # let untouched = false\n'
            "const values = [1, 2]\n"
            "let total = sum(values)\n"
            "result = total"
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 3)
        self.assertEqual(
            payload["auto_repair"]["repair"],
            "normalize_variable_declarations",
        )
        repaired_code = sandbox.call_args.kwargs["code"]
        self.assertIn('"const untouched = true"', repaired_code)
        self.assertIn("# let untouched = false", repaired_code)
        self.assertIn("values = [1, 2]", repaired_code)
        self.assertIn("total = sum(values)", repaired_code)
        self.assertNotIn("const values", repaired_code)
        self.assertNotIn("let total", repaired_code)

    def test_semantic_fingerprint_ignores_formatting_and_comments(self) -> None:
        self.assertEqual(
            _generated_python_semantic_fingerprint("result=1 # first"),
            _generated_python_semantic_fingerprint("\nresult = 1\n"),
        )

    def test_semantic_fingerprint_distinguishes_behavioral_constants(self) -> None:
        self.assertNotEqual(
            _generated_python_semantic_fingerprint("result = 1"),
            _generated_python_semantic_fingerprint("result = 2"),
        )

    def test_preflight_range_limit_is_inclusive_and_handles_negative_steps(self) -> None:
        at_limit = _parse_bounded_generated_python("result = range(1000000)")
        over_limit = _parse_bounded_generated_python(
            "result = range(1000001, 0, -1)"
        )

        self.assertFalse(_generated_python_preflight_issues(at_limit))
        issues = _generated_python_preflight_issues(over_limit)
        self.assertIn("1000001 iterations", issues[0]["message"])

    def test_preflight_handles_shifted_ranges_but_not_keyworded_range(self) -> None:
        shifted = _parse_bounded_generated_python("result = range(1 << 20)")
        keyworded = _parse_bounded_generated_python(
            "result = range(stop=1000001)"
        )

        shifted_issues = _generated_python_preflight_issues(shifted)
        self.assertIn("1048576 iterations", shifted_issues[0]["message"])
        self.assertFalse(
            any(
                "Constant range expands" in issue["message"]
                for issue in _generated_python_preflight_issues(keyworded)
            )
        )

    def test_preflight_skips_large_range_guard_after_range_rebinding(self) -> None:
        tree = _parse_bounded_generated_python(
            "range = lambda stop: [stop]\nresult = range(2000000)"
        )

        self.assertFalse(
            any(
                "Constant range expands" in issue["message"]
                for issue in _generated_python_preflight_issues(tree)
            )
        )

    def test_preflight_skips_range_parameter_inside_reachable_helper(self) -> None:
        tree = _parse_bounded_generated_python(
            "def build(range):\n"
            "    return range(2000000)\n"
            "result = build(lambda stop: [stop])"
        )

        self.assertFalse(
            any(
                "Constant range expands" in issue["message"]
                for issue in _generated_python_preflight_issues(tree)
            )
        )

    def test_preflight_skips_function_local_range_binding(self) -> None:
        tree = _parse_bounded_generated_python(
            "def build():\n"
            "    range = lambda stop: [stop]\n"
            "    return range(2000000)\n"
            "result = build()"
        )

        self.assertFalse(
            any(
                "Constant range expands" in issue["message"]
                for issue in _generated_python_preflight_issues(tree)
            )
        )

    def test_preflight_propagates_module_range_binding_into_helper(self) -> None:
        tree = _parse_bounded_generated_python(
            "range = lambda stop: [stop]\n"
            "def build():\n"
            "    return range(2000000)\n"
            "result = build()"
        )

        self.assertFalse(
            any(
                "Constant range expands" in issue["message"]
                for issue in _generated_python_preflight_issues(tree)
            )
        )

    def test_preflight_rejects_enormous_constant_range_expression(self) -> None:
        tree = _parse_bounded_generated_python("result = range(10 ** 20)")

        issues = _generated_python_preflight_issues(tree)

        self.assertIn("100000000000000000000 iterations", issues[0]["message"])

    def test_preflight_rejects_huge_two_argument_range(self) -> None:
        tree = _parse_bounded_generated_python("result = range(0, 10 ** 20)")

        issues = _generated_python_preflight_issues(tree)

        self.assertIn("100000000000000000000 iterations", issues[0]["message"])

    def test_preflight_rejects_huge_negative_step_range(self) -> None:
        tree = _parse_bounded_generated_python(
            "result = range(10 ** 20, 0, -1)"
        )

        issues = _generated_python_preflight_issues(tree)

        self.assertIn("100000000000000000000 iterations", issues[0]["message"])

    def test_preflight_ignores_invalid_operations_in_uncalled_helpers(self) -> None:
        tree = _parse_bounded_generated_python(
            "def dormant():\n"
            "    import pathlib\n"
            "    return current_frame.not_a_real_attribute\n"
            "result = 1"
        )

        self.assertEqual(_generated_python_preflight_issues(tree), [])

    def test_preflight_evaluates_constant_comparisons_in_while_conditions(self) -> None:
        tree = _parse_bounded_generated_python("while 0 < 1 < 2:\n    result = 1")

        issues = _generated_python_preflight_issues(tree)

        self.assertIn("has no break path", issues[0]["message"])

    def test_preflight_excludes_while_false_body(self) -> None:
        tree = _parse_bounded_generated_python(
            "while False:\n"
            "    import pathlib\n"
            "result = 1"
        )

        self.assertEqual(_generated_python_preflight_issues(tree), [])

    def test_preflight_includes_while_false_else_suite(self) -> None:
        tree = _parse_bounded_generated_python(
            "while False:\n"
            "    pass\n"
            "else:\n"
            "    result = current_frame.objcts"
        )

        issues = _generated_python_preflight_issues(tree)
        self.assertEqual(issues[0]["suggestions"], ["objects"])

    def test_preflight_excludes_definitely_empty_for_body(self) -> None:
        tree = _parse_bounded_generated_python(
            "for item in []:\n"
            "    import pathlib\n"
            "result = 1"
        )

        self.assertEqual(_generated_python_preflight_issues(tree), [])

    def test_preflight_includes_function_annotations(self) -> None:
        tree = _parse_bounded_generated_python(
            "def helper(value: current_frame.objcts):\n"
            "    return value\n"
            "result = 1"
        )

        issues = _generated_python_preflight_issues(tree)
        self.assertEqual(issues[0]["suggestions"], ["objects"])

    def test_structured_inspect_tool_executes_without_custom_code(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "current_frame": {
                            "grid": [[0, 1], [1, 0]],
                            "ascii": "WB\nBW",
                            "shape": [2, 2],
                            "step": 3,
                            "level": 2,
                        },
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            response = agent._dispatch_tool(
                state_path, "inspect", {"view": "history_summary"}
            )
            original_frame = load_runtime_state(state_path)[0]
            self.assertEqual(len(agent._matching_verified_programs(original_frame)), 1)
            state_path.write_text(
                json.dumps(
                    {
                        "current_frame": {
                            "grid": [[1, 1], [1, 1]],
                            "ascii": "BB\nBB",
                            "shape": [2, 2],
                            "step": 3,
                            "level": 2,
                        },
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            changed_frame = load_runtime_state(state_path)[0]

        payload = json.loads(response.content)
        self.assertEqual(payload["result"]["entries"], 0)
        self.assertFalse(response.step_executed)
        self.assertEqual(len(agent._verified_programs), 1)
        self.assertEqual(agent._matching_verified_programs(changed_frame), [])

    def test_structured_action_tool_reuses_validated_action_runtime(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._current_valid_actions = ["LEFT"]
        captured: list[dict] = []

        def step_env(request: dict) -> dict:
            captured.extend(request["actions"])
            return {
                "executed": True,
                "action_num": 1,
                "action_display": "LEFT",
                "valid_actions": ["RIGHT"],
            }

        agent._step_env_callback = step_env
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}),
                encoding="utf-8",
            )
            response = agent._dispatch_tool(
                state_path, "action", {"actions": [{"action": "LEFT"}]}
            )

        self.assertEqual(captured, [{"action": "LEFT"}])
        self.assertTrue(response.step_executed)

    def test_structured_action_dry_run_has_no_environment_side_effect(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._current_valid_actions = ["LEFT"]
        agent._step_env_callback = Mock(side_effect=AssertionError("must not execute"))

        response = agent._dispatch_tool(
            Path("unused/tool_runtime_state.json"),
            "action",
            {"actions": [{"action": "LEFT"}], "dry_run": True},
        )
        payload = json.loads(response.content)

        self.assertTrue(payload["dry_run"])
        self.assertTrue(payload["valid"])
        self.assertFalse(response.step_executed)
        agent._step_env_callback.assert_not_called()

    def test_structured_action_invalid_dry_run_is_non_throwing_and_read_only(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._current_valid_actions = ["LEFT"]
        agent._step_env_callback = Mock(side_effect=AssertionError("must not execute"))

        response = agent._dispatch_tool(
            Path("unused/tool_runtime_state.json"),
            "action",
            {"actions": [{"action": "RIGHT"}], "dry_run": True},
        )
        payload = json.loads(response.content)

        self.assertFalse(payload["valid"])
        self.assertTrue(payload["dry_run"])
        self.assertIn("not currently valid", payload["error"])
        self.assertFalse(response.step_executed)
        agent._step_env_callback.assert_not_called()

    def test_structured_mouse_dry_run_checks_current_frame_bounds(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._current_valid_actions = ["MOUSE"]
        agent._step_env_callback = Mock(side_effect=AssertionError("must not execute"))
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "current_frame": {
                            "grid": [[0, 0], [0, 0]],
                            "ascii": "WW\nWW",
                            "shape": [2, 2],
                            "step": 0,
                            "level": 1,
                        },
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )

            response = agent._dispatch_tool(
                state_path,
                "action",
                {"actions": [{"action": "MOUSE", "row": 2, "col": 1}], "dry_run": True},
            )

        payload = json.loads(response.content)
        self.assertFalse(payload["valid"])
        self.assertIn("outside the current frame shape 2x2", payload["error"])
        agent._step_env_callback.assert_not_called()

    def test_unknown_structured_tool_and_inspect_view_return_errors(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        unknown_tool = agent._dispatch_tool(
            Path("unused/tool_runtime_state.json"), "shell", {}
        )
        unknown_view = agent._dispatch_tool(
            Path("unused/tool_runtime_state.json"),
            "inspect",
            {"view": "raw_runtime"},
        )

        self.assertEqual(json.loads(unknown_tool.content)["error"], "Unknown tool: shell")
        self.assertEqual(
            json.loads(unknown_view.content)["error"],
            "Unknown inspect view: raw_runtime",
        )

    def test_candidate_verification_uses_read_only_action_handler(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._current_valid_actions = ["LEFT"]
        agent._step_env_callback = Mock(side_effect=AssertionError("must not execute"))
        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "current_frame": {
                            "grid": [[0]],
                            "ascii": "W",
                            "shape": [1, 1],
                            "step": 0,
                            "level": 1,
                        },
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            agent._session_runtime_dir = state_path

            verified = agent._verify_python_candidate("action('RIGHT')")

        self.assertFalse(verified)
        agent._step_env_callback.assert_not_called()

    def test_candidate_verification_fails_closed_for_missing_active_state(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        with TemporaryDirectory() as temp_dir:
            agent._session_runtime_dir = Path(temp_dir) / "missing-state.json"

            verified = agent._verify_python_candidate("result = 1")

        self.assertFalse(verified)
        self.assertEqual(
            agent._turn_efficiency_metrics["candidate_verification_failures"], 1
        )

    def test_candidate_verification_retries_one_missing_safe_import(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        state_path = Mock()
        state_path.is_file.return_value = True
        agent._session_runtime_dir = state_path
        failed = {
            "error": "NameError: Counter",
            "diagnostic": {
                "type": "NameError",
                "name": "Counter",
                "line": 1,
                "source": "result = Counter('ABBA')",
            },
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": {"A": 2, "B": 2},
            "action_results": [],
        }

        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[failed, successful],
            ) as sandbox,
        ):
            verified = agent._verify_python_candidate(
                "result = dict(Counter('ABBA'))"
            )

        self.assertTrue(verified)
        self.assertEqual(sandbox.call_count, 2)
        self.assertEqual(agent._candidate_verification_repair_count, 1)
        self.assertEqual(
            sandbox.call_args_list[1].kwargs["code"],
            "from collections import Counter\nresult = dict(Counter('ABBA'))",
        )
    def test_candidate_verification_chains_multiple_safe_imports(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        state_path = Mock()
        state_path.is_file.return_value = True
        agent._session_runtime_dir = state_path
        missing_counter = {
            "error": "NameError: Counter",
            "diagnostic": {"type": "NameError", "name": "Counter", "line": 1},
            "action_results": [],
        }
        missing_product = {
            "error": "NameError: product",
            "diagnostic": {"type": "NameError", "name": "product", "line": 3},
            "action_results": [],
        }
        successful = {"error": "", "result": 4, "action_results": []}
        code = (
            "counts = Counter('ABBA')\n"
            "result = len(list(product(counts, repeat=2)))"
        )

        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_counter, missing_product, successful],
            ) as sandbox,
        ):
            verified = agent._verify_python_candidate(code)

        self.assertTrue(verified)
        self.assertEqual(sandbox.call_count, 3)
        self.assertEqual(agent._candidate_verification_repair_count, 2)
        final_code = sandbox.call_args_list[2].kwargs["code"]
        self.assertIn("from collections import Counter", final_code)
        self.assertIn("from itertools import product", final_code)

    def test_candidate_verification_normalizes_json_literals(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        state_path = Mock()
        state_path.is_file.return_value = True
        agent._session_runtime_dir = state_path
        missing_true = {
            "error": "NameError: true",
            "diagnostic": {"type": "NameError", "name": "true", "line": 1},
            "action_results": [],
        }
        missing_null = {
            "error": "NameError: null",
            "diagnostic": {"type": "NameError", "name": "null", "line": 1},
            "action_results": [],
        }
        successful = {"error": "", "result": {"ok": True, "missing": None}}
        code = "result = {'ok': true, 'missing': null}"

        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_true, missing_null, successful],
            ) as sandbox,
        ):
            verified = agent._verify_python_candidate(code)

        self.assertTrue(verified)
        final_code = sandbox.call_args_list[2].kwargs["code"]
        self.assertIn("True", final_code)
        self.assertIn("None", final_code)
        self.assertNotIn("true", final_code)
        self.assertNotIn("null", final_code)

    def test_candidate_verification_rewrites_container_length(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        state_path = Mock()
        state_path.is_file.return_value = True
        agent._session_runtime_dir = state_path
        missing_length = {
            "error": "AttributeError: length",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "length",
                "object_type": "list",
                "line": 2,
            },
            "action_results": [],
        }
        successful = {"error": "", "result": 3, "action_results": []}
        code = "items = [1, 2, 3]\nresult = items.length"

        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_length, successful],
            ) as sandbox,
        ):
            verified = agent._verify_python_candidate(code)

        self.assertTrue(verified)
        self.assertIn("result = len(items)", sandbox.call_args.kwargs["code"])

        shadowed = "len = lambda value: 0\nitems = [1]\nresult = items.length"
        missing_length["diagnostic"]["line"] = 3
        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=missing_length,
            ) as sandbox,
        ):
            self.assertFalse(agent._verify_python_candidate(shadowed))
        sandbox.assert_called_once()

    def test_candidate_verification_rewrites_membership_methods(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        state_path = Mock()
        state_path.is_file.return_value = True
        agent._session_runtime_dir = state_path
        missing_includes = {
            "error": "AttributeError: includes",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "includes",
                "object_type": "list",
                "line": 3,
            },
            "action_results": [],
        }
        missing_own_property = {
            "error": "AttributeError: hasOwnProperty",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "hasOwnProperty",
                "object_type": "dict",
                "line": 3,
            },
            "action_results": [],
        }
        successful = {"error": "", "result": True, "action_results": []}
        code = (
            "items = [1, 2, 3]\n"
            "mapping = {'x': 1}\n"
            "result = items.includes(2) and mapping.hasOwnProperty('x')"
        )

        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_includes, missing_own_property, successful],
            ) as sandbox,
        ):
            verified = agent._verify_python_candidate(code)

        self.assertTrue(verified)
        final_code = sandbox.call_args.kwargs["code"]
        self.assertIn("2 in items", final_code)
        self.assertIn("'x' in mapping", final_code)

        invalid_arity = "result = [1, 2].includes(2, 0)"
        missing_includes["diagnostic"]["line"] = 1
        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=missing_includes,
            ) as sandbox,
        ):
            self.assertFalse(agent._verify_python_candidate(invalid_arity))
        sandbox.assert_called_once()

    def test_candidate_verification_rewrites_standalone_list_push(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        state_path = Mock()
        state_path.is_file.return_value = True
        agent._session_runtime_dir = state_path
        missing_push = {
            "error": "AttributeError: push",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "push",
                "object_type": "list",
                "line": 2,
            },
            "action_results": [],
        }
        successful = {"error": "", "result": [1, 2], "action_results": []}
        code = "items = []\nitems.push(1, 2)\nresult = items"

        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_push, successful],
            ) as sandbox,
        ):
            verified = agent._verify_python_candidate(code)

        self.assertTrue(verified)
        self.assertIn("items.extend([1, 2])", sandbox.call_args.kwargs["code"])

        value_used = "items = []\nresult = items.push(1)"
        with (
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(agent, "_cached_frame_payloads", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=missing_push,
            ) as sandbox,
        ):
            self.assertFalse(agent._verify_python_candidate(value_used))
        sandbox.assert_called_once()

    def test_candidate_selection_accepts_only_deterministically_repairable_code(self) -> None:
        def choice(code: str) -> dict:
            return {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "python",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }

        self.assertTrue(
            _choice_has_executable_tool(
                choice("if True\n    result = current_frame.objcts()")
            )
        )
        self.assertTrue(
            _choice_has_executable_tool(choice("result = True && False || True"))
        )
        self.assertTrue(
            _choice_has_executable_tool(choice("result = 1 === 1 and 2 !== 3"))
        )
        self.assertTrue(
            _choice_has_executable_tool(choice("result = !(1 === 2)"))
        )
        self.assertTrue(
            _choice_has_executable_tool(
                choice("const enabled = True && False\nresult = enabled")
            )
        )
        self.assertTrue(
            _choice_has_executable_tool(choice("value = 3\nreturn value"))
        )
        self.assertTrue(
            _choice_has_executable_tool(
                choice("def value():\n    return 3\nreturn value()")
            )
        )
        self.assertTrue(
            _choice_has_executable_tool(
                choice("result = current_frame.get('objects', [])")
            )
        )
        self.assertTrue(
            _choice_has_executable_tool(
                choice("result = last_transition.get('result')")
            )
        )
        self.assertTrue(
            _choice_has_executable_tool(choice("result = len([1, 2, 3"))
        )
        self.assertTrue(
            _choice_has_executable_tool(choice('result = len(["("'))
        )
        self.assertTrue(
            _choice_has_executable_tool(choice("    value = 3\n    result = value"))
        )
        self.assertTrue(
            _choice_has_executable_tool(
                choice("    result = current_frame.objcts()")
            )
        )
        self.assertTrue(
            _choice_has_executable_tool(
                choice(
                    "items = transitions\n"
                    "transition = items[0]\n"
                    "result = transition.aftr_frame"
                )
            )
        )
        self.assertFalse(
            _choice_has_executable_tool(
                choice("result = current_frame.not_a_real_attribute")
            )
        )
        self.assertFalse(
            _choice_has_executable_tool(
                choice("result = current_frame.get('not_documented')")
            )
        )
        self.assertFalse(
            _choice_has_executable_tool(
                choice("result = current_frame.get('objects', action('LEFT'))")
            )
        )
        large_default = ", ".join(str(index) for index in range(130))
        self.assertFalse(
            _choice_has_executable_tool(
                choice(f"result = current_frame.get('objects', [{large_default}])")
            )
        )
        self.assertFalse(_choice_has_executable_tool(choice("result = (")))
        self.assertFalse(
            _choice_has_executable_tool(choice("result = ([1, 2)]"))
        )
        self.assertFalse(
            _choice_has_executable_tool(choice("    value = 3\n        result = value"))
        )
        self.assertFalse(
            _choice_has_executable_tool(choice("if True:\n    return 1"))
        )
        self.assertFalse(
            _choice_has_executable_tool(choice("result = True & & False"))
        )
    def test_candidate_ranking_verifies_repaired_code_with_small_penalty(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        def choice(code: str) -> dict:
            return {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "python",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }

        with patch.object(
            agent, "_verify_python_candidate", return_value=True
        ) as verify:
            repaired_score, repaired_valid = agent._score_candidate_choice(
                choice("result = current_frame.objcts()")
            )
            native_score, native_valid = agent._score_candidate_choice(
                choice("result = current_frame.objects()")
            )

        self.assertTrue(repaired_valid)
        self.assertTrue(native_valid)
        self.assertEqual(native_score - repaired_score, 10)
        self.assertEqual(
            verify.call_args_list[0].args[0],
            "result = current_frame.objects()",
        )

    def test_candidate_ranking_penalizes_runtime_repairs(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        def choice(code: str) -> dict:
            return {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "python",
                                "arguments": json.dumps({"code": code}),
                            }
                        }
                    ]
                }
            }

        def verify(code: str) -> bool:
            agent._candidate_verification_repair_count = int("Counter" in code)
            return True

        with patch.object(agent, "_verify_python_candidate", side_effect=verify):
            repaired_score, repaired_valid = agent._score_candidate_choice(
                choice("result = Counter('A')")
            )
            native_score, native_valid = agent._score_candidate_choice(
                choice("result = {'A': 1}")
            )

        self.assertTrue(repaired_valid)
        self.assertTrue(native_valid)
        self.assertEqual(native_score - repaired_score, 5)

    def test_semantic_preflight_blocks_definite_failures_before_sandbox(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )

        cases = (
            ("import pathlib\nresult = 1", "Module 'pathlib'"),
            (
                "result = current_frame.not_a_real_attribute",
                "no documented attribute",
            ),
            ("while True:\n    result = 1", "has no break path"),
            ("while 1:\n    if False:\n        break", "has no break path"),
            ("action()", "exactly one positional argument"),
            (
                "def inspect_frame():\n"
                "    return current_frame.not_a_real_attribute\n"
                "result = inspect_frame()",
                "no documented attribute",
            ),
            (
                "def recurse():\n    return recurse()\nresult = recurse()",
                "Direct recursion",
            ),
            (
                "def recurse():\n"
                "    alias = recurse\n"
                "    return alias()\n"
                "result = recurse()",
                "Direct recursion",
            ),
            (
                "def recurse():\n"
                "    alias: object = recurse\n"
                "    return alias()\n"
                "result = recurse()",
                "Direct recursion",
            ),
            (
                "def first():\n    return second()\n"
                "def second():\n    return first()\n"
                "result = first()",
                "Mutual recursion",
            ),
            ("result = list(range(1000001))", "expands to 1000001 iterations"),
            (
                "result = list(range(500001 * 2))",
                "expands to 1000002 iterations",
            ),
        )
        for code, expected in cases:
            with (
                self.subTest(code=code),
                patch.object(tool_agent_module, "run_sandboxed_python") as sandbox,
            ):
                response = agent._run_python_tool(
                    Path("unused/tool_runtime_state.json"), {"code": code}
                )

            payload = json.loads(response.content)
            self.assertEqual(
                payload["diagnostic"]["type"], "GeneratedCodePreflightError"
            )
            self.assertIn(expected, payload["error"])
            sandbox.assert_not_called()

    def test_preflight_repairs_one_unambiguous_documented_attribute(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": [],
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "result = current_frame.objcts()"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], [])
        self.assertEqual(
            payload["auto_repair"]["repair"], "correct_documented_attribute"
        )
        self.assertEqual(payload["auto_repair"]["repair_count"], 1)
        repaired_code = sandbox.call_args.kwargs["code"]
        self.assertEqual(repaired_code, "result = current_frame.objects()")

    def test_preflight_repairs_multiple_unambiguous_attributes_atomically(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": [[], []],
            "stdout": "",
            "action_results": [],
        }
        code = (
            "first = current_frame.objcts()\n"
            "result = [first, current_frame.objcts()]"
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], [[], []])
        self.assertEqual(payload["auto_repair"]["repair_count"], 2)
        self.assertEqual(
            sandbox.call_args.kwargs["code"],
            "first = current_frame.objects()\n"
            "result = [first, current_frame.objects()]",
        )

    def test_preflight_allows_uncalled_mutually_recursive_helpers(self) -> None:
        tree = _parse_bounded_generated_python(
            "def first():\n    return second()\n"
            "def second():\n    return first()\n"
            "result = 1"
        )

        self.assertEqual(_generated_python_preflight_issues(tree), [])

    def test_preflight_discards_rebound_recursion_alias(self) -> None:
        tree = _parse_bounded_generated_python(
            "def compute():\n"
            "    alias = compute\n"
            "    alias = lambda: 7\n"
            "    return alias()\n"
            "result = compute()"
        )

        self.assertEqual(_generated_python_preflight_issues(tree), [])

    def test_preflight_preserves_recursion_in_assignment_rhs(self) -> None:
        tree = _parse_bounded_generated_python(
            "def compute():\n"
            "    alias = compute\n"
            "    alias = alias()\n"
            "    return 7\n"
            "result = compute()"
        )

        issues = _generated_python_preflight_issues(tree)
        self.assertIn("Direct recursion", issues[0]["message"])

    def test_preflight_discards_deleted_recursion_alias(self) -> None:
        tree = _parse_bounded_generated_python(
            "def compute():\n"
            "    alias = compute\n"
            "    del alias\n"
            "    return alias()\n"
            "result = compute()"
        )

        self.assertEqual(_generated_python_preflight_issues(tree), [])

    def test_deterministic_repair_applies_one_unambiguous_name_fix(self) -> None:
        repaired = _deterministic_generated_python_repair(
            "result = currnt_frame.shape",
            {
                "type": "NameError",
                "name": "currnt_frame",
                "suggestions": ["current_frame"],
            },
        )

        self.assertEqual(repaired, "result = current_frame.shape")
        self.assertIsNone(
            _deterministic_generated_python_repair(
                "result = missing",
                {
                    "type": "NameError",
                    "name": "missing",
                    "suggestions": ["memory", "history"],
                },
            )
        )

    def test_deterministic_repair_selects_one_clear_ranked_suggestion(self) -> None:
        repaired = _deterministic_generated_python_repair(
            "result = curent_frame.shape",
            {
                "type": "NameError",
                "name": "curent_frame",
                "suggestions": ["before_frame", "current_frame", "history"],
            },
        )

        self.assertEqual(repaired, "result = current_frame.shape")
        self.assertIsNone(
            _deterministic_generated_python_repair(
                "result = fram.shape",
                {
                    "type": "NameError",
                    "name": "fram",
                    "suggestions": ["frame", "frames"],
                },
            )
        )

    def test_deterministic_repair_is_syntax_aware(self) -> None:
        repaired = _deterministic_generated_python_repair(
            'label = "currnt_frame"\nresult = currnt_frame.shape',
            {
                "type": "NameError",
                "name": "currnt_frame",
                "line": 2,
                "suggestions": ["current_frame"],
            },
        )

        self.assertEqual(
            repaired,
            'label = "currnt_frame"\nresult = current_frame.shape',
        )
        self.assertIsNone(
            _deterministic_generated_python_repair(
                "first = missng\nsecond = missng",
                {
                    "type": "NameError",
                    "name": "missng",
                    "suggestions": ["missing"],
                },
            )
        )

    def test_preflight_invalidates_rebound_view_aliases(self) -> None:
        for code in (
            "frame = current_frame\ndel frame\nresult = frame.objcts",
            "frame = current_frame\nframe += 1\nresult = frame.objcts",
            "frame = current_frame\ndef frame():\n    return 1\nresult = frame.objcts",
            "frame = current_frame\nfor frame in [1]:\n    pass\nresult = frame.objcts",
            "frame = current_frame\nwith manager as frame:\n    pass\nresult = frame.objcts",
        ):
            with self.subTest(code=code):
                tree = _parse_bounded_generated_python(code)
                self.assertEqual(_generated_python_preflight_issues(tree), [])

    def test_preflight_tracks_top_level_assignment_expression_aliases(self) -> None:
        inferred = _parse_bounded_generated_python(
            "(frame := current_frame)\nresult = frame.objcts()"
        )
        invalidated = _parse_bounded_generated_python(
            "frame = current_frame\n(frame := 1)\nresult = frame.objcts"
        )

        inferred_issues = _generated_python_preflight_issues(inferred)
        self.assertEqual(len(inferred_issues), 1)
        self.assertEqual(inferred_issues[0]["suggestions"], ["objects"])
        self.assertEqual(_generated_python_preflight_issues(invalidated), [])

    def test_preflight_uses_view_provenance_at_source_position(self) -> None:
        tree = _parse_bounded_generated_python(
            "result = current_frame.objcts\ncurrent_frame = object()"
        )

        issues = _generated_python_preflight_issues(tree)
        self.assertEqual(issues[0]["suggestions"], ["objects"])

    def test_preflight_tracks_view_aliases_through_constant_branch(self) -> None:
        tree = _parse_bounded_generated_python(
            "if True:\n    frame = current_frame\nresult = frame.objcts"
        )

        issues = _generated_python_preflight_issues(tree)
        self.assertEqual(issues[0]["suggestions"], ["objects"])

    def test_static_action_analysis_skips_dead_and_uncalled_code(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": 1,
            "stdout": "",
            "action_results": [],
        }
        code = (
            "def unused():\n"
            "    action()\n"
            "    return current_frame.objcts()\n"
            "if False:\n"
            "    action()\n"
            "    result = current_frame.objcts()\n"
            "result = 1"
        )

        with patch.object(
            tool_agent_module,
            "run_sandboxed_python",
            return_value=successful,
        ) as sandbox:
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        self.assertEqual(json.loads(response.content)["result"], 1)
        sandbox.assert_called_once()


    def test_run_python_tool_retries_one_safe_deterministic_repair(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        failed = {
            "error": "NameError: currnt_frame",
            "diagnostic": {
                "type": "NameError",
                "name": "currnt_frame",
                "source": "result = currnt_frame.shape",
                "suggestions": ["current_frame"],
                "retry": "correct_and_retry",
            },
            "stdout": "",
            "action_results": [],
        }
        repaired = {
            "error": "",
            "result": [2, 2],
            "stdout": "",
            "action_results": [],
        }

        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}),
                encoding="utf-8",
            )
            with patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[failed, repaired],
            ) as sandbox:
                response = agent._run_python_tool(
                    state_path, {"code": "result = currnt_frame.shape"}
                )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], [2, 2])
        self.assertTrue(payload["auto_repair"]["applied"])
        self.assertEqual(sandbox.call_count, 2)

    def test_run_python_tool_inserts_one_missing_safe_import(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        failed = {
            "error": "NameError: Counter",
            "diagnostic": {
                "type": "NameError",
                "name": "Counter",
                "line": 1,
                "source": "result = dict(Counter('ABBA'))",
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": {"A": 2, "B": 2},
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[failed, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "result = dict(Counter('ABBA'))"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], {"A": 2, "B": 2})
        self.assertEqual(payload["auto_repair"]["repair"], "insert_safe_import")
        self.assertEqual(
            sandbox.call_args_list[1].kwargs["code"],
            "from collections import Counter\nresult = dict(Counter('ABBA'))",
        )
        self.assertEqual(
            payload["auto_repair"]["repaired_code_fingerprint"],
            _generated_python_semantic_fingerprint(
                sandbox.call_args_list[1].kwargs["code"]
            ),
        )

    def test_safe_imports_precede_fuzzy_identifier_repairs(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_mean = {
            "error": "NameError: mean",
            "diagnostic": {
                "type": "NameError",
                "name": "mean",
                "line": 1,
                "suggestions": ["memory"],
            },
            "stdout": "",
            "action_results": [],
        }
        missing_sqrt = {
            "error": "NameError: sqrt",
            "diagnostic": {
                "type": "NameError",
                "name": "sqrt",
                "line": 2,
                "suggestions": ["str"],
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": 2.5,
            "stdout": "",
            "action_results": [],
        }
        code = "result = mean([sqrt(4), sqrt(9)])"

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_mean, missing_sqrt, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 2.5)
        self.assertEqual(
            payload["auto_repair"]["repairs"],
            ["insert_safe_import", "insert_safe_import"],
        )
        repaired = sandbox.call_args.kwargs["code"]
        self.assertIn("from statistics import mean", repaired)
        self.assertIn("from math import sqrt", repaired)
        self.assertNotIn("memory([", repaired)

    def test_safe_import_completion_supports_sequence_analysis_helpers(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_prod = {
            "error": "NameError: prod",
            "diagnostic": {"type": "NameError", "name": "prod", "line": 1},
            "stdout": "",
            "action_results": [],
        }
        missing_pairwise = {
            "error": "NameError: pairwise",
            "diagnostic": {"type": "NameError", "name": "pairwise", "line": 2},
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": 6,
            "stdout": "",
            "action_results": [],
        }
        code = "result = prod(b - a for a, b in pairwise([1, 3, 6]))"

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_prod, missing_pairwise, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 6)
        self.assertEqual(payload["auto_repair"]["repair_count"], 2)
        repaired = sandbox.call_args.kwargs["code"]
        self.assertIn("from math import prod", repaired)
        self.assertIn("from itertools import pairwise", repaired)

    def test_safe_import_completion_supports_integer_grid_helpers(self) -> None:
        isqrt_repair = tool_agent_module._deterministic_safe_import_repair(
            "result = isqrt(81)",
            {"type": "NameError", "name": "isqrt", "line": 1},
        )
        namedtuple_repair = tool_agent_module._deterministic_safe_import_repair(
            "Point = namedtuple('Point', 'row col')\nresult = Point(1, 2)",
            {"type": "NameError", "name": "namedtuple", "line": 1},
        )

        self.assertEqual(isqrt_repair, "from math import isqrt\nresult = isqrt(81)")
        self.assertIn("from collections import namedtuple", namedtuple_repair)
        self.assertEqual(
            tool_agent_module._deterministic_safe_import_repair(
                "result = lcm(6, 8)",
                {"type": "NameError", "name": "lcm", "line": 1},
            ),
            "from math import lcm\nresult = lcm(6, 8)",
        )

    def test_safe_import_completion_supports_geometry_and_operator_helpers(self) -> None:
        expected = {
            "degrees": "from math import degrees",
            "radians": "from math import radians",
            "hypot": "from math import hypot",
            "attrgetter": "from operator import attrgetter",
            "methodcaller": "from operator import methodcaller",
        }
        for name, import_line in expected.items():
            with self.subTest(name=name):
                repaired = tool_agent_module._deterministic_safe_import_repair(
                    f"result = {name}(value)",
                    {"type": "NameError", "name": name, "line": 1},
                )
                self.assertTrue(repaired.startswith(f"{import_line}\n"))

    def test_run_python_tool_chains_multiple_side_effect_free_repairs(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_counter = {
            "error": "NameError: Counter",
            "diagnostic": {"type": "NameError", "name": "Counter", "line": 1},
            "stdout": "",
            "action_results": [],
        }
        missing_product = {
            "error": "NameError: product",
            "diagnostic": {"type": "NameError", "name": "product", "line": 3},
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": 4,
            "stdout": "",
            "action_results": [],
        }
        code = (
            "counts = Counter('ABBA')\n"
            "result = len(list(product(counts, repeat=2)))"
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_counter, missing_product, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 4)
        self.assertEqual(payload["auto_repair"]["repair"], "repair_chain")
        self.assertEqual(payload["auto_repair"]["repair_count"], 2)
        self.assertEqual(
            payload["auto_repair"]["repairs"],
            ["insert_safe_import", "insert_safe_import"],
        )
        self.assertEqual(sandbox.call_count, 3)

    def test_run_python_tool_supports_more_than_four_safe_repairs(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        names = ["Counter", "product", "sqrt", "mean", "pairwise"]
        failures = [
            {
                "error": f"NameError: {name}",
                "diagnostic": {
                    "type": "NameError",
                    "name": name,
                    "line": 2 * index + 1,
                },
                "stdout": "",
                "action_results": [],
            }
            for index, name in enumerate(names)
        ]
        successful = {
            "error": "",
            "result": 1,
            "stdout": "",
            "action_results": [],
        }
        code = "\n".join(
            [
                "counts = Counter('AA')",
                "pairs = list(product([1], repeat=2))",
                "root = sqrt(4)",
                "average = mean([root])",
                "steps = list(pairwise([1, 2]))",
                "result = len(counts) + len(pairs) + len(steps) - int(average)",
            ]
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[*failures, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 1)
        self.assertEqual(payload["auto_repair"]["repair_count"], 5)
        self.assertEqual(sandbox.call_count, 6)

    def test_effectful_call_detection_ignores_strings_and_comments(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_counter = {
            "error": "NameError: Counter",
            "diagnostic": {
                "type": "NameError",
                "name": "Counter",
                "line": 7,
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": {"A": 1},
            "stdout": "",
            "action_results": [],
        }
        code = (
            'label = "remember("\n'
            "# forget(\n"
            "if False:\n"
            '    remember("dead", 1)\n'
            "def unused():\n"
            '    forget("also_dead")\n'
            'result = dict(Counter("A"))'
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_counter, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], {"A": 1})
        self.assertEqual(payload["auto_repair"]["repair"], "insert_safe_import")
        self.assertEqual(sandbox.call_count, 2)

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=missing_counter,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": 'remember("key", 1)\nresult = dict(Counter("A"))'},
            )

        self.assertIn("NameError", json.loads(response.content)["error"])
        sandbox.assert_called_once()

    def test_effectful_call_detection_follows_called_function_alias(self) -> None:
        code = (
            "def mutate():\n"
            "    remember('key', 1)\n"
            "alias = mutate\n"
            "alias()\n"
            "result = Counter('A')"
        )

        self.assertTrue(
            tool_agent_module._generated_python_calls_any(
                code, frozenset({"remember", "forget", "record_strategy"})
            )
        )
        branch_alias = code.replace("alias = mutate", "if True:\n    alias = mutate")
        self.assertTrue(
            tool_agent_module._generated_python_calls_any(
                branch_alias, frozenset({"remember", "forget", "record_strategy"})
            )
        )

    def test_effectful_call_detection_respects_alias_rebinding_order(self) -> None:
        prefix = "def mutate():\n    remember('key', 1)\nalias = mutate\n"
        names = frozenset({"remember", "forget", "record_strategy"})

        self.assertFalse(
            tool_agent_module._generated_python_calls_any(
                prefix + "alias = lambda: None\nalias()", names
            )
        )
        self.assertTrue(
            tool_agent_module._generated_python_calls_any(
                prefix + "alias()\nalias = lambda: None", names
            )
        )

    def test_effectful_call_detection_preserves_assignment_rhs_call(self) -> None:
        code = (
            "def mutate():\n"
            "    remember('key', 1)\n"
            "alias = mutate\n"
            "alias = alias()"
        )

        self.assertTrue(
            tool_agent_module._generated_python_calls_any(
                code, frozenset({"remember", "forget", "record_strategy"})
            )
        )

    def test_effectful_call_detection_follows_transitive_helper_alias(self) -> None:
        code = (
            "def mutate():\n"
            "    remember('key', 1)\n"
            "def wrapper():\n"
            "    alias = mutate\n"
            "    alias()\n"
            "wrapper()"
        )

        self.assertTrue(
            tool_agent_module._generated_python_calls_any(
                code, frozenset({"remember", "forget", "record_strategy"})
            )
        )

    def test_effectful_call_detection_ignores_unrelated_object_method(self) -> None:
        self.assertFalse(
            tool_agent_module._generated_python_calls_any(
                "logger.remember('message')\nresult = 1",
                frozenset({"remember", "forget", "record_strategy"}),
            )
        )

    def test_effectful_call_detection_discards_deleted_alias(self) -> None:
        code = (
            "def mutate():\n"
            "    remember('key', 1)\n"
            "alias = mutate\n"
            "del alias\n"
            "alias()"
        )

        self.assertFalse(
            tool_agent_module._generated_python_calls_any(
                code, frozenset({"remember", "forget", "record_strategy"})
            )
        )

    def test_run_python_tool_reports_json_literal_repair_chain(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_true = {
            "error": "NameError: true",
            "diagnostic": {"type": "NameError", "name": "true", "line": 1},
            "stdout": "",
            "action_results": [],
        }
        missing_null = {
            "error": "NameError: null",
            "diagnostic": {"type": "NameError", "name": "null", "line": 1},
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": {"ok": True, "missing": None},
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_true, missing_null, successful],
            ),
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "result = {'ok': true, 'missing': null}"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], {"ok": True, "missing": None})
        self.assertEqual(payload["auto_repair"]["repair_count"], 2)
        self.assertEqual(
            payload["auto_repair"]["repairs"],
            ["normalize_json_literal", "normalize_json_literal"],
        )

    def test_length_repair_accepts_mapping_subclass_diagnostic(self) -> None:
        repaired = tool_agent_module._deterministic_length_attribute_repair(
            "result = payload.length",
            {
                "type": "AttributeError",
                "attribute": "length",
                "object_type": "ScoreMap",
                "mapping_type": True,
                "line": 1,
            },
        )

        self.assertEqual(repaired, "result = len(payload)")

    def test_length_repair_accepts_size_property(self) -> None:
        repaired = tool_agent_module._deterministic_length_attribute_repair(
            "result = values.size",
            {
                "type": "AttributeError",
                "attribute": "size",
                "object_type": "set",
                "line": 1,
            },
        )

        self.assertEqual(repaired, "result = len(values)")

    def test_length_repair_accepts_dictionary_view(self) -> None:
        repaired = tool_agent_module._deterministic_length_attribute_repair(
            "result = keys.length",
            {
                "type": "AttributeError",
                "attribute": "length",
                "object_type": "dict_keys",
                "line": 1,
            },
        )

        self.assertEqual(repaired, "result = len(keys)")

    def test_run_python_tool_rewrites_container_length(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_length = {
            "error": "AttributeError: length",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "length",
                "object_type": "list",
                "line": 2,
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": 3,
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_length, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "items = [1, 2, 3]\nresult = items.length"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 3)
        self.assertEqual(
            payload["auto_repair"]["repair"], "replace_length_with_len"
        )
        self.assertIn("result = len(items)", sandbox.call_args.kwargs["code"])

    def test_run_python_tool_rewrites_documented_view_subscription(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        not_subscriptable = {
            "error": "TypeError: 'FrameView' object is not subscriptable",
            "diagnostic": {
                "type": "TypeError",
                "object_type": "FrameView",
                "operation": "subscription",
                "line": 1,
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": 2,
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[not_subscriptable, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "result = len(current_frame['objects'])"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 2)
        self.assertEqual(
            payload["auto_repair"]["repair"], "replace_view_subscription"
        )
        self.assertIn(
            "result = len(current_frame.objects)", sandbox.call_args.kwargs["code"]
        )

    def test_view_subscription_repair_rejects_undocumented_keys(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        not_subscriptable = {
            "error": "TypeError: 'FrameView' object is not subscriptable",
            "diagnostic": {
                "type": "TypeError",
                "object_type": "FrameView",
                "operation": "subscription",
                "line": 1,
            },
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=not_subscriptable,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "result = current_frame['not_documented']"},
            )

        payload = json.loads(response.content)
        self.assertIn("not subscriptable", payload["error"])
        self.assertNotIn("auto_repair", payload)
        sandbox.assert_called_once()

    def test_run_python_tool_rewrites_documented_view_get(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        successful = {
            "error": "",
            "result": 2,
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=successful,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "result = len(current_frame.get('objects', []))"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 2)
        self.assertEqual(payload["auto_repair"]["repair"], "replace_view_get")
        self.assertIn(
            "result = len(current_frame.objects)", sandbox.call_args.kwargs["code"]
        )

    def test_run_python_tool_rewrites_exact_mapping_attribute(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_attribute = {
            "error": "AttributeError: 'dict' object has no attribute 'score'",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "score",
                "object_type": "dict",
                "mapping_keys": ["score", "reward"],
                "line": 2,
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": 7,
            "stdout": "",
            "action_results": [],
        }
        code = "payload = {'score': 7}\nresult = payload.score"

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_attribute, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], 7)
        self.assertEqual(
            payload["auto_repair"]["repair"], "replace_mapping_attribute"
        )
        self.assertIn("result = payload['score']", sandbox.call_args.kwargs["code"])

    def test_mapping_attribute_repair_requires_an_exact_runtime_key(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_attribute = {
            "error": "AttributeError: 'dict' object has no attribute 'score'",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "score",
                "object_type": "dict",
                "mapping_keys": ["reward"],
                "line": 2,
            },
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=missing_attribute,
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "payload = {'reward': 1}\nresult = payload.score"},
            )

        payload = json.loads(response.content)
        self.assertIn("has no attribute", payload["error"])
        self.assertNotIn("auto_repair", payload)
        sandbox.assert_called_once()

    def test_mapping_attribute_repair_accepts_one_clear_key_typo(self) -> None:
        diagnostic = {
            "type": "AttributeError",
            "attribute": "scroe",
            "object_type": "dict",
            "mapping_keys": ["reward", "score"],
            "line": 2,
        }
        code = "payload = {'score': 7}\nresult = payload.scroe"

        repaired = tool_agent_module._deterministic_mapping_attribute_repair(
            code, diagnostic
        )

        self.assertEqual(repaired, "payload = {'score': 7}\nresult = payload['score']")
        ambiguous = dict(diagnostic, attribute="fram", mapping_keys=["frame", "frames"])
        self.assertIsNone(
            tool_agent_module._deterministic_mapping_attribute_repair(
                "payload = {}\nresult = payload.fram", ambiguous
            )
        )

    def test_mapping_attribute_repair_accepts_dict_subclass_diagnostic(self) -> None:
        repaired = tool_agent_module._deterministic_mapping_attribute_repair(
            "result = payload.score",
            {
                "type": "AttributeError",
                "object_type": "ScoreMap",
                "mapping_type": True,
                "attribute": "score",
                "mapping_keys": ["score"],
                "line": 1,
            },
        )

        self.assertEqual(repaired, "result = payload['score']")

    def test_run_python_tool_corrects_safe_builtin_keyword_typo(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        unexpected_keyword = {
            "error": "TypeError: 'revers' is an invalid keyword argument for sort()",
            "diagnostic": {
                "type": "TypeError",
                "operation": "unexpected_keyword",
                "keyword": "revers",
                "line": 1,
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": [3, 2, 1],
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[unexpected_keyword, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "result = sorted([1, 3, 2], revers=True)"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], [3, 2, 1])
        self.assertEqual(
            payload["auto_repair"]["repair"], "correct_builtin_keyword"
        )
        self.assertIn("reverse=True", sandbox.call_args.kwargs["code"])

    def test_builtin_keyword_repair_rejects_shadowed_builtin(self) -> None:
        diagnostic = {
            "type": "TypeError",
            "operation": "unexpected_keyword",
            "keyword": "revers",
            "line": 3,
        }
        code = (
            "def sorted(values, **kwargs):\n"
            "    return values\n"
            "result = sorted([2, 1], revers=True)"
        )

        repaired = tool_agent_module._deterministic_builtin_keyword_repair(
            code, diagnostic
        )

        self.assertIsNone(repaired)

    def test_list_sort_keyword_repair_requires_static_list_receiver(self) -> None:
        diagnostic = {
            "type": "TypeError",
            "operation": "unexpected_keyword",
            "callable": "sort",
            "keyword": "revers",
            "line": 2,
        }
        code = "items = [3, 1, 2]\nitems.sort(revers=True)\nresult = items"

        repaired = tool_agent_module._deterministic_list_sort_keyword_repair(
            code, diagnostic
        )

        self.assertIn("items.sort(reverse=True)", repaired)
        self.assertIsNone(
            tool_agent_module._deterministic_list_sort_keyword_repair(
                "items = source\nitems.sort(revers=True)", diagnostic
            )
        )

    def test_list_sort_keyword_repair_respects_receiver_rebinding(self) -> None:
        diagnostic = {
            "type": "TypeError",
            "operation": "unexpected_keyword",
            "callable": "sort",
            "keyword": "revers",
            "line": 3,
        }

        self.assertIsNone(
            tool_agent_module._deterministic_list_sort_keyword_repair(
                "items = []\nitems = source\nitems.sort(revers=True)",
                diagnostic,
            )
        )

    def test_list_sort_keyword_repair_tracks_alias_and_constant_branch(self) -> None:
        diagnostic = {
            "type": "TypeError",
            "operation": "unexpected_keyword",
            "callable": "sort",
            "keyword": "revers",
            "line": 4,
        }
        code = (
            "if True:\n"
            "    items = [3, 1, 2]\n"
            "ordered = items\n"
            "ordered.sort(revers=True)"
        )

        repaired = tool_agent_module._deterministic_list_sort_keyword_repair(
            code, diagnostic
        )

        self.assertIn("ordered.sort(reverse=True)", repaired)

    def test_list_sort_keyword_repair_tracks_function_local_list(self) -> None:
        diagnostic = {
            "type": "TypeError",
            "operation": "unexpected_keyword",
            "callable": "sort",
            "keyword": "revers",
            "line": 3,
        }
        code = (
            "def order():\n"
            "    items = [3, 1, 2]\n"
            "    items.sort(revers=True)\n"
            "    return items\n"
            "result = order()"
        )

        repaired = tool_agent_module._deterministic_list_sort_keyword_repair(
            code, diagnostic
        )

        self.assertIn("items.sort(reverse=True)", repaired)

    def test_list_sort_keyword_repair_invalidates_loop_target(self) -> None:
        diagnostic = {
            "type": "TypeError",
            "operation": "unexpected_keyword",
            "callable": "sort",
            "keyword": "revers",
            "line": 4,
        }
        code = (
            "items = []\n"
            "for items in sources:\n"
            "    pass\n"
            "items.sort(revers=True)"
        )

        self.assertIsNone(
            tool_agent_module._deterministic_list_sort_keyword_repair(
                code, diagnostic
            )
        )

    def test_membership_repair_accepts_mapping_subclass_diagnostic(self) -> None:
        repaired = tool_agent_module._deterministic_membership_method_repair(
            "result = payload.hasOwnProperty('score')",
            {
                "type": "AttributeError",
                "attribute": "hasOwnProperty",
                "object_type": "ScoreMap",
                "mapping_type": True,
                "line": 1,
            },
        )

        self.assertEqual(repaired, "result = 'score' in payload")

    def test_string_method_repair_accepts_starts_with(self) -> None:
        repaired = tool_agent_module._deterministic_string_method_repair(
            "result = label.startsWith('arc', 1)",
            {
                "type": "AttributeError",
                "attribute": "startsWith",
                "object_type": "str",
                "line": 1,
            },
        )

        self.assertEqual(repaired, "result = label.startswith('arc', 1)")

    def test_string_method_repair_accepts_ends_with(self) -> None:
        repaired = tool_agent_module._deterministic_string_method_repair(
            "result = label.endsWith('arc')",
            {
                "type": "AttributeError",
                "attribute": "endsWith",
                "object_type": "str",
                "line": 1,
            },
        )

        self.assertEqual(repaired, "result = label.endswith('arc')")

    def test_run_python_tool_chains_membership_method_repairs(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_includes = {
            "error": "AttributeError: includes",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "includes",
                "object_type": "list",
                "line": 3,
            },
            "stdout": "",
            "action_results": [],
        }
        missing_own_property = {
            "error": "AttributeError: hasOwnProperty",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "hasOwnProperty",
                "object_type": "dict",
                "line": 3,
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": True,
            "stdout": "",
            "action_results": [],
        }
        code = (
            "items = [1, 2, 3]\n"
            "mapping = {'x': 1}\n"
            "result = items.includes(2) and mapping.hasOwnProperty('x')"
        )

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_includes, missing_own_property, successful],
            ),
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"), {"code": code}
            )

        payload = json.loads(response.content)
        self.assertTrue(payload["result"])
        self.assertEqual(payload["auto_repair"]["repair_count"], 2)
        self.assertEqual(
            payload["auto_repair"]["repairs"],
            ["replace_membership_method", "replace_membership_method"],
        )

    def test_run_python_tool_rewrites_standalone_list_push(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        missing_push = {
            "error": "AttributeError: push",
            "diagnostic": {
                "type": "AttributeError",
                "attribute": "push",
                "object_type": "list",
                "line": 2,
            },
            "stdout": "",
            "action_results": [],
        }
        successful = {
            "error": "",
            "result": [1],
            "stdout": "",
            "action_results": [],
        }

        with (
            patch.object(agent, "_ensure_session"),
            patch.object(tool_agent_module, "load_runtime_state", return_value=(None, [])),
            patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                side_effect=[missing_push, successful],
            ) as sandbox,
        ):
            response = agent._run_python_tool(
                Path("unused/tool_runtime_state.json"),
                {"code": "items = []\nitems.push(1)\nresult = items"},
            )

        payload = json.loads(response.content)
        self.assertEqual(payload["result"], [1])
        self.assertEqual(
            payload["auto_repair"]["repair"], "replace_list_push"
        )
        self.assertIn("items.append(1)", sandbox.call_args.kwargs["code"])

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

    def test_analyze_requests_complete_regeneration_after_truncation(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._tool_steps = 1
        agent._chat_completion = lambda *_args, **_kwargs: _ChatCompletionResult(
            message={"content": "partial tool payload"},
            finish_reason="length",
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

        self.assertFalse(result.step_executed)
        self.assertIn("truncated by the output limit", transcript)
        self.assertIn("complete tool call from the beginning", transcript)

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

    def test_analyze_tries_valid_runner_up_after_side_effect_free_failure(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._tool_steps = 3
        completion = Mock(
            return_value=_ChatCompletionResult(
                message={
                    "tool_calls": [
                        {
                            "id": "failing-candidate",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": '{"code":"result = missing_name"}',
                            },
                        }
                    ]
                },
                finish_reason="tool_calls",
                fallback_messages=[
                    {
                        "tool_calls": [
                            {
                                "id": "runner-up",
                                "type": "function",
                                "function": {
                                    "name": "python",
                                    "arguments": '{"code":"action(\\"LEFT\\")"}',
                                },
                            }
                        ]
                    }
                ],
            )
        )
        agent._chat_completion = completion

        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}),
                encoding="utf-8",
            )
            result = agent.analyze(
                state_path,
                action_num=0,
                valid_actions=["LEFT"],
                step_env=lambda _request: {
                    "executed": True,
                    "action_num": 1,
                    "action_display": "LEFT",
                    "valid_actions": ["LEFT"],
                },
            )
            transcript = state_path.with_name(
                "tool_runtime_state_analyzer.txt"
            ).read_text(encoding="utf-8")

        self.assertTrue(result.step_executed)
        self.assertEqual(completion.call_count, 1)
        self.assertIn("candidate_fallback", transcript)

    def test_analyze_suppresses_identical_failed_program(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._tool_steps = 2
        agent._chat_completion = Mock(
            return_value=_ChatCompletionResult(
                message={
                    "tool_calls": [
                        {
                            "id": "same-failure",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": '{"code":"result = missing_name"}',
                            },
                        }
                    ]
                },
                finish_reason="tool_calls",
            )
        )
        sandbox_failure = {
            "error": "NameError: missing_name",
            "diagnostic": {
                "type": "NameError",
                "name": "missing_name",
                "source": "result = missing_name",
                "suggestions": [],
                "retry": "correct_and_retry",
            },
            "stdout": "",
            "action_results": [],
        }

        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}),
                encoding="utf-8",
            )
            with patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=sandbox_failure,
            ) as sandbox:
                agent.analyze(state_path, action_num=0, valid_actions=["LEFT"])
            transcript = state_path.with_name(
                "tool_runtime_state_analyzer.txt"
            ).read_text(encoding="utf-8")

        self.assertEqual(sandbox.call_count, 1)
        self.assertIn("Duplicate failed program suppressed", transcript)

    def test_static_action_analysis_inspects_function_default(self) -> None:
        tree = _parse_bounded_generated_python(
            "def helper(moves=action([{'action': 'LEFT'}])):\n"
            "    return moves"
        )

        self.assertEqual(
            tool_agent_module._static_candidate_action_arguments(tree),
            [[{"action": "LEFT"}]],
        )

    def test_static_action_analysis_inspects_lambda_default(self) -> None:
        tree = _parse_bounded_generated_python(
            "helper = lambda moves=action([{'action': 'LEFT'}]): moves"
        )

        self.assertEqual(
            tool_agent_module._static_candidate_action_arguments(tree),
            [[{"action": "LEFT"}]],
        )

    def test_static_action_analysis_tracks_direct_walrus_binding(self) -> None:
        tree = _parse_bounded_generated_python(
            "(moves := [{'action': 'LEFT'}])\naction(moves)"
        )

        self.assertEqual(
            tool_agent_module._static_candidate_action_arguments(tree),
            [[{"action": "LEFT"}]],
        )

    def test_static_action_analysis_excludes_empty_for_body(self) -> None:
        tree = _parse_bounded_generated_python(
            "for item in []:\n"
            "    action([{'action': 'LEFT'}])\n"
            "result = 1"
        )

        self.assertEqual(
            tool_agent_module._static_candidate_action_arguments(tree), []
        )

    def test_static_action_analysis_inspects_function_annotation(self) -> None:
        tree = _parse_bounded_generated_python(
            "def helper(value: action([{'action': 'LEFT'}])):\n"
            "    return value"
        )

        self.assertEqual(
            tool_agent_module._static_candidate_action_arguments(tree),
            [[{"action": "LEFT"}]],
        )

    def test_repeated_failure_triggers_quality_model_failover(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._tool_steps = 2
        agent._chat_completion = Mock(
            side_effect=[
                _ChatCompletionResult(
                    message={
                        "tool_calls": [
                            {
                                "id": "failure-a",
                                "type": "function",
                                "function": {
                                    "name": "python",
                                    "arguments": '{"code":"result = missing_name"}',
                                },
                            }
                        ]
                    }
                ),
                _ChatCompletionResult(
                    message={
                        "tool_calls": [
                            {
                                "id": "failure-b",
                                "type": "function",
                                "function": {
                                    "name": "python",
                                    "arguments": (
                                        '{"code":"# different candidate\\n'
                                        'result = missing_name"}'
                                    ),
                                },
                            }
                        ]
                    }
                ),
            ]
        )
        failover = Mock(return_value=True)
        agent._activate_next_fallback_model = failover
        sandbox_failure = {
            "error": "NameError: missing_name",
            "diagnostic": {
                "type": "NameError",
                "name": "missing_name",
                "source": "result = missing_name",
                "suggestions": [],
                "retry": "correct_and_retry",
            },
            "stdout": "",
            "action_results": [],
        }

        with TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "tool_runtime_state.json"
            state_path.write_text(
                json.dumps({"current_frame": None, "history": []}),
                encoding="utf-8",
            )
            with patch.object(
                tool_agent_module,
                "run_sandboxed_python",
                return_value=sandbox_failure,
            ):
                agent.analyze(state_path, action_num=0, valid_actions=["LEFT"])

        failover.assert_called_once_with()

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

    def test_error_payload_preserves_committed_action_journal(self) -> None:
        payload = _python_tool_payload(
            {
                "stdout": "before failure",
                "error": "NameError: missing",
                "diagnostic": {
                    "type": "NameError",
                    "retry": "correct_and_retry",
                    "hint": "fix it",
                },
                "action_results": [
                    {"executed": True, "action_display": "LEFT", "action_num": 1}
                ],
            }
        )

        self.assertFalse(payload["retryable"])
        self.assertEqual(payload["action_results"][0]["action_display"], "LEFT")
        self.assertTrue(payload["partial_execution"]["side_effects_committed"])
        self.assertFalse(payload["partial_execution"]["rollback_available"])
        self.assertEqual(
            payload["diagnostic"]["retry"], "replan_from_current_state"
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
