from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from inference.agent.python_tool_sandbox import _send_json_line
from inference.agent.tool_agent import (
    ToolAgent,
    _PYTHON_TOOL_DESCRIPTION,
    _TranscriptBuffer,
    _build_system_prompt,
    _estimate_tokens,
    _request_tool_choice,
)


class _Response:
    def __init__(self, status_code: int, text: str, payload: dict, lines: list[str] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self.closed = False
        self._lines = lines or []

    def close(self) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self) -> dict:
        return self._payload

    def iter_lines(self, decode_unicode: bool = False):
        del decode_unicode
        return iter(self._lines)


class ToolAgentPromptEfficiencyTests(unittest.TestCase):
    def test_tool_schema_is_compact_and_system_prompt_remains_authoritative(self) -> None:
        system_prompt = _build_system_prompt(tool_output_tokens=1024)

        self.assertLess(len(_PYTHON_TOOL_DESCRIPTION), 1_200)
        self.assertLess(_estimate_tokens(_PYTHON_TOOL_DESCRIPTION), 400)
        for capability in (
            "find_pattern",
            "shortest_path",
            "compare_regions",
            "border_summary",
            "tile_summary",
        ):
            self.assertIn(capability, system_prompt)

    def test_generated_tool_schema_uses_the_compact_description(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        tools = agent._tools(Path("unused/tool_runtime_state.json"))

        self.assertEqual(
            tools[0]["function"]["description"],
            _PYTHON_TOOL_DESCRIPTION,
        )
        self.assertLess(_estimate_tokens(tools), 800)
        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["python", "action", "inspect"],
        )
        self.assertTrue(tools[0]["function"]["strict"])
        parameters = tools[0]["function"]["parameters"]
        self.assertFalse(parameters["additionalProperties"])
        self.assertGreater(parameters["properties"]["code"]["maxLength"], 0)

    def test_dynamic_turn_prompt_does_not_repeat_static_api_guidance(self) -> None:
        agent = ToolAgent(model="unit-test-model")

        prompt = agent._build_user_prompt(
            0,
            valid_actions=["LEFT", "RIGHT"],
        )

        self.assertLess(len(prompt), 1_000)
        self.assertLess(_estimate_tokens(prompt), 350)
        self.assertIn("Current state: step 1, level 1", prompt)
        self.assertIn("Valid actions right now: LEFT, RIGHT", prompt)
        self.assertIn("revised version of this working world model", prompt)
        self.assertIn("action(actions)", prompt)
        self.assertNotIn("shortest_path_to_any", prompt)
        self.assertNotIn("record_strategy(goal=", prompt)

    def test_sandbox_protocol_uses_compact_unicode_json_lines(self) -> None:
        handle = io.StringIO()
        payload = {"type": "result", "value": "π", "items": [1, 2]}

        byte_count = _send_json_line(handle, payload)

        encoded = handle.getvalue()
        self.assertEqual(encoded, '{"type":"result","value":"π","items":[1,2]}\n')
        self.assertEqual(json.loads(encoded), payload)
        self.assertEqual(byte_count, len(encoded.encode("utf-8")))

    def test_single_python_tool_is_forced_with_auto_fallback_available(self) -> None:
        tools = [{"type": "function", "function": {"name": "python"}}]

        self.assertEqual(
            _request_tool_choice(tools),
            {"type": "function", "function": {"name": "python"}},
        )
        self.assertEqual(_request_tool_choice(tools, force=False), "auto")

    def test_tool_choice_handles_empty_multiple_and_malformed_tools(self) -> None:
        python_tool = {"type": "function", "function": {"name": "python"}}

        self.assertIsNone(_request_tool_choice(None))
        self.assertIsNone(_request_tool_choice([]))
        self.assertEqual(_request_tool_choice([python_tool, python_tool]), "auto")
        self.assertEqual(_request_tool_choice([{"type": "function"}]), "auto")
        self.assertEqual(_request_tool_choice(["bad"]), "auto")

    def test_model_facing_tool_payload_is_compact(self) -> None:
        agent = ToolAgent(model="unit-test-model")

        rendered = agent._render_tool_payload({"result": [1, 2], "ok": True})

        self.assertEqual(rendered, '{"result":[1,2],"ok":true}')
        self.assertNotIn("\n", rendered)

    def test_tool_payload_truncates_only_requested_string_fields(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._tool_output_chars = 16

        rendered = json.loads(
            agent._render_tool_payload(
                {"stdout": "x" * 40, "result": "y" * 40},
                truncate_fields=("stdout",),
            )
        )

        self.assertTrue(rendered["truncated"])
        self.assertIn("truncated", rendered["stdout"])
        self.assertEqual(rendered["result"], "y" * 40)
        self.assertIn("token response budget", rendered["truncation_note"])

    def test_tool_text_exact_boundary_is_not_truncated(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._tool_output_chars = 8

        self.assertEqual(agent._trim_tool_text("12345678"), ("12345678", False))
        shortened, truncated = agent._trim_tool_text("123456789")
        self.assertTrue(truncated)
        self.assertLessEqual(len(shortened), agent._tool_output_chars + 80)

    def test_server_default_output_limit_is_not_adaptively_capped(self) -> None:
        with patch(
            "inference.agent.tool_agent._LOCAL_ANALYZER_MAX_OUTPUT", 0
        ):
            agent = ToolAgent(model="unit-test-model")

        self.assertIsNone(agent._max_output_tokens)
        self.assertEqual(agent._reply_reserve_tokens, 512)
        self.assertIsNone(agent._adaptive_output_limit(0))
        self.assertIsNone(agent._adaptive_output_limit(2))
        self.assertIsNone(agent._adaptive_output_limit(100, repair=True))

    def test_server_default_output_limit_is_omitted_from_payload(self) -> None:
        with patch(
            "inference.agent.tool_agent._LOCAL_ANALYZER_MAX_OUTPUT", 0
        ):
            agent = ToolAgent(model="unit-test-model")
        agent._http_session.post = Mock(
            return_value=_Response(
                200,
                "",
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )
        )

        agent._chat_completion([{"role": "user", "content": "go"}], tools=None)

        payload = agent._http_session.post.call_args.kwargs["json"]
        self.assertNotIn("max_tokens", payload)

    def test_efficiency_metrics_accumulate_ints_and_floats(self) -> None:
        agent = ToolAgent(model="unit-test-model")

        agent._record_efficiency("calls", 1)
        agent._record_efficiency("calls", 2)
        agent._record_efficiency("seconds", 0.25)
        agent._record_efficiency("seconds", 0.5)

        self.assertEqual(agent._turn_efficiency_metrics["calls"], 3)
        self.assertEqual(agent._turn_efficiency_metrics["seconds"], 0.75)

    def test_transcript_buffer_ignores_empty_sections_and_close_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transcript.txt"
            transcript = _TranscriptBuffer(path, "header\n")

            self.assertEqual(transcript.append("EMPTY", "  \n"), "")
            transcript.append("RESULT", " value ")
            first_render = transcript.render()
            self.assertIs(first_render, transcript.render())
            transcript.close()
            transcript.close()

            self.assertEqual(first_render, "header\n[RESULT]\nvalue\n\n")
            self.assertEqual(path.read_text(encoding="utf-8"), first_render)

    def test_chat_completion_reuses_session_and_downgrades_rejected_forced_choice(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        rejected = _Response(400, "tool_choice function form unsupported", {})
        agent._http_session.post = Mock(
            side_effect=[
                rejected,
                _Response(
                    200,
                    "",
                    {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
                ),
            ]
        )
        tools = [{"type": "function", "function": {"name": "python"}}]

        result = agent._chat_completion([{"role": "user", "content": "go"}], tools=tools)

        self.assertEqual(result.message["content"], "ok")
        self.assertEqual(result.request_attempts, 2)
        self.assertTrue(result.forced_tool_fallback)
        self.assertFalse(agent._forced_tool_choice_supported)
        self.assertEqual(agent._http_session.post.call_count, 2)
        self.assertTrue(rejected.closed)
        self.assertEqual(agent._http_session.post.call_args_list[1].kwargs["json"]["tool_choice"], "auto")

    def test_chat_completion_does_not_retry_unrelated_client_error(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._http_session.post = Mock(
            return_value=_Response(400, "messages are invalid", {})
        )
        tools = [{"type": "function", "function": {"name": "python"}}]

        with self.assertRaises(requests.RequestException):
            agent._chat_completion([{"role": "user", "content": "go"}], tools=tools)

        self.assertEqual(agent._http_session.post.call_count, 1)
        self.assertIsNone(agent._forced_tool_choice_supported)

    def test_chat_completion_sticks_to_auto_after_forced_choice_fallback(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._forced_tool_choice_supported = False
        agent._http_session.post = Mock(
            return_value=_Response(
                200,
                "",
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )
        )
        tools = [{"type": "function", "function": {"name": "python"}}]

        result = agent._chat_completion(
            [{"role": "user", "content": "go"}],
            tools=tools,
            max_output_tokens=321,
        )

        payload = agent._http_session.post.call_args.kwargs["json"]
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(result.request_attempts, 1)
        self.assertFalse(result.forced_tool_fallback)

    def test_chat_completion_accepts_bounded_thinking_and_required_tool_choice(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._http_session.post = Mock(
            return_value=_Response(
                200,
                "",
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "python",
                                            "arguments": '{"code":"result=1"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        )
        tools = [{"type": "function", "function": {"name": "python"}}]

        result = agent._chat_completion(
            [{"role": "user", "content": "go"}],
            tools=tools,
            thinking_token_budget=3072,
            tool_choice="required",
            request_attempt_limit=1,
        )

        payload = agent._http_session.post.call_args.kwargs["json"]
        self.assertEqual(payload["thinking_token_budget"], 3072)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(result.request_attempts, 1)
        self.assertIsNone(agent._forced_tool_choice_supported)

    def test_chat_completion_request_attempt_override_disables_transient_retry(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        failure = _Response(500, "forced tool parser failed", {})
        agent._http_session.post = Mock(return_value=failure)

        with self.assertRaisesRegex(requests.RequestException, "forced tool parser"):
            agent._chat_completion(
                [{"role": "user", "content": "go"}],
                tools=None,
                request_attempt_limit=1,
            )

        self.assertEqual(agent._http_session.post.call_count, 1)
        self.assertTrue(failure.closed)

    def test_chat_completion_downgrades_rejected_strict_schema(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._http_session.post = Mock(
            side_effect=[
                _Response(400, "strict schema unsupported", {}),
                _Response(200, "", {"choices": [{"message": {"content": "ok"}}]}),
            ]
        )
        tools = agent._tools(Path("unused/tool_runtime_state.json"))

        result = agent._chat_completion([{"role": "user", "content": "go"}], tools=tools)

        fallback_tool = agent._http_session.post.call_args_list[1].kwargs["json"]["tools"][0]["function"]
        self.assertEqual(result.message["content"], "ok")
        self.assertNotIn("strict", fallback_tool)
        self.assertNotIn("additionalProperties", fallback_tool["parameters"])
        self.assertFalse(agent._strict_tools_supported)

    def test_chat_completion_uses_responses_api(self) -> None:
        agent = ToolAgent(
            model="unit-test-model",
            provider="openai-responses",
            base_url="https://api.openai.test/v1",
        )
        agent._http_session.post = Mock(
            return_value=_Response(
                200,
                "",
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "python",
                            "arguments": '{"code":"result=1"}',
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        )
        tools = agent._tools(Path("unused/tool_runtime_state.json"))

        result = agent._chat_completion([{"role": "user", "content": "go"}], tools=tools)

        call = agent._http_session.post.call_args
        self.assertEqual(call.args[0], "https://api.openai.test/v1/responses")
        self.assertIn("input", call.kwargs["json"])
        self.assertNotIn("messages", call.kwargs["json"])
        self.assertEqual(result.message["tool_calls"][0]["function"]["name"], "python")
        self.assertEqual(result.usage["prompt_tokens"], 10)

    def test_chat_completion_assembles_streamed_tool_call(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        response = _Response(
            200,
            "",
            {},
            lines=[
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"python","arguments":"{\\\"code\\\":"}}]}}]}',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\\"result=1\\\"}"}}]},"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            ],
        )
        agent._http_session.post = Mock(return_value=response)
        tools = [{"type": "function", "function": {"name": "python"}}]

        with patch("inference.agent.tool_agent._LOCAL_ANALYZER_STREAM", True):
            result = agent._chat_completion([{"role": "user", "content": "go"}], tools=tools)

        self.assertTrue(agent._http_session.post.call_args.kwargs["stream"])
        self.assertEqual(
            result.message["tool_calls"][0]["function"]["arguments"],
            '{"code":"result=1"}',
        )
        self.assertTrue(agent._streaming_supported)

    def test_successful_forced_choice_marks_endpoint_supported(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._http_session.post = Mock(
            return_value=_Response(
                200,
                "",
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )
        )
        tools = [{"type": "function", "function": {"name": "python"}}]

        agent._chat_completion([{"role": "user", "content": "go"}], tools=tools)

        payload = agent._http_session.post.call_args.kwargs["json"]
        self.assertEqual(
            payload["tool_choice"],
            {"type": "function", "function": {"name": "python"}},
        )
        self.assertTrue(agent._forced_tool_choice_supported)

    def test_failed_auto_fallback_propagates_second_response(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._http_max_attempts = 2
        agent._http_retry_base_seconds = 0
        rejected = _Response(422, "function tool_choice is unsupported", {})
        fallback_failure = _Response(503, "server overloaded", {})
        agent._http_session.post = Mock(side_effect=[rejected, fallback_failure])
        tools = [{"type": "function", "function": {"name": "python"}}]

        with self.assertRaisesRegex(requests.RequestException, "server overloaded"):
            agent._chat_completion([{"role": "user", "content": "go"}], tools=tools)

        self.assertTrue(rejected.closed)
        self.assertFalse(agent._forced_tool_choice_supported)
        self.assertEqual(agent._http_session.post.call_count, 2)

    def test_chat_completion_rejects_success_response_without_choices(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._http_session.post = Mock(return_value=_Response(200, "", {"choices": []}))

        with self.assertRaisesRegex(requests.RequestException, "no choices"):
            agent._chat_completion([{"role": "user", "content": "go"}], tools=None)

    def test_chat_completion_recovers_from_transient_status_and_connection_error(self) -> None:
        for first_failure in (
            _Response(503, "temporarily unavailable", {}),
            requests.ConnectionError("connection reset"),
        ):
            with self.subTest(failure=type(first_failure).__name__):
                agent = ToolAgent(model="unit-test-model")
                agent._http_retry_base_seconds = 0
                success = _Response(
                    200,
                    "",
                    {"choices": [{"message": {"content": "ok"}}]},
                )
                agent._http_session.post = Mock(side_effect=[first_failure, success])

                result = agent._chat_completion(
                    [{"role": "user", "content": "go"}], tools=None
                )

                self.assertEqual(result.message, {"content": "ok"})
                self.assertEqual(result.request_attempts, 2)
                self.assertEqual(agent._http_session.post.call_count, 2)
                if isinstance(first_failure, _Response):
                    self.assertTrue(first_failure.closed)

    def test_chat_completion_bounds_transient_retries(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._http_max_attempts = 3
        agent._http_retry_base_seconds = 0
        failures = [_Response(503, "overloaded", {}) for _ in range(3)]
        agent._http_session.post = Mock(side_effect=failures)

        with self.assertRaisesRegex(requests.RequestException, "overloaded"):
            agent._chat_completion([{"role": "user", "content": "go"}], tools=None)

        self.assertEqual(agent._http_session.post.call_count, 3)
        self.assertTrue(all(response.closed for response in failures))

    def test_chat_completion_rejects_malformed_success_envelopes(self) -> None:
        invalid_json = _Response(200, "", {})
        invalid_json.json = Mock(side_effect=ValueError("invalid json"))
        cases = (
            (invalid_json, "invalid JSON"),
            (_Response(200, "", []), "non-object response"),
            (_Response(200, "", {"choices": [None]}), "invalid choice"),
            (_Response(200, "", {"choices": [{}]}), "without a message"),
        )
        for response, expected in cases:
            with self.subTest(expected=expected):
                agent = ToolAgent(model="unit-test-model")
                agent._http_session.post = Mock(return_value=response)

                with self.assertRaisesRegex(requests.RequestException, expected):
                    agent._chat_completion(
                        [{"role": "user", "content": "go"}], tools=None
                    )

                self.assertTrue(response.closed)

    def test_chat_completion_ranks_multiple_verified_candidates(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._candidate_count = 2
        agent._current_valid_actions = ["LEFT"]
        response = _Response(
            200,
            "",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "python",
                                        "arguments": '{"code":"for"}',
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "python",
                                        "arguments": '{"code":"action(\\"LEFT\\")"}',
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    },
                ]
            },
        )
        agent._http_session.post = Mock(return_value=response)

        result = agent._chat_completion(
            [{"role": "user", "content": "go"}], tools=None
        )

        self.assertEqual(result.selected_candidate_index, 1)
        self.assertEqual(result.candidate_count, 2)
        self.assertEqual(result.valid_candidate_count, 1)
        self.assertEqual(result.fallback_messages, [])
        request_payload = agent._http_session.post.call_args.kwargs["json"]
        self.assertEqual(request_payload["n"], 2)

    def test_chat_completion_retains_ranked_valid_runner_up(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._candidate_count = 2
        agent._current_valid_actions = ["LEFT", "RIGHT"]
        response = _Response(
            200,
            "",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "python",
                                        "arguments": '{"code":"action(\\"LEFT\\")"}',
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    },
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "python",
                                        "arguments": '{"code":"action(\\"RIGHT\\")"}',
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    },
                ]
            },
        )
        agent._http_session.post = Mock(return_value=response)

        result = agent._chat_completion(
            [{"role": "user", "content": "go"}], tools=None
        )

        self.assertEqual(result.selected_candidate_index, 0)
        self.assertEqual(len(result.fallback_messages), 1)
        self.assertIn(
            "RIGHT",
            result.fallback_messages[0]["tool_calls"][0]["function"]["arguments"],
        )

    def test_chat_completion_estimates_generated_tokens_when_usage_is_missing(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        response = _Response(
            200,
            "",
            {
                "choices": [
                    {
                        "message": {"content": "A sufficiently long generated response."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )
        agent._http_session.post = Mock(return_value=response)

        result = agent._chat_completion(
            [{"role": "user", "content": "go"}], tools=None
        )
        agent._accumulate_usage_tokens(result.usage)

        self.assertGreater(result.usage["generated_tokens"], 0)
        self.assertEqual(
            agent._session_generated_tokens, result.usage["generated_tokens"]
        )

    def test_agent_activates_configured_fallback_model(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        config_type = type(agent._model)
        agent._forced_tool_choice_supported = False
        agent._fallback_models = [
            config_type(
                provider="openai",
                base_url="https://fallback.invalid/v1",
                model_id="fallback-model",
            )
        ]

        with self.assertLogs("inference.agent.tool_agent", level="WARNING"):
            activated = agent._activate_next_fallback_model()

        self.assertTrue(activated)
        self.assertEqual(agent._model.model_id, "fallback-model")
        self.assertEqual(agent._model.provider, "openai")
        self.assertIsNone(agent._forced_tool_choice_supported)
        self.assertFalse(agent._activate_next_fallback_model())


if __name__ == "__main__":
    unittest.main()
