from __future__ import annotations

import json
from unittest import TestCase

from inference.utils.openai_compat import (
    build_chat_payload,
    build_provider_request,
    build_responses_payload,
    merge_chat_completion_stream,
    normalize_provider,
    normalize_provider_response,
    normalize_responses_response,
)


class OpenAICompatibilityTests(TestCase):
    def _payload(self, provider: str) -> dict:
        return build_chat_payload(
            provider=provider,
            model="unit-test-model",
            messages=[{"role": "user", "content": "go"}],
            max_tokens=128,
            temperature=0.2,
            top_p=0.9,
            top_k=20,
            thinking=True,
            seed=7,
            candidates=2,
        )

    def test_official_openai_does_not_receive_vllm_only_fields(self) -> None:
        payload = self._payload("openai")

        self.assertEqual(normalize_provider("openai"), "openai")
        self.assertEqual(payload["n"], 2)
        self.assertNotIn("top_k", payload)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("seed", payload)

    def test_openai_compatible_default_preserves_vllm_extensions(self) -> None:
        payload = self._payload("openai-compatible")

        self.assertEqual(normalize_provider("openai-compatible"), "vllm")
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["seed"], 7)

    def test_responses_payload_translates_tools_and_outputs(self) -> None:
        payload = build_responses_payload(
            model="unit-test-model",
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call-1", "function": {"name": "python", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "done"},
            ],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
            tools=[{"type": "function", "function": {"name": "python", "parameters": {"type": "object"}, "strict": True}}],
            tool_choice={"type": "function", "function": {"name": "python"}},
        )

        self.assertEqual(normalize_provider("responses"), "openai-responses")
        self.assertEqual(payload["max_output_tokens"], 256)
        self.assertEqual(payload["tools"][0]["name"], "python")
        self.assertTrue(payload["tools"][0]["strict"])
        self.assertEqual(payload["tool_choice"], {"type": "function", "name": "python"})
        self.assertEqual(payload["input"][-1]["type"], "function_call_output")

    def test_provider_adapter_selects_endpoint_and_normalizer(self) -> None:
        request = build_provider_request(
            provider="openai-responses",
            model="unit-test-model",
            messages=[{"role": "user", "content": "go"}],
            max_tokens=64,
            temperature=0.2,
            top_p=0.9,
            top_k=0,
            thinking=False,
        )

        self.assertEqual(request.endpoint, "responses")
        self.assertTrue(request.responses_api)
        normalized = normalize_provider_response(
            "openai-responses",
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]},
        )
        self.assertEqual(normalized["choices"][0]["message"]["content"], "ok")

    def test_chat_provider_adapter_caps_candidates_and_preserves_chat_contract(self) -> None:
        request = build_provider_request(
            provider="openai-compatible",
            model="unit-test-model",
            messages=[{"role": "user", "content": "go"}],
            max_tokens=None,
            temperature=0.0,
            top_p=1.0,
            top_k=10,
            thinking=False,
            candidates=99,
        )

        self.assertEqual(request.endpoint, "chat/completions")
        self.assertFalse(request.responses_api)
        self.assertEqual(request.payload["n"], 4)
        self.assertNotIn("max_tokens", request.payload)

    def test_non_responses_normalizer_returns_payload_unchanged(self) -> None:
        payload = {"choices": [], "usage": {"completion_tokens": 3}}

        normalized = normalize_provider_response("openrouter", payload)

        self.assertIs(normalized, payload)

    def test_responses_payload_translates_multipart_text_and_images(self) -> None:
        payload = build_responses_payload(
            model="unit-test-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                        {"type": "unknown", "value": "ignored"},
                    ],
                }
            ],
            max_tokens=None,
            temperature=0.0,
            top_p=1.0,
        )

        parts = payload["input"][0]["content"]
        self.assertEqual([part["type"] for part in parts], ["input_text", "input_image"])
        self.assertEqual(parts[1]["image_url"], "data:image/png;base64,AA==")
        self.assertNotIn("max_output_tokens", payload)

    def test_normalizes_responses_function_call_and_truncation(self) -> None:
        normalized = normalize_responses_response(
            {
                "output": [{"type": "function_call", "call_id": "c1", "name": "python", "arguments": '{"code":"result=1"}'}],
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        choice = normalized["choices"][0]
        self.assertEqual(choice["finish_reason"], "length")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "python")
        self.assertEqual(normalized["usage"]["prompt_tokens"], 10)

    def test_normalizes_responses_skips_malformed_items_and_joins_reasoning(self) -> None:
        normalized = normalize_responses_response(
            {
                "output": [
                    None,
                    {"type": "message", "content": [None, {"type": "text", "text": "one"}, {"type": "output_text", "text": "two"}]},
                    {"type": "reasoning", "summary": [{"text": "first"}, "ignored", {"text": "second"}]},
                ],
                "usage": {"input_tokens": 4, "prompt_tokens": 9, "output_tokens": 2},
            }
        )

        choice = normalized["choices"][0]
        self.assertEqual(choice["message"]["content"], "one\ntwo")
        self.assertEqual(choice["message"]["reasoning"], "first\nsecond")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(normalized["usage"]["prompt_tokens"], 9)
        self.assertEqual(normalized["usage"]["completion_tokens"], 2)

    def test_provider_normalization_rejects_non_object_top_level_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON object"):
            normalize_provider_response("openai", [])  # type: ignore[arg-type]

    def test_responses_normalizer_bounds_malformed_nested_shapes(self) -> None:
        normalized = normalize_responses_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "name": "python",
                        "arguments": {"code": "result = 1"},
                    },
                    {"type": "message", "content": "not-a-list"},
                    {"type": "reasoning", "summary": "not-a-list"},
                ],
                "usage": "not-an-object",
                "incomplete_details": "not-an-object",
            }
        )

        choice = normalized["choices"][0]
        arguments = choice["message"]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(arguments), {"code": "result = 1"})
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(normalized["usage"], {})

    def test_merges_streamed_tool_argument_fragments(self) -> None:
        merged = merge_chat_completion_stream(
            [
                {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "python", "arguments": '{"code":'}}]}}]},
                {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"result=1"}'}}]}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 7}},
            ]
        )

        choice = merged["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["arguments"], '{"code":"result=1"}')
        self.assertEqual(merged["usage"]["completion_tokens"], 7)
