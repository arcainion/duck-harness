from __future__ import annotations

from unittest import TestCase

from inference.utils.openai_compat import (
    build_chat_payload,
    build_responses_payload,
    merge_chat_completion_stream,
    normalize_provider,
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
