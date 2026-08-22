from __future__ import annotations

from unittest import TestCase

from inference.utils.openai_compat import build_chat_payload, normalize_provider


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
