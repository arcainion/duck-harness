from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from inference.agent import tool_agent
from inference.agent.tool_agent import (
    ToolAgent,
    _estimated_json_length,
    _estimated_request_base_length,
)


class ToolAgentContextTrimTests(unittest.TestCase):
    @staticmethod
    def _messages() -> list[dict]:
        return [
            {"role": "system", "content": "system " * 40},
            {"role": "user", "content": "first observation " * 20},
            {
                "role": "assistant",
                "content": "inspect",
                "tool_calls": [{"id": "one", "function": {"name": "python"}}],
            },
            {"role": "tool", "tool_call_id": "one", "content": "result " * 30},
            {"role": "user", "content": "second observation " * 18},
            {"role": "assistant", "content": "revised model " * 20},
            {"role": "user", "content": "latest observation " * 15},
        ]

    @staticmethod
    def _legacy_trim(
        agent: ToolAgent,
        messages: list[dict],
        *,
        tools: list[dict],
        preserve_recent: int,
    ) -> list[dict]:
        system_message = messages[0]
        history = list(messages[1:])
        while (
            history
            and agent._estimate_request_input_tokens(
                [system_message, *history],
                tools=tools,
            )
            > agent._context_budget_tokens
        ):
            if not agent._drop_oldest_history_block(
                history,
                preserve_recent=max(0, preserve_recent),
            ):
                break
        return [system_message, *agent._drop_until_first_user_message(history)]

    def test_incremental_json_length_matches_whole_request_serialization(self) -> None:
        messages = self._messages()
        tools = [{"type": "function", "function": {"name": "python"}}]
        forced_choice = {"type": "function", "function": {"name": "python"}}
        payload = {
            "messages": messages,
            "tool_choice": forced_choice,
            "tools": tools,
        }
        empty_payload = {"messages": [], "tool_choice": forced_choice, "tools": tools}
        incremental_length = (
            _estimated_json_length(empty_payload)
            + sum(_estimated_json_length(message) for message in messages)
            + 2 * (len(messages) - 1)
        )

        self.assertEqual(incremental_length, _estimated_json_length(payload))
        self.assertEqual(
            _estimated_json_length(payload),
            len(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)),
        )

    def test_incremental_trimming_matches_legacy_results_without_reserializing(self) -> None:
        messages = self._messages()
        tools = [{"type": "function", "function": {"name": "python"}}]
        for budget in (80, 160, 320, 640, 1280):
            for preserve_recent in (0, 1, 3):
                with self.subTest(budget=budget, preserve_recent=preserve_recent):
                    agent = ToolAgent(model="unit-test-model")
                    agent._context_budget_tokens = budget
                    expected = self._legacy_trim(
                        agent,
                        messages,
                        tools=tools,
                        preserve_recent=preserve_recent,
                    )

                    with patch.object(
                        agent,
                        "_estimate_request_input_tokens",
                        side_effect=AssertionError("whole request was reserialized"),
                    ):
                        actual = agent._trim_messages_for_context(
                            messages,
                            tools=tools,
                            preserve_recent=preserve_recent,
                        )

                    self.assertEqual(actual, expected)

    def test_turn_local_cache_serializes_each_message_only_once(self) -> None:
        agent = ToolAgent(model="unit-test-model")
        agent._context_budget_tokens = 100_000
        messages = self._messages()
        tools = [{"type": "function", "function": {"name": "python"}}]
        request_base_chars = _estimated_request_base_length(tools)
        cache: dict[int, tuple[dict, int]] = {}

        with patch.object(
            tool_agent,
            "_estimated_json_length",
            wraps=tool_agent._estimated_json_length,
        ) as estimate_length:
            first = agent._trim_messages_for_context(
                messages,
                tools=tools,
                request_base_chars=request_base_chars,
                message_length_cache=cache,
            )
            first_call_count = estimate_length.call_count
            second = agent._trim_messages_for_context(
                messages,
                tools=tools,
                request_base_chars=request_base_chars,
                message_length_cache=cache,
            )
            second_call_count = estimate_length.call_count
            extended = [*messages, {"role": "assistant", "content": "new response"}]
            third = agent._trim_messages_for_context(
                extended,
                tools=tools,
                request_base_chars=request_base_chars,
                message_length_cache=cache,
            )

        self.assertEqual(first, messages)
        self.assertEqual(second, messages)
        self.assertEqual(third, extended)
        self.assertEqual(first_call_count, len(messages))
        self.assertEqual(second_call_count, first_call_count)
        self.assertEqual(estimate_length.call_count, first_call_count + 1)


if __name__ == "__main__":
    unittest.main()
