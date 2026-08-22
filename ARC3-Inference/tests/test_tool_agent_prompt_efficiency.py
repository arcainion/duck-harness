from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from inference.agent.python_tool_sandbox import _send_json_line
from inference.agent.tool_agent import (
    ToolAgent,
    _PYTHON_TOOL_DESCRIPTION,
    _build_system_prompt,
    _estimate_tokens,
)


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
        self.assertLess(_estimate_tokens(tools), 500)

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

        _send_json_line(handle, payload)

        encoded = handle.getvalue()
        self.assertEqual(encoded, '{"type":"result","value":"π","items":[1,2]}\n')
        self.assertEqual(json.loads(encoded), payload)


if __name__ == "__main__":
    unittest.main()
