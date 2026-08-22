from __future__ import annotations

import unittest
from pathlib import Path

from inference.agent.tool_agent import ToolAgent, _bounded_python_memory


class PythonToolMemoryTests(unittest.TestCase):
    def test_accepts_and_normalizes_compact_json_memory(self) -> None:
        memory = _bounded_python_memory(
            {
                "target": [3, 7],
                "plan": {"moves": ["UP", "RIGHT"], "confidence": 0.8},
            }
        )

        self.assertEqual(memory["target"], [3, 7])
        self.assertEqual(memory["plan"]["moves"], ["UP", "RIGHT"])

    def test_rejects_non_json_and_oversized_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            _bounded_python_memory({"bad": float("nan")})
        with self.assertRaisesRegex(ValueError, "2048 bytes"):
            _bounded_python_memory({"large": "x" * 2050})
        with self.assertRaisesRegex(ValueError, "8192 bytes total"):
            _bounded_python_memory(
                {str(index): "x" * 1800 for index in range(5)}
            )

    def test_rejects_too_many_keys_and_long_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "16 keys"):
            _bounded_python_memory({str(index): index for index in range(17)})
        with self.assertRaisesRegex(ValueError, "64 characters"):
            _bounded_python_memory({"x" * 65: 1})

    def test_agent_persists_memory_until_session_reset(self) -> None:
        agent = object.__new__(ToolAgent)
        agent._python_memory = {}

        persisted = agent._record_python_memory({"route": ["LEFT", "UP"]})

        self.assertEqual(persisted, {"route": ["LEFT", "UP"]})
        self.assertEqual(agent._python_memory, persisted)

        agent._session_runtime_dir = Path("old-session")
        agent._ensure_session(Path("new-session/tool_runtime_state.json"))
        self.assertEqual(agent._python_memory, {})

    def test_distinct_state_files_in_same_directory_are_distinct_sessions(self) -> None:
        agent = object.__new__(ToolAgent)
        first_path = Path("shared-session/first.json").resolve()
        agent._session_runtime_dir = first_path
        agent._python_memory = {"private": "first-game"}

        agent._ensure_session(Path("shared-session/second.json"))

        self.assertEqual(agent._python_memory, {})
        self.assertEqual(
            agent._session_runtime_dir,
            Path("shared-session/second.json").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
