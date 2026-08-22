from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from inference.agent import tool_agent
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent


class ToolAgentFramePayloadCacheTests(unittest.TestCase):
    def test_reuses_payload_for_identical_frame_objects(self) -> None:
        agent = ToolAgent(model="test-model")
        frame = Frame(grid=((0, 1), (2, 3)), step=1, level=1)
        history = [HistoryEntry(action="RIGHT", frame=frame)]

        with patch.object(
            tool_agent,
            "_ascii_history_view_payload",
            wraps=tool_agent._ascii_history_view_payload,
        ) as render_history:
            first = agent._cached_frame_payloads(frame, history)
            second = agent._cached_frame_payloads(frame, list(history))

        self.assertIs(second[0], first[0])
        self.assertIs(second[1], first[1])
        self.assertEqual(render_history.call_count, 1)

    def test_rebuilds_payload_after_frame_identity_changes(self) -> None:
        agent = ToolAgent(model="test-model")
        first = Frame(grid=((0,),), step=1, level=1)
        second = Frame(grid=((1,),), step=2, level=1)

        first_payload, _first_history = agent._cached_frame_payloads(
            first,
            [HistoryEntry(action="LEFT", frame=first)],
        )
        second_payload, _second_history = agent._cached_frame_payloads(
            second,
            [HistoryEntry(action="RIGHT", frame=second)],
        )

        self.assertNotEqual(first_payload, second_payload)
        self.assertEqual(second_payload["step"], 2)

    def test_new_session_clears_cached_payload(self) -> None:
        agent = ToolAgent(model="test-model")
        frame = Frame(grid=((0,),), step=1, level=1)
        agent._cached_frame_payloads(frame, [])
        agent._cached_experience_snapshot(frame, [], ["LEFT"])

        agent._ensure_session(Path("first-state.json"))

        self.assertIsNone(agent._frame_payload_cache)
        self.assertIsNone(agent._experience_snapshot_cache)

    def test_reuses_experience_for_unchanged_state_and_actions(self) -> None:
        agent = ToolAgent(model="test-model")
        frame = Frame(grid=((0, 1),), step=1, level=1)
        history = [HistoryEntry(action="RIGHT", frame=frame)]

        with patch.object(
            tool_agent,
            "build_experience_snapshot",
            wraps=tool_agent.build_experience_snapshot,
        ) as build_snapshot:
            first = agent._cached_experience_snapshot(
                frame,
                history,
                ["LEFT", "RIGHT"],
            )
            second = agent._cached_experience_snapshot(
                frame,
                list(history),
                ["LEFT", "RIGHT"],
            )

        self.assertIs(second, first)
        self.assertEqual(build_snapshot.call_count, 1)

    def test_experience_rebuilds_when_valid_actions_change(self) -> None:
        agent = ToolAgent(model="test-model")
        frame = Frame(grid=((0,),), step=1, level=1)
        history = [HistoryEntry(action="", frame=frame)]

        with patch.object(
            tool_agent,
            "build_experience_snapshot",
            wraps=tool_agent.build_experience_snapshot,
        ) as build_snapshot:
            agent._cached_experience_snapshot(frame, history, ["LEFT"])
            agent._cached_experience_snapshot(frame, history, ["RIGHT"])

        self.assertEqual(build_snapshot.call_count, 2)


if __name__ == "__main__":
    unittest.main()
