from __future__ import annotations

import json
import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    run_sandboxed_python,
)
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import (
    _ascii_frame_view_payload,
    _ascii_history_view_payload,
    _current_frame_transport_payload,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class PythonToolHistoryTransportTests(unittest.TestCase):
    def test_sparse_change_uses_delta_and_reconstructs_frame_view(self) -> None:
        first_grid = tuple((0, 0, 0, 0) for _row in range(4))
        second_grid = tuple(
            (0, 1, 0, 0) if row == 0 else (0, 0, 0, 0)
            for row in range(4)
        )
        first = Frame(grid=first_grid, step=0, level=1)
        second = Frame(grid=second_grid, step=1, level=1)
        payload = _ascii_history_view_payload(
            [
                HistoryEntry(action="", frame=first),
                HistoryEntry(action="RIGHT", frame=second),
            ]
        )

        self.assertIn("frame", payload[0])
        self.assertIn("frame_delta", payload[1])
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        history = namespace["_history_from_payload"](payload)

        self.assertEqual(history[1].action, "RIGHT")
        self.assertEqual(history[1].frame.cell(0, 1), ARC_COLOR_CHARS[1])
        self.assertEqual(history[1].frame.ascii, "WwWW\nWWWW\nWWWW\nWWWW")
        self.assertEqual(history[1].frame.step, 1)

    def test_unchanged_frame_uses_empty_delta(self) -> None:
        frame = Frame(grid=((0, 1),), step=0, level=1)
        next_frame = Frame(grid=frame.grid, step=1, level=1)
        payload = _ascii_history_view_payload(
            [
                HistoryEntry(action="", frame=frame),
                HistoryEntry(action="WAIT", frame=next_frame),
            ]
        )

        self.assertEqual(payload[1]["frame_delta"]["changes"], [])

    def test_dense_or_shape_changed_frame_stays_full(self) -> None:
        first = Frame(grid=((0, 0), (0, 0)), step=0, level=1)
        dense = Frame(grid=((1, 1), (1, 1)), step=1, level=1)
        resized = Frame(grid=((1, 1, 1),), step=2, level=1)
        payload = _ascii_history_view_payload(
            [
                HistoryEntry(action="", frame=first),
                HistoryEntry(action="A", frame=dense),
                HistoryEntry(action="B", frame=resized),
            ]
        )

        self.assertIn("frame", payload[1])
        self.assertIn("frame", payload[2])

    def test_sparse_history_transport_is_materially_smaller(self) -> None:
        frames = []
        for step in range(20):
            grid = tuple(
                tuple(1 if row * 16 + col == step else 0 for col in range(16))
                for row in range(16)
            )
            frames.append(Frame(grid=grid, step=step, level=1))
        entries = [
            HistoryEntry(action="RIGHT", frame=frame) for frame in frames
        ]
        compact = _ascii_history_view_payload(entries)
        full = [
            {
                "action": entry.action,
                "frame": {
                    "ascii": entry.frame.ascii,
                    "step": entry.frame.step,
                    "level": entry.frame.level,
                    "shape": list(entry.frame.shape),
                    "grid": [list(row) for row in entry.frame.grid],
                },
            }
            for entry in entries
        ]

        compact_size = len(json.dumps(compact, separators=(",", ":")))
        full_size = len(json.dumps(full, separators=(",", ":")))
        self.assertLess(compact_size, full_size // 4)

    def test_current_frame_reuses_matching_history_frame(self) -> None:
        history_frame = Frame(grid=((0, 1),), step=1, level=2)
        current_frame = Frame(grid=((0, 1),), step=1, level=2)
        history_entries = [
            HistoryEntry(action="RIGHT", frame=history_frame)
        ]
        current_payload = _current_frame_transport_payload(
            current_frame,
            history_entries,
            _ascii_frame_view_payload(current_frame),
        )

        self.assertEqual(current_payload, {"history_index": 0})
        response = run_sandboxed_python(
            code=(
                "result = [current_frame.ascii, "
                "current_frame is history[-1].frame]"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": current_payload,
                "history": _ascii_history_view_payload(history_entries),
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "memory": {},
            },
            action_handler=lambda actions: {"action_result": {}, "state": {}},
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], ["Ww", True])

    def test_current_frame_falls_back_when_history_is_stale(self) -> None:
        old_frame = Frame(grid=((0,),), step=0, level=1)
        current_frame = Frame(grid=((1,),), step=1, level=1)
        full_payload = _ascii_frame_view_payload(current_frame)

        selected = _current_frame_transport_payload(
            current_frame,
            [HistoryEntry(action="", frame=old_frame)],
            full_payload,
        )

        self.assertIs(selected, full_payload)

    def test_delta_threshold_uses_full_frame_at_boundary(self) -> None:
        first = Frame(grid=((0, 0, 0), (0, 0, 0)), step=0, level=1)
        second = Frame(grid=((1, 0, 0), (0, 0, 0)), step=1, level=1)

        payload = _ascii_history_view_payload(
            [
                HistoryEntry(action="", frame=first),
                HistoryEntry(action="RIGHT", frame=second),
            ]
        )

        self.assertIn("frame", payload[1])
        self.assertNotIn("frame_delta", payload[1])

    def test_delta_chain_reconstructs_after_dense_keyframe(self) -> None:
        empty = Frame(
            grid=tuple((0, 0, 0, 0) for _row in range(4)),
            step=0,
            level=1,
        )
        dense = Frame(
            grid=tuple((1, 1, 1, 1) for _row in range(4)),
            step=1,
            level=1,
        )
        sparse = Frame(
            grid=((2, 1, 1, 1), *(dense.grid[1:])),
            step=2,
            level=1,
        )
        payload = _ascii_history_view_payload(
            [
                HistoryEntry(action="", frame=empty),
                HistoryEntry(action="A", frame=dense),
                HistoryEntry(action="B", frame=sparse),
            ]
        )
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS

        history = namespace["_history_from_payload"](payload)

        self.assertIn("frame", payload[1])
        self.assertIn("frame_delta", payload[2])
        self.assertEqual(history[2].frame.cell(0, 0), ARC_COLOR_CHARS[2])
        self.assertEqual(history[2].frame.step, 2)

    def test_malformed_delta_value_is_skipped(self) -> None:
        frame = Frame(grid=((0,),), step=0, level=1)
        payload = _ascii_history_view_payload([HistoryEntry(action="", frame=frame)])
        payload.append(
            {
                "action": "RIGHT",
                "frame_delta": {
                    "step": 1,
                    "level": 1,
                    "shape": [1, 1],
                    "changes": [[0, 0, "invalid"]],
                },
            }
        )
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS

        history = namespace["_history_from_payload"](payload)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].frame.cell(0, 0), ARC_COLOR_CHARS[0])

    def test_invalid_history_reference_does_not_create_empty_frame(self) -> None:
        frame = Frame(grid=((0,),), step=0, level=1)
        response = run_sandboxed_python(
            code="result = current_frame is None",
            timeout_seconds=5,
            initial_state={
                "current_frame": {"history_index": True},
                "history": _ascii_history_view_payload(
                    [HistoryEntry(action="", frame=frame)]
                ),
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "memory": {},
            },
            action_handler=lambda actions: {"action_result": {}, "state": {}},
        )

        self.assertEqual(response["error"], "")
        self.assertTrue(response["result"])


if __name__ == "__main__":
    unittest.main()
