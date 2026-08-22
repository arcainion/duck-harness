from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_neighbors,
    _frame_cell_symbol,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class PythonToolSpatialQueryTests(unittest.TestCase):
    def test_cell_returns_letter_symbol(self) -> None:
        symbol = _frame_cell_symbol(
            [[0, 1], [2, 3]],
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            row=1,
            col=0,
        )

        self.assertEqual(symbol, ARC_COLOR_CHARS[2])

    def test_neighbors_return_cardinal_directions_in_stable_order(self) -> None:
        result = _bounded_frame_neighbors(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            shape=(3, 3),
            color_chars=ARC_COLOR_CHARS,
            row=1,
            col=1,
        )

        self.assertEqual(result["center"]["symbol"], ARC_COLOR_CHARS[4])
        self.assertEqual(
            [entry["direction"] for entry in result["neighbors"]],
            ["UP", "RIGHT", "DOWN", "LEFT"],
        )
        self.assertEqual(
            [entry["symbol"] for entry in result["neighbors"]],
            [
                ARC_COLOR_CHARS[1],
                ARC_COLOR_CHARS[5],
                ARC_COLOR_CHARS[7],
                ARC_COLOR_CHARS[3],
            ],
        )

    def test_neighbors_clip_at_edges_and_optionally_include_diagonals(self) -> None:
        cardinal = _bounded_frame_neighbors(
            [[0, 1], [2, 3]],
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            row=0,
            col=0,
        )
        diagonal = _bounded_frame_neighbors(
            [[0, 1], [2, 3]],
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            row=0,
            col=0,
            diagonal=True,
        )

        self.assertEqual(len(cardinal["neighbors"]), 2)
        self.assertEqual(len(diagonal["neighbors"]), 3)
        self.assertEqual(diagonal["neighbors"][-1]["direction"], "DOWN_RIGHT")

    def test_rejects_invalid_coordinates_and_diagonal_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the frame"):
            _frame_cell_symbol(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                row=1,
                col=0,
            )
        with self.assertRaisesRegex(TypeError, "expects a boolean"):
            _bounded_frame_neighbors(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                row=0,
                col=0,
                diagonal=1,
            )

    def test_injected_frame_view_exposes_cell_and_neighbors(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=0,
            shape=(2, 2),
            grid=[[0, 1], [2, 3]],
        )

        self.assertEqual(frame.cell(0, 1), ARC_COLOR_CHARS[1])
        self.assertEqual(
            [item["direction"] for item in frame.neighbors(0, 0)["neighbors"]],
            ["RIGHT", "DOWN"],
        )


if __name__ == "__main__":
    unittest.main()
