from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_components,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class PythonToolComponentTests(unittest.TestCase):
    def test_summarizes_component_geometry(self) -> None:
        summary = _bounded_frame_components(
            [[1, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0]],
            shape=(3, 4),
            color_chars=ARC_COLOR_CHARS,
            symbol=ARC_COLOR_CHARS[1],
        )

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["connectivity"], 4)
        first = summary["components"][0]
        self.assertEqual(first["size"], 3)
        self.assertEqual(first["bbox"], [0, 0, 1, 1])
        self.assertEqual(first["centroid"], [1 / 3, 1 / 3])
        self.assertTrue(first["touches_edge"])

    def test_diagonal_connectivity_merges_corner_touching_cells(self) -> None:
        grid = [[1, 0], [0, 1]]

        orthogonal = _bounded_frame_components(
            grid,
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            symbol=ARC_COLOR_CHARS[1],
        )
        diagonal = _bounded_frame_components(
            grid,
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            symbol=ARC_COLOR_CHARS[1],
            diagonal=True,
        )

        self.assertEqual(orthogonal["count"], 2)
        self.assertEqual(diagonal["count"], 1)
        self.assertEqual(diagonal["components"][0]["size"], 2)

    def test_bounds_components_and_cell_samples(self) -> None:
        grid = [[1 if column % 2 == 0 else 0 for column in range(10)]]
        summary = _bounded_frame_components(
            grid,
            shape=(1, 10),
            color_chars=ARC_COLOR_CHARS,
            symbol=ARC_COLOR_CHARS[1],
            limit=2,
            cell_limit=1,
        )

        self.assertEqual(summary["count"], 5)
        self.assertEqual(len(summary["components"]), 2)
        self.assertEqual(summary["sampled_cells"], 1)
        self.assertEqual(summary["truncated_components"], 3)

        omitted_cells = _bounded_frame_components(
            grid,
            shape=(1, 10),
            color_chars=ARC_COLOR_CHARS,
            symbol=ARC_COLOR_CHARS[1],
            limit=2,
            cell_limit=10,
        )
        returned_cells = sum(
            len(component["cells"])
            for component in omitted_cells["components"]
        )
        self.assertEqual(omitted_cells["sampled_cells"], returned_cells)

    def test_injected_frame_view_exposes_components(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=1,
            shape=(2, 2),
            grid=[[1, 0], [0, 1]],
        )

        summary = frame.components(ARC_COLOR_CHARS[1], diagonal=True)

        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["components"][0]["bbox"], [0, 0, 1, 1])

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "expects one of"):
            _bounded_frame_components(
                [[1]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                symbol="?",
            )
        with self.assertRaisesRegex(TypeError, "boolean"):
            _bounded_frame_components(
                [[1]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                symbol=ARC_COLOR_CHARS[1],
                diagonal=1,
            )


if __name__ == "__main__":
    unittest.main()
