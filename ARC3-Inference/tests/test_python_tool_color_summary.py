from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_color_summary,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class PythonToolColorSummaryTests(unittest.TestCase):
    def test_ranks_colors_and_reports_bounds_and_edge_contacts(self) -> None:
        summary = _bounded_frame_color_summary(
            [[0, 0, 1], [0, 2, 1], [0, 1, 1]],
            shape=(3, 3),
            color_chars=ARC_COLOR_CHARS,
        )

        self.assertEqual(summary["color_count"], 3)
        self.assertEqual(
            [item["symbol"] for item in summary["colors"]],
            [ARC_COLOR_CHARS[0], ARC_COLOR_CHARS[1], ARC_COLOR_CHARS[2]],
        )
        dominant = summary["colors"][0]
        self.assertEqual(dominant["count"], 4)
        self.assertAlmostEqual(dominant["fraction"], 4 / 9)
        self.assertEqual(dominant["bbox"], [0, 0, 2, 1])
        self.assertEqual(dominant["touches_edges"], ["top", "bottom", "left"])
        self.assertEqual(dominant["edge_cells"], 4)
        self.assertEqual(
            dominant["edge_counts"],
            {"top": 2, "right": 0, "bottom": 1, "left": 3},
        )

    def test_limit_reports_omitted_palette_mass(self) -> None:
        summary = _bounded_frame_color_summary(
            [[0, 0, 1], [0, 2, 1], [0, 1, 1]],
            shape=(3, 3),
            color_chars=ARC_COLOR_CHARS,
            limit=1,
        )

        self.assertEqual(len(summary["colors"]), 1)
        self.assertEqual(summary["omitted_colors"], 2)
        self.assertEqual(summary["omitted_cells"], 5)

    def test_counts_ragged_missing_and_invalid_cells_as_unknown(self) -> None:
        summary = _bounded_frame_color_summary(
            [[0, True], [1], ["bad", 16]],
            shape=(3, 2),
            color_chars=ARC_COLOR_CHARS,
        )

        self.assertEqual(summary["total_cells"], 6)
        self.assertEqual(summary["observed_cells"], 2)
        self.assertEqual(summary["unknown_cells"], 4)
        self.assertEqual(sum(item["count"] for item in summary["colors"]), 2)

    def test_rejects_boolean_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "limit"):
            _bounded_frame_color_summary(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                limit=True,
            )

    def test_injected_frame_view_exposes_color_summary(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=1,
            shape=(1, 3),
            grid=[[0, 1, 1]],
        )

        summary = frame.color_summary(1)

        self.assertEqual(summary["colors"][0]["symbol"], ARC_COLOR_CHARS[1])
        self.assertEqual(summary["colors"][0]["count"], 2)
        self.assertEqual(len(summary["colors"]), 1)


if __name__ == "__main__":
    unittest.main()
