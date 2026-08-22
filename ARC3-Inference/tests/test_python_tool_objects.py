from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_objects,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class PythonToolObjectTests(unittest.TestCase):
    def test_groups_adjacent_colors_into_one_object(self) -> None:
        summary = _bounded_frame_objects(
            [[0, 1, 2, 0], [0, 1, 0, 3]],
            shape=(2, 4),
            color_chars=ARC_COLOR_CHARS,
            background=ARC_COLOR_CHARS[0],
        )

        self.assertEqual(summary["count"], 2)
        first = summary["objects"][0]
        self.assertEqual(first["size"], 3)
        self.assertEqual(first["bbox"], [0, 1, 1, 2])
        self.assertEqual(
            first["colors"],
            {ARC_COLOR_CHARS[1]: 2, ARC_COLOR_CHARS[2]: 1},
        )
        self.assertEqual(
            first["pattern"],
            [ARC_COLOR_CHARS[1:3], ARC_COLOR_CHARS[1] + "."],
        )

    def test_infers_modal_background(self) -> None:
        summary = _bounded_frame_objects(
            [[0, 0, 1], [0, 2, 1]],
            shape=(2, 3),
            color_chars=ARC_COLOR_CHARS,
        )

        self.assertEqual(summary["background"], ARC_COLOR_CHARS[0])
        self.assertTrue(summary["background_inferred"])
        self.assertEqual(summary["count"], 1)

    def test_diagonal_connectivity_merges_objects(self) -> None:
        grid = [[1, 0], [0, 2]]
        orthogonal = _bounded_frame_objects(
            grid,
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            background=ARC_COLOR_CHARS[0],
        )
        diagonal = _bounded_frame_objects(
            grid,
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            background=ARC_COLOR_CHARS[0],
            diagonal=True,
        )

        self.assertEqual(orthogonal["count"], 2)
        self.assertEqual(diagonal["count"], 1)
        self.assertEqual(diagonal["objects"][0]["colors"], {
            ARC_COLOR_CHARS[1]: 1,
            ARC_COLOR_CHARS[2]: 1,
        })

    def test_bounds_returned_objects_and_cell_samples(self) -> None:
        summary = _bounded_frame_objects(
            [[1, 0, 2, 0, 3]],
            shape=(1, 5),
            color_chars=ARC_COLOR_CHARS,
            background=ARC_COLOR_CHARS[0],
            limit=2,
            cell_limit=1,
        )

        self.assertEqual(summary["count"], 3)
        self.assertEqual(len(summary["objects"]), 2)
        self.assertEqual(summary["sampled_cells"], 1)
        self.assertEqual(summary["truncated_objects"], 1)

    def test_injected_frame_view_exposes_objects(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=1,
            shape=(1, 3),
            grid=[[0, 1, 2]],
        )

        summary = frame.objects(background=ARC_COLOR_CHARS[0])

        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["objects"][0]["size"], 2)

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "color symbols"):
            _bounded_frame_objects(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                background="?",
            )
        with self.assertRaisesRegex(TypeError, "boolean"):
            _bounded_frame_objects(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                diagonal=1,
            )

    def test_multiple_background_colors_are_excluded(self) -> None:
        summary = _bounded_frame_objects(
            [[0, 1, 2, 1, 0]],
            shape=(1, 5),
            color_chars=ARC_COLOR_CHARS,
            background=ARC_COLOR_CHARS[:2],
        )

        self.assertEqual(summary["background"], ARC_COLOR_CHARS[:2])
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["objects"][0]["colors"], {ARC_COLOR_CHARS[2]: 1})

    def test_all_background_returns_no_objects(self) -> None:
        summary = _bounded_frame_objects(
            [[0, 0], [0, 0]],
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            background=ARC_COLOR_CHARS[0],
        )

        self.assertEqual(summary["count"], 0)
        self.assertEqual(summary["objects"], [])
        self.assertEqual(summary["sampled_cells"], 0)

    def test_large_object_omits_unbounded_pattern(self) -> None:
        grid = [[1] * 17 for _row in range(17)]
        summary = _bounded_frame_objects(
            grid,
            shape=(17, 17),
            color_chars=ARC_COLOR_CHARS,
            background=ARC_COLOR_CHARS[0],
        )

        self.assertEqual(summary["objects"][0]["size"], 289)
        self.assertIsNone(summary["objects"][0]["pattern"])
        self.assertTrue(summary["objects"][0]["truncated_pattern"])

    def test_zero_limits_preserve_counts_without_samples(self) -> None:
        summary = _bounded_frame_objects(
            [[1, 0, 2]],
            shape=(1, 3),
            color_chars=ARC_COLOR_CHARS,
            background=ARC_COLOR_CHARS[0],
            limit=0,
            cell_limit=0,
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["objects"], [])
        self.assertEqual(summary["sampled_cells"], 0)
        self.assertEqual(summary["truncated_objects"], 2)


if __name__ == "__main__":
    unittest.main()
