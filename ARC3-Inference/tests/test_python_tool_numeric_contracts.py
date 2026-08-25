from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_color_summary,
    _bounded_frame_components,
    _bounded_frame_crop,
    _bounded_frame_enclosed_regions,
    _bounded_frame_find,
    _bounded_frame_layout_operation,
    _bounded_frame_objects,
    _bounded_frame_rectangles,
    _bounded_frame_runs,
    _bounded_frame_spatial_operation,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


GRID = [[0, 1], [1, 0]]
SHAPE = (2, 2)


class PythonToolNumericContractTests(unittest.TestCase):
    def test_crop_rejects_noninteger_area_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "positive integer"):
            _bounded_frame_crop(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                top=0,
                left=0,
                bottom=1,
                right=1,
                max_area=float("inf"),
            )
        rendered = _bounded_frame_crop(
            [[float("inf")]],
            shape=(1, 1),
            color_chars=ARC_COLOR_CHARS,
            top=0,
            left=0,
            bottom=1,
            right=1,
        )
        self.assertEqual(rendered["rows"], ["?"])

    def test_find_rejects_nonfinite_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer limit"):
            _bounded_frame_find(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                symbol=ARC_COLOR_CHARS[0],
                limit=float("inf"),
            )

    def test_color_summary_rejects_numeric_string_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects an integer"):
            _bounded_frame_color_summary(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                limit="2",  # type: ignore[arg-type]
            )

    def test_spatial_operation_rejects_nonfinite_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects an integer"):
            _bounded_frame_spatial_operation(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                operation="bounds",
                symbols=ARC_COLOR_CHARS[0],
                limit=float("inf"),
            )

    def test_layout_operation_rejects_nonfinite_thickness(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects an integer"):
            _bounded_frame_layout_operation(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                operation="border_summary",
                thickness=float("inf"),
            )

    def test_runs_rejects_nonfinite_minimum_length(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects an integer"):
            _bounded_frame_runs(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                min_length=float("inf"),
            )

    def test_rectangles_reject_nonfinite_minimum_size(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects an integer"):
            _bounded_frame_rectangles(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                min_size=float("inf"),
            )

    def test_enclosed_regions_reject_nonfinite_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects an integer"):
            _bounded_frame_enclosed_regions(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                limit=float("inf"),
            )

    def test_components_reject_nonfinite_cell_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects an integer"):
            _bounded_frame_components(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                symbol=ARC_COLOR_CHARS[0],
                cell_limit=float("inf"),
            )

    def test_objects_reject_nonfinite_cell_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "expects an integer"):
            _bounded_frame_objects(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                cell_limit=float("inf"),
            )


if __name__ == "__main__":
    unittest.main()
