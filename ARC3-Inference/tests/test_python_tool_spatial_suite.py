from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_spatial_operation,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


GRID = [[0, 1, 2], [1, 0, 2], [3, 1, 0]]


def spatial(operation: str, **options):
    return _bounded_frame_spatial_operation(
        GRID,
        shape=(3, 3),
        color_chars=ARC_COLOR_CHARS,
        operation=operation,
        **options,
    )


class PythonToolSpatialSuiteTests(unittest.TestCase):
    def test_01_bounds_combines_multiple_symbols(self) -> None:
        result = spatial("bounds", symbols=ARC_COLOR_CHARS[2:4], limit=2)

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["bbox"], [0, 0, 2, 2])
        self.assertEqual(result["cells"], [[0, 2], [1, 2]])
        self.assertEqual(result["truncated_cells"], 1)

    def test_02_region_summary_reports_local_palette(self) -> None:
        result = spatial("region_summary", bounds=[0, 0, 2, 2])

        self.assertEqual(result["shape"], [2, 2])
        self.assertEqual(result["area"], 4)
        self.assertEqual(result["colors"][0]["count"], 2)
        self.assertEqual(result["colors"][1]["count"], 2)

    def test_03_row_profile_supports_symbol_filters(self) -> None:
        result = spatial("row_profile", symbol=ARC_COLOR_CHARS[1])

        self.assertEqual(
            [profile["matched"] for profile in result["profiles"]],
            [1, 1, 1],
        )

    def test_04_column_profile_reports_all_colors(self) -> None:
        result = spatial("column_profile")

        self.assertEqual(result["profiles"][2]["counts"], {ARC_COLOR_CHARS[0]: 1, ARC_COLOR_CHARS[2]: 2})

    def test_05_nearest_cell_ranks_targets_deterministically(self) -> None:
        result = spatial(
            "nearest_cell",
            start=[2, 2],
            symbols=ARC_COLOR_CHARS[1],
            metric="manhattan",
        )

        self.assertEqual(result["nearest"]["cell"], [2, 1])
        self.assertEqual(result["nearest"]["distance"], 1)

    def test_06_distance_supports_three_metrics(self) -> None:
        manhattan = spatial("distance", start=[0, 0], end=[2, 1])
        chebyshev = spatial(
            "distance", start=[0, 0], end=[2, 1], metric="chebyshev"
        )
        squared = spatial(
            "distance",
            start=[0, 0],
            end=[2, 1],
            metric="euclidean_squared",
        )

        self.assertEqual(manhattan["distance"], 3)
        self.assertEqual(chebyshev["distance"], 2)
        self.assertEqual(squared["distance"], 5)

    def test_07_line_between_returns_bresenham_cells_and_symbols(self) -> None:
        result = spatial("line_between", start=[0, 0], end=[2, 1])

        self.assertEqual(
            [item["cell"] for item in result["cells"]],
            [[0, 0], [1, 1], [2, 1]],
        )
        self.assertEqual(result["cells"][1]["symbol"], ARC_COLOR_CHARS[0])

    def test_08_translate_cells_reports_out_of_frame_targets(self) -> None:
        result = spatial(
            "translate_cells",
            cells=[[0, 0], [2, 2]],
            delta=[1, -1],
        )

        self.assertEqual(result["mapped"][0]["target"], [1, -1])
        self.assertFalse(result["mapped"][0]["in_bounds"])
        self.assertEqual(result["outside_frame"], 2)

    def test_09_mirror_cells_supports_reflections_and_rotation(self) -> None:
        vertical = spatial(
            "mirror_cells", cells=[[0, 0]], symmetry="vertical"
        )
        rotated = spatial(
            "mirror_cells", cells=[[0, 0]], symmetry="rotate_180"
        )

        self.assertEqual(vertical["mapped"][0]["target"], [0, 2])
        self.assertEqual(rotated["mapped"][0]["target"], [2, 2])

    def test_10_compare_regions_detects_geometric_relation(self) -> None:
        grid = [[0, 1, 9, 1, 0], [1, 0, 9, 0, 1]]
        result = _bounded_frame_spatial_operation(
            grid,
            shape=(2, 5),
            color_chars=ARC_COLOR_CHARS,
            operation="compare_regions",
            first=[0, 0, 2, 2],
            second=[0, 3, 2, 5],
        )

        self.assertTrue(result["exact_matches"])
        self.assertTrue(
            {"FLIP_HORIZONTAL", "FLIP_VERTICAL"}
            & set(result["exact_matches"])
        )

    def test_injected_frame_view_exposes_all_ten_operations(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=1,
            shape=(3, 3),
            grid=GRID,
        )

        names = (
            "bounds",
            "region_summary",
            "row_profile",
            "column_profile",
            "nearest_cell",
            "distance",
            "line_between",
            "translate_cells",
            "mirror_cells",
            "compare_regions",
        )
        self.assertTrue(all(callable(getattr(frame, name)) for name in names))
        results = [
            frame.bounds(ARC_COLOR_CHARS[0]),
            frame.region_summary([0, 0, 2, 2]),
            frame.row_profile(),
            frame.column_profile(),
            frame.nearest_cell([0, 0], ARC_COLOR_CHARS[1]),
            frame.distance([0, 0], [2, 2]),
            frame.line_between([0, 0], [2, 2]),
            frame.translate_cells([[0, 0]], [1, 1]),
            frame.mirror_cells([[0, 0]], "vertical"),
            frame.compare_regions([0, 0, 2, 2], [0, 0, 2, 2]),
        ]
        self.assertEqual(len(results), 10)
        self.assertTrue(all(isinstance(result, dict) for result in results))

    def test_rejects_invalid_metrics_bounds_symbols_and_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric"):
            spatial("distance", start=[0, 0], end=[1, 1], metric="taxicab-ish")
        with self.assertRaisesRegex(ValueError, "non-empty region"):
            spatial("region_summary", bounds=[0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "expects symbols"):
            spatial("bounds", symbols="?")
        with self.assertRaisesRegex(TypeError, "limit"):
            spatial("line_between", start=[0, 0], end=[1, 1], limit=True)


if __name__ == "__main__":
    unittest.main()
