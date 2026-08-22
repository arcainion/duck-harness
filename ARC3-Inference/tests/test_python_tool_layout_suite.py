from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_layout_operation,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


GRID = [
    [0, 0, 9, 1, 1],
    [0, 2, 9, 2, 1],
    [9, 9, 9, 9, 9],
    [3, 3, 9, 4, 4],
    [3, 0, 9, 0, 4],
]


def layout(operation: str, **options):
    return _bounded_frame_layout_operation(
        GRID,
        shape=(5, 5),
        color_chars=ARC_COLOR_CHARS,
        operation=operation,
        **options,
    )


class PythonToolLayoutSuiteTests(unittest.TestCase):
    def test_01_border_summary_reports_each_side(self) -> None:
        result = layout("border_summary", thickness=1)

        self.assertEqual(set(result["sides"]), {"top", "right", "bottom", "left"})
        self.assertEqual(result["sides"]["top"]["area"], 5)
        self.assertTrue(result["corner_cells_counted_on_two_sides"])

    def test_02_corner_summary_reports_bounded_corner_palettes(self) -> None:
        result = layout("corner_summary", size=2)

        self.assertEqual(result["corners"]["top_left"]["bounds"], [0, 0, 2, 2])
        self.assertEqual(result["corners"]["bottom_right"]["area"], 4)

    def test_03_center_summary_handles_radius(self) -> None:
        center = layout("center_summary")
        expanded = layout("center_summary", radius=1)

        self.assertEqual(center["bounds"], [2, 2, 3, 3])
        self.assertEqual(center["colors"][0]["symbol"], ARC_COLOR_CHARS[9])
        self.assertEqual(expanded["shape"], [3, 3])

    def test_04_quadrant_summary_partitions_the_frame(self) -> None:
        result = layout("quadrant_summary")

        self.assertEqual(result["split"], [3, 3])
        areas = [item["area"] for item in result["quadrants"].values()]
        self.assertEqual(sum(areas), 25)

    def test_05_color_adjacency_counts_cross_color_contacts(self) -> None:
        result = layout("color_adjacency")

        pair = next(
            item
            for item in result["pairs"]
            if item["colors"] == [ARC_COLOR_CHARS[0], ARC_COLOR_CHARS[2]]
        )
        self.assertEqual(pair["contacts"], 2)
        self.assertTrue(pair["samples"])

    def test_06_distance_between_colors_selects_nearest_pair(self) -> None:
        result = layout(
            "distance_between_colors",
            first=ARC_COLOR_CHARS[0],
            second=ARC_COLOR_CHARS[4],
        )

        self.assertEqual(result["distance"], 1)
        self.assertEqual(result["cells"], [[4, 3], [3, 3]])

    def test_07_divider_lines_detects_uniform_rows_and_columns(self) -> None:
        result = layout("divider_lines", symbol=ARC_COLOR_CHARS[9])

        self.assertEqual(result["rows"], [{"index": 2, "symbol": ARC_COLOR_CHARS[9]}])
        self.assertEqual(result["columns"], [{"index": 2, "symbol": ARC_COLOR_CHARS[9]}])

    def test_08_panels_extracts_regions_between_dividers(self) -> None:
        result = layout("panels", symbol=ARC_COLOR_CHARS[9])

        self.assertEqual(result["count"], 4)
        self.assertEqual(
            [item["bounds"] for item in result["panels"]],
            [[0, 0, 2, 2], [0, 3, 2, 5], [3, 0, 5, 2], [3, 3, 5, 5]],
        )

    def test_09_tile_summary_includes_partial_edge_tiles(self) -> None:
        result = layout("tile_summary", tile_height=2, tile_width=2)

        self.assertEqual(result["grid_shape"], [3, 3])
        self.assertEqual(result["count"], 9)
        self.assertFalse(result["tiles"][-1]["complete"])

    def test_10_edge_distance_reports_distribution(self) -> None:
        result = layout("edge_distance", symbols=ARC_COLOR_CHARS[2])

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["minimum"], 1)
        self.assertEqual(result["maximum"], 1)
        self.assertEqual(result["histogram"], {"1": 2})

    def test_injected_frame_view_executes_all_ten_operations(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=1,
            shape=(5, 5),
            grid=GRID,
        )

        results = [
            frame.border_summary(),
            frame.corner_summary(),
            frame.center_summary(),
            frame.quadrant_summary(),
            frame.color_adjacency(),
            frame.distance_between_colors(ARC_COLOR_CHARS[0], ARC_COLOR_CHARS[4]),
            frame.divider_lines(ARC_COLOR_CHARS[9]),
            frame.panels(ARC_COLOR_CHARS[9]),
            frame.tile_summary(2, 2),
            frame.edge_distance(ARC_COLOR_CHARS[2]),
        ]

        self.assertEqual(len(results), 10)
        self.assertTrue(all(isinstance(result, dict) for result in results))

    def test_rejects_invalid_options(self) -> None:
        cases = [
            ("border_summary", {"thickness": True}, TypeError),
            ("corner_summary", {"size": "bad"}, TypeError),
            ("color_adjacency", {"diagonal": 1}, TypeError),
            ("distance_between_colors", {"first": "?", "second": ARC_COLOR_CHARS[0]}, ValueError),
            ("tile_summary", {"tile_height": True, "tile_width": 1}, TypeError),
            ("edge_distance", {"symbols": "?"}, ValueError),
        ]
        for operation, options, exception in cases:
            with self.subTest(operation=operation), self.assertRaises(exception):
                layout(operation, **options)


if __name__ == "__main__":
    unittest.main()
