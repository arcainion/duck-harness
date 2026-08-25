from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_color_transitions,
    _bounded_frame_diff_summary,
    _bounded_frame_find_pattern,
    _bounded_frame_object_changes,
    _bounded_frame_object_relations,
    _bounded_frame_periodicity,
    _bounded_frame_ray,
    _bounded_frame_symmetry,
    _bounded_reachable_region,
    _bounded_shortest_path,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


GRID = [[0, 1], [1, 0]]
SHAPE = (2, 2)


class PythonToolRemainingNumericContractTests(unittest.TestCase):
    def test_ray_rejects_nonfinite_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer"):
            _bounded_frame_ray(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                row=0,
                col=0,
                direction="RIGHT",
                limit=float("inf"),
            )

    def test_object_relations_rejects_numeric_string_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "object_limit"):
            _bounded_frame_object_relations(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                object_limit="2",  # type: ignore[arg-type]
            )

    def test_object_changes_rejects_nonfinite_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer"):
            _bounded_frame_object_changes(
                GRID,
                GRID,
                before_shape=SHAPE,
                after_shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                limit=float("inf"),
            )

    def test_symmetry_rejects_nonfinite_sample_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer"):
            _bounded_frame_symmetry(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                sample_limit=float("inf"),
            )

    def test_periodicity_rejects_nonfinite_candidate_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "candidate_limit"):
            _bounded_frame_periodicity(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                candidate_limit=float("inf"),
            )

    def test_pattern_matching_rejects_nonfinite_mismatch_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "mismatch_limit"):
            _bounded_frame_find_pattern(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                pattern=[ARC_COLOR_CHARS[0]],
                mismatch_limit=float("inf"),
            )

    def test_reachable_region_rejects_nonfinite_node_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "max_nodes"):
            _bounded_reachable_region(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                start=[0, 0],
                passable=ARC_COLOR_CHARS[:2],
                max_nodes=float("inf"),
            )

    def test_shortest_path_rejects_nonfinite_path_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "path_limit"):
            _bounded_shortest_path(
                GRID,
                shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                start=[0, 0],
                goal=[1, 1],
                passable=ARC_COLOR_CHARS[:2],
                path_limit=float("inf"),
            )

    def test_color_transitions_bounds_invalid_cells_and_rejects_bad_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer"):
            _bounded_frame_color_transitions(
                GRID,
                GRID,
                before_shape=SHAPE,
                after_shape=SHAPE,
                color_chars=ARC_COLOR_CHARS,
                cell_limit=float("inf"),
            )
        result = _bounded_frame_color_transitions(
            [[float("inf")]],
            [[0]],
            before_shape=(1, 1),
            after_shape=(1, 1),
            color_chars=ARC_COLOR_CHARS,
        )
        self.assertEqual(result["transitions"][0]["before"], "?")

    def test_frame_diff_bounds_invalid_cells_and_rejects_bad_limit(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer limit"):
            _bounded_frame_diff_summary(
                GRID,
                GRID,
                before_shape=SHAPE,
                after_shape=SHAPE,
                before_level=0,
                after_level=0,
                color_chars=ARC_COLOR_CHARS,
                limit=float("inf"),
            )
        result = _bounded_frame_diff_summary(
            [[float("inf")]],
            [[0]],
            before_shape=(1, 1),
            after_shape=(1, 1),
            before_level=0,
            after_level=0,
            color_chars=ARC_COLOR_CHARS,
        )
        self.assertEqual(result["changes"][0]["before"], "?")


if __name__ == "__main__":
    unittest.main()
