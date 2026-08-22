from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_shortest_path,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class BoundedShortestPathTests(unittest.TestCase):
    def test_finds_cardinal_path_over_selected_color(self) -> None:
        path = _bounded_shortest_path(
            [[1, 1, 2], [0, 1, 0], [0, 1, 1]],
            shape=(3, 3),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[0, 2],
            passable=ARC_COLOR_CHARS[1],
        )

        self.assertTrue(path["found"])
        self.assertEqual(path["distance"], 2)
        self.assertEqual(path["path"], [[0, 0], [0, 1], [0, 2]])
        self.assertEqual(path["moves"], ["RIGHT", "RIGHT"])
        self.assertEqual(path["next_step"], [0, 1])
        self.assertFalse(path["search_truncated"])
        self.assertFalse(path["path_truncated"])

    def test_goal_is_enterable_even_when_not_passable(self) -> None:
        path = _bounded_shortest_path(
            [[1, 2]],
            shape=(1, 2),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[0, 1],
            passable=ARC_COLOR_CHARS[1],
        )

        self.assertTrue(path["found"])
        self.assertEqual(path["distance"], 1)

    def test_selects_nearest_of_multiple_goals_in_one_search(self) -> None:
        path = _bounded_shortest_path(
            [[1, 1, 2, 1, 2]],
            shape=(1, 5),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[[0, 4], [0, 2]],
            passable=ARC_COLOR_CHARS[1],
        )

        self.assertTrue(path["found"])
        self.assertEqual(path["goals"], [[0, 4], [0, 2]])
        self.assertEqual(path["selected_goal"], [0, 2])
        self.assertEqual(path["goal"], [0, 2])
        self.assertEqual(path["distance"], 2)

    def test_selects_nearest_cell_matching_target_symbols(self) -> None:
        path = _bounded_shortest_path(
            [[1, 1, 2, 1, 3]],
            shape=(1, 5),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=ARC_COLOR_CHARS[2] + ARC_COLOR_CHARS[3],
            passable=ARC_COLOR_CHARS[1],
        )

        self.assertTrue(path["found"])
        self.assertEqual(
            path["target_symbols"],
            [ARC_COLOR_CHARS[2], ARC_COLOR_CHARS[3]],
        )
        self.assertEqual(path["goals"], [])
        self.assertEqual(path["selected_goal"], [0, 2])
        self.assertEqual(path["distance"], 2)
        self.assertEqual(path["moves"], ["RIGHT", "RIGHT"])

    def test_symbol_target_is_enterable_and_reports_absence(self) -> None:
        reachable = _bounded_shortest_path(
            [[1, 2]],
            shape=(1, 2),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=ARC_COLOR_CHARS[2],
            passable=ARC_COLOR_CHARS[1],
        )
        absent = _bounded_shortest_path(
            [[1, 1]],
            shape=(1, 2),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=ARC_COLOR_CHARS[2],
            passable=ARC_COLOR_CHARS[1],
        )

        self.assertEqual(reachable["distance"], 1)
        self.assertFalse(absent["found"])
        self.assertIsNone(absent["selected_goal"])

    def test_rejects_empty_or_excessive_goal_lists(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one goal"):
            _bounded_shortest_path(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                start=[0, 0],
                goal=[],
                passable=ARC_COLOR_CHARS[0],
            )
        with self.assertRaisesRegex(ValueError, "limited to 64 goals"):
            _bounded_shortest_path(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                start=[0, 0],
                goal=[[0, 0]] * 65,
                passable=ARC_COLOR_CHARS[0],
            )

    def test_reports_unreachable_and_search_truncation(self) -> None:
        unreachable = _bounded_shortest_path(
            [[1, 0, 2]],
            shape=(1, 3),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[0, 2],
            passable=ARC_COLOR_CHARS[1],
        )
        truncated = _bounded_shortest_path(
            [[1, 1, 1]],
            shape=(1, 3),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[0, 2],
            passable=ARC_COLOR_CHARS[1],
            max_nodes=1,
        )

        self.assertFalse(unreachable["found"])
        self.assertFalse(unreachable["search_truncated"])
        self.assertFalse(truncated["found"])
        self.assertTrue(truncated["search_truncated"])

    def test_bounds_returned_path_but_preserves_distance(self) -> None:
        path = _bounded_shortest_path(
            [[1, 1, 1, 1, 2]],
            shape=(1, 5),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[0, 4],
            passable=ARC_COLOR_CHARS[1],
            path_limit=2,
        )

        self.assertEqual(path["distance"], 4)
        self.assertEqual(path["path"], [[0, 0], [0, 1], [0, 2]])
        self.assertEqual(path["moves"], ["RIGHT", "RIGHT"])
        self.assertTrue(path["path_truncated"])

    def test_rejects_invalid_coordinates_and_symbols(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the frame"):
            _bounded_shortest_path(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                start=[-1, 0],
                goal=[0, 0],
                passable=ARC_COLOR_CHARS[0],
            )
        with self.assertRaisesRegex(ValueError, "expects symbols"):
            _bounded_shortest_path(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                start=[0, 0],
                goal=[0, 0],
                passable="?",
            )
        with self.assertRaisesRegex(ValueError, "target symbols"):
            _bounded_shortest_path(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                start=[0, 0],
                goal="?",
                passable=ARC_COLOR_CHARS[0],
            )

    def test_injected_frame_view_exposes_shortest_path(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=0,
            shape=(1, 3),
            grid=[[1, 1, 2]],
        )

        path = frame.shortest_path(
            [0, 0], [0, 2], passable=ARC_COLOR_CHARS[1]
        )
        self.assertEqual(path["moves"], ["RIGHT", "RIGHT"])

        nearest = frame.shortest_path_to_any(
            [0, 0], [[0, 2], [0, 1]], passable=ARC_COLOR_CHARS[1]
        )
        self.assertEqual(nearest["selected_goal"], [0, 1])

        nearest_symbol = frame.shortest_path(
            [0, 0], ARC_COLOR_CHARS[2], passable=ARC_COLOR_CHARS[1]
        )
        self.assertEqual(nearest_symbol["selected_goal"], [0, 2])

    def test_start_equal_to_goal_returns_zero_length_path(self) -> None:
        path = _bounded_shortest_path(
            [[1]],
            shape=(1, 1),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[0, 0],
            passable=ARC_COLOR_CHARS[1],
        )

        self.assertTrue(path["found"])
        self.assertEqual(path["distance"], 0)
        self.assertEqual(path["path"], [[0, 0]])
        self.assertEqual(path["moves"], [])
        self.assertIsNone(path["next_step"])

    def test_diagonal_path_reports_diagonal_move(self) -> None:
        path = _bounded_shortest_path(
            [[1, 0], [0, 2]],
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[1, 1],
            passable=ARC_COLOR_CHARS[1],
            diagonal=True,
        )

        self.assertEqual(path["distance"], 1)
        self.assertEqual(path["moves"], ["DOWN_RIGHT"])
        self.assertEqual(path["next_step"], [1, 1])

    def test_duplicate_goals_are_deduplicated_stably(self) -> None:
        path = _bounded_shortest_path(
            [[1, 1, 2]],
            shape=(1, 3),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[[0, 2], [0, 2], [0, 1]],
            passable=ARC_COLOR_CHARS[1],
        )

        self.assertEqual(path["goals"], [[0, 2], [0, 1]])
        self.assertEqual(path["selected_goal"], [0, 1])

    def test_zero_path_limit_preserves_distance_and_next_step(self) -> None:
        path = _bounded_shortest_path(
            [[1, 1, 2]],
            shape=(1, 3),
            color_chars=ARC_COLOR_CHARS,
            start=[0, 0],
            goal=[0, 2],
            passable=ARC_COLOR_CHARS[1],
            path_limit=0,
        )

        self.assertEqual(path["distance"], 2)
        self.assertEqual(path["path"], [[0, 0]])
        self.assertEqual(path["moves"], [])
        self.assertEqual(path["next_step"], [0, 1])
        self.assertTrue(path["path_truncated"])

    def test_rejects_boolean_coordinates_and_limits(self) -> None:
        with self.assertRaisesRegex(TypeError, "integer coordinates"):
            _bounded_shortest_path(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                start=[True, 0],
                goal=[0, 0],
                passable=ARC_COLOR_CHARS[0],
            )
        with self.assertRaisesRegex(TypeError, "max_nodes"):
            _bounded_shortest_path(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                start=[0, 0],
                goal=[0, 0],
                passable=ARC_COLOR_CHARS[0],
                max_nodes=True,
            )


if __name__ == "__main__":
    unittest.main()
