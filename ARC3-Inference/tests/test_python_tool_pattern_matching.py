from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_find_pattern,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class PythonToolPatternMatchingTests(unittest.TestCase):
    def test_finds_exact_pattern_in_reading_order(self) -> None:
        summary = _bounded_frame_find_pattern(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            shape=(3, 3),
            color_chars=ARC_COLOR_CHARS,
            pattern=[
                ARC_COLOR_CHARS[0] + ARC_COLOR_CHARS[1],
                ARC_COLOR_CHARS[1] + ARC_COLOR_CHARS[0],
            ],
        )

        self.assertEqual(summary["count"], 2)
        self.assertEqual(
            summary["matches"][0],
            {
                "top": 0,
                "left": 0,
                "bottom": 2,
                "right": 2,
                "transform": "IDENTITY",
            },
        )

    def test_wildcard_matches_any_color(self) -> None:
        summary = _bounded_frame_find_pattern(
            [[0, 1], [2, 3]],
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            pattern=[ARC_COLOR_CHARS[0] + "?", "?" + ARC_COLOR_CHARS[3]],
            wildcard="?",
        )

        self.assertEqual(summary["count"], 1)

    def test_transforms_find_rotated_non_square_pattern(self) -> None:
        pattern = [ARC_COLOR_CHARS[1] + ARC_COLOR_CHARS[2] + ARC_COLOR_CHARS[3]]
        grid = [[1], [2], [3]]

        exact = _bounded_frame_find_pattern(
            grid,
            shape=(3, 1),
            color_chars=ARC_COLOR_CHARS,
            pattern=pattern,
        )
        transformed = _bounded_frame_find_pattern(
            grid,
            shape=(3, 1),
            color_chars=ARC_COLOR_CHARS,
            pattern=pattern,
            transforms=True,
        )

        self.assertEqual(exact["count"], 0)
        self.assertEqual(transformed["count"], 1)
        self.assertEqual(transformed["matches"][0]["transform"], "ROTATE_90")

    def test_deduplicates_symmetric_variants_and_bounds_matches(self) -> None:
        summary = _bounded_frame_find_pattern(
            [[0, 0, 0]],
            shape=(1, 3),
            color_chars=ARC_COLOR_CHARS,
            pattern=[ARC_COLOR_CHARS[0]],
            transforms=True,
            limit=1,
        )

        self.assertEqual(summary["variants"], ["IDENTITY"])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(len(summary["matches"]), 1)
        self.assertEqual(summary["truncated_matches"], 2)

    def test_injected_frame_view_exposes_pattern_search(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=1,
            shape=(1, 2),
            grid=[[0, 1]],
        )

        summary = frame.find_pattern([ARC_COLOR_CHARS[:2]])

        self.assertEqual(summary["count"], 1)

    def test_rejects_invalid_patterns(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            _bounded_frame_find_pattern(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                pattern=[ARC_COLOR_CHARS[:2], ARC_COLOR_CHARS[:1]],
            )
        with self.assertRaisesRegex(ValueError, "non-color"):
            _bounded_frame_find_pattern(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                pattern=[ARC_COLOR_CHARS[0]],
                wildcard=ARC_COLOR_CHARS[1],
            )


if __name__ == "__main__":
    unittest.main()
