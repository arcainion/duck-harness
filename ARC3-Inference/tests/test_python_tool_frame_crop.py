from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import _SANDBOX_BOOTSTRAP, _bounded_frame_crop
from inference.utils.grid_utils import ARC_COLOR_CHARS


class BoundedFrameCropTests(unittest.TestCase):
    def test_injected_frame_view_exposes_crop(self) -> None:
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

        self.assertEqual(
            frame.crop(0, 0, 1, 2)["rows"], [ARC_COLOR_CHARS[0:2]]
        )

    def test_returns_letter_coded_half_open_region(self) -> None:
        crop = _bounded_frame_crop(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            shape=(3, 3),
            color_chars=ARC_COLOR_CHARS,
            top=0,
            left=1,
            bottom=2,
            right=3,
        )

        self.assertEqual(crop["bounds"], [0, 1, 2, 3])
        self.assertEqual(crop["shape"], [2, 2])
        self.assertEqual(
            crop["rows"], [ARC_COLOR_CHARS[1:3], ARC_COLOR_CHARS[4:6]]
        )
        self.assertEqual(crop["area"], 4)
        self.assertFalse(crop["clipped"])

    def test_clips_bounds_to_frame(self) -> None:
        crop = _bounded_frame_crop(
            [[0, 1], [2, 3]],
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            top=-4,
            left=-2,
            bottom=8,
            right=9,
        )

        self.assertEqual(crop["bounds"], [0, 0, 2, 2])
        self.assertEqual(
            crop["rows"], [ARC_COLOR_CHARS[0:2], ARC_COLOR_CHARS[2:4]]
        )
        self.assertTrue(crop["clipped"])

    def test_rejects_non_integer_coordinates(self) -> None:
        with self.assertRaisesRegex(TypeError, "coordinates must be integers"):
            _bounded_frame_crop(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                top=0.0,
                left=0,
                bottom=1,
                right=1,
            )

    def test_rejects_empty_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty region"):
            _bounded_frame_crop(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                top=0,
                left=0,
                bottom=0,
                right=1,
            )

    def test_rejects_more_than_256_cells(self) -> None:
        grid = [[0] * 17 for _ in range(17)]
        with self.assertRaisesRegex(ValueError, "limited to 256 cells"):
            _bounded_frame_crop(
                grid,
                shape=(17, 17),
                color_chars=ARC_COLOR_CHARS,
                top=0,
                left=0,
                bottom=17,
                right=17,
            )


if __name__ == "__main__":
    unittest.main()
