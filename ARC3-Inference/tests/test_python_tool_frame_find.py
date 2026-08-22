from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_find,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class BoundedFrameFindTests(unittest.TestCase):
    def test_returns_count_bbox_and_bounded_coordinate_sample(self) -> None:
        symbol = ARC_COLOR_CHARS[2]
        found = _bounded_frame_find(
            [[2, 0, 2], [0, 2, 0], [2, 0, 0]],
            shape=(3, 3),
            color_chars=ARC_COLOR_CHARS,
            symbol=symbol,
            limit=2,
        )

        self.assertEqual(found["symbol"], symbol)
        self.assertEqual(found["count"], 4)
        self.assertEqual(found["cells"], [[0, 0], [0, 2]])
        self.assertEqual(found["bbox"], [0, 0, 2, 2])
        self.assertEqual(found["truncated"], 2)

    def test_reports_absent_symbol(self) -> None:
        found = _bounded_frame_find(
            [[0]],
            shape=(1, 1),
            color_chars=ARC_COLOR_CHARS,
            symbol=ARC_COLOR_CHARS[1],
        )

        self.assertEqual(found["count"], 0)
        self.assertEqual(found["cells"], [])
        self.assertIsNone(found["bbox"])
        self.assertEqual(found["truncated"], 0)

    def test_caps_coordinate_sample_at_256(self) -> None:
        found = _bounded_frame_find(
            [[0] * 20 for _ in range(20)],
            shape=(20, 20),
            color_chars=ARC_COLOR_CHARS,
            symbol=ARC_COLOR_CHARS[0],
            limit=999,
        )

        self.assertEqual(found["count"], 400)
        self.assertEqual(len(found["cells"]), 256)
        self.assertEqual(found["truncated"], 144)

    def test_rejects_unknown_symbol_and_boolean_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "expects one of the color symbols"):
            _bounded_frame_find(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                symbol="?",
            )
        with self.assertRaisesRegex(TypeError, "expects an integer limit"):
            _bounded_frame_find(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                symbol=ARC_COLOR_CHARS[0],
                limit=True,
            )

    def test_injected_frame_view_exposes_find(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
        namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
        frame = namespace["FrameView"](
            ascii="",
            step=1,
            level=0,
            shape=(2, 2),
            grid=[[0, 1], [1, 0]],
        )

        self.assertEqual(
            frame.find(ARC_COLOR_CHARS[1])["cells"], [[0, 1], [1, 0]]
        )


if __name__ == "__main__":
    unittest.main()
