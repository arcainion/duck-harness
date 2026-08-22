from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _bounded_frame_ray,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


class PythonToolRayTests(unittest.TestCase):
    def test_scans_to_edge_with_stable_distances(self) -> None:
        summary = _bounded_frame_ray(
            [[0, 1, 2, 3]],
            shape=(1, 4),
            color_chars=ARC_COLOR_CHARS,
            row=0,
            col=0,
            direction="RIGHT",
        )

        self.assertEqual(summary["length"], 3)
        self.assertEqual(
            [(cell["symbol"], cell["distance"]) for cell in summary["cells"]],
            [
                (ARC_COLOR_CHARS[1], 1),
                (ARC_COLOR_CHARS[2], 2),
                (ARC_COLOR_CHARS[3], 3),
            ],
        )
        self.assertIsNone(summary["hit"])
        self.assertTrue(summary["reached_edge"])

    def test_stop_color_is_included_as_hit(self) -> None:
        summary = _bounded_frame_ray(
            [[0, 1, 2, 3]],
            shape=(1, 4),
            color_chars=ARC_COLOR_CHARS,
            row=0,
            col=0,
            direction="right",
            stop_at=ARC_COLOR_CHARS[2],
        )

        self.assertEqual(summary["length"], 2)
        self.assertEqual(summary["hit"], summary["cells"][-1])
        self.assertEqual(summary["hit"]["symbol"], ARC_COLOR_CHARS[2])
        self.assertFalse(summary["reached_edge"])

    def test_supports_diagonal_direction_and_hyphen_alias(self) -> None:
        summary = _bounded_frame_ray(
            [[0, 0, 0], [0, 1, 0], [0, 0, 2]],
            shape=(3, 3),
            color_chars=ARC_COLOR_CHARS,
            row=0,
            col=0,
            direction="down-right",
        )

        self.assertEqual(summary["direction"], "DOWN_RIGHT")
        self.assertEqual(summary["delta"], [1, 1])
        self.assertEqual(
            [cell["symbol"] for cell in summary["cells"]],
            [ARC_COLOR_CHARS[1], ARC_COLOR_CHARS[2]],
        )

    def test_include_start_and_zero_limit_preserve_metadata(self) -> None:
        included = _bounded_frame_ray(
            [[0, 1]],
            shape=(1, 2),
            color_chars=ARC_COLOR_CHARS,
            row=0,
            col=0,
            direction="RIGHT",
            include_start=True,
        )
        bounded = _bounded_frame_ray(
            [[0, 1]],
            shape=(1, 2),
            color_chars=ARC_COLOR_CHARS,
            row=0,
            col=0,
            direction="RIGHT",
            stop_at=ARC_COLOR_CHARS[1],
            limit=0,
        )

        self.assertEqual(included["cells"][0]["distance"], 0)
        self.assertEqual(included["cells"][0]["symbol"], ARC_COLOR_CHARS[0])
        self.assertEqual(bounded["cells"], [])
        self.assertEqual(bounded["hit"]["distance"], 1)
        self.assertEqual(bounded["truncated_cells"], 1)

    def test_ragged_cells_are_reported_as_unknown(self) -> None:
        summary = _bounded_frame_ray(
            [[0], []],
            shape=(2, 2),
            color_chars=ARC_COLOR_CHARS,
            row=0,
            col=0,
            direction="DOWN_RIGHT",
        )

        self.assertEqual(summary["cells"][0]["symbol"], "?")
        self.assertTrue(summary["reached_edge"])

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the frame"):
            _bounded_frame_ray(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                row=-1,
                col=0,
                direction="RIGHT",
            )
        with self.assertRaisesRegex(ValueError, "expects one of"):
            _bounded_frame_ray(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                row=0,
                col=0,
                direction="FORWARD",
            )
        with self.assertRaisesRegex(ValueError, "expects symbols"):
            _bounded_frame_ray(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                row=0,
                col=0,
                direction="RIGHT",
                stop_at="?",
            )
        with self.assertRaisesRegex(TypeError, "include_start"):
            _bounded_frame_ray(
                [[0]],
                shape=(1, 1),
                color_chars=ARC_COLOR_CHARS,
                row=0,
                col=0,
                direction="RIGHT",
                include_start=1,
            )

    def test_injected_frame_view_exposes_ray(self) -> None:
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

        summary = frame.ray(0, 0, "RIGHT", stop_at=ARC_COLOR_CHARS[2])

        self.assertEqual(summary["hit"]["col"], 2)


if __name__ == "__main__":
    unittest.main()
