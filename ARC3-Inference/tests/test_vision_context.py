from __future__ import annotations

from unittest import TestCase

from inference.agent.vision_context import ARC_COLOR_MAP, color_for_value


class VisionContextTests(TestCase):
    def test_unknown_grid_values_receive_stable_distinct_colors(self) -> None:
        first = color_for_value(42)

        self.assertEqual(first, color_for_value(42))
        self.assertNotEqual(first, ARC_COLOR_MAP[0])
        self.assertNotEqual(first, color_for_value(43))


if __name__ == "__main__":
    import unittest

    unittest.main()
