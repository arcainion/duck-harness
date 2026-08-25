from __future__ import annotations

import base64
import io
import os
from unittest import TestCase
from unittest.mock import patch

from PIL import Image

from inference.agent import vision_context
from inference.agent.runtime_state import Frame
from inference.agent.vision_context import (
    ARC_COLOR_MAP,
    color_for_value,
    current_grid_image_enabled,
    current_grid_image_part,
    current_grid_image_upscale,
    frame_to_png_data_url,
)


def _decoded_image(data_url: str) -> Image.Image:
    encoded = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


class VisionContextTests(TestCase):
    def test_current_grid_mode_accepts_hyphen_and_space_aliases(self) -> None:
        for value in ("current-grid", "current grid"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"MULTIMODAL_CONTEXT": value}
            ):
                self.assertTrue(current_grid_image_enabled())

    def test_environment_upscale_is_capped(self) -> None:
        with patch.dict(os.environ, {"MULTIMODAL_UPSCALE": "1000000"}):
            self.assertEqual(current_grid_image_upscale(), 64)

    def test_explicit_upscale_rejects_bool_and_non_integer(self) -> None:
        frame = Frame(grid=((1,),), step=0, level=1)

        with self.assertRaises(TypeError):
            frame_to_png_data_url(frame, upscale=True)
        with self.assertRaises(TypeError):
            frame_to_png_data_url(frame, upscale=2.5)  # type: ignore[arg-type]

    def test_explicit_upscale_is_capped(self) -> None:
        frame = Frame(grid=((1,),), step=0, level=1)

        with _decoded_image(frame_to_png_data_url(frame, upscale=1000)) as image:
            self.assertEqual(image.size, (64, 64))

    def test_unknown_grid_values_receive_stable_distinct_colors(self) -> None:
        first = color_for_value(42)

        self.assertEqual(first, color_for_value(42))
        self.assertNotEqual(first, ARC_COLOR_MAP[0])
        self.assertNotEqual(first, color_for_value(43))

    def test_signed_unknown_values_receive_distinct_colors(self) -> None:
        self.assertNotEqual(color_for_value(42), color_for_value(-42))

    def test_huge_unknown_value_does_not_overflow(self) -> None:
        value = 10**10000

        self.assertEqual(color_for_value(value), color_for_value(value))

    def test_malformed_cells_render_as_background(self) -> None:
        frame = Frame(
            grid=((True, "invalid", 1),),  # type: ignore[arg-type]
            step=0,
            level=1,
        )

        with _decoded_image(frame_to_png_data_url(frame, upscale=1)) as image:
            self.assertEqual(image.getpixel((0, 0)), ARC_COLOR_MAP[0])
            self.assertEqual(image.getpixel((1, 0)), ARC_COLOR_MAP[0])
            self.assertEqual(image.getpixel((2, 0)), ARC_COLOR_MAP[1])

    def test_oversized_grid_dimension_is_rejected(self) -> None:
        frame = Frame(grid=((0,),) * 513, step=0, level=1)

        with self.assertRaisesRegex(ValueError, "limited to 512"):
            frame_to_png_data_url(frame, upscale=1)

    def test_upscale_is_reduced_to_rendered_pixel_budget(self) -> None:
        frame = Frame(grid=((0, 1), (1, 0)), step=0, level=1)

        with patch.object(vision_context, "_MAX_RENDERED_PIXELS", 16):
            data_url = frame_to_png_data_url(frame, upscale=64)
        with _decoded_image(data_url) as image:
            self.assertEqual(image.size, (4, 4))

    def test_invalid_optional_image_is_omitted(self) -> None:
        frame = Frame(grid=(), step=0, level=1)

        with patch.dict(os.environ, {"MULTIMODAL_CONTEXT": "current_grid"}):
            self.assertIsNone(current_grid_image_part(frame))


if __name__ == "__main__":
    import unittest

    unittest.main()
