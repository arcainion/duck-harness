"""Optional multimodal context helpers for ARC analyzer prompts."""
from __future__ import annotations

import base64
import colorsys
import io
import math
import os
from typing import Any

from PIL import Image

from inference.agent.runtime_state import Frame


ARC_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}
_MAX_GRID_DIMENSION = 512
_MAX_IMAGE_UPSCALE = 64
_MAX_RENDERED_PIXELS = 4_194_304


def color_for_value(value: int) -> tuple[int, int, int]:
    """Render unknown symbols distinctly instead of silently as background."""
    if isinstance(value, bool):
        raise TypeError("Boolean grid values are not color symbols.")
    numeric = int(value)
    if numeric in ARC_COLOR_MAP:
        return ARC_COLOR_MAP[numeric]
    hue = ((numeric * 2_654_435_761) & 0xFFFFFFFF) / 2**32
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
    return int(red * 255), int(green * 255), int(blue * 255)


def multimodal_context() -> str:
    return (
        os.environ.get("MULTIMODAL_CONTEXT", "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def current_grid_image_enabled() -> bool:
    return multimodal_context() == "current_grid"


def current_grid_image_upscale() -> int:
    raw = os.environ.get("MULTIMODAL_UPSCALE", "").strip()
    if not raw:
        return 16
    try:
        return max(1, min(_MAX_IMAGE_UPSCALE, int(raw)))
    except (ValueError, OverflowError):
        return 16


def frame_to_png_data_url(frame: Frame, *, upscale: int | None = None) -> str:
    rows = len(frame.grid)
    cols = max((len(row) for row in frame.grid), default=0)
    if rows <= 0 or cols <= 0:
        raise ValueError("Cannot render an empty grid as an image.")
    if rows > _MAX_GRID_DIMENSION or cols > _MAX_GRID_DIMENSION:
        raise ValueError(
            f"Grid images are limited to {_MAX_GRID_DIMENSION} rows and columns."
        )

    if upscale is None:
        scale = current_grid_image_upscale()
    elif isinstance(upscale, bool) or not isinstance(upscale, int):
        raise TypeError("Image upscale must be an integer.")
    else:
        scale = max(1, min(_MAX_IMAGE_UPSCALE, upscale))
    base_pixels = rows * cols
    scale = min(scale, max(1, math.isqrt(_MAX_RENDERED_PIXELS // base_pixels)))
    image = Image.new("RGB", (cols, rows), ARC_COLOR_MAP[0])
    pixels = image.load()
    for row_idx, row in enumerate(frame.grid):
        for col_idx in range(cols):
            value = row[col_idx] if col_idx < len(row) else 0
            numeric = value if isinstance(value, int) and not isinstance(value, bool) else 0
            pixels[col_idx, row_idx] = color_for_value(numeric)
    if scale > 1:
        image = image.resize((cols * scale, rows * scale), Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def current_grid_image_part(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None or not current_grid_image_enabled():
        return None
    try:
        url = frame_to_png_data_url(frame)
    except (MemoryError, OSError, OverflowError, TypeError, ValueError):
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": url,
        },
    }
