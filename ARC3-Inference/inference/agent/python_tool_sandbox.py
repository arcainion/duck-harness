"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import atexit
import base64
import difflib
import hashlib
import inspect
import json
import marshal
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
import traceback
import zlib
from typing import Any, Callable

from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS

_SANDBOX_REQUIRE_OS_ISOLATION = os.environ.get(
    "LOCAL_ANALYZER_REQUIRE_OS_SANDBOX", ""
).strip().lower() in {"1", "true", "yes", "on"}


class SandboxHostActionError(Exception):
    """Raised when the sandbox host rejects an action request with a safe message."""


def _bounded_frame_crop(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    top: int,
    left: int,
    bottom: int,
    right: int,
    max_area: int = 256,
) -> dict[str, Any]:
    """Return a clipped, letter-coded crop without exposing numeric grid values."""
    coordinates = (top, left, bottom, right)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in coordinates
    ):
        raise TypeError("frame.crop(...) coordinates must be integers.")

    row_count, column_count = shape
    clipped_top = max(0, min(row_count, top))
    clipped_left = max(0, min(column_count, left))
    clipped_bottom = max(0, min(row_count, bottom))
    clipped_right = max(0, min(column_count, right))
    if clipped_bottom <= clipped_top or clipped_right <= clipped_left:
        raise ValueError("frame.crop(...) must select a non-empty region.")

    area = (clipped_bottom - clipped_top) * (clipped_right - clipped_left)
    if area > max_area:
        raise ValueError(f"frame.crop(...) is limited to {max_area} cells.")

    def color_symbol(value: Any) -> str:
        try:
            color_index = int(value)
        except (TypeError, ValueError):
            return "?"
        if 0 <= color_index < len(color_chars):
            return color_chars[color_index]
        return "?"

    rows = []
    for row in range(clipped_top, clipped_bottom):
        source_row = grid[row] if row < len(grid) else []
        rows.append(
            "".join(
                color_symbol(source_row[col]) if col < len(source_row) else "?"
                for col in range(clipped_left, clipped_right)
            )
        )

    bounds = [clipped_top, clipped_left, clipped_bottom, clipped_right]
    return {
        "bounds": bounds,
        "shape": [clipped_bottom - clipped_top, clipped_right - clipped_left],
        "rows": rows,
        "area": area,
        "clipped": bounds != [top, left, bottom, right],
    }


def _frame_cell_symbol(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    row: int,
    col: int,
) -> str:
    """Return one public color symbol without exposing its numeric value."""
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (row, col)
    ):
        raise TypeError("frame.cell(row, col) expects integer coordinates.")
    row_count, column_count = shape
    if row < 0 or row >= row_count or col < 0 or col >= column_count:
        raise ValueError("frame.cell(row, col) coordinates are outside the frame.")
    try:
        color_index = int(grid[row][col])
    except (IndexError, TypeError, ValueError):
        return "?"
    if 0 <= color_index < len(color_chars):
        return color_chars[color_index]
    return "?"


def _bounded_frame_neighbors(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    row: int,
    col: int,
    diagonal: bool = False,
) -> dict[str, Any]:
    """Return the four or eight letter-coded neighbors around one cell."""
    center_symbol = _frame_cell_symbol(
        grid,
        shape=shape,
        color_chars=color_chars,
        row=row,
        col=col,
    )
    if not isinstance(diagonal, bool):
        raise TypeError("frame.neighbors(..., diagonal=...) expects a boolean.")

    offsets = [
        (-1, 0, "UP"),
        (0, 1, "RIGHT"),
        (1, 0, "DOWN"),
        (0, -1, "LEFT"),
    ]
    if diagonal:
        offsets.extend(
            [
                (-1, 1, "UP_RIGHT"),
                (1, 1, "DOWN_RIGHT"),
                (1, -1, "DOWN_LEFT"),
                (-1, -1, "UP_LEFT"),
            ]
        )

    row_count, column_count = shape
    neighbors = []
    for row_delta, col_delta, direction in offsets:
        neighbor_row = row + row_delta
        neighbor_col = col + col_delta
        if not (
            0 <= neighbor_row < row_count and 0 <= neighbor_col < column_count
        ):
            continue
        neighbors.append(
            {
                "row": neighbor_row,
                "col": neighbor_col,
                "symbol": _frame_cell_symbol(
                    grid,
                    shape=shape,
                    color_chars=color_chars,
                    row=neighbor_row,
                    col=neighbor_col,
                ),
                "direction": direction,
                "delta": [row_delta, col_delta],
            }
        )

    return {
        "center": {"row": row, "col": col, "symbol": center_symbol},
        "neighbors": neighbors,
        "diagonal": diagonal,
    }


def _bounded_frame_ray(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    row: int,
    col: int,
    direction: str,
    stop_at: Any = None,
    include_start: bool = False,
    limit: int = 64,
) -> dict[str, Any]:
    """Scan one of eight grid directions with bounded letter-coded samples."""
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (row, col)
    ):
        raise TypeError("frame.ray(row, col, ...) expects integer coordinates.")
    row_count, column_count = shape
    if row < 0 or row >= row_count or col < 0 or col >= column_count:
        raise ValueError("frame.ray(row, col, ...) coordinates are outside the frame.")
    if not isinstance(direction, str):
        raise TypeError("frame.ray(..., direction) expects a direction name.")
    direction_name = direction.strip().upper().replace("-", "_")
    directions = {
        "UP": (-1, 0),
        "UP_RIGHT": (-1, 1),
        "RIGHT": (0, 1),
        "DOWN_RIGHT": (1, 1),
        "DOWN": (1, 0),
        "DOWN_LEFT": (1, -1),
        "LEFT": (0, -1),
        "UP_LEFT": (-1, -1),
    }
    if direction_name not in directions:
        raise ValueError(
            "frame.ray(..., direction) expects one of: "
            + ", ".join(directions)
            + "."
        )
    if not isinstance(include_start, bool):
        raise TypeError("frame.ray(..., include_start=...) expects a boolean.")
    if isinstance(limit, bool):
        raise TypeError("frame.ray(..., limit=...) expects an integer.")
    try:
        cell_limit = max(0, min(256, int(limit)))
    except (TypeError, ValueError) as exc:
        raise TypeError("frame.ray(..., limit=...) expects an integer.") from exc

    if stop_at is None:
        stop_symbols: list[str] = []
    elif isinstance(stop_at, str):
        stop_symbols = list(stop_at)
    elif isinstance(stop_at, (list, tuple, set)):
        stop_symbols = list(stop_at)
    else:
        raise TypeError("frame.ray(..., stop_at=...) expects color symbols.")
    if any(
        not isinstance(symbol, str)
        or len(symbol) != 1
        or symbol not in color_chars
        for symbol in stop_symbols
    ):
        raise ValueError(
            f"frame.ray(..., stop_at=...) expects symbols from {color_chars!r}."
        )
    stop_set = set(stop_symbols)
    normalized_stop = "".join(symbol for symbol in color_chars if symbol in stop_set)

    cells: list[dict[str, Any]] = []

    def append_cell(cell_row: int, cell_col: int, distance: int) -> dict[str, Any]:
        entry = {
            "row": cell_row,
            "col": cell_col,
            "symbol": _frame_cell_symbol(
                grid,
                shape=shape,
                color_chars=color_chars,
                row=cell_row,
                col=cell_col,
            ),
            "distance": distance,
        }
        if len(cells) < cell_limit:
            cells.append(entry)
        return entry

    sampled_total = 0
    if include_start:
        append_cell(row, col, 0)
        sampled_total += 1
    row_delta, col_delta = directions[direction_name]
    current_row = row + row_delta
    current_col = col + col_delta
    length = 0
    hit = None
    while 0 <= current_row < row_count and 0 <= current_col < column_count:
        length += 1
        entry = append_cell(current_row, current_col, length)
        sampled_total += 1
        if entry["symbol"] in stop_set:
            hit = entry
            break
        current_row += row_delta
        current_col += col_delta

    return {
        "origin": [row, col],
        "direction": direction_name,
        "delta": [row_delta, col_delta],
        "stop_at": normalized_stop,
        "length": length,
        "cells": cells,
        "hit": hit,
        "reached_edge": hit is None,
        "truncated_cells": max(0, sampled_total - len(cells)),
    }


def _bounded_frame_find(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    symbol: str,
    limit: int = 64,
) -> dict[str, Any]:
    """Find a color symbol while bounding the coordinate list returned."""
    if not isinstance(symbol, str) or len(symbol) != 1 or symbol not in color_chars:
        raise ValueError(
            f"frame.find(symbol) expects one of the color symbols {color_chars!r}."
        )
    if isinstance(limit, bool):
        raise TypeError("frame.find(..., limit=...) expects an integer limit.")
    try:
        match_limit = max(0, min(256, int(limit)))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "frame.find(..., limit=...) expects an integer limit."
        ) from exc

    target = color_chars.index(symbol)
    row_count, column_count = shape
    count = 0
    cells: list[list[int]] = []
    min_row = min_col = None
    max_row = max_col = None
    for row in range(row_count):
        source_row = grid[row] if row < len(grid) else []
        for col in range(column_count):
            if col >= len(source_row) or source_row[col] != target:
                continue
            count += 1
            min_row = row if min_row is None else min(min_row, row)
            min_col = col if min_col is None else min(min_col, col)
            max_row = row if max_row is None else max(max_row, row)
            max_col = col if max_col is None else max(max_col, col)
            if len(cells) < match_limit:
                cells.append([row, col])

    return {
        "symbol": symbol,
        "count": count,
        "cells": cells,
        "bbox": (
            [min_row, min_col, max_row, max_col]
            if min_row is not None
            else None
        ),
        "truncated": max(0, count - len(cells)),
    }


def _bounded_frame_color_summary(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    limit: int = 16,
) -> dict[str, Any]:
    """Summarize the frame palette without exposing raw numeric color ids."""
    if isinstance(limit, bool):
        raise TypeError("frame.color_summary(..., limit=...) expects an integer.")
    try:
        color_limit = max(0, min(len(color_chars), int(limit)))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "frame.color_summary(..., limit=...) expects an integer."
        ) from exc

    row_count, column_count = shape
    total_cells = row_count * column_count
    stats = [
        {
            "symbol": symbol,
            "count": 0,
            "min_row": None,
            "min_col": None,
            "max_row": None,
            "max_col": None,
            "edge_cells": 0,
            "edge_counts": {"top": 0, "right": 0, "bottom": 0, "left": 0},
        }
        for symbol in color_chars
    ]
    observed_cells = 0
    for row in range(row_count):
        source_row = grid[row] if row < len(grid) else []
        for col in range(column_count):
            value = source_row[col] if col < len(source_row) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= len(color_chars)
            ):
                continue
            observed_cells += 1
            item = stats[value]
            item["count"] += 1
            item["min_row"] = row if item["min_row"] is None else min(item["min_row"], row)
            item["min_col"] = col if item["min_col"] is None else min(item["min_col"], col)
            item["max_row"] = row if item["max_row"] is None else max(item["max_row"], row)
            item["max_col"] = col if item["max_col"] is None else max(item["max_col"], col)
            on_edge = False
            for edge, touches in (
                ("top", row == 0),
                ("right", col == column_count - 1),
                ("bottom", row == row_count - 1),
                ("left", col == 0),
            ):
                if touches:
                    item["edge_counts"][edge] += 1
                    on_edge = True
            if on_edge:
                item["edge_cells"] += 1

    ranked = sorted(
        ((index, item) for index, item in enumerate(stats) if item["count"]),
        key=lambda pair: (-pair[1]["count"], pair[0]),
    )
    colors: list[dict[str, Any]] = []
    for _index, item in ranked[:color_limit]:
        edge_counts = item["edge_counts"]
        colors.append(
            {
                "symbol": item["symbol"],
                "count": item["count"],
                "fraction": item["count"] / total_cells if total_cells else 0.0,
                "bbox": [
                    item["min_row"],
                    item["min_col"],
                    item["max_row"],
                    item["max_col"],
                ],
                "touches_edges": [
                    edge
                    for edge in ("top", "right", "bottom", "left")
                    if edge_counts[edge]
                ],
                "edge_cells": item["edge_cells"],
                "edge_counts": edge_counts,
            }
        )
    omitted = ranked[color_limit:]
    return {
        "shape": [row_count, column_count],
        "total_cells": total_cells,
        "observed_cells": observed_cells,
        "unknown_cells": max(0, total_cells - observed_cells),
        "color_count": len(ranked),
        "colors": colors,
        "omitted_colors": len(omitted),
        "omitted_cells": sum(item["count"] for _index, item in omitted),
    }


def _bounded_frame_spatial_operation(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    operation: str,
    **options: Any,
) -> dict[str, Any]:
    """Serve compact, bounded geometry operations for generated programs."""
    row_count, column_count = shape

    def bounded_limit(name: str, default: int, cap: int = 256) -> int:
        value = options.get(name, default)
        if isinstance(value, bool):
            raise TypeError(f"frame.{operation}(..., {name}=...) expects an integer.")
        try:
            return max(0, min(cap, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.{operation}(..., {name}=...) expects an integer."
            ) from exc

    def coordinate(value: Any, name: str, *, signed: bool = False) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(f"frame.{operation}(..., {name}=...) expects [row, col].")
        row, col = value
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (row, col)):
            raise TypeError(
                f"frame.{operation}(..., {name}=...) expects integer coordinates."
            )
        if not signed and not (0 <= row < row_count and 0 <= col < column_count):
            raise ValueError(f"frame.{operation}(..., {name}=...) is outside the frame.")
        return row, col

    def coordinates(value: Any, name: str = "cells") -> list[tuple[int, int]]:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise TypeError(f"frame.{operation}(..., {name}=...) expects coordinates.")
        if len(value) > 256:
            raise ValueError(f"frame.{operation}(..., {name}=...) is limited to 256 cells.")
        return [coordinate(item, f"{name}[{index}]") for index, item in enumerate(value)]

    def symbols(value: Any, name: str = "symbols") -> list[str]:
        raw = list(value) if isinstance(value, str) else list(value) if isinstance(value, (list, tuple, set)) else []
        if not raw or any(
            not isinstance(item, str) or len(item) != 1 or item not in color_chars
            for item in raw
        ):
            raise ValueError(
                f"frame.{operation}(..., {name}=...) expects symbols from {color_chars!r}."
            )
        return list(dict.fromkeys(raw))

    def bounds(value: Any, name: str = "bounds") -> tuple[int, int, int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise TypeError(
                f"frame.{operation}(..., {name}=...) expects [top, left, bottom, right]."
            )
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise TypeError(f"frame.{operation}(..., {name}=...) expects integers.")
        top, left, bottom, right = value
        if not (0 <= top < bottom <= row_count and 0 <= left < right <= column_count):
            raise ValueError(
                f"frame.{operation}(..., {name}=...) must be a non-empty region inside the frame."
            )
        return top, left, bottom, right

    def value_at(row: int, col: int) -> int | None:
        source_row = grid[row] if row < len(grid) else []
        value = source_row[col] if col < len(source_row) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return None
        return value

    def cell_payload(row: int, col: int) -> dict[str, Any]:
        value = value_at(row, col)
        return {
            "cell": [row, col],
            "symbol": color_chars[value] if value is not None else None,
        }

    if operation == "bounds":
        requested = symbols(options.get("symbols"))
        requested_values = {color_chars.index(symbol) for symbol in requested}
        cell_limit = bounded_limit("limit", 64)
        cells: list[list[int]] = []
        count = 0
        row_total = col_total = 0
        min_row = min_col = None
        max_row = max_col = None
        for row in range(row_count):
            for col in range(column_count):
                if value_at(row, col) not in requested_values:
                    continue
                count += 1
                row_total += row
                col_total += col
                min_row = row if min_row is None else min(min_row, row)
                min_col = col if min_col is None else min(min_col, col)
                max_row = row if max_row is None else max(max_row, row)
                max_col = col if max_col is None else max(max_col, col)
                if len(cells) < cell_limit:
                    cells.append([row, col])
        return {
            "symbols": requested,
            "count": count,
            "bbox": [min_row, min_col, max_row, max_col] if count else None,
            "centroid": [row_total / count, col_total / count] if count else None,
            "cells": cells,
            "truncated_cells": max(0, count - len(cells)),
        }

    if operation == "region_summary":
        top, left, bottom, right = bounds(options.get("bounds"))
        counts: dict[str, int] = {}
        unknown = 0
        for row in range(top, bottom):
            for col in range(left, right):
                value = value_at(row, col)
                if value is None:
                    unknown += 1
                else:
                    symbol = color_chars[value]
                    counts[symbol] = counts.get(symbol, 0) + 1
        area = (bottom - top) * (right - left)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], color_chars.index(item[0])))
        return {
            "bounds": [top, left, bottom, right],
            "shape": [bottom - top, right - left],
            "area": area,
            "colors": [
                {"symbol": symbol, "count": count, "fraction": count / area}
                for symbol, count in ranked
            ],
            "unknown_cells": unknown,
        }

    if operation in {"row_profile", "column_profile"}:
        requested_raw = options.get("symbol")
        requested = None if requested_raw is None else symbols(requested_raw, "symbol")
        axis_length = row_count if operation == "row_profile" else column_count
        cross_length = column_count if operation == "row_profile" else row_count
        profile_limit = bounded_limit("limit", 64)
        profiles = []
        for axis_index in range(axis_length):
            counts: dict[str, int] = {}
            for cross_index in range(cross_length):
                row, col = (axis_index, cross_index) if operation == "row_profile" else (cross_index, axis_index)
                value = value_at(row, col)
                if value is None:
                    continue
                symbol = color_chars[value]
                if requested is None or symbol in requested:
                    counts[symbol] = counts.get(symbol, 0) + 1
            profiles.append(
                {
                    "index": axis_index,
                    "counts": {
                        symbol: counts[symbol]
                        for symbol in color_chars
                        if symbol in counts
                    },
                    "matched": sum(counts.values()),
                }
            )
        return {
            "axis": "row" if operation == "row_profile" else "column",
            "symbol_filter": requested,
            "count": len(profiles),
            "profiles": profiles[:profile_limit],
            "truncated_profiles": max(0, len(profiles) - profile_limit),
        }

    if operation in {"nearest_cell", "distance"}:
        start = coordinate(options.get("start"), "start")
        metric = str(options.get("metric", "manhattan") or "").strip().lower()
        if metric not in {"manhattan", "chebyshev", "euclidean_squared"}:
            raise ValueError(
                f"frame.{operation}(..., metric=...) expects manhattan, chebyshev, or euclidean_squared."
            )

        def distance_to(cell: tuple[int, int]) -> int:
            row_delta = abs(cell[0] - start[0])
            col_delta = abs(cell[1] - start[1])
            if metric == "manhattan":
                return row_delta + col_delta
            if metric == "chebyshev":
                return max(row_delta, col_delta)
            return row_delta * row_delta + col_delta * col_delta

        if operation == "distance":
            end = coordinate(options.get("end"), "end")
            return {"start": list(start), "end": list(end), "metric": metric, "distance": distance_to(end)}
        requested = symbols(options.get("symbols"))
        requested_values = {color_chars.index(symbol) for symbol in requested}
        result_limit = bounded_limit("limit", 64)
        ranked = [
            (distance_to((row, col)), row, col)
            for row in range(row_count)
            for col in range(column_count)
            if value_at(row, col) in requested_values
        ]
        ranked.sort()
        candidates = [
            {**cell_payload(row, col), "distance": distance}
            for distance, row, col in ranked[:result_limit]
        ]
        return {
            "start": list(start),
            "symbols": requested,
            "metric": metric,
            "count": len(ranked),
            "nearest": candidates[0] if candidates else None,
            "candidates": candidates,
            "truncated_candidates": max(0, len(ranked) - len(candidates)),
        }

    if operation == "line_between":
        start = coordinate(options.get("start"), "start")
        end = coordinate(options.get("end"), "end")
        line_limit = bounded_limit("limit", 128, 512)
        row, col = start
        end_row, end_col = end
        delta_col = abs(end_col - col)
        step_col = 1 if col < end_col else -1
        delta_row = -abs(end_row - row)
        step_row = 1 if row < end_row else -1
        error = delta_col + delta_row
        line: list[tuple[int, int]] = []
        while True:
            line.append((row, col))
            if (row, col) == end:
                break
            doubled = 2 * error
            if doubled >= delta_row:
                error += delta_row
                col += step_col
            if doubled <= delta_col:
                error += delta_col
                row += step_row
        return {
            "start": list(start),
            "end": list(end),
            "count": len(line),
            "cells": [cell_payload(row, col) for row, col in line[:line_limit]],
            "truncated_cells": max(0, len(line) - line_limit),
        }

    if operation in {"translate_cells", "mirror_cells"}:
        source = coordinates(options.get("cells"))
        result_limit = bounded_limit("limit", 128, 256)
        mapped: list[dict[str, Any]] = []
        if operation == "translate_cells":
            row_delta, col_delta = coordinate(options.get("delta"), "delta", signed=True)

            def transform(row: int, col: int) -> tuple[int, int]:
                return row + row_delta, col + col_delta

            transform_name = "TRANSLATE"
        else:
            transform_name = str(options.get("symmetry") or "").strip().upper()
            transforms = {
                "HORIZONTAL": lambda row, col: (row_count - 1 - row, col),
                "VERTICAL": lambda row, col: (row, column_count - 1 - col),
                "ROTATE_180": lambda row, col: (row_count - 1 - row, column_count - 1 - col),
                "MAIN_DIAGONAL": lambda row, col: (col, row),
                "ANTI_DIAGONAL": lambda row, col: (column_count - 1 - col, row_count - 1 - row),
            }
            if transform_name not in transforms:
                raise ValueError(
                    "frame.mirror_cells(..., symmetry=...) expects HORIZONTAL, VERTICAL, "
                    "ROTATE_180, MAIN_DIAGONAL, or ANTI_DIAGONAL."
                )
            transform = transforms[transform_name]
        outside = 0
        for row, col in source:
            mapped_row, mapped_col = transform(row, col)
            in_bounds = 0 <= mapped_row < row_count and 0 <= mapped_col < column_count
            outside += not in_bounds
            if len(mapped) < result_limit:
                mapped.append(
                    {
                        "source": [row, col],
                        "target": [mapped_row, mapped_col],
                        "in_bounds": in_bounds,
                        "symbol": cell_payload(mapped_row, mapped_col)["symbol"] if in_bounds else None,
                    }
                )
        return {
            "transform": transform_name,
            "count": len(source),
            "mapped": mapped,
            "outside_frame": outside,
            "truncated_mappings": max(0, len(source) - len(mapped)),
        }

    if operation == "compare_regions":
        first = bounds(options.get("first"), "first")
        second = bounds(options.get("second"), "second")
        first_grid = [
            [value_at(row, col) for col in range(first[1], first[3])]
            for row in range(first[0], first[2])
        ]
        second_grid = [
            [value_at(row, col) for col in range(second[1], second[3])]
            for row in range(second[0], second[2])
        ]
        relation = _bounded_frame_transform_relation(
            first_grid,
            second_grid,
            before_shape=(first[2] - first[0], first[3] - first[1]),
            after_shape=(second[2] - second[0], second[3] - second[1]),
            color_chars=color_chars,
            allow_recolor=options.get("allow_recolor", True),
        )
        return {"first": list(first), "second": list(second), **relation}

    raise ValueError(f"Unknown spatial operation: {operation}")


def _bounded_frame_layout_operation(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    operation: str,
    **options: Any,
) -> dict[str, Any]:
    """Serve bounded frame-layout and color-topology operations."""
    row_count, column_count = shape

    def integer(name: str, default: int, cap: int = 256, minimum: int = 0) -> int:
        value = options.get(name, default)
        if isinstance(value, bool):
            raise TypeError(f"frame.{operation}(..., {name}=...) expects an integer.")
        try:
            return max(minimum, min(cap, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.{operation}(..., {name}=...) expects an integer."
            ) from exc

    def symbol(value: Any, name: str, *, optional: bool = False) -> str | None:
        if optional and value is None:
            return None
        if not isinstance(value, str) or len(value) != 1 or value not in color_chars:
            raise ValueError(
                f"frame.{operation}(..., {name}=...) expects a symbol from {color_chars!r}."
            )
        return value

    def symbols(value: Any) -> list[str]:
        raw = list(value) if isinstance(value, (str, list, tuple, set)) else []
        if not raw or any(
            not isinstance(item, str) or len(item) != 1 or item not in color_chars
            for item in raw
        ):
            raise ValueError(
                f"frame.{operation}(..., symbols=...) expects symbols from {color_chars!r}."
            )
        return [item for item in color_chars if item in raw]

    def value_at(row: int, col: int) -> int | None:
        source_row = grid[row] if 0 <= row < len(grid) else []
        value = source_row[col] if 0 <= col < len(source_row) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return None
        return value

    def palette(bounds: tuple[int, int, int, int]) -> dict[str, Any]:
        top, left, bottom, right = bounds
        counts: dict[str, int] = {}
        unknown = 0
        for row in range(top, bottom):
            for col in range(left, right):
                value = value_at(row, col)
                if value is None:
                    unknown += 1
                else:
                    cell_symbol = color_chars[value]
                    counts[cell_symbol] = counts.get(cell_symbol, 0) + 1
        area = max(0, bottom - top) * max(0, right - left)
        return {
            "bounds": [top, left, bottom, right],
            "shape": [max(0, bottom - top), max(0, right - left)],
            "area": area,
            "colors": [
                {"symbol": item, "count": counts[item], "fraction": counts[item] / area if area else 0.0}
                for item in sorted(counts, key=lambda item: (-counts[item], color_chars.index(item)))
            ],
            "unknown_cells": unknown,
        }

    if operation == "border_summary":
        thickness = integer("thickness", 1, max(row_count, column_count, 1), 1)
        row_thickness = min(row_count, thickness)
        col_thickness = min(column_count, thickness)
        regions = {
            "top": (0, 0, row_thickness, column_count),
            "right": (0, column_count - col_thickness, row_count, column_count),
            "bottom": (row_count - row_thickness, 0, row_count, column_count),
            "left": (0, 0, row_count, col_thickness),
        }
        return {
            "thickness": thickness,
            "sides": {name: palette(bounds) for name, bounds in regions.items()},
            "corner_cells_counted_on_two_sides": True,
        }

    if operation == "corner_summary":
        size = integer("size", 1, max(row_count, column_count, 1), 1)
        height = min(row_count, size)
        width = min(column_count, size)
        regions = {
            "top_left": (0, 0, height, width),
            "top_right": (0, column_count - width, height, column_count),
            "bottom_right": (row_count - height, column_count - width, row_count, column_count),
            "bottom_left": (row_count - height, 0, row_count, width),
        }
        return {"size": size, "corners": {name: palette(bounds) for name, bounds in regions.items()}}

    if operation == "center_summary":
        radius = integer("radius", 0, max(row_count, column_count, 1))
        top = max(0, (row_count - 1) // 2 - radius)
        bottom = min(row_count, row_count // 2 + 1 + radius)
        left = max(0, (column_count - 1) // 2 - radius)
        right = min(column_count, column_count // 2 + 1 + radius)
        return {"radius": radius, **palette((top, left, bottom, right))}

    if operation == "quadrant_summary":
        row_split = (row_count + 1) // 2
        col_split = (column_count + 1) // 2
        regions = {
            "top_left": (0, 0, row_split, col_split),
            "top_right": (0, col_split, row_split, column_count),
            "bottom_right": (row_split, col_split, row_count, column_count),
            "bottom_left": (row_split, 0, row_count, col_split),
        }
        return {
            "split": [row_split, col_split],
            "quadrants": {name: palette(bounds) for name, bounds in regions.items()},
        }

    if operation == "color_adjacency":
        diagonal = options.get("diagonal", False)
        if not isinstance(diagonal, bool):
            raise TypeError("frame.color_adjacency(..., diagonal=...) expects a boolean.")
        result_limit = integer("limit", 64)
        offsets = [(0, 1), (1, 0)]
        if diagonal:
            offsets.extend([(1, 1), (1, -1)])
        contacts: dict[tuple[str, str], dict[str, Any]] = {}
        for row in range(row_count):
            for col in range(column_count):
                first_value = value_at(row, col)
                if first_value is None:
                    continue
                for row_delta, col_delta in offsets:
                    other_row, other_col = row + row_delta, col + col_delta
                    if not (0 <= other_row < row_count and 0 <= other_col < column_count):
                        continue
                    second_value = value_at(other_row, other_col)
                    if second_value is None or second_value == first_value:
                        continue
                    values = sorted((first_value, second_value))
                    key = color_chars[values[0]], color_chars[values[1]]
                    item = contacts.setdefault(key, {"colors": list(key), "contacts": 0, "samples": []})
                    item["contacts"] += 1
                    if len(item["samples"]) < 8:
                        item["samples"].append([[row, col], [other_row, other_col]])
        ranked = sorted(contacts.values(), key=lambda item: (-item["contacts"], item["colors"]))
        return {
            "diagonal": diagonal,
            "count": len(ranked),
            "pairs": ranked[:result_limit],
            "truncated_pairs": max(0, len(ranked) - result_limit),
        }

    if operation == "distance_between_colors":
        first_symbol = symbol(options.get("first"), "first")
        second_symbol = symbol(options.get("second"), "second")
        metric = str(options.get("metric", "manhattan") or "").strip().lower()
        if metric not in {"manhattan", "chebyshev", "euclidean_squared"}:
            raise ValueError("frame.distance_between_colors(..., metric=...) received an invalid metric.")
        first_cells = [
            (row, col) for row in range(row_count) for col in range(column_count)
            if value_at(row, col) == color_chars.index(first_symbol)
        ]
        second_cells = [
            (row, col) for row in range(row_count) for col in range(column_count)
            if value_at(row, col) == color_chars.index(second_symbol)
        ]
        best = None
        for first_cell in first_cells:
            for second_cell in second_cells:
                if first_cell == second_cell:
                    continue
                row_delta = abs(first_cell[0] - second_cell[0])
                col_delta = abs(first_cell[1] - second_cell[1])
                distance = (
                    row_delta + col_delta if metric == "manhattan"
                    else max(row_delta, col_delta) if metric == "chebyshev"
                    else row_delta * row_delta + col_delta * col_delta
                )
                candidate = distance, first_cell, second_cell
                if best is None or candidate < best:
                    best = candidate
        return {
            "first": first_symbol,
            "second": second_symbol,
            "metric": metric,
            "first_count": len(first_cells),
            "second_count": len(second_cells),
            "distance": best[0] if best else None,
            "cells": [list(best[1]), list(best[2])] if best else None,
        }

    if operation in {"divider_lines", "panels"}:
        requested = symbol(options.get("symbol"), "symbol", optional=True)
        divider_rows = []
        divider_columns = []
        for row in range(row_count):
            values = [value_at(row, col) for col in range(column_count)]
            if values and values[0] is not None and all(value == values[0] for value in values):
                line_symbol = color_chars[values[0]]
                if requested is None or line_symbol == requested:
                    divider_rows.append({"index": row, "symbol": line_symbol})
        for col in range(column_count):
            values = [value_at(row, col) for row in range(row_count)]
            if values and values[0] is not None and all(value == values[0] for value in values):
                line_symbol = color_chars[values[0]]
                if requested is None or line_symbol == requested:
                    divider_columns.append({"index": col, "symbol": line_symbol})
        if operation == "divider_lines":
            result_limit = integer("limit", 64)
            return {
                "symbol_filter": requested,
                "row_count": len(divider_rows),
                "column_count": len(divider_columns),
                "rows": divider_rows[:result_limit],
                "columns": divider_columns[:result_limit],
                "truncated_lines": max(0, len(divider_rows) - result_limit) + max(0, len(divider_columns) - result_limit),
            }

        def intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
            result = []
            start = 0
            for separator in separators:
                if start < separator:
                    result.append((start, separator))
                start = separator + 1
            if start < length:
                result.append((start, length))
            return result

        row_intervals = intervals(row_count, [item["index"] for item in divider_rows])
        col_intervals = intervals(column_count, [item["index"] for item in divider_columns])
        all_panels = [palette((top, left, bottom, right)) for top, bottom in row_intervals for left, right in col_intervals]
        panel_limit = integer("limit", 64)
        return {
            "symbol_filter": requested,
            "divider_rows": [item["index"] for item in divider_rows],
            "divider_columns": [item["index"] for item in divider_columns],
            "count": len(all_panels),
            "panels": all_panels[:panel_limit],
            "truncated_panels": max(0, len(all_panels) - panel_limit),
        }

    if operation == "tile_summary":
        tile_height = integer("tile_height", 1, max(row_count, 1), 1)
        tile_width = integer("tile_width", 1, max(column_count, 1), 1)
        tile_limit = integer("limit", 64)
        tiles = []
        for top in range(0, row_count, tile_height):
            for left in range(0, column_count, tile_width):
                bottom = min(row_count, top + tile_height)
                right = min(column_count, left + tile_width)
                item = palette((top, left, bottom, right))
                item["complete"] = bottom - top == tile_height and right - left == tile_width
                tiles.append(item)
        return {
            "tile_shape": [tile_height, tile_width],
            "grid_shape": [
                (row_count + tile_height - 1) // tile_height,
                (column_count + tile_width - 1) // tile_width,
            ],
            "count": len(tiles),
            "tiles": tiles[:tile_limit],
            "truncated_tiles": max(0, len(tiles) - tile_limit),
        }

    if operation == "edge_distance":
        requested = symbols(options.get("symbols"))
        requested_values = {color_chars.index(item) for item in requested}
        sample_limit = integer("limit", 64)
        ranked = []
        histogram: dict[int, int] = {}
        for row in range(row_count):
            for col in range(column_count):
                if value_at(row, col) not in requested_values:
                    continue
                distance = min(row, col, row_count - 1 - row, column_count - 1 - col)
                histogram[distance] = histogram.get(distance, 0) + 1
                ranked.append((distance, row, col))
        ranked.sort()
        return {
            "symbols": requested,
            "count": len(ranked),
            "minimum": ranked[0][0] if ranked else None,
            "maximum": max((item[0] for item in ranked), default=None),
            "histogram": {str(distance): histogram[distance] for distance in sorted(histogram)},
            "nearest": [[row, col] for _distance, row, col in ranked[:sample_limit]],
            "truncated_nearest": max(0, len(ranked) - sample_limit),
        }

    raise ValueError(f"Unknown layout operation: {operation}")


def _bounded_frame_runs(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    symbol: str | None = None,
    directions: str = "HV",
    min_length: int = 2,
    limit: int = 64,
) -> dict[str, Any]:
    """Find maximal same-color horizontal, vertical, and diagonal runs."""
    if symbol is not None and (
        not isinstance(symbol, str) or len(symbol) != 1 or symbol not in color_chars
    ):
        raise ValueError(
            f"frame.runs(..., symbol=...) expects one of {color_chars!r} or None."
        )
    if not isinstance(directions, str) or not directions.strip():
        raise ValueError("frame.runs(..., directions=...) expects H, V, and/or D.")
    direction_codes = directions.strip().upper()
    if any(code not in "HVD" for code in direction_codes):
        raise ValueError("frame.runs(..., directions=...) expects H, V, and/or D.")

    def bounded_integer(value: Any, name: str, minimum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"frame.runs(..., {name}=...) expects an integer.")
        try:
            return max(minimum, min(256, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.runs(..., {name}=...) expects an integer."
            ) from exc

    required_length = bounded_integer(min_length, "min_length", 1)
    run_limit = bounded_integer(limit, "limit", 0)
    target = color_chars.index(symbol) if symbol is not None else None
    axes: list[tuple[str, int, int]] = []
    if "H" in direction_codes:
        axes.append(("HORIZONTAL", 0, 1))
    if "V" in direction_codes:
        axes.append(("VERTICAL", 1, 0))
    if "D" in direction_codes:
        axes.extend(
            [
                ("DIAGONAL_DOWN", 1, 1),
                ("DIAGONAL_UP", -1, 1),
            ]
        )

    row_count, column_count = shape

    def cell(row: int, col: int) -> int | None:
        if not (0 <= row < row_count and 0 <= col < column_count):
            return None
        if row >= len(grid) or col >= len(grid[row]):
            return None
        value = grid[row][col]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return None
        return value

    runs: list[dict[str, Any]] = []
    run_count = 0
    counts_by_direction = {name: 0 for name, _row_delta, _col_delta in axes}
    counts_by_symbol: dict[str, int] = {}
    for name, row_delta, col_delta in axes:
        for row in range(row_count):
            for col in range(column_count):
                value = cell(row, col)
                if value is None or (target is not None and value != target):
                    continue
                if cell(row - row_delta, col - col_delta) == value:
                    continue
                end_row, end_col = row, col
                length = 1
                while cell(end_row + row_delta, end_col + col_delta) == value:
                    end_row += row_delta
                    end_col += col_delta
                    length += 1
                if length < required_length:
                    continue
                run_count += 1
                counts_by_direction[name] += 1
                color_symbol = color_chars[value]
                counts_by_symbol[color_symbol] = counts_by_symbol.get(color_symbol, 0) + 1
                if len(runs) < run_limit:
                    runs.append(
                        {
                            "symbol": color_symbol,
                            "direction": name,
                            "start": [row, col],
                            "end": [end_row, end_col],
                            "delta": [row_delta, col_delta],
                            "length": length,
                        }
                    )

    return {
        "symbol": symbol,
        "directions": [name for name, _row_delta, _col_delta in axes],
        "min_length": required_length,
        "count": run_count,
        "runs": runs,
        "counts_by_direction": counts_by_direction,
        "counts_by_symbol": counts_by_symbol,
        "truncated_runs": max(0, run_count - len(runs)),
    }


def _bounded_frame_rectangles(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    symbol: str | None = None,
    kind: str = "any",
    min_size: int = 2,
    limit: int = 64,
) -> dict[str, Any]:
    """Detect maximal filled rectangles and closed rectangular outlines."""
    if symbol is not None and (
        not isinstance(symbol, str) or len(symbol) != 1 or symbol not in color_chars
    ):
        raise ValueError(
            f"frame.rectangles(..., symbol=...) expects one of {color_chars!r} or None."
        )
    if not isinstance(kind, str) or kind.lower() not in {"any", "filled", "outline"}:
        raise ValueError(
            "frame.rectangles(..., kind=...) expects 'any', 'filled', or 'outline'."
        )
    requested_kind = kind.lower()

    def bounded_integer(value: Any, name: str, minimum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"frame.rectangles(..., {name}=...) expects an integer.")
        try:
            return max(minimum, min(256, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.rectangles(..., {name}=...) expects an integer."
            ) from exc

    required_size = bounded_integer(min_size, "min_size", 1)
    rectangle_limit = bounded_integer(limit, "limit", 0)
    target = color_chars.index(symbol) if symbol is not None else None
    row_count, column_count = shape

    def cell(row: int, col: int) -> int | None:
        if not (0 <= row < row_count and 0 <= col < column_count):
            return None
        if row >= len(grid) or col >= len(grid[row]):
            return None
        value = grid[row][col]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return None
        return value

    visited: set[tuple[int, int]] = set()
    rectangles: list[dict[str, Any]] = []
    rectangle_count = 0
    counts_by_kind = {"filled": 0, "outline": 0}
    counts_by_symbol: dict[str, int] = {}
    for start_row in range(row_count):
        for start_col in range(column_count):
            start_value = cell(start_row, start_col)
            start = (start_row, start_col)
            if (
                start in visited
                or start_value is None
                or (target is not None and start_value != target)
            ):
                continue
            stack = [start]
            visited.add(start)
            cells: list[tuple[int, int]] = []
            min_row = max_row = start_row
            min_col = max_col = start_col
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                min_row = min(min_row, row)
                min_col = min(min_col, col)
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor = (row + row_delta, col + col_delta)
                    if neighbor not in visited and cell(*neighbor) == start_value:
                        visited.add(neighbor)
                        stack.append(neighbor)

            height = max_row - min_row + 1
            width = max_col - min_col + 1
            if height < required_size or width < required_size:
                continue
            area = height * width
            filled = len(cells) == area and all(
                cell(row, col) == start_value
                for row in range(min_row, max_row + 1)
                for col in range(min_col, max_col + 1)
            )
            border_complete = all(
                cell(row, col) == start_value
                for row in range(min_row, max_row + 1)
                for col in range(min_col, max_col + 1)
                if row in {min_row, max_row} or col in {min_col, max_col}
            )
            detected_kind = "filled" if filled else "outline" if border_complete else None
            if detected_kind is None or (
                requested_kind != "any" and requested_kind != detected_kind
            ):
                continue

            rectangle_count += 1
            counts_by_kind[detected_kind] += 1
            color_symbol = color_chars[start_value]
            counts_by_symbol[color_symbol] = counts_by_symbol.get(color_symbol, 0) + 1
            if len(rectangles) >= rectangle_limit:
                continue
            interior_colors: dict[str, int] = {}
            for row in range(min_row + 1, max_row):
                for col in range(min_col + 1, max_col):
                    value = cell(row, col)
                    interior_symbol = (
                        color_chars[value] if value is not None else "?"
                    )
                    interior_colors[interior_symbol] = (
                        interior_colors.get(interior_symbol, 0) + 1
                    )
            border_cells = area if height <= 2 or width <= 2 else 2 * height + 2 * width - 4
            rectangles.append(
                {
                    "symbol": color_symbol,
                    "kind": detected_kind,
                    "bbox": [min_row, min_col, max_row, max_col],
                    "shape": [height, width],
                    "area": area,
                    "border_cells": border_cells,
                    "interior_area": max(0, height - 2) * max(0, width - 2),
                    "component_size": len(cells),
                    "fill_ratio": len(cells) / area,
                    "touches_edge": (
                        min_row == 0
                        or min_col == 0
                        or max_row == row_count - 1
                        or max_col == column_count - 1
                    ),
                    "interior_colors": interior_colors,
                }
            )

    return {
        "symbol": symbol,
        "kind": requested_kind,
        "min_size": required_size,
        "count": rectangle_count,
        "rectangles": rectangles,
        "counts_by_kind": counts_by_kind,
        "counts_by_symbol": counts_by_symbol,
        "truncated_rectangles": max(0, rectangle_count - len(rectangles)),
    }


def _bounded_frame_enclosed_regions(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    symbol: str | None = None,
    diagonal: bool = False,
    limit: int = 64,
    cell_limit: int = 128,
) -> dict[str, Any]:
    """Summarize equal-color regions that do not touch the frame edge."""
    if symbol is not None and (
        not isinstance(symbol, str) or len(symbol) != 1 or symbol not in color_chars
    ):
        raise ValueError(
            "frame.enclosed_regions(..., symbol=...) expects a color symbol or None."
        )
    if not isinstance(diagonal, bool):
        raise TypeError(
            "frame.enclosed_regions(..., diagonal=...) expects a boolean."
        )

    def bounded_integer(value: Any, name: str, maximum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(
                f"frame.enclosed_regions(..., {name}=...) expects an integer."
            )
        try:
            return max(0, min(maximum, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.enclosed_regions(..., {name}=...) expects an integer."
            ) from exc

    region_limit = bounded_integer(limit, "limit", 128)
    sample_limit = bounded_integer(cell_limit, "cell_limit", 256)
    target = color_chars.index(symbol) if symbol is not None else None
    row_count, column_count = shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def cell(row: int, col: int) -> int | None:
        if not (0 <= row < row_count and 0 <= col < column_count):
            return None
        if row >= len(grid) or col >= len(grid[row]):
            return None
        value = grid[row][col]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return None
        return value

    visited: set[tuple[int, int]] = set()
    regions: list[dict[str, Any]] = []
    region_count = 0
    sampled_cells = 0
    counts_by_symbol: dict[str, int] = {}
    for start_row in range(row_count):
        for start_col in range(column_count):
            start = (start_row, start_col)
            start_value = cell(*start)
            if (
                start in visited
                or start_value is None
                or (target is not None and start_value != target)
            ):
                continue
            stack = [start]
            visited.add(start)
            component: list[tuple[int, int]] = []
            row_sum = 0
            col_sum = 0
            min_row = max_row = start_row
            min_col = max_col = start_col
            touches_edge = False
            while stack:
                row, col = stack.pop()
                component.append((row, col))
                row_sum += row
                col_sum += col
                min_row = min(min_row, row)
                min_col = min(min_col, col)
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                touches_edge = (
                    touches_edge
                    or row in {0, row_count - 1}
                    or col in {0, column_count - 1}
                )
                for row_delta, col_delta in directions:
                    neighbor = (row + row_delta, col + col_delta)
                    if neighbor not in visited and cell(*neighbor) == start_value:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if touches_edge:
                continue

            region_count += 1
            color_symbol = color_chars[start_value]
            counts_by_symbol[color_symbol] = counts_by_symbol.get(color_symbol, 0) + 1
            if len(regions) >= region_limit:
                continue
            component_set = set(component)
            boundary_coordinates: set[tuple[int, int]] = set()
            for row, col in component:
                for row_delta, col_delta in directions:
                    neighbor = (row + row_delta, col + col_delta)
                    if neighbor not in component_set:
                        boundary_coordinates.add(neighbor)
            boundary_colors: dict[str, int] = {}
            for row, col in sorted(boundary_coordinates):
                value = cell(row, col)
                boundary_symbol = color_chars[value] if value is not None else "?"
                boundary_colors[boundary_symbol] = (
                    boundary_colors.get(boundary_symbol, 0) + 1
                )
            remaining_samples = max(0, sample_limit - sampled_cells)
            sampled = [[row, col] for row, col in component[:remaining_samples]]
            sampled_cells += len(sampled)
            size = len(component)
            regions.append(
                {
                    "id": region_count - 1,
                    "symbol": color_symbol,
                    "size": size,
                    "bbox": [min_row, min_col, max_row, max_col],
                    "centroid": [row_sum / size, col_sum / size],
                    "boundary_cells": len(boundary_coordinates),
                    "boundary_colors": boundary_colors,
                    "cells": sampled,
                    "truncated_cells": max(0, size - len(sampled)),
                }
            )

    return {
        "symbol": symbol,
        "connectivity": 8 if diagonal else 4,
        "count": region_count,
        "regions": regions,
        "counts_by_symbol": counts_by_symbol,
        "sampled_cells": sampled_cells,
        "truncated_regions": max(0, region_count - len(regions)),
    }


def _bounded_frame_components(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    symbol: str,
    diagonal: bool = False,
    limit: int = 64,
    cell_limit: int = 128,
) -> dict[str, Any]:
    """Summarize equal-color connected components with bounded cell samples."""
    if not isinstance(symbol, str) or len(symbol) != 1 or symbol not in color_chars:
        raise ValueError(
            f"frame.components(symbol) expects one of the color symbols {color_chars!r}."
        )
    if not isinstance(diagonal, bool):
        raise TypeError("frame.components(..., diagonal=...) expects a boolean.")

    def bounded_integer(value: Any, name: str, maximum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"frame.components(..., {name}=...) expects an integer.")
        try:
            return max(0, min(maximum, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.components(..., {name}=...) expects an integer."
            ) from exc

    component_limit = bounded_integer(limit, "limit", 128)
    sample_limit = bounded_integer(cell_limit, "cell_limit", 256)
    target = color_chars.index(symbol)
    row_count, column_count = shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def is_target(row: int, col: int) -> bool:
        return (
            0 <= row < row_count
            and 0 <= col < column_count
            and row < len(grid)
            and col < len(grid[row])
            and grid[row][col] == target
        )

    visited: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    component_count = 0
    sampled_cells = 0
    for start_row in range(row_count):
        for start_col in range(column_count):
            start = (start_row, start_col)
            if start in visited or not is_target(*start):
                continue
            component_count += 1
            include_component = len(components) < component_limit
            stack = [start]
            visited.add(start)
            size = 0
            row_sum = 0
            col_sum = 0
            min_row = max_row = start_row
            min_col = max_col = start_col
            touches_edge = False
            cells: list[list[int]] = []
            while stack:
                row, col = stack.pop()
                size += 1
                row_sum += row
                col_sum += col
                min_row = min(min_row, row)
                min_col = min(min_col, col)
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                touches_edge = (
                    touches_edge
                    or row in {0, row_count - 1}
                    or col in {0, column_count - 1}
                )
                if include_component and sampled_cells < sample_limit:
                    cells.append([row, col])
                    sampled_cells += 1
                for row_delta, col_delta in directions:
                    neighbor = (row + row_delta, col + col_delta)
                    if neighbor not in visited and is_target(*neighbor):
                        visited.add(neighbor)
                        stack.append(neighbor)

            if include_component:
                components.append(
                    {
                        "id": component_count - 1,
                        "size": size,
                        "bbox": [min_row, min_col, max_row, max_col],
                        "centroid": [row_sum / size, col_sum / size],
                        "touches_edge": touches_edge,
                        "cells": cells,
                        "truncated_cells": max(0, size - len(cells)),
                    }
                )

    return {
        "symbol": symbol,
        "connectivity": 8 if diagonal else 4,
        "count": component_count,
        "components": components,
        "sampled_cells": sampled_cells,
        "truncated_components": max(0, component_count - len(components)),
    }


def _bounded_frame_objects(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    background: str | None = None,
    diagonal: bool = False,
    limit: int = 64,
    cell_limit: int = 128,
) -> dict[str, Any]:
    """Group connected non-background cells into bounded multi-color objects."""
    if background is not None and (
        not isinstance(background, str)
        or not background
        or any(symbol not in color_chars for symbol in background)
    ):
        raise ValueError(
            "frame.objects(..., background=...) expects one or more color symbols."
        )
    if not isinstance(diagonal, bool):
        raise TypeError("frame.objects(..., diagonal=...) expects a boolean.")

    def bounded_integer(value: Any, name: str, maximum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"frame.objects(..., {name}=...) expects an integer.")
        try:
            return max(0, min(maximum, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.objects(..., {name}=...) expects an integer."
            ) from exc

    object_limit = bounded_integer(limit, "limit", 128)
    sample_limit = bounded_integer(cell_limit, "cell_limit", 256)
    row_count, column_count = shape
    if background is None:
        counts: dict[int, int] = {}
        for row in range(min(row_count, len(grid))):
            for value in grid[row][:column_count]:
                if isinstance(value, int) and 0 <= value < len(color_chars):
                    counts[value] = counts.get(value, 0) + 1
        inferred = max(counts, key=lambda value: (counts[value], -value)) if counts else 0
        background_values = {inferred}
    else:
        background_values = {color_chars.index(symbol) for symbol in background}
    background_symbols = "".join(
        color_chars[value] for value in sorted(background_values)
    )

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def is_foreground(row: int, col: int) -> bool:
        return (
            0 <= row < row_count
            and 0 <= col < column_count
            and row < len(grid)
            and col < len(grid[row])
            and isinstance(grid[row][col], int)
            and 0 <= grid[row][col] < len(color_chars)
            and grid[row][col] not in background_values
        )

    visited: set[tuple[int, int]] = set()
    objects: list[dict[str, Any]] = []
    object_count = 0
    sampled_cells = 0
    for start_row in range(row_count):
        for start_col in range(column_count):
            start = (start_row, start_col)
            if start in visited or not is_foreground(*start):
                continue
            include_object = len(objects) < object_limit
            object_count += 1
            stack = [start]
            visited.add(start)
            cells: list[tuple[int, int]] = []
            palette: dict[int, int] = {}
            row_sum = 0
            col_sum = 0
            min_row = max_row = start_row
            min_col = max_col = start_col
            touches_edge = False
            sampled: list[list[int]] = []
            while stack:
                row, col = stack.pop()
                value = grid[row][col]
                cells.append((row, col))
                palette[value] = palette.get(value, 0) + 1
                row_sum += row
                col_sum += col
                min_row = min(min_row, row)
                min_col = min(min_col, col)
                max_row = max(max_row, row)
                max_col = max(max_col, col)
                touches_edge = (
                    touches_edge
                    or row in {0, row_count - 1}
                    or col in {0, column_count - 1}
                )
                if include_object and sampled_cells < sample_limit:
                    sampled.append([row, col])
                    sampled_cells += 1
                for row_delta, col_delta in directions:
                    neighbor = (row + row_delta, col + col_delta)
                    if neighbor not in visited and is_foreground(*neighbor):
                        visited.add(neighbor)
                        stack.append(neighbor)

            if not include_object:
                continue
            size = len(cells)
            bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
            pattern = None
            if bbox_area <= 256:
                cell_set = set(cells)
                pattern = [
                    "".join(
                        color_chars[grid[row][col]] if (row, col) in cell_set else "."
                        for col in range(min_col, max_col + 1)
                    )
                    for row in range(min_row, max_row + 1)
                ]
            objects.append(
                {
                    "id": object_count - 1,
                    "signature": hashlib.sha1(
                        repr(
                            sorted(
                                (
                                    row - min_row,
                                    col - min_col,
                                    color_chars[grid[row][col]],
                                )
                                for row, col in cells
                            )
                        ).encode("utf-8")
                    ).hexdigest()[:16],
                    "size": size,
                    "bbox": [min_row, min_col, max_row, max_col],
                    "centroid": [row_sum / size, col_sum / size],
                    "touches_edge": touches_edge,
                    "colors": {
                        color_chars[value]: palette[value] for value in sorted(palette)
                    },
                    "pattern": pattern,
                    "truncated_pattern": pattern is None,
                    "cells": sampled,
                    "truncated_cells": max(0, size - len(sampled)),
                }
            )

    return {
        "background": background_symbols,
        "background_inferred": background is None,
        "connectivity": 8 if diagonal else 4,
        "count": object_count,
        "objects": objects,
        "sampled_cells": sampled_cells,
        "truncated_objects": max(0, object_count - len(objects)),
    }


def _bounded_frame_object_relations(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    background: str | None = None,
    diagonal: bool = False,
    object_limit: int = 32,
    relation_limit: int = 64,
) -> dict[str, Any]:
    """Describe bounded pairwise spatial and visual relations between objects."""
    for value, name, maximum in (
        (object_limit, "object_limit", 64),
        (relation_limit, "relation_limit", 256),
    ):
        if isinstance(value, bool):
            raise TypeError(
                f"frame.object_relations(..., {name}=...) expects an integer."
            )
        try:
            parsed = max(0, min(maximum, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.object_relations(..., {name}=...) expects an integer."
            ) from exc
        if name == "object_limit":
            bounded_object_limit = parsed
        else:
            bounded_relation_limit = parsed

    object_result = _bounded_frame_objects(
        grid,
        shape=shape,
        color_chars=color_chars,
        background=background,
        diagonal=diagonal,
        limit=bounded_object_limit,
        cell_limit=0,
    )
    objects = object_result["objects"]

    def summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("id", "signature", "size", "bbox", "centroid", "colors")
        }

    relations: list[dict[str, Any]] = []
    relation_count = 0
    counts = {
        "same_signature": 0,
        "same_size": 0,
        "row_overlap": 0,
        "column_overlap": 0,
        "bbox_contains": 0,
    }
    for index, first in enumerate(objects):
        for second in objects[index + 1 :]:
            relation_count += 1
            first_box = first["bbox"]
            second_box = second["bbox"]
            row_overlap = max(
                0,
                min(first_box[2], second_box[2])
                - max(first_box[0], second_box[0])
                + 1,
            )
            column_overlap = max(
                0,
                min(first_box[3], second_box[3])
                - max(first_box[1], second_box[1])
                + 1,
            )
            row_gap = max(
                0,
                second_box[0] - first_box[2] - 1,
                first_box[0] - second_box[2] - 1,
            )
            column_gap = max(
                0,
                second_box[1] - first_box[3] - 1,
                first_box[1] - second_box[3] - 1,
            )
            vertical = (
                "ABOVE"
                if first_box[2] < second_box[0]
                else "BELOW"
                if second_box[2] < first_box[0]
                else "OVERLAP"
            )
            horizontal = (
                "LEFT"
                if first_box[3] < second_box[1]
                else "RIGHT"
                if second_box[3] < first_box[1]
                else "OVERLAP"
            )
            first_contains = (
                first_box[0] <= second_box[0]
                and first_box[1] <= second_box[1]
                and first_box[2] >= second_box[2]
                and first_box[3] >= second_box[3]
            )
            second_contains = (
                second_box[0] <= first_box[0]
                and second_box[1] <= first_box[1]
                and second_box[2] >= first_box[2]
                and second_box[3] >= first_box[3]
            )
            containment = (
                "FIRST_CONTAINS_SECOND"
                if first_contains
                else "SECOND_CONTAINS_FIRST"
                if second_contains
                else None
            )
            same_signature = first["signature"] == second["signature"]
            same_size = first["size"] == second["size"]
            counts["same_signature"] += same_signature
            counts["same_size"] += same_size
            counts["row_overlap"] += row_overlap > 0
            counts["column_overlap"] += column_overlap > 0
            counts["bbox_contains"] += containment is not None
            if len(relations) < bounded_relation_limit:
                relations.append(
                    {
                        "first_id": first["id"],
                        "second_id": second["id"],
                        "vertical": vertical,
                        "horizontal": horizontal,
                        "row_overlap": row_overlap,
                        "column_overlap": column_overlap,
                        "row_gap": row_gap,
                        "column_gap": column_gap,
                        "bbox_gap": row_gap + column_gap,
                        "containment": containment,
                        "centroid_delta": [
                            second["centroid"][0] - first["centroid"][0],
                            second["centroid"][1] - first["centroid"][1],
                        ],
                        "same_size": same_size,
                        "same_signature": same_signature,
                        "shared_colors": sorted(
                            set(first["colors"]) & set(second["colors"])
                        ),
                    }
                )

    return {
        "background": object_result["background"],
        "background_inferred": object_result["background_inferred"],
        "connectivity": object_result["connectivity"],
        "object_count": object_result["count"],
        "objects": [summary(item) for item in objects],
        "truncated_objects": object_result["truncated_objects"],
        "complete_object_set": object_result["truncated_objects"] == 0,
        "relation_count": relation_count,
        "relations": relations,
        "counts": counts,
        "truncated_relations": max(0, relation_count - len(relations)),
    }


def _bounded_frame_object_changes(
    before_grid: list[list[Any]],
    after_grid: list[list[Any]],
    *,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    color_chars: str,
    background: str | None = None,
    diagonal: bool = False,
    limit: int = 64,
) -> dict[str, Any]:
    """Match translation-invariant multi-color objects across two frames."""
    if not isinstance(diagonal, bool):
        raise TypeError("frame.track_objects(..., diagonal=...) expects a boolean.")
    if isinstance(limit, bool):
        raise TypeError("frame.track_objects(..., limit=...) expects an integer.")
    try:
        event_limit = max(0, min(256, int(limit)))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "frame.track_objects(..., limit=...) expects an integer."
        ) from exc

    inferred_background = background is None
    if inferred_background:
        counts: dict[int, int] = {}
        for grid, shape in (
            (before_grid, before_shape),
            (after_grid, after_shape),
        ):
            row_count, column_count = shape
            for row in range(min(row_count, len(grid))):
                for value in grid[row][:column_count]:
                    if isinstance(value, int) and 0 <= value < len(color_chars):
                        counts[value] = counts.get(value, 0) + 1
        inferred = max(counts, key=lambda value: (counts[value], -value)) if counts else 0
        shared_background = color_chars[inferred]
    else:
        shared_background = background

    before = _bounded_frame_objects(
        before_grid,
        shape=before_shape,
        color_chars=color_chars,
        background=shared_background,
        diagonal=diagonal,
        limit=128,
        cell_limit=0,
    )
    after = _bounded_frame_objects(
        after_grid,
        shape=after_shape,
        color_chars=color_chars,
        background=shared_background,
        diagonal=diagonal,
        limit=128,
        cell_limit=0,
    )

    before_by_signature: dict[str, list[dict[str, Any]]] = {}
    after_by_signature: dict[str, list[dict[str, Any]]] = {}
    for item in before["objects"]:
        before_by_signature.setdefault(item["signature"], []).append(item)
    for item in after["objects"]:
        after_by_signature.setdefault(item["signature"], []).append(item)

    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    removed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    all_signatures = sorted(set(before_by_signature) | set(after_by_signature))
    for signature in all_signatures:
        old_items = before_by_signature.get(signature, [])
        new_items = after_by_signature.get(signature, [])
        candidates = sorted(
            (
                abs(old["centroid"][0] - new["centroid"][0])
                + abs(old["centroid"][1] - new["centroid"][1]),
                old["id"],
                new["id"],
                old,
                new,
            )
            for old in old_items
            for new in new_items
        )
        used_old: set[int] = set()
        used_new: set[int] = set()
        for _, old_id, new_id, old, new in candidates:
            if old_id not in used_old and new_id not in used_new:
                used_old.add(old_id)
                used_new.add(new_id)
                matched.append((old, new))
        removed.extend(item for item in old_items if item["id"] not in used_old)
        added.extend(item for item in new_items if item["id"] not in used_new)

    def summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("id", "signature", "size", "bbox", "centroid", "colors")
        }

    events: list[dict[str, Any]] = []
    moved_count = 0
    unchanged_count = 0
    for old, new in sorted(matched, key=lambda pair: pair[0]["id"]):
        delta = [
            new["bbox"][0] - old["bbox"][0],
            new["bbox"][1] - old["bbox"][1],
        ]
        event_type = "unchanged" if delta == [0, 0] else "moved"
        moved_count += event_type == "moved"
        unchanged_count += event_type == "unchanged"
        events.append(
            {
                "type": event_type,
                "delta": delta,
                "distance": abs(delta[0]) + abs(delta[1]),
                "before": summary(old),
                "after": summary(new),
            }
        )
    events.extend({"type": "removed", "before": summary(item)} for item in removed)
    events.extend({"type": "added", "after": summary(item)} for item in added)

    counts = {
        "matched": len(matched),
        "moved": moved_count,
        "unchanged": unchanged_count,
        "added": len(added),
        "removed": len(removed),
    }
    return {
        "background": shared_background,
        "background_inferred": inferred_background,
        "connectivity": 8 if diagonal else 4,
        "before_count": before["count"],
        "after_count": after["count"],
        "counts": counts,
        "events": events[:event_limit],
        "truncated_events": max(0, len(events) - event_limit),
        "truncated_input": {
            "before": before["truncated_objects"],
            "after": after["truncated_objects"],
        },
    }


def _bounded_frame_transform_relation(
    before_grid: list[list[Any]],
    after_grid: list[list[Any]],
    *,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    color_chars: str,
    allow_recolor: bool = True,
) -> dict[str, Any]:
    """Score D4 transforms from a prior frame to the current frame."""
    if not isinstance(allow_recolor, bool):
        raise TypeError(
            "frame.transform_relation(..., allow_recolor=...) expects a boolean."
        )

    def materialize(
        grid: list[list[Any]], shape: tuple[int, int]
    ) -> tuple[tuple[Any, ...], ...]:
        row_count, column_count = shape
        return tuple(
            tuple(
                grid[row][col]
                if row < len(grid) and col < len(grid[row])
                else None
                for col in range(column_count)
            )
            for row in range(row_count)
        )

    def rotate(matrix: tuple[tuple[Any, ...], ...]) -> tuple[tuple[Any, ...], ...]:
        return tuple(tuple(row) for row in zip(*matrix[::-1])) if matrix else ()

    def flip_horizontal(
        matrix: tuple[tuple[Any, ...], ...],
    ) -> tuple[tuple[Any, ...], ...]:
        return tuple(tuple(reversed(row)) for row in matrix)

    source = materialize(before_grid, before_shape)
    target = materialize(after_grid, after_shape)
    transformed: list[tuple[str, tuple[tuple[Any, ...], ...]]] = [
        ("IDENTITY", source)
    ]
    rotated = source
    for name in ("ROTATE_90", "ROTATE_180", "ROTATE_270"):
        rotated = rotate(rotated)
        transformed.append((name, rotated))
    reflected = flip_horizontal(source)
    transformed.append(("FLIP_HORIZONTAL", reflected))
    for name in (
        "FLIP_HORIZONTAL_ROTATE_90",
        "FLIP_HORIZONTAL_ROTATE_180",
        "FLIP_HORIZONTAL_ROTATE_270",
    ):
        reflected = rotate(reflected)
        transformed.append((name, reflected))

    target_rows, target_cols = after_shape
    cell_count = target_rows * target_cols

    def symbol(value: Any) -> str:
        if isinstance(value, int) and 0 <= value < len(color_chars):
            return color_chars[value]
        return "?"

    candidates: list[dict[str, Any]] = []
    for name, matrix in transformed:
        matrix_shape = (len(matrix), len(matrix[0]) if matrix else 0)
        compatible = matrix_shape == after_shape
        candidate: dict[str, Any] = {
            "transform": name,
            "shape": list(matrix_shape),
            "compatible_shape": compatible,
        }
        if not compatible:
            candidates.append(candidate)
            continue

        exact_mismatches = sum(
            source_value != target[row][col]
            for row, source_row in enumerate(matrix)
            for col, source_value in enumerate(source_row)
        )
        candidate.update(
            {
                "exact_mismatches": exact_mismatches,
                "exact_match_ratio": (
                    (cell_count - exact_mismatches) / cell_count
                    if cell_count
                    else 1.0
                ),
            }
        )
        if allow_recolor:
            frequencies: dict[Any, dict[Any, int]] = {}
            for row, source_row in enumerate(matrix):
                for col, source_value in enumerate(source_row):
                    target_value = target[row][col]
                    counts = frequencies.setdefault(source_value, {})
                    counts[target_value] = counts.get(target_value, 0) + 1
            mapping: dict[Any, Any] = {}
            recolor_mismatches = 0
            for source_value, counts in frequencies.items():
                chosen = max(counts, key=lambda value: (counts[value], -int(value or 0)))
                mapping[source_value] = chosen
                recolor_mismatches += sum(counts.values()) - counts[chosen]
            candidate.update(
                {
                    "recolor_mismatches": recolor_mismatches,
                    "recolor_match_ratio": (
                        (cell_count - recolor_mismatches) / cell_count
                        if cell_count
                        else 1.0
                    ),
                    "color_map": {
                        symbol(source_value): symbol(target_value)
                        for source_value, target_value in sorted(
                            mapping.items(), key=lambda item: str(item[0])
                        )
                    },
                }
            )
        candidates.append(candidate)

    compatible_candidates = [
        candidate for candidate in candidates if candidate["compatible_shape"]
    ]
    def score(candidate: dict[str, Any]) -> tuple[int, ...]:
        if allow_recolor:
            return (
                candidate["recolor_mismatches"],
                candidate["exact_mismatches"],
            )
        return (candidate["exact_mismatches"],)

    best_score = (
        min(score(candidate) for candidate in compatible_candidates)
        if compatible_candidates
        else None
    )
    return {
        "before_shape": list(before_shape),
        "after_shape": list(after_shape),
        "allow_recolor": allow_recolor,
        "exact_matches": [
            candidate["transform"]
            for candidate in compatible_candidates
            if candidate["exact_mismatches"] == 0
        ],
        "recolor_matches": (
            [
                candidate["transform"]
                for candidate in compatible_candidates
                if candidate["recolor_mismatches"] == 0
            ]
            if allow_recolor
            else []
        ),
        "best": [
            candidate["transform"]
            for candidate in compatible_candidates
            if score(candidate) == best_score
        ],
        "candidates": candidates,
    }


def _bounded_frame_symmetry(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    sample_limit: int = 8,
) -> dict[str, Any]:
    """Score exact and approximate reflection and rotational frame symmetries."""
    if isinstance(sample_limit, bool):
        raise TypeError("frame.symmetry(..., sample_limit=...) expects an integer.")
    try:
        mismatch_limit = max(0, min(32, int(sample_limit)))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "frame.symmetry(..., sample_limit=...) expects an integer."
        ) from exc

    row_count, column_count = shape

    def cell(row: int, col: int) -> Any:
        if not (0 <= row < row_count and 0 <= col < column_count):
            return None
        if row >= len(grid) or col >= len(grid[row]):
            return None
        value = grid[row][col]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return None
        return value

    def symbol(value: Any) -> str:
        return color_chars[value] if value is not None else "?"

    square = row_count == column_count
    transforms: list[tuple[str, bool, Any]] = [
        ("HORIZONTAL_AXIS", True, lambda row, col: (row_count - 1 - row, col)),
        ("VERTICAL_AXIS", True, lambda row, col: (row, column_count - 1 - col)),
        (
            "ROTATE_180",
            True,
            lambda row, col: (row_count - 1 - row, column_count - 1 - col),
        ),
        ("ROTATE_90", square, lambda row, col: (row_count - 1 - col, row)),
        ("ROTATE_270", square, lambda row, col: (col, column_count - 1 - row)),
        ("MAIN_DIAGONAL", square, lambda row, col: (col, row)),
        (
            "ANTI_DIAGONAL",
            square,
            lambda row, col: (row_count - 1 - col, column_count - 1 - row),
        ),
    ]
    cell_count = row_count * column_count
    candidates: list[dict[str, Any]] = []
    for name, compatible, coordinate in transforms:
        candidate: dict[str, Any] = {
            "symmetry": name,
            "compatible_shape": compatible,
        }
        if not compatible:
            candidates.append(candidate)
            continue
        mismatched_cells = 0
        mismatch_samples: list[dict[str, Any]] = []
        for row in range(row_count):
            for col in range(column_count):
                other_row, other_col = coordinate(row, col)
                value = cell(row, col)
                other_value = cell(other_row, other_col)
                if value == other_value:
                    continue
                mismatched_cells += 1
                if len(mismatch_samples) < mismatch_limit:
                    mismatch_samples.append(
                        {
                            "cell": [row, col],
                            "counterpart": [other_row, other_col],
                            "actual": symbol(value),
                            "expected": symbol(other_value),
                        }
                    )
        candidate.update(
            {
                "mismatched_cells": mismatched_cells,
                "match_ratio": (
                    (cell_count - mismatched_cells) / cell_count
                    if cell_count
                    else 1.0
                ),
                "mismatches": mismatch_samples,
                "truncated_mismatches": max(
                    0, mismatched_cells - len(mismatch_samples)
                ),
            }
        )
        candidates.append(candidate)

    compatible_candidates = [
        candidate for candidate in candidates if candidate["compatible_shape"]
    ]
    best_score = min(
        (candidate["mismatched_cells"] for candidate in compatible_candidates),
        default=None,
    )
    return {
        "shape": list(shape),
        "symmetries": [
            candidate["symmetry"]
            for candidate in compatible_candidates
            if candidate["mismatched_cells"] == 0
        ],
        "best": [
            candidate["symmetry"]
            for candidate in compatible_candidates
            if candidate["mismatched_cells"] == best_score
        ],
        "candidates": candidates,
    }


def _bounded_frame_periodicity(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    candidate_limit: int = 8,
    sample_limit: int = 8,
) -> dict[str, Any]:
    """Score exact and approximate finite row and column periods."""
    parsed_limits: dict[str, int] = {}
    for value, name in (
        (candidate_limit, "candidate_limit"),
        (sample_limit, "sample_limit"),
    ):
        if isinstance(value, bool):
            raise TypeError(f"frame.periodicity(..., {name}=...) expects an integer.")
        try:
            parsed_limits[name] = max(0, min(32, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.periodicity(..., {name}=...) expects an integer."
            ) from exc
    result_limit = parsed_limits["candidate_limit"]
    mismatch_limit = parsed_limits["sample_limit"]
    row_count, column_count = shape

    def cell(row: int, col: int) -> Any:
        if not (0 <= row < row_count and 0 <= col < column_count):
            return None
        if row >= len(grid) or col >= len(grid[row]):
            return None
        value = grid[row][col]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return None
        return value

    def symbol(value: Any) -> str:
        return color_chars[value] if value is not None else "?"

    def analyze(axis: str, length: int) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for period in range(1, length):
            comparisons = (
                (row_count - period) * column_count
                if axis == "rows"
                else row_count * (column_count - period)
            )
            mismatches = 0
            samples: list[dict[str, Any]] = []
            row_range = range(period, row_count) if axis == "rows" else range(row_count)
            col_range = (
                range(column_count)
                if axis == "rows"
                else range(period, column_count)
            )
            for row in row_range:
                for col in col_range:
                    other_row = row - period if axis == "rows" else row
                    other_col = col if axis == "rows" else col - period
                    value = cell(row, col)
                    expected = cell(other_row, other_col)
                    if value == expected:
                        continue
                    mismatches += 1
                    if len(samples) < mismatch_limit:
                        samples.append(
                            {
                                "cell": [row, col],
                                "counterpart": [other_row, other_col],
                                "actual": symbol(value),
                                "expected": symbol(expected),
                            }
                        )
            candidates.append(
                {
                    "period": period,
                    "complete_tiles": length % period == 0,
                    "comparisons": comparisons,
                    "mismatches": mismatches,
                    "match_ratio": (
                        (comparisons - mismatches) / comparisons
                        if comparisons
                        else 1.0
                    ),
                    "mismatch_samples": samples,
                    "truncated_mismatches": max(0, mismatches - len(samples)),
                }
            )

        exact_periods = [
            candidate["period"]
            for candidate in candidates
            if candidate["mismatches"] == 0
        ]

        def score(candidate: dict[str, Any]) -> tuple[float, int]:
            comparisons = candidate["comparisons"]
            mismatch_ratio = (
                candidate["mismatches"] / comparisons if comparisons else 0.0
            )
            return mismatch_ratio, candidate["period"]

        ranked = sorted(candidates, key=score)
        return {
            "exact_periods": exact_periods,
            "fundamental_period": min(exact_periods, default=None),
            "best_period": ranked[0]["period"] if ranked else None,
            "scanned_candidates": len(candidates),
            "candidates": ranked[:result_limit],
            "truncated_candidates": max(0, len(ranked) - result_limit),
        }

    return {
        "shape": list(shape),
        "rows": analyze("rows", row_count),
        "columns": analyze("columns", column_count),
    }


def _bounded_frame_find_pattern(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    pattern: Any,
    wildcard: str | None = None,
    transforms: bool = False,
    max_mismatches: int = 0,
    mismatch_limit: int = 8,
    limit: int = 64,
) -> dict[str, Any]:
    """Find bounded exact or approximate patterns, optionally under D4 transforms."""
    if isinstance(pattern, (str, bytes)) or not isinstance(pattern, (list, tuple)):
        raise TypeError("frame.find_pattern(pattern) expects a non-empty list of rows.")
    if not pattern:
        raise ValueError("frame.find_pattern(pattern) expects at least one row.")
    if wildcard is not None and (
        not isinstance(wildcard, str)
        or len(wildcard) != 1
        or wildcard in color_chars
    ):
        raise ValueError(
            "frame.find_pattern(..., wildcard=...) expects one non-color character."
        )
    if not isinstance(transforms, bool):
        raise TypeError("frame.find_pattern(..., transforms=...) expects a boolean.")
    parsed_limits: dict[str, int] = {}
    for value, name, upper_bound in (
        (max_mismatches, "max_mismatches", 256),
        (mismatch_limit, "mismatch_limit", 32),
        (limit, "limit", 256),
    ):
        if isinstance(value, bool):
            raise TypeError(
                f"frame.find_pattern(..., {name}=...) expects an integer."
            )
        try:
            parsed_limits[name] = max(0, min(upper_bound, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.find_pattern(..., {name}=...) expects an integer."
            ) from exc
    allowed_mismatches = parsed_limits["max_mismatches"]
    mismatch_sample_limit = parsed_limits["mismatch_limit"]
    match_limit = parsed_limits["limit"]

    normalized_rows: list[tuple[str, ...]] = []
    width: int | None = None
    for row in pattern:
        if isinstance(row, str):
            values = tuple(row)
        elif isinstance(row, (list, tuple)):
            values = tuple(row)
        else:
            raise TypeError(
                "frame.find_pattern(pattern) rows must be strings or symbol lists."
            )
        if not values:
            raise ValueError("frame.find_pattern(pattern) rows cannot be empty.")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError("frame.find_pattern(pattern) rows must have equal lengths.")
        for value in values:
            if not isinstance(value, str) or len(value) != 1:
                raise TypeError("Pattern cells must be single-character symbols.")
            if value != wildcard and value not in color_chars:
                raise ValueError(
                    f"Pattern cells must use color symbols from {color_chars!r}"
                    " or the configured wildcard."
                )
        normalized_rows.append(values)

    cell_count = len(normalized_rows) * int(width or 0)
    if cell_count > 256:
        raise ValueError("frame.find_pattern(pattern) accepts at most 256 cells.")
    allowed_mismatches = min(allowed_mismatches, cell_count)

    identity = tuple(normalized_rows)

    def rotate(matrix: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(row) for row in zip(*matrix[::-1]))

    def flip_horizontal(
        matrix: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(reversed(row)) for row in matrix)

    candidates: list[tuple[str, tuple[tuple[str, ...], ...]]] = [
        ("IDENTITY", identity)
    ]
    if transforms:
        rotated = identity
        for name in ("ROTATE_90", "ROTATE_180", "ROTATE_270"):
            rotated = rotate(rotated)
            candidates.append((name, rotated))
        reflected = flip_horizontal(identity)
        candidates.append(("FLIP_HORIZONTAL", reflected))
        for name in (
            "FLIP_HORIZONTAL_ROTATE_90",
            "FLIP_HORIZONTAL_ROTATE_180",
            "FLIP_HORIZONTAL_ROTATE_270",
        ):
            reflected = rotate(reflected)
            candidates.append((name, reflected))

    variants: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for name, matrix in candidates:
        if matrix not in seen:
            seen.add(matrix)
            variants.append((name, matrix))

    row_count, column_count = shape
    ranked_matches: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    match_count = 0
    for variant_index, (transform, matrix) in enumerate(variants):
        pattern_rows = len(matrix)
        pattern_cols = len(matrix[0])
        for top in range(max(0, row_count - pattern_rows + 1)):
            for left in range(max(0, column_count - pattern_cols + 1)):
                mismatch_count = 0
                mismatch_samples: list[dict[str, Any]] = []
                for row_offset, pattern_row in enumerate(matrix):
                    grid_row = top + row_offset
                    for col_offset, symbol in enumerate(pattern_row):
                        grid_col = left + col_offset
                        if symbol == wildcard:
                            continue
                        actual_value = (
                            grid[grid_row][grid_col]
                            if grid_row < len(grid) and grid_col < len(grid[grid_row])
                            else None
                        )
                        expected_value = color_chars.index(symbol)
                        if actual_value != expected_value:
                            mismatch_count += 1
                            if len(mismatch_samples) < mismatch_sample_limit:
                                actual_symbol = (
                                    color_chars[actual_value]
                                    if isinstance(actual_value, int)
                                    and not isinstance(actual_value, bool)
                                    and 0 <= actual_value < len(color_chars)
                                    else None
                                )
                                mismatch_samples.append(
                                    {
                                        "cell": [grid_row, grid_col],
                                        "actual": actual_symbol,
                                        "expected": symbol,
                                    }
                                )
                        if mismatch_count > allowed_mismatches:
                            break
                    if mismatch_count > allowed_mismatches:
                        break
                if mismatch_count > allowed_mismatches:
                    continue
                match_count += 1
                match = {
                    "top": top,
                    "left": left,
                    "bottom": top + pattern_rows,
                    "right": left + pattern_cols,
                    "transform": transform,
                }
                if allowed_mismatches:
                    match["mismatches"] = mismatch_count
                    match["mismatch_samples"] = mismatch_samples
                    match["truncated_mismatches"] = max(
                        0, mismatch_count - len(mismatch_samples)
                    )
                if match_limit:
                    ranked_matches.append(
                        ((mismatch_count, variant_index, top, left), match)
                    )
                    ranked_matches.sort(key=lambda item: item[0])
                    if len(ranked_matches) > match_limit:
                        ranked_matches.pop()

    matches = [match for _rank, match in ranked_matches]

    return {
        "count": match_count,
        "matches": matches,
        "variants": [name for name, _matrix in variants],
        "truncated_matches": max(0, match_count - len(matches)),
    }


def _bounded_reachable_region(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    start: Any,
    passable: Any,
    diagonal: bool = False,
    max_nodes: int = 4096,
    cell_limit: int = 128,
    frontier_limit: int = 64,
) -> dict[str, Any]:
    """Run bounded BFS and summarize reachable space plus its blocked frontier."""
    row_count, column_count = shape
    if not isinstance(start, (list, tuple)) or len(start) != 2:
        raise TypeError("frame.reachable_region(start, ...) expects [row, col].")
    start_row, start_col = start
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (start_row, start_col)
    ):
        raise TypeError(
            "frame.reachable_region(start, ...) expects integer coordinates."
        )
    if not (0 <= start_row < row_count and 0 <= start_col < column_count):
        raise ValueError("frame.reachable_region(start, ...) is outside the frame.")
    if isinstance(passable, str):
        symbols = list(passable)
    elif isinstance(passable, (list, tuple, set)):
        symbols = list(passable)
    else:
        raise TypeError(
            "frame.reachable_region(..., passable=...) expects color symbols."
        )
    if not symbols or any(
        not isinstance(symbol, str)
        or len(symbol) != 1
        or symbol not in color_chars
        for symbol in symbols
    ):
        raise ValueError(
            "frame.reachable_region(..., passable=...) expects valid color symbols."
        )
    if not isinstance(diagonal, bool):
        raise TypeError(
            "frame.reachable_region(..., diagonal=...) expects a boolean."
        )

    def bounded_integer(value: Any, name: str, maximum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(
                f"frame.reachable_region(..., {name}=...) expects an integer."
            )
        try:
            return max(0, min(maximum, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.reachable_region(..., {name}=...) expects an integer."
            ) from exc

    node_limit = bounded_integer(max_nodes, "max_nodes", 16384)
    returned_cell_limit = bounded_integer(cell_limit, "cell_limit", 512)
    returned_frontier_limit = bounded_integer(frontier_limit, "frontier_limit", 256)
    passable_values = {color_chars.index(symbol) for symbol in symbols}
    offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    if diagonal:
        offsets.extend([(-1, 1), (1, 1), (1, -1), (-1, -1)])

    def cell(row: int, col: int) -> int | None:
        if not (0 <= row < row_count and 0 <= col < column_count):
            return None
        if row >= len(grid) or col >= len(grid[row]):
            return -1
        value = grid[row][col]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return -1
        return value

    start_cell = (start_row, start_col)
    queue = [start_cell]
    distances = {start_cell: 0}
    cursor = 0
    explored = 0
    while cursor < len(queue) and explored < node_limit:
        row, col = queue[cursor]
        cursor += 1
        explored += 1
        for row_delta, col_delta in offsets:
            neighbor = (row + row_delta, col + col_delta)
            if neighbor in distances:
                continue
            value = cell(*neighbor)
            if value not in passable_values:
                continue
            distances[neighbor] = distances[(row, col)] + 1
            queue.append(neighbor)

    reachable = list(distances)
    min_row = min((row for row, _col in reachable), default=None)
    min_col = min((col for _row, col in reachable), default=None)
    max_row = max((row for row, _col in reachable), default=None)
    max_col = max((col for _row, col in reachable), default=None)
    maximum_distance = max(distances.values(), default=0)
    farthest = [
        list(coordinate)
        for coordinate, distance in distances.items()
        if distance == maximum_distance
    ]

    frontier_coordinates: set[tuple[int, int]] = set()
    for row, col in reachable:
        for row_delta, col_delta in offsets:
            neighbor = (row + row_delta, col + col_delta)
            neighbor_value = cell(*neighbor)
            if (
                neighbor_value is not None
                and neighbor_value not in passable_values
                and neighbor not in distances
            ):
                frontier_coordinates.add(neighbor)
    frontier_colors: dict[str, int] = {}
    frontier: list[dict[str, Any]] = []
    for row, col in sorted(frontier_coordinates):
        value = cell(row, col)
        color_symbol = color_chars[value] if value != -1 else "?"
        frontier_colors[color_symbol] = frontier_colors.get(color_symbol, 0) + 1
        if len(frontier) < returned_frontier_limit:
            frontier.append({"cell": [row, col], "symbol": color_symbol})

    returned_cells = queue[:returned_cell_limit]
    returned_farthest = farthest[:32]
    return {
        "start": list(start_cell),
        "passable": sorted(set(symbols), key=color_chars.index),
        "connectivity": 8 if diagonal else 4,
        "reachable_count": len(reachable),
        "bbox": (
            [min_row, min_col, max_row, max_col]
            if min_row is not None
            else None
        ),
        "touches_edge": any(
            row in {0, row_count - 1} or col in {0, column_count - 1}
            for row, col in reachable
        ),
        "maximum_distance": maximum_distance,
        "farthest_cells": returned_farthest,
        "truncated_farthest_cells": max(0, len(farthest) - len(returned_farthest)),
        "cells": [list(coordinate) for coordinate in returned_cells],
        "truncated_cells": max(0, len(reachable) - len(returned_cells)),
        "frontier_count": len(frontier_coordinates),
        "frontier_colors": frontier_colors,
        "frontier": frontier,
        "truncated_frontier": max(0, len(frontier_coordinates) - len(frontier)),
        "explored": explored,
        "search_truncated": cursor < len(queue),
    }


def _bounded_shortest_path(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    start: Any,
    goal: Any,
    passable: Any,
    diagonal: bool = False,
    max_nodes: int = 4096,
    path_limit: int = 128,
) -> dict[str, Any]:
    """Run bounded BFS over caller-selected letter-coded passable cells."""
    row_count, column_count = shape

    def coordinate(value: Any, name: str) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(f"frame.shortest_path(..., {name}=...) expects [row, col].")
        row, col = value
        if any(isinstance(item, bool) or not isinstance(item, int) for item in (row, col)):
            raise TypeError(f"frame.shortest_path(..., {name}=...) expects integer coordinates.")
        if row < 0 or row >= row_count or col < 0 or col >= column_count:
            raise ValueError(f"frame.shortest_path(..., {name}=...) is outside the frame.")
        return row, col

    start_cell = coordinate(start, "start")

    def looks_like_coordinate(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                not isinstance(item, bool) and isinstance(item, int)
                for item in value
            )
        )

    target_symbols: list[str] = []
    if isinstance(goal, str) or (
        isinstance(goal, (list, tuple, set))
        and bool(goal)
        and all(isinstance(item, str) for item in goal)
    ):
        raw_target_symbols = list(goal)
        if not raw_target_symbols:
            raise ValueError(
                "frame.shortest_path(..., goal=...) requires at least one target symbol."
            )
        if any(
            len(symbol) != 1 or symbol not in color_chars
            for symbol in raw_target_symbols
        ):
            raise ValueError(
                "frame.shortest_path(..., goal=...) target symbols must come from "
                f"{color_chars!r}."
            )
        target_symbols = list(dict.fromkeys(raw_target_symbols))
        raw_goals = []
    elif looks_like_coordinate(goal):
        raw_goals = [goal]
    elif isinstance(goal, (list, tuple)):
        raw_goals = list(goal)
        if not raw_goals:
            raise ValueError("frame.shortest_path(...) requires at least one goal.")
        if len(raw_goals) > 64:
            raise ValueError("frame.shortest_path(...) is limited to 64 goals.")
    else:
        raise TypeError(
            "frame.shortest_path(..., goal=...) expects color symbols, [row, col], "
            "or a list of coordinates."
        )

    requested_goals = []
    goal_cells = set()
    for index, raw_goal in enumerate(raw_goals):
        goal_cell = coordinate(raw_goal, f"goals[{index}]")
        if goal_cell not in goal_cells:
            requested_goals.append(goal_cell)
            goal_cells.add(goal_cell)
    if isinstance(passable, str):
        symbols = list(passable)
    elif isinstance(passable, (list, tuple, set)):
        symbols = list(passable)
    else:
        raise TypeError(
            "frame.shortest_path(..., passable=...) expects color symbols."
        )
    if not symbols or any(
        not isinstance(symbol, str)
        or len(symbol) != 1
        or symbol not in color_chars
        for symbol in symbols
    ):
        raise ValueError(
            f"frame.shortest_path(..., passable=...) expects symbols from {color_chars!r}."
        )
    if not isinstance(diagonal, bool):
        raise TypeError("frame.shortest_path(..., diagonal=...) expects a boolean.")

    def bounded_integer(value: Any, name: str, default_cap: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"frame.shortest_path(..., {name}=...) expects an integer.")
        try:
            return max(0, min(default_cap, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.shortest_path(..., {name}=...) expects an integer."
            ) from exc

    node_limit = bounded_integer(max_nodes, "max_nodes", 16384)
    returned_path_limit = bounded_integer(path_limit, "path_limit", 512)
    passable_values = {color_chars.index(symbol) for symbol in symbols}
    target_values = {color_chars.index(symbol) for symbol in target_symbols}
    offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    if diagonal:
        offsets.extend([(-1, 1), (1, 1), (1, -1), (-1, -1)])

    queue = [start_cell]
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    cursor = 0
    explored = 0
    found = False
    reached_goal = None
    while cursor < len(queue) and explored < node_limit:
        current = queue[cursor]
        cursor += 1
        explored += 1
        current_row = grid[current[0]] if current[0] < len(grid) else []
        try:
            current_value = int(current_row[current[1]])
        except (IndexError, TypeError, ValueError):
            current_value = None
        if current in goal_cells or current_value in target_values:
            found = True
            reached_goal = current
            break
        for row_delta, col_delta in offsets:
            neighbor = (current[0] + row_delta, current[1] + col_delta)
            row, col = neighbor
            if (
                row < 0
                or row >= row_count
                or col < 0
                or col >= column_count
                or neighbor in parents
            ):
                continue
            source_row = grid[row] if row < len(grid) else []
            try:
                cell_value = int(source_row[col])
            except (IndexError, TypeError, ValueError):
                continue
            if (
                neighbor not in goal_cells
                and cell_value not in target_values
                and cell_value not in passable_values
            ):
                continue
            parents[neighbor] = current
            queue.append(neighbor)

    path: list[tuple[int, int]] = []
    if reached_goal is not None:
        current = reached_goal
        while current is not None:
            path.append(current)
            current = parents[current]
        path.reverse()

    direction_names = {
        (-1, 0): "UP",
        (0, 1): "RIGHT",
        (1, 0): "DOWN",
        (0, -1): "LEFT",
        (-1, 1): "UP_RIGHT",
        (1, 1): "DOWN_RIGHT",
        (1, -1): "DOWN_LEFT",
        (-1, -1): "UP_LEFT",
    }
    returned_path = path[: returned_path_limit + 1]
    moves = [
        direction_names[
            (
                returned_path[index][0] - returned_path[index - 1][0],
                returned_path[index][1] - returned_path[index - 1][1],
            )
        ]
        for index in range(1, len(returned_path))
    ]
    reported_goal = (
        reached_goal
        if reached_goal is not None
        else requested_goals[0] if len(requested_goals) == 1 else None
    )
    return {
        "found": found,
        "start": list(start_cell),
        "goal": list(reported_goal) if reported_goal is not None else None,
        "goals": [list(cell) for cell in requested_goals],
        "target_symbols": target_symbols,
        "selected_goal": list(reached_goal) if reached_goal is not None else None,
        "distance": len(path) - 1 if found else None,
        "path": [list(cell) for cell in returned_path],
        "moves": moves,
        "next_step": list(path[1]) if len(path) > 1 else None,
        "explored": explored,
        "search_truncated": not found and cursor < len(queue),
        "path_truncated": found and len(returned_path) < len(path),
    }


def _bounded_frame_color_transitions(
    before_grid: list[list[Any]],
    after_grid: list[list[Any]],
    *,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    color_chars: str,
    include_unchanged: bool = False,
    limit: int = 64,
    cell_limit: int = 128,
) -> dict[str, Any]:
    """Aggregate per-cell color transitions between two frames."""
    if not isinstance(include_unchanged, bool):
        raise TypeError(
            "frame.color_transitions(..., include_unchanged=...) expects a boolean."
        )

    def bounded_integer(value: Any, name: str, maximum: int) -> int:
        if isinstance(value, bool):
            raise TypeError(
                f"frame.color_transitions(..., {name}=...) expects an integer."
            )
        try:
            return max(0, min(maximum, int(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"frame.color_transitions(..., {name}=...) expects an integer."
            ) from exc

    transition_limit = bounded_integer(limit, "limit", 128)
    sample_limit = bounded_integer(cell_limit, "cell_limit", 256)
    row_count = max(before_shape[0], after_shape[0])
    column_count = max(before_shape[1], after_shape[1])

    def cell(
        grid: list[list[Any]], shape: tuple[int, int], row: int, col: int
    ) -> Any:
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            return None
        if row >= len(grid) or col >= len(grid[row]):
            return None
        value = grid[row][col]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(color_chars)
        ):
            return -1
        return value

    def symbol(value: Any) -> str | None:
        if value is None:
            return None
        return color_chars[value] if value != -1 else "?"

    def value_order(value: Any) -> int:
        return -2 if value is None else int(value)

    stats: dict[tuple[Any, Any], dict[str, Any]] = {}
    source_targets: dict[Any, dict[Any, int]] = {}
    changed_cells = 0
    unchanged_cells = 0
    for row in range(row_count):
        for col in range(column_count):
            before = cell(before_grid, before_shape, row, col)
            after = cell(after_grid, after_shape, row, col)
            changed = before != after
            changed_cells += changed
            unchanged_cells += not changed
            if not changed and not include_unchanged:
                continue
            key = (before, after)
            entry = stats.setdefault(
                key,
                {
                    "count": 0,
                    "min_row": row,
                    "min_col": col,
                    "max_row": row,
                    "max_col": col,
                },
            )
            entry["count"] += 1
            entry["min_row"] = min(entry["min_row"], row)
            entry["min_col"] = min(entry["min_col"], col)
            entry["max_row"] = max(entry["max_row"], row)
            entry["max_col"] = max(entry["max_col"], col)
            targets = source_targets.setdefault(before, {})
            targets[after] = targets.get(after, 0) + 1

    ranked_keys = sorted(
        stats,
        key=lambda key: (
            -stats[key]["count"],
            value_order(key[0]),
            value_order(key[1]),
        ),
    )
    selected_keys = ranked_keys[:transition_limit]
    selected_set = set(selected_keys)
    samples: dict[tuple[Any, Any], list[list[int]]] = {
        key: [] for key in selected_keys
    }
    sampled_cells = 0
    for row in range(row_count):
        for col in range(column_count):
            if sampled_cells >= sample_limit:
                break
            key = (
                cell(before_grid, before_shape, row, col),
                cell(after_grid, after_shape, row, col),
            )
            if key in selected_set and (
                include_unchanged or key[0] != key[1]
            ):
                samples[key].append([row, col])
                sampled_cells += 1
        if sampled_cells >= sample_limit:
            break

    transitions: list[dict[str, Any]] = []
    for key in selected_keys:
        entry = stats[key]
        transition_samples = samples[key]
        transitions.append(
            {
                "before": symbol(key[0]),
                "after": symbol(key[1]),
                "count": entry["count"],
                "bbox": [
                    entry["min_row"],
                    entry["min_col"],
                    entry["max_row"],
                    entry["max_col"],
                ],
                "cells": transition_samples,
                "truncated_cells": max(
                    0, entry["count"] - len(transition_samples)
                ),
            }
        )

    source_mappings: list[dict[str, Any]] = []
    for source, targets in sorted(
        source_targets.items(), key=lambda item: value_order(item[0])
    ):
        dominant = max(
            targets,
            key=lambda target: (targets[target], -value_order(target)),
        )
        source_mappings.append(
            {
                "source": symbol(source),
                "targets": [
                    {"symbol": symbol(target), "count": count}
                    for target, count in sorted(
                        targets.items(), key=lambda item: value_order(item[0])
                    )
                ],
                "dominant_target": symbol(dominant),
                "deterministic": len(targets) == 1,
            }
        )

    return {
        "before_shape": list(before_shape),
        "after_shape": list(after_shape),
        "shape_changed": before_shape != after_shape,
        "include_unchanged": include_unchanged,
        "changed_cells": changed_cells,
        "unchanged_cells": unchanged_cells,
        "transition_types": len(ranked_keys),
        "transitions": transitions,
        "source_mappings": source_mappings,
        "sampled_cells": sampled_cells,
        "truncated_transitions": max(0, len(ranked_keys) - len(transitions)),
    }


def _bounded_frame_diff_summary(
    before_grid: list[list[Any]],
    after_grid: list[list[Any]],
    *,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    before_level: int,
    after_level: int,
    color_chars: str,
    limit: int = 64,
) -> dict[str, Any]:
    """Build the public, bounded frame-diff value used inside the sandbox."""
    try:
        change_limit = max(0, min(128, int(limit)))
    except (TypeError, ValueError) as exc:
        raise TypeError("frame.diff(..., limit=...) expects an integer limit.") from exc

    def cell(grid: list[list[Any]], row: int, col: int) -> Any:
        if row < 0 or row >= len(grid):
            return None
        values = grid[row]
        if col < 0 or col >= len(values):
            return None
        return values[col]

    def color_symbol(value: Any) -> str | None:
        if value is None:
            return None
        try:
            color_index = int(value)
        except (TypeError, ValueError):
            return "?"
        if 0 <= color_index < len(color_chars):
            return color_chars[color_index]
        return "?"

    row_count = max(before_shape[0], after_shape[0])
    column_count = max(before_shape[1], after_shape[1])
    changed_count = 0
    changed_samples: list[dict[str, Any]] = []
    min_row = min_col = None
    max_row = max_col = None

    for row in range(row_count):
        for col in range(column_count):
            before_value = cell(before_grid, row, col)
            after_value = cell(after_grid, row, col)
            if before_value == after_value:
                continue
            changed_count += 1
            min_row = row if min_row is None else min(min_row, row)
            min_col = col if min_col is None else min(min_col, col)
            max_row = row if max_row is None else max(max_row, row)
            max_col = col if max_col is None else max(max_col, col)
            if len(changed_samples) < change_limit:
                changed_samples.append(
                    {
                        "row": row,
                        "col": col,
                        "before": color_symbol(before_value),
                        "after": color_symbol(after_value),
                    }
                )

    return {
        "before_level": before_level,
        "after_level": after_level,
        "level_changed": before_level != after_level,
        "before_shape": list(before_shape),
        "after_shape": list(after_shape),
        "shape_changed": before_shape != after_shape,
        "changed_cells": changed_count,
        "changed_bbox": (
            [min_row, min_col, max_row, max_col] if min_row is not None else None
        ),
        "changes": changed_samples,
        "truncated_changes": max(0, changed_count - len(changed_samples)),
    }


def _sandbox_exception_diagnostic(
    exc: BaseException,
    code: str,
    *,
    allowed_modules: set[str] | None = None,
) -> dict[str, Any]:
    """Build compact, structured repair context for generated-code failures."""
    line_number = getattr(exc, "lineno", None)
    column = getattr(exc, "offset", None)
    if line_number is None:
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        if user_frames:
            line_number = user_frames[-1].lineno

    try:
        parsed_line = int(line_number) if line_number is not None else None
    except (TypeError, ValueError):
        parsed_line = None
    try:
        parsed_column = int(column) if column is not None else None
    except (TypeError, ValueError):
        parsed_column = None

    source = ""
    code_lines = str(code or "").splitlines()
    if parsed_line is not None and 1 <= parsed_line <= len(code_lines):
        source = code_lines[parsed_line - 1].strip()[:240]
    elif isinstance(exc, SyntaxError) and getattr(exc, "text", None):
        source = str(exc.text).strip()[:240]

    context = [
        {
            "line": index + 1,
            "source": code_lines[index].strip()[:240],
            "current": index + 1 == parsed_line,
        }
        for index in range(
            max(0, (parsed_line or 1) - 2),
            min(len(code_lines), (parsed_line or 1) + 1),
        )
    ]
    details: dict[str, Any] = {}
    suggestions: list[str] = []
    error_type = exc.__class__.__name__[:80]
    if isinstance(exc, SyntaxError):
        hint = "Fix the Python syntax at the reported line and retry."
    elif isinstance(exc, NameError):
        missing_name = str(getattr(exc, "name", "") or "")[:120]
        if missing_name:
            details["name"] = missing_name
            available_names: set[str] = set()
            current_traceback = exc.__traceback__
            while current_traceback is not None:
                frame = current_traceback.tb_frame
                if frame.f_code.co_filename == "<python_tool>":
                    available_names.update(
                        name
                        for name in (*frame.f_globals, *frame.f_locals)
                        if isinstance(name, str) and not name.startswith("_")
                    )
                current_traceback = current_traceback.tb_next
            suggestions = difflib.get_close_matches(
                missing_name,
                sorted(available_names),
                n=3,
                cutoff=0.5,
            )
        hint = (
            "Define or import the missing name in this call; every Python call "
            "starts fresh."
        )
    elif isinstance(exc, ImportError):
        modules = ", ".join(sorted(allowed_modules or set()))
        hint = f"Use only approved modules: {modules}." if modules else "Use an approved module."
    elif isinstance(exc, AttributeError):
        attribute = str(getattr(exc, "name", "") or "")[:120]
        object_type = type(getattr(exc, "obj", None)).__name__[:80]
        if attribute:
            details["attribute"] = attribute
        if object_type:
            details["object_type"] = object_type
        documented_attributes = {
            "FrameView": [
                "ascii",
                "cell",
                "color_transitions",
                "components",
                "crop",
                "diff",
                "enclosed_regions",
                "find",
                "find_pattern",
                "level",
                "neighbors",
                "object_relations",
                "objects",
                "periodicity",
                "ray",
                "reachable_region",
                "rectangles",
                "runs",
                "segmentation",
                "shape",
                "shortest_path",
                "shortest_path_to_any",
                "step",
                "symmetry",
                "track_objects",
                "transform_relation",
            ],
            "TransitionView": [
                "action",
                "after_frame",
                "before_frame",
                "color_transitions",
                "diff",
                "frame",
                "result",
            ],
        }
        suggestions = difflib.get_close_matches(
            attribute,
            documented_attributes.get(object_type, []),
            n=3,
            cutoff=0.45,
        )
        hint = (
            "Use the closest documented sandbox attribute and retry."
            if suggestions
            else "Check the documented sandbox object attributes and method names."
        )
    elif isinstance(exc, KeyError):
        if exc.args and isinstance(exc.args[0], (str, int, float, bool, type(None))):
            details["key"] = exc.args[0]
        hint = "Check available mapping keys or use `.get(...)` for optional values."
    elif isinstance(exc, IndexError):
        hint = "Check frame shape and sequence bounds before indexing."
    elif isinstance(exc, TypeError):
        hint = "Check argument types, method signatures, and keyword-only parameters."
    elif isinstance(exc, ValueError):
        hint = "Check the reported API constraint and retry with a valid value."
    else:
        hint = "Inspect the reported source line, correct the code, and retry."

    return {
        "type": error_type,
        "line": parsed_line,
        "column": parsed_column,
        "end_line": getattr(exc, "end_lineno", None),
        "end_column": getattr(exc, "end_offset", None),
        "source": source or None,
        "context": context,
        "suggestions": suggestions,
        "hint": hint[:300],
        "retry": "correct_and_retry",
        **details,
    }


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import ast
    import builtins
    import copy
    import contextlib
    import difflib
    import io
    import json
    import os
    import sys
    import traceback
    import types
    from typing import Any

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__
    __FRAME_CROP_SOURCE__
    __FRAME_CELL_SOURCE__
    __FRAME_NEIGHBORS_SOURCE__
    __FRAME_RAY_SOURCE__
    __FRAME_FIND_SOURCE__
    __FRAME_COLOR_SUMMARY_SOURCE__
    __FRAME_SPATIAL_SOURCE__
    __FRAME_LAYOUT_SOURCE__
    __FRAME_RUNS_SOURCE__
    __FRAME_RECTANGLES_SOURCE__
    __FRAME_ENCLOSED_REGIONS_SOURCE__
    __FRAME_COMPONENTS_SOURCE__
    __FRAME_OBJECTS_SOURCE__
    __FRAME_OBJECT_RELATIONS_SOURCE__
    __FRAME_OBJECT_CHANGES_SOURCE__
    __FRAME_TRANSFORM_RELATION_SOURCE__
    __FRAME_SYMMETRY_SOURCE__
    __FRAME_PERIODICITY_SOURCE__
    __FRAME_PATTERN_SOURCE__
    __REACHABLE_REGION_SOURCE__
    __SHORTEST_PATH_SOURCE__
    __FRAME_COLOR_TRANSITIONS_SOURCE__
    __FRAME_DIFF_SOURCE__
    __EXCEPTION_DIAGNOSTIC_SOURCE__

    HOST_STDOUT = sys.stdout
    MAX_STDOUT_CHARS = 32_768
    MAX_RESULT_DEPTH = 8
    MAX_RESULT_ITEMS = 512
    MAX_RESULT_STRING_CHARS = 4_096

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hasattr",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "ImportError",
        "IndexError",
        "KeyError",
        "LookupError",
        "NameError",
        "NotImplementedError",
        "OSError",
        "StopAsyncIteration",
        "StopIteration",
        "TimeoutError",
        "TypeError",
        "UnicodeError",
        "ValueError",
        "ZeroDivisionError",
        "RuntimeError",
        "zip",
    }
    SAFE_MODULE_CACHE = {}


    class BoundedStringIO(io.StringIO):
        def __init__(self, max_chars=MAX_STDOUT_CHARS):
            super().__init__()
            self.max_chars = max(0, int(max_chars))
            self.omitted_chars = 0

        def write(self, value):
            if not isinstance(value, str):
                raise TypeError("string argument expected")
            remaining = max(0, self.max_chars - super().tell())
            if remaining:
                super().write(value[:remaining])
            self.omitted_chars += max(0, len(value) - remaining)
            return len(value)

        def getvalue(self):
            rendered = super().getvalue()
            if self.omitted_chars:
                rendered += f"\n... [truncated {self.omitted_chars} stdout chars]"
            return rendered


    class SafeModule:
        # Expose public non-module attributes from an approved module.

        def __init__(self, module):
            object.__setattr__(self, "_module", module)

        def __getattribute__(self, name):
            if str(name).startswith("_"):
                raise AttributeError("Private module attributes are not allowed.")
            module = object.__getattribute__(self, "_module")
            value = getattr(module, name)
            if isinstance(value, types.ModuleType):
                raise AttributeError(f"Module-valued attribute '{name}' is not allowed.")
            return value


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    def _freeze_cache_value(value):
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key), _freeze_cache_value(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(_freeze_cache_value(item) for item in value)
        if isinstance(value, set):
            return tuple(
                sorted(
                    (_freeze_cache_value(item) for item in value),
                    key=repr,
                )
            )
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return repr(value)


    def _cached_frame_analysis(method):
        def wrapped(self, *args, **kwargs):
            key = (
                method.__name__,
                _freeze_cache_value(args),
                _freeze_cache_value(kwargs),
            )
            if key not in self._analysis_cache:
                if len(self._analysis_cache) >= 64:
                    self._analysis_cache.pop(next(iter(self._analysis_cache)))
                self._analysis_cache[key] = method(self, *args, **kwargs)
            return copy.deepcopy(self._analysis_cache[key])

        wrapped.__name__ = method.__name__
        wrapped.__doc__ = method.__doc__
        return wrapped


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None
            self._analysis_cache = {}

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def color_transitions(
            self,
            other,
            *,
            include_unchanged=False,
            limit=64,
            cell_limit=128,
        ):
            if not isinstance(other, FrameView):
                raise TypeError(
                    "frame.color_transitions(other) expects another FrameView."
                )
            return _bounded_frame_color_transitions(
                other._grid,
                self._grid,
                before_shape=other.shape,
                after_shape=self.shape,
                color_chars=COLOR_CHARS,
                include_unchanged=include_unchanged,
                limit=limit,
                cell_limit=cell_limit,
            )

        def diff(self, other, *, limit=64):
            # Return a bounded summary of changes from other to this frame.
            if not isinstance(other, FrameView):
                raise TypeError("frame.diff(other) expects another frame.")
            return _bounded_frame_diff_summary(
                other._grid,
                self._grid,
                before_shape=other.shape,
                after_shape=self.shape,
                before_level=other.level,
                after_level=self.level,
                color_chars=COLOR_CHARS,
                limit=limit,
            )

        def crop(self, top, left, bottom, right):
            # Coordinates are half-open; output is clipped and letter-coded.
            return _bounded_frame_crop(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                top=top,
                left=left,
                bottom=bottom,
                right=right,
            )

        def cell(self, row, col):
            return _frame_cell_symbol(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                row=row,
                col=col,
            )

        def neighbors(self, row, col, *, diagonal=False):
            return _bounded_frame_neighbors(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                row=row,
                col=col,
                diagonal=diagonal,
            )

        def ray(
            self,
            row,
            col,
            direction,
            *,
            stop_at=None,
            include_start=False,
            limit=64,
        ):
            return _bounded_frame_ray(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                row=row,
                col=col,
                direction=direction,
                stop_at=stop_at,
                include_start=include_start,
                limit=limit,
            )

        def find(self, symbol, *, limit=64):
            # Return a bounded coordinate sample plus total count and bounds.
            return _bounded_frame_find(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                symbol=symbol,
                limit=limit,
            )

        def analyze(self, calls):
            # Run several bounded frame queries through one compact call.
            if not isinstance(calls, (list, tuple)):
                raise TypeError("frame.analyze(calls) expects a list of query objects.")
            if len(calls) > 32:
                raise ValueError("frame.analyze(calls) is limited to 32 queries.")
            results = []
            for index, call in enumerate(calls):
                if not isinstance(call, dict):
                    raise TypeError(f"frame.analyze(calls)[{index}] must be an object.")
                name = str(call.get("method") or call.get("name") or "").strip()
                if not name or name.startswith("_") or name == "analyze":
                    raise ValueError(f"frame.analyze(calls)[{index}] has an invalid method.")
                method = getattr(self, name, None)
                if not callable(method):
                    raise ValueError(f"Unknown frame analysis method: {name}")
                args = call.get("args") or []
                kwargs = call.get("kwargs") or {}
                if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
                    raise TypeError(f"frame.analyze(calls)[{index}] args/kwargs are invalid.")
                results.append({"method": name, "result": method(*args, **kwargs)})
            return results

        @_cached_frame_analysis
        def color_summary(self, *, limit=16):
            return _bounded_frame_color_summary(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                limit=limit,
            )

        @_cached_frame_analysis
        def bounds(self, symbols, *, limit=64):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="bounds", symbols=symbols, limit=limit,
            )

        @_cached_frame_analysis
        def region_summary(self, bounds):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="region_summary", bounds=bounds,
            )

        @_cached_frame_analysis
        def row_profile(self, symbol=None, *, limit=64):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="row_profile", symbol=symbol, limit=limit,
            )

        @_cached_frame_analysis
        def column_profile(self, symbol=None, *, limit=64):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="column_profile", symbol=symbol, limit=limit,
            )

        @_cached_frame_analysis
        def nearest_cell(self, start, symbols, *, metric="manhattan", limit=64):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="nearest_cell", start=start, symbols=symbols,
                metric=metric, limit=limit,
            )

        def distance(self, start, end, *, metric="manhattan"):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="distance", start=start, end=end, metric=metric,
            )

        def line_between(self, start, end, *, limit=128):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="line_between", start=start, end=end, limit=limit,
            )

        def translate_cells(self, cells, delta, *, limit=128):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="translate_cells", cells=cells, delta=delta, limit=limit,
            )

        def mirror_cells(self, cells, symmetry, *, limit=128):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="mirror_cells", cells=cells, symmetry=symmetry, limit=limit,
            )

        @_cached_frame_analysis
        def compare_regions(self, first, second, *, allow_recolor=True):
            return _bounded_frame_spatial_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="compare_regions", first=first, second=second,
                allow_recolor=allow_recolor,
            )

        @_cached_frame_analysis
        def border_summary(self, *, thickness=1):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="border_summary", thickness=thickness,
            )

        @_cached_frame_analysis
        def corner_summary(self, *, size=1):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="corner_summary", size=size,
            )

        @_cached_frame_analysis
        def center_summary(self, *, radius=0):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="center_summary", radius=radius,
            )

        @_cached_frame_analysis
        def quadrant_summary(self):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="quadrant_summary",
            )

        @_cached_frame_analysis
        def color_adjacency(self, *, diagonal=False, limit=64):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="color_adjacency", diagonal=diagonal, limit=limit,
            )

        @_cached_frame_analysis
        def distance_between_colors(self, first, second, *, metric="manhattan"):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="distance_between_colors", first=first, second=second,
                metric=metric,
            )

        @_cached_frame_analysis
        def divider_lines(self, symbol=None, *, limit=64):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="divider_lines", symbol=symbol, limit=limit,
            )

        @_cached_frame_analysis
        def panels(self, symbol=None, *, limit=64):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="panels", symbol=symbol, limit=limit,
            )

        @_cached_frame_analysis
        def tile_summary(self, tile_height, tile_width, *, limit=64):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="tile_summary", tile_height=tile_height,
                tile_width=tile_width, limit=limit,
            )

        @_cached_frame_analysis
        def edge_distance(self, symbols, *, limit=64):
            return _bounded_frame_layout_operation(
                self._grid, shape=self.shape, color_chars=COLOR_CHARS,
                operation="edge_distance", symbols=symbols, limit=limit,
            )

        @_cached_frame_analysis
        def runs(self, symbol=None, *, directions="HV", min_length=2, limit=64):
            return _bounded_frame_runs(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                symbol=symbol,
                directions=directions,
                min_length=min_length,
                limit=limit,
            )

        @_cached_frame_analysis
        def rectangles(self, symbol=None, *, kind="any", min_size=2, limit=64):
            return _bounded_frame_rectangles(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                symbol=symbol,
                kind=kind,
                min_size=min_size,
                limit=limit,
            )

        @_cached_frame_analysis
        def enclosed_regions(
            self, symbol=None, *, diagonal=False, limit=64, cell_limit=128
        ):
            return _bounded_frame_enclosed_regions(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                symbol=symbol,
                diagonal=diagonal,
                limit=limit,
                cell_limit=cell_limit,
            )

        @_cached_frame_analysis
        def components(self, symbol, *, diagonal=False, limit=64, cell_limit=128):
            return _bounded_frame_components(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                symbol=symbol,
                diagonal=diagonal,
                limit=limit,
                cell_limit=cell_limit,
            )

        @_cached_frame_analysis
        def objects(self, *, background=None, diagonal=False, limit=64, cell_limit=128):
            return _bounded_frame_objects(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                background=background,
                diagonal=diagonal,
                limit=limit,
                cell_limit=cell_limit,
            )

        @_cached_frame_analysis
        def object_relations(
            self,
            *,
            background=None,
            diagonal=False,
            object_limit=32,
            relation_limit=64,
        ):
            return _bounded_frame_object_relations(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                background=background,
                diagonal=diagonal,
                object_limit=object_limit,
                relation_limit=relation_limit,
            )

        def track_objects(self, other, *, background=None, diagonal=False, limit=64):
            if not isinstance(other, FrameView):
                raise TypeError("frame.track_objects(other) expects another FrameView.")
            return _bounded_frame_object_changes(
                other._grid,
                self._grid,
                before_shape=other.shape,
                after_shape=self.shape,
                color_chars=COLOR_CHARS,
                background=background,
                diagonal=diagonal,
                limit=limit,
            )

        def transform_relation(self, other, *, allow_recolor=True):
            if not isinstance(other, FrameView):
                raise TypeError(
                    "frame.transform_relation(other) expects another FrameView."
                )
            return _bounded_frame_transform_relation(
                other._grid,
                self._grid,
                before_shape=other.shape,
                after_shape=self.shape,
                color_chars=COLOR_CHARS,
                allow_recolor=allow_recolor,
            )

        @_cached_frame_analysis
        def symmetry(self, *, sample_limit=8):
            return _bounded_frame_symmetry(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                sample_limit=sample_limit,
            )

        @_cached_frame_analysis
        def periodicity(self, *, candidate_limit=8, sample_limit=8):
            return _bounded_frame_periodicity(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                candidate_limit=candidate_limit,
                sample_limit=sample_limit,
            )

        @_cached_frame_analysis
        def find_pattern(
            self,
            pattern,
            *,
            wildcard=None,
            transforms=False,
            max_mismatches=0,
            mismatch_limit=8,
            limit=64,
        ):
            return _bounded_frame_find_pattern(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                pattern=pattern,
                wildcard=wildcard,
                transforms=transforms,
                max_mismatches=max_mismatches,
                mismatch_limit=mismatch_limit,
                limit=limit,
            )

        @_cached_frame_analysis
        def reachable_region(
            self,
            start,
            *,
            passable,
            diagonal=False,
            max_nodes=4096,
            cell_limit=128,
            frontier_limit=64,
        ):
            return _bounded_reachable_region(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                start=start,
                passable=passable,
                diagonal=diagonal,
                max_nodes=max_nodes,
                cell_limit=cell_limit,
                frontier_limit=frontier_limit,
            )

        def shortest_path(
            self,
            start,
            goal,
            *,
            passable,
            diagonal=False,
            max_nodes=4096,
            path_limit=128,
        ):
            # Search only caller-selected colors; the goal is always enterable.
            return _bounded_shortest_path(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                start=start,
                goal=goal,
                passable=passable,
                diagonal=diagonal,
                max_nodes=max_nodes,
                path_limit=path_limit,
            )

        def shortest_path_to_any(
            self,
            start,
            goals,
            *,
            passable,
            diagonal=False,
            max_nodes=4096,
            path_limit=128,
        ):
            return self.shortest_path(
                start,
                goals,
                passable=passable,
                diagonal=diagonal,
                max_nodes=max_nodes,
                path_limit=path_limit,
            )

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame):
            self.action = action
            self.frame = frame

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def diff(self, *, limit=64):
            if self.before_frame is None or self.after_frame is None:
                return None
            return self.after_frame.diff(self.before_frame, limit=limit)

        def color_transitions(
            self, *, include_unchanged=False, limit=64, cell_limit=128
        ):
            if self.before_frame is None or self.after_frame is None:
                return None
            return self.after_frame.color_transitions(
                self.before_frame,
                include_unchanged=include_unchanged,
                limit=limit,
                cell_limit=cell_limit,
            )

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _frame_ascii(grid):
        if not grid:
            return "(empty grid)"
        return "\n".join(
            "".join(COLOR_CHARS[max(0, min(15, int(value)))] for value in row)
            for row in grid
        )


    def _frame_from_delta(payload, previous_frame):
        if not isinstance(payload, dict) or not isinstance(previous_frame, FrameView):
            return None
        shape = tuple(payload.get("shape", previous_frame.shape))
        if shape != previous_frame.shape:
            return None
        grid = [list(row) for row in previous_frame._grid]
        for change in payload.get("changes", []):
            if not isinstance(change, (list, tuple)) or len(change) != 3:
                return None
            row, col, value = change
            if (
                isinstance(row, bool)
                or isinstance(col, bool)
                or isinstance(value, bool)
                or not isinstance(row, int)
                or not isinstance(col, int)
                or not isinstance(value, int)
                or row < 0
                or row >= len(grid)
                or col < 0
                or col >= len(grid[row])
                or value < 0
                or value >= len(COLOR_CHARS)
            ):
                return None
            grid[row][col] = value
        return FrameView(
            ascii=_frame_ascii(grid),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=shape,
            grid=grid,
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            previous_frame = items[-1].frame if items else None
            frame = _frame_from_payload(entry.get("frame"))
            if frame is None:
                frame = _frame_from_delta(entry.get("frame_delta"), previous_frame)
            if frame is None:
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=frame,
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result={},
                )
            )
        if transitions and isinstance(last_action_result, dict):
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _json_safe(value, _depth=0, _budget=None, _seen=None):
        if _budget is None:
            _budget = [MAX_RESULT_ITEMS]
        if _seen is None:
            _seen = set()
        if isinstance(value, str):
            if len(value) <= MAX_RESULT_STRING_CHARS:
                return value
            omitted = len(value) - MAX_RESULT_STRING_CHARS
            return value[:MAX_RESULT_STRING_CHARS] + f"... [truncated {omitted} chars]"
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if _depth >= MAX_RESULT_DEPTH:
            return "... [maximum result depth reached]"
        if isinstance(value, (dict, list, tuple, set)):
            identity = id(value)
            if identity in _seen:
                return "... [cyclic reference]"
            _seen.add(identity)
            try:
                if isinstance(value, dict):
                    rendered = {}
                    for key, item in value.items():
                        if _budget[0] <= 0:
                            rendered["... [truncated]"] = "result item limit reached"
                            break
                        _budget[0] -= 1
                        rendered[str(key)[:256]] = _json_safe(
                            item,
                            _depth + 1,
                            _budget,
                            _seen,
                        )
                    return rendered
                rendered = []
                for item in value:
                    if _budget[0] <= 0:
                        rendered.append("... [result item limit reached]")
                        break
                    _budget[0] -= 1
                    rendered.append(_json_safe(item, _depth + 1, _budget, _seen))
                return rendered
            finally:
                _seen.remove(identity)
        try:
            rendered = str(value)
        except Exception:
            rendered = f"<{value.__class__.__name__}>"
        return _json_safe(rendered, _depth, _budget, _seen)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        module_name = str(name or "")
        if level != 0 or module_name not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        module = builtins.__import__(name, globals, locals, (), level)
        for imported_name in fromlist or ():
            if (
                imported_name == "*"
                or str(imported_name).startswith("_")
                or not hasattr(module, imported_name)
                or isinstance(getattr(module, imported_name), types.ModuleType)
            ):
                raise ImportError(f"Name '{imported_name}' is not allowed from module '{name}'.")
        proxy = SAFE_MODULE_CACHE.get(module_name)
        if proxy is None:
            proxy = SafeModule(module)
            SAFE_MODULE_CACHE[module_name] = proxy
        return proxy


    def _safe_getattr(obj, name, *args):
        if isinstance(name, str) and name.startswith("_"):
            raise AttributeError("Private attribute access is not allowed.")
        if len(args) > 1:
            raise TypeError(
                f"getattr expected at most 3 arguments, got {2 + len(args)}"
            )
        try:
            return builtins.getattr(obj, name)
        except AttributeError:
            if args:
                return args[0]
            raise


    def _validate_user_code(code):
        # Reject Python object-graph escape primitives before execution.
        tree = ast.parse(code, filename="<python_tool>", mode="exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError(f"Private attribute access is not allowed: {node.attr}")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ValueError(f"Dunder names are not allowed: {node.id}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name.startswith("__"):
                raise ValueError(f"Dunder definitions are not allowed: {node.name}")
            if isinstance(node, ast.arg) and node.arg.startswith("__"):
                raise ValueError(f"Dunder argument names are not allowed: {node.arg}")
            if isinstance(node, ast.keyword) and node.arg is not None and node.arg.startswith("__"):
                raise ValueError(f"Dunder keyword arguments are not allowed: {node.arg}")
            if isinstance(node, ast.alias):
                bound_name = node.asname or node.name.split(".", 1)[0]
                if node.name.startswith("_") or bound_name.startswith("_"):
                    raise ValueError(f"Private imports are not allowed: {node.name}")
        return tree


    def _prepare_user_code(code):
        # Validate code and capture a notebook-style final expression once.
        tree = _validate_user_code(code)
        if not tree.body or not isinstance(tree.body[-1], ast.Expr):
            return tree
        final_expression = tree.body[-1]
        target = ast.copy_location(
            ast.Name(id="__tool_expression_result", ctx=ast.Store()),
            final_expression,
        )
        assignment = ast.copy_location(
            ast.Assign(targets=[target], value=final_expression.value),
            final_expression,
        )
        tree.body[-1] = assignment
        return ast.fix_missing_locations(tree)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_AS", None), 512 * 1024 * 1024),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        stdout = BoundedStringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import
        runtime_globals["__builtins__"]["getattr"] = _safe_getattr

        current_state_payload = dict(initial.get("state") or {})

        def _refresh_state(state_payload):
            history = _history_from_payload(state_payload.get("history"))
            current_payload = state_payload.get("current_frame")
            history_index = (
                current_payload.get("history_index")
                if isinstance(current_payload, dict)
                else None
            )
            if (
                not isinstance(history_index, bool)
                and isinstance(history_index, int)
                and 0 <= history_index < len(history)
            ):
                current_frame = history[history_index].frame
            elif isinstance(current_payload, dict) and "history_index" in current_payload:
                current_frame = None
            else:
                current_frame = _frame_from_payload(current_payload)
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result
            runtime_globals["experience"] = dict(state_payload.get("experience") or {})
            runtime_globals["strategy"] = dict(state_payload.get("strategy") or {})
            runtime_globals["memory"] = dict(state_payload.get("memory") or {})

        def _apply_state_update(update):
            nonlocal current_state_payload
            merged = dict(current_state_payload)
            history_append = update.get("history_append")
            if isinstance(history_append, list):
                merged["history"] = [*(merged.get("history") or []), *history_append]
            for key, value in update.items():
                if key != "history_append":
                    merged[key] = value
            current_state_payload = merged
            _refresh_state(current_state_payload)

        def action(actions):
            normalized_actions = _normalize_actions(actions)
            _send({"type": "action", "actions": normalized_actions})
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            if isinstance(reply.get("state_update"), dict):
                _apply_state_update(reply["state_update"])
            else:
                current_state_payload.clear()
                current_state_payload.update(reply.get("state") or {})
                _refresh_state(current_state_payload)
            return action_result

        def record_strategy(
            *,
            goal=None,
            hypothesis=None,
            evidence=None,
            confidence=None,
            open_question=None,
            next_test=None,
            test_action=None,
            expected_outcome=None,
            fallback=None,
            contradictions=None,
        ):
            update = {
                "goal": goal,
                "hypothesis": hypothesis,
                "evidence": evidence,
                "confidence": confidence,
                "open_question": open_question,
                "next_test": next_test,
                "test_action": test_action,
                "expected_outcome": expected_outcome,
                "fallback": fallback,
                "contradictions": contradictions,
            }
            _send({"type": "strategy", "update": _json_safe(update)})
            reply = _recv()
            if reply.get("type") != "strategy_result":
                raise RuntimeError("Invalid strategy response from sandbox host.")
            persisted = dict(reply.get("strategy") or {})
            runtime_globals["strategy"] = persisted
            return persisted

        def _persist_memory(updated):
            _send({"type": "memory", "memory": _json_safe(updated)})
            reply = _recv()
            if reply.get("type") == "memory_error":
                raise ValueError(str(reply.get("error", "memory update failed")))
            if reply.get("type") != "memory_result":
                raise RuntimeError("Invalid memory response from sandbox host.")
            persisted = dict(reply.get("memory") or {})
            runtime_globals["memory"] = persisted
            return persisted

        def remember(key, value):
            if not isinstance(key, str) or not key.strip():
                raise ValueError("remember(key, value) requires a non-empty string key.")
            updated = dict(runtime_globals.get("memory") or {})
            updated[key.strip()] = _json_safe(value)
            return _persist_memory(updated)

        def forget(key=None):
            updated = dict(runtime_globals.get("memory") or {})
            if key is None:
                updated = {}
            elif not isinstance(key, str) or not key.strip():
                raise ValueError("forget(key) requires a non-empty string key or None.")
            else:
                updated.pop(key.strip(), None)
            return _persist_memory(updated)

        runtime_globals["action"] = action
        runtime_globals["record_strategy"] = record_strategy
        runtime_globals["remember"] = remember
        runtime_globals["forget"] = forget
        _refresh_state(current_state_payload)

        try:
            parsed = _prepare_user_code(str(initial.get("code", "")))
            compiled = compile(parsed, "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(
                        runtime_globals.get(
                            "result",
                            runtime_globals.get("__tool_expression_result"),
                        )
                    ),
                    "action_results": _json_safe(action_results),
                }
            )
        except Exception as exc:
            code = str(initial.get("code", ""))
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "diagnostic": _sandbox_exception_diagnostic(
                        exc,
                        code,
                        allowed_modules=SAFE_MODULES,
                    ),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation)).replace(
    "__FRAME_CROP_SOURCE__\n", inspect.getsource(_bounded_frame_crop)
).replace(
    "__FRAME_CELL_SOURCE__\n", inspect.getsource(_frame_cell_symbol)
).replace(
    "__FRAME_NEIGHBORS_SOURCE__\n", inspect.getsource(_bounded_frame_neighbors)
).replace(
    "__FRAME_RAY_SOURCE__\n", inspect.getsource(_bounded_frame_ray)
).replace(
    "__FRAME_FIND_SOURCE__\n", inspect.getsource(_bounded_frame_find)
).replace(
    "__FRAME_COLOR_SUMMARY_SOURCE__\n",
    inspect.getsource(_bounded_frame_color_summary),
).replace(
    "__FRAME_SPATIAL_SOURCE__\n",
    inspect.getsource(_bounded_frame_spatial_operation),
).replace(
    "__FRAME_LAYOUT_SOURCE__\n",
    inspect.getsource(_bounded_frame_layout_operation),
).replace(
    "__FRAME_RUNS_SOURCE__\n", inspect.getsource(_bounded_frame_runs)
).replace(
    "__FRAME_RECTANGLES_SOURCE__\n", inspect.getsource(_bounded_frame_rectangles)
).replace(
    "__FRAME_ENCLOSED_REGIONS_SOURCE__\n",
    inspect.getsource(_bounded_frame_enclosed_regions),
).replace(
    "__FRAME_COMPONENTS_SOURCE__\n", inspect.getsource(_bounded_frame_components)
).replace(
    "__FRAME_OBJECTS_SOURCE__\n", inspect.getsource(_bounded_frame_objects)
).replace(
    "__FRAME_OBJECT_RELATIONS_SOURCE__\n",
    inspect.getsource(_bounded_frame_object_relations),
).replace(
    "__FRAME_OBJECT_CHANGES_SOURCE__\n",
    inspect.getsource(_bounded_frame_object_changes),
).replace(
    "__FRAME_TRANSFORM_RELATION_SOURCE__\n",
    inspect.getsource(_bounded_frame_transform_relation),
).replace(
    "__FRAME_SYMMETRY_SOURCE__\n", inspect.getsource(_bounded_frame_symmetry)
).replace(
    "__FRAME_PERIODICITY_SOURCE__\n", inspect.getsource(_bounded_frame_periodicity)
).replace(
    "__FRAME_PATTERN_SOURCE__\n", inspect.getsource(_bounded_frame_find_pattern)
).replace(
    "__REACHABLE_REGION_SOURCE__\n", inspect.getsource(_bounded_reachable_region)
).replace(
    "__SHORTEST_PATH_SOURCE__\n", inspect.getsource(_bounded_shortest_path)
).replace(
    "__FRAME_COLOR_TRANSITIONS_SOURCE__\n",
    inspect.getsource(_bounded_frame_color_transitions),
).replace(
    "__FRAME_DIFF_SOURCE__\n", inspect.getsource(_bounded_frame_diff_summary)
).replace(
    "__EXCEPTION_DIAGNOSTIC_SOURCE__\n",
    inspect.getsource(_sandbox_exception_diagnostic),
)

_SANDBOX_BOOTSTRAP_PAYLOAD = base64.b64encode(
    zlib.compress(
        marshal.dumps(
            compile(
                _SANDBOX_BOOTSTRAP,
                "<python_tool_sandbox_bootstrap>",
                "exec",
            )
        ),
        level=9,
    )
).decode("ascii")
_SANDBOX_LAUNCHER = (
    "import base64,marshal,sys,zlib;"
    "exec(marshal.loads(zlib.decompress(base64.b64decode(sys.stdin.readline()))))"
)


def _sanitize_host_error_text(text: str) -> str:
    if not str(text or "").strip():
        return "Sandbox process exited unexpectedly."
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> int:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    handle.write(encoded)
    handle.flush()
    return len(encoded.encode("utf-8"))


def _runtime_state_update(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Return an append-friendly update for sandbox runtime state."""
    update: dict[str, Any] = {}
    previous_history = previous.get("history")
    current_history = current.get("history")
    if isinstance(previous_history, list) and isinstance(current_history, list):
        prefix_length = len(previous_history)
        if current_history[:prefix_length] == previous_history:
            if len(current_history) > prefix_length:
                update["history_append"] = current_history[prefix_length:]
        elif current_history != previous_history:
            update["history"] = current_history
    elif current_history != previous_history:
        update["history"] = current_history

    for key, value in current.items():
        if key == "history":
            continue
        if previous.get(key) != value:
            update[key] = value
    return update


def _sandbox_command() -> tuple[list[str], str | None]:
    python_command = [sys.executable, "-I", "-S", "-c", _SANDBOX_LAUNCHER]
    bubblewrap = shutil.which("bwrap") if os.name == "posix" else None
    if bubblewrap is None:
        if _SANDBOX_REQUIRE_OS_ISOLATION:
            raise OSError(
                "LOCAL_ANALYZER_REQUIRE_OS_SANDBOX is enabled, but bubblewrap is unavailable."
            )
        return python_command, None
    return (
        [
            bubblewrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/tmp/work",
            "--chdir",
            "/tmp/work",
            "--",
            *python_command,
        ],
        "/tmp/work",
    )


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (AttributeError, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
        except OSError:
            return

        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            pass
    finally:
        for handle in (process.stdin, process.stdout, process.stderr):
            if handle is not None:
                handle.close()


class _PreparedSandboxProcess:
    """A bootstrapped process that is consumed by exactly one tool call."""

    def __init__(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="rgb_python_tool_warm_")
        command, self.isolated_cwd = _sandbox_command()
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=self.temp_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except BaseException:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.stdout_queue: queue.Queue[str | None] = queue.Queue()

        def stdout_reader() -> None:
            for raw_line in self.process.stdout:
                self.stdout_queue.put(raw_line)
            self.stdout_queue.put(None)

        threading.Thread(target=stdout_reader, daemon=True).start()
        self.process.stdin.write(_SANDBOX_BOOTSTRAP_PAYLOAD + "\n")
        self.process.stdin.flush()

    def close(self, *, terminate: bool = False) -> None:
        if terminate and self.process.poll() is None:
            _kill_process_group(self.process)
        _wait_for_process_exit(self.process)
        shutil.rmtree(self.temp_dir, ignore_errors=True)


try:
    _SANDBOX_PREWARM_WORKERS = max(
        0, int(os.environ.get("PYTHON_TOOL_PREWARM_WORKERS", "1") or 0)
    )
except ValueError:
    _SANDBOX_PREWARM_WORKERS = 1
_SANDBOX_PREWARM_QUEUE: queue.Queue[_PreparedSandboxProcess] = queue.Queue(
    maxsize=max(1, _SANDBOX_PREWARM_WORKERS)
)
_SANDBOX_PREWARM_LOCK = threading.Lock()
_SANDBOX_PREWARM_STARTING = 0


def prewarm_sandbox() -> None:
    """Fill the single-use worker queue without blocking the caller."""
    global _SANDBOX_PREWARM_STARTING
    if _SANDBOX_PREWARM_WORKERS <= 0:
        return
    with _SANDBOX_PREWARM_LOCK:
        missing = (
            _SANDBOX_PREWARM_WORKERS
            - _SANDBOX_PREWARM_QUEUE.qsize()
            - _SANDBOX_PREWARM_STARTING
        )
        _SANDBOX_PREWARM_STARTING += max(0, missing)

    def start_one() -> None:
        global _SANDBOX_PREWARM_STARTING
        worker: _PreparedSandboxProcess | None = None
        try:
            worker = _PreparedSandboxProcess()
            _SANDBOX_PREWARM_QUEUE.put_nowait(worker)
            worker = None
        except (OSError, queue.Full):
            pass
        finally:
            if worker is not None:
                worker.close(terminate=True)
            with _SANDBOX_PREWARM_LOCK:
                _SANDBOX_PREWARM_STARTING -= 1

    for _ in range(max(0, missing)):
        threading.Thread(target=start_one, daemon=True).start()


def _take_prepared_sandbox() -> _PreparedSandboxProcess | None:
    try:
        worker = _SANDBOX_PREWARM_QUEUE.get_nowait()
    except queue.Empty:
        return None
    if worker.process.poll() is not None:
        worker.close()
        prewarm_sandbox()
        return None
    prewarm_sandbox()
    return worker


def _close_prewarmed_sandboxes() -> None:
    deadline = time.monotonic() + 2.0
    while True:
        try:
            worker = _SANDBOX_PREWARM_QUEUE.get_nowait()
        except queue.Empty:
            with _SANDBOX_PREWARM_LOCK:
                starting = _SANDBOX_PREWARM_STARTING
            if starting <= 0 or time.monotonic() >= deadline:
                return
            time.sleep(0.01)
            continue
        worker.close(terminate=True)


atexit.register(_close_prewarmed_sandboxes)


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
    strategy_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    memory_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    transport_state = dict(initial_state)
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        host_strategy_updates: list[dict[str, Any]] = []
        prepared = _take_prepared_sandbox()
        if prepared is not None:
            process = prepared.process
            isolated_cwd = prepared.isolated_cwd
            sandbox_dir = prepared.temp_dir
            stdout_queue = prepared.stdout_queue
        else:
            try:
                command, isolated_cwd = _sandbox_command()
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    cwd=sandbox_dir,
                    env=_sandbox_env(),
                    start_new_session=True,
                )
            except OSError:
                return {
                    "error": (
                        "Sandbox process could not start because strict OS isolation is unavailable."
                        if _SANDBOX_REQUIRE_OS_ISOLATION
                        else "Sandbox process could not start."
                    ),
                    "stdout": "",
                    "action_results": [],
                }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        if prepared is None:
            stdout_queue = queue.Queue()

            def _stdout_reader() -> None:
                for raw_line in process.stdout:
                    stdout_queue.put(raw_line)
                stdout_queue.put(None)

            threading.Thread(target=_stdout_reader, daemon=True).start()
            process.stdin.write(_SANDBOX_BOOTSTRAP_PAYLOAD + "\n")

        def finish_process() -> None:
            _wait_for_process_exit(process)
            if prepared is not None:
                shutil.rmtree(prepared.temp_dir, ignore_errors=True)

        host_to_sandbox_bytes = 0

        def send_to_sandbox(payload: dict[str, Any]) -> None:
            nonlocal host_to_sandbox_bytes
            host_to_sandbox_bytes += _send_json_line(process.stdin, payload)

        def efficiency() -> dict[str, Any]:
            return {
                "prewarmed": prepared is not None,
                "host_to_sandbox_bytes": host_to_sandbox_bytes,
                "elapsed_seconds": time.monotonic() - started_at,
            }

        send_to_sandbox(
            {
                "code": code,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": isolated_cwd or sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            if should_stop is not None:
                try:
                    cancelled = bool(should_stop())
                except Exception:  # noqa: BLE001 - cancellation checks are advisory
                    cancelled = False
                if cancelled:
                    _kill_process_group(process)
                    finish_process()
                    return {
                        "error": "Tool execution cancelled by host.",
                        "diagnostic": {
                            "type": "CancelledError",
                            "line": None,
                            "column": None,
                            "source": None,
                            "hint": "The enclosing run requested cancellation.",
                            "retry": "do_not_retry",
                        },
                        "stdout": "",
                        "action_results": list(host_action_results),
                        "efficiency": efficiency(),
                    }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                finish_process()
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "diagnostic": {
                        "type": "TimeoutError",
                        "line": None,
                        "column": None,
                        "source": None,
                        "hint": (
                            "Reduce unbounded loops or search work, then retry with "
                            "a smaller bounded computation."
                        ),
                    },
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                poll_timeout = min(remaining, 0.1) if should_stop is not None else remaining
                line = stdout_queue.get(timeout=poll_timeout)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                finish_process()
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                finish_process()
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "action":
                try:
                    action_result_payload = action_handler(list(message.get("actions") or []))
                except SandboxHostActionError as exc:
                    send_to_sandbox(
                        {
                            "type": "action_error",
                            "error": str(exc) or "action failed in sandbox host.",
                        },
                    )
                    continue
                except Exception:  # noqa: BLE001
                    send_to_sandbox(
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                next_state = action_result_payload.get("state") or {}
                state_update = _runtime_state_update(transport_state, next_state)
                transport_state = dict(next_state)
                send_to_sandbox(
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state_update": state_update,
                    },
                )
                continue

            if msg_type == "strategy":
                if strategy_handler is None:
                    persisted_strategy: dict[str, Any] = {}
                else:
                    try:
                        persisted_strategy = strategy_handler(
                            dict(message.get("update") or {})
                        )
                    except Exception:  # noqa: BLE001
                        persisted_strategy = {}
                host_strategy_updates.append(dict(persisted_strategy))
                send_to_sandbox(
                    {
                        "type": "strategy_result",
                        "strategy": persisted_strategy,
                    },
                )
                continue

            if msg_type == "memory":
                if memory_handler is None:
                    persisted_memory: dict[str, Any] = {}
                else:
                    try:
                        persisted_memory = memory_handler(
                            dict(message.get("memory") or {})
                        )
                    except (TypeError, ValueError) as exc:
                        send_to_sandbox(
                            {
                                "type": "memory_error",
                                "error": str(exc) or "memory update failed.",
                            },
                        )
                        continue
                    except Exception:  # noqa: BLE001
                        send_to_sandbox(
                            {
                                "type": "memory_error",
                                "error": "memory update failed.",
                            },
                        )
                        continue
                send_to_sandbox(
                    {
                        "type": "memory_result",
                        "memory": persisted_memory,
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                finish_process()
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "diagnostic": message.get("diagnostic"),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "strategy_updates": list(host_strategy_updates),
                    "efficiency": efficiency(),
                }

            finish_process()
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
