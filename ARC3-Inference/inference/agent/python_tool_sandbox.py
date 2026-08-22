"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
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
from typing import Any, Callable

from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS


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


def _bounded_frame_find_pattern(
    grid: list[list[Any]],
    *,
    shape: tuple[int, int],
    color_chars: str,
    pattern: Any,
    wildcard: str | None = None,
    transforms: bool = False,
    limit: int = 64,
) -> dict[str, Any]:
    """Find bounded exact or wildcard pattern matches, optionally under D4 transforms."""
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
    if isinstance(limit, bool):
        raise TypeError("frame.find_pattern(..., limit=...) expects an integer.")
    try:
        match_limit = max(0, min(256, int(limit)))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "frame.find_pattern(..., limit=...) expects an integer."
        ) from exc

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
    matches: list[dict[str, Any]] = []
    match_count = 0
    for transform, matrix in variants:
        pattern_rows = len(matrix)
        pattern_cols = len(matrix[0])
        for top in range(max(0, row_count - pattern_rows + 1)):
            for left in range(max(0, column_count - pattern_cols + 1)):
                matched = True
                for row_offset, pattern_row in enumerate(matrix):
                    grid_row = top + row_offset
                    if grid_row >= len(grid):
                        matched = False
                        break
                    for col_offset, symbol in enumerate(pattern_row):
                        grid_col = left + col_offset
                        if symbol == wildcard:
                            continue
                        if (
                            grid_col >= len(grid[grid_row])
                            or grid[grid_row][grid_col] != color_chars.index(symbol)
                        ):
                            matched = False
                            break
                    if not matched:
                        break
                if not matched:
                    continue
                match_count += 1
                if len(matches) < match_limit:
                    matches.append(
                        {
                            "top": top,
                            "left": left,
                            "bottom": top + pattern_rows,
                            "right": left + pattern_cols,
                            "transform": transform,
                        }
                    )

    return {
        "count": match_count,
        "matches": matches,
        "variants": [name for name, _matrix in variants],
        "truncated_matches": max(0, match_count - len(matches)),
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

    if looks_like_coordinate(goal):
        raw_goals = [goal]
    elif isinstance(goal, (list, tuple)):
        raw_goals = list(goal)
        if not raw_goals:
            raise ValueError("frame.shortest_path(...) requires at least one goal.")
        if len(raw_goals) > 64:
            raise ValueError("frame.shortest_path(...) is limited to 64 goals.")
    else:
        raise TypeError(
            "frame.shortest_path(..., goal=...) expects [row, col] or a list of coordinates."
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
        if current in goal_cells:
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
            if neighbor not in goal_cells and cell_value not in passable_values:
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
        "selected_goal": list(reached_goal) if reached_goal is not None else None,
        "distance": len(path) - 1 if found else None,
        "path": [list(cell) for cell in returned_path],
        "moves": moves,
        "next_step": list(path[1]) if len(path) > 1 else None,
        "explored": explored,
        "search_truncated": not found and cursor < len(queue),
        "path_truncated": found and len(returned_path) < len(path),
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

    error_type = exc.__class__.__name__[:80]
    if isinstance(exc, SyntaxError):
        hint = "Fix the Python syntax at the reported line and retry."
    elif isinstance(exc, NameError):
        hint = (
            "Define or import the missing name in this call; every Python call "
            "starts fresh."
        )
    elif isinstance(exc, ImportError):
        modules = ", ".join(sorted(allowed_modules or set()))
        hint = f"Use only approved modules: {modules}." if modules else "Use an approved module."
    elif isinstance(exc, AttributeError):
        hint = "Check the documented sandbox object attributes and method names."
    elif isinstance(exc, KeyError):
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
        "source": source or None,
        "hint": hint[:300],
    }


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import ast
    import builtins
    import contextlib
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
    __FRAME_FIND_SOURCE__
    __FRAME_COMPONENTS_SOURCE__
    __FRAME_OBJECTS_SOURCE__
    __FRAME_PATTERN_SOURCE__
    __SHORTEST_PATH_SOURCE__
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


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

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

        def find(self, symbol, *, limit=64):
            # Return a bounded coordinate sample plus total count and bounds.
            return _bounded_frame_find(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                symbol=symbol,
                limit=limit,
            )

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

        def find_pattern(self, pattern, *, wildcard=None, transforms=False, limit=64):
            return _bounded_frame_find_pattern(
                self._grid,
                shape=self.shape,
                color_chars=COLOR_CHARS,
                pattern=pattern,
                wildcard=wildcard,
                transforms=transforms,
                limit=limit,
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


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
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

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
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
            _refresh_state(reply.get("state") or {})
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
        _refresh_state(initial.get("state") or {})

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
    "__FRAME_FIND_SOURCE__\n", inspect.getsource(_bounded_frame_find)
).replace(
    "__FRAME_COMPONENTS_SOURCE__\n", inspect.getsource(_bounded_frame_components)
).replace(
    "__FRAME_OBJECTS_SOURCE__\n", inspect.getsource(_bounded_frame_objects)
).replace(
    "__FRAME_PATTERN_SOURCE__\n", inspect.getsource(_bounded_frame_find_pattern)
).replace(
    "__SHORTEST_PATH_SOURCE__\n", inspect.getsource(_bounded_shortest_path)
).replace(
    "__FRAME_DIFF_SOURCE__\n", inspect.getsource(_bounded_frame_diff_summary)
).replace(
    "__EXCEPTION_DIAGNOSTIC_SOURCE__\n",
    inspect.getsource(_sandbox_exception_diagnostic),
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


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _sandbox_command() -> tuple[list[str], str | None]:
    python_command = [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP]
    bubblewrap = shutil.which("bwrap") if os.name == "posix" else None
    if bubblewrap is None:
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
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
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


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
    strategy_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    memory_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        host_strategy_updates: list[dict[str, Any]] = []
        command, isolated_cwd = _sandbox_command()
        try:
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
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
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
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
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
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
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
                _wait_for_process_exit(process)
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
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": str(exc) or "action failed in sandbox host.",
                        },
                    )
                    continue
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
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
                _send_json_line(
                    process.stdin,
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
                        _send_json_line(
                            process.stdin,
                            {
                                "type": "memory_error",
                                "error": str(exc) or "memory update failed.",
                            },
                        )
                        continue
                    except Exception:  # noqa: BLE001
                        _send_json_line(
                            process.stdin,
                            {
                                "type": "memory_error",
                                "error": "memory update failed.",
                            },
                        )
                        continue
                _send_json_line(
                    process.stdin,
                    {
                        "type": "memory_result",
                        "memory": persisted_memory,
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "diagnostic": message.get("diagnostic"),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "strategy_updates": list(host_strategy_updates),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
