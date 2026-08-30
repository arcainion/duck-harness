"""Bounded, deterministic grid pathfinding for generated gameplay policies."""

from __future__ import annotations

import heapq
from collections import deque
from operator import index
from types import MappingProxyType
from typing import Any, Iterable, TypeAlias

import numpy as np


GridPoint: TypeAlias = tuple[int, int]
Path: TypeAlias = tuple[GridPoint, ...]

MAX_GRID_CELLS = 64 * 64
MAX_CLEARANCE_RADIUS = 8
MAX_MATCH_VALUES = 256
MAX_WAYPOINTS = 32
CARDINAL_MOVES: tuple[tuple[str, int, int], ...] = (
    ("UP", -1, 0),
    ("RIGHT", 0, 1),
    ("DOWN", 1, 0),
    ("LEFT", 0, -1),
)


def _point(value: Any, label: str) -> GridPoint:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a (row, col) pair")
    try:
        if len(value) != 2:
            raise ValueError(f"{label} must be a (row, col) pair")
        row, col = value[0], value[1]
    except (TypeError, IndexError) as exc:
        raise ValueError(f"{label} must be a (row, col) pair") from exc
    if isinstance(row, bool) or isinstance(col, bool):
        raise ValueError(f"{label} coordinates must be integers")
    try:
        return index(row), index(col)
    except TypeError as exc:
        raise ValueError(f"{label} coordinates must be integers") from exc


def _mask(passable: Any) -> np.ndarray:
    mask = np.asarray(passable, dtype=bool)
    if mask.ndim != 2 or not mask.shape[0] or not mask.shape[1]:
        raise ValueError("passable must be a non-empty two-dimensional grid")
    if int(mask.size) > MAX_GRID_CELLS:
        raise ValueError(f"passable may contain at most {MAX_GRID_CELLS} cells")
    return mask


def _grid(value: Any, label: str) -> np.ndarray:
    grid = np.asarray(value)
    if grid.ndim != 2 or not grid.shape[0] or not grid.shape[1]:
        raise ValueError(f"{label} must be a non-empty two-dimensional grid")
    if int(grid.size) > MAX_GRID_CELLS:
        raise ValueError(f"{label} may contain at most {MAX_GRID_CELLS} cells")
    return grid


def _shape(value: Any) -> tuple[int, int]:
    shape = _point(value, "shape")
    if shape[0] <= 0 or shape[1] <= 0 or shape[0] * shape[1] > MAX_GRID_CELLS:
        raise ValueError(
            f"shape must contain positive dimensions and at most {MAX_GRID_CELLS} cells"
        )
    return shape


def _cost_grid(value: Any, shape: tuple[int, int]) -> np.ndarray:
    try:
        costs = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("costs must be a numeric two-dimensional grid") from exc
    if costs.shape != shape:
        raise ValueError(
            f"costs shape {costs.shape!r} must match passable shape {shape!r}"
        )
    if not bool(np.all(np.isfinite(costs))) or bool(np.any(costs < 0)):
        raise ValueError("costs must contain only finite non-negative values")
    return costs


def _in_bounds(point: GridPoint, shape: tuple[int, int]) -> bool:
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _checked_point(value: Any, label: str, shape: tuple[int, int]) -> GridPoint:
    point = _point(value, label)
    if not _in_bounds(point, shape):
        raise ValueError(f"{label} {point!r} is outside grid shape {shape!r}")
    return point


def _expansion_limit(value: Any, cell_count: int) -> int:
    if isinstance(value, bool):
        raise ValueError("max_expansions must be a positive integer")
    try:
        limit = index(value)
    except TypeError as exc:
        raise ValueError("max_expansions must be a positive integer") from exc
    if not 1 <= limit <= MAX_GRID_CELLS:
        raise ValueError(f"max_expansions must be between 1 and {MAX_GRID_CELLS}")
    return min(limit, cell_count)


def _bounded_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(
            f"{label} must be an integer between {minimum} and {maximum}"
        ) from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return result


def _checked_points(
    values: Iterable[Any],
    label: str,
    shape: tuple[int, int],
    *,
    max_points: int = MAX_GRID_CELLS,
    preserve_order: bool = False,
) -> tuple[GridPoint, ...]:
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{label}s must be an iterable of (row, col) pairs") from exc
    points: list[GridPoint] = []
    for position, value in enumerate(iterator):
        if position >= max_points:
            raise ValueError(f"{label}s may contain at most {max_points} points")
        points.append(_checked_point(value, label, shape))
    if preserve_order:
        return tuple(points)
    return tuple(sorted(set(points)))


def _reconstruct_path(
    parents: dict[GridPoint, GridPoint | None], destination: GridPoint
) -> Path:
    reversed_path: list[GridPoint] = []
    cursor: GridPoint | None = destination
    while cursor is not None:
        reversed_path.append(cursor)
        cursor = parents[cursor]
    return tuple(reversed(reversed_path))


def cardinal_neighbors(passable: Any, point: Any) -> Path:
    """Return traversable four-way neighbors in UP, RIGHT, DOWN, LEFT order."""

    mask = _mask(passable)
    current = _checked_point(point, "point", mask.shape)
    result: list[GridPoint] = []
    for _action, row_delta, col_delta in CARDINAL_MOVES:
        candidate = (current[0] + row_delta, current[1] + col_delta)
        if _in_bounds(candidate, mask.shape) and bool(mask[candidate]):
            result.append(candidate)
    return tuple(result)


def shortest_path(
    passable: Any,
    start: Any,
    goal: Any,
    max_expansions: int = MAX_GRID_CELLS,
) -> Path:
    """Return a shortest four-way path, including both endpoints, or ``()``."""

    return shortest_path_to_any(passable, start, (goal,), max_expansions)


def shortest_path_to_any(
    passable: Any,
    start: Any,
    goals: Iterable[Any],
    max_expansions: int = MAX_GRID_CELLS,
) -> Path:
    """Return a shortest path to any goal using deterministic bounded BFS.

    The start and goal cells are allowed even when their entries in ``passable`` are
    false. This lets policies build masks from terrain while locating an actor or
    target on a distinct board value. Intermediate cells must be passable.
    """

    mask = _mask(passable)
    origin = _checked_point(start, "start", mask.shape)
    destinations = set(_checked_points(goals, "goal", mask.shape))
    if not destinations:
        return ()
    if origin in destinations:
        return (origin,)

    limit = _expansion_limit(max_expansions, int(mask.size))
    frontier: deque[GridPoint] = deque((origin,))
    parents: dict[GridPoint, GridPoint | None] = {origin: None}
    expansions = 0

    while frontier and expansions < limit:
        current = frontier.popleft()
        expansions += 1
        for _action, row_delta, col_delta in CARDINAL_MOVES:
            candidate = (current[0] + row_delta, current[1] + col_delta)
            if not _in_bounds(candidate, mask.shape) or candidate in parents:
                continue
            if candidate not in destinations and not bool(mask[candidate]):
                continue
            parents[candidate] = current
            if candidate in destinations:
                return _reconstruct_path(parents, candidate)
            frontier.append(candidate)
    return ()


def path_to_actions(path: Iterable[Any]) -> tuple[str, ...]:
    """Convert an orthogonally adjacent point path into action names."""

    try:
        iterator = iter(path)
    except TypeError as exc:
        raise ValueError("path must be an iterable of (row, col) pairs") from exc
    points: list[GridPoint] = []
    for position, value in enumerate(iterator):
        if position >= MAX_GRID_CELLS:
            raise ValueError(f"path may contain at most {MAX_GRID_CELLS} points")
        points.append(_point(value, "path point"))
    actions: list[str] = []
    action_by_delta = {
        (row_delta, col_delta): action
        for action, row_delta, col_delta in CARDINAL_MOVES
    }
    for first, second in zip(points, points[1:], strict=False):
        delta = (second[0] - first[0], second[1] - first[1])
        try:
            actions.append(action_by_delta[delta])
        except KeyError as exc:
            raise ValueError(
                f"path contains a non-cardinal step from {first!r} to {second!r}"
            ) from exc
    return tuple(actions)


def next_path_action(
    path: Iterable[Any], valid_actions: Iterable[str] = ()
) -> str | None:
    """Return the path's first action, or ``None`` if absent/currently invalid."""

    actions = path_to_actions(path)
    if not actions:
        return None
    allowed = {str(action).upper() for action in valid_actions}
    if allowed and actions[0] not in allowed:
        return None
    return actions[0]


def find_cells(grid: Any, values: Any, max_results: int = MAX_GRID_CELLS) -> Path:
    """Find matching grid values in deterministic row-major order."""

    board = _grid(grid, "grid")
    limit = _bounded_integer(max_results, "max_results", 1, MAX_GRID_CELLS)
    if np.isscalar(values):
        requested = (values,)
    else:
        try:
            iterator = iter(values)
        except TypeError as exc:
            raise ValueError("values must be a scalar or iterable") from exc
        requested_values: list[Any] = []
        for position, value in enumerate(iterator):
            if position >= MAX_MATCH_VALUES:
                raise ValueError(
                    f"values may contain at most {MAX_MATCH_VALUES} entries"
                )
            requested_values.append(value)
        requested = tuple(requested_values)
    if not requested:
        return ()
    matches = np.argwhere(np.isin(board, requested))
    return tuple((int(row), int(col)) for row, col in matches[:limit])


def distance_map(
    passable: Any,
    starts: Iterable[Any],
    max_expansions: int = MAX_GRID_CELLS,
) -> np.ndarray:
    """Return read-only four-way distances from one or more origins.

    Unreachable cells contain -1. Origin cells receive distance zero even when their
    passable entry is false, matching the endpoint behavior of ``shortest_path``.
    """

    mask = _mask(passable)
    origins = _checked_points(starts, "start", mask.shape)
    limit = _expansion_limit(max_expansions, int(mask.size))
    distances = np.full(mask.shape, -1, dtype=np.int16)
    frontier: deque[GridPoint] = deque()
    for origin in origins:
        distances[origin] = 0
        frontier.append(origin)

    expansions = 0
    while frontier and expansions < limit:
        current = frontier.popleft()
        expansions += 1
        next_distance = int(distances[current]) + 1
        for _action, row_delta, col_delta in CARDINAL_MOVES:
            candidate = (current[0] + row_delta, current[1] + col_delta)
            if not _in_bounds(candidate, mask.shape):
                continue
            if distances[candidate] >= 0 or not bool(mask[candidate]):
                continue
            distances[candidate] = next_distance
            frontier.append(candidate)
    distances.setflags(write=False)
    return distances


def reachable_points(
    passable: Any,
    start: Any,
    max_distance: int | None = None,
    max_expansions: int = MAX_GRID_CELLS,
) -> Path:
    """Return reachable points in row-major order, optionally within a distance."""

    mask = _mask(passable)
    origin = _checked_point(start, "start", mask.shape)
    distances = distance_map(mask, (origin,), max_expansions)
    if max_distance is None:
        selected = distances >= 0
    else:
        radius = _bounded_integer(max_distance, "max_distance", 0, MAX_GRID_CELLS - 1)
        selected = (distances >= 0) & (distances <= radius)
    return tuple((int(row), int(col)) for row, col in np.argwhere(selected))


def connected_components(passable: Any, min_size: int = 1) -> tuple[Path, ...]:
    """Return four-way passable components, largest first with stable tie-breaking."""

    mask = _mask(passable)
    threshold = _bounded_integer(min_size, "min_size", 1, MAX_GRID_CELLS)
    unseen = np.array(mask, dtype=bool, copy=True)
    components: list[Path] = []
    for raw_row, raw_col in np.argwhere(mask):
        origin = (int(raw_row), int(raw_col))
        if not bool(unseen[origin]):
            continue
        unseen[origin] = False
        frontier: deque[GridPoint] = deque((origin,))
        component: list[GridPoint] = []
        while frontier:
            current = frontier.popleft()
            component.append(current)
            for _action, row_delta, col_delta in CARDINAL_MOVES:
                candidate = (current[0] + row_delta, current[1] + col_delta)
                if not _in_bounds(candidate, mask.shape) or not bool(unseen[candidate]):
                    continue
                unseen[candidate] = False
                frontier.append(candidate)
        if len(component) >= threshold:
            components.append(tuple(sorted(component)))
    components.sort(key=lambda component: (-len(component), component[0]))
    return tuple(components)


def shortest_path_through(
    passable: Any,
    start: Any,
    waypoints: Iterable[Any],
    max_expansions_per_leg: int = MAX_GRID_CELLS,
) -> Path:
    """Join shortest paths through up to 32 ordered waypoints, or return ``()``."""

    mask = _mask(passable)
    origin = _checked_point(start, "start", mask.shape)
    targets = _checked_points(
        waypoints,
        "waypoint",
        mask.shape,
        max_points=MAX_WAYPOINTS,
        preserve_order=True,
    )
    limit = _expansion_limit(max_expansions_per_leg, int(mask.size))
    result: list[GridPoint] = [origin]
    current = origin
    for target in targets:
        leg = shortest_path(mask, current, target, limit)
        if not leg:
            return ()
        if len(result) + len(leg) - 1 > MAX_GRID_CELLS:
            raise ValueError(
                f"combined waypoint path may contain at most {MAX_GRID_CELLS} points"
            )
        result.extend(leg[1:])
        current = target
    return tuple(result)


def weighted_shortest_path(
    passable: Any,
    costs: Any,
    start: Any,
    goal: Any,
    max_expansions: int = MAX_GRID_CELLS,
) -> Path:
    """Return a minimum-cost path, excluding the start cell from traversal cost."""

    return weighted_shortest_path_to_any(
        passable, costs, start, (goal,), max_expansions
    )


def weighted_shortest_path_to_any(
    passable: Any,
    costs: Any,
    start: Any,
    goals: Iterable[Any],
    max_expansions: int = MAX_GRID_CELLS,
) -> Path:
    """Return a deterministic minimum-cost four-way path to any goal."""

    mask = _mask(passable)
    traversal_costs = _cost_grid(costs, mask.shape)
    origin = _checked_point(start, "start", mask.shape)
    destinations = set(_checked_points(goals, "goal", mask.shape))
    if not destinations:
        return ()
    if origin in destinations:
        return (origin,)

    limit = _expansion_limit(max_expansions, int(mask.size))
    parents: dict[GridPoint, GridPoint | None] = {origin: None}
    best_cost: dict[GridPoint, float] = {origin: 0.0}
    serial = 0
    frontier: list[tuple[float, int, GridPoint]] = [(0.0, serial, origin)]
    expansions = 0

    while frontier and expansions < limit:
        total_cost, _order, current = heapq.heappop(frontier)
        if total_cost != best_cost.get(current):
            continue
        expansions += 1
        if current in destinations:
            return _reconstruct_path(parents, current)
        for _action, row_delta, col_delta in CARDINAL_MOVES:
            candidate = (current[0] + row_delta, current[1] + col_delta)
            if not _in_bounds(candidate, mask.shape):
                continue
            if candidate not in destinations and not bool(mask[candidate]):
                continue
            candidate_cost = total_cost + float(traversal_costs[candidate])
            if candidate_cost >= best_cost.get(candidate, float("inf")):
                continue
            best_cost[candidate] = candidate_cost
            parents[candidate] = current
            serial += 1
            heapq.heappush(frontier, (candidate_cost, serial, candidate))
    return ()


def path_cost(costs: Any, path: Iterable[Any]) -> float:
    """Return the sum of costs for entered path cells; the first cell is free."""

    raw_costs = _grid(costs, "costs")
    cost_grid = _cost_grid(raw_costs, raw_costs.shape)
    try:
        iterator = iter(path)
    except TypeError as exc:
        raise ValueError("path must be an iterable of (row, col) pairs") from exc
    points: list[GridPoint] = []
    for position, value in enumerate(iterator):
        if position >= MAX_GRID_CELLS:
            raise ValueError(f"path may contain at most {MAX_GRID_CELLS} points")
        points.append(_checked_point(value, "path point", cost_grid.shape))
    path_to_actions(points)
    return float(sum(float(cost_grid[point]) for point in points[1:]))


def clearance_mask(passable: Any, radius: int = 1) -> np.ndarray:
    """Return cells whose square radius is entirely passable and inside the grid."""

    mask = _mask(passable)
    checked_radius = _bounded_integer(radius, "radius", 0, MAX_CLEARANCE_RADIUS)
    if checked_radius == 0:
        result = np.array(mask, dtype=bool, copy=True)
    else:
        rows, cols = mask.shape
        padded = np.pad(mask, checked_radius, mode="constant", constant_values=False)
        result = np.ones(mask.shape, dtype=bool)
        width = checked_radius * 2 + 1
        for row_offset in range(width):
            for col_offset in range(width):
                result &= padded[
                    row_offset : row_offset + rows,
                    col_offset : col_offset + cols,
                ]
    result.setflags(write=False)
    return result


def grid_line(start: Any, goal: Any, shape: Any = (64, 64)) -> Path:
    """Return a bounded Bresenham grid ray including both endpoints."""

    checked_shape = _shape(shape)
    first = _checked_point(start, "start", checked_shape)
    last = _checked_point(goal, "goal", checked_shape)
    row, col = first
    goal_row, goal_col = last
    col_distance = abs(goal_col - col)
    row_distance = -abs(goal_row - row)
    col_step = 1 if col < goal_col else -1
    row_step = 1 if row < goal_row else -1
    error = col_distance + row_distance
    points: list[GridPoint] = []
    while True:
        points.append((row, col))
        if row == goal_row and col == goal_col:
            return tuple(points)
        doubled_error = 2 * error
        if doubled_error >= row_distance:
            error += row_distance
            col += col_step
        if doubled_error <= col_distance:
            error += col_distance
            row += row_step


def line_of_sight(passable: Any, start: Any, goal: Any) -> bool:
    """Return whether every intermediate cell on the grid ray is passable."""

    mask = _mask(passable)
    ray = grid_line(start, goal, mask.shape)
    return all(bool(mask[point]) for point in ray[1:-1])


def action_destination(
    point: Any, action: Any, shape: Any = (64, 64)
) -> GridPoint | None:
    """Project one cardinal action, returning ``None`` when it leaves the grid."""

    checked_shape = _shape(shape)
    origin = _checked_point(point, "point", checked_shape)
    action_name = str(action).strip().upper()
    for candidate_name, row_delta, col_delta in CARDINAL_MOVES:
        if action_name != candidate_name:
            continue
        candidate = (origin[0] + row_delta, origin[1] + col_delta)
        return candidate if _in_bounds(candidate, checked_shape) else None
    raise ValueError(f"action must be one of UP, RIGHT, DOWN, LEFT; got {action!r}")


def value_mask(grid: Any, values: Any, invert: bool = False) -> np.ndarray:
    """Return a read-only mask selecting one or more grid values."""

    board = _grid(grid, "grid")
    if np.isscalar(values):
        requested = (values,)
    else:
        try:
            iterator = iter(values)
        except TypeError as exc:
            raise ValueError("values must be a scalar or iterable") from exc
        requested_values: list[Any] = []
        for position, value in enumerate(iterator):
            if position >= MAX_MATCH_VALUES:
                raise ValueError(
                    f"values may contain at most {MAX_MATCH_VALUES} entries"
                )
            requested_values.append(value)
        requested = tuple(requested_values)
    result = np.isin(board, requested)
    if bool(invert):
        result = np.logical_not(result)
    result = np.asarray(result, dtype=bool)
    result.setflags(write=False)
    return result


def component_boxes(
    mask: Any, min_size: int = 1
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Return ``(top, left, bottom, right, size)`` for each passable component."""

    boxes: list[tuple[int, int, int, int, int]] = []
    for component in connected_components(mask, min_size):
        rows = [point[0] for point in component]
        cols = [point[1] for point in component]
        boxes.append((min(rows), min(cols), max(rows), max(cols), len(component)))
    return tuple(boxes)


def component_centers(mask: Any, min_size: int = 1) -> Path:
    """Return a real component cell nearest each bounding-box center."""

    centers: list[GridPoint] = []
    for component in connected_components(mask, min_size):
        top = min(point[0] for point in component)
        left = min(point[1] for point in component)
        bottom = max(point[0] for point in component)
        right = max(point[1] for point in component)
        center_row_twice = top + bottom
        center_col_twice = left + right
        centers.append(
            min(
                component,
                key=lambda point: (
                    (2 * point[0] - center_row_twice) ** 2
                    + (2 * point[1] - center_col_twice) ** 2,
                    point,
                ),
            )
        )
    return tuple(centers)


def approach_points(passable: Any, targets: Iterable[Any], distance: int = 1) -> Path:
    """Return passable cells at an exact Manhattan distance from target cells."""

    mask = _mask(passable)
    checked_distance = _bounded_integer(distance, "distance", 0, MAX_CLEARANCE_RADIUS)
    target_points = _checked_points(targets, "target", mask.shape)
    if not target_points:
        return ()
    distances = np.full(mask.shape, -1, dtype=np.int16)
    frontier: deque[GridPoint] = deque()
    for target in target_points:
        distances[target] = 0
        frontier.append(target)
    while frontier:
        current = frontier.popleft()
        current_distance = int(distances[current])
        if current_distance >= checked_distance:
            continue
        for _action, row_delta, col_delta in CARDINAL_MOVES:
            candidate = (current[0] + row_delta, current[1] + col_delta)
            if not _in_bounds(candidate, mask.shape) or distances[candidate] >= 0:
                continue
            distances[candidate] = current_distance + 1
            frontier.append(candidate)
    selected = (distances == checked_distance) & mask
    return tuple((int(row), int(col)) for row, col in np.argwhere(selected))


def shortest_approach_path(
    passable: Any,
    start: Any,
    targets: Iterable[Any],
    distance: int = 1,
    max_expansions: int = MAX_GRID_CELLS,
) -> Path:
    """Return a shortest route to a passable target-approach cell."""

    mask = _mask(passable)
    candidates = approach_points(mask, targets, distance)
    if not candidates:
        return ()
    return shortest_path_to_any(mask, start, candidates, max_expansions)


def path_is_valid(passable: Any, path: Iterable[Any]) -> bool:
    """Check bounds, cardinal adjacency, and passable intermediate path cells."""

    mask = _mask(passable)
    try:
        iterator = iter(path)
    except TypeError:
        return False
    points: list[GridPoint] = []
    try:
        for position, value in enumerate(iterator):
            if position >= MAX_GRID_CELLS:
                return False
            points.append(_checked_point(value, "path point", mask.shape))
        if not points:
            return False
        path_to_actions(points)
    except ValueError:
        return False
    return all(bool(mask[point]) for point in points[1:-1])


def path_suffix(path: Iterable[Any], current: Any) -> Path:
    """Return the suffix after the last occurrence of the current position."""

    current_point = _point(current, "current")
    try:
        iterator = iter(path)
    except TypeError as exc:
        raise ValueError("path must be an iterable of (row, col) pairs") from exc
    points: list[GridPoint] = []
    for position, value in enumerate(iterator):
        if position >= MAX_GRID_CELLS:
            raise ValueError(f"path may contain at most {MAX_GRID_CELLS} points")
        points.append(_point(value, "path point"))
    matching_indices = [
        position for position, point in enumerate(points) if point == current_point
    ]
    if not matching_indices:
        return ()
    return tuple(points[matching_indices[-1] :])


PATHFINDING_API_VERSION = 1
POLICY_PATHFINDING_GLOBALS = MappingProxyType(
    {
        "PATHFINDING_API_VERSION": PATHFINDING_API_VERSION,
        "CARDINAL_MOVES": CARDINAL_MOVES,
        "action_destination": action_destination,
        "approach_points": approach_points,
        "cardinal_neighbors": cardinal_neighbors,
        "clearance_mask": clearance_mask,
        "component_boxes": component_boxes,
        "component_centers": component_centers,
        "connected_components": connected_components,
        "distance_map": distance_map,
        "find_cells": find_cells,
        "grid_line": grid_line,
        "line_of_sight": line_of_sight,
        "next_path_action": next_path_action,
        "path_cost": path_cost,
        "path_is_valid": path_is_valid,
        "path_suffix": path_suffix,
        "path_to_actions": path_to_actions,
        "reachable_points": reachable_points,
        "shortest_approach_path": shortest_approach_path,
        "shortest_path": shortest_path,
        "shortest_path_to_any": shortest_path_to_any,
        "shortest_path_through": shortest_path_through,
        "value_mask": value_mask,
        "weighted_shortest_path": weighted_shortest_path,
        "weighted_shortest_path_to_any": weighted_shortest_path_to_any,
    }
)
