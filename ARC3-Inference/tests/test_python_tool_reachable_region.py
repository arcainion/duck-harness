from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import (
    _bounded_reachable_region,
    _SANDBOX_BOOTSTRAP,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def reachable(grid, **kwargs):
    return _bounded_reachable_region(
        grid,
        shape=(len(grid), len(grid[0]) if grid else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def test_summarizes_reachable_space_distance_and_blocked_frontier():
    result = reachable(
        [[0, 0, 1], [1, 0, 1], [0, 0, 0]],
        start=[0, 0],
        passable=ARC_COLOR_CHARS[0],
    )

    assert result["reachable_count"] == 6
    assert result["bbox"] == [0, 0, 2, 2]
    assert result["touches_edge"] is True
    assert result["maximum_distance"] == 4
    assert {tuple(cell) for cell in result["farthest_cells"]} == {(2, 0), (2, 2)}
    assert result["frontier_count"] == 3
    assert result["frontier_colors"] == {ARC_COLOR_CHARS[1]: 3}


def test_diagonal_connectivity_can_join_corner_regions():
    grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    four = reachable(grid, start=[1, 1], passable=ARC_COLOR_CHARS[0])
    eight = reachable(
        grid,
        start=[1, 1],
        passable=ARC_COLOR_CHARS[0],
        diagonal=True,
    )
    assert four["reachable_count"] == 1
    assert eight["reachable_count"] == 5
    assert eight["connectivity"] == 8


def test_node_budget_reports_partial_search():
    result = reachable(
        [[0, 0, 0], [0, 0, 0]],
        start=[0, 0],
        passable=ARC_COLOR_CHARS[0],
        max_nodes=1,
    )

    assert result["explored"] == 1
    assert result["reachable_count"] == 3
    assert result["maximum_distance"] == 1
    assert result["search_truncated"] is True


def test_detail_limits_preserve_reachable_and_frontier_counts():
    result = reachable(
        [[0, 0, 1], [1, 0, 1], [0, 0, 0]],
        start=[0, 0],
        passable=ARC_COLOR_CHARS[0],
        cell_limit=1,
        frontier_limit=1,
    )

    assert result["reachable_count"] == 6
    assert len(result["cells"]) == 1
    assert result["truncated_cells"] == 5
    assert result["frontier_count"] == 3
    assert len(result["frontier"]) == 1
    assert result["truncated_frontier"] == 2


def test_ragged_missing_cell_is_unknown_frontier():
    result = _bounded_reachable_region(
        [[0, 0], [0]],
        shape=(2, 2),
        color_chars=ARC_COLOR_CHARS,
        start=[0, 0],
        passable=ARC_COLOR_CHARS[0],
    )

    assert result["reachable_count"] == 3
    assert result["frontier_colors"] == {"?": 1}


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"start": [0], "passable": "W"}, TypeError),
        ({"start": [True, 0], "passable": "W"}, TypeError),
        ({"start": [2, 0], "passable": "W"}, ValueError),
        ({"start": [0, 0], "passable": "?"}, ValueError),
        ({"start": [0, 0], "passable": 0}, TypeError),
        ({"start": [0, 0], "passable": "W", "diagonal": 1}, TypeError),
        ({"start": [0, 0], "passable": "W", "max_nodes": True}, TypeError),
        ({"start": [0, 0], "passable": "W", "cell_limit": True}, TypeError),
        ({"start": [0, 0], "passable": "W", "frontier_limit": True}, TypeError),
    ],
)
def test_rejects_invalid_reachability_options(kwargs, error):
    with pytest.raises(error):
        reachable([[0]], **kwargs)


def test_injected_frame_view_exposes_reachable_region():
    assert "def reachable_region(" in _SANDBOX_BOOTSTRAP
    assert "_bounded_reachable_region(" in _SANDBOX_BOOTSTRAP
