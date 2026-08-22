from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_enclosed_regions,
    _SANDBOX_BOOTSTRAP,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def enclosed(grid, **kwargs):
    return _bounded_frame_enclosed_regions(
        grid,
        shape=(len(grid), len(grid[0]) if grid else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def test_finds_arbitrary_enclosed_region_and_boundary_palette():
    result = enclosed(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        symbol=ARC_COLOR_CHARS[0],
    )

    assert result["count"] == 1
    region = result["regions"][0]
    assert region["size"] == 9
    assert region["bbox"] == [1, 1, 3, 3]
    assert region["centroid"] == [2.0, 2.0]
    assert region["boundary_cells"] == 12
    assert region["boundary_colors"] == {ARC_COLOR_CHARS[1]: 12}


def test_excludes_regions_that_reach_frame_edge():
    result = enclosed(
        [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
        symbol=ARC_COLOR_CHARS[0],
    )

    assert result["count"] == 0


def test_optional_symbol_scans_all_enclosed_colors():
    result = enclosed(
        [[1, 1, 1, 1], [1, 0, 2, 1], [1, 1, 1, 1]],
    )

    assert result["count"] == 2
    assert result["counts_by_symbol"] == {
        ARC_COLOR_CHARS[0]: 1,
        ARC_COLOR_CHARS[2]: 1,
    }


def test_diagonal_connectivity_can_merge_enclosed_regions():
    grid = [
        [1, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
    ]

    four = enclosed(grid, symbol=ARC_COLOR_CHARS[0])
    eight = enclosed(grid, symbol=ARC_COLOR_CHARS[0], diagonal=True)
    assert four["count"] == 2
    assert eight["count"] == 1
    assert eight["regions"][0]["size"] == 2


def test_limits_region_and_cell_details_without_losing_counts():
    result = enclosed(
        [[1, 1, 1, 1], [1, 0, 2, 1], [1, 1, 1, 1]],
        limit=1,
        cell_limit=0,
    )

    assert result["count"] == 2
    assert len(result["regions"]) == 1
    assert result["truncated_regions"] == 1
    assert result["regions"][0]["cells"] == []
    assert result["regions"][0]["truncated_cells"] == 1


def test_invalid_internal_cell_is_reported_as_unknown_boundary():
    result = _bounded_frame_enclosed_regions(
        [[1, 1, 1], [1, 0, True], [1, 1, 1]],
        shape=(3, 3),
        color_chars=ARC_COLOR_CHARS,
        symbol=ARC_COLOR_CHARS[0],
    )

    assert result["regions"][0]["boundary_colors"] == {
        "?": 1,
        ARC_COLOR_CHARS[1]: 3,
    }


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"symbol": "?"}, ValueError),
        ({"diagonal": 1}, TypeError),
        ({"limit": True}, TypeError),
        ({"cell_limit": True}, TypeError),
    ],
)
def test_rejects_invalid_enclosure_options(kwargs, error):
    with pytest.raises(error):
        enclosed([[1]], **kwargs)


def test_injected_frame_view_exposes_enclosed_regions():
    assert "def enclosed_regions(" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_enclosed_regions(" in _SANDBOX_BOOTSTRAP
