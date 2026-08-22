from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_rectangles,
    _SANDBOX_BOOTSTRAP,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def rectangles(grid, **kwargs):
    return _bounded_frame_rectangles(
        grid,
        shape=(len(grid), len(grid[0]) if grid else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def test_detects_maximal_filled_rectangle():
    result = rectangles(
        [[1, 1, 1], [1, 1, 1]],
        symbol=ARC_COLOR_CHARS[1],
    )

    assert result["count"] == 1
    item = result["rectangles"][0]
    assert item["kind"] == "filled"
    assert item["bbox"] == [0, 0, 1, 2]
    assert item["shape"] == [2, 3]
    assert item["fill_ratio"] == 1.0
    assert item["touches_edge"] is True


def test_detects_closed_outline_and_summarizes_interior_palette():
    result = rectangles(
        [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
        symbol=ARC_COLOR_CHARS[1],
    )

    item = result["rectangles"][0]
    assert item["kind"] == "outline"
    assert item["border_cells"] == 12
    assert item["interior_area"] == 4
    assert item["component_size"] == 12
    assert item["fill_ratio"] == 0.75
    assert item["interior_colors"] == {ARC_COLOR_CHARS[0]: 4}


def test_rejects_open_or_irregular_shapes():
    result = rectangles(
        [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
        symbol=ARC_COLOR_CHARS[1],
    )

    assert result["count"] == 0


def test_kind_and_minimum_size_filters_are_applied():
    grid = [[1, 1], [1, 1]]

    assert rectangles(grid, kind="outline")["count"] == 0
    assert rectangles(grid, kind="filled", min_size=3)["count"] == 0
    assert rectangles(grid, kind="filled", min_size=2)["count"] == 1


def test_limit_preserves_aggregate_counts_for_separate_rectangles():
    result = rectangles(
        [[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]],
        symbol=ARC_COLOR_CHARS[1],
        limit=1,
    )

    assert result["count"] == 2
    assert result["counts_by_kind"] == {"filled": 2, "outline": 0}
    assert result["counts_by_symbol"] == {ARC_COLOR_CHARS[1]: 2}
    assert len(result["rectangles"]) == 1
    assert result["truncated_rectangles"] == 1


def test_ragged_and_invalid_cells_do_not_complete_a_rectangle():
    result = _bounded_frame_rectangles(
        [[1, 1], [1], [True, 1]],
        shape=(3, 2),
        color_chars=ARC_COLOR_CHARS,
        symbol=ARC_COLOR_CHARS[1],
    )

    assert result["count"] == 0


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"symbol": "?"}, ValueError),
        ({"kind": "solid"}, ValueError),
        ({"min_size": True}, TypeError),
        ({"limit": True}, TypeError),
    ],
)
def test_rejects_invalid_rectangle_options(kwargs, error):
    with pytest.raises(error):
        rectangles([[1]], **kwargs)


def test_injected_frame_view_exposes_rectangles():
    assert "def rectangles(self, symbol=None" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_rectangles(" in _SANDBOX_BOOTSTRAP
