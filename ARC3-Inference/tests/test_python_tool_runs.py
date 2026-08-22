from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import _bounded_frame_runs, _SANDBOX_BOOTSTRAP
from inference.utils.grid_utils import ARC_COLOR_CHARS


def runs(grid, **kwargs):
    return _bounded_frame_runs(
        grid,
        shape=(len(grid), len(grid[0]) if grid else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def test_finds_only_maximal_horizontal_runs_for_selected_color():
    result = runs(
        [[1, 1, 1], [0, 1, 0], [1, 1, 1]],
        symbol=ARC_COLOR_CHARS[1],
        directions="H",
    )

    assert result["count"] == 2
    assert result["counts_by_direction"] == {"HORIZONTAL": 2}
    assert [item["start"] for item in result["runs"]] == [[0, 0], [2, 0]]
    assert [item["end"] for item in result["runs"]] == [[0, 2], [2, 2]]
    assert all(item["length"] == 3 for item in result["runs"])


def test_detects_both_diagonal_axes():
    result = runs(
        [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
        symbol=ARC_COLOR_CHARS[1],
        directions="D",
        min_length=3,
    )

    assert result["count"] == 2
    assert result["counts_by_direction"] == {
        "DIAGONAL_DOWN": 1,
        "DIAGONAL_UP": 1,
    }
    assert {tuple(item["delta"]) for item in result["runs"]} == {(1, 1), (-1, 1)}


def test_unfiltered_scan_reports_per_color_counts():
    result = runs([[1, 1], [2, 2]], directions="H")

    assert result["count"] == 2
    assert result["counts_by_symbol"] == {
        ARC_COLOR_CHARS[1]: 1,
        ARC_COLOR_CHARS[2]: 1,
    }


def test_limit_truncates_details_without_losing_aggregate_counts():
    result = runs(
        [[1, 1], [1, 1], [1, 1]],
        symbol=ARC_COLOR_CHARS[1],
        directions="H",
        limit=1,
    )

    assert result["count"] == 3
    assert len(result["runs"]) == 1
    assert result["truncated_runs"] == 2


def test_ragged_and_invalid_cells_break_runs():
    result = _bounded_frame_runs(
        [[1, 1, 1], [1], [True, 1, 1]],
        shape=(3, 3),
        color_chars=ARC_COLOR_CHARS,
        symbol=ARC_COLOR_CHARS[1],
        directions="H",
    )

    assert result["count"] == 2
    assert [item["length"] for item in result["runs"]] == [3, 2]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"symbol": "?"}, ValueError),
        ({"directions": ""}, ValueError),
        ({"directions": "HX"}, ValueError),
        ({"min_length": True}, TypeError),
        ({"limit": True}, TypeError),
    ],
)
def test_rejects_invalid_run_options(kwargs, error):
    with pytest.raises(error):
        runs([[1]], **kwargs)


def test_injected_frame_view_exposes_runs():
    assert "def runs(self, symbol=None" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_runs(" in _SANDBOX_BOOTSTRAP
