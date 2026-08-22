from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_periodicity,
    _SANDBOX_BOOTSTRAP,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def periodicity(grid, **kwargs):
    return _bounded_frame_periodicity(
        grid,
        shape=(len(grid), len(grid[0]) if grid else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def candidate(result, axis, period):
    return next(
        item for item in result[axis]["candidates"] if item["period"] == period
    )


def tiled_grid():
    return [
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2],
        [3, 4, 3, 4],
    ]


def test_detects_fundamental_row_and_column_periods():
    result = periodicity(tiled_grid())

    assert result["rows"]["exact_periods"] == [2]
    assert result["columns"]["exact_periods"] == [2]
    assert result["rows"]["fundamental_period"] == 2
    assert result["columns"]["fundamental_period"] == 2
    assert candidate(result, "rows", 2)["complete_tiles"] is True
    assert candidate(result, "columns", 2)["match_ratio"] == 1.0


def test_ranks_and_explains_near_periodicity():
    grid = tiled_grid()
    grid[3][3] = 9
    result = periodicity(grid)
    row_period = candidate(result, "rows", 2)
    column_period = candidate(result, "columns", 2)

    assert result["rows"]["best_period"] == 2
    assert result["columns"]["best_period"] == 2
    assert row_period["mismatches"] == 1
    assert row_period["mismatch_samples"][0] == {
        "cell": [3, 3],
        "counterpart": [1, 3],
        "actual": ARC_COLOR_CHARS[9],
        "expected": ARC_COLOR_CHARS[4],
    }
    assert column_period["mismatches"] == 1


def test_candidate_limit_preserves_exact_period_summary():
    result = periodicity(tiled_grid(), candidate_limit=1)

    assert result["rows"]["fundamental_period"] == 2
    assert result["rows"]["scanned_candidates"] == 3
    assert len(result["rows"]["candidates"]) == 1
    assert result["rows"]["truncated_candidates"] == 2


def test_sample_limit_preserves_mismatch_totals():
    result = periodicity(
        [[1, 2], [3, 4], [5, 6]],
        sample_limit=1,
    )
    row_period = candidate(result, "rows", 1)

    assert row_period["mismatches"] == 4
    assert len(row_period["mismatch_samples"]) == 1
    assert row_period["truncated_mismatches"] == 3


def test_ragged_cells_are_compared_as_unknown():
    result = _bounded_frame_periodicity(
        [[1, 2], [1]],
        shape=(2, 2),
        color_chars=ARC_COLOR_CHARS,
    )
    row_period = candidate(result, "rows", 1)

    assert row_period["mismatches"] == 1
    assert row_period["mismatch_samples"][0]["actual"] == "?"


def test_single_cell_frame_has_no_nontrivial_period_candidates():
    result = periodicity([[1]])

    assert result["rows"]["fundamental_period"] is None
    assert result["columns"]["best_period"] is None
    assert result["rows"]["candidates"] == []


@pytest.mark.parametrize("name", ["candidate_limit", "sample_limit"])
def test_rejects_boolean_limits(name):
    with pytest.raises(TypeError, match="expects an integer"):
        periodicity([[1]], **{name: True})


def test_injected_frame_view_exposes_periodicity():
    assert "def periodicity(self, *, candidate_limit=8" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_periodicity(" in _SANDBOX_BOOTSTRAP
