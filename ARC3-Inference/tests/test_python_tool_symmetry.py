from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import _bounded_frame_symmetry, _SANDBOX_BOOTSTRAP
from inference.utils.grid_utils import ARC_COLOR_CHARS


def symmetry(grid, **kwargs):
    return _bounded_frame_symmetry(
        grid,
        shape=(len(grid), len(grid[0]) if grid else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def candidate(result, name):
    return next(item for item in result["candidates"] if item["symmetry"] == name)


def test_detects_vertical_axis_symmetry_on_rectangular_frame():
    result = symmetry([[1, 0, 1], [2, 0, 2]])

    assert result["symmetries"] == ["VERTICAL_AXIS"]
    assert result["best"] == ["VERTICAL_AXIS"]
    assert candidate(result, "VERTICAL_AXIS")["match_ratio"] == 1.0


def test_reports_cells_that_break_near_symmetry():
    result = symmetry([[1, 0, 1], [2, 0, 3]])
    vertical = candidate(result, "VERTICAL_AXIS")

    assert vertical["mismatched_cells"] == 2
    assert vertical["match_ratio"] == pytest.approx(4 / 6)
    assert vertical["mismatches"] == [
        {
            "cell": [1, 0],
            "counterpart": [1, 2],
            "actual": ARC_COLOR_CHARS[2],
            "expected": ARC_COLOR_CHARS[3],
        },
        {
            "cell": [1, 2],
            "counterpart": [1, 0],
            "actual": ARC_COLOR_CHARS[3],
            "expected": ARC_COLOR_CHARS[2],
        },
    ]


def test_detects_rotational_and_diagonal_square_symmetries():
    result = symmetry([[1, 2], [2, 1]])

    assert "ROTATE_180" in result["symmetries"]
    assert "MAIN_DIAGONAL" in result["symmetries"]
    assert "ANTI_DIAGONAL" in result["symmetries"]


def test_marks_square_only_candidates_incompatible_for_rectangles():
    result = symmetry([[1, 2, 3], [4, 5, 6]])

    assert candidate(result, "ROTATE_90") == {
        "symmetry": "ROTATE_90",
        "compatible_shape": False,
    }
    assert candidate(result, "MAIN_DIAGONAL")["compatible_shape"] is False


def test_sample_limit_preserves_mismatch_count_and_reports_truncation():
    result = symmetry([[1, 2], [3, 4]], sample_limit=1)
    horizontal = candidate(result, "HORIZONTAL_AXIS")

    assert horizontal["mismatched_cells"] == 4
    assert len(horizontal["mismatches"]) == 1
    assert horizontal["truncated_mismatches"] == 3


def test_ragged_cells_are_compared_as_unknown():
    result = _bounded_frame_symmetry(
        [[1, 1], [1]],
        shape=(2, 2),
        color_chars=ARC_COLOR_CHARS,
    )

    vertical = candidate(result, "VERTICAL_AXIS")
    assert vertical["mismatched_cells"] == 2
    assert {item["actual"] for item in vertical["mismatches"]} == {
        "?",
        ARC_COLOR_CHARS[1],
    }


def test_empty_frame_is_symmetric_under_every_candidate():
    result = symmetry([])

    assert len(result["symmetries"]) == 7
    assert all(item["match_ratio"] == 1.0 for item in result["candidates"])


def test_rejects_boolean_sample_limit():
    with pytest.raises(TypeError, match="expects an integer"):
        symmetry([[1]], sample_limit=True)


def test_injected_frame_view_exposes_symmetry():
    assert "def symmetry(self, *, sample_limit=8)" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_symmetry(" in _SANDBOX_BOOTSTRAP
