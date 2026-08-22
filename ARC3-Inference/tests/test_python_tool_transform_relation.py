from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_transform_relation,
    _SANDBOX_BOOTSTRAP,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def relation(before, after, **kwargs):
    return _bounded_frame_transform_relation(
        before,
        after,
        before_shape=(len(before), len(before[0]) if before else 0),
        after_shape=(len(after), len(after[0]) if after else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def candidate(result, name):
    return next(item for item in result["candidates"] if item["transform"] == name)


def test_detects_rectangular_rotation_with_shape_change():
    result = relation(
        [[1, 2, 3], [4, 5, 6]],
        [[4, 1], [5, 2], [6, 3]],
    )

    assert result["exact_matches"] == ["ROTATE_90"]
    assert result["best"] == ["ROTATE_90"]
    assert candidate(result, "ROTATE_90")["exact_match_ratio"] == 1.0
    assert candidate(result, "IDENTITY")["compatible_shape"] is False


def test_reports_consistent_recolor_mapping_for_reflection():
    result = relation(
        [[1, 2], [3, 1]],
        [[5, 6], [6, 7]],
    )
    reflected = candidate(result, "FLIP_HORIZONTAL")

    assert "FLIP_HORIZONTAL" in result["recolor_matches"]
    assert reflected["recolor_mismatches"] == 0
    assert reflected["color_map"] == {
        ARC_COLOR_CHARS[1]: ARC_COLOR_CHARS[6],
        ARC_COLOR_CHARS[2]: ARC_COLOR_CHARS[5],
        ARC_COLOR_CHARS[3]: ARC_COLOR_CHARS[7],
    }


def test_ranks_approximate_transform_by_exact_mismatches():
    result = relation(
        [[1, 2], [3, 4]],
        [[3, 1], [9, 2]],
        allow_recolor=False,
    )

    assert result["best"] == ["ROTATE_90"]
    assert result["recolor_matches"] == []
    assert candidate(result, "ROTATE_90")["exact_mismatches"] == 1
    assert "recolor_mismatches" not in candidate(result, "ROTATE_90")


def test_marks_dimension_incompatible_transforms_without_scoring_them():
    result = relation([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]])

    assert candidate(result, "IDENTITY")["compatible_shape"] is True
    rotated = candidate(result, "ROTATE_90")
    assert rotated == {
        "transform": "ROTATE_90",
        "shape": [3, 2],
        "compatible_shape": False,
    }


def test_handles_empty_frames_without_division_by_zero():
    result = relation([], [])

    assert len(result["exact_matches"]) == 8
    assert all(item["exact_match_ratio"] == 1.0 for item in result["candidates"])


def test_rejects_non_boolean_recolor_option():
    with pytest.raises(TypeError, match="expects a boolean"):
        relation([[0]], [[0]], allow_recolor=1)


def test_injected_frame_view_exposes_transform_relation():
    assert "def transform_relation(self, other" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_transform_relation(" in _SANDBOX_BOOTSTRAP
