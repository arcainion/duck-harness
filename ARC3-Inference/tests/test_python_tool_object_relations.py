from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_object_relations,
    _SANDBOX_BOOTSTRAP,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def relations(grid, **kwargs):
    return _bounded_frame_object_relations(
        grid,
        shape=(len(grid), len(grid[0]) if grid else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def test_reports_pairwise_direction_overlap_and_gap():
    result = relations(
        [[1, 0, 2], [1, 0, 2]],
        background=ARC_COLOR_CHARS[0],
    )

    assert result["object_count"] == 2
    assert result["relation_count"] == 1
    pair = result["relations"][0]
    assert pair["horizontal"] == "LEFT"
    assert pair["vertical"] == "OVERLAP"
    assert pair["row_overlap"] == 2
    assert pair["column_overlap"] == 0
    assert pair["row_gap"] == 0
    assert pair["column_gap"] == 1
    assert pair["bbox_gap"] == 1
    assert pair["centroid_delta"] == [0.0, 2.0]


def test_identifies_repeated_translation_invariant_objects():
    result = relations(
        [[1, 0, 1]],
        background=ARC_COLOR_CHARS[0],
    )

    pair = result["relations"][0]
    assert pair["same_size"] is True
    assert pair["same_signature"] is True
    assert pair["shared_colors"] == [ARC_COLOR_CHARS[1]]
    assert result["counts"]["same_signature"] == 1


def test_detects_bounding_box_containment():
    result = relations(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 2, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        background=ARC_COLOR_CHARS[0],
    )

    pair = result["relations"][0]
    assert pair["containment"] == "FIRST_CONTAINS_SECOND"
    assert pair["row_overlap"] == 1
    assert pair["column_overlap"] == 1
    assert result["counts"]["bbox_contains"] == 1


def test_object_limit_marks_pair_set_as_incomplete():
    result = relations(
        [[1, 0, 2, 0, 3]],
        background=ARC_COLOR_CHARS[0],
        object_limit=2,
    )

    assert result["object_count"] == 3
    assert len(result["objects"]) == 2
    assert result["truncated_objects"] == 1
    assert result["complete_object_set"] is False
    assert result["relation_count"] == 1


def test_relation_limit_preserves_aggregate_pair_counts():
    result = relations(
        [[1, 0, 2, 0, 3]],
        background=ARC_COLOR_CHARS[0],
        relation_limit=1,
    )

    assert result["relation_count"] == 3
    assert len(result["relations"]) == 1
    assert result["truncated_relations"] == 2
    assert result["counts"]["same_size"] == 3


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"background": "?"}, ValueError),
        ({"diagonal": 1}, TypeError),
        ({"object_limit": True}, TypeError),
        ({"relation_limit": True}, TypeError),
    ],
)
def test_rejects_invalid_relation_options(kwargs, error):
    with pytest.raises(error):
        relations([[1]], **kwargs)


def test_injected_frame_view_exposes_object_relations():
    assert "def object_relations(" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_object_relations(" in _SANDBOX_BOOTSTRAP
