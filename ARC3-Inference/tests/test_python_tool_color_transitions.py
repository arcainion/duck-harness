from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_color_transitions,
    _SANDBOX_BOOTSTRAP,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def transitions(before, after, **kwargs):
    return _bounded_frame_color_transitions(
        before,
        after,
        before_shape=(len(before), len(before[0]) if before else 0),
        after_shape=(len(after), len(after[0]) if after else 0),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def test_aggregates_deterministic_recoloring_rule():
    result = transitions(
        [[1, 1, 0], [1, 0, 0]],
        [[2, 2, 0], [2, 0, 0]],
    )

    assert result["changed_cells"] == 3
    assert result["unchanged_cells"] == 3
    assert result["transition_types"] == 1
    transition = result["transitions"][0]
    assert transition["before"] == ARC_COLOR_CHARS[1]
    assert transition["after"] == ARC_COLOR_CHARS[2]
    assert transition["count"] == 3
    assert transition["bbox"] == [0, 0, 1, 1]
    assert result["source_mappings"] == [
        {
            "source": ARC_COLOR_CHARS[1],
            "targets": [{"symbol": ARC_COLOR_CHARS[2], "count": 3}],
            "dominant_target": ARC_COLOR_CHARS[2],
            "deterministic": True,
        }
    ]


def test_reports_branching_source_mapping_and_dominant_target():
    result = transitions([[1, 1, 1]], [[2, 2, 3]])
    mapping = result["source_mappings"][0]

    assert mapping["deterministic"] is False
    assert mapping["dominant_target"] == ARC_COLOR_CHARS[2]
    assert mapping["targets"] == [
        {"symbol": ARC_COLOR_CHARS[2], "count": 2},
        {"symbol": ARC_COLOR_CHARS[3], "count": 1},
    ]


def test_can_include_unchanged_transitions():
    result = transitions([[1, 0]], [[2, 0]], include_unchanged=True)

    assert result["transition_types"] == 2
    assert {(item["before"], item["after"]) for item in result["transitions"]} == {
        (ARC_COLOR_CHARS[0], ARC_COLOR_CHARS[0]),
        (ARC_COLOR_CHARS[1], ARC_COLOR_CHARS[2]),
    }


def test_shape_growth_is_reported_as_appearance_from_absence():
    result = _bounded_frame_color_transitions(
        [[1]],
        [[1, 2]],
        before_shape=(1, 1),
        after_shape=(1, 2),
        color_chars=ARC_COLOR_CHARS,
    )

    assert result["shape_changed"] is True
    assert result["changed_cells"] == 1
    assert result["transitions"][0]["before"] is None
    assert result["transitions"][0]["after"] == ARC_COLOR_CHARS[2]


def test_limits_preserve_transition_and_mapping_aggregates():
    result = transitions(
        [[1, 2]],
        [[3, 4]],
        limit=1,
        cell_limit=0,
    )

    assert result["transition_types"] == 2
    assert len(result["transitions"]) == 1
    assert result["truncated_transitions"] == 1
    assert len(result["source_mappings"]) == 2
    assert result["sampled_cells"] == 0


def test_invalid_internal_values_are_letter_coded_as_unknown():
    result = _bounded_frame_color_transitions(
        [[True]],
        [[1]],
        before_shape=(1, 1),
        after_shape=(1, 1),
        color_chars=ARC_COLOR_CHARS,
    )

    assert result["transitions"][0]["before"] == "?"
    assert result["transitions"][0]["after"] == ARC_COLOR_CHARS[1]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"include_unchanged": 1}, TypeError),
        ({"limit": True}, TypeError),
        ({"cell_limit": True}, TypeError),
    ],
)
def test_rejects_invalid_transition_options(kwargs, error):
    with pytest.raises(error):
        transitions([[1]], [[1]], **kwargs)


def test_injected_views_expose_color_transitions():
    assert "def color_transitions(" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_color_transitions(" in _SANDBOX_BOOTSTRAP
    assert "return self.after_frame.color_transitions(" in _SANDBOX_BOOTSTRAP
