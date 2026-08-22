from __future__ import annotations

import pytest

from inference.agent.python_tool_sandbox import (
    _bounded_frame_object_changes,
    _SANDBOX_BOOTSTRAP,
)
from inference.utils.grid_utils import ARC_COLOR_CHARS


def track(before, after, **kwargs):
    return _bounded_frame_object_changes(
        before,
        after,
        before_shape=(len(before), len(before[0])),
        after_shape=(len(after), len(after[0])),
        color_chars=ARC_COLOR_CHARS,
        **kwargs,
    )


def test_tracks_translated_multicolor_object_with_stable_signature():
    result = track(
        [[0, 1, 2, 0]],
        [[0, 0, 1, 2]],
        background=ARC_COLOR_CHARS[0],
    )

    assert result["counts"] == {
        "matched": 1,
        "moved": 1,
        "unchanged": 0,
        "added": 0,
        "removed": 0,
    }
    event = result["events"][0]
    assert event["type"] == "moved"
    assert event["delta"] == [0, 1]
    assert event["distance"] == 1
    assert event["before"]["signature"] == event["after"]["signature"]


def test_distinguishes_unchanged_from_color_changed_objects():
    unchanged = track(
        [[0, 1, 0]], [[0, 1, 0]], background=ARC_COLOR_CHARS[0]
    )
    changed = track([[0, 1, 0]], [[0, 2, 0]], background=ARC_COLOR_CHARS[0])

    assert unchanged["counts"]["unchanged"] == 1
    assert unchanged["events"][0]["delta"] == [0, 0]
    assert changed["counts"]["matched"] == 0
    assert {event["type"] for event in changed["events"]} == {"added", "removed"}


def test_pairs_repeated_identical_objects_by_nearest_centroid():
    result = track(
        [[1, 0, 0, 0, 1]],
        [[0, 1, 0, 1, 0]],
        background=ARC_COLOR_CHARS[0],
    )

    assert result["counts"]["moved"] == 2
    assert [event["delta"] for event in result["events"]] == [[0, 1], [0, -1]]


def test_infers_one_shared_background_across_both_frames():
    result = track([[0, 0, 1]], [[0, 1, 1]])

    assert result["background"] == ARC_COLOR_CHARS[0]
    assert result["background_inferred"] is True
    assert result["counts"]["removed"] == 1
    assert result["counts"]["added"] == 1


def test_event_limit_preserves_full_counts_and_reports_truncation():
    result = track(
        [[1, 0, 2, 0]],
        [[3, 0, 4, 0]],
        background=ARC_COLOR_CHARS[0],
        limit=1,
    )

    assert result["counts"]["removed"] == 2
    assert result["counts"]["added"] == 2
    assert len(result["events"]) == 1
    assert result["truncated_events"] == 3


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"diagonal": 1}, TypeError),
        ({"limit": True}, TypeError),
        ({"background": "?"}, ValueError),
    ],
)
def test_rejects_invalid_tracking_options(kwargs, error):
    with pytest.raises(error):
        track([[0]], [[0]], **kwargs)


def test_injected_frame_view_exposes_object_tracking():
    assert "def track_objects(self, other" in _SANDBOX_BOOTSTRAP
    assert "_bounded_frame_object_changes(" in _SANDBOX_BOOTSTRAP
