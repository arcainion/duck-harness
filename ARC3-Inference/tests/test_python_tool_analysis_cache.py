from __future__ import annotations

from inference.agent.python_tool_sandbox import _SANDBOX_BOOTSTRAP
from inference.utils.grid_utils import ARC_COLOR_CHARS


def sandbox_namespace():
    namespace = {"__name__": "sandbox_bootstrap_test"}
    exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)
    namespace["COLOR_CHARS"] = ARC_COLOR_CHARS
    return namespace


def frame(namespace, grid=None):
    grid = grid or [[0, 1, 0]]
    return namespace["FrameView"](
        ascii="",
        step=1,
        level=1,
        shape=(len(grid), len(grid[0])),
        grid=grid,
    )


def test_repeated_object_analysis_executes_once_and_returns_independent_copy():
    namespace = sandbox_namespace()
    original = namespace["_bounded_frame_objects"]
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    namespace["_bounded_frame_objects"] = counted
    view = frame(namespace)
    first = view.objects(background=ARC_COLOR_CHARS[0])
    first["objects"][0]["size"] = 999
    second = view.objects(background=ARC_COLOR_CHARS[0])

    assert calls == 1
    assert second["objects"][0]["size"] == 1
    assert first is not second


def test_changed_arguments_create_distinct_cache_entries():
    namespace = sandbox_namespace()
    original = namespace["_bounded_frame_objects"]
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    namespace["_bounded_frame_objects"] = counted
    view = frame(namespace)
    view.objects(background=ARC_COLOR_CHARS[0], diagonal=False)
    view.objects(background=ARC_COLOR_CHARS[0], diagonal=True)

    assert calls == 2
    assert len(view._analysis_cache) == 2


def test_equivalent_list_and_tuple_arguments_share_reachability_cache():
    namespace = sandbox_namespace()
    original = namespace["_bounded_reachable_region"]
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    namespace["_bounded_reachable_region"] = counted
    view = frame(namespace, [[0, 0], [0, 1]])
    view.reachable_region([0, 0], passable=[ARC_COLOR_CHARS[0]])
    view.reachable_region((0, 0), passable=(ARC_COLOR_CHARS[0],))

    assert calls == 1


def test_analysis_caches_are_isolated_between_frames():
    namespace = sandbox_namespace()
    original = namespace["_bounded_frame_symmetry"]
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    namespace["_bounded_frame_symmetry"] = counted
    first = frame(namespace)
    second = frame(namespace)
    first.symmetry()
    second.symmetry()

    assert calls == 2
    assert first._analysis_cache is not second._analysis_cache


def test_cache_is_bounded_to_sixty_four_entries():
    namespace = sandbox_namespace()
    view = frame(namespace)

    for limit in range(70):
        view.symmetry(sample_limit=limit)

    assert len(view._analysis_cache) == 64
