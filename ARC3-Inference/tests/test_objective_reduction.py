from inference.agent.objective_reduction import (
    MAX_ARCHIVES,
    MAX_DEPTH,
    MAX_EVIDENCE_ITEMS,
    MAX_NODES,
    ObjectiveReducer,
)


def initialize(reducer: ObjectiveReducer) -> dict:
    return reducer.apply({
        "op": "initialize",
        "description": "Complete the level",
        "success_criterion": "level_completed",
    })


def test_disabled_reducer_rejects_without_mutation() -> None:
    reducer = ObjectiveReducer()
    result = initialize(reducer)
    assert result["ok"] is False
    assert result["error"]["code"] == "disabled"
    assert reducer.snapshot()["graph"]["root_id"] is None


def test_ordered_reduction_and_completion_roll_up() -> None:
    reducer = ObjectiveReducer(enabled=True)
    assert initialize(reducer)["active_objective_id"] == "obj-1"
    reduced = reducer.apply({
        "op": "reduce",
        "objective_id": "obj-1",
        "children": [
            {"description": "Learn controls", "success_criterion": "prediction works"},
            {"description": "Reach target", "success_criterion": "progress"},
        ],
    })
    assert reduced["active_path"] == ["obj-1", "obj-2"]
    assert reducer.apply({"op": "complete", "objective_id": "obj-2"})["active_objective_id"] == "obj-3"
    result = reducer.apply({"op": "complete", "objective_id": "obj-3", "evidence": ["progress"]})
    assert result["active_objective_id"] is None
    statuses = {node["id"]: node["status"] for node in result["graph"]["nodes"]}
    assert statuses == {"obj-1": "completed", "obj-2": "completed", "obj-3": "completed"}


def test_failure_blocks_until_leaf_is_revised() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    reducer.apply({"op": "fail", "objective_id": "obj-1", "evidence": ["contradiction"]})
    assert reducer.active_objective_id is None
    revised = reducer.apply({
        "op": "revise",
        "objective_id": "obj-1",
        "description": "Try a different goal",
    })
    assert revised["active_objective_id"] == "obj-1"
    assert revised["graph"]["nodes"][0]["revisions"]


def test_invalid_reduction_is_transactional() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    before = reducer.snapshot()["graph"]
    result = reducer.apply({
        "op": "reduce",
        "objective_id": "obj-1",
        "children": [{"description": "missing criterion"}],
    })
    assert result["error"]["code"] == "missing_field"
    assert reducer.snapshot()["graph"] == before


def test_node_limit_is_transactional_and_evidence_remains_bounded() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    before = reducer.snapshot()["graph"]
    rejected = reducer.apply(
        {
            "op": "reduce",
            "objective_id": "obj-1",
            "children": [
                {"description": f"child {index}", "success_criterion": "done"}
                for index in range(MAX_NODES)
            ],
        }
    )
    assert rejected["error"]["code"] == "node_limit"
    assert reducer.snapshot()["graph"] == before

    for index in range(MAX_EVIDENCE_ITEMS + 2):
        reducer.apply(
            {
                "op": "revise",
                "objective_id": "obj-1",
                "evidence": [f"evidence {index}"],
            }
        )
    node = reducer.snapshot()["graph"]["nodes"][0]
    assert len(node["evidence"]) == MAX_EVIDENCE_ITEMS
    assert node["evidence"][-1] == f"evidence {MAX_EVIDENCE_ITEMS + 1}"


def test_depth_limit_and_outcome_recording() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    active = "obj-1"
    for depth in range(MAX_DEPTH):
        result = reducer.apply({
            "op": "reduce",
            "objective_id": active,
            "children": [{"description": f"depth {depth}", "success_criterion": "done"}],
        })
        assert result["ok"]
        active = result["active_objective_id"]
    rejected = reducer.apply({
        "op": "reduce",
        "objective_id": active,
        "children": [{"description": "too deep", "success_criterion": "done"}],
    })
    assert rejected["error"]["code"] == "depth_limit"
    reducer.record_outcome(active, {"executed_count": 2, "outcome_class": "novel", "board_changed": True})
    node = next(node for node in reducer.snapshot()["graph"]["nodes"] if node["id"] == active)
    assert node["attempts"] == 2
    assert node["recent_outcomes"][-1]["outcome_class"] == "novel"


def test_reroot_archives_and_retention_is_bounded() -> None:
    reducer = ObjectiveReducer(enabled=True)
    for index in range(MAX_ARCHIVES + 2):
        initialize(reducer)
        reducer.apply({"op": "complete", "objective_id": reducer.active_objective_id})
    initialize(reducer)
    snapshot = reducer.snapshot()
    assert len(snapshot["archives"]) == MAX_ARCHIVES
    assert snapshot["graph"]["root_id"] is not None
