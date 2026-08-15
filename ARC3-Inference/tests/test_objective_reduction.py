from inference.agent.objective_reduction import (
    MAX_ARCHIVES,
    MAX_DEPTH,
    MAX_EVIDENCE_ITEMS,
    MAX_NODES,
    MAX_OPERATION_EVENTS,
    MAX_REVISIONS,
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
    assert reducer.apply(
        {"op": "complete", "objective_id": "obj-2", "evidence": ["controls learned"]}
    )["active_objective_id"] == "obj-3"
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


def test_objective_api_rejects_malformed_json_types_without_coercion() -> None:
    reducer = ObjectiveReducer(enabled=True)
    invalid_initialize = reducer.apply(
        {
            "op": "initialize",
            "description": {"unexpected": "object"},
            "success_criterion": "done",
        }
    )
    assert invalid_initialize["error"]["code"] == "invalid_field_type"
    assert reducer.snapshot()["graph"]["root_id"] is None

    initialize(reducer)
    before = reducer.snapshot()["graph"]
    cases = [
        (
            {"op": False},
            "invalid_operation",
        ),
        (
            {"op": "revise", "objective_id": "obj-1", "description": 42},
            "invalid_field_type",
        ),
        (
            {"op": "revise", "objective_id": "obj-1", "evidence": ["valid", 42]},
            "invalid_evidence",
        ),
        (
            {"op": "revise", "objective_id": "obj-1", "attempt_budget": True},
            "invalid_attempt_budget",
        ),
        (
            {"op": "revise", "objective_id": "obj-1", "attempt_budget": 3.5},
            "invalid_attempt_budget",
        ),
        (
            {"op": "revise", "objective_id": "obj-1", "attempt_budget": "3"},
            "invalid_attempt_budget",
        ),
        (
            {"op": "complete", "objective_id": 1, "evidence": ["done"]},
            "invalid_objective_id",
        ),
        (
            {
                "op": "revise",
                "objective_id": "obj-1",
                "request_id": 7,
                "description": "different",
            },
            "invalid_request_id",
        ),
    ]
    for update, expected_code in cases:
        result = reducer.apply(update)
        assert result["error"]["code"] == expected_code
        assert reducer.snapshot()["graph"] == before


def test_unexpected_operation_error_rolls_back_and_returns_structured_failure() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    before = reducer.snapshot()

    def explode(_update: dict) -> None:
        reducer._nodes["obj-1"].description = "partially mutated"
        reducer._archives.append({"corrupt": True})
        raise RuntimeError("host detail must not escape")

    reducer._reduce = explode  # type: ignore[method-assign]
    result = reducer.apply({
        "op": "reduce",
        "objective_id": "obj-1",
        "children": [{"description": "child", "success_criterion": "done"}],
    })

    assert result["ok"] is False
    assert result["error"] == {
        "code": "internal_error",
        "message": "The objective operation failed without changing the graph.",
    }
    after = reducer.snapshot()
    assert after["graph"] == before["graph"]
    assert after["archives"] == before["archives"]
    assert after["discarded_branches"] == before["discarded_branches"]
    assert after["operation_events"][-1]["error_code"] == "internal_error"


def test_completion_and_failure_require_evidence_or_an_outcome() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    before = reducer.snapshot()["graph"]
    rejected = reducer.apply({"op": "complete", "objective_id": "obj-1"})
    assert rejected["error"]["code"] == "evidence_required"
    assert reducer.snapshot()["graph"] == before

    reducer.record_outcome("obj-1", {"executed_count": 1, "board_changed": True})
    completed = reducer.apply({"op": "complete", "objective_id": "obj-1"})
    assert completed["ok"]


def test_completion_requires_supportive_outcome_or_explicit_evidence() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    reducer.record_outcome(
        "obj-1", {"executed": True, "outcome_class": "exact_noop", "board_changed": False}
    )
    assert reducer.snapshot()["active_completion_ready"] is False
    rejected = reducer.apply({"op": "complete", "objective_id": "obj-1"})
    assert rejected["error"]["code"] == "insufficient_completion_evidence"
    completed = reducer.apply(
        {"op": "complete", "objective_id": "obj-1", "evidence": ["criterion verified"]}
    )
    assert completed["ok"]

    contradicted = ObjectiveReducer(enabled=True)
    initialize(contradicted)
    contradicted.record_outcome(
        "obj-1",
        {
            "executed": True,
            "outcome_class": "novel",
            "novel_state": True,
            "prediction_result": {"status": "contradicted"},
        },
    )
    assert contradicted.snapshot()["active_completion_ready"] is False
    rejected = contradicted.apply({"op": "complete", "objective_id": "obj-1"})
    assert rejected["error"]["code"] == "insufficient_completion_evidence"

    supported = ObjectiveReducer(enabled=True)
    initialize(supported)
    supported.record_outcome(
        "obj-1",
        {
            "executed": True,
            "outcome_class": "revisit",
            "prediction_result": {"status": "supported"},
        },
    )
    assert supported.snapshot()["active_completion_ready"] is True
    assert supported.apply({"op": "complete", "objective_id": "obj-1"})["ok"]


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


def test_outcome_mutation_is_recorded_in_operation_journal() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    before_sequence = reducer.snapshot()["operation_events"][-1]["sequence"]

    reducer.record_outcome(
        "obj-1",
        {"executed_count": 2, "outcome_class": "unchanged", "board_changed": False},
    )

    event = reducer.snapshot()["operation_events"][-1]
    assert event == {
        "sequence": before_sequence + 1,
        "operation": "outcome",
        "request_id": None,
        "ok": True,
        "error_code": None,
        "objective_id": "obj-1",
        "before_active_objective_id": "obj-1",
        "after_active_objective_id": "obj-1",
        "before_blocking_objective_id": None,
        "after_blocking_objective_id": None,
        "discarded_branch_count": 0,
        "outcome_class": "unchanged",
        "executed_count": 2,
        "executed": True,
        "requested_count": 1,
        "rejection_code": None,
        "prediction_status": None,
    }


def test_rejected_action_requests_trigger_review_without_inflating_attempts() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    for _ in range(3):
        reducer.record_outcome(
            "obj-1",
            {
                "executed": False,
                "requested_count": 1,
                "executed_count": 0,
                "error": {"code": "cycle_guard", "message": "cycle blocked"},
            },
        )

    snapshot = reducer.snapshot()
    node = snapshot["graph"]["nodes"][0]
    assert node["attempts"] == 0
    assert node["attempts_since_revision"] == 0
    assert node["rejected_action_requests"] == 3
    assert node["rejection_streak"] == 3
    assert node["recent_outcomes"][-1]["executed"] is False
    assert snapshot["active_objective_id"] is None
    assert snapshot["blocking_objective_id"] == "obj-1"
    assert "3 consecutive action requests" in snapshot["blocking_reason"]
    assert snapshot["operation_events"][-1]["rejection_code"] == "cycle_guard"

    rejected_completion = reducer.apply({"op": "complete", "objective_id": "obj-1"})
    assert rejected_completion["error"]["code"] == "evidence_required"
    revised = reducer.apply(
        {"op": "revise", "objective_id": "obj-1", "description": "Try another action"}
    )
    revised_node = revised["graph"]["nodes"][0]
    assert revised["active_objective_id"] == "obj-1"
    assert revised_node["rejection_streak"] == 0
    assert revised_node["rejected_action_requests"] == 3


def test_repeated_prediction_contradictions_force_early_review() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)

    def record(status: str) -> None:
        reducer.record_outcome(
            "obj-1",
            {
                "executed": True,
                "executed_count": 1,
                "board_changed": True,
                "novel_state": True,
                "outcome_class": "novel",
                "prediction_result": {"status": status},
            },
        )

    record("contradicted")
    record("supported")
    assert reducer.snapshot()["graph"]["nodes"][0]["prediction_contradiction_streak"] == 0
    record("contradicted")
    record("contradicted")

    snapshot = reducer.snapshot()
    node = snapshot["graph"]["nodes"][0]
    assert node["attempts"] == 4
    assert node["prediction_contradictions"] == 3
    assert node["prediction_contradiction_streak"] == 2
    assert snapshot["active_objective_id"] is None
    assert snapshot["blocking_objective_id"] == "obj-1"
    assert "2 consecutive strategy predictions" in snapshot["blocking_reason"]
    assert snapshot["operation_events"][-1]["prediction_status"] == "contradicted"

    revised = reducer.apply(
        {"op": "revise", "objective_id": "obj-1", "description": "Use new hypothesis"}
    )
    revised_node = revised["graph"]["nodes"][0]
    assert revised_node["prediction_contradiction_streak"] == 0
    assert revised_node["prediction_contradictions"] == 3


def test_review_cannot_be_bypassed_with_a_non_material_revision() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    for _ in range(3):
        reducer.record_outcome(
            "obj-1", {"outcome_class": "exact_noop", "board_changed": False}
        )
    before = reducer.snapshot()["graph"]

    evidence_only = reducer.apply(
        {"op": "revise", "objective_id": "obj-1", "evidence": ["still blocked"]}
    )
    same_text = reducer.apply(
        {
            "op": "revise",
            "objective_id": "obj-1",
            "description": "Complete the level",
            "evidence": ["same plan"],
        }
    )

    assert evidence_only["error"]["code"] == "non_material_revision"
    assert same_text["error"]["code"] == "non_material_revision"
    assert reducer.snapshot()["graph"] == before

    changed = reducer.apply(
        {
            "op": "revise",
            "objective_id": "obj-1",
            "success_criterion": "A different measurable condition",
        }
    )
    node = changed["graph"]["nodes"][0]
    assert changed["active_objective_id"] == "obj-1"
    assert node["attempts_since_revision"] == 0
    assert node["unproductive_streak"] == 0


def test_evidence_only_revision_preserves_pending_leaf_attempt_window() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    reducer.record_outcome(
        "obj-1", {"outcome_class": "novel", "board_changed": True}
    )

    result = reducer.apply(
        {"op": "revise", "objective_id": "obj-1", "evidence": ["new observation"]}
    )
    node = result["graph"]["nodes"][0]
    assert result["ok"]
    assert node["evidence"] == ["new observation"]
    assert node["attempts_since_revision"] == 1
    assert node["revisions"] == []


def test_material_revision_lifetime_is_bounded_transactionally() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    for index in range(MAX_REVISIONS):
        result = reducer.apply(
            {
                "op": "revise",
                "objective_id": "obj-1",
                "description": f"Plan variant {index}",
            }
        )
        assert result["ok"]

    before = reducer.snapshot()["graph"]
    rejected = reducer.apply(
        {
            "op": "revise",
            "objective_id": "obj-1",
            "description": "One variant too many",
        }
    )
    assert rejected["error"]["code"] == "revision_limit"
    assert reducer.snapshot()["graph"] == before
    node = before["nodes"][0]
    assert node["revision_count"] == MAX_REVISIONS
    assert node["revision_limit"] == MAX_REVISIONS
    assert len(node["revisions"]) == MAX_REVISIONS

    evidence_only = reducer.apply(
        {"op": "revise", "objective_id": "obj-1", "evidence": ["final observation"]}
    )
    node = evidence_only["graph"]["nodes"][0]
    assert evidence_only["ok"]
    assert node["revision_count"] == MAX_REVISIONS
    assert node["attempts_since_revision"] == 0


def test_repeated_no_progress_blocks_leaf_until_revision() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    for _ in range(3):
        reducer.record_outcome(
            "obj-1",
            {"executed_count": 1, "outcome_class": "exact_noop", "board_changed": False},
        )

    blocked = reducer.snapshot()
    assert blocked["active_objective_id"] is None
    assert blocked["blocking_objective_id"] == "obj-1"
    assert blocked["blocking_status"] == "pending"
    assert "3 consecutive actions" in blocked["blocking_reason"]

    revised = reducer.apply(
        {
            "op": "revise",
            "objective_id": "obj-1",
            "description": "Try a new mechanism",
        }
    )
    assert revised["active_objective_id"] == "obj-1"
    node = revised["graph"]["nodes"][0]
    assert node["unproductive_streak"] == 0
    assert node["review_required"] is False


def test_attempt_budget_forces_review_despite_productive_exploration() -> None:
    reducer = ObjectiveReducer(enabled=True)
    created = reducer.apply(
        {
            "op": "initialize",
            "description": "Find the mechanism",
            "success_criterion": "Mechanism confirmed",
            "attempt_budget": 2,
        }
    )
    assert created["ok"]
    for _ in range(2):
        reducer.record_outcome(
            "obj-1",
            {
                "executed_count": 1,
                "outcome_class": "novel",
                "novel_state": True,
                "board_changed": True,
            },
        )

    blocked = reducer.snapshot()
    node = blocked["graph"]["nodes"][0]
    assert blocked["active_objective_id"] is None
    assert blocked["blocking_objective_id"] == "obj-1"
    assert "Attempt budget exhausted (2/2)" in blocked["blocking_reason"]
    assert node["attempts"] == 2
    assert node["attempts_since_revision"] == 2

    revised = reducer.apply(
        {
            "op": "revise",
            "objective_id": "obj-1",
            "attempt_budget": 3,
            "evidence": ["Narrowed the search space"],
        }
    )
    node = revised["graph"]["nodes"][0]
    assert revised["active_objective_id"] == "obj-1"
    assert node["attempt_budget"] == 3
    assert node["attempts_since_revision"] == 0
    assert node["attempts"] == 2
    assert node["revisions"][-1]["attempt_budget"] == 2


def test_attempt_budget_validation_is_transactional_for_children() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    before = reducer.snapshot()["graph"]
    rejected = reducer.apply(
        {
            "op": "reduce",
            "objective_id": "obj-1",
            "children": [
                {
                    "description": "Probe",
                    "success_criterion": "done",
                    "attempt_budget": 0,
                }
            ],
        }
    )
    assert rejected["error"]["code"] == "invalid_attempt_budget"
    assert reducer.snapshot()["graph"] == before


def test_reviewed_leaf_can_be_completed_or_failed_from_final_outcome() -> None:
    for operation, expected_status in (("complete", "completed"), ("fail", "failed")):
        reducer = ObjectiveReducer(enabled=True)
        reducer.apply(
            {
                "op": "initialize",
                "description": "Test hypothesis",
                "success_criterion": "Outcome decides it",
                "attempt_budget": 1,
            }
        )
        reducer.record_outcome(
            "obj-1",
            {"outcome_class": "novel", "novel_state": True, "board_changed": True},
        )
        assert reducer.snapshot()["blocking_objective_id"] == "obj-1"
        resolved = reducer.apply({"op": operation, "objective_id": "obj-1"})
        assert resolved["ok"]
        assert resolved["graph"]["nodes"][0]["status"] == expected_status
        if operation == "complete":
            assert reducer.snapshot()["blocking_objective_id"] is None


def test_replan_preserves_completed_prefix_and_audits_removed_suffix() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    reducer.apply(
        {
            "op": "reduce",
            "objective_id": "obj-1",
            "children": [
                {"description": "Orient", "success_criterion": "scene understood"},
                {"description": "Navigate", "success_criterion": "target reached"},
                {"description": "Confirm", "success_criterion": "progress"},
            ],
        }
    )
    reducer.apply(
        {"op": "complete", "objective_id": "obj-2", "evidence": ["scene mapped"]}
    )
    reducer.apply(
        {"op": "fail", "objective_id": "obj-3", "evidence": ["no avatar exists"]}
    )

    replanned = reducer.apply(
        {
            "op": "replan",
            "objective_id": "obj-1",
            "reason": "The level is a selector puzzle",
            "children": [
                {"description": "Model selector", "success_criterion": "transition predicted"},
                {"description": "Choose goal", "success_criterion": "progress"},
            ],
        }
    )

    assert replanned["ok"]
    assert replanned["active_objective_id"] == "obj-5"
    assert replanned["discarded_branches"][-1]["target_id"] == "obj-1"
    nodes = {node["id"]: node for node in reducer.snapshot()["graph"]["nodes"]}
    assert set(nodes) == {"obj-1", "obj-2", "obj-5", "obj-6"}
    assert nodes["obj-1"]["children"] == ["obj-2", "obj-5", "obj-6"]
    audit = reducer.snapshot()["discarded_branches"][-1]
    assert audit["removed_roots"] == ["obj-3", "obj-4"]
    assert {node["id"] for node in audit["nodes"]} == {"obj-3", "obj-4"}
    archive = reducer.archive(reason="reset")
    assert archive["discarded_branches"][-1]["reason"] == "The level is a selector puzzle"
    assert reducer.snapshot()["discarded_branches"] == []


def test_replan_rejects_non_focus_target_transactionally() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    reducer.apply(
        {
            "op": "reduce",
            "objective_id": "obj-1",
            "children": [
                {"description": "First", "success_criterion": "done"},
                {"description": "Future", "success_criterion": "done"},
            ],
        }
    )
    before = reducer.snapshot()
    rejected = reducer.apply(
        {
            "op": "replan",
            "objective_id": "obj-3",
            "reason": "premature change",
            "children": [{"description": "Replacement", "success_criterion": "done"}],
        }
    )
    assert rejected["error"]["code"] == "not_on_focus_path"
    assert reducer.snapshot()["graph"] == before["graph"]
    assert reducer.snapshot()["discarded_branches"] == []


def test_request_id_makes_replan_idempotent_and_detects_conflicts() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    reducer.apply(
        {
            "op": "reduce",
            "objective_id": "obj-1",
            "children": [{"description": "Old", "success_criterion": "done"}],
        }
    )
    update = {
        "request_id": "replan-v1",
        "op": "replan",
        "objective_id": "obj-1",
        "reason": "new evidence",
        "children": [{"description": "New", "success_criterion": "done"}],
    }
    first = reducer.apply(update)
    event_count = len(reducer.snapshot()["operation_events"])
    replayed = reducer.apply(update)

    assert first["ok"] and replayed["ok"]
    assert first["replayed"] is False
    assert replayed["replayed"] is True
    assert replayed["event_sequence"] == first["event_sequence"]
    assert replayed["graph"] == first["graph"]
    assert len(reducer.snapshot()["operation_events"]) == event_count
    assert len(reducer.snapshot()["discarded_branches"]) == 1

    conflict = reducer.apply(
        {
            **update,
            "children": [{"description": "Different", "success_criterion": "done"}],
        }
    )
    assert conflict["error"]["code"] == "idempotency_conflict"
    assert conflict["replayed"] is False
    assert reducer.snapshot()["graph"] == first["graph"]
    assert reducer.snapshot()["operation_events"][-1]["error_code"] == "idempotency_conflict"


def test_operation_journal_is_bounded_and_moves_into_archive() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    for index in range(MAX_OPERATION_EVENTS + 3):
        reducer.apply({"op": f"unknown-{index}"})
    events = reducer.snapshot()["operation_events"]
    assert len(events) == MAX_OPERATION_EVENTS
    assert events[-1]["ok"] is False
    assert events[-1]["error_code"] == "unknown_operation"

    archive = reducer.archive(reason="test")
    assert archive["operation_events"][-1]["operation"] == "archive"
    assert reducer.snapshot()["operation_events"] == []


def test_reroot_archives_and_retention_is_bounded() -> None:
    reducer = ObjectiveReducer(enabled=True)
    for index in range(MAX_ARCHIVES + 2):
        initialize(reducer)
        reducer.apply(
            {
                "op": "complete",
                "objective_id": reducer.active_objective_id,
                "evidence": ["done"],
            }
        )
    initialize(reducer)
    snapshot = reducer.snapshot()
    assert len(snapshot["archives"]) == MAX_ARCHIVES
    assert all(item["successful"] for item in snapshot["archives"])
    assert snapshot["graph"]["root_id"] is not None


def test_successful_archive_marks_root_complete_without_rewriting_node_history() -> None:
    reducer = ObjectiveReducer(enabled=True)
    initialize(reducer)
    archive = reducer.archive(reason="level_transition", successful=True)
    assert archive["successful"] is True
    assert archive["root_status"] == "completed"
    assert archive["nodes"][0]["status"] == "pending"
