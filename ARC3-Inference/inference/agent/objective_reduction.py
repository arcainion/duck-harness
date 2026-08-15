"""Deterministic ordered objective reduction for tool-using agents."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


MAX_DEPTH = 4
MAX_NODES = 32
MAX_ARCHIVES = 5
MAX_TEXT_CHARS = 280
MAX_EVIDENCE_ITEMS = 5
MAX_EVIDENCE_CHARS = 200
MAX_RECENT_OUTCOMES = 5
MAX_REVISIONS = 5
UNPRODUCTIVE_REVIEW_LIMIT = 3
REJECTED_ACTION_REVIEW_LIMIT = 3
PREDICTION_CONTRADICTION_REVIEW_LIMIT = 2
MAX_DISCARDED_BRANCHES = 5
MAX_OPERATION_EVENTS = 50
MAX_IDEMPOTENCY_KEYS = 32
MAX_REQUEST_ID_CHARS = 80
DEFAULT_ATTEMPT_BUDGET = 6
MAX_ATTEMPT_BUDGET = 20


class ObjectiveValidationError(ValueError):
    """A stable, model-facing validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ObjectiveNode:
    id: str
    description: str
    success_criterion: str
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    status: str = "pending"
    evidence: list[str] = field(default_factory=list)
    attempts: int = 0
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    revision_count: int = 0
    attempt_budget: int = DEFAULT_ATTEMPT_BUDGET
    attempts_since_revision: int = 0
    unproductive_streak: int = 0
    review_required: bool = False
    blocking_reason: str | None = None
    rejected_action_requests: int = 0
    rejection_streak: int = 0
    prediction_contradictions: int = 0
    prediction_contradiction_streak: int = 0

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "success_criterion": self.success_criterion,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "status": self.status,
            "evidence": list(self.evidence),
            "attempts": self.attempts,
            "recent_outcomes": deepcopy(self.recent_outcomes),
            "revisions": deepcopy(self.revisions),
            "revision_count": self.revision_count,
            "revision_limit": MAX_REVISIONS,
            "attempt_budget": self.attempt_budget,
            "attempts_since_revision": self.attempts_since_revision,
            "unproductive_streak": self.unproductive_streak,
            "review_required": self.review_required,
            "blocking_reason": self.blocking_reason,
            "rejected_action_requests": self.rejected_action_requests,
            "rejection_streak": self.rejection_streak,
            "prediction_contradictions": self.prediction_contradictions,
            "prediction_contradiction_streak": self.prediction_contradiction_streak,
        }


class ObjectiveReducer:
    """Own and validate a single ordered AND objective tree."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._nodes: dict[str, ObjectiveNode] = {}
        self._root_id: str | None = None
        self._next_id = 1
        self._archives: list[dict[str, Any]] = []
        self._discarded_branches: list[dict[str, Any]] = []
        self._operation_events: list[dict[str, Any]] = []
        self._event_sequence = 0
        self._idempotency_results: dict[str, dict[str, Any]] = {}
        self._last_result: dict[str, Any] | None = None

    @property
    def active_node(self) -> ObjectiveNode | None:
        if self._root_id is None:
            return None
        return self._active_from(self._root_id)

    @property
    def active_objective_id(self) -> str | None:
        node = self.active_node
        return node.id if node is not None else None

    def _active_from(self, node_id: str) -> ObjectiveNode | None:
        node = self._nodes[node_id]
        if node.status != "pending" or node.review_required:
            return None
        if not node.children:
            return node
        for child_id in node.children:
            child = self._nodes[child_id]
            if child.status == "completed":
                continue
            if child.status == "failed":
                return None
            return self._active_from(child_id)
        return None

    @property
    def blocking_node(self) -> ObjectiveNode | None:
        if self._root_id is None:
            return None
        return self._blocking_from(self._root_id)

    def _blocking_from(self, node_id: str) -> ObjectiveNode | None:
        node = self._nodes[node_id]
        if node.status == "failed" or node.review_required:
            return node
        if node.status != "pending":
            return None
        for child_id in node.children:
            child = self._nodes[child_id]
            if child.status == "completed":
                continue
            return self._blocking_from(child_id)
        return None

    def path_to(self, node_id: str | None) -> list[str]:
        node = self._nodes.get(str(node_id or ""))
        path: list[str] = []
        while node is not None:
            path.append(node.id)
            node = self._nodes.get(node.parent_id) if node.parent_id else None
        return list(reversed(path))

    def active_path(self) -> list[str]:
        active = self.active_node
        if active is None:
            return []
        return self.path_to(active.id)

    def active_path_descriptions(self) -> list[str]:
        return [self._nodes[node_id].description for node_id in self.active_path()]

    def snapshot(self) -> dict[str, Any]:
        active = self.active_node
        blocking = self.blocking_node
        blocking_path = self.path_to(blocking.id if blocking else None)
        return {
            "enabled": self.enabled,
            "active_objective_id": active.id if active else None,
            "active_path": self.active_path(),
            "active_path_descriptions": self.active_path_descriptions(),
            "active_success_criterion": active.success_criterion if active else None,
            "active_completion_ready": bool(
                active
                and (
                    active.evidence
                    or self._has_supportive_completion_outcome(active)
                )
            ),
            "blocking_objective_id": blocking.id if blocking else None,
            "blocking_status": blocking.status if blocking else None,
            "blocking_path": blocking_path,
            "blocking_path_descriptions": [self._nodes[node_id].description for node_id in blocking_path],
            "blocking_reason": blocking.blocking_reason if blocking else None,
            "graph": {
                "root_id": self._root_id,
                "nodes": [node.payload() for node in self._nodes.values()],
            },
            "archives": deepcopy(self._archives),
            "discarded_branches": deepcopy(self._discarded_branches),
            "operation_events": deepcopy(self._operation_events),
            "last_result": deepcopy(self._last_result),
        }

    def apply(self, update: dict[str, Any]) -> dict[str, Any]:
        raw_operation = update.get("op")
        operation = raw_operation.strip().lower() if isinstance(raw_operation, str) else ""
        before_active = self.active_objective_id
        before_blocking = self.blocking_node.id if self.blocking_node else None
        backup = (
            deepcopy(self._nodes),
            self._root_id,
            self._next_id,
            deepcopy(self._archives),
            deepcopy(self._discarded_branches),
            deepcopy(self._operation_events),
            self._event_sequence,
            deepcopy(self._idempotency_results),
            deepcopy(self._last_result),
        )
        request_id: str | None = None
        fingerprint = self._operation_fingerprint(update)
        try:
            if raw_operation is not None and not isinstance(raw_operation, str):
                raise ObjectiveValidationError(
                    "invalid_operation", "op must be a string."
                )
            request_id = self._request_id(update.get("request_id"))
            if request_id and request_id in self._idempotency_results:
                cached = self._idempotency_results[request_id]
                if cached["fingerprint"] != fingerprint:
                    raise ObjectiveValidationError(
                        "idempotency_conflict",
                        "request_id was already used for a different objective operation.",
                    )
                cached_result = cached["result"]
                cached_error = cached_result.get("error") or {}
                replayed = self._result(
                    bool(cached_result.get("ok")),
                    operation,
                    code=cached_error.get("code"),
                    message=cached_error.get("message"),
                )
                replayed["event_sequence"] = cached_result.get("event_sequence")
                replayed["replayed"] = True
                self._last_result = deepcopy(replayed)
                return replayed
            if not self.enabled:
                raise ObjectiveValidationError("disabled", "Objective reduction is not enabled.")
            if operation not in {
                "initialize",
                "reduce",
                "complete",
                "fail",
                "revise",
                "replan",
            }:
                raise ObjectiveValidationError("unknown_operation", "Unknown objective operation.")
            getattr(self, f"_{operation}")(update)
            result = self._result(True, operation)
        except ObjectiveValidationError as exc:
            (
                self._nodes,
                self._root_id,
                self._next_id,
                self._archives,
                self._discarded_branches,
                self._operation_events,
                self._event_sequence,
                self._idempotency_results,
                self._last_result,
            ) = backup
            result = self._result(False, operation, code=exc.code, message=exc.message)
        except Exception:
            (
                self._nodes,
                self._root_id,
                self._next_id,
                self._archives,
                self._discarded_branches,
                self._operation_events,
                self._event_sequence,
                self._idempotency_results,
                self._last_result,
            ) = backup
            result = self._result(
                False,
                operation,
                code="internal_error",
                message="The objective operation failed without changing the graph.",
            )
        event = self._record_operation_event(
            operation=operation,
            request_id=request_id,
            update=update,
            result=result,
            before_active=before_active,
            before_blocking=before_blocking,
        )
        result["event_sequence"] = event["sequence"]
        result["replayed"] = False
        if request_id and (result.get("error") or {}).get("code") != "idempotency_conflict":
            self._idempotency_results[request_id] = {
                "fingerprint": fingerprint,
                "result": deepcopy(result),
            }
            while len(self._idempotency_results) > MAX_IDEMPOTENCY_KEYS:
                self._idempotency_results.pop(next(iter(self._idempotency_results)))
        self._last_result = deepcopy(result)
        return result

    def _request_id(self, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ObjectiveValidationError(
                "invalid_request_id", "request_id must be a string."
            )
        request_id = value.strip()
        if not request_id:
            raise ObjectiveValidationError(
                "invalid_request_id", "request_id must not be empty when provided."
            )
        if len(request_id) > MAX_REQUEST_ID_CHARS:
            raise ObjectiveValidationError(
                "invalid_request_id",
                f"request_id may contain at most {MAX_REQUEST_ID_CHARS} characters.",
            )
        return request_id

    def _operation_fingerprint(self, update: dict[str, Any]) -> str:
        return json.dumps(
            update,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )

    def _record_operation_event(
        self,
        *,
        operation: str,
        request_id: str | None,
        update: dict[str, Any],
        result: dict[str, Any],
        before_active: str | None,
        before_blocking: str | None,
    ) -> dict[str, Any]:
        self._event_sequence += 1
        error = result.get("error") or {}
        event = {
            "sequence": self._event_sequence,
            "operation": operation,
            "request_id": request_id,
            "ok": bool(result.get("ok")),
            "error_code": error.get("code"),
            "objective_id": str(update.get("objective_id") or "").strip() or None,
            "before_active_objective_id": before_active,
            "after_active_objective_id": self.active_objective_id,
            "before_blocking_objective_id": before_blocking,
            "after_blocking_objective_id": (
                self.blocking_node.id if self.blocking_node else None
            ),
            "discarded_branch_count": len(self._discarded_branches),
        }
        if operation == "outcome":
            event["outcome_class"] = update.get("outcome_class")
            event["executed_count"] = update.get("executed_count")
            event["executed"] = update.get("executed")
            event["requested_count"] = update.get("requested_count")
            event["rejection_code"] = update.get("rejection_code")
            event["prediction_status"] = update.get("prediction_status")
        self._operation_events.append(event)
        self._operation_events = self._operation_events[-MAX_OPERATION_EVENTS:]
        return event

    def _result(
        self,
        ok: bool,
        operation: str,
        *,
        code: str | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "ok": ok,
            "operation": operation,
            "active_objective_id": snapshot["active_objective_id"],
            "active_path": snapshot["active_path"],
            "blocking_objective_id": snapshot["blocking_objective_id"],
            "blocking_reason": snapshot["blocking_reason"],
            "graph": snapshot["graph"],
            "discarded_branches": snapshot["discarded_branches"],
            "error": None if ok else {"code": code, "message": message},
        }

    def _text(self, value: Any, field_name: str, *, required: bool = True) -> str:
        if value is None:
            text = ""
        elif not isinstance(value, str):
            raise ObjectiveValidationError(
                "invalid_field_type", f"{field_name} must be a string."
            )
        else:
            text = value.strip()
        if required and not text:
            raise ObjectiveValidationError("missing_field", f"{field_name} is required.")
        if len(text) > MAX_TEXT_CHARS:
            raise ObjectiveValidationError("text_too_long", f"{field_name} exceeds {MAX_TEXT_CHARS} characters.")
        return text

    def _evidence(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ObjectiveValidationError("invalid_evidence", "evidence must be a list of strings.")
        if len(value) > MAX_EVIDENCE_ITEMS:
            raise ObjectiveValidationError("too_much_evidence", f"evidence may contain at most {MAX_EVIDENCE_ITEMS} items.")
        evidence: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ObjectiveValidationError(
                    "invalid_evidence", "evidence must be a list of strings."
                )
            text = item.strip()
            if not text:
                continue
            if len(text) > MAX_EVIDENCE_CHARS:
                raise ObjectiveValidationError("evidence_too_long", f"evidence items may contain at most {MAX_EVIDENCE_CHARS} characters.")
            evidence.append(text)
        return evidence

    def _attempt_budget(
        self, value: Any, *, default: int = DEFAULT_ATTEMPT_BUDGET
    ) -> int:
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ObjectiveValidationError(
                "invalid_attempt_budget", "attempt_budget must be an integer."
            )
        budget = value
        if not 1 <= budget <= MAX_ATTEMPT_BUDGET:
            raise ObjectiveValidationError(
                "invalid_attempt_budget",
                f"attempt_budget must be between 1 and {MAX_ATTEMPT_BUDGET}.",
            )
        return budget

    def _new_node(
        self,
        description: str,
        success_criterion: str,
        parent_id: str | None = None,
        *,
        attempt_budget: int = DEFAULT_ATTEMPT_BUDGET,
    ) -> ObjectiveNode:
        if len(self._nodes) >= MAX_NODES:
            raise ObjectiveValidationError("node_limit", f"An objective graph may contain at most {MAX_NODES} nodes.")
        node_id = f"obj-{self._next_id}"
        self._next_id += 1
        node = ObjectiveNode(
            node_id,
            description,
            success_criterion,
            parent_id,
            attempt_budget=attempt_budget,
        )
        self._nodes[node_id] = node
        return node

    def _initialize(self, update: dict[str, Any]) -> None:
        description = self._text(update.get("description"), "description")
        criterion = self._text(update.get("success_criterion"), "success_criterion")
        attempt_budget = self._attempt_budget(update.get("attempt_budget"))
        if self._root_id is not None:
            root = self._nodes[self._root_id]
            if root.status == "pending":
                raise ObjectiveValidationError("open_root", "Complete or fail the current root before initializing another.")
            self.archive(reason="reroot")
        root = self._new_node(
            description, criterion, attempt_budget=attempt_budget
        )
        self._root_id = root.id

    def _require_node(self, value: Any) -> ObjectiveNode:
        if not isinstance(value, str):
            raise ObjectiveValidationError(
                "invalid_objective_id", "objective_id must be a string."
            )
        node_id = value.strip()
        node = self._nodes.get(node_id)
        if node is None:
            raise ObjectiveValidationError("unknown_objective", f"Unknown objective id: {node_id or '<empty>'}.")
        return node

    def _require_active(self, value: Any) -> ObjectiveNode:
        node = self._require_node(value)
        if node.id != self.active_objective_id:
            raise ObjectiveValidationError("not_active", "The operation must target the active objective leaf.")
        return node

    def _require_decidable_leaf(self, value: Any) -> ObjectiveNode:
        node = self._require_node(value)
        blocking = self.blocking_node
        is_reviewing = bool(
            blocking is not None
            and blocking.id == node.id
            and node.status == "pending"
            and node.review_required
        )
        if node.id != self.active_objective_id and not is_reviewing:
            raise ObjectiveValidationError(
                "not_active",
                "Complete or fail only the active leaf or the pending leaf currently blocked for review.",
            )
        if node.children:
            raise ObjectiveValidationError(
                "not_leaf", "Complete or fail an actionable objective leaf."
            )
        return node

    def _depth(self, node: ObjectiveNode) -> int:
        depth = 0
        while node.parent_id:
            depth += 1
            node = self._nodes[node.parent_id]
        return depth

    def _reduce(self, update: dict[str, Any]) -> None:
        parent = self._require_active(update.get("objective_id"))
        if parent.children:
            raise ObjectiveValidationError("already_reduced", "The active objective already has children.")
        raw_children = update.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise ObjectiveValidationError("invalid_children", "children must be a non-empty list.")
        if self._depth(parent) >= MAX_DEPTH:
            raise ObjectiveValidationError("depth_limit", f"Objective depth may not exceed {MAX_DEPTH}.")
        if len(self._nodes) + len(raw_children) > MAX_NODES:
            raise ObjectiveValidationError("node_limit", f"An objective graph may contain at most {MAX_NODES} nodes.")
        parsed: list[tuple[str, str, int]] = []
        for raw_child in raw_children:
            if not isinstance(raw_child, dict):
                raise ObjectiveValidationError("invalid_child", "Each child must be an object.")
            parsed.append(
                (
                    self._text(raw_child.get("description"), "description"),
                    self._text(
                        raw_child.get("success_criterion"), "success_criterion"
                    ),
                    self._attempt_budget(raw_child.get("attempt_budget")),
                )
            )
        parent.children = [
            self._new_node(
                desc, criterion, parent.id, attempt_budget=attempt_budget
            ).id
            for desc, criterion, attempt_budget in parsed
        ]

    def _complete(self, update: dict[str, Any]) -> None:
        node = self._require_decidable_leaf(update.get("objective_id"))
        evidence = self._evidence(update.get("evidence"))
        has_executed_outcome = any(
            outcome.get("executed", True) for outcome in node.recent_outcomes
        )
        if not evidence and not has_executed_outcome:
            raise ObjectiveValidationError(
                "evidence_required",
                "Complete an objective only with explicit evidence or a recorded action outcome.",
            )
        if not evidence and not self._has_supportive_completion_outcome(node):
            raise ObjectiveValidationError(
                "insufficient_completion_evidence",
                "The recorded outcomes do not support this success criterion; provide explicit evidence or continue testing.",
            )
        node.evidence.extend(evidence)
        node.evidence = node.evidence[-MAX_EVIDENCE_ITEMS:]
        node.status = "completed"
        node.review_required = False
        node.blocking_reason = None
        while node.parent_id:
            parent = self._nodes[node.parent_id]
            if not parent.children or any(self._nodes[child].status != "completed" for child in parent.children):
                break
            parent.status = "completed"
            node = parent

    def _has_supportive_completion_outcome(self, node: ObjectiveNode) -> bool:
        for outcome in node.recent_outcomes:
            if outcome.get("executed", True) is False:
                continue
            try:
                reward = float(outcome.get("reward") or 0.0)
            except (TypeError, ValueError):
                reward = 0.0
            if (
                outcome.get("level_completed")
                or outcome.get("run_complete")
                or reward > 0.0
            ):
                return True
            prediction_status = str(
                outcome.get("prediction_status") or ""
            ).strip().lower()
            if prediction_status == "supported":
                return True
            if prediction_status == "contradicted":
                continue
            outcome_class = str(outcome.get("outcome_class") or "").strip().lower()
            if outcome_class in {"level_progress", "novel"}:
                return True
            if not outcome_class and (
                outcome.get("novel_state") or outcome.get("board_changed")
            ):
                return True
        return False

    def _fail(self, update: dict[str, Any]) -> None:
        node = self._require_decidable_leaf(update.get("objective_id"))
        evidence = self._evidence(update.get("evidence"))
        has_executed_outcome = any(
            outcome.get("executed", True) for outcome in node.recent_outcomes
        )
        if not evidence and not has_executed_outcome:
            raise ObjectiveValidationError(
                "evidence_required",
                "Fail an objective only with explicit evidence or a recorded action outcome.",
            )
        node.evidence.extend(evidence)
        node.evidence = node.evidence[-MAX_EVIDENCE_ITEMS:]
        node.status = "failed"
        node.review_required = False
        node.blocking_reason = "The objective was explicitly failed; revise it before acting again."

    def _revise(self, update: dict[str, Any]) -> None:
        node = self._require_node(update.get("objective_id"))
        if node.children or node.status == "completed":
            raise ObjectiveValidationError("not_revisable", "Only pending or failed leaves may be revised.")
        description = self._text(update.get("description"), "description", required=False)
        criterion = self._text(update.get("success_criterion"), "success_criterion", required=False)
        evidence = self._evidence(update.get("evidence"))
        raw_attempt_budget = update.get("attempt_budget")
        attempt_budget = (
            self._attempt_budget(raw_attempt_budget, default=node.attempt_budget)
            if raw_attempt_budget is not None
            else node.attempt_budget
        )
        description_changed = bool(description and description != node.description)
        criterion_changed = bool(criterion and criterion != node.success_criterion)
        budget_changed = bool(
            raw_attempt_budget is not None and attempt_budget != node.attempt_budget
        )
        material_change = description_changed or criterion_changed or budget_changed
        if not material_change and not evidence:
            raise ObjectiveValidationError("empty_revision", "A revision must change text or add evidence.")
        if not material_change and (node.status == "failed" or node.review_required):
            raise ObjectiveValidationError(
                "non_material_revision",
                "Reopening a failed or reviewed objective requires a changed description, success criterion, or attempt budget.",
            )
        if not material_change:
            node.evidence.extend(evidence)
            node.evidence = node.evidence[-MAX_EVIDENCE_ITEMS:]
            return
        if node.revision_count >= MAX_REVISIONS:
            raise ObjectiveValidationError(
                "revision_limit",
                f"An objective leaf may be materially revised at most {MAX_REVISIONS} times; resolve it or replan its parent.",
            )
        node.revisions.append(
            {
                "description": node.description,
                "success_criterion": node.success_criterion,
                "attempt_budget": node.attempt_budget,
                "attempts_since_revision": node.attempts_since_revision,
                "rejection_streak": node.rejection_streak,
                "prediction_contradiction_streak": node.prediction_contradiction_streak,
            }
        )
        node.revisions = node.revisions[-MAX_REVISIONS:]
        node.revision_count += 1
        if description_changed:
            node.description = description
        if criterion_changed:
            node.success_criterion = criterion
        node.attempt_budget = attempt_budget
        node.evidence.extend(evidence)
        node.evidence = node.evidence[-MAX_EVIDENCE_ITEMS:]
        node.status = "pending"
        node.attempts_since_revision = 0
        node.unproductive_streak = 0
        node.rejection_streak = 0
        node.prediction_contradiction_streak = 0
        node.review_required = False
        node.blocking_reason = None

    def _subtree_ids(self, node_id: str) -> list[str]:
        result: list[str] = []
        stack = [node_id]
        while stack:
            current_id = stack.pop()
            result.append(current_id)
            stack.extend(reversed(self._nodes[current_id].children))
        return result

    def _replan(self, update: dict[str, Any]) -> None:
        target = self._require_node(update.get("objective_id"))
        focus = self.active_path()
        if not focus:
            blocking = self.blocking_node
            focus = self.path_to(blocking.id if blocking else None)
        if target.id not in focus:
            raise ObjectiveValidationError(
                "not_on_focus_path",
                "Replan an objective only on the current active or blocking path.",
            )
        if target.status != "pending" or not target.children:
            raise ObjectiveValidationError(
                "not_replannable",
                "Replan requires a pending objective with an existing decomposition.",
            )

        reason = self._text(update.get("reason"), "reason")
        raw_children = update.get("children")
        if not isinstance(raw_children, list) or not raw_children:
            raise ObjectiveValidationError(
                "invalid_children", "children must be a non-empty list."
            )
        if self._depth(target) >= MAX_DEPTH:
            raise ObjectiveValidationError(
                "depth_limit", f"Objective depth may not exceed {MAX_DEPTH}."
            )
        parsed: list[tuple[str, str, int]] = []
        for raw_child in raw_children:
            if not isinstance(raw_child, dict):
                raise ObjectiveValidationError(
                    "invalid_child", "Each child must be an object."
                )
            parsed.append(
                (
                    self._text(raw_child.get("description"), "description"),
                    self._text(
                        raw_child.get("success_criterion"), "success_criterion"
                    ),
                    self._attempt_budget(raw_child.get("attempt_budget")),
                )
            )

        completed_prefix: list[str] = []
        unresolved_roots: list[str] = []
        found_unresolved = False
        for child_id in target.children:
            if not found_unresolved and self._nodes[child_id].status == "completed":
                completed_prefix.append(child_id)
                continue
            found_unresolved = True
            unresolved_roots.append(child_id)
        if not unresolved_roots:
            raise ObjectiveValidationError(
                "nothing_to_replan", "The decomposition has no unresolved branch."
            )

        removed_ids = [
            node_id
            for root_id in unresolved_roots
            for node_id in self._subtree_ids(root_id)
        ]
        if len(self._nodes) - len(removed_ids) + len(parsed) > MAX_NODES:
            raise ObjectiveValidationError(
                "node_limit", f"An objective graph may contain at most {MAX_NODES} nodes."
            )
        discarded = {
            "target_id": target.id,
            "reason": reason,
            "removed_roots": list(unresolved_roots),
            "nodes": [
                {
                    "id": self._nodes[node_id].id,
                    "description": self._nodes[node_id].description,
                    "status": self._nodes[node_id].status,
                    "attempts": self._nodes[node_id].attempts,
                    "attempt_budget": self._nodes[node_id].attempt_budget,
                    "attempts_since_revision": self._nodes[
                        node_id
                    ].attempts_since_revision,
                    "revision_count": self._nodes[node_id].revision_count,
                    "evidence": list(self._nodes[node_id].evidence[-2:]),
                }
                for node_id in removed_ids
            ],
        }
        for node_id in removed_ids:
            del self._nodes[node_id]
        replacement_ids = [
            self._new_node(
                description,
                criterion,
                target.id,
                attempt_budget=attempt_budget,
            ).id
            for description, criterion, attempt_budget in parsed
        ]
        target.children = [*completed_prefix, *replacement_ids]
        target.review_required = False
        target.unproductive_streak = 0
        target.rejection_streak = 0
        target.prediction_contradiction_streak = 0
        target.blocking_reason = None
        self._discarded_branches.append(discarded)
        self._discarded_branches = self._discarded_branches[-MAX_DISCARDED_BRANCHES:]

    def record_outcome(self, objective_id: str, outcome: dict[str, Any]) -> None:
        node = self._nodes.get(str(objective_id))
        if node is None:
            return
        before_active = self.active_objective_id
        before_blocking = self.blocking_node.id if self.blocking_node else None
        executed = outcome.get("executed", True) is not False
        try:
            executed_count = max(1, int(outcome.get("executed_count") or 1)) if executed else 0
        except (TypeError, ValueError):
            executed_count = 1 if executed else 0
        try:
            requested_count = max(1, int(outcome.get("requested_count") or 1))
        except (TypeError, ValueError):
            requested_count = 1
        error = outcome.get("error")
        rejection_code = (
            str(error.get("code") or "").strip()[:80]
            if isinstance(error, dict)
            else str(outcome.get("stop_reason") or "").strip()[:80]
        )
        prediction_result = outcome.get("prediction_result")
        prediction_status = (
            str(prediction_result.get("status") or "").strip().lower()
            if isinstance(prediction_result, dict)
            else ""
        )
        if executed:
            node.attempts += executed_count
            node.attempts_since_revision += executed_count
            node.rejection_streak = 0
            if prediction_status == "contradicted":
                node.prediction_contradictions += 1
                node.prediction_contradiction_streak += 1
            elif prediction_status:
                node.prediction_contradiction_streak = 0
        else:
            node.rejected_action_requests += 1
            node.rejection_streak += 1
        compact = {
            key: outcome.get(key)
            for key in ("action_display", "executed_actions", "outcome_class", "board_changed", "novel_state", "level_completed", "run_complete", "reward", "cycle_risk", "stop_reason")
            if outcome.get(key) is not None
        }
        compact.update(
            {
                "executed": executed,
                "executed_count": executed_count,
                "requested_count": requested_count,
            }
        )
        if rejection_code:
            compact["rejection_code"] = rejection_code
        if prediction_status:
            compact["prediction_status"] = prediction_status
        node.recent_outcomes.append(compact)
        node.recent_outcomes = node.recent_outcomes[-MAX_RECENT_OUTCOMES:]

        outcome_class = str(outcome.get("outcome_class") or "").strip().lower()
        try:
            reward = float(outcome.get("reward") or 0.0)
        except (TypeError, ValueError):
            reward = 0.0
        productive = bool(
            outcome.get("level_completed")
            or outcome.get("run_complete")
            or outcome.get("novel_state")
            or reward > 0
            or outcome_class in {"level_progress", "novel"}
            or (not outcome_class and outcome.get("board_changed"))
        )
        if not executed:
            if node.rejection_streak >= REJECTED_ACTION_REVIEW_LIMIT:
                detail = f" (last reason: {rejection_code})" if rejection_code else ""
                node.review_required = True
                node.blocking_reason = (
                    f"{node.rejection_streak} consecutive action requests were rejected before execution{detail}; revise the leaf or choose a valid, non-cycling action."
                )
        elif productive:
            node.unproductive_streak = 0
        else:
            node.unproductive_streak += executed_count
        if (
            executed
            and node.prediction_contradiction_streak
            >= PREDICTION_CONTRADICTION_REVIEW_LIMIT
        ):
            node.review_required = True
            node.blocking_reason = (
                f"{node.prediction_contradiction_streak} consecutive strategy predictions were contradicted; revise the leaf or its hypothesis before acting again."
            )
        elif executed and node.unproductive_streak >= UNPRODUCTIVE_REVIEW_LIMIT:
            node.review_required = True
            node.blocking_reason = (
                f"{node.unproductive_streak} consecutive actions produced no objective progress; revise the leaf before acting again."
            )
        elif executed and node.attempts_since_revision >= node.attempt_budget:
            node.review_required = True
            node.blocking_reason = (
                f"Attempt budget exhausted ({node.attempts_since_revision}/{node.attempt_budget}); revise the leaf or raise its bounded budget before acting again."
            )
        self._record_operation_event(
            operation="outcome",
            request_id=None,
            update={
                "objective_id": node.id,
                "outcome_class": outcome_class or None,
                "executed_count": executed_count,
                "executed": executed,
                "requested_count": requested_count,
                "rejection_code": rejection_code or None,
                "prediction_status": prediction_status or None,
            },
            result={"ok": True, "error": None},
            before_active=before_active,
            before_blocking=before_blocking,
        )

    def archive(
        self,
        *,
        reason: str,
        lesson: str = "",
        successful: bool | None = None,
    ) -> dict[str, Any] | None:
        if self._root_id is None:
            return None
        root = self._nodes[self._root_id]
        resolved_success = root.status == "completed" if successful is None else bool(successful)
        self._event_sequence += 1
        self._operation_events.append(
            {
                "sequence": self._event_sequence,
                "operation": "archive",
                "request_id": None,
                "ok": True,
                "error_code": None,
                "objective_id": root.id,
                "before_active_objective_id": self.active_objective_id,
                "after_active_objective_id": None,
                "before_blocking_objective_id": (
                    self.blocking_node.id if self.blocking_node else None
                ),
                "after_blocking_objective_id": None,
                "discarded_branch_count": len(self._discarded_branches),
                "reason": str(reason or "reset")[:80],
            }
        )
        self._operation_events = self._operation_events[-MAX_OPERATION_EVENTS:]
        archive = {
            "reason": str(reason or "reset")[:80],
            "root_description": root.description,
            "root_status": "completed" if resolved_success else root.status,
            "successful": resolved_success,
            "lesson": str(lesson or "").strip()[:MAX_TEXT_CHARS],
            "nodes": [
                {
                    "id": node.id,
                    "description": node.description,
                    "status": node.status,
                    "evidence": list(node.evidence[-2:]),
                    "attempts": node.attempts,
                    "attempt_budget": node.attempt_budget,
                    "attempts_since_revision": node.attempts_since_revision,
                    "revision_count": node.revision_count,
                    "rejected_action_requests": node.rejected_action_requests,
                    "rejection_streak": node.rejection_streak,
                    "prediction_contradictions": node.prediction_contradictions,
                    "prediction_contradiction_streak": node.prediction_contradiction_streak,
                }
                for node in self._nodes.values()
            ],
            "discarded_branches": deepcopy(self._discarded_branches),
            "operation_events": deepcopy(self._operation_events),
        }
        self._archives.append(archive)
        self._archives = self._archives[-MAX_ARCHIVES:]
        self._nodes = {}
        self._root_id = None
        self._discarded_branches = []
        self._operation_events = []
        self._idempotency_results = {}
        self._last_result = None
        return deepcopy(archive)
