"""Deterministic ordered objective reduction for tool-using agents."""

from __future__ import annotations

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
    revisions: list[dict[str, str]] = field(default_factory=list)

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
        }


class ObjectiveReducer:
    """Own and validate a single ordered AND objective tree."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._nodes: dict[str, ObjectiveNode] = {}
        self._root_id: str | None = None
        self._next_id = 1
        self._archives: list[dict[str, Any]] = []
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
        if node.status != "pending":
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

    def active_path(self) -> list[str]:
        active = self.active_node
        if active is None:
            return []
        path: list[str] = []
        node: ObjectiveNode | None = active
        while node is not None:
            path.append(node.id)
            node = self._nodes.get(node.parent_id) if node.parent_id else None
        return list(reversed(path))

    def active_path_descriptions(self) -> list[str]:
        return [self._nodes[node_id].description for node_id in self.active_path()]

    def snapshot(self) -> dict[str, Any]:
        active = self.active_node
        return {
            "enabled": self.enabled,
            "active_objective_id": active.id if active else None,
            "active_path": self.active_path(),
            "active_path_descriptions": self.active_path_descriptions(),
            "active_success_criterion": active.success_criterion if active else None,
            "graph": {
                "root_id": self._root_id,
                "nodes": [node.payload() for node in self._nodes.values()],
            },
            "archives": deepcopy(self._archives),
            "last_result": deepcopy(self._last_result),
        }

    def apply(self, update: dict[str, Any]) -> dict[str, Any]:
        operation = str(update.get("op") or "").strip().lower()
        backup = (deepcopy(self._nodes), self._root_id, self._next_id, deepcopy(self._archives))
        try:
            if not self.enabled:
                raise ObjectiveValidationError("disabled", "Objective reduction is not enabled.")
            if operation not in {"initialize", "reduce", "complete", "fail", "revise"}:
                raise ObjectiveValidationError("unknown_operation", "Unknown objective operation.")
            getattr(self, f"_{operation}")(update)
            result = self._result(True, operation)
        except ObjectiveValidationError as exc:
            self._nodes, self._root_id, self._next_id, self._archives = backup
            result = self._result(False, operation, code=exc.code, message=exc.message)
        self._last_result = deepcopy(result)
        return result

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
            "graph": snapshot["graph"],
            "error": None if ok else {"code": code, "message": message},
        }

    def _text(self, value: Any, field_name: str, *, required: bool = True) -> str:
        text = str(value or "").strip()
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
            text = str(item or "").strip()
            if not text:
                continue
            if len(text) > MAX_EVIDENCE_CHARS:
                raise ObjectiveValidationError("evidence_too_long", f"evidence items may contain at most {MAX_EVIDENCE_CHARS} characters.")
            evidence.append(text)
        return evidence

    def _new_node(self, description: str, success_criterion: str, parent_id: str | None = None) -> ObjectiveNode:
        if len(self._nodes) >= MAX_NODES:
            raise ObjectiveValidationError("node_limit", f"An objective graph may contain at most {MAX_NODES} nodes.")
        node_id = f"obj-{self._next_id}"
        self._next_id += 1
        node = ObjectiveNode(node_id, description, success_criterion, parent_id)
        self._nodes[node_id] = node
        return node

    def _initialize(self, update: dict[str, Any]) -> None:
        if self._root_id is not None:
            root = self._nodes[self._root_id]
            if root.status == "pending":
                raise ObjectiveValidationError("open_root", "Complete or fail the current root before initializing another.")
            self.archive(reason="reroot")
        description = self._text(update.get("description"), "description")
        criterion = self._text(update.get("success_criterion"), "success_criterion")
        root = self._new_node(description, criterion)
        self._root_id = root.id

    def _require_node(self, value: Any) -> ObjectiveNode:
        node_id = str(value or "").strip()
        node = self._nodes.get(node_id)
        if node is None:
            raise ObjectiveValidationError("unknown_objective", f"Unknown objective id: {node_id or '<empty>'}.")
        return node

    def _require_active(self, value: Any) -> ObjectiveNode:
        node = self._require_node(value)
        if node.id != self.active_objective_id:
            raise ObjectiveValidationError("not_active", "The operation must target the active objective leaf.")
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
        parsed: list[tuple[str, str]] = []
        for raw_child in raw_children:
            if not isinstance(raw_child, dict):
                raise ObjectiveValidationError("invalid_child", "Each child must be an object.")
            parsed.append((
                self._text(raw_child.get("description"), "description"),
                self._text(raw_child.get("success_criterion"), "success_criterion"),
            ))
        parent.children = [self._new_node(desc, criterion, parent.id).id for desc, criterion in parsed]

    def _complete(self, update: dict[str, Any]) -> None:
        node = self._require_active(update.get("objective_id"))
        node.evidence.extend(self._evidence(update.get("evidence")))
        node.evidence = node.evidence[-MAX_EVIDENCE_ITEMS:]
        node.status = "completed"
        while node.parent_id:
            parent = self._nodes[node.parent_id]
            if not parent.children or any(self._nodes[child].status != "completed" for child in parent.children):
                break
            parent.status = "completed"
            node = parent

    def _fail(self, update: dict[str, Any]) -> None:
        node = self._require_active(update.get("objective_id"))
        node.evidence.extend(self._evidence(update.get("evidence")))
        node.evidence = node.evidence[-MAX_EVIDENCE_ITEMS:]
        node.status = "failed"

    def _revise(self, update: dict[str, Any]) -> None:
        node = self._require_node(update.get("objective_id"))
        if node.children or node.status == "completed":
            raise ObjectiveValidationError("not_revisable", "Only pending or failed leaves may be revised.")
        description = self._text(update.get("description"), "description", required=False)
        criterion = self._text(update.get("success_criterion"), "success_criterion", required=False)
        evidence = self._evidence(update.get("evidence"))
        if not description and not criterion and not evidence:
            raise ObjectiveValidationError("empty_revision", "A revision must change text or add evidence.")
        node.revisions.append({"description": node.description, "success_criterion": node.success_criterion})
        node.revisions = node.revisions[-MAX_REVISIONS:]
        if description:
            node.description = description
        if criterion:
            node.success_criterion = criterion
        node.evidence.extend(evidence)
        node.evidence = node.evidence[-MAX_EVIDENCE_ITEMS:]
        node.status = "pending"

    def record_outcome(self, objective_id: str, outcome: dict[str, Any]) -> None:
        node = self._nodes.get(str(objective_id))
        if node is None:
            return
        node.attempts += max(1, int(outcome.get("executed_count") or 1))
        compact = {
            key: outcome.get(key)
            for key in ("action_display", "executed_actions", "outcome_class", "board_changed", "novel_state", "level_completed", "reward")
            if outcome.get(key) is not None
        }
        node.recent_outcomes.append(compact)
        node.recent_outcomes = node.recent_outcomes[-MAX_RECENT_OUTCOMES:]

    def archive(self, *, reason: str, lesson: str = "") -> dict[str, Any] | None:
        if self._root_id is None:
            return None
        root = self._nodes[self._root_id]
        archive = {
            "reason": str(reason or "reset")[:80],
            "root_description": root.description,
            "root_status": root.status,
            "lesson": str(lesson or "").strip()[:MAX_TEXT_CHARS],
            "nodes": [
                {
                    "id": node.id,
                    "description": node.description,
                    "status": node.status,
                    "evidence": list(node.evidence[-2:]),
                    "attempts": node.attempts,
                }
                for node in self._nodes.values()
            ],
        }
        self._archives.append(archive)
        self._archives = self._archives[-MAX_ARCHIVES:]
        self._nodes = {}
        self._root_id = None
        return deepcopy(archive)
