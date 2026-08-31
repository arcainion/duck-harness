"""Host-owned objective hierarchy for orchestrated ARC gameplay.

The model may propose tactical reductions, but it never owns identifiers or
engine-level completion.  This module intentionally contains no model or game
engine dependencies so its invariants can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ObjectiveError(ValueError):
    """Raised when a proposed objective transition violates host invariants."""


class ObjectiveKind(StrEnum):
    GAME = "game"
    LEVEL = "level"
    TACTICAL = "tactical"


class ObjectiveStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SUPERSEDED = "superseded"


class ObjectiveEvidenceMode(StrEnum):
    """Host-verifiable evidence required to complete a tactical objective."""

    ENGINE_PROGRESS = "engine_progress"
    STABLE_TRANSITION = "stable_transition"
    CONTRASTIVE_TRANSITION = "contrastive_transition"


class ReductionVerdict(StrEnum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    FAIL = "fail"
    DECOMPOSE = "decompose"


@dataclass(frozen=True)
class SubgoalSpec:
    title: str
    success_criteria: str
    failure_criteria: str
    expected_evidence: str
    action_budget: int
    evidence_mode: ObjectiveEvidenceMode = ObjectiveEvidenceMode.ENGINE_PROGRESS
    minimum_evidence_actions: int = 4
    single_step: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> "SubgoalSpec":
        if not isinstance(payload, dict):
            raise ObjectiveError("each subgoal must be an object")

        def required_text(name: str, limit: int = 1200) -> str:
            value = str(payload.get(name) or "").strip()
            if not value:
                raise ObjectiveError(f"subgoal {name} is required")
            return value[:limit]

        raw_action_budget = payload.get("action_budget")
        if not isinstance(raw_action_budget, int) or isinstance(
            raw_action_budget, bool
        ):
            raise ObjectiveError("subgoal action_budget must be an integer")
        action_budget = raw_action_budget
        if not 1 <= action_budget <= 32:
            raise ObjectiveError("subgoal action_budget must be between 1 and 32")
        raw_minimum_evidence = payload.get(
            "minimum_evidence_actions", min(4, action_budget)
        )
        if not isinstance(raw_minimum_evidence, int) or isinstance(
            raw_minimum_evidence, bool
        ):
            raise ObjectiveError("subgoal minimum_evidence_actions must be an integer")
        if not 1 <= raw_minimum_evidence <= min(4, action_budget):
            raise ObjectiveError(
                "subgoal minimum_evidence_actions must be between 1 and "
                "min(4, action_budget)"
            )
        raw_single_step = payload.get("single_step", action_budget == 1)
        if not isinstance(raw_single_step, bool):
            raise ObjectiveError("subgoal single_step must be a boolean")
        if raw_single_step and action_budget != 1:
            raise ObjectiveError("single-step subgoals must request action_budget=1")
        if not raw_single_step and action_budget == 1:
            raise ObjectiveError(
                "action_budget=1 is reserved for explicitly single-step subgoals"
            )
        try:
            evidence_mode = ObjectiveEvidenceMode(
                str(payload.get("evidence_mode") or "engine_progress").strip()
            )
        except ValueError as exc:
            raise ObjectiveError(
                "subgoal evidence_mode must be engine_progress, stable_transition, "
                "or contrastive_transition"
            ) from exc
        if evidence_mode is ObjectiveEvidenceMode.CONTRASTIVE_TRANSITION and (
            action_budget < 3 or raw_minimum_evidence < 3 or raw_single_step
        ):
            raise ObjectiveError(
                "contrastive_transition subgoals require action_budget>=3, "
                "minimum_evidence_actions>=3, and single_step=false"
            )
        return cls(
            title=required_text("title", 200),
            success_criteria=required_text("success_criteria"),
            failure_criteria=required_text("failure_criteria"),
            expected_evidence=required_text("expected_evidence"),
            action_budget=action_budget,
            evidence_mode=evidence_mode,
            minimum_evidence_actions=raw_minimum_evidence,
            single_step=raw_single_step,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "success_criteria": self.success_criteria,
            "failure_criteria": self.failure_criteria,
            "expected_evidence": self.expected_evidence,
            "action_budget": self.action_budget,
            "evidence_mode": self.evidence_mode.value,
            "minimum_evidence_actions": self.minimum_evidence_actions,
            "single_step": self.single_step,
        }


@dataclass(frozen=True)
class ReductionProposal:
    objective_id: str
    verdict: ReductionVerdict
    evidence: str
    rationale: str
    subgoals: tuple[SubgoalSpec, ...] = ()
    selected_index: int = 0

    @classmethod
    def from_payload(cls, payload: Any) -> "ReductionProposal":
        if not isinstance(payload, dict):
            raise ObjectiveError("reduction proposal must be an object")
        objective_id = str(payload.get("objective_id") or "").strip()
        if not objective_id:
            raise ObjectiveError("objective_id is required")
        try:
            verdict = ReductionVerdict(str(payload.get("verdict") or "").strip())
        except ValueError as exc:
            raise ObjectiveError("invalid reduction verdict") from exc
        raw_subgoals = payload.get("subgoals", [])
        if not isinstance(raw_subgoals, list):
            raise ObjectiveError("subgoals must be an array")
        if len(raw_subgoals) > 6:
            raise ObjectiveError("a reduction may contain at most six subgoals")
        subgoals = tuple(SubgoalSpec.from_payload(item) for item in raw_subgoals)
        raw_selected_index = payload.get("selected_index", 0)
        if not isinstance(raw_selected_index, int) or isinstance(
            raw_selected_index, bool
        ):
            raise ObjectiveError("selected_index must be an integer")
        selected_index = raw_selected_index
        if verdict is ReductionVerdict.DECOMPOSE:
            if not subgoals:
                raise ObjectiveError("decompose requires at least one subgoal")
            if not 0 <= selected_index < len(subgoals):
                raise ObjectiveError("selected_index is outside subgoals")
        elif subgoals:
            raise ObjectiveError("subgoals are only valid with decompose")
        return cls(
            objective_id=objective_id,
            verdict=verdict,
            evidence=str(payload.get("evidence") or "").strip()[:2400],
            rationale=str(payload.get("rationale") or "").strip()[:2400],
            subgoals=subgoals,
            selected_index=selected_index,
        )


@dataclass
class ObjectiveNode:
    objective_id: str
    parent_id: str | None
    kind: ObjectiveKind
    title: str
    success_criteria: str
    failure_criteria: str
    expected_evidence: str
    action_budget: int
    evidence_mode: ObjectiveEvidenceMode = ObjectiveEvidenceMode.ENGINE_PROGRESS
    minimum_evidence_actions: int = 1
    single_step: bool = False
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    attempts: int = 0
    actions_used: int = 0
    children: list[str] = field(default_factory=list)
    resolution_evidence: str = ""

    @property
    def remaining_actions(self) -> int:
        return max(0, self.action_budget - self.actions_used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "parent_id": self.parent_id,
            "kind": self.kind.value,
            "title": self.title,
            "success_criteria": self.success_criteria,
            "failure_criteria": self.failure_criteria,
            "expected_evidence": self.expected_evidence,
            "action_budget": self.action_budget,
            "evidence_mode": self.evidence_mode.value,
            "minimum_evidence_actions": self.minimum_evidence_actions,
            "single_step": self.single_step,
            "status": self.status.value,
            "attempts": self.attempts,
            "actions_used": self.actions_used,
            "children": list(self.children),
            "resolution_evidence": self.resolution_evidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectiveNode":
        return cls(
            objective_id=str(payload["objective_id"]),
            parent_id=(
                str(payload["parent_id"])
                if payload.get("parent_id") is not None
                else None
            ),
            kind=ObjectiveKind(str(payload["kind"])),
            title=str(payload.get("title") or ""),
            success_criteria=str(payload.get("success_criteria") or ""),
            failure_criteria=str(payload.get("failure_criteria") or ""),
            expected_evidence=str(payload.get("expected_evidence") or ""),
            action_budget=max(0, int(payload.get("action_budget", 0) or 0)),
            evidence_mode=ObjectiveEvidenceMode(
                str(payload.get("evidence_mode") or "engine_progress")
            ),
            minimum_evidence_actions=max(
                1,
                int(
                    payload.get(
                        "minimum_evidence_actions",
                        min(4, int(payload.get("action_budget", 0) or 0)),
                    )
                    or 1
                ),
            ),
            single_step=bool(payload.get("single_step", False)),
            status=ObjectiveStatus(str(payload.get("status") or "pending")),
            attempts=max(0, int(payload.get("attempts", 0) or 0)),
            actions_used=max(0, int(payload.get("actions_used", 0) or 0)),
            children=[str(item) for item in payload.get("children") or []],
            resolution_evidence=str(payload.get("resolution_evidence") or ""),
        )


@dataclass
class ObjectiveTree:
    game_id: str
    nodes: dict[str, ObjectiveNode]
    root_id: str
    active_id: str
    current_level: int
    next_tactical_id: int = 1
    max_depth: int = 6
    max_children: int = 6

    @classmethod
    def start_game(
        cls, game_id: str, *, level: int, level_action_budget: int
    ) -> "ObjectiveTree":
        safe_game_id = str(game_id or "unknown")[:160]
        root_id = f"game:{safe_game_id}"
        level_id = f"level:{max(1, int(level))}:1"
        root = ObjectiveNode(
            objective_id=root_id,
            parent_id=None,
            kind=ObjectiveKind.GAME,
            title=f"Solve game {safe_game_id}",
            success_criteria="The engine reports WIN/run_complete.",
            failure_criteria="The engine reports terminal GAME_OVER or host limits expire.",
            expected_evidence="Engine-owned game state.",
            action_budget=max(1, int(level_action_budget)),
            status=ObjectiveStatus.ACTIVE,
            children=[level_id],
        )
        level_node = ObjectiveNode(
            objective_id=level_id,
            parent_id=root_id,
            kind=ObjectiveKind.LEVEL,
            title=f"Complete level {max(1, int(level))}",
            success_criteria="The engine reports level_completed or advances the level.",
            failure_criteria="The game ends or the host level budget expires.",
            expected_evidence="Engine-owned level transition.",
            action_budget=max(1, int(level_action_budget)),
            status=ObjectiveStatus.ACTIVE,
        )
        tree = cls(
            game_id=safe_game_id,
            nodes={root_id: root, level_id: level_node},
            root_id=root_id,
            active_id=level_id,
            current_level=max(1, int(level)),
        )
        tree.validate()
        return tree

    @property
    def active(self) -> ObjectiveNode:
        try:
            return self.nodes[self.active_id]
        except KeyError as exc:
            raise ObjectiveError("active objective does not exist") from exc

    def depth(self, objective_id: str) -> int:
        depth = 0
        seen: set[str] = set()
        current = self.nodes.get(objective_id)
        while current is not None and current.parent_id is not None:
            if current.objective_id in seen:
                raise ObjectiveError("objective tree contains a cycle")
            seen.add(current.objective_id)
            depth += 1
            current = self.nodes.get(current.parent_id)
        return depth

    def _active_path(self) -> set[str]:
        path: set[str] = set()
        current = self.nodes.get(self.active_id)
        while current is not None:
            if current.objective_id in path:
                raise ObjectiveError("objective tree contains a cycle")
            path.add(current.objective_id)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        return path

    def validate(self) -> None:
        if self.root_id not in self.nodes or self.active_id not in self.nodes:
            raise ObjectiveError("objective tree root or active objective is missing")
        root = self.nodes[self.root_id]
        if root.objective_id != self.root_id:
            raise ObjectiveError("objective node ID does not match its map key")
        if root.kind is not ObjectiveKind.GAME:
            raise ObjectiveError("objective tree root must be a game objective")
        if root.parent_id is not None:
            raise ObjectiveError("game root may not have a parent")
        active_path = self._active_path()
        if self.root_id not in active_path:
            raise ObjectiveError("active objective is detached from the game root")
        for objective_id, node in self.nodes.items():
            if node.objective_id != objective_id:
                raise ObjectiveError("objective node ID does not match its map key")
            if node.action_budget < 1:
                raise ObjectiveError("objective action budgets must be positive")
            if not isinstance(node.evidence_mode, ObjectiveEvidenceMode):
                raise ObjectiveError("objective evidence mode is invalid")
            if (
                node.kind is not ObjectiveKind.TACTICAL
                and node.evidence_mode is not ObjectiveEvidenceMode.ENGINE_PROGRESS
            ):
                raise ObjectiveError(
                    "game and level objectives require engine_progress evidence"
                )
            if not 1 <= node.minimum_evidence_actions <= min(4, node.action_budget):
                raise ObjectiveError(
                    "objective minimum evidence actions must fit its action budget"
                )
            if node.single_step and node.action_budget != 1:
                raise ObjectiveError(
                    "single-step objective must have action budget one"
                )
            if node.actions_used > node.action_budget:
                raise ObjectiveError("objective action budget was exceeded")
            if len(node.children) != len(set(node.children)):
                raise ObjectiveError("objective children may not contain duplicates")
            if self.depth(objective_id) > self.max_depth:
                raise ObjectiveError("objective tree exceeds maximum depth")
            if node.parent_id is not None:
                parent = self.nodes.get(node.parent_id)
                if parent is None or objective_id not in parent.children:
                    raise ObjectiveError(
                        "objective parent/child links are inconsistent"
                    )
                if node.kind is ObjectiveKind.GAME:
                    raise ObjectiveError("only the root may be a game objective")
                if (
                    node.kind is ObjectiveKind.LEVEL
                    and parent.kind is not ObjectiveKind.GAME
                ):
                    raise ObjectiveError(
                        "level objectives must be children of the game"
                    )
                if node.kind is ObjectiveKind.TACTICAL and parent.kind not in {
                    ObjectiveKind.LEVEL,
                    ObjectiveKind.TACTICAL,
                }:
                    raise ObjectiveError(
                        "tactical objectives require a level or tactical parent"
                    )
            for child_id in node.children:
                child = self.nodes.get(child_id)
                if child is None or child.parent_id != objective_id:
                    raise ObjectiveError(
                        "objective parent/child links are inconsistent"
                    )
            if (
                node.status is ObjectiveStatus.ACTIVE
                and objective_id not in active_path
            ):
                raise ObjectiveError("active objectives must form one path")
        if root.status is ObjectiveStatus.ACTIVE:
            if any(
                self.nodes[objective_id].status is not ObjectiveStatus.ACTIVE
                for objective_id in active_path
            ):
                raise ObjectiveError("the active path must contain active objectives")
        elif any(node.status is ObjectiveStatus.ACTIVE for node in self.nodes.values()):
            raise ObjectiveError("a resolved game may not contain active objectives")

    def start_level(self, level: int, *, level_action_budget: int) -> ObjectiveNode:
        level = max(1, int(level))
        if level == self.current_level:
            return self.nodes[self._level_id_for_active_path()]
        previous_level_id = self._level_id_for_active_path()
        self._resolve_branch(
            previous_level_id,
            ObjectiveStatus.COMPLETED,
            f"engine advanced from level {self.current_level} to {level}",
        )
        root = self.nodes[self.root_id]
        root.status = ObjectiveStatus.ACTIVE
        level_id = f"level:{level}:{len(root.children) + 1}"
        node = ObjectiveNode(
            objective_id=level_id,
            parent_id=self.root_id,
            kind=ObjectiveKind.LEVEL,
            title=f"Complete level {level}",
            success_criteria="The engine reports level_completed or advances the level.",
            failure_criteria="The game ends or the host level budget expires.",
            expected_evidence="Engine-owned level transition.",
            action_budget=max(1, int(level_action_budget)),
            status=ObjectiveStatus.ACTIVE,
        )
        root.children.append(level_id)
        self.nodes[level_id] = node
        self.active_id = level_id
        self.current_level = level
        self.validate()
        return node

    def _level_id_for_active_path(self) -> str:
        current = self.active
        while current.kind is not ObjectiveKind.LEVEL:
            if current.parent_id is None:
                raise ObjectiveError("active path does not contain a level objective")
            current = self.nodes[current.parent_id]
        return current.objective_id

    @property
    def current_level_objective(self) -> ObjectiveNode:
        """Return the host-owned level node on the active objective path."""

        return self.nodes[self._level_id_for_active_path()]

    @property
    def remaining_level_actions(self) -> int:
        """Return the authoritative remaining action budget for this level."""

        return self.current_level_objective.remaining_actions

    def sync_level_action_status(self, *, used: int, limit: int) -> None:
        """Synchronize the level node with the controller's authoritative counters."""

        checked_limit = int(limit)
        checked_used = int(used)
        if checked_limit < 1:
            raise ObjectiveError("level action limit must be positive")
        if checked_used < 0:
            raise ObjectiveError("level actions used may not be negative")
        level = self.current_level_objective
        level.action_budget = checked_limit
        level.actions_used = min(checked_used, checked_limit)
        self.validate()

    def apply_proposal(
        self, proposal: ReductionProposal, *, remaining_level_actions: int
    ) -> ObjectiveNode:
        if proposal.objective_id != self.active_id:
            raise ObjectiveError(
                f"proposal targets {proposal.objective_id!r}; active objective is {self.active_id!r}"
            )
        node = self.active
        if proposal.verdict in {ReductionVerdict.COMPLETE, ReductionVerdict.FAIL}:
            if node.kind is not ObjectiveKind.TACTICAL:
                raise ObjectiveError(
                    "only tactical objectives may be resolved by the model"
                )
            status = (
                ObjectiveStatus.COMPLETED
                if proposal.verdict is ReductionVerdict.COMPLETE
                else ObjectiveStatus.FAILED
            )
            self._resolve_active(status, proposal.evidence or proposal.rationale)
            self.validate()
            return self.active
        if proposal.verdict is ReductionVerdict.CONTINUE:
            if node.kind is not ObjectiveKind.TACTICAL:
                raise ObjectiveError("game and level objectives must be decomposed")
            node.attempts += 1
            node.status = ObjectiveStatus.ACTIVE
            self.validate()
            return node

        if self.depth(node.objective_id) >= self.max_depth:
            raise ObjectiveError("cannot decompose beyond maximum objective depth")
        for child_id in node.children:
            child = self.nodes[child_id]
            if child.status in {ObjectiveStatus.PENDING, ObjectiveStatus.ACTIVE}:
                child.status = ObjectiveStatus.SUPERSEDED
        created: list[ObjectiveNode] = []
        available = min(
            int(remaining_level_actions),
            self.remaining_level_actions,
        )
        if available <= 0:
            raise ObjectiveError(
                "no level action budget remains for a tactical subgoal"
            )
        for spec in proposal.subgoals:
            objective_id = f"tactical:{self.next_tactical_id}"
            self.next_tactical_id += 1
            requested_budget = (
                spec.action_budget if spec.single_step else max(8, spec.action_budget)
            )
            effective_budget = min(requested_budget, available)
            child = ObjectiveNode(
                objective_id=objective_id,
                parent_id=node.objective_id,
                kind=ObjectiveKind.TACTICAL,
                title=spec.title,
                success_criteria=spec.success_criteria,
                failure_criteria=spec.failure_criteria,
                expected_evidence=spec.expected_evidence,
                action_budget=effective_budget,
                evidence_mode=spec.evidence_mode,
                minimum_evidence_actions=min(
                    spec.minimum_evidence_actions, effective_budget
                ),
                single_step=spec.single_step and effective_budget == 1,
                status=ObjectiveStatus.PENDING,
            )
            self.nodes[objective_id] = child
            node.children.append(objective_id)
            created.append(child)
        selected = created[proposal.selected_index]
        selected.status = ObjectiveStatus.ACTIVE
        selected.attempts += 1
        node.status = ObjectiveStatus.ACTIVE
        self.active_id = selected.objective_id
        self.validate()
        return selected

    def record_action(self) -> None:
        node = self.active
        if node.kind is ObjectiveKind.TACTICAL:
            level = self.current_level_objective
            if level.remaining_actions <= 0:
                raise ObjectiveError("level action budget is exhausted")
            if node.remaining_actions <= 0:
                raise ObjectiveError("tactical action budget is exhausted")
            node.actions_used += 1
            level.actions_used += 1

    def complete_active_tactical(self, evidence: str) -> ObjectiveNode:
        if self.active.kind is not ObjectiveKind.TACTICAL:
            raise ObjectiveError("active objective is not tactical")
        self._resolve_active(ObjectiveStatus.COMPLETED, evidence)
        self.validate()
        return self.active

    def fail_active_tactical(self, evidence: str) -> ObjectiveNode:
        if self.active.kind is not ObjectiveKind.TACTICAL:
            raise ObjectiveError("active objective is not tactical")
        self._resolve_active(ObjectiveStatus.FAILED, evidence)
        self.validate()
        return self.active

    def _resolve_active(self, status: ObjectiveStatus, evidence: str) -> None:
        node = self.active
        node.status = status
        node.resolution_evidence = str(evidence or "")[:2400]
        if node.parent_id is None:
            return
        parent = self.nodes[node.parent_id]
        if parent.status not in {ObjectiveStatus.COMPLETED, ObjectiveStatus.FAILED}:
            parent.status = ObjectiveStatus.ACTIVE
        self.active_id = parent.objective_id

    def _resolve_branch(
        self, objective_id: str, status: ObjectiveStatus, evidence: str
    ) -> None:
        node = self.nodes[objective_id]
        node.status = status
        node.resolution_evidence = str(evidence or "")[:2400]
        stack = list(node.children)
        while stack:
            child = self.nodes[stack.pop()]
            if child.status not in {ObjectiveStatus.COMPLETED, ObjectiveStatus.FAILED}:
                child.status = ObjectiveStatus.SUPERSEDED
            stack.extend(child.children)

    def resolve_game(self, *, won: bool, evidence: str) -> None:
        self._resolve_branch(
            self.root_id,
            ObjectiveStatus.COMPLETED if won else ObjectiveStatus.FAILED,
            evidence,
        )
        self.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "root_id": self.root_id,
            "active_id": self.active_id,
            "current_level": self.current_level,
            "next_tactical_id": self.next_tactical_id,
            "max_depth": self.max_depth,
            "max_children": self.max_children,
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectiveTree":
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, dict):
            raise ObjectiveError("objective state is missing nodes")
        tree = cls(
            game_id=str(payload.get("game_id") or "unknown"),
            nodes={
                str(key): ObjectiveNode.from_dict(value)
                for key, value in raw_nodes.items()
                if isinstance(value, dict)
            },
            root_id=str(payload.get("root_id") or ""),
            active_id=str(payload.get("active_id") or ""),
            current_level=max(1, int(payload.get("current_level", 1) or 1)),
            next_tactical_id=max(1, int(payload.get("next_tactical_id", 1) or 1)),
            max_depth=max(2, int(payload.get("max_depth", 6) or 6)),
            max_children=max(1, int(payload.get("max_children", 6) or 6)),
        )
        tree.validate()
        return tree
