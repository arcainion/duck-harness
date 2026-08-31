from __future__ import annotations

import unittest

from inference.agent.objective_reduction import (
    GameSolverType,
    ObjectiveEvidenceMode,
    ObjectiveError,
    ObjectiveKind,
    ObjectiveStatus,
    ObjectiveTree,
    ReductionProposal,
    TacticalExecutionMode,
)


def reduction_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "objective_id": "level:1:1",
        "verdict": "decompose",
        "evidence": "initial board",
        "rationale": "test one direction",
        "selected_index": 0,
        "subgoals": [
            {
                "title": "Probe north",
                "success_criteria": "player moves north",
                "failure_criteria": "north is blocked",
                "expected_evidence": "changed player coordinates",
                "action_budget": 12,
                "minimum_evidence_actions": 4,
                "single_step": False,
            }
        ],
    }
    payload.update(overrides)
    return payload


class ObjectiveTreeTests(unittest.TestCase):
    def test_legacy_tactical_objective_defaults_to_hybrid_solver(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=20,
        )
        restored = ObjectiveTree.from_dict(tree.to_dict())

        self.assertIsNone(tree.nodes[tree.root_id].solver_type)
        self.assertEqual(GameSolverType.HYBRID, tactical.solver_type)
        self.assertEqual(GameSolverType.HYBRID, restored.active.solver_type)

    def test_solver_type_round_trips_and_invalid_type_is_rejected(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
        subgoal["solver_type"] = "navigation"
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal])),
            remaining_level_actions=20,
        )

        self.assertEqual(GameSolverType.NAVIGATION, tactical.solver_type)
        subgoal["solver_type"] = "unknown-solver"
        with self.assertRaisesRegex(ObjectiveError, "solver_type"):
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal]))

    def test_host_creates_game_level_and_tactical_path(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        self.assertEqual(ObjectiveKind.LEVEL, tree.active.kind)
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=7,
        )
        self.assertEqual(ObjectiveKind.TACTICAL, tactical.kind)
        self.assertEqual(7, tactical.action_budget)
        self.assertEqual(ObjectiveStatus.ACTIVE, tactical.status)
        self.assertEqual(ObjectiveEvidenceMode.ENGINE_PROGRESS, tactical.evidence_mode)
        tree.validate()

    def test_stable_transition_evidence_mode_round_trips_to_tactical_node(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
        subgoal["evidence_mode"] = "stable_transition"
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal])),
            remaining_level_actions=20,
        )
        restored = ObjectiveTree.from_dict(tree.to_dict())
        self.assertEqual(
            ObjectiveEvidenceMode.STABLE_TRANSITION,
            tactical.evidence_mode,
        )
        self.assertEqual(tactical.evidence_mode, restored.active.evidence_mode)

    def test_contrastive_transition_evidence_mode_round_trips(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
        subgoal["evidence_mode"] = "contrastive_transition"
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal])),
            remaining_level_actions=20,
        )
        restored = ObjectiveTree.from_dict(tree.to_dict())
        self.assertEqual(
            ObjectiveEvidenceMode.CONTRASTIVE_TRANSITION,
            tactical.evidence_mode,
        )
        self.assertEqual(tactical.evidence_mode, restored.active.evidence_mode)

    def test_contrastive_transition_requires_resolvable_evidence_budget(self) -> None:
        for action_budget, minimum_evidence, single_step in (
            (2, 2, False),
            (8, 2, False),
            (1, 1, True),
        ):
            with self.subTest(
                action_budget=action_budget,
                minimum_evidence=minimum_evidence,
                single_step=single_step,
            ):
                subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
                subgoal.update(
                    {
                        "evidence_mode": "contrastive_transition",
                        "action_budget": action_budget,
                        "minimum_evidence_actions": minimum_evidence,
                        "single_step": single_step,
                    }
                )
                with self.assertRaisesRegex(
                    ObjectiveError, "contrastive_transition"
                ):
                    ReductionProposal.from_payload(
                        reduction_payload(subgoals=[subgoal])
                    )

    def test_invalid_evidence_mode_is_rejected(self) -> None:
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
        subgoal["evidence_mode"] = "board_changed"
        with self.assertRaisesRegex(ObjectiveError, "evidence_mode"):
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal]))

    def test_navigation_execution_mode_round_trips_to_tactical_node(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
        subgoal["execution_mode"] = "navigate"
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal])),
            remaining_level_actions=20,
        )
        restored = ObjectiveTree.from_dict(tree.to_dict())

        self.assertEqual(TacticalExecutionMode.NAVIGATE, tactical.execution_mode)
        self.assertEqual(tactical.execution_mode, restored.active.execution_mode)

    def test_invalid_execution_mode_is_rejected(self) -> None:
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
        subgoal["execution_mode"] = "fixed_sequence"
        with self.assertRaisesRegex(ObjectiveError, "execution_mode"):
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal]))

    def test_host_expands_short_non_single_step_to_macro_horizon(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
        subgoal["action_budget"] = 4
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal])),
            remaining_level_actions=20,
        )
        self.assertEqual(8, tactical.action_budget)
        self.assertEqual(4, tactical.minimum_evidence_actions)
        self.assertFalse(tactical.single_step)

    def test_explicit_single_step_preserves_one_action_contract(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index,arg-type]
        subgoal.update(
            action_budget=1,
            minimum_evidence_actions=1,
            single_step=True,
        )
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload(subgoals=[subgoal])),
            remaining_level_actions=20,
        )
        self.assertEqual(1, tactical.action_budget)
        self.assertEqual(1, tactical.minimum_evidence_actions)
        self.assertTrue(tactical.single_step)

    def test_model_cannot_complete_engine_owned_level(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        proposal = ReductionProposal.from_payload(
            reduction_payload(
                verdict="complete",
                subgoals=[],
                evidence="looks solved",
            )
        )
        with self.assertRaisesRegex(ObjectiveError, "only tactical"):
            tree.apply_proposal(proposal, remaining_level_actions=10)

    def test_resolved_tactical_returns_control_to_parent(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=20,
        )
        tree.record_action()
        tree.complete_active_tactical("movement observed")
        self.assertEqual(
            ObjectiveStatus.COMPLETED, tree.nodes[tactical.objective_id].status
        )
        self.assertEqual(ObjectiveKind.LEVEL, tree.active.kind)
        self.assertEqual(1, tree.nodes[tactical.objective_id].actions_used)

    def test_level_transition_supersedes_unfinished_tactics(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=20,
        )
        new_level = tree.start_level(2, level_action_budget=30)
        self.assertEqual(
            ObjectiveStatus.SUPERSEDED, tree.nodes[tactical.objective_id].status
        )
        self.assertEqual(ObjectiveKind.LEVEL, new_level.kind)
        self.assertEqual(2, tree.current_level)

    def test_round_trip_preserves_invariants(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=20,
        )
        restored = ObjectiveTree.from_dict(tree.to_dict())
        self.assertEqual(tree.to_dict(), restored.to_dict())

    def test_reduction_rejects_more_than_six_children(self) -> None:
        subgoal = reduction_payload()["subgoals"][0]  # type: ignore[index]
        with self.assertRaisesRegex(ObjectiveError, "at most six"):
            ReductionProposal.from_payload(
                reduction_payload(subgoals=[dict(subgoal) for _ in range(7)])  # type: ignore[arg-type]
            )

    def test_reduction_payload_rejects_malformed_shapes(self) -> None:
        cases = (
            (None, "must be an object"),
            ({}, "objective_id is required"),
            (reduction_payload(verdict="guess"), "invalid reduction verdict"),
            (reduction_payload(subgoals={}), "subgoals must be an array"),
            (
                reduction_payload(selected_index=1),
                "selected_index is outside subgoals",
            ),
            (
                reduction_payload(verdict="continue"),
                "subgoals are only valid with decompose",
            ),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ObjectiveError, message):
                    ReductionProposal.from_payload(payload)

    def test_subgoal_requires_strict_integer_budget_and_all_contract_text(self) -> None:
        base = dict(reduction_payload()["subgoals"][0])  # type: ignore[index, arg-type]
        for budget in (0, 33, 1.5, "4", True, None):
            with self.subTest(action_budget=budget):
                malformed = {**base, "action_budget": budget}
                with self.assertRaisesRegex(ObjectiveError, "action_budget"):
                    ReductionProposal.from_payload(
                        reduction_payload(subgoals=[malformed])
                    )
        for field in (
            "title",
            "success_criteria",
            "failure_criteria",
            "expected_evidence",
        ):
            with self.subTest(field=field):
                malformed = {**base, field: "  "}
                with self.assertRaisesRegex(ObjectiveError, field):
                    ReductionProposal.from_payload(
                        reduction_payload(subgoals=[malformed])
                    )

        for value in (0, 5, 1.5, "4", True, None):
            with self.subTest(minimum_evidence_actions=value):
                malformed = {**base, "minimum_evidence_actions": value}
                with self.assertRaisesRegex(ObjectiveError, "minimum_evidence_actions"):
                    ReductionProposal.from_payload(
                        reduction_payload(subgoals=[malformed])
                    )
        for value in (0, 1, "false", None):
            with self.subTest(single_step=value):
                malformed = {**base, "single_step": value}
                with self.assertRaisesRegex(ObjectiveError, "single_step"):
                    ReductionProposal.from_payload(
                        reduction_payload(subgoals=[malformed])
                    )

    def test_selected_index_must_be_a_strict_integer(self) -> None:
        for selected_index in (0.5, "0", True):
            with self.subTest(selected_index=selected_index):
                with self.assertRaisesRegex(ObjectiveError, "selected_index"):
                    ReductionProposal.from_payload(
                        reduction_payload(selected_index=selected_index)
                    )

    def test_proposal_must_target_the_active_objective(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        proposal = ReductionProposal.from_payload(
            reduction_payload(objective_id="level:99:1")
        )
        with self.assertRaisesRegex(ObjectiveError, "active objective"):
            tree.apply_proposal(proposal, remaining_level_actions=20)

    def test_continue_is_tactical_only_and_increments_attempts(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        with self.assertRaisesRegex(ObjectiveError, "must be decomposed"):
            tree.apply_proposal(
                ReductionProposal.from_payload(
                    reduction_payload(verdict="continue", subgoals=[])
                ),
                remaining_level_actions=20,
            )
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=20,
        )
        continued = tree.apply_proposal(
            ReductionProposal.from_payload(
                reduction_payload(
                    objective_id=tactical.objective_id,
                    verdict="continue",
                    subgoals=[],
                )
            ),
            remaining_level_actions=20,
        )
        self.assertEqual(2, continued.attempts)
        self.assertEqual(tactical.objective_id, tree.active_id)

    def test_redecomposition_supersedes_only_unresolved_children(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        subgoal = dict(reduction_payload()["subgoals"][0])  # type: ignore[index, arg-type]
        selected = tree.apply_proposal(
            ReductionProposal.from_payload(
                reduction_payload(
                    subgoals=[subgoal, {**subgoal, "title": "Probe east"}]
                )
            ),
            remaining_level_actions=20,
        )
        pending_id = tree.nodes[selected.parent_id or ""].children[1]
        tree.complete_active_tactical("confirmed")
        replacement = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=20,
        )
        self.assertEqual(
            ObjectiveStatus.COMPLETED, tree.nodes[selected.objective_id].status
        )
        self.assertEqual(ObjectiveStatus.SUPERSEDED, tree.nodes[pending_id].status)
        self.assertEqual(ObjectiveStatus.ACTIVE, replacement.status)

    def test_nested_reduction_stops_at_maximum_depth(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=32)
        while tree.depth(tree.active_id) < tree.max_depth:
            proposal = ReductionProposal.from_payload(
                reduction_payload(objective_id=tree.active_id)
            )
            tree.apply_proposal(proposal, remaining_level_actions=32)
        with self.assertRaisesRegex(ObjectiveError, "maximum objective depth"):
            tree.apply_proposal(
                ReductionProposal.from_payload(
                    reduction_payload(objective_id=tree.active_id)
                ),
                remaining_level_actions=32,
            )

    def test_zero_remaining_level_budget_rejects_decomposition(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        with self.assertRaisesRegex(ObjectiveError, "no level action budget"):
            tree.apply_proposal(
                ReductionProposal.from_payload(reduction_payload()),
                remaining_level_actions=0,
            )

    def test_record_action_cannot_exceed_tactical_budget(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(
                reduction_payload(
                    subgoals=[
                        {
                            **reduction_payload()["subgoals"][0],  # type: ignore[index, dict-item]
                            "action_budget": 1,
                            "minimum_evidence_actions": 1,
                            "single_step": True,
                        }
                    ]
                )
            ),
            remaining_level_actions=20,
        )
        tree.record_action()
        self.assertEqual(0, tactical.remaining_actions)
        self.assertEqual(1, tree.current_level_objective.actions_used)
        self.assertEqual(19, tree.remaining_level_actions)
        with self.assertRaisesRegex(ObjectiveError, "budget is exhausted"):
            tree.record_action()

    def test_controller_level_budget_is_authoritative_and_cannot_be_exceeded(
        self,
    ) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.sync_level_action_status(used=19, limit=20)
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=32,
        )
        self.assertEqual(1, tactical.action_budget)
        self.assertEqual(1, tree.remaining_level_actions)
        tree.record_action()
        self.assertEqual(1, tactical.actions_used)
        self.assertEqual(0, tree.remaining_level_actions)
        with self.assertRaisesRegex(ObjectiveError, "level action budget"):
            tree.record_action()

    def test_level_action_status_sync_clamps_controller_overrun(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tree.sync_level_action_status(used=22, limit=20)
        self.assertEqual(20, tree.current_level_objective.actions_used)
        self.assertEqual(0, tree.remaining_level_actions)

    def test_same_level_transition_is_idempotent(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        original = tree.active_id
        returned = tree.start_level(1, level_action_budget=99)
        self.assertEqual(original, returned.objective_id)
        self.assertEqual(1, len(tree.nodes[tree.root_id].children))
        self.assertEqual(20, returned.action_budget)

    def test_engine_game_resolution_supersedes_unfinished_branch(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        tactical = tree.apply_proposal(
            ReductionProposal.from_payload(reduction_payload()),
            remaining_level_actions=20,
        )
        tree.resolve_game(won=False, evidence="engine GAME_OVER")
        self.assertEqual(ObjectiveStatus.FAILED, tree.nodes[tree.root_id].status)
        self.assertEqual(ObjectiveStatus.SUPERSEDED, tactical.status)
        self.assertEqual(
            "engine GAME_OVER", tree.nodes[tree.root_id].resolution_evidence
        )
        tree.validate()

    def test_deserialization_rejects_corrupt_tree_links_and_identity(self) -> None:
        tree = ObjectiveTree.start_game("game-a", level=1, level_action_budget=20)
        cases: list[tuple[str, dict[str, object], str]] = []

        missing_child = tree.to_dict()
        missing_child["nodes"][tree.root_id]["children"].append("level:missing")
        cases.append(("missing child", missing_child, "parent/child"))

        duplicate_child = tree.to_dict()
        duplicate_child["nodes"][tree.root_id]["children"].append(tree.active_id)
        cases.append(("duplicate child", duplicate_child, "duplicates"))

        mismatched_id = tree.to_dict()
        mismatched_id["nodes"][tree.active_id]["objective_id"] = "level:other:1"
        cases.append(("mismatched id", mismatched_id, "map key"))

        wrong_root_kind = tree.to_dict()
        wrong_root_kind["nodes"][tree.root_id]["kind"] = "tactical"
        cases.append(("wrong root kind", wrong_root_kind, "root must be a game"))

        for label, payload, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ObjectiveError, message):
                    ObjectiveTree.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
