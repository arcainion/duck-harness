from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.agent.trial_knowledge import TrialKnowledgeStore


def _transition(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "executed": True,
        "before_state_id": "state-a",
        "after_state_id": "state-b",
        "action_display": "LEFT",
        "outcome_class": "novel",
    }
    value.update(overrides)
    return value


class TrialKnowledgeRobustnessTests(unittest.TestCase):
    def test_invalid_constructor_limits_use_bounded_defaults(self) -> None:
        store = TrialKnowledgeStore(
            transition_limit=float("inf"), lesson_limit=True
        )

        self.assertEqual(store._transition_limit, 512)
        self.assertEqual(store._lesson_limit, 24)

    def test_switching_persistence_path_clears_previous_games(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            store = TrialKnowledgeStore(persistence_path=first)
            store.observe("game-a", _transition())

            store.configure_path(second)

            self.assertEqual(store.snapshot("game-a")["observations"], 0)

    def test_scalar_persisted_transition_collection_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(
                json.dumps({"version": 3, "games": {"game": {"transitions": 7}}}),
                encoding="utf-8",
            )

            store = TrialKnowledgeStore(persistence_path=path)

            self.assertEqual(store.snapshot("game")["observations"], 0)

    def test_scalar_persisted_lesson_collection_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(
                json.dumps({"version": 3, "games": {"game": {"lessons": "bad"}}}),
                encoding="utf-8",
            )

            store = TrialKnowledgeStore(persistence_path=path)

            self.assertEqual(store.snapshot("game")["progress_lessons"], [])

    def test_malformed_state_context_version_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(
                json.dumps(
                    {
                        "version": "invalid",
                        "games": {
                            "game": {
                                "transitions": [
                                    {**_transition(), "state_context_version": "invalid"}
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            record = TrialKnowledgeStore(persistence_path=path).snapshot("game")[
                "transition_records"
            ][0]

            self.assertEqual(record["state_context_version"], 1)
            self.assertTrue(record["legacy_state_identity"])

    def test_observe_validates_mapping_and_executed_flag(self) -> None:
        store = TrialKnowledgeStore()

        store.observe("game", None)  # type: ignore[arg-type]
        store.observe("game", _transition(executed="false"))
        store.observe("game", _transition(executed="true"))

        self.assertEqual(store.snapshot("game")["observations"], 1)

    def test_observe_normalizes_pass_index(self) -> None:
        store = TrialKnowledgeStore()

        store.observe("game", _transition(), pass_index=float("inf"))
        store.observe("game", _transition(), pass_index=10**12)
        records = store.snapshot("game")["transition_records"]

        self.assertEqual(records[0]["pass_index"], 0)
        self.assertEqual(records[1]["pass_index"], 1_000_000)

    def test_scalar_strategy_plan_steps_are_not_split_into_characters(self) -> None:
        store = TrialKnowledgeStore()

        store.observe(
            "game",
            _transition(level_completed=True),
            strategy={"plan_steps": "LEFT,RIGHT"},
        )

        self.assertEqual(
            store.snapshot("game")["progress_lessons"][0]["plan_steps"], []
        )

    def test_snapshot_handles_malformed_legacy_pass_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(
                json.dumps(
                    {
                        "games": {
                            "game": {
                                "transitions": [
                                    {**_transition(), "pass_index": "invalid"}
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            snapshot = TrialKnowledgeStore(persistence_path=path).snapshot("game")

            self.assertEqual(snapshot["prior_trials"], 1)

    def test_snapshot_handles_malformed_legacy_version_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 3,
                        "games": {
                            "game": {
                                "transitions": [
                                    {
                                        **_transition(),
                                        "state_context_version": float("inf"),
                                        "legacy_state_identity": "false",
                                    }
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            snapshot = TrialKnowledgeStore(persistence_path=path).snapshot("game")

            self.assertEqual(snapshot["legacy_state_identity_observations"], 0)


if __name__ == "__main__":
    unittest.main()
