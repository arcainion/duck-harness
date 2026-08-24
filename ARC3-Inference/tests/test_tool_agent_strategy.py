from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from inference.agent.action_names import MAX_ACTION_BATCH
from inference.agent.inference_controller import InferenceControllerConfig
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent, _remaining_deadline_seconds
from inference.agent.trial_knowledge import TrialKnowledgeStore


class ToolAgentStrategyTests(TestCase):
    def test_request_deadline_declines_and_never_becomes_negative(self) -> None:
        self.assertIsNone(_remaining_deadline_seconds(None, now=50.0))
        self.assertEqual(_remaining_deadline_seconds(60.0, now=52.5), 7.5)
        self.assertEqual(_remaining_deadline_seconds(60.0, now=61.0), 0.0)

    def _agent(self) -> ToolAgent:
        agent = ToolAgent(
            model="unit-test-model",
            provider="vllm",
            base_url="http://127.0.0.1:1/v1",
        )
        agent._controller_config = InferenceControllerConfig(enabled=True)
        return agent

    def test_python_action_batch_is_bounded_before_host_dispatch(self) -> None:
        agent = self._agent()
        with self.assertRaisesRegex(ValueError, "at most 12 actions"):
            agent._normalize_python_actions(["LEFT"] * 13)

    def test_compact_action_result_tolerates_malformed_host_counts(self) -> None:
        agent = self._agent()

        compact = agent._compact_action_result(
            {
                "executed": True,
                "requested_count": "many",
                "executed_count": "several",
                "stopped_early": True,
                "executed_actions": ["LEFT"] * (MAX_ACTION_BATCH + 3),
            }
        )

        self.assertEqual(compact["requested_count"], 1)
        self.assertEqual(compact["executed_count"], 1)
        self.assertEqual(len(compact["executed_actions"]), MAX_ACTION_BATCH)

    def test_compact_action_result_keeps_bounded_animation_summary(self) -> None:
        agent = self._agent()
        animation = {
            "frame_count": 3,
            "transient_changed_cells": 4,
            "motion_bbox": [1, 2, 5, 8],
        }

        compact = agent._compact_action_result(
            {"executed": True, "action_display": "SPACE", "animation": animation}
        )

        self.assertEqual(compact["animation"], animation)

    def test_structured_strategy_is_bounded_and_updates_world_model(self) -> None:
        agent = self._agent()

        saved = agent._record_strategy(
            {
                "goal": "g" * 500,
                "hypothesis": "buttons move the matching object",
                "evidence": ["first", "second"],
                "confidence": 4,
                "open_question": "which color is selected?",
                "next_test": "press SPACE once",
                "subgoals": [f"subgoal {index}" for index in range(10)],
                "current_subgoal": "align the selector",
                "plan_steps": [f"step {index}" for index in range(12)],
                "success_criteria": "the score increases",
                "causal_model": {
                    "entities": [
                        {"id": "selector", "kind": "cursor", "confidence": 0.8}
                    ],
                    "relations": [
                        {
                            "cause": "SPACE",
                            "effect": "selector advances",
                            "support": 2,
                            "confidence": 0.9,
                        }
                    ],
                    "subgoals": [
                        {
                            "id": "align",
                            "description": "align selector",
                            "status": "active",
                        }
                    ],
                    "predictions": [
                        {"action": "SPACE", "expected_changes": "selector advances"}
                    ],
                },
            }
        )

        self.assertLessEqual(len(saved["goal"]), 280)
        self.assertEqual(saved["confidence"], 1.0)
        self.assertEqual(
            agent._summarized_knowledge["current_plan"], "press SPACE once"
        )
        self.assertEqual(len(saved["subgoals"]), 6)
        self.assertEqual(len(saved["plan_steps"]), 8)
        self.assertEqual(saved["current_subgoal"], "align the selector")
        self.assertEqual(saved["causal_model"]["relations"][0]["support"], 2)

    def test_recording_causal_model_refreshes_live_experience_and_cache(self) -> None:
        agent = self._agent()
        agent._current_experience_snapshot = {"state_id": "state-a"}
        agent._experience_snapshot_cache = object()

        saved = agent._record_strategy(
            {
                "causal_model": {
                    "relations": [
                        {
                            "cause": "RIGHT",
                            "effect": "outcome:novel",
                            "conditions": "state:state-a",
                            "confidence": 0.7,
                        }
                    ]
                }
            }
        )

        self.assertEqual(
            agent._current_experience_snapshot["causal_model"],
            saved["causal_model"],
        )
        self.assertIsNone(agent._experience_snapshot_cache)

    def test_cross_trial_store_exposes_verified_progress_without_raw_frames(
        self,
    ) -> None:
        store = TrialKnowledgeStore()
        store.observe(
            "game",
            {
                "executed": True,
                "before_state_id": "state-a",
                "after_state_id": "state-b",
                "action_display": "SPACE",
                "outcome_class": "level_progress",
                "level_completed": True,
            },
            strategy={"hypothesis": "SPACE confirms a complete arrangement"},
            pass_index=1,
        )

        snapshot = store.snapshot("game", state_id="state-a")

        self.assertEqual(snapshot["state_action_evidence"][0]["action"], "SPACE")
        self.assertEqual(snapshot["progress_lessons"][0]["pass_index"], 1)
        self.assertNotIn("grid", str(snapshot).lower())

    def test_cross_trial_snapshot_excludes_the_live_evidence_source(self) -> None:
        store = TrialKnowledgeStore()
        for evidence_id, action in (("prior-pass", "LEFT"), ("live-pass", "RIGHT")):
            store.observe(
                "game",
                {
                    "executed": True,
                    "before_state_id": "state-a",
                    "after_state_id": f"after-{action}",
                    "action_display": action,
                    "outcome_class": "novel",
                    "level_completed": evidence_id == "live-pass",
                },
                evidence_id=evidence_id,
            )

        snapshot = store.snapshot("game", exclude_evidence_id="live-pass")

        self.assertEqual(snapshot["observations"], 1)
        self.assertEqual(snapshot["transition_records"][0]["action_display"], "LEFT")
        self.assertEqual(snapshot["progress_lessons"], [])

    def test_experience_cache_invalidates_when_shared_knowledge_changes(self) -> None:
        store = TrialKnowledgeStore()
        agent = self._agent()
        agent._knowledge_store = store
        agent._knowledge_game_id = "game"
        agent._knowledge_evidence_id = "live-pass"
        frame = Frame(grid=((1,),), step=0, level=1)
        history = [HistoryEntry(action="", frame=frame)]
        first = agent._cached_experience_snapshot(frame, history, ["RIGHT"])

        store.observe(
            "game",
            {
                "executed": True,
                "before_state_id": "prior-state",
                "after_state_id": "prior-result",
                "action_display": "RIGHT",
                "outcome_class": "novel",
            },
            evidence_id="prior-pass",
        )
        second = agent._cached_experience_snapshot(frame, history, ["RIGHT"])

        self.assertIsNot(first, second)
        self.assertEqual(second["cross_trial"]["prior_trials"], 1)

    def test_cross_trial_store_resumes_from_atomic_json(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "knowledge.json"
            store = TrialKnowledgeStore(persistence_path=path)
            store.observe(
                "game",
                {
                    "executed": True,
                    "before_state_id": "state-a",
                    "after_state_id": "state-b",
                    "behavioral_before_state_id": "behavior-a",
                    "behavioral_after_state_id": "behavior-b",
                    "action_display": "RIGHT",
                    "outcome_class": "novel",
                },
                pass_index=2,
            )

            resumed = TrialKnowledgeStore(persistence_path=path).snapshot("game")

            self.assertEqual(resumed["observations"], 1)
            self.assertEqual(resumed["transition_records"][0]["pass_index"], 2)

    def test_cross_trial_store_merges_stale_process_views_and_recovers_backup(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "knowledge.json"
            first = TrialKnowledgeStore(persistence_path=path)
            second = TrialKnowledgeStore(persistence_path=path)
            base = {
                "executed": True,
                "before_state_id": "state-a",
                "after_state_id": "state-b",
                "action_display": "RIGHT",
                "outcome_class": "novel",
            }
            first.observe("game", base, evidence_id="run:pass=0")
            second.observe(
                "game",
                {**base, "action_display": "LEFT"},
                evidence_id="run:pass=1",
            )

            merged = TrialKnowledgeStore(persistence_path=path).snapshot("game")
            self.assertEqual(merged["observations"], 2)
            self.assertEqual(merged["independent_evidence"], 2)

            path.write_text("{corrupt", encoding="utf-8")
            recovered = TrialKnowledgeStore(persistence_path=path).snapshot("game")
            self.assertGreaterEqual(recovered["observations"], 1)

    def test_full_stale_store_merge_retains_new_local_observation(self) -> None:
        store = TrialKnowledgeStore(transition_limit=32)
        store.observe(
            "game",
            {
                "executed": True,
                "before_state_id": "fresh-before",
                "after_state_id": "fresh-after",
                "action_display": "SPACE",
                "outcome_class": "novel",
            },
            evidence_id="fresh-process",
        )
        disk_records = [
            {
                "before_state_id": f"old-{index}",
                "after_state_id": f"old-{index + 1}",
                "action_display": "LEFT",
                "outcome_class": "revisit",
                "evidence_id": f"old-process-{index}",
                "observed_at": f"2025-01-01T00:00:{index:02d}Z",
            }
            for index in range(32)
        ]
        path = mock.Mock(spec=Path)
        path.exists.return_value = True
        path.read_text.return_value = json.dumps(
            {"games": {"game": {"transitions": disk_records, "lessons": []}}}
        )

        store._merge_disk_locked(path)

        merged = store.snapshot("game")
        self.assertEqual(merged["observations"], 32)
        self.assertIn(
            "fresh-process",
            {item["evidence_id"] for item in merged["transition_records"]},
        )

    def test_cross_trial_action_evidence_weights_independent_sources(self) -> None:
        store = TrialKnowledgeStore()
        base = {
            "executed": True,
            "before_state_id": "state-a",
            "after_state_id": "state-b",
            "action_display": "RIGHT",
        }
        for _ in range(6):
            store.observe(
                "game",
                {**base, "outcome_class": "exact_noop"},
                evidence_id="run-a:pass=0",
            )
        for pass_index in (1, 2, 3):
            store.observe(
                "game",
                {**base, "outcome_class": "novel"},
                evidence_id=f"run-a:pass={pass_index}",
            )

        evidence = store.snapshot("game", state_id="state-a")["state_action_evidence"][
            0
        ]

        self.assertEqual(evidence["outcomes"], {"exact_noop": 1, "novel": 3})
        self.assertEqual(evidence["trials"], 4)
        self.assertEqual(evidence["raw_observations"], 9)

    def test_adaptive_inference_budget_reduces_candidates_near_exhaustion(self) -> None:
        agent = self._agent()
        agent._candidate_count = 3
        agent._current_experience_snapshot = {
            "phase": "recover",
            "ranked_actions": [{"uncertainty": 1.0}],
        }
        with mock.patch(
            "inference.agent.tool_agent._LOCAL_ANALYZER_GAME_TOKEN_BUDGET", 100
        ):
            agent._session_generated_tokens = 10
            self.assertEqual(agent._adaptive_candidate_count(20), 3)
            agent._session_generated_tokens = 70
            self.assertEqual(agent._adaptive_candidate_count(20), 1)
            self.assertEqual(agent._adaptive_output_limit(1), 30)

    def test_semantic_candidate_score_prefers_plan_and_avoids_known_mouse_noop(
        self,
    ) -> None:
        agent = self._agent()
        agent._current_valid_actions = ["RIGHT", "MOUSE"]
        agent._current_experience_snapshot = {
            "action_budget": 1,
            "ranked_actions": [{"action": "RIGHT", "priority": 0}],
            "discouraged_actions": ["MOUSE(row=1, col=1)"],
            "mouse_search": {"recent": [{"row": 1, "col": 1, "outcome": "exact_noop"}]},
            "recommended_plan": {"actions": ["RIGHT"]},
        }

        planned = agent._score_candidate_choice(
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "python",
                                "arguments": '{"code":"action([\\"RIGHT\\"])"}',
                            }
                        }
                    ]
                }
            }
        )[0]
        noop = agent._score_candidate_choice(
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "python",
                                "arguments": '{"code":"action([{\\"action\\":\\"MOUSE\\",\\"row\\":1,\\"col\\":1}])"}',
                            }
                        }
                    ]
                }
            }
        )[0]

        self.assertGreater(planned, noop)

    def test_semantic_candidate_score_uses_empirical_outcomes(self) -> None:
        agent = self._agent()
        agent._current_experience_snapshot = {
            "action_budget": 1,
            "transition_models_here": [
                {
                    "action": "RIGHT",
                    "predicted_outcome": "level_progress",
                    "confidence": 1.0,
                    "contradictions": 0,
                },
                {
                    "action": "LEFT",
                    "predicted_outcome": "exact_noop",
                    "confidence": 1.0,
                    "contradictions": 0,
                },
            ],
        }

        self.assertGreater(
            agent._semantic_action_score([[{"action": "RIGHT"}]]),
            agent._semantic_action_score([[{"action": "LEFT"}]]),
        )

    def test_candidate_scoring_resolves_literal_action_variables(self) -> None:
        agent = self._agent()
        agent._current_valid_actions = ["RIGHT"]
        agent._current_experience_snapshot = {
            "action_budget": 1,
            "outcome_utilities": {"level_progress": 1.0},
            "transition_models_here": [
                {
                    "action": "RIGHT",
                    "predicted_outcome": "level_progress",
                    "confidence": 1.0,
                    "contradictions": 0,
                }
            ],
        }

        score, valid = agent._score_candidate_choice(
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "python",
                                "arguments": '{"code":"moves = [\\"RIGHT\\"]\\naction(moves)"}',
                            }
                        }
                    ]
                }
            }
        )

        self.assertTrue(valid)
        self.assertGreater(score, 200)

    def test_known_terminal_first_action_is_blocked_before_dispatch(self) -> None:
        agent = self._agent()
        agent._current_valid_actions = ["RIGHT", "LEFT"]
        agent._current_experience_snapshot = {
            "transition_models_here": [{"action": "RIGHT", "terminal_failures": 1}]
        }

        with self.assertRaisesRegex(ValueError, "terminal-failure"):
            agent._normalize_python_actions(["RIGHT"])
        self.assertEqual(
            agent._normalize_python_actions(["LEFT"]), [{"action": "LEFT"}]
        )

        self.assertEqual(
            agent._normalize_python_actions(["LEFT", "RIGHT"]),
            [{"action": "LEFT"}, {"action": "RIGHT"}],
        )

    def test_conflicting_harm_evidence_allows_cautious_revalidation(self) -> None:
        agent = self._agent()
        agent._current_valid_actions = ["RIGHT"]
        agent._current_experience_snapshot = {
            "transition_models_here": [
                {"action": "RIGHT", "terminal_failures": 1, "trials": 2}
            ]
        }

        self.assertEqual(
            agent._normalize_python_actions(["RIGHT"]), [{"action": "RIGHT"}]
        )

    def test_executed_actions_ground_causal_relations_and_predictions(self) -> None:
        agent = self._agent()
        agent._record_strategy(
            {
                "causal_model": {
                    "predictions": [
                        {
                            "action": "RIGHT",
                            "expected_changes": "reach a new state",
                            "expected_outcome": "novel",
                            "confidence": 0.7,
                        }
                    ]
                }
            }
        )

        agent._update_causal_model_from_actions(
            [
                {
                    "executed": True,
                    "action_display": "RIGHT",
                    "outcome_class": "novel",
                    "before_state_id": "state-a",
                }
            ]
        )

        causal = agent._strategy_memory["causal_model"]
        self.assertEqual(causal["predictions"][0]["status"], "supported")
        self.assertEqual(causal["relations"][0]["effect"], "outcome:novel")
        self.assertEqual(causal["relations"][0]["support"], 1)

    def test_grounded_causal_relations_are_conditioned_by_state(self) -> None:
        agent = self._agent()

        agent._update_causal_model_from_actions(
            [
                {
                    "executed": True,
                    "action_display": "RIGHT",
                    "outcome_class": "novel",
                    "before_state_id": "state-a",
                },
                {
                    "executed": True,
                    "action_display": "RIGHT",
                    "outcome_class": "exact_noop",
                    "before_state_id": "state-b",
                },
            ]
        )

        relations = agent._strategy_memory["causal_model"]["relations"]
        self.assertEqual(len(relations), 2)
        self.assertEqual(
            {relation["conditions"] for relation in relations},
            {"state:state-a", "state:state-b"},
        )
        self.assertTrue(all(relation["contradictions"] == 0 for relation in relations))

    def test_causal_prediction_is_only_resolved_in_its_recorded_state(self) -> None:
        agent = self._agent()
        agent._current_experience_snapshot = {
            "state_id": "exact-a",
            "behavioral_state_id": "stable-a",
        }
        agent._record_strategy(
            {
                "causal_model": {
                    "predictions": [
                        {
                            "action": "RIGHT",
                            "expected_outcome": "novel",
                            "confidence": 0.7,
                        }
                    ]
                }
            }
        )

        agent._update_causal_model_from_actions(
            [
                {
                    "executed": True,
                    "action_display": "RIGHT",
                    "outcome_class": "novel",
                    "behavioral_before_state_id": "stable-b",
                }
            ]
        )
        prediction = agent._strategy_memory["causal_model"]["predictions"][0]
        self.assertEqual(prediction["status"], "untested")
        self.assertEqual(prediction["conditions"], "state:stable-a")

        agent._update_causal_model_from_actions(
            [
                {
                    "executed": True,
                    "action_display": "RIGHT",
                    "outcome_class": "novel",
                    "behavioral_before_state_id": "stable-a",
                }
            ]
        )

        self.assertEqual(
            agent._strategy_memory["causal_model"]["predictions"][0]["status"],
            "supported",
        )

    def test_new_grounded_relation_evicts_weak_stale_automatic_relation(self) -> None:
        agent = self._agent()
        agent._strategy_memory["causal_model"] = {
            "relations": [
                {
                    "cause": "RIGHT",
                    "effect": "outcome:novel",
                    "conditions": f"state:old-{index}",
                    "evidence": "automatically grounded from executed transition",
                    "confidence": index / 100,
                    "support": index + 1,
                    "last_observed_action": index,
                }
                for index in range(24)
            ]
        }

        agent._update_causal_model_from_actions(
            [
                {
                    "executed": True,
                    "action_display": "SPACE",
                    "outcome_class": "level_progress",
                    "before_state_id": "new-state",
                    "action_num": 99,
                }
            ]
        )

        relations = agent._strategy_memory["causal_model"]["relations"]
        self.assertEqual(len(relations), 24)
        self.assertNotIn("state:old-0", {item["conditions"] for item in relations})
        new_relation = next(
            item for item in relations if item["conditions"] == "state:new-state"
        )
        self.assertEqual(new_relation["last_observed_action"], 99)

    def test_candidate_scoring_ignores_causal_relations_from_other_states(self) -> None:
        agent = self._agent()
        relations = [
            {
                "cause": "RIGHT",
                "effect": "outcome:level_progress",
                "conditions": "state:state-a",
                "confidence": 1.0,
            },
            {
                "cause": "RIGHT",
                "effect": "outcome:exact_noop",
                "conditions": "state:state-b",
                "confidence": 1.0,
            },
        ]
        agent._current_experience_snapshot = {
            "state_id": "state-b",
            "behavioral_state_id": "stable-b",
            "action_budget": 1,
            "causal_model": {"relations": relations},
        }

        state_b_score = agent._semantic_action_score([[{"action": "RIGHT"}]])
        agent._current_experience_snapshot["state_id"] = "state-a"
        state_a_score = agent._semantic_action_score([[{"action": "RIGHT"}]])

        self.assertLess(state_b_score, 0)
        self.assertGreater(state_a_score, 0)

    def test_candidate_scoring_penalizes_untriggered_reset(self) -> None:
        agent = self._agent()
        agent._current_experience_snapshot = {
            "action_budget": 1,
            "ranked_actions": [
                {"action": "RESET", "priority": 6},
                {"action": "RIGHT", "priority": 1},
            ],
            "recovery_portfolio": [],
        }

        reset_score = agent._semantic_action_score([[{"action": "RESET"}]])
        right_score = agent._semantic_action_score([[{"action": "RIGHT"}]])

        self.assertLess(reset_score, right_score)

    def test_candidate_scoring_does_not_double_count_exact_and_object_relations(
        self,
    ) -> None:
        agent = self._agent()
        exact = {
            "cause": "RIGHT",
            "effect": "outcome:level_progress",
            "conditions": "state:stable-a",
            "confidence": 0.8,
        }
        abstract = {
            "cause": "RIGHT",
            "effect": "outcome:level_progress",
            "conditions": "object_state:objects-a",
            "confidence": 0.8,
        }
        agent._current_experience_snapshot = {
            "state_id": "exact-a",
            "behavioral_state_id": "stable-a",
            "object_state_id": "objects-a",
            "action_budget": 1,
            "causal_model": {"relations": [exact, abstract]},
        }
        combined_score = agent._semantic_action_score([[{"action": "RIGHT"}]])
        agent._current_experience_snapshot["causal_model"] = {"relations": [exact]}
        exact_score = agent._semantic_action_score([[{"action": "RIGHT"}]])

        self.assertEqual(combined_score, exact_score)

    def test_empirical_model_subsumes_duplicate_cross_trial_and_grounded_evidence(
        self,
    ) -> None:
        agent = self._agent()
        base_snapshot = {
            "state_id": "state-a",
            "behavioral_state_id": "stable-a",
            "action_budget": 1,
            "transition_models_here": [
                {
                    "action": "RIGHT",
                    "predicted_outcome": "level_progress",
                    "confidence": 1.0,
                }
            ],
            "causal_model": {
                "relations": [
                    {
                        "cause": "RIGHT",
                        "effect": "outcome:level_progress",
                        "conditions": "state:stable-a",
                        "confidence": 1.0,
                        "evidence": "automatically grounded from executed transition",
                    }
                ]
            },
        }
        agent._current_experience_snapshot = {
            **base_snapshot,
            "cross_trial": {
                "state_action_evidence": [
                    {
                        "action": "RIGHT",
                        "outcomes": {"level_progress": 1},
                        "trials": 1,
                    }
                ]
            },
        }
        redundant_score = agent._semantic_action_score([[{"action": "RIGHT"}]])
        agent._current_experience_snapshot = dict(base_snapshot)
        model_only_score = agent._semantic_action_score([[{"action": "RIGHT"}]])

        self.assertEqual(redundant_score, model_only_score)

    def test_object_conditioned_causal_relation_scores_equivalent_state(self) -> None:
        agent = self._agent()
        agent._update_causal_model_from_actions(
            [
                {
                    "executed": True,
                    "action_display": "RIGHT",
                    "outcome_class": "level_progress",
                    "before_state_id": "translated-state-a",
                    "object_before_state_id": "shared-object-layout",
                }
            ]
        )
        causal = agent._strategy_memory["causal_model"]
        self.assertIn(
            "object_state:shared-object-layout",
            {relation["conditions"] for relation in causal["relations"]},
        )
        agent._current_experience_snapshot = {
            "state_id": "translated-state-b",
            "behavioral_state_id": "translated-stable-b",
            "object_state_id": "shared-object-layout",
            "action_budget": 1,
            "causal_model": causal,
        }

        equivalent_score = agent._semantic_action_score([[{"action": "RIGHT"}]])
        agent._current_experience_snapshot["object_state_id"] = "different-layout"
        unrelated_score = agent._semantic_action_score([[{"action": "RIGHT"}]])

        self.assertGreater(equivalent_score, unrelated_score)

    def test_compact_action_result_preserves_object_transition_identity(self) -> None:
        compact = self._agent()._compact_action_result(
            {
                "executed": True,
                "object_before_state_id": "object-a",
                "object_after_state_id": "object-b",
                "decision_context_changed": True,
            }
        )

        self.assertEqual(compact["object_before_state_id"], "object-a")
        self.assertEqual(compact["object_after_state_id"], "object-b")
        self.assertTrue(compact["decision_context_changed"])

    def test_batch_results_expand_every_engine_transition_for_learning(self) -> None:
        agent = self._agent()
        observations = agent._engine_transition_observations(
            [
                {
                    "executed": True,
                    "action_display": "RIGHT",
                    "steps": [
                        {
                            "action_num": 1,
                            "action_display": "LEFT",
                            "before_state_id": "state-a",
                            "outcome_class": "novel",
                        },
                        {
                            "action_num": 2,
                            "action_display": "RIGHT",
                            "before_state_id": "state-b",
                            "outcome_class": "level_progress",
                        },
                    ],
                }
            ]
        )

        self.assertEqual(
            [item["action_display"] for item in observations], ["LEFT", "RIGHT"]
        )
        self.assertTrue(all(item["executed"] for item in observations))
        agent._update_causal_model_from_actions(observations)
        self.assertEqual(
            {
                relation["cause"]
                for relation in agent._strategy_memory["causal_model"]["relations"]
            },
            {"LEFT", "RIGHT"},
        )

    def test_prediction_is_evaluated_once_then_consumed(self) -> None:
        agent = self._agent()
        agent._record_strategy(
            {
                "test_action": "space",
                "expected_outcome": "level_progress",
                "fallback": "inspect a different object",
                "contradictions": ["old evidence was ambiguous"] * 10,
            }
        )
        self.assertEqual(
            agent._strategy_memory["fallback"], "inspect a different object"
        )
        self.assertLessEqual(len(agent._strategy_memory["contradictions"]), 3)

        action_result = {
            "executed": True,
            "action_display": "SPACE",
            "outcome_class": "level_progress",
            "level_completed": True,
            "reward": 0.1,
        }
        result = agent._evaluate_strategy_prediction(action_result)

        self.assertEqual(result["status"], "supported")
        self.assertNotIn("test_action", agent._strategy_memory)
        self.assertNotIn("expected_outcome", agent._strategy_memory)
        self.assertNotIn("prediction_result", agent._strategy_memory)
        consumed = agent._strategy_memory["last_evaluated_prediction"]
        self.assertEqual(consumed["test_action"], "SPACE")
        self.assertEqual(consumed["expected_outcome"], "level_progress")
        self.assertEqual(consumed["status"], "supported")

        repeat = agent._evaluate_strategy_prediction(dict(action_result))
        self.assertIsNone(repeat)

    def test_partial_prediction_update_keeps_declared_outcome(self) -> None:
        agent = self._agent()
        agent._record_strategy({"test_action": "LEFT", "expected_outcome": "new_state"})
        agent._record_strategy({"test_action": "RIGHT"})

        self.assertEqual(agent._strategy_memory["test_action"], "RIGHT")
        self.assertEqual(agent._strategy_memory["expected_outcome"], "new_state")

    def test_prompt_contains_bounded_controller_summary_not_raw_grid(self) -> None:
        agent = self._agent()
        frame = Frame(grid=((1, 2), (3, 4)), step=0, level=1)
        history = [HistoryEntry(action="", frame=frame)]
        snapshot = {
            "enabled": True,
            "policy": "outcome_aware",
            "phase": "orient",
            "action_budget": 1,
            "state_id": "opaque-id",
            "state_visits": 1,
            "unique_states": 1,
            "actions_observed": 0,
            "no_op_streak": 0,
            "stagnation_actions": 0,
            "cycle_period": None,
            "tried_here": {},
            "suggested_actions": ["LEFT"],
            "discouraged_actions": [],
            "ranked_actions": [
                {
                    "action": "LEFT",
                    "priority": 1,
                    "reason": "untried action from this state",
                }
            ],
            "transition_models_here": [
                {
                    "action": "RIGHT",
                    "trials": 2,
                    "confidence": 1.0,
                    "verified_deterministic": True,
                }
            ],
            "model_conflicts_here": 0,
            "recent_transitions": [{"large": "x" * 10000}],
        }

        prompt = agent._build_user_prompt(
            0,
            valid_actions=["LEFT"],
            current_frame=frame,
            history_entries=history,
            experience_snapshot=snapshot,
        )

        self.assertIn('"phase":"orient"', prompt)
        self.assertIn('"action_budget":1', prompt)
        self.assertIn('"policy":"outcome_aware"', prompt)
        self.assertIn('"ranked_actions"', prompt)
        self.assertIn('"transition_models_here"', prompt)
        self.assertIn('"verified_deterministic":true', prompt)
        self.assertIn("opaque-id", prompt)
        self.assertNotIn("recent_transitions", prompt)
        self.assertNotIn("[[1, 2], [3, 4]]", prompt)
        self.assertIn("prefer batching it in one call", prompt)
        self.assertIn(
            "Before executing new actions you must always give the revised version",
            prompt,
        )
        self.assertLess(len(prompt), 10_000)

    def test_multimodal_message_includes_previous_and_current_changed_frames(
        self,
    ) -> None:
        agent = self._agent()
        previous = Frame(grid=((1,),), step=0, level=1)
        current = Frame(grid=((2,),), step=1, level=1)
        with mock.patch(
            "inference.agent.tool_agent.current_grid_image_part",
            side_effect=[
                {"type": "image_url", "image_url": {"url": "current"}},
                {"type": "image_url", "image_url": {"url": "previous"}},
            ],
        ):
            message = agent._build_user_message("prompt", current, previous)

        content = message["content"]
        self.assertEqual(content[0]["text"], "prompt")
        self.assertEqual(content[1]["text"], "Previous grid image:")
        self.assertEqual(content[3]["text"], "Current grid image:")

    def test_python_actions_are_canonicalized_and_checked_against_current_actions(
        self,
    ) -> None:
        agent = self._agent()
        agent._current_valid_actions = ["LEFT", "MOUSE"]

        self.assertEqual(
            agent._normalize_python_actions(["ACTION3"]),
            [{"action": "LEFT"}],
        )
        with self.assertRaisesRegex(ValueError, "RIGHT.*not currently valid"):
            agent._normalize_python_actions(["RIGHT"])

        self.assertEqual(
            agent._normalize_python_actions(["LEFT", "RIGHT"]),
            [{"action": "LEFT"}, {"action": "RIGHT"}],
        )

    def test_unknown_python_action_is_rejected_without_a_current_action_list(
        self,
    ) -> None:
        agent = self._agent()
        agent._current_valid_actions = []

        with self.assertRaisesRegex(ValueError, "NOT_REAL.*unknown"):
            agent._normalize_python_actions(["NOT_REAL"])

    def test_mouse_actions_require_integer_row_and_col(self) -> None:
        agent = self._agent()
        agent._current_valid_actions = ["MOUSE"]

        for action in (
            {"action": "MOUSE"},
            {"action": "MOUSE", "row": 1},
            {"action": "MOUSE", "row": True, "col": 2},
            {"action": "MOUSE", "row": 1, "col": 2.5},
        ):
            with (
                self.subTest(action=action),
                self.assertRaisesRegex(
                    ValueError, "requires integer|must be an integer"
                ),
            ):
                agent._normalize_python_actions([action])
