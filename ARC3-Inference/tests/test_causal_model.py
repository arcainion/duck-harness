from __future__ import annotations

import unittest

from inference.agent.causal_model import normalize_causal_model


class CausalModelNormalizationTests(unittest.TestCase):
    def test_scalar_sections_are_ignored(self) -> None:
        normalized = normalize_causal_model(
            {
                "entities": 3,
                "relations": {"cause": "a"},
                "subgoals": "pending",
                "predictions": object(),
            }
        )

        self.assertEqual(
            normalized,
            {"entities": [], "relations": [], "subgoals": [], "predictions": []},
        )

    def test_scalar_subgoal_dependencies_are_ignored(self) -> None:
        normalized = normalize_causal_model(
            {"subgoals": [{"id": "goal", "depends_on": "not-a-list"}]}
        )

        self.assertEqual(normalized["subgoals"][0]["depends_on"], [])

    def test_nonfinite_confidence_is_neutralized(self) -> None:
        normalized = normalize_causal_model(
            {
                "entities": [
                    {"id": "nan", "confidence": float("nan")},
                    {"id": "infinite", "confidence": float("inf")},
                ]
            }
        )

        self.assertEqual(
            [item["confidence"] for item in normalized["entities"]], [0.0, 0.0]
        )

    def test_invalid_relation_counters_are_neutralized(self) -> None:
        normalized = normalize_causal_model(
            {
                "relations": [
                    {
                        "cause": "input",
                        "effect": "output",
                        "support": float("inf"),
                        "contradictions": True,
                    }
                ]
            }
        )

        self.assertEqual(normalized["relations"][0]["support"], 0)
        self.assertEqual(normalized["relations"][0]["contradictions"], 0)


if __name__ == "__main__":
    unittest.main()
