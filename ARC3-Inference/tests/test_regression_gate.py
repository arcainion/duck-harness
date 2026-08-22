from __future__ import annotations

from unittest import TestCase

from inference.tools.regression_gate import evaluate_gate


class RegressionGateTests(TestCase):
    def test_gate_passes_metrics_within_absolute_and_relative_bounds(self) -> None:
        result = evaluate_gate(
            {
                "overall_score": 2.1,
                "rewarding_action_rate": 0.02,
                "no_op_rate": 0.08,
                "repeated_noop_rate": 0.01,
                "terminal_state_violations": 0,
                "trace_count": 25,
            },
            {
                "min_overall_score": 1.5,
                "max_noop_rate": 0.1,
                "min_score_ratio_vs_baseline": 1.0,
            },
            baseline_metrics={"overall_score": 2.0},
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.failures, ())

    def test_gate_reports_every_failed_behavioral_threshold(self) -> None:
        result = evaluate_gate(
            {
                "overall_score": 1.0,
                "rewarding_action_rate": 0.001,
                "no_op_rate": 0.3,
                "repeated_noop_rate": 0.2,
                "terminal_state_violations": 2,
                "trace_count": 2,
            },
            {
                "min_overall_score": 1.5,
                "min_rewarding_action_rate": 0.01,
                "max_noop_rate": 0.1,
                "max_repeated_noop_rate": 0.05,
                "max_terminal_state_violations": 0,
                "min_trace_count": 20,
            },
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.failures), 6)


if __name__ == "__main__":
    import unittest

    unittest.main()
