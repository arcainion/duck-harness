from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from inference.tools.codegen_quality import (
    _threshold_failures,
    evaluate_samples,
    load_jsonl,
    main,
)


class CodegenQualityTests(TestCase):
    def test_evaluator_reports_program_action_and_expected_action_quality(self) -> None:
        summary = evaluate_samples(
            [
                {
                    "code": "action('LEFT')",
                    "valid_actions": ["LEFT", "RIGHT"],
                    "expected_actions": ["LEFT"],
                    "expect_action": True,
                },
                {
                    "code": "action('UP')",
                    "valid_actions": ["LEFT"],
                    "expected_actions": ["LEFT"],
                    "expect_action": True,
                },
                {"code": "for", "expect_action": False},
                {"code": "result = current_frame.shape", "expect_action": False},
            ]
        ).to_dict()

        self.assertEqual(summary["samples"], 4)
        self.assertEqual(summary["valid_programs"], 3)
        self.assertEqual(summary["action_contract_passes"], 2)
        self.assertEqual(summary["expected_action_passes"], 2)
        self.assertEqual(summary["preflight_passes"], 3)
        self.assertEqual(summary["executable_samples"], 0)
        self.assertEqual(summary["failures"]["invalid_program"], 1)
        self.assertEqual(summary["failures"]["action_contract"], 1)
        self.assertEqual(summary["failures"]["expected_action"], 1)

    def test_jsonl_loader_rejects_non_object_records(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.jsonl"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected one JSON object"):
                load_jsonl(path)

    def test_jsonl_loader_reports_source_line_and_accepts_utf8_bom(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.jsonl"
            path.write_text(
                '\ufeff{"code":"result = 1"}\n\n{"code":',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, rf"{path}:3: invalid JSON"
            ):
                load_jsonl(path)

    def test_evaluator_executes_result_and_action_contracts(self) -> None:
        summary = evaluate_samples(
            [
                {
                    "code": "action('LEFT')\nresult = 'acted'",
                    "execute": True,
                    "initial_state": {
                        "current_frame": {"grid": [[0, 0]], "step": 0, "level": 1},
                        "history": [],
                    },
                    "valid_actions": ["LEFT"],
                    "expected_actions": ["LEFT"],
                    "expected_executed_actions": ["LEFT"],
                    "expected_result": "acted",
                    "expect_action": True,
                }
            ]
        ).to_dict()

        self.assertEqual(summary["execution_rate"], 1.0)
        self.assertEqual(summary["result_rate"], 1.0)

    def test_empty_corpus_has_zero_rates_without_division_errors(self) -> None:
        summary = evaluate_samples([]).to_dict()

        self.assertEqual(summary["samples"], 0)
        self.assertEqual(summary["valid_program_rate"], 0.0)
        self.assertEqual(summary["execution_rate"], 0.0)
        self.assertEqual(summary["result_rate"], 0.0)

    def test_dynamic_action_satisfies_contract_but_not_unseen_expected_action(self) -> None:
        summary = evaluate_samples(
            [
                {
                    "code": "choice = valid_actions[0]\naction(choice)",
                    "valid_actions": ["LEFT"],
                    "expected_actions": ["LEFT"],
                    "expect_action": True,
                }
            ]
        ).to_dict()

        self.assertEqual(summary["action_contract_passes"], 1)
        self.assertEqual(summary["expected_action_passes"], 0)
        self.assertEqual(summary["failures"], {"expected_action": 1})

    def test_evaluator_ignores_actions_in_dead_or_uncalled_code(self) -> None:
        summary = evaluate_samples(
            [
                {
                    "code": (
                        "def dormant():\n"
                        "    action('LEFT')\n"
                        "if False:\n"
                        "    action('RIGHT')\n"
                        "result = 1"
                    ),
                    "valid_actions": ["LEFT", "RIGHT"],
                    "expect_action": True,
                }
            ]
        ).to_dict()

        self.assertEqual(summary["action_contract_passes"], 0)
        self.assertEqual(summary["failures"]["action_contract"], 1)

    def test_execution_result_mismatch_is_reported_separately(self) -> None:
        summary = evaluate_samples(
            [{"code": "result = 1", "execute": True, "expected_result": 2}]
        ).to_dict()

        self.assertEqual(summary["execution_passes"], 1)
        self.assertEqual(summary["result_passes"], 0)
        self.assertEqual(summary["failures"]["result"], 1)

    def test_main_returns_failure_and_writes_report_when_threshold_is_missed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus.jsonl"
            thresholds = root / "thresholds.json"
            report = root / "reports" / "quality.json"
            corpus.write_text('{"code":"result = 1"}\n', encoding="utf-8")
            thresholds.write_text(
                json.dumps({"min_samples": 2, "min_valid_program_rate": 1.0}),
                encoding="utf-8",
            )
            output = StringIO()
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "codegen-quality",
                        str(corpus),
                        "--thresholds",
                        str(thresholds),
                        "--output",
                        str(report),
                    ],
                ),
                redirect_stdout(output),
            ):
                exit_code = main()

            self.assertEqual(exit_code, 1)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["samples"], 1)
            self.assertIn('"gate_passed": false', output.getvalue())
            self.assertIn("samples=1 < 2", output.getvalue())

    def test_threshold_validation_rejects_unknown_and_out_of_range_values(self) -> None:
        summary = evaluate_samples([]).to_dict()

        invalid = (
            ({"min_valid_progam_rate": 1.0}, "unknown quality rate"),
            ({"label": "strict"}, "unknown quality threshold"),
            ({"min_samples": True}, "non-negative integer"),
            ({"min_execution_rate": 1.1}, "between 0 and 1"),
        )
        for thresholds, expected in invalid:
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex(ValueError, expected):
                    _threshold_failures(summary, thresholds)
