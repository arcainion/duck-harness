from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from inference.tools.codegen_quality import evaluate_samples, load_jsonl


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
        self.assertEqual(summary["failures"]["invalid_program"], 1)
        self.assertEqual(summary["failures"]["action_contract"], 1)
        self.assertEqual(summary["failures"]["expected_action"], 1)

    def test_jsonl_loader_rejects_non_object_records(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.jsonl"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "expected one JSON object"):
                load_jsonl(path)
