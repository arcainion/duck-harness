from __future__ import annotations

import json
from unittest import TestCase

from viewer.data import _extract_python_code, _render_tool_arguments


PROGRAM = {
    "version": 1,
    "body": [
        {
            "kind": "expr",
            "value": {
                "kind": "call",
                "function": {"kind": "name", "name": "print"},
                "args": [{"kind": "name", "name": "current_frame"}],
            },
        }
    ],
}


class ProgramIRViewerTests(TestCase):
    def test_program_ir_renders_lowered_preview_and_metadata(self) -> None:
        rendered = _render_tool_arguments("python", {"program": PROGRAM})
        self.assertIn("compiled by duck-program-ir/1.24", rendered)
        self.assertIn("sha256=", rendered)
        self.assertIn("src_sha256=", rendered)
        self.assertIn("print(current_frame)", rendered)
        self.assertNotIn('"kind": "call"', rendered)

    def test_program_ir_json_extracts_lowered_source_for_metrics(self) -> None:
        source = _extract_python_code(json.dumps({"program": PROGRAM}))
        self.assertEqual(source, "print(current_frame)")

    def test_program_ir_markup_extracts_lowered_source_for_metrics(self) -> None:
        content = (
            "<tool_call>\n<function=python>\n<parameter=program>\n"
            f"{json.dumps(PROGRAM)}\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        self.assertEqual(_extract_python_code(content), "print(current_frame)")

    def test_legacy_code_trace_remains_readable(self) -> None:
        legacy_json = json.dumps({"code": "print(current_frame)"})
        legacy_markup = (
            "<tool_call>\n<function=python>\n<parameter=code>\n"
            "print(current_frame)\n</parameter>\n</function>\n</tool_call>"
        )
        self.assertEqual(_extract_python_code(legacy_json), "print(current_frame)")
        self.assertEqual(_extract_python_code(legacy_markup), "print(current_frame)")

    def test_mixed_raw_code_and_program_trace_is_not_counted_as_executed(self) -> None:
        mixed_json = json.dumps({"code": "print('never')", "program": PROGRAM})
        mixed_markup = (
            "<tool_call>\n<function=python>\n"
            "<parameter=code>print('never')</parameter>\n"
            f"<parameter=program>{json.dumps(PROGRAM)}</parameter>\n"
            "</function>\n</tool_call>"
        )
        self.assertEqual(_extract_python_code(mixed_json), "")
        self.assertEqual(_extract_python_code(mixed_markup), "")

    def test_invalid_program_falls_back_to_json(self) -> None:
        rendered = _render_tool_arguments("python", {"program": {"version": 1, "body": []}})
        self.assertIn('"body": []', rendered)
