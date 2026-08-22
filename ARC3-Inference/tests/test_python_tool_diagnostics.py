from __future__ import annotations

import unittest

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _sandbox_exception_diagnostic,
)
from inference.agent.tool_agent import _python_tool_payload


class PythonToolDiagnosticTests(unittest.TestCase):
    def test_runtime_error_reports_user_source_and_fresh_call_hint(self) -> None:
        code = "value = 1\nresult = missing_name + value"
        try:
            exec(compile(code, "<python_tool>", "exec"), {})
        except NameError as exc:
            diagnostic = _sandbox_exception_diagnostic(exc, code)
        else:  # pragma: no cover
            self.fail("Expected generated code to raise NameError")

        self.assertEqual(diagnostic["type"], "NameError")
        self.assertEqual(diagnostic["line"], 2)
        self.assertEqual(diagnostic["source"], "result = missing_name + value")
        self.assertIn("starts fresh", diagnostic["hint"])

    def test_syntax_error_reports_line_column_and_source(self) -> None:
        code = "if True\n    result = 1"
        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            diagnostic = _sandbox_exception_diagnostic(exc, code)
        else:  # pragma: no cover
            self.fail("Expected generated code to raise SyntaxError")

        self.assertEqual(diagnostic["type"], "SyntaxError")
        self.assertEqual(diagnostic["line"], 1)
        self.assertIsInstance(diagnostic["column"], int)
        self.assertEqual(diagnostic["source"], "if True")
        self.assertIn("syntax", diagnostic["hint"].lower())

    def test_tool_payload_preserves_structured_diagnostic(self) -> None:
        diagnostic = {
            "type": "TypeError",
            "line": 3,
            "column": None,
            "source": "current_frame.find(2)",
            "hint": "Check argument types.",
        }
        payload = _python_tool_payload(
            {
                "error": "TypeError: bad argument",
                "diagnostic": diagnostic,
                "stdout": "before failure\n",
            }
        )

        self.assertEqual(payload["diagnostic"], diagnostic)
        self.assertTrue(payload["retryable"])
        self.assertEqual(payload["stdout"], "before failure\n")

    def test_diagnostic_helper_is_injected_into_bootstrap(self) -> None:
        namespace = {"__name__": "sandbox_bootstrap_test"}
        exec(compile(_SANDBOX_BOOTSTRAP, "<sandbox-bootstrap>", "exec"), namespace)

        self.assertIn("_sandbox_exception_diagnostic", namespace)


if __name__ == "__main__":
    unittest.main()
