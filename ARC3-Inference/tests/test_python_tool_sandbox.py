from __future__ import annotations

from unittest import TestCase

from inference.agent.python_tool_sandbox import run_sandboxed_python


def _run(code: str) -> dict:
    return run_sandboxed_python(
        code=code,
        timeout_seconds=5,
        initial_state={
            "current_frame": None,
            "history": [],
            "valid_actions": [],
            "last_action_result": {},
        },
        action_handler=lambda actions: {"action_result": {}, "state": {}},
    )


class PythonToolSandboxTests(TestCase):
    def test_sandbox_executes_supported_python(self) -> None:
        response = _run("result = sum(range(5))")

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 10)

    def test_sandbox_rejects_object_graph_escape(self) -> None:
        response = _run("result = ().__class__.__base__.__subclasses__()")

        self.assertIn("Private attribute access is not allowed", response["error"])

    def test_sandbox_rejects_private_from_import(self) -> None:
        response = _run("from random import _os as host_os\nresult = host_os.getcwd()")

        self.assertIn("Private imports are not allowed", response["error"])

    def test_sandbox_rejects_unlisted_submodules(self) -> None:
        response = _run("import json.tool\nresult = json.tool.sys.version")

        self.assertIn("Module 'json.tool' is not allowed", response["error"])

    def test_sandbox_hides_module_valued_attributes(self) -> None:
        response = _run("import fractions\nresult = fractions.sys.version")

        self.assertIn("Module-valued attribute 'sys' is not allowed", response["error"])
