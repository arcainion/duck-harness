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
            "experience": {"phase": "orient"},
            "strategy": {},
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

    def test_sandbox_exposes_experience_and_persists_strategy_updates(self) -> None:
        updates: list[dict] = []

        def persist(update: dict) -> dict:
            updates.append(update)
            return {"goal": str(update.get("goal")), "confidence": 0.75}

        response = run_sandboxed_python(
            code=(
                "saved = record_strategy(goal='reach target', evidence=['moved'], confidence=0.75, "
                "test_action='RIGHT', expected_outcome='new_state', "
                "fallback='try LEFT', contradictions=['timer changed'])\n"
                "result = [experience['phase'], saved['goal'], strategy['confidence']]"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {"phase": "orient"},
                "strategy": {},
            },
            action_handler=lambda actions: {"action_result": {}, "state": {}},
            strategy_handler=persist,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], ["orient", "reach target", 0.75])
        self.assertEqual(updates[0]["evidence"], ["moved"])
        self.assertEqual(updates[0]["test_action"], "RIGHT")
        self.assertEqual(updates[0]["expected_outcome"], "new_state")
        self.assertEqual(updates[0]["fallback"], "try LEFT")
        self.assertEqual(updates[0]["contradictions"], ["timer changed"])
