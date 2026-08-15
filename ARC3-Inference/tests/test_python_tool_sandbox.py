from __future__ import annotations

from unittest import TestCase

from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.objective_reduction import ObjectiveReducer


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
                "saved = record_strategy(goal='reach target', evidence=['moved'], confidence=0.75)\n"
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

    def test_sandbox_objective_operations_refresh_runtime_state(self) -> None:
        reducer = ObjectiveReducer(enabled=True)

        def state() -> dict:
            return {
                "current_frame": None,
                "history": [],
                "valid_actions": ["LEFT"],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "objectives": reducer.snapshot(),
            }

        response = run_sandboxed_python(
            code=(
                "created = objectives({'op':'initialize','description':'win','success_criterion':'progress'})\n"
                "result = [created['ok'], objective_state['active_objective_id']]"
            ),
            timeout_seconds=5,
            initial_state=state(),
            action_handler=lambda actions: {"action_result": {}, "state": state()},
            objective_handler=reducer.apply,
            state_provider=state,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [True, "obj-1"])
        self.assertEqual(response["objective_updates"][0]["operation"], "initialize")

    def test_sandbox_returns_stable_disabled_objective_error(self) -> None:
        response = _run("result = objectives({'op':'initialize'})['error']['code']")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], "disabled")

    def test_sandbox_can_transactionally_replan_current_branch(self) -> None:
        reducer = ObjectiveReducer(enabled=True)

        def state() -> dict:
            return {
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "objectives": reducer.snapshot(),
            }

        response = run_sandboxed_python(
            code=(
                "root = objectives({'op':'initialize','description':'win','success_criterion':'progress'})\n"
                "objectives({'op':'reduce','objective_id':root['active_objective_id'],'children':[{'description':'old a','success_criterion':'a'},{'description':'old b','success_criterion':'b'}]})\n"
                "update = {'request_id':'sandbox-replan-v1','op':'replan','objective_id':'obj-1','reason':'new evidence','children':[{'description':'new plan','success_criterion':'done'}]}\n"
                "changed = objectives(update)\n"
                "replayed = objectives(update)\n"
                "result = [changed['active_objective_id'], len(objective_state['discarded_branches']), replayed['replayed']]"
            ),
            timeout_seconds=5,
            initial_state=state(),
            action_handler=lambda actions: {"action_result": {}, "state": state()},
            objective_handler=reducer.apply,
            state_provider=state,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], ["obj-4", 1, True])
