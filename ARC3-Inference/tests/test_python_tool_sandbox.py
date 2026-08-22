from __future__ import annotations

from unittest import TestCase

from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _SANDBOX_LAUNCHER,
    _sandbox_command,
    SandboxHostActionError,
    run_sandboxed_python,
)


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
    def test_sandbox_launcher_uses_compact_precompiled_bootstrap(self) -> None:
        command, _isolated_cwd = _sandbox_command()

        self.assertEqual(command[-1], _SANDBOX_LAUNCHER)
        self.assertLess(len(_SANDBOX_LAUNCHER), 256)
        self.assertLess(len(_SANDBOX_LAUNCHER), len(_SANDBOX_BOOTSTRAP) // 100)

    def test_sandbox_executes_supported_python(self) -> None:
        response = _run("result = sum(range(5))")

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 10)

    def test_sandbox_returns_notebook_style_final_expression(self) -> None:
        response = _run("values = [2, 3, 5]\nsum(values)")

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 10)

    def test_sandbox_timeout_returns_structured_diagnostic(self) -> None:
        response = run_sandboxed_python(
            code="while True:\n    pass",
            timeout_seconds=1,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "memory": {},
            },
            action_handler=lambda actions: {"action_result": {}, "state": {}},
        )

        self.assertIn("timed out after 1s", response["error"])
        self.assertEqual(response["diagnostic"]["type"], "TimeoutError")
        self.assertIn("bounded computation", response["diagnostic"]["hint"])

    def test_sandbox_bounds_stdout_before_host_transport(self) -> None:
        response = _run("print('x' * 50000)")

        self.assertEqual(response["error"], "")
        self.assertLess(len(response["stdout"]), 33_000)
        self.assertIn("truncated", response["stdout"])

    def test_sandbox_bounds_large_and_cyclic_results(self) -> None:
        large = _run("list(range(10000))")
        cyclic = _run("value = []\nvalue.append(value)\nvalue")

        self.assertEqual(large["error"], "")
        self.assertLessEqual(len(large["result"]), 513)
        self.assertIn("item limit", large["result"][-1])
        self.assertEqual(cyclic["error"], "")
        self.assertEqual(cyclic["result"], ["... [cyclic reference]"])

    def test_sandbox_rejects_object_graph_escape(self) -> None:
        response = _run("result = ().__class__.__base__.__subclasses__()")

        self.assertIn("Private attribute access is not allowed", response["error"])

    def test_sandbox_rejects_dunder_name_access(self) -> None:
        response = _run("result = __builtins__")

        self.assertIn("Dunder names are not allowed", response["error"])

    def test_sandbox_allows_underscore_loop_variables_and_helpers(self) -> None:
        response = _run(
            "total = 0\n"
            "for _ in range(3):\n"
            "    total += _\n"
            "def _double(value):\n"
            "    return value * 2\n"
            "result = [total, _double(21)]"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [3, 42])

    def test_sandbox_allows_operator_and_string_modules(self) -> None:
        response = _run(
            "import operator\n"
            "import string\n"
            "result = [operator.add(1, 2), string.ascii_uppercase[:3]]"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [3, "ABC"])

    def test_sandbox_allows_common_exception_handling(self) -> None:
        response = _run(
            "caught = []\n"
            "try:\n"
            "    {}['missing']\n"
            "except KeyError:\n"
            "    caught.append('key')\n"
            "try:\n"
            "    next(iter([]))\n"
            "except StopIteration:\n"
            "    caught.append('stop')\n"
            "try:\n"
            "    [][5]\n"
            "except IndexError:\n"
            "    caught.append('index')\n"
            "try:\n"
            "    int('x')\n"
            "except ValueError:\n"
            "    caught.append('value')\n"
            "result = caught"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], ["key", "stop", "index", "value"])

    def test_sandbox_restored_introspection_builtins(self) -> None:
        response = _run(
            "values = {'alpha': 1}\n"
            "result = [\n"
            "    isinstance(type(values), type),\n"
            "    'get' in dir(values),\n"
            "    hasattr(values, 'get'),\n"
            "    issubclass(bool, int),\n"
            "]"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [True, True, True, True])

    def test_sandbox_getattr_blocks_private_and_dunder_names(self) -> None:
        response = _run(
            "messages = []\n"
            "for target in ((current_frame, '_grid'), ('x', '__class__')):\n"
            "    try:\n"
            "        getattr(target[0], target[1])\n"
            "    except AttributeError as exc:\n"
            "        messages.append(str(exc))\n"
            "result = len(messages)"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 2)

    def test_sandbox_getattr_supports_public_names_and_default(self) -> None:
        response = _run(
            "payload = {'alpha': 1}\n"
            "result = [\n"
            "    getattr('abcd', 'upper')(),\n"
            "    getattr(payload, 'missing', 'fallback'),\n"
            "]"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], ["ABCD", "fallback"])

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

    def test_sandbox_record_strategy_forwards_prediction_fields(self) -> None:
        updates: list[dict] = []

        def persist(update: dict) -> dict:
            updates.append(update)
            return {"goal": None}

        response = run_sandboxed_python(
            code=(
                "record_strategy(\n"
                "    test_action='LEFT',\n"
                "    expected_outcome='new_state',\n"
                "    fallback='probe up instead',\n"
                "    contradictions=['old reading was stale'],\n"
                ")\n"
                "result = 'ok'"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
            },
            action_handler=lambda actions: {"action_result": {}, "state": {}},
            strategy_handler=persist,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["test_action"], "LEFT")
        self.assertEqual(updates[0]["expected_outcome"], "new_state")
        self.assertEqual(updates[0]["fallback"], "probe up instead")
        self.assertEqual(updates[0]["contradictions"], ["old reading was stale"])

    def test_sandbox_surfaces_host_action_validation_errors(self) -> None:
        def handler(actions: list) -> dict:
            if len(actions) > 2:
                raise SandboxHostActionError(
                    "action(actions) accepts at most 12 actions per batch."
                )
            return {"action_result": {"executed": True}, "state": {}}

        response = run_sandboxed_python(
            code=(
                "first = action(['UP'])\n"
                "message = ''\n"
                "try:\n"
                "    action(['A', 'B', 'C'])\n"
                "except RuntimeError as exc:\n"
                "    message = str(exc)\n"
                "result = [first.get('executed'), 'at most 12' in message]"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
            },
            action_handler=handler,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [True, True])
