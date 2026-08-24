from __future__ import annotations

import queue
from unittest import TestCase
from unittest import mock

from inference.agent import python_tool_sandbox as sandbox_module
from inference.agent.python_tool_sandbox import (
    _SANDBOX_BOOTSTRAP,
    _SANDBOX_LAUNCHER,
    _runtime_state_update,
    _sandbox_command,
    SandboxHostActionError,
    run_sandboxed_python,
    validate_sandbox_isolation,
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


def _run_with_frame(code: str) -> dict:
    return run_sandboxed_python(
        code=code,
        timeout_seconds=5,
        initial_state={
            "current_frame": {
                "ascii": "WW\nWG",
                "grid": [[0, 0], [0, 1]],
                "step": 0,
                "level": 1,
                "shape": [2, 2],
            },
            "history": [],
            "valid_actions": [],
            "last_action_result": {},
            "experience": {},
            "strategy": {},
            "memory": {},
        },
        action_handler=lambda actions: {"action_result": {}, "state": {}},
    )


class PythonToolSandboxTests(TestCase):
    def test_required_os_isolation_fails_closed_without_bubblewrap(self) -> None:
        with (
            mock.patch.object(sandbox_module, "_SANDBOX_REQUIRE_OS_ISOLATION", True),
            mock.patch.object(sandbox_module.shutil, "which", return_value=None),
            self.assertRaisesRegex(OSError, "bubblewrap is unavailable"),
        ):
            validate_sandbox_isolation()

    def test_runtime_state_update_appends_history_and_omits_unchanged_fields(self) -> None:
        previous = {"history": [{"frame": "one"}], "valid_actions": ["LEFT"], "memory": {}}
        current = {
            "history": [{"frame": "one"}, {"frame_delta": "two"}],
            "valid_actions": ["RIGHT"],
            "memory": {},
        }

        update = _runtime_state_update(previous, current)

        self.assertEqual(update["history_append"], [{"frame_delta": "two"}])
        self.assertEqual(update["valid_actions"], ["RIGHT"])
        self.assertNotIn("history", update)
        self.assertNotIn("memory", update)

    def test_runtime_state_update_covers_identical_replaced_and_shrunk_history(self) -> None:
        unchanged = {"history": [{"frame": "one"}], "valid_actions": ["LEFT"]}
        replaced = {"history": [{"frame": "different"}], "valid_actions": ["LEFT"]}
        shrunk = {"history": [], "valid_actions": ["LEFT"]}

        self.assertEqual(_runtime_state_update(unchanged, dict(unchanged)), {})
        self.assertEqual(
            _runtime_state_update(unchanged, replaced),
            {"history": [{"frame": "different"}]},
        )
        self.assertEqual(_runtime_state_update(unchanged, shrunk), {"history": []})

    def test_runtime_state_update_replaces_nonlist_history_and_tracks_none(self) -> None:
        previous = {"history": None, "experience": {"phase": "orient"}}
        current = {"history": [{"frame": "one"}], "experience": None}

        self.assertEqual(
            _runtime_state_update(previous, current),
            {"history": [{"frame": "one"}], "experience": None},
        )

    def test_take_prepared_sandbox_discards_dead_worker_and_refills(self) -> None:
        worker = mock.Mock()
        worker.process.poll.return_value = 7
        workers: queue.Queue = queue.Queue()
        workers.put(worker)

        with (
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_QUEUE", workers),
            mock.patch.object(sandbox_module, "prewarm_sandbox") as refill,
        ):
            selected = sandbox_module._take_prepared_sandbox()

        self.assertIsNone(selected)
        worker.close.assert_called_once_with()
        refill.assert_called_once_with()

    def test_take_prepared_sandbox_consumes_live_worker_once_and_refills(self) -> None:
        worker = mock.Mock()
        worker.process.poll.return_value = None
        workers: queue.Queue = queue.Queue()
        workers.put(worker)

        with (
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_QUEUE", workers),
            mock.patch.object(sandbox_module, "prewarm_sandbox") as refill,
        ):
            selected = sandbox_module._take_prepared_sandbox()
            second = sandbox_module._take_prepared_sandbox()

        self.assertIs(selected, worker)
        self.assertIsNone(second)
        refill.assert_called_once_with()

    def test_prewarm_disabled_does_not_start_background_threads(self) -> None:
        with (
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_WORKERS", 0),
            mock.patch.object(sandbox_module.threading, "Thread") as thread,
        ):
            sandbox_module.prewarm_sandbox()

        thread.assert_not_called()

    def test_prewarm_does_not_oversubscribe_full_or_inflight_capacity(self) -> None:
        workers: queue.Queue = queue.Queue(maxsize=1)
        workers.put(mock.Mock())
        with (
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_WORKERS", 1),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_QUEUE", workers),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_STARTING", 0),
            mock.patch.object(sandbox_module.threading, "Thread") as thread,
        ):
            sandbox_module.prewarm_sandbox()
        thread.assert_not_called()

        empty_workers: queue.Queue = queue.Queue(maxsize=1)
        with (
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_WORKERS", 1),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_QUEUE", empty_workers),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_STARTING", 1),
            mock.patch.object(sandbox_module.threading, "Thread") as thread,
        ):
            sandbox_module.prewarm_sandbox()
        thread.assert_not_called()

    def test_close_prewarmed_sandboxes_terminates_every_queued_worker(self) -> None:
        first = mock.Mock()
        second = mock.Mock()
        workers: queue.Queue = queue.Queue()
        workers.put(first)
        workers.put(second)

        with (
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_QUEUE", workers),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_STARTING", 0),
        ):
            sandbox_module._close_prewarmed_sandboxes()

        first.close.assert_called_once_with(terminate=True)
        second.close.assert_called_once_with(terminate=True)

    def test_sandbox_launcher_uses_compact_precompiled_bootstrap(self) -> None:
        command, _isolated_cwd = _sandbox_command()

        self.assertEqual(command[-1], _SANDBOX_LAUNCHER)
        self.assertLess(len(_SANDBOX_LAUNCHER), 256)
        self.assertLess(len(_SANDBOX_LAUNCHER), len(_SANDBOX_BOOTSTRAP) // 100)

    def test_strict_os_isolation_fails_closed_without_bubblewrap(self) -> None:
        with (
            mock.patch.object(sandbox_module, "_SANDBOX_REQUIRE_OS_ISOLATION", True),
            mock.patch.object(sandbox_module.shutil, "which", return_value=None),
        ):
            with self.assertRaisesRegex(OSError, "bubblewrap is unavailable"):
                sandbox_module._sandbox_command()

    def test_sandbox_executes_supported_python(self) -> None:
        response = _run("result = sum(range(5))")

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 10)
        self.assertGreater(response["efficiency"]["host_to_sandbox_bytes"], 0)
        self.assertGreater(response["efficiency"]["elapsed_seconds"], 0)
        self.assertIsInstance(response["efficiency"]["prewarmed"], bool)

    def test_frame_analyze_batches_bounded_queries(self) -> None:
        response = _run_with_frame(
            "result = current_frame.analyze(["
            "{'method': 'color_summary'}, "
            "{'method': 'bounds', 'args': ['W']}])"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual([item["method"] for item in response["result"]], ["color_summary", "bounds"])
        self.assertEqual(response["result"][1]["result"]["count"], 3)

    def test_frame_analyze_accepts_empty_calls_and_keyword_arguments(self) -> None:
        response = _run_with_frame(
            "empty = current_frame.analyze([])\n"
            "queries = current_frame.analyze(["
            "{'name': 'color_summary', 'kwargs': {'limit': 1}}])\n"
            "result = [empty, queries[0]['result']['colors']]"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"][0], [])
        self.assertEqual(len(response["result"][1]), 1)

    def test_frame_analyze_rejects_invalid_batches_methods_and_arguments(self) -> None:
        response = _run_with_frame(
            "errors = []\n"
            "def attempt(calls):\n"
            "    try:\n"
            "        current_frame.analyze(calls)\n"
            "    except Exception as exc:\n"
            "        errors.append(str(exc))\n"
            "attempt('bad')\n"
            "attempt([{'method': 'color_summary'}] * 33)\n"
            "attempt([{'method': '_grid'}])\n"
            "attempt([{'method': 'missing'}])\n"
            "attempt([{'method': 'color_summary', 'args': 'bad'}])\n"
            "result = errors"
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(len(response["result"]), 5)
        self.assertIn("expects a list", response["result"][0])
        self.assertIn("limited to 32", response["result"][1])
        self.assertIn("invalid method", response["result"][2])
        self.assertIn("Unknown frame analysis method", response["result"][3])
        self.assertIn("args/kwargs are invalid", response["result"][4])

    def test_frame_analyze_results_cannot_mutate_cached_results(self) -> None:
        response = _run_with_frame(
            "batch = current_frame.analyze([{'method': 'color_summary'}])\n"
            "batch[0]['result']['colors'].clear()\n"
            "result = current_frame.color_summary()['colors']"
        )

        self.assertEqual(response["error"], "")
        self.assertGreater(len(response["result"]), 0)

    def test_sandbox_workers_do_not_reuse_generated_globals(self) -> None:
        first = _run("generated_marker = 123\nresult = generated_marker")
        second = _run("result = generated_marker")

        self.assertEqual(first["result"], 123)
        self.assertIn("NameError", second["error"])

    def test_action_state_updates_append_then_replace_history(self) -> None:
        white = {"ascii": "W", "grid": [[0]], "step": 0, "level": 1, "shape": [1, 1]}
        green = {"ascii": "G", "grid": [[3]], "step": 1, "level": 1, "shape": [1, 1]}
        initial_entry = {"action": "INIT", "frame": white}
        states = iter(
            [
                {
                    "current_frame": {"history_index": 1},
                    "history": [initial_entry, {"action": "LEFT", "frame": green}],
                    "valid_actions": ["RIGHT"],
                    "last_action_result": {"executed": True},
                    "experience": {"phase": "progress"},
                    "strategy": {},
                    "memory": {"keep": 7},
                },
                {
                    "current_frame": {"history_index": 0},
                    "history": [{"action": "RESET", "frame": white}],
                    "valid_actions": ["UP"],
                    "last_action_result": {"executed": True},
                    "experience": {"phase": "recover"},
                    "strategy": {},
                    "memory": {"keep": 7},
                },
            ]
        )

        response = run_sandboxed_python(
            code=(
                "action('LEFT')\n"
                "first = [len(history), current_frame.cell(0, 0), valid_actions[0], memory['keep']]\n"
                "action('RIGHT')\n"
                "result = [first, [len(history), current_frame.cell(0, 0), valid_actions[0], memory['keep']]]"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": {"history_index": 0},
                "history": [initial_entry],
                "valid_actions": ["LEFT"],
                "last_action_result": {},
                "experience": {"phase": "orient"},
                "strategy": {},
                "memory": {"keep": 7},
            },
            action_handler=lambda actions: {
                "action_result": {"executed": True, "actions": actions},
                "state": next(states),
            },
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [[2, "G", "RIGHT", 7], [1, "W", "UP", 7]])

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

    def test_sandbox_hides_unexpected_host_action_errors(self) -> None:
        def fail_action(_actions: list[dict]) -> dict:
            raise RuntimeError("sensitive host detail")

        response = run_sandboxed_python(
            code="action('LEFT')",
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": ["LEFT"],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "memory": {},
            },
            action_handler=fail_action,
        )

        self.assertIn("action failed in sandbox host", response["error"])
        self.assertNotIn("sensitive host detail", response["error"])

    def test_sandbox_strategy_handler_failure_falls_back_to_empty_strategy(self) -> None:
        def fail_strategy(_update: dict) -> dict:
            raise RuntimeError("persistence unavailable")

        response = run_sandboxed_python(
            code="saved = record_strategy(goal='test fallback')\nresult = [saved, strategy]",
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {"old": True},
                "memory": {},
            },
            action_handler=lambda _actions: {"action_result": {}, "state": {}},
            strategy_handler=fail_strategy,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [{}, {}])
        self.assertEqual(response["strategy_updates"], [{}])

    def test_sandbox_memory_handler_persists_remember_and_forget(self) -> None:
        updates: list[dict] = []

        def persist_memory(memory: dict) -> dict:
            updates.append(dict(memory))
            return dict(memory)

        response = run_sandboxed_python(
            code=(
                "first = remember(' clue ', {'color': 'R'})\n"
                "second = forget('clue')\n"
                "result = [first, second, memory]"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "memory": {},
            },
            action_handler=lambda _actions: {"action_result": {}, "state": {}},
            memory_handler=persist_memory,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [{"clue": {"color": "R"}}, {}, {}])
        self.assertEqual(updates, [{"clue": {"color": "R"}}, {}])

    def test_sandbox_memory_validation_error_is_returned_to_generated_code(self) -> None:
        def reject_memory(_memory: dict) -> dict:
            raise ValueError("memory quota exceeded")

        response = run_sandboxed_python(
            code=(
                "try:\n"
                "    remember('large', 1)\n"
                "except ValueError as exc:\n"
                "    result = str(exc)"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "memory": {},
            },
            action_handler=lambda _actions: {"action_result": {}, "state": {}},
            memory_handler=reject_memory,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], "memory quota exceeded")

    def test_sandbox_hides_unexpected_memory_handler_errors(self) -> None:
        def fail_memory(_memory: dict) -> dict:
            raise RuntimeError("sensitive persistence detail")

        response = run_sandboxed_python(
            code=(
                "try:\n"
                "    remember('key', 'value')\n"
                "except ValueError as exc:\n"
                "    result = str(exc)"
            ),
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "memory": {},
            },
            action_handler=lambda _actions: {"action_result": {}, "state": {}},
            memory_handler=fail_memory,
        )

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], "memory update failed.")
        self.assertNotIn("sensitive persistence detail", str(response))

    def test_sandbox_process_start_failure_returns_structured_fallback(self) -> None:
        with (
            mock.patch.object(sandbox_module, "_take_prepared_sandbox", return_value=None),
            mock.patch.object(sandbox_module.subprocess, "Popen", side_effect=OSError),
        ):
            response = _run("result = 1")

        self.assertEqual(response["error"], "Sandbox process could not start.")
        self.assertEqual(response["stdout"], "")
        self.assertEqual(response["action_results"], [])

    def test_sandbox_rejects_invalid_and_unknown_worker_messages(self) -> None:
        for line, expected in (
            ("not-json\n", "invalid response"),
            ('{"type":"mystery"}\n', "unknown message type"),
        ):
            process = mock.Mock()
            process.stderr.read.return_value = ""
            worker = mock.Mock()
            worker.process = process
            worker.isolated_cwd = "sandbox"
            worker.temp_dir = "sandbox"
            worker.stdout_queue = queue.Queue()
            worker.stdout_queue.put(line)

            with (
                self.subTest(line=line),
                mock.patch.object(sandbox_module, "_take_prepared_sandbox", return_value=worker),
                mock.patch.object(sandbox_module, "_wait_for_process_exit"),
                mock.patch.object(sandbox_module, "_kill_process_group"),
                mock.patch.object(sandbox_module.shutil, "rmtree"),
            ):
                response = _run("result = 1")

            self.assertIn(expected, response["error"])

    def test_prewarm_startup_failures_restore_capacity_and_close_workers(self) -> None:
        class ImmediateThread:
            def __init__(self, *, target, daemon: bool) -> None:
                self.target = target
                self.daemon = daemon

            def start(self) -> None:
                self.target()

        workers: queue.Queue = queue.Queue(maxsize=1)
        with (
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_WORKERS", 1),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_QUEUE", workers),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_STARTING", 0),
            mock.patch.object(sandbox_module, "_PreparedSandboxProcess", side_effect=OSError),
            mock.patch.object(sandbox_module.threading, "Thread", ImmediateThread),
        ):
            sandbox_module.prewarm_sandbox()
            self.assertEqual(sandbox_module._SANDBOX_PREWARM_STARTING, 0)
            self.assertTrue(workers.empty())

        worker = mock.Mock()
        full_queue = mock.Mock()
        full_queue.qsize.return_value = 0
        full_queue.put_nowait.side_effect = queue.Full
        with (
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_WORKERS", 1),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_QUEUE", full_queue),
            mock.patch.object(sandbox_module, "_SANDBOX_PREWARM_STARTING", 0),
            mock.patch.object(sandbox_module, "_PreparedSandboxProcess", return_value=worker),
            mock.patch.object(sandbox_module.threading, "Thread", ImmediateThread),
        ):
            sandbox_module.prewarm_sandbox()
            self.assertEqual(sandbox_module._SANDBOX_PREWARM_STARTING, 0)

        worker.close.assert_called_once_with(terminate=True)

    def test_sandbox_cancellation_interrupts_running_generated_code(self) -> None:
        response = run_sandboxed_python(
            code="while True:\n    pass",
            timeout_seconds=5,
            initial_state={
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {},
                "strategy": {},
                "memory": {},
            },
            action_handler=lambda _actions: {"action_result": {}, "state": {}},
            should_stop=lambda: True,
        )

        self.assertEqual(response["error"], "Tool execution cancelled by host.")
        self.assertEqual(response["diagnostic"]["type"], "CancelledError")
        self.assertEqual(response["diagnostic"]["retry"], "do_not_retry")
        self.assertLess(response["efficiency"]["elapsed_seconds"], 2)
