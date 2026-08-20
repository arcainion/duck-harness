from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

import inference.agent.python_tool_sandbox as sandbox_module
from inference.agent.python_tool_sandbox import run_sandboxed_python


def _make_frame_payload(grid, *, step: int = 0, level: int = 1, ascii: str = ""):
    return {
        "ascii": ascii or "\n".join("".join(str(v) for v in row) for row in grid),
        "step": step,
        "level": level,
        "shape": [len(grid), len(grid[0]) if grid else 0],
        "grid": [list(row) for row in grid],
    }


def _run(code: str, **kwargs) -> dict:
    state = kwargs.pop("initial_state", None) or {
        "current_frame": None,
        "history": [],
        "valid_actions": [],
        "last_action_result": {},
        "experience": {"phase": "orient"},
        "strategy": {},
    }
    return run_sandboxed_python(
        code=code,
        timeout_seconds=kwargs.pop("timeout_seconds", 5),
        initial_state=state,
        action_handler=kwargs.pop("action_handler", lambda actions: {"action_result": {}, "state": {}}),
        **kwargs,
    )


def _run_with_frame(code: str, grid, **kwargs) -> dict:
    frame = _make_frame_payload(grid)
    state = {
        "current_frame": frame,
        "history": [],
        "valid_actions": [],
        "last_action_result": {},
        "experience": {"phase": "orient"},
        "strategy": {},
    }
    return _run(code, initial_state=state, **kwargs)


class PythonToolSandboxTests(TestCase):
    def test_sandbox_executes_supported_python(self) -> None:
        response = _run("result = sum(range(5))")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 10)

    def test_runtime_error_reports_innermost_user_source_line(self) -> None:
        response = _run("def fail():\n    missing_name()\nfail()")
        self.assertEqual(response["error_category"], "UNDEFINED_VARIABLE")
        self.assertEqual(response["error_line"], 2)

    def test_sandbox_rejects_object_graph_escape(self) -> None:
        response = _run("result = ().__class__.__base__.__subclasses__()")
        self.assertIn("Private attribute access is not allowed", response["error"])

    def test_sandbox_rejects_runtime_binding_shadowing(self) -> None:
        cases = (
            "current_frame = None",
            "def action():\n    pass",
            "def helper(history):\n    pass",
            "import math as bfs",
        )
        for code in cases:
            with self.subTest(code=code):
                response = _run(code)
                self.assertEqual(
                    response["error_category"],
                    "PROTECTED_RUNTIME_BINDING",
                )
                self.assertIn("cannot be overwritten", response["error"])

    def test_sandbox_rejects_operator_attrgetter_dunder_escape(self) -> None:
        response = _run('import operator\nresult = operator.attrgetter("__class__")(1)')
        self.assertEqual(response["error_category"], "PRIVATE_ACCESS")
        self.assertIn("operator.attrgetter", response["error"])

    def test_sandbox_rejects_operator_methodcaller_dunder_escape(self) -> None:
        response = _run('import operator\nresult = operator.methodcaller("__reduce__")(1)')
        self.assertEqual(response["error_category"], "PRIVATE_ACCESS")
        self.assertIn("operator.methodcaller", response["error"])

    def test_sandbox_rejects_from_import_of_dynamic_attribute_helper(self) -> None:
        response = _run('from operator import attrgetter\nresult = attrgetter("__class__")(1)')
        self.assertEqual(response["error_category"], "PRIVATE_ACCESS")

    def test_safe_operator_functions_remain_available(self) -> None:
        response = _run("import operator\nresult = operator.add(2, 3)")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 5)

    def test_sandbox_rejects_string_formatter_dunder_escape(self) -> None:
        response = _run(
            'import string\nresult = string.Formatter().get_field("0.__class__", [1], {})'
        )
        self.assertEqual(response["error_category"], "PRIVATE_ACCESS")
        self.assertIn("string.Formatter", response["error"])

    def test_sandbox_rejects_str_format_private_resolution(self) -> None:
        response = _run('result = "{0.__class__}".format(1)')
        self.assertEqual(response["error_category"], "PRIVATE_ACCESS")
        self.assertIn("dynamic attribute resolution", response["error"])

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


class SandboxPreInjectedHelpersTests(TestCase):
    def test_color_grid_returns_2d_char_list(self) -> None:
        grid = [[1, 2, 3], [4, 5, 6]]
        response = _run_with_frame("result = color_grid(current_frame)", grid)
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)
        self.assertEqual(len(response["result"]), 2)
        self.assertEqual(len(response["result"][0]), 3)

    def test_diff_frames_detects_changes(self) -> None:
        grid2 = [[1, 2], [1, 1]]
        frame2 = _make_frame_payload(grid2)
        state = {
            "current_frame": frame2,
            "history": [],
            "valid_actions": [],
            "last_action_result": {},
            "experience": {"phase": "orient"},
            "strategy": {},
        }
        code = (
            "import json\n"
            "f1 = current_frame\n"
            "f2 = current_frame\n"
            "result = diff_frames(f1, f2)"
        )
        response = _run(code, initial_state=state)
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], dict)
        self.assertIn("changed", response["result"])

    def test_diff_frames_with_none_frames(self) -> None:
        response = _run("result = diff_frames(None, None)")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"]["changed"], [])

    def test_find_positions_returns_matching_coords(self) -> None:
        grid = [[1, 2, 1], [2, 1, 2]]
        response = _run_with_frame(
            "result = find_positions(current_frame, 'r')",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)

    def test_find_positions_empty_when_no_match(self) -> None:
        grid = [[1, 1], [1, 1]]
        response = _run_with_frame(
            "result = find_positions(current_frame, 'z')",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [])

    def test_neighbors4_returns_adjacent_positions(self) -> None:
        response = _run("result = neighbors4(1, 1, 3, 3)")
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)
        self.assertEqual(len(response["result"]), 4)
        self.assertIn([0, 1], response["result"])
        self.assertIn([2, 1], response["result"])
        self.assertIn([1, 0], response["result"])
        self.assertIn([1, 2], response["result"])

    def test_neighbors4_corner_position(self) -> None:
        response = _run("result = neighbors4(0, 0, 3, 3)")
        self.assertEqual(response["error"], "")
        self.assertEqual(len(response["result"]), 2)

    def test_neighbors8_includes_diagonals(self) -> None:
        response = _run("result = neighbors8(1, 1, 3, 3)")
        self.assertEqual(response["error"], "")
        self.assertEqual(len(response["result"]), 8)

    def test_neighbors8_corner_position(self) -> None:
        response = _run("result = neighbors8(0, 0, 3, 3)")
        self.assertEqual(response["error"], "")
        self.assertEqual(len(response["result"]), 3)

    def test_bfs_finds_path(self) -> None:
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        response = _run_with_frame(
            "result = bfs(current_frame, (0, 0), (2, 2))",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)
        self.assertGreater(len(response["result"]), 0)
        self.assertEqual(response["result"][0], [0, 0])
        self.assertEqual(response["result"][-1], [2, 2])

    def test_bfs_no_path_returns_empty(self) -> None:
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        response = _run_with_frame(
            "result = bfs(current_frame, (0, 0), (5, 5))",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [])

    def test_flood_fill_returns_region(self) -> None:
        grid = [[1, 1, 2], [1, 2, 2], [3, 3, 3]]
        response = _run_with_frame(
            "result = flood(current_frame, (0, 0))",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)
        self.assertIn([0, 0], response["result"])
        self.assertIn([0, 1], response["result"])
        self.assertIn([1, 0], response["result"])

    def test_cell_at_returns_color_char(self) -> None:
        grid = [[1, 2], [3, 4]]
        response = _run_with_frame(
            "result = cell_at(current_frame, 0, 1)",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsNotNone(response["result"])

    def test_cell_at_out_of_bounds_returns_none(self) -> None:
        grid = [[1, 2]]
        response = _run_with_frame(
            "result = cell_at(current_frame, 5, 5)",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsNone(response["result"])

    def test_count_colors_returns_dict(self) -> None:
        grid = [[1, 1, 2], [2, 2, 2]]
        response = _run_with_frame(
            "result = count_colors(current_frame)",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], dict)

    def test_object_positions_returns_list(self) -> None:
        grid = [[1, 1, 2], [1, 1, 2]]
        response = _run_with_frame(
            "result = object_positions(current_frame, 'r')",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)


class SandboxEdgeCaseTests(TestCase):
    def test_empty_code_returns_error(self) -> None:
        response = _run("")
        self.assertEqual(response["error"], "")


    def test_syntax_error_returns_error(self) -> None:
        response = _run("def foo(")
        self.assertIn("SyntaxError", response["error"])

    def test_division_by_zero_returns_error(self) -> None:
        response = _run("result = 1 / 0")
        self.assertIn("error", response["error"].lower())

    def test_name_error_for_undefined_variable(self) -> None:
        response = _run("result = undefined_var")
        self.assertIn("error", response["error"].lower())

    def test_recursion_error_returns_category(self) -> None:
        response = _run(
            "def f(): return f()\nf()",
            timeout_seconds=3,
        )
        self.assertIn("error", response["error"].lower())

    def test_timeout_returns_error(self) -> None:
        response = _run(
            "while True:\n    pass",
            timeout_seconds=1,
        )
        self.assertIn("timed out", response["error"].lower())
        self.assertEqual(response["error_category"], "TIMEOUT")

    def test_process_start_failure_is_structured(self) -> None:
        with patch("inference.agent.python_tool_sandbox.subprocess.Popen", side_effect=OSError("boom")):
            response = _run("pass")
        self.assertEqual(response["error_category"], "SANDBOX_START_FAILED")
        self.assertIn("could not start", response["error"])

    def test_initial_payload_failure_is_structured(self) -> None:
        with patch("inference.agent.python_tool_sandbox._send_json_line", side_effect=BrokenPipeError):
            response = _run("pass")
        self.assertEqual(response["error_category"], "SANDBOX_PROTOCOL_ERROR")

    def test_unserializable_initial_payload_is_structured(self) -> None:
        response = _run("pass", initial_state={"bad": object()})
        self.assertEqual(response["error_category"], "SANDBOX_PROTOCOL_ERROR")
        self.assertIn("initial payload", response["error"])

    def test_action_result_protocol_write_failure_is_structured(self) -> None:
        original_send = sandbox_module._send_json_line
        send_count = 0

        def fail_second_send(handle, payload):
            nonlocal send_count
            send_count += 1
            if send_count == 2:
                raise BrokenPipeError
            return original_send(handle, payload)

        def handler(actions):
            return {
                "action_result": {"executed": True},
                "state": {},
            }
        with patch(
            "inference.agent.python_tool_sandbox._send_json_line",
            side_effect=fail_second_send,
        ):
            response = _run("action(['LEFT'])", action_handler=handler)
        self.assertEqual(response["error_category"], "SANDBOX_PROTOCOL_ERROR")
        self.assertIn("action result", response["error"])
        self.assertEqual(len(response["action_results"]), 1)

    def test_action_handler_is_called(self) -> None:
        calls = []

        def handler(actions):
            calls.append(actions)
            return {
                "action_result": {"executed": True, "board_changed": True},
                "state": {},
            }

        response = _run(
            "r = action(['LEFT'])\nresult = r.get('executed')",
            action_handler=handler,
        )
        self.assertEqual(response["error"], "")
        self.assertTrue(response["result"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], [{"action": "LEFT"}])

    def test_action_batch_limit_is_enforced_before_host_ipc(self) -> None:
        calls = []

        def handler(actions):
            calls.append(actions)
            return {"action_result": {}, "state": {}}

        response = _run(
            "action(['LEFT'] * 13)",
            action_handler=handler,
        )
        self.assertEqual(response["error_category"], "ACTION_BATCH_LIMIT")
        self.assertIn("at most 12", response["error_hint"])
        self.assertEqual(calls, [])

    def test_action_with_dict_format(self) -> None:
        calls = []

        def handler(actions):
            calls.append(actions)
            return {"action_result": {"executed": True}, "state": {}}

        response = _run(
            "r = action([{'action': 'MOUSE', 'row': 3, 'col': 7}])\nresult = r.get('executed')",
            action_handler=handler,
        )
        self.assertEqual(response["error"], "")
        self.assertTrue(response["result"])
        self.assertEqual(calls[0], [{"action": "MOUSE", "row": 3, "col": 7}])

    def test_action_without_list_raises_error(self) -> None:
        response = _run("action('LEFT')")
        self.assertEqual(response["error"], "")

    def test_multiple_actions_in_sequence(self) -> None:
        call_count = [0]

        def handler(actions):
            call_count[0] += 1
            return {"action_result": {"executed": True}, "state": {}}

        _run(
            "r1 = action(['LEFT'])\nr2 = action(['RIGHT'])\nresult = call_count",
            action_handler=handler,
        )
        self.assertEqual(call_count[0], 2)

    def test_result_variable_is_returned(self) -> None:
        response = _run("result = {'key': 'value', 'count': 42}")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], {"key": "value", "count": 42})

    def test_print_goes_to_stdout(self) -> None:
        response = _run("print('hello world')")
        self.assertEqual(response["error"], "")
        self.assertIn("hello world", response["stdout"])

    def test_multiple_prints_concatenated(self) -> None:
        response = _run("print('line1')\nprint('line2')")
        self.assertEqual(response["error"], "")
        self.assertIn("line1", response["stdout"])
        self.assertIn("line2", response["stdout"])

    def test_exact_stdout_limit_is_not_reported_as_truncated(self) -> None:
        response = _run("print('x' * 32767)")
        self.assertEqual(response["error"], "")
        self.assertEqual(len(response["stdout"]), 32768)
        self.assertNotIn("stdout capped", response["stdout"])

    def test_stdout_over_limit_is_capped(self) -> None:
        response = _run("print('x' * 32768)")
        self.assertEqual(response["error"], "")
        self.assertIn("stdout capped at 32KB", response["stdout"])

    def test_builtins_are_available(self) -> None:
        response = _run("result = [len([1,2,3]), max([1,5,3]), min([1,5,3]), sorted([3,1,2])]")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [3, 5, 1, [1, 2, 3]])

    def test_collections_module_available(self) -> None:
        response = _run(
            "from collections import deque, Counter\n"
            "d = deque([1, 2, 3])\n"
            "c = Counter(['a', 'b', 'a'])\n"
            "result = [list(d), dict(c)]"
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [[1, 2, 3], {"a": 2, "b": 1}])

    def test_json_module_available(self) -> None:
        response = _run("import json\nresult = json.loads('{\"a\": 1}')")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], {"a": 1})

    def test_math_module_available(self) -> None:
        response = _run("import math\nresult = math.sqrt(16)")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 4.0)

    def test_re_module_available(self) -> None:
        response = _run("import re\nresult = re.findall(r'\\d+', 'a1b2c3')")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], ["1", "2", "3"])

    def test_os_module_blocked(self) -> None:
        response = _run("import os")
        self.assertIn("error", response["error"].lower())

    def test_sys_module_blocked(self) -> None:
        response = _run("import sys")
        self.assertIn("error", response["error"].lower())

    def test_numpy_blocked(self) -> None:
        response = _run("import numpy")
        self.assertIn("error", response["error"].lower())

    def test_subprocess_blocked(self) -> None:
        response = _run("import subprocess")
        self.assertIn("error", response["error"].lower())

    def test_strategy_persists_across_calls(self) -> None:
        updates = []

        def persist(update):
            updates.append(update)
            return {"goal": "test", "confidence": 0.5}

        response = run_sandboxed_python(
            code="record_strategy(goal='test', confidence=0.5)\nresult = strategy.get('goal')",
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
        self.assertEqual(response["result"], "test")
        self.assertEqual(len(updates), 1)

    def test_experience_is_read_only_dict(self) -> None:
        response = _run(
            "experience['new_key'] = 'injected'\n"
            "result = 'new_key' in experience"
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], True)

    def test_valid_actions_available(self) -> None:
        state = {
            "current_frame": None,
            "history": [],
            "valid_actions": ["LEFT", "RIGHT", "UP"],
            "last_action_result": {},
            "experience": {"phase": "orient"},
            "strategy": {},
        }
        response = _run("result = valid_actions", initial_state=state)
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], ["LEFT", "RIGHT", "UP"])

    def test_last_action_result_available(self) -> None:
        state = {
            "current_frame": None,
            "history": [],
            "valid_actions": [],
            "last_action_result": {"board_changed": True, "reward": 0.5},
            "experience": {"phase": "orient"},
            "strategy": {},
        }
        response = _run("result = last_action_result", initial_state=state)
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"]["board_changed"], True)

    def test_grid_utils_available(self) -> None:
        response = _run("result = diff_frames(None, None)")
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], dict)

    def test_action_returns_action_result(self) -> None:
        def handler(actions):
            return {
                "action_result": {
                    "executed": True,
                    "board_changed": True,
                    "level": 2,
                    "score": 10,
                },
                "state": {},
            }

        response = _run(
            "r = action(['LEFT'])\n"
            "result = [r.get('executed'), r.get('board_changed'), r.get('level')]",
            action_handler=handler,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [True, True, 2])

    def test_action_refreshes_valid_actions(self) -> None:
        def handler(actions):
            return {
                "action_result": {"executed": True},
                "state": {"valid_actions": ["NEW_ACTION"]},
            }

        response = _run(
            "action(['LEFT'])\nresult = valid_actions",
            action_handler=handler,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], ["NEW_ACTION"])

    def test_large_grid_performance(self) -> None:
        grid = [[i % 16 for i in range(64)] for _ in range(64)]
        response = _run_with_frame(
            "result = len(count_colors(current_frame))",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], int)

    def test_frame_view_str_representation(self) -> None:
        grid = [[1, 2], [3, 4]]
        response = _run_with_frame(
            "result = str(current_frame)",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIn("AsciiFrameView", response["result"])

    def test_segmentation_available_on_frame(self) -> None:
        grid = [[1, 1, 2], [1, 2, 2]]
        response = _run_with_frame(
            "seg = current_frame.segmentation\nresult = 'nodes' in seg",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], True)

    def test_empty_grid_handled(self) -> None:
        response = _run_with_frame(
            "result = color_grid(current_frame)",
            [],
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [])

    def test_single_cell_grid(self) -> None:
        response = _run_with_frame(
            "result = count_colors(current_frame)",
            [[5]],
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], dict)

    def test_action_error_is_raised(self) -> None:
        def handler(actions):
            raise ValueError("test error")

        response = _run(
            "action(['LEFT'])",
            action_handler=handler,
        )
        self.assertIn("error", response["error"].lower())

    def test_action_invalid_response_raises_error(self) -> None:
        def handler(actions):
            return "not a dict"

        response = _run(
            "action(['LEFT'])",
            action_handler=handler,
        )
        self.assertEqual(response["error_category"], "ACTION_FAILED")
        self.assertIn("action failed", response["error"].lower())

    def test_action_invalid_nested_payload_is_structured(self) -> None:
        response = _run(
            "action(['LEFT'])",
            action_handler=lambda actions: {"action_result": "invalid", "state": {}},
        )
        self.assertEqual(response["error_category"], "ACTION_FAILED")

    def test_action_falsy_non_dict_nested_payloads_are_rejected(self) -> None:
        cases = (
            {"action_result": [], "state": {}},
            {"action_result": False, "state": {}},
            {"action_result": {}, "state": []},
            {"action_result": {}, "state": ""},
        )
        for handler_result in cases:
            with self.subTest(handler_result=handler_result):
                response = _run(
                    "action(['LEFT'])",
                    action_handler=lambda actions, value=handler_result: value,
                )
                self.assertEqual(response["error_category"], "ACTION_FAILED")

    def test_non_dict_strategy_response_does_not_crash_host(self) -> None:
        response = _run(
            "record_strategy(goal='test')\nresult = strategy",
            strategy_handler=lambda update: "invalid",
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], {})

    def test_bfs_same_start_and_goal(self) -> None:
        grid = [[0, 0], [0, 0]]
        response = _run_with_frame(
            "result = bfs(current_frame, (0, 0), (0, 0))",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [[0, 0]])

    def test_flood_fill_single_cell(self) -> None:
        grid = [[1, 2], [3, 4]]
        response = _run_with_frame(
            "result = flood(current_frame, (0, 0))",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [[0, 0]])

    def test_find_positions_all_matching(self) -> None:
        grid = [[1, 1], [1, 1]]
        response = _run_with_frame(
            "result = find_positions(current_frame, 'W')",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)

    def test_unicode_in_code_and_output(self) -> None:
        response = _run("result = '日本語: 🎮'")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], "日本語: 🎮")

    def test_unpaired_surrogate_result_does_not_break_protocol(self) -> None:
        response = _run("result = chr(55296)")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], "\ud800")

    def test_unicode_in_frame_grid(self) -> None:
        grid = [[1, 2], [3, 4]]
        response = _run_with_frame(
            "result = str(current_frame)",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIn("AsciiFrameView", response["result"])

    def test_large_result_object(self) -> None:
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}
        response = _run(f"result = {large_dict}")
        self.assertEqual(response["error"], "")
        self.assertEqual(len(response["result"]), 1000)

    def test_result_is_bounded_before_crossing_sandbox_ipc(self) -> None:
        response = _run("result = ['x' * 1000 for item in range(1000)]")
        self.assertEqual(response["error"], "")
        self.assertTrue(response["result_truncated"])
        self.assertLess(len(str(response["result"])), 50_000)

    def test_cyclic_result_is_bounded_instead_of_failing(self) -> None:
        response = _run("result = []\nresult.append(result)")
        self.assertEqual(response["error"], "")
        self.assertTrue(response["result_truncated"])
        self.assertEqual(response["result"], ["... [cycle]"])

    def test_non_finite_result_numbers_are_json_safe(self) -> None:
        response = _run("import math\nresult = [math.nan, math.inf, -math.inf]")
        self.assertEqual(response["error"], "")
        self.assertTrue(response["result_truncated"])
        self.assertEqual(response["result"], ["nan", "inf", "-inf"])

    def test_deeply_nested_result(self) -> None:
        nested = {"a": {"b": {"c": {"d": [1, 2, 3]}}}}
        response = _run(f"result = {nested}")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"]["a"]["b"]["c"]["d"], [1, 2, 3])

    def test_action_handler_exception_propagates(self) -> None:
        def handler(actions):
            raise ValueError("handler error")

        response = _run(
            "action(['LEFT'])",
            action_handler=handler,
        )
        self.assertIn("error", response["error"].lower())

    def test_complex_frame_with_segmentation(self) -> None:
        grid = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
        response = _run_with_frame(
            "seg = current_frame.segmentation\n"
            "result = 'nodes' in seg",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertTrue(response["result"])

    def test_bfs_with_custom_blocked_values(self) -> None:
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        response = _run_with_frame(
            "result = bfs(current_frame, (0, 0), (2, 2), blocked=[1])",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)

    def test_flood_fill_with_diagonal_false(self) -> None:
        grid = [[1, 1, 2], [1, 2, 2], [3, 3, 3]]
        response = _run_with_frame(
            "result = flood(current_frame, (0, 0))",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIn([0, 0], response["result"])
        self.assertIn([0, 1], response["result"])
        self.assertIn([1, 0], response["result"])

    def test_cell_at_negative_coordinates(self) -> None:
        grid = [[1, 2], [3, 4]]
        response = _run_with_frame(
            "result = cell_at(current_frame, -1, -1)",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsNone(response["result"])

    def test_count_colors_with_all_same(self) -> None:
        grid = [[5, 5, 5], [5, 5, 5]]
        response = _run_with_frame(
            "result = count_colors(current_frame)",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], dict)
        self.assertEqual(sum(response["result"].values()), 6)

    def test_object_positions_with_multiple_objects(self) -> None:
        grid = [[1, 2, 1, 2], [3, 4, 3, 4]]
        response = _run_with_frame(
            "result = object_positions(current_frame, 'W')",
            grid,
        )
        self.assertEqual(response["error"], "")
        self.assertIsInstance(response["result"], list)

    def test_diff_frames_identical(self) -> None:
        grid = [[1, 2], [3, 4]]
        frame = _make_frame_payload(grid)
        state = {
            "current_frame": frame,
            "history": [],
            "valid_actions": [],
            "last_action_result": {},
            "experience": {"phase": "orient"},
            "strategy": {},
        }
        response = _run(
            "import json\n"
            "f1 = current_frame\n"
            "f2 = current_frame\n"
            "result = diff_frames(f1, f2)",
            initial_state=state,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"]["changed"], [])

    def test_neighbors4_at_edges(self) -> None:
        response = _run("result = neighbors4(0, 0, 1, 1)")
        self.assertEqual(response["error"], "")
        self.assertEqual(len(response["result"]), 0)

    def test_neighbors8_at_edges(self) -> None:
        response = _run("result = neighbors8(0, 0, 1, 1)")
        self.assertEqual(response["error"], "")
        self.assertEqual(len(response["result"]), 0)

    def test_strategy_handler_exception_propagates(self) -> None:
        def persist(update):
            raise RuntimeError("persist failed")

        response = run_sandboxed_python(
            code="record_strategy(goal='test', confidence=0.5)\nresult = strategy.get('goal', '')",
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
        self.assertEqual(response["result"], "")

    def test_action_with_complex_return_state(self) -> None:
        def handler(actions):
            return {
                "action_result": {"executed": True, "board_changed": True},
                "state": {
                    "valid_actions": ["LEFT", "RIGHT"],
                    "current_frame": _make_frame_payload([[0, 1], [2, 3]]),
                    "history": [],
                    "last_action_result": {"reward": 1.0},
                    "experience": {"phase": "progress"},
                    "strategy": {"goal": "test"},
                },
            }

        response = _run(
            "action(['LEFT'])\n"
            "result = [valid_actions, last_action_result.get('reward'), strategy.get('goal')]",
            action_handler=handler,
        )
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [["LEFT", "RIGHT"], 1.0, "test"])

    def test_multiple_strategy_updates(self) -> None:
        updates = []

        def persist(update):
            updates.append(update)
            return {"goal": str(update.get("goal")), "confidence": 0.5}

        response = run_sandboxed_python(
            code=(
                "record_strategy(goal='first')\n"
                "record_strategy(goal='second', evidence=['e1'])\n"
                "record_strategy(goal='third', contradictions=['c1'])\n"
                "result = len(strategy.get('evidence', []))"
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
        self.assertEqual(len(updates), 3)
        self.assertEqual(updates[0]["goal"], "first")
        self.assertEqual(updates[1]["evidence"], ["e1"])
        self.assertEqual(updates[2]["contradictions"], ["c1"])

    def test_history_available_in_sandbox(self) -> None:
        frame1 = _make_frame_payload([[1, 1]], step=0)
        frame2 = _make_frame_payload([[2, 2]], step=1)
        state = {
            "current_frame": frame2,
            "history": [
                {"action": "LEFT", "frame": frame1},
                {"action": "RIGHT", "frame": frame2},
            ],
            "valid_actions": [],
            "last_action_result": {},
            "experience": {"phase": "orient"},
            "strategy": {},
        }
        response = _run("result = len(history)", initial_state=state)
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], 2)

    def test_valid_actions_empty_list(self) -> None:
        state = {
            "current_frame": None,
            "history": [],
            "valid_actions": [],
            "last_action_result": {},
            "experience": {"phase": "orient"},
            "strategy": {},
        }
        response = _run("result = valid_actions", initial_state=state)
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], [])

    def test_last_action_result_empty(self) -> None:
        state = {
            "current_frame": None,
            "history": [],
            "valid_actions": [],
            "last_action_result": {},
            "experience": {"phase": "orient"},
            "strategy": {},
        }
        response = _run("result = last_action_result", initial_state=state)
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"], {})

    def test_experience_phase_values(self) -> None:
        for phase in ["orient", "explore", "progress", "recover"]:
            state = {
                "current_frame": None,
                "history": [],
                "valid_actions": [],
                "last_action_result": {},
                "experience": {"phase": phase},
                "strategy": {},
            }
            response = _run("result = experience['phase']", initial_state=state)
            self.assertEqual(response["error"], "")
            self.assertEqual(response["result"], phase)

    def test_result_can_be_none(self) -> None:
        response = _run("result = None")
        self.assertEqual(response["error"], "")
        self.assertIsNone(response["result"])

    def test_result_can_be_boolean(self) -> None:
        response = _run("result = True")
        self.assertEqual(response["error"], "")
        self.assertTrue(response["result"])

    def test_result_can_be_float(self) -> None:
        response = _run("result = 3.14159")
        self.assertEqual(response["error"], "")
        self.assertAlmostEqual(response["result"], 3.14159)

    def test_print_output_captured(self) -> None:
        response = _run("print('captured output')")
        self.assertEqual(response["error"], "")
        self.assertIn("captured output", response["stdout"])

    def test_result_with_complex_data(self) -> None:
        response = _run("result = {'grid': [[1, 0], [0, 1]], 'count': 2}")
        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"]["count"], 2)
        self.assertEqual(response["result"]["grid"], [[1, 0], [0, 1]])
