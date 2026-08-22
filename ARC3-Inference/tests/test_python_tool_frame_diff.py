from __future__ import annotations

from unittest import TestCase

from inference.agent.python_tool_sandbox import (
    _bounded_frame_diff_summary,
    run_sandboxed_python,
)


def _frame(grid: list[list[int]], *, step: int, level: int = 1) -> dict:
    return {
        "ascii": "",
        "step": step,
        "level": level,
        "shape": [len(grid), max((len(row) for row in grid), default=0)],
        "grid": grid,
    }


def _run_diff(code: str) -> dict:
    before = _frame([[0, 1, 1], [0, 0, 2]], step=0)
    after = _frame([[0, 2, 1], [3, 0, 2]], step=1)
    return run_sandboxed_python(
        code=code,
        timeout_seconds=5,
        initial_state={
            "current_frame": after,
            "history": [
                {"action": "", "frame": before},
                {"action": "LEFT", "frame": after},
            ],
            "valid_actions": ["LEFT", "RIGHT"],
            "last_action_result": {"executed": True},
            "experience": {},
            "strategy": {},
        },
        action_handler=lambda actions: {"action_result": {}, "state": {}},
    )


class PythonToolFrameDiffTests(TestCase):
    def test_diff_algorithm_is_bounded_and_uses_color_symbols(self) -> None:
        result = _bounded_frame_diff_summary(
            [[0, 1, 1], [0, 0, 2]],
            [[0, 2, 1], [3, 0, 2]],
            before_shape=(2, 3),
            after_shape=(2, 3),
            before_level=1,
            after_level=1,
            color_chars=".abcdefghijklmno",
            limit=1,
        )

        self.assertEqual(result["changed_cells"], 2)
        self.assertEqual(result["changed_bbox"], [0, 0, 1, 1])
        self.assertEqual(
            result["changes"],
            [{"row": 0, "col": 1, "before": "a", "after": "b"}],
        )
        self.assertEqual(result["truncated_changes"], 1)

    def test_frame_diff_returns_bounded_structured_summary(self) -> None:
        response = _run_diff("result = current_frame.diff(previous_frame, limit=1)")

        self.assertEqual(response["error"], "")
        result = response["result"]
        self.assertEqual(result["changed_cells"], 2)
        self.assertEqual(result["changed_bbox"], [0, 0, 1, 1])
        self.assertEqual(len(result["changes"]), 1)
        self.assertEqual(result["truncated_changes"], 1)
        self.assertEqual(result["changes"][0]["row"], 0)
        self.assertEqual(result["changes"][0]["col"], 1)
        self.assertIsInstance(result["changes"][0]["before"], str)
        self.assertIsInstance(result["changes"][0]["after"], str)

    def test_transition_diff_uses_before_and_after_frames(self) -> None:
        response = _run_diff("result = last_transition.diff(limit=4)")

        self.assertEqual(response["error"], "")
        self.assertEqual(response["result"]["changed_cells"], 2)
        self.assertEqual(response["result"]["truncated_changes"], 0)

    def test_frame_diff_handles_shape_and_level_changes(self) -> None:
        response = run_sandboxed_python(
            code="result = current_frame.diff(previous_frame)",
            timeout_seconds=5,
            initial_state={
                "current_frame": _frame([[0, 1], [2, 3]], step=1, level=2),
                "history": [
                    {"action": "", "frame": _frame([[0]], step=0, level=1)},
                    {
                        "action": "SPACE",
                        "frame": _frame([[0, 1], [2, 3]], step=1, level=2),
                    },
                ],
                "valid_actions": ["SPACE"],
                "last_action_result": {"level_completed": True},
                "experience": {},
                "strategy": {},
            },
            action_handler=lambda actions: {"action_result": {}, "state": {}},
        )

        self.assertEqual(response["error"], "")
        self.assertTrue(response["result"]["shape_changed"])
        self.assertTrue(response["result"]["level_changed"])
        self.assertEqual(response["result"]["changed_cells"], 3)
