"""Offline quality evaluation for generated Duck Python programs."""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from inference.agent.action_names import to_engine_action, to_model_action
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.tool_agent import (
    _generated_python_preflight_issues,
    _parse_bounded_generated_python,
    _reachable_module_node_ids,
)


@dataclass(frozen=True)
class CodegenQualitySummary:
    samples: int
    valid_programs: int
    action_contract_passes: int
    expected_action_passes: int
    preflight_passes: int
    executable_samples: int
    execution_passes: int
    result_passes: int
    failures: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        denominator = max(1, self.samples)
        payload["valid_program_rate"] = self.valid_programs / denominator
        payload["action_contract_rate"] = self.action_contract_passes / denominator
        payload["expected_action_rate"] = self.expected_action_passes / denominator
        payload["preflight_rate"] = self.preflight_passes / denominator
        executable_denominator = max(1, self.executable_samples)
        payload["execution_rate"] = self.execution_passes / executable_denominator
        payload["result_rate"] = self.result_passes / executable_denominator
        return payload


def _canonical_action(value: Any) -> str:
    raw = str(value or "").strip()
    return to_model_action(to_engine_action(raw) or raw)


def _literal_actions(tree: ast.AST) -> tuple[list[str], bool]:
    actions: list[str] = []
    dynamic = False
    reachable_nodes = _reachable_module_node_ids(tree)
    for node in ast.walk(tree):
        if id(node) not in reachable_nodes:
            continue
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "action"
            and node.args
        ):
            continue
        try:
            value = ast.literal_eval(node.args[0])
        except (SyntaxError, ValueError):
            dynamic = True
            continue
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            name = item.get("action") if isinstance(item, dict) else item
            canonical = _canonical_action(name)
            if canonical:
                actions.append(canonical)
    return actions, dynamic


def evaluate_samples(samples: Iterable[dict[str, Any]]) -> CodegenQualitySummary:
    total = valid_programs = action_contract_passes = expected_action_passes = 0
    preflight_passes = executable_samples = execution_passes = result_passes = 0
    failures: dict[str, int] = {}

    def fail(category: str) -> None:
        failures[category] = failures.get(category, 0) + 1

    for sample in samples:
        total += 1
        code = sample.get("code")
        if not isinstance(code, str) or not code.strip():
            fail("empty_code")
            continue
        try:
            tree = _parse_bounded_generated_python(code)
            compile(tree, "<codegen_quality>", "exec")
        except (SyntaxError, TypeError, ValueError, OverflowError):
            fail("invalid_program")
            continue
        valid_programs += 1
        if _generated_python_preflight_issues(tree):
            fail("preflight")
            continue
        preflight_passes += 1

        actions, dynamic_actions = _literal_actions(tree)
        valid_actions = {
            _canonical_action(value) for value in sample.get("valid_actions", [])
        }
        requires_action = bool(sample.get("expect_action", False))
        action_contract_ok = not requires_action or bool(actions) or dynamic_actions
        if actions and valid_actions:
            action_contract_ok = action_contract_ok and all(
                action in valid_actions for action in actions
            )
        if action_contract_ok:
            action_contract_passes += 1
        else:
            fail("action_contract")

        expected = {
            _canonical_action(value) for value in sample.get("expected_actions", [])
        }
        expected_ok = not expected or bool(expected.intersection(actions))
        if expected_ok:
            expected_action_passes += 1
        else:
            fail("expected_action")

        if not bool(sample.get("execute", False)):
            continue
        executable_samples += 1
        initial_state = dict(sample.get("initial_state") or {})
        initial_state.setdefault(
            "current_frame",
            {"ascii": "W", "grid": [[0]], "shape": [1, 1], "step": 0, "level": 1},
        )
        initial_state.setdefault("history", [])
        initial_state.setdefault("valid_actions", list(sample.get("valid_actions") or []))
        captured_actions: list[dict[str, Any]] = []

        def action_handler(actions_payload: list[dict[str, Any]]) -> dict[str, Any]:
            captured_actions.extend(dict(item) for item in actions_payload)
            return {
                "action_result": {
                    "executed": True,
                    "steps": [
                        {"executed": True, "action_display": str(item.get("action", ""))}
                        for item in actions_payload
                    ],
                },
                "state": initial_state,
            }

        execution = run_sandboxed_python(
            code=code,
            timeout_seconds=max(1, min(10, int(sample.get("timeout_seconds") or 3))),
            initial_state=initial_state,
            action_handler=action_handler,
        )
        if execution.get("error"):
            fail("execution")
            continue
        execution_passes += 1
        expected_result_declared = "expected_result" in sample
        result_ok = not expected_result_declared or execution.get("result") == sample.get("expected_result")
        expected_executed = [_canonical_action(item) for item in sample.get("expected_executed_actions", [])]
        actual_executed = [_canonical_action(item.get("action")) for item in captured_actions]
        if expected_executed and actual_executed != expected_executed:
            result_ok = False
        if result_ok:
            result_passes += 1
        else:
            fail("result")

    return CodegenQualitySummary(
        samples=total,
        valid_programs=valid_programs,
        action_contract_passes=action_contract_passes,
        expected_action_passes=expected_action_passes,
        preflight_passes=preflight_passes,
        executable_samples=executable_samples,
        execution_passes=execution_passes,
        result_passes=result_passes,
        failures=failures,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected one JSON object per line")
        samples.append(value)
    return samples


def _threshold_failures(
    summary: dict[str, Any], thresholds: Any
) -> list[str]:
    if not isinstance(thresholds, dict):
        raise ValueError("quality thresholds must be a JSON object")
    failures: list[str] = []
    for metric, minimum in thresholds.items():
        if metric == "min_samples":
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
                raise ValueError("min_samples must be a non-negative integer")
            if int(summary["samples"]) < minimum:
                failures.append(f"samples={summary['samples']} < {minimum}")
            continue
        if not isinstance(metric, str) or not metric.startswith("min_"):
            raise ValueError(f"unknown quality threshold: {metric!r}")
        rate_name = metric[4:]
        if rate_name not in summary or not rate_name.endswith("_rate"):
            raise ValueError(f"unknown quality rate threshold: {metric}")
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise ValueError(f"{metric} must be a number between 0 and 1")
        minimum_rate = float(minimum)
        if not 0.0 <= minimum_rate <= 1.0:
            raise ValueError(f"{metric} must be between 0 and 1")
        actual = float(summary[rate_name])
        if actual < minimum_rate:
            failures.append(f"{rate_name}={actual:.6g} < {minimum_rate:.6g}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="JSONL code-generation evaluation corpus")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--thresholds", type=Path, help="Optional JSON minimum-rate gate")
    args = parser.parse_args()
    summary = evaluate_samples(load_jsonl(args.corpus)).to_dict()
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    if args.thresholds is not None:
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        failures = _threshold_failures(summary, thresholds)
        if failures:
            print(json.dumps({"gate_passed": False, "failures": failures}, indent=2))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
