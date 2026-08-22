"""Offline quality evaluation for generated Duck Python programs."""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from inference.agent.action_names import to_engine_action, to_model_action
from inference.agent.tool_agent import _parse_bounded_generated_python


@dataclass(frozen=True)
class CodegenQualitySummary:
    samples: int
    valid_programs: int
    action_contract_passes: int
    expected_action_passes: int
    failures: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        denominator = max(1, self.samples)
        payload["valid_program_rate"] = self.valid_programs / denominator
        payload["action_contract_rate"] = self.action_contract_passes / denominator
        payload["expected_action_rate"] = self.expected_action_passes / denominator
        return payload


def _canonical_action(value: Any) -> str:
    raw = str(value or "").strip()
    return to_model_action(to_engine_action(raw) or raw)


def _literal_actions(tree: ast.AST) -> tuple[list[str], bool]:
    actions: list[str] = []
    dynamic = False
    for node in ast.walk(tree):
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

    return CodegenQualitySummary(
        samples=total,
        valid_programs=valid_programs,
        action_contract_passes=action_contract_passes,
        expected_action_passes=expected_action_passes,
        failures=failures,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected one JSON object per line")
        samples.append(value)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="JSONL code-generation evaluation corpus")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    summary = evaluate_samples(load_jsonl(args.corpus)).to_dict()
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
