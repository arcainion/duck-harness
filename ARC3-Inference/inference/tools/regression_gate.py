"""Fail a run when score or closed-loop behavior regresses past configured bounds."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.tools.trace_metrics import summarize_run_traces


@dataclass(frozen=True)
class GateResult:
    passed: bool
    metrics: dict[str, float]
    failures: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def collect_metrics(run_dir: Path) -> dict[str, float]:
    evaluation = _load_json(run_dir / "evaluation.json")
    trace = summarize_run_traces(run_dir)["overall"]
    actions = float(trace.get("actions") or 0)
    repeated_noops = float(trace.get("repeated_no_ops") or 0)
    return {
        "overall_score": float(evaluation.get("score") or 0.0),
        "no_op_rate": float(trace.get("no_op_rate") or 0.0),
        "repeated_noop_rate": repeated_noops / actions if actions else 0.0,
        "rewarding_action_rate": float(trace.get("rewarding_action_rate") or 0.0),
        "terminal_state_violations": float(trace.get("terminal_state_violations") or 0),
        "trace_count": float(trace.get("trace_count") or 0),
    }


def evaluate_gate(
    metrics: dict[str, float],
    thresholds: dict[str, Any],
    *,
    baseline_metrics: dict[str, float] | None = None,
) -> GateResult:
    failures: list[str] = []
    minima = {
        "overall_score": "min_overall_score",
        "rewarding_action_rate": "min_rewarding_action_rate",
        "trace_count": "min_trace_count",
    }
    maxima = {
        "no_op_rate": "max_noop_rate",
        "repeated_noop_rate": "max_repeated_noop_rate",
        "terminal_state_violations": "max_terminal_state_violations",
    }
    for metric, threshold_name in minima.items():
        if threshold_name not in thresholds:
            continue
        limit = float(thresholds[threshold_name])
        if metrics.get(metric, 0.0) < limit:
            failures.append(f"{metric}={metrics.get(metric, 0.0):.6g} < {limit:.6g}")
    for metric, threshold_name in maxima.items():
        if threshold_name not in thresholds:
            continue
        limit = float(thresholds[threshold_name])
        if metrics.get(metric, 0.0) > limit:
            failures.append(f"{metric}={metrics.get(metric, 0.0):.6g} > {limit:.6g}")
    if baseline_metrics is not None:
        ratio = float(thresholds.get("min_score_ratio_vs_baseline", 1.0))
        required = baseline_metrics.get("overall_score", 0.0) * ratio
        if metrics.get("overall_score", 0.0) < required:
            failures.append(
                f"overall_score={metrics.get('overall_score', 0.0):.6g} < "
                f"baseline requirement {required:.6g}"
            )
    return GateResult(not failures, dict(metrics), tuple(failures))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Candidate run directory")
    parser.add_argument("--config", default="configs/regression.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_json(Path(args.config))
    run_dir = Path(args.run_dir or str(config.get("run_dir") or ""))
    if not str(run_dir):
        raise ValueError("A run directory is required.")
    metrics = collect_metrics(run_dir)
    baseline_raw = str(config.get("baseline_run_dir") or "").strip()
    baseline = collect_metrics(Path(baseline_raw)) if baseline_raw else None
    result = evaluate_gate(metrics, dict(config.get("thresholds") or {}), baseline_metrics=baseline)
    print(json.dumps({"passed": result.passed, "metrics": result.metrics, "failures": result.failures}, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

