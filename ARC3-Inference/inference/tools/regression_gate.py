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
    benchmark = _load_json(run_dir / "benchmark.json")
    run_config_path = run_dir / "run_config.json"
    run_config = _load_json(run_config_path) if run_config_path.exists() else {}
    trace = summarize_run_traces(run_dir)["overall"]
    actions = float(trace.get("actions") or 0)
    repeated_noops = float(trace.get("repeated_no_ops") or 0)
    game_runs = [item for item in benchmark.get("game_runs", []) if isinstance(item, dict)]
    trial_count = len(game_runs)
    completed_levels = sum(max(0, int(item.get("levels_completed") or 0)) for item in game_runs)
    total_levels = sum(max(0, int(item.get("number_of_levels") or 0)) for item in game_runs)
    wins = sum(
        str(item.get("state") or "").lower() == "won"
        or (
            int(item.get("number_of_levels") or 0) > 0
            and int(item.get("levels_completed") or 0) >= int(item.get("number_of_levels") or 0)
        )
        for item in game_runs
    )
    histories = [
        entry
        for item in game_runs
        for entry in (item.get("history") or [])
        if isinstance(entry, dict)
    ]
    generated_tokens = sum(max(0, int(item.get("generated_tokens") or 0)) for item in histories)
    trial_wallclock = sum(
        max(
            (max(0.0, float(entry.get("wallclock_seconds") or 0.0)) for entry in (item.get("history") or []) if isinstance(entry, dict)),
            default=0.0,
        )
        for item in game_runs
    )
    capabilities = run_config.get("inference_capabilities") or {}
    return {
        "overall_score": float(evaluation.get("score") or 0.0),
        "no_op_rate": float(trace.get("no_op_rate") or 0.0),
        "repeated_noop_rate": repeated_noops / actions if actions else 0.0,
        "rewarding_action_rate": float(trace.get("rewarding_action_rate") or 0.0),
        "terminal_state_violations": float(trace.get("terminal_state_violations") or 0),
        "trace_count": float(trace.get("trace_count") or 0),
        "trial_count": float(trial_count),
        "whole_game_win_rate": wins / trial_count if trial_count else 0.0,
        "level_completion_rate": completed_levels / total_levels if total_levels else 0.0,
        "crash_rate": sum(str(item.get("state") or "").lower() == "crashed" for item in game_runs) / trial_count if trial_count else 0.0,
        "cancel_rate": sum(str(item.get("state") or "").lower() == "cancelled" for item in game_runs) / trial_count if trial_count else 0.0,
        "generated_tokens_per_level": generated_tokens / max(1, completed_levels),
        "actions_per_level": len(histories) / max(1, completed_levels),
        "mean_trial_wallclock_seconds": trial_wallclock / trial_count if trial_count else 0.0,
        "outcome_aware_enabled": float(str(capabilities.get("controller_policy") or "") == "outcome_aware"),
        "candidate_count": float(capabilities.get("candidate_count") or 0),
        "verified_plan_min_support": float(capabilities.get("plan_min_support") or 0),
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
        "trial_count": "min_trial_count",
        "whole_game_win_rate": "min_whole_game_win_rate",
        "level_completion_rate": "min_level_completion_rate",
        "outcome_aware_enabled": "min_outcome_aware_enabled",
        "candidate_count": "min_candidate_count",
        "verified_plan_min_support": "min_verified_plan_support",
    }
    maxima = {
        "no_op_rate": "max_noop_rate",
        "repeated_noop_rate": "max_repeated_noop_rate",
        "terminal_state_violations": "max_terminal_state_violations",
        "crash_rate": "max_crash_rate",
        "cancel_rate": "max_cancel_rate",
        "generated_tokens_per_level": "max_generated_tokens_per_level",
        "actions_per_level": "max_actions_per_level",
        "mean_trial_wallclock_seconds": "max_mean_trial_wallclock_seconds",
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
