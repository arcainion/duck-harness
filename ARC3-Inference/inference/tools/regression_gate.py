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
    game_runs = [
        item for item in benchmark.get("game_runs", []) if isinstance(item, dict)
    ]
    trial_count = len(game_runs)
    completed_levels = sum(
        max(0, int(item.get("levels_completed") or 0)) for item in game_runs
    )
    total_levels = sum(
        max(0, int(item.get("number_of_levels") or 0)) for item in game_runs
    )
    wins = sum(
        str(item.get("state") or "").lower() == "won"
        or (
            int(item.get("number_of_levels") or 0) > 0
            and int(item.get("levels_completed") or 0)
            >= int(item.get("number_of_levels") or 0)
        )
        for item in game_runs
    )
    histories = [
        entry
        for item in game_runs
        for entry in (item.get("history") or [])
        if isinstance(entry, dict)
    ]
    generated_tokens = sum(
        max(0, int(item.get("generated_tokens") or 0)) for item in histories
    )
    trial_wallclock = sum(
        max(
            (
                max(0.0, float(entry.get("wallclock_seconds") or 0.0))
                for entry in (item.get("history") or [])
                if isinstance(entry, dict)
            ),
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
        "level_completion_rate": completed_levels / total_levels
        if total_levels
        else 0.0,
        "crash_rate": sum(
            str(item.get("state") or "").lower() == "crashed" for item in game_runs
        )
        / trial_count
        if trial_count
        else 0.0,
        "cancel_rate": sum(
            str(item.get("state") or "").lower() == "cancelled" for item in game_runs
        )
        / trial_count
        if trial_count
        else 0.0,
        "generated_tokens_per_level": generated_tokens / max(1, completed_levels),
        "actions_per_level": len(histories) / max(1, completed_levels),
        "mean_trial_wallclock_seconds": trial_wallclock / trial_count
        if trial_count
        else 0.0,
        "outcome_aware_enabled": float(
            str(capabilities.get("controller_policy") or "") == "outcome_aware"
        ),
        "candidate_count": float(capabilities.get("candidate_count") or 0),
        "verified_plan_min_support": float(capabilities.get("plan_min_support") or 0),
        "game_token_budget": float(capabilities.get("game_token_budget") or 0),
        "reasoning_control_v2": float(
            all(
                bool(capabilities.get(name))
                for name in (
                    "causal_model_grounded",
                    "reward_and_terminal_aware",
                    "information_gain_exploration",
                    "contextual_plan_validation",
                    "temporal_motion_evidence",
                )
            )
        ),
        "audit_controls_v3": float(
            all(
                bool(capabilities.get(name))
                for name in (
                    "independent_evidence_support",
                    "process_safe_knowledge_store",
                    "adaptive_inference_budget",
                    "terminal_game_token_budget",
                    "complete_inference_trace_contract",
                    "provider_usage_fallback",
                    "pareto_utility_planning",
                    "simple_path_planning",
                    "state_conditioned_causal_model",
                    "causal_cache_coherence",
                    "bounded_causal_eviction",
                    "full_batch_harm_guard",
                    "dynamic_host_harm_guard",
                    "partial_batch_diagnostics",
                    "guard_metric_taxonomy",
                    "reset_scoped_volatility",
                    "repeatable_exact_effect_recovery",
                    "state_dependent_batch_actions",
                    "declining_request_deadline",
                    "lossless_bounded_knowledge_merge",
                    "confidence_aware_harm_evidence",
                    "coordinate_scoped_parameterized_evidence",
                    "joint_transition_calibration",
                    "provenance_weighted_action_models",
                    "observable_decision_state",
                    "behavior_backed_regression_gates",
                    "contingent_observation_planning",
                    "delayed_credit_assignment",
                    "object_centric_temporal_state",
                    "hypothesis_directed_exploration",
                    "adaptive_recovery_portfolio",
                    "nonstationary_dynamics_detection",
                    "state_context_continuity",
                    "animation_aware_outcomes",
                    "executable_branch_policy",
                    "regime_aware_recency",
                    "object_abstract_planning",
                    "causal_credit_continuity",
                    "deterministic_budget_fallback",
                    "versioned_state_memory",
                )
            )
        ),
        "plan_follow_rate": float(trace.get("plan_follow_rate") or 0.0),
        "harm_interventions": float(trace.get("harm_interventions") or 0.0),
        "followed_plan_progress_rate": float(
            trace.get("followed_plan_progress_rate") or 0.0
        ),
        "inference_telemetry_rate": float(trace.get("inference_telemetry_rate") or 0.0),
        "prediction_evaluation_rate": float(
            trace.get("prediction_evaluation_rate") or 0.0
        ),
        "plan_recommendation_rate": float(trace.get("plan_recommendation_rate") or 0.0),
        "state_context_continuity_rate": float(
            trace.get("state_context_continuity_rate")
            if trace.get("state_context_continuity_rate") is not None
            else 1.0
        ),
        "state_context_link_rate": float(
            trace.get("state_context_link_rate")
            if trace.get("state_context_link_rate") is not None
            else 0.0
        ),
        "transient_effect_classification_rate": float(
            trace.get("transient_effect_classification_rate")
            if trace.get("transient_effect_classification_rate") is not None
            else 1.0
        ),
        "plan_policy_ready_rate": float(trace.get("plan_policy_ready_rate") or 0.0),
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
        "reasoning_control_v2": "min_reasoning_control_v2",
        "audit_controls_v3": "min_audit_controls_v3",
        "game_token_budget": "min_game_token_budget",
        "inference_telemetry_rate": "min_inference_telemetry_rate",
        "prediction_evaluation_rate": "min_prediction_evaluation_rate",
        "plan_recommendation_rate": "min_plan_recommendation_rate",
        "plan_follow_rate": "min_plan_follow_rate",
        "followed_plan_progress_rate": "min_followed_plan_progress_rate",
        "state_context_continuity_rate": "min_state_context_continuity_rate",
        "state_context_link_rate": "min_state_context_link_rate",
        "transient_effect_classification_rate": "min_transient_effect_classification_rate",
        "plan_policy_ready_rate": "min_plan_policy_ready_rate",
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
    result = evaluate_gate(
        metrics, dict(config.get("thresholds") or {}), baseline_metrics=baseline
    )
    print(
        json.dumps(
            {
                "passed": result.passed,
                "metrics": result.metrics,
                "failures": result.failures,
            },
            indent=2,
        )
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
