"""LLM objective reduction plus generated CPU/CUDA gameplay policies."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests

from inference.agent.gameplay_policy_runtime import (
    GameplayPolicyRuntime,
    PolicyDecision,
    PolicyObservation,
    PolicyRuntimeError,
    PolicyStatus,
    verify_policy_source,
)
from inference.agent.objective_reduction import (
    ObjectiveError,
    ObjectiveKind,
    ObjectiveTree,
    ReductionProposal,
)
from inference.agent.runtime_state import Frame, HistoryEntry, load_runtime_state
from inference.agent.tool_agent import (
    AnalyzerTurnResult,
    ToolAgent,
    _extract_reasoning_text,
    _normalize_message_content,
    _recover_tool_calls_from_markup,
)


log = logging.getLogger(__name__)

_STATE_VERSION = 1
_MAX_ROLE_HISTORY = 12
_MAX_STRUCTURED_ATTEMPTS = 3
_MAX_POLICY_REPAIRS = 2
_MAX_NO_ACTION_BOUNDARIES = 8
_MAX_CONSECUTIVE_POLICY_FAILURES = _MAX_POLICY_REPAIRS + 1
_DEFAULT_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS = 300.0
_DEFAULT_REDUCER_MAX_OUTPUT = 4096
_DEFAULT_CODER_MAX_OUTPUT = 8192
_DEFAULT_REDUCER_THINKING_BUDGET = 2048
_DEFAULT_CODER_THINKING_BUDGET = 3072
_ORCHESTRATION_TOOL_CHOICE = "required"


class OrchestrationFailure(RuntimeError):
    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


class OrchestrationYield(RuntimeError):
    pass


def _empty_orchestration_metrics() -> dict[str, int | float]:
    return {
        "reducer_calls": 0,
        "coder_calls": 0,
        "reducer_attempts": 0,
        "coder_attempts": 0,
        "reducer_timeouts": 0,
        "coder_timeouts": 0,
        "reducer_transport_failures": 0,
        "coder_transport_failures": 0,
        "reducer_model_seconds": 0.0,
        "coder_model_seconds": 0.0,
        "reducer_generated_tokens": 0,
        "coder_generated_tokens": 0,
        "policy_activations": 0,
        "policy_repairs": 0,
        "policy_steps": 0,
        "cuda_fallbacks": 0,
        "cpu_policy_seconds": 0.0,
        "cuda_policy_seconds": 0.0,
        "objectives_completed": 0,
        "objectives_failed": 0,
    }


def _positive_env_float(name: str, default: float) -> float | None:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning("invalid %s=%r; using %s", name, raw, default)
        value = default
    return value if value > 0 else None


def _positive_env_int(name: str, default: int) -> int | None:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning("invalid %s=%r; using %s", name, raw, default)
        value = default
    return value if value > 0 else None


def objective_reduction_enabled() -> bool:
    return os.environ.get(
        "LOCAL_ANALYZER_OBJECTIVE_REDUCTION", "false"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tool_schema(
    name: str, description: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {**parameters, "additionalProperties": False},
        },
    }


_REDUCTION_TOOL = _tool_schema(
    "submit_reduction",
    "Submit one validated objective-tree transition.",
    {
        "type": "object",
        "properties": {
            "objective_id": {"type": "string", "maxLength": 200},
            "verdict": {
                "type": "string",
                "enum": ["continue", "complete", "fail", "decompose"],
            },
            "evidence": {"type": "string", "maxLength": 2400},
            "rationale": {"type": "string", "maxLength": 2400},
            "selected_index": {"type": "integer", "minimum": 0, "maximum": 5},
            "subgoals": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "maxLength": 200},
                        "success_criteria": {"type": "string", "maxLength": 1200},
                        "failure_criteria": {"type": "string", "maxLength": 1200},
                        "expected_evidence": {"type": "string", "maxLength": 1200},
                        "action_budget": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 32,
                        },
                    },
                    "required": [
                        "title",
                        "success_criteria",
                        "failure_criteria",
                        "expected_evidence",
                        "action_budget",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "objective_id",
            "verdict",
            "evidence",
            "rationale",
            "selected_index",
            "subgoals",
        ],
    },
)

_POLICY_TOOL = _tool_schema(
    "submit_policy",
    "Submit one complete versioned gameplay policy module.",
    {
        "type": "object",
        "properties": {
            "objective_id": {"type": "string", "maxLength": 200},
            "source": {"type": "string", "maxLength": 65536},
            "backend_capabilities": {
                "type": "array",
                "items": {"type": "string", "enum": ["cpu", "cuda"]},
                "minItems": 1,
                "maxItems": 2,
            },
            "self_test_notes": {"type": "string", "maxLength": 2000},
        },
        "required": [
            "objective_id",
            "source",
            "backend_capabilities",
            "self_test_notes",
        ],
    },
)

_REDUCER_SYSTEM_PROMPT = """You are the objective-reducer role for an ARC-AGI-3 game.
Think carefully, then call submit_reduction exactly once. The host owns all IDs,
tree invariants, budgets, and game/level completion. You may complete or fail only
the active tactical objective. A game or level objective must be decomposed into
one to six falsifiable tactical subgoals. Prefer one selected subgoal whose result
will maximally reduce uncertainty. Keep reasoning compact and reserve response
space for the required tool call. Do not emit actions or Python code."""

_CODER_SYSTEM_PROMPT = """You are the gameplay-policy coder role for an ARC-AGI-3 game.
Think carefully, then call submit_policy exactly once with a complete Python module.
The module must define POLICY_API_VERSION = 1, SUPPORTED_BACKENDS containing cpu,
optional initialize(context), and decide(observation, memory). decide returns a
PolicyDecision or equivalent mapping. A continue decision must contain exactly one
currently-valid action. subgoal_succeeded/subgoal_failed contain no action. The
observation has immutable uint8 board[64,64], level, step, valid_actions,
last_transition, objective, recent_transitions, and backend. Allowed imports are
math/statistics/collections/itertools/functools/heapq/bisect/numpy and optional
lazy torch. No files, network, subprocesses, reflection, engine calls, or hidden
state. Keep all persistent state finite and JSON-serializable. CPU is mandatory;
CUDA may only be an optional optimization. Keep reasoning compact and reserve
response space for the complete policy and required tool call."""


def _compact_board(frame: Frame) -> list[str]:
    return [
        "".join(format(max(0, min(15, int(cell))), "x") for cell in row)
        for row in frame.grid
    ]


def _recent_transition_payload(
    history: list[HistoryEntry], limit: int = 8
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in history[-max(1, limit) :]:
        payload.append(
            {
                "action": entry.action,
                "reward": float(entry.reward),
                "level": entry.frame.level,
                "step": entry.frame.step,
                "engine_state": entry.frame.engine_state,
                "outcome": entry.outcome_class_override,
            }
        )
    return payload


def _json_object_from_message(
    agent: ToolAgent, message: dict[str, Any]
) -> dict[str, Any]:
    raw_calls = json.loads(json.dumps(message.get("tool_calls") or []))
    calls = agent._normalize_response_tool_calls(raw_calls)
    if not calls:
        content = _normalize_message_content(message.get("content", ""))
        calls = agent._normalize_response_tool_calls(
            _recover_tool_calls_from_markup(_extract_reasoning_text(message), content)
        )
    if calls:
        function = calls[0].get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            decoded = json.loads(arguments)
            if isinstance(decoded, dict):
                return decoded
    content = _normalize_message_content(message.get("content", "")).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S)
    decoded = json.loads(content)
    if not isinstance(decoded, dict):
        raise ValueError("structured response is not an object")
    return decoded


def _without_policy_source(entry: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(entry)
    content = sanitized.get("content")
    if not isinstance(content, str):
        return sanitized
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return sanitized
    if not isinstance(payload, dict) or "source" not in payload:
        return sanitized
    payload.pop("source", None)
    payload["source_artifact"] = "stored_by_hash"
    sanitized["content"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    return sanitized


class OrchestratedObjectiveAgent(ToolAgent):
    """Analyzer that calls the LLM only at objective boundaries and failures."""

    disable_controller_fallback = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tree: ObjectiveTree | None = None
        self._role_histories: dict[str, list[dict[str, Any]]] = {
            "reducer": [],
            "coder": [],
        }
        self._policy_runtime: GameplayPolicyRuntime | None = None
        self._policy_objective_id = ""
        self._policy_source_hash = ""
        self._policy_artifact = ""
        self._policy_memory: Any = {}
        self._policy_repairs: dict[str, int] = {}
        self._boundary_reason = "game_start"
        self._reduction_required = True
        self._last_transition: dict[str, Any] | None = None
        self._recent_transitions: list[dict[str, Any]] = []
        self._consecutive_activation_failures = 0
        self._last_reduction_step = 0
        self._orchestration_metrics = _empty_orchestration_metrics()
        self._current_transcript_path: Path | None = None
        self._orchestration_request_timeout_seconds = _positive_env_float(
            "LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS",
            _DEFAULT_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS,
        )
        self._orchestration_role_max_output = {
            "reducer": _positive_env_int(
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT",
                _DEFAULT_REDUCER_MAX_OUTPUT,
            ),
            "coder": _positive_env_int(
                "LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT",
                _DEFAULT_CODER_MAX_OUTPUT,
            ),
        }
        self._orchestration_role_thinking_budget = {
            "reducer": _positive_env_int(
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET",
                _DEFAULT_REDUCER_THINKING_BUDGET,
            ),
            "coder": _positive_env_int(
                "LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET",
                _DEFAULT_CODER_THINKING_BUDGET,
            ),
        }

    def close(self) -> None:
        if self._policy_runtime is not None:
            self._policy_runtime.close()
            self._policy_runtime = None
        super().close()

    def _durable_state_path(self) -> Path | None:
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None:
            return None
        return runtime_path.with_name(f"{runtime_path.stem}_objective_state.json")

    def _reset_orchestration_state(self) -> None:
        if self._policy_runtime is not None:
            self._policy_runtime.close()
        self._policy_runtime = None
        self._tree = None
        self._role_histories = {"reducer": [], "coder": []}
        self._policy_objective_id = ""
        self._policy_source_hash = ""
        self._policy_artifact = ""
        self._policy_memory = {}
        self._policy_repairs = {}
        self._boundary_reason = "game_start"
        self._reduction_required = True
        self._last_transition = None
        self._recent_transitions = []
        self._consecutive_activation_failures = 0
        self._last_reduction_step = 0
        self._orchestration_metrics = _empty_orchestration_metrics()

    def _load_durable_state(self) -> None:
        self._reset_orchestration_state()
        path = self._durable_state_path()
        if path is None or not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("version") != _STATE_VERSION
            ):
                raise ValueError("unsupported orchestrated objective state")
            if str(payload.get("game_id") or "") != self._knowledge_game_id:
                raise ValueError("orchestration state belongs to another game")
            raw_tree = payload.get("objective_tree")
            if isinstance(raw_tree, dict):
                self._tree = ObjectiveTree.from_dict(raw_tree)
            histories = payload.get("role_histories")
            if isinstance(histories, dict):
                for role in ("reducer", "coder"):
                    raw = histories.get(role)
                    if isinstance(raw, list):
                        self._role_histories[role] = [
                            dict(item)
                            for item in raw[-_MAX_ROLE_HISTORY:]
                            if isinstance(item, dict)
                        ]
            self._policy_objective_id = str(payload.get("policy_objective_id") or "")
            self._policy_source_hash = str(payload.get("policy_source_hash") or "")
            self._policy_artifact = str(payload.get("policy_artifact") or "")
            self._policy_memory = payload.get("policy_memory", {})
            self._policy_repairs = {
                str(key): max(0, int(value or 0))
                for key, value in (payload.get("policy_repairs") or {}).items()
            }
            self._boundary_reason = str(payload.get("boundary_reason") or "resume")
            self._reduction_required = bool(payload.get("reduction_required", False))
            self._last_transition = (
                dict(payload["last_transition"])
                if isinstance(payload.get("last_transition"), dict)
                else None
            )
            self._recent_transitions = [
                dict(item)
                for item in payload.get("recent_transitions") or []
                if isinstance(item, dict)
            ][-8:]
            self._consecutive_activation_failures = max(
                0, int(payload.get("consecutive_activation_failures", 0) or 0)
            )
            self._last_reduction_step = max(
                0, int(payload.get("last_reduction_step", 0) or 0)
            )
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                self._orchestration_metrics.update(
                    {
                        str(key): value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                )
            self._session_total_tokens = max(
                0, int(payload.get("total_tokens", 0) or 0)
            )
            self._session_generated_tokens = max(
                0, int(payload.get("generated_tokens", 0) or 0)
            )
        except (
            OSError,
            TypeError,
            ValueError,
            ObjectiveError,
            json.JSONDecodeError,
        ) as exc:
            log.warning("orchestrated objective state ignored at %s: %s", path, exc)
            self._reset_orchestration_state()

    def _persist_durable_state(self) -> None:
        path = self._durable_state_path()
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if path is None or runtime_path is None or not runtime_path.exists():
            return
        payload = {
            "version": _STATE_VERSION,
            "game_id": self._knowledge_game_id,
            "objective_tree": self._tree.to_dict() if self._tree is not None else None,
            "role_histories": {
                role: [
                    _without_policy_source(item) if role == "coder" else item
                    for item in history[-_MAX_ROLE_HISTORY:]
                ]
                for role, history in self._role_histories.items()
            },
            "policy_objective_id": self._policy_objective_id,
            "policy_source_hash": self._policy_source_hash,
            "policy_artifact": self._policy_artifact,
            "policy_memory": self._policy_memory,
            "policy_repairs": self._policy_repairs,
            "boundary_reason": self._boundary_reason,
            "reduction_required": self._reduction_required,
            "last_transition": self._last_transition,
            "recent_transitions": self._recent_transitions[-8:],
            "consecutive_activation_failures": self._consecutive_activation_failures,
            "last_reduction_step": self._last_reduction_step,
            "metrics": self._orchestration_metrics,
            "total_tokens": self._session_total_tokens,
            "generated_tokens": self._session_generated_tokens,
        }
        try:
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as exc:
            log.warning(
                "orchestrated objective state write failed at %s: %s", path, exc
            )

    def _append_transcript(self, label: str, payload: Any) -> None:
        path = self._current_transcript_path
        if path is None:
            return
        text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, indent=2, default=str)
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{label}]\n{text}\n")
        except OSError as exc:
            log.warning("orchestration transcript write failed: %s", exc)

    def _emit_event(self, event_type: str, **payload: Any) -> None:
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None:
            return
        event = {
            "type": event_type,
            "time": time.time(),
            "game_id": self._knowledge_game_id,
            **payload,
        }
        self._append_transcript(f"ORCHESTRATION {event_type}", event)
        path = runtime_path.with_name("orchestration_events.jsonl")
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            log.warning("orchestration event write failed: %s", exc)

    def _remaining_seconds(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - time.monotonic()

    def _role_request_timeout(self, request_deadline: float | None) -> float | None:
        candidates: list[float] = []
        if self._orchestration_request_timeout_seconds is not None:
            candidates.append(self._orchestration_request_timeout_seconds)
        remaining = self._remaining_seconds(request_deadline)
        if remaining is not None:
            if remaining <= 0:
                raise OrchestrationYield("analyzer time budget")
            candidates.append(remaining)
        return max(0.1, min(candidates)) if candidates else None

    def _role_output_limit(self, role: str, attempt: int) -> int | None:
        role_limit = self._orchestration_role_max_output.get(role)
        inherited_limit = self._adaptive_output_limit(attempt, repair=attempt > 1)
        limits = [
            value
            for value in (role_limit, inherited_limit)
            if value is not None and value > 0
        ]
        return min(limits) if limits else None

    def _role_thinking_budget(
        self, role: str, output_limit: int | None
    ) -> int | None:
        budget = self._orchestration_role_thinking_budget.get(role)
        if budget is None or budget <= 0:
            return None
        if output_limit is None:
            return budget
        # Preserve room for the required structured result even if deployment
        # overrides accidentally set the thinking budget above the output cap.
        return min(budget, max(1, output_limit - 1))

    def _structured_role_call(
        self,
        *,
        role: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        tool: dict[str, Any],
        validator: Callable[[dict[str, Any]], Any],
        request_deadline: float | None,
        should_stop: Callable[[], bool] | None,
    ) -> Any:
        history = self._role_histories[role]
        correction = ""
        last_error = ""
        for attempt in range(1, _MAX_STRUCTURED_ATTEMPTS + 1):
            if should_stop is not None and should_stop():
                raise OrchestrationYield("stop requested")
            request_timeout = self._role_request_timeout(request_deadline)
            if self._remaining_game_tokens() == 0:
                raise OrchestrationFailure(
                    "orchestration exhausted the per-game model token budget",
                    category="orchestration_token_budget",
                )
            user_text = json.dumps(
                user_payload, ensure_ascii=False, separators=(",", ":")
            )
            if correction:
                user_text += f"\nPrevious response was rejected: {correction}\nReturn a corrected tool call."
            messages = [
                {"role": "system", "content": system_prompt},
                *history[-_MAX_ROLE_HISTORY:],
                {"role": "user", "content": user_text},
            ]
            self._should_stop_callback = should_stop
            output_limit = self._role_output_limit(role, attempt)
            thinking_budget = self._role_thinking_budget(role, output_limit)
            attempt_metric = f"{role}_attempts"
            self._orchestration_metrics[attempt_metric] = (
                int(self._orchestration_metrics.get(attempt_metric, 0)) + 1
            )
            request_started = time.monotonic()
            self._emit_event(
                "role_request_started",
                role=role,
                structured_attempt=attempt,
                request_timeout_seconds=request_timeout,
                max_output_tokens=output_limit,
                thinking_token_budget=thinking_budget,
                tool_choice=_ORCHESTRATION_TOOL_CHOICE,
            )
            try:
                result = self.model_client.complete(
                    messages,
                    tools=[tool],
                    request_timeout_seconds=request_timeout,
                    max_output_tokens=output_limit,
                    thinking_token_budget=thinking_budget,
                    tool_choice=_ORCHESTRATION_TOOL_CHOICE,
                    request_attempt_limit=1,
                )
            except requests.RequestException as exc:
                elapsed = max(0.0, time.monotonic() - request_started)
                failure_metric = f"{role}_transport_failures"
                self._orchestration_metrics[failure_metric] = (
                    int(self._orchestration_metrics.get(failure_metric, 0)) + 1
                )
                if isinstance(exc, requests.Timeout):
                    timeout_metric = f"{role}_timeouts"
                    self._orchestration_metrics[timeout_metric] = (
                        int(self._orchestration_metrics.get(timeout_metric, 0)) + 1
                    )
                self._emit_event(
                    "role_request_failed",
                    role=role,
                    structured_attempt=attempt,
                    elapsed_seconds=elapsed,
                    timeout=isinstance(exc, requests.Timeout),
                    detail=str(exc),
                )
                last_error = f"{type(exc).__name__}: {exc}"
                if should_stop is not None and should_stop():
                    raise OrchestrationYield("stop requested") from exc
                if attempt < _MAX_STRUCTURED_ATTEMPTS:
                    continue
                raise OrchestrationFailure(
                    f"{role} transport failed after {_MAX_STRUCTURED_ATTEMPTS} attempts: {exc}",
                    category=f"orchestration_{role}_transport_exhausted",
                ) from exc
            finally:
                elapsed = max(0.0, time.monotonic() - request_started)
                seconds_metric = f"{role}_model_seconds"
                self._orchestration_metrics[seconds_metric] = (
                    float(self._orchestration_metrics.get(seconds_metric, 0.0))
                    + elapsed
                )
            self._accumulate_usage_tokens(result.usage)
            self._orchestration_metrics[f"{role}_calls"] = int(
                self._orchestration_metrics.get(f"{role}_calls", 0)
            ) + max(1, result.request_attempts)
            generated_tokens = 0
            for key in ("completion_tokens", "output_tokens", "generated_tokens"):
                try:
                    generated_tokens = max(0, int((result.usage or {}).get(key)))
                    break
                except (AttributeError, TypeError, ValueError):
                    continue
            token_metric = f"{role}_generated_tokens"
            self._orchestration_metrics[token_metric] = (
                int(self._orchestration_metrics.get(token_metric, 0)) + generated_tokens
            )
            self._emit_event(
                "role_request_completed",
                role=role,
                structured_attempt=attempt,
                request_attempts=max(1, result.request_attempts),
                elapsed_seconds=elapsed,
                generated_tokens=generated_tokens,
                finish_reason=str(getattr(result, "finish_reason", "") or ""),
            )
            reasoning = _extract_reasoning_text(result.message)
            try:
                raw = _json_object_from_message(self, result.message)
                value = validator(raw)
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ObjectiveError,
                PolicyRuntimeError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                correction = last_error[:2000]
                self._append_transcript(f"{role.upper()} REJECTED", correction)
                continue
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
            }
            if reasoning:
                assistant_entry["reasoning_content"] = reasoning[-6000:]
            history.extend(
                [
                    {"role": "user", "content": user_text[-12000:]},
                    assistant_entry,
                ]
            )
            del history[:-_MAX_ROLE_HISTORY]
            transcript_payload = raw
            if role == "coder" and "source" in raw:
                transcript_payload = {
                    **raw,
                    "source": "<stored as a hashed policy artifact>",
                }
            self._append_transcript(f"{role.upper()} ACCEPTED", transcript_payload)
            return value
        raise OrchestrationFailure(
            f"{role} did not produce a valid structured result after {_MAX_STRUCTURED_ATTEMPTS} attempts: {last_error}",
            category=f"orchestration_{role}_exhausted",
        )

    def _reducer_payload(
        self, frame: Frame, history: list[HistoryEntry]
    ) -> dict[str, Any]:
        assert self._tree is not None
        return {
            "boundary_reason": self._boundary_reason,
            "active_objective": self._tree.active.to_dict(),
            "objective_tree": self._tree.to_dict(),
            "observation": {
                "level": frame.level,
                "step": frame.step,
                "engine_state": frame.engine_state,
                "valid_actions": list(frame.valid_actions),
                "board_hex_rows": _compact_board(frame),
            },
            "recent_transitions": _recent_transition_payload(history),
            "policy_repairs_for_active": self._policy_repairs.get(
                self._tree.active_id, 0
            ),
        }

    def _reduce(
        self,
        frame: Frame,
        history: list[HistoryEntry],
        *,
        request_deadline: float | None,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        assert self._tree is not None

        def validate_reduction(raw: dict[str, Any]) -> ReductionProposal:
            proposal = ReductionProposal.from_payload(raw)
            probe = ObjectiveTree.from_dict(self._tree.to_dict())
            probe.apply_proposal(proposal, remaining_level_actions=32)
            return proposal

        proposal = self._structured_role_call(
            role="reducer",
            system_prompt=_REDUCER_SYSTEM_PROMPT,
            user_payload=self._reducer_payload(frame, history),
            tool=_REDUCTION_TOOL,
            validator=validate_reduction,
            request_deadline=request_deadline,
            should_stop=should_stop,
        )
        previous_id = self._tree.active_id
        active = self._tree.apply_proposal(proposal, remaining_level_actions=32)
        if proposal.verdict.value == "complete":
            self._orchestration_metrics["objectives_completed"] = (
                int(self._orchestration_metrics.get("objectives_completed", 0)) + 1
            )
        elif proposal.verdict.value == "fail":
            self._orchestration_metrics["objectives_failed"] = (
                int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
            )
        self._last_reduction_step = frame.step
        self._reduction_required = False
        self._emit_event(
            "objective_reduced",
            previous_objective_id=previous_id,
            active_objective_id=active.objective_id,
            verdict=proposal.verdict.value,
            rationale=proposal.rationale,
        )

    def _policy_payload(
        self, frame: Frame, history: list[HistoryEntry], repair: str
    ) -> dict[str, Any]:
        assert self._tree is not None
        return {
            "active_objective": self._tree.active.to_dict(),
            "observation": {
                "level": frame.level,
                "step": frame.step,
                "engine_state": frame.engine_state,
                "valid_actions": list(frame.valid_actions),
                "board_hex_rows": _compact_board(frame),
            },
            "recent_transitions": _recent_transition_payload(history),
            "requested_backend": os.environ.get("LOCAL_GAMEPLAY_POLICY_BACKEND", "cpu"),
            "repair_reason": repair,
        }

    def _policy_validator(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._tree is None:
            raise PolicyRuntimeError("objective tree is unavailable")
        objective_id = str(raw.get("objective_id") or "")
        if objective_id != self._tree.active_id:
            raise PolicyRuntimeError(
                f"policy targets {objective_id!r}; active objective is {self._tree.active_id!r}",
                category="policy_verification",
            )
        source = str(raw.get("source") or "")
        source_hash = verify_policy_source(source)
        capabilities = raw.get("backend_capabilities")
        normalized_capabilities = (
            [str(item).strip().lower() for item in capabilities]
            if isinstance(capabilities, list)
            else []
        )
        if (
            "cpu" not in normalized_capabilities
            or len(normalized_capabilities) != len(set(normalized_capabilities))
            or any(item not in {"cpu", "cuda"} for item in normalized_capabilities)
        ):
            raise PolicyRuntimeError(
                "backend_capabilities must contain cpu and optional cuda once each",
                category="policy_verification",
            )
        return {**raw, "source": source, "source_hash": source_hash}

    def _save_policy_artifact(self, source: str, source_hash: str) -> str:
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None or self._tree is None:
            return ""
        directory = runtime_path.parent / "policies"
        directory.mkdir(parents=True, exist_ok=True)
        safe_objective = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._tree.active_id)
        path = directory / f"{safe_objective}-{source_hash}.py"
        if not path.exists():
            path.write_text(source, encoding="utf-8")
        return str(path.relative_to(runtime_path.parent))

    def _activate_policy(
        self,
        frame: Frame,
        history: list[HistoryEntry],
        *,
        request_deadline: float | None,
        should_stop: Callable[[], bool] | None,
        repair_reason: str = "",
    ) -> None:
        assert self._tree is not None
        if self._tree.active.kind is not ObjectiveKind.TACTICAL:
            raise ObjectiveError(
                "a gameplay policy requires an active tactical objective"
            )
        raw = self._structured_role_call(
            role="coder",
            system_prompt=_CODER_SYSTEM_PROMPT,
            user_payload=self._policy_payload(frame, history, repair_reason),
            tool=_POLICY_TOOL,
            validator=self._policy_validator,
            request_deadline=request_deadline,
            should_stop=should_stop,
        )
        source = str(raw["source"])
        source_hash = str(raw["source_hash"])
        runtime = GameplayPolicyRuntime()
        try:
            activation = runtime.activate(
                source,
                context={
                    "game_id": self._knowledge_game_id,
                    "objective": self._tree.active.to_dict(),
                },
            )
        except PolicyRuntimeError:
            runtime.close()
            raise
        if activation.source_hash != source_hash:
            runtime.close()
            raise PolicyRuntimeError(
                "activated policy fingerprint does not match verified source",
                category="policy_protocol",
            )
        if self._policy_runtime is not None:
            self._policy_runtime.close()
        self._policy_runtime = runtime
        self._policy_objective_id = self._tree.active_id
        self._policy_source_hash = source_hash
        self._policy_artifact = self._save_policy_artifact(source, source_hash)
        self._policy_memory = runtime.memory
        self._orchestration_metrics["policy_activations"] = (
            int(self._orchestration_metrics.get("policy_activations", 0)) + 1
        )
        if activation.backend_fallback_reason:
            self._orchestration_metrics["cuda_fallbacks"] = (
                int(self._orchestration_metrics.get("cuda_fallbacks", 0)) + 1
            )
        self._emit_event(
            "policy_activated",
            objective_id=self._policy_objective_id,
            source_hash=source_hash,
            artifact=self._policy_artifact,
            backend=activation.backend,
            backend_fallback_reason=activation.backend_fallback_reason,
        )

    def _invalidate_policy(
        self, reason: str, *, require_reduction: bool = False
    ) -> None:
        if self._policy_runtime is not None:
            self._policy_memory = self._policy_runtime.memory
            self._policy_runtime.close()
        self._policy_runtime = None
        self._policy_objective_id = ""
        self._boundary_reason = reason
        self._reduction_required = self._reduction_required or require_reduction

    def _restore_policy_if_possible(self) -> bool:
        if (
            self._policy_runtime is not None
            or self._tree is None
            or self._policy_objective_id != self._tree.active_id
            or not self._policy_artifact
        ):
            return self._policy_runtime is not None
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None:
            return False
        artifact = (runtime_path.parent / self._policy_artifact).resolve()
        policy_root = (runtime_path.parent / "policies").resolve()
        try:
            artifact.relative_to(policy_root)
        except ValueError:
            return False
        if not artifact.is_file():
            return False
        source = artifact.read_text(encoding="utf-8")
        if verify_policy_source(source) != self._policy_source_hash:
            return False
        runtime = GameplayPolicyRuntime()
        activation = runtime.activate(
            source,
            context={
                "game_id": self._knowledge_game_id,
                "objective": self._tree.active.to_dict(),
            },
        )
        runtime.set_memory(self._policy_memory)
        self._policy_runtime = runtime
        self._emit_event(
            "policy_resumed",
            objective_id=self._policy_objective_id,
            source_hash=self._policy_source_hash,
            backend=activation.backend,
        )
        return True

    def _observation(self, frame: Frame) -> PolicyObservation:
        if (
            self._tree is None
            or self._policy_runtime is None
            or self._policy_runtime.activation is None
        ):
            raise PolicyRuntimeError(
                "policy observation requested without active runtime"
            )
        return PolicyObservation(
            board=np.asarray(frame.grid, dtype=np.uint8),
            level=frame.level,
            step=frame.step,
            valid_actions=tuple(frame.valid_actions),
            last_transition=self._last_transition,
            objective=self._tree.active.to_dict(),
            recent_transitions=tuple(self._recent_transitions[-8:]),
            backend=self._policy_runtime.activation.backend,
        )

    def _stagnated(self, frame: Frame, history: list[HistoryEntry]) -> bool:
        if frame.step - self._last_reduction_step < 6 or len(history) < 6:
            return False
        recent = history[-6:]
        return all(entry.frame.grid == recent[0].frame.grid for entry in recent[1:])

    def _ensure_tree(self, frame: Frame) -> None:
        if self._tree is None:
            self._tree = ObjectiveTree.start_game(
                self._knowledge_game_id or "unknown",
                level=frame.level,
                level_action_budget=32,
            )
            self._emit_event(
                "objective_created",
                root_id=self._tree.root_id,
                active_objective_id=self._tree.active_id,
            )
            self._boundary_reason = "game_start"
            return
        if frame.level != self._tree.current_level:
            self._invalidate_policy("level_transition", require_reduction=True)
            level = self._tree.start_level(frame.level, level_action_budget=32)
            self._emit_event(
                "level_objective_created",
                active_objective_id=level.objective_id,
                level=frame.level,
            )

    def _policy_failure(self, exc: PolicyRuntimeError) -> None:
        assert self._tree is not None
        self._consecutive_activation_failures += 1
        objective_id = self._tree.active_id
        repairs = self._policy_repairs.get(objective_id, 0) + 1
        self._policy_repairs[objective_id] = repairs
        self._orchestration_metrics["policy_repairs"] = (
            int(self._orchestration_metrics.get("policy_repairs", 0)) + 1
        )
        self._emit_event(
            "policy_failed",
            objective_id=objective_id,
            category=exc.category,
            detail=str(exc),
            repair=repairs,
        )
        self._invalidate_policy(f"{exc.category}: {exc}")
        self._reduction_required = True
        if repairs > _MAX_POLICY_REPAIRS:
            self._tree.fail_active_tactical(str(exc))
            self._orchestration_metrics["objectives_failed"] = (
                int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
            )
            self._boundary_reason = f"policy_repairs_exhausted:{objective_id}"
        if self._consecutive_activation_failures >= _MAX_CONSECUTIVE_POLICY_FAILURES:
            raise OrchestrationFailure(
                "the initial policy and two replacements failed before executing an action",
                category="orchestration_policy_activation_exhausted",
            ) from exc

    def analyze(
        self,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        transcript_path: Path | None = None,
        analysis_step: int | None = None,
        transcript_updated: Callable[[str], None] | None = None,
        request_timeout_seconds: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> AnalyzerTurnResult:
        del transcript_updated
        if not state_path.exists():
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category="state_missing",
                failure_detail=f"Runtime state does not exist: {state_path}",
            )
        previous_session = self._session_runtime_dir
        self._ensure_session(state_path)
        if previous_session != self._session_runtime_dir and self._tree is None:
            self._boundary_reason = "game_start"
        self._current_transcript_path = transcript_path or state_path.with_name(
            f"{state_path.stem}_analyzer.txt"
        )
        frame, history = load_runtime_state(state_path)
        if frame is None:
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category="state_missing",
                failure_detail="Runtime state contains no current frame.",
            )
        if valid_actions:
            frame = Frame(
                grid=frame.grid,
                step=frame.step,
                level=frame.level,
                valid_actions=tuple(
                    sorted({str(item).upper() for item in valid_actions})
                ),
                engine_state=frame.engine_state,
                score=frame.score,
            )
        if step_env is None:
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category="orchestration_missing_step_env",
                failure_detail="Orchestrated gameplay requires the guarded step_env callback.",
            )
        started = time.monotonic()
        hard_limit = request_timeout_seconds
        if hard_limit is None:
            hard_limit = self._timeout
        request_deadline = (
            started + hard_limit if hard_limit is not None and hard_limit > 0 else None
        )
        turn_deadline = (
            started + self._yield_seconds
            if self._yield_seconds is not None and self._yield_seconds > 0
            else None
        )
        self._turn_efficiency_metrics = {}
        self._should_stop_callback = should_stop
        self._append_transcript(
            "ORCHESTRATED TURN",
            {
                "analysis_step": analysis_step,
                "action_num": action_num,
                "boundary_reason": self._boundary_reason,
            },
        )
        try:
            self._ensure_tree(frame)
            assert self._tree is not None
            if frame.engine_state in {"WIN", "GAME_OVER"}:
                self._tree.resolve_game(
                    won=frame.engine_state == "WIN",
                    evidence=f"engine state {frame.engine_state}",
                )
                self._persist_durable_state()
                return AnalyzerTurnResult(
                    step_executed=False,
                    reasoning=f"Engine resolved game as {frame.engine_state}.",
                    efficiency_metrics=dict(self._orchestration_metrics),
                )
            if self._stagnated(frame, history) and self._policy_runtime is not None:
                self._invalidate_policy(
                    "controller_stagnation_window", require_reduction=True
                )
            with contextlib.suppress(PolicyRuntimeError):
                self._restore_policy_if_possible()

            no_action_boundaries = 0
            while no_action_boundaries < _MAX_NO_ACTION_BOUNDARIES:
                remaining = self._remaining_seconds(turn_deadline)
                if remaining is not None and remaining <= 0:
                    raise OrchestrationYield("turn time budget")
                if (
                    self._tree.active.kind is not ObjectiveKind.TACTICAL
                    or self._reduction_required
                ):
                    self._reduce(
                        frame,
                        history,
                        request_deadline=request_deadline,
                        should_stop=should_stop,
                    )
                    no_action_boundaries += 1
                    continue
                if self._tree.active.remaining_actions <= 0:
                    self._tree.fail_active_tactical("tactical action budget exhausted")
                    self._invalidate_policy("tactical_action_budget_exhausted")
                    no_action_boundaries += 1
                    continue
                if (
                    self._policy_runtime is None
                    or self._policy_objective_id != self._tree.active_id
                ):
                    repair_reason = self._boundary_reason
                    try:
                        self._activate_policy(
                            frame,
                            history,
                            request_deadline=request_deadline,
                            should_stop=should_stop,
                            repair_reason=repair_reason,
                        )
                    except PolicyRuntimeError as exc:
                        self._policy_failure(exc)
                        no_action_boundaries += 1
                        continue
                    remaining = self._remaining_seconds(turn_deadline)
                    if remaining is not None and remaining <= 0:
                        raise OrchestrationYield("turn time budget")
                assert self._policy_runtime is not None
                try:
                    policy_started = time.monotonic()
                    decision: PolicyDecision = self._policy_runtime.decide(
                        self._observation(frame)
                    )
                    policy_seconds = time.monotonic() - policy_started
                    backend = (
                        self._policy_runtime.activation.backend
                        if self._policy_runtime.activation is not None
                        else "cpu"
                    )
                    metric = f"{backend}_policy_seconds"
                    self._orchestration_metrics[metric] = (
                        float(self._orchestration_metrics.get(metric, 0.0))
                        + policy_seconds
                    )
                    self._policy_memory = self._policy_runtime.memory
                except PolicyRuntimeError as exc:
                    self._policy_failure(exc)
                    no_action_boundaries += 1
                    continue

                if decision.status is PolicyStatus.SUBGOAL_SUCCEEDED:
                    objective_id = self._tree.active_id
                    self._tree.complete_active_tactical(decision.evidence)
                    self._orchestration_metrics["objectives_completed"] = (
                        int(self._orchestration_metrics.get("objectives_completed", 0))
                        + 1
                    )
                    self._invalidate_policy(f"subgoal_succeeded:{objective_id}")
                    self._emit_event(
                        "objective_completed",
                        objective_id=objective_id,
                        evidence=decision.evidence,
                    )
                    no_action_boundaries += 1
                    continue
                if decision.status is PolicyStatus.SUBGOAL_FAILED:
                    objective_id = self._tree.active_id
                    self._tree.fail_active_tactical(decision.evidence)
                    self._orchestration_metrics["objectives_failed"] = (
                        int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
                    )
                    self._invalidate_policy(f"subgoal_failed:{objective_id}")
                    self._emit_event(
                        "objective_failed",
                        objective_id=objective_id,
                        evidence=decision.evidence,
                    )
                    no_action_boundaries += 1
                    continue

                action_payload = dict(decision.action or {})
                action_payload["strategy_prediction"] = {
                    "objective_id": self._tree.active_id,
                    "objective_title": self._tree.active.title,
                    "policy_hash": self._policy_source_hash,
                    "policy_backend": self._policy_runtime.activation.backend
                    if self._policy_runtime.activation is not None
                    else "cpu",
                    "evidence": decision.evidence,
                    **(decision.prediction or {}),
                }
                result = step_env(action_payload)
                transition = {
                    "action": action_payload.get("action"),
                    "row": action_payload.get("row"),
                    "col": action_payload.get("col"),
                    "executed": bool(result.get("executed")),
                    "board_changed": bool(result.get("board_changed")),
                    "level_completed": bool(result.get("level_completed")),
                    "run_complete": bool(result.get("run_complete")),
                    "game_over": bool(result.get("game_over")),
                    "stop_reason": str(result.get("stop_reason") or ""),
                    "error": str(result.get("error") or ""),
                    "objective_id": self._tree.active_id,
                    "policy_hash": self._policy_source_hash,
                }
                self._last_transition = transition
                self._recent_transitions.append(transition)
                del self._recent_transitions[:-8]
                self._emit_event("gameplay_decision", **transition)
                if result.get("executed"):
                    self._tree.record_action()
                    self._consecutive_activation_failures = 0
                    self._orchestration_metrics["policy_steps"] = (
                        int(self._orchestration_metrics.get("policy_steps", 0)) + 1
                    )
                    if result.get("run_complete"):
                        self._tree.resolve_game(
                            won=True,
                            evidence="guarded policy action produced engine run_complete",
                        )
                    if result.get("level_completed") or result.get("run_complete"):
                        self._invalidate_policy(
                            "run_complete"
                            if result.get("run_complete")
                            else "level_completed"
                        )
                    self._persist_durable_state()
                    return AnalyzerTurnResult(
                        step_executed=True,
                        reasoning=decision.evidence,
                        efficiency_metrics=dict(self._orchestration_metrics),
                        attempts=1,
                    )
                self._policy_failure(
                    PolicyRuntimeError(
                        transition["error"]
                        or transition["stop_reason"]
                        or "guarded action was not executed",
                        category="guarded_action",
                    )
                )
                no_action_boundaries += 1

            raise OrchestrationFailure(
                "orchestration crossed too many no-action objective boundaries in one turn",
                category="orchestration_no_action_exhausted",
            )
        except OrchestrationYield as exc:
            self._persist_durable_state()
            return AnalyzerTurnResult(
                step_executed=False,
                yielded_control=True,
                yield_reason="turn_time_budget",
                reasoning=str(exc),
                efficiency_metrics=dict(self._orchestration_metrics),
            )
        except requests.RequestException as exc:
            self._persist_durable_state()
            return AnalyzerTurnResult(
                step_executed=False,
                retryable_failure=True,
                failure_category="orchestration_model_transport",
                failure_detail=str(exc),
                efficiency_metrics=dict(self._orchestration_metrics),
            )
        except (OrchestrationFailure, ObjectiveError) as exc:
            category = getattr(exc, "category", "orchestration_objective_error")
            self._emit_event(
                "orchestration_exhausted", category=category, detail=str(exc)
            )
            self._persist_durable_state()
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category=category,
                failure_detail=str(exc),
                efficiency_metrics=dict(self._orchestration_metrics),
            )
        except Exception as exc:  # noqa: BLE001 - convert to visible analyzer failure
            log.exception("orchestrated objective analyzer failed")
            self._persist_durable_state()
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category="orchestration_internal",
                failure_detail=f"{type(exc).__name__}: {exc}",
                efficiency_metrics=dict(self._orchestration_metrics),
            )
        finally:
            self._should_stop_callback = None
