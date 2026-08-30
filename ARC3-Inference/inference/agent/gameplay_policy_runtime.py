"""Persistent, isolated runtime for LLM-generated gameplay policies."""

from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing
import os
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Any, Literal

import numpy as np

from inference.agent.policy_codegen_helpers import POLICY_CODEGEN_GLOBALS
from inference.agent.policy_pathfinding import POLICY_PATHFINDING_GLOBALS


POLICY_API_VERSION = 1
MAX_POLICY_SOURCE_BYTES = 65_536
MAX_POLICY_AST_NODES = 5_000
ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "numpy",
    "statistics",
    "torch",
}
FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "hasattr",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
FORBIDDEN_ATTRIBUTES = {
    "dump",
    "dumps",
    "fromfile",
    "load",
    "loads",
    "save",
    "savez",
    "savez_compressed",
    "tofile",
}
FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.Yield,
    ast.YieldFrom,
)
POLICY_OBSERVATION_ATTRIBUTES = {
    "backend",
    "board",
    "level",
    "last_transition",
    "objective",
    "recent_transitions",
    "step",
    "valid_actions",
}


class PolicyRuntimeError(RuntimeError):
    def __init__(self, message: str, *, category: str = "policy_runtime") -> None:
        super().__init__(message)
        self.category = category


class PolicyStatus(StrEnum):
    CONTINUE = "continue"
    SUBGOAL_SUCCEEDED = "subgoal_succeeded"
    SUBGOAL_FAILED = "subgoal_failed"


@dataclass(frozen=True)
class PolicyObservation:
    board: np.ndarray
    level: int
    step: int
    valid_actions: tuple[str, ...]
    last_transition: dict[str, Any] | None
    objective: dict[str, Any]
    recent_transitions: tuple[dict[str, Any], ...]
    backend: Literal["cpu", "cuda"]

    def __post_init__(self) -> None:
        board = np.asarray(self.board, dtype=np.uint8)
        if board.shape != (64, 64):
            raise ValueError(
                f"policy board must have shape (64, 64), got {board.shape}"
            )
        board = np.array(board, dtype=np.uint8, copy=True)
        board.setflags(write=False)
        object.__setattr__(self, "board", board)


@dataclass(frozen=True)
class PolicyDecision:
    status: PolicyStatus
    action: dict[str, Any] | None
    memory: Any
    evidence: str = ""
    prediction: dict[str, Any] | None = None

    @classmethod
    def from_payload(
        cls, payload: Any, *, valid_actions: tuple[str, ...]
    ) -> "PolicyDecision":
        if isinstance(payload, cls):
            try:
                normalized_status = PolicyStatus(str(payload.status))
            except ValueError as exc:
                raise PolicyRuntimeError(
                    "policy returned an invalid status", category="invalid_decision"
                ) from exc
            if payload.action is not None and not isinstance(payload.action, dict):
                raise PolicyRuntimeError(
                    "policy action must be a mapping", category="invalid_decision"
                )
            if payload.prediction is not None and not isinstance(
                payload.prediction, dict
            ):
                raise PolicyRuntimeError(
                    "policy prediction must be a mapping",
                    category="invalid_decision",
                )
            candidate = cls(
                status=normalized_status,
                action=(
                    dict(payload.action) if isinstance(payload.action, dict) else None
                ),
                memory=payload.memory,
                evidence=str(payload.evidence or "")[:2400],
                prediction=(
                    dict(payload.prediction)
                    if isinstance(payload.prediction, dict)
                    else None
                ),
            )
        elif isinstance(payload, dict):
            try:
                status = PolicyStatus(str(payload.get("status") or "continue"))
            except ValueError as exc:
                raise PolicyRuntimeError(
                    "policy returned an invalid status", category="invalid_decision"
                ) from exc
            action = payload.get("action")
            if action is not None and not isinstance(action, dict):
                raise PolicyRuntimeError(
                    "policy action must be a mapping", category="invalid_decision"
                )
            prediction = payload.get("prediction")
            if prediction is not None and not isinstance(prediction, dict):
                raise PolicyRuntimeError(
                    "policy prediction must be a mapping",
                    category="invalid_decision",
                )
            candidate = cls(
                status=status,
                action=dict(action) if isinstance(action, dict) else None,
                memory=payload.get("memory", {}),
                evidence=str(payload.get("evidence") or "")[:2400],
                prediction=dict(prediction) if isinstance(prediction, dict) else None,
            )
        else:
            raise PolicyRuntimeError(
                "policy decide() must return PolicyDecision or a mapping",
                category="invalid_decision",
            )
        _ensure_json(candidate.memory, "policy memory")
        if candidate.prediction is not None:
            _ensure_json(candidate.prediction, "policy prediction")
        if candidate.status is PolicyStatus.CONTINUE:
            if candidate.action is None:
                raise PolicyRuntimeError(
                    "continue requires exactly one action", category="invalid_decision"
                )
            action = _normalize_action(candidate.action, valid_actions)
            return cls(
                status=candidate.status,
                action=action,
                memory=candidate.memory,
                evidence=candidate.evidence,
                prediction=candidate.prediction,
            )
        if candidate.action is not None:
            raise PolicyRuntimeError(
                "terminal subgoal decisions may not include an action",
                category="invalid_decision",
            )
        return candidate


@dataclass(frozen=True)
class PolicyActivation:
    source_hash: str
    backend: Literal["cpu", "cuda"]
    supported_backends: tuple[str, ...]
    backend_fallback_reason: str = ""


def _ensure_json(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PolicyRuntimeError(
            f"{label} must be finite JSON data", category="invalid_decision"
        ) from exc


def _is_observation_board(node: ast.AST, aliases: set[str]) -> bool:
    return (isinstance(node, ast.Name) and node.id in aliases) or (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "observation"
        and node.attr == "board"
    )


def _truth_tests_board(node: ast.AST, aliases: set[str]) -> bool:
    if _is_observation_board(node, aliases):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _truth_tests_board(node.operand, aliases)
    if isinstance(node, ast.BoolOp):
        return any(_truth_tests_board(value, aliases) for value in node.values)
    return False


def _normalize_action(
    payload: dict[str, Any], valid_actions: tuple[str, ...]
) -> dict[str, Any]:
    raw_name = str(payload.get("action") or "").strip().upper()
    if not raw_name or raw_name not in valid_actions:
        raise PolicyRuntimeError(
            f"policy action {raw_name!r} is not currently valid",
            category="invalid_action",
        )
    action: dict[str, Any] = {"action": raw_name}
    has_coordinates = "row" in payload or "col" in payload
    if raw_name == "MOUSE" or has_coordinates:
        try:
            row = int(payload["row"])
            col = int(payload["col"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyRuntimeError(
                f"{raw_name} coordinates require integer row and col",
                category="invalid_action",
            ) from exc
        if not 0 <= row <= 63 or not 0 <= col <= 63:
            raise PolicyRuntimeError(
                f"{raw_name} row and col must be between 0 and 63",
                category="invalid_action",
            )
        action.update(row=row, col=col)
    return action


def verify_policy_source(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise PolicyRuntimeError(
            "policy source is empty", category="policy_verification"
        )
    if len(source.encode("utf-8")) > MAX_POLICY_SOURCE_BYTES:
        raise PolicyRuntimeError(
            "policy source exceeds 65536 bytes", category="policy_verification"
        )
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise PolicyRuntimeError(
            f"policy syntax error: {exc}", category="policy_verification"
        ) from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_POLICY_AST_NODES:
        raise PolicyRuntimeError(
            "policy AST is too large", category="policy_verification"
        )
    errors: list[str] = []

    def reject(message: str) -> None:
        if message not in errors:
            errors.append(message)

    board_aliases: set[str] = set()
    for node in nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None or not _is_observation_board(value, board_aliases):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        board_aliases.update(
            target.id for target in targets if isinstance(target, ast.Name)
        )
    for statement in tree.body:
        if not isinstance(
            statement,
            (
                ast.Assign,
                ast.AnnAssign,
                ast.Expr,
                ast.FunctionDef,
                ast.Import,
                ast.ImportFrom,
            ),
        ):
            reject(f"top-level {type(statement).__name__} is not permitted")
    for node in nodes:
        if isinstance(node, FORBIDDEN_NODES):
            reject(f"{type(node).__name__} is not permitted in policy code")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            for name in names:
                root = name.split(".", 1)[0]
                if root not in ALLOWED_IMPORTS:
                    reject(f"import {name!r} is not permitted")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            reject(f"dunder name {node.id!r} is not permitted")
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "observation"
                and node.attr not in POLICY_OBSERVATION_ATTRIBUTES
            ):
                reject(f"PolicyObservation has no attribute {node.attr!r}")
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
                reject(f"attribute {node.attr!r} is not permitted")
        if isinstance(node, (ast.If, ast.While, ast.IfExp)) and _truth_tests_board(
            node.test, board_aliases
        ):
            reject(
                "observation.board cannot be used as a boolean; use board.size or "
                "an explicit NumPy reduction"
            )
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(_is_observation_board(item, board_aliases) for item in operands):
                if any(
                    isinstance(item, ast.Constant) and item.value is None
                    for item in operands
                ):
                    reject(
                        "PolicyObservation.board is always a non-None uint8[64,64] "
                        "NumPy array; delete board is None / board is not None checks "
                        "and use board.shape or board.size directly"
                    )
                else:
                    reject(
                        "observation.board cannot be compared directly; use "
                        "np.array_equal or compare a finite scalar digest"
                    )
        if isinstance(node, ast.Dict):
            for value in node.values:
                if _is_observation_board(value, board_aliases):
                    reject(
                        "observation.board cannot be stored in a mapping; policy memory "
                        "must contain finite JSON data"
                    )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                reject(f"call {node.func.id!r} is not permitted")
    top_level_functions = {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef)
    }
    decide = top_level_functions.get("decide")
    if decide is None:
        reject("policy must define decide(observation, memory) as a top-level function")
    else:
        decide_arguments = [*decide.args.posonlyargs, *decide.args.args]
        if (
            [argument.arg for argument in decide_arguments] != ["observation", "memory"]
            or decide.args.vararg is not None
            or decide.args.kwarg is not None
            or decide.args.kwonlyargs
            or decide.args.defaults
            or decide.args.kw_defaults
        ):
            reject(
                "policy decide signature must be exactly decide(observation, memory)"
            )
    initialize = top_level_functions.get("initialize")
    if initialize is not None:
        initialize_arguments = [
            *initialize.args.posonlyargs,
            *initialize.args.args,
        ]
        if (
            [argument.arg for argument in initialize_arguments] != ["context"]
            or initialize.args.vararg is not None
            or initialize.args.kwarg is not None
            or initialize.args.kwonlyargs
            or initialize.args.defaults
            or initialize.args.kw_defaults
        ):
            reject("policy initialize signature must be exactly initialize(context)")
    if errors:
        raise PolicyRuntimeError(
            "policy verification failed:\n- " + "\n- ".join(errors),
            category="policy_verification",
        )
    material = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()


def choose_policy_backend(
    requested: str,
    supported_backends: tuple[str, ...],
    *,
    min_free_mb: int,
    torch_module: Any | None = None,
) -> tuple[Literal["cpu", "cuda"], str]:
    requested = str(requested or "cpu").strip().lower()
    if requested not in {"cpu", "auto", "cuda"}:
        raise PolicyRuntimeError(
            f"unknown gameplay backend {requested!r}", category="backend_unavailable"
        )
    supported = {str(item).strip().lower() for item in supported_backends}
    if "cpu" not in supported:
        raise PolicyRuntimeError(
            "policies must declare CPU support", category="backend_unavailable"
        )
    if requested == "cpu":
        return "cpu", ""
    if "cuda" not in supported:
        if requested == "cuda":
            raise PolicyRuntimeError(
                "policy does not declare CUDA support", category="backend_unavailable"
            )
        return "cpu", "policy_does_not_support_cuda"
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            if requested == "cuda":
                raise PolicyRuntimeError(
                    "PyTorch CUDA is unavailable", category="backend_unavailable"
                ) from exc
            return "cpu", "torch_unavailable"
    try:
        cuda_available = bool(torch_module.cuda.is_available())
        free_bytes, _total_bytes = torch_module.cuda.mem_get_info()
        free_mb = int(free_bytes) // (1024 * 1024)
    except Exception as exc:  # noqa: BLE001 - optional CUDA probing
        if requested == "cuda":
            raise PolicyRuntimeError(
                f"CUDA probe failed: {exc}", category="backend_unavailable"
            ) from exc
        return "cpu", f"cuda_probe_failed:{type(exc).__name__}"
    if not cuda_available:
        if requested == "cuda":
            raise PolicyRuntimeError(
                "CUDA is not available", category="backend_unavailable"
            )
        return "cpu", "cuda_unavailable"
    if free_mb < max(0, int(min_free_mb)):
        if requested == "cuda":
            raise PolicyRuntimeError(
                f"CUDA has {free_mb} MiB free; {min_free_mb} MiB required",
                category="backend_unavailable",
            )
        return "cpu", f"cuda_headroom:{free_mb}<{min_free_mb}"
    return "cuda", ""


def _safe_import_factory(allow_torch: bool):
    real_import = __import__

    def safe_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".", 1)[0]
        if root not in ALLOWED_IMPORTS or (root == "torch" and not allow_torch):
            raise ImportError(f"policy import {name!r} is not permitted")
        return real_import(name, globals_, locals_, fromlist, level)

    return safe_import


def _safe_builtins(allow_torch: bool) -> dict[str, Any]:
    return {
        "__import__": _safe_import_factory(allow_torch),
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "iter": iter,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "next": next,
        "ord": ord,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "ValueError": ValueError,
        "RuntimeError": RuntimeError,
        "zip": zip,
    }


def _decision_to_wire(decision: PolicyDecision) -> dict[str, Any]:
    return {
        "status": decision.status.value,
        "action": decision.action,
        "memory": decision.memory,
        "evidence": decision.evidence,
        "prediction": decision.prediction,
    }


def _policy_worker_main(connection: Connection) -> None:
    namespace: dict[str, Any] = {}
    decide: Any = None
    backend: Literal["cpu", "cuda"] = "cpu"
    try:
        while True:
            request = connection.recv()
            command = request.get("command") if isinstance(request, dict) else None
            if command == "close":
                return
            if command == "load":
                source = str(request.get("source") or "")
                source_hash = verify_policy_source(source)
                allow_torch = str(request.get("requested_backend") or "cpu") != "cpu"
                namespace = {
                    "__builtins__": _safe_builtins(allow_torch),
                    "__name__": "generated_gameplay_policy",
                    "PolicyDecision": PolicyDecision,
                    "PolicyStatus": PolicyStatus,
                    "PolicyObservation": PolicyObservation,
                    "np": np,
                }
                namespace.update(POLICY_CODEGEN_GLOBALS)
                namespace.update(POLICY_PATHFINDING_GLOBALS)
                exec(compile(source, "<generated-gameplay-policy>", "exec"), namespace)
                if namespace.get("POLICY_API_VERSION") != POLICY_API_VERSION:
                    raise PolicyRuntimeError(
                        f"POLICY_API_VERSION must be {POLICY_API_VERSION}",
                        category="policy_verification",
                    )
                raw_supported = namespace.get("SUPPORTED_BACKENDS", ("cpu",))
                if not isinstance(raw_supported, (list, tuple)):
                    raise PolicyRuntimeError(
                        "SUPPORTED_BACKENDS must be a list or tuple",
                        category="policy_verification",
                    )
                supported = tuple(str(item).lower() for item in raw_supported)
                if (
                    not supported
                    or len(supported) != len(set(supported))
                    or any(item not in {"cpu", "cuda"} for item in supported)
                ):
                    raise PolicyRuntimeError(
                        "SUPPORTED_BACKENDS may contain cpu and optional cuda once each",
                        category="policy_verification",
                    )
                backend, fallback = choose_policy_backend(
                    str(request.get("requested_backend") or "cpu"),
                    supported,
                    min_free_mb=int(request.get("min_free_mb", 4096) or 4096),
                )
                decide = namespace.get("decide")
                if not callable(decide):
                    raise PolicyRuntimeError(
                        "policy must define decide(observation, memory)",
                        category="policy_verification",
                    )
                initialize = namespace.get("initialize")
                memory: Any = {}
                if callable(initialize):
                    context = dict(request.get("context") or {})
                    context["backend"] = backend
                    memory = initialize(context)
                    _ensure_json(memory, "policy initial memory")
                self_test = namespace.get("self_test")
                if callable(self_test):
                    result = self_test()
                    if result is False:
                        raise PolicyRuntimeError(
                            "policy self_test returned false",
                            category="policy_verification",
                        )
                connection.send(
                    {
                        "ok": True,
                        "source_hash": source_hash,
                        "backend": backend,
                        "supported_backends": supported,
                        "backend_fallback_reason": fallback,
                        "memory": memory,
                    }
                )
                continue
            if command == "decide":
                if not callable(decide):
                    raise PolicyRuntimeError(
                        "no policy is active", category="policy_runtime"
                    )
                raw = dict(request.get("observation") or {})
                observation = PolicyObservation(
                    board=np.asarray(raw.get("board"), dtype=np.uint8),
                    level=max(1, int(raw.get("level", 1) or 1)),
                    step=max(0, int(raw.get("step", 0) or 0)),
                    valid_actions=tuple(
                        str(item) for item in raw.get("valid_actions") or ()
                    ),
                    last_transition=(
                        dict(raw["last_transition"])
                        if isinstance(raw.get("last_transition"), dict)
                        else None
                    ),
                    objective=dict(raw.get("objective") or {}),
                    recent_transitions=tuple(
                        dict(item)
                        for item in raw.get("recent_transitions") or ()
                        if isinstance(item, dict)
                    ),
                    backend=backend,
                )
                decision = PolicyDecision.from_payload(
                    decide(observation, request.get("memory", {})),
                    valid_actions=observation.valid_actions,
                )
                connection.send({"ok": True, "decision": _decision_to_wire(decision)})
                continue
            raise PolicyRuntimeError(
                f"unknown worker command {command!r}", category="policy_protocol"
            )
    except BaseException as exc:  # noqa: BLE001 - report and terminate worker
        category = getattr(exc, "category", "policy_runtime")
        message = f"{type(exc).__name__}: {exc}"
        if "out of memory" in message.lower():
            category = "cuda_oom"
        try:
            connection.send({"ok": False, "category": category, "error": message})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class GameplayPolicyRuntime:
    """One generated-policy subprocess whose JSON memory is owned by the host."""

    def __init__(
        self,
        *,
        requested_backend: str | None = None,
        cuda_min_free_mb: int | None = None,
        activation_timeout_seconds: float = 30.0,
        decision_timeout_seconds: float | None = None,
    ) -> None:
        self.requested_backend = (
            str(
                requested_backend
                or os.environ.get("LOCAL_GAMEPLAY_POLICY_BACKEND", "cpu")
            )
            .strip()
            .lower()
        )
        self.cuda_min_free_mb = int(
            cuda_min_free_mb
            if cuda_min_free_mb is not None
            else os.environ.get("LOCAL_GAMEPLAY_POLICY_CUDA_MIN_FREE_MB", "4096")
        )
        self.activation_timeout_seconds = max(0.1, float(activation_timeout_seconds))
        configured_decision_timeout = (
            decision_timeout_seconds
            if decision_timeout_seconds is not None
            else os.environ.get("LOCAL_GAMEPLAY_POLICY_DECISION_TIMEOUT_SECONDS", "2")
        )
        self.decision_timeout_seconds = max(0.05, float(configured_decision_timeout))
        self._process: multiprocessing.Process | None = None
        self._connection: Connection | None = None
        self._source = ""
        self._context: dict[str, Any] = {}
        self._memory: Any = {}
        self.activation: PolicyActivation | None = None

    @property
    def memory(self) -> Any:
        return self._memory

    def set_memory(self, value: Any) -> None:
        _ensure_json(value, "policy memory")
        self._memory = value

    def _start(self) -> None:
        self.close()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_policy_worker_main,
            args=(child,),
            name="objective-gameplay-policy",
            daemon=True,
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent

    def _exchange(self, request: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        connection = self._connection
        process = self._process
        if connection is None or process is None or not process.is_alive():
            raise PolicyRuntimeError(
                "policy worker is not running", category="policy_worker_exit"
            )
        try:
            connection.send(request)
            if not connection.poll(max(0.01, timeout)):
                self.close()
                raise PolicyRuntimeError(
                    f"policy worker exceeded {timeout:.2f}s timeout",
                    category="policy_timeout",
                )
            response = connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            self.close()
            raise PolicyRuntimeError(
                f"policy worker communication failed: {exc}",
                category="policy_worker_exit",
            ) from exc
        if not isinstance(response, dict) or not response.get("ok"):
            category = (
                str(response.get("category") or "policy_runtime")
                if isinstance(response, dict)
                else "policy_protocol"
            )
            error = (
                str(response.get("error") or "policy worker failed")
                if isinstance(response, dict)
                else "policy worker returned an invalid response"
            )
            self.close()
            raise PolicyRuntimeError(error, category=category)
        return response

    def activate(self, source: str, *, context: dict[str, Any]) -> PolicyActivation:
        source_hash = verify_policy_source(source)
        self._source = source
        self._context = dict(context)
        self._start()
        response = self._exchange(
            {
                "command": "load",
                "source": source,
                "context": context,
                "requested_backend": self.requested_backend,
                "min_free_mb": self.cuda_min_free_mb,
            },
            timeout=self.activation_timeout_seconds,
        )
        if str(response.get("source_hash")) != source_hash:
            self.close()
            raise PolicyRuntimeError(
                "worker policy fingerprint mismatch", category="policy_protocol"
            )
        self._memory = response.get("memory", {})
        self.activation = PolicyActivation(
            source_hash=source_hash,
            backend=str(response.get("backend") or "cpu"),  # type: ignore[arg-type]
            supported_backends=tuple(response.get("supported_backends") or ("cpu",)),
            backend_fallback_reason=str(response.get("backend_fallback_reason") or ""),
        )
        return self.activation

    def preflight(
        self,
        observation: PolicyObservation,
        *,
        minimum_actions: int = 4,
    ) -> None:
        """Exercise bounded no-progress handling, then restore a fresh activation."""

        if self.activation is None:
            raise PolicyRuntimeError("no active policy", category="policy_preflight")
        required = max(1, min(4, int(minimum_actions)))
        source = self._source
        context = dict(self._context)
        recent: list[dict[str, Any]] = []

        def signature(decision: PolicyDecision) -> tuple[Any, Any, Any]:
            action = decision.action or {}
            return action.get("action"), action.get("row"), action.get("col")

        try:
            decision = self.decide(observation)
            if decision.status is not PolicyStatus.CONTINUE:
                raise PolicyRuntimeError(
                    "policy preflight terminated before proposing its first action",
                    category="policy_preflight",
                )
            previous_signature = signature(decision)
            for evidence_count in range(1, required):
                action = dict(decision.action or {})
                transition = {
                    "action": action.get("action"),
                    "row": action.get("row"),
                    "col": action.get("col"),
                    "executed": True,
                    "post_action_observed": True,
                    "board_changed": False,
                    "reward": 0.0,
                    "score": 0,
                    "level": observation.level,
                    "engine_state": "NOT_FINISHED",
                    "outcome_class": "exact_noop",
                    "novel_state": False,
                    "decision_context_changed": False,
                    "meaningful_progress": False,
                    "loop_detected": False,
                    "cycle_risk": False,
                    "level_completed": False,
                    "run_complete": False,
                    "game_over": False,
                    "stop_reason": "",
                    "error": "",
                    "objective_id": str(
                        observation.objective.get("objective_id") or ""
                    ),
                }
                recent.append(transition)
                synthetic = PolicyObservation(
                    board=observation.board,
                    level=observation.level,
                    step=observation.step + evidence_count,
                    valid_actions=observation.valid_actions,
                    last_transition=transition,
                    objective=observation.objective,
                    recent_transitions=tuple(recent),
                    backend=observation.backend,
                )
                decision = self.decide(synthetic)
                if decision.status is not PolicyStatus.CONTINUE:
                    raise PolicyRuntimeError(
                        "policy preflight terminated after only "
                        f"{evidence_count} inconclusive action(s); the objective "
                        f"requires {required}",
                        category="policy_preflight",
                    )
                current_signature = signature(decision)
                if current_signature == previous_signature:
                    raise PolicyRuntimeError(
                        "policy preflight repeated the same action after exact_noop "
                        "instead of selecting another valid probe",
                        category="policy_preflight",
                    )
                previous_signature = current_signature
        except BaseException:
            self.close()
            raise
        # Preflight decisions must not leak memory or module globals into gameplay.
        self.activate(source, context=context)

    def decide(self, observation: PolicyObservation) -> PolicyDecision:
        if self.activation is None:
            raise PolicyRuntimeError("no active policy", category="policy_runtime")
        request = {
            "command": "decide",
            "observation": {
                "board": observation.board.tolist(),
                "level": observation.level,
                "step": observation.step,
                "valid_actions": list(observation.valid_actions),
                "last_transition": observation.last_transition,
                "objective": observation.objective,
                "recent_transitions": list(observation.recent_transitions),
            },
            "memory": self._memory,
        }
        try:
            response = self._exchange(request, timeout=self.decision_timeout_seconds)
        except PolicyRuntimeError as exc:
            if exc.category == "cuda_oom" and self.requested_backend == "auto":
                previous_memory = self._memory
                requested = self.requested_backend
                self.requested_backend = "cpu"
                try:
                    activation = self.activate(self._source, context=self._context)
                    self._memory = previous_memory
                    self.activation = PolicyActivation(
                        source_hash=activation.source_hash,
                        backend="cpu",
                        supported_backends=activation.supported_backends,
                        backend_fallback_reason="cuda_oom",
                    )
                    return self.decide(observation)
                finally:
                    self.requested_backend = requested
            raise
        decision = PolicyDecision.from_payload(
            response.get("decision"), valid_actions=observation.valid_actions
        )
        self._memory = decision.memory
        return decision

    def close(self) -> None:
        connection = self._connection
        process = self._process
        self._connection = None
        self._process = None
        self.activation = None
        if connection is not None:
            try:
                if process is not None and process.is_alive():
                    connection.send({"command": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            process.join(timeout=0.25)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    def __enter__(self) -> "GameplayPolicyRuntime":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
