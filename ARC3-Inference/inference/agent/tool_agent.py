"""Direct OpenAI-compatible tool-calling analyzer for ARC puzzle runs."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

from inference.agent.action_names import (
    MAX_ACTION_BATCH,
    to_engine_action,
    to_model_action,
)
from inference.agent.causal_model import normalize_causal_model
from inference.agent.inference_controller import (
    InferenceControllerConfig,
    action_family,
    build_experience_snapshot,
    harmful_evidence_is_decisive,
    normalize_action_key,
)
from inference.agent.prompts import (
    COMPACT_TOOL_SESSION_ADDENDUM,
    GAME_OVERVIEW_ADDENDUM,
    MULTIMODAL_CONTEXT_ADDENDUM,
    PYTHON_ADDENDUM,
    STRUCTURED_RUNTIME_STATE_ADDENDUM,
    TOOL_CALL_FORMAT_GUIDANCE,
    VISUAL_GAME_ADDENDUM,
)
from inference.agent.python_tool_sandbox import (
    SandboxHostActionError,
    _sandbox_exception_diagnostic,
    prewarm_sandbox,
    run_sandboxed_python,
)
from inference.agent.runtime_state import (
    RUNTIME_STATE_FILENAME,
    Frame,
    HistoryEntry,
    load_runtime_state,
)
from inference.agent.trial_knowledge import TrialKnowledgeStore
from inference.agent.vision_context import (
    current_grid_image_enabled,
    current_grid_image_part,
)
from inference.utils.openai_compat import build_chat_payload, build_headers

log = logging.getLogger(__name__)

_LOCAL_ANALYZER_MODEL_ID = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
_LOCAL_ANALYZER_BASE_URL = os.environ.get(
    "LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1"
)
_DEFAULT_ANALYZER_MODEL = os.environ.get(
    "INFERENCE_ANALYZER_MODEL",
    _LOCAL_ANALYZER_MODEL_ID,
)
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)
_MARKDOWN_CODE_FENCE_RE = re.compile(
    r"(?P<fence>\x60{3,}|~{3,})(?P<language>[A-Za-z0-9_+.-]*)[ \t]*\r?\n"
    r"(?P<code>.*?)(?:\r?\n)?(?P=fence)",
    flags=re.DOTALL,
)


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _contains_tool_call_markup(*chunks: str) -> bool:
    for chunk in chunks:
        lowered = chunk.lower()
        if "<tool_call" in lowered or "<function=" in lowered:
            return True
    return False


def _strip_tool_call_markup(text: str) -> str:
    if not text.strip():
        return ""
    stripped = _TOOL_CALL_BLOCK_RE.sub("", text)
    return stripped.strip()


class _GeneratedCodeLimitError(ValueError):
    """Raised when generated source exceeds a host-side parsing limit."""


class _GeneratedCodePreflightError(ValueError):
    """Raised when generated source is valid Python but unsafe to execute."""


_SANDBOX_SAFE_MODULES = frozenset(
    {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
)
_FRAME_VIEW_ATTRIBUTES = frozenset(
    {
        "analyze",
        "ascii",
        "border_summary",
        "bounds",
        "cell",
        "center_summary",
        "color_adjacency",
        "color_summary",
        "color_transitions",
        "column_profile",
        "compare_regions",
        "components",
        "corner_summary",
        "crop",
        "diff",
        "distance",
        "distance_between_colors",
        "divider_lines",
        "edge_distance",
        "enclosed_regions",
        "find",
        "find_pattern",
        "level",
        "line_between",
        "mirror_cells",
        "nearest_cell",
        "neighbors",
        "object_relations",
        "objects",
        "panels",
        "periodicity",
        "quadrant_summary",
        "ray",
        "reachable_region",
        "rectangles",
        "region_summary",
        "row_profile",
        "runs",
        "segmentation",
        "shape",
        "shortest_path",
        "shortest_path_to_any",
        "step",
        "symmetry",
        "tile_summary",
        "track_objects",
        "transform_relation",
        "translate_cells",
    }
)
_FRAME_VIEW_NAMES = frozenset(
    {"current_frame", "latest_frame", "previous_frame", "last_action_frame"}
)
_TRANSITION_VIEW_ATTRIBUTES = frozenset(
    {
        "action",
        "after_frame",
        "before_frame",
        "color_transitions",
        "diff",
        "frame",
        "result",
    }
)


def _loop_has_direct_break(loop: ast.While) -> bool:
    class BreakFinder(ast.NodeVisitor):
        found = False

        def visit_Break(self, _node: ast.Break) -> None:
            self.found = True

        def visit_For(self, _node: ast.For) -> None:
            return

        def visit_AsyncFor(self, _node: ast.AsyncFor) -> None:
            return

        def visit_While(self, _node: ast.While) -> None:
            return

        def visit_FunctionDef(self, _node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, _node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, _node: ast.ClassDef) -> None:
            return

    finder = BreakFinder()
    for statement in loop.body:
        finder.visit(statement)
    return finder.found


def _generated_python_preflight_issues(tree: ast.AST) -> list[dict[str, Any]]:
    """Return definite, side-effect-free generated-code failures.

    This deliberately reports only errors that can be proven from syntax. It
    avoids speculative lint that could reject a valid model-generated program.
    """
    issues: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            for name in names:
                if name not in _SANDBOX_SAFE_MODULES:
                    issues.append(
                        {
                            "line": getattr(node, "lineno", None),
                            "message": f"Module '{name}' is not available in the sandbox.",
                            "hint": (
                                "Use an approved module or implement the bounded operation "
                                "with the preloaded frame APIs."
                            ),
                        }
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "action"
            and (len(node.args) != 1 or node.keywords)
        ):
            issues.append(
                {
                    "line": getattr(node, "lineno", None),
                    "message": "action(...) requires exactly one positional argument.",
                    "hint": "Pass one action or one ordered action list.",
                }
            )
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in _FRAME_VIEW_NAMES
            and node.attr not in _FRAME_VIEW_ATTRIBUTES
        ):
            issues.append(
                {
                    "line": getattr(node, "lineno", None),
                    "message": (
                        f"{node.value.id} has no documented attribute '{node.attr}'."
                    ),
                    "hint": "Use a documented FrameView property or method.",
                }
            )
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "last_transition"
            and node.attr not in _TRANSITION_VIEW_ATTRIBUTES
        ):
            issues.append(
                {
                    "line": getattr(node, "lineno", None),
                    "message": (
                        f"last_transition has no documented attribute '{node.attr}'."
                    ),
                    "hint": "Use a documented TransitionView property or method.",
                }
            )
        if (
            isinstance(node, ast.While)
            and isinstance(node.test, ast.Constant)
            and node.test.value is True
            and not _loop_has_direct_break(node)
        ):
            issues.append(
                {
                    "line": getattr(node, "lineno", None),
                    "message": "Unbounded while-True loop has no break path.",
                    "hint": "Use a finite range, explicit work limit, or reachable break.",
                }
            )
    return issues


def _choice_has_executable_python_tool(choice: Any) -> bool:
    if not isinstance(choice, dict):
        return False
    message = choice.get("message")
    if not isinstance(message, dict):
        return False
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function") if isinstance(raw_call, dict) else None
        if (
            not isinstance(function, dict)
            or str(function.get("name") or "").strip() != "python"
        ):
            continue
        try:
            arguments = _normalize_tool_call_arguments(
                function.get("arguments", "{}")
            )
            code = _normalize_generated_python_code(arguments.get("code", ""))
            tree = _parse_bounded_generated_python(code)
            compile(tree, "<python_tool_candidate>", "exec")
        except (SyntaxError, TypeError, ValueError, OverflowError):
            continue
        if code and not _generated_python_preflight_issues(tree):
            return True
    return False


def _deterministic_generated_python_repair(
    code: str, diagnostic: dict[str, Any]
) -> str | None:
    """Apply one conservative identifier repair from a structured diagnostic."""
    suggestions = [
        str(item)
        for item in diagnostic.get("suggestions") or []
        if isinstance(item, str) and item.isidentifier()
    ]
    if len(suggestions) != 1:
        return None
    replacement = suggestions[0]
    error_type = str(diagnostic.get("type") or "")
    if error_type == "NameError":
        target = str(diagnostic.get("name") or "")
        pattern = rf"\b{re.escape(target)}\b" if target.isidentifier() else ""
    elif error_type == "AttributeError":
        target = str(diagnostic.get("attribute") or "")
        pattern = (
            rf"(?<=\.){re.escape(target)}\b" if target.isidentifier() else ""
        )
    else:
        return None
    if not pattern:
        return None
    repaired, replacements = re.subn(pattern, replacement, code)
    if replacements != 1 or repaired == code:
        return None
    try:
        tree = _parse_bounded_generated_python(repaired)
        compile(tree, "<python_tool_auto_repair>", "exec")
    except (SyntaxError, TypeError, ValueError, OverflowError):
        return None
    if _generated_python_preflight_issues(tree):
        return None
    return repaired


def _parse_bounded_generated_python(code: str) -> ast.AST:
    source_bytes = len(code.encode("utf-8"))
    if source_bytes > _LOCAL_ANALYZER_MAX_CODE_BYTES:
        raise _GeneratedCodeLimitError(
            f"Python code is limited to {_LOCAL_ANALYZER_MAX_CODE_BYTES} UTF-8 bytes; "
            f"received {source_bytes}."
        )
    try:
        tree = ast.parse(code, filename="<python_tool>", mode="exec")
    except (MemoryError, RecursionError) as exc:
        raise _GeneratedCodeLimitError(
            "Python code is too deeply nested or complex to parse safely."
        ) from exc
    for node_count, _node in enumerate(ast.walk(tree), start=1):
        if node_count > _LOCAL_ANALYZER_MAX_AST_NODES:
            raise _GeneratedCodeLimitError(
                f"Python code is limited to {_LOCAL_ANALYZER_MAX_AST_NODES} syntax nodes."
            )
    return tree


def _static_candidate_value(node: ast.AST, bindings: dict[str, Any]) -> Any:
    """Resolve bounded literal data and simple variable aliases without execution."""
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_static_candidate_value(item, bindings) for item in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.Dict):
        return {
            _static_candidate_value(key, bindings): _static_candidate_value(
                value, bindings
            )
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _static_candidate_value(node.operand, bindings)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("non-numeric unary candidate value")
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("candidate action is dynamic")


def _static_candidate_action_arguments(tree: ast.AST) -> list[Any]:
    bindings: dict[str, Any] = {}
    for statement in getattr(tree, "body", []):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            try:
                bindings[statement.targets[0].id] = _static_candidate_value(
                    statement.value, bindings
                )
            except ValueError:
                bindings.pop(statement.targets[0].id, None)
    arguments: list[Any] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "action"
            and node.args
        ):
            try:
                arguments.append(_static_candidate_value(node.args[0], bindings))
            except ValueError:
                continue
    return arguments


def _normalize_generated_python_code(value: Any) -> str:
    """Recover unambiguous executable Python from common model wrappers.

    Exact nested code payloads are unwrapped first. Otherwise raw Python remains
    authoritative. If it does not compile, accept one unambiguous Python wrapper
    whose body does compile. Multiple blocks stay untouched so we never guess.
    """
    python_languages = {"", "py", "py3", "python", "python3"}

    def executable(candidate: Any) -> str | None:
        if not isinstance(candidate, str):
            return None
        candidate = candidate.rstrip()
        try:
            tree = _parse_bounded_generated_python(candidate)
            compile(tree, "<python_tool>", "exec")
        except (
            SyntaxError,
            ValueError,
            OverflowError,
            MemoryError,
            RecursionError,
        ):
            return None
        return candidate

    def nested_code(payload: Any) -> str | None:
        if not isinstance(payload, dict) or set(payload) not in (
            {"code"},
            {"code", "language"},
        ):
            return None
        language = str(payload.get("language") or "").strip().lower()
        if language not in python_languages:
            return None
        return executable(payload.get("code"))

    direct_nested = nested_code(value)
    if direct_nested is not None:
        return direct_nested

    code = str(value or "").rstrip()
    try:
        decoded = json.loads(code)
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = None
    decoded_nested = nested_code(decoded)
    if decoded_nested is not None:
        return decoded_nested

    raw = executable(code)
    if raw is not None:
        return raw

    # Models sometimes include an illustrative non-Python block alongside the
    # program, or make one failed attempt before emitting valid Python. Recover
    # the program when (and only when) the fenced candidates have one unique
    # executable value. This keeps the previous no-guessing behavior for two
    # competing valid programs while accepting standard backtick and tilde
    # Markdown fences, with or without a newline before the closing fence.
    fenced_candidates: list[str] = []
    for match in _MARKDOWN_CODE_FENCE_RE.finditer(code):
        if str(match.group("language") or "").lower() not in python_languages:
            continue
        fenced = executable(str(match.group("code") or ""))
        if fenced is not None and fenced not in fenced_candidates:
            fenced_candidates.append(fenced)
    if len(fenced_candidates) == 1:
        return fenced_candidates[0]

    xml_match = re.fullmatch(
        r"\s*<(?:python|code(?:\s+language=['\"](?:py|python|python3)['\"])?)>"
        r"(?P<code>.*?)</(?:python|code)>\s*",
        code,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if xml_match is not None:
        xml_code = executable(str(xml_match.group("code") or "").strip("\r\n"))
        if xml_code is not None:
            return xml_code

    lines = code.splitlines()
    if lines and lines[0].strip().lower() in {"py", "py:", "python", "python:"}:
        labelled = executable("\n".join(lines[1:]))
        if labelled is not None:
            return labelled
    return code


def _recover_tool_calls_from_markup(*chunks: str) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not chunk.strip():
            continue
        for match in _TOOL_CALL_BLOCK_RE.finditer(chunk):
            tool_name = str(match.group(1) or "").strip()
            if not tool_name:
                continue
            raw_body = str(match.group(2) or "")
            arguments = {
                str(parameter_name).strip(): value
                for parameter_name, value in _TOOL_CALL_PARAMETER_RE.findall(raw_body)
                if str(parameter_name).strip()
            }
            cache_key = (
                tool_name,
                json.dumps(arguments, ensure_ascii=True, sort_keys=True),
            )
            if cache_key in seen:
                continue
            seen.add(cache_key)
            recovered.append(
                {
                    "id": f"markup-call-{len(recovered) + 1}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=True),
                    },
                }
            )
    return recovered


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_LOCAL_ANALYZER_MAX_OUTPUT = _get_env_int("LOCAL_ANALYZER_MAX_OUTPUT", 0)
_LOCAL_ANALYZER_INITIAL_MAX_OUTPUT = _get_env_int(
    "LOCAL_ANALYZER_INITIAL_MAX_OUTPUT", 2048
)
_LOCAL_ANALYZER_FOLLOWUP_MAX_OUTPUT = _get_env_int(
    "LOCAL_ANALYZER_FOLLOWUP_MAX_OUTPUT", 1024
)
_LOCAL_ANALYZER_REPAIR_MAX_OUTPUT = _get_env_int(
    "LOCAL_ANALYZER_REPAIR_MAX_OUTPUT", 512
)
_LOCAL_ANALYZER_FORCE_PYTHON_TOOL = _get_env_bool(
    "LOCAL_ANALYZER_FORCE_PYTHON_TOOL", True
)
_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
_LOCAL_ANALYZER_TIMEOUT = _get_env_float("LOCAL_ANALYZER_TIMEOUT", 0.0)
_LOCAL_ANALYZER_TOOL_STEPS = _get_env_int("LOCAL_ANALYZER_TOOL_STEPS", 12)
_LOCAL_ANALYZER_TOOL_TIMEOUT = _get_env_int("LOCAL_ANALYZER_TOOL_TIMEOUT", 30)
_LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS = _get_env_int(
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", 1024
)
_LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
_LOCAL_ANALYZER_ENABLE_THINKING = _get_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True)
_LOCAL_ANALYZER_TEMPERATURE = _get_env_float("LOCAL_ANALYZER_TEMPERATURE", 0.6)
_LOCAL_ANALYZER_TOP_P = _get_env_float("LOCAL_ANALYZER_TOP_P", 0.95)
_LOCAL_ANALYZER_TOP_K = _get_env_int("LOCAL_ANALYZER_TOP_K", 20)
_LOCAL_ANALYZER_SEED = _get_env_int("LOCAL_ANALYZER_SEED", -1)
_LOCAL_ANALYZER_HTTP_MAX_ATTEMPTS = max(
    1, _get_env_int("LOCAL_ANALYZER_HTTP_MAX_ATTEMPTS", 3)
)
_LOCAL_ANALYZER_HTTP_RETRY_BASE_SECONDS = max(
    0.0, _get_env_float("LOCAL_ANALYZER_HTTP_RETRY_BASE_SECONDS", 0.25)
)
_LOCAL_ANALYZER_HTTP_RETRY_MAX_SECONDS = max(
    _LOCAL_ANALYZER_HTTP_RETRY_BASE_SECONDS,
    _get_env_float("LOCAL_ANALYZER_HTTP_RETRY_MAX_SECONDS", 2.0),
)
_LOCAL_ANALYZER_MAX_CODE_BYTES = max(
    1024, _get_env_int("LOCAL_ANALYZER_MAX_CODE_BYTES", 65_536)
)
_LOCAL_ANALYZER_MAX_AST_NODES = max(
    256, _get_env_int("LOCAL_ANALYZER_MAX_AST_NODES", 20_000)
)
_LOCAL_ANALYZER_CANDIDATES = min(
    4, max(1, _get_env_int("LOCAL_ANALYZER_CANDIDATES", 2))
)
_LOCAL_ANALYZER_GAME_TOKEN_BUDGET = max(
    0, _get_env_int("LOCAL_ANALYZER_GAME_TOKEN_BUDGET", 250_000)
)
_LOCAL_ANALYZER_DURABLE_STATE_BYTES = max(
    16_384, _get_env_int("LOCAL_ANALYZER_DURABLE_STATE_BYTES", 1_048_576)
)
_REQUEST_SAFETY_MARGIN_TOKENS = 512
_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS = 512
_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30
_RESPONSE_META_MAX_CHARS = 4000
_TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

_PYTHON_TOOL_DESCRIPTION = (
    "Run one ephemeral Python snippet against preloaded letter-coded game state. Globals: "
    "`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, "
    "`valid_actions`, `last_action_result`, read-only `experience`, persisted `strategy`, "
    "`record_strategy(...)`, bounded `memory`, `remember(key, value)`, `forget(key=None)`, "
    "and `action(actions)` for real environment actions. Frame and transition objects expose the "
    "bounded, mutation-safe analysis/search methods documented in the system prompt; repeated structural "
    "queries are memoized. Combine related inspections and scoring in one snippet before acting. "
    "Raw numeric colors are unavailable: prefer `.segmentation`, compact summaries, "
    "targeted searches, and small `.crop(...)` regions. `history[-1].frame` is current; use "
    "`previous_frame` or `last_transition` for before/after analysis. A final expression is returned "
    "notebook-style; use `print(...)` or assign `result` for compact output."
)


def _normalize_valid_actions(valid_actions: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in valid_actions or []:
        engine_name = to_engine_action(value)
        name = to_model_action(engine_name or value)
        if name and name not in names:
            names.append(name)
    return names


def _format_valid_action_line(valid_actions: list[str] | None) -> str:
    names = _normalize_valid_actions(valid_actions)
    if not names:
        return "unknown"
    return ", ".join(names)


def _terminal_action_reason(result: dict[str, Any]) -> str | None:
    if result.get("run_complete"):
        return "run_complete"
    if result.get("game_over"):
        return "game_over"
    if result.get("level_completed"):
        return "level_completed"
    if result.get("done"):
        return "done"
    return None


def _terminal_action_stop_detail(reason: str | None) -> str:
    if reason == "run_complete":
        return "No further actions were executed because the run is already complete."
    if reason == "game_over":
        return (
            "No further actions were executed because the previous action reached GAME_OVER; "
            "the runner will auto-reset before the next analyzer turn."
        )
    if reason == "level_completed":
        return (
            "No further actions were executed because the previous action completed a level; "
            "re-ground on the new scene before acting again."
        )
    if reason == "done":
        return "No further actions were executed because the environment reported done."
    return "No further actions were executed because the previous action reached a terminal state."


def _display_action_number(action_num: int) -> int:
    return max(1, int(action_num) + 1)


def _remaining_deadline_seconds(
    deadline: float | None, *, now: float | None = None
) -> float | None:
    if deadline is None:
        return None
    current = time.monotonic() if now is None else float(now)
    return max(0.0, float(deadline) - current)


def _normalize_summary_text(value: Any, *, max_chars: int | None = 280) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars].rstrip()}... [{omitted} chars omitted]"


def _extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
    normalized_labels = {label.lower(): label for label in labels}
    targets = tuple(f"{label.lower()}:" for label in labels)
    extracted: dict[str, list[str]] = {label: [] for label in labels}
    current_label: str | None = None

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        candidate = stripped
        while candidate.startswith(("-", "*")):
            candidate = candidate[1:].lstrip()
        lowered = candidate.lower()

        matched_label: str | None = None
        inline_value = ""
        for target in targets:
            if lowered.startswith(target):
                matched_label = normalized_labels[target[:-1]]
                inline_value = candidate[len(target) :].strip()
                break

        if matched_label is not None:
            current_label = matched_label
            if inline_value:
                extracted[current_label].append(inline_value)
            continue

        if current_label is not None and stripped:
            extracted[current_label].append(stripped)

    return {
        label: _normalize_summary_text("\n".join(lines).strip(), max_chars=None)
        for label, lines in extracted.items()
        if "\n".join(lines).strip()
    }


def _extract_scientist_note(content: str) -> dict[str, str]:
    if not content.strip():
        return {}
    extracted = _extract_labeled_blocks(
        content,
        [
            "World model",
            "Goal model",
            "Action model",
            "Recent findings",
            "Open questions",
            "Plan",
            "Cross-level notes",
            "Hypothesis",
            "History check",
            "Next test",
        ],
    )
    result = {
        "world_model": extracted.get("World model", ""),
        "goal_model": extracted.get("Goal model", ""),
        "action_model": extracted.get("Action model", ""),
        "recent_findings": extracted.get("Recent findings", ""),
        "open_questions": extracted.get("Open questions", ""),
        "current_plan": extracted.get("Plan", ""),
        "cross_level_notes": extracted.get("Cross-level notes", ""),
    }
    if not result["world_model"]:
        result["world_model"] = extracted.get("Hypothesis", "")
    if not result["recent_findings"]:
        result["recent_findings"] = extracted.get("History check", "")
    if not result["current_plan"]:
        result["current_plan"] = extracted.get("Next test", "")
    return result


def _empty_world_model() -> dict[str, str]:
    return {
        "world_model": "",
        "goal_model": "",
        "action_model": "",
        "recent_findings": "",
        "open_questions": "",
        "current_plan": "",
        "cross_level_notes": "",
    }


def _request_tool_choice(
    tools: list[dict[str, Any]] | None,
    *,
    force: bool = _LOCAL_ANALYZER_FORCE_PYTHON_TOOL,
) -> str | dict[str, Any] | None:
    if not tools:
        return None
    if force and len(tools) == 1:
        function = tools[0].get("function", {}) if isinstance(tools[0], dict) else {}
        name = str(function.get("name", "")).strip()
        if name:
            return {"type": "function", "function": {"name": name}}
    return "auto"


def _trim_log_text(text: str, *, max_chars: int = _RESPONSE_META_MAX_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    omitted = len(stripped) - max_chars
    return f"{stripped[:max_chars].rstrip()}\n... [truncated {omitted} chars]"


def _format_model_response_meta(
    *,
    finish_reason: str,
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    tool_call_markup_in_text: bool,
    recovered_tool_calls_from_markup: bool,
    malformed_argument_errors: list[str],
) -> str:
    lines = [
        f"finish_reason: {finish_reason or '(empty)'}",
        f"tool_call_count: {len(tool_calls)}",
        f"content_chars: {len(content)}",
        f"reasoning_chars: {len(reasoning)}",
        f"tool_call_markup_in_text: {'yes' if tool_call_markup_in_text else 'no'}",
        f"tool_calls_recovered_from_markup: {'yes' if recovered_tool_calls_from_markup else 'no'}",
    ]
    if malformed_argument_errors:
        lines.append("tool_call_argument_issues:")
        lines.extend(f"- {issue}" for issue in malformed_argument_errors)
    if tool_calls:
        lines.append("raw_tool_calls:")
        lines.append(
            _trim_log_text(json.dumps(tool_calls, indent=2, ensure_ascii=True))
        )
    return "\n".join(lines)


def _build_system_prompt(*, tool_output_tokens: int) -> str:
    prompt = "You are a coding agent solving a grid-based puzzle game."
    prompt += GAME_OVERVIEW_ADDENDUM
    prompt += STRUCTURED_RUNTIME_STATE_ADDENDUM
    if current_grid_image_enabled():
        prompt += MULTIMODAL_CONTEXT_ADDENDUM
    prompt += VISUAL_GAME_ADDENDUM
    prompt += PYTHON_ADDENDUM
    prompt += COMPACT_TOOL_SESSION_ADDENDUM.format(
        tool_output_tokens=tool_output_tokens
    )
    return prompt


@dataclass(frozen=True)
class AnalyzerModelConfig:
    provider: str
    base_url: str
    model_id: str


@dataclass(frozen=True)
class AnalyzerTurnResult:
    step_executed: bool
    retryable_failure: bool = False
    reasoning: str = ""
    yielded_control: bool = False
    yield_reason: str | None = None
    efficiency_metrics: dict[str, float | int] | None = None
    failure_category: str | None = None
    failure_detail: str = ""
    attempts: int = 0
    exhausted: bool = False


@dataclass(frozen=True)
class _ToolDispatchResult:
    content: str
    step_executed: bool = False


@dataclass(frozen=True)
class _AsciiFrameView:
    ascii: str
    step: int
    level: int
    shape: tuple[int, int]

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"
        )

    __repr__ = __str__


@dataclass(frozen=True)
class _AsciiHistoryEntryView:
    action: str
    frame: _AsciiFrameView

    def __str__(self) -> str:
        return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

    __repr__ = __str__


@dataclass(frozen=True)
class _FramePayloadCache:
    current_frame: Frame | None
    history_entries: tuple[HistoryEntry, ...]
    current_payload: dict[str, Any] | None
    history_payload: list[dict[str, Any]]


@dataclass(frozen=True)
class _ExperienceSnapshotCache:
    current_frame: Frame | None
    history_entries: tuple[HistoryEntry, ...]
    valid_actions: tuple[str, ...]
    knowledge_revision: int
    payload: dict[str, Any]


def _to_ascii_frame_view(frame: Frame | None) -> _AsciiFrameView | None:
    if frame is None:
        return None
    return _AsciiFrameView(
        ascii=frame.ascii,
        step=frame.step,
        level=frame.level,
        shape=frame.shape,
    )


def _to_ascii_history_views(
    history_entries: list[HistoryEntry],
) -> list[_AsciiHistoryEntryView]:
    views: list[_AsciiHistoryEntryView] = []
    for entry in history_entries:
        frame_view = _to_ascii_frame_view(entry.frame)
        if frame_view is None:
            continue
        views.append(_AsciiHistoryEntryView(action=entry.action, frame=frame_view))
    return views


def _ascii_frame_view_payload(frame: Frame | None) -> dict[str, Any] | None:
    view = _to_ascii_frame_view(frame)
    if view is None:
        return None
    return {
        "ascii": view.ascii,
        "step": view.step,
        "level": view.level,
        "shape": [int(view.shape[0]), int(view.shape[1])],
        "grid": [list(row) for row in frame.grid],
    }


def _ascii_frame_delta_payload(
    before: Frame,
    after: Frame,
) -> dict[str, Any] | None:
    if len(before.grid) != len(after.grid) or any(
        len(before_row) != len(after_row)
        for before_row, after_row in zip(before.grid, after.grid)
    ):
        return None
    cell_count = sum(len(row) for row in after.grid)
    changes = [
        [row, col, after_value]
        for row, (before_row, after_row) in enumerate(zip(before.grid, after.grid))
        for col, (before_value, after_value) in enumerate(zip(before_row, after_row))
        if before_value != after_value
    ]
    if changes and len(changes) * 6 >= max(1, cell_count):
        return None
    return {
        "step": after.step,
        "level": after.level,
        "shape": [int(after.shape[0]), int(after.shape[1])],
        "changes": changes,
    }


def _ascii_history_view_payload(
    history_entries: list[HistoryEntry],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    previous_frame: Frame | None = None
    for entry in history_entries:
        delta_payload = (
            _ascii_frame_delta_payload(previous_frame, entry.frame)
            if previous_frame is not None
            else None
        )
        if delta_payload is not None:
            payload.append({"action": entry.action, "frame_delta": delta_payload})
        else:
            frame_payload = _ascii_frame_view_payload(entry.frame)
            if frame_payload is not None:
                payload.append({"action": entry.action, "frame": frame_payload})
        previous_frame = entry.frame
    return payload


def _current_frame_transport_payload(
    current_frame: Frame | None,
    history_entries: list[HistoryEntry],
    full_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if current_frame is None or not history_entries:
        return full_payload
    latest = history_entries[-1].frame
    if (
        current_frame.step == latest.step
        and current_frame.level == latest.level
        and current_frame.grid == latest.grid
    ):
        return {"history_index": len(history_entries) - 1}
    return full_payload


def _format_action_span(
    start_action_num: int | None, end_action_num: int | None
) -> str | None:
    if start_action_num is None or end_action_num is None:
        return None
    if start_action_num <= 0 or end_action_num <= 0:
        return None
    if start_action_num == end_action_num:
        return f"{start_action_num}"
    return f"{start_action_num}-{end_action_num}"


def _estimate_tokens(value: Any) -> int:
    return max(1, (_estimated_json_length(value) + 2) // 3)


def _estimated_json_length(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str))
    except TypeError:
        return len(str(value))


def _estimated_request_base_length(
    tools: list[dict[str, Any]] | None,
) -> int:
    payload: dict[str, Any] = {"messages": []}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = _request_tool_choice(tools)
    return _estimated_json_length(payload)


def _host_accessible_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname != "host.docker.internal":
        return base_url
    netloc = "127.0.0.1"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _resolve_analyzer_model(model: str) -> AnalyzerModelConfig:
    requested = (model or "").strip()
    lowered = requested.lower()
    if lowered in {"local", "local-qwen", "qwen-local", "qwen"}:
        configured_base_url = os.environ.get(
            "LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL
        ).strip()
        if not configured_base_url:
            raise ValueError(
                "LOCAL_ANALYZER_BASE_URL must be set for the local analyzer preset."
            )

        provider = (
            os.environ.get(
                "LOCAL_ANALYZER_PROVIDER", os.environ.get("OPENAI_PROVIDER", "vllm")
            )
            .strip()
            .lower()
        )
        if not provider:
            provider = "vllm"
        model_id = (
            os.environ.get("LOCAL_ANALYZER_MODEL_ID", "").strip()
            or _LOCAL_ANALYZER_MODEL_ID.strip()
        )
        if not model_id:
            raise ValueError(
                "LOCAL_ANALYZER_MODEL_ID must be set for the local analyzer preset."
            )
        return AnalyzerModelConfig(
            provider=provider,
            base_url=_host_accessible_base_url(configured_base_url),
            model_id=model_id,
        )

    if not requested:
        requested = _LOCAL_ANALYZER_MODEL_ID.strip()
    if not requested:
        raise ValueError(
            "Analyzer model id is required. Set analyzer.model_id in config, pass --model, "
            "or set LOCAL_ANALYZER_MODEL_ID / INFERENCE_ANALYZER_MODEL."
        )

    provider = (
        os.environ.get(
            "OPENAI_PROVIDER", os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")
        )
        .strip()
        .lower()
    )
    if not provider:
        provider = "vllm"
    base_url = _host_accessible_base_url(
        os.environ.get(
            "OPENAI_BASE_URL",
            os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL),
        ).strip()
    )
    if not base_url:
        raise ValueError(
            "OPENAI_BASE_URL or LOCAL_ANALYZER_BASE_URL must be set for direct model ids."
        )
    return AnalyzerModelConfig(provider=provider, base_url=base_url, model_id=requested)


def _resolve_fallback_models() -> list[AnalyzerModelConfig]:
    raw = os.environ.get("LOCAL_ANALYZER_FALLBACKS_JSON", "").strip()
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("LOCAL_ANALYZER_FALLBACKS_JSON ignored: %s", exc)
        return []
    if not isinstance(values, list):
        log.warning("LOCAL_ANALYZER_FALLBACKS_JSON must contain a JSON list")
        return []
    resolved: list[AnalyzerModelConfig] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        model_id = str(value.get("model") or value.get("model_id") or "").strip()
        base_url = str(value.get("base_url") or "").strip()
        provider = str(value.get("provider") or "vllm").strip().lower()
        if model_id and base_url:
            resolved.append(
                AnalyzerModelConfig(
                    provider=provider or "vllm",
                    base_url=_host_accessible_base_url(base_url),
                    model_id=model_id,
                )
            )
    return resolved


def _append_transcript_section(log_path: Path, label: str, content: str) -> None:
    rendered_content = content.strip()
    if not rendered_content:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{label}]\n")
        f.write(rendered_content)
        f.write("\n\n")


class _TranscriptBuffer:
    """Append-only transcript with one file handle and lazy full rendering."""

    def __init__(self, log_path: Path, header: str) -> None:
        self._handle: Any = io.StringIO()
        self._parts = [header]
        self._rendered: str | None = header
        self._closed = False
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(log_path, "a", encoding="utf-8")
            self._handle.write(header)
            self._handle.flush()
        except OSError as exc:
            log.warning("analyzer transcript unavailable at %s: %s", log_path, exc)
            try:
                self._handle.close()
            except (OSError, ValueError):
                pass
            self._handle = io.StringIO()

    def append(self, label: str, content: str) -> str:
        section = _render_transcript_section(label, content)
        if not section:
            return ""
        self._parts.append(section)
        self._rendered = None
        try:
            self._handle.write(section)
            self._handle.flush()
        except (OSError, ValueError) as exc:
            log.warning("analyzer transcript write failed: %s", exc)
            try:
                self._handle.close()
            except (OSError, ValueError):
                pass
            self._handle = io.StringIO()
        return section

    def render(self) -> str:
        if self._rendered is None:
            self._rendered = "".join(self._parts)
        return self._rendered

    def close(self) -> None:
        if not self._closed:
            try:
                self._handle.close()
            except (OSError, ValueError):
                pass
            self._closed = True

    def __del__(self) -> None:
        self.close()


def _render_transcript_section(label: str, content: str) -> str:
    rendered_content = content.strip()
    if not rendered_content:
        return ""
    return f"[{label}]\n{rendered_content}\n\n"


def _json_like_payload(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _render_scalar_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _render_human_readable_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}{key_text}: |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}{key_text}: {_render_scalar_value(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}- |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}- {_render_scalar_value(item)}")
        return lines
    if isinstance(value, str):
        if "\n" in value:
            multiline = value.splitlines() or [""]
            return [f"{prefix}|", *(f"{prefix}  {line}" for line in multiline)]
        return [f"{prefix}{value}"]
    return [f"{prefix}{_render_scalar_value(value)}"]


def _render_human_readable_value(value: Any) -> str:
    return "\n".join(_render_human_readable_lines(value))


def _render_jsonish_text(value: Any) -> str:
    parsed = _json_like_payload(value)
    if parsed is not None:
        return _render_human_readable_value(parsed)
    return (
        _normalize_message_content(value)
        if not isinstance(value, str)
        else value.strip()
    )


def _render_tool_parameter_text(value: Any) -> str:
    if isinstance(value, str):
        return value.rstrip("\n")
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def _normalize_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return json.loads(json.dumps(arguments))
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        if stripped.startswith("<tool_call>"):
            recovered_tool_calls = _recover_tool_calls_from_markup(stripped)
            if recovered_tool_calls:
                recovered_arguments = (
                    recovered_tool_calls[0].get("function", {}).get("arguments", "{}")
                )
                return _normalize_tool_call_arguments(recovered_arguments)
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as json_error:
            code = _normalize_generated_python_code(stripped)
            try:
                compile(code, "<python_tool_arguments>", "exec")
            except (SyntaxError, TypeError, ValueError):
                raise json_error
            return {"code": code}
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("tool call arguments must decode to a JSON object")
    raise ValueError("tool call arguments must be a JSON object or JSON object string")


def _render_tool_call_markup(
    tool_name: str,
    arguments: Any,
    *,
    arguments_normalized: bool = False,
) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return ""
    if arguments_normalized:
        if not isinstance(arguments, dict):
            return ""
        parsed_arguments = arguments
    else:
        try:
            parsed_arguments = _normalize_tool_call_arguments(arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ""

    lines = ["<tool_call>", f"<function={name}>"]
    for parameter_name, parameter_value in parsed_arguments.items():
        lines.append(f"<parameter={parameter_name}>")
        rendered_value = _render_tool_parameter_text(parameter_value)
        if rendered_value:
            lines.extend(rendered_value.splitlines())
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _render_tool_result_display(content: Any) -> str:
    parsed = (
        _json_like_payload(content)
        if isinstance(content, str)
        else (content if isinstance(content, dict) else None)
    )
    if isinstance(parsed, dict):
        stdout = str(parsed.get("stdout", "") or "").rstrip("\n")
        error = str(parsed.get("error", "") or "").rstrip("\n")
        result = parsed.get("result")
        has_result = result not in (None, "", [], {})
        if stdout and not error and not has_result:
            return stdout

        blocks: list[str] = []
        if stdout:
            blocks.append(stdout)
        if has_result:
            rendered_result = _render_human_readable_value(result)
            if stdout:
                blocks.append(f"result:\n{rendered_result}")
            else:
                blocks.append(rendered_result)
        if error:
            if stdout or has_result:
                blocks.append(f"error:\n{error}")
            else:
                blocks.append(error)
        if blocks:
            return "\n\n".join(block for block in blocks if block.strip())

    return _render_jsonish_text(content)


def _python_tool_payload(sandbox_result: dict[str, Any]) -> dict[str, Any]:
    """Preserve independent stdout and result channels from generated code."""
    payload: dict[str, Any] = {"tool": "python"}
    rendered_stdout = str(sandbox_result.get("stdout", "") or "")
    rendered_error = str(sandbox_result.get("error", "") or "")
    action_results = [
        item
        for item in sandbox_result.get("action_results") or []
        if isinstance(item, dict)
    ]
    auto_repair = sandbox_result.get("auto_repair")
    if isinstance(auto_repair, dict):
        payload["auto_repair"] = dict(auto_repair)
    if rendered_error:
        payload["error"] = rendered_error
        diagnostic = sandbox_result.get("diagnostic")
        if isinstance(diagnostic, dict):
            payload["diagnostic"] = dict(diagnostic)
            payload["retryable"] = diagnostic.get("retry") != "do_not_retry"
        if rendered_stdout:
            payload["stdout"] = rendered_stdout
        if action_results:
            payload["action_results"] = action_results
            executed_calls = sum(
                1 for item in action_results if bool(item.get("executed"))
            )
            payload["partial_execution"] = {
                "action_calls": len(action_results),
                "executed_action_calls": executed_calls,
                "side_effects_committed": executed_calls > 0,
                "rollback_available": False,
            }
            payload["retryable"] = False
            if isinstance(payload.get("diagnostic"), dict):
                payload["diagnostic"]["retry"] = "replan_from_current_state"
                payload["diagnostic"]["hint"] = (
                    "Some environment actions already executed and cannot be rolled "
                    "back. Inspect action_results and replan from the current state; "
                    "do not rerun the same program."
                )
        return payload

    payload["returncode"] = 0
    if rendered_stdout:
        payload["stdout"] = rendered_stdout
    if sandbox_result.get("result") is not None:
        payload["result"] = sandbox_result.get("result")
    elif not rendered_stdout and action_results:
        if len(action_results) == 1:
            payload["result"] = action_results[-1]
        else:
            payload["result"] = {
                "action_calls": len(action_results),
                "last_action_result": action_results[-1],
            }
    return payload


def _bounded_python_memory(
    update: dict[str, Any],
    *,
    max_entries: int = 16,
    max_key_chars: int = 64,
    max_value_bytes: int = 2048,
    max_total_bytes: int = 8192,
) -> dict[str, Any]:
    """Validate and normalize the JSON scratchpad persisted between tool calls."""
    if not isinstance(update, dict):
        raise TypeError("memory must be a JSON object.")
    if len(update) > max_entries:
        raise ValueError(f"memory is limited to {max_entries} keys.")

    normalized: dict[str, Any] = {}
    for raw_key, value in update.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError("memory keys must be non-empty strings.")
        key = raw_key.strip()
        if len(key) > max_key_chars:
            raise ValueError(f"memory keys are limited to {max_key_chars} characters.")
        try:
            encoded_value = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("memory values must be finite JSON data.") from exc
        if len(encoded_value) > max_value_bytes:
            raise ValueError(
                f"each memory value is limited to {max_value_bytes} bytes."
            )
        normalized[key] = value

    encoded_memory = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded_memory) > max_total_bytes:
        raise ValueError(f"memory is limited to {max_total_bytes} bytes total.")
    return json.loads(encoded_memory.decode("utf-8"))


def _resolve_run_artifact_location(state_path: Path) -> tuple[Path, str | None]:
    parent = state_path.parent
    if parent.name == "artifacts" and parent.parent != parent:
        run_root = parent.parent
        runtime_state_files = list(parent.glob(f"*_{RUNTIME_STATE_FILENAME}"))
        if len(runtime_state_files) <= 1:
            return run_root, None
        runtime_state_stem = Path(RUNTIME_STATE_FILENAME).stem
        suffix = f"_{runtime_state_stem}"
        state_stem = state_path.stem
        game_stem = (
            state_stem[: -len(suffix)] if state_stem.endswith(suffix) else state_stem
        )
        return run_root, game_stem
    return parent, None


def _resolve_named_run_artifact(
    state_path: Path,
    *,
    default_name: str,
    per_game_suffix: str,
    directory_name: str | None = None,
) -> Path:
    run_root, game_stem = _resolve_run_artifact_location(state_path)
    output_root = run_root / directory_name if directory_name else run_root
    if game_stem:
        return output_root / f"{game_stem}{per_game_suffix}"
    return output_root / default_name


def _render_prompt_log_message(message: dict[str, Any]) -> str:
    role = str(message.get("role", "")).strip().upper() or "UNKNOWN"
    header = f"[{role}]"
    tool_call_id = str(message.get("tool_call_id", "")).strip()
    if role == "TOOL" and tool_call_id:
        header = f"[TOOL RESULT: {tool_call_id}]"
    blocks = [header]

    content = _normalize_message_content(message.get("content", ""))
    if content:
        blocks.append(
            _render_tool_result_display(content) if role == "TOOL" else content
        )

    reasoning = _extract_reasoning_text(message)
    if reasoning:
        blocks.append("[REASONING]")
        blocks.append(reasoning)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for tool_call in tool_calls:
            function = (
                tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            )
            name = str(function.get("name", "")).strip() or "unknown"
            blocks.append(f"[ASSISTANT TOOL CALL: {name}]")
            tool_call_id = str(tool_call.get("id", "")).strip()
            if tool_call_id:
                blocks.append(f"id: {tool_call_id}")
            rendered_tool_call = _render_tool_call_markup(
                name, function.get("arguments", "{}")
            )
            if rendered_tool_call:
                blocks.append(rendered_tool_call)
            else:
                raw_arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                    rendered_arguments = json.dumps(
                        parsed_arguments, indent=2, ensure_ascii=True
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    rendered_arguments = str(raw_arguments)
                blocks.append("arguments:")
                blocks.append(
                    rendered_arguments if rendered_arguments.strip() else "{}"
                )

    return "\n".join(blocks)


def _resolve_prompt_log_path(state_path: Path) -> Path:
    return _resolve_named_run_artifact(
        state_path,
        default_name="prompt.log",
        per_game_suffix=".log",
        directory_name="prompts",
    )


def _resolve_request_log_path(state_path: Path) -> Path:
    return _resolve_named_run_artifact(
        state_path,
        default_name="requests.jsonl",
        per_game_suffix="_requests.jsonl",
    )


def _append_request_snapshot(
    log_path: Path,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    event: str | None = None,
    tool_choice: Any = None,
    finish_reason: str | None = None,
    analysis_step: int | None = None,
    action: int | None = None,
    request_index_within_turn: int | None = None,
) -> None:
    payload = {
        "messages": messages,
        "tools": tools or [],
    }
    if event:
        payload["event"] = event
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if finish_reason is not None:
        payload["finish_reason"] = str(finish_reason)
    if analysis_step is not None:
        payload["analysis_step"] = analysis_step
    if action is not None:
        payload["action"] = action
    if request_index_within_turn is not None:
        payload["request_index_within_turn"] = request_index_within_turn
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=True,
            )
        )
        f.write("\n")


def _write_prompt_log_snapshot(
    log_path: Path,
    *,
    model_id: str,
    base_url: str,
    display_action_num: int,
    analysis_step: int | None,
    request_index: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    transcript: str,
) -> None:
    rendered_messages = "\n\n".join(
        _render_prompt_log_message(message) for message in messages
    )
    rendered_tools: list[str] = []
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name", "")).strip() or "unknown"
        description = str(function.get("description", "")).strip()
        if description:
            rendered_tools.append(f"- {name}: {description}")
        else:
            rendered_tools.append(f"- {name}")
    analysis_label = str(analysis_step) if analysis_step is not None else "n/a"
    transcript_text = transcript.strip()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LATEST MODEL CALL SNAPSHOT\n")
        f.write(f"model: {model_id}\n")
        f.write(f"base_url: {base_url}\n")
        f.write(f"analysis_step: {analysis_label}\n")
        f.write(f"action: {display_action_num}\n")
        f.write(f"request_index_within_turn: {request_index}\n")
        f.write(f"message_count: {len(messages)}\n")
        f.write(f"tool_choice: {tool_choice or '(none)'}\n")
        f.write("\n[AVAILABLE TOOLS]\n")
        f.write("\n".join(rendered_tools) if rendered_tools else "(none)")
        f.write("\n\n[MODEL INPUT]\n")
        f.write(rendered_messages.strip())
        f.write("\n\n[TURN TRANSCRIPT SO FAR]\n")
        f.write(transcript_text)
        f.write("\n")


def _safe_append_request_snapshot(*args: Any, **kwargs: Any) -> None:
    try:
        _append_request_snapshot(*args, **kwargs)
    except OSError as exc:
        log.warning("analyzer request log write failed: %s", exc)


def _safe_write_prompt_log_snapshot(*args: Any, **kwargs: Any) -> None:
    try:
        _write_prompt_log_snapshot(*args, **kwargs)
    except OSError as exc:
        log.warning("analyzer prompt log write failed: %s", exc)


def _normalize_message_content(content: Any) -> str:
    def _strip_think_tags(text: str) -> str:
        cleaned = _THINK_TAG_RE.sub("", text)
        cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
        return cleaned.strip()

    if isinstance(content, str):
        return _strip_think_tags(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return _strip_think_tags("\n".join(part for part in parts if part))
    return ""


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning")
    if reasoning in (None, ""):
        reasoning = message.get("reasoning_content", "")
    return _normalize_message_content(reasoning)


def _is_context_length_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "maximum context length" in message
        or "reduce the length of the input prompt" in message
        or "parameter=input_tokens" in message
        or '"param":"input_tokens"' in message
    )


@dataclass
class _ChatCompletionResult:
    message: dict[str, Any]
    finish_reason: str = ""
    usage: dict[str, Any] | None = None
    latency_seconds: float = 0.0
    request_attempts: int = 1
    forced_tool_fallback: bool = False
    candidate_count: int = 1
    selected_candidate_index: int = 0
    valid_candidate_count: int = 1
    fallback_messages: list[dict[str, Any]] = dataclass_field(default_factory=list)


class ToolAgent:
    """Direct tool-calling analyzer compatible with OpenAI-style endpoints."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_ANALYZER_MODEL,
        timeout: float | None = None,
        save_request_logs: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        knowledge_store: TrialKnowledgeStore | None = None,
        game_id: str = "",
        pass_index: int = 0,
        evidence_id: str = "",
    ) -> None:
        resolved_model = _resolve_analyzer_model(model)
        if base_url is not None or provider is not None:
            resolved_model = AnalyzerModelConfig(
                provider=str(provider or resolved_model.provider).strip()
                or resolved_model.provider,
                base_url=(
                    _host_accessible_base_url(str(base_url).strip())
                    if base_url is not None and str(base_url).strip()
                    else resolved_model.base_url
                ),
                model_id=resolved_model.model_id,
            )
        self._model = resolved_model
        self._fallback_models = _resolve_fallback_models()
        configured_timeout = _LOCAL_ANALYZER_TIMEOUT if timeout is None else timeout
        self._timeout = (
            None
            if configured_timeout is None or configured_timeout <= 0
            else float(configured_timeout)
        )
        self._api_key = str(api_key or "").strip()
        self._tool_steps = (
            None
            if _LOCAL_ANALYZER_TOOL_STEPS <= 0
            else max(1, _LOCAL_ANALYZER_TOOL_STEPS)
        )
        self._python_timeout = min(30, max(1, _LOCAL_ANALYZER_TOOL_TIMEOUT))
        self._yield_seconds = (
            None
            if _LOCAL_ANALYZER_YIELD_SECONDS <= 0
            else float(_LOCAL_ANALYZER_YIELD_SECONDS)
        )
        configured_max_output = _LOCAL_ANALYZER_MAX_OUTPUT
        self._max_output_tokens = (
            max(1, configured_max_output)
            if configured_max_output > 0
            else max(1, _LOCAL_ANALYZER_INITIAL_MAX_OUTPUT)
        )
        self._reply_reserve_tokens = self._max_output_tokens
        self._tool_output_tokens = max(64, _LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS)
        self._tool_output_chars = max(256, self._tool_output_tokens * 4)
        self._save_request_logs = bool(save_request_logs)
        self._system_prompt = _build_system_prompt(
            tool_output_tokens=self._tool_output_tokens,
        )
        self._request_safety_margin_tokens = _REQUEST_SAFETY_MARGIN_TOKENS
        self._context_budget_tokens = max(
            1024,
            _LOCAL_ANALYZER_CONTEXT_WINDOW
            - self._reply_reserve_tokens
            - self._request_safety_margin_tokens,
        )
        self._history_messages: list[dict[str, Any]] = []
        self._session_runtime_dir: Path | None = None
        self._session_total_tokens = 0
        self._session_generated_tokens = 0
        self._step_env_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = (
            None
        )
        self._current_valid_actions: list[str] = []
        self._last_step_summary: dict[str, Any] | None = None
        self._last_action_result: dict[str, Any] | None = None
        self._summarized_knowledge = _empty_world_model()
        self._controller_config = InferenceControllerConfig.from_env()
        self._knowledge_store = knowledge_store
        self._knowledge_game_id = str(game_id or "")
        self._knowledge_pass_index = max(0, int(pass_index))
        self._knowledge_evidence_id = str(evidence_id or "")[:160]
        self._current_experience_snapshot: dict[str, Any] = {}
        self._strategy_memory: dict[str, Any] = {}
        self._python_memory: dict[str, Any] = {}
        self._frame_payload_cache: _FramePayloadCache | None = None
        self._experience_snapshot_cache: _ExperienceSnapshotCache | None = None
        self._generated_tool_call_count = 0
        self._http_session = requests.Session()
        self._forced_tool_choice_supported: bool | None = None
        self._turn_efficiency_metrics: dict[str, float | int] = {}
        self._http_max_attempts = _LOCAL_ANALYZER_HTTP_MAX_ATTEMPTS
        self._http_retry_base_seconds = _LOCAL_ANALYZER_HTTP_RETRY_BASE_SECONDS
        self._http_retry_max_seconds = _LOCAL_ANALYZER_HTTP_RETRY_MAX_SECONDS
        self._candidate_count = _LOCAL_ANALYZER_CANDIDATES
        self._tokenizer_path = os.environ.get(
            "LOCAL_ANALYZER_TOKENIZER_PATH", ""
        ).strip()
        self._tokenizer: Any | None = None
        self._tokenizer_load_attempted = False
        self._should_stop_callback: Callable[[], bool] | None = None

    def _activate_next_fallback_model(self) -> bool:
        if not self._fallback_models:
            return False
        previous = self._model
        self._model = self._fallback_models.pop(0)
        self._forced_tool_choice_supported = None
        log.warning(
            "analyzer failover: %s at %s -> %s at %s",
            previous.model_id,
            previous.base_url,
            self._model.model_id,
            self._model.base_url,
        )
        return True

    def close(self) -> None:
        self._http_session.close()

    def __del__(self) -> None:
        session = getattr(self, "_http_session", None)
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    def _adaptive_output_limit(
        self, request_index: int, *, repair: bool = False
    ) -> int:
        if _LOCAL_ANALYZER_MAX_OUTPUT > 0:
            limit = self._max_output_tokens
        elif repair:
            limit = max(
                1, min(self._max_output_tokens, _LOCAL_ANALYZER_REPAIR_MAX_OUTPUT)
            )
        elif request_index <= 1:
            limit = self._max_output_tokens
        else:
            limit = max(
                1, min(self._max_output_tokens, _LOCAL_ANALYZER_FOLLOWUP_MAX_OUTPUT)
            )
        remaining = self._remaining_game_tokens()
        return min(limit, max(1, remaining)) if remaining is not None else limit

    def _remaining_game_tokens(self) -> int | None:
        if _LOCAL_ANALYZER_GAME_TOKEN_BUDGET <= 0:
            return None
        return max(
            0, _LOCAL_ANALYZER_GAME_TOKEN_BUDGET - self._session_generated_tokens
        )

    def _adaptive_candidate_count(self, output_limit: int) -> int:
        remaining = self._remaining_game_tokens()
        if remaining is not None and remaining < max(1, output_limit) * 2:
            return 1
        snapshot = self._current_experience_snapshot or {}
        if not snapshot:
            return self._candidate_count
        uncertainty = max(
            (
                float(item.get("uncertainty") or 0.0)
                for item in snapshot.get("ranked_actions") or []
                if isinstance(item, dict)
            ),
            default=0.0,
        )
        if str(snapshot.get("phase") or "") == "recover" or uncertainty >= 0.6:
            return self._candidate_count
        return 1

    def _exact_request_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> int | None:
        if not self._tokenizer_path:
            return None
        if not self._tokenizer_load_attempted:
            self._tokenizer_load_attempted = True
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._tokenizer_path,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception as exc:  # noqa: BLE001 - optional optimization
                log.warning(
                    "analyzer tokenizer unavailable at %s: %s",
                    self._tokenizer_path,
                    exc,
                )
                self._tokenizer = None
        if self._tokenizer is None:
            return None
        try:
            tokens = self._tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=True,
                add_generation_prompt=True,
            )
            return max(1, len(tokens))
        except Exception as exc:  # noqa: BLE001 - fall back to conservative estimate
            log.warning("analyzer exact token count failed: %s", exc)
            return None

    def _record_efficiency(self, key: str, value: float | int) -> None:
        existing = self._turn_efficiency_metrics.get(key, 0)
        self._turn_efficiency_metrics[key] = existing + value

    def _headers(self) -> dict[str, str]:
        api_key = (
            self._api_key
            or os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        site_url = os.environ.get("LOCAL_ANALYZER_SITE_URL", "").strip()
        app_name = os.environ.get(
            "LOCAL_ANALYZER_APP_NAME", "ARC3 Agent Harness"
        ).strip()
        return build_headers(
            provider=self._model.provider,
            api_key=api_key,
            referer=site_url,
            title=app_name,
        )

    def _ensure_session(self, state_path: Path) -> None:
        session_path = state_path.resolve()
        if self._session_runtime_dir != session_path:
            self._session_runtime_dir = session_path
            self._history_messages = []
            self._session_total_tokens = 0
            self._session_generated_tokens = 0
            self._last_step_summary = None
            self._last_action_result = None
            self._summarized_knowledge = _empty_world_model()
            self._strategy_memory = {}
            self._python_memory = {}
            self._frame_payload_cache = None
            self._experience_snapshot_cache = None
            self._generated_tool_call_count = 0
            self._load_durable_state()

    def _durable_state_path(self) -> Path | None:
        if getattr(self, "_session_runtime_dir", None) is None:
            return None
        path = self._session_runtime_dir
        return path.with_name(f"{path.stem}_agent_state.json")

    def _load_durable_state(self) -> None:
        path = self._durable_state_path()
        if path is None or not path.is_file():
            return
        try:
            if path.stat().st_size > _LOCAL_ANALYZER_DURABLE_STATE_BYTES:
                raise ValueError(
                    "durable agent state exceeds its configured size limit"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                raise ValueError("unsupported durable agent state")
            for attribute, key in (
                ("_summarized_knowledge", "summarized_knowledge"),
                ("_strategy_memory", "strategy_memory"),
                ("_python_memory", "python_memory"),
            ):
                value = payload.get(key)
                if isinstance(value, dict):
                    setattr(self, attribute, dict(value))
            history = payload.get("history_messages")
            if isinstance(history, list):
                self._history_messages = [
                    dict(item) for item in history if isinstance(item, dict)
                ]
            for attribute, key in (
                ("_last_step_summary", "last_step_summary"),
                ("_last_action_result", "last_action_result"),
            ):
                value = payload.get(key)
                if isinstance(value, dict):
                    setattr(self, attribute, dict(value))
            self._session_total_tokens = max(
                0, int(payload.get("total_tokens", 0) or 0)
            )
            self._session_generated_tokens = max(
                0, int(payload.get("generated_tokens", 0) or 0)
            )
        except (OSError, TypeError, ValueError) as exc:
            log.warning("durable analyzer state ignored at %s: %s", path, exc)

    def _persist_durable_state(self) -> None:
        path = self._durable_state_path()
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if path is None or runtime_path is None or not runtime_path.exists():
            return
        history = list(self._history_messages)
        payload = {
            "version": 1,
            "summarized_knowledge": self._summarized_knowledge,
            "strategy_memory": self._strategy_memory,
            "python_memory": self._python_memory,
            "history_messages": history,
            "last_step_summary": self._last_step_summary,
            "last_action_result": self._last_action_result,
            "total_tokens": self._session_total_tokens,
            "generated_tokens": self._session_generated_tokens,
        }
        try:
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), default=str
            )
            while (
                len(encoded.encode("utf-8")) > _LOCAL_ANALYZER_DURABLE_STATE_BYTES
                and history
            ):
                history.pop(0)
                payload["history_messages"] = history
                encoded = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), default=str
                )
            if len(encoded.encode("utf-8")) > _LOCAL_ANALYZER_DURABLE_STATE_BYTES:
                payload["history_messages"] = []
                encoded = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), default=str
                )
            if len(encoded.encode("utf-8")) > _LOCAL_ANALYZER_DURABLE_STATE_BYTES:
                raise ValueError(
                    "durable analyzer state exceeds its configured size limit"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as exc:
            log.warning("durable analyzer state write failed at %s: %s", path, exc)

    def _cached_frame_payloads(
        self,
        current_frame: Frame | None,
        history_entries: list[HistoryEntry],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        cached = self._frame_payload_cache
        if (
            cached is not None
            and cached.current_frame is current_frame
            and len(cached.history_entries) == len(history_entries)
            and all(
                cached_entry is current_entry
                for cached_entry, current_entry in zip(
                    cached.history_entries, history_entries
                )
            )
        ):
            return cached.current_payload, cached.history_payload

        current_payload = _ascii_frame_view_payload(current_frame)
        history_payload = _ascii_history_view_payload(history_entries)
        self._frame_payload_cache = _FramePayloadCache(
            current_frame=current_frame,
            history_entries=tuple(history_entries),
            current_payload=current_payload,
            history_payload=history_payload,
        )
        return current_payload, history_payload

    def _cached_experience_snapshot(
        self,
        current_frame: Frame | None,
        history_entries: list[HistoryEntry],
        valid_actions: list[str],
    ) -> dict[str, Any]:
        action_key = tuple(valid_actions)
        knowledge_revision = (
            self._knowledge_store.revision if self._knowledge_store is not None else 0
        )
        cached = self._experience_snapshot_cache
        if (
            cached is not None
            and cached.current_frame is current_frame
            and cached.valid_actions == action_key
            and cached.knowledge_revision == knowledge_revision
            and len(cached.history_entries) == len(history_entries)
            and all(
                cached_entry is current_entry
                for cached_entry, current_entry in zip(
                    cached.history_entries, history_entries
                )
            )
        ):
            return cached.payload

        cross_trial = None
        if self._knowledge_store is not None and self._knowledge_game_id:
            cross_trial = self._knowledge_store.snapshot(
                self._knowledge_game_id,
                exclude_evidence_id=self._knowledge_evidence_id,
            )
        payload = build_experience_snapshot(
            history_entries,
            current_frame,
            valid_actions,
            self._controller_config,
            external_transitions=(cross_trial or {}).get("transition_records"),
            evidence_id=self._knowledge_evidence_id,
        )
        if cross_trial is not None:
            payload = dict(payload)
            payload["cross_trial"] = self._knowledge_store.snapshot(
                self._knowledge_game_id,
                state_id=str(payload.get("state_id") or ""),
                exclude_evidence_id=self._knowledge_evidence_id,
            )
        if self._strategy_memory.get("causal_model"):
            payload["causal_model"] = normalize_causal_model(
                self._strategy_memory["causal_model"]
            )
        self._experience_snapshot_cache = _ExperienceSnapshotCache(
            current_frame=current_frame,
            history_entries=tuple(history_entries),
            valid_actions=action_key,
            knowledge_revision=knowledge_revision,
            payload=payload,
        )
        return payload

    def _normalize_response_tool_calls(self, value: Any) -> list[dict[str, Any]]:
        raw_calls = value if isinstance(value, list) else []
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw_call in raw_calls:
            call = dict(raw_call) if isinstance(raw_call, dict) else {}
            raw_function = call.get("function")
            call["function"] = (
                dict(raw_function) if isinstance(raw_function, dict) else {}
            )
            call["type"] = str(call.get("type") or "function")
            call_id = str(call.get("id") or "").strip()
            if not call_id or call_id in seen_ids:
                while True:
                    self._generated_tool_call_count += 1
                    call_id = f"generated-tool-call-{self._generated_tool_call_count}"
                    if call_id not in seen_ids:
                        break
            call["id"] = call_id
            seen_ids.add(call_id)
            normalized.append(call)
        return normalized

    def _record_python_memory(self, update: dict[str, Any]) -> dict[str, Any]:
        persisted = _bounded_python_memory(update)
        self._python_memory = persisted
        self._persist_durable_state()
        return dict(persisted)

    def _record_strategy(self, update: dict[str, Any]) -> dict[str, Any]:
        def short_text(value: Any, max_chars: int = 280) -> str:
            return _normalize_summary_text(value, max_chars=max_chars)[:max_chars]

        persisted = dict(self._strategy_memory)
        for key in (
            "goal",
            "hypothesis",
            "open_question",
            "next_test",
            "fallback",
            "current_subgoal",
            "success_criteria",
            "risk",
            "abort_condition",
        ):
            value = short_text(update.get(key))
            if value:
                persisted[key] = value

        test_action = short_text(update.get("test_action"), 80)
        if test_action:
            persisted["test_action"] = normalize_action_key(test_action)
        expected_outcome = str(update.get("expected_outcome") or "").strip().lower()
        if expected_outcome in {
            "no_change",
            "state_change",
            "new_state",
            "level_progress",
            "unknown",
        }:
            persisted["expected_outcome"] = expected_outcome
        if test_action or expected_outcome:
            persisted.pop("prediction_result", None)

        raw_evidence = update.get("evidence")
        if isinstance(raw_evidence, (list, tuple)):
            evidence = [short_text(item, 160) for item in raw_evidence]
            evidence = [item for item in evidence if item][:5]
        else:
            single_evidence = short_text(raw_evidence, 160)
            evidence = [single_evidence] if single_evidence else []
        if evidence:
            persisted["evidence"] = evidence

        for key, limit in (("subgoals", 6), ("plan_steps", 8)):
            raw_items = update.get(key)
            if isinstance(raw_items, (list, tuple)):
                items = [short_text(item, 200) for item in raw_items]
                items = [item for item in items if item][:limit]
                if items:
                    persisted[key] = items

        raw_contradictions = update.get("contradictions")
        if isinstance(raw_contradictions, (list, tuple)):
            contradictions = [short_text(item, 160) for item in raw_contradictions]
            contradictions = [item for item in contradictions if item][:3]
        else:
            single_contradiction = short_text(raw_contradictions, 160)
            contradictions = [single_contradiction] if single_contradiction else []
        if contradictions:
            persisted["contradictions"] = contradictions

        causal_model_updated = isinstance(update.get("causal_model"), dict)
        if causal_model_updated:
            causal_model = normalize_causal_model(update["causal_model"])
            snapshot_state = self._current_experience_snapshot.get(
                "behavioral_state_id"
            ) or self._current_experience_snapshot.get("state_id")
            if snapshot_state:
                for prediction in causal_model["predictions"]:
                    if not prediction.get("conditions"):
                        prediction["conditions"] = f"state:{snapshot_state}"
            persisted["causal_model"] = causal_model

        try:
            if update.get("confidence") is not None:
                persisted["confidence"] = max(
                    0.0, min(1.0, float(update.get("confidence")))
                )
        except (TypeError, ValueError):
            pass

        self._strategy_memory = persisted
        if causal_model_updated:
            self._current_experience_snapshot["causal_model"] = persisted[
                "causal_model"
            ]
            self._experience_snapshot_cache = None
        if persisted.get("goal"):
            self._summarized_knowledge["goal_model"] = str(persisted["goal"])
        if persisted.get("hypothesis"):
            confidence = persisted.get("confidence")
            suffix = (
                f" (confidence={confidence:.2f})"
                if isinstance(confidence, float)
                else ""
            )
            self._summarized_knowledge["world_model"] = (
                f"{persisted['hypothesis']}{suffix}"
            )
        if persisted.get("evidence"):
            self._summarized_knowledge["recent_findings"] = "; ".join(
                str(item) for item in persisted["evidence"]
            )
        if persisted.get("open_question"):
            self._summarized_knowledge["open_questions"] = str(
                persisted["open_question"]
            )
        if persisted.get("next_test"):
            self._summarized_knowledge["current_plan"] = str(persisted["next_test"])
        self._persist_durable_state()
        return dict(self._strategy_memory)

    def _consume_strategy_prediction(self, result: dict[str, Any]) -> dict[str, Any]:
        consumed = {
            "test_action": normalize_action_key(
                self._strategy_memory.get("test_action", "")
            )
            or str(result.get("action") or ""),
            "expected_outcome": (
                str(self._strategy_memory.get("expected_outcome") or "").strip().lower()
                or str(result.get("expected") or "")
            ),
            "status": result.get("status"),
            "actual": result.get("actual"),
        }
        self._strategy_memory["last_evaluated_prediction"] = consumed
        for key in ("test_action", "expected_outcome", "prediction_result"):
            self._strategy_memory.pop(key, None)
        return dict(consumed)

    def _evaluate_strategy_prediction(
        self, action_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        existing = action_result.get("prediction_result")
        if isinstance(existing, dict):
            self._consume_strategy_prediction(dict(existing))
            return dict(existing)
        test_action = normalize_action_key(self._strategy_memory.get("test_action", ""))
        expected = str(self._strategy_memory.get("expected_outcome") or "").strip()
        if not test_action or not expected:
            return None
        candidates = [
            item for item in action_result.get("steps") or [] if isinstance(item, dict)
        ]
        candidates.append(action_result)
        observed = next(
            (
                item
                for item in candidates
                if normalize_action_key(item.get("action_display", "")) == test_action
                or (
                    test_action == "MOUSE"
                    and action_family(item.get("action_display", "")) == "MOUSE"
                )
            ),
            None,
        )
        if observed is None:
            return None
        actual = str(observed.get("outcome_class") or "unknown")
        if (
            not observed.get("executed", action_result.get("executed"))
            or expected == "unknown"
        ):
            status = "inconclusive"
        else:
            matched = {
                "no_change": not bool(observed.get("board_changed")),
                "state_change": bool(observed.get("board_changed")),
                "new_state": bool(observed.get("novel_state")),
                "level_progress": bool(
                    observed.get("level_completed")
                    or observed.get("run_complete")
                    or float(observed.get("reward") or 0.0) > 0.0
                ),
            }.get(expected)
            status = "supported" if matched else "contradicted"
        result = {
            "status": status,
            "action": normalize_action_key(observed.get("action_display", test_action)),
            "expected": expected,
            "actual": actual,
        }
        self._consume_strategy_prediction(result)
        return dict(result)

    def _update_causal_model_from_actions(
        self, action_results: list[dict[str, Any]]
    ) -> None:
        """Ground causal relations and structured predictions in observations."""
        causal = normalize_causal_model(self._strategy_memory.get("causal_model") or {})
        changed = False
        relations = causal["relations"]
        predictions = causal["predictions"]

        def ground_relation(
            *, action: str, effect: str, condition: str, action_num: int
        ) -> bool:
            matching = next(
                (
                    relation
                    for relation in relations
                    if normalize_action_key(relation.get("cause") or "") == action
                    and str(relation.get("effect") or "") == effect
                    and str(relation.get("conditions") or "") == condition
                ),
                None,
            )
            if matching is None and len(relations) >= 24:
                evictable = [
                    relation
                    for relation in relations
                    if str(relation.get("evidence") or "").startswith(
                        "automatically grounded"
                    )
                    and not (
                        normalize_action_key(relation.get("cause") or "") == action
                        and str(relation.get("effect") or "") == effect
                        and int(relation.get("last_observed_action") or 0) == action_num
                    )
                ]
                if evictable:
                    relations.remove(
                        min(
                            evictable,
                            key=lambda relation: (
                                float(relation.get("confidence") or 0.0),
                                int(relation.get("support") or 0),
                                int(relation.get("last_observed_action") or 0),
                            ),
                        )
                    )
            if matching is None and len(relations) < 24:
                matching = {
                    "cause": action,
                    "effect": effect,
                    "conditions": condition,
                    "evidence": "automatically grounded from executed transition",
                    "confidence": 0.5,
                    "support": 0,
                    "contradictions": 0,
                    "last_observed_action": action_num,
                }
                relations.append(matching)
            if matching is None:
                return False
            matching["support"] = int(matching.get("support") or 0) + 1
            matching["last_observed_action"] = action_num
            for relation in relations:
                if (
                    relation is not matching
                    and normalize_action_key(relation.get("cause") or "") == action
                    and str(relation.get("effect") or "").startswith("outcome:")
                    and str(relation.get("conditions") or "") == condition
                ):
                    relation["contradictions"] = (
                        int(relation.get("contradictions") or 0) + 1
                    )
            for relation in relations:
                if (
                    normalize_action_key(relation.get("cause") or "") != action
                    or str(relation.get("conditions") or "") != condition
                ):
                    continue
                support = int(relation.get("support") or 0)
                contradictions = int(relation.get("contradictions") or 0)
                relation["confidence"] = round(
                    (support + 1) / (support + contradictions + 2), 3
                )
            return True

        for observed in action_results:
            if not observed.get("executed"):
                continue
            action = normalize_action_key(observed.get("action_display") or "")
            outcome = str(observed.get("outcome_class") or "unknown")
            effect = f"outcome:{outcome}"
            state_condition = (
                "state:"
                f"{observed.get('behavioral_before_state_id') or observed.get('before_state_id') or 'unknown'}"
            )
            observed_conditions = [state_condition]
            object_state_id = str(observed.get("object_before_state_id") or "")
            if object_state_id:
                observed_conditions.append(f"object_state:{object_state_id}")
            action_num = int(observed.get("action_num") or 0)
            for condition in observed_conditions:
                changed = (
                    ground_relation(
                        action=action,
                        effect=effect,
                        condition=condition,
                        action_num=action_num,
                    )
                    or changed
                )
            for prediction in predictions:
                prediction_conditions = str(prediction.get("conditions") or "")
                if (
                    str(prediction.get("status") or "untested") == "untested"
                    and normalize_action_key(prediction.get("action") or "") == action
                    and (
                        not prediction_conditions.startswith(
                            ("state:", "object_state:")
                        )
                        or prediction_conditions in observed_conditions
                    )
                ):
                    expected = str(prediction.get("expected_outcome") or "")
                    if expected:
                        prediction["status"] = (
                            "supported" if expected == outcome else "contradicted"
                        )
                        changed = True
        if changed:
            self._strategy_memory["causal_model"] = normalize_causal_model(causal)
            self._persist_durable_state()

    @property
    def total_tokens(self) -> int:
        return max(0, int(self._session_total_tokens))

    @property
    def generated_tokens(self) -> int:
        return max(0, int(self._session_generated_tokens))

    def _accumulate_usage_tokens(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return

        def first_token_count(*keys: str) -> int:
            for key in keys:
                raw_value = usage.get(key)
                try:
                    return max(0, int(raw_value))
                except (TypeError, ValueError):
                    continue
            return 0

        generated_token_count = first_token_count(
            "completion_tokens",
            "output_tokens",
            "generated_tokens",
        )
        self._session_generated_tokens += generated_token_count

        total_tokens = usage.get("total_tokens")
        try:
            if total_tokens is not None:
                self._session_total_tokens += max(0, int(total_tokens))
                return
        except (TypeError, ValueError):
            pass

        input_token_count = first_token_count("prompt_tokens", "input_tokens")
        self._session_total_tokens += input_token_count + generated_token_count

    def _summarize_step_sequence(
        self, action_results: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not action_results:
            return None
        executed_results = [item for item in action_results if item.get("executed")]
        if not executed_results:
            return None

        total_executed = 0
        executed_actions: list[str] = []
        for item in executed_results:
            count = item.get("executed_count")
            try:
                parsed = int(count) if count is not None else 1
            except (TypeError, ValueError):
                parsed = 1
            total_executed += max(1, parsed)
            action_names = item.get("executed_actions")
            if isinstance(action_names, list):
                executed_actions.extend(
                    str(name).strip() for name in action_names if str(name).strip()
                )
            else:
                fallback_action = str(item.get("action_display") or "").strip()
                if fallback_action:
                    executed_actions.append(fallback_action)

        last = executed_results[-1]
        try:
            end_action_num = int(last.get("action_num"))
        except (TypeError, ValueError):
            end_action_num = None
        start_action_num = None
        if end_action_num is not None and total_executed > 0:
            start_action_num = max(1, end_action_num - total_executed + 1)

        return {
            "start_action_num": start_action_num,
            "end_action_num": end_action_num,
            "executed_count": total_executed,
            "executed_actions": executed_actions,
            "level": last.get("level"),
            "level_transition": any(
                bool(item.get("level_completed")) for item in executed_results
            ),
            "run_complete": any(
                bool(item.get("run_complete")) for item in executed_results
            ),
            "game_over": any(bool(item.get("game_over")) for item in executed_results),
            "board_changed": any(
                bool(item.get("board_changed")) for item in executed_results
            ),
            "stop_reason": last.get("stop_reason"),
        }

    def _describe_last_outcome(self, summary: dict[str, Any] | None) -> str:
        if not summary:
            return ""
        span = _format_action_span(
            summary.get("start_action_num"),
            summary.get("end_action_num"),
        )
        count = summary.get("executed_count")
        prefix = "Last executed sequence"
        if span and count:
            prefix = f"Actions {span} ({count} total)"
        elif span:
            prefix = f"Action span {span}"
        elif count:
            prefix = f"Last executed sequence ({count} total)"

        level = summary.get("level")
        if summary.get("level_transition"):
            level_text = f" to level {level}" if level is not None else ""
            return f"{prefix} triggered a level transition{level_text}; re-ground on the new scene."
        if summary.get("run_complete"):
            return f"{prefix} completed the run."
        if summary.get("game_over"):
            return f"{prefix} reached GAME_OVER."

        pieces = [prefix]
        if summary.get("board_changed"):
            pieces.append(
                "produced a board change; verify that it affected gameplay objects rather than only HUD elements."
            )
        else:
            pieces.append(
                "did not show a confirmed board change; treat this as weak evidence until verified."
            )
        stop_reason = _normalize_summary_text(summary.get("stop_reason"))
        if stop_reason:
            pieces.append(f"stop_reason={stop_reason}.")
        return " ".join(pieces)

    def _update_summarized_knowledge_from_assistant(self, content: str) -> None:
        note = _extract_scientist_note(content)
        if not note:
            return
        for key, value in note.items():
            if value:
                self._summarized_knowledge[key] = value

    def _update_summarized_knowledge_from_step_summary(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        if (
            summary.get("level_transition")
            or summary.get("run_complete")
            or summary.get("game_over")
        ):
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""

    def _summarized_knowledge_lines(self) -> list[str]:
        entries = [
            ("World model", self._summarized_knowledge.get("world_model", "")),
            ("Goal model", self._summarized_knowledge.get("goal_model", "")),
            ("Action model", self._summarized_knowledge.get("action_model", "")),
            ("Recent findings", self._summarized_knowledge.get("recent_findings", "")),
            ("Open questions", self._summarized_knowledge.get("open_questions", "")),
            ("Plan", self._summarized_knowledge.get("current_plan", "")),
            (
                "Cross-level notes",
                self._summarized_knowledge.get("cross_level_notes", ""),
            ),
        ]
        lines = [f"- {label}: {value}" for label, value in entries if value]
        if not lines:
            return []
        return [
            "Working world model carried from earlier turns:",
            *lines,
            "- Revise any item above immediately if `current_frame` or `history` contradicts it.",
        ]

    def _build_user_message(
        self,
        user_prompt: str,
        current_frame: Frame | None,
        previous_frame: Frame | None = None,
    ) -> dict[str, Any]:
        image_part = current_grid_image_part(current_frame)
        if image_part is None:
            return {"role": "user", "content": user_prompt}

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        previous_image = (
            current_grid_image_part(previous_frame)
            if previous_frame is not None
            and current_frame is not None
            and previous_frame.grid != current_frame.grid
            else None
        )
        if previous_image is not None:
            content.extend(
                [
                    {"type": "text", "text": "Previous grid image:"},
                    previous_image,
                ]
            )
        content.extend(
            [
                {"type": "text", "text": "Current grid image:"},
                image_part,
            ]
        )
        return {
            "role": "user",
            "content": content,
        }

    def _build_user_prompt(
        self,
        action_num: int,
        *,
        valid_actions: list[str] | None,
        current_frame: Frame | None = None,
        history_entries: list[HistoryEntry] | None = None,
        previous_step_summary: dict[str, Any] | None = None,
        experience_snapshot: dict[str, Any] | None = None,
    ) -> str:
        history_entries = history_entries or []
        current_step = (
            max(
                current_frame.step if current_frame is not None else 0,
                max(0, action_num),
            )
            + 1
        )
        current_level = current_frame.level if current_frame is not None else 1
        summary_level = None
        if previous_step_summary is not None:
            try:
                summary_level = int(previous_step_summary.get("level"))
            except (TypeError, ValueError):
                summary_level = None
        if summary_level is not None:
            current_level = max(current_level, summary_level)
        observed_max_level = max(
            [
                current_level,
                *[
                    entry.frame.level
                    for entry in history_entries
                    if entry.frame is not None
                ],
            ],
            default=current_level,
        )
        lines: list[str] = []
        if previous_step_summary:
            count = previous_step_summary.get("executed_count")
            try:
                normalized_count = int(count) if count is not None else None
            except (TypeError, ValueError):
                normalized_count = None
            action_label = "action" if normalized_count == 1 else "actions"
            lines.append(
                f"The code executed {normalized_count or 0} {action_label} in the previous sequence."
            )
            executed_actions = previous_step_summary.get("executed_actions")
            rendered_actions: list[str] = []
            if isinstance(executed_actions, list):
                rendered_actions = [
                    str(name).strip() for name in executed_actions if str(name).strip()
                ]
            if rendered_actions:
                action_prefix = (
                    "Executed actions (first 10):"
                    if len(rendered_actions) > 10
                    else "Executed actions:"
                )
                lines.append(f"{action_prefix} {', '.join(rendered_actions[:10])}.")
            else:
                lines.append("Executed actions: none.")
            if previous_step_summary.get("run_complete"):
                lines.append("You have completed the run!")
            elif previous_step_summary.get("level_transition"):
                lines.append("You have progressed to a new level!")
            else:
                lines.append("You are still on the same level.")
            if previous_step_summary.get("game_over"):
                lines.append("The game is over.")
        elif (current_frame is not None and current_frame.step > 0) or action_num > 0:
            lines.append("No previous action sequence was captured.")
        else:
            lines.append("No previous sequence has been executed yet.")
        state_line = f"Current state: step {current_step}, level {current_level}"
        if observed_max_level > current_level:
            state_line += f" out of observed max level {observed_max_level} so far"
        state_line += "."
        lines.extend(
            [
                state_line,
                f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",
                "Use the documented `python` state and bounded helpers to inspect current evidence, update the working world model, and select the shortest reliable action or batch.",
                "Keep output decision-focused; record material strategy changes and stop acting on any terminal result.",
            ]
        )
        if experience_snapshot and experience_snapshot.get("enabled"):
            compact_experience = {
                key: experience_snapshot.get(key)
                for key in (
                    "policy",
                    "phase",
                    "action_budget",
                    "state_id",
                    "behavioral_state_id",
                    "state_visits",
                    "unique_states",
                    "unique_behavioral_states",
                    "volatile_cells",
                    "actions_observed",
                    "no_op_streak",
                    "behavioral_no_op_streak",
                    "stagnation_actions",
                    "cycle_period",
                    "cycle_signal",
                    "latest_outcome",
                    "recovery_reasons",
                    "tried_here",
                    "suggested_actions",
                    "discouraged_actions",
                    "ranked_actions",
                    "mouse_search",
                    "plan_candidates",
                    "recommended_plan",
                    "transition_models_here",
                    "model_conflicts_here",
                    "outcome_utilities",
                    "causal_model",
                )
            }
            cross_trial = experience_snapshot.get("cross_trial")
            if isinstance(cross_trial, dict):
                compact_experience["cross_trial"] = {
                    key: cross_trial.get(key)
                    for key in (
                        "prior_trials",
                        "observations",
                        "independent_evidence",
                        "state_action_evidence",
                        "progress_lessons",
                    )
                }
            lines.extend(
                [
                    "Deterministic experience controller snapshot:",
                    json.dumps(
                        compact_experience, separators=(",", ":"), sort_keys=True
                    ),
                    "Honor the controller phase, action budget, rankings, and deterministic transition evidence; never retry a discouraged exact action, and revise the hypothesis on conflicts.",
                ]
            )
        lines.extend(self._summarized_knowledge_lines())
        lines.append(
            "Before executing new actions you must always give the revised version of this working world model, updated from the newest evidence."
        )
        if action_num == 0:
            lines.append(
                "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame."
            )
        else:
            lines.append(
                "Focus on what changed most recently in `history`, update the target environment change if needed, and separate gameplay-object changes from HUD-only changes."
            )
        lines.extend(
            [
                "Then call `action(actions)` inside `python` with the selected valid action; prefer batching it in one call only when the short sequence is reliable.",
            ]
        )
        if "MOUSE" in _normalize_valid_actions(valid_actions):
            lines.append("If you use MOUSE, include integer row and col arguments.")
        return "\n".join(lines)

    def _tools(self, state_path: Path) -> list[dict[str, Any]]:
        self._ensure_session(state_path)
        return [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": _PYTHON_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to run. The snippet is ephemeral and is not saved across tool calls."
                                ),
                            },
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def _score_candidate_choice(self, choice: Any) -> tuple[int, bool]:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            return -10_000, False
        message = choice["message"]
        score = 0
        valid = False
        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            for raw_call in raw_tool_calls:
                if not isinstance(raw_call, dict):
                    score -= 100
                    continue
                function = raw_call.get("function")
                if (
                    not isinstance(function, dict)
                    or str(function.get("name", "")).strip() != "python"
                ):
                    score -= 50
                    continue
                try:
                    arguments = _normalize_tool_call_arguments(
                        function.get("arguments", "{}")
                    )
                    code = _normalize_generated_python_code(arguments.get("code", ""))
                    if not code:
                        raise ValueError("empty Python code")
                    tree = _parse_bounded_generated_python(code)
                    compile(tree, "<python_tool_candidate>", "exec")
                except (SyntaxError, TypeError, ValueError, OverflowError):
                    score -= 100
                    continue
                score += 100
                valid = True
                literal_action_batches: list[list[dict[str, Any]]] = []
                candidate_action_arguments = _static_candidate_action_arguments(tree)
                for literal_actions in candidate_action_arguments:
                    try:
                        normalized_actions = self._normalize_python_actions(
                            literal_actions
                        )
                    except (TypeError, ValueError):
                        score -= 100
                    else:
                        score += 20
                        literal_action_batches.append(normalized_actions)
                score += self._semantic_action_score(literal_action_batches)
        content = _normalize_message_content(message.get("content", ""))
        reasoning = _extract_reasoning_text(message)
        if content or reasoning:
            score += 10
            valid = True
        if str(choice.get("finish_reason", "")) == "length":
            score -= 20
        return score, valid

    def _semantic_action_score(self, action_batches: list[list[dict[str, Any]]]) -> int:
        """Rank executable candidates against current empirical evidence."""
        snapshot = self._current_experience_snapshot or {}
        if not snapshot or not action_batches:
            return 0
        actions = [item for batch in action_batches for item in batch]
        budget = max(1, int(snapshot.get("action_budget") or 1))
        score = -40 * max(0, len(actions) - budget)
        discouraged = {
            normalize_action_key(item)
            for item in snapshot.get("discouraged_actions") or []
        }
        rankings = {
            str(item.get("action") or ""): item
            for item in snapshot.get("ranked_actions") or []
            if isinstance(item, dict)
        }
        empirical = {
            normalize_action_key(item.get("action") or ""): item
            for item in snapshot.get("transition_models_here") or []
            if isinstance(item, dict) and item.get("action")
        }
        cross_evidence = {
            normalize_action_key(item.get("action") or ""): item
            for item in (snapshot.get("cross_trial") or {}).get(
                "state_action_evidence", []
            )
            if isinstance(item, dict) and item.get("action")
        }
        causal_relations = [
            item
            for item in (snapshot.get("causal_model") or {}).get("relations", [])
            if isinstance(item, dict)
        ]
        current_state_conditions = {
            f"state:{state_id}"
            for state_id in (
                snapshot.get("state_id"),
                snapshot.get("behavioral_state_id"),
            )
            if state_id
        }
        object_state_id = str(snapshot.get("object_state_id") or "")
        if object_state_id:
            current_state_conditions.add(f"object_state:{object_state_id}")
        mouse_recent = {
            (int(item["row"]), int(item["col"])): str(item.get("outcome") or "")
            for item in (snapshot.get("mouse_search") or {}).get("recent", [])
            if isinstance(item, dict) and "row" in item and "col" in item
        }
        recommended_plan = snapshot.get("recommended_plan") or {}
        planned = list(recommended_plan.get("actions") or [])
        plan_confidence = float(recommended_plan.get("confidence") or 0.0)
        reset_recommended = any(
            isinstance(item, dict) and item.get("strategy") == "reset_episode"
            for item in snapshot.get("recovery_portfolio") or ()
        )
        configured_utilities = snapshot.get("outcome_utilities") or {}
        outcome_values = (
            {
                outcome: round(120 * float(value))
                for outcome, value in configured_utilities.items()
            }
            if configured_utilities
            else {
                "level_progress": 120,
                "novel": 24,
                "revisit": -6,
                "volatile_only": -24,
                "exact_noop": -48,
                "terminal_failure": -240,
                "negative_reward": -240,
            }
        )
        outcome_values.setdefault("unknown", 0)
        recommended_mouse = {
            (int(item["row"]), int(item["col"]))
            for item in (snapshot.get("mouse_search") or {}).get(
                "recommended_coordinates", []
            )
            if isinstance(item, dict) and "row" in item and "col" in item
        }
        for index, item in enumerate(actions):
            name = str(item.get("action") or "")
            if name == "MOUSE":
                coordinate = (int(item["row"]), int(item["col"]))
                key = f"MOUSE(ROW={coordinate[0]}, COL={coordinate[1]})"
                score += (
                    -70
                    if mouse_recent.get(coordinate) in {"exact_noop", "volatile_only"}
                    else 20
                )
                if coordinate in recommended_mouse:
                    score += 35
            else:
                key = normalize_action_key(name)
            if key in discouraged:
                score -= 100
            if key == "RESET":
                score += 60 if reset_recommended else -160
            ranking = rankings.get(action_family(key))
            if ranking is not None:
                score += max(0, 40 - 10 * int(ranking.get("priority") or 0))
            model = empirical.get(key)
            if model is not None:
                confidence = float(model.get("confidence") or 0.0)
                score += round(
                    outcome_values.get(
                        str(model.get("predicted_outcome") or "unknown"), 0
                    )
                    * confidence
                )
                score += round(120 * float(model.get("mean_reward") or 0.0))
                score -= 25 * int(model.get("contradictions") or 0)
            evidence = cross_evidence.get(key)
            if evidence is not None and model is None:
                outcomes = evidence.get("outcomes") or {}
                trials = max(1, int(evidence.get("trials") or 1))
                score += round(
                    sum(
                        outcome_values.get(str(outcome), 0) * int(count or 0)
                        for outcome, count in outcomes.items()
                    )
                    / trials
                )
            applicable_relations: dict[
                str, tuple[tuple[int, float, int, int], dict[str, Any]]
            ] = {}
            for relation in causal_relations:
                if normalize_action_key(relation.get("cause") or "") != key:
                    continue
                conditions = str(relation.get("conditions") or "")
                if (
                    conditions.startswith(("state:", "object_state:"))
                    and conditions not in current_state_conditions
                ):
                    continue
                effect = str(relation.get("effect") or "")
                if not effect.startswith("outcome:"):
                    continue
                if model is not None and str(relation.get("evidence") or "").startswith(
                    "automatically grounded"
                ):
                    continue
                specificity = (
                    2
                    if conditions.startswith("state:")
                    else 1
                    if conditions.startswith("object_state:")
                    else 0
                )
                existing = applicable_relations.get(effect)
                relation_rank = (
                    specificity,
                    float(relation.get("confidence") or 0.0),
                    -int(relation.get("contradictions") or 0),
                    int(relation.get("support") or 0),
                )
                if existing is None or relation_rank > existing[0]:
                    applicable_relations[effect] = (relation_rank, relation)
            for effect, (_rank, relation) in applicable_relations.items():
                causal_outcome = effect.split(":", 1)[1]
                confidence = float(relation.get("confidence") or 0.0)
                score += round(outcome_values.get(causal_outcome, 0) * confidence)
                score -= 10 * int(relation.get("contradictions") or 0)
            if index < len(planned) and normalize_action_key(planned[index]) == key:
                score += round(80 * plan_confidence)
        return score

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        request_timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
    ) -> _ChatCompletionResult:
        force_tool = self._forced_tool_choice_supported is not False
        tool_choice = _request_tool_choice(tools, force=force_tool)
        output_limit = max_output_tokens or self._max_output_tokens
        payload = build_chat_payload(
            provider=self._model.provider,
            model=self._model.model_id,
            messages=messages,
            max_tokens=output_limit,
            temperature=_LOCAL_ANALYZER_TEMPERATURE,
            top_p=_LOCAL_ANALYZER_TOP_P,
            top_k=_LOCAL_ANALYZER_TOP_K,
            thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING),
            tools=tools,
            tool_choice=tool_choice,
            seed=_LOCAL_ANALYZER_SEED,
            candidates=self._adaptive_candidate_count(output_limit),
        )
        started_at = time.monotonic()
        deadline = (
            started_at + request_timeout_seconds
            if request_timeout_seconds is not None
            else None
        )
        request_attempts = 0

        def stop_requested() -> bool:
            if self._should_stop_callback is None:
                return False
            try:
                return bool(self._should_stop_callback())
            except Exception as exc:  # noqa: BLE001
                log.warning("analyzer HTTP stop check failed: %s", exc)
                return False

        def post_chat(request_payload: dict[str, Any]) -> requests.Response:
            nonlocal request_attempts
            if stop_requested():
                raise requests.RequestException("analyzer request cancelled")
            timeout = (
                request_timeout_seconds
                if request_timeout_seconds is not None
                else self._timeout
            )
            if deadline is not None:
                timeout = max(0.1, deadline - time.monotonic())
            request_attempts += 1
            return self._http_session.post(
                f"{self._model.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=request_payload,
                timeout=timeout,
            )

        def retry_delay(response: requests.Response | None) -> float:
            delay = min(
                self._http_retry_max_seconds,
                self._http_retry_base_seconds * (2 ** max(0, request_attempts - 1)),
            )
            headers = getattr(response, "headers", {}) if response is not None else {}
            try:
                retry_after = float(headers.get("Retry-After", ""))
            except (AttributeError, TypeError, ValueError):
                retry_after = 0.0
            if retry_after > 0:
                delay = min(self._http_retry_max_seconds, retry_after)
            return max(0.0, delay)

        def wait_to_retry(response: requests.Response | None) -> bool:
            delay = retry_delay(response)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                delay = min(delay, remaining)
            end = time.monotonic() + delay
            while True:
                if stop_requested():
                    raise requests.RequestException("analyzer request cancelled")
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return True
                time.sleep(min(0.05, remaining))

        def post_with_retry(request_payload: dict[str, Any]) -> requests.Response:
            while True:
                try:
                    candidate = post_chat(request_payload)
                except requests.RequestException:
                    if request_attempts >= self._http_max_attempts or not wait_to_retry(
                        None
                    ):
                        raise
                    continue
                if (
                    candidate.status_code in _TRANSIENT_HTTP_STATUS_CODES
                    and request_attempts < self._http_max_attempts
                ):
                    candidate.close()
                    if wait_to_retry(candidate):
                        continue
                return candidate

        response = post_with_retry(payload)
        forced_tool_fallback = False
        if (
            isinstance(tool_choice, dict)
            and response.status_code in {400, 404, 422}
            and request_attempts < self._http_max_attempts
        ):
            detail = response.text.lower()
            if (
                "tool_choice" in detail
                or "function" in detail
                or "tool choice" in detail
            ):
                fallback_payload = dict(payload)
                fallback_payload["tool_choice"] = "auto"
                response.close()
                response = post_with_retry(fallback_payload)
                forced_tool_fallback = True
                self._forced_tool_choice_supported = False
        elif isinstance(tool_choice, dict) and response.status_code < 400:
            self._forced_tool_choice_supported = True
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            message = f"{exc}"
            if detail:
                message += f" | response: {detail}"
            response.close()
            raise requests.RequestException(message) from exc
        if getattr(response, "status_code", 200) >= 400:
            detail = response.text.strip()
            message = f"{response.status_code} Error"
            if detail:
                message += f" | response: {detail}"
            response.close()
            raise requests.RequestException(message)
        try:
            payload = response.json()
        except ValueError as exc:
            response.close()
            raise requests.RequestException("server returned invalid JSON") from exc
        if not isinstance(payload, dict):
            response.close()
            raise requests.RequestException("server returned a non-object response")
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            response.close()
            raise requests.RequestException("server returned no choices")
        candidate_scores = [self._score_candidate_choice(choice) for choice in choices]
        selected_candidate_index = max(
            range(len(choices)),
            key=lambda index: (candidate_scores[index][0], -index),
        )
        fallback_candidate_indices = sorted(
            (
                index
                for index, (_score, valid) in enumerate(candidate_scores)
                if valid
                and index != selected_candidate_index
                and _choice_has_executable_python_tool(choices[index])
            ),
            key=lambda index: (-candidate_scores[index][0], index),
        )
        choice = choices[selected_candidate_index]
        if not isinstance(choice, dict):
            response.close()
            raise requests.RequestException("server returned an invalid choice")
        message = choice.get("message")
        if not isinstance(message, dict):
            response.close()
            raise requests.RequestException(
                "server returned a choice without a message"
            )
        usage = payload.get("usage")
        usage_payload = dict(usage) if isinstance(usage, dict) else {}
        has_generated_usage = False
        for usage_key in ("completion_tokens", "output_tokens", "generated_tokens"):
            try:
                if usage_key in usage_payload and int(usage_payload[usage_key]) >= 0:
                    has_generated_usage = True
                    break
            except (TypeError, ValueError):
                continue
        if not has_generated_usage:
            estimated_generated_tokens = sum(
                max(
                    1,
                    (
                        len(
                            json.dumps(
                                candidate.get("message") or {},
                                ensure_ascii=False,
                                separators=(",", ":"),
                                default=str,
                            )
                        )
                        + 3
                    )
                    // 4,
                )
                for candidate in choices
                if isinstance(candidate, dict)
            )
            usage_payload["generated_tokens"] = max(1, estimated_generated_tokens)
        return _ChatCompletionResult(
            message=message,
            finish_reason=str(choice.get("finish_reason", "") or ""),
            usage=usage_payload,
            latency_seconds=time.monotonic() - started_at,
            request_attempts=request_attempts,
            forced_tool_fallback=forced_tool_fallback,
            candidate_count=len(choices),
            selected_candidate_index=selected_candidate_index,
            valid_candidate_count=sum(1 for _score, valid in candidate_scores if valid),
            fallback_messages=[
                json.loads(json.dumps(choices[index].get("message") or {}))
                for index in fallback_candidate_indices
                if isinstance(choices[index], dict)
                and isinstance(choices[index].get("message"), dict)
            ],
        )

    def _trim_tool_text(self, text: str) -> tuple[str, bool]:
        if len(text) <= self._tool_output_chars:
            return text, False
        omitted = len(text) - self._tool_output_chars
        return (
            f"{text[: self._tool_output_chars]}\n... [truncated {omitted} chars]",
            True,
        )

    def _summarize_planned_actions(self, value: Any) -> Any:
        if isinstance(value, dict):
            compacted = {
                key: self._summarize_planned_actions(item)
                for key, item in value.items()
            }
            planned_actions = compacted.pop("planned_actions", None)
            if isinstance(planned_actions, list):
                compacted["planned_action_count"] = len(planned_actions)
                action_result = compacted.get("action_result")
                if isinstance(action_result, dict):
                    executed_count = action_result.get("executed_count")
                    try:
                        compacted["executed_action_count"] = int(executed_count)
                    except (TypeError, ValueError):
                        compacted["executed_action_count"] = (
                            1 if action_result.get("executed") else 0
                        )
            return compacted
        if isinstance(value, list):
            return [self._summarize_planned_actions(item) for item in value]
        return value

    def _render_tool_payload(
        self, payload: dict[str, Any], *, truncate_fields: tuple[str, ...] = ()
    ) -> str:
        result = self._summarize_planned_actions(dict(payload))
        truncated = False
        for field in truncate_fields:
            value = result.get(field)
            if isinstance(value, str):
                result[field], field_truncated = self._trim_tool_text(value)
                truncated = truncated or field_truncated
        if truncated:
            result["truncated"] = True
            result["truncation_note"] = (
                f"Tool output was cut off to stay within the ~{self._tool_output_tokens}-token response budget."
            )
        return json.dumps(result, ensure_ascii=True, separators=(",", ":"))

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, dict):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")
        if len(items) > MAX_ACTION_BATCH:
            raise ValueError(
                f"action(actions) accepts at most {MAX_ACTION_BATCH} actions per batch."
            )

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                engine_name = to_engine_action(action_name)
                if engine_name is None:
                    raise ValueError(f"Action {index} ({action_name}) is unknown.")
                canonical_name = to_model_action(engine_name)
                if (
                    index == 1
                    and self._current_valid_actions
                    and canonical_name not in self._current_valid_actions
                ):
                    raise ValueError(
                        f"Action {index} ({canonical_name}) is not currently valid; "
                        f"choose one of {self._current_valid_actions}."
                    )
                normalized.append({"action": canonical_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                engine_name = to_engine_action(action_name)
                if engine_name is None:
                    raise ValueError(f"Action {index} ({action_name}) is unknown.")
                canonical_name = to_model_action(engine_name)
                if (
                    index == 1
                    and self._current_valid_actions
                    and canonical_name not in self._current_valid_actions
                ):
                    raise ValueError(
                        f"Action {index} ({canonical_name}) is not currently valid; "
                        f"choose one of {self._current_valid_actions}."
                    )
                entry = {"action": canonical_name}
                if canonical_name == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if canonical_name == "MOUSE":
                    if "row" not in item or "col" not in item:
                        raise ValueError(
                            f"Action {index} MOUSE requires integer `row` and `col` fields."
                        )
                    for coordinate in ("row", "col"):
                        value = item.get(coordinate)
                        if isinstance(value, bool) or not isinstance(value, int):
                            raise ValueError(
                                f"Action {index} MOUSE `{coordinate}` must be an integer."
                            )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        if normalized and self._is_known_harmful_action(normalized[0]):
            raise ValueError(
                "The first action has observed terminal-failure or negative-reward "
                "evidence in the current state. Choose a safer probe or split the "
                "batch so the action can be reevaluated after the state changes."
            )
        return normalized

    def _is_known_harmful_action(self, action: dict[str, Any]) -> bool:
        snapshot = self._current_experience_snapshot or {}
        name = str(action.get("action") or "")
        key = (
            f"MOUSE(ROW={int(action['row'])}, COL={int(action['col'])})"
            if name == "MOUSE" and "row" in action and "col" in action
            else normalize_action_key(name)
        )
        for model in snapshot.get("transition_models_here") or []:
            if not isinstance(model, dict):
                continue
            harmful_trials = int(model.get("terminal_failures") or 0)
            total_trials = max(harmful_trials, int(model.get("trials") or 0))
            if normalize_action_key(model.get("action") or "") == key and (
                harmful_evidence_is_decisive(harmful_trials, total_trials)
            ):
                return True
        for evidence in (snapshot.get("cross_trial") or {}).get(
            "state_action_evidence", []
        ):
            if not isinstance(evidence, dict):
                continue
            outcomes = evidence.get("outcomes") or {}
            harmful_trials = int(outcomes.get("terminal_failure") or 0) + int(
                outcomes.get("negative_reward") or 0
            )
            total_trials = max(
                harmful_trials,
                int(evidence.get("trials") or 0),
                sum(int(count or 0) for count in outcomes.values()),
            )
            if normalize_action_key(evidence.get("action") or "") == key and (
                harmful_evidence_is_decisive(harmful_trials, total_trials)
            ):
                return True
        return False

    def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "executed": bool(payload.get("executed")),
            "action_num": payload.get("action_num"),
            "level": payload.get("level"),
            "score": payload.get("score"),
            "reward": payload.get("reward"),
            "state": payload.get("state"),
            "valid_actions": payload.get("valid_actions", []),
            "board_changed": bool(payload.get("board_changed")),
            "decision_context_changed": bool(payload.get("decision_context_changed")),
            "done": bool(payload.get("done")),
            "level_completed": bool(payload.get("level_completed")),
            "game_over": bool(payload.get("game_over")),
            "run_complete": bool(payload.get("run_complete")),
            "action_display": payload.get("action_display")
            or payload.get("action_name"),
        }
        for controller_key in (
            "guarded",
            "guard_reason_code",
            "loop_detected",
            "cycle_risk",
            "cycle_period",
            "controller_policy",
            "controller_phase",
            "controller_reason_codes",
            "no_op_streak",
            "behavioral_no_op_streak",
            "stagnation_actions",
            "before_state_id",
            "after_state_id",
            "behavioral_before_state_id",
            "behavioral_after_state_id",
            "object_before_state_id",
            "object_after_state_id",
            "novel_state",
            "outcome_class",
            "action_rank",
            "action_rank_reason",
            "prediction_result",
            "animation",
        ):
            if controller_key in payload:
                compact[controller_key] = payload.get(controller_key)
        steps = payload.get("steps")
        if isinstance(steps, list):
            compact["steps"] = [
                dict(item) for item in steps[:12] if isinstance(item, dict)
            ]
        executed_actions = payload.get("executed_actions")
        if isinstance(executed_actions, list) and executed_actions:
            compact["executed_actions"] = [
                str(action).strip()
                for action in executed_actions[:MAX_ACTION_BATCH]
                if str(action).strip()
            ]
        elif compact.get("action_display"):
            compact["executed_actions"] = [str(compact["action_display"]).strip()]
        try:
            batch_size = min(
                MAX_ACTION_BATCH,
                max(
                    1,
                    int(
                        payload.get("requested_count")
                        or payload.get("executed_count")
                        or 1
                    ),
                ),
            )
        except (TypeError, ValueError):
            batch_size = 1
        if batch_size > 1 or bool(payload.get("stopped_early")):
            compact["requested_count"] = batch_size
            if not compact["executed"]:
                compact["executed_count"] = 0
            else:
                try:
                    compact["executed_count"] = min(
                        batch_size,
                        max(0, int(payload.get("executed_count", batch_size))),
                    )
                except (TypeError, ValueError):
                    compact["executed_count"] = 1
            compact["stopped_early"] = bool(payload.get("stopped_early"))
        if payload.get("stop_reason"):
            compact["stop_reason"] = payload.get("stop_reason")
        if payload.get("stop_detail"):
            compact["stop_detail"] = payload.get("stop_detail")
        for timing_key in ("run_elapsed_seconds", "time_remaining_seconds"):
            if timing_key in payload:
                compact[timing_key] = payload.get(timing_key)
        if payload.get("error"):
            compact["error"] = payload.get("error")
        return compact

    @staticmethod
    def _engine_transition_observations(
        action_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expand action-call aggregates into independently learnable steps."""
        observations: list[dict[str, Any]] = []
        for result in action_results:
            steps = result.get("steps")
            if isinstance(steps, list) and steps:
                for raw_step in steps:
                    if not isinstance(raw_step, dict):
                        continue
                    step = dict(raw_step)
                    step["executed"] = True
                    observations.append(step)
            elif result.get("executed"):
                observations.append(dict(result))
        return observations

    def _run_python_tool(
        self, state_path: Path, arguments: dict[str, Any]
    ) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        code = _normalize_generated_python_code(arguments.get("code", ""))
        if not code:
            return _ToolDispatchResult(
                json.dumps(
                    {"error": "python requires a non-empty `code` string."},
                    separators=(",", ":"),
                )
            )
        try:
            tree = _parse_bounded_generated_python(code)
            compile(tree, "<python_tool>", "exec")
            preflight_issues = _generated_python_preflight_issues(tree)
            for static_actions in _static_candidate_action_arguments(tree)[:1]:
                try:
                    self._normalize_python_actions(static_actions)
                except (TypeError, ValueError) as exc:
                    preflight_issues.append(
                        {
                            "line": None,
                            "message": str(exc),
                            "hint": (
                                "Use only currently valid actions and respect the "
                                "current controller action budget."
                            ),
                        }
                    )
            if preflight_issues:
                first_issue = preflight_issues[0]
                raise _GeneratedCodePreflightError(
                    str(first_issue.get("message") or "Generated-code preflight failed.")
                )
        except _GeneratedCodeLimitError as exc:
            payload = _python_tool_payload(
                {
                    "error": f"Python validation error: {exc}",
                    "diagnostic": {
                        "type": "GeneratedCodeLimitError",
                        "line": None,
                        "column": None,
                        "source": None,
                        "hint": "Use a smaller bounded program and retry.",
                        "retry": "correct_and_retry",
                    },
                    "stdout": "",
                    "action_results": [],
                }
            )
            return _ToolDispatchResult(
                self._render_tool_payload(payload, truncate_fields=("error",))
            )
        except _GeneratedCodePreflightError as exc:
            issue = preflight_issues[0] if preflight_issues else {}
            line = issue.get("line")
            code_lines = code.splitlines()
            source = (
                code_lines[line - 1].strip()[:240]
                if isinstance(line, int) and 1 <= line <= len(code_lines)
                else None
            )
            payload = _python_tool_payload(
                {
                    "error": f"Python preflight error: {exc}",
                    "diagnostic": {
                        "type": "GeneratedCodePreflightError",
                        "line": line,
                        "column": None,
                        "source": source,
                        "context": [],
                        "suggestions": [],
                        "hint": str(issue.get("hint") or "Correct the program and retry."),
                        "retry": "correct_and_retry",
                    },
                    "stdout": "",
                    "action_results": [],
                }
            )
            return _ToolDispatchResult(
                self._render_tool_payload(payload, truncate_fields=("error",))
            )
        except SyntaxError as exc:
            payload = _python_tool_payload(
                {
                    "error": f"Python syntax error: {exc.msg}",
                    "diagnostic": _sandbox_exception_diagnostic(exc, code),
                    "stdout": "",
                    "action_results": [],
                }
            )
            return _ToolDispatchResult(
                self._render_tool_payload(
                    payload,
                    truncate_fields=("error",),
                )
            )

        current_frame, history_entries = load_runtime_state(state_path)

        def _serialized_runtime_state(
            *,
            next_valid_actions: list[str] | None = None,
            last_action_result: dict[str, Any] | None = None,
            runtime_state: tuple[Frame | None, list[HistoryEntry]] | None = None,
        ) -> dict[str, Any]:
            if runtime_state is None:
                refreshed_frame, refreshed_history = load_runtime_state(state_path)
            else:
                refreshed_frame, refreshed_history = runtime_state
            current_frame_payload, history_payload = self._cached_frame_payloads(
                refreshed_frame,
                refreshed_history,
            )
            if isinstance(next_valid_actions, list):
                sanitized_actions = [
                    str(item).strip()
                    for item in next_valid_actions
                    if str(item).strip()
                ]
            else:
                sanitized_actions = list(self._current_valid_actions)
            persisted_action_result = (
                last_action_result
                if isinstance(last_action_result, dict)
                else self._last_action_result
            )
            return {
                "current_frame": _current_frame_transport_payload(
                    refreshed_frame,
                    refreshed_history,
                    current_frame_payload,
                ),
                "history": history_payload,
                "valid_actions": sanitized_actions,
                "experience": self._cached_experience_snapshot(
                    refreshed_frame,
                    refreshed_history,
                    sanitized_actions,
                ),
                "strategy": dict(self._strategy_memory),
                "memory": dict(self._python_memory),
                "last_action_result": (
                    dict(persisted_action_result)
                    if isinstance(persisted_action_result, dict)
                    else {}
                ),
            }

        terminal_action_result: dict[str, Any] | None = None

        def _handle_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal terminal_action_result
            if self._step_env_callback is None:
                raise RuntimeError("action(actions) is not available in this session.")
            try:
                normalized_actions = self._normalize_python_actions(actions)
            except (TypeError, ValueError) as exc:
                raise SandboxHostActionError(str(exc)) from exc
            live_frame, _live_history = load_runtime_state(state_path)
            if live_frame is not None:
                rows, cols = live_frame.shape
                for index, action in enumerate(normalized_actions, start=1):
                    if action.get("action") != "MOUSE":
                        continue
                    row = int(action["row"])
                    col = int(action["col"])
                    if not (0 <= row < rows and 0 <= col < cols):
                        raise SandboxHostActionError(
                            f"Action {index} MOUSE coordinate ({row}, {col}) is outside "
                            f"the current frame shape {rows}x{cols}."
                        )
            if terminal_action_result is not None:
                reason = (
                    _terminal_action_reason(terminal_action_result) or "terminal_state"
                )
                compact_payload = {
                    "executed": False,
                    "action_num": terminal_action_result.get("action_num"),
                    "level": terminal_action_result.get("level"),
                    "score": terminal_action_result.get("score"),
                    "reward": 0.0,
                    "state": terminal_action_result.get("state"),
                    "valid_actions": [],
                    "board_changed": False,
                    "done": bool(terminal_action_result.get("done")),
                    "level_completed": bool(
                        terminal_action_result.get("level_completed")
                    ),
                    "game_over": bool(terminal_action_result.get("game_over")),
                    "run_complete": bool(terminal_action_result.get("run_complete")),
                    "requested_count": len(normalized_actions),
                    "executed_count": 0,
                    "stopped_early": True,
                    "stop_reason": f"previous_{reason}",
                    "stop_detail": _terminal_action_stop_detail(reason),
                }
                self._last_action_result = dict(compact_payload)
                return {
                    "action_result": compact_payload,
                    "state": _serialized_runtime_state(
                        next_valid_actions=[],
                        last_action_result=compact_payload,
                    ),
                }
            raw_payload = self._step_env_callback(
                {
                    "actions": normalized_actions,
                    "strategy_prediction": {
                        key: self._strategy_memory.get(key)
                        for key in ("test_action", "expected_outcome")
                        if self._strategy_memory.get(key)
                    },
                }
            )
            if not isinstance(raw_payload, dict):
                raise RuntimeError(
                    "action(actions) did not return a JSON-like payload."
                )
            compact_payload = self._compact_action_result(raw_payload)
            prediction_result = self._evaluate_strategy_prediction(compact_payload)
            if prediction_result is not None:
                compact_payload["prediction_result"] = prediction_result
            next_valid_actions = raw_payload.get("valid_actions")
            if isinstance(next_valid_actions, list):
                self._current_valid_actions = _normalize_valid_actions(
                    next_valid_actions
                )
            if compact_payload.get("executed") and _terminal_action_reason(
                compact_payload
            ):
                terminal_action_result = compact_payload
            self._last_action_result = dict(compact_payload)
            return {
                "action_result": compact_payload,
                "state": _serialized_runtime_state(
                    next_valid_actions=next_valid_actions
                    if isinstance(next_valid_actions, list)
                    else None,
                    last_action_result=compact_payload,
                ),
            }

        state_started_at = time.monotonic()
        initial_state = _serialized_runtime_state(
            runtime_state=(current_frame, history_entries)
        )
        self._record_efficiency(
            "state_build_seconds",
            time.monotonic() - state_started_at,
        )
        self._record_efficiency(
            "initial_state_bytes",
            len(json.dumps(initial_state, ensure_ascii=False, separators=(",", ":"))),
        )
        self._record_efficiency("generated_code_chars", len(code))
        sandbox_started_at = time.monotonic()
        sandbox_result = run_sandboxed_python(
            code=code,
            timeout_seconds=self._python_timeout,
            initial_state=initial_state,
            action_handler=_handle_action,
            strategy_handler=self._record_strategy,
            memory_handler=self._record_python_memory,
            should_stop=self._should_stop_callback,
        )
        sandbox_calls = 1
        first_diagnostic = sandbox_result.get("diagnostic")
        first_action_results = [
            item
            for item in sandbox_result.get("action_results") or []
            if isinstance(item, dict)
        ]
        repaired_code = (
            _deterministic_generated_python_repair(code, first_diagnostic)
            if sandbox_result.get("error")
            and not first_action_results
            and not re.search(r"\b(?:record_strategy|remember|forget)\s*\(", code)
            and isinstance(first_diagnostic, dict)
            else None
        )
        if repaired_code is not None:
            repaired_result = run_sandboxed_python(
                code=repaired_code,
                timeout_seconds=self._python_timeout,
                initial_state=initial_state,
                action_handler=_handle_action,
                strategy_handler=self._record_strategy,
                memory_handler=self._record_python_memory,
                should_stop=self._should_stop_callback,
            )
            sandbox_calls += 1
            repaired_result["auto_repair"] = {
                "applied": True,
                "original_diagnostic_type": first_diagnostic.get("type"),
                "original_source": first_diagnostic.get("source"),
                "repaired_code_fingerprint": hashlib.blake2b(
                    repaired_code.encode("utf-8"), digest_size=12
                ).hexdigest(),
            }
            sandbox_result = repaired_result
            self._record_efficiency("automatic_code_repairs", 1)
        self._record_efficiency("sandbox_calls", sandbox_calls)
        self._record_efficiency(
            "sandbox_seconds",
            time.monotonic() - sandbox_started_at,
        )
        sandbox_efficiency = sandbox_result.get("efficiency")
        if isinstance(sandbox_efficiency, dict):
            if sandbox_efficiency.get("prewarmed"):
                self._record_efficiency("sandbox_prewarmed_calls", 1)
            try:
                self._record_efficiency(
                    "sandbox_transport_bytes",
                    int(sandbox_efficiency.get("host_to_sandbox_bytes") or 0),
                )
            except (TypeError, ValueError):
                pass

        action_results = [
            item
            for item in sandbox_result.get("action_results") or []
            if isinstance(item, dict)
        ]
        transition_observations = self._engine_transition_observations(action_results)
        if self._knowledge_store is not None and self._knowledge_game_id:
            for action_result in transition_observations:
                self._knowledge_store.observe(
                    self._knowledge_game_id,
                    action_result,
                    strategy=self._strategy_memory,
                    pass_index=self._knowledge_pass_index,
                    evidence_id=self._knowledge_evidence_id,
                )
        self._update_causal_model_from_actions(transition_observations)
        payload = _python_tool_payload(sandbox_result)

        step_executed = any(bool(item.get("executed")) for item in action_results)
        if step_executed:
            self._last_step_summary = self._summarize_step_sequence(action_results)
            self._update_summarized_knowledge_from_step_summary()
        return _ToolDispatchResult(
            self._render_tool_payload(
                payload, truncate_fields=("stdout", "error", "result")
            ),
            step_executed=step_executed,
        )

    def _dispatch_tool(
        self, state_path: Path, name: str, arguments: dict[str, Any]
    ) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        if name == "python":
            return self._run_python_tool(state_path, arguments)
        return _ToolDispatchResult(
            json.dumps({"error": f"Unknown tool: {name}"}, separators=(",", ":"))
        )

    def _estimate_request_input_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        exact = self._exact_request_tokens(messages, tools)
        if exact is not None:
            return exact
        payload: dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _request_tool_choice(tools)
        return _estimate_tokens(payload)

    def _drop_oldest_history_block(
        self, history: list[dict[str, Any]], *, preserve_recent: int
    ) -> bool:
        removable = len(history) - preserve_recent
        if removable <= 0:
            return False
        first = history.pop(0)
        first_role = str(first.get("role", "")).strip()
        if first_role in {"assistant", "tool"}:
            while (
                history
                and history[0].get("role") == "tool"
                and len(history) > preserve_recent
            ):
                history.pop(0)
            return True
        while (
            history
            and history[0].get("role") == "tool"
            and len(history) > preserve_recent
        ):
            history.pop(0)
        while (
            history
            and history[0].get("role") != "user"
            and len(history) > preserve_recent
        ):
            history.pop(0)
        return True

    def _keep_recent_history_turns(
        self,
        messages: list[dict[str, Any]],
        *,
        max_turns: int,
    ) -> list[dict[str, Any]]:
        if max_turns <= 0 or not messages:
            return []

        kept_reversed: list[dict[str, Any]] = []
        assistant_turns = 0
        for message in reversed(messages):
            kept_reversed.append(message)
            if str(message.get("role", "")).strip() == "assistant":
                assistant_turns += 1
                if assistant_turns >= max_turns:
                    break

        kept = list(reversed(kept_reversed))
        while kept and str(kept[0].get("role", "")).strip() == "tool":
            kept.pop(0)
        return kept

    def _drop_until_first_user_message(
        self, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        trimmed = list(history)
        while trimmed and str(trimmed[0].get("role", "")).strip() != "user":
            trimmed.pop(0)
        return trimmed

    def _persistent_history_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        request_base_chars: int | None = None,
        message_length_cache: dict[int, tuple[dict[str, Any], int]] | None = None,
    ) -> list[dict[str, Any]]:
        trimmed = self._trim_messages_for_context(
            messages,
            tools=tools,
            request_base_chars=request_base_chars,
            message_length_cache=message_length_cache,
        )
        if not trimmed:
            return []
        trimmed_history = trimmed[1:]
        history = self._keep_recent_history_turns(
            trimmed_history,
            max_turns=_PERSISTENT_HISTORY_ASSISTANT_TURNS,
        )
        if (
            history
            and str(history[0].get("role", "")).strip() != "user"
            and len(trimmed_history) > len(history)
        ):
            previous_message = trimmed_history[len(trimmed_history) - len(history) - 1]
            if str(previous_message.get("role", "")).strip() == "user":
                history = [previous_message, *history]
        return self._drop_until_first_user_message(history)

    def _trim_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
        request_base_chars: int | None = None,
        message_length_cache: dict[int, tuple[dict[str, Any], int]] | None = None,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        preserve_recent = max(0, preserve_recent)
        budget_tokens = max(
            1, self._context_budget_tokens - max(0, extra_safety_tokens)
        )
        request_chars = (
            _estimated_request_base_length(tools)
            if request_base_chars is None
            else request_base_chars
        )

        def message_chars(message: dict[str, Any]) -> int:
            if message_length_cache is None:
                return _estimated_json_length(message)
            cache_key = id(message)
            cached = message_length_cache.get(cache_key)
            if cached is not None and cached[0] is message:
                return cached[1]
            length = _estimated_json_length(message)
            message_length_cache[cache_key] = (message, length)
            return length

        system_chars = message_chars(system_message)
        history_chars = [message_chars(message) for message in history]
        history_chars_total = sum(history_chars)

        def estimated_tokens() -> int:
            exact = self._exact_request_tokens(
                [system_message, *history],
                tools,
            )
            if exact is not None:
                return exact
            # Replacing the payload's empty `[]` adds each rendered message plus
            # `, ` between adjacent items. The system message is always retained.
            rendered_chars = (
                request_chars
                + system_chars
                + history_chars_total
                + 2 * len(history_chars)
            )
            return max(1, (rendered_chars + 2) // 3)

        while history and estimated_tokens() > budget_tokens:
            previous_length = len(history)
            if not self._drop_oldest_history_block(
                history, preserve_recent=preserve_recent
            ):
                break
            removed_count = previous_length - len(history)
            history_chars_total -= sum(history_chars[:removed_count])
            del history_chars[:removed_count]
        history = self._drop_until_first_user_message(history)
        return [system_message, *history]

    def _force_reduce_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_recent: int = 1,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        if not self._drop_oldest_history_block(
            history, preserve_recent=max(0, preserve_recent)
        ):
            return list(messages)
        return [system_message, *history]

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
    ) -> AnalyzerTurnResult | None:
        if not state_path.exists():
            return AnalyzerTurnResult(
                step_executed=False,
                failure_category="state_missing",
                failure_detail=f"Runtime state does not exist: {state_path}",
            )
        self._ensure_session(state_path)
        self._step_env_callback = step_env
        self._current_valid_actions = _normalize_valid_actions(valid_actions)
        self._turn_efficiency_metrics = {}

        analyzer_log = transcript_path or (
            state_path.parent / f"{state_path.stem}_analyzer.txt"
        )
        prompt_log = _resolve_prompt_log_path(state_path)
        current_frame, history_entries = load_runtime_state(state_path)
        cross_trial = None
        if self._knowledge_store is not None and self._knowledge_game_id:
            cross_trial = self._knowledge_store.snapshot(self._knowledge_game_id)
        experience_snapshot = build_experience_snapshot(
            history_entries,
            current_frame,
            self._current_valid_actions,
            self._controller_config,
            external_transitions=(cross_trial or {}).get("transition_records"),
            evidence_id=self._knowledge_evidence_id,
        )
        if cross_trial is not None:
            experience_snapshot = dict(experience_snapshot)
            experience_snapshot["cross_trial"] = self._knowledge_store.snapshot(
                self._knowledge_game_id,
                state_id=str(experience_snapshot.get("state_id") or ""),
            )
        if self._strategy_memory.get("causal_model"):
            experience_snapshot["causal_model"] = normalize_causal_model(
                self._strategy_memory["causal_model"]
            )
        self._current_experience_snapshot = experience_snapshot
        user_prompt = self._build_user_prompt(
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=history_entries,
            previous_step_summary=self._last_step_summary,
            experience_snapshot=experience_snapshot,
        )
        display_action_num = _display_action_number(action_num)

        step_label = (
            f"analysis_step={analysis_step} | " if analysis_step is not None else ""
        )
        transcript_header = (
            f"\n--- {step_label}action={display_action_num} | "
            f"{time.strftime('%H:%M:%S')} | tool-agent ---\n"
        )
        transcript_buffer = _TranscriptBuffer(analyzer_log, transcript_header)

        def append_transcript(label: str, content: str) -> None:
            section = transcript_buffer.append(label, content)
            if section and transcript_updated is not None:
                try:
                    transcript_updated(transcript_buffer.render())
                except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
                    log.warning("analyzer transcript callback failed: %s", exc)

        append_transcript("SYSTEM PROMPT", self._system_prompt)
        append_transcript("USER PROMPT", user_prompt)

        previous_history_messages = list(self._history_messages)
        preserve_history = True
        tools = self._tools(state_path)
        prewarm_sandbox()
        tool_choice = _request_tool_choice(tools)
        request_tools_snapshot = tools
        request_base_chars = _estimated_request_base_length(tools)
        message_length_cache: dict[int, tuple[dict[str, Any], int]] = {}
        messages: list[dict[str, Any]] = self._trim_messages_for_context(
            [
                {"role": "system", "content": self._system_prompt},
                *self._history_messages,
                self._build_user_message(
                    user_prompt,
                    current_frame,
                    history_entries[-2].frame if len(history_entries) >= 2 else None,
                ),
            ],
            tools=tools,
            preserve_recent=1,
            request_base_chars=request_base_chars,
            message_length_cache=message_length_cache,
        )
        step_executed = False
        captured_reasoning = ""
        latest_request_messages: list[dict[str, Any]] | None = None
        latest_request_tools: list[dict[str, Any]] | None = None
        latest_request_tool_choice: Any = None
        latest_request_index = 0
        turn_started_at = time.monotonic()
        request_deadline = (
            turn_started_at + max(0.0, float(request_timeout_seconds))
            if request_timeout_seconds is not None
            else None
        )
        yielded_control_reason: str | None = None
        repair_next_request = False
        queued_candidate_messages: list[dict[str, Any]] = []
        failed_code_fingerprints: set[str] = set()
        failure_counts: dict[str, int] = {}
        escalated_failures: set[str] = set()

        def control_yield_reason() -> str | None:
            if should_stop is not None:
                try:
                    if should_stop():
                        return "stop_requested"
                except Exception as exc:
                    log.warning(
                        "analyzer stop check failed at action %d: %s",
                        display_action_num,
                        exc,
                    )
            if (
                self._yield_seconds is not None
                and (time.monotonic() - turn_started_at) >= self._yield_seconds
            ):
                return "turn_time_budget"
            remaining_request_time = _remaining_deadline_seconds(request_deadline)
            if remaining_request_time is not None and remaining_request_time <= 0:
                return "request_time_budget"
            if self._remaining_game_tokens() == 0:
                return "game_token_budget"
            return None

        self._should_stop_callback = should_stop
        try:
            turn_count = 0
            while self._tool_steps is None or turn_count < self._tool_steps:
                yielded_control_reason = control_yield_reason()
                if yielded_control_reason is not None:
                    break
                turn_count += 1
                messages = self._trim_messages_for_context(
                    messages,
                    tools=tools,
                    request_base_chars=request_base_chars,
                    message_length_cache=message_length_cache,
                )
                latest_request_messages = list(messages)
                latest_request_tools = request_tools_snapshot
                latest_request_tool_choice = tool_choice
                latest_request_index = turn_count
                try:
                    request_kwargs: dict[str, Any] = {
                        "tools": tools,
                        "max_output_tokens": self._adaptive_output_limit(
                            turn_count,
                            repair=repair_next_request,
                        ),
                    }
                    repair_next_request = False
                    remaining_request_time = _remaining_deadline_seconds(
                        request_deadline
                    )
                    if remaining_request_time is not None:
                        request_kwargs["request_timeout_seconds"] = max(
                            0.1, remaining_request_time
                        )
                    if self._save_request_logs:
                        _safe_append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            messages=latest_request_messages,
                            tools=latest_request_tools,
                            event="request",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                        )
                    used_candidate_fallback = bool(queued_candidate_messages)
                    if used_candidate_fallback:
                        fallback_message = queued_candidate_messages.pop(0)
                        result = _ChatCompletionResult(
                            message=fallback_message,
                            finish_reason="candidate_fallback",
                            request_attempts=0,
                            candidate_count=1,
                            valid_candidate_count=1,
                        )
                        append_transcript(
                            "ANALYZER STATUS",
                            "candidate_fallback: trying the next statically valid "
                            "candidate without another model request.",
                        )
                    else:
                        result = self._chat_completion(messages, **request_kwargs)
                    self._record_efficiency("model_calls", result.request_attempts)
                    self._record_efficiency("model_seconds", result.latency_seconds)
                    self._record_efficiency("model_candidates", result.candidate_count)
                    self._record_efficiency(
                        "valid_model_candidates", result.valid_candidate_count
                    )
                    self._record_efficiency(
                        "selected_candidate_index", result.selected_candidate_index
                    )
                    if result.forced_tool_fallback:
                        self._record_efficiency("forced_tool_fallbacks", 1)
                    self._accumulate_usage_tokens(result.usage)
                    if self._save_request_logs:
                        _safe_append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            messages=latest_request_messages,
                            tools=latest_request_tools,
                            event="response",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                            finish_reason=result.finish_reason,
                        )
                except requests.RequestException as exc:
                    if not _is_context_length_error(exc):
                        if self._activate_next_fallback_model():
                            self._record_efficiency("model_failovers", 1)
                            append_transcript(
                                "ANALYZER STATUS",
                                f"model_failover: retrying with {self._model.model_id} after {exc}",
                            )
                            continue
                        raise
                    trimmed_messages = self._trim_messages_for_context(
                        messages,
                        tools=tools,
                        extra_safety_tokens=_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS,
                        request_base_chars=request_base_chars,
                        message_length_cache=message_length_cache,
                    )
                    if trimmed_messages == messages:
                        trimmed_messages = self._force_reduce_messages(messages)
                    if trimmed_messages == messages:
                        raise
                    append_transcript(
                        "ANALYZER STATUS",
                        "context_overflow_recovered: dropped older history after server rejected the request as too long.",
                    )
                    messages = trimmed_messages
                    continue
                raw_reasoning = _extract_reasoning_text(result.message)
                raw_content = _normalize_message_content(
                    result.message.get("content", "")
                )
                tool_calls = self._normalize_response_tool_calls(
                    json.loads(json.dumps(result.message.get("tool_calls") or []))
                )
                tool_call_markup_in_text = _contains_tool_call_markup(
                    raw_reasoning, raw_content
                )
                recovered_tool_calls_from_markup = False
                if not tool_calls and tool_call_markup_in_text:
                    tool_calls = self._normalize_response_tool_calls(
                        _recover_tool_calls_from_markup(raw_reasoning, raw_content)
                    )
                    recovered_tool_calls_from_markup = bool(tool_calls)
                reasoning = (
                    _strip_tool_call_markup(raw_reasoning)
                    if tool_call_markup_in_text
                    else raw_reasoning
                )
                content = (
                    _strip_tool_call_markup(raw_content)
                    if tool_call_markup_in_text
                    else raw_content
                )
                normalized_tool_arguments: list[dict[str, Any] | None] = []
                tool_argument_errors: list[str | None] = []
                malformed_argument_errors: list[str] = []
                response_fallback_messages = list(result.fallback_messages)
                for tool_call in tool_calls:
                    function = (
                        tool_call.get("function", {})
                        if isinstance(tool_call, dict)
                        else {}
                    )
                    tool_name = str(function.get("name", "")).strip() or "unknown"
                    raw_arguments = function.get("arguments", "{}")
                    try:
                        normalized_tool_arguments.append(
                            _normalize_tool_call_arguments(raw_arguments)
                        )
                        tool_argument_errors.append(None)
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        issue = f"{tool_name}: invalid arguments ({exc})"
                        normalized_tool_arguments.append(None)
                        tool_argument_errors.append(
                            f"Invalid arguments for tool {tool_name}: {exc}. "
                            "Provide one JSON object and retry."
                        )
                        malformed_argument_errors.append(issue)
                response_meta = _format_model_response_meta(
                    finish_reason=result.finish_reason,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_markup_in_text=tool_call_markup_in_text,
                    recovered_tool_calls_from_markup=recovered_tool_calls_from_markup,
                    malformed_argument_errors=malformed_argument_errors,
                )
                append_transcript(
                    "MODEL RESPONSE META",
                    response_meta,
                )
                assistant_message: dict[str, Any] = {"role": "assistant"}

                if reasoning:
                    captured_reasoning = reasoning
                    append_transcript("THINKING", reasoning)
                    assistant_message["reasoning"] = reasoning

                if not tool_calls:
                    if content:
                        self._update_summarized_knowledge_from_assistant(content)
                        append_transcript("ASSISTANT", content)
                        assistant_message["content"] = content
                    elif reasoning:
                        assistant_message["content"] = None

                    if content or reasoning:
                        messages.append(assistant_message)
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        break
                    followup_prefix = "No action was executed. "
                    if tool_call_markup_in_text:
                        followup_prefix = (
                            "Tool markup appeared as text and was not executable. "
                        )
                    followup_prompt = (
                        f"{followup_prefix}"
                        "Emit exactly one parsed `python` tool call now. Combine useful inspections in one snippet, "
                        "update the world model if needed, and call `action(actions)` with the selected valid action or batch. "
                        f"{TOOL_CALL_FORMAT_GUIDANCE}"
                    )
                    append_transcript("USER PROMPT", followup_prompt)
                    messages.append({"role": "user", "content": followup_prompt})
                    repair_next_request = True
                    continue

                if content:
                    self._update_summarized_knowledge_from_assistant(content)
                    append_transcript("ASSISTANT", content)
                    assistant_message["content"] = content
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)

                repair_instruction: str | None = None
                for tool_index, tool_call in enumerate(tool_calls):
                    function = (
                        tool_call.get("function", {})
                        if isinstance(tool_call, dict)
                        else {}
                    )
                    tool_name = str(function.get("name", "")).strip()
                    raw_args = function.get("arguments", "{}")
                    normalized_arguments = normalized_tool_arguments[tool_index]
                    arguments = normalized_arguments or {}
                    argument_error = tool_argument_errors[tool_index]
                    generated_code = (
                        _normalize_generated_python_code(arguments.get("code", ""))
                        if tool_name == "python" and arguments
                        else ""
                    )
                    code_fingerprint = (
                        hashlib.blake2b(
                            generated_code.encode("utf-8"), digest_size=12
                        ).hexdigest()
                        if generated_code
                        else ""
                    )
                    rendered_tool_call = _render_tool_call_markup(
                        tool_name,
                        arguments if normalized_arguments is not None else raw_args,
                        arguments_normalized=normalized_arguments is not None,
                    )
                    append_transcript(
                        f"TOOL CALL: {tool_name}",
                        rendered_tool_call
                        or (json.dumps(arguments, indent=2) if arguments else "{}"),
                    )
                    if argument_error is not None:
                        repair_next_request = True
                        dispatch = _ToolDispatchResult(
                            json.dumps(
                                {
                                    "error": argument_error,
                                    "retryable": True,
                                },
                                separators=(",", ":"),
                            )
                        )
                    elif (
                        code_fingerprint
                        and code_fingerprint in failed_code_fingerprints
                    ):
                        dispatch = _ToolDispatchResult(
                            json.dumps(
                                {
                                    "error": (
                                        "Duplicate failed program suppressed before "
                                        "sandbox execution."
                                    ),
                                    "retryable": True,
                                    "diagnostic": {
                                        "type": "DuplicateGeneratedCodeError",
                                        "line": None,
                                        "column": None,
                                        "source": None,
                                        "hint": (
                                            "Produce a materially different program "
                                            "that addresses the prior diagnostic."
                                        ),
                                        "retry": "change_approach",
                                    },
                                },
                                separators=(",", ":"),
                            )
                        )
                    else:
                        dispatch = self._dispatch_tool(state_path, tool_name, arguments)
                    if dispatch.step_executed:
                        step_executed = True
                    try:
                        dispatch_payload = json.loads(dispatch.content)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        dispatch_payload = {}
                    if isinstance(dispatch_payload, dict) and dispatch_payload.get(
                        "error"
                    ):
                        diagnostic = dispatch_payload.get("diagnostic")
                        diagnostic = (
                            diagnostic if isinstance(diagnostic, dict) else {}
                        )
                        partial_execution = dispatch_payload.get("partial_execution")
                        side_effects_committed = bool(
                            isinstance(partial_execution, dict)
                            and partial_execution.get("side_effects_committed")
                        )
                        failure_material = "|".join(
                            (
                                str(diagnostic.get("type") or "ToolError"),
                                str(diagnostic.get("source") or ""),
                                str(dispatch_payload.get("error") or "")[:500],
                            )
                        )
                        failure_fingerprint = hashlib.blake2b(
                            failure_material.encode("utf-8"), digest_size=12
                        ).hexdigest()
                        failure_counts[failure_fingerprint] = (
                            failure_counts.get(failure_fingerprint, 0) + 1
                        )
                        if code_fingerprint and not side_effects_committed:
                            failed_code_fingerprints.add(code_fingerprint)
                        retry_directive = str(
                            diagnostic.get("retry")
                            or (
                                "correct_and_retry"
                                if dispatch_payload.get("retryable", True)
                                else "do_not_retry"
                            )
                        )
                        repeated = failure_counts[failure_fingerprint]
                        quality_failed_over = False
                        if (
                            repeated >= 2
                            and failure_fingerprint not in escalated_failures
                        ):
                            escalated_failures.add(failure_fingerprint)
                            if self._activate_next_fallback_model():
                                quality_failed_over = True
                                queued_candidate_messages.clear()
                                response_fallback_messages = []
                                self._record_efficiency("model_failovers", 1)
                                append_transcript(
                                    "ANALYZER STATUS",
                                    "quality_failover: repeated generated-code "
                                    f"failure; switched to {self._model.model_id}.",
                                )
                        if side_effects_committed:
                            repair_instruction = (
                                "The failed program committed environment actions. "
                                "Do not replay it. Re-ground from current_frame and "
                                "action_results on the next solver turn."
                            )
                        elif retry_directive == "do_not_retry":
                            yielded_control_reason = "tool_do_not_retry"
                        else:
                            repair_next_request = True
                            if response_fallback_messages and not quality_failed_over:
                                queued_candidate_messages.extend(
                                    response_fallback_messages
                                )
                                response_fallback_messages = []
                            repair_instruction = (
                                "Repair policy: address the structured diagnostic "
                                "directly and emit a materially different bounded "
                                "program."
                            )
                            if repeated >= 2:
                                repair_instruction += (
                                    " The same failure repeated; change approach "
                                    "instead of making a cosmetic edit."
                                )
                    append_transcript(
                        f"TOOL RESULT: {tool_name}",
                        _render_tool_result_display(dispatch.content),
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": dispatch.content,
                        }
                    )
                    if dispatch.step_executed:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                if yielded_control_reason is not None:
                    break
                if step_executed:
                    break
                if repair_instruction:
                    append_transcript("USER PROMPT", repair_instruction)
                    messages.append({"role": "user", "content": repair_instruction})

        except requests.RequestException as exc:
            append_transcript("ANALYZER STATUS", f"request_error: {exc}")
            preserve_history = False
            if latest_request_messages is not None:
                _safe_write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript=transcript_buffer.render(),
                )
            log.warning(
                "analyzer request failed at action %d: %s", display_action_num, exc
            )
            transcript_buffer.close()
            return AnalyzerTurnResult(
                step_executed=False,
                retryable_failure=True,
                reasoning=captured_reasoning,
                efficiency_metrics=dict(self._turn_efficiency_metrics),
                failure_category="transport",
                failure_detail=str(exc),
                attempts=latest_request_index,
            )
        except Exception as exc:
            append_transcript("ANALYZER STATUS", f"error: {exc}")
            preserve_history = False
            if latest_request_messages is not None:
                _safe_write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript=transcript_buffer.render(),
                )
            log.warning("analyzer failed at action %d: %s", display_action_num, exc)
            transcript_buffer.close()
            return AnalyzerTurnResult(
                step_executed=False,
                reasoning=captured_reasoning,
                efficiency_metrics=dict(self._turn_efficiency_metrics),
                failure_category="internal",
                failure_detail=f"{type(exc).__name__}: {exc}",
                attempts=latest_request_index,
            )
        finally:
            if preserve_history:
                self._history_messages = self._persistent_history_messages(
                    messages,
                    tools=tools,
                    request_base_chars=request_base_chars,
                    message_length_cache=message_length_cache,
                )
            else:
                self._history_messages = previous_history_messages
            self._persist_durable_state()
            self._step_env_callback = None
            self._current_valid_actions = []
            self._should_stop_callback = None

        if step_executed:
            status_message = "Step executed."
        elif yielded_control_reason is not None:
            status_message = f"Yielded control to solver: {yielded_control_reason}."
        else:
            status_message = "No action(...) call was captured."

        status = (
            f"model: {self._model.model_id}\n"
            f"base_url: {self._model.base_url}\n"
            f"max_output_tokens: {self._max_output_tokens if self._max_output_tokens is not None else 'server default'}\n"
            f"reply_reserve_tokens: {self._reply_reserve_tokens}\n"
            f"context_budget_tokens: {self._context_budget_tokens}\n"
            f"request_safety_margin_tokens: {self._request_safety_margin_tokens}\n"
            f"tool_output_tokens: {self._tool_output_tokens}\n"
            f"yield_seconds: {self._yield_seconds if self._yield_seconds is not None else 'disabled'}\n"
            f"available_tools: python\n"
            f"python_timeout_seconds: {self._python_timeout}\n"
            f"history_messages: {len(self._history_messages)}\n"
            f"efficiency_metrics: {json.dumps(self._turn_efficiency_metrics, separators=(',', ':'))}\n"
            f"step_executed: {step_executed}\n"
            f"message: {status_message}"
        )
        append_transcript("ANALYZER STATUS", status)
        if latest_request_messages is not None:
            _safe_write_prompt_log_snapshot(
                prompt_log,
                model_id=self._model.model_id,
                base_url=self._model.base_url,
                display_action_num=display_action_num,
                analysis_step=analysis_step,
                request_index=latest_request_index,
                messages=latest_request_messages,
                tools=latest_request_tools,
                tool_choice=latest_request_tool_choice,
                transcript=transcript_buffer.render(),
            )
        transcript_buffer.close()
        exhausted = bool(
            not step_executed
            and yielded_control_reason is None
            and self._tool_steps is not None
            and turn_count >= self._tool_steps
        )
        return AnalyzerTurnResult(
            step_executed=step_executed,
            reasoning=captured_reasoning,
            yielded_control=yielded_control_reason is not None,
            yield_reason=yielded_control_reason,
            efficiency_metrics=dict(self._turn_efficiency_metrics),
            failure_category="tool_step_exhausted" if exhausted else None,
            failure_detail=(
                "Analyzer exhausted its tool-step budget without executing an action."
                if exhausted
                else ""
            ),
            attempts=latest_request_index,
            exhausted=exhausted,
        )
