"""Direct OpenAI-compatible tool-calling analyzer for ARC puzzle runs."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

from inference.agent.action_names import MAX_ACTION_BATCH, to_engine_action, to_model_action
from inference.agent.inference_controller import (
    InferenceControllerConfig,
    action_family,
    build_experience_snapshot,
    evaluate_outcome_match,
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
    build_small_context_prompt,
)
from inference.agent.python_tool_policy import (
    allowed_modules_text,
    runtime_bindings_text,
    runtime_helper_signatures_text,
)
from inference.agent.program_ir import (
    ProgramCompileError,
    compiler_runtime_metadata,
    compile_program,
    fast_program_tool_parameters_schema,
    program_tool_parameters_schema,
)
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.runtime_state import (
    RUNTIME_STATE_FILENAME,
    Frame,
    HistoryEntry,
    load_runtime_state,
)
from inference.agent.vision_context import (
    current_grid_image_enabled,
    current_grid_image_part,
    multimodal_context,
)
from inference.utils.openai_compat import build_chat_payload, build_headers, normalize_provider

log = logging.getLogger(__name__)

_LOCAL_ANALYZER_MODEL_ID = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
_LOCAL_ANALYZER_BASE_URL = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
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
            arguments: dict[str, Any] = {}
            for parameter_name, value in _TOOL_CALL_PARAMETER_RE.findall(raw_body):
                normalized_name = str(parameter_name).strip()
                if not normalized_name:
                    continue
                normalized_value: Any = value
                if normalized_name == "program":
                    try:
                        normalized_value = json.loads(str(value).strip())
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                arguments[normalized_name] = normalized_value
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
_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
_LOCAL_ANALYZER_TIMEOUT = _get_env_float("LOCAL_ANALYZER_TIMEOUT", 0.0)
_LOCAL_ANALYZER_TOOL_STEPS = _get_env_int("LOCAL_ANALYZER_TOOL_STEPS", 12)
_LOCAL_ANALYZER_TOOL_TIMEOUT = _get_env_int("LOCAL_ANALYZER_TOOL_TIMEOUT", 30)
_LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS = _get_env_int("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", 1024)
_LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
_LOCAL_ANALYZER_ENABLE_THINKING = _get_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True)
_LOCAL_ANALYZER_TEMPERATURE = _get_env_float("LOCAL_ANALYZER_TEMPERATURE", 0.6)
_LOCAL_ANALYZER_TOP_P = _get_env_float("LOCAL_ANALYZER_TOP_P", 0.95)
_LOCAL_ANALYZER_TOP_K = _get_env_int("LOCAL_ANALYZER_TOP_K", 20)
_LOCAL_ANALYZER_SEED = _get_env_int("LOCAL_ANALYZER_SEED", -1)
_REQUEST_SAFETY_MARGIN_TOKENS = 512
_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS = 512
_PERSISTENT_HISTORY_ASSISTANT_TURNS = 6
_PERSISTENT_HISTORY_TOKEN_BUDGET = 4096
_RESPONSE_META_MAX_CHARS = 4000
_TOOL_NAME_MAX_CHARS = 80
_TOOL_CALL_ID_MAX_CHARS = 128

_PYTHON_TOOL_DESCRIPTION = (
    "Compile and run one ephemeral structured ProgramIR against preloaded ASCII game state. "
    "Submit `program` with version 1 and a non-empty statement body; raw Python source is not accepted. "
    "Available globals: "
    "`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, "
    "`valid_actions`, `last_action_result`, read-only `experience`, persisted `strategy`, "
    "`record_strategy(...)`, and `action(actions)` for executing one or more real environment actions. "
    "Pre-injected helpers: `color_grid(frame)`, `diff_frames(f1,f2)`, `find_positions(frame,char)`, "
    "`neighbors4(r,c,rows,cols)`, `neighbors8(...)`, `bfs(frame,start,goal,blocked=None)`, "
    "`flood(frame,start,color=None)`, `cell_at(frame,r,c)`, `count_colors(frame)`, `object_positions(frame,color)`."
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
    if result.get("stopped_early") and result.get("stop_reason") == "stopped":
        return "stopped"
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
    if reason == "stopped":
        return (
            "No further actions were executed because the solver is stopping or its "
            "runtime/action budget is exhausted."
        )
    return "No further actions were executed because the previous action reached a terminal state."


def _display_action_number(action_num: int) -> int:
    return max(1, int(action_num) + 1)


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
                inline_value = candidate[len(target):].strip()
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
    required: bool = False,
) -> str | None:
    if not tools:
        return None
    return "required" if required else "auto"


def _turn_tool_choice(
    tools: list[dict[str, Any]] | None,
    *,
    provider: str,
    turn_count: int,
    step_executed: bool,
) -> str | None:
    require_schema_constrained_tool = not step_executed and (
        turn_count > 1 or normalize_provider(provider) == "vllm"
    )
    return _request_tool_choice(
        tools,
        required=require_schema_constrained_tool,
    )


def _tool_result_requires_strict_retry(content: str) -> bool:
    """Use guided decoding only after the fast path produced invalid structure."""
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("stage") == "schema" and bool(
        payload.get("error")
    )


def _normalize_tool_name(value: Any) -> str:
    raw = str(value or "").strip()
    cleaned = "".join(
        character
        for character in raw
        if character.isprintable() and character not in "\r\n\t"
    )
    if not cleaned:
        return "unknown"
    if len(cleaned) <= _TOOL_NAME_MAX_CHARS:
        return cleaned
    return cleaned[: _TOOL_NAME_MAX_CHARS - 3].rstrip() + "..."


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
        lines.append(_trim_log_text(json.dumps(tool_calls, indent=2, ensure_ascii=True)))
    return "\n".join(lines)


def _build_system_prompt(
    *,
    tool_output_tokens: int,
    context_window_tokens: int | None = None,
) -> str:
    if context_window_tokens is not None and context_window_tokens <= 32_768:
        return build_small_context_prompt(tool_output_tokens=tool_output_tokens)
    prompt = "You are a coding agent solving a grid-based puzzle game."
    prompt += GAME_OVERVIEW_ADDENDUM
    prompt += STRUCTURED_RUNTIME_STATE_ADDENDUM
    if current_grid_image_enabled():
        prompt += MULTIMODAL_CONTEXT_ADDENDUM
    prompt += VISUAL_GAME_ADDENDUM
    prompt += PYTHON_ADDENDUM
    prompt += COMPACT_TOOL_SESSION_ADDENDUM.format(tool_output_tokens=tool_output_tokens)
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
        return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

    __repr__ = __str__


@dataclass(frozen=True)
class _AsciiHistoryEntryView:
    action: str
    frame: _AsciiFrameView

    def __str__(self) -> str:
        return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

    __repr__ = __str__


def _to_ascii_frame_view(frame: Frame | None) -> _AsciiFrameView | None:
    if frame is None:
        return None
    return _AsciiFrameView(
        ascii=frame.ascii,
        step=frame.step,
        level=frame.level,
        shape=frame.shape,
    )


def _to_ascii_history_views(history_entries: list[HistoryEntry]) -> list[_AsciiHistoryEntryView]:
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


def _ascii_history_view_payload(history_entries: list[HistoryEntry]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in history_entries:
        frame_payload = _ascii_frame_view_payload(entry.frame)
        if frame_payload is None:
            continue
        payload.append({"action": entry.action, "frame": frame_payload})
    return payload


def _format_action_span(start_action_num: int | None, end_action_num: int | None) -> str | None:
    if start_action_num is None or end_action_num is None:
        return None
    if start_action_num <= 0 or end_action_num <= 0:
        return None
    if start_action_num == end_action_num:
        return f"{start_action_num}"
    return f"{start_action_num}-{end_action_num}"


def _estimate_tokens(value: Any) -> int:
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    length = len(rendered)
    if length == 0:
        return 0
    ascii_chars = sum(1 for c in rendered if ord(c) < 128)
    ascii_ratio = ascii_chars / length
    if ascii_ratio > 0.9:
        chars_per_token = 4.0
    elif ascii_ratio > 0.7:
        chars_per_token = 3.5
    else:
        chars_per_token = 2.5
    return max(1, int(length / chars_per_token))


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
        configured_base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL).strip()
        if not configured_base_url:
            raise ValueError("LOCAL_ANALYZER_BASE_URL must be set for the local analyzer preset.")

        provider = os.environ.get("LOCAL_ANALYZER_PROVIDER", os.environ.get("OPENAI_PROVIDER", "vllm")).strip().lower()
        if not provider:
            provider = "vllm"
        model_id = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "").strip() or _LOCAL_ANALYZER_MODEL_ID.strip()
        if not model_id:
            raise ValueError("LOCAL_ANALYZER_MODEL_ID must be set for the local analyzer preset.")
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

    provider = os.environ.get("OPENAI_PROVIDER", os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")).strip().lower()
    if not provider:
        provider = "vllm"
    base_url = _host_accessible_base_url(
        os.environ.get("OPENAI_BASE_URL", os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL)).strip()
    )
    if not base_url:
        raise ValueError("OPENAI_BASE_URL or LOCAL_ANALYZER_BASE_URL must be set for direct model ids.")
    return AnalyzerModelConfig(provider=provider, base_url=base_url, model_id=requested)


def _append_transcript_section(log_path: Path, label: str, content: str) -> None:
    rendered_content = content.strip()
    if not rendered_content:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{label}]\n")
            f.write(rendered_content)
            f.write("\n\n")
    except OSError:
        logging.warning("Failed to write transcript section %s to %s", label, log_path)


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
    return _normalize_message_content(value) if not isinstance(value, str) else value.strip()


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
                recovered_arguments = recovered_tool_calls[0].get("function", {}).get("arguments", "{}")
                return json.loads(str(recovered_arguments))
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("tool call arguments must decode to a JSON object")
    raise ValueError("tool call arguments must be a JSON object or JSON object string")


def _render_tool_call_markup(tool_name: str, arguments: Any) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return ""
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
    parsed = _json_like_payload(content) if isinstance(content, str) else (content if isinstance(content, dict) else None)
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
        game_stem = state_stem[:-len(suffix)] if state_stem.endswith(suffix) else state_stem
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
        blocks.append(_render_tool_result_display(content) if role == "TOOL" else content)

    reasoning = _extract_reasoning_text(message)
    if reasoning:
        blocks.append("[REASONING]")
        blocks.append(reasoning)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            name = str(function.get("name", "")).strip() or "unknown"
            blocks.append(f"[ASSISTANT TOOL CALL: {name}]")
            tool_call_id = str(tool_call.get("id", "")).strip()
            if tool_call_id:
                blocks.append(f"id: {tool_call_id}")
            rendered_tool_call = _render_tool_call_markup(name, function.get("arguments", "{}"))
            if rendered_tool_call:
                blocks.append(rendered_tool_call)
            else:
                raw_arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    rendered_arguments = json.dumps(parsed_arguments, indent=2, ensure_ascii=True)
                except (TypeError, ValueError, json.JSONDecodeError):
                    rendered_arguments = str(raw_arguments)
                blocks.append("arguments:")
                blocks.append(rendered_arguments if rendered_arguments.strip() else "{}")

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
    tool_choice: str | None = None,
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
    tool_choice: str | None,
    transcript: str,
) -> None:
    rendered_messages = "\n\n".join(_render_prompt_log_message(message) for message in messages)
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
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL CALL SNAPSHOT\n")
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
    ) -> None:
        resolved_model = _resolve_analyzer_model(model)
        if base_url is not None or provider is not None:
            resolved_model = AnalyzerModelConfig(
                provider=str(provider or resolved_model.provider).strip() or resolved_model.provider,
                base_url=(
                    _host_accessible_base_url(str(base_url).strip())
                    if base_url is not None and str(base_url).strip()
                    else resolved_model.base_url
                ),
                model_id=resolved_model.model_id,
            )
        self._model = resolved_model
        configured_timeout = _LOCAL_ANALYZER_TIMEOUT if timeout is None else timeout
        self._timeout = None if configured_timeout is None or configured_timeout <= 0 else float(configured_timeout)
        self._api_key = str(api_key or "").strip()
        self._tool_steps = None if _LOCAL_ANALYZER_TOOL_STEPS <= 0 else max(1, _LOCAL_ANALYZER_TOOL_STEPS)
        self._python_timeout = min(30, max(1, _LOCAL_ANALYZER_TOOL_TIMEOUT))
        self._yield_seconds = None if _LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(_LOCAL_ANALYZER_YIELD_SECONDS)
        configured_max_output = _LOCAL_ANALYZER_MAX_OUTPUT
        self._max_output_tokens = None if configured_max_output <= 0 else max(1, configured_max_output)
        self._reply_reserve_tokens = self._max_output_tokens or 512
        self._tool_output_tokens = max(64, _LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS)
        self._tool_output_chars = max(256, self._tool_output_tokens * 4)
        self._save_request_logs = bool(save_request_logs)
        self._system_prompt = _build_system_prompt(
            tool_output_tokens=self._tool_output_tokens,
            context_window_tokens=_LOCAL_ANALYZER_CONTEXT_WINDOW,
        )
        self._request_safety_margin_tokens = _REQUEST_SAFETY_MARGIN_TOKENS
        self._context_budget_tokens = max(
            1024,
            _LOCAL_ANALYZER_CONTEXT_WINDOW - self._reply_reserve_tokens - self._request_safety_margin_tokens,
        )
        self._history_messages: list[dict[str, Any]] = []
        self._session_runtime_dir: Path | None = None
        self._session_total_tokens = 0
        self._session_generated_tokens = 0
        self._step_env_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._current_valid_actions: list[str] = []
        self._last_step_summary: dict[str, Any] | None = None
        self._last_action_result: dict[str, Any] | None = None
        self._summarized_knowledge = _empty_world_model()
        self._controller_config = InferenceControllerConfig.from_env()
        self._strategy_memory: dict[str, Any] = {}
        self._last_image_level: int | None = None
        self._tool_specs: dict[bool, list[dict[str, Any]]] = {}
        self._tool_token_estimates: dict[bool, int] = {}
        self._next_synthetic_tool_call_id = 1
        # One ToolAgent is scoped to one game. Reuse its localhost vLLM
        # connection across turns instead of paying TCP setup for every call.
        self._http_session = requests.Session()

    def close(self) -> None:
        """Release HTTP connection-pool resources owned by this game agent."""
        self._http_session.close()

    def _normalize_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Ensure assistant/tool message pairing has unique, non-empty IDs."""
        normalized: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for raw_tool_call in tool_calls:
            tool_call = dict(raw_tool_call)
            tool_call_id = "".join(
                character
                for character in str(tool_call.get("id") or "")
                if character.isprintable()
            ).strip()[:_TOOL_CALL_ID_MAX_CHARS]
            if not tool_call_id or tool_call_id in used_ids:
                while True:
                    candidate = (
                        f"duck-synthetic-call-{self._next_synthetic_tool_call_id}"
                    )
                    self._next_synthetic_tool_call_id += 1
                    if candidate not in used_ids:
                        tool_call_id = candidate
                        break
            used_ids.add(tool_call_id)
            tool_call["id"] = tool_call_id
            tool_call["type"] = "function"
            tool_call["function"] = dict(tool_call["function"])
            normalized.append(tool_call)
        return normalized

    def _headers(self) -> dict[str, str]:
        api_key = (
            self._api_key
            or os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        site_url = os.environ.get("LOCAL_ANALYZER_SITE_URL", "").strip()
        app_name = os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Agent Harness").strip()
        return build_headers(
            provider=self._model.provider,
            api_key=api_key,
            referer=site_url,
            title=app_name,
        )

    def _ensure_session(self, state_path: Path) -> None:
        runtime_dir = state_path.parent
        if self._session_runtime_dir != runtime_dir:
            self._session_runtime_dir = runtime_dir
            self._history_messages = []
            self._session_total_tokens = 0
            self._session_generated_tokens = 0
            self._last_step_summary = None
            self._last_action_result = None
            self._summarized_knowledge = _empty_world_model()
            self._strategy_memory = {}
            self._last_image_level = None

    def _record_strategy(self, update: dict[str, Any]) -> dict[str, Any]:
        def short_text(value: Any, max_chars: int = 280) -> str:
            return _normalize_summary_text(value, max_chars=max_chars)[:max_chars]

        persisted = dict(self._strategy_memory)
        for key in ("goal", "hypothesis", "open_question", "next_test", "fallback"):
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

        raw_contradictions = update.get("contradictions")
        if isinstance(raw_contradictions, (list, tuple)):
            contradictions = [short_text(item, 160) for item in raw_contradictions]
            contradictions = [item for item in contradictions if item][:3]
        else:
            single_contradiction = short_text(raw_contradictions, 160)
            contradictions = [single_contradiction] if single_contradiction else []
        if contradictions:
            persisted["contradictions"] = contradictions

        try:
            if update.get("confidence") is not None:
                persisted["confidence"] = max(
                    0.0, min(1.0, float(update.get("confidence")))
                )
        except (TypeError, ValueError):
            pass

        self._strategy_memory = persisted
        if persisted.get("goal"):
            self._summarized_knowledge["goal_model"] = str(persisted["goal"])
        if persisted.get("hypothesis"):
            confidence = persisted.get("confidence")
            suffix = f" (confidence={confidence:.2f})" if isinstance(confidence, float) else ""
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
        if persisted.get("contradictions"):
            self._summarized_knowledge["contradictions"] = "; ".join(
                str(item) for item in persisted["contradictions"]
            )
        return dict(self._strategy_memory)

    def _evaluate_strategy_prediction(
        self, action_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        existing = action_result.get("prediction_result")
        if isinstance(existing, dict):
            self._strategy_memory["prediction_result"] = dict(existing)
            return dict(existing)
        test_action = normalize_action_key(self._strategy_memory.get("test_action", ""))
        expected = str(self._strategy_memory.get("expected_outcome") or "").strip()
        if not test_action or not expected:
            return None
        candidates = [
            item
            for item in action_result.get("steps") or []
            if isinstance(item, dict)
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
        status = evaluate_outcome_match(expected, observed)
        result = {
            "status": status,
            "action": normalize_action_key(observed.get("action_display", test_action)),
            "expected": expected,
            "actual": actual,
        }
        self._strategy_memory["prediction_result"] = result
        return result

    @property
    def total_tokens(self) -> int:
        return max(0, int(self._session_total_tokens))

    @property
    def generated_tokens(self) -> int:
        return max(0, int(self._session_generated_tokens))

    def _accumulate_usage_tokens(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return

        def first_token_count(keys: tuple[str, ...]) -> int:
            for key in keys:
                raw_value = usage.get(key)
                try:
                    if raw_value is not None:
                        return max(0, int(raw_value))
                except (TypeError, ValueError):
                    continue
            return 0

        generated_token_count = first_token_count(
            ("completion_tokens", "output_tokens", "generated_tokens")
        )
        self._session_generated_tokens += generated_token_count

        total_tokens = usage.get("total_tokens")
        try:
            if total_tokens is not None:
                self._session_total_tokens += max(0, int(total_tokens))
                return
        except (TypeError, ValueError):
            pass

        input_token_count = first_token_count(("prompt_tokens", "input_tokens"))
        self._session_total_tokens += input_token_count + generated_token_count

    def _summarize_step_sequence(self, action_results: list[dict[str, Any]]) -> dict[str, Any] | None:
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
                executed_actions.extend(str(name).strip() for name in action_names if str(name).strip())
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
            "level_transition": any(bool(item.get("level_completed")) for item in executed_results),
            "run_complete": any(bool(item.get("run_complete")) for item in executed_results),
            "game_over": any(bool(item.get("game_over")) for item in executed_results),
            "board_changed": any(bool(item.get("board_changed")) for item in executed_results),
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
            pieces.append("produced a board change; verify that it affected gameplay objects rather than only HUD elements.")
        else:
            pieces.append("did not show a confirmed board change; treat this as weak evidence until verified.")
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
        if summary.get("board_changed"):
            self._summarized_knowledge["recent_findings"] = _normalize_summary_text(
                f"Board changed. Level={summary.get('level')}. "
                + (self._summarized_knowledge.get("recent_findings") or ""),
                max_chars=200,
            )
        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            existing_notes = self._summarized_knowledge.get("cross_level_notes", "")
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""
            if summary.get("level_transition") and existing_notes:
                self._summarized_knowledge["cross_level_notes"] = existing_notes
            else:
                self._summarized_knowledge["cross_level_notes"] = ""
            for stale_key in ("evidence", "contradictions", "test_action", "expected_outcome", "prediction_result"):
                self._strategy_memory.pop(stale_key, None)
            self._summarized_knowledge.pop("contradictions", None)

    def _summarized_knowledge_lines(self) -> list[str]:
        entries = [
            ("World model", self._summarized_knowledge.get("world_model", "")),
            ("Goal model", self._summarized_knowledge.get("goal_model", "")),
            ("Action model", self._summarized_knowledge.get("action_model", "")),
            ("Recent findings", self._summarized_knowledge.get("recent_findings", "")),
            ("Open questions", self._summarized_knowledge.get("open_questions", "")),
            ("Plan", self._summarized_knowledge.get("current_plan", "")),
            ("Contradictions", self._summarized_knowledge.get("contradictions", "")),
            ("Cross-level notes", self._summarized_knowledge.get("cross_level_notes", "")),
        ]
        max_value_chars = 200
        lines = []
        for label, value in entries:
            if not value:
                continue
            if len(value) > max_value_chars:
                value = value[:max_value_chars].rstrip() + "..."
            lines.append(f"- {label}: {value}")
        if not lines:
            return []
        return [
            "Working world model carried from earlier turns:",
            *lines,
            "- Revise any item above immediately if `current_frame` or `history` contradicts it.",
        ]

    def _build_user_message(self, user_prompt: str, current_frame: Frame | None) -> dict[str, Any]:
        include_image = False
        image_mode = multimodal_context()
        if current_frame is not None and image_mode == "current_grid":
            include_image = True
        elif current_frame is not None and image_mode == "level_start":
            include_image = current_frame.level != self._last_image_level
            if include_image:
                self._last_image_level = current_frame.level
        image_part = current_grid_image_part(current_frame) if include_image else None
        if image_part is None:
            return {"role": "user", "content": user_prompt}

        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{user_prompt}\n\nCurrent grid image:"},
                image_part,
            ],
        }

    def _compress_experience_snapshot(self, exp: dict[str, Any]) -> str:
        if not exp or not exp.get("enabled"):
            return ""
        parts = []
        phase = exp.get("phase", "unknown")
        budget = exp.get("action_budget", "?")
        parts.append(f"Phase: {phase} (budget: {budget} actions)")
        visits = exp.get("state_visits", 0)
        unique = exp.get("unique_states", 0)
        parts.append(f"States: {unique} unique, {visits} visits")
        no_op_streak = exp.get("behavioral_no_op_streak", 0)
        if no_op_streak > 0:
            parts.append(f"No-op streak: {no_op_streak}")
        cycle = exp.get("cycle_period")
        if cycle:
            confidence = exp.get("cycle_confidence", 0)
            parts.append(f"Cycle detected: period={cycle}, confidence={confidence:.0%}")
        outcome = exp.get("latest_outcome")
        if outcome:
            parts.append(f"Last outcome: {outcome}")
        suggested = exp.get("suggested_actions", [])
        if suggested:
            parts.append(f"Suggested: {', '.join(str(a) for a in suggested[:3])}")
        discouraged = exp.get("discouraged_actions", [])
        if discouraged:
            parts.append(f"Discouraged: {', '.join(str(a) for a in discouraged[:3])}")
        tried = exp.get("tried_here", {})
        if tried:
            summaries = []
            for action_name, stats in tried.items():
                trials = stats.get("trials", 0) if isinstance(stats, dict) else 0
                changes = stats.get("changes", 0) if isinstance(stats, dict) else 0
                no_ops = stats.get("no_ops", 0) if isinstance(stats, dict) else 0
                if changes > 0:
                    summaries.append(f"{action_name}({trials}x,{changes}change)")
                elif no_ops > 0:
                    summaries.append(f"{action_name}({trials}x,{no_ops}noop)")
                else:
                    summaries.append(f"{action_name}({trials}x)")
            if summaries:
                parts.append(f"Tried here: {', '.join(summaries[:6])}")
        recovery = exp.get("recovery_reasons", [])
        if recovery:
            parts.append(f"Recovery: {', '.join(str(r) for r in recovery[:2])}")
        ranked = exp.get("ranked_actions", [])
        if ranked:
            top3 = ranked[:3]
            rank_parts = []
            for item in top3:
                act = item.get("action", "?")
                reason = item.get("reason", "")
                rank_parts.append(f"{act}({reason})" if reason else act)
            parts.append(f"Ranked: {', '.join(rank_parts)}")
        models = exp.get("transition_models_here", [])
        if models:
            model_lines = []
            for m in models[:3]:
                action = m.get("action", "?")
                outcome = m.get("predicted_outcome", "?")
                conf = m.get("confidence", 0)
                det = "det" if m.get("verified_deterministic") else f"conf={conf:.0%}"
                model_lines.append(f"{action}→{outcome}({det})")
            parts.append(f"Models: {'; '.join(model_lines)}")
        chains = exp.get("chain_predictions", [])
        if chains:
            chain_lines = [f"{c['first_action']}→{c['second_action']}(conf={c['chain_confidence']:.0%})" for c in chains[:2]]
            parts.append(f"Chains: {'; '.join(chain_lines)}")
        return "; ".join(parts)

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
        current_step = max(current_frame.step if current_frame is not None else 0, max(0, action_num)) + 1
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
            [current_level, *[entry.frame.level for entry in history_entries if entry.frame is not None]],
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
            lines.append(f"The code executed {normalized_count or 0} {action_label} in the previous sequence.")
            executed_actions = previous_step_summary.get("executed_actions")
            rendered_actions: list[str] = []
            if isinstance(executed_actions, list):
                rendered_actions = [str(name).strip() for name in executed_actions if str(name).strip()]
            if rendered_actions:
                action_prefix = "Executed actions (first 10):" if len(rendered_actions) > 10 else "Executed actions:"
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
        lines.append(state_line)
        total_actions = len(history_entries)
        if total_actions > 0 and experience_snapshot and experience_snapshot.get("enabled"):
            exp = experience_snapshot
            no_ops = int(exp.get("no_op_actions", 0))
            changes = total_actions - no_ops
            lines.append(
                f"Cumulative: {total_actions} actions, {no_ops} no-ops, {changes} board changes, "
                f"{exp.get('unique_behavioral_states', 0)} unique states visited."
            )
        lines.extend(
            [
                f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",
                "See system prompt for runtime variable reference. Use Python to inspect the evidence and call `action(actions)` with the best valid action.",
            ]
        )
        if experience_snapshot and experience_snapshot.get("enabled"):
            compressed_exp = self._compress_experience_snapshot(experience_snapshot)
            if compressed_exp:
                lines.extend(
                    [
                        "Experience controller:",
                        compressed_exp,
                        "Follow the controller phase and action budget: orient with one compact inspection, explore with a controlled falsifiable probe, continue progress when evidence supports it, and abandon or revise the current hypothesis in recover. Never retry a discouraged action in the same visible state.",
                    ]
                )
        lines.append(
            "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, "
            "but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`."
        )
        lines.extend(self._summarized_knowledge_lines())
        lines.append("end of world model. ")
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
                "Before writing code, quickly state in one line: (a) what you think the board contains, (b) what the goal is, (c) what you will test. Then write minimal code to test it.",
                "When ready, call `action(actions)` from inside the `python` tool with the best valid action or ordered batch selected by your code. Batch only a short sequence supported by observed action effects; use single probes while orienting, exploring, or recovering.",
                "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it.",
                "If you include assistant text before a tool call, keep it short and use it to update the world model. Helpful optional prefixes are `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and `Cross-level notes:`.",
                TOOL_CALL_FORMAT_GUIDANCE,
            ]
        )
        if "MOUSE" in _normalize_valid_actions(valid_actions):
            lines.append("If you use MOUSE, include integer row and col arguments.")
        return "\n".join(lines)

    def _tools(
        self,
        state_path: Path,
        *,
        strict: bool = True,
    ) -> list[dict[str, Any]]:
        self._ensure_session(state_path)
        strict_vllm = normalize_provider(self._model.provider) == "vllm" and strict
        cached = self._tool_specs.get(strict_vllm)
        if cached is not None:
            return cached
        function: dict[str, Any] = {
            "name": "python",
            "description": _PYTHON_TOOL_DESCRIPTION,
            "parameters": (
                program_tool_parameters_schema()
                if strict or normalize_provider(self._model.provider) != "vllm"
                else fast_program_tool_parameters_schema()
            ),
        }
        # Modern vLLM applies schema-constrained decoding to auto-selected tool
        # calls only when the function opts into strict mode. Other compatible
        # providers retain their existing request shape because support varies
        # by upstream route/model.
        if strict_vllm:
            function["strict"] = True
        tools = [
            {
                "type": "function",
                "function": function,
            }
        ]
        self._tool_specs[strict_vllm] = tools
        return tools

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None = None,
        request_timeout_seconds: float | None = None,
    ) -> _ChatCompletionResult:
        payload = build_chat_payload(
            provider=self._model.provider,
            model=self._model.model_id,
            messages=messages,
            max_tokens=self._max_output_tokens,
            temperature=_LOCAL_ANALYZER_TEMPERATURE,
            top_p=_LOCAL_ANALYZER_TOP_P,
            top_k=_LOCAL_ANALYZER_TOP_K,
            thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING),
            tools=tools,
            tool_choice=(
                tool_choice
                if tool_choice is not None
                else _request_tool_choice(tools)
            ),
            seed=_LOCAL_ANALYZER_SEED,
        )
        configured_timeout = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else self._timeout
        )
        request_deadline = (
            time.monotonic() + configured_timeout
            if configured_timeout is not None
            else None
        )

        def remaining_request_timeout() -> float | None:
            if request_deadline is None:
                return None
            remaining = request_deadline - time.monotonic()
            if remaining <= 0.0:
                raise requests.Timeout("analyzer request deadline exceeded")
            return max(0.001, remaining)

        def sleep_before_retry(retry_index: int) -> None:
            delay = min(2 ** retry_index * 0.5, 4.0)
            if request_deadline is not None:
                remaining = request_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise requests.Timeout("analyzer request deadline exceeded")
                delay = min(delay, remaining)
            time.sleep(delay)

        def post_chat(request_payload: dict[str, Any]) -> requests.Response:
            return self._http_session.post(
                f"{self._model.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=request_payload,
                timeout=remaining_request_timeout(),
            )

        _RETRYABLE_HTTP_CODES = {408, 429, 502, 503, 504}
        _max_retries = 3
        response = None
        for _retry in range(_max_retries):
            try:
                response = post_chat(payload)
            except requests.ConnectionError:
                if _retry == _max_retries - 1:
                    raise
                sleep_before_retry(_retry)
                continue
            if response.status_code not in _RETRYABLE_HTTP_CODES:
                break
            if _retry < _max_retries - 1:
                response.close()
                sleep_before_retry(_retry)
        assert response is not None
        try:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = response.text.strip()
                message = f"{exc}"
                if detail:
                    message += f" | response: {detail}"
                raise requests.RequestException(message) from exc
            if getattr(response, "status_code", 200) >= 400:
                detail = response.text.strip()
                message = f"{response.status_code} Error"
                if detail:
                    message += f" | response: {detail}"
                raise requests.RequestException(message)
            try:
                payload = response.json()
            except ValueError as exc:
                raise requests.RequestException(
                    "server returned invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise requests.RequestException("server returned a non-object response")
            choices = payload.get("choices", [])
            if not isinstance(choices, list) or not choices:
                raise requests.RequestException("server returned no choices")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise requests.RequestException("server returned an invalid choice")
            message = choice.get("message", {})
            if not isinstance(message, dict):
                raise requests.RequestException("server returned an invalid message")
            tool_calls = message.get("tool_calls")
            if tool_calls is not None and not isinstance(tool_calls, list):
                raise requests.RequestException("server returned invalid tool_calls")
            if isinstance(tool_calls, list) and any(
                not isinstance(tool_call, dict) for tool_call in tool_calls
            ):
                raise requests.RequestException("server returned an invalid tool call")
            if isinstance(tool_calls, list) and any(
                not isinstance(tool_call.get("function"), dict)
                for tool_call in tool_calls
            ):
                raise requests.RequestException(
                    "server returned an invalid tool function"
                )
            if isinstance(tool_calls, list):
                message = dict(message)
                message["tool_calls"] = self._normalize_tool_calls(tool_calls)
            return _ChatCompletionResult(
                message=message,
                finish_reason=str(choice.get("finish_reason", "") or ""),
                usage=payload.get("usage"),
            )
        finally:
            response.close()

    def _trim_tool_text(self, text: str) -> tuple[str, bool]:
        if len(text) <= self._tool_output_chars:
            return text, False
        omitted = len(text) - self._tool_output_chars
        return f"{text[:self._tool_output_chars]}\n... [truncated {omitted} chars]", True

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
                        compacted["executed_action_count"] = 1 if action_result.get("executed") else 0
            return compacted
        if isinstance(value, list):
            return [self._summarize_planned_actions(item) for item in value]
        return value

    def _render_tool_payload(self, payload: dict[str, Any], *, truncate_fields: tuple[str, ...] = ()) -> str:
        result = self._summarize_planned_actions(dict(payload))
        truncated = False
        for field in truncate_fields:
            value = result.get(field)
            if isinstance(value, str):
                result[field], field_truncated = self._trim_tool_text(value)
                truncated = truncated or field_truncated
        rendered = json.dumps(result, indent=2)
        max_chars = self._tool_output_chars * 2
        if len(rendered) > max_chars:
            # Preserve a valid JSON tool response. Slicing the serialized object
            # makes it impossible for the model (and trace consumers) to parse.
            for field in sorted(
                truncate_fields,
                key=lambda name: len(json.dumps(result.get(name), ensure_ascii=True)),
                reverse=True,
            ):
                if field not in result:
                    continue
                value = result[field]
                encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
                if len(encoded) <= 512:
                    continue
                result[field] = {
                    "truncated": True,
                    "type": type(value).__name__,
                    "preview": encoded[:512],
                    "omitted_chars": len(encoded) - 512,
                }
                truncated = True
                rendered = json.dumps(result, indent=2)
                if len(rendered) <= max_chars:
                    break
        if truncated:
            truncation_note = (
                f"Tool output was cut off to stay within the ~{self._tool_output_tokens}-token response budget."
            )
            result["truncated"] = True
            result["truncation_note"] = truncation_note
            rendered = json.dumps(result, indent=2)
        if len(rendered) > max_chars:
            # Defensive fallback for an unexpectedly large non-truncatable
            # payload. Keep protocol-level fields and enough structured error
            # context for the model to repair its next call.
            compact = {
                key: value
                for key, value in result.items()
                if key in {
                    "success",
                    "tool",
                    "stage",
                    "error_category",
                    "compiler_version",
                    "result_truncated",
                }
                and isinstance(value, (str, int, float, bool, type(None)))
            }
            compiler = result.get("compiler")
            if isinstance(compiler, dict) and isinstance(compiler.get("version"), str):
                compact["compiler_version"] = compiler["version"]
            recovery_hint = result.get("recovery_hint") or result.get("error")
            if isinstance(recovery_hint, str) and recovery_hint:
                compact["recovery_hint"] = recovery_hint[: max(80, max_chars // 3)]
            compact.update(
                {
                    "truncated": True,
                    "truncation_note": "Large tool fields were omitted; repair context was preserved.",
                }
            )
            diagnostics = result.get("diagnostics")
            if isinstance(diagnostics, list) and diagnostics and isinstance(diagnostics[0], dict):
                first = diagnostics[0]
                compact_diagnostic = {
                    key: str(first[key])[:160]
                    for key in ("code", "path", "message", "hint")
                    if first.get(key) is not None
                }
                candidate = {**compact, "diagnostics": [compact_diagnostic]}
                if len(json.dumps(candidate, indent=2)) <= max_chars:
                    compact = candidate
            rendered = json.dumps(compact, indent=2)
        return rendered

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
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(f"Action {index} uses legacy MOUSE x/y fields; use row and col.")
                unexpected_fields = sorted(set(item) - {"action", "row", "col"})
                if unexpected_fields:
                    raise ValueError(
                        f"Action {index} has unsupported field(s): "
                        f"{', '.join(str(field) for field in unexpected_fields)}."
                    )
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE":
                    if type(item.get("row")) is not int or type(item.get("col")) is not int:
                        raise ValueError(
                            f"Action {index} MOUSE requires integer row and col fields."
                        )
                    entry["row"] = item["row"]
                    entry["col"] = item["col"]
                elif "row" in item or "col" in item:
                    raise ValueError(
                        f"Action {index} row and col fields are only valid for MOUSE."
                    )
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized

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
            "done": bool(payload.get("done")),
            "level_completed": bool(payload.get("level_completed")),
            "game_over": bool(payload.get("game_over")),
            "run_complete": bool(payload.get("run_complete")),
            "action_display": payload.get("action_display") or payload.get("action_name"),
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
            compact["steps"] = [dict(item) for item in steps[:12] if isinstance(item, dict)]
        requested_actions = payload.get("requested_actions")
        if isinstance(requested_actions, list) and requested_actions:
            compact["requested_actions"] = [
                str(action).strip()
                for action in requested_actions[:MAX_ACTION_BATCH]
                if str(action).strip()
            ]
        executed_actions = payload.get("executed_actions")
        if not compact["executed"]:
            compact["executed_actions"] = []
        elif isinstance(executed_actions, list) and executed_actions:
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

    def _validate_code_common_mistakes(self, code: str) -> str | None:
        if "current_frame._grid" in code or "frame._grid" in code:
            return "Warning: Direct _grid access is discouraged. Use current_frame.segmentation or pre-injected helpers instead."
        if code.count("action(") > 3:
            return "Warning: Many action() calls in one snippet. Consider batching actions."
        if re.search(r"while\s+True\s*:", code):
            return "Warning: Infinite loop detected. Ensure a break condition exists."
        if "import os" in code or "import sys" in code or "import subprocess" in code:
            return "Warning: System modules are not available in sandbox. Use allowed modules only."
        if "import numpy" in code or "import np" in code:
            return "Warning: numpy is not available. Use built-in list operations."
        if re.search(r"while\s+\d+\s*[<>=!]", code):
            return "Warning: Numeric while loop may timeout. Use for-loop with range() instead."
        if code.count("for ") > 3 and "def " in code:
            return "Warning: Complex nested code may timeout. Simplify and use pre-injected helpers."
        if re.search(r"history\[\d+\]\[['\"]", code):
            return "Warning: history entries use .action and .frame attributes, not dict keys. Use history[i].action."
        if re.search(r"action\(\s*['\"]", code):
            return "Warning: action() expects a list. Use action(['LEFT']) not action('LEFT')."
        if re.search(r"print\s*\(\s*\w+\.ascii\s*\)", code):
            return "Warning: Printing full board ascii wastes output tokens. Print compact derived data instead."
        return None

    def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        unexpected_arguments = [key for key in arguments if key != "program"]
        if unexpected_arguments:
            diagnostics = []
            for key in unexpected_arguments:
                is_raw_source = key == "code"
                message = (
                    "python requires structured `program` version 1; raw `code` is not accepted."
                    if is_raw_source
                    else f"unknown python tool argument: {key}"
                )
                diagnostics.append({
                    "code": "IR_RAW_SOURCE_REJECTED" if is_raw_source else "IR_UNKNOWN_TOOL_ARGUMENT",
                    "path": key,
                    "message": message,
                })
            error_category = (
                "RAW_SOURCE_NOT_ACCEPTED"
                if "code" in unexpected_arguments
                else "INVALID_TOOL_ARGUMENTS"
            )
            payload = {
                "tool": "python",
                "compiler": compiler_runtime_metadata(),
                "stage": "schema",
                "error_category": error_category,
                "error": diagnostics[0]["message"],
                "diagnostics": diagnostics,
            }
            return _ToolDispatchResult(self._render_tool_payload(payload))
        if "program" not in arguments:
            message = "python requires the `program` argument."
            payload = {
                "tool": "python",
                "compiler": compiler_runtime_metadata(),
                "stage": "schema",
                "error_category": "INVALID_TOOL_ARGUMENTS",
                "error": message,
                "diagnostics": [{
                    "code": "IR_REQUIRED_FIELD",
                    "path": "program",
                    "message": message,
                }],
            }
            return _ToolDispatchResult(self._render_tool_payload(payload))
        try:
            compiled_program = compile_program(arguments["program"])
        except ProgramCompileError as exc:
            payload = {"tool": "python", "compiler": compiler_runtime_metadata(), **exc.payload()}
            return _ToolDispatchResult(self._render_tool_payload(payload))
        code = compiled_program.source.rstrip()
        validation_warning = self._validate_code_common_mistakes(code)
        if validation_warning:
            log.info("Code validation warning: %s", validation_warning)

        valid_actions = list(_normalize_valid_actions(self._current_valid_actions))

        def _serialized_runtime_state(
            *,
            next_valid_actions: list[str] | None = None,
            last_action_result: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            refreshed_frame, refreshed_history = load_runtime_state(state_path)
            current_frame_payload = _ascii_frame_view_payload(refreshed_frame)
            if isinstance(next_valid_actions, list):
                sanitized_actions = [str(item).strip() for item in next_valid_actions if str(item).strip()]
            else:
                sanitized_actions = list(valid_actions)
            persisted_action_result = (
                last_action_result
                if isinstance(last_action_result, dict)
                else self._last_action_result
            )
            return {
                "current_frame": current_frame_payload,
                "history": _ascii_history_view_payload(refreshed_history),
                "valid_actions": sanitized_actions,
                "experience": build_experience_snapshot(
                    refreshed_history,
                    refreshed_frame,
                    sanitized_actions,
                    self._controller_config,
                    strategy_confidence=float(self._strategy_memory.get("confidence", 1.0)),
                ),
                "strategy": dict(self._strategy_memory),
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
            normalized_actions = self._normalize_python_actions(actions)
            if terminal_action_result is not None:
                reason = _terminal_action_reason(terminal_action_result) or "terminal_state"
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
                    "level_completed": bool(terminal_action_result.get("level_completed")),
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
                raise RuntimeError("action(actions) did not return a JSON-like payload.")
            compact_payload = self._compact_action_result(raw_payload)
            prediction_result = self._evaluate_strategy_prediction(compact_payload)
            if prediction_result is not None:
                compact_payload["prediction_result"] = prediction_result
            next_valid_actions = raw_payload.get("valid_actions")
            if isinstance(next_valid_actions, list):
                self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
            if _terminal_action_reason(compact_payload):
                terminal_action_result = compact_payload
            self._last_action_result = dict(compact_payload)
            return {
                "action_result": compact_payload,
                "state": _serialized_runtime_state(
                    next_valid_actions=next_valid_actions if isinstance(next_valid_actions, list) else None,
                    last_action_result=compact_payload,
                ),
            }

        sandbox_result = run_sandboxed_python(
            code=code,
            timeout_seconds=self._python_timeout,
            initial_state=_serialized_runtime_state(),
            action_handler=_handle_action,
            strategy_handler=self._record_strategy,
        )

        action_results = [
            item
            for item in sandbox_result.get("action_results") or []
            if isinstance(item, dict)
        ]
        payload: dict[str, Any] = {"tool": "python", "compiler": compiled_program.metadata()}
        rendered_stdout = str(sandbox_result.get("stdout", "") or "")
        rendered_error = str(sandbox_result.get("error", "") or "")
        error_hint = str(sandbox_result.get("error_hint", "") or "")
        error_category = str(sandbox_result.get("error_category", "") or "")
        if rendered_error:
            payload["error"] = rendered_error
            if error_hint:
                payload["error_hint"] = error_hint
            if error_category:
                payload["error_category"] = error_category
                payload["stage"] = (
                    "sandbox"
                    if error_category in {
                        "SANDBOX_START_FAILED",
                        "SANDBOX_PROCESS_ERROR",
                        "SANDBOX_PROTOCOL_ERROR",
                    }
                    else "runtime"
                )
                error_line = sandbox_result.get("error_line")
                runtime_path = compiled_program.path_for_line(error_line) or "program"
                diagnostic_message = error_hint or rendered_error.splitlines()[-1]
                payload["diagnostics"] = [{
                    "code": "IR_RUNTIME_ERROR",
                    "path": runtime_path,
                    "message": diagnostic_message,
                }]
                recovery_map = {
                    "UNDEFINED_VARIABLE": "Define the variable or check spelling.",
                    "MISSING_RUNTIME_VARIABLE": "Do not redefine runtime variables.",
                    "MISUSED_RUNTIME_FUNCTION": "Call the function, e.g. action(['LEFT']).",
                    "BLOCKED_MODULE": "Use only allowed modules.",
                    "TYPE_MISMATCH": "Ensure operands are compatible types.",
                    "WRONG_ARGUMENTS": "Check function signature.",
                    "MISSING_KEY": "Check available keys before accessing.",
                    "INDEX_OUT_OF_RANGE": "Check bounds before indexing.",
                    "PRIVATE_ACCESS": "Use public attributes only.",
                    "TIMEOUT": "Optimize or reduce computation.",
                    "ACTION_FAILED": "Inspect the action payload and current valid_actions before retrying.",
                    "ACTION_BATCH_LIMIT": f"Use at most {MAX_ACTION_BATCH} actions in one action() call.",
                    "PROTECTED_RUNTIME_BINDING": "Use injected runtime variables and helpers without redefining them.",
                    "SANDBOX_START_FAILED": "Retry with a smaller program; the isolated runtime could not start.",
                    "SANDBOX_PROCESS_ERROR": "Simplify the program and retry in a fresh tool call.",
                    "SANDBOX_PROTOCOL_ERROR": "Retry in a fresh tool call with a smaller result.",
                }
                recovery = recovery_map.get(error_category, "Simplify code and retry.")
                if error_category == "UNDEFINED_VARIABLE":
                    recovery += (
                        f" Available runtime bindings: {runtime_bindings_text()}."
                        f" Helper signatures: {runtime_helper_signatures_text()}."
                    )
                elif error_category == "BLOCKED_MODULE":
                    recovery = f"Allowed: {allowed_modules_text()}."
                elif error_category == "MISSING_ATTRIBUTE":
                    recovery += " FrameView has: .ascii, .step, .level, .shape, .segmentation. HistoryEntry has: .action, .frame. TransitionView has: .action, .before_frame, .after_frame, .frame, .result."
                elif error_category == "TYPE_ERROR":
                    recovery = f"Type error: {error_hint}. Use isinstance() to check types, or convert with int()/str()/list()."
                elif "recursion" in rendered_error.lower():
                    recovery = "Recursion depth exceeded. Use iterative loops with bounded ranges instead of recursive calls."
                elif "memory" in rendered_error.lower():
                    recovery = "Memory exceeded. Use smaller data structures and avoid copying large grids."
                payload["recovery_hint"] = f"{error_hint} {recovery}" if error_hint else recovery
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
        else:
            payload["returncode"] = 0
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
            if sandbox_result.get("result") is not None:
                payload["result"] = sandbox_result.get("result")
            elif action_results:
                if len(action_results) == 1:
                    payload["result"] = action_results[-1]
                else:
                    payload["result"] = {
                        "action_calls": len(action_results),
                        "last_action_result": action_results[-1],
                    }
            if sandbox_result.get("result_truncated"):
                payload["result_truncated"] = True
                payload["result_warning"] = "Result was truncated inside the sandbox; return a smaller summary."

        step_executed = any(bool(item.get("executed")) for item in action_results)
        if validation_warning:
            payload["validation_warning"] = validation_warning
        if step_executed:
            self._last_step_summary = self._summarize_step_sequence(action_results)
            self._update_summarized_knowledge_from_step_summary()
        return _ToolDispatchResult(
            self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result")),
            step_executed=step_executed,
        )

    def _dispatch_tool(self, state_path: Path, name: str, arguments: Any) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        if name == "python":
            if not isinstance(arguments, dict):
                message = "python tool arguments must be a JSON object containing `program`."
                return _ToolDispatchResult(
                    self._render_tool_payload(
                        {
                            "tool": "python",
                            "compiler": compiler_runtime_metadata(),
                            "stage": "schema",
                            "error_category": "INVALID_TOOL_ARGUMENTS",
                            "error": message,
                            "diagnostics": [
                                {
                                    "code": "IR_TOOL_ARGUMENTS_NOT_OBJECT",
                                    "path": "arguments",
                                    "message": message,
                                }
                            ],
                        }
                    )
                )
            return self._run_python_tool(state_path, arguments)
        rendered_name = _normalize_tool_name(name)
        message = f"Unknown tool: {rendered_name}. The only available tool is `python`."
        return _ToolDispatchResult(
            self._render_tool_payload(
                {
                    "tool": rendered_name,
                    "stage": "schema",
                    "error_category": "INVALID_TOOL_NAME",
                    "error": message,
                    "diagnostics": [
                        {
                            "code": "IR_UNKNOWN_TOOL",
                            "path": "tool.name",
                            "message": message,
                        }
                    ],
                    "recovery_hint": "Call `python` with one JSON object containing `program`.",
                }
            )
        )

    def _decode_tool_arguments(
        self,
        tool_name: str,
        raw_arguments: Any,
    ) -> tuple[Any, _ToolDispatchResult | None]:
        """Decode provider arguments without disguising malformed JSON as `{}`."""
        if not isinstance(raw_arguments, str):
            if isinstance(raw_arguments, dict):
                return json.loads(json.dumps(raw_arguments)), None
            return raw_arguments, None
        try:
            return json.loads(raw_arguments), None
        except json.JSONDecodeError as exc:
            rendered_name = _normalize_tool_name(tool_name)
            message = (
                f"{rendered_name} tool arguments are invalid JSON at "
                f"line {exc.lineno}, column {exc.colno}: {exc.msg}."
            )
            payload: dict[str, Any] = {
                "tool": rendered_name,
                "stage": "schema",
                "error_category": "MALFORMED_TOOL_ARGUMENTS",
                "error": message,
                "diagnostics": [
                    {
                        "code": "IR_TOOL_ARGUMENTS_INVALID_JSON",
                        "path": "arguments",
                        "message": message,
                    }
                ],
                "recovery_hint": (
                    "Emit one valid JSON object. For `python`, the object must contain "
                    "a structured `program`; do not emit raw source or tool-call markup."
                ),
            }
            if rendered_name == "python":
                payload["compiler"] = compiler_runtime_metadata()
            return None, _ToolDispatchResult(self._render_tool_payload(payload))

    def _estimate_request_input_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        if not tools:
            return _estimate_tokens({"messages": messages})
        for strict_vllm, cached_tools in self._tool_specs.items():
            if tools is not cached_tools:
                continue
            tool_tokens = self._tool_token_estimates.get(strict_vllm)
            if tool_tokens is None:
                tool_tokens = _estimate_tokens(
                    {
                        "tools": tools,
                        "tool_choice": _request_tool_choice(tools),
                    }
                )
                self._tool_token_estimates[strict_vllm] = tool_tokens
            # Eight conservative tokens cover the separators between the two
            # independently rendered payload fragments.
            return _estimate_tokens({"messages": messages}) + tool_tokens + 8
        return _estimate_tokens(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": _request_tool_choice(tools),
            }
        )

    def _drop_oldest_history_block(self, history: list[dict[str, Any]], *, preserve_recent: int) -> bool:
        removable = len(history) - preserve_recent
        if removable <= 0:
            return False
        first = history.pop(0)
        first_role = str(first.get("role", "")).strip()
        if first_role in {"assistant", "tool"}:
            while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
                history.pop(0)
            return True
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            history.pop(0)
        while history and history[0].get("role") != "user" and len(history) > preserve_recent:
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

    def _drop_until_first_user_message(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed = list(history)
        while trimmed and str(trimmed[0].get("role", "")).strip() != "user":
            trimmed.pop(0)
        return trimmed

    def _persistent_history_messages(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        trimmed = self._trim_messages_for_context(messages, tools=tools)
        if not trimmed:
            return []
        trimmed_history = trimmed[1:]
        history = self._keep_recent_history_turns(
            trimmed_history,
            max_turns=_PERSISTENT_HISTORY_ASSISTANT_TURNS,
        )
        while (
            sum(
                str(message.get("role", "")).strip() == "assistant"
                for message in history
            )
            > 1
            and _estimate_tokens(history) > _PERSISTENT_HISTORY_TOKEN_BUDGET
        ):
            self._drop_oldest_history_block(history, preserve_recent=0)
        history = self._drop_until_first_user_message(history)
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
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        preserve_recent = max(0, preserve_recent)
        budget_tokens = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        initial_len = len(history)
        while history and self._estimate_request_input_tokens([system_message, *history], tools=tools) > budget_tokens:
            if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                break
        history = self._drop_until_first_user_message(history)
        dropped_count = initial_len - len(history)
        summary_msg: dict[str, Any] | None = None
        if dropped_count > 0:
            summary_parts = []
            actions_seen = set()
            errors_seen = set()
            for msg in messages[1:initial_len - len(history)]:
                if msg.get("role") == "tool":
                    content = str(msg.get("content", ""))
                    if "executed" in content and "action" in content:
                        actions_seen.add("actions")
                    if "error" in content.lower():
                        errors_seen.add("errors")
            if actions_seen:
                summary_parts.append(f"{dropped_count} earlier turns with actions were trimmed")
            else:
                summary_parts.append(f"{dropped_count} earlier turns were trimmed")
            if errors_seen:
                summary_parts.append("some earlier errors occurred")
            summary_text = "; ".join(summary_parts) + "."
            summary_msg = {"role": "user", "content": f"[Context summary: {summary_text} Re-ground from current_frame and experience.]"}
            history = [summary_msg, *history]
        candidate = [system_message, *history]
        if self._estimate_request_input_tokens(candidate, tools=tools) <= budget_tokens:
            return candidate

        if summary_msg is not None and history and history[0] is summary_msg:
            history = history[1:]
            candidate = [system_message, *history]
            if self._estimate_request_input_tokens(candidate, tools=tools) <= budget_tokens:
                return candidate

        reground_message = {
            "role": "user",
            "content": (
                "[Current turn content exceeded the input budget. Call python with a small "
                "ProgramIR inspection and re-ground from current_frame, experience, and "
                "valid_actions.]"
            ),
        }
        candidate = [system_message, reground_message]
        estimated_tokens = self._estimate_request_input_tokens(candidate, tools=tools)
        if estimated_tokens > budget_tokens:
            raise ValueError(
                "System prompt and tool schema exceed the configured analyzer input "
                f"budget ({estimated_tokens} estimated tokens > {budget_tokens}). "
                "Increase LOCAL_ANALYZER_CONTEXT_WINDOW or reduce reserved output tokens."
            )
        return candidate

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
        if not self._drop_oldest_history_block(history, preserve_recent=max(0, preserve_recent)):
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
            return None
        self._ensure_session(state_path)
        self._step_env_callback = step_env
        self._current_valid_actions = _normalize_valid_actions(valid_actions)

        analyzer_log = transcript_path or (state_path.parent / f"{state_path.stem}_analyzer.txt")
        prompt_log = _resolve_prompt_log_path(state_path)
        current_frame, history_entries = load_runtime_state(state_path)
        experience_snapshot = build_experience_snapshot(
            history_entries,
            current_frame,
            self._current_valid_actions,
            self._controller_config,
            strategy_confidence=float(self._strategy_memory.get("confidence", 1.0)),
        )
        user_prompt = self._build_user_prompt(
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=history_entries,
            previous_step_summary=self._last_step_summary,
            experience_snapshot=experience_snapshot,
        )
        display_action_num = _display_action_number(action_num)

        with open(analyzer_log, "a", encoding="utf-8") as f:
            step_label = f"analysis_step={analysis_step} | " if analysis_step is not None else ""
            transcript_header = (
                f"\n--- {step_label}action={display_action_num} | "
                f"{time.strftime('%H:%M:%S')} | tool-agent ---\n"
            )
            f.write(transcript_header)
        transcript_parts = [transcript_header]

        def append_transcript(label: str, content: str) -> None:
            _append_transcript_section(analyzer_log, label, content)
            transcript_parts.append(_render_transcript_section(label, content))
            if transcript_updated is not None:
                transcript_updated("".join(transcript_parts))

        append_transcript("SYSTEM PROMPT", self._system_prompt)
        append_transcript("USER PROMPT", user_prompt)

        previous_history_messages = list(self._history_messages)
        preserve_history = True
        strict_tool_retry = False
        initial_tools = self._tools(state_path, strict=False)
        messages: list[dict[str, Any]] = self._trim_messages_for_context(
            [{"role": "system", "content": self._system_prompt}, *self._history_messages, self._build_user_message(user_prompt, current_frame)],
            tools=initial_tools,
            preserve_recent=1,
        )
        step_executed = False
        captured_reasoning = ""
        latest_request_messages: list[dict[str, Any]] | None = None
        latest_request_tools: list[dict[str, Any]] | None = None
        latest_request_tool_choice: str | None = None
        latest_request_index = 0
        turn_started_at = time.monotonic()
        yielded_control_reason: str | None = None
        last_tool_error: str = ""

        def control_yield_reason() -> str | None:
            if should_stop is not None:
                try:
                    if should_stop():
                        return "stop_requested"
                except Exception as exc:
                    log.warning("analyzer stop check failed at action %d: %s", display_action_num, exc)
            if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:
                return "turn_time_budget"
            return None

        try:
            turn_count = 0
            while self._tool_steps is None or turn_count < self._tool_steps:
                yielded_control_reason = control_yield_reason()
                if yielded_control_reason is not None:
                    break
                turn_count += 1
                tools = self._tools(state_path, strict=strict_tool_retry)
                tool_choice = _turn_tool_choice(
                    tools,
                    provider=self._model.provider,
                    turn_count=turn_count,
                    step_executed=step_executed,
                )
                messages = self._trim_messages_for_context(messages, tools=tools)
                latest_request_tool_choice = tool_choice
                latest_request_index = turn_count
                if self._save_request_logs:
                    latest_request_messages = json.loads(json.dumps(messages))
                    latest_request_tools = json.loads(json.dumps(tools))
                    _write_prompt_log_snapshot(
                        prompt_log,
                        model_id=self._model.model_id,
                        base_url=self._model.base_url,
                        display_action_num=display_action_num,
                        analysis_step=analysis_step,
                        request_index=turn_count,
                        messages=latest_request_messages,
                        tools=latest_request_tools,
                        tool_choice=tool_choice,
                        transcript="".join(transcript_parts),
                    )
                try:
                    request_kwargs: dict[str, Any] = {
                        "tools": tools,
                        "tool_choice": tool_choice,
                    }
                    if request_timeout_seconds is not None:
                        request_kwargs["request_timeout_seconds"] = request_timeout_seconds
                    if self._save_request_logs:
                        _append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            messages=latest_request_messages,
                            tools=latest_request_tools,
                            event="request",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                        )
                    result = self._chat_completion(messages, **request_kwargs)
                    self._accumulate_usage_tokens(result.usage)
                    if self._save_request_logs:
                        _append_request_snapshot(
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
                        raise
                    trimmed_messages = self._trim_messages_for_context(
                        messages,
                        tools=tools,
                        extra_safety_tokens=_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS,
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
                raw_content = _normalize_message_content(result.message.get("content", ""))
                tool_calls = json.loads(json.dumps(result.message.get("tool_calls") or []))
                tool_call_markup_in_text = _contains_tool_call_markup(raw_reasoning, raw_content)
                recovered_tool_calls_from_markup = False
                if not tool_calls and tool_call_markup_in_text:
                    tool_calls = _recover_tool_calls_from_markup(raw_reasoning, raw_content)
                    recovered_tool_calls_from_markup = bool(tool_calls)
                if tool_calls:
                    tool_calls = self._normalize_tool_calls(tool_calls)
                reasoning = _strip_tool_call_markup(raw_reasoning) if tool_call_markup_in_text else raw_reasoning
                content = _strip_tool_call_markup(raw_content) if tool_call_markup_in_text else raw_content
                malformed_argument_errors: list[str] = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = _normalize_tool_name(function.get("name", ""))
                    raw_arguments = function.get("arguments", "{}")
                    if isinstance(raw_arguments, str):
                        try:
                            json.loads(raw_arguments)
                        except json.JSONDecodeError as exc:
                            malformed_argument_errors.append(f"{tool_name}: invalid JSON arguments ({exc})")
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
                    strict_tool_retry = True
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
                    followup_prefix = "You have not acted yet. Investigate first. "
                    if tool_call_markup_in_text:
                        followup_prefix = (
                            "You did not call a tool. We detected `<tool_call>` markup inside your reasoning or assistant text, "
                            "so no parsed tool call was executed. On this retry, do not add a note or explanation first. "
                            "Emit exactly one `python` tool call directly as your next response. "
                            "Do not place `<tool_call>` markup inside reasoning, explanation, or notes. "
                        )
                    else:
                        followup_prefix = "You have not acted yet. "
                        if content:
                            analysis_excerpt = content[:150].rstrip()
                            if len(content) > 150:
                                analysis_excerpt += "..."
                            followup_prefix += f"Your analysis was: {analysis_excerpt}. "
                            followup_prefix += "You already described the board and plan. Now emit exactly one `python` tool call implementing that plan. Do not repeat your analysis. "
                        exp_snap = experience_snapshot if experience_snapshot else {}
                        phase = exp_snap.get("phase", "unknown")
                        last_outcome = exp_snap.get("latest_outcome")
                        if last_outcome:
                            followup_prefix += f"Last outcome was '{last_outcome}'. "
                        if phase in ("orient", "explore"):
                            followup_prefix += (
                                f"Phase is {phase}: write code that diffs previous_frame vs current_frame using diff_frames(), "
                                "identifies key objects with segmentation, then calls action() with your best probe. "
                            )
                        elif phase == "recover":
                            no_op_streak = exp_snap.get("behavioral_no_op_streak", 0)
                            suggested = exp_snap.get("suggested_actions", [])
                            suggest_text = f"Try one of {suggested[:3]}" if suggested else "Try a different action family"
                            followup_prefix += (
                                f"Phase is recover (no-op streak={no_op_streak}). {suggest_text} "
                                "and call action() with it. Do not retry discouraged actions. "
                            )
                        elif phase == "progress":
                            followup_prefix += "Phase is progress: call action() with the next step in the sequence that was producing board changes. "
                    followup_prompt = (
                        f"{followup_prefix}"
                        f"{TOOL_CALL_FORMAT_GUIDANCE}"
                    )
                    if last_tool_error:
                        followup_prompt += f"\nPrevious tool error: {last_tool_error}. Write simpler code and try a different approach."
                    append_transcript("USER PROMPT", followup_prompt)
                    messages.append({"role": "user", "content": followup_prompt})
                    continue

                if content:
                    self._update_summarized_knowledge_from_assistant(content)
                    append_transcript("ASSISTANT", content)
                    assistant_message["content"] = content
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)

                for tool_index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = _normalize_tool_name(function.get("name", ""))
                    raw_args = function.get("arguments", "{}")
                    arguments, argument_error = self._decode_tool_arguments(
                        tool_name,
                        raw_args,
                    )
                    rendered_tool_call = _render_tool_call_markup(tool_name, raw_args)
                    append_transcript(
                        f"TOOL CALL: {tool_name}",
                        rendered_tool_call or (json.dumps(arguments, indent=2) if arguments else "{}"),
                    )
                    dispatch = argument_error or self._dispatch_tool(
                        state_path,
                        tool_name,
                        arguments,
                    )
                    if _tool_result_requires_strict_retry(dispatch.content):
                        strict_tool_retry = True
                    if dispatch.step_executed:
                        step_executed = True
                        last_tool_error = ""
                    elif dispatch.content:
                        try:
                            parsed_content = json.loads(dispatch.content)
                            if isinstance(parsed_content, dict) and parsed_content.get("error"):
                                last_tool_error = str(parsed_content.get("recovery_hint") or parsed_content["error"])[:200]
                        except (json.JSONDecodeError, TypeError):
                            pass
                    append_transcript(f"TOOL RESULT: {tool_name}", _render_tool_result_display(dispatch.content))
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

        except requests.RequestException as exc:
            append_transcript("ANALYZER STATUS", f"request_error: {exc}")
            preserve_history = False
            if latest_request_messages is not None:
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript="".join(transcript_parts),
                )
            log.warning("analyzer request failed at action %d: %s", display_action_num, exc)
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning=captured_reasoning)
        except Exception as exc:
            append_transcript("ANALYZER STATUS", f"error: {exc}")
            preserve_history = False
            if latest_request_messages is not None:
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript="".join(transcript_parts),
                )
            log.warning("analyzer failed at action %d: %s", display_action_num, exc)
            return None
        finally:
            if preserve_history:
                self._history_messages = self._persistent_history_messages(
                    messages,
                    tools=self._tools(state_path, strict=strict_tool_retry),
                )
            else:
                self._history_messages = previous_history_messages
            self._step_env_callback = None
            self._current_valid_actions = []

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
            f"step_executed: {step_executed}\n"
            f"message: {status_message}"
        )
        append_transcript("ANALYZER STATUS", status)
        if latest_request_messages is not None:
            _write_prompt_log_snapshot(
                prompt_log,
                model_id=self._model.model_id,
                base_url=self._model.base_url,
                display_action_num=display_action_num,
                analysis_step=analysis_step,
                request_index=latest_request_index,
                messages=latest_request_messages,
                tools=latest_request_tools,
                tool_choice=latest_request_tool_choice,
                transcript="".join(transcript_parts),
            )
        return AnalyzerTurnResult(
            step_executed=step_executed,
            reasoning=captured_reasoning,
            yielded_control=yielded_control_reason is not None,
        )
