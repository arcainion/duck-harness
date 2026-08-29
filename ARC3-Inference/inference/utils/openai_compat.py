"""Helpers for provider-specific OpenAI-compatible requests and responses."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderRequest:
    endpoint: str
    payload: dict[str, Any]
    responses_api: bool = False


def normalize_provider(value: str | None) -> str:
    provider = str(value or "").strip().lower()
    if provider in {"", "openai-compatible", "compat"}:
        return "vllm"
    if provider in {"openai", "official-openai"}:
        return "openai"
    if provider in {"responses", "openai-responses", "official-openai-responses"}:
        return "openai-responses"
    if provider in {"openrouter", "router"}:
        return "openrouter"
    return provider


def build_headers(
    *,
    provider: str,
    api_key: str,
    referer: str = "",
    title: str = "",
) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
    return headers


def build_chat_payload(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    thinking_token_budget: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    seed: int | None = None,
    candidates: int = 1,
    stream: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if candidates > 1:
        payload["n"] = min(4, max(1, int(candidates)))
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

    normalized = normalize_provider(provider)
    if normalized == "vllm":
        if top_k > 0:
            payload["top_k"] = top_k
        payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if thinking and thinking_token_budget is not None and thinking_token_budget > 0:
            payload["thinking_token_budget"] = int(thinking_token_budget)
        if seed is not None and seed >= 0:
            payload["seed"] = seed

    return payload


def build_responses_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Translate the harness' Chat Completions shape to the Responses API."""
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id", "")),
                    "output": str(message.get("content", "")),
                }
            )
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    parts.append({"type": "input_text", "text": str(part.get("text", ""))})
                elif part.get("type") == "image_url":
                    image = part.get("image_url")
                    image_url = image.get("url", "") if isinstance(image, dict) else image
                    parts.append({"type": "input_image", "image_url": str(image_url or "")})
            content = parts
        input_items.append({"role": role, "content": content})
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id", "")),
                        "name": str(function.get("name", "")),
                        "arguments": str(function.get("arguments", "{}")),
                    }
                )

    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "stream": bool(stream),
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    if tools:
        response_tools: list[dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            response_tools.append(
                {
                    "type": "function",
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                    "strict": bool(function.get("strict", False)),
                }
            )
        payload["tools"] = response_tools
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function", {})
            payload["tool_choice"] = {
                "type": "function",
                "name": str(function.get("name", "")),
            }
        elif tool_choice:
            payload["tool_choice"] = tool_choice
    return payload


def build_provider_request(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    thinking: bool,
    thinking_token_budget: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    seed: int | None = None,
    candidates: int = 1,
    stream: bool = False,
) -> ProviderRequest:
    """Build one normalized request through the selected provider adapter."""
    normalized = normalize_provider(provider)
    if normalized == "openai-responses":
        return ProviderRequest(
            endpoint="responses",
            responses_api=True,
            payload=build_responses_payload(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
            ),
        )
    return ProviderRequest(
        endpoint="chat/completions",
        payload=build_chat_payload(
            provider=normalized,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            thinking=thinking,
            thinking_token_budget=thinking_token_budget,
            tools=tools,
            tool_choice=tool_choice,
            seed=seed,
            candidates=candidates,
            stream=stream,
        ),
    )


def normalize_provider_response(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("provider response must be a JSON object")
    if normalize_provider(provider) == "openai-responses":
        return normalize_responses_response(payload)
    return payload


def normalize_responses_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses API result into the chat-shaped internal contract."""
    if not isinstance(payload, dict):
        raise ValueError("Responses API response must be a JSON object")
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    output = payload.get("output")
    for item in output if isinstance(output, list) else []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            raw_arguments = item.get("arguments", "{}")
            arguments = (
                raw_arguments
                if isinstance(raw_arguments, str)
                else json.dumps(raw_arguments, separators=(",", ":"))
                if isinstance(raw_arguments, (dict, list))
                else "{}"
            )
            tool_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name", "")),
                        "arguments": arguments,
                    },
                }
            )
        elif item_type == "message":
            item_content = item.get("content")
            for part in item_content if isinstance(item_content, list) else []:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                    content.append(str(part.get("text", "")))
        elif item_type == "reasoning":
            summary = item.get("summary")
            for part in summary if isinstance(summary, list) else []:
                if isinstance(part, dict):
                    reasoning.append(str(part.get("text", "")))
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(content)}
    if reasoning:
        message["reasoning"] = "\n".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls
    incomplete_value = payload.get("incomplete_details")
    incomplete = incomplete_value if isinstance(incomplete_value, dict) else {}
    reason = str(incomplete.get("reason", ""))
    finish_reason = "length" if reason in {"max_output_tokens", "max_tokens"} else (
        "tool_calls" if tool_calls else "stop"
    )
    usage_value = payload.get("usage")
    usage = dict(usage_value) if isinstance(usage_value, dict) else {}
    if "input_tokens" in usage:
        usage.setdefault("prompt_tokens", usage["input_tokens"])
    if "output_tokens" in usage:
        usage.setdefault("completion_tokens", usage["output_tokens"])
    return {"choices": [{"message": message, "finish_reason": finish_reason}], "usage": usage}


def merge_chat_completion_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble OpenAI-compatible SSE deltas into a regular completion."""
    choices: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    for event in events:
        if isinstance(event.get("usage"), dict):
            usage.update(event["usage"])
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = int(choice.get("index", 0))
            target = choices.setdefault(index, {"message": {"role": "assistant", "content": ""}})
            delta = choice.get("delta") or choice.get("message") or {}
            message = target["message"]
            for key in ("content", "reasoning", "reasoning_content"):
                if isinstance(delta.get(key), str):
                    message[key] = str(message.get(key, "")) + delta[key]
            for call in delta.get("tool_calls") or []:
                call_index = int(call.get("index", 0))
                calls = message.setdefault("tool_calls", [])
                while len(calls) <= call_index:
                    calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                target_call = calls[call_index]
                if call.get("id"):
                    target_call["id"] += str(call["id"])
                function = call.get("function") or {}
                target_call["function"]["name"] += str(function.get("name", ""))
                target_call["function"]["arguments"] += str(function.get("arguments", ""))
            if choice.get("finish_reason") is not None:
                target["finish_reason"] = choice["finish_reason"]
    return {"choices": [choices[index] for index in sorted(choices)], "usage": usage}
