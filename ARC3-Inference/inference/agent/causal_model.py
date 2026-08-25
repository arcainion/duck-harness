"""Bounded structured causal world-model persistence."""
from __future__ import annotations

import math
from typing import Any


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _confidence(value: Any) -> float:
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return 0.0
        return round(max(0.0, min(1.0, numeric)), 3)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def normalize_causal_model(value: Any) -> dict[str, list[dict[str, Any]]]:
    """Validate the model's causal graph while keeping prompt/state size bounded."""
    raw = value if isinstance(value, dict) else {}
    result: dict[str, list[dict[str, Any]]] = {
        "entities": [],
        "relations": [],
        "subgoals": [],
        "predictions": [],
    }
    for item in _items(raw.get("entities"))[:16]:
        if not isinstance(item, dict) or not _text(item.get("id"), 64):
            continue
        result["entities"].append({
            "id": _text(item.get("id"), 64),
            "kind": _text(item.get("kind"), 64),
            "attributes": _text(item.get("attributes"), 200),
            "evidence": _text(item.get("evidence"), 180),
            "confidence": _confidence(item.get("confidence")),
        })
    for item in _items(raw.get("relations"))[:24]:
        if not isinstance(item, dict):
            continue
        cause, effect = _text(item.get("cause"), 96), _text(item.get("effect"), 96)
        if not cause or not effect:
            continue
        result["relations"].append({
            "cause": cause,
            "effect": effect,
            "conditions": _text(item.get("conditions"), 180),
            "evidence": _text(item.get("evidence"), 180),
            "confidence": _confidence(item.get("confidence")),
            "support": _count(item.get("support")),
            "contradictions": _count(item.get("contradictions")),
            "last_observed_action": _count(item.get("last_observed_action")),
        })
    for item in _items(raw.get("subgoals"))[:12]:
        if not isinstance(item, dict) or not _text(item.get("id"), 64):
            continue
        status = _text(item.get("status"), 24).lower()
        result["subgoals"].append({
            "id": _text(item.get("id"), 64),
            "description": _text(item.get("description"), 200),
            "status": status if status in {"pending", "active", "complete", "blocked"} else "pending",
            "success_criteria": _text(item.get("success_criteria"), 180),
            "depends_on": [
                _text(entry, 64)
                for entry in _items(item.get("depends_on"))[:6]
                if _text(entry, 64)
            ],
        })
    for item in _items(raw.get("predictions"))[:8]:
        if not isinstance(item, dict) or not _text(item.get("action"), 80):
            continue
        status = _text(item.get("status"), 24).lower()
        result["predictions"].append({
            "action": _text(item.get("action"), 80),
            "expected_changes": _text(item.get("expected_changes"), 200),
            "expected_outcome": _text(item.get("expected_outcome"), 40).lower(),
            "conditions": _text(item.get("conditions"), 180),
            "confidence": _confidence(item.get("confidence")),
            "status": status if status in {"untested", "supported", "contradicted"} else "untested",
        })
    return result
