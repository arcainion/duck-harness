"""Streaming inference-quality metrics from replay event sidecars."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


_EVENT_NAME_RE = re.compile(r"^(?P<game>.+)_p(?P<pass>\d+)_events\.jsonl$")


def _fallback_state_id(event: dict[str, Any]) -> str:
    explicit = str(event.get("after_state_id") or event.get("state_id") or "").strip()
    if explicit:
        return explicit
    board = event.get("board")
    if not isinstance(board, list):
        return ""
    encoded = json.dumps(board, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def iter_events(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one event at a time; large replay directories are never loaded whole."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if isinstance(payload, dict):
                yield payload


def summarize_event_file(path: Path) -> dict[str, Any]:
    actions = 0
    no_ops = 0
    repeated_no_ops = 0
    rewarding_actions = 0
    loop_interventions = 0
    terminal_violations = 0
    unique_states: set[str] = set()
    phase_counts: Counter[str] = Counter()
    previous_action = ""
    previous_no_op = False
    terminal_seen = False

    for event in iter_events(path):
        state_id = _fallback_state_id(event)
        if state_id:
            unique_states.add(state_id)
        loop_interventions += int(
            bool(event.get("guarded"))
            or str(event.get("stop_reason") or "") in {"loop_guard", "loop_detected"}
        )
        if event.get("type") != "action":
            continue
        actions += 1
        action = str(event.get("action_display") or event.get("action_name") or "")
        no_op = not bool(event.get("board_changed"))
        no_ops += int(no_op)
        repeated_no_ops += int(no_op and previous_no_op and action == previous_action)
        rewarding_actions += int(float(event.get("reward") or 0.0) > 0.0)
        phase = str(event.get("controller_phase") or "").strip()
        if phase:
            phase_counts[phase] += 1
        if terminal_seen and action != "RESET":
            terminal_violations += 1
        if action == "RESET":
            terminal_seen = False
        terminal_seen = terminal_seen or any(
            bool(event.get(key))
            for key in ("done", "game_over", "run_complete")
        )
        previous_action = action
        previous_no_op = no_op

    return {
        "actions": actions,
        "no_op_actions": no_ops,
        "no_op_rate": no_ops / actions if actions else 0.0,
        "repeated_no_ops": repeated_no_ops,
        "rewarding_actions": rewarding_actions,
        "rewarding_action_rate": rewarding_actions / actions if actions else 0.0,
        "unique_states_observed": len(unique_states),
        "loop_interventions": loop_interventions,
        "terminal_state_violations": terminal_violations,
        "phase_counts": dict(sorted(phase_counts.items())),
    }


def _combine(items: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        key: sum(int(item.get(key, 0) or 0) for item in items)
        for key in (
            "actions",
            "no_op_actions",
            "repeated_no_ops",
            "rewarding_actions",
            "unique_states_observed",
            "loop_interventions",
            "terminal_state_violations",
        )
    }
    phases: Counter[str] = Counter()
    for item in items:
        phases.update(item.get("phase_counts") or {})
    actions = totals["actions"]
    totals["no_op_rate"] = totals["no_op_actions"] / actions if actions else 0.0
    totals["rewarding_action_rate"] = (
        totals["rewarding_actions"] / actions if actions else 0.0
    )
    totals["phase_counts"] = dict(sorted(phases.items()))
    totals["trace_count"] = len(items)
    return totals


def summarize_run_traces(run_dir: Path) -> dict[str, Any]:
    artifacts_dir = run_dir / "artifacts"
    per_game_items: dict[str, list[dict[str, Any]]] = {}
    all_items: list[dict[str, Any]] = []
    if artifacts_dir.is_dir():
        for path in sorted(artifacts_dir.glob("*_events.jsonl")):
            match = _EVENT_NAME_RE.match(path.name)
            if match is None:
                continue
            item = summarize_event_file(path)
            per_game_items.setdefault(match.group("game"), []).append(item)
            all_items.append(item)
    return {
        "overall": _combine(all_items),
        "games": {
            game_id: _combine(items)
            for game_id, items in sorted(per_game_items.items())
        },
    }
