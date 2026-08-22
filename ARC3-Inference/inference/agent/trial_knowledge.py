"""Thread-safe, bounded knowledge shared by passes of the same game."""
from __future__ import annotations

import threading
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any


class TrialKnowledgeStore:
    """Accumulate compact transition evidence without retaining raw frames."""

    def __init__(
        self,
        *,
        transition_limit: int = 512,
        lesson_limit: int = 24,
        persistence_path: Path | None = None,
    ) -> None:
        self._transition_limit = max(32, int(transition_limit))
        self._lesson_limit = max(4, int(lesson_limit))
        self._lock = threading.RLock()
        self._transitions: dict[str, deque[dict[str, Any]]] = {}
        self._lessons: dict[str, deque[dict[str, Any]]] = {}
        self._persistence_path: Path | None = None
        if persistence_path is not None:
            self.configure_path(persistence_path)

    def configure_path(self, path: Path) -> None:
        """Load a resumable store and atomically persist future observations."""
        resolved = Path(path)
        with self._lock:
            self._persistence_path = resolved
            if not resolved.exists():
                return
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            games = payload.get("games") if isinstance(payload, dict) else None
            if not isinstance(games, dict):
                raise ValueError(f"Invalid trial knowledge store: {resolved}")
            for game_id, game_payload in games.items():
                if not isinstance(game_payload, dict):
                    continue
                self._transitions[str(game_id)] = deque(
                    [dict(item) for item in game_payload.get("transitions", []) if isinstance(item, dict)],
                    maxlen=self._transition_limit,
                )
                self._lessons[str(game_id)] = deque(
                    [dict(item) for item in game_payload.get("lessons", []) if isinstance(item, dict)],
                    maxlen=self._lesson_limit,
                )

    def _persist_locked(self) -> None:
        path = self._persistence_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "games": {
                game_id: {
                    "transitions": list(self._transitions.get(game_id, ())),
                    "lessons": list(self._lessons.get(game_id, ())),
                }
                for game_id in sorted(set(self._transitions) | set(self._lessons))
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)

    def observe(
        self,
        game_id: str,
        transition: dict[str, Any],
        *,
        strategy: dict[str, Any] | None = None,
        pass_index: int = 0,
    ) -> None:
        if not game_id or not transition.get("executed"):
            return
        record = {
            key: transition.get(key)
            for key in (
                "before_state_id",
                "after_state_id",
                "behavioral_before_state_id",
                "behavioral_after_state_id",
                "action_display",
                "outcome_class",
                "level",
                "level_completed",
                "run_complete",
                "board_changed",
            )
        }
        record["pass_index"] = max(0, int(pass_index))
        with self._lock:
            self._transitions.setdefault(
                game_id, deque(maxlen=self._transition_limit)
            ).append(record)
            if record.get("level_completed") or record.get("run_complete"):
                plan = strategy or {}
                lesson = {
                    "pass_index": record["pass_index"],
                    "level": record.get("level"),
                    "state_id": record.get("before_state_id"),
                    "action": record.get("action_display"),
                    "goal": str(plan.get("goal") or "")[:280],
                    "hypothesis": str(plan.get("hypothesis") or "")[:280],
                    "current_subgoal": str(plan.get("current_subgoal") or "")[:200],
                    "plan_steps": list(plan.get("plan_steps") or [])[:8],
                }
                lessons = self._lessons.setdefault(
                    game_id, deque(maxlen=self._lesson_limit)
                )
                if lesson not in lessons:
                    lessons.append(lesson)
            self._persist_locked()

    def snapshot(self, game_id: str, *, state_id: str = "") -> dict[str, Any]:
        with self._lock:
            transitions = list(self._transitions.get(game_id, ()))
            lessons = list(self._lessons.get(game_id, ()))
        local = [item for item in transitions if state_id and item.get("before_state_id") == state_id]
        outcomes: dict[str, Counter[str]] = {}
        for item in local:
            action = str(item.get("action_display") or "")
            if action:
                outcomes.setdefault(action, Counter())[str(item.get("outcome_class") or "unknown")] += 1
        return {
            "prior_trials": len({int(item.get("pass_index") or 0) for item in transitions}),
            "observations": len(transitions),
            "state_action_evidence": [
                {"action": action, "outcomes": dict(sorted(counts.items())), "trials": sum(counts.values())}
                for action, counts in sorted(outcomes.items())
            ][:12],
            "progress_lessons": lessons[-8:],
            "transition_records": transitions[-128:],
        }
