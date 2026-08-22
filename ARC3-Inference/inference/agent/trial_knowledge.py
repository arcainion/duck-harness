"""Thread-safe, bounded knowledge shared by passes of the same game."""
from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any


class TrialKnowledgeStore:
    """Accumulate compact transition evidence without retaining raw frames."""

    def __init__(self, *, transition_limit: int = 512, lesson_limit: int = 24) -> None:
        self._transition_limit = max(32, int(transition_limit))
        self._lesson_limit = max(4, int(lesson_limit))
        self._lock = threading.RLock()
        self._transitions: dict[str, deque[dict[str, Any]]] = {}
        self._lessons: dict[str, deque[dict[str, Any]]] = {}

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
        }

