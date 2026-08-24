"""Thread-safe, bounded knowledge shared by passes of the same game."""

from __future__ import annotations

import threading
import json
import os
import time
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
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
            try:
                payload = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                backup = resolved.with_suffix(resolved.suffix + ".bak")
                if not backup.exists():
                    return
                try:
                    payload = json.loads(backup.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return
            games = payload.get("games") if isinstance(payload, dict) else None
            try:
                schema_version = int(payload.get("version") or 1)
            except (AttributeError, TypeError, ValueError):
                schema_version = 1
            if not isinstance(games, dict):
                raise ValueError(f"Invalid trial knowledge store: {resolved}")
            for game_id, game_payload in games.items():
                if not isinstance(game_payload, dict):
                    continue
                loaded_transitions = []
                for item in game_payload.get("transitions", []):
                    if not isinstance(item, dict):
                        continue
                    record = dict(item)
                    record.setdefault(
                        "state_context_version", 1 if schema_version < 3 else 2
                    )
                    record["legacy_state_identity"] = (
                        int(record.get("state_context_version") or 1) < 2
                    )
                    loaded_transitions.append(record)
                self._transitions[str(game_id)] = deque(
                    loaded_transitions, maxlen=self._transition_limit
                )
                self._lessons[str(game_id)] = deque(
                    [
                        dict(item)
                        for item in game_payload.get("lessons", [])
                        if isinstance(item, dict)
                    ],
                    maxlen=self._lesson_limit,
                )

    def _persist_locked(self) -> None:
        path = self._persistence_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._process_file_lock(path):
                self._merge_disk_locked(path)
                self._write_payload_locked(path)
        except (OSError, TimeoutError):
            # Durable memory is best-effort: a contended or temporarily
            # unavailable sidecar must never abort a live game.
            return

    @contextmanager
    def _process_file_lock(self, path: Path):
        lock_path = path.with_suffix(path.suffix + ".lock")
        deadline = time.monotonic() + 5.0
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    stale = time.time() - lock_path.stat().st_mtime > 30.0
                except FileNotFoundError:
                    continue
                if stale:
                    lock_path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out locking trial knowledge: {path}")
                time.sleep(0.01)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    def _merge_disk_locked(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        games = payload.get("games") if isinstance(payload, dict) else None
        if not isinstance(games, dict):
            return
        try:
            schema_version = int(payload.get("version") or 1)
        except (AttributeError, TypeError, ValueError):
            schema_version = 1
        for game_id, raw_game in games.items():
            if not isinstance(raw_game, dict):
                continue
            for target, key, limit in (
                (self._transitions, "transitions", self._transition_limit),
                (self._lessons, "lessons", self._lesson_limit),
            ):
                disk_items = raw_game.get(key)
                if not isinstance(disk_items, list):
                    disk_items = []
                combined: list[dict[str, Any]] = []
                seen: set[str] = set()
                for item in (
                    *disk_items,
                    *target.get(str(game_id), ()),
                ):
                    if not isinstance(item, dict):
                        continue
                    record = dict(item)
                    if key == "transitions":
                        record.setdefault(
                            "state_context_version", 1 if schema_version < 3 else 2
                        )
                        record["legacy_state_identity"] = (
                            int(record.get("state_context_version") or 1) < 2
                        )
                    signature = json.dumps(record, sort_keys=True, default=str)
                    if signature not in seen:
                        combined.append(record)
                        seen.add(signature)
                if key == "transitions":
                    combined.sort(key=lambda item: str(item.get("observed_at") or ""))
                target[str(game_id)] = deque(combined, maxlen=limit)

    def _write_payload_locked(self, path: Path) -> None:
        payload = {
            "version": 3,
            "state_identity_version": 2,
            "games": {
                game_id: {
                    "transitions": list(self._transitions.get(game_id, ())),
                    "lessons": list(self._lessons.get(game_id, ())),
                }
                for game_id in sorted(set(self._transitions) | set(self._lessons))
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        if path.exists():
            backup = path.with_suffix(path.suffix + ".bak")
            backup_temporary = path.with_suffix(path.suffix + ".bak.tmp")
            backup_temporary.write_bytes(path.read_bytes())
            backup_temporary.replace(backup)
        temporary.replace(path)

    def observe(
        self,
        game_id: str,
        transition: dict[str, Any],
        *,
        strategy: dict[str, Any] | None = None,
        pass_index: int = 0,
        evidence_id: str = "",
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
                "object_before_state_id",
                "object_after_state_id",
                "action_display",
                "outcome_class",
                "level",
                "level_completed",
                "run_complete",
                "board_changed",
                "reward",
                "game_over",
                "state",
                "valid_actions_before",
                "valid_actions_after",
                "state_context_version",
            )
        }
        record["pass_index"] = max(0, int(pass_index))
        record["evidence_id"] = str(evidence_id or "")[:160]
        record["observed_at"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
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
        local = [
            item
            for item in transitions
            if state_id and item.get("before_state_id") == state_id
        ]
        grouped_outcomes: dict[str, dict[str, Counter[str]]] = {}
        raw_action_counts: Counter[str] = Counter()
        for item in local:
            action = str(item.get("action_display") or "")
            if action:
                evidence_id = str(
                    item.get("evidence_id")
                    or f"legacy-pass:{int(item.get('pass_index') or 0)}"
                )
                grouped_outcomes.setdefault(action, {}).setdefault(
                    evidence_id, Counter()
                )[str(item.get("outcome_class") or "unknown")] += 1
                raw_action_counts[action] += 1
        outcomes: dict[str, Counter[str]] = {}
        for action, evidence_groups in grouped_outcomes.items():
            independent = outcomes.setdefault(action, Counter())
            for counts in evidence_groups.values():
                modal_outcome = min(
                    counts.items(), key=lambda item: (-item[1], item[0])
                )[0]
                independent[modal_outcome] += 1
        evidence_ids = {
            str(
                item.get("evidence_id")
                or f"legacy-pass:{int(item.get('pass_index') or 0)}"
            )
            for item in transitions
        }
        return {
            "prior_trials": len(evidence_ids),
            "observations": len(transitions),
            "independent_evidence": len(evidence_ids),
            "legacy_state_identity_observations": sum(
                bool(item.get("legacy_state_identity"))
                or int(item.get("state_context_version") or 1) < 2
                for item in transitions
            ),
            "state_action_evidence": [
                {
                    "action": action,
                    "outcomes": dict(sorted(counts.items())),
                    "trials": sum(counts.values()),
                    "raw_observations": raw_action_counts[action],
                }
                for action, counts in sorted(outcomes.items())
            ][:12],
            "progress_lessons": lessons[-8:],
            "transition_records": transitions[-128:],
        }
