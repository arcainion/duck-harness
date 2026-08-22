"""Structured runtime state shared with created Python tools."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.utils.grid_utils import format_grid_ascii


RUNTIME_STATE_FILENAME = "tool_runtime_state.json"


@dataclass(frozen=True)
class Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        return rows, cols

    @property
    def ascii(self) -> str:
        return format_grid_ascii(self.grid)

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"Level: {self.level}\n"
            f"Step: {self.step}\n"
            f"Grid shape: {rows} x {cols}\n"
            f"Grid contents:\n{self.ascii}"
        )


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    frame: Frame


_RUNTIME_STATE_CACHE_LIMIT = 64
_runtime_state_cache: OrderedDict[
    str,
    tuple[
        tuple[int, int, int],
        Frame | None,
        tuple[HistoryEntry, ...],
    ],
] = OrderedDict()
_runtime_state_cache_lock = threading.Lock()


def _runtime_state_signature(path: Path) -> tuple[int, int, int]:
    metadata = path.stat()
    return metadata.st_mtime_ns, metadata.st_size, metadata.st_ino


def _invalidate_runtime_state_cache(path: Path) -> None:
    cache_key = str(path.resolve())
    with _runtime_state_cache_lock:
        _runtime_state_cache.pop(cache_key, None)


def normalize_grid(raw: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    rows: list[tuple[int, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells: list[int] = []
        for cell in row:
            try:
                cells.append(int(cell))
            except (TypeError, ValueError):
                cells.append(0)
        rows.append(tuple(cells))
    return tuple(rows)


def frame_from_payload(payload: Any) -> Frame | None:
    if not isinstance(payload, dict):
        return None
    try:
        step = max(0, int(payload.get("step", 0) or 0))
    except (TypeError, ValueError):
        step = 0
    try:
        level = max(1, int(payload.get("level", 1) or 1))
    except (TypeError, ValueError):
        level = 1
    return Frame(
        grid=normalize_grid(payload.get("grid")),
        step=step,
        level=level,
    )


def frame_to_payload(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "grid": [list(row) for row in frame.grid],
        "step": frame.step,
        "level": frame.level,
    }


def history_entry_from_payload(payload: Any) -> HistoryEntry | None:
    if not isinstance(payload, dict):
        return None
    frame = frame_from_payload(payload.get("frame"))
    if frame is None:
        return None
    return HistoryEntry(action=str(payload.get("action", "")).strip(), frame=frame)


def history_entry_to_payload(entry: HistoryEntry) -> dict[str, Any]:
    return {
        "action": entry.action,
        "frame": frame_to_payload(entry.frame),
    }


def _decode_runtime_state(text: str) -> tuple[Frame | None, list[HistoryEntry]]:
    payload = json.loads(text)
    current_frame = frame_from_payload(payload.get("current_frame"))
    history_entries = [
        entry
        for raw_entry in payload.get("history", [])
        for entry in [history_entry_from_payload(raw_entry)]
        if entry is not None
    ]
    return current_frame, history_entries


def load_runtime_state(path: Path) -> tuple[Frame | None, list[HistoryEntry]]:
    cache_key = str(path.resolve())
    latest: tuple[Frame | None, list[HistoryEntry]] = (None, [])
    for _attempt in range(3):
        try:
            signature = _runtime_state_signature(path)
        except FileNotFoundError:
            with _runtime_state_cache_lock:
                _runtime_state_cache.pop(cache_key, None)
            return None, []

        with _runtime_state_cache_lock:
            cached = _runtime_state_cache.get(cache_key)
            if cached is not None and cached[0] == signature:
                _runtime_state_cache.move_to_end(cache_key)
                return cached[1], list(cached[2])

        try:
            latest = _decode_runtime_state(path.read_text(encoding="utf-8"))
            final_signature = _runtime_state_signature(path)
        except FileNotFoundError:
            continue
        if final_signature != signature:
            continue

        current_frame, history_entries = latest
        with _runtime_state_cache_lock:
            _runtime_state_cache[cache_key] = (
                final_signature,
                current_frame,
                tuple(history_entries),
            )
            _runtime_state_cache.move_to_end(cache_key)
            while len(_runtime_state_cache) > _RUNTIME_STATE_CACHE_LIMIT:
                _runtime_state_cache.popitem(last=False)
        return current_frame, history_entries
    return latest


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "current_frame": frame_to_payload(current_frame),
        "history": [history_entry_to_payload(entry) for entry in history],
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    _invalidate_runtime_state_cache(path)
