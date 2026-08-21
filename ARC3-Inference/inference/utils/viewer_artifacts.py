from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


_VIEWER_DATA_SUFFIX = "_viewer_data.json"
_RAW_EVENTS_SUFFIX = "_events.json.gz"
_RAW_EVENTS_JSONL_SUFFIX = "_events.jsonl"
_TAIL_SCAN_CHUNK_BYTES = 64 * 1024


def raw_events_sidecar_path(viewer_data_path: Path) -> Path:
    name = viewer_data_path.name
    if name.endswith(_VIEWER_DATA_SUFFIX):
        stem = name[: -len(_VIEWER_DATA_SUFFIX)]
        sidecar_name = f"{stem}{_RAW_EVENTS_SUFFIX}" if stem else f"viewer_data{_RAW_EVENTS_SUFFIX}"
        return viewer_data_path.with_name(sidecar_name)
    if name.endswith(".json"):
        return viewer_data_path.with_name(f"{viewer_data_path.stem}_events.json.gz")
    return viewer_data_path.with_name(f"{name}_events.json.gz")


def raw_events_jsonl_sidecar_path(viewer_data_path: Path) -> Path:
    name = viewer_data_path.name
    if name.endswith(_VIEWER_DATA_SUFFIX):
        stem = name[: -len(_VIEWER_DATA_SUFFIX)]
        sidecar_name = f"{stem}{_RAW_EVENTS_JSONL_SUFFIX}" if stem else f"viewer_data{_RAW_EVENTS_JSONL_SUFFIX}"
        return viewer_data_path.with_name(sidecar_name)
    if name.endswith(".json"):
        return viewer_data_path.with_name(f"{viewer_data_path.stem}_events.jsonl")
    return viewer_data_path.with_name(f"{name}_events.jsonl")


def reset_raw_events_sidecar(viewer_data_path: Path) -> None:
    raw_events_jsonl_sidecar_path(viewer_data_path).unlink(missing_ok=True)
    raw_events_sidecar_path(viewer_data_path).unlink(missing_ok=True)


def _repair_jsonl_tail(sidecar_path: Path) -> None:
    """Drop an interrupted trailing record or terminate a valid final record."""
    if not sidecar_path.exists():
        return
    with sidecar_path.open("r+b") as file:
        file.seek(0, 2)
        end = file.tell()
        if end == 0:
            return
        file.seek(end - 1)
        if file.read(1) == b"\n":
            return

        cursor = end
        later_chunks: list[bytes] = []
        tail_start = 0
        tail = b""
        while cursor > 0:
            chunk_size = min(_TAIL_SCAN_CHUNK_BYTES, cursor)
            cursor -= chunk_size
            file.seek(cursor)
            chunk = file.read(chunk_size)
            newline_index = chunk.rfind(b"\n")
            if newline_index >= 0:
                tail_start = cursor + newline_index + 1
                tail = chunk[newline_index + 1 :] + b"".join(
                    reversed(later_chunks)
                )
                break
            later_chunks.append(chunk)
        else:
            tail = b"".join(reversed(later_chunks))

        try:
            parsed = json.loads(tail.decode("utf-8"))
            valid_tail = isinstance(parsed, dict)
        except (UnicodeDecodeError, json.JSONDecodeError):
            valid_tail = False

        file.seek(end if valid_tail else tail_start)
        if valid_tail:
            file.write(b"\n")
        else:
            file.truncate()


def append_raw_events_sidecar(viewer_data_path: Path, events: list[dict[str, Any]]) -> Path:
    sidecar_path = raw_events_jsonl_sidecar_path(viewer_data_path)
    if not events:
        return sidecar_path
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    _repair_jsonl_tail(sidecar_path)
    encoded = b"".join(
        (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")
        for event in events
    )
    with sidecar_path.open("ab") as file:
        file.write(encoded)
    return sidecar_path


def write_raw_events_sidecar(viewer_data_path: Path, events: list[dict[str, Any]]) -> Path:
    sidecar_path = raw_events_sidecar_path(viewer_data_path)
    payload = json.dumps(events, separators=(",", ":")).encode("utf-8")
    sidecar_path.write_bytes(gzip.compress(payload))
    return sidecar_path


def _load_jsonl_events(sidecar_path: Path) -> list[dict[str, Any]] | None:
    if not sidecar_path.exists():
        return None
    try:
        lines = sidecar_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return []

    events: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def load_raw_events(
    payload: dict[str, Any],
    *,
    viewer_data_path: Path | None = None,
) -> list[dict[str, Any]]:
    raw_events = payload.get("events")
    if isinstance(raw_events, list):
        return [event for event in raw_events if isinstance(event, dict)]

    if viewer_data_path is None:
        return []

    jsonl_events = _load_jsonl_events(raw_events_jsonl_sidecar_path(viewer_data_path))
    if jsonl_events is not None:
        return jsonl_events

    sidecar_path = raw_events_sidecar_path(viewer_data_path)
    if not sidecar_path.exists():
        return []

    try:
        decoded = gzip.decompress(sidecar_path.read_bytes()).decode("utf-8")
        parsed = json.loads(decoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [event for event in parsed if isinstance(event, dict)]
