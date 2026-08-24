from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference.agent.runtime_state import (
    _RUNTIME_STATE_CACHE_LIMIT,
    _runtime_state_cache,
    Frame,
    HistoryEntry,
    load_runtime_state,
    write_runtime_state,
)


class RuntimeStateCacheTests(unittest.TestCase):
    def test_unchanged_state_is_served_without_rereading_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            frame = Frame(grid=((0, 1),), step=1, level=1)
            write_runtime_state(
                path,
                current_frame=frame,
                history=[
                    HistoryEntry(
                        action="RIGHT",
                        frame=frame,
                        animation={"transient_changed_cells": 2},
                    )
                ],
            )
            first = load_runtime_state(path)

            with patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("unchanged state should use the cache"),
            ):
                second = load_runtime_state(path)

        self.assertEqual(second, first)
        self.assertIsNot(second[1], first[1])
        self.assertEqual(second[1][0].animation["transient_changed_cells"], 2)

    def test_write_invalidates_cached_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            first_frame = Frame(grid=((0,),), step=1, level=1)
            second_frame = Frame(grid=((1,),), step=2, level=1)
            write_runtime_state(path, current_frame=first_frame, history=[])
            self.assertEqual(load_runtime_state(path)[0], first_frame)

            write_runtime_state(path, current_frame=second_frame, history=[])
            refreshed, _history = load_runtime_state(path)

        self.assertEqual(refreshed, second_frame)

    def test_atomic_external_replacement_invalidates_by_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            replacement = Path(directory) / "replacement.json"
            first_frame = Frame(grid=((0,),), step=1, level=1)
            second_frame = Frame(grid=((1, 2),), step=2, level=1)
            write_runtime_state(path, current_frame=first_frame, history=[])
            self.assertEqual(load_runtime_state(path)[0], first_frame)

            write_runtime_state(
                replacement,
                current_frame=second_frame,
                history=[],
            )
            replacement.replace(path)
            refreshed, _history = load_runtime_state(path)

        self.assertEqual(refreshed, second_frame)

    def test_deleted_cached_file_returns_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            frame = Frame(grid=((0,),), step=1, level=1)
            write_runtime_state(path, current_frame=frame, history=[])
            self.assertEqual(load_runtime_state(path)[0], frame)

            path.unlink()
            current, history = load_runtime_state(path)

        self.assertIsNone(current)
        self.assertEqual(history, [])

    def test_cache_capacity_is_enforced_across_many_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index in range(_RUNTIME_STATE_CACHE_LIMIT + 5):
                path = Path(directory) / f"state-{index}.json"
                frame = Frame(grid=((index % 16,),), step=index, level=1)
                write_runtime_state(path, current_frame=frame, history=[])
                load_runtime_state(path)

            self.assertLessEqual(
                len(_runtime_state_cache),
                _RUNTIME_STATE_CACHE_LIMIT,
            )


if __name__ == "__main__":
    unittest.main()
