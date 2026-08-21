from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from inference.agent.runtime_state import load_runtime_state


class RuntimeStateRecoveryTests(unittest.TestCase):
    def test_non_object_json_degrades_to_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            for payload in ("null", "[]", '"state"'):
                path.write_text(payload, encoding="utf-8")
                self.assertEqual(load_runtime_state(path), (None, []))

    def test_null_history_is_treated_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            path.write_text(
                '{"current_frame":{"grid":[[1]],"step":2,"level":3},'
                '"history":null}',
                encoding="utf-8",
            )

            frame, history = load_runtime_state(path)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.grid, ((1,),))
        self.assertEqual(history, [])

    def test_non_finite_frame_numbers_use_safe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            path.write_text(
                '{"current_frame":{"grid":[[Infinity]],'
                '"step":Infinity,"level":-Infinity},"history":[]}',
                encoding="utf-8",
            )

            frame, history = load_runtime_state(path)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.grid, ((0,),))
        self.assertEqual(frame.step, 0)
        self.assertEqual(frame.level, 1)
        self.assertEqual(history, [])

    def test_file_disappearing_after_exists_check_is_contained(self) -> None:
        path = MagicMock(spec=Path)
        path.exists.return_value = True
        path.read_text.side_effect = OSError("file disappeared")

        self.assertEqual(load_runtime_state(path), (None, []))


if __name__ == "__main__":
    unittest.main()
