from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference.utils.viewer_artifacts import (
    append_raw_events_sidecar,
    load_raw_events,
    raw_events_jsonl_sidecar_path,
)


class ViewerArtifactRecoveryTests(unittest.TestCase):
    def test_append_repairs_interrupted_jsonl_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            viewer_path = Path(temp_dir) / "game_viewer_data.json"
            sidecar = raw_events_jsonl_sidecar_path(viewer_path)
            sidecar.write_bytes(b'{"step":1}\n{"step":')

            append_raw_events_sidecar(viewer_path, [{"step": 2}])

            self.assertEqual(
                load_raw_events({}, viewer_data_path=viewer_path),
                [{"step": 1}, {"step": 2}],
            )

    def test_append_preserves_valid_final_record_without_newline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            viewer_path = Path(temp_dir) / "game_viewer_data.json"
            sidecar = raw_events_jsonl_sidecar_path(viewer_path)
            sidecar.write_text(json.dumps({"step": 1}), encoding="utf-8")

            append_raw_events_sidecar(viewer_path, [{"step": 2}])

            self.assertEqual(
                load_raw_events({}, viewer_data_path=viewer_path),
                [{"step": 1}, {"step": 2}],
            )

    def test_loader_salvages_valid_events_around_corrupt_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            viewer_path = Path(temp_dir) / "game_viewer_data.json"
            sidecar = raw_events_jsonl_sidecar_path(viewer_path)
            sidecar.write_text(
                '{"step":1}\nnot-json\n{"step":2}\n',
                encoding="utf-8",
            )

            self.assertEqual(
                load_raw_events({}, viewer_data_path=viewer_path),
                [{"step": 1}, {"step": 2}],
            )

    def test_loader_salvages_events_around_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            viewer_path = Path(temp_dir) / "game_viewer_data.json"
            sidecar = raw_events_jsonl_sidecar_path(viewer_path)
            sidecar.write_bytes(b'{"step":1}\n\xff\xfe\n{"step":2}\n')

            self.assertEqual(
                load_raw_events({}, viewer_data_path=viewer_path),
                [{"step": 1}, {"step": 2}],
            )


if __name__ == "__main__":
    unittest.main()
