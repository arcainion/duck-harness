from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from viewer import data
from viewer.server import _ViewerHandler, _requested_run_dir


class ViewerSecurityTests(TestCase):
    def test_client_disconnect_during_response_is_tolerated(self) -> None:
        handler = object.__new__(_ViewerHandler)
        handler.wfile = mock.Mock()
        handler.wfile.write.side_effect = ConnectionAbortedError("client closed")

        handler._write_response_body(b"payload")

        handler.wfile.write.assert_called_once_with(b"payload")

    def test_requested_run_is_confined_to_runs_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / "runs"
            runs_dir.mkdir()
            run_name = "20260814_120000_audit"

            self.assertEqual(
                _requested_run_dir(
                    runs_dir=runs_dir,
                    default_run_dir=None,
                    requested_run=run_name,
                ),
                runs_dir / run_name,
            )

            for malicious_name in ("../outside", str((root / "outside").resolve())):
                with self.assertRaises(FileNotFoundError):
                    _requested_run_dir(
                        runs_dir=runs_dir,
                        default_run_dir=None,
                        requested_run=malicious_name,
                    )

    def test_explicit_default_run_remains_selectable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / "runs"
            default_run = root / "example-run"

            self.assertEqual(
                _requested_run_dir(
                    runs_dir=runs_dir,
                    default_run_dir=default_run,
                    requested_run="example-run",
                ),
                default_run,
            )

    def test_payload_cache_updates_are_thread_safe(self) -> None:
        with data._RUN_PAYLOAD_CACHE_LOCK:
            data._RUN_PAYLOAD_CACHE.clear()

        def store(version: int) -> dict:
            key = ("run", "summary", (version,))
            return data._store_cached_payload(key, {"version": version}, scope_length=2)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(store, range(100)))

        with data._RUN_PAYLOAD_CACHE_LOCK:
            self.assertEqual(len(data._RUN_PAYLOAD_CACHE), 1)

    def test_empty_top_level_artifacts_do_not_hide_pass_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260814_120000_audit"
            (run_dir / "artifacts").mkdir(parents=True)
            pass_artifacts = run_dir / "passes" / "0" / "artifacts"
            pass_artifacts.mkdir(parents=True)
            viewer_path = pass_artifacts / "game_viewer_data.json"
            viewer_path.write_text(json.dumps({"viewer_steps": []}), encoding="utf-8")

            self.assertEqual(data._viewer_data_paths(run_dir), [viewer_path])

    def test_direct_and_split_artifacts_are_both_discoverable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260814_120000_audit"
            direct_artifacts = run_dir / "artifacts"
            direct_artifacts.mkdir(parents=True)
            direct_path = direct_artifacts / "direct_viewer_data.json"
            direct_path.write_text(json.dumps({"viewer_steps": []}), encoding="utf-8")
            seed_artifacts = run_dir / "seeds" / "2" / "artifacts"
            seed_artifacts.mkdir(parents=True)
            seed_path = seed_artifacts / "seed_viewer_data.json"
            seed_path.write_text(json.dumps({"viewer_steps": []}), encoding="utf-8")

            self.assertEqual(data._viewer_data_paths(run_dir), [direct_path, seed_path])

    def test_viewer_artifacts_sort_passes_numerically_within_each_game(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260814_120000_audit"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            names = [
                "game-b_p1_viewer_data.json",
                "game-a_p10_viewer_data.json",
                "game-a_p2_viewer_data.json",
                "game-a_p1_viewer_data.json",
            ]
            for name in names:
                (artifacts / name).write_text(
                    json.dumps({"viewer_steps": []}), encoding="utf-8"
                )

            self.assertEqual(
                [path.name for path in data._viewer_data_paths(run_dir)],
                [
                    "game-a_p1_viewer_data.json",
                    "game-a_p2_viewer_data.json",
                    "game-a_p10_viewer_data.json",
                    "game-b_p1_viewer_data.json",
                ],
            )

    def test_pass_request_log_changes_invalidate_run_fingerprint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "20260814_120000_audit"
            request_log = run_dir / "passes" / "0" / "game_requests.jsonl"
            request_log.parent.mkdir(parents=True)
            request_log.write_text("{}\n", encoding="utf-8")
            before = data._run_dir_fingerprint(run_dir)

            request_log.write_text('{"request":1}\n', encoding="utf-8")

            self.assertNotEqual(data._run_dir_fingerprint(run_dir), before)
