from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from viewer import data
from viewer.server import _requested_run_dir


class ViewerSecurityTests(TestCase):
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

    def test_requested_run_with_empty_name(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / "runs"
            runs_dir.mkdir()
            result = _requested_run_dir(
                runs_dir=runs_dir,
                default_run_dir=root / "default-run",
                requested_run="",
            )
            self.assertEqual(result, root / "default-run")

    def test_requested_run_parent_traversal(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / "runs"
            runs_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                _requested_run_dir(
                    runs_dir=runs_dir,
                    default_run_dir=None,
                    requested_run="../etc/passwd",
                )

    def test_requested_run_with_dots(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / "runs"
            runs_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                _requested_run_dir(
                    runs_dir=runs_dir,
                    default_run_dir=None,
                    requested_run=".",
                )
            with self.assertRaises(FileNotFoundError):
                _requested_run_dir(
                    runs_dir=runs_dir,
                    default_run_dir=None,
                    requested_run="..",
                )

    def test_payload_cache_concurrent_writes(self) -> None:
        with data._RUN_PAYLOAD_CACHE_LOCK:
            data._RUN_PAYLOAD_CACHE.clear()

        def store(version: int) -> dict:
            key = ("run", "summary", (version,))
            return data._store_cached_payload(key, {"version": version}, scope_length=2)

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(store, range(200)))

        for r in results:
            self.assertIn("version", r)
