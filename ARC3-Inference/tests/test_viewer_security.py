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

    def test_objective_panel_escapes_model_authored_fields(self) -> None:
        index_html = (Path(__file__).parents[1] / "viewer" / "index.html").read_text(
            encoding="utf-8"
        )
        self.assertIn('escapeHtml(path.length ? path.join(" › ")', index_html)
        self.assertIn("selected?.success_criterion", index_html)
        self.assertIn("state.blocking_reason", index_html)
        self.assertIn('escapeHtml(evidence.join("; "))', index_html)
        self.assertIn("escapeHtml(lastOperation.operation)", index_html)
        self.assertIn("lastOperation.error_code", index_html)
        self.assertIn("selected.attempts_since_revision", index_html)
        self.assertIn("selected.attempt_budget", index_html)
