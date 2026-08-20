from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from taaf.benchmark import Benchmark


def _write_benchmark_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "label": "audit",
                "n_passes": 1,
                "game_runs": [],
            }
        ),
        encoding="utf-8",
    )


class BenchmarkSecurityTests(TestCase):
    def test_from_json_does_not_load_pickle_sidecars_by_default(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "benchmark.json"
            _write_benchmark_json(path)

            with patch.object(
                Benchmark,
                "_load_intermediate_states",
                side_effect=AssertionError("pickle sidecar was loaded"),
            ):
                Benchmark.from_json(path)
                with self.assertRaisesRegex(AssertionError, "pickle sidecar was loaded"):
                    Benchmark.from_json(path, with_intermediate_states=True)

    def test_reset_run_state_discards_previous_invocation(self) -> None:
        benchmark = Benchmark(label="audit")
        benchmark.game_runs = [object()]  # type: ignore[list-item]
        benchmark.solver_label = "old solver"
        benchmark._deadline_fired = True
        benchmark.start_time = datetime(2000, 1, 1)
        benchmark.end_time = datetime(2000, 1, 2)

        benchmark._reset_run_state()

        self.assertEqual(benchmark.game_runs, [])
        self.assertEqual(benchmark.solver_label, "")
        self.assertFalse(benchmark._deadline_fired)
        self.assertIsNotNone(benchmark.start_time)
        assert benchmark.start_time is not None
        self.assertNotEqual(benchmark.start_time.year, 2000)
        self.assertIsNone(benchmark.end_time)

    def test_from_json_empty_game_runs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty_runs.json"
            path.write_text(
                json.dumps({"label": "empty", "n_passes": 1, "game_runs": []}),
                encoding="utf-8",
            )
            benchmark = Benchmark.from_json(path)
            self.assertEqual(len(benchmark.game_runs), 0)
            self.assertEqual(benchmark.label, "empty")

    def test_from_json_preserves_label(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labeled.json"
            path.write_text(
                json.dumps({"label": "custom-label", "n_passes": 2, "game_runs": []}),
                encoding="utf-8",
            )
            benchmark = Benchmark.from_json(path)
            self.assertEqual(benchmark.label, "custom-label")
            self.assertEqual(benchmark.n_passes, 2)

    def test_reset_run_state_clears_game_runs(self) -> None:
        benchmark = Benchmark(label="test")
        benchmark.game_runs = [object(), object(), object()]
        benchmark._reset_run_state()
        self.assertEqual(benchmark.game_runs, [])

    def test_reset_run_state_preserves_label(self) -> None:
        benchmark = Benchmark(label="keep-me")
        benchmark._reset_run_state()
        self.assertEqual(benchmark.label, "keep-me")

    def test_corrupted_json_raises_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corrupted.json"
            path.write_text("not valid json {{{", encoding="utf-8")
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                Benchmark.from_json(path)

    def test_missing_file_raises_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nonexistent.json"
            with self.assertRaises((FileNotFoundError, OSError)):
                Benchmark.from_json(path)
