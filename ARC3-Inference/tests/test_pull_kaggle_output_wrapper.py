from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class PullKaggleOutputWrapperTests(unittest.TestCase):
    def _fake_kaggle(self, root: Path) -> tuple[Path, dict[str, str]]:
        fake_bin = root / "bin"
        fake_bin.mkdir()
        fake_kaggle = fake_bin / "kaggle"
        fake_kaggle.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "kernels" ] && [ "$2" = "status" ] '
            '&& [ "${FAIL_STATUS:-}" = "true" ]; then exit 1; fi\n'
            'if [ "$1" = "kernels" ] && [ "$2" = "output" ]; then\n'
            "  shift 2\n"
            "  while [ \"$#\" -gt 0 ]; do\n"
            '    if [ "$1" = "-p" ]; then destination="$2"; shift 2; '
            "else shift; fi\n"
            "  done\n"
            '  touch "$destination/fresh-output.txt"\n'
            "fi\n",
            encoding="utf-8",
        )
        fake_kaggle.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
        return fake_kaggle, env

    def test_pull_cleans_existing_destination_contents(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        wrapper = project_dir / "pull-kaggle-output.sh"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "results"
            destination.mkdir()
            destination.joinpath("stale.txt").write_text("old", encoding="utf-8")
            destination.joinpath(".stale-hidden").mkdir()
            destination.joinpath("nested").mkdir()
            destination.joinpath("nested", "old.txt").write_text(
                "old", encoding="utf-8"
            )
            _, env = self._fake_kaggle(root)

            completed = subprocess.run(
                [
                    "bash",
                    str(wrapper),
                    "--skip-status",
                    "--path",
                    str(destination),
                ],
                cwd=project_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                ["fresh-output.txt"],
                [item.name for item in destination.iterdir()],
            )
            self.assertIn("Cleaning previous Kaggle output", completed.stdout)

    def test_failed_status_check_preserves_existing_output(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        wrapper = project_dir / "pull-kaggle-output.sh"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "results"
            destination.mkdir()
            stale = destination / "stale.txt"
            stale.write_text("keep", encoding="utf-8")
            _, env = self._fake_kaggle(root)
            env["FAIL_STATUS"] = "true"

            completed = subprocess.run(
                ["bash", str(wrapper), "--path", str(destination)],
                cwd=project_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(1, completed.returncode)
            self.assertEqual("keep", stale.read_text(encoding="utf-8"))
            self.assertFalse(destination.joinpath("fresh-output.txt").exists())

    def test_pull_refuses_filesystem_root(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        wrapper = project_dir / "pull-kaggle-output.sh"

        with tempfile.TemporaryDirectory() as temp_dir:
            _, env = self._fake_kaggle(Path(temp_dir))
            completed = subprocess.run(
                ["bash", str(wrapper), "--skip-status", "--path", "/"],
                cwd=project_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(2, completed.returncode)
        self.assertIn("refusing to clean unsafe output destination", completed.stderr)


if __name__ == "__main__":
    unittest.main()
