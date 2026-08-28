from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class KaggleDuckWrapperTests(unittest.TestCase):
    def test_diagnostic_mode_selects_five_games_and_safe_concurrency(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        wrapper = project_dir / "kaggle-duck.sh"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            make_log = root / "make-arguments.json"
            fake_bin.joinpath("make").write_text(
                "#!/bin/sh\n"
                "python3 -c 'import json, os, sys; json.dump(sys.argv[1:], "
                "open(os.environ[\"MAKE_LOG\"], \"w\"))' \"$@\"\n",
                encoding="utf-8",
            )
            fake_bin.joinpath("make").chmod(0o755)
            env = dict(os.environ)
            env.update(
                {
                    "HOME": str(root),
                    "MAKE_LOG": str(make_log),
                    "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                    "KAGGLE_DRY_RUN": "true",
                    "KAGGLE_DUCK_DIAGNOSTIC": "true",
                }
            )

            subprocess.run(
                ["bash", str(wrapper)],
                cwd=project_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = json.loads(make_log.read_text(encoding="utf-8"))

        self.assertIn("KAGGLE_DUCK_PUBLIC_HARNESS=false", arguments)
        self.assertIn("CONCURRENT_JOBS=3", arguments)
        self.assertIn("ANALYZER_TIMEOUT=900", arguments)
        game_argument = next(item for item in arguments if item.startswith("GAME="))
        self.assertIn("ft09-0d8bbf25", game_argument)
        self.assertIn("r11l-495a7899", game_argument)

    def test_file_credentials_are_safe_and_available_before_make(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        wrapper = project_dir / "kaggle-duck.sh"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            home = root / "home"
            credentials_dir = home / ".kaggle"
            credentials_dir.mkdir(parents=True)
            marker = root / "credential-was-executed"
            credentials_dir.joinpath("kaggle.json").write_text(
                json.dumps(
                    {
                        "username": "safe-owner",
                        "key": f"$(touch {marker})",
                    }
                ),
                encoding="utf-8",
            )

            make_log = root / "make-environment.json"
            kaggle_log = root / "kaggle-arguments.txt"
            fake_bin.joinpath("make").write_text(
                "#!/bin/sh\n"
                "python3 -c 'import json, os, sys; "
                "json.dump({\"args\": sys.argv[1:], "
                "\"username\": os.environ.get(\"KAGGLE_USERNAME\"), "
                "\"key\": os.environ.get(\"KAGGLE_KEY\"), "
                "\"token\": os.environ.get(\"KAGGLE_API_TOKEN\")}, "
                "open(os.environ[\"MAKE_LOG\"], \"w\"))' \"$@\"\n",
                encoding="utf-8",
            )
            fake_bin.joinpath("kaggle").write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" > \"$KAGGLE_LOG\"\n",
                encoding="utf-8",
            )
            fake_bin.joinpath("make").chmod(0o755)
            fake_bin.joinpath("kaggle").chmod(0o755)

            env = dict(os.environ)
            env.update(
                {
                    "HOME": str(home),
                    "MAKE_LOG": str(make_log),
                    "KAGGLE_LOG": str(kaggle_log),
                    "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                    "RUN_NAME": "audit-test",
                    "KAGGLE_DRY_RUN": "false",
                }
            )
            for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_DATASET_REF"):
                env.pop(key, None)

            completed = subprocess.run(
                ["bash", str(wrapper)],
                cwd=project_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            recorded = json.loads(make_log.read_text(encoding="utf-8"))
            status_arguments = kaggle_log.read_text(encoding="utf-8").strip()

        self.assertFalse(marker.exists())
        self.assertEqual(recorded["username"], "safe-owner")
        self.assertEqual(recorded["key"], f"$(touch {marker})")
        self.assertEqual(recorded["token"], f"$(touch {marker})")
        self.assertIn(
            "KAGGLE_DATASET_REF=safe-owner/taaf-kaggle-source-audit-test",
            recorded["args"],
        )
        self.assertIn(
            "Notebook: https://www.kaggle.com/code/safe-owner/taaf-audit-test",
            completed.stdout,
        )
        self.assertEqual(
            status_arguments,
            "kernels status safe-owner/taaf-audit-test",
        )


if __name__ == "__main__":
    unittest.main()
