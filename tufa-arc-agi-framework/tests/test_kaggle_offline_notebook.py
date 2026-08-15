from __future__ import annotations

import json
from pathlib import Path


def test_normal_kaggle_notebook_configures_explicit_offline_environment_dir() -> None:
    notebook_path = (
        Path(__file__).parents[1]
        / "src"
        / "taaf"
        / "kaggle"
        / "taaf_kaggle_run.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "def _configure_offline_games(games, env_dir: Path)" in code
    assert "dataclasses.replace(game.arcade_spec, environments_dir=str(env_dir))" in code
    assert '_configure_offline_games(bm.games, wheelhouse.parent / "environment_files")' in code
    assert code.index("if true_submission:") < code.index("else:\n        # Ordinary Save & Run")
