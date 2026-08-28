from __future__ import annotations

import json
from pathlib import Path

from taaf.deploy_kaggle import _write_kernel_bundle, normalize_model_ref


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


def test_kernel_bundle_declares_and_renders_model_sources(tmp_path: Path) -> None:
    model_source = "owner/model/pyTorch/fp8/3"

    _write_kernel_bundle(
        bundle_dir=tmp_path,
        kernel_id="owner/notebook",
        kernel_title="notebook",
        dataset_sources=["owner/source"],
        kernel_sources=[],
        model_sources=[model_source],
        private=True,
        enable_gpu=True,
        enable_internet=False,
        accelerator="NvidiaRtxPro6000",
        run_as_submission=False,
    )

    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    notebook = (tmp_path / "taaf_kaggle_run.ipynb").read_text()
    assert metadata["model_sources"] == [model_source]
    assert model_source in notebook
    assert "__TAAF_MODEL_SOURCES__" not in notebook


def test_model_ref_requires_a_complete_optional_numeric_version() -> None:
    assert normalize_model_ref("owner/model/pyTorch/fp8") == "owner/model/pyTorch/fp8"
    assert normalize_model_ref("owner/model/pyTorch/fp8/3") == "owner/model/pyTorch/fp8/3"
