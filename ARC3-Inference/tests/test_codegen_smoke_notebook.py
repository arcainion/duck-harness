from __future__ import annotations

import json
import unittest
from pathlib import Path


NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "duck-programir-codegen-smoke"
)
NOTEBOOK_PATH = NOTEBOOK_DIR / "duck_programir_codegen_smoke.ipynb"
METADATA_PATH = NOTEBOOK_DIR / "kernel-metadata.json"


class CodegenSmokeNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_json_with_compilable_code_cells(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code_cells = [
            cell for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        self.assertEqual(len(code_cells), 4)
        for index, cell in enumerate(code_cells):
            with self.subTest(cell=index):
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    def test_notebook_runs_real_model_compiler_and_sandbox_path(self) -> None:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        for required in (
            "duck_kaggle_setup_command",
            "program_tool_parameters_schema",
            "'tool_choice': 'required'",
            "requests.post",
            "compile_program(generated_program)",
            "run_sandboxed_python",
            "generator_comprehension",
            "execution.get('result') == 220",
            "duck_programir_codegen_smoke.json",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_kaggle_metadata_attaches_offline_runtime_inputs(self) -> None:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(metadata["code_file"], NOTEBOOK_PATH.name)
        self.assertFalse(metadata["enable_internet"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaRtxPro6000")
        self.assertEqual(
            metadata["dataset_sources"],
            [
                "arcainionprime/taaf-kaggle-source",
                "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
            ],
        )


if __name__ == "__main__":
    unittest.main()
