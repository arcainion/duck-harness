from __future__ import annotations

import ast
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import TestCase, mock

from inference.framework.kaggle import (
    DEFAULT_VLLM_MAX_MODEL_LEN,
    duck_kaggle_setup_command,
    duck_kaggle_vllm_config_for_accelerator,
)
from inference.framework.solver import HarnessSolver


class KaggleHardwareProfileTests(TestCase):
    def test_t4_profile_matches_kaggle_dual_gpu_shape(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaTeslaT4")

        self.assertEqual(config.expected_gpu_type, "t4")
        self.assertEqual(config.expected_gpu_count, 2)
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertEqual(config.max_model_len, 8192)

    def test_t4_setup_clamps_analyzer_context_to_server_limit(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaTeslaT4")
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_PROVIDER": "vllm",
                "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
            },
            clear=False,
        ):
            command = duck_kaggle_setup_command(config)

        self.assertIn("VLLM_MAX_MODEL_LEN = 8192", command)
        self.assertIn("ANALYZER_CONTEXT_WINDOW = 8192", command)
        self.assertIn("VLLM_TENSOR_PARALLEL_SIZE = 2", command)
        self.assertIn("EXPECTED_GPU_TYPE = 't4'", command)
        self.assertIn("EXPECTED_GPU_COUNT = 2", command)

    def test_rtx_pro_6000_profile_uses_single_gpu_defaults(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaRtxPro6000")

        self.assertEqual(config.max_model_len, DEFAULT_VLLM_MAX_MODEL_LEN)
        self.assertEqual(config.tensor_parallel_size, 1)
        self.assertEqual(config.expected_gpu_type, "rtx-pro-6000")
        self.assertEqual(config.expected_gpu_count, 1)

        command = duck_kaggle_setup_command(config)
        self.assertIn("VLLM_MAX_MODEL_LEN = 65536", command)
        self.assertIn("VLLM_TENSOR_PARALLEL_SIZE = 1", command)
        self.assertIn("EXPECTED_GPU_TYPE = 'rtx-pro-6000'", command)
        self.assertIn("EXPECTED_GPU_COUNT = 1", command)

    def test_solver_wires_resolved_accelerator_profile_into_setup_command(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaTeslaT4")
        solver = HarnessSolver(
            kaggle_vllm_max_model_len=config.max_model_len,
            kaggle_vllm_tensor_parallel_size=config.tensor_parallel_size,
            kaggle_expected_gpu_type=config.expected_gpu_type,
            kaggle_expected_gpu_count=config.expected_gpu_count,
        )

        command = solver.kaggle_setup_commands[0]
        self.assertIn("VLLM_MAX_MODEL_LEN = 8192", command)
        self.assertIn("VLLM_TENSOR_PARALLEL_SIZE = 2", command)
        self.assertIn("EXPECTED_GPU_TYPE = 't4'", command)
        self.assertIn("EXPECTED_GPU_COUNT = 2", command)

    def test_setup_detects_server_exit_and_emits_complete_log(self) -> None:
        command = duck_kaggle_setup_command(
            duck_kaggle_vllm_config_for_accelerator("NvidiaTeslaT4")
        )

        self.assertIn("def read_server_log() -> str:", command)
        self.assertIn("return VLLM_SERVER_LOG.read_text", command)
        self.assertIn("returncode = process.poll()", command)
        self.assertIn("wait_for_vllm_server(process)", command)
        self.assertIn("Complete vLLM server log:", command)
        self.assertNotIn("os.kill(", command)
        self.assertNotIn("splitlines()[-lines:]", command)

        script = command.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
        parsed = ast.parse(script)
        wanted = {"read_server_log", "server_failure_message", "wait_for_vllm_server"}
        functions = ast.Module(
            body=[
                node
                for node in parsed.body
                if isinstance(node, ast.FunctionDef) and node.name in wanted
            ],
            type_ignores=[],
        )

        class ExitedProcess:
            def poll(self) -> int:
                return 17

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "vllm.log"
            log_path.write_text("earliest worker failure\nlatest engine failure\n", encoding="utf-8")
            namespace = {
                "Path": Path,
                "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
                "VLLM_SERVER_LOG": log_path,
                "request_json": mock.Mock(),
                "subprocess": subprocess,
                "time": time,
            }
            exec(compile(functions, "<kaggle-vllm-setup-test>", "exec"), namespace)

            with self.assertRaises(RuntimeError) as raised:
                namespace["wait_for_vllm_server"](ExitedProcess())

        message = str(raised.exception)
        self.assertIn("exited with code 17 before becoming ready", message)
        self.assertIn("earliest worker failure", message)
        self.assertIn("latest engine failure", message)

    def test_setup_runs_real_programir_codegen_smoke_before_benchmark(self) -> None:
        command = duck_kaggle_setup_command(
            duck_kaggle_vllm_config_for_accelerator("NvidiaRtxPro6000")
        )

        for required in (
            "def run_programir_codegen_smoke_test() -> None:",
            "program_tool_parameters_schema",
            "'tool_choice': 'required'",
            "compile_program(generated_program)",
            "run_sandboxed_python(",
            "execution.get('result') == 220",
            "duck_programir_codegen_smoke.json",
            "run_programir_codegen_smoke_test()",
        ):
            with self.subTest(required=required):
                self.assertIn(required, command)

        self.assertLess(
            command.index("run_programir_codegen_smoke_test()\nsetup_env"),
            command.index("setup_env_path ="),
        )

    def test_unknown_accelerator_uses_defaults(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaUnknownGPU")
        self.assertEqual(config.tensor_parallel_size, 1)
        self.assertGreater(config.max_model_len, 0)

    def test_rtx_pro_6000_setup_command_structure(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaRtxPro6000")
        command = duck_kaggle_setup_command(config)
        self.assertIn("def wait_for_vllm_server(", command)
        self.assertIn("VLLM_BASE_URL", command)

    def test_t4_single_gpu_config(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaTeslaT4")
        self.assertGreater(config.tensor_parallel_size, 0)

    def test_config_has_expected_attributes(self) -> None:
        for acc in ("NvidiaTeslaT4", "NvidiaRtxPro6000"):
            config = duck_kaggle_vllm_config_for_accelerator(acc)
            self.assertTrue(hasattr(config, "max_model_len"))
            self.assertTrue(hasattr(config, "tensor_parallel_size"))
            self.assertTrue(hasattr(config, "expected_gpu_type"))
            self.assertTrue(hasattr(config, "expected_gpu_count"))

    def test_solver_default_kaggle_commands(self) -> None:
        solver = HarnessSolver()
        self.assertIsInstance(solver.kaggle_setup_commands, list)

    def test_setup_command_mentions_vllm_server(self) -> None:
        for acc in ("NvidiaTeslaT4", "NvidiaRtxPro6000"):
            config = duck_kaggle_vllm_config_for_accelerator(acc)
            command = duck_kaggle_setup_command(config)
            self.assertIn("vllm", command.lower())
