from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import TestCase, mock

from inference.framework.kaggle import (
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DuckKaggleVllmConfig,
    duck_kaggle_setup_command,
    duck_kaggle_teardown_command,
    duck_kaggle_vllm_config_for_accelerator,
)
from inference.framework.solver import HarnessSolver


class KaggleHardwareProfileTests(TestCase):
    def test_invalid_vllm_profiles_fail_before_deployment(self) -> None:
        invalid_configs = (
            DuckKaggleVllmConfig(vllm_port=0),
            DuckKaggleVllmConfig(max_model_len=0),
            DuckKaggleVllmConfig(gpu_memory_utilization=float("nan")),
            DuckKaggleVllmConfig(gpu_memory_utilization=1.01),
            DuckKaggleVllmConfig(served_model_name=" "),
        )
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                duck_kaggle_setup_command(config)

    def test_setup_cleans_started_server_after_boot_or_smoke_failure(self) -> None:
        command = duck_kaggle_setup_command()

        self.assertIn("def stop_started_vllm_server(", command)
        self.assertIn("except BaseException:\n        stop_started_vllm_server(process)", command)
        self.assertIn("except BaseException:\n    stop_started_vllm_server(vllm_process)", command)

        script = command.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
        parsed = ast.parse(script)
        cleanup_function = ast.Module(
            body=[
                node
                for node in parsed.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "stop_started_vllm_server"
            ],
            type_ignores=[],
        )

        class StuckProcess:
            def __init__(self) -> None:
                self.terminated = False
                self.killed = False
                self.wait_calls = 0

            def poll(self):
                return None

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout: int) -> None:
                self.wait_calls += 1
                if self.wait_calls == 1:
                    raise subprocess.TimeoutExpired("vllm", timeout)

            def kill(self) -> None:
                self.killed = True

        with tempfile.TemporaryDirectory() as temp_dir:
            pid_path = Path(temp_dir) / "vllm.pid"
            pid_path.write_text("123", encoding="utf-8")
            namespace = {
                "subprocess": subprocess,
                "VLLM_SERVER_PID": pid_path,
            }
            exec(compile(cleanup_function, "<kaggle-cleanup-test>", "exec"), namespace)
            process = StuckProcess()
            namespace["stop_started_vllm_server"](process)

            self.assertTrue(process.terminated)
            self.assertTrue(process.killed)
            self.assertEqual(process.wait_calls, 2)
            self.assertFalse(pid_path.exists())

    def test_teardown_refuses_to_signal_unrelated_stale_pid(self) -> None:
        command = duck_kaggle_teardown_command()

        self.assertIn("Path('/proc') / str(pid) / 'cmdline'", command)
        self.assertIn("vllm.entrypoints.openai.api_server", command)
        self.assertIn("Refusing to signal PID", command)

    def test_default_analyzer_profile_disables_hidden_thinking(self) -> None:
        config_path = Path(__file__).parents[1] / "configs" / "inference.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertIs(config["analyzer"]["thinking"], False)

    def test_default_multimodal_profile_sends_one_small_image_per_level(self) -> None:
        config_path = Path(__file__).parents[1] / "configs" / "inference.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["multimodal"]["context"], "level_start")
        self.assertEqual(config["multimodal"]["upscale"], 2)

        command = duck_kaggle_setup_command()
        self.assertIn("'MULTIMODAL_CONTEXT': 'level_start'", command)
        self.assertIn("'MULTIMODAL_UPSCALE': '2'", command)

    def test_t4_profile_matches_kaggle_dual_gpu_shape(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaTeslaT4")

        self.assertEqual(config.expected_gpu_type, "t4")
        self.assertEqual(config.expected_gpu_count, 2)
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertEqual(config.max_model_len, 8192)
        self.assertEqual(config.max_num_batched_tokens, 4096)
        self.assertEqual(config.max_num_seqs, 16)

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
        self.assertIn("VLLM_MAX_NUM_BATCHED_TOKENS = 4096", command)
        self.assertIn("VLLM_MAX_NUM_SEQS = 16", command)
        self.assertIn("EXPECTED_GPU_TYPE = 't4'", command)
        self.assertIn("EXPECTED_GPU_COUNT = 2", command)

    def test_rtx_pro_6000_profile_uses_single_gpu_defaults(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaRtxPro6000")

        self.assertEqual(config.max_model_len, DEFAULT_VLLM_MAX_MODEL_LEN)
        self.assertEqual(config.tensor_parallel_size, 1)
        self.assertEqual(config.expected_gpu_type, "rtx-pro-6000")
        self.assertEqual(config.expected_gpu_count, 1)
        self.assertEqual(config.gpu_memory_utilization, 0.92)
        self.assertEqual(config.max_num_batched_tokens, 16384)
        self.assertEqual(config.max_num_seqs, 32)

        command = duck_kaggle_setup_command(config)
        self.assertIn("VLLM_MAX_MODEL_LEN = 65536", command)
        self.assertIn("VLLM_TENSOR_PARALLEL_SIZE = 1", command)
        self.assertIn("VLLM_GPU_MEMORY_UTILIZATION = 0.92", command)
        self.assertIn("VLLM_MAX_NUM_BATCHED_TOKENS = 16384", command)
        self.assertIn("VLLM_MAX_NUM_SEQS = 32", command)
        self.assertIn("'--enable-chunked-prefill'", command)
        self.assertIn("'--uvicorn-log-level'", command)
        self.assertIn("'warning'", command)
        self.assertIn("'--disable-uvicorn-access-log'", command)
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
            command.index("run_programir_codegen_smoke_test()\nexcept BaseException:"),
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
