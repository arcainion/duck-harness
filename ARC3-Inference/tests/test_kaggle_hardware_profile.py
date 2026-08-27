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
    DuckKaggleVllmConfig,
    duck_kaggle_setup_command,
    duck_kaggle_vllm_config_for_accelerator,
)
from inference.framework.solver import HarnessSolver


class KaggleHardwareProfileTests(TestCase):
    def test_solver_propagates_expected_gpu_shape(self) -> None:
        solver = HarnessSolver(
            kaggle_expected_gpu_type="t4",
            kaggle_expected_gpu_count=2,
        )

        config = solver._kaggle_vllm_config()

        self.assertEqual(config.expected_gpu_type, "t4")
        self.assertEqual(config.expected_gpu_count, 2)

    def test_solver_propagates_scheduler_tuning(self) -> None:
        solver = HarnessSolver(
            kaggle_vllm_gpu_memory_utilization=0.81,
            kaggle_vllm_max_num_seqs=7,
            kaggle_vllm_max_num_batched_tokens=4096,
            kaggle_vllm_enable_chunked_prefill=False,
        )

        config = solver._kaggle_vllm_config()

        self.assertEqual(config.gpu_memory_utilization, 0.81)
        self.assertEqual(config.max_num_seqs, 7)
        self.assertEqual(config.max_num_batched_tokens, 4096)
        self.assertFalse(config.enable_chunked_prefill)

    def test_t4_profile_matches_kaggle_dual_gpu_shape(self) -> None:
        config = duck_kaggle_vllm_config_for_accelerator("NvidiaTeslaT4")

        self.assertEqual(config.expected_gpu_type, "t4")
        self.assertEqual(config.expected_gpu_count, 2)
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertEqual(config.max_model_len, 8192)
        self.assertEqual(config.max_num_seqs, 16)
        self.assertEqual(config.max_num_batched_tokens, 8192)

    def test_t4_accelerator_matching_ignores_case_and_punctuation(self) -> None:
        for value in ("nvidia-tesla-t4", "NVIDIA TESLA T4", "NvidiaTeslaT4"):
            with self.subTest(value=value):
                self.assertEqual(
                    duck_kaggle_vllm_config_for_accelerator(value).expected_gpu_type,
                    "t4",
                )

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
        self.assertIn("VLLM_MAX_NUM_SEQS = 16", command)
        self.assertIn("VLLM_MAX_NUM_BATCHED_TOKENS = 8192", command)
        self.assertIn("--enable-chunked-prefill", command)

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
        self.assertIn("VLLM_GPU_MEMORY_UTILIZATION = 0.92", command)
        self.assertIn("VLLM_MAX_NUM_SEQS = 16", command)
        self.assertIn("VLLM_MAX_NUM_BATCHED_TOKENS = 8192", command)

    def test_custom_scheduler_values_are_rendered_and_capacities_are_bounded(self) -> None:
        command = duck_kaggle_setup_command(
            DuckKaggleVllmConfig(
                gpu_memory_utilization=0.77,
                max_num_seqs=0,
                max_num_batched_tokens=-9,
                enable_chunked_prefill=False,
            )
        )

        self.assertIn("VLLM_GPU_MEMORY_UTILIZATION = 0.77", command)
        self.assertIn("VLLM_MAX_NUM_SEQS = 1", command)
        self.assertIn("VLLM_MAX_NUM_BATCHED_TOKENS = 1", command)
        self.assertIn("VLLM_ENABLE_CHUNKED_PREFILL = False", command)

    def test_setup_preserves_context_window_when_already_below_server_limit(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_PROVIDER": "vllm",
                "LOCAL_ANALYZER_CONTEXT_WINDOW": "4096",
            },
            clear=False,
        ):
            command = duck_kaggle_setup_command(
                duck_kaggle_vllm_config_for_accelerator("NvidiaTeslaT4")
            )

        self.assertIn("ANALYZER_CONTEXT_WINDOW = 4096", command)

    def test_setup_forwards_next_run_safeguards(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_CANDIDATES": "1",
                "LOCAL_ANALYZER_MAX_OUTPUT": "4096",
                "LOCAL_ANALYZER_GAME_TOKEN_BUDGET": "100000",
                "LOCAL_ANALYZER_STAGNATION_WINDOW": "6",
                "LOCAL_ANALYZER_CYCLE_WINDOW": "4",
                "LOCAL_ANALYZER_CYCLE_STOP_LIMIT": "8",
                "LOCAL_ANALYZER_REPEAT_ACTION_LIMIT": "2",
                "LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_WINDOW": "16",
                "LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_LIMIT": "12",
                "LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STRIKE_LIMIT": "3",
                "LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STOP_LIMIT": "8",
                "LOCAL_ANALYZER_IGNORE_EDGE_HUD_CHANGES": "true",
                "LOCAL_ANALYZER_PROGRESS_UTILITY": "4.0",
                "LOCAL_ANALYZER_NOVEL_UTILITY": "0.05",
                "LOCAL_ANALYZER_EXPLORATION_WEIGHT": "0.5",
                "LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MULTIPLIER": "2.0",
                "LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM": "16",
            },
            clear=False,
        ):
            command = duck_kaggle_setup_command()

        self.assertIn("'LOCAL_ANALYZER_CANDIDATES': '1'", command)
        self.assertIn("'LOCAL_ANALYZER_MAX_OUTPUT': '4096'", command)
        self.assertIn("'LOCAL_ANALYZER_GAME_TOKEN_BUDGET': '100000'", command)
        self.assertIn("'LOCAL_ANALYZER_STAGNATION_WINDOW': '6'", command)
        self.assertIn("'LOCAL_ANALYZER_CYCLE_WINDOW': '4'", command)
        self.assertIn("'LOCAL_ANALYZER_CYCLE_STOP_LIMIT': '8'", command)
        self.assertIn("'LOCAL_ANALYZER_REPEAT_ACTION_LIMIT': '2'", command)
        self.assertIn(
            "'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_WINDOW': '16'", command
        )
        self.assertIn(
            "'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_LIMIT': '12'", command
        )
        self.assertIn(
            "'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STRIKE_LIMIT': '3'", command
        )
        self.assertIn(
            "'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STOP_LIMIT': '8'", command
        )
        self.assertIn("'LOCAL_ANALYZER_IGNORE_EDGE_HUD_CHANGES': 'true'", command)
        self.assertIn("'LOCAL_ANALYZER_PROGRESS_UTILITY': '4.0'", command)
        self.assertIn("'LOCAL_ANALYZER_NOVEL_UTILITY': '0.05'", command)
        self.assertIn("'LOCAL_ANALYZER_EXPLORATION_WEIGHT': '0.5'", command)
        self.assertIn("'LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MULTIPLIER': '2.0'", command)
        self.assertIn("'LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM': '16'", command)

    def test_setup_rejects_provider_incompatible_with_local_vllm(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"LOCAL_ANALYZER_PROVIDER": "openrouter"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "must be vLLM/OpenAI-compatible"):
                duck_kaggle_setup_command()

    def test_setup_rejects_malformed_dataset_references(self) -> None:
        for dataset_ref in ("", "owner-only", "owner/slug/extra", "/slug", "owner/"):
            with self.subTest(dataset_ref=dataset_ref):
                with self.assertRaisesRegex(ValueError, "owner/slug"):
                    duck_kaggle_setup_command(
                        DuckKaggleVllmConfig(wheelhouse_dataset_source=dataset_ref)
                    )

    def test_setup_rejects_non_numeric_context_window(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LOCAL_ANALYZER_PROVIDER": "vllm",
                "LOCAL_ANALYZER_CONTEXT_WINDOW": "many",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError):
                duck_kaggle_setup_command()

    def test_setup_rejects_non_numeric_scheduler_values(self) -> None:
        invalid_configs = (
            DuckKaggleVllmConfig(gpu_memory_utilization="high"),
            DuckKaggleVllmConfig(max_num_seqs="many"),
            DuckKaggleVllmConfig(max_num_batched_tokens="many"),
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    duck_kaggle_setup_command(config)

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
