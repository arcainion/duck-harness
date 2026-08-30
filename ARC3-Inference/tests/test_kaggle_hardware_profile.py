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
    DEFAULT_QWEN_MODEL_SOURCE,
    DEFAULT_SERVED_MODEL_NAME,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_REASONING_CONFIG,
    DuckKaggleVllmConfig,
    duck_kaggle_model_sources,
    duck_kaggle_setup_command,
    duck_kaggle_vllm_config_for_accelerator,
)
from inference.framework.solver import HarnessSolver


class KaggleHardwareProfileTests(TestCase):
    def test_default_analyzer_enables_and_preserves_thinking_for_qwen38(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "inference.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertTrue(config["analyzer"]["thinking"])
        self.assertTrue(config["chat"]["thinking"])
        self.assertTrue(
            config["server"]["default_chat_template_kwargs"]["preserve_thinking"]
        )

    def test_default_model_is_qwen38_fp8_repacked_kaggle_model(self) -> None:
        solver = HarnessSolver()

        self.assertEqual(
            solver.kaggle_model_sources,
            ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pyTorch/hf-fp8/1"],
        )
        self.assertEqual(solver.kaggle_model_sources, [DEFAULT_QWEN_MODEL_SOURCE])
        self.assertEqual(solver.kaggle_served_model_name, DEFAULT_SERVED_MODEL_NAME)
        self.assertEqual(
            solver.kaggle_dataset_sources,
            ["driessmit1/arc3-vllm-h100-wheelhouse-v3"],
        )

        command = duck_kaggle_setup_command()
        self.assertIn(f"MODEL_SOURCE = {DEFAULT_QWEN_MODEL_SOURCE!r}", command)
        self.assertIn("kagglehub.model_download(model_source)", command)
        self.assertIn(f"SERVED_MODEL_NAME = {DEFAULT_SERVED_MODEL_NAME!r}", command)

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
        self.assertIn('{"preserve_thinking": true}', command)

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

    def test_setup_configures_bounded_reasoning_and_raw_transport_smoke(self) -> None:
        command = duck_kaggle_setup_command()

        self.assertIn("'--reasoning-parser',\n        'qwen3'", command)
        self.assertIn("'--reasoning-config'", command)
        self.assertIn(repr(DEFAULT_VLLM_REASONING_CONFIG), command)
        self.assertIn("'chat_template_kwargs': {'enable_thinking': True}", command)
        self.assertIn("'thinking_token_budget': 64", command)
        self.assertNotIn("'tool_choice': 'required'", command)
        self.assertIn("BEGIN_REDUCTION", command)
        self.assertIn("BEGIN_POLICY", command)
        self.assertIn("Raw reduction JSON fidelity: passed", command)
        self.assertIn("Raw policy source fidelity: passed", command)
        self.assertIn("run_vllm_api_smoke_test()", command)

    def test_bounded_reasoning_smoke_accepts_raw_orchestration_envelopes(self) -> None:
        command = duck_kaggle_setup_command()
        script = command.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
        parsed = ast.parse(script)
        functions = ast.Module(
            body=[
                node
                for node in parsed.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "run_vllm_api_smoke_test"
            ],
            type_ignores=[],
        )
        reduction_fixture = (
            "BEGIN_REDUCTION\n"
            "{\n"
            '  "objective_id": "level:1:1",\n'
            '  "verdict": "continue",\n'
            '  "evidence": "board unchanged",\n'
            '  "rationale": "continue the active probe",\n'
            '  "selected_index": 0,\n'
            '  "subgoals": []\n'
            "}\n"
            "END_REDUCTION"
        )
        policy_fixture = (
            "BEGIN_POLICY\n"
            "POLICY_API_VERSION = 1\n"
            'SUPPORTED_BACKENDS = ("cpu",)\n'
            "def decide(observation, memory):\n"
            '    return {"status": "continue", "action": '
            '{"action": "ACTION6"}, "memory": memory}\n'
            "END_POLICY"
        )
        request_json = mock.Mock(
            side_effect=[
                {
                    "choices": [
                        {
                            "message": {
                                "content": reduction_fixture,
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": policy_fixture,
                            }
                        }
                    ]
                },
            ]
        )
        namespace = {
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason,
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        namespace["run_vllm_api_smoke_test"]()

        self.assertEqual(2, request_json.call_count)
        reduction_payload = request_json.call_args_list[0].kwargs["payload"]
        self.assertEqual(64, reduction_payload["thinking_token_budget"])
        self.assertEqual(
            {"enable_thinking": True}, reduction_payload["chat_template_kwargs"]
        )
        self.assertNotIn("tools", reduction_payload)
        self.assertNotIn("tool_choice", reduction_payload)
        self.assertIn(reduction_fixture, reduction_payload["messages"][0]["content"])
        raw_payload = request_json.call_args_list[1].kwargs["payload"]
        self.assertEqual(64, raw_payload["thinking_token_budget"])
        self.assertNotIn("tools", raw_payload)
        self.assertNotIn("tool_choice", raw_payload)
        self.assertIn(policy_fixture, raw_payload["messages"][0]["content"])

    def test_bounded_reasoning_smoke_fails_on_raw_reduction_corruption(
        self,
    ) -> None:
        command = duck_kaggle_setup_command()
        script = command.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
        parsed = ast.parse(script)
        functions = ast.Module(
            body=[
                node
                for node in parsed.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "run_vllm_api_smoke_test"
            ],
            type_ignores=[],
        )
        namespace = {
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": mock.Mock(
                return_value={"choices": [{"message": {"content": "4"}}]}
            ),
            "server_failure_message": lambda reason: reason + "\ncomplete server log",
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        with self.assertRaisesRegex(
            RuntimeError, "raw-reduction fidelity check"
        ) as raised:
            namespace["run_vllm_api_smoke_test"]()

        self.assertIn("complete server log", str(raised.exception))

    def test_bounded_reasoning_smoke_fails_on_raw_policy_corruption(self) -> None:
        command = duck_kaggle_setup_command()
        script = command.split("\n", 1)[1].rsplit("\nPYSETUP", 1)[0]
        parsed = ast.parse(script)
        functions = ast.Module(
            body=[
                node
                for node in parsed.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "run_vllm_api_smoke_test"
            ],
            type_ignores=[],
        )
        request_json = mock.Mock(
            side_effect=[
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "BEGIN_REDUCTION\n"
                                    "{\n"
                                    '  "objective_id": "level:1:1",\n'
                                    '  "verdict": "continue",\n'
                                    '  "evidence": "board unchanged",\n'
                                    '  "rationale": "continue the active probe",\n'
                                    '  "selected_index": 0,\n'
                                    '  "subgoals": []\n'
                                    "}\n"
                                    "END_REDUCTION"
                                ),
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": "BEGIN_POLICY POLICY_API_VERSION = 1 END_POLICY"
                            }
                        }
                    ]
                },
            ]
        )
        namespace = {
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason + "\ncomplete server log",
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        with self.assertRaisesRegex(
            RuntimeError, "raw-policy fidelity check"
        ) as raised:
            namespace["run_vllm_api_smoke_test"]()

        self.assertIn("complete server log", str(raised.exception))

    def test_custom_scheduler_values_are_rendered_and_capacities_are_bounded(
        self,
    ) -> None:
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

    def test_setup_preserves_context_window_when_already_below_server_limit(
        self,
    ) -> None:
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
                "LOCAL_ANALYZER_MAX_OUTPUT": "0",
                "LOCAL_ANALYZER_TOOL_STEPS": "0",
                "LOCAL_ANALYZER_ENABLE_THINKING": "true",
                "LOCAL_ANALYZER_GAME_TOKEN_BUDGET": "100000",
                "LOCAL_ANALYZER_OBJECTIVE_REDUCTION": "true",
                "LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS": "300",
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT": "4096",
                "LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT": "8192",
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET": "2048",
                "LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET": "3072",
                "LOCAL_GAMEPLAY_POLICY_BACKEND": "cpu",
                "LOCAL_GAMEPLAY_POLICY_CUDA_MIN_FREE_MB": "4096",
                "LOCAL_GAMEPLAY_POLICY_DECISION_TIMEOUT_SECONDS": "2",
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
                "LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM": "20",
                "LOCAL_ANALYZER_LEVEL_NO_PROGRESS_TOKEN_LIMIT": "75000",
            },
            clear=False,
        ):
            command = duck_kaggle_setup_command()

        self.assertIn("'LOCAL_ANALYZER_CANDIDATES': '1'", command)
        self.assertIn("'LOCAL_ANALYZER_MAX_OUTPUT': '0'", command)
        self.assertIn("'LOCAL_ANALYZER_TOOL_STEPS': '0'", command)
        self.assertIn("'LOCAL_ANALYZER_ENABLE_THINKING': 'true'", command)
        self.assertIn("'LOCAL_ANALYZER_GAME_TOKEN_BUDGET': '100000'", command)
        self.assertIn("'LOCAL_ANALYZER_OBJECTIVE_REDUCTION': 'true'", command)
        self.assertIn(
            "'LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS': '300'",
            command,
        )
        self.assertIn(
            "'LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT': '4096'",
            command,
        )
        self.assertIn(
            "'LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT': '8192'",
            command,
        )
        self.assertIn(
            "'LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET': '2048'",
            command,
        )
        self.assertIn(
            "'LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET': '3072'",
            command,
        )
        self.assertIn("'LOCAL_GAMEPLAY_POLICY_BACKEND': 'cpu'", command)
        self.assertIn("'LOCAL_GAMEPLAY_POLICY_CUDA_MIN_FREE_MB': '4096'", command)
        self.assertIn("'LOCAL_GAMEPLAY_POLICY_DECISION_TIMEOUT_SECONDS': '2'", command)
        self.assertIn("'LOCAL_ANALYZER_STAGNATION_WINDOW': '6'", command)
        self.assertIn("'LOCAL_ANALYZER_CYCLE_WINDOW': '4'", command)
        self.assertIn("'LOCAL_ANALYZER_CYCLE_STOP_LIMIT': '8'", command)
        self.assertIn("'LOCAL_ANALYZER_REPEAT_ACTION_LIMIT': '2'", command)
        self.assertIn("'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_WINDOW': '16'", command)
        self.assertIn("'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_LIMIT': '12'", command)
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
        self.assertIn("'LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM': '20'", command)
        self.assertIn(
            "'LOCAL_ANALYZER_LEVEL_NO_PROGRESS_TOKEN_LIMIT': '75000'", command
        )

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

    def test_setup_rejects_malformed_model_references(self) -> None:
        for model_ref in ("", "owner/model", "owner/model/pytorch/fp8/latest"):
            with self.subTest(model_ref=model_ref):
                with self.assertRaisesRegex(ValueError, "Kaggle Model handle|numeric"):
                    duck_kaggle_model_sources(
                        DuckKaggleVllmConfig(model_source=model_ref)
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
            log_path.write_text(
                "earliest worker failure\nlatest engine failure\n", encoding="utf-8"
            )
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
