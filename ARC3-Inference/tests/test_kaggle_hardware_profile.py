from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest import TestCase, mock

import numpy as np

from inference.agent.gameplay_policy_runtime import PolicyObservation
from inference.agent.policy_solver_helpers import solver_decide
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


GENERATED_NAVIGATION_POLICY = """BEGIN_POLICY
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
POLICY_REUSE_SCOPE = "none"
POLICY_SOLVER_TYPE = "navigation"
POLICY_SOLVER_CONFIG = {
    "actor_values": [1],
    "target_values": [2],
    "passable_values": [0, 1, 2],
    "approach_distance": 1,
    "probe_actions": ["UP", "RIGHT"],
}

def decide(observation, memory):
    return solver_decide(
        POLICY_SOLVER_TYPE, observation, memory, POLICY_SOLVER_CONFIG
    )
END_POLICY"""

PATHFINDING_POLICY_BEHAVIOR = {
    "corridor": {
        "status": "continue",
        "path_actions": ["RIGHT", "RIGHT"],
        "action": "RIGHT",
    },
    "detour": {
        "status": "continue",
        "path_actions": ["DOWN", "RIGHT", "RIGHT", "UP"],
        "action": "DOWN",
    },
    "at_approach": {
        "status": "subgoal_succeeded",
        "path_actions": [],
        "action": None,
    },
    "unreachable": {
        "status": "subgoal_failed",
        "path_actions": [],
        "action": None,
    },
    "engine_progress_unreachable": {
        "status": "continue",
        "path_actions": [],
        "action": "UP",
    },
}

GAME_INFERENCE_BEHAVIOR = {
    "directional_motion": {
        "inferred_effect": "actor_translation",
        "hypothesis_verdict": "supported",
        "meaningful_progress": False,
        "novel_state": True,
        "cycle_risk": False,
        "recommended_action": "RIGHT",
    },
    "exact_noop": {
        "inferred_effect": "exact_noop",
        "hypothesis_verdict": "contradicted",
        "meaningful_progress": False,
        "novel_state": False,
        "cycle_risk": False,
        "recommended_action": "UP",
    },
    "engine_progress": {
        "inferred_effect": "level_advance",
        "hypothesis_verdict": "supported",
        "meaningful_progress": True,
        "novel_state": False,
        "cycle_risk": False,
        "recommended_action": None,
    },
    "visual_novelty": {
        "inferred_effect": "visual_change_only",
        "hypothesis_verdict": "contradicted",
        "meaningful_progress": False,
        "novel_state": True,
        "cycle_risk": False,
        "recommended_action": "UP",
    },
    "inverse_cycle": {
        "inferred_effect": "inverse_cycle",
        "hypothesis_verdict": "contradicted",
        "meaningful_progress": False,
        "novel_state": False,
        "cycle_risk": True,
        "recommended_action": "DOWN",
    },
}

GAME_SAFETY_BEHAVIOR = {
    "edge_hud_flicker": {
        "classification": "hud_only",
        "meaningful_progress": False,
        "novel_state": False,
        "action_blocked": False,
        "stop": False,
        "recommended_action": {"action": "UP"},
    },
    "pure_translation": {
        "classification": "object_translation",
        "meaningful_progress": False,
        "novel_state": False,
        "action_blocked": False,
        "stop": False,
        "recommended_action": {"action": "UP"},
    },
    "repeated_click_noop": {
        "classification": "repeated_click_noop",
        "meaningful_progress": False,
        "novel_state": False,
        "action_blocked": True,
        "stop": False,
        "recommended_action": {"action": "MOUSE", "row": 2, "col": 4},
    },
    "directional_saturation": {
        "classification": "directional_saturation",
        "meaningful_progress": False,
        "novel_state": False,
        "action_blocked": True,
        "stop": False,
        "recommended_action": {"action": "RIGHT"},
    },
    "incomplete_observation": {
        "classification": "awaiting_post_action_observation",
        "meaningful_progress": False,
        "novel_state": False,
        "action_blocked": False,
        "stop": False,
        "recommended_action": None,
    },
    "terminal_completion": {
        "classification": "terminal_success",
        "meaningful_progress": True,
        "novel_state": False,
        "action_blocked": False,
        "stop": True,
        "recommended_action": None,
    },
}

RAW_REDUCTION_FIXTURE = (
    "BEGIN_REDUCTION\n{\n"
    '  "objective_id": "level:1:1",\n'
    '  "verdict": "continue",\n'
    '  "evidence": "board unchanged",\n'
    '  "rationale": "continue the active probe",\n'
    '  "selected_index": 0,\n'
    '  "subgoals": []\n}\nEND_REDUCTION'
)
RAW_POLICY_FIXTURE = (
    "BEGIN_POLICY\nPOLICY_API_VERSION = 1\n"
    'SUPPORTED_BACKENDS = ("cpu",)\n'
    "def decide(observation, memory):\n"
    '    return {"status": "continue", "action": '
    '{"action": "ACTION6"}, "memory": memory}\nEND_POLICY'
)
PROBE_ACTIONS_BEHAVIOR = {
    "contrastive": {"probe_actions": ["UP", "RIGHT", "UP", "RIGHT"]},
    "clicks": {
        "mouse_points": [[28, 30], [28, 38], [28, 30]],
        "probe_actions": ["MOUSE", "MOUSE", "MOUSE"],
    },
    "navigation": {
        "actor_values": [1],
        "target_values": [2],
        "passable_values": [0, 1, 2],
        "approach_distance": 1,
        "probe_actions": ["UP", "RIGHT"],
    },
}


class KaggleHardwareProfileTests(TestCase):
    def test_pathfinding_smoke_answers_match_trusted_navigation_solver(self) -> None:
        boards = {
            "corridor": [[1, 0, 0, 2]],
            "detour": [[1, 9, 0, 2], [0, 0, 0, 9]],
            "at_approach": [[1, 2]],
            "unreachable": [[1, 9, 2], [9, 9, 9]],
            "engine_progress_unreachable": [[1, 9, 2], [9, 9, 9]],
        }
        config = {
            "actor_values": [1],
            "target_values": [2],
            "passable_values": [0, 1, 2],
            "approach_distance": 1,
            "probe_actions": ["UP", "RIGHT"],
        }
        deltas = {
            (-1, 0): "UP",
            (0, 1): "RIGHT",
            (1, 0): "DOWN",
            (0, -1): "LEFT",
        }

        for case, small_board in boards.items():
            with self.subTest(case=case):
                board = np.full((64, 64), 9, dtype=np.uint8)
                height = len(small_board)
                width = len(small_board[0])
                board[:height, :width] = small_board
                objective = (
                    {"evidence_mode": "engine_progress"}
                    if case == "engine_progress_unreachable"
                    else {}
                )
                observation = PolicyObservation(
                    board=board,
                    level=1,
                    step=0,
                    valid_actions=("UP", "RIGHT", "DOWN", "LEFT"),
                    last_transition=None,
                    objective=objective,
                    recent_transitions=(),
                    backend="cpu",
                )

                decision = solver_decide("navigation", observation, {}, config)
                path = decision.get("memory", {}).get("path", [])
                path_actions = [
                    deltas[(end[0] - start[0], end[1] - start[1])]
                    for start, end in zip(path, path[1:], strict=False)
                ]
                actual = {
                    "status": decision["status"],
                    "path_actions": path_actions,
                    "action": (decision.get("action") or {}).get("action"),
                }

                self.assertEqual(PATHFINDING_POLICY_BEHAVIOR[case], actual)

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
        self.assertIn("thinking_token_budget: int = 64", command)
        self.assertIn("'thinking_token_budget': thinking_token_budget", command)
        self.assertIn("thinking_token_budget=256", command)
        self.assertNotIn("'tool_choice': 'required'", command)
        self.assertIn("BEGIN_REDUCTION", command)
        self.assertIn("BEGIN_POLICY", command)
        self.assertIn("Raw reduction JSON fidelity: passed", command)
        self.assertIn("Raw policy source fidelity: passed", command)
        self.assertIn("LLM probe_actions contract matrix: passed", command)
        self.assertIn("LLM generated policy behavior: passed", command)
        self.assertIn("LLM pathfinding policy behavior: passed", command)
        self.assertIn("LLM game inference behavior: passed", command)
        self.assertIn("LLM game safety behavior: passed", command)
        self.assertIn("def assert_generated_policy_contract()", command)
        self.assertIn("def assert_pathfinding_policy_behavior()", command)
        self.assertIn("def assert_game_inference_behavior()", command)
        self.assertIn("def assert_game_safety_behavior()", command)
        self.assertIn("import ast", command)
        self.assertIn("one scalar MOUSE probe_actions entry per coordinate", command)
        self.assertIn("instead of treating probe_actions as the route", command)
        self.assertIn("run_vllm_api_smoke_test()", command)
        self.assertIn("'KAGGLE_DUCK_SMOKE_TEST_ONLY': 'false'", command)

    def test_setup_embeds_smoke_test_only_option(self) -> None:
        with mock.patch.dict(
            os.environ, {"KAGGLE_DUCK_SMOKE_TEST_ONLY": "true"}
        ):
            command = duck_kaggle_setup_command()

        self.assertIn("'KAGGLE_DUCK_SMOKE_TEST_ONLY': 'true'", command)

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
        probe_actions_fixture = json.dumps(
            {
                "contrastive": {"probe_actions": ["RIGHT", "UP", "RIGHT", "UP"]},
                "clicks": {
                    "mouse_points": [[28, 30], [28, 38], [28, 30]],
                    "probe_actions": ["MOUSE", "MOUSE", "MOUSE"],
                },
                "navigation": {
                    "actor_values": [1],
                    "target_values": [2],
                    "passable_values": [2, 0, 1],
                    "approach_distance": 1,
                    "probe_actions": ["RIGHT", "UP"],
                    "max_plan_length": 64,
                },
            }
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
                {
                    "choices": [
                        {"message": {"content": probe_actions_fixture}}
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": GENERATED_NAVIGATION_POLICY}}
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(PATHFINDING_POLICY_BEHAVIOR)
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(GAME_INFERENCE_BEHAVIOR)
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": json.dumps(GAME_SAFETY_BEHAVIOR)}}
                    ]
                },
            ]
        )
        namespace = {
            "ast": ast,
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason,
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        namespace["run_vllm_api_smoke_test"]()

        self.assertEqual(7, request_json.call_count)
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
        probe_payload = request_json.call_args_list[2].kwargs["payload"]
        self.assertEqual(768, probe_payload["max_tokens"])
        self.assertEqual(256, probe_payload["thinking_token_budget"])
        self.assertEqual({"type": "json_object"}, probe_payload["response_format"])
        self.assertNotIn("tools", probe_payload)
        self.assertNotIn("tool_choice", probe_payload)
        self.assertIn(
            "minimum_evidence_actions is 4",
            probe_payload["messages"][0]["content"],
        )
        self.assertIn("ordered mouse_points", probe_payload["messages"][0]["content"])
        self.assertIn("not an output field", probe_payload["messages"][0]["content"])
        policy_payload = request_json.call_args_list[3].kwargs["payload"]
        self.assertEqual(1024, policy_payload["max_tokens"])
        self.assertEqual(256, policy_payload["thinking_token_budget"])
        self.assertIn(
            "directly calls solver_decide", policy_payload["messages"][0]["content"]
        )
        pathfinding_payload = request_json.call_args_list[4].kwargs["payload"]
        self.assertEqual(1024, pathfinding_payload["max_tokens"])
        self.assertEqual(256, pathfinding_payload["thinking_token_budget"])
        self.assertEqual(
            {"type": "json_object"}, pathfinding_payload["response_format"]
        )
        self.assertIn(
            "engine_progress_unreachable",
            pathfinding_payload["messages"][0]["content"],
        )
        self.assertIn("must never enter value 9", pathfinding_payload["messages"][0]["content"])
        inference_payload = request_json.call_args_list[5].kwargs["payload"]
        self.assertEqual(1536, inference_payload["max_tokens"])
        self.assertEqual(384, inference_payload["thinking_token_budget"])
        self.assertEqual(
            {"type": "json_object"}, inference_payload["response_format"]
        )
        self.assertIn("visual_novelty", inference_payload["messages"][0]["content"])
        self.assertIn("inverse_cycle", inference_payload["messages"][0]["content"])
        safety_payload = request_json.call_args_list[6].kwargs["payload"]
        self.assertEqual(1792, safety_payload["max_tokens"])
        self.assertEqual(384, safety_payload["thinking_token_budget"])
        self.assertEqual(
            {"type": "json_object"}, safety_payload["response_format"]
        )
        self.assertIn("edge_hud_flicker", safety_payload["messages"][0]["content"])
        self.assertIn(
            "incomplete_observation", safety_payload["messages"][0]["content"]
        )

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
            "ast": ast,
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

    def test_bounded_reasoning_smoke_repairs_abbreviated_navigation_keys(self) -> None:
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
        reduction = (
            "BEGIN_REDUCTION\n{\n"
            '  "objective_id": "level:1:1",\n'
            '  "verdict": "continue",\n'
            '  "evidence": "board unchanged",\n'
            '  "rationale": "continue the active probe",\n'
            '  "selected_index": 0,\n'
            '  "subgoals": []\n}\nEND_REDUCTION'
        )
        policy = (
            "BEGIN_POLICY\nPOLICY_API_VERSION = 1\n"
            'SUPPORTED_BACKENDS = ("cpu",)\n'
            "def decide(observation, memory):\n"
            '    return {"status": "continue", "action": '
            '{"action": "ACTION6"}, "memory": memory}\nEND_POLICY'
        )
        bad_result = {
            "contrastive": {"probe_actions": ["UP", "RIGHT", "UP", "RIGHT"]},
            "clicks": {
                "mouse_points": [[28, 30], [28, 38], [28, 30]],
                "probe_actions": ["MOUSE", "MOUSE", "MOUSE"],
            },
            "navigation": {
                "actor": 1,
                "target": 2,
                "passable": [0, 1, 2],
                "approach_distance": 1,
                "probe_actions": ["UP", "RIGHT"],
            },
        }
        repaired_result = {
            **bad_result,
            "clicks": {
                "mouse_points": [[28, 30], [28, 38], [28, 30]],
                "probe_actions": [
                    {"x": 28, "y": 30},
                    {"x": 28, "y": 38},
                    {"x": 28, "y": 30},
                ],
            },
            "navigation": {
                "actor_values": [1],
                "target_values": [2],
                "passable_values": [0, 1, 2],
                "approach_distance": 1,
                "probe_actions": ["UP", "RIGHT"],
            },
        }
        request_json = mock.Mock(
            side_effect=[
                {"choices": [{"message": {"content": reduction}}]},
                {"choices": [{"message": {"content": policy}}]},
                {"choices": [{"message": {"content": json.dumps(bad_result)}}]},
                {
                    "choices": [
                        {"message": {"content": json.dumps(repaired_result)}}
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": GENERATED_NAVIGATION_POLICY}}
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(PATHFINDING_POLICY_BEHAVIOR)
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(GAME_INFERENCE_BEHAVIOR)
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": json.dumps(GAME_SAFETY_BEHAVIOR)}}
                    ]
                },
            ]
        )
        namespace = {
            "ast": ast,
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason,
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        namespace["run_vllm_api_smoke_test"]()

        self.assertEqual(8, request_json.call_count)
        repair_payload = request_json.call_args_list[3].kwargs["payload"]
        self.assertIn(
            "exact registered field names", repair_payload["messages"][0]["content"]
        )
        self.assertIn(
            "only these failed top-level cases: navigation",
            repair_payload["messages"][0]["content"],
        )
        self.assertIn(
            "mouse coordinates belong only in mouse_points",
            repair_payload["messages"][0]["content"],
        )
        self.assertIn('"actor": 1', repair_payload["messages"][0]["content"])

    def test_bounded_reasoning_smoke_repairs_direct_action_policy(self) -> None:
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
        reduction = (
            "BEGIN_REDUCTION\n{\n"
            '  "objective_id": "level:1:1",\n'
            '  "verdict": "continue",\n'
            '  "evidence": "board unchanged",\n'
            '  "rationale": "continue the active probe",\n'
            '  "selected_index": 0,\n'
            '  "subgoals": []\n}\nEND_REDUCTION'
        )
        raw_policy = (
            "BEGIN_POLICY\nPOLICY_API_VERSION = 1\n"
            'SUPPORTED_BACKENDS = ("cpu",)\n'
            "def decide(observation, memory):\n"
            '    return {"status": "continue", "action": '
            '{"action": "ACTION6"}, "memory": memory}\nEND_POLICY'
        )
        probe_actions = json.dumps(
            {
                "contrastive": {
                    "probe_actions": ["UP", "RIGHT", "UP", "RIGHT"]
                },
                "clicks": {
                    "mouse_points": [[28, 30], [28, 38], [28, 30]],
                    "probe_actions": ["MOUSE", "MOUSE", "MOUSE"],
                },
                "navigation": {
                    "actor_values": [1],
                    "target_values": [2],
                    "passable_values": [2, 1, 0],
                    "approach_distance": 1,
                    "probe_actions": ["UP", "RIGHT"],
                },
            }
        )
        direct_action_policy = GENERATED_NAVIGATION_POLICY.replace(
            "    return solver_decide(\n"
            "        POLICY_SOLVER_TYPE, observation, memory, POLICY_SOLVER_CONFIG\n"
            "    )",
            '    return {"status": "continue", "action": {"action": "UP"}}',
        )
        request_json = mock.Mock(
            side_effect=[
                {"choices": [{"message": {"content": reduction}}]},
                {"choices": [{"message": {"content": raw_policy}}]},
                {"choices": [{"message": {"content": probe_actions}}]},
                {"choices": [{"message": {"content": direct_action_policy}}]},
                {
                    "choices": [
                        {"message": {"content": GENERATED_NAVIGATION_POLICY}}
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(PATHFINDING_POLICY_BEHAVIOR)
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(GAME_INFERENCE_BEHAVIOR)
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": json.dumps(GAME_SAFETY_BEHAVIOR)}}
                    ]
                },
            ]
        )
        namespace = {
            "ast": ast,
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason,
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        namespace["run_vllm_api_smoke_test"]()

        self.assertEqual(8, request_json.call_count)
        repair_prompt = request_json.call_args_list[4].kwargs["payload"]["messages"][
            0
        ]["content"]
        self.assertIn("canonical dispatcher call", repair_prompt)
        self.assertIn("Return one complete replacement module", repair_prompt)
        self.assertIn(direct_action_policy, repair_prompt)

    def test_bounded_reasoning_smoke_repairs_pathfinding_behavior(self) -> None:
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
        recovered_pathfinding = {
            case: PATHFINDING_POLICY_BEHAVIOR[case]
            for case in ("corridor", "detour")
        }
        malformed_pathfinding = (
            '{"corridor":{"status":"continue","path_actions":[\n'
            "I have to give the solution based on the reasoning directly now.</think>\n"
            + json.dumps(recovered_pathfinding)
        )
        request_json = mock.Mock(
            side_effect=[
                {"choices": [{"message": {"content": RAW_REDUCTION_FIXTURE}}]},
                {"choices": [{"message": {"content": RAW_POLICY_FIXTURE}}]},
                {
                    "choices": [
                        {"message": {"content": json.dumps(PROBE_ACTIONS_BEHAVIOR)}}
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": GENERATED_NAVIGATION_POLICY}}
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": malformed_pathfinding}}
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "at_approach": PATHFINDING_POLICY_BEHAVIOR[
                                            "at_approach"
                                        ],
                                        "unreachable": PATHFINDING_POLICY_BEHAVIOR[
                                            "unreachable"
                                        ],
                                    }
                                )
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "engine_progress_unreachable": (
                                            PATHFINDING_POLICY_BEHAVIOR[
                                                "engine_progress_unreachable"
                                            ]
                                        )
                                    }
                                )
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(GAME_INFERENCE_BEHAVIOR)
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": json.dumps(GAME_SAFETY_BEHAVIOR)}}
                    ]
                },
            ]
        )
        namespace = {
            "ast": ast,
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason,
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        namespace["run_vllm_api_smoke_test"]()

        self.assertEqual(9, request_json.call_count)
        repair_prompt = request_json.call_args_list[5].kwargs["payload"]["messages"][
            0
        ]["content"]
        self.assertIn(
            "only these failed top-level cases: at_approach, unreachable, "
            "engine_progress_unreachable",
            repair_prompt,
        )
        self.assertIn("Do not return or modify passing cases", repair_prompt)
        self.assertIn("trusted solver oracle", repair_prompt)
        self.assertIn('"action": "UP"', repair_prompt)
        self.assertIn("</think>", repair_prompt)
        self.assertEqual(
            {"type": "json_object"},
            request_json.call_args_list[5].kwargs["payload"]["response_format"],
        )
        second_repair_prompt = request_json.call_args_list[6].kwargs["payload"][
            "messages"
        ][0]["content"]
        self.assertIn(
            "only these failed top-level cases: engine_progress_unreachable",
            second_repair_prompt,
        )

    def test_bounded_reasoning_smoke_repairs_game_inference_behavior(self) -> None:
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
        bad_inference = {
            **GAME_INFERENCE_BEHAVIOR,
            "visual_novelty": {
                "inferred_effect": "visual_change_only",
                "hypothesis_verdict": "supported",
                "meaningful_progress": True,
                "novel_state": True,
                "cycle_risk": False,
                "recommended_action": "MOUSE",
            },
        }
        request_json = mock.Mock(
            side_effect=[
                {"choices": [{"message": {"content": RAW_REDUCTION_FIXTURE}}]},
                {"choices": [{"message": {"content": RAW_POLICY_FIXTURE}}]},
                {
                    "choices": [
                        {"message": {"content": json.dumps(PROBE_ACTIONS_BEHAVIOR)}}
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": GENERATED_NAVIGATION_POLICY}}
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(PATHFINDING_POLICY_BEHAVIOR)
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"content": json.dumps(bad_inference)}}]},
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "visual_novelty": GAME_INFERENCE_BEHAVIOR[
                                            "visual_novelty"
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": json.dumps(GAME_SAFETY_BEHAVIOR)}}
                    ]
                },
            ]
        )
        namespace = {
            "ast": ast,
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason,
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        namespace["run_vllm_api_smoke_test"]()

        self.assertEqual(8, request_json.call_count)
        repair_prompt = request_json.call_args_list[6].kwargs["payload"]["messages"][
            0
        ]["content"]
        self.assertIn("only these failed top-level cases: visual_novelty", repair_prompt)
        self.assertIn("trusted inference oracle", repair_prompt)
        self.assertIn('"meaningful_progress": false', repair_prompt)
        self.assertIn('"recommended_action": "UP"', repair_prompt)

    def test_bounded_reasoning_smoke_repairs_game_safety_behavior(self) -> None:
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
        bad_safety = {
            **GAME_SAFETY_BEHAVIOR,
            "pure_translation": {
                "classification": "object_translation",
                "meaningful_progress": False,
                "novel_state": True,
                "action_blocked": False,
                "stop": False,
                "recommended_action": {"action": "RIGHT"},
            },
        }
        request_json = mock.Mock(
            side_effect=[
                {"choices": [{"message": {"content": RAW_REDUCTION_FIXTURE}}]},
                {"choices": [{"message": {"content": RAW_POLICY_FIXTURE}}]},
                {
                    "choices": [
                        {"message": {"content": json.dumps(PROBE_ACTIONS_BEHAVIOR)}}
                    ]
                },
                {
                    "choices": [
                        {"message": {"content": GENERATED_NAVIGATION_POLICY}}
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(PATHFINDING_POLICY_BEHAVIOR)
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(GAME_INFERENCE_BEHAVIOR)
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"content": json.dumps(bad_safety)}}]},
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "pure_translation": GAME_SAFETY_BEHAVIOR[
                                            "pure_translation"
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                },
            ]
        )
        namespace = {
            "ast": ast,
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason,
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        namespace["run_vllm_api_smoke_test"]()

        self.assertEqual(8, request_json.call_count)
        repair_payload = request_json.call_args_list[7].kwargs["payload"]
        repair_prompt = repair_payload["messages"][0]["content"]
        self.assertIn("only these failed top-level cases: pure_translation", repair_prompt)
        self.assertIn("trusted safety oracle", repair_prompt)
        self.assertIn('"novel_state": false', repair_prompt)
        self.assertIn('"action": "UP"', repair_prompt)
        self.assertEqual({"type": "json_object"}, repair_payload["response_format"])

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
            "ast": ast,
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

    def test_bounded_reasoning_smoke_fails_on_probe_actions_contract(self) -> None:
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
                                    "BEGIN_REDUCTION\n{\n"
                                    '  "objective_id": "level:1:1",\n'
                                    '  "verdict": "continue",\n'
                                    '  "evidence": "board unchanged",\n'
                                    '  "rationale": "continue the active probe",\n'
                                    '  "selected_index": 0,\n'
                                    '  "subgoals": []\n}\nEND_REDUCTION'
                                )
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "BEGIN_POLICY\nPOLICY_API_VERSION = 1\n"
                                    'SUPPORTED_BACKENDS = ("cpu",)\n'
                                    "def decide(observation, memory):\n"
                                    '    return {"status": "continue", "action": '
                                    '{"action": "ACTION6"}, "memory": memory}\n'
                                    "END_POLICY"
                                )
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"content": "{}"}}]},
                {"choices": [{"message": {"content": "{}"}}]},
            ]
        )
        namespace = {
            "ast": ast,
            "json": json,
            "VLLM_BASE_URL": "http://127.0.0.1:1234/v1",
            "SERVED_MODEL_NAME": "unit-test-model",
            "request_json": request_json,
            "server_failure_message": lambda reason: reason + "\ncomplete server log",
        }
        exec(compile(functions, "<kaggle-vllm-smoke-test>", "exec"), namespace)

        with self.assertRaisesRegex(
            RuntimeError, "probe-actions contract check"
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
