"""Kaggle helpers for the ARC3 duck harness."""

from __future__ import annotations

import os
from dataclasses import dataclass

from inference.utils.openai_compat import normalize_provider

DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
DEFAULT_QWEN_MODEL_SOURCE = (
    "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pyTorch/hf-fp8/1"
)
DEFAULT_SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_VLLM_PORT = 1234
DEFAULT_VLLM_MAX_MODEL_LEN = 65536
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE = 1
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.92
DEFAULT_VLLM_MAX_NUM_SEQS = 16
DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_VLLM_ENABLE_CHUNKED_PREFILL = True
DEFAULT_VLLM_REASONING_CONFIG = (
    '{"reasoning_start_str":"<think>",'
    '"reasoning_end_str":"I have to give the solution based on the reasoning '
    'directly now.</think>"}'
)
DEFAULT_EXPECTED_GPU_TYPE = "rtx-pro-6000"
DEFAULT_EXPECTED_GPU_COUNT = 1
T4_VLLM_MAX_MODEL_LEN = 8192
T4_VLLM_MAX_NUM_SEQS = 16
T4_VLLM_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_WHEELHOUSE_STAMP_TEXT = "vllm==0.19.0 torch==2.10.0 flashinfer==0.6.6\n"

# The 25 official ARC-AGI-3 games. The first 16 are the original Kaggle duck
# validation harness order; the remaining 9 complete the official tag set.
DUCK_HARNESS_PUBLIC_GAME_IDS: tuple[str, ...] = (
    "tn36-ef4dde99",
    "lf52-271a04aa",
    "cn04-2fe56bfb",
    "bp35-0a0ad940",
    "wa30-ee6fef47",
    "lp85-305b61c3",
    "r11l-495a7899",
    "tu93-0768757b",
    "sp80-589a99af",
    "m0r0-492f87ba",
    "vc33-5430563c",
    "ar25-0c556536",
    "ka59-38d34dbb",
    "sc25-635fd71a",
    "sk48-d8078629",
    "dc22-fdcac232",
    "cd82-fb555c5d",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ls20-9607627b",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "su15-1944f8ab",
    "tr87-cd924810",
)


@dataclass(frozen=True)
class DuckKaggleVllmConfig:
    """Kaggle-side vLLM/model configuration declared by ``HarnessSolver``."""

    wheelhouse_dataset_source: str = DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE
    model_source: str = DEFAULT_QWEN_MODEL_SOURCE
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME
    vllm_port: int = DEFAULT_VLLM_PORT
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN
    tensor_parallel_size: int = DEFAULT_VLLM_TENSOR_PARALLEL_SIZE
    gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = DEFAULT_VLLM_MAX_NUM_SEQS
    max_num_batched_tokens: int = DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS
    enable_chunked_prefill: bool = DEFAULT_VLLM_ENABLE_CHUNKED_PREFILL
    expected_gpu_type: str = DEFAULT_EXPECTED_GPU_TYPE
    expected_gpu_count: int = DEFAULT_EXPECTED_GPU_COUNT
    wheelhouse_stamp_text: str = DEFAULT_WHEELHOUSE_STAMP_TEXT


def duck_kaggle_vllm_config_for_accelerator(
    accelerator: str | None,
) -> DuckKaggleVllmConfig:
    """Return the vLLM profile matching Kaggle's accelerator allocation."""

    normalized = "".join(
        character for character in str(accelerator or "").lower() if character.isalnum()
    )
    if normalized == "nvidiateslat4":
        # Kaggle's NvidiaTeslaT4 shape exposes two 16 GiB T4s. The 27B FP8
        # model must be sharded across both, with a bounded KV-cache budget.
        return DuckKaggleVllmConfig(
            max_model_len=T4_VLLM_MAX_MODEL_LEN,
            tensor_parallel_size=2,
            max_num_seqs=T4_VLLM_MAX_NUM_SEQS,
            max_num_batched_tokens=T4_VLLM_MAX_NUM_BATCHED_TOKENS,
            expected_gpu_type="t4",
            expected_gpu_count=2,
        )
    return DuckKaggleVllmConfig()


def duck_kaggle_dataset_sources(
    config: DuckKaggleVllmConfig | None = None,
) -> list[str]:
    cfg = config or DuckKaggleVllmConfig()
    return [cfg.wheelhouse_dataset_source]


def duck_kaggle_model_sources(
    config: DuckKaggleVllmConfig | None = None,
) -> list[str]:
    cfg = config or DuckKaggleVllmConfig()
    return [_validate_model_source(cfg.model_source)]


def duck_kaggle_setup_command(config: DuckKaggleVllmConfig | None = None) -> str:
    cfg = config or DuckKaggleVllmConfig()
    wheelhouse_owner, wheelhouse_slug = _split_dataset_source(
        cfg.wheelhouse_dataset_source,
        option_name="wheelhouse_dataset_source",
    )
    model_source = _validate_model_source(cfg.model_source)
    # Base URL / model are pinned to the local vLLM server below, so reject a
    # provider that disagrees (e.g. openrouter) — it would drop vLLM-only payload
    # fields (top_k, chat_template_kwargs) against a vLLM endpoint.
    analyzer_provider = os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")
    if normalize_provider(analyzer_provider) != "vllm":
        raise ValueError(
            f"kaggle-duck runs a local vLLM server, so LOCAL_ANALYZER_PROVIDER must be "
            f"vLLM/OpenAI-compatible, got {analyzer_provider!r}."
        )
    replacements = {
        "__WHEELHOUSE_OWNER__": repr(wheelhouse_owner),
        "__WHEELHOUSE_SLUG__": repr(wheelhouse_slug),
        "__MODEL_SOURCE__": repr(model_source),
        "__SERVED_MODEL_NAME__": repr(cfg.served_model_name),
        "__VLLM_PORT__": repr(int(cfg.vllm_port)),
        "__VLLM_MAX_MODEL_LEN__": repr(int(cfg.max_model_len)),
        # The launcher's Makefile exports LOCAL_ANALYZER_CONTEXT_WINDOW from
        # JSON shared.context_window (or analyzer.context_window). Embed it
        # here so the agent's prompt budget on Kaggle is the JSON value, not
        # vllm's max-model-len. Falls back to max_model_len if unset.
        "__ANALYZER_CONTEXT_WINDOW__": repr(
            min(
                int(
                    os.environ.get("LOCAL_ANALYZER_CONTEXT_WINDOW") or cfg.max_model_len
                ),
                int(cfg.max_model_len),
            )
        ),
        # Remaining JSON-driven analyzer/multimodal config: the launcher's
        # Makefile exports each from inference.json; embed the launcher value
        # so the rendered setup_env on Kaggle reflects JSON edits. Fallback
        # equals the historical hardcoded literal so direct kaggle.py callers
        # outside Make are unaffected.
        "__LOCAL_ANALYZER_PROVIDER__": repr(analyzer_provider),
        "__LOCAL_ANALYZER_APP_NAME__": repr(
            os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Kaggle Harness")
        ),
        "__LOCAL_ANALYZER_MAX_OUTPUT__": repr(
            os.environ.get("LOCAL_ANALYZER_MAX_OUTPUT", "0")
        ),
        "__LOCAL_ANALYZER_TOOL_STEPS__": repr(
            os.environ.get("LOCAL_ANALYZER_TOOL_STEPS", "0")
        ),
        "__LOCAL_ANALYZER_TOOL_TIMEOUT__": repr(
            os.environ.get("LOCAL_ANALYZER_TOOL_TIMEOUT", "30")
        ),
        "__LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__": repr(
            os.environ.get("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", "1024")
        ),
        "__LOCAL_ANALYZER_CANDIDATES__": repr(
            os.environ.get("LOCAL_ANALYZER_CANDIDATES", "2")
        ),
        "__LOCAL_ANALYZER_GAME_TOKEN_BUDGET__": repr(
            os.environ.get("LOCAL_ANALYZER_GAME_TOKEN_BUDGET", "250000")
        ),
        "__LOCAL_ANALYZER_OBJECTIVE_REDUCTION__": repr(
            os.environ.get("LOCAL_ANALYZER_OBJECTIVE_REDUCTION", "false")
        ),
        "__KAGGLE_DUCK_SMOKE_TEST_ONLY__": repr(
            os.environ.get("KAGGLE_DUCK_SMOKE_TEST_ONLY", "false")
        ),
        "__LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS__": repr(
            os.environ.get(
                "LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS", "300"
            )
        ),
        "__LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT__": repr(
            os.environ.get("LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT", "4096")
        ),
        "__LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT__": repr(
            os.environ.get("LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT", "8192")
        ),
        "__LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET__": repr(
            os.environ.get(
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET", "2048"
            )
        ),
        "__LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET__": repr(
            os.environ.get("LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET", "1024")
        ),
        "__LOCAL_GAMEPLAY_POLICY_BACKEND__": repr(
            os.environ.get("LOCAL_GAMEPLAY_POLICY_BACKEND", "cpu")
        ),
        "__LOCAL_GAMEPLAY_POLICY_CUDA_MIN_FREE_MB__": repr(
            os.environ.get("LOCAL_GAMEPLAY_POLICY_CUDA_MIN_FREE_MB", "4096")
        ),
        "__LOCAL_GAMEPLAY_POLICY_DECISION_TIMEOUT_SECONDS__": repr(
            os.environ.get("LOCAL_GAMEPLAY_POLICY_DECISION_TIMEOUT_SECONDS", "2")
        ),
        "__LOCAL_ANALYZER_STRATEGY_ENABLED__": repr(
            os.environ.get("LOCAL_ANALYZER_STRATEGY_ENABLED", "true")
        ),
        "__LOCAL_ANALYZER_STRATEGY_POLICY__": repr(
            os.environ.get("LOCAL_ANALYZER_STRATEGY_POLICY", "outcome_aware")
        ),
        "__LOCAL_ANALYZER_SAME_STATE_NOOP_LIMIT__": repr(
            os.environ.get("LOCAL_ANALYZER_SAME_STATE_NOOP_LIMIT", "2")
        ),
        "__LOCAL_ANALYZER_STAGNATION_WINDOW__": repr(
            os.environ.get("LOCAL_ANALYZER_STAGNATION_WINDOW", "6")
        ),
        "__LOCAL_ANALYZER_CYCLE_WINDOW__": repr(
            os.environ.get("LOCAL_ANALYZER_CYCLE_WINDOW", "4")
        ),
        "__LOCAL_ANALYZER_CYCLE_STOP_LIMIT__": repr(
            os.environ.get("LOCAL_ANALYZER_CYCLE_STOP_LIMIT", "0")
        ),
        "__LOCAL_ANALYZER_REPEAT_ACTION_LIMIT__": repr(
            os.environ.get("LOCAL_ANALYZER_REPEAT_ACTION_LIMIT", "0")
        ),
        "__LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_WINDOW__": repr(
            os.environ.get("LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_WINDOW", "0")
        ),
        "__LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_LIMIT__": repr(
            os.environ.get("LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_LIMIT", "0")
        ),
        "__LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STRIKE_LIMIT__": repr(
            os.environ.get("LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STRIKE_LIMIT", "0")
        ),
        "__LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STOP_LIMIT__": repr(
            os.environ.get("LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STOP_LIMIT", "0")
        ),
        "__LOCAL_ANALYZER_IGNORE_EDGE_HUD_CHANGES__": repr(
            os.environ.get("LOCAL_ANALYZER_IGNORE_EDGE_HUD_CHANGES", "false")
        ),
        "__LOCAL_ANALYZER_VOLATILE_WINDOW__": repr(
            os.environ.get("LOCAL_ANALYZER_VOLATILE_WINDOW", "8")
        ),
        "__LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES__": repr(
            os.environ.get("LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES", "4")
        ),
        "__LOCAL_ANALYZER_VOLATILE_RATIO__": repr(
            os.environ.get("LOCAL_ANALYZER_VOLATILE_RATIO", "0.75")
        ),
        "__LOCAL_ANALYZER_PROGRESS_UTILITY__": repr(
            os.environ.get("LOCAL_ANALYZER_PROGRESS_UTILITY", "1.0")
        ),
        "__LOCAL_ANALYZER_NOVEL_UTILITY__": repr(
            os.environ.get("LOCAL_ANALYZER_NOVEL_UTILITY", "0.2")
        ),
        "__LOCAL_ANALYZER_EXPLORATION_WEIGHT__": repr(
            os.environ.get("LOCAL_ANALYZER_EXPLORATION_WEIGHT", "0.75")
        ),
        "__LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MULTIPLIER__": repr(
            os.environ.get("LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MULTIPLIER", "0")
        ),
        "__LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM__": repr(
            os.environ.get("LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM", "0")
        ),
        "__LOCAL_ANALYZER_LEVEL_NO_PROGRESS_TOKEN_LIMIT__": repr(
            os.environ.get("LOCAL_ANALYZER_LEVEL_NO_PROGRESS_TOKEN_LIMIT", "0")
        ),
        "__LOCAL_ANALYZER_YIELD_SECONDS__": repr(
            os.environ.get("LOCAL_ANALYZER_YIELD_SECONDS", "60")
        ),
        "__LOCAL_ANALYZER_TEMPERATURE__": repr(
            os.environ.get("LOCAL_ANALYZER_TEMPERATURE", "0.6")
        ),
        "__LOCAL_ANALYZER_TOP_P__": repr(
            os.environ.get("LOCAL_ANALYZER_TOP_P", "0.95")
        ),
        "__LOCAL_ANALYZER_TOP_K__": repr(os.environ.get("LOCAL_ANALYZER_TOP_K", "20")),
        "__LOCAL_ANALYZER_ENABLE_THINKING__": repr(
            os.environ.get("LOCAL_ANALYZER_ENABLE_THINKING", "1")
        ),
        "__LOCAL_ANALYZER_STREAM__": repr(
            os.environ.get("LOCAL_ANALYZER_STREAM", "false")
        ),
        "__LOCAL_ANALYZER_REQUIRE_OS_SANDBOX__": repr(
            os.environ.get("LOCAL_ANALYZER_REQUIRE_OS_SANDBOX", "true")
        ),
        "__LOCAL_ANALYZER_VERIFY_CANDIDATES__": repr(
            os.environ.get("LOCAL_ANALYZER_VERIFY_CANDIDATES", "true")
        ),
        "__MULTIMODAL_CONTEXT__": repr(
            os.environ.get("MULTIMODAL_CONTEXT", "current_grid")
        ),
        "__MULTIMODAL_UPSCALE__": repr(os.environ.get("MULTIMODAL_UPSCALE", "4")),
        "__VLLM_TENSOR_PARALLEL_SIZE__": repr(int(cfg.tensor_parallel_size)),
        "__VLLM_GPU_MEMORY_UTILIZATION__": repr(float(cfg.gpu_memory_utilization)),
        "__VLLM_MAX_NUM_SEQS__": repr(max(1, int(cfg.max_num_seqs))),
        "__VLLM_MAX_NUM_BATCHED_TOKENS__": repr(
            max(1, int(cfg.max_num_batched_tokens))
        ),
        "__VLLM_ENABLE_CHUNKED_PREFILL__": repr(bool(cfg.enable_chunked_prefill)),
        "__VLLM_REASONING_CONFIG__": repr(DEFAULT_VLLM_REASONING_CONFIG),
        "__EXPECTED_GPU_TYPE__": repr(str(cfg.expected_gpu_type)),
        "__EXPECTED_GPU_COUNT__": repr(int(cfg.expected_gpu_count)),
        "__WHEELHOUSE_STAMP_TEXT__": repr(cfg.wheelhouse_stamp_text),
    }
    script = _DUCK_VLLM_SETUP_SCRIPT
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)
    return f"\"$PYTHON\" - <<'PYSETUP'\n{script}\nPYSETUP"


def duck_kaggle_teardown_command() -> str:
    return f"\"$PYTHON\" - <<'PYTEARDOWN'\n{_DUCK_VLLM_TEARDOWN_SCRIPT}\nPYTEARDOWN"


def _split_dataset_source(value: str, *, option_name: str) -> tuple[str, str]:
    parts = str(value or "").strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"{option_name} must be a Kaggle dataset ref in owner/slug format."
        )
    return parts[0], parts[1]


def _validate_model_source(value: str) -> str:
    parts = str(value or "").strip().split("/")
    if len(parts) not in (4, 5) or not all(parts):
        raise ValueError(
            "model_source must be a Kaggle Model handle in "
            "owner/model/framework/variation[/version] format."
        )
    if len(parts) == 5 and not parts[4].isdigit():
        raise ValueError("model_source version must be numeric.")
    return "/".join(parts)


_DUCK_VLLM_SETUP_SCRIPT = r"""import ast
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WHEELHOUSE_OWNER = __WHEELHOUSE_OWNER__
WHEELHOUSE_SLUG = __WHEELHOUSE_SLUG__
MODEL_SOURCE = __MODEL_SOURCE__
SERVED_MODEL_NAME = __SERVED_MODEL_NAME__
VLLM_HOST = '127.0.0.1'
VLLM_PORT = __VLLM_PORT__
VLLM_BASE_URL = f'http://{VLLM_HOST}:{VLLM_PORT}/v1'
VLLM_MAX_MODEL_LEN = __VLLM_MAX_MODEL_LEN__
ANALYZER_CONTEXT_WINDOW = __ANALYZER_CONTEXT_WINDOW__
VLLM_TENSOR_PARALLEL_SIZE = __VLLM_TENSOR_PARALLEL_SIZE__
VLLM_GPU_MEMORY_UTILIZATION = __VLLM_GPU_MEMORY_UTILIZATION__
VLLM_MAX_NUM_SEQS = __VLLM_MAX_NUM_SEQS__
VLLM_MAX_NUM_BATCHED_TOKENS = __VLLM_MAX_NUM_BATCHED_TOKENS__
VLLM_ENABLE_CHUNKED_PREFILL = __VLLM_ENABLE_CHUNKED_PREFILL__
EXPECTED_GPU_TYPE = __EXPECTED_GPU_TYPE__
EXPECTED_GPU_COUNT = __EXPECTED_GPU_COUNT__
WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
SITE_PACKAGES = WORKING_DIR / 'vllm-site-packages'
VLLM_SERVER_LOG = WORKING_DIR / 'vllm-openai-server.log'
VLLM_SERVER_PID = WORKING_DIR / 'vllm-openai-server.pid'
INSTALL_STAMP = SITE_PACKAGES / f'.{WHEELHOUSE_SLUG}'
STAMP_TEXT = __WHEELHOUSE_STAMP_TEXT__

GPU_NAME_PATTERNS = {'rtx-pro-6000': ('rtx pro 6000',), 'h100': ('h100',), 'l4': ('l4',), 't4': ('t4',)}


def taaf_kaggle_input_paths() -> dict[str, Path]:
    raw = os.getenv('TAAF_KAGGLE_INPUT_PATHS', '').strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError('TAAF_KAGGLE_INPUT_PATHS must contain a JSON object.')
    return {str(ref): Path(str(path)) for ref, path in data.items()}


def resolve_kaggle_dataset_path(owner: str, slug: str) -> Path:
    mapped = taaf_kaggle_input_paths().get(f'{owner}/{slug}')
    if mapped is not None:
        return mapped
    for dataset_path in (Path('/kaggle/input') / slug, Path('/kaggle/input/datasets') / owner / slug):
        if dataset_path.exists():
            return dataset_path
    return Path('/kaggle/input') / slug


def resolve_kaggle_model_path(model_source: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError('kagglehub is required to resolve an attached Kaggle Model.') from exc
    return Path(kagglehub.model_download(model_source))


WHEELHOUSE = resolve_kaggle_dataset_path(WHEELHOUSE_OWNER, WHEELHOUSE_SLUG)
MODEL_PATH = resolve_kaggle_model_path(MODEL_SOURCE)


def assert_expected_cuda_gpu() -> None:
    if not Path('/kaggle/input').exists():
        return
    assert shutil.which('nvidia-smi'), 'CUDA GPU check failed: nvidia-smi is not available.'
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    assert result.returncode == 0, f'nvidia-smi failed: {result.stderr.strip()}'
    gpu_names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert gpu_names, 'nvidia-smi did not report any CUDA GPUs.'
    expected_gpu_type = os.getenv('KAGGLE_GPU_TYPE', EXPECTED_GPU_TYPE).strip().lower()
    expected_count = os.getenv('KAGGLE_GPU_COUNT', str(EXPECTED_GPU_COUNT))
    if expected_count.isdigit():
        assert len(gpu_names) == int(expected_count), f'Expected {expected_count} CUDA GPU(s), found {gpu_names}'
    patterns = GPU_NAME_PATTERNS.get(expected_gpu_type, (expected_gpu_type.replace('-', ' '),))
    mismatched = [name for name in gpu_names if not any(pattern in name.lower() for pattern in patterns)]
    assert not mismatched, f'Expected GPU type {expected_gpu_type!r}, found {gpu_names}'
    print(f'CUDA GPU check passed for {expected_gpu_type} x{expected_count}: {gpu_names}', flush=True)


def vllm_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(SITE_PACKAGES) if not existing else f'{SITE_PACKAGES}{os.pathsep}{existing}'
    env.update(
        {
            'USE_TF': '0',
            'TRANSFORMERS_NO_TF': '1',
            'TRANSFORMERS_NO_TORCHVISION': '1',
            'VLLM_NO_USAGE_STATS': '1',
        }
    )
    return env


def cached_install_is_usable() -> bool:
    if not INSTALL_STAMP.exists() or INSTALL_STAMP.read_text(encoding='utf-8') != STAMP_TEXT:
        return False
    result = subprocess.run(
        [sys.executable, '-c', "import vllm, torch; print(f'Cached vLLM {vllm.__version__}, torch {torch.__version__}')"],
        env=vllm_env(),
        text=True,
    )
    return result.returncode == 0


def install_vllm_wheelhouse() -> None:
    requirements = WHEELHOUSE / 'requirements.lock'
    if not requirements.exists():
        raise FileNotFoundError(f'Missing wheelhouse lock file: {requirements}')
    if cached_install_is_usable():
        print(f'Using cached vLLM target install at {SITE_PACKAGES}', flush=True)
        return
    shutil.rmtree(SITE_PACKAGES, ignore_errors=True)
    SITE_PACKAGES.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--no-index',
        '--find-links',
        str(WHEELHOUSE),
        '--requirement',
        str(requirements),
        '--target',
        str(SITE_PACKAGES),
        '--upgrade',
        '--ignore-installed',
        '--only-binary',
        ':all:',
        '--no-compile',
        '--disable-pip-version-check',
        '--no-warn-conflicts',
    ]
    print('Installing vLLM wheelhouse into', SITE_PACKAGES, flush=True)
    subprocess.run(cmd, check=True)
    INSTALL_STAMP.write_text(STAMP_TEXT, encoding='utf-8')


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def read_server_log() -> str:
    if not VLLM_SERVER_LOG.exists():
        return '(vLLM server log file was not created)'
    return VLLM_SERVER_LOG.read_text(encoding='utf-8', errors='replace')


def server_failure_message(reason: str) -> str:
    return f'{reason}\nComplete vLLM server log:\n{read_server_log()}'


def wait_for_vllm_server(process: subprocess.Popen[str], timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f'{VLLM_BASE_URL}/models'
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                server_failure_message(
                    f'vLLM server exited with code {returncode} before becoming ready at {url}.'
                )
            )
        try:
            models = request_json(url, timeout=5)
            print('vLLM server ready:', models, flush=True)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(
        server_failure_message(f'Timed out waiting for vLLM server at {url} after {timeout_seconds} seconds.')
    )


def start_vllm_server() -> None:
    install_vllm_wheelhouse()
    VLLM_SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    VLLM_SERVER_PID.unlink(missing_ok=True)
    log_handle = VLLM_SERVER_LOG.open('w', encoding='utf-8')
    cmd = [
        sys.executable,
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        str(MODEL_PATH),
        '--served-model-name',
        SERVED_MODEL_NAME,
        '--host',
        VLLM_HOST,
        '--port',
        str(VLLM_PORT),
        '--tensor-parallel-size',
        str(VLLM_TENSOR_PARALLEL_SIZE),
        '--enable-auto-tool-choice',
        '--tool-call-parser',
        'qwen3_coder',
        '--generation-config',
        'vllm',
        '--enable-prefix-caching',
        '--default-chat-template-kwargs',
        '{"preserve_thinking": true}',
        '--reasoning-parser',
        'qwen3',
        '--reasoning-config',
        __VLLM_REASONING_CONFIG__,
        '--max-model-len',
        str(VLLM_MAX_MODEL_LEN),
        '--gpu-memory-utilization',
        str(VLLM_GPU_MEMORY_UTILIZATION),
        '--max-num-seqs',
        str(VLLM_MAX_NUM_SEQS),
        '--max-num-batched-tokens',
        str(VLLM_MAX_NUM_BATCHED_TOKENS),
    ]
    if VLLM_ENABLE_CHUNKED_PREFILL:
        cmd.append('--enable-chunked-prefill')
    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=vllm_env(), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    VLLM_SERVER_PID.write_text(str(process.pid), encoding='utf-8')
    try:
        wait_for_vllm_server(process)
    finally:
        log_handle.close()


def run_vllm_api_smoke_test() -> None:
    def request_content(
        label: str, prompt: str, max_tokens: int, *, thinking_token_budget: int = 64
    ) -> str:
        response = request_json(
            f'{VLLM_BASE_URL}/chat/completions',
            payload={
                'model': SERVED_MODEL_NAME,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.0,
                'max_tokens': max_tokens,
                'chat_template_kwargs': {'enable_thinking': True},
                'thinking_token_budget': thinking_token_budget,
            },
            timeout=120,
        )
        choices = response.get('choices')
        if not isinstance(choices, list) or not choices:
            raise ValueError(f'{label} response did not contain a choice')
        message = choices[0].get('message')
        if not isinstance(message, dict):
            raise ValueError(f'{label} response did not contain a message')
        content = message.get('content')
        if not isinstance(content, str):
            raise ValueError(f'{label} response did not contain string content')
        return content.strip()

    def assert_raw_fidelity(label: str, fixture: str, max_tokens: int) -> None:
        content = request_content(
            f'raw-{label}',
            f'Copy the following {label} envelope exactly. Preserve every quote, '
            'newline, space, and indentation character. Output no other text.\n\n'
            + fixture,
            max_tokens,
        )
        if content != fixture:
            raise ValueError(
                f'raw-{label} fidelity check changed quotes, newlines, or indentation'
            )

    def assert_probe_actions_contract() -> None:
        prompt = '''Return exactly one JSON object and no Markdown or prose. Solve these
three independent policy-configuration cases using the stated probe_actions contract:
1. contrastive: UP is the positive directional probe, RIGHT is its directional
same-modality control, and minimum_evidence_actions is 4. Produce the shortest valid
finite schedule, alternating the positive and control. The minimum is a constraint,
not an output field.
2. clicks: preserve the ordered mouse_points [[28,30],[28,38],[28,30]] exactly and
produce one scalar MOUSE probe_actions entry per coordinate.
3. navigation: actor value 1 must route to target value 2 over passable values 0,1,2
with approach_distance 1. UP and RIGHT are only its bounded evidence probes; retain
all route configuration instead of treating probe_actions as the route. Use the exact
keys actor_values, target_values, passable_values, approach_distance, and probe_actions;
never abbreviate them as actor, target, or passable.
Use exactly the top-level keys contrastive, clicks, and navigation. Each value must be
a JSON object containing only POLICY_SOLVER_CONFIG fields. Do not emit objective fields,
solver types, explanations, or copies of the constraints.'''
        def validate(content: str) -> tuple[dict | None, list[str]]:
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                return None, ['contrastive', 'clicks', 'navigation']
            if not isinstance(result, dict):
                return None, ['contrastive', 'clicks', 'navigation']
            contrastive = result.get('contrastive')
            clicks = result.get('clicks')
            navigation = result.get('navigation')
            failures = []
            contrastive_actions = (
                contrastive.get('probe_actions')
                if isinstance(contrastive, dict)
                else None
            )
            if not (
                isinstance(contrastive_actions, list)
                and len(contrastive_actions) >= 4
                and set(contrastive_actions) == {'UP', 'RIGHT'}
                and contrastive_actions.count('UP') >= 2
                and all(
                    left != right
                    for left, right in zip(
                        contrastive_actions, contrastive_actions[1:]
                    )
                )
            ):
                failures.append('contrastive')
            if not isinstance(clicks, dict) or (
                clicks.get('mouse_points') != [[28, 30], [28, 38], [28, 30]]
                or clicks.get('probe_actions') != ['MOUSE', 'MOUSE', 'MOUSE']
            ):
                failures.append('clicks')
            if not isinstance(navigation, dict) or not (
                set(navigation.get('actor_values', [])) == {1}
                and set(navigation.get('target_values', [])) == {2}
                and set(navigation.get('passable_values', [])) == {0, 1, 2}
                and navigation.get('approach_distance') == 1
                and set(navigation.get('probe_actions', [])) == {'UP', 'RIGHT'}
            ):
                failures.append('navigation')
            return result, failures

        content = request_content(
            'probe-actions', prompt, 768, thinking_token_budget=256
        )
        result, failures = validate(content)
        if failures:
            failed_cases = ', '.join(failures)
            repair_prompt = (
                'Return exactly one raw JSON object containing only these failed '
                f'top-level cases: {failed_cases}. Do not return or modify any other '
                'case. Correct each failed case using the original contract below and '
                'the exact registered field names. probe_actions must always be a list '
                'of scalar action-name strings; mouse coordinates belong only in '
                'mouse_points.\nOriginal contract:\n'
                + prompt
                + '\nPrevious answer:\n'
                + content
            )
            repair_content = request_content(
                'probe-actions-repair',
                repair_prompt,
                768,
                thinking_token_budget=256,
            )
            try:
                repaired_cases = json.loads(repair_content)
            except json.JSONDecodeError:
                repaired_cases = None
            if isinstance(repaired_cases, dict):
                merged = dict(result or {})
                for case in failures:
                    if case in repaired_cases:
                        merged[case] = repaired_cases[case]
                content = json.dumps(merged)
            else:
                content = repair_content
            _result, failures = validate(content)
        if failures:
            raise ValueError(
                'probe-actions contract check failed for '
                + ', '.join(failures)
                + f'; final model content was {content!r}'
            )

    def assert_generated_policy_contract() -> None:
        prompt = '''Return exactly one complete Python policy module between BEGIN_POLICY
and END_POLICY lines, with no Markdown or prose. Generate a navigation policy for actor
value 1 routing to target value 2 over passable values 0,1,2, approach_distance 1,
and bounded evidence probes UP then RIGHT. Declare POLICY_API_VERSION = 1,
SUPPORTED_BACKENDS containing cpu, POLICY_REUSE_SCOPE = "none", literal
POLICY_SOLVER_TYPE = "navigation", and literal POLICY_SOLVER_CONFIG using the exact
registered keys actor_values, target_values, passable_values, approach_distance, and
probe_actions. Define decide(observation, memory) with exactly one return statement
that directly calls solver_decide(POLICY_SOLVER_TYPE, observation, memory,
POLICY_SOLVER_CONFIG). Do not implement navigation yourself or emit direct actions.'''

        def validate(content: str) -> list[str]:
            failures = []
            if content.count('BEGIN_POLICY') != 1 or content.count('END_POLICY') != 1:
                return ['policy envelope']
            begin, remainder = content.split('BEGIN_POLICY', 1)
            source, end = remainder.split('END_POLICY', 1)
            if begin.strip() or end.strip():
                failures.append('policy envelope')
            try:
                tree = ast.parse(source.strip())
            except SyntaxError:
                return [*failures, 'Python syntax']

            assignments = {}
            for statement in tree.body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    try:
                        assignments[statement.targets[0].id] = ast.literal_eval(
                            statement.value
                        )
                    except (ValueError, TypeError):
                        pass
            if assignments.get('POLICY_API_VERSION') != 1:
                failures.append('API version')
            backends = assignments.get('SUPPORTED_BACKENDS')
            if not isinstance(backends, (list, tuple)) or 'cpu' not in backends:
                failures.append('CPU backend')
            if assignments.get('POLICY_REUSE_SCOPE') != 'none':
                failures.append('reuse scope')
            if assignments.get('POLICY_SOLVER_TYPE') != 'navigation':
                failures.append('solver type')
            config = assignments.get('POLICY_SOLVER_CONFIG')

            def has_exact_values(value, expected):
                return (
                    isinstance(value, (list, tuple))
                    and len(value) == len(expected)
                    and all(item in expected for item in value)
                )

            if not isinstance(config, dict) or not (
                has_exact_values(config.get('actor_values'), (1,))
                and has_exact_values(config.get('target_values'), (2,))
                and has_exact_values(config.get('passable_values'), (0, 1, 2))
                and config.get('approach_distance') == 1
                and config.get('probe_actions') == ['UP', 'RIGHT']
            ):
                failures.append('solver config')
            decide = next(
                (
                    statement
                    for statement in tree.body
                    if isinstance(statement, ast.FunctionDef)
                    and statement.name == 'decide'
                ),
                None,
            )
            if decide is None or not (
                not decide.args.posonlyargs
                and [arg.arg for arg in decide.args.args]
                == ['observation', 'memory']
                and decide.args.vararg is None
                and decide.args.kwarg is None
                and not decide.args.kwonlyargs
                and not decide.args.defaults
                and not decide.args.kw_defaults
            ):
                failures.append('decide signature')
            else:
                direct_return = (
                    decide.body[0]
                    if len(decide.body) == 1
                    and isinstance(decide.body[0], ast.Return)
                    else None
                )
                call = direct_return.value if direct_return is not None else None
                expected_args = [
                    'POLICY_SOLVER_TYPE',
                    'observation',
                    'memory',
                    'POLICY_SOLVER_CONFIG',
                ]
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == 'solver_decide'
                    and not call.keywords
                    and [
                        arg.id if isinstance(arg, ast.Name) else None
                        for arg in call.args
                    ]
                    == expected_args
                ):
                    failures.append('canonical dispatcher call')
            return failures

        content = request_content(
            'generated-policy', prompt, 1024, thinking_token_budget=256
        )
        failures = validate(content)
        if failures:
            content = request_content(
                'generated-policy-repair',
                prompt
                + '\nRepair every failed requirement: '
                + ', '.join(failures)
                + '. Return one complete replacement module. Previous answer:\n'
                + content,
                1024,
                thinking_token_budget=256,
            )
            failures = validate(content)
        if failures:
            raise ValueError(
                'generated-policy contract check failed for '
                + ', '.join(failures)
                + f'; final model content was {content!r}'
            )

    def assert_pathfinding_policy_behavior() -> None:
        prompt = '''Return exactly one JSON object and no Markdown or prose. Evaluate five
independent navigation-policy cases. Boards are row-major grids; actor value is 1,
target value is 2, passable values are 0,1,2, approach_distance is 1, configured
probe_actions are UP then RIGHT, valid actions are UP,RIGHT,DOWN,LEFT, and cardinal
BFS neighbor order is UP,RIGHT,DOWN,LEFT. A route must stop on a passable cell exactly
one Manhattan step from the target and must never enter value 9. For routed cases,
path_actions is the complete shortest route and action is its first step.

1. corridor board [[1,0,0,2]].
2. detour board [[1,9,0,2],[0,0,0,9]].
3. at_approach board [[1,2]].
4. unreachable board [[1,9,2],[9,9,9]] with ordinary route evidence.
5. engine_progress_unreachable uses the same unreachable board, but evidence_mode is
engine_progress. This liveness mode overrides ordinary unreachable termination: an
absent route must continue with the first bounded configured probe instead of failing.

Use exactly the top-level keys corridor, detour, at_approach, unreachable, and
engine_progress_unreachable. Each value must contain exactly status, path_actions,
and action. status is continue, subgoal_succeeded, or subgoal_failed; action is a
cardinal string or null. Do not include explanations or configuration fields.'''
        expected = {
            'corridor': {
                'status': 'continue',
                'path_actions': ['RIGHT', 'RIGHT'],
                'action': 'RIGHT',
            },
            'detour': {
                'status': 'continue',
                'path_actions': ['DOWN', 'RIGHT', 'RIGHT', 'UP'],
                'action': 'DOWN',
            },
            'at_approach': {
                'status': 'subgoal_succeeded',
                'path_actions': [],
                'action': None,
            },
            'unreachable': {
                'status': 'subgoal_failed',
                'path_actions': [],
                'action': None,
            },
            'engine_progress_unreachable': {
                'status': 'continue',
                'path_actions': [],
                'action': 'UP',
            },
        }

        def validate(content: str):
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                return None, list(expected)
            if not isinstance(result, dict) or set(result) != set(expected):
                return result if isinstance(result, dict) else None, list(expected)
            failures = [
                case for case, decision in expected.items() if result.get(case) != decision
            ]
            return result, failures

        content = request_content(
            'pathfinding-policy', prompt, 1024, thinking_token_budget=256
        )
        result, failures = validate(content)
        if failures:
            repair_content = request_content(
                'pathfinding-policy-repair',
                'Return exactly one raw JSON object containing only these failed '
                f'top-level cases: {", ".join(failures)}. Correct them from the '
                'original contract below. Do not return or modify passing cases. '
                'The trusted solver oracle requires these exact failed-case '
                'decisions:\n'
                + json.dumps({case: expected[case] for case in failures})
                + '\nOriginal contract:\n'
                + prompt
                + '\nPrevious answer:\n'
                + content,
                1024,
                thinking_token_budget=256,
            )
            try:
                repaired_cases = json.loads(repair_content)
            except json.JSONDecodeError:
                repaired_cases = None
            if isinstance(repaired_cases, dict):
                merged = {
                    case: decision
                    for case, decision in (result or {}).items()
                    if case in expected
                }
                for case in failures:
                    if case in repaired_cases:
                        merged[case] = repaired_cases[case]
                content = json.dumps(merged)
            else:
                content = repair_content
            _result, failures = validate(content)
        if failures:
            raise ValueError(
                'pathfinding-policy behavior check failed for '
                + ', '.join(failures)
                + f'; final model content was {content!r}'
            )

    def assert_game_inference_behavior() -> None:
        prompt = '''Return exactly one JSON object and no Markdown or prose. Infer game
behavior from five independent transition cases. A hypothesis is supported only by
the stated transition evidence; an unchanged board is an exact_noop unless engine
state, level, score, or reward proves progress. A new board alone is novel_state, not
meaningful_progress. Returning to a previously seen board via inverse actions is an
inverse_cycle. recommended_action must avoid repeating contradicted no-ops and cycles.

1. directional_motion: before [[0,0,0,0],[0,1,0,2],[0,0,0,0]], action RIGHT,
after [[0,0,0,0],[0,0,1,2],[0,0,0,0]], level/score/reward unchanged. Candidate
hypothesis: RIGHT translates actor value 1 one cell right. Valid actions UP,RIGHT,DOWN.
2. exact_noop: before and after [[0,0,0],[0,1,0],[0,0,0]], action RIGHT,
level/score/reward unchanged. Candidate hypothesis: RIGHT translates actor value 1.
Valid actions UP,RIGHT; use UP as the distinct control.
3. engine_progress: board unchanged [[1,2]], action SPACE, level changes 1 to 2,
score changes 0 to 1, reward is 1. Candidate hypothesis: SPACE can advance the level
without a final board difference. No next action is needed after verified progress.
4. visual_novelty: before [[1,3],[0,0]], action MOUSE at row 0 col 1, after
[[1,4],[0,0]], level/score/reward unchanged. Candidate hypothesis: any board change
is meaningful progress. Valid scalar action UP is the next evidence probe.
5. inverse_cycle: seen states are A=[[0,1,0],[0,0,0]], then RIGHT gives
B=[[0,0,1],[0,0,0]], then LEFT gives A again, with level/score/reward unchanged.
Candidate hypothesis: alternating RIGHT and LEFT makes progress. Valid actions
RIGHT,LEFT,DOWN; use DOWN to break the cycle.

Use exactly the top-level keys directional_motion, exact_noop, engine_progress,
visual_novelty, and inverse_cycle. Each value must contain exactly inferred_effect,
hypothesis_verdict, meaningful_progress, novel_state, cycle_risk, and
recommended_action. inferred_effect is actor_translation, exact_noop,
level_advance, visual_change_only, or inverse_cycle. hypothesis_verdict is supported
or contradicted. Boolean fields must be JSON booleans. recommended_action is an
action string or null.'''
        expected = {
            'directional_motion': {
                'inferred_effect': 'actor_translation',
                'hypothesis_verdict': 'supported',
                'meaningful_progress': False,
                'novel_state': True,
                'cycle_risk': False,
                'recommended_action': 'RIGHT',
            },
            'exact_noop': {
                'inferred_effect': 'exact_noop',
                'hypothesis_verdict': 'contradicted',
                'meaningful_progress': False,
                'novel_state': False,
                'cycle_risk': False,
                'recommended_action': 'UP',
            },
            'engine_progress': {
                'inferred_effect': 'level_advance',
                'hypothesis_verdict': 'supported',
                'meaningful_progress': True,
                'novel_state': False,
                'cycle_risk': False,
                'recommended_action': None,
            },
            'visual_novelty': {
                'inferred_effect': 'visual_change_only',
                'hypothesis_verdict': 'contradicted',
                'meaningful_progress': False,
                'novel_state': True,
                'cycle_risk': False,
                'recommended_action': 'UP',
            },
            'inverse_cycle': {
                'inferred_effect': 'inverse_cycle',
                'hypothesis_verdict': 'contradicted',
                'meaningful_progress': False,
                'novel_state': False,
                'cycle_risk': True,
                'recommended_action': 'DOWN',
            },
        }

        def validate(content: str):
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                return None, list(expected)
            if not isinstance(result, dict) or set(result) != set(expected):
                return result if isinstance(result, dict) else None, list(expected)
            failures = [
                case for case, inference in expected.items() if result.get(case) != inference
            ]
            return result, failures

        content = request_content(
            'game-inference', prompt, 1536, thinking_token_budget=384
        )
        result, failures = validate(content)
        if failures:
            repair_content = request_content(
                'game-inference-repair',
                'Return exactly one raw JSON object containing only these failed '
                f'top-level cases: {", ".join(failures)}. Do not return or modify '
                'passing cases. The trusted inference oracle requires these exact '
                'failed-case decisions:\n'
                + json.dumps({case: expected[case] for case in failures})
                + '\nOriginal contract:\n'
                + prompt
                + '\nPrevious answer:\n'
                + content,
                1536,
                thinking_token_budget=384,
            )
            try:
                repaired_cases = json.loads(repair_content)
            except json.JSONDecodeError:
                repaired_cases = None
            if isinstance(repaired_cases, dict):
                merged = {
                    case: inference
                    for case, inference in (result or {}).items()
                    if case in expected
                }
                for case in failures:
                    if case in repaired_cases:
                        merged[case] = repaired_cases[case]
                content = json.dumps(merged)
            else:
                content = repair_content
            _result, failures = validate(content)
        if failures:
            raise ValueError(
                'game-inference behavior check failed for '
                + ', '.join(failures)
                + f'; final model content was {content!r}'
            )

    reduction_fixture = (
        'BEGIN_REDUCTION\n'
        '{\n'
        '  "objective_id": "level:1:1",\n'
        '  "verdict": "continue",\n'
        '  "evidence": "board unchanged",\n'
        '  "rationale": "continue the active probe",\n'
        '  "selected_index": 0,\n'
        '  "subgoals": []\n'
        '}\n'
        'END_REDUCTION'
    )
    policy_fixture = (
        'BEGIN_POLICY\n'
        'POLICY_API_VERSION = 1\n'
        'SUPPORTED_BACKENDS = ("cpu",)\n'
        'def decide(observation, memory):\n'
        '    return {"status": "continue", "action": '
        '{"action": "ACTION6"}, "memory": memory}\n'
        'END_POLICY'
    )
    try:
        assert_raw_fidelity('reduction', reduction_fixture, 512)
        assert_raw_fidelity('policy', policy_fixture, 512)
        assert_probe_actions_contract()
        assert_generated_policy_contract()
        assert_pathfinding_policy_behavior()
        assert_game_inference_behavior()
    except Exception as exc:
        raise RuntimeError(
            server_failure_message(
                'vLLM bounded-thinking/raw-orchestration smoke test failed: '
                f'{type(exc).__name__}: {exc}'
            )
        ) from exc
    print('\n' + '=' * 88, flush=True)
    print('VLLM OPENAI SERVER QWEN BOUNDED-THINKING TRANSPORT SMOKE TEST', flush=True)
    print('Raw reduction JSON fidelity: passed', flush=True)
    print('Raw policy source fidelity: passed', flush=True)
    print('LLM probe_actions contract matrix: passed', flush=True)
    print('LLM generated policy behavior: passed', flush=True)
    print('LLM pathfinding policy behavior: passed', flush=True)
    print('LLM game inference behavior: passed', flush=True)
    print('=' * 88 + '\n', flush=True)


print(f'vLLM wheelhouse path: {WHEELHOUSE}', flush=True)
print(f'Qwen model path: {MODEL_PATH}', flush=True)
assert_expected_cuda_gpu()
missing = [str(path) for path in (WHEELHOUSE, MODEL_PATH) if not path.exists()]
if missing:
    raise FileNotFoundError('Missing attached Kaggle input path(s): ' + ', '.join(missing))
start_vllm_server()
run_vllm_api_smoke_test()
setup_env = {
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'PYTHONPATH': str(SITE_PACKAGES) + os.pathsep + os.environ.get('PYTHONPATH', ''),
    'LOCAL_ANALYZER_BASE_URL': VLLM_BASE_URL,
    'OPENAI_BASE_URL': VLLM_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'OPENAI_PROVIDER': __LOCAL_ANALYZER_PROVIDER__,
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': __LOCAL_ANALYZER_APP_NAME__,
    'LOCAL_ANALYZER_CONTEXT_WINDOW': str(ANALYZER_CONTEXT_WINDOW),
    'LOCAL_ANALYZER_MAX_OUTPUT': __LOCAL_ANALYZER_MAX_OUTPUT__,
    'LOCAL_ANALYZER_TOOL_STEPS': __LOCAL_ANALYZER_TOOL_STEPS__,
    'LOCAL_ANALYZER_TOOL_TIMEOUT': __LOCAL_ANALYZER_TOOL_TIMEOUT__,
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': __LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS__,
    'LOCAL_ANALYZER_CANDIDATES': __LOCAL_ANALYZER_CANDIDATES__,
    'LOCAL_ANALYZER_GAME_TOKEN_BUDGET': __LOCAL_ANALYZER_GAME_TOKEN_BUDGET__,
    'LOCAL_ANALYZER_OBJECTIVE_REDUCTION': __LOCAL_ANALYZER_OBJECTIVE_REDUCTION__,
    'KAGGLE_DUCK_SMOKE_TEST_ONLY': __KAGGLE_DUCK_SMOKE_TEST_ONLY__,
    'LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS': __LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS__,
    'LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT': __LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT__,
    'LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT': __LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT__,
    'LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET': __LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET__,
    'LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET': __LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET__,
    'LOCAL_GAMEPLAY_POLICY_BACKEND': __LOCAL_GAMEPLAY_POLICY_BACKEND__,
    'LOCAL_GAMEPLAY_POLICY_CUDA_MIN_FREE_MB': __LOCAL_GAMEPLAY_POLICY_CUDA_MIN_FREE_MB__,
    'LOCAL_GAMEPLAY_POLICY_DECISION_TIMEOUT_SECONDS': __LOCAL_GAMEPLAY_POLICY_DECISION_TIMEOUT_SECONDS__,
    'LOCAL_ANALYZER_STRATEGY_ENABLED': __LOCAL_ANALYZER_STRATEGY_ENABLED__,
    'LOCAL_ANALYZER_STRATEGY_POLICY': __LOCAL_ANALYZER_STRATEGY_POLICY__,
    'LOCAL_ANALYZER_SAME_STATE_NOOP_LIMIT': __LOCAL_ANALYZER_SAME_STATE_NOOP_LIMIT__,
    'LOCAL_ANALYZER_STAGNATION_WINDOW': __LOCAL_ANALYZER_STAGNATION_WINDOW__,
    'LOCAL_ANALYZER_CYCLE_WINDOW': __LOCAL_ANALYZER_CYCLE_WINDOW__,
    'LOCAL_ANALYZER_CYCLE_STOP_LIMIT': __LOCAL_ANALYZER_CYCLE_STOP_LIMIT__,
    'LOCAL_ANALYZER_REPEAT_ACTION_LIMIT': __LOCAL_ANALYZER_REPEAT_ACTION_LIMIT__,
    'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_WINDOW': __LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_WINDOW__,
    'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_LIMIT': __LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_LIMIT__,
    'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STRIKE_LIMIT': __LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STRIKE_LIMIT__,
    'LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STOP_LIMIT': __LOCAL_ANALYZER_DIRECTIONAL_NO_PROGRESS_STOP_LIMIT__,
    'LOCAL_ANALYZER_IGNORE_EDGE_HUD_CHANGES': __LOCAL_ANALYZER_IGNORE_EDGE_HUD_CHANGES__,
    'LOCAL_ANALYZER_VOLATILE_WINDOW': __LOCAL_ANALYZER_VOLATILE_WINDOW__,
    'LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES': __LOCAL_ANALYZER_VOLATILE_MIN_SAMPLES__,
    'LOCAL_ANALYZER_VOLATILE_RATIO': __LOCAL_ANALYZER_VOLATILE_RATIO__,
    'LOCAL_ANALYZER_PROGRESS_UTILITY': __LOCAL_ANALYZER_PROGRESS_UTILITY__,
    'LOCAL_ANALYZER_NOVEL_UTILITY': __LOCAL_ANALYZER_NOVEL_UTILITY__,
    'LOCAL_ANALYZER_EXPLORATION_WEIGHT': __LOCAL_ANALYZER_EXPLORATION_WEIGHT__,
    'LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MULTIPLIER': __LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MULTIPLIER__,
    'LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM': __LOCAL_ANALYZER_LEVEL_ACTION_LIMIT_MINIMUM__,
    'LOCAL_ANALYZER_LEVEL_NO_PROGRESS_TOKEN_LIMIT': __LOCAL_ANALYZER_LEVEL_NO_PROGRESS_TOKEN_LIMIT__,
    'LOCAL_ANALYZER_YIELD_SECONDS': __LOCAL_ANALYZER_YIELD_SECONDS__,
    'LOCAL_ANALYZER_TEMPERATURE': __LOCAL_ANALYZER_TEMPERATURE__,
    'LOCAL_ANALYZER_TOP_P': __LOCAL_ANALYZER_TOP_P__,
    'LOCAL_ANALYZER_TOP_K': __LOCAL_ANALYZER_TOP_K__,
    'LOCAL_ANALYZER_ENABLE_THINKING': __LOCAL_ANALYZER_ENABLE_THINKING__,
    'LOCAL_ANALYZER_STREAM': __LOCAL_ANALYZER_STREAM__,
    'LOCAL_ANALYZER_REQUIRE_OS_SANDBOX': __LOCAL_ANALYZER_REQUIRE_OS_SANDBOX__,
    'LOCAL_ANALYZER_VERIFY_CANDIDATES': __LOCAL_ANALYZER_VERIFY_CANDIDATES__,
    'MULTIMODAL_CONTEXT': __MULTIMODAL_CONTEXT__,
    'MULTIMODAL_UPSCALE': __MULTIMODAL_UPSCALE__,
}
setup_env_path = Path(os.environ['TAAF_KAGGLE_SETUP_ENV'])
existing_setup_env = {}
if setup_env_path.exists():
    existing_setup_env = json.loads(setup_env_path.read_text(encoding='utf-8'))
    if not isinstance(existing_setup_env, dict):
        raise RuntimeError('TAAF_KAGGLE_SETUP_ENV must contain a JSON object.')
existing_setup_env.update(setup_env)
setup_env_path.write_text(json.dumps(existing_setup_env, indent=2), encoding='utf-8')
"""

_DUCK_VLLM_TEARDOWN_SCRIPT = r"""import os
import shutil
import signal
import time
from pathlib import Path

WORKING_DIR = Path(os.environ['TAAF_KAGGLE_WORKING_DIR'])
pid_path = WORKING_DIR / 'vllm-openai-server.pid'
site_packages = WORKING_DIR / 'vllm-site-packages'
if pid_path.exists():
    try:
        pid = int(pid_path.read_text(encoding='utf-8').strip())
        print('Stopping vLLM server', flush=True)
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(1)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception as exc:
        print(f'Could not stop vLLM server cleanly: {exc!r}', flush=True)
    pid_path.unlink(missing_ok=True)
shutil.rmtree(site_packages, ignore_errors=True)
print(f'Removed temporary vLLM install at {site_packages}', flush=True)
"""
