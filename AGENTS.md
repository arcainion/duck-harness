# Agent Notes

## Docker / Kaggle development container

- Docker Desktop is installed per-user. In PowerShell sessions where `docker`
  is not on `PATH`, invoke the CLI directly at:
  `C:\Users\arcai\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`.
- Do not assume Docker is installed inside WSL; use the Windows CLI above.
- The persistent local Kaggle development container is named
  `duck-kaggle-dev` and uses `kaggle/python:latest`.
- The repository is bind-mounted in that container at
  `/workspace/duck-harness`, which is also its working directory.
- The container was created with `--gpus all`. Confirm the currently exposed
  GPU with `nvidia-smi` before running GPU-dependent tests; local hardware is
  not identical to Kaggle's requested T4 environment.
- Use container names rather than transient container IDs. For example:

  ```powershell
  $dockerCli = 'C:\Users\arcai\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
  & $dockerCli ps -a
  & $dockerCli start duck-kaggle-dev
  & $dockerCli exec duck-kaggle-dev sh -lc 'cd /workspace/duck-harness && nvidia-smi'
  ```

- The older `nervous_mirzakhani` container is a stopped one-shot container
  with no persistent command, repository mount, or GPU request. Do not use it
  for Duck deployment testing.

## Running `make kaggle-duck` (Kaggle deployment)

- The target lives in `ARC3-Inference/Makefile` (lines 407-422) and must be run
  from inside `ARC3-Inference/`. It delegates to TAAF's `kaggle` deployment:
  the current `tufa-arc-agi-framework` + `ARC3-Inference` sources are bundled
  into a Kaggle source dataset, a private notebook is pushed, and the notebook
  boots a local vLLM server and runs the duck against it.
- It must be run inside the `duck-kaggle-dev` container, which has the repo at
  `/workspace/duck-harness` (see "Docker / Kaggle development container"
  above). The container is a `kaggle/python` image, so the `kaggle` CLI and
  `make install`ed venv must be set up there. Use the wrapper
  `ARC3-Inference/kaggle-duck.sh`, which runs `make kaggle-duck` with sensible
  defaults and prints the notebook/dataset URLs plus a best-effort
  `kaggle kernels status` after a live push. Run it from the Windows host as:

  ```powershell
  $dockerCli = 'C:\Users\arcai\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
  & $dockerCli exec duck-kaggle-dev bash /workspace/duck-harness/ARC3-Inference/kaggle-duck.sh
  ```

  or interactively from a shell inside the container
  (`& $dockerCli exec -it duck-kaggle-dev bash`), working from
  `/workspace/duck-harness/ARC3-Inference/`:

  ```bash
  ./kaggle-duck.sh
  ```

- The wrapper defaults to `RUN_NAME=duck-harness-kaggle`,
  `KAGGLE_KERNEL_SLUG=taaf-<RUN_NAME>`, and
  `KAGGLE_DATASET_REF=<KAGGLE_USERNAME>/taaf-kaggle-source-<RUN_NAME>`.
  Override any value by exporting it first, e.g.
  `RUN_NAME=duck-harness-20260816 ./kaggle-duck.sh`; `KAGGLE_DRY_RUN=true`
  stages without pushing. Its status check uses the credential export
  workaround for the container's `kaggle` CLI.
- Set `KAGGLE_DUCK_DIAGNOSTIC=true` to run the five-game diagnostic suite at
  concurrency 3 with a 900-second analyzer timeout before committing to the public harness. Override the suite
  with `KAGGLE_DUCK_DIAGNOSTIC_GAMES='["game-id", ...]'` when needed.
- Set `KAGGLE_DUCK_OBJECTIVE_REDUCTION=true` to use the host-owned objective
  tree and generated gameplay-policy runtime. Gameplay defaults to CPU; select
  `LOCAL_GAMEPLAY_POLICY_BACKEND=auto` or `cuda` explicitly to permit CUDA.
- Prerequisites: run `make install` first (the target uses `uv run --no-sync`),
  the `kaggle` CLI must be installed and on `PATH` inside the container
  (`python -m pip install kaggle`), and credentials must resolve inside the
  container from `KAGGLE_USERNAME` + `KAGGLE_KEY` (or
  `~/.kaggle/kaggle.json`). The CLI is verified with `kaggle kernels list
  --mine` before pushing.
- `LOCAL_ANALYZER_PROVIDER` must be vLLM/OpenAI-compatible; the setup rejects
  anything else (e.g. `openrouter`) because the notebook runs its own vLLM.
- Typical per-run overrides (inside the container, via the wrapper):

  ```bash
  RUN_NAME=duck-harness-20260816 \
  KAGGLE_KERNEL_SLUG=taaf-duck-harness-20260816 \
  KAGGLE_DATASET_REF=arcainionprime/taaf-kaggle-source-duck-harness-20260816 \
  ./kaggle-duck.sh
  ```

  The kernel title must slugify to the kernel slug. Use
  `DEPLOYMENT_WAIT=true` to block until the notebook finishes and pull the
  output back into the run directory, and `KAGGLE_DRY_RUN=true` to stage the
  bundles and metadata without calling the Kaggle API.
- To pull the latest completed notebook output explicitly for analysis, run
  the following inside `duck-kaggle-dev`, replacing the destination with an
  existing or desired directory:

  ```bash
  kaggle kernels output arcainionprime/taaf-duck-harness-kaggle -p /path/to/dest
  ```

  The repository wrapper performs the status check, credential export, directory
  creation, and download with these defaults:

  ```bash
  cd /workspace/duck-harness/ARC3-Inference
  bash ./pull-kaggle-output.sh
  ```

  Override the destination with `--path /path/to/dest`, or the kernel with
  `--kernel owner/slug`. The equivalent environment variables are
  `KAGGLE_OUTPUT_DIR` and `KAGGLE_KERNEL_REF`.

  For the repository-mounted analysis directory, use
  `/workspace/duck-harness/results` as the destination. The command requires
  working Kaggle credentials in the container and downloads the kernel's
  current published output, so check `kaggle kernels status
  arcainionprime/taaf-duck-harness-kaggle` first when a run may still be active.
- Target defaults (Makefile, not the README): `AGENT=duck-harness`,
  `MODEL=local`, `KAGGLE_DUCK_PUBLIC_HARNESS=true` (the 25 official
  ARC-AGI-3 games), `N_PASSES=1`, `CONCURRENT_JOBS=12`,
  `MAX_RUNTIME_MINUTES=132`, `MAX_EXPERIMENT_RUNTIME_MINUTES=540`,
  `ANALYZER_TIMEOUT=900`, one candidate, unlimited tool steps, server-default
  response sizing, thinking enabled and preserved across turns, a 100,000-token
  per-game budget, and a per-level action
  ceiling of twice the baseline with a minimum of 20 actions, plus a 75,000-token
  no-progress ceiling that resets after level progress. Up to two cooperative
  60-second analyzer yields resume the same analysis step; a third yield at the
  same state uses controller fallback. An unavailable fallback is recorded and
  rolls into a fresh analyzer step instead of terminating the game. Fallback
  tries all safe mouse coordinates and deprioritizes an immediately repeated
  no-progress fallback action. Controller execution also falls back
  after two genuine no-action analyzer turns at the same state and
  reports final no-action generated tokens in benchmark totals. Controller
  stagnation/cycle windows are 6/4; edge-only HUD changes are ignored, exact
  click coordinates are blocked after two failed attempts, inverse movement
  cycles are guarded, pure object translations are not rewarded as novel, a
  direction used at least 12 times in a 16-action no-progress window is guarded,
  persistently blocked after three guard strikes, and the level stops after
  eight directional no-progress guards,
  and eight consecutive cycle-risk actions stop the run.
  Progress utility is 4.0, novel-state utility is 0.05, and exploration weight
  is 0.5.
- The notebook attaches the duck solver's declared datasets:
  `driessmit1/arc3-vllm-h100-wheelhouse-v3` (vLLM wheelhouse, installed
  offline into a temporary target dir), plus the Kaggle Model
  `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pyTorch/hf-fp8/1` (served as
  `Qwen/Qwen3.8-27B-FP8`). The setup resolves the attached model via
  KaggleHub and asserts the expected GPU shape before starting vLLM.
- Accelerator defaults to `NvidiaRtxPro6000` (rtx-pro-6000 profile,
  `max_model_len=65536`, TP=1). With `KAGGLE_ACCELERATOR=NvidiaTeslaT4` the
  profile switches to `max_model_len=8192`, TP=2 across the two exposed T4s.
  Pass `KAGGLE_ACCELERATOR=...` as a Make override to match the requested
  allocation; Kaggle hardware is not identical to the local T4 machine.
