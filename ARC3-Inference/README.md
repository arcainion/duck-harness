# The Duck 🦆

The Duck is the ARC3 inference harness in this repo: a tool-using solver that
plays ARC-AGI-3 games through TAAF.

It ties together:

- TAAF `Benchmark` / `GameAPI` execution
- a local OpenAI-compatible vLLM server, or OpenRouter
- the duck's single ephemeral `python` tool
- structured run artifacts for scoring, viewing, and trace export

The Python package lives under `inference/`. The run viewer lives under
`viewer/`.

## Quick Start

You need Python 3.12 and `uv`.

```bash
make install
```

Run with the default local vLLM config:

```bash
make server
make interactive
```

Submit the default Slurm run:

```bash
make sbatch
```

Run through OpenRouter instead:

```bash
export OPENROUTER_API_KEY=<your-openrouter-api-key>
CONFIG_PATH=configs/inference.openrouter.json make interactive
```

Open the viewer:

```bash
make view
```

If your runs are under the checked-in default experiment root, point the viewer
there:

```bash
make view VIEW_RUNS_DIR=/shared/arc_3_results/$USER
```

## What The Duck Does

For each TAAF game run, the harness starts a `HarnessSolver`. The solver gives
the duck the latest game state, valid actions, history, and a Python tool. The
duck inspects the board, writes small bits of code to reason about it, and calls
`action(...)` from inside Python to execute real game actions.

The duck can use:

- `current_frame.ascii` for a compact symbolic grid
- `current_frame.segmentation` for connected components, object hashes,
  boundaries, containment, and adjacency
- `current_frame.find(symbol, limit=64)` for a bounded coordinate sample,
  total count, and bounding box for one letter-coded color
- `current_frame.cell(row, col)` and `current_frame.neighbors(row, col)` for
  bounded local color, direction, and coordinate queries
- `current_frame.shortest_path(start, goal, passable=...)` for bounded BFS over
  caller-selected letter-coded colors, including the next step and move sequence
- `current_frame.shortest_path_to_any(start, goals, passable=...)` for selecting
  and routing to the nearest of up to 64 candidate coordinates in one search
- `current_frame.crop(top, left, bottom, right)` for a clipped letter-coded
  region, and `current_frame.diff(other)` for bounded before/after changes
- `history`, `previous_frame`, `transitions`, and `last_transition` for
  before/after reasoning
- `valid_actions` for the current action set
- `last_action_result` for fields such as `board_changed`, `level_completed`,
  `game_over`, `run_complete`, `reward`, and a bounded `animation` summary that
  preserves transient motion evidence even when the final board looks unchanged
- `experience` for the controller phase, opaque state identity, state visits,
  tried/no-op actions, short-cycle detection, outcome-aware action rankings,
  suggested probes, and verified local transition models that expose when the
  same state-action pair has produced consistent or contradictory results
- phase-specific action budgets: one-action probes while orienting, exploring,
  or recovering, and short batches only after the controller detects progress
- `strategy` plus `record_strategy(...)` for bounded goal, hypothesis, evidence,
  confidence, open-question, next-test, expected-outcome, fallback, and
  contradiction memory within one game run

The raw numeric grid is intentionally hidden from the Python tool. The preferred
view is `current_frame.segmentation`; use `current_frame.find(...)` to locate a
color and `current_frame.crop(...)` for small local checks without parsing or
printing the whole board.

Generated-code failures include bounded structured diagnostics with the error
type, source line and column, offending source text, and a repair hint so the
model can correct and retry an ephemeral Python call directly.

The model-facing actions are:

- `UP`, `DOWN`, `LEFT`, `RIGHT`
- `SPACE`
- `MOUSE(row=..., col=...)`

`MOUSE` uses `row` and `col`. Legacy `x` / `y` mouse fields are rejected.

Every Python tool call starts with fresh executable state. A bounded JSON
scratchpad remains available as `memory` within one game run: use
`remember(key, value)` to persist computed data and `forget(key)` or `forget()`
to remove it. The scratchpad allows 16 keys, 64 characters per key, 2 KiB per
value, and 8 KiB total. The tool can also import a small allowlist of standard
library modules, print compact summaries, assign a final value to `result`, and
call `action(...)` once or many times. The tool call timeout defaults to 30
seconds. Batched action results include a bounded `steps` list so each action can
be attributed to its individual before/after state and outcome.
Each batch is capped at 12 entries at both the sandbox and solver boundaries;
larger batches are rejected before any environment action executes.
Every queued action is screened for current-state terminal/negative-reward
evidence. The host repeats that check against cross-trial knowledge immediately
before each action, after any preceding state changes. Partial-batch failures
return a bounded `stop_detail` for the next reasoning turn.

## Configuration

The main config is strict JSON. Comments are not supported.

- `configs/inference.json` is the default local-vLLM / Slurm config.
- `configs/inference.openrouter.json` uses OpenRouter.
- `configs/eval.json` selects runs for `make eval`.
- `configs/significance.json` selects score files for `make significance`.

Use a different config with:

```bash
CONFIG_PATH=/path/to/config.json make interactive
CONFIG_PATH=/path/to/config.json make sbatch
```

Useful sections in `configs/inference.json`:

- `shared.*`: model name, base URL, provider, and context window.
- `experiments.root_dir`: where timestamped run directories are written.
- `environment.*`: games, tags, passes, concurrency, and runtime limits.
- `deployment.*`: inline vs Slurm and source repos bundled into Slurm jobs.
- `deployment.slurm.*`: GPU, walltime, partition, local-server startup, and
  extra `sbatch` flags.
- Kaggle runs are configured by CLI/Make overrides because the notebook slug
  and source dataset are usually per run.
- `server.*`: vLLM model-serving settings.
- `analyzer.*`: duck sampling/tool settings. This key is still named
  `analyzer` for compatibility with existing code and configs. The inference
  controller is configured by `strategy_enabled`, `same_state_noop_limit`,
  `stagnation_window`, and `cycle_window`. `strategy_policy` accepts
  `outcome_aware` (the default) or `legacy`. Outcome-aware mode exposes exact
  mouse-coordinate outcomes, support-weighted transition-graph plans, and
  phase-specific action budgets. Plans require repeated evidence by default;
  tune `plan_min_support`, `plan_min_confidence`, and `plan_max_depth`.
  Repeated actions from one run/pass count as one evidence source, preventing
  correlated observations from falsely satisfying the planner's support gate.
  Cross-trial action rankings use the same independent-source weighting while
  retaining raw observation totals for diagnostics.
  Plans are filtered against current and observed intermediate action
  availability, reject any edge with terminal/negative-reward evidence, and
  retain non-dominated confidence/utility routes during bounded search before
  selecting the best candidates. Candidate routes cannot revisit a state, so
  positive novelty cannot make cycles look like useful plans. Action rankings combine
  empirical value with uncertainty-weighted information gain; configure
  `progress_utility`, `novel_utility`, `revisit_utility`, `noop_utility`,
  `terminal_failure_utility`, and `exploration_weight`.
  The outcome-aware volatility detector can be tuned with `volatile_window`,
  `volatile_min_samples`, and `volatile_ratio`. Volatility evidence resets at
  each `RESET`, and deterministically repeated exact state/action effects are
  retained as controllable behavior even when their cells are otherwise masked.

Restore the previous controller for an ablation:

```bash
LOCAL_ANALYZER_STRATEGY_POLICY=legacy make interactive
```
- `analyzer.candidates` defaults to two. Candidate fan-out is reduced to one
  when the controller is confident or the per-game generation budget is low.
  Set `analyzer.game_token_budget` (or
  `LOCAL_ANALYZER_GAME_TOKEN_BUDGET`) to cap generated tokens per game.
  Exhausting that budget finishes the pass cleanly instead of repeatedly
  resubmitting an analyzer turn. If a compatible provider omits token usage,
  response size supplies a conservative generated-token estimate so the budget
  remains enforceable.
  Compilable candidates are ranked
  against empirical transition outcomes and contradictions as well as current
  action rankings, discouraged exact actions, mouse search history, action
  budget, and the confidence of any recommended plan.
- Passes of the same game run sequentially and share bounded progress lessons;
  different games remain concurrent. Shared knowledge is atomically persisted
  to `cross_trial_knowledge.json` in the run directory (or the path in
  `LOCAL_ANALYZER_KNOWLEDGE_PATH`) so interrupted runs can resume. It contains
  opaque state IDs and compact strategy/causal evidence, never raw frames.
  Writes use a process lock and merge the latest disk state to prevent parallel
  writers losing observations; a last-known-good backup is used after a
  truncated or corrupt primary file. Persistence failure never aborts gameplay.
  Executed transitions automatically ground causal action→outcome relations
  and resolve structured predictions that declare `expected_outcome`.
  Automatically grounded relations and predictions are conditioned on the
  behavioral state where they were observed, and causal-model updates refresh
  same-turn candidate scoring immediately. When the bounded relation store is
  full, weak stale automatically-grounded relations are evicted before new
  state evidence; model-authored rules are preserved.
- Mouse probes expose untried spatial-frontier coordinates around productive
  clicks. Animation summaries retain bounded per-frame change counts, motion
  vectors/directions, and reversibility; multimodal turns include both previous
  and current grids when the board changed.
- `chat.*`: direct chat probing with `make chat`.
- `viewer.port`: default viewer port.
- `multimodal.*`: image context for the current grid.

The checked-in default config currently runs the official tagged game set with
20 passes, 45 minutes per game, and `concurrent_jobs=16`. On Slurm it requests
two B200 GPUs and starts one local vLLM server per GPU. In that mode
`concurrent_jobs` is interpreted per GPU/server, so the effective concurrency is
32.

## Running Games

Run inline in the current process:

```bash
make interactive
```

Submit to Slurm:

```bash
make sbatch
```

Launch the validated duck harness on Kaggle:

```bash
make kaggle-duck \
  RUN_NAME=duck-harness-20260527 \
  KAGGLE_KERNEL_SLUG=taaf-duck-harness-20260527 \
  KAGGLE_DATASET_REF=driessmit1/taaf-kaggle-source-duck-harness-20260527
```

Name a run:

```bash
make sbatch RUN_NAME=baseline-qwen
```

Run one game or short-prefix:

```bash
make interactive GAME=taps N_PASSES=1 MAX_RUNTIME_MINUTES=10
```

Run the official tag set with a whole-experiment cap:

```bash
make sbatch GAME=[] GAME_TAGS=official MAX_EXPERIMENT_RUNTIME_MINUTES=360
```

Run the duck locally against TAAF's competition Arcade simulator:

```bash
make interactive \
  GAME=[] GAME_TAGS=official \
  SIMULATE_COMPETITION_ARCADE=true \
  COMPETITION_CLONE_RUNS=110 \
  N_PASSES=1
```

The simulator is inline-only. `COMPETITION_CLONE_RUNS=110` repeats the selected
25 official games with unique competition-safe IDs, which catches submission
Arcade issues without waiting for a Kaggle rerun.

Common overrides:

- `GAME`: one game, comma-separated games, or a JSON list.
- `GAME_TAGS`: include tags such as `official`.
- `EXCLUDE_GAME_TAGS`: exclude tags.
- `N_PASSES`: TAAF passes per selected game.
- `CONCURRENT_JOBS`: TAAF concurrency. With Slurm local servers this is per
  GPU/server.
- `MAX_ACTIONS`: optional per-game action cap.
- `MAX_RUNTIME_MINUTES`: per-game wall-clock cap.
- `MAX_EXPERIMENT_RUNTIME_MINUTES` or `MAX_EXPERIMENT_RUNTIME_HOURS`: whole-run
  wall-clock budget. If the per-game cap is unset, the runner derives it from
  the number of games, passes, and effective concurrency.
- `EXPERIMENTS_DIR`: base directory for timestamped runs.
- `EXPERIMENT_DIR`: exact output directory for one run.

List resolved official games without running them:

```bash
uv run --no-sync inference-taaf-run --include-tags official --list-games
```

## Local vLLM

Start the server:

```bash
make server
```

Check or stop it:

```bash
make check-server
make stop-server
```

The default local base URL is `http://127.0.0.1:1234/v1`. `make server`
generates a local server API key unless `SERVER_REQUIRE_API_KEY=false`.

On cluster machines, the Makefile moves Hugging Face, Torch, Triton, and related
caches under `/shared/<user>` when that directory exists.

## Slurm Flow

`make sbatch` runs through TAAF's Slurm deployment. The job directory includes
the run config, generated Slurm script, dependency override file, benchmark
artifacts, diagnostics, and logs.

For local-vLLM Slurm runs, the solver starts local server processes inside the
allocation before the duck begins playing. Each run gets its own API key and
run-scoped localhost ports, so a run either talks to its own server or fails
fast. The server is started from the bundled `src/ARC3-Inference` snapshot in
the run directory, using the worker's per-run virtualenv.

`deployment.source_repos` is bundled into the Slurm job directory. The worker
uses a generated dependency override file so it installs those bundled repos
instead of fetching private dependencies from GitHub.

## Kaggle Flow

`make kaggle-duck` runs through TAAF's Kaggle deployment. It packages the
current TAAF and ARC3-Inference sources into a Kaggle source dataset, pushes a
private Kaggle notebook, and attaches the vLLM wheelhouse dataset plus the
version-pinned `Qwen3.8 27B FP8 Repacked` Kaggle Model. The duck solver owns the
Kaggle setup hooks, so the launcher only chooses the run shape.

By default the target uses the 25 official ARC-AGI-3 games, `model=local`, 12
concurrent games, 132 minutes per game, and a 540-minute Kaggle runtime. The
next-run safeguards use one candidate, seven tool steps, a 4,096-token
per-response cap, a 100,000-token per-game budget, and a per-level action
ceiling of twice the baseline (minimum 20) and a 75,000-token per-level
no-progress ceiling that resets after progress. The seventh and final analyzer
tool step exposes only the direct `action` tool; two no-action turns at the same
state trigger a controller-ranked fallback. Final no-action generated tokens are
included in benchmark totals. The controller ignores edge-only
HUD changes, blocks a third failed click at the same coordinate and a repeated
inverse movement cycle, treats pure object translation as a revisit, blocks a
direction used at least 12 times in a 16-action window without score or level
progress, persistently blocks it after three guard strikes, stops a level after
eight directional no-progress guards, and stops after eight consecutive
cycle-risk actions.
Its stagnation/cycle windows are 6/4. Add `DEPLOYMENT_WAIT=true` if you want the
command to block and pull the finished Kaggle output back into the run directory. Set
`KAGGLE_DUCK_DIAGNOSTIC=true` when invoking `kaggle-duck.sh` to first run the
single-game `ar25-0c556536` diagnostic run at concurrency 1 with a 900-second analyzer timeout.
The host-owned Orchestrated Objective Reduction analyzer is opt-in. Enable it
for a CPU-first diagnostic run with:

```bash
KAGGLE_DUCK_DIAGNOSTIC=true \
KAGGLE_DUCK_OBJECTIVE_REDUCTION=true \
LOCAL_GAMEPLAY_POLICY_BACKEND=cpu \
./kaggle-duck.sh
```

This keeps objective reduction and policy generation on the configured local
LLM while ordinary gameplay executes the generated policy without an LLM call.
Reducer/coder requests have an independent 300-second model timeout, so the
60-second cooperative analyzer yield is checked only between orchestration
boundaries. Thinking remains enabled; reducer and coder responses are capped at
4096 and 8192 tokens respectively, with separate reasoning budgets of 2048 and
1024 tokens. The reducer emits JSON between strict
`BEGIN_REDUCTION`/`END_REDUCTION` markers and the coder emits raw Python between
strict `BEGIN_POLICY`/`END_POLICY` markers. This avoids the vLLM native tool
grammar and prevents tool-call escaping from corrupting generated content. Each
role uses one HTTP attempt per orchestration attempt; three rejected attempts
exhaust the role visibly instead of restarting through the controller. Reducer,
coder, and policy observations receive only the exact model-facing action names
`UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, `MOUSE`, and `ACTION7`; conversion to
engine aliases happens only at the controller boundary, and `MOUSE` always
requires bounded row/column coordinates. Pre-action failure streaks are scoped
to one tactical leaf, and a policy receives one terminal-only evaluation of the
post-action observation when its action budget reaches zero.
Policy transition observations include controller-owned reward, outcome class,
novelty, cycle, no-op, stagnation, and bounded animation evidence plus a
host-computed `meaningful_progress` flag. An identical action proposed after an
`exact_noop`, `volatile_only`, guarded, or cyclic transition fails the tactical
objective before another controller call. Static policy verification reports all
detectable violations together; coder retries receive cumulative errors and the
bounded rejected source so the model can make a minimal repair rather than
regenerating blindly.
Override these with
`LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS`,
`LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT`, and
`LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT`, plus
`LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET` and
`LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET`.
`LOCAL_GAMEPLAY_POLICY_BACKEND=auto` permits CUDA only when the policy declares
support and at least `LOCAL_GAMEPLAY_POLICY_CUDA_MIN_FREE_MB` (default 4096 MiB)
is available; otherwise it falls back to CPU. Explicit `cuda` is strict.

The equivalent direct CLI form is:

```bash
uv run --no-sync inference-taaf-run \
  --deployment-target kaggle \
  --kaggle-duck-public-harness \
  --agent duck-harness \
  --model local \
  --run-name duck-harness-20260527 \
  --kaggle-kernel-slug taaf-duck-harness-20260527 \
  --kaggle-dataset-ref driessmit1/taaf-kaggle-source-duck-harness-20260527 \
  --max-runtime-minutes 132 \
  --max-experiment-runtime-minutes 540 \
  --concurrent-jobs 12 \
  --analyzer-timeout 120
```

## Run Artifacts

Each run writes a timestamped directory under `experiments.root_dir`, or under
`EXPERIMENTS_DIR` / `EXPERIMENT_DIR` when overridden.

Important files include:

- `run_config.json`: resolved games, passes, concurrency, runtime caps, model,
  Slurm settings, and hardware metadata.
- `benchmark.json`: saved TAAF benchmark and per-game `GameRun` state.
- `diagnostics.html`: TAAF diagnostics.
- `artifacts/*_viewer_data.json`: compact viewer payloads.
- `artifacts/*_events.jsonl`: append-only full viewer event sidecars.
- duck transcript HTML/text files linked from the viewer.
- `stdout.log` and `stderr.log` for Slurm jobs.
- `requests.jsonl` files when `analyzer.save_request_logs` is true.

## Viewer

Start the viewer on the default port from `configs/inference.json`:

```bash
make view
```

Override the port:

```bash
make view VIEW_PORT=8012
```

Point at a run root:

```bash
make view VIEW_RUNS_DIR=/shared/arc_3_results/$USER
```

Point at one exact run:

```bash
make view VIEW_RUN_DIR=/shared/arc_3_results/$USER/<run-name>
```

The viewer shows run summaries, per-game progress, boards, actions, rewards,
level transitions, and the duck's transcript.

## Scoring

Score one run directory:

```bash
make score_run SCORE_RUN_DIR=/path/to/run
```

Evaluate runs from `configs/eval.json`:

```bash
make eval
```

Write a score file somewhere specific:

```bash
make score_run SCORE_RUN_DIR=/path/to/run SCORE_OUTPUT_PATH=docs/candidate-score.json
```

The scorer reads TAAF `benchmark.json`, uses persisted `final_score` values when
present, and otherwise asks TAAF's `GameRun` scorer to compute the score from
the saved state. It writes `evaluation.json` plus the lightweight `score.json`
format used by significance checks.

## Significance

Compare a candidate score file against a current best:

```bash
make significance \
  BASELINE_SCORE=docs/current-best-score.json \
  CANDIDATE_SCORE=docs/candidate-score.json
```

Or configure those paths in `configs/significance.json` and run:

```bash
make significance
```

The comparison aligns by `game_id`, averages repeated trials within each game,
and uses games as the paired unit. It checks runtime budget, hardware, dataset
metadata, and trial counts before reporting whether the candidate passes the
internal-highscore threshold:

```text
P(true_delta > 0 | results) >= 0.90
```

The output also includes win rate, a bootstrap 90% interval, and TAAF paired
test p-values as robustness checks.

## Behavioral Regression Gate

Check both score and closed-loop behavior before accepting a run:

```bash
make regression-gate REGRESSION_RUN_DIR=/path/to/run
```

`configs/regression.json` requires whole-game wins and level completion in
addition to score, trace coverage, and the declared outcome-aware/candidate/
verified-planner configuration. It also verifies the independent-evidence,
process-safe-memory, and adaptive-budget controls, and records how often the
duck follows a recommended plan and whether followed plans produce progress.
Raw action traces preserve the animation and planner-decision fields consumed
by these metrics. Loop interventions and empirical harm interventions are
reported separately.
The gate caps no-ops, crashes, cancellations,
tokens per completed level, actions per completed level, and terminal-state
violations. This intentionally rejects the old score-only zero-win example as
evidence. It can also require a score ratio against `baseline_run_dir`. The
command exits nonzero and reports every failed threshold.

## Trace Export

Export machine-readable per-episode duck traces:

```bash
make traces
```

For runs outside `runs/`, call the tool directly:

```bash
uv run --no-sync inference-traces --runs-dir /shared/arc_3_results/$USER
```

Traces are written in live-chat `messages` format. They preserve assistant
reasoning, tool calls, compact tool results, actions, scores, and level
transitions linked back to message indices.

## Useful Commands

- `make install`: create `.venv` and install all locked dependencies.
- `make prepare-ci`: run Ruff and the test suite.
- `make server`: start local vLLM.
- `make interactive`: run through TAAF inline deployment.
- `make sbatch`: submit through TAAF Slurm deployment.
- `make chat PROMPT="..."`: send a direct chat probe to the configured model.
- `make view`: serve the run viewer.
- `make score_run SCORE_RUN_DIR=...`: score one saved run.
- `make eval`: score runs selected by an eval config.
- `make significance`: compare two score files.
- `make traces`: export trace JSON.
- `make zip`: zip the local `runs/` directory.
- `make clean`: remove local `runs/` artifacts.

## Repo Map

- `inference/framework/run.py`: CLI entry point and TAAF deployment setup.
- `inference/framework/solver.py`: TAAF solver adapter, action execution,
  viewer events, transcripts, and local-server orchestration.
- `inference/agent/tool_agent.py`: OpenAI-compatible tool-calling duck.
- `inference/agent/python_tool_sandbox.py`: isolated Python tool runtime.
- `inference/utils/segmentation.py`: connected-component board segmentation.
- `inference/tools/eval.py`: TAAF score export.
- `inference/tools/significance.py`: paired score comparison.
- `inference/tools/traces.py`: trace export.
- `viewer/`: local browser UI for saved runs.
- `tests/`: unit coverage for config, duck runtime, TAAF runner, viewer,
  scoring, significance, and traces.
