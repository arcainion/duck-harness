"""LLM objective reduction plus generated CPU/CUDA gameplay policies."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests

from inference.agent.action_names import to_model_action, to_model_actions
from inference.agent.gameplay_policy_runtime import (
    GameplayPolicyRuntime,
    PolicyDecision,
    PolicyObservation,
    PolicyRuntimeError,
    PolicyStatus,
    verify_policy_source,
)
from inference.agent.objective_reduction import (
    ObjectiveEvidenceMode,
    ObjectiveError,
    ObjectiveKind,
    ObjectiveNode,
    ObjectiveStatus,
    ObjectiveTree,
    ReductionProposal,
    ReductionVerdict,
    SubgoalSpec,
    TacticalExecutionMode,
)
from inference.agent.policy_solver_helpers import (
    NAVIGATION_SOLVER_FAMILIES,
    POLICY_SOLVER_TYPES,
    solver_family,
    validate_solver_config,
)
from inference.agent.policy_codegen_helpers import (
    contrastive_transition_evidence_status,
    stable_transition_evidence_status,
    transition_has_stable_change,
)
from inference.agent.runtime_state import Frame, HistoryEntry, load_runtime_state
from inference.agent.tool_agent import (
    AnalyzerTurnResult,
    ToolAgent,
    _extract_reasoning_text,
    _is_context_length_error,
    _normalize_message_content,
)


log = logging.getLogger(__name__)

_STATE_VERSION = 1
_MAX_ROLE_HISTORY = 12
_MAX_STRUCTURED_ATTEMPTS = 3
_MAX_POLICY_REPAIRS = 2
_MAX_NO_ACTION_BOUNDARIES = 8
_ACTION_FAMILY_SATURATION_ATTEMPTS = 12
_STRATEGIC_RECALIBRATION_FAILURES = 3
_DEFAULT_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS = 300.0
_DEFAULT_REDUCER_MAX_OUTPUT = 4096
_DEFAULT_CODER_MAX_OUTPUT = 8192
_DEFAULT_REDUCER_THINKING_BUDGET = 2048
_DEFAULT_CODER_THINKING_BUDGET = 1024
_REDUCTION_BEGIN_MARKER = "BEGIN_REDUCTION"
_REDUCTION_END_MARKER = "END_REDUCTION"
_POLICY_BEGIN_MARKER = "BEGIN_POLICY"
_POLICY_END_MARKER = "END_POLICY"
_MAX_REJECTED_POLICY_REPAIR_CHARS = 24000
_CONTEXT_RE = re.compile(
    r"maximum context length is\s+(\d+)\s+tokens.*?requested\s+(\d+)\s+"
    r"output tokens.*?prompt contains at least\s+(\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)
_NON_PROGRESS_OUTCOME_CLASSES = frozenset(
    {
        "exact_noop",
        "behavioral_noop",
        "volatile_only",
        "transient_effect",
        "cycle",
        "cycle_risk",
        "guarded",
        "invalid_action",
    }
)
_MODEL_ACTION_CONTRACT = {
    "UP": "move up",
    "DOWN": "move down",
    "LEFT": "move left",
    "RIGHT": "move right",
    "SPACE": "press space",
    "MOUSE": "click one board cell; integer row and col from 0 through 63 are required",
}


class OrchestrationFailure(RuntimeError):
    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


class OrchestrationYield(RuntimeError):
    pass


def _empty_orchestration_metrics() -> dict[str, int | float]:
    return {
        "reducer_calls": 0,
        "coder_calls": 0,
        "reducer_attempts": 0,
        "coder_attempts": 0,
        "reducer_timeouts": 0,
        "coder_timeouts": 0,
        "reducer_transport_failures": 0,
        "coder_transport_failures": 0,
        "reducer_model_seconds": 0.0,
        "coder_model_seconds": 0.0,
        "reducer_generated_tokens": 0,
        "coder_generated_tokens": 0,
        "policy_activations": 0,
        "policy_reuses": 0,
        "policy_reuse_rejections": 0,
        "policy_reuse_evictions": 0,
        "policy_reuse_promotions": 0,
        "policy_provisional_caches": 0,
        "role_context_adjustments": 0,
        "policy_repairs": 0,
        "policy_activation_exhaustion_recoveries": 0,
        "policy_steps": 0,
        "cuda_fallbacks": 0,
        "cpu_policy_seconds": 0.0,
        "cuda_policy_seconds": 0.0,
        "objectives_completed": 0,
        "objectives_failed": 0,
        "objective_completion_rejections": 0,
        "objective_completion_reinterpretations": 0,
        "objective_failure_rejections": 0,
        "repeated_action_repairs": 0,
        "action_family_saturation_guards": 0,
        "guard_resolved_objectives": 0,
    }


def _rejected_policy_repair_guidance(user_payload: dict[str, Any]) -> str:
    """Return repair advice scoped to the active solver contract."""

    solver_contract = user_payload.get("solver_contract")
    objective = user_payload.get("active_objective")
    solver_contract = solver_contract if isinstance(solver_contract, dict) else {}
    objective = objective if isinstance(objective, dict) else {}
    selected_type = str(solver_contract.get("selected_type") or "").strip().lower()
    selected_family = str(solver_contract.get("selected_family") or "").strip().lower()
    evidence_mode = str(objective.get("evidence_mode") or "").strip().lower()

    guidance = (
        "Do not wrap, retry, or rebuild the result of solver_decide. The decide "
        "function must contain one canonical return: return solver_decide("
        "POLICY_SOLVER_TYPE, observation, memory, POLICY_SOLVER_CONFIG). Repair "
        "behavior only through POLICY_SOLVER_CONFIG."
    )
    if selected_family == "navigation":
        return (
            guidance
            + " For a navigation policy, probe_actions control only the bounded "
            "stable/contrastive evidence phase; actor_values, target_values, "
            "passable_values, and approach_distance control route execution. If "
            "preflight reports subgoal_succeeded or that the configured target is "
            "already reached while engine progress is still required, reconsider "
            "actor_values, target_values, passable_values, approach_distance, and "
            "interaction_actions from the board evidence rather than changing only "
            "probe_actions."
        )
    if selected_type == "static" and evidence_mode == "contrastive_transition":
        return (
            guidance
            + " For contrastive static probes, separate repeated positives with a "
            "same-modality control instead of placing duplicate actions consecutively."
        )
    return guidance


def _positive_env_float(name: str, default: float) -> float | None:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning("invalid %s=%r; using %s", name, raw, default)
        value = default
    return value if value > 0 else None


def _positive_env_int(name: str, default: int) -> int | None:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning("invalid %s=%r; using %s", name, raw, default)
        value = default
    return value if value > 0 else None


def objective_reduction_enabled() -> bool:
    return os.environ.get(
        "LOCAL_ANALYZER_OBJECTIVE_REDUCTION", "false"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_REDUCER_SYSTEM_PROMPT = """You are the objective-reducer role for an ARC-AGI-3 game.
Think carefully. After thinking, your visible answer must be exactly one JSON object
between BEGIN_REDUCTION and END_REDUCTION lines. Do not use a tool call, Markdown,
or prose outside the markers. The object must contain objective_id, verdict, evidence,
rationale, selected_index, and subgoals. verdict is continue, complete, fail, or
decompose. Each subgoal contains title, success_criteria, failure_criteria,
expected_evidence, evidence_mode, execution_mode, solver_type, an integer action_budget from 1 to 32, an integer
minimum_evidence_actions from 1 through min(4, action_budget), and a boolean
single_step. evidence_mode is engine_progress, stable_transition, or
contrastive_transition. Use stable_transition only for descriptive repeatability: it
requires reproducible executed, nonvolatile, non-cyclic board changes from the same
action or coordinate but does not prove that action is uniquely causal. Use
contrastive_transition when identifying a causal control, interaction, or coordinate;
it requires the same exact positive action or coordinate to produce stable change at
least twice plus a distinct executed negative-control action or coordinate from the
same modality without a corresponding stable change: direction versus direction,
MOUSE coordinate versus MOUSE coordinate, or button versus button. Contrastive
subgoals must set action_budget at least 3,
minimum_evidence_actions at least 3, and single_step=false. Use engine_progress when
success requires meaningful_progress, reward, level_completed, or run_complete. The
execution_mode is probe, navigate, or interact. Use navigate whenever success requires
an actor/object to approach, reach, contact, merge with, or move toward a spatial
destination. Navigate policies are host-required to localize board entities, build a
passability mask, call trusted pathfinding, and replan from transition evidence. Use
probe for bounded control experiments and interact for non-routing buttons or clicks.
Do not describe a spatial route while declaring probe or interact. The
solver_type must be one registered portable solver label. Family headings such as
routing, physics, manipulation, interaction, alignment, observation, sequence/search,
coverage/transform, and field mechanics are explanatory categories, not valid
solver_type values. Choose the concrete label inside the matching category: routing
(navigation, lattice-corridor, glyph-transform-route, guided-attraction), physics
(sliding, inertial-block, trajectory-replay, gravity), manipulation
(push-pull, carrier-placement, inventory), interaction (click-interaction,
relation-toggle, switch-bridge, signal), alignment (connector-align,
linked-centroid, mirror-merge, paired-platform-alignment), coverage/transform
(beam-coverage, marker-coverage, template-paint, pattern-transform,
transform-program), sequence/search (cycle-rotation, paired-sequence-arm,
symbol-rule-sequence, peg-jump), field mechanics (beam, flow-deflector, mirror,
puzzle), multi-agent, observation (static, cellular-automata), or hybrid. A
navigate objective must select a navigation-capable family, not static,
interaction, sequence, transform, or hybrid. The
host owns all IDs,
tree invariants, budgets, and game/level completion. You may complete or fail only
the active tactical objective. A game or level objective must be decomposed into one
to six falsifiable tactical subgoals. Prefer the selected subgoal whose result will
maximally reduce uncertainty. Copy objective_id exactly from
active_objective.objective_id in the user payload; the example below illustrates only
the response shape. The selected tactical subgoal should normally be one coherent
macro objective with an 8-16 action horizon: include alternative probes, adaptation
after negative evidence, and an execution sequence so gameplay can continue without
another LLM call. The host expands non-single-step horizons shorter than eight when
level budget remains. Set single_step=true and action_budget=1 only when one action plus
its following observation conclusively resolves the entire subgoal; otherwise set
single_step=false. Set minimum_evidence_actions to the number of real post-action
observations needed to prove the declared criteria, normally 4 for exploration and 1
only for conclusive evidence. Use valid JSON with
double-quoted keys and strings. Keep reasoning compact and reserve response space for
the complete JSON object. All valid action names are already model-facing: UP, DOWN,
LEFT, RIGHT, SPACE, or MOUSE. These meanings are exact, not hypotheses.
MOUSE always requires integer row and col coordinates from 0 through 63. Never invent
or reason about hidden ACTION1 through ACTION6 aliases. A mere board change is not
necessarily progress: volatile_only and exact_noop outcomes are negative evidence.
Novel states are exploration evidence, not host-confirmed progress. Do not repeat a
tactical contract already present in objective_tree; use its resolution evidence and
select a materially different falsifiable objective.
level_action_evidence is authoritative host evidence accumulated across every tactical
objective in the current level. Treat saturated=true as a hard prohibition: do not
select a subgoal that requires that action family. High no_progress with zero stable
changes or meaningful progress is strong negative evidence even when coordinates or
object labels differ. Do not evade failed MOUSE evidence by scanning fresh coordinates.
host_control_model is authoritative, salience-weighted component-motion evidence across
the current level. When linked_high_confidence=true, select multi-agent, linked-centroid,
or paired-platform-alignment for objectives that manipulate the moving structures.
Do not fall back to static or single-actor routing merely because no engine progress has
occurred. If linked motion truly does not govern the selected objective, include a
specific control_model_override string of at least 20 characters in that selected
subgoal; the host validates this exception and otherwise rejects the mismatch.
After three consecutive failed engine_progress tactical objectives, the host requires a
contrastive_transition recalibration before accepting another execution hypothesis.
Use it to falsify the assumed control/action/coordinate mapping, not to rename the same
object-manipulation story. MOUSE interaction-learning objectives must also use
contrastive_transition; stable click repeatability alone is not causal evidence.
The host accepts engine_progress tactical success only after controller-confirmed
meaningful_progress, reward, level_completed, or run_complete. A stable_transition
tactical objective may complete on repeated host-classified stable changes after its
minimum evidence count; this resolves only that tactical learning goal, never the
engine-owned level or game objective.
A contrastive_transition objective has the same limited tactical scope and additionally
requires the host-validated negative control.

BEGIN_REDUCTION
{"objective_id":"level:1:1","verdict":"decompose","evidence":"initial board","rationale":"run a bounded adaptive probe sequence","selected_index":0,"subgoals":[{"title":"Identify a causal control action","success_criteria":"the same exact positive action produces stable change at least twice while a distinct negative-control action does not","failure_criteria":"eight adaptive probes produce no contrastive causal evidence","expected_evidence":"repeated positive transitions plus a distinct executed negative control","evidence_mode":"contrastive_transition","execution_mode":"probe","solver_type":"static","action_budget":8,"minimum_evidence_actions":4,"single_step":false}]}
END_REDUCTION"""

_CODER_SYSTEM_PROMPT = """You are the gameplay-policy coder role for an ARC-AGI-3 game.
Think about the gameplay logic compactly. After thinking, your visible answer must be
exactly one complete Python module between BEGIN_POLICY and END_POLICY lines. Do not
use a tool call, JSON, Markdown fences, prose, placeholders, or truncated functions.
The host owns and associates the active objective; do not emit objective metadata.
Preserve all Python quotes, newlines, spaces, and indentation exactly.

PolicyObservation is an immutable object, not a mapping. Read its public attributes:
observation.board, observation.level, observation.step, observation.valid_actions,
observation.last_transition, observation.objective, observation.recent_transitions,
and observation.backend. Never call observation.get(). Return mappings with the exact
PolicyDecision wire contract. Start from this runnable envelope and place decide before
all helper functions so the required entrypoint cannot be truncated:

BEGIN_POLICY
POLICY_API_VERSION = 1
SUPPORTED_BACKENDS = ("cpu",)
POLICY_REUSE_SCOPE = "none"
POLICY_SOLVER_TYPE = "static"
POLICY_SOLVER_CONFIG = {"probe_actions": ["SPACE"]}

def initialize(context):
    return {}

def decide(observation, memory):
    return solver_decide(
        POLICY_SOLVER_TYPE, observation, memory, POLICY_SOLVER_CONFIG
    )
END_POLICY

All observation.valid_actions names are already model-facing and have exact meanings:
UP, DOWN, LEFT, RIGHT, SPACE, or MOUSE. Never emit or reason about hidden
ACTION1 through ACTION6 aliases. For MOUSE, action is
{"action": "MOUSE", "row": row, "col": col}; integer row and col coordinates from
0 through 63 are always required. A terminal
decision is {"status": "subgoal_succeeded" or "subgoal_failed", "action": None,
"memory": JSON_VALUE, "evidence": TEXT}. Keep memory finite JSON data. Optional
helpers may follow decide. Allowed imports are math/statistics/collections/itertools/
functools/heapq/bisect/numpy and optional lazy torch. No files, network, subprocesses,
reflection, engine calls, or hidden state. CPU support is mandatory; CUDA is optional.
Keep the complete source small. BEGIN_POLICY and END_POLICY must each appear exactly
once on their own lines and are not part of the Python module.

The generation payload's level_action_evidence is authoritative across prior tactical
objectives in this level. Never emit an action whose entry has saturated=true. Use its
executed, no_progress, stable_changes, meaningful_progress, and distinct_points counts
to avoid repeating an exhausted action family through superficially new coordinates.
Read observation.objective["execution_mode"]. For navigate, the reachable decide path
must call solver_decide with a navigation-capable selected solver type. The trusted
dispatcher localizes the board, builds passability, plans a route, and replans after
blocking or relevant board change. Supply correct board-derived color roles rather
than hardcoded coordinates or a fixed directional sequence.

Every new policy must declare literal POLICY_SOLVER_TYPE and
POLICY_SOLVER_CONFIG values matching observation.objective["solver_type"], and its
reachable decide path must call the trusted dispatcher exactly as
solver_decide(POLICY_SOLVER_TYPE, observation, memory, POLICY_SOLVER_CONFIG).
The dispatcher returns a complete PolicyDecision-compatible mapping. Configure only
finite JSON using actor_values, target_values, passable_values, hazard_values,
interactive_values, source_values, coverage_values, interaction_actions,
probe_actions, action_sequences, approach_distance, and max_plan_length.
Routing-capable solvers also support clearance_radius from 0 through 8 for multi-cell
actors. Hybrid
requires non-hybrid fallback_types and matching fallback_configs. Sequence types
require action_sequences; static/cellular-automata require probe_actions. Routing,
momentum, and gravity require actor_values, target_values, and passable_values;
alignment/manipulation/multi-agent also require actor and passability plus a target or
interactive role; field types require source/actor, target, and passability; coverage
requires target_values. Do not
reimplement a selected solver with a fixed action loop or call another solver type.
Prefer returning the dispatcher mapping directly. If post-processing it is necessary,
propagate decision["memory"] in every rebuilt continue, mouse, or terminal decision;
passing the input memory back discards the trusted solver's probe/plan position.

Declare POLICY_REUSE_SCOPE = "tactical" only when the module is genuinely generic
across tactical objectives in the same level: it must derive actions and terminal
criteria from observation.objective, observation.board, valid_actions, and transition
evidence rather than embedding the current objective's title, fixed route, target
coordinates, or success threshold. Prefer reusable board-driven exploration and
pathfinding policies when they can honor different tactical contracts. Read
observation.objective["evidence_mode"]. For engine_progress, a local board change,
novelty, adjacency, digest change, or inferred feature merge is not sufficient: require
meaningful_progress, reward, level_completed, or run_complete. For stable_transition,
require repeated executed, nonvolatile, non-cyclic changes from the same action or
coordinate after the minimum evidence count; this completes only the tactical learning
goal and proves repeatability, not causality. For contrastive_transition, require the
same exact positive action or coordinate to produce stable change at least twice and a
distinct executed same-modality negative-control action or coordinate without a
corresponding stable change. Pair directions only with directions and MOUSE coordinates
only with MOUSE coordinates. SPACE has no distinct same-modality control. For static
contrastive probe_actions, separate repeated positives with the negative
control so synthetic exact_noop preflight observations advance through different
actions; for example, UP, RIGHT, UP, RIGHT rather than consecutive UP.
Default to POLICY_REUSE_SCOPE = "none" when uncertain. The declaration is
only a reuse candidate; the host qualifies, limits, and may evict it. The host always
supplies fresh memory/context and preflights a reused policy against the new objective;
reuse never crosses a game or level boundary.

Trusted pathfinding helpers are already available as globals; do not import or redefine
them. A passable grid is any non-empty 2D boolean-like array with at most 4096 cells;
True cells are traversable; for example, use np.equal(observation.board, FLOOR_VALUE)
to build one without a forbidden direct board comparison.
cardinal_neighbors(passable, point) returns traversable
(row, col) neighbors. shortest_path(passable, start, goal, max_expansions=4096) and
shortest_path_to_any(passable, start, goals, max_expansions=4096) return a tuple path
including both endpoints, or () when unreachable. Start and goal cells may be false in
the passable mask, but intermediate cells may not. path_to_actions(path) returns exact
UP/RIGHT/DOWN/LEFT action names. next_path_action(path, observation.valid_actions)
returns the first currently valid action or None. All searches use deterministic
four-way movement ordered UP, RIGHT, DOWN, LEFT and have a hard 4096-expansion ceiling.
Use find_cells(observation.board, VALUES, max_results=4096) for deterministic row-major
object coordinates (VALUES accepts at most 256 distinct candidates). distance_map(passable, starts, max_expansions=4096) returns a
read-only int16 array with -1 for unreachable cells; never place it in policy memory.
reachable_points(passable, start, max_distance=None, max_expansions=4096) returns a
row-major tuple. connected_components(passable, min_size=1) returns passable regions
largest-first. shortest_path_through(passable, start, ordered_waypoints,
max_expansions_per_leg=4096) joins routes through at most 32 targets, or returns ().
When hazards have different penalties, use weighted_shortest_path(passable, costs,
start, goal, max_expansions=4096) or weighted_shortest_path_to_any(...); costs must be
finite and non-negative, and path_cost(costs, path) excludes the starting cell.
clearance_mask(passable, radius=1) makes paths respect a square actor footprint (radius
0-8). grid_line(start, goal, shape=(64, 64)) returns a deterministic ray, while
line_of_sight(passable, start, goal) requires its intermediate cells to be passable.
action_destination(point, action, shape=(64, 64)) projects UP/RIGHT/DOWN/LEFT and
returns None at an edge. Helper arrays are read-only and must never enter JSON memory.
For multi-cell objects, value_mask(board, VALUES) creates a read-only mask;
component_boxes(mask, min_size=1) returns (top,left,bottom,right,size), and
component_centers(mask, min_size=1) returns a real cell nearest each box center.
approach_points(passable, targets, distance=1) finds safe interaction cells, while
shortest_approach_path(passable, start, targets, distance=1, max_expansions=4096)
routes to one without stepping onto the target. Before reusing a JSON-stored route,
call path_suffix(path, current), then path_is_valid(passable, suffix); replan if false.
PATHFINDING_API_VERSION is 1. These globals are host-owned; do not redefine them.

Prefer host decision builders over handwritten contract mappings. continue_decision(
action, memory, evidence="", prediction=None, point=None) canonicalizes one action;
mouse_decision(point, memory, ...) validates click coordinates. path_decision(path,
observation.valid_actions, memory, ...) continues or returns subgoal_failed when no next
action is valid. subgoal_succeeded(memory, evidence) and subgoal_failed(memory, evidence)
build action-free terminal decisions. transition_outcome(last_transition) returns one of
terminal/progress/guarded/failed/no_progress/unknown; transition_has_progress(...) and
transition_requires_replan(..., replan_on_no_progress=True) implement host semantics.
Before retrying, transition_repeats_nonprogress_action(last_transition, action, point=None)
detects the exact action/coordinate repeat. POLICY_CODEGEN_API_VERSION is 1.
transition_facts(last_transition) provides one bounded JSON snapshot, while
transition_change_class(last_transition) preserves host state-change classes such as
novel, revisit, volatile_only, and exact_noop independently of progress.
infer_game_state(observation) returns a bounded JSON snapshot of board palette,
component counts, foreground bounds, controls, objective budget, latest transition,
spatial symmetry/topology cues, motion summaries, and classified per-action effects.
controls["dynamics"] empirically maps directional actions to observed stable motion and
classifies the scheme as standard, inverted, rotated, remapped, or state-dependent.
Its object_motion section aggregates per-action component displacement and identifies
linked opposing or mixed multi-object controls when no global motion direction exists.
Treat its confidence as advisory: animation motion can describe world motion or scrolling.
Pass the prior result or its compact result["state_token"] as previous_state to obtain
a state_delta that distinguishes unchanged boards, candidate translations, growth,
shrinkage, topology changes, and recolor/transform changes without storing a board.
Tokens are bound to level and board shape; a mismatch returns change_type=scope_reset
instead of comparing unrelated states. A substantially present prior background remains
stable across temporary palette-dominance flips and board["background_source"] reports
whether continuity or dominant-color inference selected it.
The compact token also contains at most 32 same-color object descriptors; state_delta[
"object_changes"] deterministically reports stable, moved, resized, added, and removed
objects, including per-object doubled-center shifts. Treat objects_truncated=true as
partial evidence rather than a complete inventory.
board["object_layout"] summarizes at most 32 nearest bounding-box relations, alignment
overlaps/gaps, containment candidates, and repeated shape groups. Bounding-box contact
is only a candidate relation, not proof that irregular object cells touch.
infer_game_type(observation) ranks advisory routing, interaction, sequence, transform,
multi_agent, hybrid, or observe families with explicit confidence, evidence,
coverage, objective-alignment diagnostics, recommended registered solver types, and
ranked unresolved probes. Its control_scheme repeats the empirical directional mapping
so solver selection can avoid assuming standard controls. MOUSE probe suggestions always require a separately derived
safe coordinate. infer_game_type accepts the same optional previous_state and uses the
delta to distinguish routing-like translation from transform-like dynamics and coherent
translation from divergent multi-object motion;
the latter raises a multi_agent hypothesis and recommends registered cooperative solvers.
Use these helpers to form or check hypotheses, not as proof of engine progress or as a
replacement for the objective's declared solver.
transition_has_stable_change(last_transition) accepts only executed, board-changing,
non-cyclic, nonvolatile learning evidence. For evidence_mode=stable_transition,
call stable_transition_evidence_ready(observation.objective,
observation.recent_transitions); it applies the host's exact same-action/coordinate,
minimum-evidence, nonvolatile, and non-cyclic rules. Do not replace it with a custom
counter or group distinct directions under a generic MOVE key. It intentionally cannot
resolve a contrastive_transition contract. For evidence_mode=contrastive_transition,
call contrastive_transition_evidence_ready(observation.objective,
observation.recent_transitions); plan both repeated positive probes and at least one
distinct same-modality negative-control probe. An unrelated MOUSE no-op cannot control
for a directional effect, and vice versa. Do not use
stable_transition_evidence_ready for a causal or paired-control claim. For
evidence_mode=engine_progress, require transition_has_progress instead.
accumulate_transition_evidence(memory, last_transition, key="transition_evidence",
limit=16) stores a rolling history. objective_evidence_ready(observation.objective,
observation.recent_transitions) checks the host-requested minimum evidence count.
Use board_digest(observation.board) for a compact stable state fingerprint;
region_digest(board, (top,left,bottom,right)) uses inclusive bounds and
cells_digest(board, cells) fingerprints an order-independent coordinate set. Never
store the board itself. memory_with_defaults(memory, defaults) fills missing fields;
memory_update(memory, updates), memory_push(memory, key, value, limit=16), and
memory_increment(memory, key, amount=1, minimum=None, maximum=None) return new finite
JSON memory with 64-key, 32-KiB, and rolling-history limits. For a standalone list use
history_push(history, value, limit=16), then store the returned list; do not call
memory_push with a list as its first argument. memory_increment keys are literal and
never dotted paths. To increment memory[FIELD][KEY], use memory_mapping_increment(
memory, FIELD, KEY, amount=1, minimum=None, maximum=None).
recent_outcome_counts(observation.recent_transitions), consecutive_outcome_count(...),
and recent_action_counts(..., only_nonprogress=False) summarize at most 64 transitions.
least_tried_action(observation.valid_actions, observation.recent_transitions, exclude=(),
include_mouse=False) provides deterministic exploration without emitting an unlocated
MOUSE action. For clicks, use least_tried_mouse_point(candidates,
observation.recent_transitions, exclude=(), only_nonprogress=False,
allow_edge_hud=False), then pass its result to mouse_decision;
recent_mouse_point_counts(...) exposes JSON-safe "row,col" counts. Exclude accepts
point pairs or those canonical keys, so exclude=recent_mouse_point_counts(...).keys()
is safe. The two-cell outer HUD band is excluded by default. Set allow_edge_hud=True
only after persistent interior
evidence specifically identifies an edge control as relevant. The helper returns None
when no safe candidate remains; never pass None to mouse_decision—choose a safe
non-mouse action or return subgoal_failed instead.
Use first_matching_cell(board, values), nearest_matching_cell(board, values, origin), or
matching_region_center(board, values) instead of rewriting marker scans. Use
line_value_count(board, values, axis, index), line_run_length(..., from_end=False),
edge_value_count(board, values, edge), and edge_run_length(board, values, edge, offset)
for rows, columns, progress bars, and inward edge runs. All helpers are host-owned and
bounded. The reducer/coder payload's board_hex_rows use hexadecimal symbols
0123456789abcdef, while observation.board stores the corresponding integers 0 through
15. Convert payload symbols with palette_value("b") == 11 or
palette_values("b5") == (11, 5). Never use ord() for board values.

observation.board is a NumPy uint8 array. Never use it directly as a boolean or compare
it with == or !=; use board.size, np.array_equal, or a finite Python integer digest.
It is always non-None with shape (64, 64): never check board is None and never call
hasattr(board, "shape"); read board.shape and board.size directly.
Never store the board, a NumPy array, or a NumPy scalar in memory; memory must contain
only finite JSON values. engine_state and board_hex_rows are not observation fields.
Transition mappings expose action, row, col, executed, board_changed, reward, score,
level, engine_state, outcome_class, novel_state, decision_context_changed,
meaningful_progress, loop_detected, cycle_risk, no_op_streak, stagnation_actions,
level_completed, run_complete, game_over, stop_reason, error, objective_id, and a
bounded animation_summary. animation_summary["object_motion"] reports bounded
per-object displacement, including coherent, opposing, divergent, stationary,
ambiguous, and edge-only classifications. Read mappings with mapping.get(), never
getattr().

After issuing an action, the next decide call must inspect the resulting
last_transition for the active objective. Issuing an action is never evidence that the
subgoal succeeded. exact_noop, behavioral_noop, volatile_only, guarded, and cyclic
outcomes are not meaningful progress. A one-shot probe must return
subgoal_succeeded/subgoal_failed after inspecting its post-action transition; it must
not repeat the same non-progress action. Use a finite rolling integer digest when a
board digest is needed; hashlib/json imports, getattr, and observation.engine_state are
not permitted. The active objective exposes minimum_evidence_actions and single_step.
Unless the engine reports level/run completion, the host requires exactly that many
executed post-action observations before accepting tactical success or failure. Do not
return subgoal_succeeded from one board_changed, novel_state, or meaningful_progress
observation; continue gathering evidence until the complete declared success criteria
and evidence_mode are satisfied. volatile_only and transient_effect can never prove
tactical success. Runtime transition history is objective-scoped: at entry to a new tactical
objective observation.last_transition is None, and recent_transitions contains only
transitions from that objective. Never evaluate post-action evidence until
last_transition is a mapping produced by this objective. The host preflights a new
policy against up to four synthetic exact_noop observations: it must keep selecting a
different valid probe rather than repeat or terminate early. Unless game_over or the
action budget is exhausted, do not return subgoal_failed before
observation.objective["minimum_evidence_actions"] observations."""


def _compact_board(frame: Frame) -> list[str]:
    return [
        "".join(format(max(0, min(15, int(cell))), "x") for cell in row)
        for row in frame.grid
    ]


def _recent_transition_payload(
    history: list[HistoryEntry], limit: int = 8
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in history[-max(1, limit) :]:
        payload.append(
            {
                "action": to_model_action(entry.action),
                "reward": float(entry.reward),
                "level": entry.frame.level,
                "step": entry.frame.step,
                "engine_state": entry.frame.engine_state,
                "outcome": entry.outcome_class_override,
            }
        )
    return payload


def _model_transition_payload(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if "action" in payload:
        payload["action"] = to_model_action(payload.get("action"))
    return payload


def _finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return parsed if np.isfinite(parsed) else 0.0


def _bounded_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _animation_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in (
            "frame_count",
            "changed_frame_count",
            "total_changed_cells",
            "peak_changed_cells",
            "final_changed_cells",
            "transient_changed_cells",
            "motion_direction",
            "temporally_reversible",
            "object_motion",
        )
        if key in value
    }


def _host_control_model(
    history: list[HistoryEntry], *, level: int, limit: int = 64
) -> dict[str, Any]:
    """Aggregate bounded host-observed component motion for objective reduction."""

    allowed_classes = {
        "coherent",
        "opposing",
        "divergent",
        "stationary",
        "ambiguous",
        "edge_only",
    }
    directional = {"UP", "RIGHT", "DOWN", "LEFT"}
    evidence: dict[str, dict[str, Any]] = {}
    for entry in history[-max(1, min(256, int(limit))) :]:
        if entry.frame.level != level:
            continue
        motion = entry.animation.get("object_motion")
        if not isinstance(motion, dict) or motion.get("tracking_available") is not True:
            continue
        classification = str(motion.get("classification") or "").strip()
        if classification not in allowed_classes:
            continue
        action = to_model_action(entry.action)
        if action not in {*directional, "SPACE", "MOUSE"}:
            continue
        action_evidence = evidence.setdefault(
            action,
            {
                "classifications": {},
                "motion_confidences": {},
                "shift_sets": {},
                "decisive_opposing_samples": 0,
            },
        )
        classifications = action_evidence["classifications"]
        classifications[classification] = classifications.get(classification, 0) + 1
        motion_confidence = str(motion.get("confidence") or "unknown").strip().lower()
        if motion_confidence not in {"low", "medium", "high"}:
            motion_confidence = "unknown"
        motion_confidences = action_evidence["motion_confidences"]
        motion_confidences[motion_confidence] = (
            motion_confidences.get(motion_confidence, 0) + 1
        )
        raw_shifts = motion.get("salient_distinct_shifts_twice")
        if not isinstance(raw_shifts, list):
            raw_shifts = motion.get("distinct_shifts_twice")
        shift_set: tuple[tuple[int, int], ...] = ()
        if isinstance(raw_shifts, list):
            try:
                shift_set = tuple(
                    sorted(
                        (int(item[0]), int(item[1]))
                        for item in raw_shifts[:8]
                        if isinstance(item, (list, tuple)) and len(item) == 2
                    )
                )
            except (TypeError, ValueError, OverflowError):
                shift_set = ()
        shift_sets = action_evidence["shift_sets"]
        shift_sets[shift_set] = shift_sets.get(shift_set, 0) + 1
        try:
            salient_moved_count = int(motion.get("salient_moved_count"))
        except (TypeError, ValueError, OverflowError):
            salient_moved_count = len(shift_set)
        if (
            action in directional
            and classification == "opposing"
            and motion_confidence == "high"
            and salient_moved_count >= 2
        ):
            action_evidence["decisive_opposing_samples"] += 1

    by_action: dict[str, dict[str, Any]] = {}
    totals: dict[str, int] = {}
    directional_samples = 0
    decisive_opposing_samples = 0
    for action in ("UP", "RIGHT", "DOWN", "LEFT", "SPACE", "MOUSE"):
        action_evidence = evidence.get(action)
        if action_evidence is None:
            continue
        classifications = action_evidence["classifications"]
        samples = sum(int(count) for count in classifications.values())
        if action in directional:
            directional_samples += samples
            decisive_opposing_samples += int(
                action_evidence["decisive_opposing_samples"]
            )
        for classification, count in classifications.items():
            totals[classification] = totals.get(classification, 0) + int(count)
        dominant, dominant_count = sorted(
            classifications.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )[0]
        shift_sets = sorted(
            action_evidence["shift_sets"].items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:4]
        by_action[action] = {
            "samples": samples,
            "classifications": dict(sorted(classifications.items())),
            "motion_confidences": dict(
                sorted(action_evidence["motion_confidences"].items())
            ),
            "decisive_opposing_samples": int(
                action_evidence["decisive_opposing_samples"]
            ),
            "dominant_classification": str(dominant),
            "consistency": round(float(dominant_count) / float(samples), 4),
            "salient_shift_sets_twice": [
                {
                    "shifts": [list(shift) for shift in shift_set],
                    "count": int(count),
                }
                for shift_set, count in shift_sets
            ],
        }

    opposing = int(totals.get("opposing", 0))
    coherent = int(totals.get("coherent", 0))
    divergent = int(totals.get("divergent", 0))
    if opposing and coherent:
        scheme = "linked_mixed"
    elif opposing:
        scheme = "linked_opposing"
    elif divergent:
        scheme = "divergent_multi_object"
    elif coherent:
        scheme = "coherent"
    elif totals:
        scheme = "stationary_or_ambiguous"
    else:
        scheme = "unknown"
    repeated_opposing_evidence = opposing >= 2 and directional_samples >= 4
    confidence = (
        "high"
        if decisive_opposing_samples >= 1 or repeated_opposing_evidence
        else "medium"
        if opposing or divergent >= 2 or coherent >= 2
        else "low"
    )
    linked_high_confidence = confidence == "high" and scheme in {
        "linked_mixed",
        "linked_opposing",
    }
    ineffective_actions = [
        action
        for action, item in by_action.items()
        if item["dominant_classification"] in {"stationary", "edge_only"}
        and float(item["consistency"]) >= 0.75
    ]
    return {
        "schema_version": 2,
        "scheme": scheme,
        "confidence": confidence,
        "confidence_reason": (
            "decisive_high_confidence_opposing_motion"
            if decisive_opposing_samples >= 1
            else "repeated_opposing_motion"
            if repeated_opposing_evidence
            else "mixed_or_repeated_motion_evidence"
            if confidence == "medium"
            else "insufficient_motion_evidence"
        ),
        "linked_high_confidence": linked_high_confidence,
        "decisive_opposing_samples": decisive_opposing_samples,
        "samples": sum(int(count) for count in totals.values()),
        "directional_samples": directional_samples,
        "classifications": dict(sorted(totals.items())),
        "by_action": by_action,
        "ineffective_actions": ineffective_actions,
        "recommended_solver_types": (
            ["multi-agent", "linked-centroid", "paired-platform-alignment"]
            if linked_high_confidence
            else ["navigation", "guided-attraction"]
            if scheme == "coherent"
            else []
        ),
        "selection_constraint": (
            "Prefer a linked multi-object or alignment solver. Generic static and "
            "single-actor routing require control_model_override in the selected subgoal."
            if linked_high_confidence
            else "Advisory until additional repeatable motion evidence is observed."
        ),
    }


def _meaningful_progress(result: dict[str, Any]) -> bool:
    if not result.get("executed") or result.get("error"):
        return False
    outcome = str(result.get("outcome_class") or "").strip().lower()
    if (
        result.get("loop_detected")
        or result.get("cycle_risk")
        or outcome in _NON_PROGRESS_OUTCOME_CLASSES
    ):
        return False
    if (
        result.get("level_completed")
        or result.get("run_complete")
        or _finite_float(result.get("reward")) > 0
    ):
        return True
    # The controller owns progress classification. In particular, a novel board
    # state can still be an unproductive translation or animation. Older
    # controllers that omit the explicit field are treated conservatively; the
    # engine-owned terminal/reward checks above remain sufficient for real progress.
    explicit = result.get("meaningful_progress")
    if isinstance(explicit, bool):
        return explicit
    return False


_TACTICAL_CONTRACT_STOPWORDS = frozenset(
    {
        "a",
        "after",
        "and",
        "at",
        "before",
        "by",
        "for",
        "from",
        "if",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "the",
        "then",
        "to",
        "until",
        "using",
        "while",
        "with",
        "within",
    }
)

_TACTICAL_ACTION_TERMS = {
    "click": "MOUSE",
    "clicked": "MOUSE",
    "clicking": "MOUSE",
    "clicks": "MOUSE",
    "cursor": "MOUSE",
    "down": "DOWN",
    "downward": "DOWN",
    "downwards": "DOWN",
    "left": "LEFT",
    "leftward": "LEFT",
    "leftwards": "LEFT",
    "mouse": "MOUSE",
    "right": "RIGHT",
    "rightward": "RIGHT",
    "rightwards": "RIGHT",
    "space": "SPACE",
    "up": "UP",
    "upward": "UP",
    "upwards": "UP",
}


def _tactical_contract_terms(value: Any) -> frozenset[str]:
    """Return deterministic salient terms for host-side anti-churn checks."""

    if isinstance(value, SubgoalSpec):
        text = " ".join(
            (value.title, value.success_criteria, value.expected_evidence)
        )
    else:
        text = " ".join(
            str(getattr(value, name, "") or "")
            for name in ("title", "success_criteria", "expected_evidence")
        )
    return frozenset(
        token
        for token in re.findall(r"[a-z]+|\d+", text.lower())
        if len(token) > 1 and token not in _TACTICAL_CONTRACT_STOPWORDS
    )


def _tactical_title_terms(value: Any) -> frozenset[str]:
    """Return target-bearing title terms for anti-churn identity checks."""

    text = value.title if isinstance(value, SubgoalSpec) else getattr(value, "title", "")
    return frozenset(
        token
        for token in re.findall(r"[a-z]+|\d+", str(text or "").lower())
        if len(token) > 1 and token not in _TACTICAL_CONTRACT_STOPWORDS
    )


def _tactical_title_actions(value: Any) -> frozenset[str]:
    """Return canonical controls explicitly named by a tactical title."""

    text = value.title if isinstance(value, SubgoalSpec) else getattr(value, "title", "")
    return frozenset(
        action
        for token in re.findall(r"[a-z]+", str(text or "").lower())
        if (action := _TACTICAL_ACTION_TERMS.get(token)) is not None
    )


def _tactical_contract_similarity(left: Any, right: Any) -> float:
    """Measure salient-term containment for deterministic paraphrase detection."""

    left_terms = _tactical_contract_terms(left)
    right_terms = _tactical_contract_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    overlap = len(left_terms & right_terms)
    if overlap < 6:
        return 0.0
    return overlap / min(len(left_terms), len(right_terms))


def _tactical_title_similarity(left: Any, right: Any) -> float:
    """Measure whether two contracts name the same action or spatial target."""

    left_terms = _tactical_title_terms(left)
    right_terms = _tactical_title_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    overlap = len(left_terms & right_terms)
    if overlap < 2:
        return 0.0
    return overlap / min(len(left_terms), len(right_terms))


def _contract_requests_mouse(spec: SubgoalSpec) -> bool:
    """Return whether a tactical contract explicitly depends on mouse input."""

    text = " ".join(
        (spec.title, spec.success_criteria, spec.expected_evidence)
    ).lower()
    terms = set(re.findall(r"[a-z]+", text))
    return bool(terms & {"click", "clicks", "cursor", "mouse"})


def _contract_requires_navigation(spec: SubgoalSpec) -> bool:
    """Identify contracts whose success explicitly depends on spatial routing."""

    text = " ".join(
        (spec.title, spec.success_criteria, spec.expected_evidence)
    ).lower()
    terms = set(re.findall(r"[a-z]+", text))
    if terms & {"navigate", "pathfind", "pathfinding", "route"}:
        return True
    if terms & {"approach", "contact", "merge", "reach"}:
        return True
    return bool(
        terms & {"drive", "move"}
        and terms
        & {
            "boundary",
            "column",
            "destination",
            "goal",
            "row",
            "target",
            "toward",
            "wall",
        }
    )


def _reachable_policy_calls(source: str, target: str) -> tuple[ast.Call, ...]:
    """Return calls to ``target`` reachable through generated entrypoints."""

    tree = ast.parse(source, mode="exec")
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = [name for name in ("decide", "initialize") if name in functions]
    visited: set[str] = set()
    matches: list[ast.Call] = []
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            called = node.func.id
            if called == target:
                matches.append(node)
            if called in functions and called not in visited:
                pending.append(called)
    return tuple(matches)


def _valid_solver_dispatch_call(call: ast.Call) -> bool:
    if call.keywords or len(call.args) != 4:
        return False
    expected_names = (
        "POLICY_SOLVER_TYPE",
        "observation",
        "memory",
        "POLICY_SOLVER_CONFIG",
    )
    return all(
        isinstance(argument, ast.Name) and argument.id == expected
        for argument, expected in zip(call.args, expected_names)
    )


def _validate_solver_declaration_usage(source: str) -> None:
    """Keep validated solver declarations immutable and single-purpose."""

    tree = ast.parse(source, mode="exec")
    declaration_names = {"POLICY_SOLVER_TYPE", "POLICY_SOLVER_CONFIG"}
    allowed_stores: set[int] = set()
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in declaration_names:
                allowed_stores.add(id(target))
    allowed_loads: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "solver_decide" or len(node.args) != 4:
            continue
        for index in (0, 3):
            argument = node.args[index]
            if isinstance(argument, ast.Name):
                allowed_loads.add(id(argument))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id not in declaration_names:
            continue
        if isinstance(node.ctx, ast.Store) and id(node) not in allowed_stores:
            raise PolicyRuntimeError(
                f"policy {node.id} may not be reassigned after declaration",
                category="policy_solver_contract",
            )
        if isinstance(node.ctx, ast.Load) and id(node) not in allowed_loads:
            raise PolicyRuntimeError(
                f"policy {node.id} may only be read as a solver_decide argument",
                category="policy_solver_contract",
            )


def _action_family_saturation_reason(
    evidence: dict[str, dict[str, Any]], action: Any
) -> str:
    """Explain when an action family has exhausted useful level evidence."""

    action_name = str(action or "").strip().upper()
    if action_name != "MOUSE":
        return ""
    stats = evidence.get(action_name)
    if not isinstance(stats, dict):
        return ""
    executed = max(0, int(stats.get("executed", 0) or 0))
    no_progress = max(0, int(stats.get("no_progress", 0) or 0))
    stable_changes = max(0, int(stats.get("stable_changes", 0) or 0))
    meaningful_progress = max(0, int(stats.get("meaningful_progress", 0) or 0))
    if (
        executed < _ACTION_FAMILY_SATURATION_ATTEMPTS
        or no_progress < _ACTION_FAMILY_SATURATION_ATTEMPTS
        or stable_changes > 0
        or meaningful_progress > 0
    ):
        return ""
    distinct_points = stats.get("distinct_points")
    point_count = len(distinct_points) if isinstance(distinct_points, list) else 0
    return (
        f"MOUSE is saturated for this level after {executed} executed probes "
        f"at {point_count} distinct coordinates produced no stable change or "
        "controller-confirmed progress"
    )


def _failed_engine_progress_since_recalibration(tree: ObjectiveTree) -> int:
    """Count failed execution hypotheses since grounded progress or recalibration."""

    parent_id = tree.current_level_objective.objective_id
    siblings = [
        node
        for node in tree.nodes.values()
        if node.kind is ObjectiveKind.TACTICAL and node.parent_id == parent_id
    ]
    count = 0
    for node in reversed(siblings):
        if node.status is ObjectiveStatus.COMPLETED and node.evidence_mode in {
            ObjectiveEvidenceMode.ENGINE_PROGRESS,
            ObjectiveEvidenceMode.CONTRASTIVE_TRANSITION,
        }:
            break
        if (
            node.status is ObjectiveStatus.FAILED
            and node.evidence_mode is ObjectiveEvidenceMode.ENGINE_PROGRESS
        ):
            count += 1
    return count


def _equivalent_attempted_tactical(
    tree: ObjectiveTree, spec: SubgoalSpec, *, threshold: float = 0.72
) -> ObjectiveNode | None:
    """Find an already-attempted sibling with an equivalent tactical contract."""

    parent_id = tree.active_id
    for node in tree.nodes.values():
        if (
            node.kind is not ObjectiveKind.TACTICAL
            or node.parent_id != parent_id
            or (node.attempts <= 0 and node.actions_used <= 0)
        ):
            continue
        if (
            node.evidence_mode is not spec.evidence_mode
            or node.execution_mode is not spec.execution_mode
            or node.solver_type is not spec.solver_type
        ):
            continue
        node_actions = _tactical_title_actions(node)
        spec_actions = _tactical_title_actions(spec)
        if node_actions and spec_actions and node_actions.isdisjoint(spec_actions):
            continue
        if (
            _tactical_contract_similarity(node, spec) >= threshold
            and _tactical_title_similarity(node, spec) >= 0.5
        ):
            return node
    return None


def _objective_contract_hash(objective: Any) -> str:
    """Return a stable hash of the tactical contract, excluding runtime counters."""

    payload = objective.to_dict() if hasattr(objective, "to_dict") else objective
    if not isinstance(payload, dict):
        return ""
    material = {
        key: re.sub(r"\s+", " ", str(payload.get(key) or "")).strip().lower()
        for key in (
            "title",
            "success_criteria",
            "failure_criteria",
            "expected_evidence",
        )
    }
    material["single_step"] = bool(payload.get("single_step"))
    material["evidence_mode"] = str(payload.get("evidence_mode") or "engine_progress")
    material["execution_mode"] = str(payload.get("execution_mode") or "probe")
    material["solver_type"] = str(payload.get("solver_type") or "hybrid")
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(encoded.encode("utf-8"), digest_size=12).hexdigest()


def _context_error_limits(exc: BaseException) -> tuple[int, int, int] | None:
    """Extract server context, requested output, and measured input token counts."""

    match = _CONTEXT_RE.search(str(exc))
    if match is None:
        return None
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _policy_transition_payload(
    action_payload: dict[str, Any],
    result: dict[str, Any],
    *,
    objective_id: str,
    policy_hash: str,
) -> dict[str, Any]:
    return {
        "action": action_payload.get("action"),
        "row": action_payload.get("row"),
        "col": action_payload.get("col"),
        "executed": bool(result.get("executed")),
        "post_action_observed": True,
        "board_changed": bool(result.get("board_changed")),
        "reward": _finite_float(result.get("reward")),
        "score": _bounded_int(result.get("score")),
        "level": _bounded_int(result.get("level")),
        "engine_state": str(result.get("state") or result.get("engine_state") or ""),
        "outcome_class": str(result.get("outcome_class") or ""),
        "novel_state": bool(result.get("novel_state")),
        "decision_context_changed": bool(result.get("decision_context_changed")),
        "meaningful_progress": _meaningful_progress(result),
        "loop_detected": bool(result.get("loop_detected")),
        "cycle_risk": bool(result.get("cycle_risk")),
        "cycle_period": result.get("cycle_period"),
        "no_op_streak": _bounded_int(
            result.get("behavioral_no_op_streak", result.get("no_op_streak"))
        ),
        "stagnation_actions": _bounded_int(result.get("stagnation_actions")),
        "animation_summary": _animation_summary(result.get("animation")),
        "level_completed": bool(result.get("level_completed")),
        "run_complete": bool(result.get("run_complete")),
        "game_over": bool(result.get("game_over")),
        "stop_reason": str(result.get("stop_reason") or ""),
        "error": str(result.get("error") or ""),
        "objective_id": objective_id,
        "policy_hash": policy_hash,
    }


def _action_signature(value: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (value.get("action"), value.get("row"), value.get("col"))


def _repeats_non_progress_action(
    previous: dict[str, Any] | None,
    action: dict[str, Any],
    *,
    objective_id: str,
) -> bool:
    if not isinstance(previous, dict) or previous.get("objective_id") != objective_id:
        return False
    outcome = str(previous.get("outcome_class") or "").strip().lower()
    non_progress = (
        outcome in _NON_PROGRESS_OUTCOME_CLASSES
        or bool(previous.get("loop_detected"))
        or bool(previous.get("cycle_risk"))
    )
    return non_progress and _action_signature(previous) == _action_signature(action)


def _content_between_markers(
    message: dict[str, Any], *, begin: str, end: str, label: str
) -> str:
    if message.get("tool_calls"):
        raise ValueError(f"{label} must return raw content, not a tool call")
    content = _normalize_message_content(message.get("content", "")).strip()
    if content.count(begin) != 1:
        raise ValueError(f"{label} response must contain exactly one {begin} marker")
    if content.count(end) != 1:
        raise ValueError(f"{label} response must contain exactly one {end} marker")
    for newline in ("\r\n", "\n"):
        prefix = f"{begin}{newline}"
        suffix = f"{newline}{end}"
        if content.startswith(prefix) and content.endswith(suffix):
            body = content[len(prefix) : -len(suffix)]
            if not body.strip():
                raise ValueError(f"{label} response contained an empty body")
            return body
    raise ValueError(
        f"{label} response must contain only {begin}, a newline, the body, "
        f"a newline, and {end}"
    )


def _reduction_from_message(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("tool_calls"):
        raise ValueError("reducer must return raw content, not a tool call")
    content = _normalize_message_content(message.get("content", "")).strip()
    if _REDUCTION_BEGIN_MARKER not in content and _REDUCTION_END_MARKER not in content:
        decoder = json.JSONDecoder()
        decoded, end_index = decoder.raw_decode(content)
        if content[end_index:].strip():
            raise ValueError(
                "bare reducer JSON fallback may not contain surrounding prose"
            )
        if not isinstance(decoded, dict):
            raise ValueError("bare reducer response must be a JSON object")
        return decoded
    body = _content_between_markers(
        message,
        begin=_REDUCTION_BEGIN_MARKER,
        end=_REDUCTION_END_MARKER,
        label="reducer",
    )
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise ValueError("reducer response body must be a JSON object")
    return decoded


def _policy_source_from_message(message: dict[str, Any]) -> str:
    return _content_between_markers(
        message,
        begin=_POLICY_BEGIN_MARKER,
        end=_POLICY_END_MARKER,
        label="coder",
    )


def _policy_reuse_scope_from_source(source: str) -> str:
    """Read the opt-in reuse declaration without executing generated code."""

    try:
        module = ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return "none"
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else []
        if isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "POLICY_REUSE_SCOPE"
            for target in targets
        ):
            continue
        value_node = statement.value
        if isinstance(value_node, ast.Constant) and value_node.value in {
            "none",
            "tactical",
        }:
            return str(value_node.value)
        return "none"
    return "none"


def _literal_policy_assignment(source: str, name: str) -> Any:
    """Read one top-level generated-policy literal without executing source."""

    module = ast.parse(source, mode="exec")
    matches: list[ast.AST] = []
    for statement in module.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            if statement.value is not None:
                matches.append(statement.value)
    if len(matches) != 1:
        raise PolicyRuntimeError(
            f"policy must declare exactly one literal {name}",
            category="policy_solver_contract",
        )
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError, SyntaxError) as exc:
        raise PolicyRuntimeError(
            f"policy {name} must be a literal value",
            category="policy_solver_contract",
        ) from exc


def _without_policy_source(entry: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(entry)
    content = sanitized.get("content")
    if not isinstance(content, str):
        return sanitized
    if (
        content.count(_POLICY_BEGIN_MARKER) == 1
        and content.count(_POLICY_END_MARKER) == 1
    ):
        sanitized["content"] = "<policy source stored as a hashed artifact>"
        return sanitized
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return sanitized
    if not isinstance(payload, dict) or "source" not in payload:
        return sanitized
    payload.pop("source", None)
    payload["source_artifact"] = "stored_by_hash"
    sanitized["content"] = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )
    return sanitized


class OrchestratedObjectiveAgent(ToolAgent):
    """Analyzer that calls the LLM only at objective boundaries and failures."""

    disable_controller_fallback = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tree: ObjectiveTree | None = None
        self._role_histories: dict[str, list[dict[str, Any]]] = {
            "reducer": [],
            "coder": [],
        }
        self._policy_runtime: GameplayPolicyRuntime | None = None
        self._policy_objective_id = ""
        self._policy_source_hash = ""
        self._policy_artifact = ""
        self._rejected_policy_objective_id = ""
        self._rejected_policy_source_hash = ""
        self._rejected_policy_artifact = ""
        self._policy_solver_type = ""
        self._policy_solver_family = ""
        self._active_policy_reuse_scope = "none"
        self._reusable_policies: list[dict[str, Any]] = []
        self._policy_memory: Any = {}
        self._policy_repairs: dict[str, int] = {}
        self._premature_completion_repairs: dict[str, int] = {}
        self._premature_failure_repairs: dict[str, int] = {}
        self._repeated_action_repairs: dict[str, int] = {}
        self._policy_executed_action = False
        self._policy_observed_host_progress = False
        self._policy_was_reused = False
        self._boundary_reason = "game_start"
        self._reduction_required = True
        self._last_transition: dict[str, Any] | None = None
        self._recent_transitions: list[dict[str, Any]] = []
        self._level_action_evidence: dict[str, dict[str, Any]] = {}
        self._action_evidence_level = 0
        self._consecutive_activation_failures = 0
        self._failure_streak_objective_id = ""
        self._last_reduction_step = 0
        self._level_action_status: tuple[int, int, int] | None = None
        self._orchestration_metrics = _empty_orchestration_metrics()
        self._current_transcript_path: Path | None = None
        self._orchestration_request_timeout_seconds = _positive_env_float(
            "LOCAL_ANALYZER_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS",
            _DEFAULT_ORCHESTRATION_REQUEST_TIMEOUT_SECONDS,
        )
        self._orchestration_role_max_output = {
            "reducer": _positive_env_int(
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_MAX_OUTPUT",
                _DEFAULT_REDUCER_MAX_OUTPUT,
            ),
            "coder": _positive_env_int(
                "LOCAL_ANALYZER_ORCHESTRATION_CODER_MAX_OUTPUT",
                _DEFAULT_CODER_MAX_OUTPUT,
            ),
        }
        self._orchestration_role_thinking_budget = {
            "reducer": _positive_env_int(
                "LOCAL_ANALYZER_ORCHESTRATION_REDUCER_THINKING_BUDGET",
                _DEFAULT_REDUCER_THINKING_BUDGET,
            ),
            "coder": _positive_env_int(
                "LOCAL_ANALYZER_ORCHESTRATION_CODER_THINKING_BUDGET",
                _DEFAULT_CODER_THINKING_BUDGET,
            ),
        }

    def set_level_action_status(self, level: int, used: int, limit: int) -> None:
        """Receive the controller's authoritative current-level action counters."""

        checked_level = max(1, int(level))
        checked_used = max(0, int(used))
        checked_limit = max(1, int(limit))
        self._level_action_status = (
            checked_level,
            min(checked_used, checked_limit),
            checked_limit,
        )

    def _record_level_action_evidence(self, transition: dict[str, Any]) -> None:
        """Accumulate bounded host-classified action evidence across objectives."""

        if transition.get("executed") is not True:
            return
        action = str(transition.get("action") or "").strip().upper()
        if action not in _MODEL_ACTION_CONTRACT:
            return
        stats = self._level_action_evidence.setdefault(
            action,
            {
                "executed": 0,
                "no_progress": 0,
                "stable_changes": 0,
                "meaningful_progress": 0,
                "distinct_points": [],
            },
        )
        stats["executed"] = min(4096, int(stats.get("executed", 0)) + 1)
        stable = transition_has_stable_change(transition)
        progress = bool(transition.get("meaningful_progress"))
        if stable:
            stats["stable_changes"] = min(
                4096, int(stats.get("stable_changes", 0)) + 1
            )
        if progress:
            stats["meaningful_progress"] = min(
                4096, int(stats.get("meaningful_progress", 0)) + 1
            )
        if not stable and not progress:
            stats["no_progress"] = min(
                4096, int(stats.get("no_progress", 0)) + 1
            )
        if action == "MOUSE":
            row = transition.get("row")
            col = transition.get("col")
            if isinstance(row, int) and isinstance(col, int):
                key = f"{row},{col}"
                points = stats.setdefault("distinct_points", [])
                if isinstance(points, list) and key not in points:
                    points.append(key)
                    del points[:-64]

    def _level_action_evidence_payload(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-safe summary with deterministic saturation state."""

        payload: dict[str, dict[str, Any]] = {}
        for action in sorted(self._level_action_evidence):
            stats = self._level_action_evidence[action]
            item = {
                "executed": max(0, int(stats.get("executed", 0) or 0)),
                "no_progress": max(0, int(stats.get("no_progress", 0) or 0)),
                "stable_changes": max(
                    0, int(stats.get("stable_changes", 0) or 0)
                ),
                "meaningful_progress": max(
                    0, int(stats.get("meaningful_progress", 0) or 0)
                ),
                "distinct_points": list(stats.get("distinct_points") or [])[-64:],
            }
            item["saturated"] = bool(
                _action_family_saturation_reason(self._level_action_evidence, action)
            )
            payload[action] = item
        return payload

    def close(self) -> None:
        if self._policy_runtime is not None:
            self._policy_runtime.close()
            self._policy_runtime = None
        super().close()

    def _durable_state_path(self) -> Path | None:
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None:
            return None
        return runtime_path.with_name(f"{runtime_path.stem}_objective_state.json")

    def _reset_orchestration_state(self) -> None:
        if self._policy_runtime is not None:
            self._policy_runtime.close()
        self._policy_runtime = None
        self._tree = None
        self._role_histories = {"reducer": [], "coder": []}
        self._policy_objective_id = ""
        self._policy_source_hash = ""
        self._policy_artifact = ""
        self._rejected_policy_objective_id = ""
        self._rejected_policy_source_hash = ""
        self._rejected_policy_artifact = ""
        self._policy_solver_type = ""
        self._policy_solver_family = ""
        self._active_policy_reuse_scope = "none"
        self._reusable_policies = []
        self._policy_memory = {}
        self._policy_repairs = {}
        self._premature_completion_repairs = {}
        self._premature_failure_repairs = {}
        self._repeated_action_repairs = {}
        self._policy_executed_action = False
        self._policy_observed_host_progress = False
        self._policy_was_reused = False
        self._boundary_reason = "game_start"
        self._reduction_required = True
        self._last_transition = None
        self._recent_transitions = []
        self._level_action_evidence = {}
        self._action_evidence_level = 0
        self._consecutive_activation_failures = 0
        self._failure_streak_objective_id = ""
        self._last_reduction_step = 0
        self._orchestration_metrics = _empty_orchestration_metrics()

    def _load_durable_state(self) -> None:
        self._reset_orchestration_state()
        path = self._durable_state_path()
        if path is None or not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("version") != _STATE_VERSION
            ):
                raise ValueError("unsupported orchestrated objective state")
            if str(payload.get("game_id") or "") != self._knowledge_game_id:
                raise ValueError("orchestration state belongs to another game")
            raw_tree = payload.get("objective_tree")
            if isinstance(raw_tree, dict):
                self._tree = ObjectiveTree.from_dict(raw_tree)
            histories = payload.get("role_histories")
            if isinstance(histories, dict):
                for role in ("reducer", "coder"):
                    raw = histories.get(role)
                    if isinstance(raw, list):
                        self._role_histories[role] = [
                            dict(item)
                            for item in raw[-_MAX_ROLE_HISTORY:]
                            if isinstance(item, dict)
                        ]
            self._policy_objective_id = str(payload.get("policy_objective_id") or "")
            self._policy_source_hash = str(payload.get("policy_source_hash") or "")
            self._policy_artifact = str(payload.get("policy_artifact") or "")
            self._rejected_policy_objective_id = str(
                payload.get("rejected_policy_objective_id") or ""
            )
            self._rejected_policy_source_hash = str(
                payload.get("rejected_policy_source_hash") or ""
            )
            self._rejected_policy_artifact = str(
                payload.get("rejected_policy_artifact") or ""
            )
            self._policy_solver_type = str(payload.get("policy_solver_type") or "")
            self._policy_solver_family = str(payload.get("policy_solver_family") or "")
            if self._policy_solver_type:
                try:
                    self._policy_solver_family = solver_family(self._policy_solver_type)
                except ValueError:
                    self._policy_solver_type = ""
                    self._policy_solver_family = ""
            else:
                self._policy_solver_family = ""
            self._active_policy_reuse_scope = str(
                payload.get("active_policy_reuse_scope") or "none"
            )
            raw_reusable = payload.get("reusable_policies")
            if isinstance(raw_reusable, list):
                self._reusable_policies = [
                    dict(item)
                    for item in raw_reusable[-4:]
                    if isinstance(item, dict)
                    and str(item.get("game_id") or "") == self._knowledge_game_id
                ]
            self._policy_memory = payload.get("policy_memory", {})
            self._policy_repairs = {
                str(key): max(0, int(value or 0))
                for key, value in (payload.get("policy_repairs") or {}).items()
            }
            self._premature_completion_repairs = {
                str(key): max(0, int(value or 0))
                for key, value in (
                    payload.get("premature_completion_repairs") or {}
                ).items()
            }
            self._premature_failure_repairs = {
                str(key): max(0, int(value or 0))
                for key, value in (
                    payload.get("premature_failure_repairs") or {}
                ).items()
            }
            self._repeated_action_repairs = {
                str(key): max(0, int(value or 0))
                for key, value in (payload.get("repeated_action_repairs") or {}).items()
            }
            self._policy_executed_action = bool(
                payload.get("policy_executed_action", False)
            )
            self._policy_observed_host_progress = bool(
                payload.get("policy_observed_host_progress", False)
            )
            self._policy_was_reused = bool(payload.get("policy_was_reused", False))
            self._boundary_reason = str(payload.get("boundary_reason") or "resume")
            self._reduction_required = bool(payload.get("reduction_required", False))
            self._last_transition = (
                _model_transition_payload(payload["last_transition"])
                if isinstance(payload.get("last_transition"), dict)
                else None
            )
            self._recent_transitions = [
                _model_transition_payload(item)
                for item in payload.get("recent_transitions") or []
                if isinstance(item, dict)
            ][-8:]
            self._action_evidence_level = max(
                0, int(payload.get("action_evidence_level", 0) or 0)
            )
            raw_action_evidence = payload.get("level_action_evidence")
            if isinstance(raw_action_evidence, dict):
                for action, raw_stats in raw_action_evidence.items():
                    action_name = str(action).strip().upper()
                    if (
                        action_name not in _MODEL_ACTION_CONTRACT
                        or not isinstance(raw_stats, dict)
                    ):
                        continue
                    points = raw_stats.get("distinct_points")
                    self._level_action_evidence[action_name] = {
                        "executed": min(
                            4096, max(0, int(raw_stats.get("executed", 0) or 0))
                        ),
                        "no_progress": min(
                            4096,
                            max(0, int(raw_stats.get("no_progress", 0) or 0)),
                        ),
                        "stable_changes": min(
                            4096,
                            max(0, int(raw_stats.get("stable_changes", 0) or 0)),
                        ),
                        "meaningful_progress": min(
                            4096,
                            max(
                                0,
                                int(raw_stats.get("meaningful_progress", 0) or 0),
                            ),
                        ),
                        "distinct_points": [
                            str(point)[:16]
                            for point in (points if isinstance(points, list) else [])
                            if isinstance(point, str)
                        ][-64:],
                    }
            self._consecutive_activation_failures = max(
                0, int(payload.get("consecutive_activation_failures", 0) or 0)
            )
            self._failure_streak_objective_id = str(
                payload.get("failure_streak_objective_id") or ""
            )
            self._last_reduction_step = max(
                0, int(payload.get("last_reduction_step", 0) or 0)
            )
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                self._orchestration_metrics.update(
                    {
                        str(key): value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                )
            self._session_total_tokens = max(
                0, int(payload.get("total_tokens", 0) or 0)
            )
            self._session_generated_tokens = max(
                0, int(payload.get("generated_tokens", 0) or 0)
            )
        except (
            OSError,
            TypeError,
            ValueError,
            ObjectiveError,
            json.JSONDecodeError,
        ) as exc:
            log.warning("orchestrated objective state ignored at %s: %s", path, exc)
            self._reset_orchestration_state()

    def _persist_durable_state(self) -> None:
        path = self._durable_state_path()
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if path is None or runtime_path is None or not runtime_path.exists():
            return
        payload = {
            "version": _STATE_VERSION,
            "game_id": self._knowledge_game_id,
            "objective_tree": self._tree.to_dict() if self._tree is not None else None,
            "role_histories": {
                role: [
                    _without_policy_source(item) if role == "coder" else item
                    for item in history[-_MAX_ROLE_HISTORY:]
                ]
                for role, history in self._role_histories.items()
            },
            "policy_objective_id": self._policy_objective_id,
            "policy_source_hash": self._policy_source_hash,
            "policy_artifact": self._policy_artifact,
            "rejected_policy_objective_id": self._rejected_policy_objective_id,
            "rejected_policy_source_hash": self._rejected_policy_source_hash,
            "rejected_policy_artifact": self._rejected_policy_artifact,
            "policy_solver_type": self._policy_solver_type,
            "policy_solver_family": self._policy_solver_family,
            "active_policy_reuse_scope": self._active_policy_reuse_scope,
            "reusable_policies": self._reusable_policies[-4:],
            "policy_memory": self._policy_memory,
            "policy_repairs": self._policy_repairs,
            "premature_completion_repairs": self._premature_completion_repairs,
            "premature_failure_repairs": self._premature_failure_repairs,
            "repeated_action_repairs": self._repeated_action_repairs,
            "policy_executed_action": self._policy_executed_action,
            "policy_observed_host_progress": self._policy_observed_host_progress,
            "policy_was_reused": self._policy_was_reused,
            "boundary_reason": self._boundary_reason,
            "reduction_required": self._reduction_required,
            "last_transition": self._last_transition,
            "recent_transitions": self._recent_transitions[-8:],
            "action_evidence_level": self._action_evidence_level,
            "level_action_evidence": self._level_action_evidence_payload(),
            "consecutive_activation_failures": self._consecutive_activation_failures,
            "failure_streak_objective_id": self._failure_streak_objective_id,
            "last_reduction_step": self._last_reduction_step,
            "metrics": self._orchestration_metrics,
            "total_tokens": self._session_total_tokens,
            "generated_tokens": self._session_generated_tokens,
        }
        try:
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as exc:
            log.warning(
                "orchestrated objective state write failed at %s: %s", path, exc
            )

    def _append_transcript(self, label: str, payload: Any) -> None:
        path = self._current_transcript_path
        if path is None:
            return
        text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, indent=2, default=str)
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{label}]\n{text}\n")
        except OSError as exc:
            log.warning("orchestration transcript write failed: %s", exc)

    def _emit_event(self, event_type: str, **payload: Any) -> None:
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None:
            return
        if event_type.startswith(("objective_", "policy_")):
            active = self._tree.active if self._tree is not None else None
            selected = active.solver_type if active is not None else None
            payload.setdefault(
                "solver_type", selected.value if selected is not None else ""
            )
            payload.setdefault(
                "solver_family", solver_family(selected.value) if selected is not None else ""
            )
        event = {
            "type": event_type,
            "time": time.time(),
            "game_id": self._knowledge_game_id,
            **payload,
        }
        self._append_transcript(f"ORCHESTRATION {event_type}", event)
        path = runtime_path.with_name("orchestration_events.jsonl")
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            log.warning("orchestration event write failed: %s", exc)

    def _remaining_seconds(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return deadline - time.monotonic()

    def _role_request_timeout(self, request_deadline: float | None) -> float | None:
        candidates: list[float] = []
        if self._orchestration_request_timeout_seconds is not None:
            candidates.append(self._orchestration_request_timeout_seconds)
        remaining = self._remaining_seconds(request_deadline)
        if remaining is not None:
            if remaining <= 0:
                raise OrchestrationYield("analyzer time budget")
            candidates.append(remaining)
        return max(0.1, min(candidates)) if candidates else None

    def _role_output_limit(self, role: str, attempt: int) -> int | None:
        role_limit = self._orchestration_role_max_output.get(role)
        inherited_limit = self._adaptive_output_limit(attempt, repair=attempt > 1)
        limits = [
            value
            for value in (role_limit, inherited_limit)
            if value is not None and value > 0
        ]
        return min(limits) if limits else None

    def _role_thinking_budget(
        self, role: str, output_limit: int | None, *, attempt: int = 1
    ) -> int | None:
        budget = self._orchestration_role_thinking_budget.get(role)
        if budget is None or budget <= 0:
            return None
        # Keep thinking enabled on repair attempts while progressively reserving
        # more visible-output room for the required raw envelope.
        budget = max(64, budget // (2 ** max(0, attempt - 1)))
        if output_limit is None:
            return budget
        # Preserve room for the required structured result even if deployment
        # overrides accidentally set the thinking budget above the output cap.
        return min(budget, max(1, output_limit - 1))

    def _structured_role_call(
        self,
        *,
        role: str,
        system_prompt: str,
        user_payload: dict[str, Any],
        validator: Callable[[dict[str, Any]], Any],
        request_deadline: float | None,
        should_stop: Callable[[], bool] | None,
        rejected_policy_source: str = "",
    ) -> Any:
        history = self._role_histories[role]
        corrections: list[str] = []
        last_error = ""
        context_output_limit: int | None = None
        context_adjustments = 0
        for attempt in range(1, _MAX_STRUCTURED_ATTEMPTS + 1):
            if should_stop is not None and should_stop():
                raise OrchestrationYield("stop requested")
            request_timeout = self._role_request_timeout(request_deadline)
            if self._remaining_game_tokens() == 0:
                raise OrchestrationFailure(
                    "orchestration exhausted the per-game model token budget",
                    category="orchestration_token_budget",
                )
            user_text = json.dumps(
                user_payload, ensure_ascii=False, separators=(",", ":")
            )
            if corrections:
                reducer_requires_decomposition = role == "reducer" and any(
                    "game and level objectives must be decomposed" in item.lower()
                    for item in corrections
                )
                response_instruction = (
                    (
                        "The active objective is host-controlled. Copy its objective_id "
                        'exactly, use verdict="decompose", provide one to six tactical '
                        "subgoals, and select one. Do not return continue, complete, or fail."
                        if reducer_requires_decomposition
                        else "Return a corrected raw BEGIN_REDUCTION/END_REDUCTION JSON object."
                    )
                    if role == "reducer"
                    else (
                        "Minimally repair the rejected module and return one complete "
                        "raw BEGIN_POLICY/END_POLICY module. Fix every listed error; "
                        "do not introduce reflection or undocumented observation fields."
                    )
                )
                user_text += (
                    "\nPrevious validation errors; every item must be fixed:\n- "
                    + "\n- ".join(corrections)
                    + "\n"
                    f"{response_instruction}"
                )
            if role == "coder" and rejected_policy_source:
                bounded_source = rejected_policy_source[
                    :_MAX_REJECTED_POLICY_REPAIR_CHARS
                ]
                truncation = (
                    "\n[rejected source truncated by host]"
                    if len(rejected_policy_source) > len(bounded_source)
                    else ""
                )
                user_text += (
                    "\nThe policy below is the most recent rejected candidate. "
                    "Minimally repair it and preserve working changes. "
                    f"{_rejected_policy_repair_guidance(user_payload)}\n"
                    "<REJECTED_POLICY_SOURCE>\n"
                    f"{bounded_source}{truncation}\n"
                    "</REJECTED_POLICY_SOURCE>"
                )
            messages = [
                {"role": "system", "content": system_prompt},
                *history[-_MAX_ROLE_HISTORY:],
                {"role": "user", "content": user_text},
            ]
            self._should_stop_callback = should_stop
            output_limit = self._role_output_limit(role, attempt)
            if context_output_limit is not None:
                output_limit = (
                    min(output_limit, context_output_limit)
                    if output_limit is not None
                    else context_output_limit
                )
            thinking_budget = self._role_thinking_budget(
                role, output_limit, attempt=attempt
            )
            attempt_metric = f"{role}_attempts"
            self._orchestration_metrics[attempt_metric] = (
                int(self._orchestration_metrics.get(attempt_metric, 0)) + 1
            )
            request_started = time.monotonic()
            self._emit_event(
                "role_request_started",
                role=role,
                structured_attempt=attempt,
                request_timeout_seconds=request_timeout,
                max_output_tokens=output_limit,
                thinking_token_budget=thinking_budget,
                tool_choice=None,
                response_mode="raw_reduction" if role == "reducer" else "raw_policy",
            )
            try:
                result = self.model_client.complete(
                    messages,
                    tools=None,
                    request_timeout_seconds=request_timeout,
                    max_output_tokens=output_limit,
                    thinking_token_budget=thinking_budget,
                    tool_choice=None,
                    request_attempt_limit=1,
                )
            except requests.RequestException as exc:
                elapsed = max(0.0, time.monotonic() - request_started)
                failure_metric = f"{role}_transport_failures"
                self._orchestration_metrics[failure_metric] = (
                    int(self._orchestration_metrics.get(failure_metric, 0)) + 1
                )
                if isinstance(exc, requests.Timeout):
                    timeout_metric = f"{role}_timeouts"
                    self._orchestration_metrics[timeout_metric] = (
                        int(self._orchestration_metrics.get(timeout_metric, 0)) + 1
                    )
                self._emit_event(
                    "role_request_failed",
                    role=role,
                    structured_attempt=attempt,
                    elapsed_seconds=elapsed,
                    timeout=isinstance(exc, requests.Timeout),
                    detail=str(exc),
                )
                last_error = f"{type(exc).__name__}: {exc}"
                if should_stop is not None and should_stop():
                    raise OrchestrationYield("stop requested") from exc
                if _is_context_length_error(exc) and attempt < _MAX_STRUCTURED_ATTEMPTS:
                    context_adjustments += 1
                    limits = _context_error_limits(exc)
                    previous_output_limit = output_limit
                    if limits is not None:
                        context_window, requested_output, input_tokens = limits
                        available_output = max(256, context_window - input_tokens - 256)
                        context_output_limit = min(requested_output, available_output)
                    elif output_limit is not None:
                        context_output_limit = max(256, output_limit // 2)
                    else:
                        context_output_limit = 2048
                    history_before = len(history)
                    if context_adjustments == 1 and len(history) > 2:
                        del history[:-2]
                    else:
                        history.clear()
                    self._orchestration_metrics["role_context_adjustments"] = int(
                        self._orchestration_metrics.get("role_context_adjustments", 0)
                    ) + 1
                    self._emit_event(
                        "role_context_adjusted",
                        role=role,
                        structured_attempt=attempt,
                        previous_max_output_tokens=previous_output_limit,
                        adjusted_max_output_tokens=context_output_limit,
                        history_messages_removed=history_before - len(history),
                        server_context_tokens=limits[0] if limits is not None else None,
                        measured_input_tokens=limits[2] if limits is not None else None,
                    )
                    continue
                if attempt < _MAX_STRUCTURED_ATTEMPTS:
                    continue
                raise OrchestrationFailure(
                    f"{role} transport failed after {_MAX_STRUCTURED_ATTEMPTS} attempts: {exc}",
                    category=f"orchestration_{role}_transport_exhausted",
                ) from exc
            finally:
                elapsed = max(0.0, time.monotonic() - request_started)
                seconds_metric = f"{role}_model_seconds"
                self._orchestration_metrics[seconds_metric] = (
                    float(self._orchestration_metrics.get(seconds_metric, 0.0))
                    + elapsed
                )
            self._accumulate_usage_tokens(result.usage)
            self._orchestration_metrics[f"{role}_calls"] = int(
                self._orchestration_metrics.get(f"{role}_calls", 0)
            ) + max(1, result.request_attempts)
            generated_tokens = 0
            for key in ("completion_tokens", "output_tokens", "generated_tokens"):
                try:
                    generated_tokens = max(0, int((result.usage or {}).get(key)))
                    break
                except (AttributeError, TypeError, ValueError):
                    continue
            token_metric = f"{role}_generated_tokens"
            self._orchestration_metrics[token_metric] = (
                int(self._orchestration_metrics.get(token_metric, 0)) + generated_tokens
            )
            self._emit_event(
                "role_request_completed",
                role=role,
                structured_attempt=attempt,
                request_attempts=max(1, result.request_attempts),
                elapsed_seconds=elapsed,
                generated_tokens=generated_tokens,
                finish_reason=str(getattr(result, "finish_reason", "") or ""),
            )
            reasoning = _extract_reasoning_text(result.message)
            raw: dict[str, Any] | None = None
            try:
                if role == "coder":
                    raw = {"source": _policy_source_from_message(result.message)}
                elif role == "reducer":
                    raw = _reduction_from_message(result.message)
                else:
                    raise ValueError(f"unsupported orchestration role {role!r}")
                value = validator(raw)
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                ObjectiveError,
                PolicyRuntimeError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                correction = last_error[:4000]
                if correction not in corrections:
                    corrections.append(correction)
                if role == "coder" and isinstance(raw, dict):
                    rejected_source = raw.get("source")
                    if isinstance(rejected_source, str) and rejected_source.strip():
                        rejected_policy_source = rejected_source
                        self._record_rejected_policy_candidate(
                            rejected_source,
                            phase="raw_content_validation",
                            detail=str(exc),
                            category=str(getattr(exc, "category", "structured_output")),
                            structured_attempt=attempt,
                        )
                if role == "reducer":
                    self._record_rejected_structured_response(
                        role=role,
                        message=result.message,
                        detail=str(exc),
                        structured_attempt=attempt,
                    )
                self._append_transcript(
                    f"{role.upper()} REJECTED",
                    {"current_error": correction, "all_errors": corrections},
                )
                continue
            if role == "coder":
                assistant_content = (
                    f"{_POLICY_BEGIN_MARKER}\n{raw['source']}\n{_POLICY_END_MARKER}"
                )
            else:
                encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
                assistant_content = (
                    f"{_REDUCTION_BEGIN_MARKER}\n{encoded}\n{_REDUCTION_END_MARKER}"
                )
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if reasoning:
                assistant_entry["reasoning_content"] = reasoning[-6000:]
            history.extend(
                [
                    {"role": "user", "content": user_text[-12000:]},
                    assistant_entry,
                ]
            )
            del history[:-_MAX_ROLE_HISTORY]
            transcript_payload = raw
            if role == "coder" and "source" in raw:
                transcript_payload = {
                    **raw,
                    "source": "<stored as a hashed policy artifact>",
                }
            self._append_transcript(f"{role.upper()} ACCEPTED", transcript_payload)
            return value
        raise OrchestrationFailure(
            f"{role} did not produce a valid structured result after {_MAX_STRUCTURED_ATTEMPTS} attempts: {last_error}",
            category=f"orchestration_{role}_exhausted",
        )

    def _reducer_payload(
        self, frame: Frame, history: list[HistoryEntry]
    ) -> dict[str, Any]:
        assert self._tree is not None
        level = self._tree.current_level_objective
        return {
            "boundary_reason": self._boundary_reason,
            "active_objective": self._tree.active.to_dict(),
            "objective_tree": self._tree.to_dict(),
            "level_action_budget": {
                "used": level.actions_used,
                "limit": level.action_budget,
                "remaining": level.remaining_actions,
            },
            "observation": {
                "level": frame.level,
                "step": frame.step,
                "engine_state": frame.engine_state,
                "valid_actions": to_model_actions(frame.valid_actions),
                "board_hex_rows": _compact_board(frame),
            },
            "action_contract": dict(_MODEL_ACTION_CONTRACT),
            "recent_transitions": _recent_transition_payload(history),
            "level_action_evidence": self._level_action_evidence_payload(),
            "host_control_model": _host_control_model(history, level=frame.level),
            "runtime_transition_scope": (
                "PolicyObservation.last_transition and recent_transitions contain only "
                "transitions whose objective_id matches active_objective. At objective "
                "entry last_transition is None until this policy executes an action."
            ),
            "policy_repairs_for_active": self._policy_repairs.get(
                self._tree.active_id, 0
            ),
        }

    def _reduce(
        self,
        frame: Frame,
        history: list[HistoryEntry],
        *,
        request_deadline: float | None,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        assert self._tree is not None
        remaining_level_actions = self._tree.remaining_level_actions
        if remaining_level_actions <= 0:
            raise OrchestrationFailure(
                "the host level action budget is exhausted",
                category="orchestration_level_action_budget_exhausted",
            )
        host_control_model = _host_control_model(history, level=frame.level)

        def validate_reduction(raw: dict[str, Any]) -> ReductionProposal:
            proposal = ReductionProposal.from_payload(raw)
            if proposal.verdict is ReductionVerdict.DECOMPOSE:
                raw_subgoals = raw.get("subgoals")
                if not isinstance(raw_subgoals, list) or any(
                    not isinstance(item, dict) or "solver_type" not in item
                    for item in raw_subgoals
                ):
                    raise ObjectiveError(
                        "every new tactical subgoal must declare solver_type"
                    )
                selected = proposal.subgoals[proposal.selected_index]
                saturation_reason = _action_family_saturation_reason(
                    self._level_action_evidence, "MOUSE"
                )
                if saturation_reason and _contract_requests_mouse(selected):
                    raise ObjectiveError(
                        f"selected subgoal requires a saturated action family: "
                        f"{saturation_reason}. Select a non-MOUSE objective supported "
                        "by remaining action evidence"
                    )
                if (
                    _contract_requires_navigation(selected)
                    and selected.execution_mode is not TacticalExecutionMode.NAVIGATE
                ):
                    raise ObjectiveError(
                        "selected subgoal describes spatial routing but execution_mode "
                        "is not navigate; declare execution_mode=navigate so the policy "
                        "must localize the board, plan a route, and replan after changes"
                    )
                selected_family = solver_family(selected.solver_type.value)
                raw_selected = raw_subgoals[proposal.selected_index]
                control_model_override = str(
                    raw_selected.get("control_model_override") or ""
                ).strip()[:400]
                if (
                    host_control_model["linked_high_confidence"]
                    and selected_family in {"observe", "routing"}
                    and len(control_model_override) < 20
                ):
                    recommended = ", ".join(
                        host_control_model["recommended_solver_types"]
                    )
                    raise ObjectiveError(
                        f"host control model is high-confidence "
                        f"{host_control_model['scheme']}; solver type "
                        f"{selected.solver_type.value!r} ({selected_family}) assumes "
                        "static or single-actor dynamics. Select one of "
                        f"{recommended}, or provide a specific control_model_override "
                        "explaining why linked motion is irrelevant to this subgoal"
                    )
                if (
                    selected.execution_mode is TacticalExecutionMode.NAVIGATE
                    and selected_family not in NAVIGATION_SOLVER_FAMILIES
                ):
                    raise ObjectiveError(
                        f"navigate subgoal selected non-navigation solver type "
                        f"{selected.solver_type.value!r} ({selected_family}); select a "
                        "routing, physics, manipulation, alignment, coverage, field, "
                        "gravity, or multi-agent solver"
                    )
                if (
                    _contract_requests_mouse(selected)
                    and selected.evidence_mode
                    is ObjectiveEvidenceMode.STABLE_TRANSITION
                ):
                    raise ObjectiveError(
                        "a MOUSE interaction-learning contract cannot use "
                        "stable_transition because repeatability alone does not prove "
                        "the clicked coordinate is causal; use contrastive_transition "
                        "with a repeated positive and distinct negative control"
                    )
                failed_execution_hypotheses = (
                    _failed_engine_progress_since_recalibration(self._tree)
                )
                if (
                    failed_execution_hypotheses
                    >= _STRATEGIC_RECALIBRATION_FAILURES
                    and selected.evidence_mode
                    is not ObjectiveEvidenceMode.CONTRASTIVE_TRANSITION
                ):
                    raise ObjectiveError(
                        f"{failed_execution_hypotheses} consecutive engine_progress "
                        "objectives failed without a successful recalibration; select "
                        "a contrastive_transition control experiment before another "
                        "execution hypothesis"
                    )
                repeated = _equivalent_attempted_tactical(self._tree, selected)
                if repeated is not None:
                    raise ObjectiveError(
                        "selected subgoal repeats already-attempted tactical contract "
                        f"{repeated.objective_id!r} ({repeated.title!r}); selected "
                        f"title was {selected.title!r}. Select a materially different "
                        "action or spatial target and use the prior resolution evidence"
                    )
            probe = ObjectiveTree.from_dict(self._tree.to_dict())
            probe.apply_proposal(
                proposal,
                remaining_level_actions=remaining_level_actions,
            )
            return proposal

        proposal = self._structured_role_call(
            role="reducer",
            system_prompt=_REDUCER_SYSTEM_PROMPT,
            user_payload=self._reducer_payload(frame, history),
            validator=validate_reduction,
            request_deadline=request_deadline,
            should_stop=should_stop,
        )
        previous_id = self._tree.active_id
        active = self._tree.apply_proposal(
            proposal,
            remaining_level_actions=remaining_level_actions,
        )
        if proposal.verdict.value == "complete":
            self._orchestration_metrics["objectives_completed"] = (
                int(self._orchestration_metrics.get("objectives_completed", 0)) + 1
            )
        elif proposal.verdict.value == "fail":
            self._orchestration_metrics["objectives_failed"] = (
                int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
            )
        self._last_reduction_step = frame.step
        self._reduction_required = False
        self._emit_event(
            "objective_reduced",
            previous_objective_id=previous_id,
            active_objective_id=active.objective_id,
            verdict=proposal.verdict.value,
            rationale=proposal.rationale,
            solver_type=active.solver_type.value if active.solver_type is not None else "",
            solver_family=(
                solver_family(active.solver_type.value)
                if active.solver_type is not None
                else ""
            ),
            host_control_scheme=host_control_model["scheme"],
            host_control_confidence=host_control_model["confidence"],
            host_control_confidence_reason=host_control_model["confidence_reason"],
            host_control_decisive_opposing_samples=host_control_model[
                "decisive_opposing_samples"
            ],
        )

    def _policy_payload(
        self, frame: Frame, history: list[HistoryEntry], repair: str
    ) -> dict[str, Any]:
        assert self._tree is not None
        active = self._tree.active
        selected_solver = active.solver_type.value if active.solver_type is not None else ""
        return {
            "active_objective": active.to_dict(),
            "observation": {
                "level": frame.level,
                "step": frame.step,
                "engine_state": frame.engine_state,
                "valid_actions": to_model_actions(frame.valid_actions),
                "board_hex_rows": _compact_board(frame),
            },
            "action_contract": dict(_MODEL_ACTION_CONTRACT),
            "recent_transitions": _recent_transition_payload(history),
            "level_action_evidence": self._level_action_evidence_payload(),
            "requested_backend": os.environ.get("LOCAL_GAMEPLAY_POLICY_BACKEND", "cpu"),
            "solver_contract": {
                "api_version": 1,
                "selected_type": selected_solver,
                "selected_family": solver_family(selected_solver) if selected_solver else "",
                "registered_types": list(POLICY_SOLVER_TYPES),
                "required_entrypoint": (
                    "solver_decide(POLICY_SOLVER_TYPE, observation, memory, "
                    "POLICY_SOLVER_CONFIG)"
                ),
            },
            "repair_reason": repair,
        }

    def _policy_validator(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._tree is None:
            raise PolicyRuntimeError("objective tree is unavailable")
        source = str(raw.get("source") or "")
        source_hash = verify_policy_source(source)
        active = self._tree.active
        if active.solver_type is None:
            raise PolicyRuntimeError(
                "active tactical objective has no solver type",
                category="policy_solver_contract",
            )
        declared_type = _literal_policy_assignment(source, "POLICY_SOLVER_TYPE")
        if not isinstance(declared_type, str):
            raise PolicyRuntimeError(
                "POLICY_SOLVER_TYPE must be a literal string",
                category="policy_solver_contract",
            )
        declared_type = declared_type.strip().lower()
        if declared_type != active.solver_type.value:
            raise PolicyRuntimeError(
                f"policy solver type {declared_type!r} does not match active objective "
                f"solver type {active.solver_type.value!r}",
                category="policy_solver_contract",
            )
        declared_config = _literal_policy_assignment(source, "POLICY_SOLVER_CONFIG")
        try:
            normalized_config = validate_solver_config(declared_type, declared_config)
        except ValueError as exc:
            raise PolicyRuntimeError(
                f"invalid POLICY_SOLVER_CONFIG: {exc}",
                category="policy_solver_contract",
            ) from exc
        solver_calls = _reachable_policy_calls(source, "solver_decide")
        if not solver_calls:
            raise PolicyRuntimeError(
                "generated policy decide path must call solver_decide",
                category="policy_solver_contract",
            )
        if any(not _valid_solver_dispatch_call(call) for call in solver_calls):
            raise PolicyRuntimeError(
                "solver_decide must be called with exactly POLICY_SOLVER_TYPE, "
                "observation, memory, POLICY_SOLVER_CONFIG",
                category="policy_solver_contract",
            )
        _validate_solver_declaration_usage(source)
        declared_family = solver_family(declared_type)
        if (
            active.evidence_mode is ObjectiveEvidenceMode.CONTRASTIVE_TRANSITION
            and declared_family == "observe"
        ):
            probes = list(normalized_config["probe_actions"])
            repeated = {action for action in probes if probes.count(action) >= 2}
            modalities = {
                "direction": {"UP", "DOWN", "LEFT", "RIGHT"},
            }
            has_same_modality_control = any(
                positive in actions
                and any(candidate != positive for candidate in probes if candidate in actions)
                for positive in repeated
                for actions in modalities.values()
            )
            if len(probes) < active.minimum_evidence_actions:
                raise PolicyRuntimeError(
                    "contrastive observation policy probe_actions must contain at least "
                    f"{active.minimum_evidence_actions} actions to meet the objective's "
                    "minimum evidence count",
                    category="policy_solver_contract",
                )
            if not repeated or not has_same_modality_control:
                raise PolicyRuntimeError(
                    "contrastive observation policy probe_actions must repeat one exact "
                    "positive action and include a distinct same-modality negative control "
                    "(direction versus direction; SPACE has no distinct button control)",
                    category="policy_solver_contract",
                )
        if (
            active.execution_mode is TacticalExecutionMode.NAVIGATE
            and declared_family not in NAVIGATION_SOLVER_FAMILIES
        ):
            raise PolicyRuntimeError(
                f"navigate policy solver family {declared_family!r} is not "
                "navigation-capable",
                category="policy_navigation_contract",
            )
        return {
            "objective_id": self._tree.active_id,
            "source": source,
            "source_hash": source_hash,
            "solver_type": declared_type,
            "solver_family": declared_family,
        }

    def _save_policy_artifact(self, source: str, source_hash: str) -> str:
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None or self._tree is None:
            return ""
        directory = runtime_path.parent / "policies"
        directory.mkdir(parents=True, exist_ok=True)
        safe_objective = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._tree.active_id)
        path = directory / f"{safe_objective}-{source_hash}.py"
        if not path.exists():
            path.write_text(source, encoding="utf-8")
        return str(path.relative_to(runtime_path.parent))

    def _save_rejected_policy_artifact(self, source: str) -> tuple[str, str]:
        content_hash = hashlib.blake2b(
            source.encode("utf-8"), digest_size=16
        ).hexdigest()
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None:
            return "", content_hash
        objective_id = self._tree.active_id if self._tree is not None else "unknown"
        safe_objective = re.sub(r"[^A-Za-z0-9_.-]+", "_", objective_id)
        directory = runtime_path.parent / "policies" / "rejected"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{safe_objective}-{content_hash}.py"
        if not path.exists():
            path.write_text(source, encoding="utf-8")
        return str(path.relative_to(runtime_path.parent)), content_hash

    def _record_rejected_policy_candidate(
        self,
        source: str,
        *,
        phase: str,
        detail: str,
        category: str,
        structured_attempt: int | None = None,
        semantic_hash: str = "",
    ) -> tuple[str, str]:
        artifact, content_hash = self._save_rejected_policy_artifact(source)
        self._emit_event(
            "policy_candidate_rejected",
            objective_id=self._tree.active_id if self._tree is not None else "",
            phase=phase,
            category=category,
            detail=detail[:2000],
            structured_attempt=structured_attempt,
            content_hash=content_hash,
            semantic_hash=semantic_hash,
            source_bytes=len(source.encode("utf-8")),
            artifact=artifact,
        )
        return artifact, content_hash

    def _record_rejected_structured_response(
        self,
        *,
        role: str,
        message: dict[str, Any],
        detail: str,
        structured_attempt: int,
    ) -> None:
        content = _normalize_message_content(message.get("content", ""))
        reasoning = _extract_reasoning_text(message)
        payload = {
            "role": role,
            "structured_attempt": structured_attempt,
            "detail": detail[:4000],
            "content": content,
            "reasoning_content": reasoning,
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        content_hash = hashlib.blake2b(
            encoded.encode("utf-8"), digest_size=16
        ).hexdigest()
        runtime_path = getattr(self, "_session_runtime_dir", None)
        artifact = ""
        if runtime_path is not None:
            directory = runtime_path.parent / "structured_rejected"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{role}-{content_hash}.json"
            if not path.exists():
                path.write_text(encoded, encoding="utf-8")
            artifact = str(path.relative_to(runtime_path.parent))
        self._emit_event(
            "structured_candidate_rejected",
            role=role,
            structured_attempt=structured_attempt,
            detail=detail[:2000],
            content_hash=content_hash,
            content_bytes=len(content.encode("utf-8")),
            reasoning_bytes=len(reasoning.encode("utf-8")),
            artifact=artifact,
        )

    def _policy_artifact_source(
        self, artifact_name: str, expected_hash: str
    ) -> str | None:
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None or not artifact_name or not expected_hash:
            return None
        artifact = (runtime_path.parent / artifact_name).resolve()
        policy_root = (runtime_path.parent / "policies").resolve()
        try:
            artifact.relative_to(policy_root)
        except ValueError:
            return None
        try:
            if not artifact.is_file():
                return None
            source = artifact.read_text(encoding="utf-8")
            if verify_policy_source(source) != expected_hash:
                return None
        except (OSError, PolicyRuntimeError, ValueError):
            return None
        return source

    def _rejected_policy_source_for_repair(self) -> str:
        if (
            self._tree is None
            or self._rejected_policy_objective_id != self._tree.active_id
        ):
            return ""
        return (
            self._policy_artifact_source(
                self._rejected_policy_artifact,
                self._rejected_policy_source_hash,
            )
            or ""
        )

    def _remember_rejected_policy(
        self, *, objective_id: str, artifact: str, source_hash: str
    ) -> None:
        self._rejected_policy_objective_id = objective_id
        self._rejected_policy_artifact = artifact
        self._rejected_policy_source_hash = source_hash

    def _clear_rejected_policy(self) -> None:
        self._rejected_policy_objective_id = ""
        self._rejected_policy_artifact = ""
        self._rejected_policy_source_hash = ""

    def _evict_reusable_policy(self, source_hash: str, reason: str) -> None:
        before = len(self._reusable_policies)
        self._reusable_policies = [
            item
            for item in self._reusable_policies
            if str(item.get("source_hash") or "") != source_hash
        ]
        if len(self._reusable_policies) == before:
            return
        self._orchestration_metrics["policy_reuse_evictions"] = int(
            self._orchestration_metrics.get("policy_reuse_evictions", 0)
        ) + 1
        self._emit_event(
            "policy_reuse_evicted",
            source_hash=source_hash,
            objective_id=self._policy_objective_id,
            reason=reason,
            host_progress=self._policy_observed_host_progress,
        )

    def _cache_current_policy_for_reuse(self, reason: str) -> None:
        if self._tree is None or not self._policy_source_hash:
            return
        successful = reason.startswith("subgoal_succeeded:")
        if self._policy_was_reused:
            if successful and self._policy_observed_host_progress:
                promoted = False
                for item in self._reusable_policies:
                    if str(item.get("source_hash") or "") == self._policy_source_hash:
                        if item.get("qualification") != "proven":
                            item["qualification"] = "proven"
                            promoted = True
                        item["last_progress_objective_id"] = self._policy_objective_id
                if promoted:
                    self._orchestration_metrics["policy_reuse_promotions"] = int(
                        self._orchestration_metrics.get("policy_reuse_promotions", 0)
                    ) + 1
                    self._emit_event(
                        "policy_reuse_promoted",
                        source_hash=self._policy_source_hash,
                        objective_id=self._policy_objective_id,
                    )
                return
            self._evict_reusable_policy(
                self._policy_source_hash,
                "reused policy ended without host-confirmed progress: " + reason,
            )
            return
        if (
            self._active_policy_reuse_scope != "tactical"
            or not self._policy_executed_action
            or not successful
            or not self._policy_artifact
        ):
            return
        origin = self._tree.nodes.get(self._policy_objective_id)
        if origin is None or origin.kind is not ObjectiveKind.TACTICAL:
            return
        qualification = (
            "proven" if self._policy_observed_host_progress else "provisional"
        )
        entry = {
            "game_id": self._knowledge_game_id,
            "level": self._tree.current_level,
            "source_hash": self._policy_source_hash,
            "artifact": self._policy_artifact,
            "origin_objective_id": self._policy_objective_id,
            "contract_hash": _objective_contract_hash(origin),
            "solver_type": origin.solver_type.value if origin.solver_type is not None else "",
            "solver_family": (
                solver_family(origin.solver_type.value)
                if origin.solver_type is not None
                else ""
            ),
            "qualification": qualification,
        }
        self._reusable_policies = [
            item
            for item in self._reusable_policies
            if not (
                item.get("source_hash") == entry["source_hash"]
                and item.get("level") == entry["level"]
            )
        ]
        self._reusable_policies.append(entry)
        del self._reusable_policies[:-4]
        if qualification == "provisional":
            self._orchestration_metrics["policy_provisional_caches"] = int(
                self._orchestration_metrics.get("policy_provisional_caches", 0)
            ) + 1
        self._emit_event("policy_reuse_cached", **entry)

    def _try_reuse_policy(self, frame: Frame) -> bool:
        assert self._tree is not None
        objective = self._tree.active
        if objective.kind is not ObjectiveKind.TACTICAL:
            return False
        contract_hash = _objective_contract_hash(objective)
        for entry in reversed(self._reusable_policies):
            try:
                entry_level = int(entry.get("level") or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                str(entry.get("game_id") or "") != self._knowledge_game_id
                or entry_level != self._tree.current_level
                or str(entry.get("solver_type") or "")
                != (objective.solver_type.value if objective.solver_type is not None else "")
            ):
                continue
            qualification = str(entry.get("qualification") or "provisional")
            if qualification not in {"proven", "provisional"}:
                continue
            if qualification == "provisional" and str(
                entry.get("contract_hash") or ""
            ) != contract_hash:
                continue
            source_hash = str(entry.get("source_hash") or "")
            artifact = str(entry.get("artifact") or "")
            source = self._policy_artifact_source(artifact, source_hash)
            if source is None or _policy_reuse_scope_from_source(source) != "tactical":
                continue
            runtime = GameplayPolicyRuntime()
            try:
                validated = self._policy_validator({"source": source})
                if str(validated.get("source_hash") or "") != source_hash:
                    raise PolicyRuntimeError(
                        "reused policy solver validation changed source fingerprint",
                        category="policy_protocol",
                    )
                activation = runtime.activate(
                    source,
                    context={
                        "game_id": self._knowledge_game_id,
                        "objective": objective.to_dict(),
                    },
                )
                runtime.preflight(
                    PolicyObservation(
                        board=np.asarray(frame.grid, dtype=np.uint8),
                        level=frame.level,
                        step=frame.step,
                        valid_actions=tuple(to_model_actions(frame.valid_actions)),
                        last_transition=None,
                        objective=objective.to_dict(),
                        recent_transitions=(),
                        backend=activation.backend,
                    ),
                    minimum_actions=objective.minimum_evidence_actions,
                )
                assert runtime.activation is not None
                activation = runtime.activation
                if activation.source_hash != source_hash:
                    raise PolicyRuntimeError(
                        "reused policy fingerprint mismatch",
                        category="policy_protocol",
                    )
            except (OSError, PolicyRuntimeError, ValueError) as exc:
                runtime.close()
                self._orchestration_metrics["policy_reuse_rejections"] = (
                    int(self._orchestration_metrics.get("policy_reuse_rejections", 0))
                    + 1
                )
                self._emit_event(
                    "policy_reuse_rejected",
                    objective_id=objective.objective_id,
                    origin_objective_id=entry.get("origin_objective_id"),
                    source_hash=source_hash,
                    category=getattr(exc, "category", "policy_reuse"),
                    detail=str(exc),
                )
                continue
            if self._policy_runtime is not None:
                self._policy_runtime.close()
            self._policy_runtime = runtime
            self._policy_objective_id = objective.objective_id
            self._policy_source_hash = source_hash
            self._policy_artifact = artifact
            self._policy_solver_type = (
                objective.solver_type.value if objective.solver_type is not None else ""
            )
            self._policy_solver_family = (
                solver_family(self._policy_solver_type) if self._policy_solver_type else ""
            )
            self._active_policy_reuse_scope = "tactical"
            self._policy_memory = runtime.memory
            self._policy_executed_action = False
            self._policy_observed_host_progress = False
            self._policy_was_reused = True
            self._orchestration_metrics["policy_activations"] = (
                int(self._orchestration_metrics.get("policy_activations", 0)) + 1
            )
            self._orchestration_metrics["policy_reuses"] = (
                int(self._orchestration_metrics.get("policy_reuses", 0)) + 1
            )
            if activation.backend_fallback_reason:
                self._orchestration_metrics["cuda_fallbacks"] = (
                    int(self._orchestration_metrics.get("cuda_fallbacks", 0)) + 1
                )
            self._emit_event(
                "policy_reused",
                objective_id=objective.objective_id,
                origin_objective_id=entry.get("origin_objective_id"),
                source_hash=source_hash,
                artifact=artifact,
                qualification=qualification,
                contract_hash=contract_hash,
                backend=activation.backend,
                backend_fallback_reason=activation.backend_fallback_reason,
                solver_type=self._policy_solver_type,
                solver_family=self._policy_solver_family,
            )
            return True
        return False

    def _activate_policy(
        self,
        frame: Frame,
        history: list[HistoryEntry],
        *,
        request_deadline: float | None,
        should_stop: Callable[[], bool] | None,
        repair_reason: str = "",
    ) -> None:
        assert self._tree is not None
        if self._tree.active.kind is not ObjectiveKind.TACTICAL:
            raise ObjectiveError(
                "a gameplay policy requires an active tactical objective"
            )
        if repair_reason.startswith("subgoal_succeeded:") and self._try_reuse_policy(
            frame
        ):
            return
        raw = self._structured_role_call(
            role="coder",
            system_prompt=_CODER_SYSTEM_PROMPT,
            user_payload=self._policy_payload(frame, history, repair_reason),
            validator=self._policy_validator,
            request_deadline=request_deadline,
            should_stop=should_stop,
            rejected_policy_source=self._rejected_policy_source_for_repair(),
        )
        source = str(raw["source"])
        source_hash = str(raw["source_hash"])
        runtime = GameplayPolicyRuntime()
        try:
            activation = runtime.activate(
                source,
                context={
                    "game_id": self._knowledge_game_id,
                    "objective": self._tree.active.to_dict(),
                },
            )
            if self._tree.active.actions_used == 0:
                runtime.preflight(
                    PolicyObservation(
                        board=np.asarray(frame.grid, dtype=np.uint8),
                        level=frame.level,
                        step=frame.step,
                        valid_actions=tuple(to_model_actions(frame.valid_actions)),
                        last_transition=None,
                        objective=self._tree.active.to_dict(),
                        recent_transitions=(),
                        backend=activation.backend,
                    ),
                    minimum_actions=self._tree.active.minimum_evidence_actions,
                )
                assert runtime.activation is not None
                activation = runtime.activation
        except PolicyRuntimeError as exc:
            runtime.close()
            artifact, _ = self._record_rejected_policy_candidate(
                source,
                phase=(
                    "preflight" if exc.category == "policy_preflight" else "activation"
                ),
                detail=str(exc),
                category=exc.category,
                semantic_hash=source_hash,
            )
            self._remember_rejected_policy(
                objective_id=self._tree.active_id,
                artifact=artifact,
                source_hash=source_hash,
            )
            raise
        if activation.source_hash != source_hash:
            runtime.close()
            artifact, _ = self._record_rejected_policy_candidate(
                source,
                phase="activation_fingerprint",
                detail="activated policy fingerprint does not match verified source",
                category="policy_protocol",
                semantic_hash=source_hash,
            )
            self._remember_rejected_policy(
                objective_id=self._tree.active_id,
                artifact=artifact,
                source_hash=source_hash,
            )
            raise PolicyRuntimeError(
                "activated policy fingerprint does not match verified source",
                category="policy_protocol",
            )
        if self._policy_runtime is not None:
            self._policy_runtime.close()
        self._policy_runtime = runtime
        self._clear_rejected_policy()
        self._policy_objective_id = self._tree.active_id
        self._policy_source_hash = source_hash
        self._policy_artifact = self._save_policy_artifact(source, source_hash)
        self._policy_solver_type = str(raw["solver_type"])
        self._policy_solver_family = str(raw["solver_family"])
        self._active_policy_reuse_scope = _policy_reuse_scope_from_source(source)
        self._policy_memory = runtime.memory
        self._policy_executed_action = False
        self._policy_observed_host_progress = False
        self._policy_was_reused = False
        self._orchestration_metrics["policy_activations"] = (
            int(self._orchestration_metrics.get("policy_activations", 0)) + 1
        )
        if activation.backend_fallback_reason:
            self._orchestration_metrics["cuda_fallbacks"] = (
                int(self._orchestration_metrics.get("cuda_fallbacks", 0)) + 1
            )
        self._emit_event(
            "policy_activated",
            objective_id=self._policy_objective_id,
            source_hash=source_hash,
            artifact=self._policy_artifact,
            backend=activation.backend,
            backend_fallback_reason=activation.backend_fallback_reason,
            solver_type=self._policy_solver_type,
            solver_family=self._policy_solver_family,
        )

    def _invalidate_policy(
        self, reason: str, *, require_reduction: bool = False
    ) -> None:
        self._cache_current_policy_for_reuse(reason)
        if self._policy_runtime is not None:
            self._policy_memory = self._policy_runtime.memory
            self._policy_runtime.close()
        self._policy_runtime = None
        self._policy_objective_id = ""
        self._active_policy_reuse_scope = "none"
        self._policy_executed_action = False
        self._policy_observed_host_progress = False
        self._policy_was_reused = False
        self._boundary_reason = reason
        self._reduction_required = self._reduction_required or require_reduction

    def _restore_policy_if_possible(self) -> bool:
        if (
            self._policy_runtime is not None
            or self._tree is None
            or self._policy_objective_id != self._tree.active_id
            or not self._policy_artifact
        ):
            return self._policy_runtime is not None
        runtime_path = getattr(self, "_session_runtime_dir", None)
        if runtime_path is None:
            return False
        artifact = (runtime_path.parent / self._policy_artifact).resolve()
        policy_root = (runtime_path.parent / "policies").resolve()
        try:
            artifact.relative_to(policy_root)
        except ValueError:
            return False
        if not artifact.is_file():
            return False
        source = artifact.read_text(encoding="utf-8")
        if verify_policy_source(source) != self._policy_source_hash:
            return False
        try:
            restored_solver = _literal_policy_assignment(source, "POLICY_SOLVER_TYPE")
        except PolicyRuntimeError:
            restored_solver = ""
        self._policy_solver_type = (
            str(restored_solver).strip().lower()
            if isinstance(restored_solver, str)
            else ""
        )
        try:
            self._policy_solver_family = (
                solver_family(self._policy_solver_type)
                if self._policy_solver_type
                else ""
            )
        except ValueError:
            return False
        self._active_policy_reuse_scope = _policy_reuse_scope_from_source(source)
        runtime = GameplayPolicyRuntime()
        activation = runtime.activate(
            source,
            context={
                "game_id": self._knowledge_game_id,
                "objective": self._tree.active.to_dict(),
            },
        )
        runtime.set_memory(self._policy_memory)
        self._policy_runtime = runtime
        self._emit_event(
            "policy_resumed",
            objective_id=self._policy_objective_id,
            source_hash=self._policy_source_hash,
            backend=activation.backend,
            solver_type=self._policy_solver_type,
            solver_family=self._policy_solver_family,
        )
        return True

    def _observation(self, frame: Frame) -> PolicyObservation:
        if (
            self._tree is None
            or self._policy_runtime is None
            or self._policy_runtime.activation is None
        ):
            raise PolicyRuntimeError(
                "policy observation requested without active runtime"
            )
        active_id = self._tree.active_id
        active_transitions = tuple(
            item
            for item in self._recent_transitions
            if str(item.get("objective_id") or "") == active_id
        )[-8:]
        last_transition = (
            self._last_transition
            if isinstance(self._last_transition, dict)
            and str(self._last_transition.get("objective_id") or "") == active_id
            else None
        )
        return PolicyObservation(
            board=np.asarray(frame.grid, dtype=np.uint8),
            level=frame.level,
            step=frame.step,
            valid_actions=tuple(to_model_actions(frame.valid_actions)),
            last_transition=last_transition,
            objective=self._tree.active.to_dict(),
            recent_transitions=active_transitions,
            backend=self._policy_runtime.activation.backend,
        )

    def _stagnated(self, frame: Frame, history: list[HistoryEntry]) -> bool:
        if frame.step - self._last_reduction_step < 6 or len(history) < 6:
            return False
        recent = history[-6:]
        return all(entry.frame.grid == recent[0].frame.grid for entry in recent[1:])

    def _ensure_tree(self, frame: Frame) -> None:
        status = self._level_action_status
        status_matches_frame = status is not None and status[0] == frame.level
        level_action_budget = status[2] if status_matches_frame else 32
        if self._tree is None:
            self._action_evidence_level = frame.level
            self._tree = ObjectiveTree.start_game(
                self._knowledge_game_id or "unknown",
                level=frame.level,
                level_action_budget=level_action_budget,
            )
            self._emit_event(
                "objective_created",
                root_id=self._tree.root_id,
                active_objective_id=self._tree.active_id,
            )
            self._boundary_reason = "game_start"
        elif frame.level != self._tree.current_level:
            self._invalidate_policy("level_transition", require_reduction=True)
            self._level_action_evidence = {}
            self._action_evidence_level = frame.level
            level = self._tree.start_level(
                frame.level,
                level_action_budget=level_action_budget,
            )
            self._emit_event(
                "level_objective_created",
                active_objective_id=level.objective_id,
                level=frame.level,
            )
        elif self._action_evidence_level != frame.level:
            self._level_action_evidence = {}
            self._action_evidence_level = frame.level
        if status_matches_frame:
            self._tree.sync_level_action_status(used=status[1], limit=status[2])

    def _policy_failure(
        self,
        exc: PolicyRuntimeError,
        *,
        counts_activation_failure: bool = True,
    ) -> None:
        assert self._tree is not None
        objective_id = self._tree.active_id
        if counts_activation_failure:
            if self._failure_streak_objective_id != objective_id:
                self._failure_streak_objective_id = objective_id
                self._consecutive_activation_failures = 0
            self._consecutive_activation_failures += 1
        repairs = self._policy_repairs.get(objective_id, 0) + 1
        self._policy_repairs[objective_id] = repairs
        require_reduction = repairs > _MAX_POLICY_REPAIRS
        self._orchestration_metrics["policy_repairs"] = (
            int(self._orchestration_metrics.get("policy_repairs", 0)) + 1
        )
        self._emit_event(
            "policy_failed",
            objective_id=objective_id,
            category=exc.category,
            detail=str(exc),
            repair=repairs,
            repair_route="reducer" if require_reduction else "coder",
            counts_activation_failure=counts_activation_failure,
        )
        self._invalidate_policy(
            f"{exc.category}: {exc}", require_reduction=require_reduction
        )
        self._reduction_required = require_reduction
        if repairs > _MAX_POLICY_REPAIRS:
            evidence = f"policy repairs exhausted: {exc}"
            self._tree.fail_active_tactical(evidence)
            self._orchestration_metrics["objectives_failed"] = (
                int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
            )
            self._boundary_reason = f"policy_repairs_exhausted:{objective_id}"
            self._reduction_required = True
            self._emit_event(
                "objective_failed",
                objective_id=objective_id,
                evidence=evidence,
            )
            if counts_activation_failure:
                self._orchestration_metrics[
                    "policy_activation_exhaustion_recoveries"
                ] = (
                    int(
                        self._orchestration_metrics.get(
                            "policy_activation_exhaustion_recoveries", 0
                        )
                    )
                    + 1
                )
                self._emit_event(
                    "policy_activation_exhaustion_routed_to_reducer",
                    objective_id=objective_id,
                    activation_failures=self._consecutive_activation_failures,
                    no_action_boundary_limit=_MAX_NO_ACTION_BOUNDARIES,
                )
            # This streak is scoped to the failed tactical leaf. The enclosing
            # analyze loop retains its independent no-action boundary ceiling.
            self._consecutive_activation_failures = 0
            self._failure_streak_objective_id = ""

    def _tactical_completion_evidence(self) -> tuple[bool, str]:
        """Decide whether host evidence is strong enough to resolve a tactical leaf."""

        assert self._tree is not None
        active = self._tree.active
        required_actions = active.minimum_evidence_actions
        transition = self._last_transition
        if not isinstance(transition, dict):
            return False, "no post-action transition has been observed"
        if str(transition.get("objective_id") or "") != active.objective_id:
            return False, "the latest transition belongs to another objective"
        if not bool(transition.get("executed")):
            return False, "the latest proposed action was not executed"
        if bool(transition.get("level_completed")) or bool(
            transition.get("run_complete")
        ):
            return True, "engine terminal progress confirms completion"
        outcome_class = str(transition.get("outcome_class") or "").strip().lower()
        if (
            active.evidence_mode is not ObjectiveEvidenceMode.CONTRASTIVE_TRANSITION
            and outcome_class in _NON_PROGRESS_OUTCOME_CLASSES
        ):
            return False, f"latest transition is non-progress outcome {outcome_class}"
        if active.actions_used < required_actions:
            return False, (
                f"only {active.actions_used} of {required_actions} required exploratory "
                "actions have post-action evidence"
            )
        relevant = [
            item
            for item in self._recent_transitions
            if str(item.get("objective_id") or "") == active.objective_id
            and bool(item.get("executed"))
        ]
        if len(relevant) < required_actions:
            return False, (
                f"only {len(relevant)} of {required_actions} required transition "
                "observations are retained"
            )
        if active.evidence_mode is ObjectiveEvidenceMode.CONTRASTIVE_TRANSITION:
            allowed, reason = contrastive_transition_evidence_status(
                active.to_dict(), relevant
            )
            if not allowed:
                return False, reason
            return True, "host minimum-evidence and " + reason
        if active.evidence_mode is ObjectiveEvidenceMode.STABLE_TRANSITION:
            allowed, reason = stable_transition_evidence_status(
                active.to_dict(), relevant
            )
            if not allowed:
                return False, reason
            return True, "host minimum-evidence and " + reason
        if not any(bool(item.get("meaningful_progress")) for item in relevant):
            return False, (
                "novel or changed states alone do not prove tactical completion; "
                "no controller-confirmed meaningful progress was observed"
            )
        return True, "host persistence and meaningful-progress requirements are met"

    def _tactical_failure_evidence(self) -> tuple[bool, str]:
        """Require bounded evidence before accepting an action-free failure claim."""

        assert self._tree is not None
        active = self._tree.active
        required_actions = active.minimum_evidence_actions
        transition = self._last_transition
        if (
            not isinstance(transition, dict)
            or str(transition.get("objective_id") or "") != active.objective_id
        ):
            return False, "no post-action evidence exists for this objective"
        if bool(transition.get("game_over")):
            return True, "engine game-over evidence confirms tactical failure"
        if active.remaining_actions <= 0 or active.actions_used >= required_actions:
            return True, "minimum failure evidence or tactical budget is exhausted"
        return False, (
            f"only {active.actions_used} of {required_actions} required exploratory "
            "actions have post-action evidence"
        )

    def _resolve_loop_guard_as_tactical_failure(
        self, transition: dict[str, Any]
    ) -> bool:
        """Route a mature controller loop guard to reduction instead of code repair."""

        if self._tree is None or self._tree.active.kind is not ObjectiveKind.TACTICAL:
            return False
        if str(transition.get("stop_reason") or "").strip() != "loop_guard":
            return False
        active = self._tree.active
        if active.actions_used < active.minimum_evidence_actions:
            return False
        objective_id = active.objective_id
        action = str(transition.get("action") or "unknown")
        point = ""
        if action == "MOUSE":
            point = f" at ({transition.get('row')},{transition.get('col')})"
        evidence = (
            f"controller loop guard falsified the tactical strategy after "
            f"{active.actions_used} executed evidence actions: {action}{point} "
            "would repeat a guarded cycle"
        )
        self._tree.fail_active_tactical(evidence)
        self._orchestration_metrics["objectives_failed"] = (
            int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
        )
        self._orchestration_metrics["guard_resolved_objectives"] = (
            int(self._orchestration_metrics.get("guard_resolved_objectives", 0)) + 1
        )
        self._consecutive_activation_failures = 0
        self._failure_streak_objective_id = ""
        self._invalidate_policy("loop_guard_falsified_objective", require_reduction=True)
        self._emit_event(
            "objective_guard_falsified",
            objective_id=objective_id,
            action=transition.get("action"),
            row=transition.get("row"),
            col=transition.get("col"),
            actions_used=active.actions_used,
            minimum_evidence_actions=active.minimum_evidence_actions,
            evidence=evidence,
        )
        self._emit_event(
            "objective_failed",
            objective_id=objective_id,
            evidence=evidence,
        )
        return True

    def _repair_premature_failure(
        self, *, policy_evidence: str, failure_reason: str
    ) -> bool:
        """Return True after scheduling one coder repair, False on repeated failure."""

        assert self._tree is not None
        objective_id = self._tree.active_id
        repairs = self._premature_failure_repairs.get(objective_id, 0)
        if repairs >= 1:
            return False
        self._premature_failure_repairs[objective_id] = repairs + 1
        self._orchestration_metrics["objective_failure_rejections"] = (
            int(self._orchestration_metrics.get("objective_failure_rejections", 0)) + 1
        )
        self._emit_event(
            "objective_failure_rejected",
            objective_id=objective_id,
            evidence=policy_evidence,
            reason=failure_reason,
            actions_used=self._tree.active.actions_used,
            action_budget=self._tree.active.action_budget,
        )
        self._policy_failure(
            PolicyRuntimeError(
                "premature subgoal_failed rejected by host: "
                f"{failure_reason}; continue with a different valid probe",
                category="premature_subgoal_failure",
            ),
            counts_activation_failure=False,
        )
        return True

    def _repair_repeated_action(
        self,
        *,
        action_payload: dict[str, Any],
        previous_outcome_class: str,
    ) -> bool:
        """Give one replacement policy the same leaf and remaining action budget."""

        assert self._tree is not None
        objective_id = self._tree.active_id
        repairs = self._repeated_action_repairs.get(objective_id, 0)
        if repairs >= 1:
            return False
        self._repeated_action_repairs[objective_id] = repairs + 1
        self._orchestration_metrics["repeated_action_repairs"] = (
            int(self._orchestration_metrics.get("repeated_action_repairs", 0)) + 1
        )
        self._emit_event(
            "policy_non_progress_repeat_repair",
            objective_id=objective_id,
            action=action_payload.get("action"),
            row=action_payload.get("row"),
            col=action_payload.get("col"),
            previous_outcome_class=previous_outcome_class,
            remaining_actions=self._tree.active.remaining_actions,
        )
        self._policy_failure(
            PolicyRuntimeError(
                "repeated non-progress action rejected by host; keep the same "
                "objective and remaining budget, inspect the latest transition, "
                "and choose a distinct valid action or coordinate",
                category="repeated_non_progress_action",
            ),
            counts_activation_failure=False,
        )
        return True

    def _adjudicate_rejected_completion(
        self,
        *,
        policy_evidence: str,
        completion_reason: str,
    ) -> None:
        """Repair one early positive claim, but fail non-progress or repeated claims."""

        assert self._tree is not None
        objective_id = self._tree.active_id
        transition = self._last_transition
        outcome_class = (
            str(transition.get("outcome_class") or "").strip().lower()
            if isinstance(transition, dict)
            and str(transition.get("objective_id") or "") == objective_id
            else ""
        )
        previous_repairs = self._premature_completion_repairs.get(objective_id, 0)
        fail_tactical = (
            outcome_class in _NON_PROGRESS_OUTCOME_CLASSES or previous_repairs >= 1
        )
        if not fail_tactical:
            self._premature_completion_repairs[objective_id] = previous_repairs + 1
            self._policy_failure(
                PolicyRuntimeError(
                    "premature subgoal_succeeded rejected by host: "
                    f"{completion_reason}; continue the same tactical objective and "
                    "satisfy its full success criteria",
                    category="premature_subgoal_success",
                ),
                counts_activation_failure=False,
            )
            return

        evidence = (
            "host reinterpreted rejected subgoal success as tactical failure: "
            f"{completion_reason}"
        )
        if policy_evidence:
            evidence += f"; policy evidence: {policy_evidence}"
        self._tree.fail_active_tactical(evidence)
        self._orchestration_metrics["objectives_failed"] = (
            int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
        )
        self._orchestration_metrics["objective_completion_reinterpretations"] = (
            int(
                self._orchestration_metrics.get(
                    "objective_completion_reinterpretations", 0
                )
            )
            + 1
        )
        self._consecutive_activation_failures = 0
        self._failure_streak_objective_id = ""
        self._invalidate_policy(
            f"rejected_subgoal_success:{objective_id}", require_reduction=True
        )
        self._emit_event(
            "objective_completion_reinterpreted",
            objective_id=objective_id,
            adjudication="failed",
            reason=completion_reason,
            outcome_class=outcome_class or None,
            prior_completion_repairs=previous_repairs,
        )
        self._emit_event(
            "objective_failed",
            objective_id=objective_id,
            evidence=evidence,
        )

    def analyze(
        self,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        transcript_path: Path | None = None,
        analysis_step: int | None = None,
        transcript_updated: Callable[[str], None] | None = None,
        request_timeout_seconds: float | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> AnalyzerTurnResult:
        del transcript_updated
        if not state_path.exists():
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category="state_missing",
                failure_detail=f"Runtime state does not exist: {state_path}",
            )
        previous_session = self._session_runtime_dir
        self._ensure_session(state_path)
        if previous_session != self._session_runtime_dir and self._tree is None:
            self._boundary_reason = "game_start"
        self._current_transcript_path = transcript_path or state_path.with_name(
            f"{state_path.stem}_analyzer.txt"
        )
        frame, history = load_runtime_state(state_path)
        if frame is None:
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category="state_missing",
                failure_detail="Runtime state contains no current frame.",
            )
        if valid_actions:
            frame = Frame(
                grid=frame.grid,
                step=frame.step,
                level=frame.level,
                valid_actions=tuple(
                    sorted({str(item).upper() for item in valid_actions})
                ),
                engine_state=frame.engine_state,
                score=frame.score,
            )
        if step_env is None:
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category="orchestration_missing_step_env",
                failure_detail="Orchestrated gameplay requires the guarded step_env callback.",
            )
        started = time.monotonic()
        hard_limit = request_timeout_seconds
        if hard_limit is None:
            hard_limit = self._timeout
        request_deadline = (
            started + hard_limit if hard_limit is not None and hard_limit > 0 else None
        )
        turn_deadline = (
            started + self._yield_seconds
            if self._yield_seconds is not None and self._yield_seconds > 0
            else None
        )
        self._turn_efficiency_metrics = {}
        self._should_stop_callback = should_stop
        self._append_transcript(
            "ORCHESTRATED TURN",
            {
                "analysis_step": analysis_step,
                "action_num": action_num,
                "boundary_reason": self._boundary_reason,
            },
        )
        try:
            self._ensure_tree(frame)
            assert self._tree is not None
            if frame.engine_state in {"WIN", "GAME_OVER"}:
                self._tree.resolve_game(
                    won=frame.engine_state == "WIN",
                    evidence=f"engine state {frame.engine_state}",
                )
                self._persist_durable_state()
                return AnalyzerTurnResult(
                    step_executed=False,
                    reasoning=f"Engine resolved game as {frame.engine_state}.",
                    efficiency_metrics=dict(self._orchestration_metrics),
                )
            if self._tree.remaining_level_actions <= 0:
                level = self._tree.current_level_objective
                detail = (
                    f"level {frame.level} exhausted its host action budget "
                    f"({level.actions_used}/{level.action_budget})"
                )
                self._boundary_reason = "level_action_budget_exhausted"
                self._emit_event(
                    "level_action_budget_exhausted",
                    level=frame.level,
                    actions_used=level.actions_used,
                    action_budget=level.action_budget,
                )
                self._persist_durable_state()
                return AnalyzerTurnResult(
                    step_executed=False,
                    exhausted=True,
                    failure_category="orchestration_level_action_budget_exhausted",
                    failure_detail=detail,
                    reasoning=detail,
                    efficiency_metrics=dict(self._orchestration_metrics),
                )
            if self._stagnated(frame, history) and self._policy_runtime is not None:
                self._invalidate_policy(
                    "controller_stagnation_window", require_reduction=True
                )
            with contextlib.suppress(PolicyRuntimeError):
                self._restore_policy_if_possible()

            no_action_boundaries = 0
            while no_action_boundaries < _MAX_NO_ACTION_BOUNDARIES:
                remaining = self._remaining_seconds(turn_deadline)
                if remaining is not None and remaining <= 0:
                    raise OrchestrationYield("turn time budget")
                if (
                    self._tree.active.kind is not ObjectiveKind.TACTICAL
                    or self._reduction_required
                ):
                    self._reduce(
                        frame,
                        history,
                        request_deadline=request_deadline,
                        should_stop=should_stop,
                    )
                    no_action_boundaries += 1
                    continue
                budget_exhausted = self._tree.active.remaining_actions <= 0
                if budget_exhausted and (
                    self._policy_runtime is None
                    or self._policy_objective_id != self._tree.active_id
                ):
                    objective_id = self._tree.active_id
                    evidence = (
                        "tactical action budget exhausted before a final policy "
                        "evaluation was available"
                    )
                    self._tree.fail_active_tactical(evidence)
                    self._orchestration_metrics["objectives_failed"] = (
                        int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
                    )
                    self._invalidate_policy("tactical_action_budget_exhausted")
                    self._emit_event(
                        "objective_failed",
                        objective_id=objective_id,
                        evidence=evidence,
                    )
                    no_action_boundaries += 1
                    continue
                if (
                    self._policy_runtime is None
                    or self._policy_objective_id != self._tree.active_id
                ):
                    repair_reason = self._boundary_reason
                    try:
                        self._activate_policy(
                            frame,
                            history,
                            request_deadline=request_deadline,
                            should_stop=should_stop,
                            repair_reason=repair_reason,
                        )
                    except PolicyRuntimeError as exc:
                        self._policy_failure(exc, counts_activation_failure=True)
                        no_action_boundaries += 1
                        continue
                    remaining = self._remaining_seconds(turn_deadline)
                    if remaining is not None and remaining <= 0:
                        raise OrchestrationYield("turn time budget")
                assert self._policy_runtime is not None
                try:
                    policy_started = time.monotonic()
                    decision: PolicyDecision = self._policy_runtime.decide(
                        self._observation(frame)
                    )
                    policy_seconds = time.monotonic() - policy_started
                    backend = (
                        self._policy_runtime.activation.backend
                        if self._policy_runtime.activation is not None
                        else "cpu"
                    )
                    metric = f"{backend}_policy_seconds"
                    self._orchestration_metrics[metric] = (
                        float(self._orchestration_metrics.get(metric, 0.0))
                        + policy_seconds
                    )
                    self._policy_memory = self._policy_runtime.memory
                except PolicyRuntimeError as exc:
                    self._policy_failure(
                        exc,
                        counts_activation_failure=not self._policy_executed_action,
                    )
                    no_action_boundaries += 1
                    continue

                if decision.status is PolicyStatus.SUBGOAL_SUCCEEDED:
                    objective_id = self._tree.active_id
                    completion_allowed, completion_reason = (
                        self._tactical_completion_evidence()
                    )
                    if not completion_allowed:
                        self._orchestration_metrics[
                            "objective_completion_rejections"
                        ] = (
                            int(
                                self._orchestration_metrics.get(
                                    "objective_completion_rejections", 0
                                )
                            )
                            + 1
                        )
                        self._emit_event(
                            "objective_completion_rejected",
                            objective_id=objective_id,
                            evidence=decision.evidence,
                            reason=completion_reason,
                            actions_used=self._tree.active.actions_used,
                            action_budget=self._tree.active.action_budget,
                            latest_outcome_class=(
                                self._last_transition.get("outcome_class")
                                if self._last_transition is not None
                                else None
                            ),
                        )
                        self._adjudicate_rejected_completion(
                            policy_evidence=decision.evidence,
                            completion_reason=completion_reason,
                        )
                        no_action_boundaries += 1
                        continue
                    self._consecutive_activation_failures = 0
                    self._failure_streak_objective_id = ""
                    self._tree.complete_active_tactical(decision.evidence)
                    self._orchestration_metrics["objectives_completed"] = (
                        int(self._orchestration_metrics.get("objectives_completed", 0))
                        + 1
                    )
                    self._invalidate_policy(f"subgoal_succeeded:{objective_id}")
                    self._emit_event(
                        "objective_completed",
                        objective_id=objective_id,
                        evidence=decision.evidence,
                    )
                    no_action_boundaries += 1
                    continue
                if decision.status is PolicyStatus.SUBGOAL_FAILED:
                    objective_id = self._tree.active_id
                    failure_allowed, failure_reason = self._tactical_failure_evidence()
                    if not failure_allowed and self._repair_premature_failure(
                        policy_evidence=decision.evidence,
                        failure_reason=failure_reason,
                    ):
                        no_action_boundaries += 1
                        continue
                    self._consecutive_activation_failures = 0
                    self._failure_streak_objective_id = ""
                    evidence = decision.evidence
                    if not failure_allowed:
                        evidence = (
                            "repeated premature subgoal failure accepted as tactical "
                            f"failure after one coder repair: {failure_reason}; "
                            f"policy evidence: {decision.evidence}"
                        )
                    self._tree.fail_active_tactical(evidence)
                    self._orchestration_metrics["objectives_failed"] = (
                        int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
                    )
                    self._invalidate_policy(f"subgoal_failed:{objective_id}")
                    self._emit_event(
                        "objective_failed",
                        objective_id=objective_id,
                        evidence=evidence,
                    )
                    no_action_boundaries += 1
                    continue

                if budget_exhausted:
                    objective_id = self._tree.active_id
                    evidence = (
                        "policy requested another action during the final post-budget "
                        "evaluation"
                    )
                    if decision.evidence:
                        evidence += f": {decision.evidence}"
                    self._tree.fail_active_tactical(evidence)
                    self._orchestration_metrics["objectives_failed"] = (
                        int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
                    )
                    self._invalidate_policy("tactical_action_budget_exhausted")
                    self._emit_event(
                        "objective_failed",
                        objective_id=objective_id,
                        evidence=evidence,
                    )
                    no_action_boundaries += 1
                    continue

                action_payload = dict(decision.action or {})
                action_payload["strategy_prediction"] = {
                    "objective_id": self._tree.active_id,
                    "objective_title": self._tree.active.title,
                    "policy_hash": self._policy_source_hash,
                    "solver_type": self._policy_solver_type,
                    "solver_family": self._policy_solver_family,
                    "policy_backend": self._policy_runtime.activation.backend
                    if self._policy_runtime.activation is not None
                    else "cpu",
                    "evidence": decision.evidence,
                    **(decision.prediction or {}),
                }
                if _repeats_non_progress_action(
                    self._last_transition,
                    action_payload,
                    objective_id=self._tree.active_id,
                ):
                    objective_id = self._tree.active_id
                    evidence = (
                        "policy repeated the same action after a non-progress "
                        f"{self._last_transition.get('outcome_class') or 'guarded'} "
                        "transition instead of evaluating the post-action evidence"
                    )
                    previous_outcome_class = str(
                        self._last_transition.get("outcome_class") or "guarded"
                    )
                    if self._repair_repeated_action(
                        action_payload=action_payload,
                        previous_outcome_class=previous_outcome_class,
                    ):
                        no_action_boundaries += 1
                        continue
                    self._tree.fail_active_tactical(evidence)
                    self._orchestration_metrics["objectives_failed"] = (
                        int(self._orchestration_metrics.get("objectives_failed", 0)) + 1
                    )
                    self._invalidate_policy("repeated_non_progress_action")
                    self._emit_event(
                        "policy_non_progress_repeat_rejected",
                        objective_id=objective_id,
                        action=action_payload.get("action"),
                        row=action_payload.get("row"),
                        col=action_payload.get("col"),
                        previous_outcome_class=previous_outcome_class,
                    )
                    self._emit_event(
                        "objective_failed",
                        objective_id=objective_id,
                        evidence=evidence,
                    )
                    no_action_boundaries += 1
                    continue
                saturation_reason = _action_family_saturation_reason(
                    self._level_action_evidence, action_payload.get("action")
                )
                if saturation_reason:
                    objective_id = self._tree.active_id
                    evidence = (
                        "policy action rejected by the host action-family guard: "
                        f"{saturation_reason}"
                    )
                    self._tree.fail_active_tactical(evidence)
                    self._orchestration_metrics["objectives_failed"] = (
                        int(self._orchestration_metrics.get("objectives_failed", 0))
                        + 1
                    )
                    self._orchestration_metrics[
                        "action_family_saturation_guards"
                    ] = (
                        int(
                            self._orchestration_metrics.get(
                                "action_family_saturation_guards", 0
                            )
                        )
                        + 1
                    )
                    self._invalidate_policy(
                        "action_family_saturated", require_reduction=True
                    )
                    self._emit_event(
                        "action_family_saturated",
                        objective_id=objective_id,
                        action=action_payload.get("action"),
                        row=action_payload.get("row"),
                        col=action_payload.get("col"),
                        detail=saturation_reason,
                    )
                    self._emit_event(
                        "objective_failed",
                        objective_id=objective_id,
                        evidence=evidence,
                    )
                    no_action_boundaries += 1
                    continue
                result = step_env(action_payload)
                transition = _policy_transition_payload(
                    action_payload,
                    result,
                    objective_id=self._tree.active_id,
                    policy_hash=self._policy_source_hash,
                )
                self._last_transition = transition
                self._recent_transitions.append(transition)
                del self._recent_transitions[:-8]
                self._record_level_action_evidence(transition)
                if bool(transition.get("meaningful_progress")):
                    self._policy_observed_host_progress = True
                self._emit_event("gameplay_decision", **transition)
                if result.get("executed"):
                    self._tree.record_action()
                    self._policy_executed_action = True
                    self._consecutive_activation_failures = 0
                    self._failure_streak_objective_id = ""
                    self._orchestration_metrics["policy_steps"] = (
                        int(self._orchestration_metrics.get("policy_steps", 0)) + 1
                    )
                    if result.get("run_complete"):
                        self._tree.resolve_game(
                            won=True,
                            evidence="guarded policy action produced engine run_complete",
                        )
                    if result.get("level_completed") or result.get("run_complete"):
                        self._invalidate_policy(
                            "run_complete"
                            if result.get("run_complete")
                            else "level_completed"
                        )
                    self._persist_durable_state()
                    return AnalyzerTurnResult(
                        step_executed=True,
                        reasoning=decision.evidence,
                        efficiency_metrics=dict(self._orchestration_metrics),
                        attempts=1,
                    )
                if self._resolve_loop_guard_as_tactical_failure(transition):
                    no_action_boundaries += 1
                    continue
                self._policy_failure(
                    PolicyRuntimeError(
                        transition["error"]
                        or transition["stop_reason"]
                        or "guarded action was not executed",
                        category="guarded_action",
                    ),
                    counts_activation_failure=False,
                )
                no_action_boundaries += 1

            raise OrchestrationFailure(
                "orchestration crossed too many no-action objective boundaries in one turn",
                category="orchestration_no_action_exhausted",
            )
        except OrchestrationYield as exc:
            self._persist_durable_state()
            return AnalyzerTurnResult(
                step_executed=False,
                yielded_control=True,
                yield_reason="turn_time_budget",
                reasoning=str(exc),
                efficiency_metrics=dict(self._orchestration_metrics),
            )
        except requests.RequestException as exc:
            self._persist_durable_state()
            return AnalyzerTurnResult(
                step_executed=False,
                retryable_failure=True,
                failure_category="orchestration_model_transport",
                failure_detail=str(exc),
                efficiency_metrics=dict(self._orchestration_metrics),
            )
        except (OrchestrationFailure, ObjectiveError) as exc:
            category = getattr(exc, "category", "orchestration_objective_error")
            self._emit_event(
                "orchestration_exhausted", category=category, detail=str(exc)
            )
            self._persist_durable_state()
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category=category,
                failure_detail=str(exc),
                efficiency_metrics=dict(self._orchestration_metrics),
            )
        except Exception as exc:  # noqa: BLE001 - convert to visible analyzer failure
            log.exception("orchestrated objective analyzer failed")
            self._persist_durable_state()
            return AnalyzerTurnResult(
                step_executed=False,
                exhausted=True,
                failure_category="orchestration_internal",
                failure_detail=f"{type(exc).__name__}: {exc}",
                efficiency_metrics=dict(self._orchestration_metrics),
            )
        finally:
            self._should_stop_callback = None
