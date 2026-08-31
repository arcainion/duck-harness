"""Sandbox-safe solver families for generated gameplay policies.

The public labels mirror the dynamics solvers in ``agi_core.knowledge.solvers``.
The implementations here are deliberately self-contained: they operate only on a
``PolicyObservation``-shaped value, bounded JSON memory, and trusted policy helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from inference.agent.objective_reduction import GameSolverType
from inference.agent.policy_codegen_helpers import (
    POLICY_ACTIONS,
    board_digest,
    transition_has_progress,
    transition_repeats_nonprogress_action,
    transition_requires_replan,
)
from inference.agent.policy_pathfinding import (
    clearance_mask,
    component_centers,
    find_cells,
    line_of_sight,
    next_path_action,
    shortest_approach_path,
    shortest_path_to_any,
    value_mask,
)

POLICY_SOLVER_API_VERSION = 1

_SOLVER_FAMILIES = {
    "beam": "field",
    "beam-coverage": "coverage",
    "carrier-placement": "manipulation",
    "cellular-automata": "observe",
    "click-interaction": "interaction",
    "connector-align": "alignment",
    "cycle-rotation": "sequence",
    "flow-deflector": "field",
    "glyph-transform-route": "routing",
    "gravity": "gravity",
    "guided-attraction": "routing",
    "hybrid": "hybrid",
    "inertial-block": "momentum",
    "inventory": "manipulation",
    "lattice-corridor": "routing",
    "linked-centroid": "alignment",
    "marker-coverage": "coverage",
    "mirror": "field",
    "mirror-merge": "alignment",
    "multi-agent": "multi_agent",
    "navigation": "routing",
    "paired-platform-alignment": "alignment",
    "paired-sequence-arm": "sequence",
    "pattern-transform": "transform",
    "peg-jump": "sequence",
    "puzzle": "field",
    "push-pull": "manipulation",
    "relation-toggle": "interaction",
    "signal": "interaction",
    "sliding": "momentum",
    "static": "observe",
    "switch-bridge": "interaction",
    "symbol-rule-sequence": "sequence",
    "template-paint": "coverage",
    "trajectory-replay": "momentum",
    "transform-program": "transform",
}

POLICY_SOLVER_TYPES = tuple(sorted(_SOLVER_FAMILIES))
POLICY_SOLVER_FAMILIES = MappingProxyType(dict(_SOLVER_FAMILIES))
_OBJECTIVE_SOLVER_TYPES = frozenset(item.value for item in GameSolverType)
if frozenset(POLICY_SOLVER_TYPES) != _OBJECTIVE_SOLVER_TYPES:
    missing = sorted(_OBJECTIVE_SOLVER_TYPES.difference(POLICY_SOLVER_TYPES))
    extra = sorted(set(POLICY_SOLVER_TYPES).difference(_OBJECTIVE_SOLVER_TYPES))
    raise RuntimeError(
        f"policy solver catalog differs from GameSolverType; missing={missing}, extra={extra}"
    )
NAVIGATION_SOLVER_FAMILIES = frozenset(
    {"alignment", "coverage", "field", "gravity", "manipulation", "momentum", "multi_agent", "routing"}
)

_COLOR_KEYS = frozenset(
    {
        "actor_values",
        "coverage_values",
        "hazard_values",
        "interactive_values",
        "passable_values",
        "source_values",
        "target_values",
    }
)
_ACTION_LIST_KEYS = frozenset({"interaction_actions", "probe_actions"})
_KNOWN_CONFIG_KEYS = frozenset(
    {
        *_COLOR_KEYS,
        *_ACTION_LIST_KEYS,
        "action_sequences",
        "approach_distance",
        "clearance_radius",
        "fallback_configs",
        "fallback_types",
        "max_plan_length",
    }
)
_MAX_COLOR_VALUES = 16
_MAX_ACTIONS = 32
_MAX_SEQUENCES = 8
_MAX_FALLBACKS = 8


def solver_family(solver_type: Any) -> str:
    """Return the portable family for one registered dynamics label."""

    normalized = str(solver_type or "").strip().lower()
    try:
        return _SOLVER_FAMILIES[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown policy solver type {normalized!r}") from exc


def _bounded_colors(value: Any, key: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"solver config {key} must be a list of color values")
    if len(value) > _MAX_COLOR_VALUES:
        raise ValueError(f"solver config {key} may contain at most {_MAX_COLOR_VALUES} values")
    result: list[int] = []
    for item in value:
        if type(item) is not int:
            raise ValueError(f"solver config {key} values must be JSON integers")
        color = item
        if not 0 <= color <= 255:
            raise ValueError(f"solver config {key} values must be between 0 and 255")
        if color not in result:
            result.append(color)
    return tuple(result)


def _bounded_actions(value: Any, key: str, *, maximum: int = _MAX_ACTIONS) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"solver config {key} must be a list of actions")
    if len(value) > maximum:
        raise ValueError(f"solver config {key} may contain at most {maximum} actions")
    result: list[str] = []
    for item in value:
        action = str(item or "").strip().upper()
        if action not in POLICY_ACTIONS:
            raise ValueError(f"solver config {key} contains unknown action {action!r}")
        result.append(action)
    return tuple(result)


def validate_solver_config(solver_type: Any, config: Any) -> dict[str, Any]:
    """Validate and normalize one finite-JSON solver configuration."""

    normalized_type = str(solver_type or "").strip().lower()
    family = solver_family(normalized_type)
    if not isinstance(config, Mapping):
        raise ValueError("policy solver config must be a mapping")
    unknown = set(str(key) for key in config).difference(_KNOWN_CONFIG_KEYS)
    if unknown:
        raise ValueError("unknown policy solver config key(s): " + ", ".join(sorted(unknown)))

    result: dict[str, Any] = {}
    for key in _COLOR_KEYS:
        result[key] = list(_bounded_colors(config.get(key), key))
    for key in _ACTION_LIST_KEYS:
        result[key] = list(_bounded_actions(config.get(key), key))

    approach = config.get("approach_distance", 0)
    clearance = config.get("clearance_radius", 0)
    max_plan = config.get("max_plan_length", 128)
    if isinstance(approach, bool) or not isinstance(approach, int) or not 0 <= approach <= 8:
        raise ValueError("solver config approach_distance must be an integer from 0 to 8")
    if isinstance(max_plan, bool) or not isinstance(max_plan, int) or not 1 <= max_plan <= 4096:
        raise ValueError("solver config max_plan_length must be an integer from 1 to 4096")
    if (
        isinstance(clearance, bool)
        or not isinstance(clearance, int)
        or not 0 <= clearance <= 8
    ):
        raise ValueError("solver config clearance_radius must be an integer from 0 to 8")
    result["approach_distance"] = approach
    result["clearance_radius"] = clearance
    result["max_plan_length"] = max_plan

    raw_sequences = config.get("action_sequences") or []
    if isinstance(raw_sequences, (str, bytes)) or not isinstance(raw_sequences, Sequence):
        raise ValueError("solver config action_sequences must be a list of action lists")
    if len(raw_sequences) > _MAX_SEQUENCES:
        raise ValueError(f"solver config action_sequences may contain at most {_MAX_SEQUENCES} sequences")
    result["action_sequences"] = [
        list(_bounded_actions(sequence, "action_sequences", maximum=_MAX_ACTIONS))
        for sequence in raw_sequences
    ]

    raw_fallbacks = config.get("fallback_types") or []
    if isinstance(raw_fallbacks, (str, bytes)) or not isinstance(raw_fallbacks, Sequence):
        raise ValueError("solver config fallback_types must be a list")
    if len(raw_fallbacks) > _MAX_FALLBACKS:
        raise ValueError(f"solver config fallback_types may contain at most {_MAX_FALLBACKS} values")
    fallbacks: list[str] = []
    for item in raw_fallbacks:
        candidate = str(item or "").strip().lower()
        solver_family(candidate)
        if candidate == "hybrid":
            raise ValueError("hybrid solver fallback may not recursively select hybrid")
        if candidate not in fallbacks:
            fallbacks.append(candidate)
    result["fallback_types"] = fallbacks

    raw_fallback_configs = config.get("fallback_configs") or {}
    if not isinstance(raw_fallback_configs, Mapping):
        raise ValueError("solver config fallback_configs must be a mapping")
    if set(str(key) for key in raw_fallback_configs).difference(fallbacks):
        raise ValueError("solver fallback_configs keys must appear in fallback_types")
    result["fallback_configs"] = {
        candidate: validate_solver_config(candidate, raw_fallback_configs.get(candidate) or {})
        for candidate in fallbacks
    }

    if family == "hybrid" and not fallbacks:
        raise ValueError("hybrid solver config requires at least one fallback type")
    if family == "sequence" and not result["action_sequences"]:
        raise ValueError("sequence solver config requires action_sequences")
    if family == "observe" and not result["probe_actions"]:
        raise ValueError("observe solver config requires probe_actions")
    if family == "interaction" and not (
        result["interactive_values"]
        or result["interaction_actions"]
        or result["probe_actions"]
    ):
        raise ValueError(
            "interaction solver config requires interactive_values or configured actions"
        )
    if family == "sequence" and not (
        result["interaction_actions"] or result["probe_actions"] or result["action_sequences"]
    ):
        raise ValueError(f"{family} solver config requires configured actions")
    if family in {"routing", "momentum", "gravity"}:
        missing = [
            key
            for key in ("actor_values", "target_values", "passable_values")
            if not result[key]
        ]
        if missing:
            raise ValueError(
                f"{family} solver config requires " + ", ".join(missing)
            )
    if family in {"alignment", "manipulation", "multi_agent"}:
        missing = [
            key for key in ("actor_values", "passable_values") if not result[key]
        ]
        if not (result["target_values"] or result["interactive_values"]):
            missing.append("target_values or interactive_values")
        if missing:
            raise ValueError(
                f"{family} solver config requires " + ", ".join(missing)
            )
    if family == "coverage" and not result["target_values"]:
        raise ValueError("coverage solver config requires target_values")
    if family == "field":
        missing = [
            key for key in ("target_values", "passable_values") if not result[key]
        ]
        if not (result["source_values"] or result["actor_values"]):
            missing.append("source_values or actor_values")
        if missing:
            raise ValueError("field solver config requires " + ", ".join(missing))
    if family == "transform" and not (
        result["interactive_values"] or result["target_values"]
    ):
        raise ValueError(
            "transform solver config requires interactive_values or target_values"
        )
    return result


def _memory(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _continue(action: str, memory: Mapping[str, Any], evidence: str, **prediction: Any) -> dict[str, Any]:
    return {
        "status": "continue",
        "action": {"action": action},
        "memory": dict(memory),
        "evidence": evidence,
        "prediction": prediction or None,
    }


def _mouse(point: tuple[int, int], memory: Mapping[str, Any], evidence: str) -> dict[str, Any]:
    return {
        "status": "continue",
        "action": {"action": "MOUSE", "row": point[0], "col": point[1]},
        "memory": dict(memory),
        "evidence": evidence,
    }


def _terminal(status: str, memory: Mapping[str, Any], evidence: str) -> dict[str, Any]:
    return {"status": status, "action": None, "memory": dict(memory), "evidence": evidence}


def _valid_action(observation: Any, candidates: Sequence[str]) -> str | None:
    valid = set(str(item).upper() for item in observation.valid_actions)
    return next((item for item in candidates if item in valid), None)


def _points(board: np.ndarray, values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return find_cells(board, values) if values else ()


def _passability(board: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    values = config["passable_values"] or config["actor_values"]
    passable = np.array(value_mask(board, values), dtype=bool, copy=True)
    if config["hazard_values"]:
        passable[value_mask(board, config["hazard_values"])] = False
    radius = int(config["clearance_radius"])
    if radius:
        passable = np.array(clearance_mask(passable, radius=radius), copy=True)
    return passable


def _route_decision(
    observation: Any,
    memory: dict[str, Any],
    config: Mapping[str, Any],
    *,
    approach: bool = False,
) -> dict[str, Any]:
    board = observation.board
    actors = _points(board, config["actor_values"])
    targets = _points(board, config["target_values"])
    if not actors or not targets:
        return _terminal("subgoal_failed", memory, "solver could not localize actor and target")
    actor = min(
        actors,
        key=lambda candidate: (
            min(
                abs(candidate[0] - target[0]) + abs(candidate[1] - target[1])
                for target in targets
            ),
            candidate,
        ),
    )
    passable = _passability(board, config)
    passable[actor] = True
    for target in targets:
        passable[target] = True
    if transition_requires_replan(observation.last_transition):
        memory.pop("path", None)
    distance = int(config["approach_distance"])
    path = (
        shortest_approach_path(passable, actor, targets, distance=distance)
        if approach or distance > 0
        else shortest_path_to_any(passable, actor, targets)
    )
    if not path:
        return _terminal("subgoal_failed", memory, "solver found no traversable route")
    if len(path) == 1:
        return _terminal("subgoal_succeeded", memory, "solver reached the configured target")
    path = path[: int(config["max_plan_length"])]
    action = next_path_action(path, observation.valid_actions)
    if action is None:
        return _terminal("subgoal_failed", memory, "solver route has no currently valid action")
    memory.update(
        {
            "actor": list(actor),
            "path": [list(point) for point in path],
            "board_digest": board_digest(board),
            "last_action": action,
        }
    )
    return _continue(action, memory, "trusted solver followed a board-derived route", target=list(path[-1]))


def _momentum_decision(observation: Any, memory: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    actors = _points(observation.board, config["actor_values"])
    if not actors:
        return _terminal("subgoal_failed", memory, "momentum solver could not localize actor")
    actor = actors[0]
    previous = memory.get("actor")
    if isinstance(previous, list) and len(previous) == 2:
        velocity = (actor[0] - int(previous[0]), actor[1] - int(previous[1]))
        if velocity != (0, 0):
            memory["velocity"] = list(velocity)
            predicted = (actor[0] + velocity[0], actor[1] + velocity[1])
            passable = _passability(observation.board, config)
            normalized_velocity = (
                (
                    -1 if velocity[0] < 0 else 1,
                    0,
                )
                if velocity[0] and not velocity[1]
                else (
                    (0, -1 if velocity[1] < 0 else 1)
                    if velocity[1] and not velocity[0]
                    else velocity
                )
            )
            height, width = passable.shape
            if not (
                0 <= predicted[0] < height
                and 0 <= predicted[1] < width
                and bool(passable[predicted])
            ):
                inverse = {
                    (-1, 0): "DOWN",
                    (1, 0): "UP",
                    (0, -1): "RIGHT",
                    (0, 1): "LEFT",
                }.get(normalized_velocity)
                if inverse and inverse in observation.valid_actions:
                    memory["actor"] = list(actor)
                    return _continue(inverse, memory, "momentum solver issued collision compensation")
    memory["actor"] = list(actor)
    return _route_decision(observation, memory, config)


def _gravity_decision(observation: Any, memory: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    actors = _points(observation.board, config["actor_values"])
    targets = _points(observation.board, config["target_values"])
    if not actors or not targets:
        return _terminal("subgoal_failed", memory, "gravity solver could not localize actor and landing target")
    actor = actors[0]
    target = min(targets, key=lambda point: (abs(point[1] - actor[1]), abs(point[0] - actor[0]), point))
    if target[1] != actor[1]:
        action = "RIGHT" if target[1] > actor[1] else "LEFT"
        if action in observation.valid_actions:
            if transition_repeats_nonprogress_action(
                observation.last_transition, action
            ):
                return _terminal(
                    "subgoal_failed",
                    memory,
                    "gravity solver rejected a repeated blocked lane correction",
                )
            memory.update(actor=list(actor), target=list(target), last_action=action)
            return _continue(action, memory, "gravity solver aligned the actor with a landing lane")
    interaction = _valid_action(observation, config["interaction_actions"])
    if interaction:
        return _continue(interaction, memory, "gravity solver activated the aligned support interaction")
    return _route_decision(observation, memory, config)


def _interaction_decision(observation: Any, memory: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if transition_has_progress(observation.last_transition):
        return _terminal("subgoal_succeeded", memory, "interaction solver observed meaningful progress")
    points = (
        tuple(
            point
            for point in component_centers(
                value_mask(observation.board, config["interactive_values"])
            )
            if 2 <= point[0] <= 61 and 2 <= point[1] <= 61
        )
        if config["interactive_values"]
        else ()
    )
    tried = {tuple(item) for item in memory.get("tried_points", []) if isinstance(item, list) and len(item) == 2}
    if "MOUSE" in observation.valid_actions:
        target = next((point for point in points if point not in tried), None)
        if target is not None:
            tried.add(target)
            memory["tried_points"] = [list(point) for point in sorted(tried)][-64:]
            return _mouse(target, memory, "interaction solver selected an untried component")
    actions = config["interaction_actions"] or config["probe_actions"]
    counts = dict(memory.get("action_counts") or {})
    available = [
        action
        for action in actions
        if action in observation.valid_actions
        and not transition_repeats_nonprogress_action(
            observation.last_transition, action
        )
    ]
    if not available:
        return _terminal("subgoal_failed", memory, "interaction solver has no valid configured action")
    action = min(available, key=lambda item: (int(counts.get(item, 0)), item))
    counts[action] = int(counts.get(action, 0)) + 1
    memory["action_counts"] = counts
    return _continue(action, memory, "interaction solver selected the least-tried valid action")


def _approach_then_interact(
    observation: Any, memory: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    actors = _points(observation.board, config["actor_values"])
    targets = _points(observation.board, config["target_values"] or config["interactive_values"])
    if not actors or not targets:
        return _terminal("subgoal_failed", memory, "solver could not localize movable and destination components")
    distance = min(abs(actors[0][0] - point[0]) + abs(actors[0][1] - point[1]) for point in targets)
    if distance <= int(config["approach_distance"]):
        action = _valid_action(observation, config["interaction_actions"])
        if action:
            return _continue(action, memory, "solver reached the interaction distance")
        return _terminal("subgoal_succeeded", memory, "solver aligned the configured components")
    return _route_decision(observation, memory, config, approach=True)


def _coverage_decision(observation: Any, memory: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if transition_has_progress(observation.last_transition):
        return _terminal("subgoal_succeeded", memory, "coverage solver observed meaningful progress")
    targets = _points(observation.board, config["target_values"])
    covered = set(_points(observation.board, config["coverage_values"]))
    remaining = tuple(point for point in targets if point not in covered)
    if not remaining:
        return _terminal("subgoal_succeeded", memory, "coverage solver found no uncovered targets")
    if "MOUSE" in observation.valid_actions and (not config["actor_values"]):
        safe = next(
            (point for point in remaining if 2 <= point[0] <= 61 and 2 <= point[1] <= 61),
            None,
        )
        if safe is None:
            return _terminal(
                "subgoal_failed", memory, "coverage targets are confined to the HUD edge"
            )
        return _mouse(safe, memory, "coverage solver selected the next uncovered target")
    patched = dict(config)
    patched["target_values"] = list(config["target_values"])
    return _route_decision(observation, memory, patched, approach=True)


def _sequence_decision(observation: Any, memory: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if transition_has_progress(observation.last_transition):
        return _terminal("subgoal_succeeded", memory, "sequence solver observed meaningful progress")
    sequences = config["action_sequences"]
    sequence_index = min(max(0, int(memory.get("sequence_index", 0))), len(sequences) - 1)
    offset = max(0, int(memory.get("sequence_offset", 0)))
    if transition_requires_replan(observation.last_transition) and offset:
        sequence_index += 1
        offset = 0
    while sequence_index < len(sequences):
        sequence = sequences[sequence_index]
        while offset < len(sequence):
            action = sequence[offset]
            offset += 1
            if action in observation.valid_actions:
                memory.update(sequence_index=sequence_index, sequence_offset=offset)
                return _continue(action, memory, "sequence solver executed a bounded candidate")
        sequence_index += 1
        offset = 0
    return _terminal("subgoal_failed", memory, "sequence solver exhausted all bounded candidates")


def _field_decision(observation: Any, memory: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    sources = _points(observation.board, config["source_values"] or config["actor_values"])
    targets = _points(observation.board, config["target_values"])
    if not sources or not targets:
        return _terminal("subgoal_failed", memory, "field solver could not localize source and target")
    passable = _passability(observation.board, config)
    pair = min(((source, target) for source in sources for target in targets), key=lambda pair: abs(pair[0][0]-pair[1][0]) + abs(pair[0][1]-pair[1][1]))
    if line_of_sight(passable, pair[0], pair[1]):
        action = _valid_action(observation, config["interaction_actions"])
        if action:
            return _continue(action, memory, "field solver activated a clear source-target line")
        return _terminal("subgoal_succeeded", memory, "field solver verified source-target line of sight")
    if config["interactive_values"]:
        return _interaction_decision(observation, memory, config)
    if config["actor_values"]:
        return _approach_then_interact(observation, memory, config)
    return _terminal(
        "subgoal_failed",
        memory,
        "field solver found a blocked line without a controllable deflector",
    )


def _transform_decision(
    observation: Any, memory: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if transition_has_progress(observation.last_transition):
        return _terminal("subgoal_succeeded", memory, "transform solver observed progress")
    if config["interactive_values"]:
        return _interaction_decision(observation, memory, config)
    targets = _points(observation.board, config["target_values"])
    if config["actor_values"]:
        return _route_decision(observation, memory, config, approach=True)
    if "MOUSE" in observation.valid_actions:
        target = next(
            (point for point in targets if 2 <= point[0] <= 61 and 2 <= point[1] <= 61),
            None,
        )
        if target is not None:
            return _mouse(target, memory, "transform solver selected a mismatched target")
    return _terminal(
        "subgoal_failed", memory, "transform solver has no safe controllable target"
    )


def _observe_decision(observation: Any, memory: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if transition_has_progress(observation.last_transition):
        return _terminal("subgoal_succeeded", memory, "observation solver detected progress")
    index = max(0, int(memory.get("probe_index", 0)))
    probes = config["probe_actions"]
    while index < len(probes):
        action = probes[index]
        index += 1
        if action in observation.valid_actions:
            memory["probe_index"] = index
            return _continue(action, memory, "observation solver issued a bounded probe")
    return _terminal("subgoal_failed", memory, "observation solver exhausted bounded probes")


def _dispatch(solver_type: str, observation: Any, memory: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    family = solver_family(solver_type)
    if family == "routing":
        return _route_decision(observation, memory, config)
    if family == "momentum":
        return _momentum_decision(observation, memory, config)
    if family == "gravity":
        return _gravity_decision(observation, memory, config)
    if family in {"manipulation", "alignment", "multi_agent"}:
        return _approach_then_interact(observation, memory, config)
    if family == "interaction":
        return _interaction_decision(observation, memory, config)
    if family == "coverage":
        return _coverage_decision(observation, memory, config)
    if family == "sequence":
        return _sequence_decision(observation, memory, config)
    if family == "field":
        return _field_decision(observation, memory, config)
    if family == "transform":
        return _transform_decision(observation, memory, config)
    if family == "observe":
        return _observe_decision(observation, memory, config)
    if family == "hybrid":
        states = dict(memory.get("fallback_memory") or {})
        failures: list[str] = []
        for fallback in config["fallback_types"]:
            child_memory = _memory(states.get(fallback))
            result = _dispatch(
                fallback,
                observation,
                child_memory,
                config["fallback_configs"][fallback],
            )
            states[fallback] = result.get("memory", {})
            if result.get("status") != "subgoal_failed":
                result["memory"] = {"fallback_memory": states, "active_solver": fallback}
                return result
            failures.append(fallback)
        return _terminal("subgoal_failed", {"fallback_memory": states}, "hybrid solver exhausted: " + ", ".join(failures))
    raise ValueError(f"unsupported solver family {family!r}")


def solver_decide(solver_type: Any, observation: Any, memory: Any, config: Any) -> dict[str, Any]:
    """Run one bounded trusted solver decision."""

    normalized_type = str(solver_type or "").strip().lower()
    normalized_config = validate_solver_config(normalized_type, config)
    return _dispatch(normalized_type, observation, _memory(memory), normalized_config)


POLICY_SOLVER_GLOBALS = MappingProxyType(
    {
        "POLICY_SOLVER_API_VERSION": POLICY_SOLVER_API_VERSION,
        "POLICY_SOLVER_TYPES": POLICY_SOLVER_TYPES,
        "solver_family": solver_family,
        "validate_solver_config": validate_solver_config,
        "solver_decide": solver_decide,
    }
)
