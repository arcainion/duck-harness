"""Shared compiler and sandbox policy for ephemeral Python tool programs."""
from __future__ import annotations


POLICY_VERSION = "duck-python-tool-policy/2"

SAFE_MODULES = frozenset({
    "bisect",
    "collections",
    "copy",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
})

BLOCKED_MODULE_ATTRIBUTES = {
    "operator": frozenset({"attrgetter", "methodcaller"}),
    "string": frozenset({"Formatter"}),
}

# These public-looking attributes can resolve private/dunder names supplied
# through strings, bypassing a simple AST check for leading underscores.
BLOCKED_DYNAMIC_ATTRIBUTES = frozenset({
    "Formatter",
    "attrgetter",
    "format",
    "format_map",
    "methodcaller",
})

PROTECTED_RUNTIME_BINDINGS = frozenset({
    "action",
    "bfs",
    "cell_at",
    "color_grid",
    "count_colors",
    "current_frame",
    "diff_frames",
    "experience",
    "find_positions",
    "flood",
    "grid_utils",
    "history",
    "last_action",
    "last_action_frame",
    "last_action_result",
    "last_transition",
    "latest_frame",
    "neighbors4",
    "neighbors8",
    "object_positions",
    "previous_frame",
    "record_strategy",
    "strategy",
    "transitions",
    "valid_actions",
})


def allowed_modules_text() -> str:
    return ", ".join(sorted(SAFE_MODULES))
