"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.agent.action_names import MAX_ACTION_BATCH
from inference.agent.python_tool_policy import (
    BLOCKED_DYNAMIC_ATTRIBUTES,
    BLOCKED_MODULE_ATTRIBUTES,
    PROTECTED_RUNTIME_BINDINGS,
    SAFE_MODULES,
)
from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import ast
    import builtins
    import contextlib
    import io
    import json
    import math
    import os
    import re
    import sys
    import traceback
    import types

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""
    MAX_ACTION_BATCH = __MAX_ACTION_BATCH__

    __SEGMENTATION_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = set(__SAFE_MODULES__)
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "Exception",
        "AttributeError",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "iter",
        "IndexError",
        "KeyError",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "OverflowError",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "ValueError",
        "RuntimeError",
        "ZeroDivisionError",
        "zip",
    }
    SAFE_MODULE_CACHE = {}
    BLOCKED_MODULE_ATTRIBUTES = __BLOCKED_MODULE_ATTRIBUTES__
    BLOCKED_DYNAMIC_ATTRIBUTES = set(__BLOCKED_DYNAMIC_ATTRIBUTES__)
    PROTECTED_RUNTIME_BINDINGS = set(__PROTECTED_RUNTIME_BINDINGS__)


    class SafeModule:
        # Expose public non-module attributes from an approved module.

        def __init__(self, module):
            object.__setattr__(self, "_module", module)

        def __getattribute__(self, name):
            if str(name).startswith("_"):
                raise AttributeError("Private module attributes are not allowed.")
            module = object.__getattribute__(self, "_module")
            module_name = str(getattr(module, "__name__", "")).split(".", 1)[0]
            if str(name) in BLOCKED_MODULE_ATTRIBUTES.get(module_name, set()):
                raise ValueError(
                    f"Private dynamic attribute access is not allowed: {module_name}.{name}"
                )
            value = getattr(module, name)
            if isinstance(value, types.ModuleType):
                raise AttributeError(f"Module-valued attribute '{name}' is not allowed.")
            return value


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=True) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class GridUtils:
        @staticmethod
        def _to_char_grid(frame):
            if frame is None:
                return []
            grid = frame._grid if hasattr(frame, '_grid') else None
            if grid is None:
                return []
            cc = COLOR_CHARS
            return [[cc[max(0, min(15, int(v)))] if isinstance(v, int) else str(v) for v in row] for row in grid]

        @staticmethod
        def diff_frames(frame1, frame2):
            g1 = GridUtils._to_char_grid(frame1)
            g2 = GridUtils._to_char_grid(frame2)
            if not g1 or not g2 or len(g1) != len(g2):
                return {"changed": [], "appeared": [], "disappeared": []}
            changed, appeared, disappeared = [], [], []
            for r in range(len(g1)):
                if r >= len(g2):
                    break
                for c in range(min(len(g1[r]), len(g2[r]))):
                    if g1[r][c] != g2[r][c]:
                        changed.append({"pos": (r, c), "from": g1[r][c], "to": g2[r][c]})
                        if g2[r][c] == '.':
                            disappeared.append((r, c))
                        if g1[r][c] == '.':
                            appeared.append((r, c))
            return {"changed": changed, "appeared": appeared, "disappeared": disappeared}

        @staticmethod
        def get_cell(frame, row, col):
            g = GridUtils._to_char_grid(frame)
            if 0 <= row < len(g) and 0 <= col < len(g[row]):
                return g[row][col]
            return None

        @staticmethod
        def set_cell(frame, row, col, value):
            grid = frame._grid if hasattr(frame, '_grid') else None
            if grid is not None and 0 <= row < len(grid) and 0 <= col < len(grid[row]):
                cc = COLOR_CHARS
                if isinstance(value, str) and len(value) == 1:
                    idx = cc.find(value)
                    if idx >= 0:
                        grid[row][col] = idx
                else:
                    grid[row][col] = int(value)

        @staticmethod
        def neighbors(pos, shape, diagonals=False):
            r, c = pos
            rows, cols = shape
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if diagonals:
                dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            return [(r + dr, c + dc) for dr, dc in dirs if 0 <= r + dr < rows and 0 <= c + dc < cols]

        @staticmethod
        def find_color(frame, color_char):
            return [(r, c) for r, row in enumerate(GridUtils._to_char_grid(frame)) for c, cell in enumerate(row) if cell == color_char]

        @staticmethod
        def flood_fill(frame, start, target_color=None):
            g = GridUtils._to_char_grid(frame)
            if not g:
                return set()
            rows, cols = len(g), max((len(row) for row in g), default=0)
            if not (0 <= start[0] < rows and 0 <= start[1] < cols):
                return set()
            if target_color is None:
                target_color = g[start[0]][start[1]]
            visited = set()
            stack = [start]
            while stack:
                r, c = stack.pop()
                if (r, c) in visited or not (0 <= r < rows and 0 <= c < cols) or g[r][c] != target_color:
                    continue
                visited.add((r, c))
                stack.extend([(r-1, c), (r+1, c), (r, c-1), (r, c+1)])
            return visited

        @staticmethod
        def bfs_path(frame, start, goal, blocked_colors=None):
            g = GridUtils._to_char_grid(frame)
            if not g or not start or not goal:
                return []
            rows, cols = len(g), max((len(row) for row in g), default=0)
            if not (0 <= start[0] < rows and 0 <= start[1] < cols and 0 <= goal[0] < rows and 0 <= goal[1] < cols):
                return []
            blocked = set(blocked_colors) if blocked_colors else set()
            from collections import deque
            q = deque([(start, [start])])
            visited = {start}
            while q:
                (r, c), path = q.popleft()
                if (r, c) == goal:
                    return path
                for nr, nc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_r, new_c = r + nr, c + nc
                    if (0 <= new_r < rows and 0 <= new_c < cols and (new_r, new_c) not in visited and g[new_r][new_c] not in blocked):
                        visited.add((new_r, new_c))
                        q.append(((new_r, new_c), path + [(new_r, new_c)]))
            return []

    grid_utils = GridUtils()

    def color_grid(frame):
        return GridUtils._to_char_grid(frame)

    def diff_frames(f1, f2):
        return GridUtils.diff_frames(f1, f2)

    def find_positions(frame, color_char):
        return GridUtils.find_color(frame, color_char)

    def neighbors4(r, c, rows, cols):
        return [(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)] if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def neighbors8(r, c, rows, cols):
        return [(r+dr, c+dc) for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] if 0 <= r+dr < rows and 0 <= c+dc < cols]

    def bfs(frame, start, goal, blocked=None):
        return GridUtils.bfs_path(frame, start, goal, blocked_colors=blocked)

    def flood(frame, start, color=None):
        return GridUtils.flood_fill(frame, start, target_color=color)

    def cell_at(frame, r, c):
        return GridUtils.get_cell(frame, r, c)

    def count_colors(frame):
        cc = color_grid(frame)
        counts = {}
        for row in cc:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1
        return counts

    def object_positions(frame, color_char):
        seg = frame.segmentation
        nodes = seg.get('nodes', []) if isinstance(seg, dict) else []
        return [{'id': n.get('id'), 'pixels': n.get('pixels'), 'boundary': n.get('boundary', [])} for n in nodes if n.get('color') == color_char]


    class HistoryEntryView:
        def __init__(self, *, action, frame):
            self.action = action
            self.frame = frame

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result={},
                )
            )
        if transitions and isinstance(last_action_result, dict):
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _bounded_json_result(value, max_chars=32768, max_items=2048, max_depth=24):
        # Convert a result without allowing one IPC message to grow unbounded.
        remaining_chars = [max_chars]
        remaining_items = [max_items]
        truncated = [False]
        active_containers = set()

        def marker(text):
            truncated[0] = True
            return text

        def convert(item, depth=0):
            if remaining_items[0] <= 0:
                return marker("... [item limit reached]")
            remaining_items[0] -= 1

            if isinstance(item, float) and not math.isfinite(item):
                return marker(str(item))
            if item is None or isinstance(item, (bool, int, float)):
                remaining_chars[0] -= min(remaining_chars[0], len(str(item)))
                return item
            if isinstance(item, str):
                allowance = max(0, remaining_chars[0])
                if len(item) <= allowance:
                    remaining_chars[0] -= len(item)
                    return item
                remaining_chars[0] = 0
                return marker(item[:allowance] + "... [truncated]")
            if depth >= max_depth:
                return marker("... [depth limit reached]")

            if isinstance(item, (dict, list, tuple, set)):
                identity = id(item)
                if identity in active_containers:
                    return marker("... [cycle]")
                active_containers.add(identity)
                try:
                    if isinstance(item, dict):
                        converted = {}
                        for key, child in item.items():
                            if remaining_items[0] <= 0 or remaining_chars[0] <= 0:
                                converted["... [truncated]"] = True
                                truncated[0] = True
                                break
                            rendered_key = str(key)
                            if len(rendered_key) > 256:
                                rendered_key = rendered_key[:256] + "..."
                                truncated[0] = True
                            remaining_chars[0] -= min(remaining_chars[0], len(rendered_key))
                            converted[rendered_key] = convert(child, depth + 1)
                        return converted

                    converted = []
                    for child in item:
                        if remaining_items[0] <= 0 or remaining_chars[0] <= 0:
                            converted.append("... [truncated]")
                            truncated[0] = True
                            break
                        converted.append(convert(child, depth + 1))
                    return converted
                finally:
                    active_containers.remove(identity)

            return convert(str(item), depth)

        return convert(value), truncated[0]


    def _classify_error(exc):
        error_type = type(exc).__name__
        error_msg = str(exc)
        if error_type == "NameError":
            match = re.search(r"name '(\w+)' is not defined", error_msg)
            name = match.group(1) if match else "unknown"
            if name in ("action", "record_strategy"):
                return "MISUSED_RUNTIME_FUNCTION", f"'{name}()' must be called, not referenced. Use: action(['LEFT'])"
            if name in ("current_frame", "history", "valid_actions", "last_action_result",
                        "experience", "strategy", "transitions", "last_transition",
                        "previous_frame", "last_action"):
                return "MISSING_RUNTIME_VARIABLE", f"'{name}' is available. Do not redefine it."
            return "UNDEFINED_VARIABLE", f"'{name}' is not defined. Check spelling or define it first."
        if error_type == "TypeError":
            if "argument" in error_msg and "positional" in error_msg:
                return "WRONG_ARGUMENTS", f"Wrong arguments: {error_msg}. Check function signature."
            if "unsupported operand" in error_msg or "cannot unpack" in error_msg:
                return "TYPE_MISMATCH", f"Type mismatch: {error_msg}. Ensure operands are compatible types."
            return "TYPE_ERROR", f"TypeError: {error_msg}"
        if error_type == "AttributeError":
            match = re.search(r"object has no attribute '(\w+)'", error_msg)
            attr = match.group(1) if match else "unknown"
            if attr.startswith("_"):
                return "PRIVATE_ACCESS", f"Private attribute '{attr}' is not allowed. Use public attributes only."
            return "MISSING_ATTRIBUTE", f"Attribute '{attr}' not found. Check available attributes."
        if error_type == "ImportError" or error_type == "ModuleNotFoundError":
            if "not allowed" in error_msg:
                return "BLOCKED_MODULE", f"Module blocked: {error_msg}. Use only allowed modules."
            return "IMPORT_ERROR", f"Import failed: {error_msg}"
        if error_type == "KeyError":
            return "MISSING_KEY", f"Key not found: {error_msg}. Check available keys."
        if error_type == "IndexError":
            return "INDEX_OUT_OF_RANGE", f"Index out of range: {error_msg}. Check bounds."
        if error_type == "ValueError":
            if "action(actions) accepts at most" in error_msg:
                return "ACTION_BATCH_LIMIT", error_msg
            if "Protected runtime binding" in error_msg:
                return "PROTECTED_RUNTIME_BINDING", error_msg
            if "Private" in error_msg:
                return "PRIVATE_ACCESS", f"Private access blocked: {error_msg}"
            return "VALUE_ERROR", f"Value error: {error_msg}"
        if error_type == "TimeoutError" or "timed out" in error_msg.lower():
            return "TIMEOUT", f"Code timed out. Optimize or reduce computation."
        if error_type == "RuntimeError" and error_msg.startswith("action failed:"):
            return "ACTION_FAILED", "Action execution failed. Check the action payload and current valid_actions."
        if error_type == "RecursionError":
            return "RECURSION_ERROR", "Recursion depth exceeded. Use iterative loops with bounded ranges instead of recursive calls."
        if error_type == "MemoryError":
            return "MEMORY_ERROR", "Memory exceeded. Use smaller data structures, avoid copying large grids, and process cells lazily."
        return "RUNTIME_ERROR", f"{error_type}: {error_msg}"

    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _exception_user_line(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        return user_frames[-1].lineno if user_frames else None


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        module_name = str(name or "")
        if level != 0 or module_name not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        module = builtins.__import__(name, globals, locals, (), level)
        for imported_name in fromlist or ():
            if (
                imported_name == "*"
                or str(imported_name).startswith("_")
                or not hasattr(module, imported_name)
                or isinstance(getattr(module, imported_name), types.ModuleType)
            ):
                raise ImportError(f"Name '{imported_name}' is not allowed from module '{name}'.")
        proxy = SAFE_MODULE_CACHE.get(module_name)
        if proxy is None:
            proxy = SafeModule(module)
            SAFE_MODULE_CACHE[module_name] = proxy
        return proxy


    def _validate_user_code(code):
        # Reject Python object-graph escape primitives before execution.
        tree = ast.parse(code, filename="<python_tool>", mode="exec")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id in PROTECTED_RUNTIME_BINDINGS
            ):
                raise ValueError(
                    f"Protected runtime binding cannot be overwritten: {node.id}"
                )
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ValueError(f"Private attribute access is not allowed: {node.attr}")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise ValueError(f"Private names are not allowed: {node.id}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name.startswith("_"):
                raise ValueError(f"Private definitions are not allowed: {node.name}")
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name in PROTECTED_RUNTIME_BINDINGS
            ):
                raise ValueError(
                    f"Protected runtime binding cannot be overwritten: {node.name}"
                )
            if isinstance(node, ast.arg) and node.arg.startswith("_"):
                raise ValueError(f"Private argument names are not allowed: {node.arg}")
            if isinstance(node, ast.arg) and node.arg in PROTECTED_RUNTIME_BINDINGS:
                raise ValueError(
                    f"Protected runtime binding cannot be overwritten: {node.arg}"
                )
            if isinstance(node, ast.keyword) and node.arg is not None and node.arg.startswith("_"):
                raise ValueError(f"Private keyword arguments are not allowed: {node.arg}")
            if isinstance(node, ast.Attribute) and node.attr in BLOCKED_DYNAMIC_ATTRIBUTES:
                owner = node.value.id if isinstance(node.value, ast.Name) else ""
                qualified_name = f"{owner}.{node.attr}" if owner else node.attr
                raise ValueError(
                    f"Private dynamic attribute resolution is not allowed: {qualified_name}"
                )
            if isinstance(node, ast.alias):
                bound_name = node.asname or node.name.split(".", 1)[0]
                if node.name.startswith("_") or bound_name.startswith("_"):
                    raise ValueError(f"Private imports are not allowed: {node.name}")
                if bound_name in PROTECTED_RUNTIME_BINDINGS:
                    raise ValueError(
                        f"Protected runtime binding cannot be overwritten: {bound_name}"
                    )
        return tree


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_AS", None), 512 * 1024 * 1024),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            if len(actions) > MAX_ACTION_BATCH:
                raise ValueError(
                    f"action(actions) accepts at most {MAX_ACTION_BATCH} actions per batch."
                )
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        max_stdout_chars = 32768
        stdout = io.StringIO()
        _stdout_len = [0]
        _stdout_truncated = [False]

        class _BoundedStdout:
            def write(self, s):
                remaining = max_stdout_chars - _stdout_len[0]
                if remaining <= 0:
                    if s:
                        _stdout_truncated[0] = True
                    return len(s)
                chunk = s[:remaining]
                stdout.write(chunk)
                _stdout_len[0] += len(chunk)
                if len(s) > remaining:
                    _stdout_truncated[0] = True
                return len(s)
            def flush(self):
                stdout.flush()
            def getvalue(self):
                val = stdout.getvalue()
                if _stdout_truncated[0]:
                    val += "\n... [stdout capped at 32KB]"
                return val

        bounded_stdout = _BoundedStdout()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result
            runtime_globals["experience"] = dict(state_payload.get("experience") or {})
            runtime_globals["strategy"] = dict(state_payload.get("strategy") or {})

        def action(actions):
            normalized_actions = _normalize_actions(actions)
            _send({"type": "action", "actions": normalized_actions})
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            return action_result

        def record_strategy(
            *,
            goal=None,
            hypothesis=None,
            evidence=None,
            confidence=None,
            open_question=None,
            next_test=None,
            test_action=None,
            expected_outcome=None,
            fallback=None,
            contradictions=None,
        ):
            update = {
                "goal": goal,
                "hypothesis": hypothesis,
                "evidence": evidence,
                "confidence": confidence,
                "open_question": open_question,
                "next_test": next_test,
                "test_action": test_action,
                "expected_outcome": expected_outcome,
                "fallback": fallback,
                "contradictions": contradictions,
            }
            safe_update, _ = _bounded_json_result(
                update,
                max_chars=8192,
                max_items=256,
                max_depth=8,
            )
            _send({"type": "strategy", "update": safe_update})
            reply = _recv()
            if reply.get("type") != "strategy_result":
                raise RuntimeError("Invalid strategy response from sandbox host.")
            persisted = dict(reply.get("strategy") or {})
            runtime_globals["strategy"] = persisted
            return persisted

        runtime_globals["action"] = action
        runtime_globals["record_strategy"] = record_strategy
        runtime_globals["grid_utils"] = grid_utils
        runtime_globals["color_grid"] = color_grid
        runtime_globals["diff_frames"] = diff_frames
        runtime_globals["find_positions"] = find_positions
        runtime_globals["neighbors4"] = neighbors4
        runtime_globals["neighbors8"] = neighbors8
        runtime_globals["bfs"] = bfs
        runtime_globals["flood"] = flood
        runtime_globals["cell_at"] = cell_at
        runtime_globals["count_colors"] = count_colors
        runtime_globals["object_positions"] = object_positions
        _refresh_state(initial.get("state") or {})

        try:
            parsed = _validate_user_code(str(initial.get("code", "")))
            compiled = compile(parsed, "<python_tool>", "exec")
            with contextlib.redirect_stdout(bounded_stdout):
                exec(compiled, runtime_globals, runtime_globals)
            safe_result, result_truncated = _bounded_json_result(runtime_globals.get("result"))
            safe_action_results, _ = _bounded_json_result(action_results)
            _send(
                {
                    "type": "final",
                    "stdout": bounded_stdout.getvalue(),
                    "result": safe_result,
                    "result_truncated": result_truncated,
                    "action_results": safe_action_results,
                }
            )
        except Exception as exc:
            error_category, error_hint = _classify_error(exc)
            safe_action_results, _ = _bounded_json_result(action_results)
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "error_line": _exception_user_line(exc),
                    "error_category": error_category,
                    "error_hint": error_hint,
                    "stdout": bounded_stdout.getvalue(),
                    "action_results": safe_action_results,
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation)).replace(
    "__MAX_ACTION_BATCH__", str(MAX_ACTION_BATCH)
).replace(
    "__SAFE_MODULES__", repr(sorted(SAFE_MODULES))
).replace(
    "__BLOCKED_MODULE_ATTRIBUTES__",
    repr({name: sorted(values) for name, values in BLOCKED_MODULE_ATTRIBUTES.items()}),
).replace(
    "__BLOCKED_DYNAMIC_ATTRIBUTES__", repr(sorted(BLOCKED_DYNAMIC_ATTRIBUTES))
).replace(
    "__PROTECTED_RUNTIME_BINDINGS__", repr(sorted(PROTECTED_RUNTIME_BINDINGS))
)


def _sanitize_host_error_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "Sandbox process exited unexpectedly."
    cleaned = "".join(ch for ch in raw if ch.isprintable() or ch in "\n\r\t")
    cleaned = cleaned.strip()
    if not cleaned:
        return "Sandbox process exited unexpectedly."
    if len(cleaned) > 512:
        cleaned = cleaned[:512] + "..."
    return cleaned


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    handle.flush()


def _sandbox_command() -> tuple[list[str], str | None]:
    python_command = [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP]
    bubblewrap = shutil.which("bwrap") if os.name == "posix" else None
    if bubblewrap is None:
        return python_command, None
    return (
        [
            bubblewrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/tmp/work",
            "--chdir",
            "/tmp/work",
            "--",
            *python_command,
        ],
        "/tmp/work",
    )


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            _kill_process_group(process)
        except OSError:
            return

        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            pass
    finally:
        for handle in (process.stdin, process.stdout, process.stderr):
            if handle is not None:
                try:
                    handle.close()
                except (OSError, ValueError):
                    pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
    strategy_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        host_strategy_updates: list[dict[str, Any]] = []
        command, isolated_cwd = _sandbox_command()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "error_category": "SANDBOX_START_FAILED",
                "error_hint": "The isolated Python runtime could not start. Retry with a smaller program.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        def _send_to_runtime(
            payload: dict[str, Any],
            *,
            error: str,
            error_hint: str,
        ) -> dict[str, Any] | None:
            try:
                _send_json_line(process.stdin, payload)
                return None
            except Exception:  # noqa: BLE001 - contain IPC and serialization failures
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": error,
                    "error_category": "SANDBOX_PROTOCOL_ERROR",
                    "error_hint": error_hint,
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

        send_failure = _send_to_runtime(
            {
                "code": code,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": isolated_cwd or sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
            },
            error="Sandbox process rejected its initial payload.",
            error_hint="The isolated runtime could not receive the program payload.",
        )
        if send_failure is not None:
            return send_failure

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                executed_count = len(host_action_results)
                attribution = ""
                if executed_count > 0:
                    attribution = f" {executed_count} action(s) completed before timeout."
                return {
                    "error": f"Tool timed out after {timeout_seconds}s.{attribution}",
                    "error_category": "TIMEOUT",
                    "error_hint": "Code timed out. Optimize or reduce computation.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "error_category": "SANDBOX_PROCESS_ERROR",
                    "error_hint": "The isolated Python process exited unexpectedly.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "error_category": "SANDBOX_PROTOCOL_ERROR",
                    "error_hint": "The isolated runtime returned malformed protocol data.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }
            if not isinstance(message, dict):
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned a non-object response.",
                    "error_category": "SANDBOX_PROTOCOL_ERROR",
                    "error_hint": "The isolated runtime returned invalid protocol data.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "action":
                try:
                    action_result_payload = action_handler(list(message.get("actions") or []))
                    if not isinstance(action_result_payload, dict):
                        raise TypeError("action handler must return a dictionary")
                    raw_action_result = action_result_payload.get("action_result")
                    refreshed_state = action_result_payload.get("state")
                    if raw_action_result is None:
                        raw_action_result = {}
                    if refreshed_state is None:
                        refreshed_state = {}
                    if not isinstance(raw_action_result, dict):
                        raise TypeError("action_result must be a dictionary")
                    if not isinstance(refreshed_state, dict):
                        raise TypeError("action state must be a dictionary")
                except Exception as exc:  # noqa: BLE001
                    send_failure = _send_to_runtime(
                        {
                            "type": "action_error",
                            "error": f"action failed: {type(exc).__name__}: {exc}",
                        },
                        error="Sandbox process could not receive an action failure.",
                        error_hint="The isolated runtime protocol closed during action handling.",
                    )
                    if send_failure is not None:
                        return send_failure
                    continue
                host_action_results.append(dict(raw_action_result))
                send_failure = _send_to_runtime(
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": refreshed_state,
                    },
                    error="Sandbox process could not receive an action result.",
                    error_hint="The isolated runtime protocol closed after an action executed.",
                )
                if send_failure is not None:
                    return send_failure
                continue

            if msg_type == "strategy":
                if strategy_handler is None:
                    persisted_strategy: dict[str, Any] = {
                        "_feedback": "no_strategy_handler",
                        "_warning": "strategy updates are not supported in this sandbox",
                    }
                else:
                    try:
                        persisted_strategy = strategy_handler(
                            dict(message.get("update") or {})
                        )
                        if not isinstance(persisted_strategy, dict):
                            raise TypeError("strategy handler must return a dictionary")
                    except Exception:  # noqa: BLE001
                        persisted_strategy = {}
                host_strategy_updates.append(dict(persisted_strategy))
                send_failure = _send_to_runtime(
                    {
                        "type": "strategy_result",
                        "strategy": persisted_strategy,
                    },
                    error="Sandbox process could not receive a strategy result.",
                    error_hint="The isolated runtime protocol closed during strategy handling.",
                )
                if send_failure is not None:
                    return send_failure
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                result = {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "result_truncated": bool(message.get("result_truncated")),
                    "error": str(message.get("error", "") or ""),
                    # Action callbacks execute in the host. Never let the child
                    # process replace that authoritative execution record.
                    "action_results": list(host_action_results),
                    "strategy_updates": list(host_strategy_updates),
                }
                if msg_type == "error":
                    result["error_category"] = str(message.get("error_category", "") or "")
                    result["error_hint"] = str(message.get("error_hint", "") or "")
                    error_line = message.get("error_line")
                    if isinstance(error_line, int) and not isinstance(error_line, bool):
                        result["error_line"] = error_line
                return result

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "error_category": "SANDBOX_PROTOCOL_ERROR",
                "error_hint": "The isolated runtime returned an unknown protocol message.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
