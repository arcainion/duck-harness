"""Prompt templates for the analyzer agent."""

from inference.agent.python_tool_policy import (
    allowed_modules_text,
    runtime_bindings_text,
    runtime_helper_signatures_text,
)
from inference.agent.vision_context import current_grid_image_enabled
from inference.utils.grid_utils import ARC_COLOR_LEGEND

TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
    "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
    "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
)

GAME_OVERVIEW_ADDENDUM = (
    "\n\nGame overview:\n"
    "- You are solving a multi-level grid puzzle game. \n"
    "- You are called repeatedly over the course of a run. Treat each turn as one observe-plan-act cycle: re-understand the current state from the newest frame, update your working world model in Python, choose the next best action or short sequence against the goal as currently understood, execute it, and expect to re-evaluate on the next turn from the updated state.\n"
    "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
    "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n"
    "- Optimize for as few in-game actions as possible while still being reliable.\n"
    "- In this environment, boards are presented as 64 x 64 color grids rendered with ARC color symbols.\n"
    f"- Color legend: {ARC_COLOR_LEGEND}.\n"
)

VISUAL_GAME_ADDENDUM = (
    "\n\nVisual-game guidance:\n"
    "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
    "- Game entities are usually rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometimes they might also be 1x1 tokens.\n"
    "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
    "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
    "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
    "A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment by segment. DON'T DO THIS!\n"
    "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
    "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n"
    "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n"
    "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
)

STRUCTURED_RUNTIME_STATE_ADDENDUM = (
    "\n\nRuntime variables inside every `python` tool call:\n"
    "- `current_frame` is a lightweight frame view for the latest environment state.\n"
    "- `current_frame` exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- `current_frame.ascii` is a single newline-delimited string containing the latest board rendered with the letter-coded ARC color symbols.\n"
    "- `current_frame.segmentation` parses the board into objects. It returns `{'nodes': [...], 'adjacency_list': [...]}`.\n"
    "- Each node in `segmentation['nodes']` is one 4-connected same-color object with: `id` (index, ordered top-most-left-most), `color` (ARC color character), `hash` (a signature of the object's color and shape that ignores its position -- equal hashes mean the same object regardless of where it is, so use it to track an object across frames or to spot multiple identical objects in one frame), `pixels` (cell count), `boundary` (clockwise outer-perimeter corner points as `[row, col]`), and `children` (ids of objects fully enclosed by this one).\n"
    "- `segmentation['adjacency_list']` is a list of `[i, j]` node-id pairs whose objects share an edge.\n"
    
    "- `current_frame.step` is the current environment step count.\n"
    "- `current_frame.level` is the current level number.\n"
    "- `current_frame.shape` is a `(rows, cols)` tuple.\n"
    "- The private raw grid (`current_frame._grid`) is not accessible. Use `current_frame.segmentation` as your primary view of the board -- objects, colors, shapes, containment, adjacency, and cross-frame object hashes. Use `current_frame.ascii` only to read a small, specific region; do not scan the whole board with it.\n"

    "- `history` is a chronological list of action/frame snapshots.\n"
    "- `history` is a Python list of objects, not a dict.\n"
    "- Each history entry exposes only `.action` and `.frame`; entries are not subscriptable like `entry['action']`.\n"
    "- Each `history[i].frame` is the frame after `history[i].action`; each frame exposes only `.ascii`, `.step`, `.level`, `.shape`, and `.segmentation`.\n"
    "- Important history semantics: when `history` is non-empty, `history[-1].frame` is the same latest/post-action board as `current_frame`. It is not the previous board. To inspect the state before the latest action, use `previous_frame` or `history[-2].frame` when available.\n"
    "- `previous_frame` is the frame before the most recent real environment action, or `None` if no previous frame is available.\n"
    "- `last_action` is the most recent real environment action name/display, or `None` before any real action.\n"
    "- `last_action_frame` is the post-action frame for `last_action`; it matches `current_frame` after a real action.\n"
    "- `transitions` is a chronological list of actual action transitions, excluding the initial seeded frame. Each transition exposes `.action`, `.before_frame`, `.after_frame`, `.frame` (alias of `.after_frame`), and `.result`.\n"
    "- `last_transition` is `transitions[-1]` or `None`. Its `.result` mirrors `last_action_result`; older transitions may have an empty `.result`. For before/after diffs, compare `last_transition.before_frame` to `last_transition.after_frame`; do not compare `current_frame` to `history[-1].frame`.\n"
    "- `last_action_result` is the persisted result dict from the most recent `action(...)` call. It remains available across later Python inspection calls that do not call `action(...)`, and is `{}` before any action result exists. Read transition metadata from fields/keys such as `last_action_result['board_changed']`, `last_action_result['done']`, `last_action_result['level_completed']`, `last_action_result['game_over']`, `last_action_result['run_complete']`, `last_action_result['reward']`, and `last_action_result['valid_actions']`. Its bounded `animation` summary reports transient changed cells, motion bounds, and dominant letter-coded color transitions across all frames returned by the action.\n"
    "- `valid_actions` is the current list of valid action names.\n"
    "- `experience` is a read-only compact controller snapshot with the current phase, opaque state id, visits, tried/no-op actions, recent transitions, cycle/stagnation signals, and suggested probes. It never contains the raw numeric grid.\n"
    "- `strategy` is the latest structured strategy memory recorded during this run.\n"
    "- `latest_frame` is a compatibility alias for `current_frame`; prefer `current_frame` in new programs.\n"
    f"- Available protected runtime bindings are: {runtime_bindings_text()}. Read or call them, but never overwrite them.\n"
    f"- Grid helper signatures are: {runtime_helper_signatures_text()}. `grid_utils` exposes the same helper operations as methods.\n"
    "- Call `record_strategy(goal=..., hypothesis=..., evidence=[...], confidence=0.0_to_1.0, open_question=..., next_test=...)` whenever evidence materially changes the plan. Omitted fields preserve their previous values.\n"
    "- Call `action(actions)` to execute one or more real environment actions from Python.\n"
    "- Pass `action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
    "- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n"
    "- After `action(actions)` returns, `current_frame`, `previous_frame`, `history`, `transitions`, `valid_actions`, and `last_action_result` are refreshed.\n"
)

MULTIMODAL_CONTEXT_ADDENDUM = (
    "\n\nMultimodal context:\n"
    "- User turns may include an attached image of the current ARC grid, especially when a level starts.\n"
    "- The image and `current_frame.ascii` are two representations of the same current frame.\n"
    "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
)

PYTHON_ADDENDUM = (
    "\n\nPython tool guidance:\n"
    "- Use `current_frame.segmentation` as your primary view of the board -- objects, colors, containment, adjacency, and cross-frame object hashes.\n"
    "- Use `current_frame.ascii` only to read a small, specific region of the board when `segmentation` is not enough; never use it to scan or summarize the whole board.\n"
    "- Every `python` tool call starts fresh. Submit structured `program` version 1; raw Python source is not accepted. Re-import modules or re-define custom utility logic in each program.\n"
    f"- The only importable standard-library modules are: {allowed_modules_text()}. Dynamic attribute helpers (operator.attrgetter/operator.methodcaller, string.Formatter, and str.format/format_map) are blocked.\n"
    "- The only tool is `python`; its JSON schema is authoritative. Build expressions and statements from the discriminated `kind` variants in that schema.\n"
    "- Common ProgramIR forms: names use `{'kind':'name','name':'x'}`; constants use `{'kind':'constant','value':...}`; calls use `{'kind':'call','function':..., 'args':[...]}`; assignments use `targets` containing `name_target`; field/key access uses `attribute`/`subscript`; starred unpacking uses `starred_target` only inside a `tuple_target` or `list_target`.\n"
    "- Compose data-driven calls with `star_args` for `*sequences` and `star_keywords` for `**mappings`. In a dict expression, an entry whose `key` is null lowers to `**entry.value`; later entries override earlier unpacked keys.\n"
    "- Reusable `function_def` helpers may declare optional `vararg` and `kwarg` names to receive dynamic positional and keyword inputs; those names must be unique and cannot shadow runtime bindings.\n"
    "- Use `keyword_only_parameters` for required or defaulted named configuration inputs; unlike positional parameters, a required keyword-only parameter may follow one with a default.\n"
    "- Use `generator_comprehension` for lazy inputs to `sum`, `any`, `all`, `min`, or `max`; use list/set/dict comprehensions only when the collection itself is needed.\n"
    "- Build reusable lazy traversal helpers with `yield` statements inside `function_def`; set `delegate` true to lower the value as `yield from`.\n"
    "- Use `try` for a small local fallback when uncertain data may raise AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError, or ZeroDivisionError. A handler may bind `name` for compact diagnostics. Use constrained `raise` to signal one of those same errors from a helper; broad/runtime failures remain unavailable.\n"
    "- Always inspect `current_frame`, `history`, and `valid_actions` from Python instead of reasoning from the raw board by eye.\n"
    "- For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`. `history[-1].frame` is the current frame, so comparing it to `current_frame` only compares the board to itself.\n"
    "- Maintain a compact working world model: what entities or regions exist, what actions seem to do, what the goal likely is, what remains uncertain, and what plan best fits the evidence so far.\n"
    "- Start from `experience['phase']`: use one compact inspection in `orient`, a discriminating single-action probe in `explore`, evidence-backed movement in `progress`, and a different hypothesis or action family in `recover`.\n"
    "- Never repeat an action listed in `experience['discouraged_actions']` in the same visible state.\n"
    "- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
    "- Optimize for the shortest reliable sequence that advances the current goal as described by your world model. If confidence is low, program a discriminating probe and revise the world model from the result.\n"
    "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
    "- Inspect current and history frames from Python instead of describing frames freehand.\n"
    "- Never print or echo full board frames. Return only compact derived summaries such as object lists, diffs, coordinates, counts, or tiny local crops.\n"
    "- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. Programs may be substantial, but output must stay short and interpretable.\n"
    "- A strong default loop is: summarize the board, infer the desired environment change, write a small scorer or search over candidate sequences, execute the best probe or plan with `action(...)`, then inspect again until you understand exactly what changed.\n"
    "- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n"
    "- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n"
    "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\n"
    "- Call `action(...)` inside Python rather than returning action text in the chat.\n"
    "- `action(...)` accepts an ordered list of 1-12 actions. Batch only a short sequence whose effects are supported by evidence; otherwise use a single probe so outcomes remain attributable.\n"
    "- You can call `action(...)` multiple times in one compiled program, including inside loops. Each call updates the preloaded variables before execution continues.\n"
    "- If an action result reports `game_over`, `run_complete`, `level_completed`, or `done`, stop acting immediately and re-ground on the next turn.\n"
    "- IMPORTANT: If ProgramIR validation, compilation, or execution fails, use the diagnostic path to fix or drastically simplify it. Do not retry the same failing program. Prefer a direct `action(...)` call when recovery is needed.\n"
    "- IMPORTANT: Names must be documented runtime bindings, safe builtins, allowed imports, or names defined earlier in the same program. Do not invent runtime variable names; undefined names raise NameError and waste a tool call.\n"
    "- Runtime variables and pre-injected helpers are protected bindings. Read or call them, but never assign to them or reuse their names for functions, parameters, or import aliases.\n"
    "- If you lose track of where you are or what the goal is, discard your current world model and start fresh from `current_frame` and `experience`. Do not accumulate stale beliefs.\n"
    "- When you complete a level and move to the next, check `experience['recent_transitions']` and `experience['tried_here']` for evidence of what worked in previous levels. Successful strategies often transfer: if mouse coordinates, pathfinding heuristics, or object interactions solved the prior level, try them first on the new layout.\n"
    "- Pre-injected helpers (no import needed): `color_grid(frame)` returns 2D list of color chars; `diff_frames(f1, f2)` returns changed/appeared/disappeared; `find_positions(frame, char)` returns [(r,c),...]; `neighbors4(r,c,rows,cols)` and `neighbors8(...)`; `bfs(frame,start,goal,blocked=None)` returns path; `flood(frame,start,color=None)` returns set; `cell_at(frame,r,c)` returns color char; `count_colors(frame)` returns dict; `object_positions(frame,color)` returns objects of that color.\n"
)

COMPACT_TOOL_SESSION_ADDENDUM = (
    "\n\nTool session rules:\n"
    "- You have exactly one tool: `python`.\n"
    f"- {TOOL_CALL_FORMAT_GUIDANCE}\n"
    "- The compiled ProgramIR is not saved between calls, so include any custom utility logic you still need.\n"
    "- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
    "- Do not ration tool calls when the state is unclear. Spend extra tool calls to confirm what changed between frames and whether the last action affected gameplay state or only HUD elements such as countdown bars.\n"
    "- After `action(...)` returns, the structured runtime state is refreshed before the next Python statement and before the next tool call. Inspection-only Python calls do not clear `last_action_result`.\n"
    "- Each `python` tool call has a hard time limit of 30 seconds.\n"
    "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
    "- Keep ProgramIR short and purpose-built rather than building a large framework in one call.\n"
    "- If `last_action_result` shows `run_complete=True`, `level_completed=True`, `game_over=True`, or `done=True`, do NOT call `action(...)` again. Instead, print a brief summary of what happened and return. The next turn will re-ground you on the updated state.\n"
    "- If you have called `action(...)` and it succeeded, and your program has already determined the next best action, call `action(...)` again in the same program to save a turn. But stop the loop immediately if any result reports completion.\n"
    "\nProgramIR patterns (express them using the supplied tool schema):\n"
    "- Inspect by assigning `current_frame.segmentation`, subscripting `nodes`, deriving a compact collection, and calling `print`.\n"
    "- Diff by calling `diff_frames(previous_frame, current_frame)` and printing a short slice of `changed`.\n"
    "- Search by calling pre-injected `bfs` or using structured `for`/`while` statements, then call `action` with a list expression.\n"
    "- Recover from an expected data-shape error with `try`: give each handler a non-empty `exceptions` list and `body`, plus optional `name`. A constrained `raise` supplies an allowed `exception` and optional `message`; broad and runtime-failure errors are unavailable.\n"
    "- Unpack variable-length rows with one `starred_target` per tuple/list target level, for example first, *middle, last.\n"
    "- Expand reusable sequences/mappings in calls with `star_args`/`star_keywords`; merge a mapping into a dict with an entry whose `key` is null.\n"
    "- Receive expanded inputs in a `function_def` with optional `vararg` and `kwarg` parameter names.\n"
    "- Declare named-only helper inputs with `keyword_only_parameters`; omit `default` to require the keyword.\n"
    "- Stream values from a helper with `yield`; set `delegate` true to emit every value from an iterable.\n"
    "- Minimal action example: program body contains an `expr` statement whose value calls the name `action` with one argument: a list containing the constant `LEFT`.\n"
)


def build_small_context_prompt(*, tool_output_tokens: int) -> str:
    """Return the complete, non-redundant contract for <=32K contexts."""
    image_guidance = (
        " When attached, the image and current_frame describe the same current board."
        if current_grid_image_enabled()
        else ""
    )
    return (
        "Solve every level of this grid puzzle with reliable actions."
        f"{image_guidance}\n\n"
        f"ARC colors: {ARC_COLOR_LEGEND}. Inspect current_frame.segmentation first; use "
        "current_frame.ascii only for small local regions and never print a full board. "
        "current_frame has .ascii, .step, .level, .shape, and .segmentation. "
        "history[-1].frame is the current frame; use previous_frame or last_transition for "
        "before/after comparisons. Re-ground after every action and level change.\n\n"
        f"Runtime bindings: {runtime_bindings_text()}. Grid helpers: "
        f"{runtime_helper_signatures_text()}. Protected bindings may be read or called but "
        "not overwritten. Names may also be safe builtins, allowed imports, or names defined "
        f"earlier in the same program. Allowed modules: {allowed_modules_text()}.\n\n"
        "The only tool is python. Submit ephemeral ProgramIR with integer version 1 and a "
        "non-empty body; raw source is rejected. Follow schema kind discriminators; state starts "
        "fresh each call. Use generator_comprehension for lazy aggregates. Unpack with "
        "star_args/star_keywords or null-key dict entries. Functions support vararg, kwarg, "
        "keyword_only_parameters, and yield; delegate=true means yield from. "
        "Use constrained try/raise only for expected data-shape or conversion errors. "
        f"{TOOL_CALL_FORMAT_GUIDANCE}\n\n"
        "Inspect or search compactly; assign result or print a short summary. Act with "
        "action([...]); MOUSE needs integer row/col. Calls accept 1-12 ordered actions and refresh "
        "runtime state. Batch only evidence-backed moves; stop on done, game_over, "
        "level_completed, or run_complete. Repair errors from diagnostic path/recovery_hint; "
        "never repeat unchanged failures. Persist changed hypotheses with record_strategy(...).\n\n"
        f"Tool responses are capped near {tool_output_tokens} tokens; keep output decision-oriented."
    )
