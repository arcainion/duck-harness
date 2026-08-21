from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import requests

from inference.agent.program_ir import (
    COMPILER_VERSION,
    MAX_COMPILER_DIAGNOSTICS,
    MAX_CONTAINER_ITEMS,
    MAX_INTEGER_BITS,
    MAX_IR_DEPTH,
    MAX_IR_NODES,
    MAX_PROGRAM_CHARS,
    MAX_RAW_PAYLOAD_DEPTH,
    MAX_RAW_PAYLOAD_VALUES,
    ProgramCompileError,
    compile_program,
    fast_program_tool_parameters_schema,
    program_tool_parameters_schema,
)
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.tool_agent import (
    ToolAgent,
    _tool_result_requires_strict_retry,
    _turn_tool_choice,
)


def name(value: str) -> dict:
    return {"kind": "name", "name": value}


def const(value) -> dict:
    return {"kind": "constant", "value": value}


def target(value: str) -> dict:
    return {"kind": "name_target", "name": value}


def call(function: str, *args: dict, **keywords: dict) -> dict:
    return {
        "kind": "call",
        "function": name(function),
        "args": list(args),
        "keywords": keywords,
    }


def program(*body: dict) -> dict:
    return {"version": 1, "body": list(body)}


class ProgramIRCompilerTests(unittest.TestCase):
    def test_assignment_call_and_metadata(self) -> None:
        compiled = compile_program(program(
            {"kind": "assign", "targets": [target("x")], "value": const(3)},
            {"kind": "expr", "value": call("print", name("x"))},
        ))
        self.assertEqual(compiled.source, "x = 3\nprint(x)\n")
        self.assertEqual(compiled.metadata()["version"], COMPILER_VERSION)
        self.assertGreater(compiled.node_count, 0)
        self.assertEqual(len(compiled.metadata()["program_sha256"]), 64)
        self.assertEqual(len(compiled.metadata()["source_sha256"]), 64)
        self.assertRegex(compiled.metadata()["python_version"], r"^\d+\.\d+\.\d+")

    def test_generated_lines_map_to_nested_ir_statement_paths(self) -> None:
        compiled = compile_program(program({
            "kind": "function_def",
            "name": "fail",
            "body": [
                {"kind": "assign", "targets": [target("value")], "value": const(1)},
                {"kind": "expr", "value": call("missing")},
            ],
        }))
        self.assertEqual(compiled.path_for_line(1), "program.body.0")
        self.assertEqual(compiled.path_for_line(2), "program.body.0.body.0")
        self.assertEqual(compiled.path_for_line(3), "program.body.0.body.1")
        self.assertIsNone(compiled.path_for_line(99))
        self.assertEqual(compiled.metadata()["source_map_entries"], 3)

    def test_collections_subscripts_slice_and_operators(self) -> None:
        compiled = compile_program(program(
            {
                "kind": "assign",
                "targets": [target("items")],
                "value": {"kind": "list", "items": [const(1), const(2), const(3)]},
            },
            {
                "kind": "assign",
                "targets": [target("piece")],
                "value": {
                    "kind": "subscript",
                    "value": name("items"),
                    "index": {"kind": "slice", "lower": const(0), "upper": const(2)},
                },
            },
            {
                "kind": "assign",
                "targets": [target("ok")],
                "value": {
                    "kind": "bool",
                    "op": "and",
                    "values": [
                        {"kind": "compare", "left": const(1), "ops": ["lt", "lte"], "comparators": [const(2), const(2)]},
                        {"kind": "unary", "op": "not", "operand": const(False)},
                    ],
                },
            },
            {
                "kind": "aug_assign",
                "target": target("ok"),
                "op": "bit_or",
                "value": const(False),
            },
        ))
        self.assertIn("items[0:2]", compiled.source)
        self.assertIn("1 < 2 <= 2 and (not False)", compiled.source)

    def test_dictionary_and_call_unpacking_lower_and_execute(self) -> None:
        def dict_expr(*entries: dict) -> dict:
            return {"kind": "dict", "entries": list(entries)}

        def entry(key, value) -> dict:
            return {"key": const(key), "value": const(value)}

        payload = program(
            {
                "kind": "assign",
                "targets": [target("base")],
                "value": dict_expr(entry("base", 1)),
            },
            {
                "kind": "assign",
                "targets": [target("more")],
                "value": dict_expr(entry("extra", 2)),
            },
            {
                "kind": "assign",
                "targets": [target("combined")],
                "value": {
                    "kind": "call",
                    "function": name("dict"),
                    "star_args": [{"kind": "list", "items": [name("base")]}],
                    "star_keywords": [name("more")],
                },
            },
            {
                "kind": "assign",
                "targets": [target("result")],
                "value": dict_expr(
                    {"key": None, "value": name("combined")},
                    entry("extra", 5),
                    entry("last", 3),
                ),
            },
        )

        compiled = compile_program(payload)
        self.assertIn("dict(*[base], **more)", compiled.source)
        self.assertIn("{**combined, 'extra': 5, 'last': 3}", compiled.source)
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response["result"], {"base": 1, "extra": 5, "last": 3})

    def test_dictionary_entry_requires_explicit_key_or_null_unpack_marker(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "assign",
                "targets": [target("result")],
                "value": {
                    "kind": "dict",
                    "entries": [{"value": const(1)}],
                },
            }))

        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(diagnostic.code, "IR_REQUIRED_FIELD")
        self.assertEqual(
            diagnostic.path,
            "program.body.0.value.entries.0.key",
        )

    def test_variadic_function_receives_unpacked_call_inputs(self) -> None:
        payload = program(
            {
                "kind": "function_def",
                "name": "collect",
                "parameters": [{"name": "head"}],
                "vararg": "items",
                "kwarg": "options",
                "body": [{
                    "kind": "return",
                    "value": {
                        "kind": "list",
                        "items": [
                            name("head"),
                            call("list", name("items")),
                            name("options"),
                        ],
                    },
                }],
            },
            {
                "kind": "assign",
                "targets": [target("result")],
                "value": {
                    "kind": "call",
                    "function": name("collect"),
                    "args": [const(0)],
                    "star_args": [{"kind": "list", "items": [const(1), const(2)]}],
                    "star_keywords": [{
                        "kind": "dict",
                        "entries": [{"key": const("flag"), "value": const(True)}],
                    }],
                },
            },
        )

        compiled = compile_program(payload)
        self.assertIn("def collect(head, *items, **options):", compiled.source)
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response["result"], [0, [1, 2], {"flag": True}])

    def test_variadic_function_parameter_names_must_be_unique(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "function_def",
                "name": "invalid",
                "parameters": [{"name": "items"}],
                "vararg": "items",
                "body": [{"kind": "pass"}],
            }))

        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(diagnostic.code, "IR_INVALID_VALUE")
        self.assertIn("duplicate parameter: items", diagnostic.message)

    def test_keyword_only_parameters_lower_and_execute(self) -> None:
        payload = program(
            {
                "kind": "function_def",
                "name": "score",
                "parameters": [{"name": "values"}],
                "keyword_only_parameters": [
                    {"name": "scale", "default": const(2)},
                    {"name": "offset"},
                ],
                "body": [{
                    "kind": "return",
                    "value": {
                        "kind": "binary",
                        "op": "add",
                        "left": {
                            "kind": "binary",
                            "op": "mul",
                            "left": call("sum", name("values")),
                            "right": name("scale"),
                        },
                        "right": name("offset"),
                    },
                }],
            },
            {
                "kind": "assign",
                "targets": [target("result")],
                "value": call(
                    "score",
                    {"kind": "list", "items": [const(1), const(2)]},
                    offset=const(4),
                ),
            },
        )

        compiled = compile_program(payload)
        self.assertIn("def score(values, *, scale=2, offset):", compiled.source)
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response["result"], 10)

    def test_keyword_only_parameter_names_share_function_namespace(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "function_def",
                "name": "invalid",
                "parameters": [{"name": "scale"}],
                "keyword_only_parameters": [{"name": "scale"}],
                "body": [{"kind": "pass"}],
            }))

        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(captured.exception.stage, "schema")
        self.assertIn("duplicate parameter: scale", diagnostic.message)

    def test_generator_function_yield_and_delegation_execute_lazily(self) -> None:
        payload = program(
            {
                "kind": "function_def",
                "name": "numbers",
                "body": [
                    {"kind": "yield", "value": const(0)},
                    {
                        "kind": "yield",
                        "value": {
                            "kind": "list",
                            "items": [const(1), const(2)],
                        },
                        "delegate": True,
                    },
                    {"kind": "yield", "value": const(3)},
                ],
            },
            {
                "kind": "assign",
                "targets": [target("result")],
                "value": call("list", call("numbers")),
            },
        )

        compiled = compile_program(payload)
        self.assertIn("yield 0", compiled.source)
        self.assertIn("yield from [1, 2]", compiled.source)
        self.assertEqual(compiled.path_for_line(2), "program.body.0.body.0")
        self.assertEqual(compiled.path_for_line(3), "program.body.0.body.1")
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response["result"], [0, 1, 2, 3])

    def test_yield_outside_function_has_precise_structural_diagnostic(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "yield", "value": const(1)}))

        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(captured.exception.stage, "structure")
        self.assertEqual(diagnostic.code, "IR_YIELD_OUTSIDE_FUNCTION")
        self.assertEqual(diagnostic.path, "program.body.0")

    def test_comprehensions_and_conditional_expression(self) -> None:
        clause = {
            "target": target("x"),
            "iterable": call("range", const(5)),
            "conditions": [{"kind": "compare", "left": name("x"), "ops": ["gt"], "comparators": [const(1)]}],
        }
        compiled = compile_program(program(
            {
                "kind": "assign",
                "targets": [target("values")],
                "value": {
                    "kind": "list_comprehension",
                    "element": {"kind": "binary", "op": "mul", "left": name("x"), "right": const(2)},
                    "clauses": [clause],
                },
            },
            {
                "kind": "assign",
                "targets": [target("mapping")],
                "value": {"kind": "dict_comprehension", "key": name("x"), "value": name("x"), "clauses": [clause]},
            },
            {
                "kind": "assign",
                "targets": [target("unique")],
                "value": {"kind": "set_comprehension", "element": name("x"), "clauses": [clause]},
            },
            {
                "kind": "assign",
                "targets": [target("answer")],
                "value": {"kind": "if_expr", "condition": name("values"), "then": const("yes"), "otherwise": const("no")},
            },
        ))
        self.assertIn("[x * 2 for x in range(5) if x > 1]", compiled.source)
        self.assertIn("'yes' if values else 'no'", compiled.source)

    def test_generator_comprehension_lowers_and_executes_lazily(self) -> None:
        clause = {
            "target": target("x"),
            "iterable": call("range", const(5)),
            "conditions": [{
                "kind": "compare",
                "left": name("x"),
                "ops": ["gt"],
                "comparators": [const(1)],
            }],
        }
        payload = program({
            "kind": "assign",
            "targets": [target("result")],
            "value": call(
                "sum",
                {
                    "kind": "generator_comprehension",
                    "element": {
                        "kind": "binary",
                        "op": "mul",
                        "left": name("x"),
                        "right": const(2),
                    },
                    "clauses": [clause],
                },
            ),
        })

        compiled = compile_program(payload)
        self.assertIn("sum((x * 2 for x in range(5) if x > 1))", compiled.source)
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response["result"], 18)

    def test_constrained_try_recovers_from_expected_data_error(self) -> None:
        payload = program({
            "kind": "try",
            "body": [{
                "kind": "assign",
                "targets": [target("result")],
                "value": {
                    "kind": "subscript",
                    "value": {"kind": "dict", "entries": []},
                    "index": const("missing"),
                },
            }],
            "handlers": [{
                "exceptions": ["KeyError", "IndexError"],
                "body": [{
                    "kind": "assign",
                    "targets": [target("result")],
                    "value": const("fallback"),
                }],
            }],
            "finalbody": [{"kind": "pass"}],
        })

        compiled = compile_program(payload)
        self.assertIn("except (KeyError, IndexError):", compiled.source)
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response["result"], "fallback")
        self.assertEqual(
            compiled.path_for_line(4),
            "program.body.0.handlers.0.body.0",
        )

    def test_try_rejects_broad_exception_handler_with_precise_path(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "try",
                "body": [{"kind": "pass"}],
                "handlers": [{
                    "exceptions": ["Exception"],
                    "body": [{"kind": "pass"}],
                }],
            }))

        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(diagnostic.code, "IR_INVALID_VALUE")
        self.assertEqual(
            diagnostic.path,
            "program.body.0.handlers.0.exceptions.0",
        )

    def test_try_rejects_duplicate_exception_handlers(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "try",
                "body": [{"kind": "pass"}],
                "handlers": [
                    {
                        "exceptions": ["KeyError"],
                        "body": [{"kind": "pass"}],
                    },
                    {
                        "exceptions": ["KeyError"],
                        "body": [{"kind": "pass"}],
                    },
                ],
            }))

        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(captured.exception.stage, "structure")
        self.assertEqual(diagnostic.code, "IR_DUPLICATE_EXCEPTION_HANDLER")
        self.assertEqual(
            diagnostic.path,
            "program.body.0.handlers.1.exceptions.0",
        )

    def test_typed_raise_and_bound_handler_execute_in_sandbox(self) -> None:
        payload = program({
            "kind": "try",
            "body": [{
                "kind": "raise",
                "exception": "ValueError",
                "message": const("bad shape"),
            }],
            "handlers": [{
                "exceptions": ["ValueError"],
                "name": "error",
                "body": [{
                    "kind": "assign",
                    "targets": [target("result")],
                    "value": call("str", name("error")),
                }],
            }],
        })

        compiled = compile_program(payload)
        self.assertIn("raise ValueError('bad shape')", compiled.source)
        self.assertIn("except ValueError as error:", compiled.source)
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response["result"], "bad shape")

    def test_raise_rejects_runtime_exception_type(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "raise",
                "exception": "RuntimeError",
                "message": const("do not suppress action failures"),
            }))

        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(diagnostic.code, "IR_INVALID_VALUE")
        self.assertEqual(diagnostic.path, "program.body.0.exception")

    def test_function_for_while_imports_and_control_flow(self) -> None:
        compiled = compile_program(program(
            {"kind": "import", "names": [{"name": "math", "asname": "mathlib"}]},
            {"kind": "from_import", "module": "collections", "names": [{"name": "deque"}]},
            {
                "kind": "function_def",
                "name": "search",
                "parameters": [{"name": "limit", "default": const(3)}],
                "body": [
                    {"kind": "assign", "targets": [target("total")], "value": const(0)},
                    {
                        "kind": "for",
                        "target": target("item"),
                        "iterable": call("range", name("limit")),
                        "body": [
                            {"kind": "if", "condition": {"kind": "compare", "left": name("item"), "ops": ["eq"], "comparators": [const(1)]}, "body": [{"kind": "continue"}]},
                            {"kind": "aug_assign", "target": target("total"), "op": "add", "value": name("item")},
                        ],
                    },
                    {
                        "kind": "while",
                        "condition": {"kind": "compare", "left": name("total"), "ops": ["lt"], "comparators": [const(10)]},
                        "body": [
                            {"kind": "aug_assign", "target": target("total"), "op": "add", "value": const(1)},
                            {"kind": "if", "condition": {"kind": "compare", "left": name("total"), "ops": ["gte"], "comparators": [const(5)]}, "body": [{"kind": "break"}], "orelse": [{"kind": "pass"}]},
                        ],
                    },
                    {"kind": "return", "value": name("total")},
                ],
            },
        ))
        self.assertIn("def search(limit=3):", compiled.source)
        self.assertIn("while total < 10:", compiled.source)

    def test_unpack_attribute_and_subscript_targets(self) -> None:
        compiled = compile_program(program(
            {
                "kind": "assign",
                "targets": [{"kind": "tuple_target", "items": [target("left"), target("right")]}],
                "value": {"kind": "tuple", "items": [const(1), const(2)]},
            },
            {
                "kind": "assign",
                "targets": [{"kind": "subscript_target", "value": name("values"), "index": const(0)}],
                "value": name("left"),
            },
            {
                "kind": "assign",
                "targets": [{"kind": "attribute_target", "value": name("obj"), "attr": "value"}],
                "value": name("right"),
            },
        ))
        self.assertIn("left, right = (1, 2)", compiled.source)
        self.assertIn("values[0] = left", compiled.source)

    def test_starred_tuple_and_list_unpacking(self) -> None:
        compiled = compile_program(program(
            {
                "kind": "assign",
                "targets": [{
                    "kind": "tuple_target",
                    "items": [
                        target("first"),
                        {"kind": "starred_target", "target": target("middle")},
                        target("last"),
                    ],
                }],
                "value": name("values"),
            },
            {
                "kind": "for",
                "target": {
                    "kind": "list_target",
                    "items": [
                        {"kind": "starred_target", "target": target("prefix")},
                        target("tail"),
                    ],
                },
                "iterable": name("rows"),
                "body": [{"kind": "pass"}],
            },
        ))
        self.assertIn("first, *middle, last = values", compiled.source)
        self.assertIn("for [*prefix, tail] in rows:", compiled.source)

    def test_multiple_starred_targets_at_one_level_are_rejected(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "assign",
                "targets": [{
                    "kind": "tuple_target",
                    "items": [
                        {"kind": "starred_target", "target": target("left")},
                        {"kind": "starred_target", "target": target("right")},
                    ],
                }],
                "value": name("values"),
            }))
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_INVALID_VALUE")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.targets.0")

    def test_starred_target_is_not_valid_at_assignment_top_level(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "assign",
                "targets": [{"kind": "starred_target", "target": target("items")}],
                "value": name("values"),
            }))
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_INVALID_KIND")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.targets.0")

    def test_augmented_assignment_rejects_unpacking_target_at_schema_stage(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "aug_assign",
                "target": {"kind": "tuple_target", "items": [target("left"), target("right")]},
                "op": "add",
                "value": const(1),
            }))
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_INVALID_KIND")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.target")

    def test_from_import_rejects_dotted_imported_name_with_exact_path(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "from_import",
                "module": "collections",
                "names": [{"name": "abc.Sequence"}],
            }))
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_INVALID_VALUE")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.names.0.name")

    def test_unlisted_dotted_module_is_rejected_by_safety_policy(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "import",
                "names": [{"name": "collections.abc", "asname": "abc"}],
            }))
        self.assertEqual(captured.exception.stage, "safety")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_BLOCKED_MODULE")
        self.assertEqual(
            captured.exception.diagnostics[0].path,
            "program.body.0.names.0.name",
        )

    def test_invalid_identifier_and_compare_shape_have_paths(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "assign",
                "targets": [target("_private")],
                "value": const(1),
            }))
        payload = captured.exception.payload()
        self.assertEqual(payload["stage"], "schema")
        self.assertEqual(payload["diagnostics"][0]["code"], "IR_INVALID_VALUE")
        self.assertEqual(payload["diagnostics"][0]["path"], "program.body.0.targets.0.name")

        with self.assertRaises(ProgramCompileError):
            compile_program(program({
                "kind": "expr",
                "value": {"kind": "compare", "left": const(1), "ops": ["eq"], "comparators": [const(1), const(2)]},
            }))

    def test_unknown_kind_and_extra_field_rejected(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program({"version": 1, "body": [{"kind": "eval", "source": "print(1)"}]})
        self.assertEqual(captured.exception.error_category, "INVALID_PROGRAM_IR")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_INVALID_KIND")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0")

    def test_missing_kind_reports_direct_structured_repair_hint(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program({
                "version": 1,
                "body": [{"type": "expr", "value": {"kind": "constant", "value": 1}}],
            })

        payload = captured.exception.payload()
        diagnostic = payload["diagnostics"][0]
        self.assertEqual(diagnostic["code"], "IR_INVALID_KIND")
        self.assertEqual(diagnostic["path"], "program.body.0")
        self.assertEqual(diagnostic["message"], "Node requires the `kind` discriminator.")
        self.assertIn("rename `type` to `kind`", diagnostic["hint"])
        self.assertEqual(payload["recovery_hint"], diagnostic["hint"])

    def test_near_miss_kind_suggests_closest_valid_kind(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "expression",
                "value": const(1),
            }))

        diagnostic = captured.exception.payload()["diagnostics"][0]
        self.assertEqual(diagnostic["message"], "Unknown node kind 'expression'.")
        self.assertIn("replace `kind`: 'expression' with 'expr'", diagnostic["hint"])

    def test_missing_and_extra_fields_include_actionable_hints(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "expr", "unexpected": True}))

        diagnostics = {item.path: item for item in captured.exception.diagnostics}
        self.assertEqual(
            diagnostics["program.body.0.value"].hint,
            "Add required field `value` at program.body.0.value.",
        )
        self.assertEqual(
            diagnostics["program.body.0.unexpected"].hint,
            "Remove unsupported field `unexpected` at program.body.0.unexpected.",
        )

    def test_type_coercion_is_rejected(self) -> None:
        coercive_payloads = (
            ({"version": "1", "body": [{"kind": "pass"}]}, "program.version"),
            ({"version": True, "body": [{"kind": "pass"}]}, "program.version"),
            ({"version": 1, "body": ({"kind": "pass"},)}, "program.body"),
        )
        for payload, expected_path in coercive_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ProgramCompileError) as captured:
                    compile_program(payload)
                self.assertEqual(captured.exception.stage, "schema")
                self.assertEqual(captured.exception.diagnostics[0].path, expected_path)

    def test_protocol_version_is_explicitly_required(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program({"body": [{"kind": "pass"}]})

        payload = captured.exception.payload()
        diagnostic = payload["diagnostics"][0]
        self.assertEqual(diagnostic["code"], "IR_REQUIRED_FIELD")
        self.assertEqual(diagnostic["path"], "program.version")
        self.assertEqual(
            diagnostic["hint"],
            "Add required field `version` at program.version.",
        )

    def test_strict_validation_preserves_integer_and_float_identity(self) -> None:
        integer = compile_program(program({"kind": "expr", "value": const(1)}))
        floating = compile_program(program({"kind": "expr", "value": const(1.0)}))
        self.assertEqual(integer.source, "1\n")
        self.assertEqual(floating.source, "1.0\n")
        self.assertNotEqual(integer.program_sha256, floating.program_sha256)

    def test_missing_and_extra_fields_use_stable_diagnostic_codes(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "expr", "unexpected": True}))
        diagnostics = {(item.code, item.path) for item in captured.exception.diagnostics}
        self.assertIn(("IR_REQUIRED_FIELD", "program.body.0.value"), diagnostics)
        self.assertIn(("IR_UNKNOWN_FIELD", "program.body.0.unexpected"), diagnostics)

    def test_schema_diagnostics_are_bounded(self) -> None:
        malformed = {"version": 1, "body": [{"kind": "invalid"}] * (MAX_COMPILER_DIAGNOSTICS + 10)}
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(malformed)
        self.assertEqual(len(captured.exception.diagnostics), MAX_COMPILER_DIAGNOSTICS + 1)
        self.assertEqual(captured.exception.diagnostics[-1].code, "IR_DIAGNOSTICS_TRUNCATED")

    def test_dynamic_source_execution_name_is_rejected(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "expr", "value": call("eval", const("1 + 1"))}))
        self.assertEqual(captured.exception.stage, "schema")
        self.assertIn("not allowed", str(captured.exception))

    def test_non_finite_constants_are_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ProgramCompileError) as captured:
                compile_program(program({"kind": "expr", "value": const(value)}))
            self.assertEqual(captured.exception.diagnostics[0].code, "IR_INVALID_VALUE")
            self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.value.value")

    def test_oversized_integer_is_rejected_before_canonical_serialization(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "expr", "value": const(1 << MAX_INTEGER_BITS)}))
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_INVALID_VALUE")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.value.value")

    def test_keyword_lowering_is_canonical(self) -> None:
        first = program({
            "kind": "expr",
            "value": {
                "kind": "call",
                "function": name("print"),
                "keywords": {"zebra": const(1), "alpha": const(2)},
            },
        })
        second = program({
            "kind": "expr",
            "value": {
                "kind": "call",
                "function": name("print"),
                "keywords": {"alpha": const(2), "zebra": const(1)},
            },
        })
        first_compiled = compile_program(first)
        second_compiled = compile_program(second)
        self.assertEqual(first_compiled.program_sha256, second_compiled.program_sha256)
        self.assertEqual(first_compiled.source, second_compiled.source)
        self.assertEqual(first_compiled.source_sha256, second_compiled.source_sha256)
        self.assertIn("alpha=2, zebra=1", first_compiled.source)

    def test_unstable_ast_roundtrip_is_rejected(self) -> None:
        payload = program({"kind": "assign", "targets": [target("x")], "value": const(1)})
        with patch("inference.agent.program_ir.ast.unparse", side_effect=["x = 1", "x = 2"]):
            with self.assertRaises(ProgramCompileError) as captured:
                compile_program(payload)
        self.assertEqual(captured.exception.stage, "verify")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_AST_ROUNDTRIP_MISMATCH")

    def test_top_level_return_becomes_structural_diagnostic(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "return", "value": const(1)}))
        self.assertEqual(captured.exception.stage, "structure")
        self.assertEqual(captured.exception.error_category, "INVALID_CONTROL_FLOW")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_RETURN_OUTSIDE_FUNCTION")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0")

    def test_invalid_loop_control_reports_all_exact_paths(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program(
                {"kind": "break"},
                {"kind": "if", "condition": const(True), "body": [{"kind": "continue"}]},
            ))
        self.assertEqual(
            [(item.code, item.path) for item in captured.exception.diagnostics],
            [
                ("IR_BREAK_OUTSIDE_LOOP", "program.body.0"),
                ("IR_CONTINUE_OUTSIDE_LOOP", "program.body.1.body.0"),
            ],
        )

    def test_loop_else_and_nested_function_do_not_inherit_loop_context(self) -> None:
        cases = (
            (
                [{"kind": "pass"}],
                [{"kind": "break"}],
                "program.body.0.orelse.0",
            ),
            (
                [{"kind": "function_def", "name": "helper", "body": [{"kind": "break"}]}],
                [],
                "program.body.0.body.0.body.0",
            ),
        )
        for loop_body, loop_orelse, expected_path in cases:
            loop = {
                "kind": "for",
                "target": target("item"),
                "iterable": call("range", const(1)),
                "body": loop_body,
                "orelse": loop_orelse,
            }
            with self.subTest(expected_path=expected_path), self.assertRaises(ProgramCompileError) as captured:
                compile_program(program(loop))
            self.assertEqual(captured.exception.diagnostics[0].path, expected_path)

    def test_statement_after_return_is_rejected_as_unreachable(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "function_def",
                "name": "answer",
                "body": [
                    {"kind": "return", "value": const(1)},
                    {"kind": "expr", "value": call("print", const("never"))},
                ],
            }))
        self.assertEqual(captured.exception.stage, "structure")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_UNREACHABLE_STATEMENT")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.body.1")

    def test_statement_after_raise_is_rejected_as_unreachable(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program(
                {"kind": "raise", "exception": "ValueError"},
                {"kind": "pass"},
            ))
        self.assertEqual(captured.exception.stage, "structure")
        self.assertEqual(
            captured.exception.diagnostics[0].code,
            "IR_UNREACHABLE_STATEMENT",
        )
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.1")

    def test_both_terminating_if_branches_make_following_statement_unreachable(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "function_def",
                "name": "choose",
                "parameters": [{"name": "condition"}],
                "body": [
                    {
                        "kind": "if",
                        "condition": name("condition"),
                        "body": [{"kind": "return", "value": const(1)}],
                        "orelse": [{"kind": "return", "value": const(2)}],
                    },
                    {"kind": "pass"},
                ],
            }))
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.body.1")

    def test_single_terminating_if_branch_allows_following_statement(self) -> None:
        compiled = compile_program(program({
            "kind": "function_def",
            "name": "choose",
            "parameters": [{"name": "condition"}],
            "body": [
                {
                    "kind": "if",
                    "condition": name("condition"),
                    "body": [{"kind": "return", "value": const(1)}],
                },
                {"kind": "return", "value": const(2)},
            ],
        }))
        self.assertIn("return 2", compiled.source)

    def test_structural_diagnostics_are_bounded(self) -> None:
        body = [{"kind": "return"}] + [{"kind": "pass"}] * (MAX_COMPILER_DIAGNOSTICS + 10)
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "function_def", "name": "done", "body": body}))
        self.assertEqual(len(captured.exception.diagnostics), MAX_COMPILER_DIAGNOSTICS + 1)
        self.assertEqual(captured.exception.diagnostics[-1].code, "IR_DIAGNOSTICS_TRUNCATED")

    def test_program_fingerprint_is_canonical_and_content_sensitive(self) -> None:
        first = program({"kind": "expr", "value": call("print", const(1))})
        reordered = {"body": first["body"], "version": 1}
        changed = program({"kind": "expr", "value": call("print", const(2))})
        self.assertEqual(compile_program(first).program_sha256, compile_program(reordered).program_sha256)
        self.assertNotEqual(compile_program(first).program_sha256, compile_program(changed).program_sha256)

    def test_node_limit(self) -> None:
        body = [{"kind": "expr", "value": const(index)} for index in range(MAX_IR_NODES + 1)]
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program(*body))
        self.assertEqual(captured.exception.error_category, "PROGRAM_TOO_LARGE")

    def test_container_cardinality_is_rejected_during_schema_validation(self) -> None:
        expression = {
            "kind": "list",
            "items": [const(None)] * (MAX_CONTAINER_ITEMS + 1),
        }
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "expr", "value": expression}))
        self.assertEqual(captured.exception.stage, "schema")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_TOO_LONG")
        self.assertEqual(
            captured.exception.diagnostics[0].path,
            "program.body.0.value.items",
        )

    def test_aggregate_program_size_limit(self) -> None:
        large_value = "x" * 8192
        body = [{"kind": "expr", "value": const(large_value)} for _ in range(10)]
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program(*body))
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_SIZE_LIMIT")
        self.assertIn(str(MAX_PROGRAM_CHARS), captured.exception.diagnostics[0].message)

    def test_depth_limit(self) -> None:
        expression = const(True)
        for _ in range(MAX_IR_DEPTH + 2):
            expression = {"kind": "unary", "op": "not", "operand": expression}
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "expr", "value": expression}))
        self.assertEqual(captured.exception.error_category, "PROGRAM_TOO_DEEP")

    def test_raw_payload_depth_is_bounded_before_model_validation(self) -> None:
        payload: object = None
        for _ in range(MAX_RAW_PAYLOAD_DEPTH + 2):
            payload = {"nested": payload}
        with patch("inference.agent.program_ir.ProgramIR.model_validate") as validate:
            with self.assertRaises(ProgramCompileError) as captured:
                compile_program(payload)
        validate.assert_not_called()
        self.assertEqual(captured.exception.stage, "limits")
        self.assertEqual(captured.exception.error_category, "PROGRAM_TOO_DEEP")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_RAW_DEPTH_LIMIT")

    def test_raw_payload_value_count_is_bounded_before_model_validation(self) -> None:
        payload = {"unused": [None] * MAX_RAW_PAYLOAD_VALUES}
        with patch("inference.agent.program_ir.ProgramIR.model_validate") as validate:
            with self.assertRaises(ProgramCompileError) as captured:
                compile_program(payload)
        validate.assert_not_called()
        self.assertEqual(captured.exception.stage, "limits")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_RAW_VALUE_LIMIT")

    def test_raw_payload_size_is_bounded_before_model_validation(self) -> None:
        payload = {"unused": "x" * (MAX_PROGRAM_CHARS + 1)}
        with patch("inference.agent.program_ir.ProgramIR.model_validate") as validate:
            with self.assertRaises(ProgramCompileError) as captured:
                compile_program(payload)
        validate.assert_not_called()
        self.assertEqual(captured.exception.stage, "limits")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_SIZE_LIMIT")

    def test_cyclic_direct_payload_is_rejected_before_model_validation(self) -> None:
        payload: dict = {"version": 1}
        payload["body"] = [payload]
        with patch("inference.agent.program_ir.ProgramIR.model_validate") as validate:
            with self.assertRaises(ProgramCompileError) as captured:
                compile_program(payload)
        validate.assert_not_called()
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_CYCLIC_PAYLOAD")

    def test_tool_schema_has_hoisted_definitions(self) -> None:
        schema = program_tool_parameters_schema()
        self.assertEqual(schema["required"], ["program"])
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("$defs", schema)
        program_schema = schema["properties"]["program"]
        self.assertIn("body", program_schema["properties"])
        self.assertIn("version", program_schema["required"])
        self.assertNotIn("default", program_schema["properties"]["version"])
        self.assertIn("raw Python source is not accepted", schema["properties"]["program"]["description"])
        self.assertIn("never coerced", schema["properties"]["program"]["description"])
        self.assertIn(f"{MAX_INTEGER_BITS} bits", schema["properties"]["program"]["description"])
        self.assertIn(
            f"{MAX_RAW_PAYLOAD_VALUES} values before schema validation",
            schema["properties"]["program"]["description"],
        )
        self.assertIn("kind discriminator", schema["$defs"]["Stmt"]["description"])
        self.assertIn("unreachable statements", schema["$defs"]["Stmt"]["description"])
        self.assertIn("attribute, or subscript", schema["$defs"]["AugTarget"]["description"])
        self.assertIn("starred_target", schema["$defs"]["UnpackTarget"]["description"])
        self.assertIn("GeneratorComprehensionExpr", schema["$defs"])
        self.assertIn("ExceptHandlerIR", schema["$defs"])
        self.assertIn("TryStmt", schema["$defs"])
        self.assertIn("RaiseStmt", schema["$defs"])
        function_schema = schema["$defs"]["FunctionDefStmt"]["properties"]
        self.assertIn("vararg", function_schema)
        self.assertIn("kwarg", function_schema)
        self.assertIn("keyword_only_parameters", function_schema)

    def test_tool_schema_omits_nonsemantic_generated_titles(self) -> None:
        schema = program_tool_parameters_schema()
        pending = [schema]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                self.assertNotIn("title", value)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        self.assertLess(len(json.dumps(schema)), 20_000)

    def test_cached_tool_schema_returns_isolated_copies(self) -> None:
        first = program_tool_parameters_schema()
        first["properties"]["program"]["description"] = "mutated"

        second = program_tool_parameters_schema()

        self.assertIsNot(first, second)
        self.assertIn(
            "raw Python source is not accepted",
            second["properties"]["program"]["description"],
        )

    def test_fast_tool_schema_keeps_structure_with_less_prompt_bulk(self) -> None:
        full = program_tool_parameters_schema()
        fast = fast_program_tool_parameters_schema()

        self.assertEqual(fast["required"], ["program"])
        self.assertFalse(fast["additionalProperties"])
        self.assertFalse(fast["properties"]["program"]["additionalProperties"])
        self.assertEqual(fast["$defs"].keys(), full["$defs"].keys())
        self.assertIn("required", fast["$defs"]["AssignStmt"])
        self.assertIn("enum", fast["$defs"]["BinaryExpr"]["properties"]["op"])
        self.assertLess(len(json.dumps(fast)), len(json.dumps(full)) * 0.75)


class ProgramIRSandboxTests(unittest.TestCase):
    def test_cyclic_strategy_evidence_is_bounded_instead_of_recursing(self) -> None:
        result = run_sandboxed_python(
            code=(
                "evidence = []\n"
                "evidence.append(evidence)\n"
                "record_strategy(evidence=evidence)\n"
                "result = 'ok'\n"
            ),
            timeout_seconds=2,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )

        self.assertEqual(result.get("error"), "")
        self.assertEqual(result.get("result"), "ok")
        strategy_updates = result.get("strategy_updates") or []
        self.assertEqual(len(strategy_updates), 1)
        self.assertIn("cycle", json.dumps(strategy_updates[0]).lower())

    def test_lowered_program_executes_in_existing_sandbox(self) -> None:
        compiled = compile_program(program(
            {"kind": "assign", "targets": [target("counts")], "value": call("count_colors", name("current_frame"))},
            {"kind": "assign", "targets": [target("result")], "value": name("counts")},
        ))
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={
                "current_frame": {"ascii": "00", "step": 0, "level": 1, "shape": [1, 2], "grid": [[0, 0]]},
                "history": [],
                "valid_actions": ["LEFT"],
                "experience": {},
                "strategy": {},
                "last_action_result": {},
            },
            action_handler=lambda actions: {"action_result": {}, "state": {}},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response.get("result"), {"W": 2})

    def test_blocked_import_is_rejected_by_compiler_and_sandbox(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "import", "names": [{"name": "os"}]}))
        self.assertEqual(captured.exception.stage, "safety")
        self.assertEqual(captured.exception.error_category, "UNSAFE_PROGRAM_IR")
        self.assertEqual(captured.exception.diagnostics[0].code, "IR_BLOCKED_MODULE")
        self.assertEqual(captured.exception.diagnostics[0].path, "program.body.0.names.0.name")

        response = run_sandboxed_python(
            code="import os",
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response.get("error_category"), "BLOCKED_MODULE")

    def test_from_import_dynamic_helper_has_path_addressed_safety_error(self) -> None:
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({
                "kind": "from_import",
                "module": "operator",
                "names": [{"name": "attrgetter", "asname": "getter"}],
            }))
        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, "IR_BLOCKED_DYNAMIC_ATTRIBUTE")
        self.assertEqual(diagnostic.path, "program.body.0.names.0.name")

    def test_attribute_dynamic_helper_has_path_addressed_safety_error(self) -> None:
        expression = {
            "kind": "call",
            "function": {
                "kind": "attribute",
                "value": const("{0.__class__}"),
                "attr": "format",
            },
            "args": [const(1)],
        }
        with self.assertRaises(ProgramCompileError) as captured:
            compile_program(program({"kind": "expr", "value": expression}))
        diagnostic = captured.exception.diagnostics[0]
        self.assertEqual(diagnostic.code, "IR_BLOCKED_DYNAMIC_ATTRIBUTE")
        self.assertEqual(diagnostic.path, "program.body.0.value.function.attr")

    def test_runtime_bindings_cannot_be_shadowed(self) -> None:
        cases = (
            (
                {"kind": "assign", "targets": [target("current_frame")], "value": const(None)},
                "program.body.0.targets.0.name",
            ),
            (
                {"kind": "function_def", "name": "action", "parameters": [], "body": [{"kind": "pass"}]},
                "program.body.0.name",
            ),
            (
                {
                    "kind": "function_def",
                    "name": "helper",
                    "parameters": [{"name": "history"}],
                    "body": [{"kind": "pass"}],
                },
                "program.body.0.parameters.0.name",
            ),
            (
                {"kind": "import", "names": [{"name": "math", "asname": "bfs"}]},
                "program.body.0.names.0.asname",
            ),
            (
                {
                    "kind": "from_import",
                    "module": "math",
                    "names": [{"name": "sqrt", "asname": "valid_actions"}],
                },
                "program.body.0.names.0.asname",
            ),
            (
                {
                    "kind": "try",
                    "body": [{"kind": "pass"}],
                    "handlers": [{
                        "exceptions": ["ValueError"],
                        "name": "action",
                        "body": [{"kind": "pass"}],
                    }],
                },
                "program.body.0.handlers.0.name",
            ),
            (
                {
                    "kind": "function_def",
                    "name": "helper",
                    "vararg": "action",
                    "body": [{"kind": "pass"}],
                },
                "program.body.0.vararg",
            ),
            (
                {
                    "kind": "function_def",
                    "name": "helper",
                    "kwarg": "current_frame",
                    "body": [{"kind": "pass"}],
                },
                "program.body.0.kwarg",
            ),
            (
                {
                    "kind": "function_def",
                    "name": "helper",
                    "keyword_only_parameters": [{"name": "valid_actions"}],
                    "body": [{"kind": "pass"}],
                },
                "program.body.0.keyword_only_parameters.0.name",
            ),
        )
        for statement, expected_path in cases:
            with self.subTest(statement=statement), self.assertRaises(ProgramCompileError) as captured:
                compile_program(program(statement))
            diagnostic = captured.exception.diagnostics[0]
            self.assertEqual(captured.exception.stage, "safety")
            self.assertEqual(diagnostic.code, "IR_PROTECTED_RUNTIME_BINDING")
            self.assertEqual(diagnostic.path, expected_path)

    def test_result_remains_a_writable_output_binding(self) -> None:
        compiled = compile_program(program({
            "kind": "assign",
            "targets": [target("result")],
            "value": const(42),
        }))
        self.assertIn("result = 42", compiled.source)


class ProgramIRToolAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ToolAgent(model="m", provider="vllm", base_url="http://127.0.0.1:1/v1")

    def test_tool_requires_program_not_code(self) -> None:
        with TemporaryDirectory() as tmp:
            function = self.agent._tools(Path(tmp) / "state.json")[0]["function"]
            schema = function["parameters"]
        self.assertEqual(schema["required"], ["program"])
        self.assertNotIn("code", schema["properties"])
        self.assertIs(function["strict"], True)

    def test_non_vllm_provider_retains_compatible_tool_shape(self) -> None:
        agent = ToolAgent(
            model="m",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
        )
        with TemporaryDirectory() as tmp:
            function = agent._tools(Path(tmp) / "state.json")[0]["function"]
        self.assertNotIn("strict", function)

    def test_vllm_fast_path_omits_guided_decoding(self) -> None:
        with TemporaryDirectory() as tmp:
            function = self.agent._tools(
                Path(tmp) / "state.json",
                strict=False,
            )[0]["function"]

        self.assertNotIn("strict", function)
        self.assertLess(
            len(json.dumps(function["parameters"])),
            len(json.dumps(program_tool_parameters_schema())) * 0.75,
        )

    def test_schema_failure_enables_strict_retry(self) -> None:
        self.assertTrue(
            _tool_result_requires_strict_retry(
                json.dumps(
                    {
                        "stage": "schema",
                        "error_category": "MALFORMED_TOOL_ARGUMENTS",
                        "error": "invalid JSON",
                    }
                )
            )
        )
        self.assertFalse(
            _tool_result_requires_strict_retry(
                json.dumps({"stage": "runtime", "error": "division by zero"})
            )
        )

    def test_vllm_requires_constrained_tool_until_action_executes(self) -> None:
        tools = [{"type": "function", "function": {"name": "python"}}]
        self.assertEqual(
            _turn_tool_choice(
                tools,
                provider="vllm",
                turn_count=1,
                step_executed=False,
            ),
            "required",
        )
        self.assertEqual(
            _turn_tool_choice(
                tools,
                provider="vllm",
                turn_count=2,
                step_executed=True,
            ),
            "auto",
        )
        self.assertEqual(
            _turn_tool_choice(
                tools,
                provider="openrouter",
                turn_count=1,
                step_executed=False,
            ),
            "auto",
        )

    def test_chat_completion_sends_selected_tool_choice_and_strict_schema(self) -> None:
        with TemporaryDirectory() as tmp:
            tools = self.agent._tools(Path(tmp) / "state.json")
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
        }
        with patch.object(
            self.agent._http_session,
            "post",
            return_value=response,
        ) as post:
            self.agent._chat_completion(
                [{"role": "user", "content": "inspect"}],
                tools=tools,
                tool_choice="required",
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["tool_choice"], "required")
        self.assertIs(payload["tools"][0]["function"]["strict"], True)

    def test_chat_completion_reuses_session_and_close_releases_it(self) -> None:
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
        }
        with (
            patch.object(
                self.agent._http_session,
                "post",
                return_value=response,
            ) as post,
            patch.object(self.agent._http_session, "close") as close,
        ):
            for _ in range(2):
                self.agent._chat_completion(
                    [{"role": "user", "content": "inspect"}],
                    tools=None,
                )
            self.agent.close()

        self.assertEqual(post.call_count, 2)
        close.assert_called_once_with()

    def test_chat_completion_closes_retryable_and_success_responses(self) -> None:
        retryable = MagicMock(status_code=503)
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
        }
        with (
            patch.object(
                self.agent._http_session,
                "post",
                side_effect=[retryable, response],
            ) as post,
            patch("inference.agent.tool_agent.time.sleep"),
        ):
            self.agent._chat_completion(
                [{"role": "user", "content": "inspect"}],
                tools=None,
            )

        self.assertEqual(post.call_count, 2)
        retryable.close.assert_called_once_with()
        response.close.assert_called_once_with()

    def test_chat_completion_retries_share_one_request_deadline(self) -> None:
        retryable = MagicMock(status_code=503)
        clock = [100.0]

        def advance(delay: float) -> None:
            clock[0] += delay

        with (
            patch.object(
                self.agent._http_session,
                "post",
                return_value=retryable,
            ) as post,
            patch(
                "inference.agent.tool_agent.time.monotonic",
                side_effect=lambda: clock[0],
            ),
            patch(
                "inference.agent.tool_agent.time.sleep",
                side_effect=advance,
            ) as sleep,
        ):
            with self.assertRaisesRegex(requests.Timeout, "deadline exceeded"):
                self.agent._chat_completion(
                    [{"role": "user", "content": "inspect"}],
                    tools=None,
                    request_timeout_seconds=0.25,
                )

        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["timeout"], 0.25)
        sleep.assert_called_once_with(0.25)
        retryable.close.assert_called_once_with()

    def test_chat_completion_rejects_malformed_provider_payload(self) -> None:
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"tool_calls": [None]}}],
        }
        with patch.object(
            self.agent._http_session,
            "post",
            return_value=response,
        ):
            with self.assertRaisesRegex(requests.RequestException, "invalid tool call"):
                self.agent._chat_completion(
                    [{"role": "user", "content": "inspect"}],
                    tools=None,
                )

        response.close.assert_called_once_with()

    def test_chat_completion_repairs_missing_and_duplicate_tool_call_ids(self) -> None:
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "duplicate" + ("x" * 500) + "\x00",
                                "function": {"name": "python", "arguments": "{}"},
                            },
                            {
                                "id": "duplicate" + ("x" * 500) + "\x00",
                                "function": {"name": "python", "arguments": "{}"},
                            },
                            {
                                "function": {"name": "python", "arguments": "{}"},
                            },
                        ]
                    }
                }
            ]
        }
        with patch.object(
            self.agent._http_session,
            "post",
            return_value=response,
        ):
            result = self.agent._chat_completion(
                [{"role": "user", "content": "inspect"}],
                tools=None,
            )

        tool_calls = result.message["tool_calls"]
        tool_call_ids = [tool_call["id"] for tool_call in tool_calls]
        self.assertEqual(len(tool_call_ids[0]), 128)
        self.assertNotIn("\x00", tool_call_ids[0])
        self.assertEqual(len(tool_call_ids), len(set(tool_call_ids)))
        self.assertTrue(all(tool_call_ids))
        self.assertTrue(
            all(tool_call["type"] == "function" for tool_call in tool_calls)
        )

    def test_malformed_provider_arguments_preserve_json_diagnostic(self) -> None:
        arguments, dispatch = self.agent._decode_tool_arguments(
            "python",
            '{"program":{"version":1,"body":',
        )

        self.assertIsNone(arguments)
        self.assertIsNotNone(dispatch)
        payload = json.loads(dispatch.content)
        self.assertEqual(payload["stage"], "schema")
        self.assertEqual(payload["error_category"], "MALFORMED_TOOL_ARGUMENTS")
        self.assertEqual(
            payload["diagnostics"][0]["code"],
            "IR_TOOL_ARGUMENTS_INVALID_JSON",
        )
        self.assertEqual(payload["diagnostics"][0]["path"], "arguments")
        self.assertIn("line 1, column", payload["error"])
        self.assertEqual(payload["compiler"]["version"], COMPILER_VERSION)

    def test_valid_provider_arguments_decode_without_dispatch_error(self) -> None:
        raw = json.dumps({"program": program({"kind": "pass"})})
        arguments, dispatch = self.agent._decode_tool_arguments("python", raw)

        self.assertIsNone(dispatch)
        self.assertEqual(arguments, {"program": program({"kind": "pass"})})

    def test_unknown_tool_returns_structured_repair_diagnostic(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python_source",
                {},
            )

        payload = json.loads(result.content)
        self.assertEqual(payload["stage"], "schema")
        self.assertEqual(payload["error_category"], "INVALID_TOOL_NAME")
        self.assertEqual(payload["diagnostics"][0]["code"], "IR_UNKNOWN_TOOL")
        self.assertEqual(payload["diagnostics"][0]["path"], "tool.name")
        self.assertIn("only available tool is `python`", payload["error"])

    def test_unknown_tool_name_is_sanitized_and_bounded(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "bad\nname\t" + ("x" * 500),
                {},
            )

        payload = json.loads(result.content)
        self.assertLessEqual(len(payload["tool"]), 80)
        self.assertNotIn("\n", payload["tool"])
        self.assertNotIn("\t", payload["tool"])
        self.assertTrue(payload["tool"].endswith("..."))

    def test_action_result_compaction_preserves_partial_batch_stop_detail(self) -> None:
        compact = self.agent._compact_action_result({
            "executed": True,
            "requested_count": 2,
            "executed_count": 1,
            "stopped_early": True,
            "stop_reason": "action_error",
            "stop_detail": "RuntimeError: second action failed",
        })
        self.assertEqual(compact["stop_reason"], "action_error")
        self.assertEqual(
            compact["stop_detail"],
            "RuntimeError: second action failed",
        )

    def test_raw_code_is_rejected_before_sandbox(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(Path(tmp) / "state.json", "python", {"code": "print(1)"})
        payload = json.loads(result.content)
        self.assertEqual(payload["error_category"], "RAW_SOURCE_NOT_ACCEPTED")
        self.assertEqual(payload["diagnostics"][0]["path"], "code")

    def test_raw_code_is_rejected_even_with_valid_program(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {
                    "program": program({"kind": "assign", "targets": [target("result")], "value": const(42)}),
                    "code": "result = 99",
                },
            )
        payload = json.loads(result.content)
        self.assertEqual(payload["error_category"], "RAW_SOURCE_NOT_ACCEPTED")
        self.assertNotIn("result", payload)

    def test_unsafe_program_is_rejected_before_sandbox_start(self) -> None:
        unsafe_program = program({"kind": "import", "names": [{"name": "subprocess"}]})
        with TemporaryDirectory() as tmp, patch(
            "inference.agent.tool_agent.run_sandboxed_python"
        ) as sandbox:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": unsafe_program},
            )
        payload = json.loads(result.content)
        self.assertEqual(payload["stage"], "safety")
        self.assertEqual(payload["error_category"], "UNSAFE_PROGRAM_IR")
        sandbox.assert_not_called()

    def test_missing_and_unknown_tool_arguments_are_rejected(self) -> None:
        cases = (
            ({}, "IR_REQUIRED_FIELD", "program"),
            ({"program": program({"kind": "pass"}), "extra": True}, "IR_UNKNOWN_TOOL_ARGUMENT", "extra"),
        )
        for arguments, code, path in cases:
            with self.subTest(arguments=arguments), TemporaryDirectory() as tmp:
                result = self.agent._dispatch_tool(Path(tmp) / "state.json", "python", arguments)
                payload = json.loads(result.content)
                self.assertEqual(payload["error_category"], "INVALID_TOOL_ARGUMENTS")
                self.assertEqual(payload["diagnostics"][0]["code"], code)
                self.assertEqual(payload["diagnostics"][0]["path"], path)

    def test_non_object_tool_arguments_are_rejected_structurally(self) -> None:
        for arguments in ([], [program({"kind": "pass"})], "program"):
            with self.subTest(arguments=arguments), TemporaryDirectory() as tmp:
                result = self.agent._dispatch_tool(
                    Path(tmp) / "state.json",
                    "python",
                    arguments,
                )
                payload = json.loads(result.content)
                self.assertEqual(payload["error_category"], "INVALID_TOOL_ARGUMENTS")
                self.assertEqual(
                    payload["diagnostics"][0]["code"],
                    "IR_TOOL_ARGUMENTS_NOT_OBJECT",
                )
                self.assertEqual(payload["diagnostics"][0]["path"], "arguments")

    def test_compiler_failure_metadata_is_complete(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": {"version": 1, "body": []}},
            )
        payload = json.loads(result.content)
        self.assertEqual(payload["compiler"]["version"], COMPILER_VERSION)
        self.assertEqual(payload["compiler"]["policy_version"], "duck-python-tool-policy/2")
        self.assertIn("python_version", payload["compiler"])
        self.assertEqual(payload["compiler"]["supported_ir_version"], 1)
        self.assertIn("max_ir_nodes", payload["compiler"]["limits"])
        self.assertEqual(
            payload["compiler"]["limits"]["max_raw_payload_depth"],
            MAX_RAW_PAYLOAD_DEPTH,
        )
        self.assertEqual(
            payload["compiler"]["limits"]["max_raw_payload_values"],
            MAX_RAW_PAYLOAD_VALUES,
        )
        self.assertEqual(
            payload["compiler"]["limits"]["max_container_items"],
            MAX_CONTAINER_ITEMS,
        )

    def test_tiny_tool_budget_preserves_compiler_repair_context(self) -> None:
        self.agent._tool_output_chars = 256
        malformed = {
            "version": 1,
            "body": [{"type": "expr", "value": {"kind": "constant", "value": 1}}],
        }

        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": malformed},
            )

        payload = json.loads(result.content)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["stage"], "schema")
        self.assertEqual(payload["error_category"], "INVALID_PROGRAM_IR")
        self.assertIn("rename `type` to `kind`", payload["recovery_hint"])
        self.assertLessEqual(len(result.content), self.agent._tool_output_chars * 2)

    def test_compiler_metadata_in_successful_result(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": program({"kind": "assign", "targets": [target("result")], "value": const(42)})},
            )
        payload = json.loads(result.content)
        self.assertEqual(payload["result"], 42)
        self.assertEqual(payload["compiler"]["version"], COMPILER_VERSION)
        self.assertEqual(len(payload["compiler"]["program_sha256"]), 64)
        self.assertEqual(len(payload["compiler"]["source_sha256"]), 64)

    def test_runtime_failure_reports_program_ir_statement_path(self) -> None:
        failing_program = program(
            {
                "kind": "function_def",
                "name": "fail",
                "body": [{"kind": "expr", "value": call("missing_name")}],
            },
            {"kind": "expr", "value": call("fail")},
        )
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": failing_program},
            )
        payload = json.loads(result.content)
        self.assertEqual(payload["stage"], "runtime")
        self.assertEqual(payload["error_category"], "UNDEFINED_VARIABLE")
        self.assertEqual(payload["diagnostics"][0]["code"], "IR_RUNTIME_ERROR")
        self.assertEqual(payload["diagnostics"][0]["path"], "program.body.0.body.0")
        self.assertIn("last_action_frame", payload["recovery_hint"])
        self.assertIn("bfs(frame, start, goal, blocked=None)", payload["recovery_hint"])

    def test_already_terminal_action_result_suppresses_later_action_calls(self) -> None:
        action_call = call(
            "action",
            {"kind": "list", "items": [const("LEFT")]},
        )
        action_program = program(
            {
                "kind": "assign",
                "targets": [target("first")],
                "value": action_call,
            },
            {
                "kind": "assign",
                "targets": [target("result")],
                "value": action_call,
            },
        )
        callback_count = 0

        def step_env(arguments):
            nonlocal callback_count
            callback_count += 1
            return {
                "executed": False,
                "action_num": 0,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "GAME_OVER",
                "valid_actions": [],
                "board_changed": False,
                "done": False,
                "level_completed": False,
                "game_over": True,
                "run_complete": False,
                "stopped_early": True,
                "stop_reason": "game_over",
            }

        self.agent._step_env_callback = step_env
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": action_program},
            )
        payload = json.loads(result.content)
        self.assertEqual(callback_count, 1)
        self.assertFalse(payload["result"]["executed"])
        self.assertEqual(payload["result"]["stop_reason"], "previous_game_over")
        self.assertEqual(payload["result"]["valid_actions"], [])

    def test_solver_stop_result_suppresses_later_action_calls(self) -> None:
        action_call = call(
            "action",
            {"kind": "list", "items": [const("LEFT")]},
        )
        action_program = program(
            {
                "kind": "assign",
                "targets": [target("first")],
                "value": action_call,
            },
            {
                "kind": "assign",
                "targets": [target("result")],
                "value": action_call,
            },
        )
        callback_count = 0

        def step_env(arguments):
            nonlocal callback_count
            callback_count += 1
            return {
                "executed": False,
                "action_num": 12,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "state": "PLAYING",
                "valid_actions": [],
                "board_changed": False,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "stopped_early": True,
                "stop_reason": "stopped",
                "error": "The solver action budget is exhausted.",
            }

        self.agent._step_env_callback = step_env
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": action_program},
            )
        payload = json.loads(result.content)
        self.assertEqual(callback_count, 1)
        self.assertFalse(payload["result"]["executed"])
        self.assertEqual(payload["result"]["stop_reason"], "previous_stopped")
        self.assertIn("budget is exhausted", payload["result"]["stop_detail"])

    def test_stdout_and_result_are_both_preserved(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": program(
                    {"kind": "expr", "value": call("print", const("summary"))},
                    {"kind": "assign", "targets": [target("result")], "value": const(42)},
                )},
            )
        payload = json.loads(result.content)
        self.assertIn("summary", payload["stdout"])
        self.assertEqual(payload["result"], 42)

    def test_truncated_result_metadata_reaches_tool_response(self) -> None:
        large_result = {
            "kind": "list_comprehension",
            "element": {"kind": "binary", "op": "mul", "left": const("x"), "right": const(1000)},
            "clauses": [{"target": target("item"), "iterable": call("range", const(1000))}],
        }
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(
                Path(tmp) / "state.json",
                "python",
                {"program": program({"kind": "assign", "targets": [target("result")], "value": large_result})},
            )
        payload = json.loads(result.content)
        self.assertTrue(payload["result_truncated"])
        self.assertIn("smaller summary", payload["result_warning"])


if __name__ == "__main__":
    unittest.main()
