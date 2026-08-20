from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from inference.agent.program_ir import (
    COMPILER_VERSION,
    MAX_COMPILER_DIAGNOSTICS,
    MAX_IR_DEPTH,
    MAX_IR_NODES,
    MAX_PROGRAM_CHARS,
    ProgramCompileError,
    compile_program,
    program_tool_parameters_schema,
)
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.tool_agent import ToolAgent


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

    def test_regular_import_still_allows_dotted_module(self) -> None:
        compiled = compile_program(program({
            "kind": "import",
            "names": [{"name": "collections.abc", "asname": "abc"}],
        }))
        self.assertEqual(compiled.source, "import collections.abc as abc\n")

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
        self.assertIn(captured.exception.error_category, {"PROGRAM_TOO_DEEP", "INVALID_PROGRAM_IR"})

    def test_tool_schema_has_hoisted_definitions(self) -> None:
        schema = program_tool_parameters_schema()
        self.assertEqual(schema["required"], ["program"])
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("$defs", schema)
        self.assertIn("body", schema["properties"]["program"]["properties"])
        self.assertIn("raw Python source is not accepted", schema["properties"]["program"]["description"])
        self.assertIn("never coerced", schema["properties"]["program"]["description"])
        self.assertIn("kind discriminator", schema["$defs"]["Stmt"]["description"])
        self.assertIn("unreachable statements", schema["$defs"]["Stmt"]["description"])
        self.assertIn("attribute, or subscript", schema["$defs"]["AugTarget"]["description"])
        self.assertIn("starred_target", schema["$defs"]["UnpackTarget"]["description"])
        self.assertLess(len(json.dumps(schema)), 20_000)


class ProgramIRSandboxTests(unittest.TestCase):
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

    def test_blocked_import_still_rejected_by_sandbox(self) -> None:
        compiled = compile_program(program({"kind": "import", "names": [{"name": "os"}]}))
        response = run_sandboxed_python(
            code=compiled.source,
            timeout_seconds=5,
            initial_state={},
            action_handler=lambda actions: {},
            strategy_handler=lambda update: update,
        )
        self.assertEqual(response.get("error_category"), "BLOCKED_MODULE")


class ProgramIRToolAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ToolAgent(model="m", provider="vllm", base_url="http://127.0.0.1:1/v1")

    def test_tool_requires_program_not_code(self) -> None:
        with TemporaryDirectory() as tmp:
            schema = self.agent._tools(Path(tmp) / "state.json")[0]["function"]["parameters"]
        self.assertEqual(schema["required"], ["program"])
        self.assertNotIn("code", schema["properties"])

    def test_raw_code_is_rejected_before_sandbox(self) -> None:
        with TemporaryDirectory() as tmp:
            result = self.agent._dispatch_tool(Path(tmp) / "state.json", "python", {"code": "print(1)"})
        payload = json.loads(result.content)
        self.assertEqual(payload["error_category"], "RAW_SOURCE_NOT_ACCEPTED")
        self.assertEqual(payload["diagnostics"][0]["path"], "program")

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


if __name__ == "__main__":
    unittest.main()
