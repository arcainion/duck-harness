"""Structured runtime compiler for The Duck's ephemeral Python tool.

The strict IR and deterministic AST lowering are derived from the user-supplied
LLM AST Compiler Prototype v0.17.  Repository editing, LibCST, optimizer, and
self-modification features are intentionally excluded from the runtime.
"""
from __future__ import annotations

import ast
from difflib import get_close_matches
import hashlib
import json
import keyword
import math
import platform
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from typing_extensions import TypeAliasType

from inference.agent.python_tool_policy import (
    BLOCKED_DYNAMIC_ATTRIBUTES,
    BLOCKED_MODULE_ATTRIBUTES,
    POLICY_VERSION,
    PROTECTED_RUNTIME_BINDINGS,
    SAFE_MODULES,
)


COMPILER_VERSION = "duck-program-ir/1.22"
PROGRAM_IR_VERSION = 1
MAX_IR_NODES = 512
MAX_IR_DEPTH = 32
MAX_PROGRAM_CHARS = 65_536
MAX_RAW_PAYLOAD_DEPTH = MAX_IR_DEPTH * 2 + 8
MAX_RAW_PAYLOAD_VALUES = MAX_IR_NODES * 32
MAX_COMPILER_DIAGNOSTICS = 24
MAX_IDENTIFIER_CHARS = 128
MAX_STRING_CHARS = 8192
MAX_INTEGER_BITS = 4096
MAX_CONTAINER_ITEMS = 1024
DISALLOWED_RUNTIME_NAMES = frozenset({
    "__import__", "breakpoint", "compile", "delattr", "eval", "exec",
    "getattr", "globals", "input", "locals", "open", "setattr", "vars",
})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _identifier(value: str, *, label: str = "identifier") -> str:
    if not isinstance(value, str) or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{label} must be a valid non-keyword Python identifier")
    if value.startswith("_"):
        raise ValueError(f"private {label}s are not allowed")
    if value in DISALLOWED_RUNTIME_NAMES:
        raise ValueError(f"{label} {value!r} is not allowed in runtime programs")
    if len(value) > MAX_IDENTIFIER_CHARS:
        raise ValueError(f"{label} exceeds {MAX_IDENTIFIER_CHARS} characters")
    return value


class NameExpr(StrictModel):
    kind: Literal["name"]
    name: str

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _identifier(value, label="name")


class ConstantExpr(StrictModel):
    kind: Literal["constant"]
    value: str | int | float | bool | None

    @field_validator("value")
    @classmethod
    def bounded_string(cls, value: str | int | float | bool | None):
        if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
            raise ValueError(f"string constant exceeds {MAX_STRING_CHARS} characters")
        if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > MAX_INTEGER_BITS:
            raise ValueError(f"integer constant exceeds {MAX_INTEGER_BITS} bits")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("floating-point constants must be finite")
        return value


class AttributeExpr(StrictModel):
    kind: Literal["attribute"]
    value: "Expr"
    attr: str

    @field_validator("attr")
    @classmethod
    def valid_attr(cls, value: str) -> str:
        return _identifier(value, label="attribute")


class SliceExpr(StrictModel):
    kind: Literal["slice"]
    lower: "Expr | None" = None
    upper: "Expr | None" = None
    step: "Expr | None" = None


class SubscriptExpr(StrictModel):
    kind: Literal["subscript"]
    value: "Expr"
    index: "Expr | SliceExpr"


class ListExpr(StrictModel):
    kind: Literal["list"]
    items: list["Expr"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)


class TupleExpr(StrictModel):
    kind: Literal["tuple"]
    items: list["Expr"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)


class SetExpr(StrictModel):
    kind: Literal["set"]
    items: list["Expr"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)


class DictEntry(StrictModel):
    key: "Expr | None"
    value: "Expr"


class DictExpr(StrictModel):
    kind: Literal["dict"]
    entries: list[DictEntry] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)


class UnaryExpr(StrictModel):
    kind: Literal["unary"]
    op: Literal["positive", "negative", "not", "invert"]
    operand: "Expr"


BinaryOperator = Literal[
    "add", "sub", "mul", "div", "floordiv", "mod", "pow",
    "bit_or", "bit_xor", "bit_and", "left_shift", "right_shift",
]


class BinaryExpr(StrictModel):
    kind: Literal["binary"]
    op: BinaryOperator
    left: "Expr"
    right: "Expr"


class BoolExpr(StrictModel):
    kind: Literal["bool"]
    op: Literal["and", "or"]
    values: list["Expr"] = Field(min_length=2, max_length=MAX_CONTAINER_ITEMS)


class CompareExpr(StrictModel):
    kind: Literal["compare"]
    left: "Expr"
    ops: list[Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in", "is", "is_not"]] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)
    comparators: list["Expr"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)

    @model_validator(mode="after")
    def matching_operands(self):
        if len(self.ops) != len(self.comparators):
            raise ValueError("ops and comparators must have the same length")
        return self


class CallExpr(StrictModel):
    kind: Literal["call"]
    function: "Expr"
    args: list["Expr"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)
    star_args: list["Expr"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)
    keywords: dict[str, "Expr"] = Field(default_factory=dict, max_length=MAX_CONTAINER_ITEMS)
    star_keywords: list["Expr"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)

    @field_validator("keywords")
    @classmethod
    def valid_keywords(cls, value: dict[str, "Expr"]):
        for name in value:
            _identifier(name, label="keyword argument")
        return value


class IfExpr(StrictModel):
    kind: Literal["if_expr"]
    condition: "Expr"
    then: "Expr"
    otherwise: "Expr"


class NameTarget(StrictModel):
    kind: Literal["name_target"]
    name: str

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _identifier(value, label="target name")


class AttributeTarget(StrictModel):
    kind: Literal["attribute_target"]
    value: "Expr"
    attr: str

    @field_validator("attr")
    @classmethod
    def valid_attr(cls, value: str) -> str:
        return _identifier(value, label="target attribute")


class SubscriptTarget(StrictModel):
    kind: Literal["subscript_target"]
    value: "Expr"
    index: "Expr | SliceExpr"


AugTarget = TypeAliasType(
    "AugTarget",
    Annotated[
        Union[NameTarget, AttributeTarget, SubscriptTarget],
        Field(discriminator="kind"),
    ],
)


class StarredTarget(StrictModel):
    kind: Literal["starred_target"]
    target: AugTarget


class TupleTarget(StrictModel):
    kind: Literal["tuple_target"]
    items: list["UnpackTarget"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)

    @model_validator(mode="after")
    def single_star(self):
        if sum(isinstance(item, StarredTarget) for item in self.items) > 1:
            raise ValueError("tuple unpacking may contain at most one starred target")
        return self


class ListTarget(StrictModel):
    kind: Literal["list_target"]
    items: list["UnpackTarget"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)

    @model_validator(mode="after")
    def single_star(self):
        if sum(isinstance(item, StarredTarget) for item in self.items) > 1:
            raise ValueError("list unpacking may contain at most one starred target")
        return self


StoreTarget = TypeAliasType(
    "StoreTarget",
    Annotated[
        Union[NameTarget, AttributeTarget, SubscriptTarget, TupleTarget, ListTarget],
        Field(discriminator="kind"),
    ],
)

UnpackTarget = TypeAliasType(
    "UnpackTarget",
    Annotated[
        Union[
            NameTarget, AttributeTarget, SubscriptTarget, TupleTarget,
            ListTarget, StarredTarget,
        ],
        Field(discriminator="kind"),
    ],
)


class ComprehensionClause(StrictModel):
    target: StoreTarget
    iterable: "Expr"
    conditions: list["Expr"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)


class ListComprehensionExpr(StrictModel):
    kind: Literal["list_comprehension"]
    element: "Expr"
    clauses: list[ComprehensionClause] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)


class SetComprehensionExpr(StrictModel):
    kind: Literal["set_comprehension"]
    element: "Expr"
    clauses: list[ComprehensionClause] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)


class GeneratorComprehensionExpr(StrictModel):
    kind: Literal["generator_comprehension"]
    element: "Expr"
    clauses: list[ComprehensionClause] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)


class DictComprehensionExpr(StrictModel):
    kind: Literal["dict_comprehension"]
    key: "Expr"
    value: "Expr"
    clauses: list[ComprehensionClause] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)


Expr = TypeAliasType(
    "Expr",
    Annotated[
        Union[
            NameExpr, ConstantExpr, AttributeExpr, SubscriptExpr, ListExpr,
            TupleExpr, SetExpr, DictExpr, UnaryExpr, BinaryExpr, BoolExpr,
            CompareExpr, CallExpr, IfExpr, ListComprehensionExpr,
            SetComprehensionExpr, GeneratorComprehensionExpr,
            DictComprehensionExpr,
        ],
        Field(discriminator="kind"),
    ],
)


class AssignStmt(StrictModel):
    kind: Literal["assign"]
    targets: list[StoreTarget] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)
    value: Expr


class AugAssignStmt(StrictModel):
    kind: Literal["aug_assign"]
    target: AugTarget
    op: BinaryOperator
    value: Expr


class ExprStmt(StrictModel):
    kind: Literal["expr"]
    value: Expr


class IfStmt(StrictModel):
    kind: Literal["if"]
    condition: Expr
    body: list["Stmt"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)
    orelse: list["Stmt"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)


class ForStmt(StrictModel):
    kind: Literal["for"]
    target: StoreTarget
    iterable: Expr
    body: list["Stmt"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)
    orelse: list["Stmt"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)


class WhileStmt(StrictModel):
    kind: Literal["while"]
    condition: Expr
    body: list["Stmt"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)
    orelse: list["Stmt"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)


class ParameterIR(StrictModel):
    name: str
    default: Expr | None = None

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _identifier(value, label="parameter")


class FunctionDefStmt(StrictModel):
    kind: Literal["function_def"]
    name: str
    parameters: list[ParameterIR] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)
    vararg: str | None = None
    kwarg: str | None = None
    body: list["Stmt"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _identifier(value, label="function name")

    @field_validator("vararg", "kwarg")
    @classmethod
    def valid_variadic_name(cls, value: str | None) -> str | None:
        return None if value is None else _identifier(value, label="variadic parameter")

    @model_validator(mode="after")
    def valid_parameter_order(self):
        seen_default = False
        seen_names: set[str] = set()
        for parameter in self.parameters:
            if parameter.name in seen_names:
                raise ValueError(f"duplicate parameter: {parameter.name}")
            seen_names.add(parameter.name)
            if parameter.default is not None:
                seen_default = True
            elif seen_default:
                raise ValueError("non-default parameter follows default parameter")
        for parameter_name in (self.vararg, self.kwarg):
            if parameter_name is None:
                continue
            if parameter_name in seen_names:
                raise ValueError(f"duplicate parameter: {parameter_name}")
            seen_names.add(parameter_name)
        return self


class ReturnStmt(StrictModel):
    kind: Literal["return"]
    value: Expr | None = None


class ImportAlias(StrictModel):
    name: str
    asname: str | None = None

    @field_validator("name")
    @classmethod
    def valid_import_name(cls, value: str) -> str:
        parts = value.split(".")
        if not parts or any(_identifier(part, label="import name") != part for part in parts):
            raise ValueError("import name must be dotted valid identifiers")
        return value

    @field_validator("asname")
    @classmethod
    def valid_alias(cls, value: str | None) -> str | None:
        return None if value is None else _identifier(value, label="import alias")


class ImportStmt(StrictModel):
    kind: Literal["import"]
    names: list[ImportAlias] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)


class FromImportAlias(StrictModel):
    name: str
    asname: str | None = None

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return _identifier(value, label="imported name")

    @field_validator("asname")
    @classmethod
    def valid_alias(cls, value: str | None) -> str | None:
        return None if value is None else _identifier(value, label="import alias")


class FromImportStmt(StrictModel):
    kind: Literal["from_import"]
    module: str
    names: list[FromImportAlias] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)

    @field_validator("module")
    @classmethod
    def valid_module(cls, value: str) -> str:
        parts = value.split(".")
        if not parts or any(_identifier(part, label="module") != part for part in parts):
            raise ValueError("module must be dotted valid identifiers")
        return value


class BreakStmt(StrictModel):
    kind: Literal["break"]


class ContinueStmt(StrictModel):
    kind: Literal["continue"]


class PassStmt(StrictModel):
    kind: Literal["pass"]


HandledException = Literal[
    "AttributeError",
    "IndexError",
    "KeyError",
    "OverflowError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
]


class ExceptHandlerIR(StrictModel):
    exceptions: list[HandledException] = Field(
        min_length=1,
        max_length=MAX_CONTAINER_ITEMS,
    )
    name: str | None = None
    body: list["Stmt"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str | None) -> str | None:
        return None if value is None else _identifier(value, label="exception name")


class TryStmt(StrictModel):
    kind: Literal["try"]
    body: list["Stmt"] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)
    handlers: list[ExceptHandlerIR] = Field(
        min_length=1,
        max_length=MAX_CONTAINER_ITEMS,
    )
    orelse: list["Stmt"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)
    finalbody: list["Stmt"] = Field(default_factory=list, max_length=MAX_CONTAINER_ITEMS)


class RaiseStmt(StrictModel):
    kind: Literal["raise"]
    exception: HandledException
    message: Expr | None = None


Stmt = TypeAliasType(
    "Stmt",
    Annotated[
        Union[
            AssignStmt, AugAssignStmt, ExprStmt, IfStmt, ForStmt, WhileStmt,
            FunctionDefStmt, ReturnStmt, ImportStmt, FromImportStmt, BreakStmt,
            ContinueStmt, PassStmt, TryStmt, RaiseStmt,
        ],
        Field(discriminator="kind"),
    ],
)


class ProgramIR(StrictModel):
    version: Literal[1]
    body: list[Stmt] = Field(min_length=1, max_length=MAX_CONTAINER_ITEMS)

    @field_validator("version", mode="before")
    @classmethod
    def strict_version_type(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("version must be the integer 1")
        return value


@dataclass(frozen=True)
class CompilerDiagnostic:
    code: str
    path: str
    message: str
    hint: str | None = None

    def to_dict(self) -> dict[str, str]:
        result = {"code": self.code, "path": self.path, "message": self.message}
        if self.hint:
            result["hint"] = self.hint
        return result


class ProgramCompileError(ValueError):
    def __init__(self, *, stage: str, error_category: str, diagnostics: list[CompilerDiagnostic]):
        self.stage = stage
        self.error_category = error_category
        self.diagnostics = diagnostics
        super().__init__(diagnostics[0].message if diagnostics else error_category)

    def payload(self) -> dict[str, Any]:
        payload = {
            "stage": self.stage,
            "error_category": self.error_category,
            "error": str(self),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }
        hints = list(dict.fromkeys(item.hint for item in self.diagnostics if item.hint))
        if hints:
            payload["recovery_hint"] = " ".join(hints[:3])
        return payload


@dataclass(frozen=True)
class SourceMapEntry:
    path: str
    line: int
    end_line: int


@dataclass(frozen=True)
class CompiledProgram:
    program: ProgramIR
    source: str
    node_count: int
    program_sha256: str
    source_sha256: str
    source_map: tuple[SourceMapEntry, ...]

    def path_for_line(self, line: Any) -> str | None:
        """Resolve a generated Python line to its most specific IR statement."""
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            return None
        for entry in reversed(self.source_map):
            if entry.line <= line <= entry.end_line:
                return entry.path
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            **compiler_runtime_metadata(),
            "ir_version": self.program.version,
            "node_count": self.node_count,
            "generated_chars": len(self.source),
            "program_sha256": self.program_sha256,
            "source_sha256": self.source_sha256,
            "source_map_entries": len(self.source_map),
        }


def compiler_runtime_metadata() -> dict[str, Any]:
    """Return compiler identity and limits for success and failure traces."""
    return {
        "version": COMPILER_VERSION,
        "python_version": platform.python_version(),
        "policy_version": POLICY_VERSION,
        "supported_ir_version": PROGRAM_IR_VERSION,
        "limits": {
            "max_ir_nodes": MAX_IR_NODES,
            "max_ir_depth": MAX_IR_DEPTH,
            "max_program_chars": MAX_PROGRAM_CHARS,
            "max_raw_payload_depth": MAX_RAW_PAYLOAD_DEPTH,
            "max_raw_payload_values": MAX_RAW_PAYLOAD_VALUES,
            "max_integer_bits": MAX_INTEGER_BITS,
            "max_container_items": MAX_CONTAINER_ITEMS,
        },
    }


def _path(location: tuple[Any, ...]) -> str:
    # Pydantic inserts tagged-union choices (for example ``assign`` and
    # ``name_target``) into error locations. They are schema implementation
    # details, not addressable ProgramIR fields, so omit them from repair paths.
    variant_tags = {
        "name", "constant", "attribute", "slice", "subscript", "list",
        "tuple", "set", "dict", "unary", "binary", "bool", "compare",
        "call", "if_expr", "list_comprehension", "set_comprehension",
        "dict_comprehension", "generator_comprehension", "name_target", "attribute_target",
        "subscript_target", "tuple_target", "list_target", "starred_target", "assign",
        "aug_assign", "expr", "if", "for", "while", "function_def",
        "return", "import", "from_import", "break", "continue", "pass", "try",
        "raise",
    }
    union_parent_fields = {
        "args", "body", "clauses", "comparators", "condition", "conditions",
        "element", "function", "index", "items", "iterable", "key", "left",
        "lower", "message", "operand", "orelse", "otherwise", "right", "step", "target",
        "star_args", "star_keywords", "targets", "then", "upper", "value", "values",
    }
    clean_location: list[Any] = []
    for index, item in enumerate(location):
        previous = location[index - 1] if index else None
        is_union_choice = item in variant_tags and (
            isinstance(previous, int) or previous in union_parent_fields
        )
        if item == "name" and index == len(location) - 1:
            is_union_choice = False
        if not is_union_choice:
            clean_location.append(item)
    return ".".join(str(item) for item in ("program", *clean_location))


def _schema_diagnostic_code(error_type: str) -> str:
    """Map dependency-specific validation errors to stable public codes."""
    if error_type == "missing":
        return "IR_REQUIRED_FIELD"
    if error_type == "extra_forbidden":
        return "IR_UNKNOWN_FIELD"
    if error_type in {"union_tag_invalid", "union_tag_not_found"}:
        return "IR_INVALID_KIND"
    if error_type in {"too_short", "greater_than_equal"}:
        return "IR_TOO_SHORT"
    if error_type in {"too_long", "less_than_equal"}:
        return "IR_TOO_LONG"
    if error_type.endswith("_type") or error_type.endswith("_parsing"):
        return "IR_INVALID_TYPE"
    if error_type in {"literal_error", "value_error"}:
        return "IR_INVALID_VALUE"
    return "IR_SCHEMA_INVALID"


def _schema_diagnostic(error: dict[str, Any]) -> CompilerDiagnostic:
    error_type = str(error["type"])
    path = _path(tuple(error["loc"]))
    message = str(error["msg"])
    hint: str | None = None
    context = error.get("ctx") if isinstance(error.get("ctx"), dict) else {}

    if error_type == "union_tag_not_found":
        message = "Node requires the `kind` discriminator."
        input_value = error.get("input")
        if isinstance(input_value, dict) and "type" in input_value:
            hint = f"At {path}, rename `type` to `kind`."
        else:
            hint = f"At {path}, add a string `kind` field for the intended node."
    elif error_type == "union_tag_invalid":
        tag = str(context.get("tag", "")).strip()
        expected = [
            item.strip().strip("'")
            for item in str(context.get("expected_tags", "")).split(",")
            if item.strip().strip("'")
        ]
        message = f"Unknown node kind {tag!r}."
        closest = get_close_matches(tag, expected, n=1, cutoff=0.35)
        if closest:
            hint = f"At {path}, replace `kind`: {tag!r} with {closest[0]!r}."
        elif expected:
            hint = f"At {path}, set `kind` to one of: {', '.join(expected)}."
    elif error_type == "missing":
        field = str(tuple(error["loc"])[-1])
        message = "Required field is missing."
        hint = f"Add required field `{field}` at {path}."
    elif error_type == "extra_forbidden":
        field = str(tuple(error["loc"])[-1])
        message = "Field is not allowed for this node."
        hint = f"Remove unsupported field `{field}` at {path}."

    return CompilerDiagnostic(
        code=_schema_diagnostic_code(error_type),
        path=path,
        message=message,
        hint=hint,
    )


def _schema_diagnostics(exc: ValidationError) -> list[CompilerDiagnostic]:
    errors = exc.errors(include_url=False)
    diagnostics = [
        _schema_diagnostic(error)
        for error in errors[:MAX_COMPILER_DIAGNOSTICS]
    ]
    omitted = len(errors) - len(diagnostics)
    if omitted:
        diagnostics.append(CompilerDiagnostic(
            "IR_DIAGNOSTICS_TRUNCATED",
            "program",
            f"{omitted} additional validation diagnostic(s) omitted",
        ))
    return diagnostics


def _preflight_payload(payload: Any) -> None:
    """Bound untrusted JSON-like input before recursive model validation.

    Validated IR complexity remains the public semantic limit. These wider
    raw-tree limits prevent malformed payloads, including unknown fields, from
    making Pydantic traverse an effectively unbounded object graph first.
    """
    value_count = 0
    canonical_chars = 0
    active_containers: set[int] = set()
    stack: list[tuple[Any, tuple[Any, ...], int, bool]] = [
        (payload, (), 0, False)
    ]

    def fail(*, category: str, code: str, path: tuple[Any, ...], message: str) -> None:
        raise ProgramCompileError(
            stage="limits",
            error_category=category,
            diagnostics=[CompilerDiagnostic(code, _path(path), message)],
        )

    while stack:
        value, path, depth, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(value))
            continue

        value_count += 1
        if value_count > MAX_RAW_PAYLOAD_VALUES:
            fail(
                category="PROGRAM_TOO_LARGE",
                code="IR_RAW_VALUE_LIMIT",
                path=path,
                message=(
                    f"raw program payload contains more than "
                    f"{MAX_RAW_PAYLOAD_VALUES} values"
                ),
            )
        if depth > MAX_RAW_PAYLOAD_DEPTH:
            fail(
                category="PROGRAM_TOO_DEEP",
                code="IR_RAW_DEPTH_LIMIT",
                path=path,
                message=(
                    f"raw program payload depth exceeds "
                    f"{MAX_RAW_PAYLOAD_DEPTH} before schema validation"
                ),
            )

        if isinstance(value, (dict, list)):
            identity = id(value)
            if identity in active_containers:
                fail(
                    category="INVALID_PROGRAM_IR",
                    code="IR_CYCLIC_PAYLOAD",
                    path=path,
                    message="program payload must be an acyclic JSON value",
                )
            active_containers.add(identity)
            stack.append((value, path, depth, True))

        if isinstance(value, dict):
            if len(value) > MAX_RAW_PAYLOAD_VALUES - value_count:
                fail(
                    category="PROGRAM_TOO_LARGE",
                    code="IR_RAW_VALUE_LIMIT",
                    path=path,
                    message=(
                        f"raw program payload contains more than "
                        f"{MAX_RAW_PAYLOAD_VALUES} values"
                    ),
                )
            canonical_chars += 2 + max(0, len(value) - 1) + len(value)
            for key, item in value.items():
                if isinstance(key, str):
                    canonical_chars += len(json.dumps(key, ensure_ascii=True))
                stack.append((item, (*path, key), depth + 1, False))
        elif isinstance(value, list):
            if len(value) > MAX_RAW_PAYLOAD_VALUES - value_count:
                fail(
                    category="PROGRAM_TOO_LARGE",
                    code="IR_RAW_VALUE_LIMIT",
                    path=path,
                    message=(
                        f"raw program payload contains more than "
                        f"{MAX_RAW_PAYLOAD_VALUES} values"
                    ),
                )
            canonical_chars += 2 + max(0, len(value) - 1)
            for index in range(len(value) - 1, -1, -1):
                stack.append((value[index], (*path, index), depth + 1, False))
        elif value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, str) and len(value) > MAX_PROGRAM_CHARS:
                canonical_chars = MAX_PROGRAM_CHARS + 1
            else:
                try:
                    canonical_chars += len(json.dumps(value, ensure_ascii=True))
                except (TypeError, ValueError):
                    # Schema validation supplies the more useful type/value error.
                    pass

        if canonical_chars > MAX_PROGRAM_CHARS:
            fail(
                category="PROGRAM_TOO_LARGE",
                code="IR_SIZE_LIMIT",
                path=path,
                message=(
                    f"raw canonical program exceeds {MAX_PROGRAM_CHARS} "
                    "characters before schema validation"
                ),
            )


def _validate_statement_context(program: ProgramIR) -> None:
    """Reject context-sensitive syntax with exact ProgramIR paths."""
    diagnostics: list[CompilerDiagnostic] = []

    def statement_terminates(statement: Stmt, *, function_depth: int, loop_depth: int) -> bool:
        if isinstance(statement, ReturnStmt):
            return function_depth > 0
        if isinstance(statement, RaiseStmt):
            return True
        if isinstance(statement, (BreakStmt, ContinueStmt)):
            return loop_depth > 0
        if isinstance(statement, IfStmt) and statement.orelse:
            return block_terminates(
                statement.body,
                function_depth=function_depth,
                loop_depth=loop_depth,
            ) and block_terminates(
                statement.orelse,
                function_depth=function_depth,
                loop_depth=loop_depth,
            )
        return False

    def block_terminates(statements: list[Stmt], *, function_depth: int, loop_depth: int) -> bool:
        return any(
            statement_terminates(
                statement,
                function_depth=function_depth,
                loop_depth=loop_depth,
            )
            for statement in statements
        )

    def visit_statements(
        statements: list[Stmt],
        path: tuple[Any, ...],
        *,
        function_depth: int,
        loop_depth: int,
    ) -> None:
        terminated = False
        for index, statement in enumerate(statements):
            statement_path = (*path, index)
            rendered_path = _path(statement_path)
            if terminated:
                diagnostics.append(CompilerDiagnostic(
                    "IR_UNREACHABLE_STATEMENT",
                    rendered_path,
                    "statement is unreachable because an earlier statement always terminates this block",
                ))
                continue
            if isinstance(statement, ReturnStmt) and function_depth == 0:
                diagnostics.append(CompilerDiagnostic(
                    "IR_RETURN_OUTSIDE_FUNCTION",
                    rendered_path,
                    "return is only valid inside a function body",
                ))
            elif isinstance(statement, BreakStmt) and loop_depth == 0:
                diagnostics.append(CompilerDiagnostic(
                    "IR_BREAK_OUTSIDE_LOOP",
                    rendered_path,
                    "break is only valid inside a for or while body",
                ))
            elif isinstance(statement, ContinueStmt) and loop_depth == 0:
                diagnostics.append(CompilerDiagnostic(
                    "IR_CONTINUE_OUTSIDE_LOOP",
                    rendered_path,
                    "continue is only valid inside a for or while body",
                ))

            if isinstance(statement, IfStmt):
                visit_statements(statement.body, (*statement_path, "body"), function_depth=function_depth, loop_depth=loop_depth)
                visit_statements(statement.orelse, (*statement_path, "orelse"), function_depth=function_depth, loop_depth=loop_depth)
            elif isinstance(statement, (ForStmt, WhileStmt)):
                visit_statements(statement.body, (*statement_path, "body"), function_depth=function_depth, loop_depth=loop_depth + 1)
                # The else suite runs after this loop; only an enclosing loop
                # remains a valid target for break or continue.
                visit_statements(statement.orelse, (*statement_path, "orelse"), function_depth=function_depth, loop_depth=loop_depth)
            elif isinstance(statement, FunctionDefStmt):
                # A nested function cannot target loops in its enclosing scope.
                visit_statements(statement.body, (*statement_path, "body"), function_depth=function_depth + 1, loop_depth=0)
            elif isinstance(statement, TryStmt):
                seen_exceptions: set[str] = set()
                visit_statements(statement.body, (*statement_path, "body"), function_depth=function_depth, loop_depth=loop_depth)
                for handler_index, handler in enumerate(statement.handlers):
                    for exception_index, exception_name in enumerate(handler.exceptions):
                        if exception_name in seen_exceptions:
                            diagnostics.append(CompilerDiagnostic(
                                "IR_DUPLICATE_EXCEPTION_HANDLER",
                                _path((*statement_path, "handlers", handler_index, "exceptions", exception_index)),
                                f"{exception_name} is already handled by an earlier handler",
                            ))
                        seen_exceptions.add(exception_name)
                    visit_statements(
                        handler.body,
                        (*statement_path, "handlers", handler_index, "body"),
                        function_depth=function_depth,
                        loop_depth=loop_depth,
                    )
                visit_statements(statement.orelse, (*statement_path, "orelse"), function_depth=function_depth, loop_depth=loop_depth)
                visit_statements(statement.finalbody, (*statement_path, "finalbody"), function_depth=function_depth, loop_depth=loop_depth)
            terminated = statement_terminates(
                statement,
                function_depth=function_depth,
                loop_depth=loop_depth,
            )

    visit_statements(program.body, ("body",), function_depth=0, loop_depth=0)
    if diagnostics:
        omitted = len(diagnostics) - MAX_COMPILER_DIAGNOSTICS
        if omitted > 0:
            diagnostics = diagnostics[:MAX_COMPILER_DIAGNOSTICS]
            diagnostics.append(CompilerDiagnostic(
                "IR_DIAGNOSTICS_TRUNCATED",
                "program",
                f"{omitted} additional structural diagnostic(s) omitted",
            ))
        raise ProgramCompileError(
            stage="structure",
            error_category="INVALID_CONTROL_FLOW",
            diagnostics=diagnostics,
        )


def _validate_ir_safety(program: ProgramIR) -> None:
    """Reject known unsafe capabilities before AST lowering or subprocess startup."""
    diagnostics: list[CompilerDiagnostic] = []
    omitted = 0

    def add(code: str, path: tuple[Any, ...], message: str) -> None:
        nonlocal omitted
        if len(diagnostics) < MAX_COMPILER_DIAGNOSTICS:
            diagnostics.append(CompilerDiagnostic(code, _path(path), message))
        else:
            omitted += 1

    def protect(name: str, path: tuple[Any, ...]) -> None:
        if name in PROTECTED_RUNTIME_BINDINGS:
            add(
                "IR_PROTECTED_RUNTIME_BINDING",
                path,
                f"runtime binding {name!r} is injected and cannot be overwritten",
            )

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, ImportStmt):
            for index, alias in enumerate(value.names):
                if alias.name not in SAFE_MODULES:
                    add(
                        "IR_BLOCKED_MODULE",
                        (*path, "names", index, "name"),
                        f"module {alias.name!r} is not available in the sandbox",
                    )
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                protect(
                    bound_name,
                    (*path, "names", index, "asname" if alias.asname else "name"),
                )
        elif isinstance(value, FromImportStmt):
            if value.module not in SAFE_MODULES:
                add(
                    "IR_BLOCKED_MODULE",
                    (*path, "module"),
                    f"module {value.module!r} is not available in the sandbox",
                )
            blocked_names = BLOCKED_MODULE_ATTRIBUTES.get(value.module, frozenset())
            for index, alias in enumerate(value.names):
                if alias.name in blocked_names:
                    add(
                        "IR_BLOCKED_DYNAMIC_ATTRIBUTE",
                        (*path, "names", index, "name"),
                        f"dynamic attribute helper {value.module}.{alias.name} is not allowed",
                    )
                protect(
                    alias.asname or alias.name,
                    (*path, "names", index, "asname" if alias.asname else "name"),
                )
        elif isinstance(value, (AttributeExpr, AttributeTarget)):
            if value.attr in BLOCKED_DYNAMIC_ATTRIBUTES:
                add(
                    "IR_BLOCKED_DYNAMIC_ATTRIBUTE",
                    (*path, "attr"),
                    f"dynamic attribute helper {value.attr!r} is not allowed",
                )

        if isinstance(value, NameTarget):
            protect(value.name, (*path, "name"))
        elif isinstance(value, FunctionDefStmt):
            protect(value.name, (*path, "name"))
            if value.vararg is not None:
                protect(value.vararg, (*path, "vararg"))
            if value.kwarg is not None:
                protect(value.kwarg, (*path, "kwarg"))
        elif isinstance(value, ParameterIR):
            protect(value.name, (*path, "name"))
        elif isinstance(value, ExceptHandlerIR) and value.name is not None:
            protect(value.name, (*path, "name"))

        if isinstance(value, BaseModel):
            for field_name in type(value).model_fields:
                visit(getattr(value, field_name), (*path, field_name))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, (*path, index))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, (*path, key))

    visit(program, ())
    if diagnostics:
        if omitted:
            diagnostics.append(CompilerDiagnostic(
                "IR_DIAGNOSTICS_TRUNCATED",
                "program",
                f"{omitted} additional safety diagnostic(s) omitted",
            ))
        raise ProgramCompileError(
            stage="safety",
            error_category="UNSAFE_PROGRAM_IR",
            diagnostics=diagnostics,
        )


def _canonical_program_json(program: ProgramIR) -> str:
    return json.dumps(
        program.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _program_sha256(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _complexity(value: Any, *, depth: int = 0) -> tuple[int, int]:
    if isinstance(value, BaseModel):
        own = 0 if isinstance(value, ProgramIR) else 1
        child_depth = depth + own
        child_rows = [
            _complexity(getattr(value, field_name), depth=child_depth)
            for field_name in type(value).model_fields
        ]
        return own + sum(row[0] for row in child_rows), max(
            [child_depth, *(row[1] for row in child_rows)]
        )
    if isinstance(value, dict):
        child_rows = [_complexity(item, depth=depth) for item in value.values()]
    elif isinstance(value, list):
        child_rows = [_complexity(item, depth=depth) for item in value]
    else:
        return 0, depth
    return sum(row[0] for row in child_rows), max([depth, *(row[1] for row in child_rows)])


_BINOPS: dict[str, type[ast.operator]] = {
    "add": ast.Add, "sub": ast.Sub, "mul": ast.Mult, "div": ast.Div,
    "floordiv": ast.FloorDiv, "mod": ast.Mod, "pow": ast.Pow,
    "bit_or": ast.BitOr, "bit_xor": ast.BitXor, "bit_and": ast.BitAnd,
    "left_shift": ast.LShift, "right_shift": ast.RShift,
}
_UNARYOPS: dict[str, type[ast.unaryop]] = {
    "positive": ast.UAdd, "negative": ast.USub, "not": ast.Not, "invert": ast.Invert,
}
_CMPOPS: dict[str, type[ast.cmpop]] = {
    "eq": ast.Eq, "ne": ast.NotEq, "lt": ast.Lt, "lte": ast.LtE,
    "gt": ast.Gt, "gte": ast.GtE, "in": ast.In, "not_in": ast.NotIn,
    "is": ast.Is, "is_not": ast.IsNot,
}


def _lower_target(node: StoreTarget | StarredTarget) -> ast.expr:
    if isinstance(node, NameTarget):
        return ast.Name(id=node.name, ctx=ast.Store())
    if isinstance(node, AttributeTarget):
        return ast.Attribute(value=_lower_expr(node.value), attr=node.attr, ctx=ast.Store())
    if isinstance(node, SubscriptTarget):
        return ast.Subscript(value=_lower_expr(node.value), slice=_lower_slice(node.index), ctx=ast.Store())
    if isinstance(node, StarredTarget):
        return ast.Starred(value=_lower_target(node.target), ctx=ast.Store())
    if isinstance(node, TupleTarget):
        return ast.Tuple(elts=[_lower_target(item) for item in node.items], ctx=ast.Store())
    if isinstance(node, ListTarget):
        return ast.List(elts=[_lower_target(item) for item in node.items], ctx=ast.Store())
    raise TypeError(f"unsupported target: {type(node).__name__}")


def _lower_slice(node: Expr | SliceExpr) -> ast.expr | ast.slice:
    if isinstance(node, SliceExpr):
        return ast.Slice(
            lower=None if node.lower is None else _lower_expr(node.lower),
            upper=None if node.upper is None else _lower_expr(node.upper),
            step=None if node.step is None else _lower_expr(node.step),
        )
    return _lower_expr(node)


def _lower_comprehensions(clauses: list[ComprehensionClause]) -> list[ast.comprehension]:
    return [
        ast.comprehension(
            target=_lower_target(clause.target),
            iter=_lower_expr(clause.iterable),
            ifs=[_lower_expr(item) for item in clause.conditions],
            is_async=0,
        )
        for clause in clauses
    ]


def _lower_expr(node: Expr) -> ast.expr:
    if isinstance(node, NameExpr):
        return ast.Name(id=node.name, ctx=ast.Load())
    if isinstance(node, ConstantExpr):
        return ast.Constant(value=node.value)
    if isinstance(node, AttributeExpr):
        return ast.Attribute(value=_lower_expr(node.value), attr=node.attr, ctx=ast.Load())
    if isinstance(node, SubscriptExpr):
        return ast.Subscript(value=_lower_expr(node.value), slice=_lower_slice(node.index), ctx=ast.Load())
    if isinstance(node, ListExpr):
        return ast.List(elts=[_lower_expr(item) for item in node.items], ctx=ast.Load())
    if isinstance(node, TupleExpr):
        return ast.Tuple(elts=[_lower_expr(item) for item in node.items], ctx=ast.Load())
    if isinstance(node, SetExpr):
        return ast.Set(elts=[_lower_expr(item) for item in node.items])
    if isinstance(node, DictExpr):
        return ast.Dict(
            keys=[None if item.key is None else _lower_expr(item.key) for item in node.entries],
            values=[_lower_expr(item.value) for item in node.entries],
        )
    if isinstance(node, UnaryExpr):
        return ast.UnaryOp(op=_UNARYOPS[node.op](), operand=_lower_expr(node.operand))
    if isinstance(node, BinaryExpr):
        return ast.BinOp(left=_lower_expr(node.left), op=_BINOPS[node.op](), right=_lower_expr(node.right))
    if isinstance(node, BoolExpr):
        return ast.BoolOp(op=(ast.And() if node.op == "and" else ast.Or()), values=[_lower_expr(item) for item in node.values])
    if isinstance(node, CompareExpr):
        return ast.Compare(left=_lower_expr(node.left), ops=[_CMPOPS[item]() for item in node.ops], comparators=[_lower_expr(item) for item in node.comparators])
    if isinstance(node, CallExpr):
        # JSON object order is not semantic. Sort keywords so canonical-equal
        # ProgramIR payloads also evaluate keyword values in the same order.
        return ast.Call(
            func=_lower_expr(node.function),
            args=[
                *[_lower_expr(item) for item in node.args],
                *[
                    ast.Starred(value=_lower_expr(item), ctx=ast.Load())
                    for item in node.star_args
                ],
            ],
            keywords=[
                *[
                    ast.keyword(arg=name, value=_lower_expr(node.keywords[name]))
                    for name in sorted(node.keywords)
                ],
                *[
                    ast.keyword(arg=None, value=_lower_expr(item))
                    for item in node.star_keywords
                ],
            ],
        )
    if isinstance(node, IfExpr):
        return ast.IfExp(test=_lower_expr(node.condition), body=_lower_expr(node.then), orelse=_lower_expr(node.otherwise))
    if isinstance(node, ListComprehensionExpr):
        return ast.ListComp(elt=_lower_expr(node.element), generators=_lower_comprehensions(node.clauses))
    if isinstance(node, SetComprehensionExpr):
        return ast.SetComp(elt=_lower_expr(node.element), generators=_lower_comprehensions(node.clauses))
    if isinstance(node, GeneratorComprehensionExpr):
        return ast.GeneratorExp(
            elt=_lower_expr(node.element),
            generators=_lower_comprehensions(node.clauses),
        )
    if isinstance(node, DictComprehensionExpr):
        return ast.DictComp(key=_lower_expr(node.key), value=_lower_expr(node.value), generators=_lower_comprehensions(node.clauses))
    raise TypeError(f"unsupported expression: {type(node).__name__}")


def _lower_parameters(function: FunctionDefStmt) -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=item.name) for item in function.parameters],
        vararg=None if function.vararg is None else ast.arg(arg=function.vararg),
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None if function.kwarg is None else ast.arg(arg=function.kwarg),
        defaults=[
            _lower_expr(item.default)
            for item in function.parameters
            if item.default is not None
        ],
    )


def _lower_stmt(node: Stmt) -> ast.stmt:
    if isinstance(node, AssignStmt):
        return ast.Assign(targets=[_lower_target(item) for item in node.targets], value=_lower_expr(node.value))
    if isinstance(node, AugAssignStmt):
        return ast.AugAssign(target=_lower_target(node.target), op=_BINOPS[node.op](), value=_lower_expr(node.value))
    if isinstance(node, ExprStmt):
        return ast.Expr(value=_lower_expr(node.value))
    if isinstance(node, IfStmt):
        return ast.If(test=_lower_expr(node.condition), body=[_lower_stmt(item) for item in node.body], orelse=[_lower_stmt(item) for item in node.orelse])
    if isinstance(node, ForStmt):
        return ast.For(target=_lower_target(node.target), iter=_lower_expr(node.iterable), body=[_lower_stmt(item) for item in node.body], orelse=[_lower_stmt(item) for item in node.orelse])
    if isinstance(node, WhileStmt):
        return ast.While(test=_lower_expr(node.condition), body=[_lower_stmt(item) for item in node.body], orelse=[_lower_stmt(item) for item in node.orelse])
    if isinstance(node, FunctionDefStmt):
        return ast.FunctionDef(
            name=node.name,
            args=_lower_parameters(node),
            body=[_lower_stmt(item) for item in node.body],
            decorator_list=[],
        )
    if isinstance(node, ReturnStmt):
        return ast.Return(value=None if node.value is None else _lower_expr(node.value))
    if isinstance(node, ImportStmt):
        return ast.Import(names=[ast.alias(name=item.name, asname=item.asname) for item in node.names])
    if isinstance(node, FromImportStmt):
        return ast.ImportFrom(module=node.module, names=[ast.alias(name=item.name, asname=item.asname) for item in node.names], level=0)
    if isinstance(node, BreakStmt):
        return ast.Break()
    if isinstance(node, ContinueStmt):
        return ast.Continue()
    if isinstance(node, PassStmt):
        return ast.Pass()
    if isinstance(node, RaiseStmt):
        exception = ast.Name(id=node.exception, ctx=ast.Load())
        if node.message is not None:
            exception = ast.Call(
                func=exception,
                args=[_lower_expr(node.message)],
                keywords=[],
            )
        return ast.Raise(exc=exception, cause=None)
    if isinstance(node, TryStmt):
        handlers = []
        for handler in node.handlers:
            exception_nodes = [ast.Name(id=name, ctx=ast.Load()) for name in handler.exceptions]
            exception_type: ast.expr = (
                exception_nodes[0]
                if len(exception_nodes) == 1
                else ast.Tuple(elts=exception_nodes, ctx=ast.Load())
            )
            handlers.append(ast.ExceptHandler(
                type=exception_type,
                name=handler.name,
                body=[_lower_stmt(item) for item in handler.body],
            ))
        return ast.Try(
            body=[_lower_stmt(item) for item in node.body],
            handlers=handlers,
            orelse=[_lower_stmt(item) for item in node.orelse],
            finalbody=[_lower_stmt(item) for item in node.finalbody],
        )
    raise TypeError(f"unsupported statement: {type(node).__name__}")


def _build_source_map(program: ProgramIR, module: ast.Module) -> tuple[SourceMapEntry, ...]:
    """Map verified generated-source ranges back to ProgramIR statement paths."""
    entries: list[SourceMapEntry] = []

    def visit_block(
        ir_statements: list[Stmt],
        python_statements: list[ast.stmt],
        path: tuple[Any, ...],
    ) -> None:
        for index, (ir_statement, python_statement) in enumerate(
            zip(ir_statements, python_statements, strict=False)
        ):
            statement_path = (*path, index)
            line = getattr(python_statement, "lineno", None)
            end_line = getattr(python_statement, "end_lineno", line)
            if isinstance(line, int):
                entries.append(SourceMapEntry(
                    path=_path(statement_path),
                    line=line,
                    end_line=end_line if isinstance(end_line, int) else line,
                ))

            if isinstance(ir_statement, IfStmt) and isinstance(python_statement, ast.If):
                visit_block(ir_statement.body, python_statement.body, (*statement_path, "body"))
                visit_block(ir_statement.orelse, python_statement.orelse, (*statement_path, "orelse"))
            elif isinstance(ir_statement, ForStmt) and isinstance(python_statement, ast.For):
                visit_block(ir_statement.body, python_statement.body, (*statement_path, "body"))
                visit_block(ir_statement.orelse, python_statement.orelse, (*statement_path, "orelse"))
            elif isinstance(ir_statement, WhileStmt) and isinstance(python_statement, ast.While):
                visit_block(ir_statement.body, python_statement.body, (*statement_path, "body"))
                visit_block(ir_statement.orelse, python_statement.orelse, (*statement_path, "orelse"))
            elif isinstance(ir_statement, FunctionDefStmt) and isinstance(python_statement, ast.FunctionDef):
                visit_block(ir_statement.body, python_statement.body, (*statement_path, "body"))
            elif isinstance(ir_statement, TryStmt) and isinstance(python_statement, ast.Try):
                visit_block(ir_statement.body, python_statement.body, (*statement_path, "body"))
                for handler_index, (ir_handler, python_handler) in enumerate(
                    zip(ir_statement.handlers, python_statement.handlers, strict=False)
                ):
                    visit_block(
                        ir_handler.body,
                        python_handler.body,
                        (*statement_path, "handlers", handler_index, "body"),
                    )
                visit_block(ir_statement.orelse, python_statement.orelse, (*statement_path, "orelse"))
                visit_block(ir_statement.finalbody, python_statement.finalbody, (*statement_path, "finalbody"))

    visit_block(program.body, module.body, ("body",))
    return tuple(entries)


def compile_program(payload: Any) -> CompiledProgram:
    _preflight_payload(payload)
    try:
        program = ProgramIR.model_validate(payload)
    except RecursionError as exc:
        raise ProgramCompileError(
            stage="limits",
            error_category="PROGRAM_TOO_DEEP",
            diagnostics=[CompilerDiagnostic(
                "IR_RAW_DEPTH_LIMIT",
                "program",
                "program payload exceeded the validator recursion limit",
            )],
        ) from exc
    except ValidationError as exc:
        raise ProgramCompileError(
            stage="schema",
            error_category="INVALID_PROGRAM_IR",
            diagnostics=_schema_diagnostics(exc),
        ) from exc

    canonical_json = _canonical_program_json(program)
    node_count, depth = _complexity(program)
    if len(canonical_json) > MAX_PROGRAM_CHARS:
        raise ProgramCompileError(
            stage="limits",
            error_category="PROGRAM_TOO_LARGE",
            diagnostics=[CompilerDiagnostic(
                "IR_SIZE_LIMIT",
                "program",
                f"canonical program has {len(canonical_json)} characters; maximum is {MAX_PROGRAM_CHARS}",
            )],
        )
    if node_count > MAX_IR_NODES:
        raise ProgramCompileError(
            stage="limits",
            error_category="PROGRAM_TOO_LARGE",
            diagnostics=[CompilerDiagnostic("IR_NODE_LIMIT", "program", f"program has {node_count} IR nodes; maximum is {MAX_IR_NODES}")],
        )
    if depth > MAX_IR_DEPTH:
        raise ProgramCompileError(
            stage="limits",
            error_category="PROGRAM_TOO_DEEP",
            diagnostics=[CompilerDiagnostic("IR_DEPTH_LIMIT", "program", f"program depth is {depth}; maximum is {MAX_IR_DEPTH}")],
        )

    _validate_ir_safety(program)
    _validate_statement_context(program)

    try:
        module = ast.fix_missing_locations(ast.Module(body=[_lower_stmt(item) for item in program.body], type_ignores=[]))
        source = ast.unparse(module)
        parsed = ast.parse(source, mode="exec")
        compile(parsed, "<ProgramIR>", "exec")
        verified_source = ast.unparse(parsed)
        verified = ast.parse(verified_source, mode="exec")
    except (SyntaxError, TypeError, ValueError) as exc:
        raise ProgramCompileError(
            stage="compile",
            error_category="LOWERING_FAILED",
            diagnostics=[CompilerDiagnostic("IR_COMPILE_ERROR", "program", str(exc))],
        ) from exc
    if source != verified_source or ast.dump(parsed, include_attributes=False) != ast.dump(verified, include_attributes=False):
        raise ProgramCompileError(
            stage="verify",
            error_category="AST_ROUNDTRIP_MISMATCH",
            diagnostics=[CompilerDiagnostic(
                "IR_AST_ROUNDTRIP_MISMATCH",
                "program",
                "lowered program was not stable across an AST unparse/reparse round trip",
            )],
        )
    compiled_source = source + ("\n" if source else "")
    return CompiledProgram(
        program=program,
        source=compiled_source,
        node_count=node_count,
        program_sha256=_program_sha256(canonical_json),
        source_sha256=_source_sha256(compiled_source),
        source_map=_build_source_map(program, verified),
    )


def program_tool_parameters_schema() -> dict[str, Any]:
    """Return a function-tool schema with ProgramIR definitions hoisted correctly."""
    program_schema = ProgramIR.model_json_schema()
    definitions = program_schema.pop("$defs", {})
    program_schema["description"] = (
        "A fresh version-1 structured Python program. Build statements through "
        "Stmt and expressions through Expr; raw Python source is not accepted. "
        "JSON types are strict and are never coerced. "
        f"Limits: {MAX_IR_NODES} IR nodes, depth {MAX_IR_DEPTH}, and "
        f"{MAX_PROGRAM_CHARS} canonical characters; integers are limited to "
        f"{MAX_INTEGER_BITS} bits and individual containers to "
        f"{MAX_CONTAINER_ITEMS} items. Malformed raw payloads are preflighted "
        f"at {MAX_RAW_PAYLOAD_DEPTH} container levels and "
        f"{MAX_RAW_PAYLOAD_VALUES} values before schema validation."
    )
    guidance = {
        "AugTarget": "Augmented-assignment target: name, attribute, or subscript only.",
        "Expr": "Expression node selected by its kind discriminator.",
        "Stmt": "Statement node selected by its kind discriminator; unreachable statements are rejected.",
        "StoreTarget": "Assignment target selected by its kind discriminator.",
        "UnpackTarget": "Nested unpacking target; starred_target is allowed once per tuple/list level.",
    }
    for definition_name, description in guidance.items():
        if definition_name in definitions:
            definitions[definition_name]["description"] = description
    schema = {
        "type": "object",
        "description": "Compile and execute one ephemeral ProgramIR program in The Duck sandbox.",
        "additionalProperties": False,
        "properties": {"program": program_schema},
        "required": ["program"],
        "$defs": definitions,
    }
    # Pydantic-generated titles repeat class/field names but add no validation
    # semantics. Removing them saves substantial context on small model profiles.
    stack: list[Any] = [schema]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            current.pop("title", None)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return schema


def program_to_source(payload: Any) -> str:
    """Best-effort public helper for trace viewers; never executes the program."""
    return compile_program(payload).source
