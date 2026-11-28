"""Static source harvesting. Never imports or executes the application.

Supported v1 profile: literal FastAPI method decorators; sdk_operation markers;
TypedDicts, scalar types, nullable unions, Literal scalars, list[T], dict[str, T],
and explicit JsonValue. Unsupported syntax is an error, never inferred as Any.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
OP_IDENT = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class ContractError(ValueError):
    """A source or backend capability needs an explicit maintainer decision."""


def canonical(value: Any) -> bytes:
    # Project canonical JSON v1, NOT a claim of RFC 8785/JCS compliance.
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def literal(node: ast.AST, label: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ContractError(f"{label}: expected a literal, not executable syntax") from exc


def keywords(call: ast.Call) -> dict[str, ast.AST]:
    if any(k.arg is None for k in call.keywords):
        raise ContractError("**kwargs are not supported on exported operations")
    return {str(k.arg): k.value for k in call.keywords}


def nullable(schema: dict[str, Any]) -> bool:
    return schema.get("type") == "null" or any(nullable(item) for item in schema.get("anyOf", []))


class Compiler:
    def __init__(self, sources: dict[str, str]) -> None:
        self.trees: dict[str, ast.Module] = {}
        self.classes: dict[str, list[ast.ClassDef]] = {}
        self.schemas: dict[str, Any] = {}
        self.building: set[str] = set()
        for path, source in sorted(sources.items()):
            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError as exc:
                raise ContractError(f"{path}:{exc.lineno}: {exc.msg}") from exc
            self.trees[path] = tree
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.classes.setdefault(node.name, []).append(node)

    def type_schema(self, node: ast.AST) -> dict[str, Any]:
        if isinstance(node, ast.Constant) and node.value is None:
            return {"type": "null"}
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return self.type_schema(ast.parse(node.value, mode="eval").body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            members: list[dict[str, Any]] = []
            for part in (self.type_schema(node.left), self.type_schema(node.right)):
                members.extend(part.get("anyOf", [part]))
            # v1 permits only nullable unions, not ambiguous general unions.
            nonnull = [m for m in members if m.get("type") != "null"]
            if len(nonnull) != 1 or len(members) != 2:
                raise ContractError("Only T | None unions are in portable profile v1")
            return {"anyOf": [nonnull[0], {"type": "null"}]}
        if isinstance(node, ast.Attribute):
            raise ContractError("Qualified types require an explicit source adapter")
        if isinstance(node, ast.Name):
            n = name(node)
            scalar = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "None": "null",
            }
            if n in scalar:
                return {"type": scalar[n]}
            if n == "JsonValue":
                return {"x-headroom-opaque": "intentional-json-value"}
            if n == "Any":
                raise ContractError("Implicit Any is forbidden; declare an intentional JsonValue")
            self.model(n)
            return {"$ref": f"#/components/schemas/{n}"}
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name):
                raise ContractError("Qualified generic types require an explicit source adapter")
            n = name(node.value)
            parts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            if n in {"Required", "NotRequired"} and len(parts) == 1:
                return self.type_schema(parts[0])
            if n in {"Optional"} and len(parts) == 1:
                return {"anyOf": [self.type_schema(parts[0]), {"type": "null"}]}
            if n in {"list", "List"} and len(parts) == 1:
                return {"type": "array", "items": self.type_schema(parts[0])}
            if n in {"dict", "Dict"} and len(parts) == 2 and name(parts[0]) == "str":
                return {"type": "object", "additionalProperties": self.type_schema(parts[1])}
            if n == "Literal":
                values = [literal(v, "Literal") for v in parts]
                if not values or any(type(v) not in (str, bool, int) for v in values):
                    raise ContractError("Literal supports strings, booleans, or integers")
                if len({type(v) for v in values}) != 1:
                    raise ContractError("Mixed Literal types are not in portable profile v1")
                if any(type(v) is int and abs(v) > 9007199254740991 for v in values):
                    raise ContractError(
                        "Integer Literal exceeds the portable TypeScript safe range"
                    )
                typ = {str: "string", bool: "boolean", int: "integer"}[type(values[0])]
                return {"type": typ, "enum": sorted(set(values))}
        raise ContractError(f"Unsupported type syntax: {ast.dump(node, include_attributes=False)}")

    def model(self, model_name: str) -> None:
        if model_name in self.schemas:
            return
        if model_name in self.building:
            raise ContractError(f"Recursive model {model_name} is not in portable profile v1")
        candidates = self.classes.get(model_name, [])
        if len(candidates) != 1:
            raise ContractError(f"Model {model_name}: expected one unambiguous source declaration")
        cls = candidates[0]
        if (
            len(cls.bases) != 1
            or not isinstance(cls.bases[0], ast.Name)
            or name(cls.bases[0]) != "TypedDict"
        ):
            raise ContractError(f"{model_name}: only direct TypedDict declarations are supported")
        if cls.decorator_list:
            raise ContractError(f"{model_name}: decorated models need a supported source adapter")
        settings = {k.arg: literal(k.value, model_name) for k in cls.keywords}
        if set(settings) - {"total"} or type(settings.get("total", True)) is not bool:
            raise ContractError(f"{model_name}: unsupported TypedDict settings")
        self.building.add(model_name)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for field in cls.body:
            if (
                isinstance(field, ast.Expr)
                and isinstance(field.value, ast.Constant)
                and isinstance(field.value.value, str)
            ):
                continue
            if isinstance(field, ast.Pass):
                continue
            if not isinstance(field, ast.AnnAssign) or not isinstance(field.target, ast.Name):
                raise ContractError(f"{model_name}: expected annotated fields only")
            if field.value is not None:
                raise ContractError(f"{model_name}: TypedDict field defaults are not wire defaults")
            key = field.target.id
            if key in properties:
                raise ContractError(f"Duplicate field {model_name}.{key}")
            needed = settings.get("total", True)
            annotation = field.annotation
            if isinstance(annotation, ast.Subscript):
                wrapper = name(annotation.value)
                if wrapper in {"Required", "NotRequired"}:
                    needed = wrapper == "Required"
            properties[key] = self.type_schema(annotation)
            if needed:
                required.append(key)
        self.schemas[model_name] = {
            "type": "object",
            "properties": dict(sorted(properties.items())),
            "required": sorted(required),
            "additionalProperties": True,
        }
        self.building.remove(model_name)

    def inventory(self) -> dict[str, Any]:
        routes: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        for source, tree in self.trees.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    exported = any(
                        isinstance(d, ast.Call) and name(d.func) == "sdk_operation"
                        for d in node.decorator_list
                    )
                    for d in node.decorator_list:
                        if not isinstance(d, ast.Call) or not isinstance(d.func, ast.Attribute):
                            continue
                        kind = d.func.attr
                        if kind not in METHODS | {"api_route", "websocket"}:
                            continue
                        path_arg = (
                            d.args[0]
                            if d.args
                            else next((k.value for k in d.keywords if k.arg == "path"), None)
                        )
                        if not isinstance(path_arg, ast.Constant) or not isinstance(
                            path_arg.value, str
                        ):
                            unresolved.append(
                                {
                                    "source": source,
                                    "line": node.lineno,
                                    "reason": "dynamic route path",
                                    "handler": node.name,
                                }
                            )
                            continue
                        if kind == "api_route":
                            methods_node = next(
                                (k.value for k in d.keywords if k.arg == "methods"), None
                            )
                            try:
                                methods = (
                                    literal(methods_node, "methods") if methods_node else ["GET"]
                                )
                                if not isinstance(methods, (list, tuple)) or any(
                                    not isinstance(m, str) for m in methods
                                ):
                                    raise ContractError("nonliteral methods")
                            except ContractError:
                                unresolved.append(
                                    {
                                        "source": source,
                                        "line": node.lineno,
                                        "reason": "dynamic methods",
                                        "handler": node.name,
                                    }
                                )
                                continue
                        else:
                            methods = [kind.upper()]
                        for method in methods:
                            routes.append(
                                {
                                    "method": method.upper(),
                                    "path": path_arg.value,
                                    "handler": node.name,
                                    "source": source,
                                    "line": node.lineno,
                                    "exported": exported,
                                }
                            )
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in {
                        "add_api_route",
                        "add_route",
                        "include_router",
                        "mount",
                        "add_websocket_route",
                    }:
                        unresolved.append(
                            {
                                "source": source,
                                "line": node.lineno,
                                "reason": f"registration requires runtime reconciliation: {node.func.attr}",
                            }
                        )
        return {
            "scope": "static candidates; not a complete runtime route graph",
            "routes": sorted(
                routes, key=lambda r: (r["path"], r["method"], r["source"], r["line"])
            ),
            "unresolved": sorted(unresolved, key=lambda r: (r["source"], r["line"], r["reason"])),
        }

    def compile(self) -> dict[str, Any]:
        paths: dict[str, Any] = {}
        operation_ids: set[str] = set()
        for source, tree in self.trees.items():
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                markers = [
                    d
                    for d in fn.decorator_list
                    if isinstance(d, ast.Call) and name(d.func) == "sdk_operation"
                ]
                if not markers:
                    continue
                if len(markers) != 1 or markers[0].args:
                    raise ContractError(
                        f"{source}:{fn.lineno}: expected one keyword-only sdk_operation"
                    )
                meta = keywords(markers[0])
                if set(meta) - {"operation_id", "request", "response", "access", "errors"}:
                    raise ContractError(f"{fn.name}: unknown contract metadata")
                if not {"operation_id", "response", "access"} <= set(meta):
                    raise ContractError(
                        f"{fn.name}: operation_id, response, and access are required"
                    )
                op_id = literal(meta["operation_id"], "operation_id")
                if not isinstance(op_id, str) or not OP_IDENT.fullmatch(op_id):
                    raise ContractError("operation_id must be stable lowercase snake_case")
                if op_id in operation_ids:
                    raise ContractError(f"Duplicate operation_id {op_id}")
                operation_ids.add(op_id)
                access = literal(meta["access"], "access")
                if access not in {"loopback", "loopback-same-origin", "public", "authenticated"}:
                    raise ContractError(f"{op_id}: unknown access policy")
                if access == "authenticated":
                    raise ContractError(
                        "Authenticated endpoints require an explicit security-scheme adapter"
                    )
                routes = [
                    d
                    for d in fn.decorator_list
                    if isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Attribute)
                    and d.func.attr in METHODS
                ]
                if len(routes) != 1:
                    raise ContractError(f"{op_id}: exactly one literal HTTP route is required")
                route = routes[0]
                route_kw = keywords(route)
                path_node = route.args[0] if route.args else route_kw.get("path")
                if path_node is None:
                    raise ContractError(f"{op_id}: missing route path")
                path = literal(path_node, "route path")
                if (
                    not isinstance(path, str)
                    or not path.startswith("/")
                    or any(c in path for c in ("?", "#", "\r", "\n"))
                ):
                    raise ContractError(f"{op_id}: invalid route path")
                method = route.func.attr  # type: ignore[attr-defined]
                if method not in {"get", "post", "put", "patch", "delete"}:
                    raise ContractError(
                        f"{op_id}: {method} is not supported by the JSON client profile"
                    )
                if (
                    "status_code" in route_kw
                    and literal(route_kw["status_code"], "status_code") != 200
                ):
                    raise ContractError("v1 supports a 200 JSON success response only")
                if "response_class" in route_kw:
                    raise ContractError(
                        "Custom/streaming response classes require a transport adapter"
                    )
                if "response_model" in route_kw and not (
                    isinstance(route_kw["response_model"], ast.Constant)
                    and route_kw["response_model"].value is None
                ):
                    raise ContractError(
                        "Explicit FastAPI response_model needs reconciliation with sdk_operation"
                    )
                placeholders = re.findall(r"\{([^{}]+)\}", path)
                if any(not IDENT.fullmatch(p) for p in placeholders) or len(placeholders) != len(
                    set(placeholders)
                ):
                    raise ContractError(f"{op_id}: unsupported/duplicate path converter")
                if "{" in re.sub(r"\{[^{}]+\}", "", path) or "}" in re.sub(r"\{[^{}]+\}", "", path):
                    raise ContractError("Malformed path template")
                args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
                argmap = {a.arg: a for a in args}
                parameters = []
                for param in sorted(placeholders):
                    if param not in argmap or argmap[param].annotation is None:
                        raise ContractError(f"{op_id}: untyped path parameter {param}")
                    ps = self.type_schema(argmap[param].annotation)  # type: ignore[arg-type]
                    if ps != {"type": "string"}:
                        raise ContractError("v1 path parameters must be plain strings")
                    parameters.append({"name": param, "in": "path", "required": True, "schema": ps})
                for arg in args:
                    if arg.arg in placeholders or arg.arg in {"self", "cls"}:
                        continue
                    if arg.annotation is not None and name(arg.annotation) == "Request":
                        continue
                    raise ContractError(
                        f"{op_id}: non-path parameter {arg.arg} needs a query/header/body adapter"
                    )
                if fn.args.vararg or fn.args.kwarg:
                    raise ContractError(f"{op_id}: variadic handlers are not supported")
                response = self.type_schema(meta["response"])
                if "$ref" not in response:
                    raise ContractError("v1 operation responses must be named object models")
                if fn.returns is not None and self.type_schema(fn.returns) != response:
                    raise ContractError(
                        f"{op_id}: handler return annotation conflicts with declared response"
                    )
                operation: dict[str, Any] = {
                    "operationId": op_id,
                    "parameters": parameters,
                    "x-headroom-access": access,
                    "x-headroom-retry": "never-by-default",
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "Successful JSON response",
                            "content": {"application/json": {"schema": response}},
                        },
                        "default": {
                            "description": "Undeclared HTTP error; preserve status, headers, and raw body"
                        },
                    },
                }
                if "request" in meta:
                    if method not in {"post", "put", "patch", "delete"}:
                        raise ContractError("v1 forbids JSON bodies on GET")
                    request = self.type_schema(meta["request"])
                    if "$ref" not in request:
                        raise ContractError("v1 request bodies must be named object models")
                    operation["requestBody"] = {
                        "required": True,
                        "content": {"application/json": {"schema": request}},
                    }
                if "errors" in meta:
                    errors = meta["errors"]
                    if not isinstance(errors, ast.Dict):
                        raise ContractError("errors must be a literal status-to-model mapping")
                    for k, value in zip(errors.keys, errors.values):
                        status = literal(k, "error status") if k is not None else None
                        if (
                            type(status) is not int
                            or not 400 <= status <= 599
                            or str(status) in operation["responses"]
                        ):
                            raise ContractError("Invalid or duplicate HTTP error status")
                        operation["responses"][str(status)] = {
                            "description": "Declared HTTP error",
                            "content": {"application/json": {"schema": self.type_schema(value)}},
                        }
                if method in paths.setdefault(path, {}):
                    raise ContractError(f"Duplicate exported route {method.upper()} {path}")
                paths[path][method] = operation
        if not operation_ids:
            raise ContractError("No annotated operations found; refusing an accidentally empty SDK")
        # Name collisions after language casing must never overwrite generated symbols.
        symbols = list(self.schemas) + list(operation_ids)
        folded: set[str] = set()
        for symbol in symbols:
            normalized = symbol.replace("_", "").lower()
            if normalized in folded:
                raise ContractError(f"Cross-language symbol collision: {symbol}")
            folded.add(normalized)
        return {
            "openapi": "3.1.1",
            "info": {"title": "Headroom HTTP contract pilot", "version": "0.1.0"},
            "x-headroom-profile": "json-http-v1-pilot",
            "paths": dict(sorted(paths.items())),
            "components": {"schemas": dict(sorted(self.schemas.items()))},
        }


def load_sources(root: Path, paths: list[str]) -> dict[str, str]:
    root = root.resolve()
    result = {}
    for relative in sorted(paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ContractError(f"Missing or unsafe source path: {relative}")
        result[relative] = path.read_text(encoding="utf-8")
    return result
