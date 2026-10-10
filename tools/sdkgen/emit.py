"""Deterministic reference emitters; no remote templates or post-processors."""

from __future__ import annotations

import json
import keyword
from pathlib import Path
from typing import Any

from .compiler import ContractError, canonical, digest, nullable

HERE = Path(__file__).parent


def pascal(value: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in value.split("_"))


def camel(value: str) -> str:
    result = pascal(value)
    return result[:1].lower() + result[1:]


def without_null(schema: dict[str, Any]) -> dict[str, Any]:
    return (
        next(x for x in schema["anyOf"] if x.get("type") != "null") if "anyOf" in schema else schema
    )


def typ(schema: dict[str, Any], lang: str) -> str:
    if "$ref" in schema:
        return str(schema["$ref"]).rsplit("/", 1)[1]
    if "anyOf" in schema:
        base = typ(without_null(schema), lang)
        return {"python": f"{base} | None", "typescript": f"({base} | null)", "go": f"*{base}"}[
            lang
        ]
    if "enum" in schema and lang in {"python", "typescript"}:
        if lang == "python":
            return "Literal[" + ", ".join(repr(v) for v in schema["enum"]) + "]"
        return "(" + " | ".join(json.dumps(v, ensure_ascii=False) for v in schema["enum"]) + ")"
    t = str(schema.get("type", ""))
    if t == "array":
        base = typ(schema["items"], lang)
        return {"python": f"list[{base}]", "typescript": f"Array<{base}>", "go": f"[]{base}"}[lang]
    if t == "object":
        extra = schema.get("additionalProperties", {})
        base = typ(extra if isinstance(extra, dict) else {}, lang)
        return {
            "python": f"dict[str, {base}]",
            "typescript": f"Record<string, {base}>",
            "go": f"map[string]{base}",
        }[lang]
    mappings = {
        "python": {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "null": "None",
        },
        "typescript": {
            "string": "string",
            "integer": "number",
            "number": "number",
            "boolean": "boolean",
            "null": "null",
        },
        "go": {
            "string": "string",
            "integer": "int64",
            "number": "float64",
            "boolean": "bool",
            "null": "json.RawMessage",
        },
    }
    return mappings[lang].get(
        t,
        {"python": "JsonValue", "typescript": "JsonValue", "go": "json.RawMessage"}[lang],  # type: ignore[arg-type]
    )


def operations(document: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for path, methods in document["paths"].items():
        for method, operation in methods.items():
            response = operation["responses"]["200"]["content"]["application/json"]["schema"][
                "$ref"
            ].rsplit("/", 1)[1]
            request = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
                .get("$ref")
            )
            result.append(
                {
                    "id": operation["operationId"],
                    "method": method.upper(),
                    "path": path,
                    "params": [p["name"] for p in operation["parameters"]],
                    "response": response,
                    "request": request.rsplit("/", 1)[1] if request else None,
                }
            )
    return sorted(result, key=lambda o: o["id"])


def guard_names(document: dict[str, Any]) -> None:
    reserved = set(keyword.kwlist) | {
        "constructor",
        "request",
        "transport",
        "options",
        "schemas",
        "new_client",
    }
    reserved_models = {
        "Client",
        "Transport",
        "Optional",
        "APIError",
        "ProtocolError",
        "JsonValue",
        "Options",
        "Schema",
    }
    for op in operations(document):
        if op["id"] in reserved or camel(op["id"]) in reserved:
            raise ContractError(f"Reserved client operation name: {op['id']}")
        if any(
            p in reserved | {"ctx", "body", "response", "path", "err", "out"} for p in op["params"]
        ):
            raise ContractError("Path parameter collides with a client runtime symbol")
    for model_name, schema in document["components"]["schemas"].items():
        if model_name in reserved_models:
            raise ContractError(f"Reserved model name {model_name}")
        fields = set()
        for field in schema["properties"]:
            converted = pascal(field)
            if converted in fields or converted in {
                "AdditionalProperties",
                "MarshalJSON",
                "UnmarshalJSON",
            }:
                raise ContractError(f"Go field name collision in {model_name}: {field}")
            fields.add(converted)


def emit_python(models: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, bytes]:
    lines = [
        "# Generated by tools.sdkgen; DO NOT EDIT.",
        "from __future__ import annotations",
        "",
        "from typing import Literal, NotRequired, TypeAlias, TypedDict",
        "",
        'JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"',
        "",
    ]
    # Functional TypedDict form preserves any valid source field name, including language keywords.
    for model, schema in models.items():
        lines += [f"{model} = TypedDict({model!r}, {{"]
        for field, fs in schema["properties"].items():
            ts = typ(fs, "python")
            if field not in schema["required"]:
                ts = f"NotRequired[{ts}]"
            lines.append(f"    {field!r}: {ts!r},")
        lines += ["})", ""]
    client = [
        "# Generated by tools.sdkgen; DO NOT EDIT.",
        "from __future__ import annotations",
        "",
        "from typing import Any, cast",
        "from .runtime import Transport, path_segment",
        "from .models import " + ", ".join(models),
        "",
        "",
        "class Client:",
        '    def __init__(self, base_url: str = "http://localhost:8787", *, transport: Transport | None = None, **options: Any) -> None:',
        "        self._transport = transport or Transport(base_url, **options)",
        "",
    ]
    for op in ops:
        params = [f"{p}: str" for p in op["params"]]
        if op["request"]:
            params.append(f"body: {op['request']}")
        client.append(
            f"    def {op['id']}(self{', ' if params else ''}{', '.join(params)}) -> {op['response']}:"
        )
        client.append(f"        path = {op['path']!r}")
        for p in op["params"]:
            client.append(f"        path = path.replace({('{' + p + '}')!r}, path_segment({p}))")
        body = "body" if op["request"] else "None"
        client += [
            f"        return cast({op['response']}, self._transport.request({op['method']!r}, path, {body}, {op['request']!r}, {op['response']!r}))",
            "",
        ]
    return {
        "models.py": ("\n".join(lines).rstrip() + "\n").encode(),
        "client.py": ("\n".join(client).rstrip() + "\n").encode(),
        "__init__.py": b'from .client import Client\nfrom .runtime import APIError, ProtocolError, Transport\n\n__all__ = ["Client", "APIError", "ProtocolError", "Transport"]\n',
        "runtime.py": (HERE / "runtime/python.py").read_bytes(),
        "schemas.json": canonical(models),
    }


def emit_typescript(models: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, bytes]:
    lines = [
        "// Generated by tools.sdkgen; DO NOT EDIT.",
        'import type { JsonValue } from "./runtime.js";',
        "",
    ]
    for model, schema in models.items():
        lines.append(f"export interface {model} {{")
        for field, fs in schema["properties"].items():
            marker = "" if field in schema["required"] else "?"
            lines.append(f"  {json.dumps(field)}{marker}: {typ(fs, 'typescript')};")
        lines += [
            "  [key: string]: unknown; // Preserve future wire fields without renaming.",
            "}",
            "",
        ]
    client = [
        "// Generated by tools.sdkgen; DO NOT EDIT.",
        'import { Transport, pathSegment } from "./runtime.js";',
        'import type { RequestOptions, TransportOptions } from "./runtime.js";',
        'import { schemas } from "./schemas.js";',
        "import type { " + ", ".join(models) + ' } from "./models.js";',
        "",
        "export class Client {",
        "  private readonly transport: Transport;",
        '  constructor(baseUrl = "http://localhost:8787", options: TransportOptions = {}) {',
        "    this.transport = new Transport(baseUrl, options);",
        "  }",
    ]
    for op in ops:
        params = [f"{camel(p)}: string" for p in op["params"]]
        if op["request"]:
            params.append(f"body: {op['request']}")
        params.append("options: RequestOptions = {}")
        client.append(
            f"  async {camel(op['id'])}({', '.join(params)}): Promise<{op['response']}> {{"
        )
        client.append(f"    let path = {json.dumps(op['path'])};")
        for p in op["params"]:
            client.append(
                f"    path = path.replace({json.dumps('{' + p + '}')}, pathSegment({camel(p)}));"
            )
        body = "body" if op["request"] else "undefined"
        client += [
            f"    return this.transport.request<{op['response']}>({json.dumps(op['method'])}, path, {body}, {json.dumps(op['request'])}, {json.dumps(op['response'])}, schemas, options);",
            "  }",
        ]
    client.append("}")
    package = {
        "name": "@headroom/experimental-generated-wire",
        "version": "0.0.0",
        "private": True,
        "type": "module",
    }
    tsconfig = {
        "compilerOptions": {
            "target": "ES2022",
            "module": "NodeNext",
            "moduleResolution": "NodeNext",
            "strict": True,
            "lib": ["ES2023", "DOM", "DOM.Iterable"],
            "declaration": True,
            "outDir": "dist",
            "skipLibCheck": False,
        },
        "include": ["*.ts"],
    }
    return {
        "models.ts": ("\n".join(lines).rstrip() + "\n").encode(),
        "client.ts": ("\n".join(client).rstrip() + "\n").encode(),
        "runtime.ts": (HERE / "runtime/typescript.ts").read_bytes(),
        "schemas.ts": (
            '// Generated by tools.sdkgen; DO NOT EDIT.\nimport type { Schema } from "./runtime.js";\nexport const schemas: Record<string, Schema> = '
            + canonical(models).decode().strip()
            + ";\n"
        ).encode(),
        "index.ts": b'export { Client } from "./client.js";\nexport * from "./models.js";\nexport { APIError, ProtocolError } from "./runtime.js";\n',
        "package.json": canonical(package),
        "tsconfig.json": canonical(tsconfig),
    }


def emit_go(models: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, bytes]:
    lines = [
        "// Code generated by tools.sdkgen. DO NOT EDIT.",
        "package headroomwire",
        "",
        'import ("encoding/json"; "fmt")',
        "",
    ]
    for model, schema in models.items():
        lines.append(f"type {model} struct {{")
        for field, fs in schema["properties"].items():
            needed = field in schema["required"]
            field_type = typ(fs, "go")
            if not needed:
                field_type = (
                    f"Optional[{typ(without_null(fs), 'go')}]" if nullable(fs) else "*" + field_type
                )
            lines.append(f'    {pascal(field)} {field_type} `json:"{field}"`')
        lines += [
            '    AdditionalProperties map[string]json.RawMessage `json:"-"`',
            "}",
            f"func (x {model}) MarshalJSON() ([]byte, error) {{",
            "    fields := make(map[string]json.RawMessage, len(x.AdditionalProperties))",
            "    for key, value := range x.AdditionalProperties { fields[key] = value }",
        ]
        for field, fs in schema["properties"].items():
            needed = field in schema["required"]
            var = "x." + pascal(field)
            lines.append(
                f'    if _, exists := fields[{json.dumps(field)}]; exists {{ return nil, fmt.Errorf("additional property collides with declared field {field}") }}'
            )
            condition = "true" if needed else (f"{var}.Set" if nullable(fs) else f"{var} != nil")
            lines += [
                f"    if {condition} {{",
                f"        raw, err := json.Marshal({var}); if err != nil {{ return nil, err }}",
                f"        fields[{json.dumps(field)}] = raw",
                "    }",
            ]
        lines += [
            "    raw, err := json.Marshal(fields); if err != nil { return nil, err }",
            f'    if err := validateModel(raw, "{model}"); err != nil {{ return nil, err }}',
            "    return raw, nil",
            "}",
            f"func (x *{model}) UnmarshalJSON(raw []byte) error {{",
            f'    if err := validateModel(raw, "{model}"); err != nil {{ return err }}',
            f"    type plain {model}",
            "    var value plain",
            "    if err := json.Unmarshal(raw, &value); err != nil { return err }",
            "    var extra map[string]json.RawMessage",
            "    if err := json.Unmarshal(raw, &extra); err != nil { return err }",
        ]
        for field in schema["properties"]:
            lines.append(f"    delete(extra, {json.dumps(field)})")
        lines += [
            f"    *x = {model}(value); x.AdditionalProperties = extra",
            "    return nil",
            "}",
            "",
        ]
    imports = '"context"'
    if any(op["params"] for op in ops):
        imports += '; "strings"'
    client = [
        "// Code generated by tools.sdkgen. DO NOT EDIT.",
        "package headroomwire",
        "",
        f"import ({imports})",
        "",
        "type Client struct { transport *Transport }",
        "func NewClient(baseURL string, options *Options) (*Client, error) {",
        "    transport, err := newTransport(baseURL, options); if err != nil { return nil, err }",
        "    return &Client{transport: transport}, nil",
        "}",
    ]
    for op in ops:
        args = ["ctx context.Context"] + [f"{camel(p)} string" for p in op["params"]]
        if op["request"]:
            args.append(f"body {op['request']}")
        client += [
            f"func (c *Client) {pascal(op['id'])}({', '.join(args)}) (*{op['response']}, error) {{",
            f"    path := {json.dumps(op['path'])}",
        ]
        for p in op["params"]:
            var = "encoded" + pascal(p)
            client += [
                f"    {var}, err := pathSegment({camel(p)}); if err != nil {{ return nil, err }}",
                f"    path = strings.ReplaceAll(path, {json.dumps('{' + p + '}')}, {var})",
            ]
        client += [
            f"    var out {op['response']}",
            f"    if err := c.transport.request(ctx, {json.dumps(op['method'])}, path, {'body' if op['request'] else 'nil'}, &out, {json.dumps(op['request'] or '')}); err != nil {{ return nil, err }}",
            "    return &out, nil",
            "}",
        ]
    return {
        "models.go": ("\n".join(lines).rstrip() + "\n").encode(),
        "client.go": ("\n".join(client).rstrip() + "\n").encode(),
        "runtime.go": (HERE / "runtime/go.go").read_bytes(),
        "schemas.json": canonical(models),
        "go.mod": b"module github.com/headroomlabs-ai/headroom/sdk/generated-pilot/go\n\ngo 1.23.0\n",
    }


def render_with_emitters(document: dict[str, Any], emitters: Any) -> dict[str, bytes]:
    guard_names(document)
    output = {
        "openapi.json": canonical(document),
        ".ruff.toml": b'# Generated bytes are checked by sdkgen, not rewritten by formatters.\nexclude = ["**/*.py"]\n',
    }
    for emitter in emitters:
        for path, content in emitter(document).items():
            if path in output or path == "manifest.json":
                raise ContractError(f"Emitter output path collision: {path}")
            output[path] = content
    inputs = {
        p.relative_to(HERE).as_posix(): digest(p.read_bytes())
        for p in sorted(HERE.rglob("*"))
        if p.is_file()
        and p.suffix in {".py", ".ts", ".go", ".rs", ".cs", ".lock"}
        and "__pycache__" not in p.parts
    }
    manifest = {
        "format_version": 1,
        "generator_version": "0.1.0",
        "canonical_json_profile": "headroom-json-v1",
        "schema_sha256": digest(output["openapi.json"]),
        "generator_inputs_sha256": digest(canonical(inputs)),
        "files": {p: digest(data) for p, data in sorted(output.items())},
    }
    output["manifest.json"] = canonical(manifest)
    return dict(sorted(output.items()))


def render(document: dict[str, Any]) -> dict[str, bytes]:
    from .emitters import default_emitters

    return render_with_emitters(document, default_emitters())
