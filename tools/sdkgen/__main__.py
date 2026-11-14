"""Run with python -m tools.sdkgen. Standard library only."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath

from .compiler import Compiler, ContractError, canonical, load_sources
from .emit import render


def validate_output_paths(generated: dict[str, bytes]) -> None:
    """Reject generated names that could escape or alias the output tree."""
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    for relative, content in generated.items():
        path = PurePosixPath(relative)
        unsafe_windows = any(
            ":" in part or part.endswith((".", " ")) or part.split(".", 1)[0].upper() in reserved
            for part in path.parts
        )
        if (
            not relative
            or "\\" in relative
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or unsafe_windows
        ):
            raise ContractError(f"Unsafe generated output path: {relative}")
        if not isinstance(content, bytes):
            raise ContractError(f"Generated output must be bytes: {relative}")


def compare(before: dict, after: dict) -> list[dict[str, str]]:
    """Conservative review gate; deliberately not a JSON-Schema inclusion solver."""

    annotations = {"description", "summary", "examples", "title"}
    schema_maps = {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
        "dependencies",
    }
    schema_children = {
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
    }
    component_maps = {
        "schemas",
        "responses",
        "parameters",
        "examples",
        "requestBodies",
        "headers",
        "securitySchemes",
        "links",
        "callbacks",
        "pathItems",
    }
    openapi_maps = {
        "paths",
        "webhooks",
        "responses",
        "headers",
        "parameters",
        "content",
        "encoding",
        "variables",
        "mapping",
        "scopes",
        "links",
        "callbacks",
    }
    openapi_children = {
        "info",
        "requestBody",
        "get",
        "put",
        "post",
        "delete",
        "options",
        "head",
        "patch",
        "trace",
        "servers",
        "tags",
        "flows",
        "implicit",
        "password",
        "clientCredentials",
        "authorizationCode",
    }

    def children(value: object, schema: bool, node_kind: str = "object") -> object:
        if isinstance(value, list):
            return [clean(item, schema, node_kind) for item in value]
        return clean(value, schema, node_kind)

    def namespace(value: object, schema: bool, node_kind: str = "object") -> object:
        if isinstance(value, dict):
            # Names are identifiers, even when spelled "title" or "examples".
            return {name: clean(item, schema, node_kind) for name, item in value.items()}
        return children(value, schema, node_kind)

    def clean(value: object, schema: bool, node_kind: str = "object") -> object:
        if not isinstance(value, dict):
            return deepcopy(value)
        if not schema and node_kind == "callback" and "$ref" not in value:
            # Callback expression keys name Path Items; they are not annotations.
            return {
                name: deepcopy(item)
                if name == "$ref" or name.startswith("x-")
                else clean(item, False)
                for name, item in value.items()
            }
        result = {}
        for key, item in value.items():
            if key in annotations:
                continue
            if not schema and node_kind == "link":
                # Link parameters and requestBody accept literal JSON, not Parameter
                # or Request Body Objects. Only the optional server is structural.
                result[key] = children(item, False) if key == "server" else deepcopy(item)
                continue
            if schema and key in schema_maps:
                result[key] = namespace(item, True)
            elif schema and key in schema_children:
                result[key] = children(item, True)
            elif not schema and key == "components" and isinstance(item, dict):
                result[key] = {
                    kind: namespace(
                        entries,
                        kind == "schemas",
                        {"links": "link", "callbacks": "callback"}.get(kind, "object"),
                    )
                    if kind in component_maps
                    else deepcopy(entries)
                    for kind, entries in item.items()
                }
            elif not schema and key == "schema":
                result[key] = children(item, True)
            elif not schema and key in openapi_maps:
                result[key] = namespace(
                    item, False, {"links": "link", "callbacks": "callback"}.get(key, "object")
                )
            elif not schema and key in openapi_children:
                result[key] = children(item, False)
            else:
                # Defaults, constants, enums, dependentRequired, security requirements,
                # and unknown extension/vocabulary payloads are data, not annotations.
                result[key] = deepcopy(item)
        return result

    schema_document = not any(
        key in document for document in (before, after) for key in ("openapi", "paths")
    )

    findings = []
    remaining_before = deepcopy(before)
    for path, methods in before.get("paths", {}).items():
        for method, old in methods.items():
            new = after.get("paths", {}).get(path, {}).get(method)
            if new is None:
                remaining_methods = remaining_before["paths"].get(path, {})
                remaining_methods.pop(method, None)
                if not remaining_methods:
                    remaining_before["paths"].pop(path, None)
                findings.append(
                    {"severity": "breaking", "change": f"removed operation {method.upper()} {path}"}
                )
            elif old.get("operationId") != new.get("operationId"):
                remaining_before["paths"][path][method]["operationId"] = new.get("operationId")
                findings.append(
                    {"severity": "breaking", "change": f"renamed operation {method.upper()} {path}"}
                )
    if clean(remaining_before, schema_document) != clean(after, schema_document):
        findings.append(
            {
                "severity": "review",
                "change": "Contract changed; request/response variance and target compatibility require review",
            }
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "check", "inventory"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, default=Path.cwd())
        p.add_argument(
            "--fixture",
            action="store_true",
            help="Use explicitly labeled source fixtures, not the live repository",
        )
        if command == "inventory":
            p.add_argument("--scan", default="headroom", help="Directory of .py files to inventory")
            p.add_argument(
                "--strict",
                action="store_true",
                help="Fail on any nonexported candidate or unresolved registration",
            )
        else:
            p.add_argument("--out", type=Path)
    diff = sub.add_parser("diff")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "diff":
            findings = compare(
                json.loads(args.before.read_text()), json.loads(args.after.read_text())
            )
            print(canonical(findings).decode(), end="")
            return 2 if findings else 0
        root = args.root.resolve()
        config = json.loads((root / "sdk/codegen/config.json").read_text())
        if args.command == "inventory" and not args.fixture:
            scan = (root / args.scan).resolve()
            if not scan.is_relative_to(root) or not scan.is_dir():
                raise ContractError(
                    "Inventory scan must name an existing directory inside the repository"
                )
            paths = [
                p.relative_to(root).as_posix() for p in scan.rglob("*.py") if not p.is_symlink()
            ]
        else:
            paths = config["fixture_sources" if args.fixture else "sources"]
        compiler = Compiler(load_sources(root, paths))
        if args.command == "inventory":
            report = compiler.inventory()
            print(canonical(report).decode(), end="")
            return (
                2
                if args.strict
                and (report["unresolved"] or any(not r["exported"] for r in report["routes"]))
                else 0
            )
        generated = render(compiler.compile())
        validate_output_paths(generated)
        lexical_out = (
            (Path.cwd() / args.out if not args.out.is_absolute() else args.out)
            if args.out
            else root / config["output"]
        )
        cursor = lexical_out
        while True:
            if cursor.exists() and (
                cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction())
            ):
                raise ContractError("Symlinks are not allowed in the generated output path")
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        out = lexical_out.resolve()
        if out == root or root.is_relative_to(out):
            raise ContractError("Output directory must not be the repository or its parent")
        existing = (
            {
                p.relative_to(out).as_posix(): p
                for p in out.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            }
            if out.exists()
            else {}
        )
        if any(p.is_symlink() for p in out.rglob("*")) if out.exists() else False:
            raise ContractError("Symlinks are not allowed in the generated output tree")
        if existing and "manifest.json" not in existing:
            raise ContractError("Nonempty output directory has no ownership manifest")
        old_owned = set()
        if "manifest.json" in existing:
            old_owned = set(json.loads(existing["manifest.json"].read_text())["files"]) | {
                "manifest.json"
            }
            validate_output_paths(dict.fromkeys(old_owned, b""))
        unknown = set(existing) - old_owned
        if unknown:
            raise ContractError("Unowned files in output directory: " + ", ".join(sorted(unknown)))
        changed = [
            p
            for p, data in generated.items()
            if p not in existing or existing[p].read_bytes() != data
        ]
        stale = sorted(set(existing) - set(generated))
        if args.command == "check":
            if changed or stale:
                print("Generated drift: " + ", ".join(changed + stale), file=sys.stderr)
                return 1
            print(
                f"PASS: {len(generated)} generated files match byte-for-byte ({'fixture' if args.fixture else 'source'} mode)"
            )
            return 0
        staged: list[tuple[Path, Path]] = []
        try:
            for relative, content in generated.items():
                target = out / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".sdkgen-tmp")
                temporary.write_bytes(content)
                staged.append((temporary, target))
        except BaseException:
            for temporary, _target in staged:
                temporary.unlink(missing_ok=True)
            raise
        for path in stale:
            existing[path].unlink()
        for temporary, target in staged:
            temporary.replace(target)
        print(
            f"Generated {len(generated)} files under {out} ({'fixture' if args.fixture else 'source'} mode)"
        )
        return 0
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"sdkgen: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
