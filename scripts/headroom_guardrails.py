#!/usr/bin/env python3
"""Static architectural guardrails for recurring Headroom failure classes.

Ruff, mypy, Clippy, and tests catch broad correctness issues. This runner
captures project-specific contracts that have repeatedly regressed in PRs:
message metadata preservation, CCR marker/store parity, pipeline event
metadata, CORS scoping, workflow scoping, and Rust lint-policy opt-in.

The rules are intentionally small and dependency-free. Add a new ``Rule`` when
review finds a reusable invariant that would have prevented a class of bugs.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Finding:
    rule: str
    path: Path
    line: int
    message: str

    def render(self, root: Path = ROOT) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}: {self.rule}: {self.message}"


class Rule(Protocol):
    id: str
    summary: str

    def check(self, root: Path) -> list[Finding]: ...


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dict_keys(node: ast.Dict) -> set[str]:
    keys: set[str] = set()
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
    return keys


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _contains_subscript_name_index(node: ast.AST, target_name: str, index_name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        if not isinstance(child.value, ast.Name) or child.value.id != target_name:
            continue
        slice_node = child.slice
        if isinstance(slice_node, ast.Name) and slice_node.id == index_name:
            return True
    return False


class BackendMessagePreservationRule:
    id = "PY001"
    summary = "backend message reconstruction must preserve provider metadata"

    paths = (
        Path("headroom/backends/litellm.py"),
        Path("headroom/backends/anyllm.py"),
    )

    def check(self, root: Path) -> list[Finding]:
        findings: list[Finding] = []
        for rel in self.paths:
            path = root / rel
            if not path.exists():
                findings.append(
                    Finding(
                        self.id,
                        path,
                        1,
                        f"expected backend file not found: {rel}",
                    )
                )
                continue
            tree = _parse(path)
            source = _read(path)
            if "preserve_message_fields" not in source:
                findings.append(
                    Finding(
                        self.id,
                        path,
                        1,
                        "backend conversion file must import and use preserve_message_fields",
                    )
                )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _call_name(node.func) != "append" or not node.args:
                    continue
                first = node.args[0]
                if not isinstance(first, ast.Dict):
                    continue
                keys = _dict_keys(first)
                if {"role", "content"}.issubset(keys) and _contains_name(first, "role"):
                    findings.append(
                        Finding(
                            self.id,
                            path,
                            node.lineno,
                            "wrap role/content reconstruction in preserve_message_fields(...)",
                        )
                    )
        return findings


class NoPositionalMessageRestoreRule:
    id = "PY002"
    summary = "do not restore original message metadata by optimized-message index"

    paths = (
        Path("headroom/proxy/handlers/openai.py"),
        Path("headroom/proxy/handlers/anthropic.py"),
        Path("headroom/proxy/handlers/gemini.py"),
        Path("headroom/backends/litellm.py"),
        Path("headroom/backends/anyllm.py"),
    )

    def check(self, root: Path) -> list[Finding]:
        findings: list[Finding] = []
        for rel in self.paths:
            path = root / rel
            if not path.exists():
                continue
            tree = _parse(path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.For):
                    continue
                if not isinstance(node.iter, ast.Call):
                    continue
                if _call_name(node.iter.func) != "enumerate" or not node.iter.args:
                    continue
                arg = node.iter.args[0]
                if not isinstance(arg, ast.Name) or arg.id != "optimized_messages":
                    continue
                index_name = None
                if isinstance(node.target, ast.Tuple) and node.target.elts:
                    maybe_index = node.target.elts[0]
                    if isinstance(maybe_index, ast.Name):
                        index_name = maybe_index.id
                if index_name and _contains_subscript_name_index(
                    node, "original_messages", index_name
                ):
                    findings.append(
                        Finding(
                            self.id,
                            path,
                            node.lineno,
                            "preserve fields at reconstruction sites; index-based restore is fragile",
                        )
                    )
        return findings


class CcrExplicitHashRule:
    id = "PY003"
    summary = "Rust CCR marker shims must store under the marker hash"

    paths = (
        Path("headroom/transforms/search_compressor.py"),
        Path("headroom/transforms/diff_compressor.py"),
        Path("headroom/transforms/log_compressor.py"),
    )

    def check(self, root: Path) -> list[Finding]:
        findings: list[Finding] = []
        for rel in self.paths:
            path = root / rel
            tree = _parse(path)
            functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_persist_to_python_ccr"
            ]
            if not functions:
                findings.append(
                    Finding(self.id, path, 1, "missing _persist_to_python_ccr marker bridge")
                )
                continue
            for fn in functions:
                store_calls = [
                    node
                    for node in ast.walk(fn)
                    if isinstance(node, ast.Call) and _call_name(node.func) == "store"
                ]
                if not store_calls:
                    findings.append(
                        Finding(self.id, path, fn.lineno, "no store.store(...) call found")
                    )
                    continue
                for call in store_calls:
                    kwargs = {kw.arg for kw in call.keywords if kw.arg is not None}
                    if "explicit_hash" not in kwargs:
                        findings.append(
                            Finding(
                                self.id,
                                path,
                                call.lineno,
                                "store.store(...) must pass explicit_hash=cache_key",
                            )
                        )
        return findings


class InputCompressedOriginalMessagesRule:
    id = "PY004"
    summary = "INPUT_COMPRESSED events must expose original_messages"

    paths = (
        Path("headroom/proxy/handlers/openai.py"),
        Path("headroom/proxy/handlers/anthropic.py"),
    )

    def check(self, root: Path) -> list[Finding]:
        findings: list[Finding] = []
        for rel in self.paths:
            path = root / rel
            tree = _parse(path)
            compressed_emits: list[ast.Call] = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _call_name(node.func) != "emit":
                    continue
                if "PipelineStage.INPUT_COMPRESSED" not in ast.unparse(node):
                    continue
                compressed_emits.append(node)
                metadata_nodes = [
                    kw.value
                    for kw in node.keywords
                    if kw.arg == "metadata" and isinstance(kw.value, ast.Dict)
                ]
                if not metadata_nodes or not any(
                    "original_messages" in _dict_keys(meta) for meta in metadata_nodes
                ):
                    findings.append(
                        Finding(
                            self.id,
                            path,
                            node.lineno,
                            "INPUT_COMPRESSED emit metadata must include original_messages",
                        )
                    )
            if not compressed_emits:
                findings.append(Finding(self.id, path, 1, "no INPUT_COMPRESSED emit found"))
        return findings


class CorsScopeRule:
    id = "PY005"
    summary = "proxy CORS defaults must be localhost-scoped and config-port aware"

    server_path = Path("headroom/proxy/server.py")
    policy_path = Path("headroom/proxy/cors.py")

    def check(self, root: Path) -> list[Finding]:
        server_path = root / self.server_path
        policy_path = root / self.policy_path
        server_text = _read(server_path)
        policy_text = _read(policy_path)
        findings: list[Finding] = []
        server_checks = [
            (
                "cors_origins_for_config(config)",
                "CORSMiddleware must use cors_origins_for_config(config)",
            ),
            ("allow_credentials=False", "CORS credentials must be disabled"),
        ]
        for needle, message in server_checks:
            if needle not in server_text:
                findings.append(Finding(self.id, server_path, 1, message))
        policy_checks = [
            ("CORS_ORIGINS_ENV", "CORS env override must be centralized"),
            ("os.environ.get(CORS_ORIGINS_ENV", "wildcard/custom CORS must be env-explicit"),
            ("config.port", "default CORS origins must use the effective config port"),
        ]
        for needle, message in policy_checks:
            if needle not in policy_text:
                findings.append(Finding(self.id, policy_path, 1, message))
        non_comment = "\n".join(
            line for line in server_text.splitlines() if not line.lstrip().startswith("#")
        )
        if 'allow_origins=["*"]' in non_comment or "allow_credentials=True" in non_comment:
            findings.append(
                Finding(
                    self.id,
                    server_path,
                    1,
                    "wildcard/default credentialed CORS is forbidden; require explicit opt-in",
                )
            )
        return findings


class CiWorkflowGuardrailsRule:
    id = "CI001"
    summary = "CI must run guardrails and scope commitlint to PRs"

    path = Path(".github/workflows/ci.yml")

    def check(self, root: Path) -> list[Finding]:
        path = root / self.path
        text = _read(path)
        findings: list[Finding] = []
        required = [
            ("architectural-guardrails:", "missing architectural-guardrails CI job"),
            (
                "python scripts/headroom_guardrails.py",
                "CI must execute scripts/headroom_guardrails.py",
            ),
            (
                "github.event_name == 'pull_request'",
                "commitlint must stay scoped to pull_request events",
            ),
        ]
        for needle, message in required:
            if needle not in text:
                findings.append(Finding(self.id, path, 1, message))
        return findings


class PreCommitGuardrailsRule:
    id = "CI002"
    summary = "pre-commit must run repo guardrails"

    path = Path(".pre-commit-config.yaml")

    def check(self, root: Path) -> list[Finding]:
        path = root / self.path
        text = _read(path)
        findings: list[Finding] = []
        if "headroom-guardrails" not in text or "scripts/headroom_guardrails.py" not in text:
            findings.append(
                Finding(
                    self.id,
                    path,
                    1,
                    "pre-commit must include the local headroom-guardrails hook",
                )
            )
        return findings


class RustWorkspaceLintRule:
    id = "RS001"
    summary = "Rust crates must opt into workspace lint policy"

    def check(self, root: Path) -> list[Finding]:
        findings: list[Finding] = []
        workspace = root / "Cargo.toml"
        text = _read(workspace)
        if "[workspace.lints.rust]" not in text or 'unsafe_code = "forbid"' not in text:
            findings.append(
                Finding(
                    self.id,
                    workspace,
                    1,
                    "workspace must define rust unsafe_code = forbid lint policy",
                )
            )
        for cargo in sorted((root / "crates").glob("*/Cargo.toml")):
            cargo_text = _read(cargo)
            if "[lints]" not in cargo_text or "workspace = true" not in cargo_text:
                findings.append(
                    Finding(
                        self.id,
                        cargo,
                        1,
                        "crate must opt into workspace lints with [lints] workspace = true",
                    )
                )
        return findings


RULES: tuple[Rule, ...] = (
    BackendMessagePreservationRule(),
    NoPositionalMessageRestoreRule(),
    CcrExplicitHashRule(),
    InputCompressedOriginalMessagesRule(),
    CorsScopeRule(),
    CiWorkflowGuardrailsRule(),
    PreCommitGuardrailsRule(),
    RustWorkspaceLintRule(),
)


def run(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rule in RULES:
        findings.extend(rule.check(root))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    args = parser.parse_args()

    root = args.root.resolve()
    findings = run(root)
    if findings:
        print("Headroom architectural guardrails failed:")
        for finding in findings:
            print(f"  {finding.render(root)}")
        return 1
    print(f"Headroom architectural guardrails passed ({len(RULES)} rules).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
