"""ctx-forge context-tool support.

``HEADROOM_CONTEXT_TOOL=ctx-forge`` selects a `ctx-forge
<https://github.com/dvk31/ctx-forge>`_ generated toolset as the CLI context
tool, alongside ``rtk`` and ``lean-ctx``. Stdlib-only on purpose: tomllib
(3.11+) with a tomli fallback.

Unlike rtk/lean-ctx, ctx-forge is not a downloadable binary: the toolset is
generated *into the target repo* by the ctx-forge skill and lives under
``.ctx/``. Headroom's job is therefore detection + guidance, not
installation:

1. find the repo's ``.ctx/ctx.toml`` (walk up from cwd),
2. confirm the toolset is trusted (last selftest passed) — volatile run
   state lives in the gitignored ``.ctx/cache/state.json``; pre-state-file
   toolsets kept it in the manifest's ``[verify]`` section, which is read
   as a legacy fallback,
3. produce the short guidance block wrap.py injects into agent marker files
   (``.clinerules``, ``.goosehints``, ``.cursorrules``, ...) instead of the
   rtk guidance.

A repo without a toolset degrades gracefully: detection reports the absence
and wrap continues with plain proxy compression (generation requires the
ctx-forge agent skill, not a download — there is nothing to auto-install).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

CTX_DIR_NAME = ".ctx"
MANIFEST_NAME = "ctx.toml"


@dataclass(frozen=True)
class CtxToolset:
    """A detected, parsed ctx-forge toolset."""

    repo_root: Path
    entrypoint: Path
    contract_version: str
    conformance: str
    commands: dict[str, str] = field(default_factory=dict)  # name -> description
    selftest_result: str = "never"

    @property
    def trusted(self) -> bool:
        return self.selftest_result == "pass"


def find_toolset(start: Path | None = None) -> CtxToolset | None:
    """Locate and parse the nearest ctx-forge toolset at or above ``start``.

    Returns ``None`` when no ``.ctx/ctx.toml`` exists on the path to the
    filesystem root, or when the manifest is unreadable/invalid (a broken
    manifest is treated as no toolset; Headroom falls back to its default
    context tool rather than failing the wrap).
    """
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        manifest_path = candidate / CTX_DIR_NAME / MANIFEST_NAME
        if manifest_path.is_file():
            return _parse_manifest(candidate, manifest_path)
    return None


def _parse_manifest(repo_root: Path, manifest_path: Path) -> CtxToolset | None:
    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    ctx_section = data.get("ctx") or {}
    commands_section = data.get("commands") or {}
    verify_section = data.get("verify") or {}

    commands = {
        name: str(entry.get("description", f"ctx {name}"))
        for name, entry in commands_section.items()
        if isinstance(entry, dict)
    }
    if not commands:
        return None

    entrypoint = repo_root / CTX_DIR_NAME / "ctx"
    if os.name == "nt" and not entrypoint.exists():  # pragma: no cover
        entrypoint = repo_root / CTX_DIR_NAME / "ctx.cmd"

    return CtxToolset(
        repo_root=repo_root,
        entrypoint=entrypoint,
        contract_version=str(ctx_section.get("contract_version", "unknown")),
        conformance=str(ctx_section.get("conformance", "unknown")),
        commands=commands,
        selftest_result=_selftest_result(repo_root, verify_section),
    )


def _selftest_result(repo_root: Path, verify_section: dict) -> str:
    """Selftest verdict from the volatile state file, manifest as fallback.

    ``ctx regen``/``ctx selftest`` record results in the gitignored
    ``.ctx/cache/state.json`` — absent in a fresh checkout, which is
    correctly reported as ``never`` (untrusted until one regen runs).
    Pre-state-file toolsets wrote ``last_selftest_result`` into the
    manifest's ``[verify]`` section; honor it when no state file exists.
    """
    state_path = repo_root / CTX_DIR_NAME / "cache" / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = None
    if isinstance(state, dict) and "last_selftest_result" in state:
        return str(state["last_selftest_result"])
    return str(verify_section.get("last_selftest_result", "never"))


def guidance_text(toolset: CtxToolset) -> str:
    """Return the context-tool guidance block for agent marker files.

    Mirrors the role of Headroom's rtk guidance: tell the agent which
    commands exist and when to prefer them over raw exploration. Kept under
    ~25 lines because this text rides along in every session.
    """
    lines = [
        "## Context tools (ctx-forge)",
        "",
        "This repo has a generated, verified `ctx` toolset. Prefer it over",
        "exploratory grep/read for codebase questions — one call, file:line",
        "anchored, token-dense.",
        "",
    ]
    entry = "./.ctx/ctx"
    for name, description in sorted(toolset.commands.items()):
        lines.append(f"- `{entry} {name}` — {description}")
    lines += [
        "",
        "Flags: `--json` (machine-readable), `--locate` (path:line:name),",
        "`--help`. Exit code 2 means the index is stale: run",
        f"`{entry} regen` and retry.",
    ]
    if not toolset.trusted:
        lines += [
            "",
            "WARNING: last selftest did not pass — treat answers as untrusted,",
            f"prefer raw exploration, and run `{entry} regen`.",
        ]
    return "\n".join(lines)


def setup_summary(toolset: CtxToolset | None) -> str:
    """One-line status for wrap-time console output."""
    if toolset is None:
        return (
            "ctx-forge: no .ctx/ctx.toml found in this repo — run the "
            "ctx-forge skill to generate a toolset, or unset "
            "HEADROOM_CONTEXT_TOOL."
        )
    state = "trusted" if toolset.trusted else f"UNTRUSTED (selftest: {toolset.selftest_result})"
    return (
        f"ctx-forge: {len(toolset.commands)} commands "
        f"(contract {toolset.contract_version}, {toolset.conformance}, {state}) "
        f"at {toolset.repo_root}"
    )
