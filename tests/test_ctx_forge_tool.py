"""Tests for the ctx-forge context-tool integration."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from headroom import ctx_forge
from headroom.cli import wrap

MANIFEST_MINIMAL = """\
[ctx]
contract_version = "0.1"
conformance = "standard"

[commands.map]
tier = 0
source = "tools/map.py"
description = "Repo overview"

[commands.find]
tier = 0
source = "tools/find.py"
description = "Semantic find over the symbol index"
"""


def _write_toolset(
    root: Path,
    manifest: str = MANIFEST_MINIMAL,
    state: dict | None = None,
) -> None:
    ctx_dir = root / ".ctx"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    (ctx_dir / "ctx.toml").write_text(manifest)
    if state is not None:
        cache = ctx_dir / "cache"
        cache.mkdir(exist_ok=True)
        (cache / "state.json").write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# detection


def test_find_toolset_walks_up_from_subdirectory(tmp_path: Path) -> None:
    _write_toolset(tmp_path, state={"last_selftest_result": "pass"})
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)

    toolset = ctx_forge.find_toolset(nested)

    assert toolset is not None
    assert toolset.repo_root == tmp_path
    assert toolset.entrypoint == tmp_path / ".ctx" / "ctx"
    assert toolset.contract_version == "0.1"
    assert sorted(toolset.commands) == ["find", "map"]


def test_find_toolset_returns_none_without_manifest(tmp_path: Path) -> None:
    assert ctx_forge.find_toolset(tmp_path) is None


def test_find_toolset_rejects_commandless_manifest(tmp_path: Path) -> None:
    _write_toolset(tmp_path, manifest='[ctx]\ncontract_version = "0.1"\n')
    assert ctx_forge.find_toolset(tmp_path) is None


def test_find_toolset_treats_malformed_manifest_as_absent(tmp_path: Path) -> None:
    ctx_dir = tmp_path / ".ctx"
    ctx_dir.mkdir()
    (ctx_dir / "ctx.toml").write_text("not [ valid toml ===")
    assert ctx_forge.find_toolset(tmp_path) is None


# ---------------------------------------------------------------------------
# trust: state file first, legacy manifest fallback, fresh checkout untrusted


def test_trust_comes_from_state_file(tmp_path: Path) -> None:
    _write_toolset(tmp_path, state={"last_selftest_result": "pass"})
    toolset = ctx_forge.find_toolset(tmp_path)
    assert toolset is not None and toolset.trusted

    (tmp_path / ".ctx" / "cache" / "state.json").write_text(
        json.dumps({"last_selftest_result": "fail"})
    )
    toolset = ctx_forge.find_toolset(tmp_path)
    assert toolset is not None and not toolset.trusted
    assert toolset.selftest_result == "fail"


def test_fresh_checkout_without_state_is_untrusted(tmp_path: Path) -> None:
    _write_toolset(tmp_path)  # no cache/state.json — gitignored, absent on clone
    toolset = ctx_forge.find_toolset(tmp_path)
    assert toolset is not None
    assert toolset.selftest_result == "never"
    assert not toolset.trusted


def test_legacy_manifest_verify_is_honored_without_state_file(tmp_path: Path) -> None:
    legacy = MANIFEST_MINIMAL + '\n[verify]\nlast_selftest_result = "pass"\n'
    _write_toolset(tmp_path, manifest=legacy)
    toolset = ctx_forge.find_toolset(tmp_path)
    assert toolset is not None and toolset.trusted

    # The volatile state file always wins over the legacy manifest field.
    _write_toolset(tmp_path, manifest=legacy, state={"last_selftest_result": "fail"})
    toolset = ctx_forge.find_toolset(tmp_path)
    assert toolset is not None and not toolset.trusted


# ---------------------------------------------------------------------------
# guidance / summary


def test_guidance_text_lists_commands_and_staleness_recovery(tmp_path: Path) -> None:
    _write_toolset(tmp_path, state={"last_selftest_result": "pass"})
    toolset = ctx_forge.find_toolset(tmp_path)
    assert toolset is not None

    guidance = ctx_forge.guidance_text(toolset)

    assert "./.ctx/ctx map" in guidance
    assert "./.ctx/ctx find" in guidance
    assert "Semantic find over the symbol index" in guidance
    assert "regen" in guidance
    assert "WARNING" not in guidance


def test_guidance_text_warns_when_untrusted(tmp_path: Path) -> None:
    _write_toolset(tmp_path, state={"last_selftest_result": "fail"})
    toolset = ctx_forge.find_toolset(tmp_path)
    assert toolset is not None

    guidance = ctx_forge.guidance_text(toolset)

    assert "WARNING" in guidance
    assert "untrusted" in guidance


def test_setup_summary_without_toolset_points_at_the_skill() -> None:
    summary = ctx_forge.setup_summary(None)
    assert "no .ctx/ctx.toml" in summary
    assert "ctx-forge skill" in summary


# ---------------------------------------------------------------------------
# env selection


@pytest.mark.parametrize("spelling", ["ctx-forge", "ctxforge", "ctx_forge", "CTX-FORGE"])
def test_selected_context_tool_accepts_ctx_forge_spellings(monkeypatch, spelling: str) -> None:
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", spelling)
    assert wrap._selected_context_tool() == "ctx-forge"


def test_selected_context_tool_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "ctx-fudge")
    with pytest.raises(click.ClickException):
        wrap._selected_context_tool()


# ---------------------------------------------------------------------------
# wrap-side injection


def test_inject_ctx_forge_instructions_is_idempotent(tmp_path: Path) -> None:
    marker = tmp_path / ".cursorrules"
    marker.write_text("# my existing rules\n")

    assert wrap._inject_ctx_forge_instructions(marker, "GUIDANCE BODY")
    assert wrap._inject_ctx_forge_instructions(marker, "GUIDANCE BODY")

    content = marker.read_text()
    assert content.startswith("# my existing rules")
    assert content.count(wrap._CTX_FORGE_MARKER) == 1
    assert content.count("GUIDANCE BODY") == 1
    assert wrap._CTX_FORGE_END_MARKER in content


def test_setup_ctx_forge_agent_injects_into_plain_marker(tmp_path: Path, monkeypatch) -> None:
    _write_toolset(tmp_path, state={"last_selftest_result": "pass"})
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / ".clinerules"

    wrap._setup_ctx_forge_agent("Cline", marker_path=marker)

    content = marker.read_text()
    assert wrap._CTX_FORGE_MARKER in content
    assert "./.ctx/ctx map" in content


def test_setup_ctx_forge_agent_skips_structured_configs(tmp_path: Path, monkeypatch) -> None:
    _write_toolset(tmp_path, state={"last_selftest_result": "pass"})
    monkeypatch.chdir(tmp_path)
    config = tmp_path / ".continue" / "config.json"

    wrap._setup_ctx_forge_agent("Continue", marker_path=config)

    assert not config.exists()


def test_setup_ctx_forge_agent_without_toolset_touches_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / ".goosehints"

    wrap._setup_ctx_forge_agent("Goose", marker_path=marker)

    assert not marker.exists()


def test_setup_context_tool_for_agent_routes_ctx_forge(tmp_path: Path, monkeypatch) -> None:
    """The central context-tool fork must take the ctx-forge branch, never rtk."""
    _write_toolset(tmp_path, state={"last_selftest_result": "pass"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "ctx-forge")
    marker = tmp_path / ".cursorrules"

    def _fail_rtk(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("rtk installer must not run in ctx-forge mode")

    monkeypatch.setattr(wrap, "_ensure_rtk_binary", _fail_rtk)

    result = wrap._setup_context_tool_for_agent(
        agent="cursor",
        agent_display="Cursor",
        marker_path=marker,
        on_rtk_ready=lambda _rtk: _fail_rtk(),
    )

    assert result is None
    assert wrap._CTX_FORGE_MARKER in marker.read_text()
