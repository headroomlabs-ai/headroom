"""Tests for the declarative WrapTarget registry.

The per-tool suites (test_wrap_goose / _openhands / _openclaude) still cover
env routing, missing binaries, and --prepare-only end to end against the
generated commands. These tests pin what they don't: the banner, parity with
the aider builder openclaude used to call, and the option surface.
"""

from __future__ import annotations

from click.testing import CliRunner

import headroom.cli.wrap as wrap_mod
from headroom.cli.wrap import wrap
from headroom.providers.wrap_registry import WRAP_TARGETS, build_launch_env


def test_goose_banner_hides_openai_api_base_alias():
    _, display = build_launch_env(WRAP_TARGETS["goose"], 8787, environ={}, project="p")
    # Legacy goose never encoded the project prefix and showed two vars only.
    assert display == [
        "OPENAI_BASE_URL=http://127.0.0.1:8787/v1",
        "ANTHROPIC_BASE_URL=http://127.0.0.1:8787",
    ]


def test_openhands_banner_matches_legacy_body():
    _, display = build_launch_env(WRAP_TARGETS["openhands"], 9000, environ={})
    assert display == [
        "OPENAI_BASE_URL=http://127.0.0.1:9000/v1",
        "ANTHROPIC_BASE_URL=http://127.0.0.1:9000",
        "LLM_BASE_URL=http://127.0.0.1:9000/v1",
    ]


def test_openclaude_matches_legacy_aider_builder():
    from headroom.providers.aider import build_launch_env as aider_build_launch_env

    environ = {"PATH": "/bin"}  # non-empty: the aider builder treats {} as os.environ
    got = build_launch_env(WRAP_TARGETS["openclaude"], 8787, environ, project="myproj")
    assert got == aider_build_launch_env(8787, environ, project="myproj")


def test_option_surface_is_preserved():
    for name in WRAP_TARGETS:
        params = {p.name for p in wrap.commands[name].params}
        assert {
            "port",
            "code_graph",
            "no_proxy",
            "learn",
            "memory",
            "backend",
            "anyllm_provider",
            "region",
            "verbose",
            "prepare_only",
            "tool_args",
        } <= params, name


def test_launch_passes_flags_through(monkeypatch):
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    captured: dict = {}
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(
        wrap,
        ["goose", "--port", "9001", "--learn", "--backend", "anyllm", "--", "session"],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"] == ("session",)
    assert captured["port"] == 9001
    assert captured["learn"] is True
    assert captured["backend"] == "anyllm"
    assert captured["tool_label"] == "GOOSE"
    assert captured["agent_type"] == "goose"
