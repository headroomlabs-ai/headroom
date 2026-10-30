"""Tests for the declarative WrapTarget registry and its generated commands.

Every target gets the same end-to-end checks (missing binary, --prepare-only,
env routing on launch, Bob included); the rest pins the banner, parity with the aider builder
openclaude used to call, and the option surface. Bob's preflight has its own
suite in test_wrap_bob.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest
from click.testing import CliRunner

import headroom.cli.wrap as wrap_mod
from headroom.cli.main import main
from headroom.cli.wrap import wrap
from headroom.providers.wrap_registry import WRAP_TARGETS, build_launch_env


@pytest.mark.parametrize("name", sorted(WRAP_TARGETS))
def test_missing_binary_fails_with_install_hint(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: None)

    result = CliRunner().invoke(main, ["wrap", name])

    target = WRAP_TARGETS[name]
    assert result.exit_code == 1
    assert f"'{target.binary}' not found in PATH" in result.output
    assert target.install_hint in result.output


@pytest.mark.parametrize("name", sorted(WRAP_TARGETS))
def test_prepare_only_succeeds_unpatched(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--prepare-only` fetches and writes nothing, so it exits 0 on its own."""
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(main, ["wrap", name, "--prepare-only"])

    assert result.exit_code == 0, result.output


def _project_prefix(cwd: Path) -> str:
    return f"/p/{quote(cwd.name, safe='')}"


@pytest.mark.parametrize(
    ("name", "args", "expected_env"),
    [
        (
            "bob",
            ("--port", "9000", "--", "-p", "hello"),
            lambda cwd: {
                # Bare origin: Bob appends /inference/v1/... itself.
                "BOB_GATEWAY_URL": f"http://127.0.0.1:9000{_project_prefix(cwd)}",
            },
        ),
        (
            "goose",
            ("--port", "9000", "--", "session"),
            lambda _cwd: {
                "OPENAI_BASE_URL": "http://127.0.0.1:9000/v1",
                "OPENAI_API_BASE": "http://127.0.0.1:9000/v1",
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:9000",
                # Goose's Anthropic provider reads ANTHROPIC_HOST, not _BASE_URL.
                "ANTHROPIC_HOST": "http://127.0.0.1:9000",
            },
        ),
        (
            "openhands",
            ("--port", "9000", "--", "--task", "demo"),
            lambda _cwd: {
                "OPENAI_BASE_URL": "http://127.0.0.1:9000/v1",
                "OPENAI_API_BASE": "http://127.0.0.1:9000/v1",
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:9000",
                "LLM_BASE_URL": "http://127.0.0.1:9000/v1",
            },
        ),
        (
            "openclaude",
            ("--", "--model", "gpt-4o"),
            lambda cwd: {
                "OPENAI_API_BASE": f"http://127.0.0.1:8787{_project_prefix(cwd)}/v1",
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:8787{_project_prefix(cwd)}",
            },
        ),
    ],
)
def test_launch_routes_tool_env_through_proxy(
    name: str,
    args: tuple[str, ...],
    expected_env,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    # Bob's preflight reads ~/.bob/settings; keep it off the real home.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    captured: dict = {}
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kw: captured.update(kw))

    result = CliRunner().invoke(main, ["wrap", name, *args])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    for key, value in expected_env(Path.cwd()).items():
        assert env[key] == value, key
    assert captured["tool_label"] == name.upper()
    assert captured["agent_type"] == name
    assert captured["args"] == tuple(args[args.index("--") + 1 :])


def test_goose_banner_hides_openai_api_base_alias():
    _, display = build_launch_env(WRAP_TARGETS["goose"], 8787, environ={}, project="p")
    # Goose has no project prefix; its endpoint override is visible in the banner.
    assert display == [
        "OPENAI_BASE_URL=http://127.0.0.1:8787/v1",
        "ANTHROPIC_BASE_URL=http://127.0.0.1:8787",
        "ANTHROPIC_HOST=http://127.0.0.1:8787",
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
