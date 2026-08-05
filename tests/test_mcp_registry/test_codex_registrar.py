"""Tests for the OpenAI Codex MCP registrar."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from headroom.mcp_registry.base import RegisterStatus, ServerSpec
from headroom.mcp_registry.codex import CodexRegistrar
from headroom.mcp_registry.install import build_headroom_spec

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


_RESOLVED_COMMAND = ("/usr/bin/python", "-m", "headroom.cli")
_RESOLVED_ARGS = ("-m", "headroom.cli", "mcp", "serve")


def _make_registrar(tmp_path: Path) -> CodexRegistrar:
    return CodexRegistrar(home_dir=tmp_path)


def _spec(env: dict[str, str] | None = None) -> ServerSpec:
    return ServerSpec(
        name="headroom",
        command=_RESOLVED_COMMAND[0],
        args=_RESOLVED_ARGS,
        env=env or {},
    )


def _install_spec(monkeypatch: pytest.MonkeyPatch) -> ServerSpec:
    monkeypatch.setattr(
        "headroom.mcp_registry.install.resolve_headroom_command",
        lambda: list(_RESOLVED_COMMAND),
    )
    return build_headroom_spec()


def _serena_spec() -> ServerSpec:
    return ServerSpec(
        name="serena",
        command="uvx",
        args=(
            "--from",
            "git+https://github.com/oraios/serena",
            "serena",
            "start-mcp-server",
            "--project-from-cwd",
            "--context",
            "codex",
        ),
    )


def _config_path(tmp_path: Path) -> Path:
    return tmp_path / ".codex" / "config.toml"


def _codex_home_config_path(codex_home: Path) -> Path:
    return codex_home / "config.toml"


# ----------------------------------------------------------------------
# detect()
# ----------------------------------------------------------------------


def test_detect_true_when_codex_dir_exists(tmp_path: Path) -> None:
    (tmp_path / ".codex").mkdir()
    assert _make_registrar(tmp_path).detect() is True


def test_detect_false_when_codex_dir_missing(tmp_path: Path) -> None:
    assert _make_registrar(tmp_path).detect() is False


def test_detect_true_when_codex_home_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert CodexRegistrar().detect() is True


def test_register_uses_codex_home_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    codex_home = tmp_path / "custom-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = CodexRegistrar().register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    assert _codex_home_config_path(codex_home).exists()
    assert not _config_path(tmp_path).exists()
    text = _codex_home_config_path(codex_home).read_text()
    assert "[mcp_servers.headroom]" in text


# ----------------------------------------------------------------------
# get_server()
# ----------------------------------------------------------------------


def test_get_server_returns_none_when_config_missing(tmp_path: Path) -> None:
    assert _make_registrar(tmp_path).get_server("headroom") is None


def test_get_server_returns_none_when_no_table(tmp_path: Path) -> None:
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    cfg.write_text('model = "gpt-4o"\n')
    assert _make_registrar(tmp_path).get_server("headroom") is None


def test_get_server_returns_spec_when_table_present(tmp_path: Path) -> None:
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    cfg.write_text(
        "[mcp_servers.headroom]\n"
        f"command = {_RESOLVED_COMMAND[0]!r}\n"
        f"args = {list(_RESOLVED_ARGS)!r}\n"
        "\n"
        "[mcp_servers.headroom.env]\n"
        'HEADROOM_PROXY_URL = "http://127.0.0.1:9000"\n'
    )
    got = _make_registrar(tmp_path).get_server("headroom")
    assert got is not None
    assert got.command == _RESOLVED_COMMAND[0]
    assert got.args == _RESOLVED_ARGS
    assert got.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"}


def test_get_server_robust_to_unparseable_toml(tmp_path: Path) -> None:
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    cfg.write_text("this = is = not = valid\n")
    assert _make_registrar(tmp_path).get_server("headroom") is None


def test_register_refuses_unparseable_config(tmp_path: Path) -> None:
    """An unparseable config.toml must not be appended to (that would corrupt it
    further); refuse and leave it byte-for-byte untouched."""
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    original = "this = is = not = valid\n"
    cfg.write_text(original)

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.FAILED
    assert "not valid TOML" in result.detail
    assert cfg.read_text() == original


def test_register_refuses_non_table_mcp_servers_entry(tmp_path: Path) -> None:
    """A valid config whose mcp_servers.headroom is a non-table must not get a
    duplicate `[mcp_servers.headroom]` table appended (which tomllib rejects)."""
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    original = '[mcp_servers]\nheadroom = "not-a-table"\n'
    cfg.write_text(original)

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.FAILED
    assert "non-table" in result.detail
    # Untouched — still the original single (string) definition.
    assert cfg.read_text() == original


def test_register_refuses_non_table_mcp_servers(tmp_path: Path) -> None:
    """A non-table top-level mcp_servers is also refused, not clobbered."""
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    original = 'mcp_servers = "oops"\n'
    cfg.write_text(original)

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.FAILED
    assert cfg.read_text() == original


# ----------------------------------------------------------------------
# register_server() — happy paths
# ----------------------------------------------------------------------


def test_register_creates_config_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reg = _make_registrar(tmp_path)
    result = reg.register_server(_install_spec(monkeypatch))
    assert result.status == RegisterStatus.REGISTERED
    cfg = _config_path(tmp_path)
    assert cfg.exists()
    text = cfg.read_text()
    assert "# --- Headroom MCP server ---" in text
    assert "[mcp_servers.headroom]" in text
    parsed = tomllib.loads(text)
    assert parsed["mcp_servers"]["headroom"]["command"] == _RESOLVED_COMMAND[0]
    assert parsed["mcp_servers"]["headroom"]["args"] == list(_RESOLVED_ARGS)


def test_register_appends_to_existing_config_preserves_other_keys(tmp_path: Path) -> None:
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    cfg.write_text('# user comment\nmodel = "gpt-4o"\n\n[other_section]\nvalue = 42\n')
    result = _make_registrar(tmp_path).register_server(_spec())
    assert result.status == RegisterStatus.REGISTERED
    text = cfg.read_text()
    # Existing content survived.
    assert "# user comment" in text
    assert 'model = "gpt-4o"' in text
    assert "[other_section]" in text
    # Plus our block.
    assert "[mcp_servers.headroom]" in text
    parsed = tomllib.loads(text)
    assert parsed["model"] == "gpt-4o"
    assert parsed["other_section"]["value"] == 42
    assert parsed["mcp_servers"]["headroom"]["command"] == _RESOLVED_COMMAND[0]


def test_register_includes_env_subtable(tmp_path: Path) -> None:
    spec = _spec(env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"})
    _make_registrar(tmp_path).register_server(spec)
    text = _config_path(tmp_path).read_text()
    assert "[mcp_servers.headroom.env]" in text
    parsed = tomllib.loads(text)
    assert parsed["mcp_servers"]["headroom"]["env"] == {
        "HEADROOM_PROXY_URL": "http://127.0.0.1:9000"
    }


def test_register_headroom_and_serena_coexist(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path)

    assert reg.register_server(_spec()).status == RegisterStatus.REGISTERED
    assert reg.register_server(_serena_spec()).status == RegisterStatus.REGISTERED

    text = _config_path(tmp_path).read_text()
    assert "[mcp_servers.headroom]" in text
    assert "[mcp_servers.serena]" in text
    assert "# --- Headroom MCP server ---" in text
    assert "# --- Headroom MCP server: serena ---" in text

    parsed = tomllib.loads(text)
    assert parsed["mcp_servers"]["headroom"]["command"] == _RESOLVED_COMMAND[0]
    assert parsed["mcp_servers"]["serena"]["command"] == "uvx"


def test_register_omits_env_subtable_when_env_empty(tmp_path: Path) -> None:
    _make_registrar(tmp_path).register_server(_spec())
    text = _config_path(tmp_path).read_text()
    assert "[mcp_servers.headroom.env]" not in text


# ----------------------------------------------------------------------
# Idempotency: ALREADY / MISMATCH
# ----------------------------------------------------------------------


def test_register_already_when_block_matches_spec(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path)
    reg.register_server(_spec())  # first install
    text_before = _config_path(tmp_path).read_text()
    result = reg.register_server(_spec())  # second install, same spec
    assert result.status == RegisterStatus.ALREADY
    # File unchanged.
    assert _config_path(tmp_path).read_text() == text_before


def test_register_mismatch_when_block_differs_no_force(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path)
    reg.register_server(_spec(env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"}))
    text_before = _config_path(tmp_path).read_text()

    result = reg.register_server(_spec())  # no env
    assert result.status == RegisterStatus.MISMATCH
    assert "env" in (result.detail or "")
    assert _config_path(tmp_path).read_text() == text_before  # unchanged


def test_register_force_overwrites_block(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path)
    reg.register_server(_spec(env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"}))
    result = reg.register_server(_spec(), force=True)
    assert result.status == RegisterStatus.REGISTERED
    text = _config_path(tmp_path).read_text()
    assert "9999" not in text


def test_register_force_preserves_user_managed_entry(tmp_path: Path) -> None:
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    cfg.write_text(
        '[mcp_servers.headroom]\ncommand = "/usr/local/bin/custom-headroom"\nargs = ["serve"]\n'
    )

    result = _make_registrar(tmp_path).register_server(_spec(), force=True)

    assert result.status == RegisterStatus.MISMATCH
    assert "user-managed" in (result.detail or "").lower()
    assert "/usr/local/bin/custom-headroom" in cfg.read_text()
    assert cfg.read_text().count("[mcp_servers.headroom]") == 1


def test_register_mismatch_when_user_managed_outside_markers(tmp_path: Path) -> None:
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    # User has manually put [mcp_servers.headroom] with different config — no markers.
    cfg.write_text(
        '[mcp_servers.headroom]\ncommand = "/usr/local/bin/custom-headroom"\nargs = ["serve"]\n'
    )
    result = _make_registrar(tmp_path).register_server(_spec())
    assert result.status == RegisterStatus.MISMATCH
    assert "user-managed" in (result.detail or "").lower()
    # Don't overwrite.
    assert "/usr/local/bin/custom-headroom" in cfg.read_text()


# ----------------------------------------------------------------------
# unregister
# ----------------------------------------------------------------------


def test_unregister_removes_marker_block(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path)
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    cfg.write_text("[other_section]\nvalue = 42\n")
    reg.register_server(_spec())
    assert "[mcp_servers.headroom]" in cfg.read_text()

    assert reg.unregister_server("headroom") is True
    text = cfg.read_text()
    assert "[mcp_servers.headroom]" not in text
    assert "# --- Headroom MCP server ---" not in text
    # Surrounding content survives.
    assert "[other_section]" in text


def test_unregister_serena_preserves_headroom_block(tmp_path: Path) -> None:
    reg = _make_registrar(tmp_path)
    reg.register_server(_spec())
    reg.register_server(_serena_spec())

    assert reg.unregister_server("serena") is True
    text = _config_path(tmp_path).read_text()
    assert "[mcp_servers.headroom]" in text
    assert "[mcp_servers.serena]" not in text
    assert "# --- Headroom MCP server ---" in text
    assert "# --- Headroom MCP server: serena ---" not in text


def test_unregister_returns_false_when_no_block(tmp_path: Path) -> None:
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    cfg.write_text('model = "gpt-4o"\n')
    assert _make_registrar(tmp_path).unregister_server("headroom") is False


def test_unregister_preserves_user_managed_entry(tmp_path: Path) -> None:
    """User-managed [mcp_servers.headroom] without our markers stays put."""
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir()
    cfg.write_text('[mcp_servers.headroom]\ncommand = "/custom/headroom"\n')
    # No markers => unregister is a no-op.
    assert _make_registrar(tmp_path).unregister_server("headroom") is False
    assert "/custom/headroom" in cfg.read_text()


# ----------------------------------------------------------------------
# Round-trip: write → re-read produces equivalent ServerSpec
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        ServerSpec(name="headroom", command=_RESOLVED_COMMAND[0], args=_RESOLVED_ARGS),
        ServerSpec(
            name="headroom",
            command=_RESOLVED_COMMAND[0],
            args=_RESOLVED_ARGS,
            env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"},
        ),
        ServerSpec(name="headroom", command="/usr/bin/headroom", args=()),
    ],
)
def test_round_trip(tmp_path: Path, spec: ServerSpec) -> None:
    reg = _make_registrar(tmp_path)
    reg.register_server(spec)
    got = reg.get_server("headroom")
    assert got is not None
    assert got.command == spec.command
    assert got.args == spec.args
    assert got.env == spec.env


# ---------------------------------------------------------------------------
# #733 — encoding / line-ending safety on GBK / CRLF Windows configs
# ---------------------------------------------------------------------------


def test_register_does_not_double_crlf(tmp_path: Path) -> None:
    """A pre-existing CRLF config must not gain ``\\r\\r\\n`` after register."""
    import tomllib

    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_bytes(b'model = "gpt-5"\r\nworkers = 1\r\n')

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    raw = cfg.read_bytes()
    assert b"\r\r\n" not in raw
    tomllib.loads(raw.decode("utf-8"))  # still valid TOML


def test_register_preserves_non_ascii_values(tmp_path: Path) -> None:
    """A config with non-ASCII (Chinese) values survives register and stays parseable."""
    import tomllib

    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('model = "gpt-5"\nproject = "比赛/机器人"\n', encoding="utf-8")

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    data = tomllib.loads(cfg.read_text(encoding="utf-8"))
    assert data["project"] == "比赛/机器人"
    assert "headroom" in data.get("mcp_servers", {})
