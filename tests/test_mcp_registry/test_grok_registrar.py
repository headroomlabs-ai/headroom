"""Tests for the Grok Build MCP registrar."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.mcp_registry.base import RegisterStatus, ServerSpec
from headroom.mcp_registry.grok import GrokRegistrar


def _make_registrar(tmp_path: Path) -> GrokRegistrar:
    return GrokRegistrar(home_dir=tmp_path)


def _spec() -> ServerSpec:
    return ServerSpec(
        name="headroom",
        command="/usr/bin/python",
        args=("-m", "headroom.cli", "mcp", "serve"),
    )


def test_detect_true_when_grok_dir_exists(tmp_path: Path) -> None:
    (tmp_path / ".grok").mkdir()
    assert _make_registrar(tmp_path).detect() is True


def test_detect_false_when_grok_dir_missing(tmp_path: Path) -> None:
    assert _make_registrar(tmp_path).detect() is False


def test_register_uses_grok_home_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    grok_home = tmp_path / "custom-grok-home"
    monkeypatch.setenv("GROK_HOME", str(grok_home))

    result = GrokRegistrar().register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    config = grok_home / "config.toml"
    assert config.exists()
    assert "[mcp_servers.headroom]" in config.read_text()


def test_unregister_keeps_foreign_table_inside_span(tmp_path: Path) -> None:
    """A table another writer appended before our end marker survives unregister."""
    import tomllib

    reg = _make_registrar(tmp_path)
    reg.register_server(_spec())
    cfg = tmp_path / ".grok" / "config.toml"
    end = "# --- end Headroom MCP server ---"
    cfg.write_text(cfg.read_text().replace(end, '[mcp_servers.other]\ncommand = "other"\n' + end))

    assert reg.unregister_server("headroom") is True

    assert tomllib.loads(cfg.read_text())["mcp_servers"] == {"other": {"command": "other"}}


def _insert_before_end(cfg: Path, text: str) -> None:
    end = "# --- end Headroom MCP server ---"
    cfg.write_text(cfg.read_text().replace(end, text + end))


def test_unregister_keeps_quoted_key_table_inside_span(tmp_path: Path) -> None:
    import tomllib

    reg = _make_registrar(tmp_path)
    reg.register_server(_spec())
    cfg = tmp_path / ".grok" / "config.toml"
    _insert_before_end(cfg, '[mcp_servers."foo#bar"]\ncommand = "foreign"\n')

    assert reg.unregister_server("headroom") is True

    assert tomllib.loads(cfg.read_text())["mcp_servers"] == {"foo#bar": {"command": "foreign"}}


@pytest.mark.parametrize(
    "inside_span",
    ['[notes]\ntext = """\n[mcp_servers.headroom.env]\n"""\n', "broken =\n"],
)
def test_refuses_span_rewrite_that_would_change_other_entries(
    tmp_path: Path, inside_span: str
) -> None:
    reg = _make_registrar(tmp_path)
    reg.register_server(_spec())
    cfg = tmp_path / ".grok" / "config.toml"
    _insert_before_end(cfg, inside_span)
    before = cfg.read_text()

    assert reg.unregister_server("headroom") is False
    moved = ServerSpec(name="headroom", command="/opt/python", args=_spec().args)
    assert reg.register_server(moved, force=True).status == RegisterStatus.FAILED
    assert cfg.read_text() == before


def test_register_escapes_control_chars_in_env_value(tmp_path: Path) -> None:
    """Grok shares the codex TOML escaper: a newline env value must round-trip."""
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover
        import tomli as tomllib

    reg = _make_registrar(tmp_path)
    pem = "-----BEGIN KEY-----\nabc\n-----END KEY-----"
    spec = ServerSpec(
        name="headroom",
        command="/usr/bin/python",
        args=("-m", "headroom.cli", "mcp", "serve"),
        env={"PEM": pem},
    )

    result = reg.register_server(spec, force=True)

    assert result.status == RegisterStatus.REGISTERED
    parsed = tomllib.loads(reg._config_file.read_text())
    assert parsed["mcp_servers"]["headroom"]["env"]["PEM"] == pem
    assert reg.get_server("headroom").env["PEM"] == pem
