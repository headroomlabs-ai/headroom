"""A removal whose config write fails returns False and says why, per registrar."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from headroom.mcp_registry.base import MCPRegistrar, RegisterStatus, ServerSpec
from headroom.mcp_registry.claude import ClaudeRegistrar
from headroom.mcp_registry.codex import CodexRegistrar
from headroom.mcp_registry.grok import GrokRegistrar


def _no_space(fd: int) -> None:
    raise OSError(28, "No space left on device")


@pytest.mark.parametrize(
    ("make", "config", "logger_name"),
    [
        (
            lambda home: ClaudeRegistrar(claude_cli=None, home_dir=home),
            ".claude.json",
            "headroom.mcp_registry.claude",
        ),
        (
            lambda home: CodexRegistrar(home_dir=home),
            ".codex/config.toml",
            "headroom.mcp_registry.codex",
        ),
        (
            lambda home: GrokRegistrar(home_dir=home),
            ".grok/config.toml",
            # Grok shares the marker-block registrar, and its logger, with Codex.
            "headroom.mcp_registry.codex",
        ),
    ],
    ids=["claude", "codex", "grok"],
)
def test_failed_removal_keeps_the_config_and_logs_why(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    make,
    config: str,
    logger_name: str,
) -> None:
    monkeypatch.delenv("GROK_HOME", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    registrar: MCPRegistrar = make(tmp_path)
    spec = ServerSpec(name="headroom", command="/usr/bin/python", args=("-m", "headroom.cli"))
    assert registrar.register_server(spec).status == RegisterStatus.REGISTERED
    path = tmp_path / config
    before = path.read_bytes()

    monkeypatch.setattr("headroom.fsutil.os.fsync", _no_space)
    with caplog.at_level(logging.WARNING, logger=logger_name):
        assert registrar.unregister_server("headroom") is False

    assert path.read_bytes() == before
    messages = [r.getMessage() for r in caplog.records if r.name == logger_name]
    assert messages
    assert all(
        m.startswith(f"could not remove 'headroom' from {path}") and "No space left" in m
        for m in messages
    )
