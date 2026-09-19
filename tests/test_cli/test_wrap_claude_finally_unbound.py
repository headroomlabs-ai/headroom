"""Regression: `wrap claude`'s finally must not raise UnboundLocalError when the
proxy fails to start before `_wrap_settings_path` is assigned inside the try."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_finally_survives_early_proxy_start_failure(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"restore_called": False, "cleanup_called": False}

    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "HEADROOM_CLAUDE_PROJECT_SETTINGS",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "VERTEX_TARGET_API_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda _port: None)
    monkeypatch.setattr(wrap_mod.signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(wrap_mod, "_push_runtime_env", lambda *_a, **_k: None)
    monkeypatch.setattr(wrap_mod, "_setup_coding_compressor", lambda *_a, **_k: None)
    monkeypatch.setattr(wrap_mod, "_print_telemetry_notice", lambda: None)
    monkeypatch.setattr(wrap_mod, "_write_claude_wrap_base_url", lambda *_a, **_k: None)

    def _fake_make_cleanup(_holder, _port):
        def _cleanup() -> None:
            state["cleanup_called"] = True

        return _cleanup

    monkeypatch.setattr(wrap_mod, "_make_cleanup", _fake_make_cleanup)

    def _fake_restore(*_a, **_k) -> None:
        state["restore_called"] = True

    monkeypatch.setattr(wrap_mod, "_restore_claude_wrap_base_url", _fake_restore)

    def _boom(*_a, **_k):
        raise RuntimeError("proxy failed to start")

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", _boom)

    result = runner.invoke(
        main,
        ["wrap", "claude", "--no-mcp", "--no-tokensave", "--no-serena"],
    )

    # The finally must complete without masking the real failure, skip restoring settings that this run never wrote, and still clean up.
    assert not isinstance(result.exception, UnboundLocalError), result.output
    assert state["restore_called"] is False
    assert state["cleanup_called"] is True
