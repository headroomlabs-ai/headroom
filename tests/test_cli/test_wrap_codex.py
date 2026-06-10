"""Tests for `headroom wrap codex` and `headroom unwrap codex`.

These exercise the Codex-specific ``config.toml`` injection and restoration
helpers that route Codex through the Headroom proxy.  They are deliberately
end-to-end-ish: the unit tests call the helpers directly against a temp
``$HOME``, and the integration tests invoke the real Click commands the same
way a user would from the shell.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


def _set_test_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("CODEX_HOME", raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Unit tests: helpers operating on ~/.codex/config.toml
# ---------------------------------------------------------------------------


class TestStripCodexHeadroomBlocks:
    """Tests for the regex-based cleanup helper."""

    def test_empty_content_returns_empty(self) -> None:
        assert wrap_mod._strip_codex_headroom_blocks("") == ""

    def test_returns_content_unchanged_when_no_markers(self) -> None:
        original = '[profiles.default]\nmodel = "gpt-4o"\n'
        cleaned = wrap_mod._strip_codex_headroom_blocks(original)
        # Trailing whitespace normalization only — semantic content preserved.
        assert 'model = "gpt-4o"' in cleaned
        assert "[profiles.default]" in cleaned

    def test_removes_complete_headroom_block(self) -> None:
        wrapped = (
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n'
            "\n"
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n"
        )
        assert wrap_mod._strip_codex_headroom_blocks(wrapped) == ""

    def test_preserves_user_content_around_block(self) -> None:
        user_pre = '[profiles.default]\nmodel = "gpt-4o"\n'
        user_post = '[mcp_servers.foo]\ncommand = "echo"\n'
        wrapped = (
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n" + user_pre + "\n"
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n" + user_post
        )
        cleaned = wrap_mod._strip_codex_headroom_blocks(wrapped)
        assert wrap_mod._CODEX_TOP_LEVEL_MARKER not in cleaned
        assert wrap_mod._CODEX_END_MARKER not in cleaned
        assert 'model = "gpt-4o"' in cleaned
        assert "[mcp_servers.foo]" in cleaned

    def test_removes_stray_top_level_model_provider_line(self) -> None:
        # Old wrap versions left `model_provider = "headroom"` outside markers.
        content = 'foo = 1\nmodel_provider = "headroom"\nbar = 2\n'
        cleaned = wrap_mod._strip_codex_headroom_blocks(content, remove_mcp=True)
        assert 'model_provider = "headroom"' not in cleaned
        assert "foo = 1" in cleaned
        assert "bar = 2" in cleaned

    def test_removes_codex_mcp_blocks(self) -> None:
        content = (
            '[profiles.default]\nmodel = "gpt-4o"\n\n'
            f"{wrap_mod._CODEX_MCP_MARKER}\n"
            "[mcp_servers.headroom]\n"
            'command = "headroom"\n'
            f"{wrap_mod._CODEX_MCP_END}\n\n"
            "# --- Headroom MCP server: serena ---\n"
            "[mcp_servers.serena]\n"
            'command = "uvx"\n'
            "# --- end Headroom MCP server: serena ---\n\n"
            f"{wrap_mod._MEMORY_MCP_MARKER}\n"
            "[mcp_servers.headroom_memory]\n"
            'command = "python"\n'
            f"{wrap_mod._MEMORY_MCP_END}\n"
        )

        cleaned = wrap_mod._strip_codex_headroom_blocks(content, remove_mcp=True)

        assert "[mcp_servers.headroom]" not in cleaned
        assert "[mcp_servers.serena]" not in cleaned
        assert "[mcp_servers.headroom_memory]" not in cleaned
        assert 'model = "gpt-4o"' in cleaned


class TestSnapshotCodexConfig:
    """Tests for ``_snapshot_codex_config_if_unwrapped``."""

    def test_creates_backup_on_first_call(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text('model = "gpt-4o"\n')

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        assert backup_file.exists()
        assert backup_file.read_text() == 'model = "gpt-4o"\n'

    def test_does_not_overwrite_existing_backup(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text("second-wrap content\n")
        backup_file.write_text("original-pre-wrap content\n")

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        # Backup must still contain the *original* pre-wrap content.
        assert backup_file.read_text() == "original-pre-wrap content\n"

    def test_no_backup_when_config_missing(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        assert not backup_file.exists()

    def test_no_backup_when_config_already_wrapped(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text(
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n"
        )

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        # Pre-wrap snapshot must never snapshot an already-wrapped file.
        assert not backup_file.exists()


class TestInjectAndRestoreRoundTrip:
    """End-to-end wrap → unwrap cycle operating directly on a temp $HOME."""

    def test_wrap_unwrap_restores_empty_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_file = tmp_path / ".codex" / "config.toml"

        wrap_mod._inject_codex_provider_config(8787)
        assert config_file.exists()
        assert 'model_provider = "headroom"' in config_file.read_text()

        status, _ = wrap_mod._restore_codex_provider_config()
        # No prior config existed → the injected file is fully removed.
        assert status == "removed"
        assert not config_file.exists()
        assert not (tmp_path / ".codex" / "config.toml.headroom-backup").exists()

    def test_wrap_unwrap_respects_codex_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        codex_home = tmp_path / "custom-codex-home"
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        config_file = codex_home / "config.toml"

        wrap_mod._inject_codex_provider_config(8787)
        assert config_file.exists()
        assert 'model_provider = "headroom"' in config_file.read_text()
        assert not (tmp_path / ".codex" / "config.toml").exists()

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "removed"
        assert not config_file.exists()

    def test_wrap_unwrap_restores_prior_model_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = (
            'model_provider = "openai"\n'
            "\n"
            "[model_providers.openai]\n"
            'name = "OpenAI"\n'
            'base_url = "https://api.openai.com/v1"\n'
        )
        config_file.write_text(original)

        wrap_mod._inject_codex_provider_config(8787)
        wrapped = config_file.read_text()
        assert 'model_provider = "headroom"' in wrapped
        assert "[model_providers.headroom]" in wrapped

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "restored"
        assert config_file.read_text() == original
        assert not (config_dir / "config.toml.headroom-backup").exists()

    def test_wrap_is_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = '[profiles.default]\nmodel = "gpt-4o"\n'
        config_file.write_text(original)

        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(9999)  # port change

        content = config_file.read_text()
        # Exactly two Headroom blocks — a top-level-key block and the
        # provider-table block.  Re-wrapping must not duplicate them.
        assert content.count(wrap_mod._CODEX_TOP_LEVEL_MARKER) == 2
        assert content.count(wrap_mod._CODEX_END_MARKER) == 2
        # Latest port is honoured in both keys.
        assert 'base_url = "http://127.0.0.1:9999/v1"' in content
        assert 'openai_base_url = "http://127.0.0.1:9999/v1"' in content
        assert 'base_url = "http://127.0.0.1:8787/v1"' not in content
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' not in content
        # User's original content is preserved.
        assert 'model = "gpt-4o"' in content

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "restored"
        assert config_file.read_text() == original

    def test_unwrap_is_noop_when_never_wrapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "noop"

    def test_unwrap_cleans_block_without_backup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Handles crash-case where wrap injected but backup was wiped."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        user_content = '[profiles.default]\nmodel = "gpt-4o"\n'
        config_file.write_text(
            user_content + f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n\n'
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n"
        )

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "cleaned"
        cleaned = config_file.read_text()
        assert wrap_mod._CODEX_TOP_LEVEL_MARKER not in cleaned
        assert wrap_mod._CODEX_END_MARKER not in cleaned
        assert 'model_provider = "headroom"' not in cleaned
        assert 'model = "gpt-4o"' in cleaned

    def test_unwrap_without_backup_removes_provider_and_mcp_blocks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            '[profiles.default]\nmodel = "gpt-4o"\n\n'
            f"{wrap_mod._CODEX_TOP_LEVEL_MARKER}\n"
            'model_provider = "headroom"\n'
            f"{wrap_mod._CODEX_END_MARKER}\n\n"
            f"{wrap_mod._CODEX_MCP_MARKER}\n"
            "[mcp_servers.headroom]\n"
            'command = "headroom"\n'
            f"{wrap_mod._CODEX_MCP_END}\n\n"
            f"{wrap_mod._MEMORY_MCP_MARKER}\n"
            "[mcp_servers.headroom_memory]\n"
            'command = "python"\n'
            f"{wrap_mod._MEMORY_MCP_END}\n"
        )

        status, _ = wrap_mod._restore_codex_provider_config()

        assert status == "cleaned"
        cleaned = config_file.read_text()
        assert 'model = "gpt-4o"' in cleaned
        assert "headroom" not in cleaned

    def test_unwrap_handles_malformed_prior_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unwrap preserves backup content verbatim — TOML validity isn't required."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        malformed = 'this is not valid toml ][ "" \x00\n'
        config_file.write_text(malformed)

        wrap_mod._inject_codex_provider_config(8787)
        status, _ = wrap_mod._restore_codex_provider_config()

        assert status == "restored"
        assert config_file.read_text() == malformed


# ---------------------------------------------------------------------------
# Subscription routing: openai_base_url intercepts ChatGPT plan traffic
# ---------------------------------------------------------------------------


class TestSubscriptionRouting:
    """Codex subscription (ChatGPT plan) bypasses OPENAI_BASE_URL and the
    custom model_provider; it uses the built-in ``openai`` provider whose
    base_url defaults to ``https://chatgpt.com/backend-api/codex``.
    Setting ``openai_base_url`` overrides that default for all auth modes."""

    def test_inject_writes_openai_base_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)

        wrap_mod._inject_codex_provider_config(8787)

        content = (tmp_path / ".codex" / "config.toml").read_text()
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in content

    def test_openai_base_url_port_updates_on_rewrap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)

        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(9999)

        content = (tmp_path / ".codex" / "config.toml").read_text()
        assert 'openai_base_url = "http://127.0.0.1:9999/v1"' in content
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' not in content

    def test_openai_base_url_removed_on_unwrap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = '[profiles.default]\nmodel = "gpt-4o"\n'
        config_file.write_text(original)

        wrap_mod._inject_codex_provider_config(8787)
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text()

        wrap_mod._restore_codex_provider_config()
        assert config_file.read_text() == original

    def test_strip_cleans_orphaned_openai_base_url(self) -> None:
        """Safety net: orphaned openai_base_url lines are cleaned up."""
        content = (
            '[profiles.default]\nmodel = "gpt-4o"\nopenai_base_url = "http://127.0.0.1:8787/v1"\n'
        )
        cleaned = wrap_mod._strip_codex_headroom_blocks(content)
        assert "openai_base_url" not in cleaned
        assert 'model = "gpt-4o"' in cleaned

    def test_no_env_key_in_injected_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """env_key must be absent so Codex doesn't require OPENAI_API_KEY.

        Codex treats env_key as a hard requirement — if the env var is missing
        it throws "Missing environment variable" at startup.  Subscription
        (ChatGPT Plus) users don't have OPENAI_API_KEY set, so injecting
        env_key breaks them (issue #393).
        """
        _set_test_home(monkeypatch, tmp_path)

        wrap_mod._inject_codex_provider_config(8787)

        content = (tmp_path / ".codex" / "config.toml").read_text()
        assert "env_key" not in content


# ---------------------------------------------------------------------------
# Integration tests: full `headroom wrap codex` / `headroom unwrap codex`
# ---------------------------------------------------------------------------


def test_wrap_codex_prepare_only_creates_backup_and_config(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    original = 'model_provider = "openai"\n'
    config_file.write_text(original)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    assert 'model_provider = "headroom"' in config_file.read_text()
    backup = tmp_path / ".codex" / "config.toml.headroom-backup"
    assert backup.exists()
    assert backup.read_text() == original


def test_wrap_codex_prepare_only_respects_codex_home(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        result = runner.invoke(
            main,
            ["wrap", "codex", "--prepare-only", "--no-serena", "--port", "8787"],
        )

    assert result.exit_code == 0, result.output
    config_file = codex_home / "config.toml"
    assert config_file.exists()
    content = config_file.read_text()
    assert 'model_provider = "headroom"' in content
    assert "[mcp_servers.headroom]" in content
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_unwrap_codex_without_codex_home_warns_on_ambiguous_noop(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        wrap_result = runner.invoke(
            main,
            [
                "wrap",
                "codex",
                "--prepare-only",
                "--no-mcp",
                "--no-serena",
                "--port",
                "8787",
            ],
        )

    assert wrap_result.exit_code == 0, wrap_result.output
    config_file = codex_home / "config.toml"
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text()

    monkeypatch.delenv("CODEX_HOME", raising=False)
    unwrap_result = runner.invoke(main, ["unwrap", "codex", "--no-stop-proxy"])

    assert unwrap_result.exit_code == 0, unwrap_result.output
    assert "Warning: found no Headroom wrap markers in the default Codex config" in (
        unwrap_result.output
    )
    assert "If you wrapped Codex with CODEX_HOME" in unwrap_result.output
    assert "CODEX_HOME=/path/to/codex-home headroom unwrap codex" in unwrap_result.output
    assert "Nothing to undo" in unwrap_result.output
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text()


def test_start_proxy_uses_separate_session_for_signal_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Proxy child should not receive Ctrl-C intended for the wrapped CLI."""
    popen_kwargs: dict[str, object] = {}

    class FakeProc:
        returncode = None

        def poll(self) -> None:
            return None

    def fake_popen(*args: object, **kwargs: object) -> FakeProc:
        popen_kwargs.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

    proc = wrap_mod._start_proxy(8787, agent_type="codex")

    assert isinstance(proc, FakeProc)
    assert popen_kwargs["start_new_session"] == (wrap_mod.os.name == "posix")


def test_launch_tool_ignores_sigint_in_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C should be handled by the child CLI, not kill the proxy from wrapper."""
    signal_handlers: dict[object, object] = {}

    class FakeCompleted:
        returncode = 0

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        wrap_mod.signal, "signal", lambda sig, fn: signal_handlers.setdefault(sig, fn)
    )
    monkeypatch.setattr(wrap_mod.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    with pytest.raises(SystemExit) as exc:
        wrap_mod._launch_tool(
            binary="codex",
            args=(),
            env={},
            port=8787,
            no_proxy=True,
            tool_label="CODEX",
            env_vars_display=[],
        )

    assert exc.value.code == 0
    assert signal_handlers[wrap_mod.signal.SIGINT] is wrap_mod._ignore_child_sigint


def test_wrap_codex_prepare_only_updates_stale_mcp_proxy_url(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text(
        "# --- Headroom MCP server ---\n"
        "[mcp_servers.headroom]\n"
        'command = "headroom"\n'
        'args = ["mcp", "serve"]\n'
        "\n"
        "[mcp_servers.headroom.env]\n"
        'HEADROOM_PROXY_URL = "http://127.0.0.1:9000"\n'
        "# --- end Headroom MCP server ---\n"
    )

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    content = config_file.read_text()
    assert "[mcp_servers.headroom]" in content
    assert 'command = "headroom"' in content
    assert 'args = ["mcp", "serve"]' in content
    assert "http://127.0.0.1:9000" not in content


def test_wrap_codex_prepare_only_registers_serena_when_uvx_exists(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)

    def fake_which(cmd: str) -> str | None:
        if cmd == "uvx":
            return "/usr/local/bin/uvx"
        return None

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        with patch("headroom.cli.wrap.shutil.which", side_effect=fake_which):
            result = runner.invoke(main, ["wrap", "codex", "--prepare-only"])

    assert result.exit_code == 0, result.output
    content = config_file.read_text()
    assert "[mcp_servers.serena]" in content
    assert 'command = "uvx"' in content
    assert '"--context", "codex"' in content


def test_wrap_codex_prepare_only_no_serena_skips_serena(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--no-serena"])

    assert result.exit_code == 0, result.output
    assert "[mcp_servers.serena]" not in config_file.read_text()


def test_unwrap_codex_restores_prior_config_end_to_end(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bug report, reproduced: wrap → unwrap must round-trip cleanly."""
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    original = (
        "[profiles.default]\n"
        'model = "gpt-4o"\n'
        "\n"
        "[model_providers.openai]\n"
        'base_url = "https://api.openai.com/v1"\n'
    )
    config_file.write_text(original)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        wrap_result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])
    assert wrap_result.exit_code == 0, wrap_result.output
    assert 'model_provider = "headroom"' in config_file.read_text()

    stopped: list[int] = []

    with patch(
        "headroom.cli.wrap._stop_local_proxy_for_unwrap",
        side_effect=lambda port: stopped.append(port) or "stopped",
    ):
        unwrap_result = runner.invoke(main, ["unwrap", "codex", "--port", "9999"])
    assert unwrap_result.exit_code == 0, unwrap_result.output

    # Config must be byte-for-byte what the user had before wrap, and the
    # injected block must be gone — no more "Missing OPENAI_API_KEY" when the
    # proxy is stopped.
    assert config_file.read_text() == original
    assert 'model_provider = "headroom"' not in config_file.read_text()
    assert not (tmp_path / ".codex" / "config.toml.headroom-backup").exists()
    assert stopped == [9999]
    assert "Stopped local Headroom proxy on port 9999" in unwrap_result.output


def test_unwrap_codex_no_stop_proxy_leaves_proxy_alone(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "explicit-codex-home"))

    with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy:
        result = runner.invoke(main, ["unwrap", "codex", "--no-stop-proxy"])

    assert result.exit_code == 0, result.output
    stop_proxy.assert_not_called()


def test_stop_local_proxy_for_unwrap_kills_identified_headroom_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[tuple[int, int]] = []

    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda port: {"pid": "12345"})
    monkeypatch.setattr(
        wrap_mod,
        "_kill_proxy_by_pid",
        lambda pid, port: killed.append((pid, port)) or True,
    )

    assert wrap_mod._stop_local_proxy_for_unwrap(8787) == "stopped"
    assert killed == [(12345, 8787)]


def test_stop_local_proxy_for_unwrap_refuses_unidentified_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda port: None)

    with patch("headroom.cli.wrap._kill_proxy_by_pid") as kill_proxy:
        assert wrap_mod._stop_local_proxy_for_unwrap(8787) == "unidentified"

    kill_proxy.assert_not_called()


def test_unwrap_codex_is_safe_noop_with_explicit_codex_home(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "explicit-codex-home"))

    result = runner.invoke(main, ["unwrap", "codex"])
    assert result.exit_code == 0, result.output
    assert "Nothing to undo" in result.output
    assert "Warning:" not in result.output
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_unwrap_codex_removes_headroom_only_config_file(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        wrap_result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])
    assert wrap_result.exit_code == 0, wrap_result.output

    config_file = tmp_path / ".codex" / "config.toml"
    assert config_file.exists()

    unwrap_result = runner.invoke(main, ["unwrap", "codex"])
    assert unwrap_result.exit_code == 0, unwrap_result.output
    assert not config_file.exists()


def test_unwrap_codex_preserves_unrelated_sections(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    # A config with an MCP server the user configured by hand.
    original = '[mcp_servers.local_thing]\ncommand = "/usr/local/bin/thing"\nargs = ["--serve"]\n'
    config_file.write_text(original)

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    result = runner.invoke(main, ["unwrap", "codex"])
    assert result.exit_code == 0, result.output
    restored = config_file.read_text()
    assert restored == original
