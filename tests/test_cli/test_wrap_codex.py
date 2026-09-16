"""Tests for `headroom wrap codex` and `headroom unwrap codex`.

These exercise the Codex-specific ``config.toml`` injection and restoration
helpers that route Codex through the Headroom proxy.  They are deliberately
end-to-end-ish: the unit tests call the helpers directly against a temp
``$HOME``, and the integration tests invoke the real Click commands the same
way a user would from the shell.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised in the Python 3.10 test job
    import tomli as tomllib

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main
from headroom.mcp_registry.install import build_headroom_spec


def _set_test_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("CODEX_HOME", raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


_PROXY_DEP_TESTS = frozenset(
    {
        "test_wrap_codex_aborts_before_mutating_config_when_proxy_deps_missing",
        "test_wrap_codex_skips_proxy_dependency_check_with_no_proxy",
    }
)


@pytest.fixture(autouse=True)
def _skip_wrap_proxy_dependency_gate_unless_exercised(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wrap-codex integration tests run in the base CI env without [proxy] extras."""
    if request.node.name in _PROXY_DEP_TESTS:
        return
    monkeypatch.setattr("headroom.cli.wrap.ensure_proxy_dependencies", lambda: None)


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

    def test_preserves_named_mcp_blocks_when_remove_named_mcp_false(self) -> None:
        content = (
            "# --- Headroom MCP server: serena ---\n"
            "[mcp_servers.serena]\n"
            'command = "uvx"\n'
            "# --- end Headroom MCP server: serena ---\n\n"
            f"{wrap_mod._MEMORY_MCP_MARKER}\n"
            "[mcp_servers.headroom_memory]\n"
            'command = "python"\n'
            f"{wrap_mod._MEMORY_MCP_END}\n"
        )

        cleaned = wrap_mod._strip_codex_headroom_blocks(
            content, remove_mcp=True, remove_named_mcp=False
        )

        assert "[mcp_servers.serena]" in cleaned
        assert "[mcp_servers.headroom_memory]" not in cleaned


class TestSnapshotCodexConfig:
    """Tests for ``_snapshot_codex_config_if_unwrapped``."""

    def test_creates_backup_on_first_call(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text('model = "gpt-4o"\n', encoding="utf-8")

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        assert backup_file.exists()
        assert backup_file.read_text(encoding="utf-8") == 'model = "gpt-4o"\n'

    def test_does_not_overwrite_existing_backup(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text("second-wrap content\n", encoding="utf-8")
        backup_file.write_text("original-pre-wrap content\n", encoding="utf-8")

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        # Backup must still contain the *original* pre-wrap content.
        assert backup_file.read_text(encoding="utf-8") == "original-pre-wrap content\n"

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
            f"{wrap_mod._CODEX_END_MARKER}\n",
            encoding="utf-8",
        )

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        # Pre-wrap snapshot must never snapshot an already-wrapped file.
        assert not backup_file.exists()

    def test_no_backup_when_config_already_contains_memory_mcp_block(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        config_file.write_text(
            f"{wrap_mod._MEMORY_MCP_MARKER}\n"
            "[mcp_servers.headroom_memory]\n"
            'command = "python"\n'
            'args = ["-m", "headroom.memory.mcp_server", "--user", "codex-user"]\n'
            f"{wrap_mod._MEMORY_MCP_END}\n"
        )

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        assert not backup_file.exists()

    def test_backup_when_config_contains_named_mcp_marker(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        backup_file = tmp_path / "config.toml.headroom-backup"
        original = (
            "# --- Headroom MCP server: headroom ---\n"
            "[mcp_servers.headroom]\n"
            'command = "headroom"\n'
            "# --- end Headroom MCP server: headroom ---\n"
        )
        config_file.write_text(original)

        wrap_mod._snapshot_codex_config_if_unwrapped(config_file, backup_file)

        assert backup_file.exists()
        assert backup_file.read_text() == original


class TestCodexMemoryMcpConfig:
    """Tests for the persisted Codex memory MCP block."""

    def test_inject_omits_db_and_replaces_existing_memory_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_file = tmp_path / ".codex" / "config.toml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(
            '[profiles.default]\nmodel = "gpt-4o"\n\n'
            f"{wrap_mod._MEMORY_MCP_MARKER}\n"
            "[mcp_servers.headroom_memory]\n"
            'command = "python"\n'
            'args = ["-m", "headroom.memory.mcp_server", "--db", "/tmp/project-a/.headroom/memory.db", "--user", "old-user"]\n'
            f"{wrap_mod._MEMORY_MCP_END}\n"
        )

        wrap_mod._inject_memory_mcp_config("codex-user")

        content = config_file.read_text()
        assert content.count(wrap_mod._MEMORY_MCP_MARKER) == 1
        assert "[mcp_servers.headroom_memory]" in content
        assert '"--user", "codex-user"' in content
        assert "--db" not in content
        assert 'model = "gpt-4o"' in content


class TestInjectAndRestoreRoundTrip:
    """End-to-end wrap → unwrap cycle operating directly on a temp $HOME."""

    def test_wrap_unwrap_restores_empty_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_file = tmp_path / ".codex" / "config.toml"

        wrap_mod._inject_codex_provider_config(8787)
        assert config_file.exists()
        assert 'model_provider = "headroom"' in config_file.read_text(encoding="utf-8")

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
        assert 'model_provider = "headroom"' in config_file.read_text(encoding="utf-8")
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
        config_file.write_text(original, encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)
        wrapped = config_file.read_text(encoding="utf-8")
        assert 'model_provider = "headroom"' in wrapped
        assert "[model_providers.headroom]" in wrapped

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "restored"
        assert config_file.read_text(encoding="utf-8") == original
        assert not (config_dir / "config.toml.headroom-backup").exists()

    def test_wrap_is_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = '[profiles.default]\nmodel = "gpt-4o"\n'
        config_file.write_text(original, encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(9999)  # port change

        content = config_file.read_text(encoding="utf-8")
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
        assert config_file.read_text(encoding="utf-8") == original

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
            f"{wrap_mod._CODEX_END_MARKER}\n",
            encoding="utf-8",
        )

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "cleaned"
        cleaned = config_file.read_text(encoding="utf-8")
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
            f"{wrap_mod._MEMORY_MCP_END}\n",
            encoding="utf-8",
        )

        status, _ = wrap_mod._restore_codex_provider_config()

        assert status == "cleaned"
        cleaned = config_file.read_text(encoding="utf-8")
        assert 'model = "gpt-4o"' in cleaned
        assert "headroom" not in cleaned

    def test_memory_only_wrap_restores_preexisting_named_mcp_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = (
            "# --- Headroom MCP server: headroom ---\n"
            "[mcp_servers.headroom]\n"
            'command = "headroom"\n'
            "# --- end Headroom MCP server: headroom ---\n"
        )
        config_file.write_text(original)

        wrap_mod._inject_memory_mcp_config("codex-user")

        status, _ = wrap_mod._restore_codex_provider_config()

        assert status == "restored"
        assert config_file.read_text() == original

    def test_memory_only_wrap_without_backup_preserves_named_mcp_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        backup_file = config_dir / "config.toml.headroom-backup"
        original = (
            "# --- Headroom MCP server: headroom ---\n"
            "[mcp_servers.headroom]\n"
            'command = "headroom"\n'
            "# --- end Headroom MCP server: headroom ---\n"
        )
        config_file.write_text(original)

        wrap_mod._inject_memory_mcp_config("codex-user")
        backup_file.unlink()

        status, _ = wrap_mod._restore_codex_provider_config()

        assert status == "cleaned"
        assert config_file.read_text() == original

    def test_unwrap_handles_malformed_prior_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Unwrap preserves backup content verbatim — TOML validity isn't required."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        malformed = 'this is not valid toml ][ "" \x00\n'
        config_file.write_text(malformed, encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)
        status, _ = wrap_mod._restore_codex_provider_config()

        assert status == "restored"
        assert config_file.read_text(encoding="utf-8") == malformed

    def test_unwrap_is_safe_when_no_global_agents(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No Codex AGENTS.md → unwrap is a clean no-op, not a crash."""
        _set_test_home(monkeypatch, tmp_path)

        wrap_mod.unwrap_codex.callback(port=8787, no_stop_proxy=True)

        assert not (tmp_path / ".codex" / "AGENTS.md").exists()


# ---------------------------------------------------------------------------
# Thread retag: wrap pulls native threads into the headroom menu, unwrap hands
# them back, so the Codex history list stays whole across the proxy boundary.
# ---------------------------------------------------------------------------


class TestWrapRetagsThreadProviders:
    """``wrap codex`` retags ``openai`` threads to ``headroom`` and back."""

    @staticmethod
    def _seed_threads(db: Path, rows: list[tuple[str, str]]) -> None:
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, model_provider TEXT NOT NULL)")
            conn.executemany("INSERT INTO threads (id, model_provider) VALUES (?, ?)", rows)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _count(db: Path, provider: str) -> int:
        conn = sqlite3.connect(str(db))
        try:
            (n,) = conn.execute(
                "SELECT COUNT(*) FROM threads WHERE model_provider = ?", (provider,)
            ).fetchone()
            return n
        finally:
            conn.close()

    def test_wrap_unwrap_round_trips_thread_providers(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        gui_db = tmp_path / ".codex" / "sqlite" / "state_5.sqlite"
        cli_db = tmp_path / ".codex" / "state_5.sqlite"
        self._seed_threads(gui_db, [("a", "openai"), ("b", "headroom"), ("c", "anthropic")])
        self._seed_threads(cli_db, [("d", "openai")])

        wrap_result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])
        assert wrap_result.exit_code == 0, wrap_result.output
        # Native threads are now visible under the headroom provider menu;
        # third-party providers are left untouched.
        assert self._count(gui_db, "headroom") == 2
        assert self._count(gui_db, "openai") == 0
        assert self._count(gui_db, "anthropic") == 1
        assert self._count(cli_db, "headroom") == 1

        with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="stopped"):
            unwrap_result = runner.invoke(main, ["unwrap", "codex", "--port", "8787"])
        assert unwrap_result.exit_code == 0, unwrap_result.output
        # Back to native so the unproxied Codex menu is whole again.
        assert self._count(gui_db, "openai") == 2
        assert self._count(gui_db, "headroom") == 0
        assert self._count(gui_db, "anthropic") == 1
        assert self._count(cli_db, "openai") == 1


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

        content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in content

    def test_inject_emits_requires_openai_auth_for_chatgpt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        (config_dir / "auth.json").write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)

        assert "requires_openai_auth = true" in (config_dir / "config.toml").read_text(
            encoding="utf-8"
        )

    def test_inject_omits_requires_openai_auth_for_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        (config_dir / "auth.json").write_text('{"auth_mode": "apikey"}', encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)

        assert "requires_openai_auth" not in (config_dir / "config.toml").read_text(
            encoding="utf-8"
        )

    def test_openai_base_url_port_updates_on_rewrap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)

        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(9999)

        content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
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
        config_file.write_text(original, encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text(
            encoding="utf-8"
        )

        wrap_mod._restore_codex_provider_config()
        assert config_file.read_text(encoding="utf-8") == original

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

        content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert "env_key" not in content


# ---------------------------------------------------------------------------
# Custom upstream preservation (#1614): wrap must not silently reroute a
# pre-existing custom [model_providers.*] base_url to api.openai.com.
# ---------------------------------------------------------------------------


class TestDetectCustomCodexUpstreamBaseUrl:
    """Unit tests for the detection helper used by ``_inject_codex_provider_config``."""

    def test_no_config_returns_none(self) -> None:
        assert wrap_mod._detect_custom_codex_upstream_base_url("") is None

    def test_no_custom_provider_returns_none(self) -> None:
        content = (
            'model_provider = "openai"\n\n'
            "[model_providers.openai]\n"
            'base_url = "https://api.openai.com/v1"\n'
        )
        assert wrap_mod._detect_custom_codex_upstream_base_url(content) is None

    def test_sole_candidate_used_without_explicit_selection(self) -> None:
        """Matches the #1614 repro: a custom table with no static top-level pin."""
        content = (
            "[model_providers.freemodel]\n"
            'base_url = "https://api.freemodel.dev"\n'
            'wire_api = "responses"\n'
        )
        assert (
            wrap_mod._detect_custom_codex_upstream_base_url(content) == "https://api.freemodel.dev"
        )

    def test_explicit_top_level_selection_wins(self) -> None:
        content = (
            'model_provider = "freemodel"\n\n'
            "[model_providers.freemodel]\n"
            'base_url = "https://api.freemodel.dev"\n\n'
            "[model_providers.other]\n"
            'base_url = "https://api.other.example"\n'
        )
        assert (
            wrap_mod._detect_custom_codex_upstream_base_url(content) == "https://api.freemodel.dev"
        )

    def test_was_comment_recovers_selection_on_rewrap(self) -> None:
        """After a prior wrap, model_provider reads 'headroom  # was: freemodel'."""
        content = (
            'model_provider = "headroom"  # was: freemodel\n\n'
            "[model_providers.freemodel]\n"
            'base_url = "https://api.freemodel.dev"\n'
        )
        assert (
            wrap_mod._detect_custom_codex_upstream_base_url(content) == "https://api.freemodel.dev"
        )

    def test_ambiguous_multiple_candidates_returns_none(self) -> None:
        content = (
            "[model_providers.freemodel]\n"
            'base_url = "https://api.freemodel.dev"\n\n'
            "[model_providers.other]\n"
            'base_url = "https://api.other.example"\n'
        )
        assert wrap_mod._detect_custom_codex_upstream_base_url(content) is None

    def test_builtin_provider_tables_excluded(self) -> None:
        content = (
            'model_provider = "openai"\n\n'
            "[model_providers.openai]\n"
            'base_url = "https://api.openai.com/v1"\n\n'
            "[model_providers.anthropic]\n"
            'base_url = "https://api.anthropic.com/v1"\n'
        )
        assert wrap_mod._detect_custom_codex_upstream_base_url(content) is None

    def test_own_headroom_table_excluded(self) -> None:
        content = (
            'model_provider = "headroom"\n\n'
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n'
        )
        assert wrap_mod._detect_custom_codex_upstream_base_url(content) is None


class TestInjectPreservesCustomUpstreamBaseUrl:
    """``_inject_codex_provider_config`` must preserve a pre-existing custom
    provider's ``base_url`` instead of silently rerouting to api.openai.com."""

    def test_inject_returns_and_carries_custom_base_url(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            "[model_providers.freemodel]\n"
            'base_url = "https://api.freemodel.dev"\n'
            'wire_api = "responses"\n',
            encoding="utf-8",
        )

        result = wrap_mod._inject_codex_provider_config(8787)

        assert result == "https://api.freemodel.dev"
        content = config_file.read_text(encoding="utf-8")
        parsed = tomllib.loads(content)
        headers = parsed["model_providers"]["headroom"]["env_http_headers"]
        assert (
            headers[wrap_mod._UPSTREAM_BASE_URL_HEADER_NAME] == wrap_mod._UPSTREAM_BASE_URL_ENV_VAR
        )
        # The user's own table is left untouched — only headroom's own is managed.
        assert parsed["model_providers"]["freemodel"]["base_url"] == "https://api.freemodel.dev"

    def test_inject_without_custom_provider_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)

        result = wrap_mod._inject_codex_provider_config(8787)

        assert result is None
        content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert wrap_mod._UPSTREAM_BASE_URL_HEADER_NAME not in content

    def test_preserved_upstream_survives_rewrap_and_port_change(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            'model_provider = "freemodel"\n\n'
            "[model_providers.freemodel]\n"
            'base_url = "https://api.freemodel.dev"\n',
            encoding="utf-8",
        )

        first = wrap_mod._inject_codex_provider_config(8787)
        second = wrap_mod._inject_codex_provider_config(9999)  # port change / re-wrap

        assert first == "https://api.freemodel.dev"
        assert second == "https://api.freemodel.dev"
        content = config_file.read_text(encoding="utf-8")
        parsed = tomllib.loads(content)
        assert parsed["model_providers"]["headroom"]["base_url"] == "http://127.0.0.1:9999/v1"
        headers = parsed["model_providers"]["headroom"]["env_http_headers"]
        assert (
            headers[wrap_mod._UPSTREAM_BASE_URL_HEADER_NAME] == wrap_mod._UPSTREAM_BASE_URL_ENV_VAR
        )


class TestInjectAvoidsDuplicateTopLevelKeys:
    """Wrap must not produce a TOML-validity-breaking duplicate-key error.

    Codex's ``config.toml`` is parsed strictly: two top-level
    ``model_provider = …`` (or two ``openai_base_url = …``) declarations
    cause ``codex`` to refuse to start with
    ``Error loading config.toml: …: …:1: duplicate key``.  The injector
    used to unconditionally prepend a top-level block, breaking any user
    who had already configured their own provider (e.g. ``ccswitch``).
    """

    def test_inject_does_not_create_duplicate_model_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            'model_provider = "ccswitch"\n'
            'openai_base_url = "http://llm-gateway-proxy/v1"\n'
            'model = "azure-gpt-5_5"\n'
            "\n"
            "[model_providers.ccswitch]\n"
            'name = "OpenAI"\n'
            'base_url = "http://llm-gateway-proxy/v1"\n'
            'wire_api = "responses"\n',
            encoding="utf-8",
        )

        wrap_mod._inject_codex_provider_config(8787)

        content = config_file.read_text(encoding="utf-8")
        # The wrapped file must be TOML-parseable — duplicate keys were
        # the failure mode the user reported.
        tomllib.loads(content)
        # No duplicate top-level key for either redirectable key.
        assert content.count("model_provider =") == 1
        assert content.count("openai_base_url =") == 1
        # And the rewritten values are the headroom ones.
        assert 'model_provider = "headroom"' in content
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in content

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t\n"])
    def test_redirect_existing_top_level_keys_noop_on_blank(self, blank: str) -> None:
        # No redirectable keys to rewrite in blank/whitespace content — the
        # helper returns it unchanged so the caller falls back to prepending
        # the marker-delimited top-level block.
        assert wrap_mod._redirect_existing_top_level_keys(blank, 8787) == blank

    def test_inject_preserves_user_value_in_trailing_comment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            'model_provider = "ccswitch"\nopenai_base_url = "http://llm-gateway-proxy/v1"\n',
            encoding="utf-8",
        )

        wrap_mod._inject_codex_provider_config(8787)

        content = config_file.read_text(encoding="utf-8")
        # Original value kept in a comment so the user can recover it.
        # The comment intentionally drops the surrounding quotes — the
        # value is a single TOML string and the comment is human-facing.
        assert "was: ccswitch" in content
        assert "was: http://llm-gateway-proxy/v1" in content

    def test_inject_rewrap_updates_existing_redirected_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Idempotent re-wrap on a config that already has top-level keys."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text('model_provider = "ccswitch"\n', encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)
        wrap_mod._inject_codex_provider_config(9999)  # port change

        content = config_file.read_text(encoding="utf-8")
        tomllib.loads(content)
        assert content.count("model_provider =") == 1
        assert 'model_provider = "headroom"' in content
        # Port updated in the openai_base_url we injected.
        assert 'openai_base_url = "http://127.0.0.1:9999/v1"' in content
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' not in content

    def test_inject_empty_file_still_uses_marker_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No existing top-level keys → fall back to the marker-delimited block."""
        _set_test_home(monkeypatch, tmp_path)
        wrap_mod._inject_codex_provider_config(8787)

        content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert wrap_mod._CODEX_TOP_LEVEL_MARKER in content
        assert 'model_provider = "headroom"' in content
        assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in content
        assert "[model_providers.headroom]" in content

    def test_inject_replaces_existing_headroom_provider_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Existing headroom provider table must not create duplicate TOML keys."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        config_file.write_text(
            "[model_providers.headroom]\n"
            'name = "Existing custom headroom"\n'
            'base_url = "http://example.invalid/v1"\n'
            "supports_websockets = true\n"
            'env_http_headers = { "X-Headroom-Project" = "HEADROOM_PROJECT" }\n'
            "\n"
            "[profiles.default]\n"
            'model = "gpt-5"\n'
        )

        wrap_mod._inject_codex_provider_config(8787)
        content = config_file.read_text()

        tomllib.loads(content)
        assert content.count("[model_providers.headroom]") == 1
        assert content.count("env_http_headers") == 1
        assert 'base_url = "http://127.0.0.1:8787/v1"' in content
        assert 'env_http_headers = { "X-Headroom-Project" = "HEADROOM_PROJECT" }' in content
        assert "[profiles.default]" in content
        assert 'model = "gpt-5"' in content

    def test_unwrap_restores_prior_headroom_provider_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pre-wrap headroom provider table is restored from snapshot."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = (
            "[model_providers.headroom]\n"
            'name = "Existing custom headroom"\n'
            'base_url = "http://example.invalid/v1"\n'
            "supports_websockets = true\n"
            'env_http_headers = { "X-Headroom-Project" = "HEADROOM_PROJECT" }\n'
        )
        config_file.write_text(original)

        wrap_mod._inject_codex_provider_config(8787)
        status, _ = wrap_mod._restore_codex_provider_config()

        assert status == "restored"
        assert config_file.read_text() == original

    def test_unwrap_restores_prior_model_provider_after_rewrite(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The snapshot mechanism must still restore the pre-wrap state byte-for-byte."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = 'model_provider = "ccswitch"\nopenai_base_url = "http://llm-gateway-proxy/v1"\n'
        config_file.write_text(original, encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)

        status, _ = wrap_mod._restore_codex_provider_config()
        assert status == "restored"
        assert config_file.read_text(encoding="utf-8") == original


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
    config_file.write_text(original, encoding="utf-8")

    result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    assert 'model_provider = "headroom"' in config_file.read_text(encoding="utf-8")
    backup = tmp_path / ".codex" / "config.toml.headroom-backup"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original


def test_wrap_codex_aborts_before_mutating_config_when_proxy_deps_missing(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    original = 'model_provider = "openai"\n'
    config_file.write_text(original, encoding="utf-8")

    with patch("headroom.cli.wrap.ensure_proxy_dependencies", side_effect=SystemExit(1)):
        result = runner.invoke(
            main,
            ["wrap", "codex", "--prepare-only", "--no-serena", "--port", "8787"],
        )

    assert result.exit_code == 1, result.output
    assert config_file.read_text(encoding="utf-8") == original
    assert "[mcp_servers.headroom]" not in config_file.read_text(encoding="utf-8")
    assert not (tmp_path / ".codex" / "config.toml.headroom-backup").exists()


def test_wrap_codex_skips_proxy_dependency_check_with_no_proxy(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('model_provider = "openai"\n', encoding="utf-8")

    with patch(
        "headroom.cli.wrap.ensure_proxy_dependencies",
        side_effect=AssertionError("should not run with --no-proxy"),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "codex",
                "--prepare-only",
                "--no-proxy",
                "--no-serena",
                "--port",
                "8787",
            ],
        )

    assert result.exit_code == 0, result.output


def test_wrap_codex_registers_mcp_when_codex_home_does_not_exist_yet(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MCP must register on a machine where Codex was installed but never launched.

    ``CodexRegistrar.detect()`` is just ``~/.codex`` being a directory, and that
    directory used to be created as a side effect of writing the rtk guidance
    into ``$CODEX_HOME/AGENTS.md``. Once the CLI context tools were removed,
    nothing created it, so detect() said "Codex not detected" and Headroom
    silently skipped MCP registration — leaving every compression marker the
    proxy emits unresolvable, with no error shown.

    Every other codex test pre-creates ``~/.codex``, which is exactly why none of
    them caught it; this one deliberately does not.
    """
    _set_test_home(monkeypatch, tmp_path)
    codex_dir = tmp_path / ".codex"
    assert not codex_dir.exists()  # the whole point

    result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    config = codex_dir / "config.toml"
    assert config.exists(), "wrap codex did not persist config in the durable Codex home"
    assert "[mcp_servers.headroom]" in config.read_text(encoding="utf-8")
    assert "not detected" not in result.output


def test_wrap_codex_prepare_only_respects_codex_home(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = runner.invoke(
        main,
        ["wrap", "codex", "--prepare-only", "--no-serena", "--port", "8787"],
    )

    assert result.exit_code == 0, result.output
    config_file = codex_home / "config.toml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert 'model_provider = "headroom"' in content
    assert "[mcp_servers.headroom]" in content
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_wrap_codex_launch_uses_durable_codex_home(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    config_file = codex_home / "config.toml"
    auth_file = codex_home / "auth.json"
    original_config = '[profiles.default]\nmodel = "gpt-4o"\n'
    original_auth = '{"auth_mode": "apikey"}'
    config_file.write_text(original_config, encoding="utf-8")
    auth_file.write_text(original_auth, encoding="utf-8")

    launch_env: dict[str, str] = {}
    rollout = codex_home / "sessions" / "2026" / "07" / "14" / "rollout-thread.jsonl"

    def fake_launch(
        *,
        binary: str,
        args: tuple,
        env: dict[str, str],
        port: int,
        no_proxy: bool,
        tool_label: str,
        env_vars_display: list[str],
        **kwargs: object,
    ) -> None:
        del args, port, no_proxy, tool_label, env_vars_display, kwargs
        assert binary == "/fake/codex"
        launch_env.update(env)
        assert Path(env["CODEX_HOME"]) == codex_home
        rollout.parent.mkdir(parents=True)
        rollout.write_text('{"type":"session_meta"}\n', encoding="utf-8")

    with patch(
        "headroom.cli.wrap.shutil.which",
        side_effect=lambda cmd: "/fake/codex" if cmd == "codex" else None,
    ):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "codex",
                    "--port",
                    "8787",
                    "--no-tokensave",
                    "--no-serena",
                ],
            )

    assert result.exit_code == 0, result.output
    assert launch_env["CODEX_HOME"] == str(codex_home)
    assert launch_env["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
    persisted_config = config_file.read_text(encoding="utf-8")
    assert original_config in persisted_config
    assert "[mcp_servers.headroom]" in persisted_config
    assert 'model_provider = "headroom"' not in persisted_config
    assert "[model_providers.headroom]" not in persisted_config
    assert auth_file.read_text(encoding="utf-8") == original_auth
    assert rollout.read_text(encoding="utf-8") == '{"type":"session_meta"}\n'


def test_codex_session_launch_settings_keep_routing_process_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(wrap_mod, "_project_name_from_cwd", lambda: None)
    config_file = codex_home / "config.toml"
    original_config = 'model = "gpt-5"\n'
    config_file.write_text(original_config, encoding="utf-8")

    args, env, display = wrap_mod._codex_session_launch_settings(
        port=9898,
        codex_args=("exec", "hello"),
        environ={"CODEX_HOME": str(codex_home)},
    )

    assert args == (
        "--config",
        'openai_base_url="http://127.0.0.1:9898/v1"',
        "exec",
        "hello",
    )
    assert env["CODEX_HOME"] == str(codex_home)
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9898/v1"
    assert display == ["OPENAI_BASE_URL=http://127.0.0.1:9898/v1"]
    assert config_file.read_text(encoding="utf-8") == original_config


def test_codex_session_launch_settings_preserve_custom_provider_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(wrap_mod, "_project_name_from_cwd", lambda: None)
    config_file = codex_home / "config.toml"
    original_config = (
        '[profiles.work]\nmodel_provider = "company"\n\n'
        '[model_providers.company]\nbase_url = "https://api.example.test/v1"\n'
    )
    config_file.write_text(original_config, encoding="utf-8")

    args, env, _ = wrap_mod._codex_session_launch_settings(
        port=9898,
        codex_args=("--profile", "work"),
        environ={"CODEX_HOME": str(codex_home)},
    )

    assert "model_provider=headroom" not in " ".join(args)
    # Bare dotted keys — Codex (0.144.x) silently ignores quoted segments (#2358).
    assert 'model_providers.company.base_url="http://127.0.0.1:9898/v1"' in args
    assert "model_providers.company.supports_websockets=true" in args
    assert (
        "model_providers.company.env_http_headers.X-Headroom-Base-Url"
        '="HEADROOM_CODEX_UPSTREAM_BASE_URL"'
    ) in args
    assert env[wrap_mod._UPSTREAM_BASE_URL_ENV_VAR] == "https://api.example.test/v1"
    assert config_file.read_text(encoding="utf-8") == original_config


def test_codex_dotted_key_emits_bare_segments_when_safe() -> None:
    """#2358: quoted segments are silently ignored by Codex's --config parser."""
    assert (
        wrap_mod._codex_dotted_key("model_providers", "litellm_prod", "base_url")
        == "model_providers.litellm_prod.base_url"
    )
    # Hyphens are valid in bare keys (header names under env_http_headers).
    assert (
        wrap_mod._codex_dotted_key("env_http_headers", "X-Headroom-Base-Url")
        == "env_http_headers.X-Headroom-Base-Url"
    )


def test_codex_dotted_key_quotes_only_unsafe_segments() -> None:
    # A provider name that would corrupt the dotted path if emitted bare keeps
    # its quotes; every safe neighbor stays bare.
    assert (
        wrap_mod._codex_dotted_key("model_providers", "my.provider", "base_url")
        == 'model_providers."my.provider".base_url'
    )


def test_wrap_codex_rejects_custom_provider_without_upstream_base_url(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    (codex_home / "config.toml").write_text(
        'model_provider = "company"\n[model_providers.company]\nname = "Company"\n',
        encoding="utf-8",
    )

    def fake_launch(**kwargs: object) -> None:
        configure_launch = kwargs["configure_launch"]
        assert callable(configure_launch)
        configure_launch(
            8787,
            kwargs["args"],
            kwargs["env"],
            kwargs["env_vars_display"],
        )

    with patch(
        "headroom.cli.wrap.shutil.which",
        side_effect=lambda cmd: "/fake/codex" if cmd == "codex" else None,
    ):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "codex",
                    "--port",
                    "8787",
                    "--no-mcp",
                    "--no-tokensave",
                    "--no-serena",
                ],
            )

    assert result.exit_code != 0
    assert "custom provider 'company' has no upstream base_url" in result.output


def test_wrap_codex_routes_model_provider_selected_by_config_argument(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(wrap_mod, "_project_name_from_cwd", lambda: None)
    (codex_home / "config.toml").write_text(
        '[model_providers.company]\nbase_url = "https://api.example.test/v1"\n',
        encoding="utf-8",
    )
    configured_env: dict[str, str] = {}

    def fake_launch(**kwargs: object) -> None:
        configure_launch = kwargs["configure_launch"]
        assert callable(configure_launch)
        _, env, _ = configure_launch(
            8787,
            kwargs["args"],
            kwargs["env"],
            kwargs["env_vars_display"],
        )
        configured_env.update(env)

    with patch(
        "headroom.cli.wrap.shutil.which",
        side_effect=lambda cmd: "/fake/codex" if cmd == "codex" else None,
    ):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "codex",
                    "--no-mcp",
                    "--no-tokensave",
                    "--no-serena",
                    "--",
                    "--config",
                    'model_provider="company"',
                ],
            )

    assert result.exit_code == 0, result.output
    assert configured_env[wrap_mod._UPSTREAM_BASE_URL_ENV_VAR] == ("https://api.example.test/v1")


def test_unwrap_codex_without_codex_home_warns_on_ambiguous_noop(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    codex_home = tmp_path / "custom-codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

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
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text(encoding="utf-8")

    monkeypatch.delenv("CODEX_HOME", raising=False)
    unwrap_result = runner.invoke(main, ["unwrap", "codex", "--no-stop-proxy"])

    assert unwrap_result.exit_code == 0, unwrap_result.output
    assert "Warning: found no Headroom wrap markers in the default Codex config" in (
        unwrap_result.output
    )
    assert "If you wrapped Codex with CODEX_HOME" in unwrap_result.output
    assert "CODEX_HOME=/path/to/codex-home headroom unwrap codex" in unwrap_result.output
    assert "Nothing to undo" in unwrap_result.output
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text(encoding="utf-8")


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

    monkeypatch.setattr(wrap_mod, "_get_log_path", lambda port=None: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

    proc = wrap_mod._start_proxy(8787, agent_type="codex")

    assert isinstance(proc, FakeProc)
    assert popen_kwargs["start_new_session"] == (wrap_mod.os.name == "posix")


@pytest.mark.parametrize("agent_type", ["claude", "codex", "cursor"])
def test_start_proxy_does_not_apply_agent_90_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, agent_type: str
) -> None:
    """Wrapped coding agents keep agent-savings opt-in by default."""
    # Clean baseline: the proxy's out-of-box coding profile seeds these into the
    # process env at startup (``seed_proxy_env_defaults``), which another test in
    # the shard can leave behind in ``os.environ``. This test is about what the
    # WRAPPER adds, so start from an unset env rather than inheriting pollution.
    for _var in (
        "HEADROOM_SAVINGS_PROFILE",
        "HEADROOM_TARGET_RATIO",
        "HEADROOM_MAX_ITEMS",
        "HEADROOM_SMART_CRUSHER_COMPACTION",
    ):
        monkeypatch.delenv(_var, raising=False)
    popen_kwargs: dict[str, object] = {}

    class FakeProc:
        returncode = None

        def poll(self) -> None:
            return None

    def fake_popen(*args: object, **kwargs: object) -> FakeProc:
        popen_kwargs.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(wrap_mod, "_get_log_path", lambda port=None: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

    wrap_mod._start_proxy(8787, agent_type=agent_type)

    env = popen_kwargs["env"]
    assert isinstance(env, dict)
    assert "HEADROOM_SAVINGS_PROFILE" not in env
    assert "HEADROOM_TARGET_RATIO" not in env
    assert "HEADROOM_MAX_ITEMS" not in env
    assert "HEADROOM_SMART_CRUSHER_COMPACTION" not in env


def test_start_proxy_preserves_explicit_savings_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """User-provided savings env vars should override wrapper defaults."""
    popen_kwargs: dict[str, object] = {}

    class FakeProc:
        returncode = None

        def poll(self) -> None:
            return None

    def fake_popen(*args: object, **kwargs: object) -> FakeProc:
        popen_kwargs.update(kwargs)
        return FakeProc()

    monkeypatch.setenv("HEADROOM_TARGET_RATIO", "0.20")
    monkeypatch.setenv("HEADROOM_MAX_ITEMS", "12")
    monkeypatch.setattr(wrap_mod, "_get_log_path", lambda port=None: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

    wrap_mod._start_proxy(8787, agent_type="codex")

    env = popen_kwargs["env"]
    assert isinstance(env, dict)
    assert env["HEADROOM_TARGET_RATIO"] == "0.20"
    assert env["HEADROOM_MAX_ITEMS"] == "12"


def test_launch_tool_ignores_sigint_in_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C should be handled by the child CLI, not kill the proxy from wrapper."""
    signal_handlers: dict[object, object] = {}

    class FakeCompleted:
        returncode = 0

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *args, **kwargs: (None, 8787))
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
        "# --- end Headroom MCP server ---\n",
        encoding="utf-8",
    )

    result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    content = config_file.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    expected = build_headroom_spec()
    headroom_mcp = parsed["mcp_servers"]["headroom"]
    assert "[mcp_servers.headroom]" in content
    assert headroom_mcp["command"] == expected.command
    assert headroom_mcp["args"] == list(expected.args)
    assert "env" not in headroom_mcp or "HEADROOM_PROXY_URL" not in headroom_mcp["env"]
    assert "http://127.0.0.1:9000" not in content


def test_wrap_codex_memory_prepare_only_uses_local_db_without_persisting_it(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("USER", "codex-user")
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    backend_paths: list[str] = []
    imported_users: list[str] = []

    class FakeBackend:
        async def _ensure_initialized(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class FakeClaudeCodeAdapter:
        def __init__(self, memory_dir: Path) -> None:
            self.memory_dir = memory_dir

    def fake_build_sync_backend(db_path: str) -> FakeBackend:
        backend_paths.append(db_path)
        return FakeBackend()

    async def fake_sync_import(
        backend: FakeBackend, adapter: FakeClaudeCodeAdapter, user_id: str
    ) -> int:
        imported_users.append(user_id)
        return 0

    with patch("headroom.memory.sync._build_sync_backend", side_effect=fake_build_sync_backend):
        with patch("headroom.memory.sync.sync_import", side_effect=fake_sync_import):
            with patch(
                "headroom.memory.sync_adapters.claude_code.ClaudeCodeAdapter",
                FakeClaudeCodeAdapter,
            ):
                with patch(
                    "headroom.memory.sync_adapters.claude_code.get_claude_memory_dir",
                    return_value=tmp_path / "claude-memory",
                ):
                    result = runner.invoke(
                        main,
                        [
                            "wrap",
                            "codex",
                            "--memory",
                            "--prepare-only",
                            "--no-mcp",
                            "--no-serena",
                        ],
                    )

    assert result.exit_code == 0, result.output
    assert backend_paths == [str(project_dir / ".headroom" / "memory.db")]
    assert imported_users == ["codex-user"]

    content = (tmp_path / ".codex" / "config.toml").read_text()
    assert "[mcp_servers.headroom_memory]" in content
    assert '"--user", "codex-user"' in content
    assert "--db" not in content


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

    with patch("headroom.cli.wrap.shutil.which", side_effect=fake_which):
        # Serena is the code-memory MCP; assert it lands in the codex config.
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only"])

    assert result.exit_code == 0, result.output
    content = config_file.read_text(encoding="utf-8")
    assert "[mcp_servers.serena]" in content
    assert 'command = "uvx"' in content
    assert '"--context", "codex"' in content


def test_wrap_codex_prepare_only_no_serena_skips_serena(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    config_file = tmp_path / ".codex" / "config.toml"
    config_file.parent.mkdir(parents=True)

    result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--no-serena"])

    assert result.exit_code == 0, result.output
    assert "[mcp_servers.serena]" not in config_file.read_text(encoding="utf-8")


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
    config_file.write_text(original, encoding="utf-8")

    wrap_result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])
    assert wrap_result.exit_code == 0, wrap_result.output
    assert 'model_provider = "headroom"' in config_file.read_text(encoding="utf-8")

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
    assert config_file.read_text(encoding="utf-8") == original
    assert 'model_provider = "headroom"' not in config_file.read_text(encoding="utf-8")
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


def test_wrap_codex_memory_prepare_only_unwrap_removes_memory_mcp_without_prior_config(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("USER", "codex-user")
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    class FakeBackend:
        async def _ensure_initialized(self) -> None:
            return None

        async def close(self) -> None:
            return None

    async def fake_sync_import(backend: FakeBackend, adapter: object, user_id: str) -> int:
        return 0

    with patch("headroom.memory.sync._build_sync_backend", return_value=FakeBackend()):
        with patch("headroom.memory.sync.sync_import", side_effect=fake_sync_import):
            with patch(
                "headroom.memory.sync_adapters.claude_code.ClaudeCodeAdapter",
                autospec=True,
            ):
                with patch(
                    "headroom.memory.sync_adapters.claude_code.get_claude_memory_dir",
                    return_value=tmp_path / "claude-memory",
                ):
                    wrap_result = runner.invoke(
                        main,
                        [
                            "wrap",
                            "codex",
                            "--memory",
                            "--prepare-only",
                            "--no-mcp",
                            "--no-serena",
                        ],
                    )

    assert wrap_result.exit_code == 0, wrap_result.output
    config_file = tmp_path / ".codex" / "config.toml"
    content = config_file.read_text()
    assert "[mcp_servers.headroom_memory]" in content
    assert '"--user", "codex-user"' in content

    with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy:
        unwrap_result = runner.invoke(main, ["unwrap", "codex", "--no-stop-proxy"])

    assert unwrap_result.exit_code == 0, unwrap_result.output
    assert not config_file.exists()
    assert not (tmp_path / ".codex" / "config.toml.headroom-backup").exists()
    stop_proxy.assert_not_called()


def test_wrap_codex_memory_launch_failure_unwrap_cleans_memory_only_config(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("USER", "codex-user")
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    class FakeBackend:
        async def _ensure_initialized(self) -> None:
            return None

        async def close(self) -> None:
            return None

    async def fake_sync_import(backend: FakeBackend, adapter: object, user_id: str) -> int:
        return 0

    def fake_which(cmd: str) -> str | None:
        return None if cmd == "codex" else shutil.which(cmd)

    with patch("headroom.cli.wrap.shutil.which", side_effect=fake_which):
        with patch("headroom.memory.sync._build_sync_backend", return_value=FakeBackend()):
            with patch("headroom.memory.sync.sync_import", side_effect=fake_sync_import):
                with patch(
                    "headroom.memory.sync_adapters.claude_code.ClaudeCodeAdapter",
                    autospec=True,
                ):
                    with patch(
                        "headroom.memory.sync_adapters.claude_code.get_claude_memory_dir",
                        return_value=tmp_path / "claude-memory",
                    ):
                        wrap_result = runner.invoke(
                            main,
                            ["wrap", "codex", "--memory", "--no-mcp", "--no-serena"],
                        )

    assert wrap_result.exit_code == 1
    config_file = tmp_path / ".codex" / "config.toml"
    assert not config_file.exists()
    assert not (tmp_path / ".codex" / "config.toml.headroom-backup").exists()

    with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy:
        unwrap_result = runner.invoke(main, ["unwrap", "codex", "--no-stop-proxy"])

    assert unwrap_result.exit_code == 0, unwrap_result.output
    assert "Nothing to undo" in unwrap_result.output
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
    config_file.write_text(original, encoding="utf-8")

    runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    result = runner.invoke(main, ["unwrap", "codex"])
    assert result.exit_code == 0, result.output
    restored = config_file.read_text(encoding="utf-8")
    assert restored == original


# ---------------------------------------------------------------------------
# Per-project savings: env_http_headers in the injected provider block
# ---------------------------------------------------------------------------


class TestCodexProjectHeaderConfig:
    """The injected provider maps X-Headroom-Project to HEADROOM_PROJECT.

    Codex's ``env_http_headers`` sends a header only when the mapped env var
    is set at Codex runtime, so `headroom wrap codex` exports
    ``HEADROOM_PROJECT`` and the proxy attributes savings per project.
    """

    def test_inject_writes_env_http_headers_mapping(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_test_home(monkeypatch, tmp_path)

        wrap_mod._inject_codex_provider_config(8787)

        content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert 'env_http_headers = { "X-Headroom-Project" = "HEADROOM_PROJECT" }' in content

    def test_env_http_headers_inside_provider_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The mapping must live inside [model_providers.headroom], before
        the closing marker, so it applies to the Headroom provider."""
        _set_test_home(monkeypatch, tmp_path)

        wrap_mod._inject_codex_provider_config(8787)

        content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
        section_start = content.index("[model_providers.headroom]")
        mapping_pos = content.index("env_http_headers")
        end_marker_pos = content.index(wrap_mod._CODEX_END_MARKER, section_start)
        assert section_start < mapping_pos < end_marker_pos

    def test_strip_removes_block_with_env_http_headers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_strip_codex_headroom_blocks removes the whole injected block,
        including the new env_http_headers line, leaving user content."""
        _set_test_home(monkeypatch, tmp_path)
        config_dir = tmp_path / ".codex"
        config_dir.mkdir()
        config_file = config_dir / "config.toml"
        original = '[profiles.default]\nmodel = "gpt-4o"\n'
        config_file.write_text(original, encoding="utf-8")

        wrap_mod._inject_codex_provider_config(8787)
        wrapped = config_file.read_text(encoding="utf-8")
        assert "env_http_headers" in wrapped

        cleaned = wrap_mod._strip_codex_headroom_blocks(wrapped)
        assert "env_http_headers" not in cleaned
        assert "X-Headroom-Project" not in cleaned
        assert "[model_providers.headroom]" not in cleaned
        assert 'model = "gpt-4o"' in cleaned


# ---------------------------------------------------------------------------
# Regression: codex preserves the requested port through the session-scoped runner
# ---------------------------------------------------------------------------


class TestCodexPortResolution:
    """codex() hands the requested port to the session-scoped wrap runner.

    Regression for headroom#1406 round 2 review: the codex command must keep
    the selected-port contract intact after the session-home refactor instead
    of silently dropping or rewriting the requested port before the shared
    launch path handles proxy reuse and fallback.
    """

    def test_delegates_to_session_runner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """codex() passes the requested port through to _run_codex_wrap."""
        _set_test_home(monkeypatch, Path("/tmp/test_headroom_codex"))

        call_kw: dict = {}

        def mock_run_codex_wrap(**kwargs: object) -> None:
            call_kw.update(kwargs)

        monkeypatch.setattr(wrap_mod, "_run_codex_wrap", mock_run_codex_wrap)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["wrap", "codex", "--port", "8787", "--no-mcp", "--no-serena"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert call_kw.get("port") == 8787
        assert call_kw.get("no_proxy") is False
        assert call_kw.get("prepare_only") is False


class TestCodexLaunchExportsCustomUpstream:
    """`_run_codex_wrap` must export the detected custom upstream base URL into
    the launch env so Codex emits the ``X-Headroom-Base-Url`` header. Otherwise
    the proxy falls back to api.openai.com and the user's gateway key is sent to
    the wrong host (regression of #1614)."""

    def _launch_env(self, monkeypatch, tmp_path, *, custom_upstream):
        captured: dict = {}

        monkeypatch.setattr(wrap_mod.shutil, "which", lambda name: "/usr/bin/codex")
        monkeypatch.setattr(wrap_mod, "_codex_home_dir", lambda: tmp_path)
        monkeypatch.setattr(wrap_mod, "_offer_dangling_codex_recovery", lambda active_home: None)
        monkeypatch.setattr(wrap_mod, "_prepare_codex_wrap_state", lambda **kwargs: None)
        if custom_upstream:
            (tmp_path / "config.toml").write_text(
                "\n".join(
                    (
                        'model_provider = "gateway"',
                        "[model_providers.gateway]",
                        f'base_url = "{custom_upstream}"',
                    )
                )
                + "\n",
                encoding="utf-8",
            )

        def _fake_launch(*, env, port, configure_launch, args=(), env_vars_display=(), **kwargs):
            if configure_launch is not None:
                _args, env, _display = configure_launch(port, args, env, list(env_vars_display))
            captured["env"] = env

        monkeypatch.setattr(wrap_mod, "_launch_tool", _fake_launch)

        wrap_mod._run_codex_wrap(
            port=8787,
            no_mcp=True,
            no_tokensave=True,
            serena=False,
            no_serena=True,
            code_graph=False,
            no_proxy=True,
            learn=False,
            memory=False,
            backend=None,
            anyllm_provider=None,
            region=None,
            verbose=False,
            prepare_only=False,
            codex_args=(),
        )
        return captured["env"]

    def test_custom_upstream_exported_into_launch_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env = self._launch_env(monkeypatch, tmp_path, custom_upstream="https://api.freemodel.dev")
        assert env[wrap_mod._UPSTREAM_BASE_URL_ENV_VAR] == "https://api.freemodel.dev"

    def test_no_custom_upstream_leaves_env_var_unset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env = self._launch_env(monkeypatch, tmp_path, custom_upstream=None)
        assert wrap_mod._UPSTREAM_BASE_URL_ENV_VAR not in env
