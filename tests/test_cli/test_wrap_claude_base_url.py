"""Tests for _write_claude_wrap_base_url / _restore_claude_wrap_base_url (issue #951)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main


def _settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def test_write_creates_env_key_in_fresh_file(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_write_preserves_other_env_keys(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"env": {"KEEP": "1", "ANOTHER": "2"}}), encoding="utf-8")
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["KEEP"] == "1"
    assert payload["env"]["ANOTHER"] == "2"
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_tool_search_write_and_restore_reaches_daemon_worker_settings(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ENABLE_TOOL_SEARCH": "true", "KEEP": "1"}}),
        encoding="utf-8",
    )

    previous = wrap_cli._write_claude_wrap_tool_search("false", settings_path=path)

    assert previous == "true"
    assert json.loads(path.read_text(encoding="utf-8"))["env"] == {
        "ENABLE_TOOL_SEARCH": "false",
        "KEEP": "1",
    }

    wrap_cli._restore_claude_wrap_tool_search(previous, settings_path=path)
    assert json.loads(path.read_text(encoding="utf-8"))["env"] == {
        "ENABLE_TOOL_SEARCH": "true",
        "KEEP": "1",
    }


def test_write_returns_none_when_key_absent(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev is None


def test_write_returns_previous_value_when_key_present(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://old.proxy:9000"}}),
        encoding="utf-8",
    )
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev == "http://old.proxy:9000"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_write_foundry_mode_sets_foundry_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url(
        "http://127.0.0.1:8787", foundry_mode=True, settings_path=path
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_FOUNDRY_BASE_URL"] == "http://127.0.0.1:8787"
    assert "ANTHROPIC_BASE_URL" not in payload["env"]


def test_restore_removes_key_when_previous_none(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)
    # file is deleted when payload becomes empty — key is gone
    assert not path.exists()


def test_restore_removes_env_dict_when_empty(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)
    # entire payload was {"env": {...only our key...}} — file deleted rather than left as {}
    assert not path.exists()


def test_restore_preserves_sibling_env_keys(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787", "KEEP": "1"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in payload["env"]
    assert payload["env"]["KEEP"] == "1"


def test_restore_sets_key_back_to_previous_value(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url("http://old.proxy:9000", settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://old.proxy:9000"


def test_restore_foundry_mode_removes_foundry_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_FOUNDRY_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, foundry_mode=True, settings_path=path)
    # file deleted when payload empties
    assert not path.exists()


def test_restore_noop_when_file_absent(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # must not raise


def test_restore_noop_when_key_not_present(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"env": {"OTHER": "1"}}), encoding="utf-8")
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # key absent — no-op
    assert json.loads(path.read_text())["env"]["OTHER"] == "1"


def test_restore_noop_when_env_not_dict(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"env": "not-a-dict"}), encoding="utf-8")
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # must not raise


def test_restore_noop_when_payload_not_dict(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON but not a dict
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # must not raise


def test_restore_noop_when_file_corrupt(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not valid json {{{{", encoding="utf-8")
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)  # must not raise


def test_write_refuses_to_clobber_a_corrupt_file(tmp_path: Path) -> None:
    """A file that will not parse is DATA, not a blank slate — never overwrite it.

    This previously "recovered" by resetting the payload to ``{}`` and writing
    that back, so a single hand-edited typo (or a transient read error) silently
    destroyed the user's whole settings file — permissions, env and hooks — on
    every ``headroom wrap claude``. Refusing leaves the file for the user to fix.
    """
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    original = '{"permissions": {"allow": ["Bash"]}, oops'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(click.ClickException, match="not valid JSON"):
        wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)

    assert path.read_text(encoding="utf-8") == original  # untouched


def test_write_refuses_non_dict_payload(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    original = "[1, 2, 3]"  # valid JSON but not a settings object
    path.write_text(original, encoding="utf-8")

    with pytest.raises(click.ClickException, match="does not contain a JSON object"):
        wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)

    assert path.read_text(encoding="utf-8") == original  # untouched


def test_write_recovers_from_an_empty_file(tmp_path: Path) -> None:
    """An empty file has no settings to lose, so recover rather than strand the user.

    A zero-byte settings.json is the classic residue of an interrupted
    non-atomic write, so this is the one case where treating the file as fresh
    is both safe and the helpful thing to do.
    """
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("   \n", encoding="utf-8")

    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)

    assert prev is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_write_restore_roundtrip(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": "opus", "env": {"OTHER": "x"}}), encoding="utf-8")
    prev = wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert prev is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert payload["model"] == "opus"

    wrap_cli._restore_claude_wrap_base_url(prev, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "ANTHROPIC_BASE_URL" not in payload.get("env", {})
    assert payload["env"]["OTHER"] == "x"
    assert payload["model"] == "opus"


def test_claude_project_settings_enabled_respects_flag_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_CLAUDE_PROJECT_SETTINGS", raising=False)
    assert wrap_cli._claude_project_settings_enabled(False) is False
    assert wrap_cli._claude_project_settings_enabled(True) is True

    monkeypatch.setenv("HEADROOM_CLAUDE_PROJECT_SETTINGS", "1")
    assert wrap_cli._claude_project_settings_enabled(False) is True


def test_wrap_claude_skips_project_settings_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    completed = SimpleNamespace(returncode=0)
    monkeypatch.delenv("HEADROOM_CLAUDE_PROJECT_SETTINGS", raising=False)

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        settings_path = Path(".claude") / "settings.local.json"
        settings_path.parent.mkdir()
        original = {
            "env": {
                "ANTHROPIC_BASE_URL": "http://direct.example",
                "KEEP": "1",
            }
        }
        settings_path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

        with (
            patch("headroom.cli.wrap.shutil.which", return_value="claude"),
            patch("headroom.cli.wrap._ensure_proxy", return_value=(None, 8787)),
            patch("headroom.cli.wrap._setup_headroom_mcp", return_value=None),
            patch("headroom.cli.wrap._setup_coding_compressor", return_value=None),
            patch("headroom.cli.wrap._write_claude_wrap_base_url") as write_mock,
            patch("headroom.cli.wrap._restore_claude_wrap_base_url") as restore_mock,
            patch("headroom.cli.wrap._ensure_claude_wrap_selfheal_hook") as selfheal_mock,
            patch("headroom.cli.wrap.subprocess.run", return_value=completed) as run_mock,
        ):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "claude",
                    "--no-mcp",
                    "--no-tokensave",
                    "--no-serena",
                ],
            )

        assert result.exit_code == 0, result.output
        assert json.loads(settings_path.read_text(encoding="utf-8")) == original
        write_mock.assert_not_called()
        restore_mock.assert_not_called()
        selfheal_mock.assert_not_called()
        launched_env = run_mock.call_args.kwargs["env"]
        assert launched_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"


def test_wrap_claude_project_settings_flag_writes_and_restores(tmp_path: Path) -> None:
    runner = CliRunner()
    completed = SimpleNamespace(returncode=0)

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with (
            patch("headroom.cli.wrap.shutil.which", return_value="claude"),
            patch("headroom.cli.wrap._ensure_proxy", return_value=(None, 8787)),
            patch("headroom.cli.wrap._setup_headroom_mcp", return_value=None),
            patch("headroom.cli.wrap._setup_coding_compressor", return_value=None),
            patch(
                "headroom.cli.wrap._write_claude_wrap_base_url", return_value="old"
            ) as write_mock,
            patch("headroom.cli.wrap._restore_claude_wrap_base_url") as restore_mock,
            patch(
                "headroom.cli.wrap._write_claude_wrap_tool_search", return_value="old-tool-search"
            ) as write_tool_search_mock,
            patch("headroom.cli.wrap._restore_claude_wrap_tool_search") as restore_tool_search_mock,
            patch("headroom.cli.wrap._ensure_claude_wrap_selfheal_hook") as selfheal_mock,
            patch("headroom.cli.wrap.subprocess.run", return_value=completed),
        ):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "claude",
                    "--project-settings",
                    "--no-mcp",
                    "--no-tokensave",
                    "--no-serena",
                ],
            )

        assert result.exit_code == 0, result.output
        write_mock.assert_called_once()
        write_args, write_kwargs = write_mock.call_args
        assert write_args == ("http://127.0.0.1:8787",)
        assert write_kwargs["foundry_mode"] is False
        assert write_kwargs["vertex_mode"] is False
        assert write_kwargs["port"] == 8787
        assert write_kwargs["settings_path"].name == "settings.local.json"
        assert write_kwargs["settings_path"].parent.name == ".claude"
        selfheal_mock.assert_called_once_with(write_kwargs["settings_path"])
        write_tool_search_mock.assert_called_once_with(
            "true", settings_path=write_kwargs["settings_path"]
        )
        restore_tool_search_mock.assert_called_once_with(
            "old-tool-search", settings_path=write_kwargs["settings_path"]
        )
        restore_mock.assert_called_once_with(
            "old",
            foundry_mode=False,
            vertex_mode=False,
            settings_path=write_kwargs["settings_path"],
        )


# --- stale wrap marker (issue #1768) --------------------------------------


def _marker(tmp_path: Path) -> Path:
    return wrap_cli._wrap_marker_path(_settings(tmp_path))


def test_write_with_port_creates_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path, port=8787)
    marker = json.loads(_marker(tmp_path).read_text(encoding="utf-8"))
    assert marker["port"] == 8787
    assert marker["key"] == "ANTHROPIC_BASE_URL"
    assert marker["previous"] is None
    assert marker["pid"] > 0


def test_write_without_port_skips_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path)
    assert not _marker(tmp_path).exists()


def test_restore_clears_marker_for_matching_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path, port=8787)
    assert _marker(tmp_path).exists()
    wrap_cli._restore_claude_wrap_base_url(None, settings_path=path)
    assert not _marker(tmp_path).exists()


def test_wrap_marker_is_stale_when_pid_missing() -> None:
    assert wrap_cli._wrap_marker_is_stale({}) is True


def test_wrap_marker_is_stale_when_pid_dead(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path, port=8787)
    marker = json.loads(_marker(tmp_path).read_text(encoding="utf-8"))
    marker["pid"] = 999_999_999  # astronomically unlikely to be a live pid
    assert wrap_cli._wrap_marker_is_stale(marker) is True


def test_wrap_marker_is_not_stale_for_live_pid(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path, port=8787)
    marker = json.loads(_marker(tmp_path).read_text(encoding="utf-8"))
    assert wrap_cli._wrap_marker_is_stale(marker) is False


def test_wrap_marker_is_stale_when_pid_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject a deterministic PID identity: _proc_identity returns None on
    # macOS without psutil, where reuse detection is deliberately best-effort
    # and this scenario would be undetectable.
    monkeypatch.setattr(wrap_cli, "_proc_identity", lambda pid: ("test", 50_000.0))
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path, port=8787)
    marker = json.loads(_marker(tmp_path).read_text(encoding="utf-8"))
    marker["start_time"] = marker["start_time"] - 10_000  # fabricate a mismatched identity
    assert wrap_cli._wrap_marker_is_stale(marker) is True


def test_check_and_clear_stale_wrap_marker_restores_previous(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://old.proxy:9000"}}), encoding="utf-8"
    )
    wrap_cli._write_wrap_marker(
        path, port=8787, key="ANTHROPIC_BASE_URL", previous="http://old.proxy:9000"
    )
    marker = json.loads(_marker(tmp_path).read_text(encoding="utf-8"))
    marker["pid"] = 999_999_999
    _marker(tmp_path).write_text(json.dumps(marker), encoding="utf-8")

    restored = wrap_cli._check_and_clear_stale_wrap_marker(path, key="ANTHROPIC_BASE_URL")
    assert restored == "http://old.proxy:9000"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://old.proxy:9000"
    assert not _marker(tmp_path).exists()


def test_check_and_clear_stale_wrap_marker_leaves_live_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    wrap_cli._write_claude_wrap_base_url("http://127.0.0.1:8787", settings_path=path, port=8787)
    restored = wrap_cli._check_and_clear_stale_wrap_marker(path, key="ANTHROPIC_BASE_URL")
    assert restored is None
    assert _marker(tmp_path).exists()


def test_check_and_clear_stale_wrap_marker_noop_when_no_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    assert wrap_cli._check_and_clear_stale_wrap_marker(path, key="ANTHROPIC_BASE_URL") is None
