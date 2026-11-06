"""Tests for the `headroom wrap dsh` command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _capture(captured: dict[str, object]):
    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    return fake_launch_tool


def _hermetic_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Redirect every home-adjacent path the wrap flow can write to.

    The wrap flow registers Serena/headroom MCP entries into the dsh home and
    records installs in the Headroom workspace ledger. Without redirects these
    tests would mutate the developer's real `~/.dsh` / `~/.headroom` when
    the harness is installed on the machine running the suite.
    """
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "dsh"))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "headroom"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # The repo root carries a `.serena/project.yml`, so a successful Serena
    # registration would synchronously run `serena project index` (a real
    # subprocess, up to _SERENA_INDEX_TIMEOUT). Registration is what these
    # tests assert; the pre-index is orthogonal and must not run here.
    monkeypatch.setattr("headroom.cli.wrap._index_serena_project", lambda **_kwargs: None)


def test_wrap_dsh_launches_web_with_proxy_env(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _hermetic_home(monkeypatch, tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr("headroom.cli.wrap._launch_tool", _capture(captured))
    monkeypatch.setattr(
        "headroom.providers.dsh.runtime.shutil.which",
        lambda _name: "/usr/bin/dsh",
    )

    result = runner.invoke(main, ["wrap", "dsh", "--port", "9000"])
    assert result.exit_code == 0, result.output

    env = captured["env"]
    assert env["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert captured["binary"] == "/usr/bin/dsh"
    assert captured["args"] == ("web",)
    assert captured["agent_type"] == "dsh"


def test_wrap_dsh_headless_profile(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _hermetic_home(monkeypatch, tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr("headroom.cli.wrap._launch_tool", _capture(captured))
    monkeypatch.setattr(
        "headroom.providers.dsh.runtime.shutil.which",
        lambda _name: "/usr/bin/dsh",
    )

    result = runner.invoke(main, ["wrap", "dsh", "--profile", "headless", "explain foo"])
    assert result.exit_code == 0, result.output
    assert captured["args"] == ("--profile", "headless", "explain foo")


def test_wrap_dsh_forwards_deepseek_api_url(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _hermetic_home(monkeypatch, tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr("headroom.cli.wrap._launch_tool", _capture(captured))
    monkeypatch.setattr(
        "headroom.providers.dsh.runtime.shutil.which",
        lambda _name: "/usr/bin/dsh",
    )

    result = runner.invoke(main, ["wrap", "dsh", "--deepseek-api-url", "https://deepseek.internal"])
    assert result.exit_code == 0, result.output
    assert captured["deepseek_api_url"] == "https://deepseek.internal"


def test_wrap_dsh_captures_ambient_deepseek_base_url(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _hermetic_home(monkeypatch, tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr("headroom.cli.wrap._launch_tool", _capture(captured))
    monkeypatch.setattr("headroom.providers.dsh.runtime.shutil.which", lambda _name: "/usr/bin/dsh")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://gateway.internal")
    result = runner.invoke(main, ["wrap", "dsh"])
    assert result.exit_code == 0, result.output
    assert captured["deepseek_api_url"] == "https://gateway.internal"


def test_wrap_dsh_missing_binary_fails(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _hermetic_home(monkeypatch, tmp_path)
    monkeypatch.setattr("headroom.providers.dsh.runtime.shutil.which", lambda _name: None)

    result = runner.invoke(main, ["wrap", "dsh"])
    assert result.exit_code == 1
    assert "not found in PATH" in result.output


def test_unwrap_dsh_stops_proxy(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _hermetic_home(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "headroom.cli.wrap._stop_local_proxy_for_unwrap",
        lambda _port: "stopped",
    )
    monkeypatch.setattr(
        "headroom.cli.wrap._echo_unwrap_proxy_stop_status",
        lambda _status, _port: None,
    )

    result = runner.invoke(main, ["unwrap", "dsh"])
    assert result.exit_code == 0, result.output


def test_wrap_dsh_establishes_managed_entry_before_launch_without_home(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Fresh install: no DSH home exists — the managed entry must land before
    dsh launches (review #4985851059: detect() alone would skip registration).
    """
    _hermetic_home(monkeypatch, tmp_path)
    dsh_home = tmp_path / "dsh"
    assert not dsh_home.exists()

    captured: dict[str, object] = {}
    launch_saw_entry: list[bool] = []

    def fake_launch_tool(**kwargs: object) -> None:
        # Prove the managed entry already exists at launch time.
        launch_saw_entry.append((dsh_home / "cordis.patch.yml").exists())
        captured.update(kwargs)

    monkeypatch.setattr("headroom.cli.wrap._launch_tool", fake_launch_tool)
    monkeypatch.setattr(
        "headroom.providers.dsh.runtime.shutil.which",
        lambda _name: "/usr/bin/dsh",
    )

    result = runner.invoke(main, ["wrap", "dsh", "--port", "9000"])
    assert result.exit_code == 0, result.output

    patch = dsh_home / "cordis.patch.yml"
    assert patch.exists(), "managed entry must exist before dsh launches"
    text = patch.read_text(encoding="utf-8")
    assert "@deepseek-ai/dsh-mcp-client" in text
    assert "serverName: serena" in text
    # The proxy emits `[Retrieve more: hash=…]` markers, so the headroom MCP
    # entry must land too — without it dsh has no headroom_retrieve tool to
    # resolve them. Every other wrapped agent registers it; dsh did not.
    assert "serverName: headroom" in text
    assert "mcp-headroom" in text
    assert captured["agent_type"] == "dsh"
    assert launch_saw_entry == [True]
