"""CLI-level tests for `headroom dashboard` port auto-detection.

Regression coverage for: `headroom wrap copilot` (isolated-subscription path)
starting the proxy on 8788 while `headroom dashboard` kept defaulting to the
hardcoded 8787, so the printed/opened URL pointed at nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom import paths as paths_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def clients_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(paths_mod, "workspace_dir", lambda: workspace)
    return workspace / "clients"


def _write_marker(clients_root: Path, port: int, pid: int, *, started_at: float = 0.0) -> None:
    marker = clients_root / str(port) / f"{pid}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"pid": pid, "started_at": started_at}))


def test_dashboard_falls_back_to_8787_with_no_live_session(
    runner: CliRunner, clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.cli._utils import proxy_discovery as pd

    # Stub the health probe: without this the test would make a real request to
    # 127.0.0.1:8787 and pass/fail depending on whether the developer happens to
    # have a proxy running.
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: False)
    monkeypatch.delenv("HEADROOM_PORT", raising=False)

    result = runner.invoke(main, ["dashboard", "--no-open"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8787/dashboard" in result.output
    assert "Warning" in result.output  # nothing is actually listening


def test_dashboard_discovers_live_session_on_fallback_port(
    runner: CliRunner, clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the reported bug: wrap started the proxy on 8788, dashboard must follow."""
    from headroom.cli._utils import proxy_discovery as pd

    _write_marker(clients_root, 8788, pid=1, started_at=100.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: port == 8788)
    monkeypatch.delenv("HEADROOM_PORT", raising=False)

    result = runner.invoke(main, ["dashboard", "--no-open"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8788/dashboard" in result.output
    assert "Detected live proxy on port 8788" in result.output
    assert "Warning" not in result.output


def test_dashboard_explicit_port_overrides_discovery(
    runner: CliRunner, clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.cli._utils import proxy_discovery as pd

    _write_marker(clients_root, 8788, pid=1, started_at=100.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: True)

    result = runner.invoke(main, ["dashboard", "--no-open", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9999/dashboard" in result.output
    assert "Detected live proxy" not in result.output


def test_dashboard_env_port_overrides_discovery(
    runner: CliRunner, clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.cli._utils import proxy_discovery as pd

    _write_marker(clients_root, 8788, pid=1, started_at=100.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: True)
    monkeypatch.setenv("HEADROOM_PORT", "7000")

    result = runner.invoke(main, ["dashboard", "--no-open"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:7000/dashboard" in result.output
