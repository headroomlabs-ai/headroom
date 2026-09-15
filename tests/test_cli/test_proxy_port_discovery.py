"""Unit tests for `headroom.cli._utils.proxy_discovery`.

`headroom wrap copilot` (isolated-subscription path) or a plain port-busy
fallback can start the proxy on a port other than 8787. Read-only commands
(`dashboard`, `doctor`, `inspect`, `learn`) need to find that real port from
the wrap-client marker files instead of blindly defaulting to 8787. These
tests pin the discovery/resolution contract directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from headroom import paths as paths_mod
from headroom.cli._utils import proxy_discovery as pd


@pytest.fixture
def clients_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``paths.workspace_dir()`` (and therefore ``proxy_clients_dir``,
    which is built from it) into a throwaway tmp tree."""
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(paths_mod, "workspace_dir", lambda: workspace)
    return workspace / "clients"


def _write_marker(clients_root: Path, port: int, pid: int, *, started_at: float = 0.0) -> Path:
    marker = clients_root / str(port) / f"{pid}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"pid": pid, "started_at": started_at}))
    return marker


# ---------------------------------------------------------------------------
# live_client_pids / live_proxy_ports
# ---------------------------------------------------------------------------


def test_no_markers_means_no_live_ports(clients_root: Path) -> None:
    assert pd.live_proxy_ports() == []
    assert pd.live_client_pids(8787) == []


def test_live_marker_is_discovered(clients_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_marker(clients_root, 8788, pid=4242, started_at=100.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: pid == 4242)

    assert pd.live_client_pids(8788) == [4242]
    assert pd.live_proxy_ports() == [(8788, 100.0)]


def test_dead_marker_is_pruned_and_not_counted(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = _write_marker(clients_root, 8788, pid=999, started_at=100.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: False)

    assert pd.live_client_pids(8788) == []
    assert pd.live_proxy_ports() == []
    assert not marker.exists()


def test_multiple_ports_sorted_newest_first(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_marker(clients_root, 8787, pid=1, started_at=50.0)
    _write_marker(clients_root, 8788, pid=2, started_at=200.0)
    _write_marker(clients_root, 8790, pid=3, started_at=100.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)

    assert pd.live_proxy_ports() == [(8788, 200.0), (8790, 100.0), (8787, 50.0)]


def test_non_numeric_client_dirs_are_ignored(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (clients_root / "not-a-port").mkdir(parents=True)
    _write_marker(clients_root, 8788, pid=1, started_at=1.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)

    assert pd.live_proxy_ports() == [(8788, 1.0)]


# ---------------------------------------------------------------------------
# resolve_proxy_port
# ---------------------------------------------------------------------------


def test_explicit_port_always_wins(clients_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_marker(clients_root, 8788, pid=1, started_at=1.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: True)

    assert pd.resolve_proxy_port(9999) == (9999, "explicit")


def test_env_var_wins_over_discovery(clients_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_marker(clients_root, 8788, pid=1, started_at=1.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: True)
    monkeypatch.setenv("HEADROOM_PORT", "7000")

    assert pd.resolve_proxy_port(None) == (7000, "env")


def test_no_markers_falls_back_to_default(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HEADROOM_PORT", raising=False)

    assert pd.resolve_proxy_port(None) == (8787, "default")


def test_healthy_live_marker_is_discovered(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_marker(clients_root, 8788, pid=1, started_at=1.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: port == 8788)
    monkeypatch.delenv("HEADROOM_PORT", raising=False)

    assert pd.resolve_proxy_port(None) == (8788, "discovered")


def test_unhealthy_marker_falls_back_to_default(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live client marker with no proxy actually answering /health must not
    be trusted — falls back to the historical default instead of a dead URL."""
    _write_marker(clients_root, 8788, pid=1, started_at=1.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: False)
    monkeypatch.delenv("HEADROOM_PORT", raising=False)

    assert pd.resolve_proxy_port(None) == (8787, "default")


def test_newest_session_preferred_when_multiple_healthy(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_marker(clients_root, 8787, pid=1, started_at=50.0)
    _write_marker(clients_root, 8788, pid=2, started_at=200.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: True)
    monkeypatch.delenv("HEADROOM_PORT", raising=False)

    assert pd.resolve_proxy_port(None) == (8788, "discovered")


def test_falls_through_to_next_candidate_when_newest_is_unhealthy(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The newest session wins only if it is actually reachable; a stale/dead
    newest candidate must not shadow an older but healthy one."""
    _write_marker(clients_root, 8787, pid=1, started_at=50.0)
    _write_marker(clients_root, 8788, pid=2, started_at=200.0)
    monkeypatch.setattr(pd, "pid_alive", lambda pid: True)
    monkeypatch.setattr(pd, "proxy_is_healthy", lambda port, **_: port == 8787)
    monkeypatch.delenv("HEADROOM_PORT", raising=False)

    assert pd.resolve_proxy_port(None) == (8787, "discovered")


def test_invalid_env_port_is_rejected(clients_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_PORT", "not-a-number")

    with pytest.raises(click.ClickException, match="HEADROOM_PORT must be an integer"):
        pd.resolve_proxy_port(None)


def test_out_of_range_env_port_is_rejected(
    clients_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADROOM_PORT", "70000")

    with pytest.raises(click.ClickException, match="HEADROOM_PORT must be between 1 and 65535"):
        pd.resolve_proxy_port(None)


# ---------------------------------------------------------------------------
# proxy_is_healthy
# ---------------------------------------------------------------------------


def test_proxy_is_healthy_true_when_health_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pd,
        "probe_json",
        lambda url, timeout=1.0: {"service": "headroom-proxy", "status": "healthy"},
    )
    assert pd.proxy_is_healthy(8787) is True


def test_proxy_is_healthy_false_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd, "probe_json", lambda url, timeout=1.0: None)
    assert pd.proxy_is_healthy(8787) is False


def test_proxy_is_healthy_rejects_unrelated_json_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pd,
        "probe_json",
        lambda url, timeout=1.0: {"service": "something-else", "status": "healthy"},
    )
    assert pd.proxy_is_healthy(8787) is False
