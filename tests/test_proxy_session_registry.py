from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from headroom.proxy import session_registry as sr
from headroom.proxy.session_registry import ActiveSessionRegistry, ClusterConfig


def test_active_session_registry_writes_local_and_cluster_manifests(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(
        agent_type="codex",
        session_id="sess-1",
        instance_id="inst-1",
        local_sessions_dir=tmp_path / "sessions",
        cluster=ClusterConfig(
            enabled=True,
            cluster_id="team-gamma",
            cluster_dir=tmp_path / "cluster",
        ),
    )

    payload = registry.heartbeat({"requests": 2, "tokens_saved": 50})

    assert payload["session_id"] == "sess-1"
    local = json.loads((tmp_path / "sessions" / "sess-1" / "session.json").read_text())
    cluster = json.loads(
        (
            tmp_path / "cluster" / "team-gamma" / "sessions" / "inst-1" / "sess-1" / "session.json"
        ).read_text()
    )
    assert local["metrics"]["tokens_saved"] == 50
    assert cluster["cluster"]["cluster_id"] == "team-gamma"

    sessions = sr.list_active_sessions(tmp_path / "sessions")
    assert [item["session_id"] for item in sessions] == ["sess-1"]
    assert sr.aggregate_sessions(sessions)["totals"]["tokens_saved"] == 50

    registry.close()
    assert not registry.local_manifest_path.exists()
    assert registry.cluster_manifest_path is not None
    assert not registry.cluster_manifest_path.exists()


def test_heartbeat_tolerates_unavailable_manifest_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write(path: Path, payload: dict[str, object]) -> None:
        raise PermissionError(path)

    monkeypatch.setattr(sr, "_atomic_write_json", fail_write)
    registry = ActiveSessionRegistry(
        session_id="sess-unwritable",
        instance_id="inst-unwritable",
        local_sessions_dir=tmp_path / "sessions",
        cluster=ClusterConfig(
            enabled=True,
            cluster_id="team-gamma",
            cluster_dir=tmp_path / "cluster",
        ),
    )

    payload = registry.heartbeat({"requests": 3})

    assert payload["session_id"] == "sess-unwritable"
    assert payload["metrics"]["requests"] == 3
    assert registry.snapshot({"requests": 4})["metrics"]["requests"] == 4


def test_list_active_sessions_prunes_stale_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(sr, "_utc_now", lambda: now)
    registry = ActiveSessionRegistry(
        session_id="fresh",
        instance_id="inst",
        local_sessions_dir=tmp_path / "sessions",
    )
    registry.heartbeat({"requests": 1})

    stale_dir = tmp_path / "sessions" / "stale"
    stale_dir.mkdir(parents=True)
    stale_payload = {
        "session_id": "stale",
        "last_heartbeat_at": (now - timedelta(seconds=300)).isoformat().replace("+00:00", "Z"),
        "metrics": {"requests": 99},
    }
    (stale_dir / "session.json").write_text(json.dumps(stale_payload), encoding="utf-8")

    sessions = sr.list_active_sessions(
        tmp_path / "sessions",
        stale_after_seconds=120,
        prune_stale=True,
    )

    assert [item["session_id"] for item in sessions] == ["fresh"]
    assert not (stale_dir / "session.json").exists()
