from __future__ import annotations

import asyncio
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
    assert "cluster_dir" not in cluster["cluster"]
    assert str(tmp_path) not in json.dumps(cluster)

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


def test_manifest_keeps_only_numeric_aggregate_metrics(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(local_sessions_dir=tmp_path / "sessions")

    payload = registry.heartbeat(
        {
            "requests": 3,
            "tokens_saved": 12.5,
            "savings_storage_path": "/Users/example/.headroom/savings.json",
            "arbitrary_label": "private",
            "failed_requests": True,
        }
    )

    assert payload["metrics"] == {"requests": 3, "tokens_saved": 12.5}
    assert "/Users/example" not in registry.local_manifest_path.read_text(encoding="utf-8")


def test_disabled_persistence_never_writes_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[Path] = []

    def record_write(path: Path, payload: dict[str, object]) -> None:
        writes.append(path)

    monkeypatch.setattr(sr, "_atomic_write_json", record_write)
    registry = ActiveSessionRegistry(
        local_sessions_dir=tmp_path / "sessions",
        persistence_enabled=False,
    )

    payload = registry.heartbeat({"requests": 1})
    registry.close()

    assert payload["metrics"] == {"requests": 1}
    assert writes == []
    assert not (tmp_path / "sessions").exists()


@pytest.mark.parametrize(
    "cluster_id",
    ["../outside", "../../outside", ".", "..", "a/b", "a\\b", "a" * 129],
)
def test_cluster_id_rejects_path_traversal(cluster_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single path-safe component"):
        ClusterConfig(enabled=True, cluster_id=cluster_id, cluster_dir=tmp_path)


@pytest.mark.parametrize("field", ["session_id", "instance_id"])
@pytest.mark.parametrize("value", ["../outside", "a/b", "a\\b", ".", ".."])
def test_registry_rejects_path_bearing_identifiers(field: str, value: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single path-safe component"):
        ActiveSessionRegistry(
            local_sessions_dir=tmp_path / "sessions",
            **{field: value},
        )


def test_close_tolerates_unavailable_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ActiveSessionRegistry(
        local_sessions_dir=tmp_path / "sessions",
        cluster=ClusterConfig(True, "team-gamma", tmp_path / "cluster"),
    )
    registry.heartbeat({"requests": 1})
    real_unlink = Path.unlink

    def fail_manifest_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.name == "session.json":
            raise PermissionError(path)
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_manifest_unlink)

    registry.close()


@pytest.mark.asyncio
async def test_periodic_heartbeat_keeps_idle_session_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(sr, "_utc_now", lambda: now)
    registry = ActiveSessionRegistry(
        local_sessions_dir=tmp_path / "sessions",
        stale_after_seconds=6,
    )
    registry.heartbeat({"requests": 1})
    sleeps = 0

    async def advance_clock(_seconds: float) -> None:
        nonlocal now, sleeps
        sleeps += 1
        now += timedelta(seconds=2)
        if sleeps > 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(sr.asyncio, "sleep", advance_clock)

    with pytest.raises(asyncio.CancelledError):
        await registry.run_heartbeat_loop(lambda: {"requests": 1}, interval_seconds=2)

    assert now - registry.started_at > timedelta(seconds=6)
    assert [item["session_id"] for item in sr.list_active_sessions(tmp_path / "sessions")] == [
        registry.session_id
    ]


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
        "schema_version": sr.SESSION_SCHEMA_VERSION,
        "session_id": "stale",
        "instance_id": "inst",
        "agent_type": "proxy",
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


def test_stale_pruning_is_best_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(sr, "_utc_now", lambda: now)
    stale_manifest = tmp_path / "sessions" / "stale" / "session.json"
    stale_manifest.parent.mkdir(parents=True)
    stale_manifest.write_text(
        json.dumps(
            {
                "schema_version": sr.SESSION_SCHEMA_VERSION,
                "session_id": "stale",
                "instance_id": "inst",
                "agent_type": "proxy",
                "last_heartbeat_at": (now - timedelta(seconds=300)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    real_unlink = Path.unlink

    def fail_stale_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == stale_manifest:
            raise PermissionError(path)
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_stale_unlink)

    assert sr.list_active_sessions(tmp_path / "sessions", prune_stale=True) == []


def test_aggregate_sessions_ignores_malformed_shared_metrics() -> None:
    summary = sr.aggregate_sessions(
        [
            {
                "agent_type": "proxy",
                "instance_id": "instance-1",
                "metrics": {
                    "requests": "not-a-number",
                    "tokens_saved": float("nan"),
                    "input_tokens": True,
                    "output_tokens": 2.5,
                },
            }
        ]
    )

    assert summary["totals"] == {
        "requests": 0,
        "tokens_saved": 0,
        "input_tokens": 0,
        "output_tokens": 2.5,
    }


def test_shared_manifest_reader_drops_unknown_fields_and_invalid_metrics(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(
        session_id="safe-session",
        instance_id="safe-instance",
        local_sessions_dir=tmp_path / "sessions",
    )
    payload = registry.heartbeat({"requests": 3})
    payload["secret"] = "must-not-escape"
    payload["metrics"] = {
        "requests": 4,
        "tokens_saved": float("nan"),
        "prompt": "private",
    }
    registry.local_manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    sessions = sr.list_active_sessions(tmp_path / "sessions")

    assert len(sessions) == 1
    assert "secret" not in sessions[0]
    assert sessions[0]["metrics"] == {"requests": 4}


def test_shared_manifest_reader_rejects_invalid_identity(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(
        session_id="safe-session",
        instance_id="safe-instance",
        local_sessions_dir=tmp_path / "sessions",
    )
    payload = registry.heartbeat({"requests": 3})
    payload["agent_type"] = "../../private"
    registry.local_manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert sr.list_active_sessions(tmp_path / "sessions") == []


def test_malformed_neighbor_does_not_hide_healthy_sessions(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(
        session_id="healthy", instance_id="instance", local_sessions_dir=tmp_path
    )
    payload = registry.heartbeat({"requests": 3})
    payload["session_id"] = "oversized"
    payload["metrics"] = {"requests": 10**400, "tokens_saved": 5}
    oversized = tmp_path / "oversized" / "session.json"
    oversized.parent.mkdir()
    oversized.write_text(json.dumps(payload), encoding="utf-8")
    corrupt = tmp_path / "corrupt" / "session.json"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"\xff")

    sessions = sr.list_active_sessions(tmp_path)

    assert {s["session_id"] for s in sessions} == {"healthy", "oversized"}
    assert sr.aggregate_sessions(sessions)["totals"]["requests"] == 3
    assert sr.aggregate_sessions(sessions)["totals"]["tokens_saved"] == 5
    # The public aggregation path must also tolerate unsanitized callers.
    assert sr.aggregate_sessions([payload])["totals"]["requests"] == 0


@pytest.mark.parametrize("large", [1e308, 10**308])
def test_aggregate_overflow_keeps_summary_json_safe(large: int | float) -> None:
    sessions = [
        {"metrics": {"requests": large, "tokens_saved": 2}},
        {"metrics": {"requests": large, "tokens_saved": 3}},
        {"metrics": {"requests": 1.0, "tokens_saved": 4}},
    ]

    summary = sr.aggregate_sessions(sessions)

    assert summary["count"] == 3
    assert summary["totals"]["tokens_saved"] == 9
    assert summary["totals"]["requests"] == pytest.approx(float(large))
    json.dumps(summary, allow_nan=False)


def test_close_removes_empty_session_and_instance_directories(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(
        session_id="sess-clean",
        instance_id="inst-clean",
        local_sessions_dir=tmp_path / "sessions",
        cluster=ClusterConfig(enabled=True, cluster_id="team", cluster_dir=tmp_path / "cluster"),
    )
    registry.heartbeat({"requests": 1})

    registry.close()

    assert not (tmp_path / "sessions" / "sess-clean").exists()
    assert not (tmp_path / "cluster" / "team" / "sessions" / "inst-clean").exists()


def test_free_form_agent_label_stays_visible(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(
        agent_type="Claude Code",
        session_id="sess-label",
        instance_id="inst-label",
        local_sessions_dir=tmp_path / "sessions",
    )
    registry.heartbeat({"requests": 1})

    sessions = sr.list_active_sessions(tmp_path / "sessions")

    assert [item["agent_type"] for item in sessions] == ["Claude Code"]


@pytest.mark.windows_newline
def test_manifest_writer_pins_lf_newlines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    seen: dict[str, object] = {}
    real_fdopen = os.fdopen

    def spy_fdopen(fd, mode="r", *args, **kwargs):
        seen.update(kwargs)
        return real_fdopen(fd, mode, *args, **kwargs)

    monkeypatch.setattr(sr.os, "fdopen", spy_fdopen)
    target = tmp_path / "session.json"

    sr._atomic_write_json(target, {"a": 1})

    assert seen.get("newline") == "\n"
    assert target.read_bytes().endswith(b"}\n")
    assert b"\r" not in target.read_bytes()


def test_stateless_stats_does_not_prune_stale_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import create_app

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    stale = ActiveSessionRegistry(
        session_id="sess-stale",
        instance_id="inst-stale",
        local_sessions_dir=tmp_path / "workspace" / "sessions",
        stale_after_seconds=1,
    )
    stale.heartbeat({"requests": 1})
    manifest = json.loads(stale.local_manifest_path.read_text())
    manifest["last_heartbeat_at"] = "2000-01-01T00:00:00Z"
    stale.local_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            log_requests=False,
            subscription_tracking_enabled=False,
            stateless=True,
        )
    )
    app.state.proxy.active_session_registry.local_sessions_dir = tmp_path / "workspace" / "sessions"
    with TestClient(app) as client:
        assert client.get("/stats").status_code == 200

    assert stale.local_manifest_path.exists()
