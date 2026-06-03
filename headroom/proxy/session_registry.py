"""Filesystem-backed active session registry for local and clustered proxies."""

from __future__ import annotations

import json
import logging
import os
import platform
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headroom import paths

SESSION_SCHEMA_VERSION = 1
DEFAULT_STALE_AFTER_SECONDS = 120
logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ClusterConfig:
    enabled: bool
    cluster_id: str
    cluster_dir: Path

    @classmethod
    def from_env(cls) -> ClusterConfig:
        return cls(
            enabled=paths.cluster_enabled(),
            cluster_id=paths.cluster_id(),
            cluster_dir=paths.cluster_dir(),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cluster_id": self.cluster_id,
            "cluster_dir": str(self.cluster_dir),
        }


class ActiveSessionRegistry:
    """Writes one manifest per running Headroom process.

    Each process owns only its own ``session.json`` files. Aggregation reads all
    manifests and prunes stale local manifests opportunistically.
    """

    def __init__(
        self,
        *,
        agent_type: str = "proxy",
        session_id: str | None = None,
        instance_id: str | None = None,
        local_sessions_dir: Path | None = None,
        cluster: ClusterConfig | None = None,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        hostname = platform.node() or "localhost"
        self.instance_id = instance_id or f"{hostname}-{os.getpid()}"
        self.agent_type = agent_type
        self.local_sessions_dir = local_sessions_dir or paths.sessions_dir()
        self.cluster = cluster or ClusterConfig.from_env()
        self.stale_after_seconds = max(int(stale_after_seconds), 1)
        self.started_at = _utc_now()
        self._last_payload: dict[str, Any] | None = None

    @property
    def local_session_dir(self) -> Path:
        return self.local_sessions_dir / self.session_id

    @property
    def local_manifest_path(self) -> Path:
        return self.local_session_dir / "session.json"

    @property
    def cluster_manifest_path(self) -> Path | None:
        if not self.cluster.enabled:
            return None
        return (
            self.cluster.cluster_dir
            / self.cluster.cluster_id
            / "sessions"
            / self.instance_id
            / self.session_id
            / "session.json"
        )

    def heartbeat(self, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        now = _utc_now()
        payload = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "instance_id": self.instance_id,
            "agent_type": self.agent_type,
            "pid": os.getpid(),
            "started_at": _to_iso(self.started_at),
            "last_heartbeat_at": _to_iso(now),
            "cluster": self.cluster.snapshot(),
            "metrics": metrics or {},
        }
        try:
            _atomic_write_json(self.local_manifest_path, payload)
        except OSError as exc:
            logger.warning(
                "event=active_session_manifest_write_failed scope=local path=%s error=%s",
                self.local_manifest_path,
                exc,
            )
        cluster_path = self.cluster_manifest_path
        if cluster_path is not None:
            try:
                _atomic_write_json(cluster_path, payload)
            except OSError as exc:
                logger.warning(
                    "event=active_session_manifest_write_failed scope=cluster path=%s error=%s",
                    cluster_path,
                    exc,
                )
        self._last_payload = payload
        return payload

    def close(self) -> None:
        self.local_manifest_path.unlink(missing_ok=True)
        cluster_path = self.cluster_manifest_path
        if cluster_path is not None:
            cluster_path.unlink(missing_ok=True)

    def snapshot(self, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.heartbeat(metrics)


def _read_manifest(path: Path, *, now: datetime, stale_after_seconds: int) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    heartbeat = _parse_iso(payload.get("last_heartbeat_at"))
    if heartbeat is None:
        return None
    stale = (now - heartbeat).total_seconds() > stale_after_seconds
    payload["stale"] = stale
    payload["age_seconds"] = max(0, round((now - heartbeat).total_seconds(), 3))
    return payload


def list_active_sessions(
    directory: Path | None = None,
    *,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    prune_stale: bool = False,
) -> list[dict[str, Any]]:
    root = directory or paths.sessions_dir()
    now = _utc_now()
    sessions: list[dict[str, Any]] = []
    if not root.exists():
        return sessions
    for manifest in root.glob("**/session.json"):
        payload = _read_manifest(manifest, now=now, stale_after_seconds=stale_after_seconds)
        if payload is None:
            continue
        if payload.get("stale"):
            if prune_stale:
                manifest.unlink(missing_ok=True)
            continue
        sessions.append(payload)
    sessions.sort(key=lambda item: str(item.get("last_heartbeat_at", "")), reverse=True)
    return sessions


def aggregate_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "requests": 0,
        "tokens_saved": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    by_agent: dict[str, int] = {}
    by_instance: dict[str, int] = {}
    for session in sessions:
        raw_metrics = session.get("metrics")
        metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
        totals["requests"] += int(metrics.get("requests", 0) or 0)
        totals["tokens_saved"] += int(metrics.get("tokens_saved", 0) or 0)
        totals["input_tokens"] += int(metrics.get("input_tokens", 0) or 0)
        totals["output_tokens"] += int(metrics.get("output_tokens", 0) or 0)
        agent = str(session.get("agent_type") or "unknown")
        instance = str(session.get("instance_id") or "unknown")
        by_agent[agent] = by_agent.get(agent, 0) + 1
        by_instance[instance] = by_instance.get(instance, 0) + 1
    return {
        "count": len(sessions),
        "totals": totals,
        "by_agent": by_agent,
        "by_instance": by_instance,
    }
