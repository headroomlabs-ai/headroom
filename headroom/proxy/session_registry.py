"""Filesystem-backed active session registry for local and clustered proxies."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headroom import paths

SESSION_SCHEMA_VERSION = 1
DEFAULT_STALE_AFTER_SECONDS = 120
logger = logging.getLogger(__name__)
_SAFE_CLUSTER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_COMPONENT_LENGTH = 128
_AGGREGATE_METRIC_KEYS = frozenset(
    {
        "requests",
        "tokens_saved",
        "input_tokens",
        "output_tokens",
        "failed_requests",
        "cached_requests",
        "rate_limited_requests",
    }
)


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


def _validate_path_component(value: str, *, label: str) -> str:
    if (
        value in {".", ".."}
        or len(value) > _MAX_COMPONENT_LENGTH
        or not _SAFE_CLUSTER_ID.fullmatch(value)
    ):
        raise ValueError(
            f"{label} must be a single path-safe component containing only "
            "letters, numbers, dots, underscores, and hyphens"
        )
    return value


def _aggregate_metrics(metrics: Any) -> dict[str, int | float]:
    if not isinstance(metrics, dict):
        return {}
    return {
        key: value
        for key, value in metrics.items()
        if key in _AGGREGATE_METRIC_KEYS
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


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

    def __post_init__(self) -> None:
        _validate_path_component(self.cluster_id, label="HEADROOM_CLUSTER_ID")

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
        persistence_enabled: bool = True,
    ) -> None:
        self.session_id = _validate_path_component(
            session_id or str(uuid.uuid4()), label="session_id"
        )
        self.instance_id = _validate_path_component(
            instance_id or str(uuid.uuid4()), label="instance_id"
        )
        self.agent_type = agent_type
        self.local_sessions_dir = local_sessions_dir or paths.sessions_dir()
        self.cluster = cluster or ClusterConfig.from_env()
        self.stale_after_seconds = max(int(stale_after_seconds), 1)
        self.persistence_enabled = persistence_enabled
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
        aggregate_metrics = _aggregate_metrics(metrics)
        payload = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "instance_id": self.instance_id,
            "agent_type": self.agent_type,
            "pid": os.getpid(),
            "started_at": _to_iso(self.started_at),
            "last_heartbeat_at": _to_iso(now),
            "cluster": self.cluster.snapshot(),
            "metrics": aggregate_metrics,
        }
        if self.persistence_enabled:
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

    async def run_heartbeat_loop(
        self,
        metrics_provider: Callable[[], dict[str, Any]],
        *,
        interval_seconds: float | None = None,
    ) -> None:
        """Refresh this process manifest until the lifecycle task is cancelled."""

        interval = interval_seconds or max(1.0, self.stale_after_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                self.heartbeat(metrics_provider())
            except Exception as exc:  # noqa: BLE001 - best-effort sidecar must stay alive
                logger.warning("event=active_session_heartbeat_failed error=%s", exc)

    def close(self) -> None:
        if not self.persistence_enabled:
            return
        try:
            self.local_manifest_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "event=active_session_manifest_cleanup_failed scope=local path=%s error=%s",
                self.local_manifest_path,
                exc,
            )
        cluster_path = self.cluster_manifest_path
        if cluster_path is not None:
            try:
                cluster_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "event=active_session_manifest_cleanup_failed scope=cluster path=%s error=%s",
                    cluster_path,
                    exc,
                )

    def snapshot(self, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.heartbeat(metrics)


def _read_manifest(path: Path, *, now: datetime, stale_after_seconds: int) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != SESSION_SCHEMA_VERSION:
        return None
    try:
        session_id = _validate_path_component(
            str(payload.get("session_id") or ""), label="session_id"
        )
        instance_id = _validate_path_component(
            str(payload.get("instance_id") or ""), label="instance_id"
        )
        agent_type = _validate_path_component(
            str(payload.get("agent_type") or "unknown"), label="agent_type"
        )
    except ValueError:
        return None
    heartbeat = _parse_iso(payload.get("last_heartbeat_at"))
    if heartbeat is None:
        return None
    started_at = _parse_iso(payload.get("started_at")) or heartbeat
    stale = (now - heartbeat).total_seconds() > stale_after_seconds
    cluster_payload = payload.get("cluster")
    cluster: dict[str, Any] = {"enabled": False, "cluster_id": "default"}
    if isinstance(cluster_payload, dict):
        try:
            cluster_id = _validate_path_component(
                str(cluster_payload.get("cluster_id") or "default"),
                label="cluster_id",
            )
        except ValueError:
            return None
        cluster = {
            "enabled": bool(cluster_payload.get("enabled", False)),
            "cluster_id": cluster_id,
        }
    sanitized: dict[str, Any] = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": session_id,
        "instance_id": instance_id,
        "agent_type": agent_type,
        "started_at": _to_iso(started_at),
        "last_heartbeat_at": _to_iso(heartbeat),
        "cluster": cluster,
        "metrics": _aggregate_metrics(payload.get("metrics")),
        "stale": stale,
        "age_seconds": max(0, round((now - heartbeat).total_seconds(), 3)),
    }
    pid = payload.get("pid")
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        sanitized["pid"] = pid
    return sanitized


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
                try:
                    manifest.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "event=active_session_manifest_cleanup_failed scope=stale path=%s error=%s",
                        manifest,
                        exc,
                    )
            continue
        sessions.append(payload)
    sessions.sort(key=lambda item: str(item.get("last_heartbeat_at", "")), reverse=True)
    return sessions


def aggregate_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int | float] = {
        "requests": 0,
        "tokens_saved": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    by_agent: dict[str, int] = {}
    by_instance: dict[str, int] = {}

    def metric_value(metrics: dict[str, Any], key: str) -> int | float:
        value = metrics.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return 0
        return value if math.isfinite(value) else 0

    for session in sessions:
        raw_metrics = session.get("metrics")
        metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
        totals["requests"] += metric_value(metrics, "requests")
        totals["tokens_saved"] += metric_value(metrics, "tokens_saved")
        totals["input_tokens"] += metric_value(metrics, "input_tokens")
        totals["output_tokens"] += metric_value(metrics, "output_tokens")
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
