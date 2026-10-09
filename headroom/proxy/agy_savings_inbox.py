"""Cross-process savings inbox: agy -> shared proxy replay.

``agy`` runs as a **separate process** from the shared Headroom proxy, but its
per-request savings must show up on the shared dashboard, counted once, without
agy ever writing any shared durable state. This module is the bridge.

Mechanism (AT-LEAST-ONCE with best-effort dedup — *not* exactly-once):

* In the agy process, :func:`emit_event` drops one JSON file per request into a
  canonical inbox directory (``workspace_dir()/savings.d``). Each file carries
  the exact keyword arguments that :meth:`PrometheusMetrics.record_request`
  (the single dashboard funnel) expects, plus a unique ``event_id``.
* In the shared proxy, :func:`drain_inbox` reads those files and replays each
  event through its *own* ``record_request`` funnel — the one writer of shared
  durable state (savings ledger, SavingsTracker, OTEL). agy itself redirects all
  three of those to throwaway paths, so the proxy replay is the sole writer and
  savings are counted exactly once on the dashboard.

Everything on the agy side is best-effort: emit never raises into the request
path, and drain never raises out into the proxy's lifespan / stats handler.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from headroom.paths import workspace_dir

logger = logging.getLogger("headroom.proxy")

# Bump when the on-disk envelope shape changes incompatibly.
SCHEMA_VERSION = 1

# Env var (set only in the agy process) that turns on emit at the outcome hook.
AGY_INBOX_EMIT_ENV = "HEADROOM_AGY_INBOX_EMIT"

# Hard cap on pending event files; oldest are dropped (disclosed) past this.
MAX_INBOX = 5000

# Keep at most this many processed ids in the dedup file so it stays bounded.
MAX_PROCESSED_IDS = 20000

_INBOX_SUBDIR = "savings.d"
_PROCESSED_FILE = ".processed"

# Monotonic per-process sequence so two events from the same pid never collide.
_seq = itertools.count()

# Serialize drains so the periodic task and /stats-triggered drain never race.
_drain_lock = asyncio.Lock()


def inbox_dir() -> Path:
    """Return the canonical inbox directory, creating it on demand."""

    path = workspace_dir() / _INBOX_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def agy_emit_enabled() -> bool:
    """True when the agy emit marker env var is set to ``"1"``."""

    return os.environ.get(AGY_INBOX_EMIT_ENV, "").strip() == "1"


def _new_event_id() -> str:
    """Return a process-unique, collision-resistant event id."""

    return f"{os.getpid()}-{next(_seq)}-{random.getrandbits(48):012x}"


def _json_safe(value: Any) -> Any:
    """Return ``value`` if it round-trips through JSON, else ``None``.

    Non-scalar funnel args (``pipeline_timing``, ``waste_signals``) are dicts of
    scalars and normally survive; anything that does not is dropped so a single
    weird value can never make the whole envelope unwritable.
    """

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return None


def _enforce_cap() -> None:
    """Drop the oldest event files if the inbox is at/over :data:`MAX_INBOX`."""

    try:
        files = sorted(
            inbox_dir().glob("evt-*.json"),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return
    excess = len(files) - MAX_INBOX
    if excess <= 0:
        return
    for stale in files[: excess + 1]:
        try:
            stale.unlink()
        except OSError:
            continue
    logger.warning(
        "agy savings inbox at cap (%d); dropped %d oldest event(s)",
        MAX_INBOX,
        excess + 1,
    )


def emit_event(**funnel_kwargs: Any) -> None:
    """Atomically write one inbox event carrying ``record_request`` kwargs.

    Best-effort: any failure is swallowed (logged at debug) so emit can never
    break the request that triggered it.
    """

    try:
        directory = inbox_dir()
        _enforce_cap()

        safe_kwargs = {key: _json_safe(val) for key, val in funnel_kwargs.items()}
        event_id = _new_event_id()
        envelope = {
            "v": SCHEMA_VERSION,
            "event_id": event_id,
            "kwargs": safe_kwargs,
        }

        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".tmp-evt-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(envelope, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, directory / f"evt-{event_id}.json")
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    except Exception as exc:  # noqa: BLE001 - best-effort, never raise into caller
        logger.debug("agy savings emit failed: %s", exc)


def _load_processed(path: Path) -> list[str]:
    """Return the processed-id list (order preserved), or ``[]`` if unreadable."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    ids: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            ids.append(line)
    return ids


def _write_processed(path: Path, ids: list[str]) -> None:
    """Atomically persist the processed-id list, pruned to the newest N."""

    pruned = ids[-MAX_PROCESSED_IDS:]
    try:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-proc-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(pruned))
            if pruned:
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        logger.debug("agy savings processed-file write failed: %s", exc)


async def drain_inbox(metrics: Any, *, max_events: int = 1000) -> int:
    """Replay pending inbox events through ``metrics.record_request``.

    Delivery is at-least-once with best-effort dedup: an event is only unlinked
    *after* it has been recorded (or found already-processed), so a crash
    between record and unlink re-delivers it next drain — the ``.processed`` set
    then suppresses the duplicate. Returns the number of events recorded.

    Never raises: the whole body is defended so a drain error can never crash
    the proxy lifespan loop or the stats handler.
    """

    recorded = 0
    async with _drain_lock:
        try:
            directory = inbox_dir()
            processed_path = directory / _PROCESSED_FILE
            processed_list = _load_processed(processed_path)
            processed_set = set(processed_list)

            try:
                files = sorted(directory.glob("evt-*.json"))
            except OSError:
                return 0

            dirty = False
            for event_file in files[:max_events]:
                try:
                    try:
                        raw = event_file.read_text(encoding="utf-8")
                        envelope = json.loads(raw)
                    except (OSError, ValueError):
                        # Malformed / unreadable: skip and remove, never fatal.
                        logger.debug("agy savings: dropping malformed %s", event_file.name)
                        _safe_unlink(event_file)
                        continue

                    event_id = envelope.get("event_id")
                    if not isinstance(event_id, str) or not event_id:
                        _safe_unlink(event_file)
                        continue

                    if event_id in processed_set:
                        # Crash-window duplicate: already recorded, just remove.
                        _safe_unlink(event_file)
                        continue

                    kwargs = envelope.get("kwargs")
                    if not isinstance(kwargs, dict):
                        _safe_unlink(event_file)
                        continue

                    await metrics.record_request(**kwargs)
                    recorded += 1

                    processed_set.add(event_id)
                    processed_list.append(event_id)
                    dirty = True
                    _safe_unlink(event_file)
                except Exception as exc:  # noqa: BLE001 - one bad event never aborts drain
                    logger.debug("agy savings: error replaying %s: %s", event_file.name, exc)
                    continue

            if dirty:
                _write_processed(processed_path, processed_list)
        except Exception as exc:  # noqa: BLE001 - drain never raises out
            logger.debug("agy savings drain failed: %s", exc)

    return recorded


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


__all__ = [
    "AGY_INBOX_EMIT_ENV",
    "SCHEMA_VERSION",
    "MAX_INBOX",
    "inbox_dir",
    "agy_emit_enabled",
    "emit_event",
    "drain_inbox",
]
