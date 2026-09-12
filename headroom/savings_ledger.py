"""Durable append-only savings event ledger.

Every compression — interactive ``headroom_compress`` MCP calls *and* proxy
requests — appends a single JSON line to a file-locked JSONL ledger. Unlike the
in-memory ``SessionStats`` and the 2-hour ``session_stats.jsonl`` window, this
ledger survives proxy/agent restarts and is safe across concurrent writers
(the main MCP server, each subagent's MCP server, and the proxy all append to
the same file under an advisory lock). ``headroom savings`` aggregates it on
read, so there is no shared mutable state to clobber and totals stay accurate.

Cost is computed and stored at write time so historical numbers do not drift if
model pricing changes later. litellm list pricing is used where the model is
known; a blended input-token rate is used as a fallback (MCP-tool compressions
do not know the agent's upstream model, so they record ``model="unknown"`` and
fall back to the blended rate rather than recording ``$0``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from headroom import paths as _paths

# Reuse the proxy tracker's pricing + normalization so MCP and proxy events
# bucket models identically and price them through one implementation.
from headroom.proxy.savings_tracker import (
    _estimate_compression_savings_usd,
    _normalize_model,
    _parse_timestamp,
    sanitize_project_name,
)

# fcntl is Unix-only; on Windows we skip locking (append is still best-effort).
fcntl: Any | None = None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

SCHEMA_VERSION = 1
UNKNOWN = "unknown"

# Report windows never look back further than 30 days, and events older than
# this are pruned from disk, keeping the JSONL small.
MAX_RETENTION_DAYS = 30
DEFAULT_RETENTION_DAYS = 30
# Blended input price ($/token) used only when litellm cannot price the model.
# Mirrors the ~$3 / 1M input-token assumption the MCP stats path already uses.
DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000

# Disk hygiene: compact the ledger once it grows past this size. Retention is
# also enforced on read, so accuracy never depends on compaction having run.
# Small because retention is only 30 days — the file should never be large.
_COMPACT_SIZE_BYTES = 1 * 1024 * 1024


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return aware.astimezone(timezone.utc)
    parsed = _parse_timestamp(value)
    if parsed is not None:
        return parsed
    return _utc_now()


def _label(value: Any) -> str:
    """Sanitize a free-form client label, defaulting to ``unknown``."""

    cleaned = sanitize_project_name(value)
    return cleaned or UNKNOWN


def _resolve_path(path: str | os.PathLike[str] | None) -> Path:
    return _paths.savings_events_path(path)


def estimate_cost_usd(
    model: str,
    tokens_saved: int,
    *,
    fallback_rate: float = DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
) -> float:
    """Dollar value of saved input tokens.

    Uses litellm list pricing when the model resolves; otherwise falls back to a
    blended per-token rate so unknown-model traffic still accrues a non-zero
    cost-avoided figure.
    """

    if tokens_saved <= 0:
        return 0.0
    # Skip the litellm lookup for unknown models: it can't price them and emits
    # noisy "Provider List" warnings, and the MCP path is unknown-model by far
    # the most often. Go straight to the blended fallback.
    if model and model != UNKNOWN:
        # Trust _estimate_compression_savings_usd's return verbatim: it already
        # falls back to the blended rate for models litellm can't price, and
        # returns a legitimate 0.0 for genuinely free (0-priced) models. A
        # `priced > 0` gate here re-billed those free models at the $3/M fallback
        # -- phantom cost-avoided -- undoing the same fix already applied inside
        # that helper.
        return round(_estimate_compression_savings_usd(model, tokens_saved), 6)
    return round(float(tokens_saved) * float(fallback_rate), 6)


def record_savings_event(
    *,
    tokens_before: int,
    tokens_after: int,
    model: Any = None,
    client: Any = None,
    source: str = "mcp",
    timestamp: Any = None,
    cost_usd: float | None = None,
    fallback_rate: float = DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
    path: str | os.PathLike[str] | None = None,
    new_input_tokens: int | None = None,
    deferred_tokens: int = 0,
) -> bool:
    """Append one savings event to the durable ledger. Never raises.

    Returns ``True`` when a line was written. ``cost_usd`` is computed from the
    model + tokens saved when not supplied by the caller.

    ``new_input_tokens`` is the provider-billed input that newly entered
    context this request (uncached + cache-write), the denominator /stats
    reports as ``new_input_savings_percent``. ``deferred_tokens`` is the part
    of ``saved`` that is tool-schema deferral, which rides the cached prefix
    and is never part of new input; it is subtracted from the numerator so the
    ledger pairs compression-only savings with new input exactly as /stats
    does. Both are optional so MCP-tool events and older callers keep writing
    the same line they always did.

    A request that saved nothing but carries ``new_input_tokens`` is still
    written, as a denominator-only observation: the new-input basis must see
    every request that newly billed input, not only the ones compression
    touched, or a 100-saved/100-new turn followed by a 0-saved/10,000-new turn
    reads as 50 percent instead of under 1. Without ``new_input_tokens`` a
    zero-saving request is skipped exactly as before.
    """

    try:
        before = max(int(tokens_before), 0)
        after = max(int(tokens_after), 0)
    except (TypeError, ValueError):
        return False

    saved = max(before - after, 0)
    if saved <= 0 and new_input_tokens is None:
        return False

    model_label = _normalize_model(model)
    if cost_usd is None:
        cost = estimate_cost_usd(model_label, saved, fallback_rate=fallback_rate)
    else:
        try:
            cost = max(float(cost_usd), 0.0)
        except (TypeError, ValueError):
            cost = 0.0

    event = {
        "v": SCHEMA_VERSION,
        "ts": _coerce_timestamp(timestamp).isoformat(),
        "before": before,
        "after": after,
        "saved": saved,
        "cost_usd": round(cost, 6),
        "model": model_label,
        "client": _label(client),
        "source": str(source or UNKNOWN),
        "pid": os.getpid(),
    }
    if new_input_tokens is not None:
        try:
            event["new_input"] = max(int(new_input_tokens), 0)
            event["deferred"] = min(max(int(deferred_tokens), 0), saved)
        except (TypeError, ValueError):
            event.pop("new_input", None)
            event.pop("deferred", None)

    target = _resolve_path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, separators=(",", ":")) + "\n"
        with open(target, "a", encoding="utf-8") as handle:
            if _HAS_FCNTL and fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(line)
            finally:
                if _HAS_FCNTL and fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
    except Exception:
        return False

    _maybe_compact(target)
    return True


def _read_events(
    path: str | os.PathLike[str] | None,
    *,
    retention_days: int,
    now: datetime,
) -> list[dict[str, Any]]:
    target = _resolve_path(path)
    if not target.exists():
        return []

    cutoff = now - timedelta(days=retention_days) if retention_days else None
    events: list[dict[str, Any]] = []
    try:
        with open(target, encoding="utf-8") as handle:
            if _HAS_FCNTL and fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_SH)
            try:
                for raw in handle:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    parsed = _parse_timestamp(event.get("ts"))
                    if parsed is None:
                        continue
                    if cutoff is not None and parsed < cutoff:
                        continue
                    event["_ts"] = parsed
                    events.append(event)
            finally:
                if _HAS_FCNTL and fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
    except Exception:
        return []
    return events


@dataclass
class _Bucket:
    tokens_saved: int = 0
    tokens_before: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    # Only events that recorded new input contribute to BOTH of these, so the
    # ratio never pairs one event's savings with another event's denominator.
    new_input_tokens: int = 0
    new_input_saved: int = 0

    def add(
        self,
        *,
        saved: int,
        before: int,
        cost: float,
        new_input: int | None = None,
        deferred: int = 0,
    ) -> None:
        if saved <= 0 and new_input is not None:
            # Denominator-only observation (see record_savings_event): it
            # belongs in the new-input basis and nowhere else, so calls,
            # before/saved and cost keep their saved-event semantics.
            self.new_input_tokens += new_input
            return
        self.tokens_saved += saved
        self.tokens_before += before
        self.cost_usd += cost
        self.calls += 1
        if new_input is not None:
            self.new_input_tokens += new_input
            self.new_input_saved += max(saved - deferred, 0)

    @property
    def savings_percent(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        return round(self.tokens_saved / self.tokens_before * 100, 1)

    @property
    def new_input_savings_percent(self) -> float:
        """Compression-only savings as a share of what newly entered context.

        The whole-wire ratio above recounts a session's cached history every
        turn, so long sessions read as ~0% no matter how well compression does
        on new content. Same definition as /stats ``new_input_savings_percent``:
        saved / (new input + saved), since removed tokens never reached the
        provider and are added back to form the baseline.
        """
        if self.new_input_tokens <= 0:
            return 0.0
        return round(self.new_input_saved / (self.new_input_tokens + self.new_input_saved) * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_saved": self.tokens_saved,
            "tokens_before": self.tokens_before,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
            "savings_percent": self.savings_percent,
            "new_input_tokens": self.new_input_tokens,
            "new_input_savings_percent": self.new_input_savings_percent,
        }


@dataclass
class SavingsReport:
    path: str
    schema_version: int
    lifetime: dict[str, Any]
    windows: dict[str, dict[str, Any]]
    by_model: list[dict[str, Any]]
    by_client: list[dict[str, Any]]
    top_model: str = UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "top_model": self.top_model,
            "lifetime": self.lifetime,
            "windows": self.windows,
            "by_model": self.by_model,
            "by_client": self.by_client,
        }


def _ranked(buckets: dict[str, _Bucket], key_name: str) -> list[dict[str, Any]]:
    rows = []
    for name, bucket in buckets.items():
        row = {key_name: name, **bucket.to_dict()}
        rows.append(row)
    rows.sort(key=lambda r: (r["cost_usd"], r["tokens_saved"]), reverse=True)
    return rows


def aggregate_savings(
    path: str | os.PathLike[str] | None = None,
    *,
    now: datetime | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> SavingsReport:
    """Aggregate the durable ledger into lifetime / windowed / per-dimension views."""

    now = now or _utc_now()
    # Hard-cap the lookback at 30 days regardless of caller input (0/None means
    # "use the cap", never "unbounded"), keeping the report bounded and small.
    retention_days = max(1, min(retention_days or MAX_RETENTION_DAYS, MAX_RETENTION_DAYS))
    events = _read_events(path, retention_days=retention_days, now=now)

    # "Today" is local-calendar-day; the 7-day window is a rolling 168h.
    today_cutoff = (
        now.astimezone().replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    )
    week_cutoff = now - timedelta(days=7)

    # All retained events are within the 30-day cap, so this bucket doubles as
    # both the "Last 30 days" window and the (now 30-day-bounded) lifetime view.
    windowed = _Bucket()
    today = _Bucket()
    last_7 = _Bucket()
    by_model: dict[str, _Bucket] = {}
    by_client: dict[str, _Bucket] = {}

    for event in events:
        ts: datetime = event["_ts"]
        saved = max(int(event.get("saved", 0) or 0), 0)
        before = max(int(event.get("before", 0) or 0), 0)
        try:
            cost = max(float(event.get("cost_usd", 0.0) or 0.0), 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        new_input: int | None = None
        deferred = 0
        if event.get("new_input") is not None:
            try:
                new_input = max(int(event.get("new_input") or 0), 0)
                deferred = max(int(event.get("deferred", 0) or 0), 0)
            except (TypeError, ValueError):
                new_input = None

        targets = [windowed]
        if ts >= today_cutoff:
            targets.append(today)
        if ts >= week_cutoff:
            targets.append(last_7)
        targets.append(by_model.setdefault(str(event.get("model") or UNKNOWN), _Bucket()))
        targets.append(by_client.setdefault(str(event.get("client") or UNKNOWN), _Bucket()))
        for bucket in targets:
            bucket.add(
                saved=saved, before=before, cost=cost, new_input=new_input, deferred=deferred
            )

    model_rows = _ranked(by_model, "model")
    top_model = model_rows[0]["model"] if model_rows else UNKNOWN

    return SavingsReport(
        path=str(_resolve_path(path)),
        schema_version=SCHEMA_VERSION,
        lifetime=windowed.to_dict(),
        windows={
            "today": today.to_dict(),
            "last_7_days": last_7.to_dict(),
            "last_30_days": windowed.to_dict(),
        },
        by_model=model_rows,
        by_client=_ranked(by_client, "client"),
        top_model=top_model,
    )


def _maybe_compact(target: Path) -> None:
    """Rewrite the ledger dropping out-of-retention events once it grows large."""

    try:
        if target.stat().st_size <= _COMPACT_SIZE_BYTES:
            return
    except OSError:
        return

    now = _utc_now()
    cutoff = now - timedelta(days=DEFAULT_RETENTION_DAYS)
    try:
        with open(target, "r+", encoding="utf-8") as handle:
            if _HAS_FCNTL and fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                kept: list[str] = []
                handle.seek(0)
                for raw in handle:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        event = json.loads(stripped)
                    except (ValueError, TypeError):
                        continue
                    parsed = _parse_timestamp(event.get("ts"))
                    if parsed is None or parsed < cutoff:
                        continue
                    kept.append(stripped)
                handle.seek(0)
                handle.truncate()
                if kept:
                    handle.write("\n".join(kept) + "\n")
            finally:
                if _HAS_FCNTL and fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
    except Exception:
        return


__all__ = [
    "SCHEMA_VERSION",
    "MAX_RETENTION_DAYS",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN",
    "SavingsReport",
    "estimate_cost_usd",
    "record_savings_event",
    "aggregate_savings",
]
