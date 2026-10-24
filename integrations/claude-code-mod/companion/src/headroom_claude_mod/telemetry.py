"""Bounded, read-only adapter for Headroom's retained RequestLog records.

Compatibility baseline: headroomlabs-ai/headroom@a493f559a1a85a5462fc3bf3f363d64c3c9444ce.
The single private integration is RequestLogger._logs (a bounded deque).
No wrappers, monkey patches, callbacks, model calls, or additional transcript store.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections import deque
from dataclasses import is_dataclass
from typing import Any

TAG = "mod-session"
MAX_RECORDS = 10_000
MAX_PREVIEW_CHARS = 32_768
PAGE_CHARS = 1_400
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")
REQUEST_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


class IncompatibleLogger(RuntimeError):
    """The installed Headroom does not meet the inspected adapter contract."""


def text(value: Any, limit: int = 160) -> str:
    return CONTROL.sub("\ufffd", str(value or "")[:limit])


def number(value: Any, *, signed: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if abs(value) > 2**53 - 1 or not math.isfinite(value) or (not signed and value < 0):
        return None
    return value


def snapshot(logger: Any) -> tuple[Any, ...]:
    """Copy references, not message bodies; called synchronously on the event loop."""
    logs = getattr(logger, "_logs", None)
    if not isinstance(logs, deque) or logs.maxlen is None or not 1 <= logs.maxlen <= MAX_RECORDS:
        raise IncompatibleLogger("A bounded Headroom RequestLogger is required")
    entries = tuple(logs)
    if any(
        not is_dataclass(e)
        or not isinstance(getattr(e, "tags", None), dict)
        or not isinstance(getattr(e, "request_id", None), str)
        for e in entries
    ):
        raise IncompatibleLogger("Headroom RequestLog/tags contract has changed")
    return entries


def belongs(entry: Any, session_id: str) -> bool:
    # Deliberately NOT x-headroom-session-id: that header changes cache/session behavior.
    return entry.tags.get(TAG) == session_id


def record(entry: Any) -> dict[str, Any]:
    """Explicit allowlist: no arbitrary tags, errors, prompts, or response bodies."""
    before = number(getattr(entry, "input_tokens_original", None))
    after = number(getattr(entry, "input_tokens_optimized", None))
    saved = number(getattr(entry, "tokens_saved", None), signed=True)
    complete = before is not None and after is not None and saved is not None
    # Anthropic may clamp expansion to zero or subtract cache replay debt.
    # The logger's savings remain authoritative within the raw token delta.
    consistent = complete and (saved == before - after or 0 <= saved <= max(0, before - after))
    request_id = text(getattr(entry, "request_id", None), 128)
    transforms = getattr(entry, "transforms_applied", [])
    return {
        "request_id": request_id,
        "inspectable_id": bool(REQUEST_ID.fullmatch(request_id)),
        "timestamp": text(getattr(entry, "timestamp", None), 64),
        "provider": text(getattr(entry, "provider", None), 80),
        "model": text(getattr(entry, "model", None), 120),
        "before": before,
        "after": after,
        "saved": saved,
        "accounting": "complete" if consistent else "inconsistent" if complete else "missing",
        "percent": round(saved / before * 100, 2) if consistent and before else None,
        "overhead_ms": number(getattr(entry, "optimization_latency_ms", None)),
        "latency_ms": number(getattr(entry, "total_latency_ms", None)),
        "cache_read": number(getattr(entry, "cache_read_tokens", None)),
        "cache_write": number(getattr(entry, "cache_write_tokens", None)),
        "uncached_input": number(getattr(entry, "uncached_input_tokens", None)),
        "output": number(getattr(entry, "output_tokens", None)),
        "failed": bool(getattr(entry, "error", None)),
        "transforms": [text(v, 100) for v in transforms[:16]]
        if isinstance(transforms, list)
        else [],
        "has_messages": isinstance(getattr(entry, "request_messages", None), list)
        and isinstance(getattr(entry, "compressed_messages", None), list),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Errors do not become claimed savings; missing/inconsistent data isn't coerced to zero.
    accounted = [r for r in rows if r["accounting"] == "complete" and not r["failed"]]
    before = sum(r["before"] for r in accounted)
    after = sum(r["after"] for r in accounted)
    saved = sum(r["saved"] for r in accounted)
    timings = [r["overhead_ms"] for r in accounted if r["overhead_ms"] is not None]
    output_rows = [r for r in accounted if r["output"] is not None]
    cache_rows = [
        r
        for r in accounted
        if all(r[k] is not None for k in ("cache_read", "cache_write", "uncached_input"))
    ]
    cache_input = sum(r["cache_read"] + r["cache_write"] + r["uncached_input"] for r in cache_rows)
    cache_read = sum(r["cache_read"] for r in cache_rows)
    transforms: dict[str, int] = {}
    for r in accounted:
        for name in set(r["transforms"]):
            transforms[name] = transforms.get(name, 0) + 1
    return {
        "requests": len(rows),
        "accounted_requests": len(accounted),
        "failed_requests": sum(r["failed"] for r in rows),
        "unaccounted_requests": len(rows) - len(accounted),
        "before": before if accounted else None,
        "after": after if accounted else None,
        "saved": saved if accounted else None,
        "percent": round(saved / before * 100, 2) if before else None,
        "output": sum(r["output"] for r in output_rows) if output_rows else None,
        "output_accounted_requests": len(output_rows),
        "average_overhead_ms": round(sum(timings) / len(timings), 2) if timings else None,
        "cache_read": cache_read if cache_rows else None,
        "cache_write": sum(r["cache_write"] for r in cache_rows) if cache_rows else None,
        "cache_read_percent": round(cache_read / cache_input * 100, 2) if cache_input else None,
        "cache_accounted_requests": len(cache_rows),
        "transforms": dict(sorted(transforms.items(), key=lambda p: (-p[1], p[0]))[:16]),
    }


def preview(value: Any) -> tuple[str, bool]:
    """Bound all walks and strings BEFORE JSON encoding, then bound displayed output."""
    remaining = 1_024
    chars = MAX_PREVIEW_CHARS
    truncated = False

    def visit(v: Any, depth: int = 0) -> Any:
        nonlocal remaining, chars, truncated
        remaining -= 1
        if remaining < 0 or depth > 12 or chars <= 0:
            truncated = True
            return "[preview limit]"
        if isinstance(v, str):
            take = min(chars, 8_192)
            if len(v) > take:
                truncated = True
            result = text(v, take)
            chars -= len(result)
            return result + (" [preview truncated]" if len(v) > take else "")
        if isinstance(v, dict):
            out = {}
            for i, (k, item) in enumerate(v.items()):
                if i >= 128 or remaining <= 0 or chars <= 0:
                    truncated = True
                    out["_preview"] = "additional fields omitted"
                    break
                out[text(k, 120)] = visit(item, depth + 1)
            return out
        if isinstance(v, list):
            out = []
            for i, item in enumerate(v):
                if i >= 128 or remaining <= 0 or chars <= 0:
                    truncated = True
                    out.append("[additional items omitted]")
                    break
                out.append(visit(item, depth + 1))
            return out
        if v is None or isinstance(v, (bool, int, float)):
            return (
                v
                if not isinstance(v, (int, float))
                or isinstance(v, bool)
                or number(v, signed=True) is not None
                else "[nonfinite/oversized number]"
            )
        truncated = True
        return "[unsupported value]"

    result = json.dumps(visit(value), ensure_ascii=False, indent=2)
    if len(result) > MAX_PREVIEW_CHARS:
        result = result[:MAX_PREVIEW_CHARS] + "\n[preview truncated]"
        truncated = True
    return result, truncated


def inspect(entry: Any, *, side: str, message: int, page: int) -> dict[str, Any]:
    original = getattr(entry, "request_messages", None)
    compressed = getattr(entry, "compressed_messages", None)
    if not isinstance(original, list) or not isinstance(compressed, list):
        return {
            "available": False,
            "reason": "Message capture is disabled or this request's snapshots have expired.",
        }
    counts = {"original": len(original), "compressed": len(compressed)}
    if side == "diff":
        # Request-level diff, never a guessed index-pairing across deleted/reordered messages.
        before, bt = preview(original)
        after, at = preview(compressed)
        bl, al = before.splitlines(), after.splitlines()
        truncated = bt or at or len(bl) > 250 or len(al) > 250
        lines = list(
            difflib.unified_diff(
                bl[:250],
                al[:250],
                fromfile="original request",
                tofile="compressed request",
                lineterm="",
            )
        )
        body = "\n".join(lines) or "No differences in the compared preview."
        if len(body) > MAX_PREVIEW_CHARS:
            truncated = True
            body = body[:MAX_PREVIEW_CHARS]
        count = 1
        message = 0
    else:
        values = original if side == "original" else compressed
        count = len(values)
        if message >= count:
            return {
                "available": False,
                "reason": "This side has no message at that index.",
                "counts": counts,
            }
        body, truncated = preview(values[message])
    pages = max(1, math.ceil(len(body) / PAGE_CHARS))
    page = min(page, pages - 1)
    return {
        "available": True,
        "side": side,
        "message": message,
        "message_count": count,
        "counts": counts,
        "page": page,
        "pages": pages,
        "truncated": truncated,
        "text": body[page * PAGE_CHARS : (page + 1) * PAGE_CHARS],
        "note": "Indices belong to each side independently. Diff compares bounded, ordered request snapshots.",
    }
