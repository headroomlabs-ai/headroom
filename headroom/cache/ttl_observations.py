"""Cache-TTL learning seam — the OSS half of the cross-provider TTL learner.

The proxy already attributes each turn's cache outcome
(:meth:`PrefixCacheTracker.classify_cache_miss`: ``hit`` / ``ttl_expiry`` /
``prefix_change`` / ``cold_start``). This module is the thin seam around that:

1. :func:`record_cache_observation` appends each attribution to a local JSONL
   (``cache_ttl_observations.jsonl``) when ``HEADROOM_CACHE_TTL_LEARN`` is set —
   one row per turn: provider, model, idle, reason, hit/miss.
2. :func:`resolve_learned_ttl` reads a per-provider/model TTL table
   (``cache_ttl_learned.json``) that an **offline** estimator (the
   ``headroom-cache-ttl`` plugin) writes from those observations.

``is_cold_prefix`` consults the learned TTL so cold detection is accurate for
providers that don't expose their TTL (Kimi / OpenAI / Codex). Claude Code reads
its TTL from config directly (see :func:`cold_prefix.anthropic_cache_ttl_seconds`),
so it doesn't need this — but its observations still feed the learner as ground
truth. The estimator lives out-of-process (offline batch over the JSONL) so there
is zero live-feedback risk on the hot path; here we only record + read. Every
function is best-effort and never raises.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_TRUTHY = ("1", "true", "yes", "on")
_LEARN_ENV = "HEADROOM_CACHE_TTL_LEARN"
_HEADROOM_DIR = os.path.expanduser("~/.headroom")
_OBS_DEFAULT = os.path.join(_HEADROOM_DIR, "cache_ttl_observations.jsonl")
_LEARNED_DEFAULT = os.path.join(_HEADROOM_DIR, "cache_ttl_learned.json")
# Bound the observation log so it can never grow unbounded (a prerequisite for ever
# defaulting learning on). When exceeded, rotate to ".1" (single backup); the
# estimator reads the current window, which is plenty for a TTL boundary.
_OBS_MAX_BYTES = 50 * 1024 * 1024


def observations_enabled() -> bool:
    return os.environ.get(_LEARN_ENV, "").strip().lower() in _TRUTHY


def _obs_path() -> str:
    return os.environ.get("HEADROOM_CACHE_TTL_OBS_PATH") or _OBS_DEFAULT


def _learned_path() -> str:
    return os.environ.get("HEADROOM_CACHE_TTL_LEARNED_PATH") or _LEARNED_DEFAULT


def record_cache_observation(*, provider: str, model: str, attribution: Any) -> None:
    """Append one turn's cache outcome to the observation log (best-effort).

    Gated by ``HEADROOM_CACHE_TTL_LEARN`` (off by default → no file writes).
    ``attribution`` is a ``CacheMissAttribution`` (duck-typed via getattr, so a
    plain object works too). Records hits AND misses — the estimator needs the
    idle gaps that still HIT to bound the TTL from below, and the ``ttl_expiry``
    misses to bound it from above. Never raises.
    """
    if not observations_enabled():
        return
    # Respect the global no-filesystem-writes switch (safe to default learning on
    # without forcing writes in stateless/CI deployments).
    if os.environ.get("HEADROOM_STATELESS", "").strip().lower() in _TRUTHY:
        return
    try:
        path = _obs_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if os.path.getsize(path) > _OBS_MAX_BYTES:
                os.replace(path, path + ".1")  # bounded: single-backup rotation
        except OSError:
            pass
        row = {
            "ts": round(time.time(), 3),
            "provider": provider,
            "model": model,
            "reason": getattr(attribution, "reason", None),
            "idle_seconds": round(float(getattr(attribution, "idle_seconds", 0.0) or 0.0), 1),
            "ttl_assumed": int(getattr(attribution, "cache_ttl_seconds", 0) or 0),
            "is_miss": bool(getattr(attribution, "is_miss", False)),
            "cache_read": int(getattr(attribution, "cache_read_tokens", 0) or 0),
            "expected_cached": int(getattr(attribution, "expected_cached_tokens", 0) or 0),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass  # observability must never break a request


_learned_cache: dict[str, Any] | None = None
_learned_mtime: float = -1.0


def resolve_learned_ttl(provider: str, model: str) -> int | None:
    """Learned TTL (seconds) for ``(provider, model)``, or None if not learned yet.

    Reads ``cache_ttl_learned.json`` (written by the offline estimator), keyed by
    ``"provider/model"`` first, then ``"provider"``. Cached by mtime so a hot-path
    call is a dict lookup. Best-effort: returns None on any error, so the caller
    falls back to its own default (config TTL for CC, static default elsewhere).
    """
    global _learned_cache, _learned_mtime
    path = _learned_path()
    try:
        mt = os.path.getmtime(path)
        if _learned_cache is None or mt != _learned_mtime:
            with open(path, encoding="utf-8") as f:
                _learned_cache = json.load(f)
            _learned_mtime = mt
    except Exception:
        return None
    table = _learned_cache or {}
    for key in (f"{provider}/{model}", provider):
        v = table.get(key)
        if isinstance(v, dict) and isinstance(v.get("ttl_seconds"), int | float):
            return int(v["ttl_seconds"])
        if isinstance(v, int | float):
            return int(v)
    return None


def _demo() -> None:
    import tempfile

    d = tempfile.mkdtemp()
    os.environ["HEADROOM_CACHE_TTL_OBS_PATH"] = os.path.join(d, "obs.jsonl")
    os.environ["HEADROOM_CACHE_TTL_LEARNED_PATH"] = os.path.join(d, "learned.json")

    class _Attr:
        reason = "ttl_expiry"
        idle_seconds = 420.0
        cache_ttl_seconds = 300
        is_miss = True
        cache_read_tokens = 0
        expected_cached_tokens = 9000

    # gated off by default -> no write
    os.environ.pop("HEADROOM_CACHE_TTL_LEARN", None)
    record_cache_observation(provider="openai", model="gpt-5.5", attribution=_Attr())
    assert not os.path.exists(os.environ["HEADROOM_CACHE_TTL_OBS_PATH"]), (
        "must not write when disabled"
    )

    os.environ["HEADROOM_CACHE_TTL_LEARN"] = "1"
    record_cache_observation(provider="openai", model="gpt-5.5", attribution=_Attr())
    rows = [json.loads(x) for x in open(os.environ["HEADROOM_CACHE_TTL_OBS_PATH"])]
    assert rows[0]["reason"] == "ttl_expiry" and rows[0]["idle_seconds"] == 420.0, rows[0]

    # learned-table read (exact model, then provider fallback)
    with open(os.environ["HEADROOM_CACHE_TTL_LEARNED_PATH"], "w") as f:
        json.dump({"openai/gpt-5.5": {"ttl_seconds": 1800}, "openai": 900}, f)
    assert resolve_learned_ttl("openai", "gpt-5.5") == 1800
    assert resolve_learned_ttl("openai", "gpt-4o") == 900  # provider fallback
    assert resolve_learned_ttl("anthropic", "claude-x") is None  # not learned

    print("ttl_observations self-check OK")


if __name__ == "__main__":
    _demo()
