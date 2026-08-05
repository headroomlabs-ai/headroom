"""Cold-prefix cache-miss hook: decide what to rewrite when the prompt cache is dead.

When the prefix cache has lapsed (idle since the last turn exceeded the provider
TTL), forwarding the byte-identical prefix buys nothing — the cache is gone — so
this is the safe moment for rewrites that would otherwise bust a warm cache. What
we rewrite depends on the model's reasoning shape:

* **Plain-text reasoning (Kimi / GLM / DeepSeek-R1)** — reasoning is resent as
  billable text (`reasoning_content` field or inline ``<think>``). On a cold turn
  we can DROP the old reasoning outright (full block), not just Kompress it.
* **Encrypted reasoning (Claude / OpenAI Codex)** — the reasoning is an opaque
  server-side handle billed free/light; touching it saves nothing. Instead, on a
  cold turn, dedupe + drop superseded reads across the (now-unfreezable) prefix.

This module is the *decision* surface (is-it-cold + which-shape); the handlers
apply the chosen rewrite. Never raises.

Cache note: dropping/deduping the prefix is cache-safe here **because the cache is
already dead** — nothing to bust. The one cost is the cold turn re-caches a
smaller prefix, which then benefits every subsequent warm turn until the next
lapse. (For plain-text reasoning, deterministic Kompress-every-turn — see
``thinking_compactor`` — stays cache-stable on warm turns; the cold DROP is the
extra, aggressive step reserved for when the cache is confirmed gone.)
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_MARGIN_SECONDS = 60.0


def is_cold_prefix(
    prefix_tracker: Any,
    *,
    margin_seconds: float = _DEFAULT_MARGIN_SECONDS,
    ttl_seconds: float | None = None,
) -> bool:
    """True when the prompt-cache prefix has (confidently) lapsed.

    Compares the idle gap captured at fetch (``_idle_seconds_at_fetch``) to the
    provider cache TTL. Pass ``ttl_seconds`` to use a KNOWN TTL (e.g. from
    :func:`anthropic_cache_ttl_seconds`, which reads CC's actual 5m/1h config) —
    this is what makes cold detection reliable. When ``ttl_seconds`` is None it
    falls back to the tracker's ``resolved_cache_ttl_seconds()`` (a static
    per-provider guess), reliable only where that default is documented-correct.

    The margin makes us *confident* it's past TTL before treating it as cold — we
    would rather miss a just-expired cache than rewrite a still-warm one (a wrong
    TTL here is exactly what busts a warm cache). Returns False on any error
    (conservative: never assume cold).
    """
    try:
        idle = float(getattr(prefix_tracker, "_idle_seconds_at_fetch", 0.0) or 0.0)
        if ttl_seconds is not None:
            ttl = float(ttl_seconds)
        else:
            ttl_fn = getattr(prefix_tracker, "resolved_cache_ttl_seconds", None)
            if ttl_fn is None:
                return False
            ttl = float(ttl_fn())
    except Exception:
        return False
    return idle > ttl + margin_seconds


_ANTHROPIC_FAMILIES = ("opus", "sonnet", "haiku", "fable", "mythos")
_TRUTHY = ("1", "true", "yes", "on")


def _env_truthy(name: str) -> bool:
    import os

    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _anthropic_family(model: str) -> str | None:
    m = model.lower()
    for fam in _ANTHROPIC_FAMILIES:
        if fam in m:
            return fam
    return None


def _cache_control_ttls(messages: list[dict[str, Any]], system: Any) -> set[str]:
    """Collect explicit ``cache_control.ttl`` strings from a client request.

    Anthropic prompt caching: ``{"type":"ephemeral"}`` is the 5m default;
    ``{"type":"ephemeral","ttl":"1h"}`` (with the extended-cache-ttl beta) is 1h.
    We read whatever Claude Code actually sent — the authoritative TTL signal.
    """
    ttls: set[str] = set()

    def _scan_blocks(blocks: Any) -> None:
        if not isinstance(blocks, list):
            return
        for b in blocks:
            if isinstance(b, dict):
                cc = b.get("cache_control")
                if isinstance(cc, dict) and isinstance(cc.get("ttl"), str):
                    ttls.add(cc["ttl"])

    _scan_blocks(system)
    for m in messages:
        if not isinstance(m, dict):
            continue
        cc = m.get("cache_control")
        if isinstance(cc, dict) and isinstance(cc.get("ttl"), str):
            ttls.add(cc["ttl"])
        _scan_blocks(m.get("content"))
    return ttls


def anthropic_cache_ttl_seconds(
    model: str, messages: list[dict[str, Any]], system: Any = None
) -> int | None:
    """The prompt-cache TTL Claude Code is actually using — not a guess.

    Returns:
        * ``None`` — prompt caching is OFF (``DISABLE_PROMPT_CACHING`` or the
          per-model ``DISABLE_PROMPT_CACHING_<FAMILY>``). With no cache there is
          nothing to bust, so the caller can recompact EVERY turn.
        * ``3600`` — 1h caching (request ``cache_control.ttl == "1h"``, or
          ``ENABLE_PROMPT_CACHING_1H``).
        * ``300`` — 5m caching (default, ``FORCE_PROMPT_CACHING_5M``, or
          ``cache_control.ttl == "5m"``).

    Priority: the request's ``cache_control.ttl`` is authoritative for 1h-vs-5m
    (it already reflects CC's env config + overage checks and needs no env
    sharing); the env vars are the OFF signal + a fallback. Reading a wrong TTL
    is exactly what would bust a warm cache, so this replaces the hardcoded 300s
    guess for CC. Never raises.
    """
    try:
        if _env_truthy("DISABLE_PROMPT_CACHING"):
            return None
        fam = _anthropic_family(model)
        if fam and _env_truthy(f"DISABLE_PROMPT_CACHING_{fam.upper()}"):
            return None
        ttls = _cache_control_ttls(messages, system)
        if "1h" in ttls:
            return 3600
        if "5m" in ttls:
            return 300
        # cache_control present without an explicit ttl (or none parsed): env hint,
        # else Anthropic's 5m default. (We do NOT infer "off" from absence — a false
        # off would recompact a warm cache every turn.)
        if _env_truthy("FORCE_PROMPT_CACHING_5M"):
            return 300
        if _env_truthy("ENABLE_PROMPT_CACHING_1H"):
            return 3600
        return 300
    except Exception:
        return 300  # safe default on any parse error


def has_plaintext_reasoning(messages: list[dict[str, Any]]) -> bool:
    """True if any assistant turn carries reasoning as PLAIN TEXT we can drop/compress.

    Two shapes: a Kimi-style ``reasoning_content`` field, or an inline
    ``<think>…</think>`` span in string content (GLM / DeepSeek-R1). Encrypted
    reasoning (Claude signature / OpenAI ``encrypted_content``) never appears in
    these forms, so this is False for those — which routes them to the dedupe /
    superseded-read branch instead.
    """
    for m in messages:
        if m.get("role") != "assistant":
            continue
        rc = m.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            return True
        c = m.get("content")
        if isinstance(c, str) and "<think>" in c and "</think>" in c:
            return True
    return False


def cold_recompact_messages(
    messages: list[dict[str, Any]], *, tokenizer: Any, context: str = ""
) -> tuple[list[dict[str, Any]], list[str]]:
    """Lossless whole-prefix recompaction for a confirmed-cold turn.

    Runs a fresh lossless + cross-turn-dedup ContentRouter over the *entire*
    conversation (``frozen_message_count=0``): superseded/stale-read drop +
    verbatim dedupe + lossless folds — the safe, information-preserving rewrites
    (never lossy, so old context the model relies on is not mangled). Used when
    the prompt cache is dead (idle past TTL) and the byte-identical splice would
    preserve nothing. Lossless + prefix-monotonic ⇒ deterministic per content ⇒
    the recompacted prefix re-caches and stays byte-stable on later warm turns.

    Returns (new_messages, transforms_applied). Fail-open: returns the input
    unchanged on any error (never breaks the request).
    """
    try:
        from headroom.transforms.content_router import (
            ContentRouter,
            ContentRouterConfig,
        )

        router = ContentRouter(ContentRouterConfig(lossless=True, enable_cross_turn_dedup=True))
        res = router.apply(list(messages), tokenizer, frozen_message_count=0, context=context)
        return res.messages, list(res.transforms_applied)
    except Exception as e:  # never break the request
        log.warning("cold-prefix recompaction failed (%s); leaving prefix unchanged", e)
        return list(messages), []


def _demo() -> None:
    class _T:
        def __init__(self, idle: float, ttl: float) -> None:
            self._idle_seconds_at_fetch = idle
            self._ttl = ttl

        def resolved_cache_ttl_seconds(self) -> float:
            return self._ttl

    assert is_cold_prefix(_T(400, 300))  # 400 idle > 300 ttl + 60 margin? 400 > 360 ✓
    assert not is_cold_prefix(_T(350, 300))  # 350 < 360 → warm
    assert not is_cold_prefix(_T(10, 300))  # back-to-back → warm
    assert not is_cold_prefix(object())  # missing attrs → conservative False

    assert has_plaintext_reasoning([{"role": "assistant", "reasoning_content": "abc"}])
    assert has_plaintext_reasoning([{"role": "assistant", "content": "<think>x</think> y"}])
    assert not has_plaintext_reasoning([{"role": "assistant", "content": "plain"}])
    assert not has_plaintext_reasoning([{"role": "user", "reasoning_content": "x"}])

    # --- CC cache-TTL detection (read the real TTL, don't guess 300s) ---
    import os as _os

    _1h_msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ],
        }
    ]
    _5m_msg = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    assert anthropic_cache_ttl_seconds("claude-opus-4-6", _1h_msg) == 3600  # request says 1h
    assert anthropic_cache_ttl_seconds("claude-opus-4-6", _5m_msg) == 300  # default 5m

    for _v in (
        "DISABLE_PROMPT_CACHING",
        "DISABLE_PROMPT_CACHING_OPUS",
        "ENABLE_PROMPT_CACHING_1H",
        "FORCE_PROMPT_CACHING_5M",
    ):
        _os.environ.pop(_v, None)
    _os.environ["DISABLE_PROMPT_CACHING"] = "1"
    assert (
        anthropic_cache_ttl_seconds("claude-opus-4-6", _5m_msg) is None
    )  # OFF -> recompact freely
    del _os.environ["DISABLE_PROMPT_CACHING"]
    _os.environ["DISABLE_PROMPT_CACHING_OPUS"] = "1"
    assert anthropic_cache_ttl_seconds("claude-opus-4-6", []) is None  # per-model OFF
    assert anthropic_cache_ttl_seconds("claude-sonnet-4-6", []) == 300  # other family unaffected
    del _os.environ["DISABLE_PROMPT_CACHING_OPUS"]
    _os.environ["ENABLE_PROMPT_CACHING_1H"] = "1"
    assert anthropic_cache_ttl_seconds("claude-opus-4-6", []) == 3600  # env 1h
    _os.environ["FORCE_PROMPT_CACHING_5M"] = "1"
    assert anthropic_cache_ttl_seconds("claude-opus-4-6", []) == 300  # 5m forced overrides 1h
    del _os.environ["ENABLE_PROMPT_CACHING_1H"], _os.environ["FORCE_PROMPT_CACHING_5M"]

    # The exact bug: at 400s idle, the hardcoded 300s guess says "cold" (would bust a
    # warm 1h cache); the real 1h TTL correctly says "warm".
    assert is_cold_prefix(_T(400, 300), ttl_seconds=3600) is False  # warm at 1h
    assert is_cold_prefix(_T(400, 300)) is True  # "cold" under the 300s guess (the bug)
    assert is_cold_prefix(_T(3700, 300), ttl_seconds=3600) is True  # genuinely cold past 1h

    print("cold_prefix self-check OK")


if __name__ == "__main__":
    _demo()
