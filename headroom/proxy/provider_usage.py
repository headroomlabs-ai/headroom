"""The provider's own billed input-token count, from its own usage block.

Every volume and cost figure Headroom reports ("input after Headroom", the
licence usage report, the dashboard's per-model table) should be the number
the provider billed, not Headroom's local tokenizer estimate. Headroom counts
Claude with ``o200k_base`` as a stand-in for a private tokenizer; on real
Claude Code transcripts that runs 15-25% under Claude 4.x and 35-38% under
Claude 5.x, and it never sees the system prompt or tool definitions at all.

The providers disagree about what their headline input field means, which is
why this lives in one place:

* Anthropic (and Bedrock/Vertex, which forward Anthropic-shape usage):
  ``usage.input_tokens`` is only the tokens AFTER the last cache breakpoint.
  Cache reads and writes are separate, disjoint buckets, so the billed total
  is ``input_tokens + cache_read_input_tokens + cache_creation_input_tokens``.
  The Anthropic Console reports the same three columns.
* OpenAI (chat ``prompt_tokens``, Responses ``input_tokens``): inclusive.
  ``*_tokens_details.cached_tokens`` is a SUBSET of the headline figure.
* Gemini: ``promptTokenCount`` is inclusive; ``cachedContentTokenCount`` is a
  subset of it.

A return of 0 means "the provider reported nothing", never "zero tokens":
callers fall back to the local estimate and the request is counted as
estimated rather than provider-reported (see ``RequestOutcome``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "anthropic_billed_input",
    "is_anthropic_dialect",
    "billed_input_from_usage",
    "billed_input_for_provider",
]

# Providers whose usage blocks are Anthropic-shaped (disjoint cache buckets).
_ANTHROPIC_SHAPE = frozenset({"anthropic", "bedrock", "vertex", "vertex_ai", "claude"})


def usage_int(value: Any, default: int = 0) -> int:
    """One token count from a provider usage block, as a non-negative int.

    Missing (``None``), non-numeric and boolean values give ``default``;
    negative counts clamp to 0. Never raises: usage fields are
    upstream-controlled and must not fail a response.
    """
    if isinstance(value, bool):
        return default
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def anthropic_cache_ttl_buckets(usage: Any) -> tuple[int, int]:
    """Cache-write tokens by TTL bucket ``(5m, 1h)`` from an Anthropic usage block.

    Reads ``usage.cache_creation.ephemeral_{5m,1h}_input_tokens``; returns
    ``(0, 0)`` when the block or its ``cache_creation`` object is absent.
    """
    if not isinstance(usage, dict):
        return (0, 0)
    cache_creation = usage.get("cache_creation")
    if not isinstance(cache_creation, dict):
        return (0, 0)
    return (
        int(cache_creation.get("ephemeral_5m_input_tokens", 0) or 0),
        int(cache_creation.get("ephemeral_1h_input_tokens", 0) or 0),
    )


def is_anthropic_dialect(provider: str | None) -> bool:
    """True for providers whose usage blocks keep cache buckets disjoint.

    Decided by the provider, never by which keys a usage block happens to
    carry: OpenAI-compatible gateways (LiteLLM among them) mirror top-level
    ``cache_read_input_tokens`` beside an INCLUSIVE ``input_tokens``, so the
    keys alone cannot tell the two dialects apart.
    """
    name = (provider or "").lower()
    return name in _ANTHROPIC_SHAPE or "anthropic" in name


def anthropic_billed_input(input_tokens: Any, cache_read: Any = 0, cache_write: Any = 0) -> int:
    """Billed input for an Anthropic-shape usage block: the three disjoint buckets."""
    return usage_int(input_tokens) + usage_int(cache_read) + usage_int(cache_write)


def billed_input_for_provider(
    provider: str | None,
    input_tokens: Any,
    *,
    cache_read: Any = 0,
    cache_write: Any = 0,
) -> int:
    """Billed input from already-extracted usage fields, by provider family.

    ``input_tokens`` is the provider's headline input field exactly as it
    reported it (Anthropic's uncached-only ``input_tokens``, OpenAI's inclusive
    ``prompt_tokens``/``input_tokens``, Gemini's inclusive ``promptTokenCount``).
    Returns 0 when the provider reported no input count at all.
    """
    if input_tokens is None:
        return 0
    if is_anthropic_dialect(provider):
        total = anthropic_billed_input(input_tokens, cache_read, cache_write)
        # message_start can carry input_tokens=0 alongside real cache buckets
        # on a fully cached turn; that is still a provider-reported count.
        return total
    return usage_int(input_tokens)


def billed_input_from_usage(payload: Mapping[str, Any] | None, provider: str | None) -> int:
    """Billed input from a raw provider response (or its ``usage`` block).

    ``provider`` selects the dialect (see ``is_anthropic_dialect``); it is
    required because the same keys mean different things across dialects.
    Gemini's ``usageMetadata`` is unambiguous and recognised by shape. Returns
    0 when no input count is present.
    """
    if not isinstance(payload, Mapping):
        return 0

    meta = payload.get("usageMetadata")
    if isinstance(meta, Mapping):
        return usage_int(meta.get("promptTokenCount"))

    usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else payload
    if not isinstance(usage, Mapping):
        return 0

    if "promptTokenCount" in usage:
        return usage_int(usage.get("promptTokenCount"))

    if is_anthropic_dialect(provider):
        if "input_tokens" not in usage:
            return 0
        return anthropic_billed_input(
            usage.get("input_tokens"),
            usage.get("cache_read_input_tokens"),
            usage.get("cache_creation_input_tokens"),
        )

    # OpenAI and OpenAI-compatible dialects: the headline figure is inclusive.
    if "prompt_tokens" in usage:
        return usage_int(usage.get("prompt_tokens"))
    if "input_tokens" in usage:
        return usage_int(usage.get("input_tokens"))
    return 0
