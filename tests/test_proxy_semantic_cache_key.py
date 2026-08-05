"""Regression tests for the proxy SemanticCache key (headroom/proxy/semantic_cache.py).

The key previously hashed only {model, messages}, so two requests with identical
messages but a different system prompt, tool set, or sampling config collided and
the second caller was served the first's response. These tests assert BOTH
directions:

- too loose -> the bug: different response-shaping inputs must produce different
  keys (cross-request contamination).
- too tight -> a hit-rate regression (#327): truly-identical requests must still
  hit, and a moved ``cache_control`` breakpoint must not fragment the key.
"""

from __future__ import annotations

import pytest

from headroom.proxy.semantic_cache import SemanticCache

MESSAGES = [{"role": "user", "content": "hello"}]
MODEL = "claude-haiku-4-5"


def _key(cache: SemanticCache, **kw) -> str:
    return cache._compute_key(MESSAGES, MODEL, **kw)


# --- too loose: different inputs must NOT collide (the bug) --------------------


def test_different_system_distinct_keys():
    cache = SemanticCache()
    assert _key(cache, system="Answer only in French.") != _key(
        cache, system="Answer only in English."
    )


def test_different_tools_distinct_keys():
    cache = SemanticCache()
    tools_a = [{"name": "read", "description": "read a file"}]
    tools_b = [{"name": "bash", "description": "run a command"}]
    assert _key(cache, tools=tools_a) != _key(cache, tools=tools_b)


@pytest.mark.parametrize(
    "field,a,b",
    [
        # sampling
        ("temperature", 0.0, 1.0),
        ("top_p", 0.1, 0.9),
        ("top_k", 10, 40),
        ("max_tokens", 100, 200),
        ("stop", ["STOP"], ["HALT"]),
        # OpenAI response-shaping (the #1473 review additions)
        ("tool_choice", "auto", "none"),
        ("response_format", {"type": "json_object"}, {"type": "text"}),
        ("parallel_tool_calls", True, False),
        ("seed", 1, 2),
        ("presence_penalty", 0.0, 1.5),
        ("frequency_penalty", 0.0, 1.5),
        ("logit_bias", {"50256": -100}, {"50256": 100}),
        ("n", 1, 2),
        ("logprobs", True, False),
        ("top_logprobs", 1, 5),
        ("reasoning_effort", "low", "high"),
        ("verbosity", "low", "high"),
        ("modalities", ["text"], ["text", "audio"]),
        # Anthropic response-shaping
        ("thinking", {"type": "enabled", "budget_tokens": 1024}, {"type": "disabled"}),
        ("output_config", {"format": "json"}, {"format": "text"}),
    ],
)
def test_response_shaping_fields_distinct_keys(field, a, b):
    cache = SemanticCache()
    assert _key(cache, **{field: a}) != _key(cache, **{field: b})


# --- too tight: identical / canonically-equal inputs MUST hit -----------------


def test_identical_request_same_key():
    cache = SemanticCache()
    kw = {"system": "sys", "tools": [{"name": "t"}], "temperature": 0.5, "max_tokens": 50}
    assert _key(cache, **kw) == _key(cache, **kw)


def test_legacy_call_stable_with_itself():
    """Backward-compat: a call passing no new fields is stable (existing callers)."""
    cache = SemanticCache()
    assert _key(cache) == _key(cache)


def test_cache_control_breakpoint_move_same_key():
    """Claude Code moves the cache_control breakpoint between turns; a moved
    breakpoint on the system prompt must not fragment the key."""
    cache = SemanticCache()
    system_with_cc = [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    system_without_cc = [{"type": "text", "text": "sys"}]
    assert _key(cache, system=system_with_cc) == _key(cache, system=system_without_cc)


def test_tools_cache_control_ignored():
    cache = SemanticCache()
    tools_cc = [{"name": "t", "cache_control": {"type": "ephemeral"}}]
    tools_plain = [{"name": "t"}]
    assert _key(cache, tools=tools_cc) == _key(cache, tools=tools_plain)


# --- behavioral get/set: collision prevented end to end -----------------------


async def test_get_set_collision_prevented():
    """Store under system A; fetching with system B is a MISS (no contamination),
    fetching with system A is a HIT."""
    cache = SemanticCache()
    await cache.set(MESSAGES, MODEL, b"french-body", {}, system="Answer only in French.")

    miss = await cache.get(MESSAGES, MODEL, system="Answer only in English.")
    assert miss is None

    hit = await cache.get(MESSAGES, MODEL, system="Answer only in French.")
    assert hit is not None
    assert hit.response_body == b"french-body"
