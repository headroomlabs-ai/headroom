"""Tests for pure proxy semantic cache key policy."""

from __future__ import annotations

from headroom.proxy.semantic_cache import SemanticCache
from headroom.proxy.semantic_cache_key_policy import (
    compute_semantic_cache_key,
    strip_cache_control,
)

MESSAGES = [{"role": "user", "content": "hello"}]
MODEL = "claude-haiku-4-5"


def test_strip_cache_control_recurses_through_dicts_and_lists() -> None:
    payload = {
        "system": [
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
            {"nested": {"cache_control": "drop", "value": 1}},
        ],
        "cache_control": "drop-root",
    }
    assert strip_cache_control(payload) == {
        "system": [
            {"type": "text", "text": "sys"},
            {"nested": {"value": 1}},
        ]
    }


def test_compute_semantic_cache_key_is_stable_for_identical_inputs() -> None:
    kwargs = {"system": "sys", "tools": [{"name": "read"}], "temperature": 0.2}
    assert compute_semantic_cache_key(MESSAGES, MODEL, **kwargs) == compute_semantic_cache_key(
        MESSAGES,
        MODEL,
        **kwargs,
    )


def test_compute_semantic_cache_key_distinguishes_response_shaping_fields() -> None:
    assert compute_semantic_cache_key(
        MESSAGES, MODEL, temperature=0.0
    ) != compute_semantic_cache_key(
        MESSAGES,
        MODEL,
        temperature=1.0,
    )


def test_compute_semantic_cache_key_ignores_moved_cache_control_breakpoints() -> None:
    with_breakpoint = [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    without_breakpoint = [{"type": "text", "text": "sys"}]
    assert compute_semantic_cache_key(
        MESSAGES,
        MODEL,
        system=with_breakpoint,
    ) == compute_semantic_cache_key(
        MESSAGES,
        MODEL,
        system=without_breakpoint,
    )


def test_semantic_cache_private_key_wrapper_delegates_to_policy() -> None:
    cache = SemanticCache()
    kwargs = {"system": "sys", "tools": [{"name": "read"}], "temperature": 0.2}
    assert cache._compute_key(MESSAGES, MODEL, **kwargs) == compute_semantic_cache_key(
        MESSAGES,
        MODEL,
        **kwargs,
    )
