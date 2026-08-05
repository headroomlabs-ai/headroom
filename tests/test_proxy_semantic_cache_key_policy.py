"""Tests for pure proxy semantic-cache key policy."""

from __future__ import annotations

from headroom.proxy.semantic_cache_key import (
    compute_semantic_cache_key,
    strip_cache_control,
)

MESSAGES = [{"role": "user", "content": "hello"}]
MODEL = "claude-haiku-4-5"


def test_semantic_cache_key_distinguishes_response_shaping_fields() -> None:
    assert compute_semantic_cache_key(MESSAGES, MODEL, system="French") != (
        compute_semantic_cache_key(MESSAGES, MODEL, system="English")
    )
    assert compute_semantic_cache_key(MESSAGES, MODEL, temperature=0.0) != (
        compute_semantic_cache_key(MESSAGES, MODEL, temperature=1.0)
    )


def test_semantic_cache_key_ignores_moved_cache_control() -> None:
    with_cache_control = [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    without_cache_control = [{"type": "text", "text": "sys"}]

    assert compute_semantic_cache_key(MESSAGES, MODEL, system=with_cache_control) == (
        compute_semantic_cache_key(MESSAGES, MODEL, system=without_cache_control)
    )


def test_strip_cache_control_recurses_through_dicts_and_lists() -> None:
    assert strip_cache_control(
        {
            "system": [
                {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
            ],
            "tools": [{"name": "read", "cache_control": {"type": "ephemeral"}}],
        }
    ) == {
        "system": [{"type": "text", "text": "sys"}],
        "tools": [{"name": "read"}],
    }
