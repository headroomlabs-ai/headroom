"""A request that carries the 5m default cache_control (Claude Code's idle
compaction request) must not read as cold while the conversation's earlier
turns cached with 1h: the prefix is still warm and recompacting it busts it."""

from __future__ import annotations

from headroom.cache.prefix_tracker import PrefixCacheTracker, PrefixFreezeConfig
from headroom.transforms.cold_prefix import is_cold_prefix


def _tracker(idle: float, observed_ttl: int | None) -> PrefixCacheTracker:
    t = PrefixCacheTracker("anthropic", PrefixFreezeConfig())
    t.observe_cache_ttl(observed_ttl)
    t._idle_seconds_at_fetch = idle
    return t


def test_5m_request_on_1h_conversation_is_warm_at_50_minutes():
    assert not is_cold_prefix(_tracker(3039, 3600), ttl_seconds=300)


def test_1h_conversation_is_cold_after_the_hour():
    assert is_cold_prefix(_tracker(3700, 3600), ttl_seconds=300)


def test_5m_conversation_still_goes_cold_after_5_minutes():
    assert is_cold_prefix(_tracker(400, 300), ttl_seconds=300)
    assert is_cold_prefix(_tracker(400, None), ttl_seconds=300)
