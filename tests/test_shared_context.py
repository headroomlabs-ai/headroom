"""Tests for SharedContext — compressed inter-agent context sharing."""

from headroom.shared_context import SharedContext


class TestPutGet:
    def test_put_and_get_compressed(self) -> None:
        ctx = SharedContext()
        content = " ".join(f"item_{i}: data value {i} with details" for i in range(100))
        entry = ctx.put("research", content, agent="researcher")
        assert entry.original_tokens > 0
        assert entry.key == "research"
        assert entry.agent == "researcher"

        compressed = ctx.get("research")
        assert compressed is not None
        assert len(compressed) <= len(content)

    def test_get_full(self) -> None:
        ctx = SharedContext()
        content = "short content that may not compress much"
        ctx.put("data", content)
        full = ctx.get("data", full=True)
        assert full == content

    def test_get_missing_key(self) -> None:
        ctx = SharedContext()
        assert ctx.get("nonexistent") is None

    def test_overwrite_key(self) -> None:
        ctx = SharedContext()
        ctx.put("k", "first version")
        ctx.put("k", "second version")
        assert ctx.get("k", full=True) == "second version"

    def test_get_entry_metadata(self) -> None:
        ctx = SharedContext()
        ctx.put("findings", "some data", agent="agent_a")
        entry = ctx.get_entry("findings")
        assert entry is not None
        assert entry.agent == "agent_a"
        assert entry.original_tokens >= 0
        assert isinstance(entry.savings_percent, float)

    def test_get_entry_missing(self) -> None:
        ctx = SharedContext()
        assert ctx.get_entry("missing") is None


class TestExpiry:
    def test_expired_entry_returns_none(self) -> None:
        ctx = SharedContext(ttl=0)  # Expire immediately
        ctx.put("k", "value")
        import time

        time.sleep(0.01)
        assert ctx.get("k") is None

    def test_expired_entry_cleaned_from_get_entry(self) -> None:
        ctx = SharedContext(ttl=0)
        ctx.put("k", "value")
        import time

        time.sleep(0.01)
        assert ctx.get_entry("k") is None


class TestKeys:
    def test_lists_active_keys(self) -> None:
        ctx = SharedContext()
        ctx.put("a", "data a")
        ctx.put("b", "data b")
        keys = ctx.keys()
        assert "a" in keys
        assert "b" in keys

    def test_excludes_expired_keys(self) -> None:
        ctx = SharedContext(ttl=0)
        ctx.put("expired", "gone")
        import time

        time.sleep(0.01)
        assert "expired" not in ctx.keys()


class TestStats:
    def test_stats_aggregates(self) -> None:
        ctx = SharedContext()
        content = " ".join(f"word_{i}" for i in range(50))
        ctx.put("a", content)
        ctx.put("b", content)
        stats = ctx.stats()
        assert stats.entries == 2
        assert stats.total_original_tokens > 0

    def test_stats_empty(self) -> None:
        ctx = SharedContext()
        stats = ctx.stats()
        assert stats.entries == 0
        assert stats.savings_percent == 0.0


class TestEviction:
    def test_evicts_oldest_at_capacity(self) -> None:
        ctx = SharedContext(max_entries=2)
        ctx.put("first", "data 1")
        ctx.put("second", "data 2")
        ctx.put("third", "data 3")  # Should evict "first"
        assert ctx.get("first") is None
        assert ctx.get("second") is not None
        assert ctx.get("third") is not None


class TestClear:
    def test_clear_removes_all(self) -> None:
        ctx = SharedContext()
        ctx.put("a", "x")
        ctx.put("b", "y")
        ctx.clear()
        assert ctx.keys() == []


class TestImport:
    def test_importable_from_headroom(self) -> None:
        from headroom import SharedContext as SC

        assert SC is not None
        ctx = SC()
        assert isinstance(ctx, SharedContext)
