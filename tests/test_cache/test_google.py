"""Tests for GoogleCacheOptimizer."""

from datetime import datetime, timedelta, timezone

import pytest

from headroom.cache import CacheConfig, GoogleCacheOptimizer, OptimizationContext
from headroom.cache.base import CacheStrategy
from headroom.cache.google import (
    GOOGLE_CACHE_DISCOUNT,
    GOOGLE_MIN_CACHE_TOKENS,
    CacheabilityAnalysis,
    CachedContentInfo,
)


class TestGoogleCacheOptimizer:
    """Test GoogleCacheOptimizer functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return GoogleCacheOptimizer()

    @pytest.fixture
    def context(self):
        """Create optimization context."""
        return OptimizationContext(
            provider="google",
            model="gemini-1.5-pro",
        )

    def test_optimizer_properties(self, optimizer):
        """Test optimizer properties."""
        assert optimizer.name == "google-cached-content"
        assert optimizer.provider == "google"
        assert optimizer.strategy == CacheStrategy.CACHED_CONTENT

    def test_enforces_minimum_tokens(self):
        """Test that Google's minimum is enforced."""
        config = CacheConfig(min_cacheable_tokens=100)
        optimizer = GoogleCacheOptimizer(config)
        assert optimizer.config.min_cacheable_tokens >= GOOGLE_MIN_CACHE_TOKENS

    def test_optimize_below_threshold(self, optimizer, context):
        """Test optimization with content below threshold."""
        messages = [
            {"role": "system", "content": "Short system prompt."},
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        assert result.metrics.cacheable_tokens < GOOGLE_MIN_CACHE_TOKENS
        assert any("32K" in w for w in result.warnings)

    def test_optimize_above_threshold(self, optimizer, context):
        """Test optimization with content above threshold."""
        # Create content above 32K tokens
        messages = [
            {"role": "system", "content": "You are helpful. " * 15000},
            {"role": "user", "content": "Hello!"},
        ]

        result = optimizer.optimize(messages, context)

        assert result.metrics.cacheable_tokens > 0
        if result.metrics.cacheable_tokens >= GOOGLE_MIN_CACHE_TOKENS:
            assert result.metrics.estimated_savings_percent == GOOGLE_CACHE_DISCOUNT * 100

    def test_analyze_cacheability(self, optimizer, context):
        """Test cacheability analysis."""
        messages = [
            {"role": "system", "content": "Short"},
            {"role": "user", "content": "Hello!"},
        ]

        analysis = optimizer.analyze_cacheability(messages, context)

        assert isinstance(analysis, CacheabilityAnalysis)
        assert analysis.total_tokens > 0
        assert analysis.tokens_below_minimum > 0
        assert analysis.is_cacheable is False
        assert len(analysis.recommendations) > 0

    def test_cache_registration(self, optimizer):
        """Test registering a cache."""
        cache_info = optimizer.register_cache(
            cache_id="test-cache-123",
            content_hash="abc123",
            token_count=50000,
            expires_at=datetime.now() + timedelta(hours=1),
        )

        assert cache_info.cache_id == "test-cache-123"
        assert cache_info.content_hash == "abc123"
        assert cache_info.token_count == 50000
        assert not cache_info.is_expired
        assert cache_info.ttl_remaining_seconds > 0

    def test_cache_lookup(self, optimizer):
        """Test looking up a registered cache."""
        optimizer.register_cache(
            cache_id="test-cache-456",
            content_hash="def456",
            token_count=50000,
            expires_at=datetime.now() + timedelta(hours=1),
        )

        found = optimizer.get_reusable_cache("def456")
        assert found is not None
        assert found.cache_id == "test-cache-456"

    def test_cache_lookup_expired(self, optimizer):
        """Test that expired caches are not returned."""
        optimizer.register_cache(
            cache_id="expired-cache",
            content_hash="expired123",
            token_count=50000,
            expires_at=datetime.now() - timedelta(hours=1),
        )

        found = optimizer.get_reusable_cache("expired123")
        assert found is None

    def test_cache_registration_accepts_aware_expiry(self, optimizer):
        """A timezone-aware ``expires_at`` must not crash lifecycle checks.

        Google's ``genai.caching.CachedContent.expire_time`` is an aware
        RFC3339 value, and the documented integration registers it directly.
        Comparing it against a naive ``datetime.now()`` raised
        ``TypeError: can't compare offset-naive and offset-aware datetimes``,
        which crashed ``register_cache`` itself (it logs ``ttl_remaining_seconds``)
        and every ``is_expired`` consumer.
        """
        aware_future = datetime.now(timezone.utc) + timedelta(hours=1)
        cache_info = optimizer.register_cache(
            cache_id="aware-cache",
            content_hash="aware123",
            token_count=50000,
            expires_at=aware_future,
        )
        assert not cache_info.is_expired
        assert cache_info.ttl_remaining_seconds > 0
        assert cache_info.age_seconds >= 0

        # Lookup and cleanup iterate is_expired over the registry.
        assert optimizer.get_reusable_cache("aware123") is not None
        assert optimizer.cleanup_expired_caches() == []

    def test_cache_lookup_expired_aware(self, optimizer):
        """An already-expired aware cache is recognized as expired, not crashed."""
        aware_past = datetime.now(timezone.utc) - timedelta(hours=1)
        cache_info = optimizer.register_cache(
            cache_id="aware-expired",
            content_hash="awareexp123",
            token_count=50000,
            expires_at=aware_past,
        )
        assert cache_info.is_expired
        assert optimizer.get_reusable_cache("awareexp123") is None

    def test_from_dict_roundtrip_preserves_aware_expiry(self):
        """``to_dict``/``from_dict`` of an aware timestamp stays comparable."""
        info = CachedContentInfo(
            cache_id="rt",
            content_hash="rt123",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            token_count=50000,
        )
        restored = CachedContentInfo.from_dict(info.to_dict())
        assert restored.is_expired is False
        assert restored.ttl_remaining_seconds > 0

    def test_cache_lookup_insufficient_ttl(self, optimizer):
        """Test that caches with insufficient TTL are not returned."""
        optimizer.register_cache(
            cache_id="short-ttl-cache",
            content_hash="shortttl123",
            token_count=50000,
            expires_at=datetime.now() + timedelta(seconds=30),
        )

        # Default min_ttl is 60 seconds
        found = optimizer.get_reusable_cache("shortttl123")
        assert found is None

    def test_extend_cache_ttl(self, optimizer):
        """Test extending cache TTL."""
        optimizer.register_cache(
            cache_id="extend-cache",
            content_hash="extend123",
            token_count=50000,
            expires_at=datetime.now() + timedelta(hours=1),
        )

        new_expires = datetime.now() + timedelta(hours=2)
        updated = optimizer.extend_cache_ttl("extend-cache", new_expires)

        assert updated is not None
        assert updated.expires_at == new_expires

    def test_remove_cache(self, optimizer):
        """Test removing a cache."""
        optimizer.register_cache(
            cache_id="remove-cache",
            content_hash="remove123",
            token_count=50000,
            expires_at=datetime.now() + timedelta(hours=1),
        )

        removed = optimizer.remove_cache("remove-cache")
        assert removed is True

        found = optimizer.get_reusable_cache("remove123")
        assert found is None

    def test_cleanup_expired_caches(self, optimizer):
        """Test cleaning up expired caches."""
        # Register expired cache
        optimizer.register_cache(
            cache_id="cleanup-expired",
            content_hash="cleanup123",
            token_count=50000,
            expires_at=datetime.now() - timedelta(hours=1),
        )

        expired_ids = optimizer.cleanup_expired_caches()
        assert "cleanup-expired" in expired_ids

    def test_list_caches(self, optimizer):
        """Test listing caches."""
        optimizer.register_cache(
            cache_id="list-cache-1",
            content_hash="list1",
            token_count=50000,
            expires_at=datetime.now() + timedelta(hours=1),
        )
        optimizer.register_cache(
            cache_id="list-cache-2",
            content_hash="list2",
            token_count=50000,
            expires_at=datetime.now() + timedelta(hours=2),
        )

        caches = optimizer.list_caches()
        assert len(caches) >= 2

    def test_get_statistics(self, optimizer):
        """Test getting statistics."""
        optimizer.register_cache(
            cache_id="stats-cache",
            content_hash="stats123",
            token_count=50000,
            expires_at=datetime.now() + timedelta(hours=1),
        )

        stats = optimizer.get_statistics()
        assert "active_caches" in stats
        assert "caches_created" in stats
        assert stats["caches_created"] >= 1

    def test_prepare_cache_creation(self, optimizer, context):
        """Test preparing cache creation parameters."""
        messages = [
            {"role": "system", "content": "You are helpful. " * 15000},
            {"role": "user", "content": "Hello!"},
        ]

        params = optimizer.prepare_cache_creation(messages, context)

        if params is not None:
            assert "contents" in params
            assert "ttl" in params
            assert "display_name" in params

    def test_build_request_with_cache(self, optimizer):
        """Test building request with cache."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]

        request = optimizer.build_request_with_cache(messages, "cache-123")

        assert "cached_content" in request
        assert request["cached_content"] == "cache-123"
        assert "contents" in request

    def test_export_import_registry(self, optimizer):
        """Test exporting and importing cache registry."""
        optimizer.register_cache(
            cache_id="export-cache",
            content_hash="export123",
            token_count=50000,
            expires_at=datetime.now() + timedelta(hours=1),
        )

        exported = optimizer.export_cache_registry()
        assert len(exported) >= 1

        new_optimizer = GoogleCacheOptimizer()
        imported = new_optimizer.import_cache_registry(exported)
        assert imported >= 1

    def test_cached_content_info_serialization(self):
        """Test CachedContentInfo serialization."""
        info = CachedContentInfo(
            cache_id="test",
            content_hash="hash",
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            token_count=50000,
        )

        data = info.to_dict()
        restored = CachedContentInfo.from_dict(data)

        assert restored.cache_id == info.cache_id
        assert restored.content_hash == info.content_hash
        assert restored.token_count == info.token_count
