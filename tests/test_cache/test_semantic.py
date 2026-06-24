"""Tests for SemanticCache and SemanticCacheLayer."""

import time

import pytest

from headroom.cache import (
    AnthropicCacheOptimizer,
    OptimizationContext,
    SemanticCache,
    SemanticCacheLayer,
)
from headroom.cache.semantic import SemanticCacheConfig


class TestSemanticCacheConfig:
    """Test SemanticCacheConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SemanticCacheConfig()
        assert config.similarity_threshold == 0.95
        assert config.max_entries == 1000
        assert config.ttl_seconds == 300
        assert config.use_exact_matching is True


class TestSemanticCache:
    """Test SemanticCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        config = SemanticCacheConfig(
            max_entries=10,
            ttl_seconds=60,
        )
        return SemanticCache(config)

    def test_put_and_get_exact_match(self, cache):
        """Test storing and retrieving with exact hash matching."""
        response = {"text": "Hello, how can I help?"}
        cache.put("What is the weather?", response, messages_hash="hash123")

        entry = cache.get("What is the weather?", messages_hash="hash123")
        assert entry is not None
        assert entry.response == response

    def test_get_miss(self, cache):
        """Test cache miss."""
        entry = cache.get("Unknown query", messages_hash="unknown")
        assert entry is None

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        config = SemanticCacheConfig(max_entries=3)
        cache = SemanticCache(config)

        # Fill cache
        cache.put("query1", "response1", messages_hash="h1")
        cache.put("query2", "response2", messages_hash="h2")
        cache.put("query3", "response3", messages_hash="h3")

        # Access query1 to make it recently used
        cache.get("query1", messages_hash="h1")

        # Add new entry, should evict query2 (oldest unused)
        cache.put("query4", "response4", messages_hash="h4")

        # query1 should still be there (recently accessed)
        assert cache.get("query1", messages_hash="h1") is not None
        # query2 should be evicted
        assert cache.get("query2", messages_hash="h2") is None
        # query3 and query4 should be there
        assert cache.get("query3", messages_hash="h3") is not None
        assert cache.get("query4", messages_hash="h4") is not None

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        config = SemanticCacheConfig(ttl_seconds=1)
        cache = SemanticCache(config)

        cache.put("expiring query", "response", messages_hash="exp1")

        # Should be available immediately
        assert cache.get("expiring query", messages_hash="exp1") is not None

        # Wait for TTL
        time.sleep(1.1)

        # Should be expired
        assert cache.get("expiring query", messages_hash="exp1") is None

    def test_invalidate(self, cache):
        """Test invalidating an entry."""
        key = cache.put("query", "response", messages_hash="inv1")

        assert cache.get("query", messages_hash="inv1") is not None

        cache.invalidate(key)

        assert cache.get("query", messages_hash="inv1") is None

    def test_clear(self, cache):
        """Test clearing cache."""
        cache.put("query1", "response1", messages_hash="c1")
        cache.put("query2", "response2", messages_hash="c2")

        cache.clear()

        stats = cache.get_stats()
        assert stats["entries"] == 0

    def test_stats(self, cache):
        """Test statistics."""
        cache.put("query", "response", messages_hash="s1")
        cache.get("query", messages_hash="s1")  # hit
        cache.get("unknown", messages_hash="unknown")  # miss

        stats = cache.get_stats()

        assert stats["entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_access_count(self, cache):
        """Test that access count is tracked."""
        cache.put("query", "response", messages_hash="ac1")

        # Access multiple times
        for _ in range(5):
            entry = cache.get("query", messages_hash="ac1")

        # Initial count is 1, plus 5 accesses = 6
        assert entry.access_count == 6

    def test_semantic_similarity_with_embedding_fn(self):
        """Test semantic similarity with custom embedding function."""

        def mock_embedding(text: str) -> list[float]:
            # Simple mock: return consistent embedding for similar queries
            if "weather" in text.lower():
                return [1.0, 0.0, 0.0]
            elif "time" in text.lower():
                return [0.0, 1.0, 0.0]
            else:
                return [0.0, 0.0, 1.0]

        config = SemanticCacheConfig(similarity_threshold=0.9)
        cache = SemanticCache(config, embedding_fn=mock_embedding)

        # Store a weather query
        cache.put("What is the weather today?", "It's sunny", messages_hash="w1")

        # Similar weather query should hit
        entry = cache.get("How is the weather?")
        assert entry is not None
        assert entry.response == "It's sunny"

        # Different query should miss
        entry = cache.get("What time is it?")
        assert entry is None


class TestSemanticCacheLayer:
    """Test SemanticCacheLayer functionality."""

    @pytest.fixture
    def layer(self):
        """Create cache layer with Anthropic optimizer."""
        optimizer = AnthropicCacheOptimizer()
        return SemanticCacheLayer(
            optimizer,
            similarity_threshold=0.95,
            max_entries=100,
            ttl_seconds=60,
        )

    @pytest.fixture
    def context(self):
        """Create optimization context."""
        return OptimizationContext(
            provider="anthropic",
            model="claude-3-opus",
        )

    def test_process_no_cache_hit(self, layer, context):
        """Test processing with no cache hit."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        result = layer.process(messages, context)

        assert result.semantic_cache_hit is False
        assert result.cached_response is None

    def test_process_with_cache_hit(self, layer, context):
        """Test processing with cache hit."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        # First, store a response
        layer.store_response(messages, {"text": "4"}, context)

        # Now process same messages
        result = layer.process(messages, context)

        assert result.semantic_cache_hit is True
        assert result.cached_response == {"text": "4"}

    def test_store_response(self, layer, context):
        """Test storing a response."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Tell me a joke"},
        ]

        key = layer.store_response(messages, {"text": "Why did..."}, context)

        assert key is not None
        assert len(key) > 0

    def test_get_stats(self, layer, context):
        """Test getting statistics."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        layer.process(messages, context)
        stats = layer.get_stats()

        assert "semantic_cache" in stats
        assert "provider_optimizer" in stats
        assert stats["provider_optimizer"] == "anthropic-cache-optimizer"

    def test_query_extraction(self, layer, context):
        """Test query extraction from messages."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second question"},
        ]

        # Store response
        layer.store_response(messages, {"text": "Response"}, context)

        # The query should be the last user message
        result = layer.process(messages, context)
        assert result.semantic_cache_hit is True

    def test_query_from_context(self, layer):
        """Test using query from context."""
        messages = [
            {"role": "user", "content": "Some message"},
        ]
        context = OptimizationContext(
            query="Specific query for caching",
        )

        layer.store_response(messages, {"text": "Response"}, context)
        result = layer.process(messages, context)

        assert result.semantic_cache_hit is True

    def test_provider_optimizer_fallback(self, layer, context):
        """Test that provider optimizer is used on cache miss."""
        messages = [
            {"role": "system", "content": "You are helpful. " * 500},
            {"role": "user", "content": "New uncached question"},
        ]

        result = layer.process(messages, context)

        # Should have used provider optimizer
        assert result.semantic_cache_hit is False
        # Provider optimizer should have processed
        assert result.metrics.stable_prefix_hash != ""

    def test_content_block_query_extraction(self, layer, context):
        """Test query extraction from content block format."""
        messages = [
            {"role": "system", "content": "System"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Block format question"}],
            },
        ]

        layer.store_response(messages, {"text": "Response"}, context)
        result = layer.process(messages, context)

        assert result.semantic_cache_hit is True
