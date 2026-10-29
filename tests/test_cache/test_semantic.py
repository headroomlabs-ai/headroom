"""Tests for SemanticCache and SemanticCacheLayer."""

import hashlib
import time
from datetime import datetime

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

    def test_same_query_different_context_does_not_collide(self, cache):
        """Two requests that share a trailing user message but differ in earlier
        context (distinct messages_hash) must not overwrite each other. Before the
        fix both were keyed by sha256(query), so the second clobbered the first and
        the first's hash resolved to the second's response."""
        cache.put("run the tests", {"text": "response A"}, messages_hash="ctxA")
        cache.put("run the tests", {"text": "response B"}, messages_hash="ctxB")

        got_a = cache.get("run the tests", messages_hash="ctxA")
        got_b = cache.get("run the tests", messages_hash="ctxB")

        assert got_a is not None and got_a.response == {"text": "response A"}
        assert got_b is not None and got_b.response == {"text": "response B"}

    def test_exact_match_verifies_messages_hash(self, cache):
        """A stored entry is only returned when its messages_hash matches the
        looked-up hash — never another conversation's cached response."""
        cache.put("continue", {"text": "A"}, messages_hash="hA")
        # A lookup for a hash that isn't stored is a miss, not a wrong hit.
        assert cache.get("continue", messages_hash="hB") is None

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

    def test_update_at_capacity_does_not_evict_unrelated_entry(self):
        """Re-storing an existing key at capacity must not drop another entry.

        The eviction loop used to run before the cache key was computed, so
        overwriting a key that was already present (a retried/duplicate store)
        still evicted the LRU-oldest distinct entry even though the update grows
        nothing. That silently dropped a live entry and turned a later lookup for
        it into a false miss.
        """
        config = SemanticCacheConfig(max_entries=2)
        cache = SemanticCache(config)

        cache.put("query1", "response1", messages_hash="h1")
        cache.put("query2", "response2", messages_hash="h2")

        # Re-store the already-present h2 (e.g. a duplicate/retried request).
        cache.put("query2", "response2b", messages_hash="h2")

        # h1 must still be there — updating h2 must not evict it.
        got1 = cache.get("query1", messages_hash="h1")
        assert got1 is not None and got1.response == "response1"
        # h2 reflects the update.
        got2 = cache.get("query2", messages_hash="h2")
        assert got2 is not None and got2.response == "response2b"

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

    def test_empty_query_never_semantic_matches(self):
        """An empty extracted query must not trigger cross-context false hits.

        The query is the last user message; in agent/tool traffic most turns are
        tool_result continuations whose extracted query is "". A real embedder
        maps "" to a fixed non-zero vector, so without a guard every empty-query
        turn would be ~identical to every other and serve one conversation's
        response to an unrelated one. An empty query may only ever hit via the
        exact messages_hash (which is context-complete).
        """

        def const_embedding(text: str) -> list[float]:
            # Realistic: a non-zero, identical vector for every input (incl. "").
            return [0.5, 0.5, 0.5]

        config = SemanticCacheConfig(similarity_threshold=0.9)
        cache = SemanticCache(config, embedding_fn=const_embedding)

        # Conversation A: an empty-query turn (unique full-context hash).
        cache.put("", "response-A", messages_hash="ctxA")

        # Conversation B: a different empty-query turn — must NOT get A's answer.
        assert cache.get("", messages_hash="ctxB") is None
        # Its own exact hash still works.
        assert cache.get("", messages_hash="ctxA").response == "response-A"

        # A whitespace-only query is treated the same as empty.
        cache.put("   \n\t", "response-C", messages_hash="ctxC")
        assert cache.get("   ", messages_hash="ctxD") is None

    def test_namespace_isolation_exact_match(self, cache):
        """Same messages_hash in different namespaces must not collide."""
        cache.put("continue", "response A", messages_hash="shared", namespace="ns-a")
        cache.put("continue", "response B", messages_hash="shared", namespace="ns-b")

        assert (
            cache.get("continue", messages_hash="shared", namespace="ns-a").response == "response A"
        )
        assert (
            cache.get("continue", messages_hash="shared", namespace="ns-b").response == "response B"
        )
        # A third namespace that never stored the hash is a miss.
        assert cache.get("continue", messages_hash="shared", namespace="ns-c") is None

    def test_namespace_isolation_semantic(self):
        """Semantic matches never cross namespace boundaries."""

        def embedding(text: str) -> list[float]:
            return [1.0, 0.0, 0.0] if "weather" in text.lower() else [0.0, 1.0, 0.0]

        config = SemanticCacheConfig(similarity_threshold=0.9)
        cache = SemanticCache(config, embedding_fn=embedding)

        cache.put("What is the weather?", "sunny", namespace="ns-a")

        # Same query, different namespace: no hit.
        assert cache.get("How is the weather?", namespace="ns-b") is None
        # Same namespace: still hits.
        assert cache.get("How is the weather?", namespace="ns-a").response == "sunny"

    def test_query_derived_key_includes_namespace(self, cache):
        """Without messages_hash the query-derived key is still namespace-scoped."""
        key_a = cache.put("run the tests", "response A", namespace="ns-a")
        key_b = cache.put("run the tests", "response B", namespace="ns-b")

        # Different namespaces produce different keys for the same query.
        assert key_a != key_b
        # Re-storing in the same namespace reuses the key (in-place update).
        key_a_again = cache.put("run the tests", "response A2", namespace="ns-a")
        assert key_a_again == key_a

    def test_max_entries_enforced_globally_across_namespaces(self):
        """Eviction counts entries across all namespaces, not per namespace."""
        config = SemanticCacheConfig(max_entries=2)
        cache = SemanticCache(config)

        cache.put("query1", "response1", messages_hash="h1", namespace="ns-a")
        cache.put("query2", "response2", messages_hash="h2", namespace="ns-b")

        # Cache is full: a new entry in a third namespace evicts the LRU entry
        # even though that entry belongs to a different namespace.
        cache.put("query3", "response3", messages_hash="h3", namespace="ns-c")

        assert cache.get("query1", messages_hash="h1", namespace="ns-a") is None
        assert cache.get("query2", messages_hash="h2", namespace="ns-b") is not None
        assert cache.get("query3", messages_hash="h3", namespace="ns-c") is not None


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

    def test_different_model_no_hit(self, layer):
        """Cached responses must not replay across models."""
        messages = [{"role": "user", "content": "Find alpha."}]
        context_a = OptimizationContext(provider="anthropic", model="claude-3-opus")
        context_b = OptimizationContext(provider="anthropic", model="claude-3-sonnet")

        layer.store_response(messages, {"answer": "opus response"}, context_a)

        result = layer.process(messages, context_b)
        assert result.semantic_cache_hit is False

    def test_different_provider_no_hit(self, layer):
        """Cached responses must not replay across providers."""
        messages = [{"role": "user", "content": "Find alpha."}]
        context_a = OptimizationContext(provider="anthropic", model="claude-3-opus")
        context_b = OptimizationContext(provider="bedrock", model="claude-3-opus")

        layer.store_response(messages, {"answer": "opus response"}, context_a)

        result = layer.process(messages, context_b)
        assert result.semantic_cache_hit is False

    def test_different_metadata_no_hit(self, layer):
        """Cached responses must not replay across different metadata."""
        messages = [{"role": "user", "content": "Find alpha."}]
        context_a = OptimizationContext(
            provider="anthropic",
            model="claude-3-opus",
            metadata={"temperature": 0.7},
        )
        context_b = OptimizationContext(
            provider="anthropic",
            model="claude-3-opus",
            metadata={"temperature": 0.9},
        )

        layer.store_response(messages, {"answer": "opus response"}, context_a)

        result = layer.process(messages, context_b)
        assert result.semantic_cache_hit is False

    def test_embedding_different_model_no_hit_same_context_hit(self):
        """Semantic embedding lookup is namespace-scoped too."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        def deterministic_embedding(text: str) -> list[float]:
            digest = hashlib.sha256(text.encode()).digest()
            return [float(b) / 255.0 for b in digest[:4]]

        layer = SemanticCacheLayer(
            AnthropicCacheOptimizer(),
            similarity_threshold=0.9,
            max_entries=100,
            ttl_seconds=60,
            embedding_fn=deterministic_embedding,
        )

        context_a = OptimizationContext(provider="anthropic", model="claude-3-opus")
        context_b = OptimizationContext(provider="anthropic", model="claude-3-sonnet")

        layer.store_response(messages, {"text": "4"}, context_a)

        # Same context: exact messages_hash hit via the embedding path's store.
        assert layer.process(messages, context_a).semantic_cache_hit is True
        # Different model: namespace differs, so no hit even though the query
        # embedding is identical.
        assert layer.process(messages, context_b).semantic_cache_hit is False

    def test_request_id_and_timestamp_do_not_affect_matching(self, layer):
        """request_id and timestamp are not part of the namespace."""
        messages = [{"role": "user", "content": "Hello!"}]
        context_a = OptimizationContext(
            provider="anthropic",
            model="claude-3-opus",
            request_id="req-1",
            timestamp=datetime(2024, 1, 1),
        )
        context_b = OptimizationContext(
            provider="anthropic",
            model="claude-3-opus",
            request_id="req-2",
            timestamp=datetime(2025, 6, 15),
        )

        layer.store_response(messages, {"text": "hi"}, context_a)

        result = layer.process(messages, context_b)
        assert result.semantic_cache_hit is True
        assert result.cached_response == {"text": "hi"}

    def test_unserializable_metadata_store_returns_empty_and_no_hit(self, layer):
        """Unserializable metadata yields a None namespace: store refuses and
        lookup skips the cache entirely."""
        messages = [{"role": "user", "content": "Hello!"}]
        context = OptimizationContext(
            provider="anthropic",
            model="claude-3-opus",
            metadata={"x": object()},
        )

        key = layer.store_response(messages, {"text": "hi"}, context)
        assert key == ""

        result = layer.process(messages, context)
        assert result.semantic_cache_hit is False


class TestSemanticCoverage:
    def test_process_namespace_none_delegates(self):
        from headroom.cache import AnthropicCacheOptimizer, OptimizationContext, SemanticCacheLayer

        layer = SemanticCacheLayer(AnthropicCacheOptimizer())
        messages = [{"role": "user", "content": "hi"}]
        context = OptimizationContext(provider="anthropic", model="m", metadata={"x": object()})
        result = layer.process(messages, context)
        assert result.semantic_cache_hit is False

    def test_invalidate_namespaced_entry(self):
        from headroom.cache.semantic import SemanticCache, SemanticCacheConfig

        cache = SemanticCache(SemanticCacheConfig())
        key = cache.put("q", "r", messages_hash="h1", namespace="ns-a")
        assert cache.get("q", messages_hash="h1", namespace="ns-a") is not None
        assert cache.invalidate(key) is True
        assert cache.get("q", messages_hash="h1", namespace="ns-a") is None
        cache.put("q", "r2", messages_hash="h1", namespace="ns-a")
        assert cache.get("q", messages_hash="h1", namespace="ns-a").response == "r2"

    def test_evict_oldest_removes_namespaced_index(self):
        from headroom.cache.semantic import SemanticCache, SemanticCacheConfig

        cache = SemanticCache(SemanticCacheConfig(max_entries=1))
        cache.put("q1", "r1", messages_hash="h1", namespace="ns-a")
        cache.put("q2", "r2", messages_hash="h2", namespace="ns-b")
        assert cache.get("q1", messages_hash="h1", namespace="ns-a") is None

    def test_context_free_store_policy(self):
        from headroom.cache import AnthropicCacheOptimizer, OptimizationContext, SemanticCacheLayer

        layer = SemanticCacheLayer(AnthropicCacheOptimizer())
        messages = [{"role": "user", "content": "Hello!"}]
        layer.store_response(messages, {"text": "hi"})
        assert layer.process(messages, OptimizationContext()).semantic_cache_hit is True
        ctx = OptimizationContext(provider="anthropic", model="claude-3-opus")
        assert layer.process(messages, ctx).semantic_cache_hit is False
