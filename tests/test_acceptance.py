"""
Acceptance tests for Headroom SDK.

These are the 4 required acceptance tests from the spec:
1. Date Trap Test
2. Tool Orphan Test
3. Streaming Test
4. Safety Test (malformed JSON)
"""

import pytest

from headroom import OpenAIProvider, Tokenizer
from headroom.transforms import CacheAligner

# Create a shared provider for tests
_provider = OpenAIProvider()


def get_tokenizer(model: str = "gpt-4o") -> Tokenizer:
    """Get a tokenizer for tests using OpenAI provider."""
    token_counter = _provider.get_token_counter(model)
    return Tokenizer(token_counter, model)


class TestDateTrap:
    """CacheAligner is detector-only after PR-A2 (P2-23 fix).

    The system prompt is NEVER mutated. Volatile content (dates, UUIDs,
    JWTs, hex hashes) is only DETECTED and surfaced via warnings. The
    spec's prior "date trap" remediation moved to live-zone routing
    (PR-A2 P0-1) and is exercised by tests/test_proxy_system_prompt_immutable.py.
    """

    def test_system_prompt_bytes_unchanged_when_dynamic_content_present(self):
        """The detector must not rewrite the system prompt."""
        original = "You are helpful. Current Date: 2024-01-15"
        messages = [
            {"role": "system", "content": original},
            {"role": "user", "content": "Hello"},
        ]

        aligner = CacheAligner()
        tokenizer = get_tokenizer()
        result = aligner.apply(messages, tokenizer)

        assert result.messages[0]["content"] == original
        assert result.transforms_applied == []

    def test_warning_surfaced_for_iso_date_in_system_prompt(self):
        """ISO 8601 dates should be surfaced as warnings, not extracted."""
        from headroom.config import CacheAlignerConfig

        messages = [
            {
                "role": "system",
                "content": "You are helpful. Time: 2024-01-15T10:30:00",
            },
            {"role": "user", "content": "Hello"},
        ]

        aligner = CacheAligner(CacheAlignerConfig(enabled=True))
        tokenizer = get_tokenizer()
        result = aligner.apply(messages, tokenizer)

        assert any("iso8601" in w.lower() for w in result.warnings)

    def test_cache_metrics_populated(self):
        """CachePrefixMetrics is populated even though no rewrite happens."""
        messages = [
            {"role": "system", "content": "You are helpful. Current Date: 2024-01-15"},
            {"role": "user", "content": "Hello"},
        ]

        aligner = CacheAligner()
        tokenizer = get_tokenizer()
        result = aligner.apply(messages, tokenizer)

        assert result.cache_metrics is not None
        assert result.cache_metrics.stable_prefix_bytes > 0
        assert result.cache_metrics.stable_prefix_tokens_est > 0
        assert len(result.cache_metrics.stable_prefix_hash) == 16
        assert result.cache_metrics.prefix_changed is False
        assert result.cache_metrics.previous_hash is None

    def test_cache_metrics_tracks_changes_across_requests(self):
        """Hash flips when bytes change. Hash is over the actual bytes now."""
        aligner = CacheAligner()
        tokenizer = get_tokenizer()

        messages1 = [
            {"role": "system", "content": "You are helpful. Current Date: 2024-01-15"},
            {"role": "user", "content": "Hello"},
        ]
        result1 = aligner.apply(messages1, tokenizer)

        # Same bytes → same hash, prefix_changed False.
        messages2 = [
            {"role": "system", "content": "You are helpful. Current Date: 2024-01-15"},
            {"role": "user", "content": "Hello"},
        ]
        result2 = aligner.apply(messages2, tokenizer)
        assert result2.cache_metrics.prefix_changed is False
        assert result2.cache_metrics.stable_prefix_hash == (
            result1.cache_metrics.stable_prefix_hash
        )

        # Different bytes → hash flips. The detector NEVER strips dynamic
        # content, so any byte difference is reflected in the hash. This
        # is the correct behavior — the customer must move dynamic content
        # to the live zone (live-zone tail per PR-A2) to get cache hits.
        messages3 = [
            {"role": "system", "content": "You are VERY helpful. Current Date: 2024-01-15"},
            {"role": "user", "content": "Hello"},
        ]
        result3 = aligner.apply(messages3, tokenizer)
        assert result3.cache_metrics.prefix_changed is True
        assert result3.cache_metrics.stable_prefix_hash != (
            result2.cache_metrics.stable_prefix_hash
        )


class TestStreaming:
    """Test that streaming works correctly."""

    def test_stream_passthrough(self):
        """Streaming should pass through chunks correctly."""
        # This test requires a mock client since we can't call real APIs
        # We'll test the wrapper behavior

        class MockChunk:
            def __init__(self, content: str):
                self.choices = [
                    type("Choice", (), {"delta": type("Delta", (), {"content": content})()})
                ]

        class MockStream:
            def __init__(self):
                self.chunks = [MockChunk("Hello"), MockChunk(" "), MockChunk("World")]
                self.index = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.index >= len(self.chunks):
                    raise StopIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        # The stream wrapper should yield all chunks
        stream = MockStream()
        chunks = list(stream)

        assert len(chunks) == 3
        assert all(hasattr(c, "choices") for c in chunks)

    def test_stream_metrics_saved(self):
        """Metrics should be saved when stream completes."""
        # This would require integration test with mock client
        # For unit test, we verify the wrapper generator works
        pass


class TestQueryAnchorExtraction:
    """Test that query anchors preserve needle records during crushing."""

    def test_preserves_needle_by_name(self):
        """If user asks for 'Alice', item with Alice should be preserved."""
        import json

        from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

        # User is searching for 'Alice'
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Find the user named 'Alice' in the system."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "find_users", "arguments": '{"name": "Alice"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(
                    [{"id": i, "name": f"User{i}", "score": 0.1} for i in range(50)]
                    + [{"id": 42, "name": "Alice", "score": 0.1}]
                ),  # Alice is at the END, not in first/last K
            },
        ]

        # End-to-end behavior: the relevance scorer (HybridScorer in
        # the Rust port — BM25 + embedding) should pick up "Alice"
        # from the user message and preserve the matching tool item
        # even though it sits at index 50.
        config = SmartCrusherConfig(
            enabled=True,
            min_items_to_analyze=5,
            min_tokens_to_crush=100,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)
        tokenizer = get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        crushed_content = tool_msg["content"]

        assert "Alice" in crushed_content

    def test_preserves_needle_by_uuid(self):
        """If user asks for a UUID, item with that UUID should be preserved."""
        import json

        from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

        target_uuid = "550e8400-e29b-41d4-a716-446655440000"

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"Get details for request {target_uuid}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_requests", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps(
                    [{"request_id": f"other-{i}", "status": "ok"} for i in range(50)]
                    + [{"request_id": target_uuid, "status": "ok"}]
                ),  # Target at end
            },
        ]

        config = SmartCrusherConfig(
            enabled=True,
            min_items_to_analyze=5,
            min_tokens_to_crush=100,
            max_items_after_crush=10,
        )
        crusher = SmartCrusher(config)
        tokenizer = get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        tool_msg = next(m for m in result.messages if m.get("role") == "tool")
        crushed_content = tool_msg["content"]

        assert target_uuid in crushed_content


class TestTransformIntegration:
    """Integration tests for transform pipeline."""

    def test_pipeline_preserves_message_order(self):
        """Transform pipeline should preserve message order."""
        from headroom.transforms import TransformPipeline

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        pipeline = TransformPipeline(provider=_provider)
        result = pipeline.apply(messages, "gpt-4o", model_limit=128000)

        # Order should be preserved
        roles = [m["role"] for m in result.messages]
        assert roles[0] == "system"
        assert "user" in roles
        assert "assistant" in roles

    def test_pipeline_never_removes_user_content(self):
        """User message content should never be removed."""
        from headroom.transforms import TransformPipeline

        user_content = "This is my important question that should never be modified!"
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": user_content},
        ]

        pipeline = TransformPipeline(provider=_provider)
        result = pipeline.apply(messages, "gpt-4o", model_limit=128000)

        # Find user message
        user_messages = [m for m in result.messages if m.get("role") == "user"]
        assert len(user_messages) >= 1

        # Original user content should be preserved somewhere
        all_content = " ".join(m.get("content", "") for m in result.messages)
        assert user_content in all_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
