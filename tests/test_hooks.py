"""Tests for Compression Hooks interface."""

from headroom.hooks import CompressContext, CompressEvent, CompressionHooks


class TestCompressionHooksDefaults:
    """Default (no-op) hooks don't modify anything."""

    def test_pre_compress_returns_messages_unchanged(self):
        hooks = CompressionHooks()
        messages = [{"role": "user", "content": "hello"}]
        ctx = CompressContext(model="test")
        result = hooks.pre_compress(messages, ctx)
        assert result is messages

    def test_compute_biases_returns_empty(self):
        hooks = CompressionHooks()
        messages = [{"role": "user", "content": "hello"}]
        ctx = CompressContext(model="test")
        result = hooks.compute_biases(messages, ctx)
        assert result == {}

    def test_post_compress_is_noop(self):
        hooks = CompressionHooks()
        event = CompressEvent(tokens_before=100, tokens_after=50)
        hooks.post_compress(event)  # Should not raise


class TestCustomHooks:
    """Custom hook implementations work correctly."""

    def test_pre_compress_can_modify_messages(self):
        class FilterHooks(CompressionHooks):
            def pre_compress(self, messages, ctx):
                return [m for m in messages if m.get("role") != "system"]

        hooks = FilterHooks()
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
        result = hooks.pre_compress(messages, CompressContext())
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_compute_biases_position_aware(self):
        class PositionAwareHooks(CompressionHooks):
            def compute_biases(self, messages, ctx):
                biases = {}
                n = len(messages)
                for i in range(n):
                    pos = i / max(n - 1, 1)
                    # U-curve: middle gets higher bias
                    biases[i] = 1.0 + 0.5 * (1.0 - abs(2 * pos - 1))
                return biases

        hooks = PositionAwareHooks()
        messages = [{"role": "user"}] * 10
        biases = hooks.compute_biases(messages, CompressContext())

        # Edges should have lower bias, middle should have higher
        assert biases[0] < biases[5]  # start < middle
        assert biases[9] < biases[5]  # end < middle
        assert biases[5] > 1.0  # middle is above default

    def test_post_compress_records_event(self):
        events = []

        class LoggingHooks(CompressionHooks):
            def post_compress(self, event):
                events.append(event)

        hooks = LoggingHooks()
        event = CompressEvent(
            tokens_before=1000,
            tokens_after=300,
            tokens_saved=700,
            compression_ratio=0.7,
            model="claude-sonnet",
            provider="anthropic",
        )
        hooks.post_compress(event)
        assert len(events) == 1
        assert events[0].tokens_saved == 700

    def test_hooks_receive_correct_context(self):
        received_ctx = []

        class ContextCapture(CompressionHooks):
            def pre_compress(self, messages, ctx):
                received_ctx.append(ctx)
                return messages

        hooks = ContextCapture()
        ctx = CompressContext(
            model="gpt-4o",
            user_query="find errors",
            provider="openai",
            turn_number=5,
            tool_calls=["read_file", "grep"],
        )
        hooks.pre_compress([], ctx)

        assert received_ctx[0].model == "gpt-4o"
        assert received_ctx[0].user_query == "find errors"
        assert received_ctx[0].provider == "openai"
        assert received_ctx[0].turn_number == 5
        assert "read_file" in received_ctx[0].tool_calls


class TestCompressEvent:
    def test_event_fields(self):
        event = CompressEvent(
            tokens_before=1000,
            tokens_after=200,
            tokens_saved=800,
            compression_ratio=0.8,
            transforms_applied=["smart:relevance(500->20)", "router:code_aware:0.45"],
            ccr_hashes=["abc123", "def456"],
            model="claude-sonnet-4-5-20250929",
            user_query="What are the test failures?",
            provider="anthropic",
        )
        assert event.compression_ratio == 0.8
        assert len(event.transforms_applied) == 2
        assert len(event.ccr_hashes) == 2

    def test_event_defaults(self):
        event = CompressEvent()
        assert event.tokens_before == 0
        assert event.transforms_applied == []
        assert event.provider == ""


class TestCompressContext:
    def test_context_defaults(self):
        ctx = CompressContext()
        assert ctx.model == ""
        assert ctx.tool_calls == []
        assert ctx.turn_number == 0

    def test_context_with_values(self):
        ctx = CompressContext(
            model="gpt-4o",
            user_query="find the bug",
            turn_number=3,
            tool_calls=["read_file", "bash"],
            provider="openai",
        )
        assert ctx.model == "gpt-4o"
        assert len(ctx.tool_calls) == 2
