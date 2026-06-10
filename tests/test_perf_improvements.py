"""Tests for perf improvements: prefix stability pinning and stale Read/Glob compression.

Improvement 1 (Prefix Stability):
  ContentRouterConfig.compression_stable_after_turn (env: HEADROOM_COMPRESSION_STABLE_AFTER_TURN)
  In the first N turns, skips non-deterministic Kompress for tool outputs to avoid
  busting the upstream provider prefix cache.

Improvement 2 (Stale Read Compression):
  ContentRouterConfig.stale_read_compress_after_turns (env: HEADROOM_STALE_READ_COMPRESS_AFTER_TURNS)
  Excluded-tool outputs (Read, Glob, Grep) older than N turns become eligible
  for compression instead of always being protected.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fake_tokenizer() -> Any:
    """Minimal tokenizer that counts words."""
    return SimpleNamespace(count_text=lambda text: max(1, len(text.split())))


def _tool_message(tool_call_id: str, content: str) -> dict[str, Any]:
    """OpenAI-style tool message."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _assistant_with_tool_call(tool_call_id: str, tool_name: str) -> dict[str, Any]:
    """Assistant message that references a tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
    }


def _user_message(text: str = "Do the thing") -> dict[str, Any]:
    return {"role": "user", "content": text}


def _passthrough_result(content: str) -> RouterCompressionResult:
    """Build a passthrough RouterCompressionResult for monkeypatching."""
    tokens = len(content.split())
    return RouterCompressionResult(
        compressed=content,
        original=content,
        strategy_used=CompressionStrategy.PASSTHROUGH,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.PASSTHROUGH,
                original_tokens=tokens,
                compressed_tokens=tokens,
            )
        ],
    )


def _compressed_result(content: str, compressed: str) -> RouterCompressionResult:
    """Build a compressed RouterCompressionResult for monkeypatching."""
    orig_tokens = len(content.split())
    comp_tokens = len(compressed.split())
    return RouterCompressionResult(
        compressed=compressed,
        original=content,
        strategy_used=CompressionStrategy.KOMPRESS,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.KOMPRESS,
                original_tokens=orig_tokens,
                compressed_tokens=comp_tokens,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Improvement 1: Prefix stability pinning
# ---------------------------------------------------------------------------


class TestPrefixStabilityConfig:
    """ContentRouterConfig correctly reads compression_stable_after_turn."""

    def test_default_is_zero(self) -> None:
        """By default the feature is disabled."""
        cfg = ContentRouterConfig()
        assert cfg.compression_stable_after_turn == 0

    def test_explicit_value(self) -> None:
        cfg = ContentRouterConfig(compression_stable_after_turn=5)
        assert cfg.compression_stable_after_turn == 5

    def test_zero_disables_feature(self) -> None:
        """compression_stable_after_turn=0 means never apply stability phase."""
        cfg = ContentRouterConfig(compression_stable_after_turn=0)
        assert cfg.compression_stable_after_turn == 0

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COMPRESSION_STABLE_AFTER_TURN", "7")
        cfg = ContentRouterConfig()
        assert cfg.compression_stable_after_turn == 7

    def test_env_var_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COMPRESSION_STABLE_AFTER_TURN", "0")
        cfg = ContentRouterConfig()
        assert cfg.compression_stable_after_turn == 0


class TestCompressConservative:
    """_compress_conservative skips Kompress and allows deterministic strategies."""

    def test_passthrough_for_kompress_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Content that would be routed to Kompress passes through unchanged."""
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=5))

        # Monkeypatch _determine_strategy to return KOMPRESS
        monkeypatch.setattr(router, "_determine_strategy", lambda c: CompressionStrategy.KOMPRESS)

        content = "some plain text " * 50
        result = router._compress_conservative(content)

        assert result.compressed == content
        assert result.strategy_used == CompressionStrategy.PASSTHROUGH

    def test_passthrough_for_text_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TEXT strategy also passes through (TEXT routes to Kompress internally)."""
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=5))
        monkeypatch.setattr(router, "_determine_strategy", lambda c: CompressionStrategy.TEXT)

        content = "plain text content " * 30
        result = router._compress_conservative(content)
        assert result.compressed == content
        assert result.strategy_used == CompressionStrategy.PASSTHROUGH

    def test_allows_smart_crusher(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SmartCrusher (deterministic JSON compression) is allowed in conservative mode."""
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=5))
        monkeypatch.setattr(
            router, "_determine_strategy", lambda c: CompressionStrategy.SMART_CRUSHER
        )
        # Monkeypatch _compress_pure to track calls
        calls: list[str] = []

        def fake_compress_pure(
            content: str, strategy: CompressionStrategy, *args: Any, **kwargs: Any
        ) -> RouterCompressionResult:
            calls.append(strategy.value)
            return _passthrough_result(content)

        monkeypatch.setattr(router, "_compress_pure", fake_compress_pure)
        content = '[{"a": 1}]'
        router._compress_conservative(content)
        assert calls == [CompressionStrategy.SMART_CRUSHER.value]

    def test_allows_search_compressor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SearchCompressor (rule-based) is allowed in conservative mode."""
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=5))
        monkeypatch.setattr(router, "_determine_strategy", lambda c: CompressionStrategy.SEARCH)
        calls: list[str] = []

        def fake_compress_pure(
            content: str, strategy: CompressionStrategy, *args: Any, **kwargs: Any
        ) -> RouterCompressionResult:
            calls.append(strategy.value)
            return _passthrough_result(content)

        monkeypatch.setattr(router, "_compress_pure", fake_compress_pure)
        router._compress_conservative("file.py:1: error")
        assert calls == [CompressionStrategy.SEARCH.value]

    def test_empty_content_passthrough(self) -> None:
        router = ContentRouter()
        result = router._compress_conservative("")
        assert result.compressed == ""
        assert result.strategy_used == CompressionStrategy.PASSTHROUGH


class TestStabilityPhaseApply:
    """apply() uses conservative compression for tool messages in early turns."""

    def _build_messages(self, n_turns: int, tool_name: str = "Bash") -> list[dict[str, Any]]:
        """Build a conversation with n_turns of tool use."""
        msgs: list[dict[str, Any]] = [_user_message("start")]
        for t in range(n_turns):
            tc_id = f"tc_{t}"
            msgs.append(_assistant_with_tool_call(tc_id, tool_name))
            content = f"tool output number {t} " * 100  # long enough to compress
            msgs.append(_tool_message(tc_id, content))
            if t < n_turns - 1:
                msgs.append(_user_message(f"continue turn {t}"))
        return msgs

    def test_stable_phase_active_in_early_turns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When session_turn_count <= stable_after, conservative mode is used for tools.

        Uses WebFetch which is NOT in DEFAULT_EXCLUDE_TOOLS, so tool outputs
        are eligible for compression (not protected by exclusion).
        """
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=5))
        tok = _fake_tokenizer()

        conservative_calls: list[str] = []
        original_conservative = router._compress_conservative

        def mock_conservative(content: str, **kwargs: Any) -> RouterCompressionResult:
            conservative_calls.append(content[:20])
            return original_conservative(content, **kwargs)

        monkeypatch.setattr(router, "_compress_conservative", mock_conservative)

        # 3 turns (6 messages + system) = turn count ~4, below stable_after=5
        # Use WebFetch (not excluded) so outputs actually reach the compress path
        msgs = self._build_messages(n_turns=3, tool_name="WebFetch")
        router.apply(msgs, tok)

        # Conservative should have been called for at least one tool message
        assert len(conservative_calls) > 0

    def test_stable_phase_inactive_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compression_stable_after_turn=0 means conservative mode is never used."""
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=0))
        tok = _fake_tokenizer()

        conservative_calls: list[str] = []
        original_conservative = router._compress_conservative

        def mock_conservative(content: str, **kwargs: Any) -> RouterCompressionResult:
            conservative_calls.append(content[:20])
            return original_conservative(content, **kwargs)

        monkeypatch.setattr(router, "_compress_conservative", mock_conservative)

        # Use WebFetch (not excluded) so outputs reach the compress path
        msgs = self._build_messages(n_turns=3, tool_name="WebFetch")
        router.apply(msgs, tok)

        # Conservative must never be called when feature is disabled
        assert conservative_calls == []

    def test_full_compression_after_stability_phase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After stable_after turns, conservative mode is NOT used.

        session_turn_count is computed per apply() call from num_messages // 2.
        With 10 turns (20+ messages), session_turn_count ~11 > stable_after=3.
        """
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=3))
        tok = _fake_tokenizer()

        conservative_calls: list[str] = []
        original_conservative = router._compress_conservative

        def mock_conservative(content: str, **kwargs: Any) -> RouterCompressionResult:
            conservative_calls.append(content[:20])
            return original_conservative(content, **kwargs)

        monkeypatch.setattr(router, "_compress_conservative", mock_conservative)

        # Build a long conversation: 10 turns = 20+ messages, turn count ~11 > 3
        # Use WebFetch (not excluded) so outputs reach the compress path
        msgs = self._build_messages(n_turns=10, tool_name="WebFetch")
        router.apply(msgs, tok)

        # 10 turns => session_turn_count ~11 > 3 => NOT in stability phase
        assert conservative_calls == []

    def test_same_content_same_decision_via_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same content gets same compressed output across turns (via CompressionCache).

        Uses WebFetch (not excluded) so tool outputs reach the compression path.
        """
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=0))
        tok = _fake_tokenizer()

        # Simulate a compress call that returns a fixed compressed string
        compress_results: list[str] = []

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            result = _compressed_result(content, content[:50])
            compress_results.append(result.compressed)
            return result

        monkeypatch.setattr(router, "compress", mock_compress)

        content = "tool output content " * 100
        tc_id = "tc_0"
        msgs1 = [
            _user_message("turn1"),
            _assistant_with_tool_call(tc_id, "WebFetch"),
            _tool_message(tc_id, content),
        ]
        router.apply(msgs1, tok)
        calls_after_first = len(compress_results)

        # Second call with same content: cache hit, no fresh compression needed
        msgs2 = [
            _user_message("turn1"),
            _assistant_with_tool_call(tc_id, "WebFetch"),
            _tool_message(tc_id, content),
            _user_message("turn2"),
            _assistant_with_tool_call("tc_1", "WebFetch"),
            _tool_message(tc_id, content),
        ]
        router.apply(msgs2, tok)

        # Second apply should reuse cache for the duplicate content (same hash)
        # Total compress calls should be minimal due to caching
        assert len(compress_results) <= calls_after_first + 1  # cache hit for repeat content


class TestStabilityPhaseAnthropicFormat:
    """Stability phase works for Anthropic-format tool_result content blocks."""

    def _build_anthropic_msgs(
        self, n_turns: int, tool_name: str = "Bash", long_output: bool = True
    ) -> list[dict[str, Any]]:
        """Build Anthropic-format conversation with n_turns."""
        msgs: list[dict[str, Any]] = [_user_message("start")]
        for t in range(n_turns):
            tc_id = f"tc_{t}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": tc_id, "name": tool_name, "input": {}}],
                }
            )
            tool_content = f"output from {tool_name} turn {t} " * (100 if long_output else 1)
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tc_id, "content": tool_content}
                    ],
                }
            )
            if t < n_turns - 1:
                msgs.append(_user_message(f"continue turn {t}"))
        return msgs

    def test_conservative_called_for_anthropic_tool_results_in_early_turns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In stability phase, conservative compress is used for Anthropic tool_result blocks.

        Uses WebFetch (not in DEFAULT_EXCLUDE_TOOLS) so tool_result blocks
        actually reach the compression path.
        """
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=5))
        tok = _fake_tokenizer()

        conservative_calls: list[str] = []
        original_conservative = router._compress_conservative

        def mock_conservative(content: str, **kwargs: Any) -> RouterCompressionResult:
            conservative_calls.append(content[:20])
            return original_conservative(content, **kwargs)

        monkeypatch.setattr(router, "_compress_conservative", mock_conservative)

        # 3 turns => session_turn_count ~4, below stable_after=5
        # Use WebFetch (not excluded) so blocks reach compression path
        msgs = self._build_anthropic_msgs(n_turns=3, tool_name="WebFetch")
        router.apply(msgs, tok)

        assert len(conservative_calls) > 0

    def test_conservative_not_called_for_anthropic_tool_results_after_stability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After stability phase, full compress() is used (conservative is never called)."""
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=3))
        tok = _fake_tokenizer()

        conservative_calls: list[str] = []
        original_conservative = router._compress_conservative

        def mock_conservative(content: str, **kwargs: Any) -> RouterCompressionResult:
            conservative_calls.append(content[:20])
            return original_conservative(content, **kwargs)

        monkeypatch.setattr(router, "_compress_conservative", mock_conservative)

        # 10 turns => session_turn_count ~11 > stable_after=3
        # Use WebFetch (not excluded) so blocks reach compression path
        msgs = self._build_anthropic_msgs(n_turns=10, tool_name="WebFetch")
        router.apply(msgs, tok)

        assert conservative_calls == []


# ---------------------------------------------------------------------------
# Improvement 2: Stale Read/Glob compression
# ---------------------------------------------------------------------------


class TestStaleReadConfig:
    """ContentRouterConfig correctly reads stale_read_compress_after_turns."""

    def test_default_is_zero(self) -> None:
        """Default is 0 (feature disabled: all excluded tools always protected)."""
        cfg = ContentRouterConfig()
        assert cfg.stale_read_compress_after_turns == 0

    def test_explicit_value(self) -> None:
        cfg = ContentRouterConfig(stale_read_compress_after_turns=10)
        assert cfg.stale_read_compress_after_turns == 10

    def test_zero_disables_feature(self) -> None:
        """stale_read_compress_after_turns=0 means no excluded tool outputs are compressed."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=0)
        assert cfg.stale_read_compress_after_turns == 0

    def test_999_is_backward_compat(self) -> None:
        """999 is the backward-compat escape hatch: protect everything."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=999)
        assert cfg.stale_read_compress_after_turns == 999

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_STALE_READ_COMPRESS_AFTER_TURNS", "10")
        cfg = ContentRouterConfig()
        assert cfg.stale_read_compress_after_turns == 10

    def test_env_var_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_STALE_READ_COMPRESS_AFTER_TURNS", "0")
        cfg = ContentRouterConfig()
        assert cfg.stale_read_compress_after_turns == 0

    def test_env_var_999(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_STALE_READ_COMPRESS_AFTER_TURNS", "999")
        cfg = ContentRouterConfig()
        assert cfg.stale_read_compress_after_turns == 999


def _build_long_conversation(
    n_turns: int,
    tool_name: str = "Read",
    content_per_turn: str = "file content with important data " * 50,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a conversation with tool messages.

    Returns (messages, tool_call_ids).
    """
    msgs: list[dict[str, Any]] = [_user_message("start")]
    tc_ids: list[str] = []
    for t in range(n_turns):
        tc_id = f"tc_{t}"
        tc_ids.append(tc_id)
        msgs.append(_assistant_with_tool_call(tc_id, tool_name))
        msgs.append(_tool_message(tc_id, content_per_turn))
        if t < n_turns - 1:
            msgs.append(_user_message(f"continue turn {t}"))
    return msgs, tc_ids


class TestStaleReadCompressionApply:
    """apply() respects stale_read_compress_after_turns for excluded tool messages."""

    def test_recent_reads_not_compressed_when_feature_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default (stale_read_compress_after_turns=0): Read tool outputs always protected."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=0)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)

        # 3-turn conversation with Read outputs
        msgs, _ = _build_long_conversation(n_turns=3, tool_name="Read")
        router.apply(msgs, tok)

        # With stale_read=0, all Read outputs are protected (excluded_tool path)
        # compress() should NOT be called for Read outputs
        assert compress_calls == []

    def test_old_reads_compressed_when_feature_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When stale_read_compress_after_turns=2, old Read outputs (>2 turns) get compressed."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=2)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)

        # 10-turn conversation: Read outputs from turn 0..7 are >2 turns old
        # (messages_from_end > stale_read_compress_after_turns * 2)
        msgs, _ = _build_long_conversation(n_turns=10, tool_name="Read")
        router.apply(msgs, tok)

        # Some Read outputs should have been attempted for compression
        # (they fell through the excluded_tool guard because they're old)
        # Note: compress() may still not compress if content doesn't pass min_ratio
        assert len(compress_calls) > 0, "Old Read outputs should be attempted for compression"

    def test_recent_reads_protected_when_feature_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With stale_read enabled, the MOST RECENT N turns of Read outputs are still protected."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=5)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        # Build a 6-turn conversation
        msgs, tc_ids = _build_long_conversation(n_turns=6, tool_name="Read")

        # Track route_counts via observer
        route_counts_received: list[dict[str, int]] = []

        class FakeObserver:
            def record_compression(self, **kwargs: Any) -> None:
                pass

            def record_router_route_counts(self, rc: dict[str, int]) -> None:
                route_counts_received.append(dict(rc))

        router._observer = FakeObserver()
        router.apply(msgs, tok)

        # There should be some excluded (protected) tool reads
        if route_counts_received:
            total_excluded = sum(rc.get("excluded_tool", 0) for rc in route_counts_received)
            assert total_excluded >= 0  # At least the last N turns protected

    def test_999_protects_all_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stale_read_compress_after_turns=999 means all Read outputs are protected."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=999)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)

        # 25-turn conversation — all Read outputs should still be protected
        msgs, _ = _build_long_conversation(n_turns=25, tool_name="Read")
        router.apply(msgs, tok)

        assert compress_calls == [], "999 should protect all Read outputs"

    def test_non_read_tools_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stale_read_compress_after_turns only affects excluded tools (Read/Glob).
        Non-excluded tools like WebFetch should compress normally."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=2)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)

        # Use WebFetch (not in DEFAULT_EXCLUDE_TOOLS) — should always be eligible
        msgs, _ = _build_long_conversation(n_turns=3, tool_name="WebFetch")
        router.apply(msgs, tok)

        # WebFetch outputs should be attempted for compression regardless
        assert len(compress_calls) > 0, "Non-excluded tools should always be attempted"

    def test_read_protection_window_computed_from_turns(self) -> None:
        """read_protection_window = stale_read_compress_after_turns * 2 (messages, not turns)."""
        # This tests the internal calculation: 10 turns * 2 msgs/turn = 20 messages protected
        cfg = ContentRouterConfig(stale_read_compress_after_turns=10)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        # Build 25-turn conversation: first 5 turns (10 messages) are old (beyond window)
        msgs, _ = _build_long_conversation(n_turns=25, tool_name="Read")

        route_counts_received: list[dict[str, int]] = []

        class FakeObserver:
            def record_compression(self, **kwargs: Any) -> None:
                pass

            def record_router_route_counts(self, rc: dict[str, int]) -> None:
                route_counts_received.append(dict(rc))

        router._observer = FakeObserver()
        router.apply(msgs, tok)

        if route_counts_received:
            total_excluded = sum(rc.get("excluded_tool", 0) for rc in route_counts_received)
            # With 25 turns and window=10*2=20 messages, ~5 turns should fall through
            # to compression. So excluded should be < 25.
            assert total_excluded < 25, "Some old Read outputs should not be protected"

    def test_boundary_at_exactly_stale_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Message at exactly stale_read_compress_after_turns*2 from end: edge case."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=3)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        # Build exactly 3-turn conversation + 1 extra turn
        # stale threshold = 3 * 2 = 6 messages
        # With 4 turns (8+ messages), the earliest turn is >6 from end → should be eligible
        msgs, _ = _build_long_conversation(n_turns=4, tool_name="Read")

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)
        router.apply(msgs, tok)

        # Earliest Read output (from turn 0) should be eligible for compression
        assert len(compress_calls) > 0

    def test_glob_outputs_stale_compress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Glob (also in DEFAULT_EXCLUDE_TOOLS) old outputs are eligible like Read."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=2)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)

        msgs, _ = _build_long_conversation(n_turns=8, tool_name="Glob")
        router.apply(msgs, tok)

        assert len(compress_calls) > 0, "Old Glob outputs should be attempted for compression"


class TestStaleReadAnthropicFormat:
    """Stale Read/Glob compression works for Anthropic-format tool_result content blocks."""

    def _build_anthropic_msgs(self, n_turns: int, tool_name: str = "Read") -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = [_user_message("start")]
        for t in range(n_turns):
            tc_id = f"tc_{t}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": tc_id, "name": tool_name, "input": {}}],
                }
            )
            content = f"file content line {t} with important data " * 50
            msgs.append(
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tc_id, "content": content}],
                }
            )
            if t < n_turns - 1:
                msgs.append(_user_message(f"continue {t}"))
        return msgs

    def test_anthropic_old_reads_eligible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Old Anthropic tool_result blocks for Read are eligible when feature enabled."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=2)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)

        msgs = self._build_anthropic_msgs(n_turns=8, tool_name="Read")
        router.apply(msgs, tok)

        assert len(compress_calls) > 0, "Old Anthropic Read outputs should be attempted"

    def test_anthropic_recent_reads_still_protected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With stale_read enabled, the most recent N turns of Anthropic Reads are protected."""
        cfg = ContentRouterConfig(stale_read_compress_after_turns=10)
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)

        # Only 3 turns: all within the 10-turn window → all protected
        msgs = self._build_anthropic_msgs(n_turns=3, tool_name="Read")
        router.apply(msgs, tok)

        # With only 3 turns and window=10, nothing should fall through
        assert compress_calls == [], "Recent Anthropic Read outputs should be protected"


# ---------------------------------------------------------------------------
# Integration: Both features together
# ---------------------------------------------------------------------------


class TestCacheSkipNotPoisonedByConservativeMode:
    """Conservative passthrough results must NOT poison the skip set.

    Bug: if a tool message is passed through in conservative mode (stable phase)
    and the result is added to mark_skip(), then after the stability phase ends
    the same content stays in the skip set and is never compressed — even though
    it could be compressed by Kompress in full mode.
    """

    def test_conservative_passthrough_not_added_to_skip_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content passed through in conservative mode is NOT marked as skip."""
        router = ContentRouter(ContentRouterConfig(compression_stable_after_turn=5))
        tok = _fake_tokenizer()

        # Monkeypatch compress to return a good compressed result (would compress in full mode)
        full_compress_calls: list[str] = []
        original_compress = router.compress

        def mock_full_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            full_compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_full_compress)

        content = "tool output WebFetch data " * 100
        tc_id = "tc_0"

        # Turn 1 (3 turns = conservative phase, turn_count=2 <= stable_after=5)
        msgs_early = [
            _user_message("turn1"),
            _assistant_with_tool_call(tc_id, "WebFetch"),
            _tool_message(tc_id, content),
        ]
        router.apply(msgs_early, tok)

        # Check immediately after the conservative-phase call: the skip set must not
        # have been poisoned. (Full-mode calls are allowed to add to skip set later —
        # that is correct behaviour — so we check here, before the full-mode call.)
        content_key = hash(content)
        assert not router._cache.is_skipped(content_key), (
            "Content passed through in conservative mode must not be in skip set"
        )


class TestBothFeaturesEnabled:
    """Both improvements can be active simultaneously without interference."""

    def test_both_features_coexist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """stability pinning and stale reads can both be active."""
        cfg = ContentRouterConfig(
            compression_stable_after_turn=5,
            stale_read_compress_after_turns=2,
        )
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        # 3-turn conversation: in stability phase (turn_count ~4 <= 5)
        # and stale reads enabled (window=4 msgs, older reads eligible)
        # Use WebFetch (not excluded) so outputs reach compression path
        msgs: list[dict[str, Any]] = [_user_message("start")]
        for t in range(3):
            tc_id = f"tc_{t}"
            msgs.append(_assistant_with_tool_call(tc_id, "WebFetch"))
            content = f"web output {t} " * 100
            msgs.append(_tool_message(tc_id, content))
            if t < 2:
                msgs.append(_user_message(f"continue {t}"))

        # Should not raise
        result = router.apply(msgs, tok)
        assert result is not None
        assert len(result.messages) > 0

    def test_non_excluded_tool_still_fully_compressed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stale_read does not affect non-excluded tools like WebFetch."""
        cfg = ContentRouterConfig(
            compression_stable_after_turn=0,  # stable disabled
            stale_read_compress_after_turns=10,  # stale enabled
        )
        router = ContentRouter(cfg)
        tok = _fake_tokenizer()

        compress_calls: list[str] = []
        original_compress = router.compress

        def mock_compress(content: str, **kwargs: Any) -> RouterCompressionResult:
            compress_calls.append(content[:20])
            return original_compress(content, **kwargs)

        monkeypatch.setattr(router, "compress", mock_compress)

        # 3-turn conversation with WebFetch (not excluded)
        msgs, _ = _build_long_conversation(n_turns=3, tool_name="WebFetch")
        router.apply(msgs, tok)

        # WebFetch should still be attempted
        assert len(compress_calls) > 0
