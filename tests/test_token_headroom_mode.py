"""Integration tests for token mode (legacy token_headroom behavior).

Tests the CompressionCache working across simulated multi-turn conversations,
verifying the critical invariants: no message injection, correct frozen counts,
proper handling of both Anthropic and OpenAI formats, and correct behavior
when Claude Code drops messages.
"""

import copy

from headroom.cache.compression_cache import CompressionCache


def _make_user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def _make_assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _make_tool_use_msg(tool_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
    }


def _make_tool_result_msg(tool_id: str, content: str) -> dict:
    """Anthropic-format tool result."""
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
    }


def _make_openai_tool_msg(tool_call_id: str, content: str) -> dict:
    """OpenAI-format tool result."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _large_code_content(n: int = 200) -> str:
    """Generate realistic Python code content."""
    parts = ["import os\nimport sys\nfrom typing import List, Dict\n\n"]
    for i in range(n // 10):
        parts.append(
            f"def function_{i}(arg: str) -> str:\n"
            f'    """Docstring for function {i}."""\n'
            f"    result = arg.strip()\n"
            f"    for j in range({i}):\n"
            f"        result += str(j)\n"
            f"    return result\n\n"
        )
    return "".join(parts)


class TestMultiTurnCompression:
    """Simulate multi-turn conversations to verify compression cascade."""

    def test_first_turn_nothing_cached(self):
        """On first turn, no cache hits, frozen count is minimal."""
        cache = CompressionCache()
        messages = [
            _make_user_msg("Read file.py"),
            _make_tool_use_msg("t1", "Read"),
            _make_tool_result_msg("t1", _large_code_content(100)),
        ]
        frozen = cache.compute_frozen_count(messages)
        # user (stable) + tool_use (stable) + tool_result (miss) → 2
        assert frozen == 2

    def test_second_turn_cache_hits(self):
        """After caching, same content gets cache hits."""
        cache = CompressionCache()
        code = _large_code_content(100)
        compressed = "# compressed version"

        # Simulate first turn: pipeline compressed the code
        h = CompressionCache.content_hash(code)
        cache.store_compressed(h, compressed, tokens_saved=500)

        # Second turn: same messages
        messages = [
            _make_user_msg("Read file.py"),
            _make_tool_use_msg("t1", "Read"),
            _make_tool_result_msg("t1", code),
            _make_user_msg("now edit it"),
        ]
        frozen = cache.compute_frozen_count(messages)
        # First 3 stable; trailing user message ("now edit it") is the
        # live zone by construction — it has not been sent upstream
        # before, so it cannot be in any provider prefix cache. Cap at
        # len - 1 prevents the over-freeze pattern that produced 0 %
        # compression for prose-format clients (issue observed
        # 2026-05-07 with Cline+DeepSeek).
        assert frozen == 3

        # apply_cached should swap the content
        result = cache.apply_cached(messages)
        tool_result = result[2]["content"][0]
        assert tool_result["content"] == compressed

    def test_multi_turn_waterfall(self):
        """Messages age out progressively across turns."""
        cache = CompressionCache()

        # Build conversation with 3 read results
        code_a = "code A " * 200
        code_b = "code B " * 200
        code_c = "code C " * 200

        messages = [
            _make_user_msg("read A"),
            _make_tool_result_msg("t1", code_a),
            _make_user_msg("read B"),
            _make_tool_result_msg("t2", code_b),
            _make_user_msg("read C"),
            _make_tool_result_msg("t3", code_c),
        ]

        # Turn 1: nothing cached
        frozen = cache.compute_frozen_count(messages)
        assert frozen == 1  # only first user msg

        # Simulate pipeline compressing A and B (not C — in protection window)
        cache.store_compressed(CompressionCache.content_hash(code_a), "ca", tokens_saved=100)
        cache.store_compressed(CompressionCache.content_hash(code_b), "cb", tokens_saved=100)

        # Turn 2: A and B cached, C still uncached
        frozen = cache.compute_frozen_count(messages)
        # user(stable) + tool_result_A(cached) + user(stable) + tool_result_B(cached) + user(stable) + tool_result_C(miss)
        assert frozen == 5

        # Now cache C too
        cache.store_compressed(CompressionCache.content_hash(code_c), "cc", tokens_saved=100)

        # Turn 3: all 6 messages structurally stable, but the trailing
        # message is reserved as live zone. Frozen prefix = 5.
        frozen = cache.compute_frozen_count(messages)
        assert frozen == 5


class TestNoMessageInjection:
    """Critical invariant: proxy never adds messages."""

    def test_output_length_equals_input(self):
        cache = CompressionCache()
        messages = [
            _make_user_msg("hello"),
            _make_tool_result_msg("t1", _large_code_content(50)),
            _make_user_msg("bye"),
        ]
        result = cache.apply_cached(messages)
        assert len(result) == len(messages)

    def test_orphan_cache_entries_not_injected(self):
        """Cache entries with no matching message are NOT injected."""
        cache = CompressionCache()
        cache.store_compressed("orphan_hash_1", "orphan content 1", tokens_saved=100)
        cache.store_compressed("orphan_hash_2", "orphan content 2", tokens_saved=200)

        messages = [_make_user_msg("hello")]
        result = cache.apply_cached(messages)
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_input_not_mutated(self):
        """apply_cached must NOT mutate the input list or messages."""
        cache = CompressionCache()
        code = "original code content"
        h = CompressionCache.content_hash(code)
        cache.store_compressed(h, "compressed", tokens_saved=50)

        messages = [
            _make_tool_result_msg("t1", code),
        ]
        original = copy.deepcopy(messages)
        cache.apply_cached(messages)
        assert messages == original


class TestClaudeCodeDropsMessages:
    """When Claude Code drops messages via its own context management."""

    def test_dropped_messages_not_readded(self):
        cache = CompressionCache()
        content_a = "content A " * 100
        content_b = "content B " * 100
        cache.store_compressed(CompressionCache.content_hash(content_a), "ca", tokens_saved=100)
        cache.store_compressed(CompressionCache.content_hash(content_b), "cb", tokens_saved=100)

        # CC dropped the message with content_b
        messages = [
            _make_user_msg("hello"),
            _make_tool_result_msg("t1", content_a),
            _make_user_msg("continue"),
        ]
        result = cache.apply_cached(messages)
        assert len(result) == 3  # NOT 4

    def test_frozen_count_breaks_at_gap(self):
        """Dropped cached message creates a gap that stops frozen count."""
        cache = CompressionCache()
        content_a = "content A " * 100
        content_c = "content C " * 100
        cache.store_compressed(CompressionCache.content_hash(content_a), "ca", tokens_saved=100)
        cache.store_compressed(CompressionCache.content_hash(content_c), "cc", tokens_saved=100)

        # CC dropped content_b, content_c is still here but preceded by uncached gap
        messages = [
            _make_tool_result_msg("t1", content_a),
            _make_tool_result_msg("t2", "UNCACHED content_b replacement"),
            _make_tool_result_msg("t3", content_c),
        ]
        frozen = cache.compute_frozen_count(messages)
        # t1 (cached, stable), t2 (NOT cached, stop)
        assert frozen == 1


class TestOpenAIFormat:
    """Verify OpenAI-format tool messages work correctly."""

    def test_openai_tool_result_cached(self):
        cache = CompressionCache()
        content = "large openai output " * 100
        h = CompressionCache.content_hash(content)
        cache.store_compressed(h, "compressed openai output", tokens_saved=300)

        messages = [
            _make_user_msg("run command"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            _make_openai_tool_msg("tc1", content),
        ]

        result = cache.apply_cached(messages)
        assert len(result) == 3
        assert result[2]["content"] == "compressed openai output"

    def test_openai_frozen_count(self):
        cache = CompressionCache()
        content = "openai tool output " * 100
        h = CompressionCache.content_hash(content)
        cache.store_compressed(h, "compressed", tokens_saved=200)

        messages = [
            _make_user_msg("hello"),
            _make_openai_tool_msg("tc1", content),
            _make_openai_tool_msg("tc2", "uncached content"),
        ]
        frozen = cache.compute_frozen_count(messages)
        # user (stable), tool tc1 (cached), tool tc2 (miss)
        assert frozen == 2


class TestUpdateFromResult:
    """Verify update_from_result correctly caches compression results."""

    def test_caches_compressed_anthropic(self):
        cache = CompressionCache()
        original_content = "long original " * 100
        compressed_content = "short compressed"

        originals = [
            _make_user_msg("hello"),
            _make_tool_result_msg("t1", original_content),
        ]
        compressed = [
            _make_user_msg("hello"),
            _make_tool_result_msg("t1", compressed_content),
        ]

        cache.update_from_result(originals, compressed)

        h = CompressionCache.content_hash(original_content)
        assert cache.get_compressed(h) == compressed_content

    def test_caches_compressed_openai(self):
        cache = CompressionCache()
        original_content = "long openai output " * 100
        compressed_content = "short compressed"

        originals = [_make_openai_tool_msg("tc1", original_content)]
        compressed = [_make_openai_tool_msg("tc1", compressed_content)]

        cache.update_from_result(originals, compressed)

        h = CompressionCache.content_hash(original_content)
        assert cache.get_compressed(h) == compressed_content

    def test_length_mismatch_no_crash(self):
        """If pipeline somehow changes message count, don't crash."""
        cache = CompressionCache()
        originals = [_make_user_msg("a"), _make_user_msg("b")]
        compressed = [_make_user_msg("a")]  # shorter
        # Should not raise, just log warning
        cache.update_from_result(originals, compressed)
        assert cache.get_stats()["entries"] == 0

    def test_unchanged_content_not_cached(self):
        cache = CompressionCache()
        msg = _make_user_msg("same content")
        cache.update_from_result([msg], [msg])
        assert cache.get_stats()["entries"] == 0


class TestProseFormatLiveZoneInvariant:
    """Cline / OpenClaude / Aider — prose-format clients send tool calls
    embedded in plain assistant text and tool results pasted into plain
    user messages. There are no `tool_use`, `tool_result`, or
    ``role: "tool"`` blocks anywhere in the conversation.

    Pre-fix: ``compute_frozen_count`` walked all messages and found no
    "unstable" boundary, returning ``len(messages)``. The pipeline then
    froze every message — including the brand-new user turn — leaving
    the live zone empty. ContentRouter saw ``saved 0`` on every request.
    Bug observed 2026-05-07 with Cline+DeepSeek over /v1/chat/completions
    in token mode.

    Post-fix: cap at ``len(messages) - 1`` always reserves the trailing
    message as the live zone. These tests lock that invariant.
    """

    def test_pure_user_assistant_turns_leave_live_zone(self):
        cache = CompressionCache()
        # A 6-turn Cline-shaped conversation: alternating user/assistant
        # plain-text. No tool blocks of any kind.
        messages = [
            _make_user_msg("system instructions baked into first user msg"),
            _make_assistant_msg("<execute_command>ls</execute_command>"),
            _make_user_msg("[tool_result]\nfile1.py\nfile2.py\n[/tool_result]"),
            _make_assistant_msg("<read_file>file1.py</read_file>"),
            _make_user_msg("[tool_result]\n<contents...>\n[/tool_result]"),
            _make_user_msg("now please refactor it"),
        ]
        frozen = cache.compute_frozen_count(messages)
        # Pre-fix would return 6 (every plain message is "stable").
        # Post-fix: 6 messages stable, capped at len-1 = 5.
        assert frozen == 5
        assert frozen < len(messages), (
            "Live zone must never be empty; trailing user message must "
            "always be available for compression"
        )

    def test_single_message_yields_zero_frozen(self):
        # Edge case: only the user's first message, nothing to freeze.
        cache = CompressionCache()
        messages = [_make_user_msg("first turn")]
        assert cache.compute_frozen_count(messages) == 0

    def test_empty_messages_yields_zero(self):
        cache = CompressionCache()
        assert cache.compute_frozen_count([]) == 0

    def test_two_messages_first_is_frozen_second_is_live(self):
        cache = CompressionCache()
        messages = [
            _make_user_msg("turn 1 content"),
            _make_user_msg("turn 2 — the live zone"),
        ]
        # First message structurally stable; trailing is live → 1 frozen.
        assert cache.compute_frozen_count(messages) == 1

    def test_anthropic_format_last_tool_result_is_still_live(self):
        """Even when the trailing message is a tool_result whose content
        IS in the cache, it stays in the live zone. Trailing == live by
        construction; the cache-read decision is upstream's job."""
        cache = CompressionCache()
        code = _large_code_content(50)
        cache.store_compressed(
            CompressionCache.content_hash(code), "compressed code", tokens_saved=200
        )
        messages = [
            _make_user_msg("hi"),
            _make_assistant_msg("ok"),
            _make_tool_result_msg("t1", code),
        ]
        frozen = cache.compute_frozen_count(messages)
        # Walk gets to 3, cap clamps to 2 (= len-1).
        assert frozen == 2

    def test_openai_format_last_tool_msg_is_live(self):
        cache = CompressionCache()
        content = "tool output " * 100
        cache.store_compressed(
            CompressionCache.content_hash(content), "compressed", tokens_saved=300
        )
        messages = [
            _make_user_msg("run cmd"),
            _make_openai_tool_msg("tc1", content),
        ]
        # Walk: user (stable, 1), tool (cached, 2). Cap → 1.
        assert cache.compute_frozen_count(messages) == 1


# ── List-of-blocks tool_result content (Claude Code modern format) ──────────


def _make_tool_result_list_content_msg(tool_id: str, texts: list[str]) -> dict:
    """Anthropic-format tool result with list-of-blocks content.

    Modern Claude Code sends ``tool_result`` content as a list of typed
    blocks instead of a plain string.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [{"type": "text", "text": t} for t in texts],
            }
        ],
    }


def _make_openai_tool_list_content_msg(tool_call_id: str, texts: list[str]) -> dict:
    """OpenAI-format tool message with list-of-blocks content."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": [{"type": "text", "text": t} for t in texts],
    }


class TestExtractToolResultListContent:
    """_extract_tool_result_content handles list-of-blocks content."""

    def test_anthropic_string_content_preserved(self):
        """Plain string content in Anthropic format still works."""
        from headroom.cache.compression_cache import _extract_tool_result_content as f

        msg = _make_tool_result_msg("t1", "hello world")
        assert f(msg) == "hello world"

    def test_anthropic_list_content_extracted(self):
        """List-of-blocks content is extracted and joined."""
        from headroom.cache.compression_cache import _extract_tool_result_content as f

        msg = _make_tool_result_list_content_msg("t1", ["Line 1", "Line 2"])
        assert f(msg) == "Line 1\nLine 2"

    def test_anthropic_mixed_blocks(self):
        """Non-text blocks (e.g. image) are skipped, only text blocks joined."""
        from headroom.cache.compression_cache import _extract_tool_result_content as f

        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image", "source": {"type": "base64", "data": "..."}},
                        {"type": "text", "text": "World"},
                    ],
                }
            ],
        }
        assert f(msg) == "Hello\nWorld"

    def test_anthropic_list_empty_returns_none(self):
        """Empty text-only list returns None."""
        from headroom.cache.compression_cache import _extract_tool_result_content as f

        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": []}],
        }
        assert f(msg) is None

    def test_openai_list_content_extracted(self):
        """OpenAI format tool message with list content is extracted."""
        from headroom.cache.compression_cache import _extract_tool_result_content as f

        msg = _make_openai_tool_list_content_msg("tc1", ["Result 1", "Result 2"])
        assert f(msg) == "Result 1\nResult 2"

    def test_openai_string_content_still_works(self):
        """OpenAI format with plain string content is unchanged."""
        from headroom.cache.compression_cache import _extract_tool_result_content as f

        msg = _make_openai_tool_msg("tc1", "plain result")
        assert f(msg) == "plain result"

    def test_non_tool_msg_returns_none(self):
        """Regular user/assistant messages return None."""
        from headroom.cache.compression_cache import _extract_tool_result_content as f

        assert f(_make_user_msg("hello")) is None
        assert f(_make_assistant_msg("response")) is None


class TestSwapToolResultListContent:
    """_swap_tool_result_content preserves list-of-blocks structure."""

    def test_swap_anthropic_list_content_preserves_structure(self):
        """Swap on list-of-blocks content replaces text in place."""
        from headroom.cache.compression_cache import _swap_tool_result_content

        msg = _make_tool_result_list_content_msg("t1", ["original"])
        swapped = _swap_tool_result_content(msg, "compressed")
        inner = swapped["content"][0]["content"]
        assert isinstance(inner, list)
        assert inner[0]["type"] == "text"
        assert inner[0]["text"] == "compressed"

    def test_swap_anthropic_string_content_preserved(self):
        """Swap on plain-string content still works."""
        from headroom.cache.compression_cache import _swap_tool_result_content

        msg = _make_tool_result_msg("t1", "original")
        swapped = _swap_tool_result_content(msg, "compressed")
        assert swapped["content"][0]["content"] == "compressed"

    def test_swap_openai_list_content(self):
        """Swap on OpenAI list-content replaces text."""
        from headroom.cache.compression_cache import _swap_tool_result_content

        msg = _make_openai_tool_list_content_msg("tc1", ["original"])
        swapped = _swap_tool_result_content(msg, "compressed")
        assert swapped["content"] == "compressed"

    def test_swap_does_not_mutate_original(self):
        """_swap_tool_result_content performs a deep copy."""
        from headroom.cache.compression_cache import _swap_tool_result_content

        msg = _make_tool_result_list_content_msg("t1", ["original"])
        _swap_tool_result_content(msg, "compressed")
        assert msg["content"][0]["content"][0]["text"] == "original"

    def test_swap_list_content_adds_text_block_when_missing(self):
        """When list has no text block, collapses to a single text block."""
        from headroom.cache.compression_cache import _swap_tool_result_content

        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "image", "source": {"type": "base64", "data": "..."}}],
                }
            ],
        }
        swapped = _swap_tool_result_content(msg, "compressed")
        inner = swapped["content"][0]["content"]
        assert isinstance(inner, list)
        assert len(inner) == 1
        assert inner[0]["type"] == "text"
        assert inner[0]["text"] == "compressed"
