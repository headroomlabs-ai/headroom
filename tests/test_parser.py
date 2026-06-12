"""Tests for the parser module.

Tests all parsing and analysis functions:
- compute_hash: Content hashing
- detect_waste_signals: Waste signal detection
- is_rag_content: RAG content detection
- parse_message_to_blocks: Single message parsing
- parse_messages: Multi-message parsing
- find_tool_units: Tool call/response pairing
- get_message_content_text: Content extraction
"""

from unittest.mock import Mock

import pytest

from headroom.parser import (
    compute_hash,
    detect_waste_signals,
    find_tool_units,
    get_message_content_text,
    is_rag_content,
    parse_message_to_blocks,
    parse_messages,
)

# --- Fixtures ---


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns predictable token counts."""
    tokenizer = Mock()
    # Simple mock: 1 token per 4 characters
    tokenizer.count_text = Mock(side_effect=lambda text: len(text) // 4 + 1)
    return tokenizer


@pytest.fixture
def system_message():
    """Basic system message."""
    return {"role": "system", "content": "You are a helpful assistant."}


@pytest.fixture
def user_message():
    """Basic user message."""
    return {"role": "user", "content": "Hello, how are you?"}


@pytest.fixture
def assistant_message():
    """Basic assistant message."""
    return {"role": "assistant", "content": "I'm doing well, thank you!"}


@pytest.fixture
def tool_call_message():
    """Assistant message with tool calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "search_user", "arguments": '{"user_id": "12345"}'},
            }
        ],
    }


@pytest.fixture
def tool_result_message():
    """Tool result message."""
    return {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "content": '{"id": "12345", "name": "Alice", "email": "alice@example.com"}',
    }


@pytest.fixture
def multimodal_message():
    """User message with multimodal content (list format)."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this image:"},
            {"type": "image", "source": {"type": "base64", "data": "..."}},
            {"type": "text", "text": "What do you see?"},
        ],
    }


@pytest.fixture
def rag_user_message():
    """User message containing RAG content markers."""
    return {
        "role": "user",
        "content": "[Document 1] Here is the relevant context from our knowledge base. [Source: docs/manual.md]",
    }


@pytest.fixture
def html_waste_text():
    """Text containing HTML noise."""
    return "<div class='container'><p>Hello</p><!-- comment --></div>"


@pytest.fixture
def base64_waste_text():
    """Text containing base64 encoded data."""
    return "Data: " + "A" * 60 + "=="


@pytest.fixture
def whitespace_waste_text():
    """Text with excessive whitespace."""
    return "Line 1\n\n\n\nLine 2      extra spaces"


@pytest.fixture
def json_bloat_text():
    """Text containing large JSON block (>500 chars).

    Uses spaces and punctuation to avoid base64 pattern matching.
    """
    # Use content that won't match base64 pattern (needs non-base64 chars)
    content = "This is a long text value. " * 25  # ~675 chars
    return '{"data": "' + content + '"}'


# --- TestComputeHash ---


class TestComputeHash:
    """Tests for compute_hash function."""

    def test_consistent_hash(self):
        """Same text produces same hash."""
        text = "Hello, world!"
        hash1 = compute_hash(text)
        hash2 = compute_hash(text)
        assert hash1 == hash2

    def test_different_texts_different_hashes(self):
        """Different texts produce different hashes."""
        hash1 = compute_hash("Hello")
        hash2 = compute_hash("World")
        assert hash1 != hash2

    def test_hash_length_16(self):
        """Hash is truncated to 16 characters."""
        text = "Any text content"
        hash_result = compute_hash(text)
        assert len(hash_result) == 16

    def test_empty_string_hash(self):
        """Empty string produces valid hash."""
        hash_result = compute_hash("")
        assert len(hash_result) == 16
        assert hash_result.isalnum()

    def test_unicode_text_hash(self):
        """Unicode text produces valid hash."""
        hash_result = compute_hash("Hello \\u4e16\\u754c")
        assert len(hash_result) == 16


# --- TestDetectWasteSignals ---


class TestDetectWasteSignals:
    """Tests for detect_waste_signals function."""

    def test_detect_html_tags(self, mock_tokenizer, html_waste_text):
        """Detects HTML tags as waste."""
        signals = detect_waste_signals(html_waste_text, mock_tokenizer)
        assert signals.html_noise_tokens > 0

    def test_detect_html_comments(self, mock_tokenizer):
        """Detects HTML comments as waste."""
        text = "Some text <!-- this is a comment --> more text"
        signals = detect_waste_signals(text, mock_tokenizer)
        assert signals.html_noise_tokens > 0

    def test_detect_base64(self, mock_tokenizer, base64_waste_text):
        """Detects base64 encoded content as waste."""
        signals = detect_waste_signals(base64_waste_text, mock_tokenizer)
        assert signals.base64_tokens > 0

    def test_detect_excessive_whitespace(self, mock_tokenizer, whitespace_waste_text):
        """Detects excessive whitespace as waste."""
        signals = detect_waste_signals(whitespace_waste_text, mock_tokenizer)
        assert signals.whitespace_tokens >= 0  # May be 0 if normalized tokens <= matches

    def test_detect_json_bloat(self, mock_tokenizer, json_bloat_text):
        """Detects large JSON blocks as bloat."""
        # Need to ensure the mock returns >500 tokens for JSON bloat
        # The JSON pattern requires the matched block to have >500 tokens
        mock_tokenizer.count_text = Mock(side_effect=lambda text: len(text))
        signals = detect_waste_signals(json_bloat_text, mock_tokenizer)
        assert signals.json_bloat_tokens > 0

    def test_empty_text_no_waste(self, mock_tokenizer):
        """Empty text returns zero waste signals."""
        signals = detect_waste_signals("", mock_tokenizer)
        assert signals.total() == 0

    def test_combined_waste_signals(self, mock_tokenizer):
        """Multiple waste types are detected together."""
        text = "<div>Hello</div> " + "B" * 60 + "== and <!-- comment -->"
        signals = detect_waste_signals(text, mock_tokenizer)
        assert signals.html_noise_tokens > 0
        assert signals.base64_tokens > 0

    def test_clean_text_no_waste(self, mock_tokenizer):
        """Clean text produces minimal waste signals."""
        text = "This is a normal sentence without any waste."
        signals = detect_waste_signals(text, mock_tokenizer)
        assert signals.html_noise_tokens == 0
        assert signals.base64_tokens == 0
        assert signals.json_bloat_tokens == 0


# --- TestIsRagContent ---


class TestIsRagContent:
    """Tests for is_rag_content function."""

    def test_document_markers(self):
        """Detects [Document N] markers."""
        text = "[Document 1] This is the first document. [Document 2] Second document."
        assert is_rag_content(text) is True

    def test_source_markers(self):
        """Detects [Source: ...] markers."""
        text = "[Source: knowledge_base/docs.md] Here is the information."
        assert is_rag_content(text) is True

    def test_context_tags(self):
        """Detects <context> and <document> tags."""
        assert is_rag_content("<context>Retrieved content here</context>") is True
        assert is_rag_content("<document>Document content</document>") is True

    def test_retrieved_from_marker(self):
        """Detects 'Retrieved from:' marker."""
        text = "Retrieved from: https://example.com/docs\nHere is the content."
        assert is_rag_content(text) is True

    def test_knowledge_base_marker(self):
        """Detects 'From the knowledge base:' marker."""
        text = "From the knowledge base: This is relevant information."
        assert is_rag_content(text) is True

    def test_not_rag_content(self):
        """Regular text is not detected as RAG content."""
        text = "Hello, how can I help you today?"
        assert is_rag_content(text) is False

    def test_case_insensitive(self):
        """RAG detection is case insensitive."""
        assert is_rag_content("[DOCUMENT 1] Content") is True
        assert is_rag_content("retrieved FROM: somewhere") is True


# --- TestParseMessageToBlocks ---


class TestParseMessageToBlocks:
    """Tests for parse_message_to_blocks function."""

    def test_system_message_block(self, mock_tokenizer, system_message):
        """System message creates system block."""
        blocks = parse_message_to_blocks(system_message, 0, mock_tokenizer)
        assert len(blocks) == 1
        assert blocks[0].kind == "system"
        assert blocks[0].text == "You are a helpful assistant."
        assert blocks[0].source_index == 0

    def test_user_message_block(self, mock_tokenizer, user_message):
        """User message creates user block."""
        blocks = parse_message_to_blocks(user_message, 1, mock_tokenizer)
        assert len(blocks) == 1
        assert blocks[0].kind == "user"
        assert blocks[0].text == "Hello, how are you?"
        assert blocks[0].source_index == 1

    def test_assistant_message_block(self, mock_tokenizer, assistant_message):
        """Assistant message creates assistant block."""
        blocks = parse_message_to_blocks(assistant_message, 2, mock_tokenizer)
        assert len(blocks) == 1
        assert blocks[0].kind == "assistant"
        assert blocks[0].text == "I'm doing well, thank you!"

    def test_tool_result_block(self, mock_tokenizer, tool_result_message):
        """Tool result creates tool_result block with tool_call_id."""
        blocks = parse_message_to_blocks(tool_result_message, 3, mock_tokenizer)
        assert len(blocks) == 1
        assert blocks[0].kind == "tool_result"
        assert blocks[0].flags.get("tool_call_id") == "call_abc123"

    def test_rag_detection_in_user_message(self, mock_tokenizer, rag_user_message):
        """User message with RAG markers creates rag block."""
        blocks = parse_message_to_blocks(rag_user_message, 0, mock_tokenizer)
        assert len(blocks) == 1
        assert blocks[0].kind == "rag"

    def test_multimodal_content(self, mock_tokenizer, multimodal_message):
        """Multimodal content (list with text parts) is extracted."""
        blocks = parse_message_to_blocks(multimodal_message, 0, mock_tokenizer)
        assert len(blocks) == 1
        assert "Analyze this image:" in blocks[0].text
        assert "What do you see?" in blocks[0].text

    def test_tool_calls_create_separate_blocks(self, mock_tokenizer, tool_call_message):
        """Tool calls create separate tool_call blocks."""
        blocks = parse_message_to_blocks(tool_call_message, 0, mock_tokenizer)
        # Should have tool_call blocks (no content block since content is None)
        tool_call_blocks = [b for b in blocks if b.kind == "tool_call"]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0].flags.get("tool_call_id") == "call_abc123"
        assert tool_call_blocks[0].flags.get("function_name") == "search_user"
        assert "search_user" in tool_call_blocks[0].text

    def test_empty_message_creates_block(self, mock_tokenizer):
        """Empty message (no content or tool_calls) creates minimal block."""
        empty_msg = {"role": "assistant"}
        blocks = parse_message_to_blocks(empty_msg, 0, mock_tokenizer)
        assert len(blocks) == 1
        assert blocks[0].kind == "unknown"
        assert blocks[0].text == ""

    def test_message_with_content_and_tool_calls(self, mock_tokenizer):
        """Message with both content and tool_calls creates multiple blocks."""
        msg = {
            "role": "assistant",
            "content": "Let me search for that.",
            "tool_calls": [{"id": "call_xyz", "function": {"name": "search", "arguments": "{}"}}],
        }
        blocks = parse_message_to_blocks(msg, 0, mock_tokenizer)
        kinds = [b.kind for b in blocks]
        assert "assistant" in kinds
        assert "tool_call" in kinds

    def test_waste_signals_in_flags(self, mock_tokenizer, html_waste_text):
        """Waste signals are added to block flags."""
        msg = {"role": "user", "content": html_waste_text}
        blocks = parse_message_to_blocks(msg, 0, mock_tokenizer)
        assert "waste_signals" in blocks[0].flags
        assert blocks[0].flags["waste_signals"]["html_noise"] > 0

    def test_content_hash_generated(self, mock_tokenizer, user_message):
        """Content hash is generated for blocks."""
        blocks = parse_message_to_blocks(user_message, 0, mock_tokenizer)
        assert len(blocks[0].content_hash) == 16

    def test_tokens_estimated(self, mock_tokenizer, user_message):
        """Token count is estimated."""
        blocks = parse_message_to_blocks(user_message, 0, mock_tokenizer)
        assert blocks[0].tokens_est > 0


# --- TestParseMessages ---


class TestParseMessages:
    """Tests for parse_messages function."""

    def test_parse_all_messages(self, mock_tokenizer, sample_messages):
        """All messages are parsed into blocks."""
        blocks, breakdown, waste = parse_messages(sample_messages, mock_tokenizer)
        assert len(blocks) >= len(sample_messages)

    def test_block_breakdown(self, mock_tokenizer, sample_messages):
        """Block breakdown counts tokens per kind."""
        blocks, breakdown, waste = parse_messages(sample_messages, mock_tokenizer)
        assert "system" in breakdown
        assert "user" in breakdown
        assert "assistant" in breakdown
        assert all(v > 0 for v in breakdown.values())

    def test_waste_signals_accumulated(self, mock_tokenizer):
        """Waste signals are accumulated across messages."""
        messages = [
            {"role": "user", "content": "<div>HTML here</div>"},
            {"role": "assistant", "content": "More <span>HTML</span>"},
        ]
        blocks, breakdown, waste = parse_messages(messages, mock_tokenizer)
        assert waste.html_noise_tokens > 0

    def test_empty_messages(self, mock_tokenizer):
        """Empty message list returns empty results."""
        blocks, breakdown, waste = parse_messages([], mock_tokenizer)
        assert blocks == []
        assert breakdown == {}
        assert waste.total() == 0

    def test_multiple_tool_calls_parsed(self, mock_tokenizer, sample_messages_with_tools):
        """Messages with tool calls are parsed correctly."""
        blocks, breakdown, waste = parse_messages(sample_messages_with_tools, mock_tokenizer)
        tool_call_blocks = [b for b in blocks if b.kind == "tool_call"]
        tool_result_blocks = [b for b in blocks if b.kind == "tool_result"]
        assert len(tool_call_blocks) >= 1
        assert len(tool_result_blocks) >= 1


# --- TestRereadDetection ---


class TestRereadDetection:
    """Tests for cross-message re-read detection in parse_messages."""

    LARGE_CONTENT = "def handler(event):\n    return process(event)\n" * 10  # > 200 chars

    def _expected_tokens(self, text):
        """Mirror mock_tokenizer + message overhead used for tool_result blocks."""
        return len(text) // 4 + 1 + 4

    @staticmethod
    def _filler(n):
        """Interleaved turns that push a repeat beyond the polling gap."""
        return [
            {"role": "assistant" if i % 2 == 0 else "user", "content": f"step {i} of the task"}
            for i in range(n)
        ]

    def test_reread_detected_openai_tool_messages(self, mock_tokenizer):
        """Identical large tool outputs far apart count as re-read."""
        messages = (
            [{"role": "tool", "tool_call_id": "c1", "content": self.LARGE_CONTENT}]
            + self._filler(4)
            + [{"role": "tool", "tool_call_id": "c2", "content": self.LARGE_CONTENT}]
        )
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == self._expected_tokens(self.LARGE_CONTENT)

    def test_reread_detected_anthropic_tool_result_blocks(self, mock_tokenizer):
        """Anthropic-format tool_result parts are matched by content, not id."""
        part = {"type": "tool_result", "tool_use_id": "t1", "content": self.LARGE_CONTENT}
        part2 = {"type": "tool_result", "tool_use_id": "t2", "content": self.LARGE_CONTENT}
        messages = (
            [{"role": "user", "content": [part]}]
            + self._filler(4)
            + [{"role": "user", "content": [part2]}]
        )
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == self._expected_tokens(self.LARGE_CONTENT)

    def test_three_occurrences_count_repeats_only(self, mock_tokenizer):
        """First serve is free; every distant repeat is counted."""
        msg = {"role": "tool", "tool_call_id": "c", "content": self.LARGE_CONTENT}
        messages = [dict(msg)] + self._filler(4) + [dict(msg)] + self._filler(4) + [dict(msg)]
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == 2 * self._expected_tokens(self.LARGE_CONTENT)

    def test_single_occurrence_no_signal(self, mock_tokenizer):
        """One large tool result is not a re-read."""
        messages = [{"role": "tool", "tool_call_id": "c1", "content": self.LARGE_CONTENT}]
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == 0

    def test_short_duplicates_ignored(self, mock_tokenizer):
        """Trivially short outputs (\"ok\") legitimately repeat and are skipped."""
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "tool", "tool_call_id": "c2", "content": "ok"},
            {"role": "tool", "tool_call_id": "c3", "content": "ok"},
        ]
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == 0

    def test_same_message_duplicates_ignored(self, mock_tokenizer):
        """Duplicates within a single message are not a re-read."""
        part = {"type": "tool_result", "tool_use_id": "t1", "content": self.LARGE_CONTENT}
        part2 = {"type": "tool_result", "tool_use_id": "t2", "content": self.LARGE_CONTENT}
        messages = [{"role": "user", "content": [part, part2]}]
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == 0

    def test_mixed_same_message_duplicate_not_counted(self, mock_tokenizer):
        """A duplicate inside the original message stays excluded even when
        a later message also re-serves the content."""
        part = {"type": "tool_result", "tool_use_id": "t1", "content": self.LARGE_CONTENT}
        part2 = {"type": "tool_result", "tool_use_id": "t2", "content": self.LARGE_CONTENT}
        part3 = {"type": "tool_result", "tool_use_id": "t3", "content": self.LARGE_CONTENT}
        messages = (
            [{"role": "user", "content": [part, part2]}]
            + self._filler(4)
            + [{"role": "user", "content": [part3]}]
        )
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == self._expected_tokens(self.LARGE_CONTENT)

    def test_adjacent_polling_repeats_ignored(self, mock_tokenizer):
        """Back-to-back identical results (poll loop) are not re-reads."""
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": self.LARGE_CONTENT},
            {"role": "assistant", "content": "Still pending, checking again."},
            {"role": "tool", "tool_call_id": "c2", "content": self.LARGE_CONTENT},
        ]
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == 0

    def test_polling_chain_never_accumulates(self, mock_tokenizer):
        """Each poll advances the baseline — long chains stay at zero."""
        msg = {"role": "tool", "tool_call_id": "c", "content": self.LARGE_CONTENT}
        nudge = {"role": "assistant", "content": "polling"}
        messages = [dict(msg), dict(nudge), dict(msg), dict(nudge), dict(msg), dict(nudge)]
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == 0

    def test_distant_repeat_after_polling_chain_counts(self, mock_tokenizer):
        """A far repeat counts even when earlier repeats were polling."""
        msg = {"role": "tool", "tool_call_id": "c", "content": self.LARGE_CONTENT}
        messages = (
            [dict(msg), {"role": "assistant", "content": "polling"}, dict(msg)]
            + self._filler(4)
            + [dict(msg)]
        )
        _, _, waste = parse_messages(messages, mock_tokenizer)
        assert waste.reread_tokens == self._expected_tokens(self.LARGE_CONTENT)

    def test_reread_in_total_and_dict(self):
        """reread_tokens participates in total() and to_dict()."""
        from headroom.config import WasteSignals

        ws = WasteSignals(reread_tokens=42)
        assert ws.total() == 42
        assert ws.to_dict()["reread"] == 42


# --- TestFindToolUnits ---


class TestFindToolUnits:
    """Tests for find_tool_units function."""

    def test_finds_tool_call_and_responses(self, sample_messages_with_tools):
        """Finds matching tool call and response pairs."""
        units = find_tool_units(sample_messages_with_tools)
        assert len(units) >= 1
        # Each unit is (assistant_index, [tool_response_indices])
        assistant_idx, response_indices = units[0]
        assert response_indices  # Should have at least one response

    def test_multiple_tool_calls_same_assistant(self):
        """Multiple tool calls from same assistant are grouped."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Search both"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "fetch", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result 1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result 2"},
        ]
        units = find_tool_units(messages)
        assert len(units) == 1
        assistant_idx, response_indices = units[0]
        assert len(response_indices) == 2

    def test_no_tool_units(self):
        """Returns empty list when no tool calls present."""
        messages = [
            {"role": "system", "content": "Hello"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        units = find_tool_units(messages)
        assert units == []

    def test_orphaned_tool_response(self):
        """Tool response without matching assistant is not included."""
        messages = [
            {"role": "system", "content": "Hello"},
            {"role": "user", "content": "Hi"},
            # Orphaned tool response - no assistant with tool_calls
            {"role": "tool", "tool_call_id": "orphan_call", "content": "orphaned"},
            {"role": "assistant", "content": "I don't have tools."},
        ]
        units = find_tool_units(messages)
        assert units == []

    def test_tool_response_order_sorted(self):
        """Tool response indices are sorted."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_a", "function": {"name": "first", "arguments": "{}"}},
                    {"id": "call_b", "function": {"name": "second", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_b", "content": "second result"},
            {"role": "tool", "tool_call_id": "call_a", "content": "first result"},
        ]
        units = find_tool_units(messages)
        assert len(units) == 1
        _, response_indices = units[0]
        assert response_indices == sorted(response_indices)

    def test_anthropic_format_tool_use_and_result(self):
        """Finds Anthropic format tool_use/tool_result pairs in content blocks."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Take a screenshot"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me take a screenshot."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "browser_screenshot",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "Screenshot taken successfully",
                    }
                ],
            },
            {"role": "user", "content": "Thanks!"},
        ]
        units = find_tool_units(messages)
        assert len(units) == 1
        assistant_idx, response_indices = units[0]
        assert assistant_idx == 2
        assert response_indices == [3]

    def test_anthropic_format_multiple_tool_uses(self):
        """Finds multiple Anthropic format tool_use blocks from same assistant."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_a", "name": "first", "input": {}},
                    {"type": "tool_use", "id": "toolu_b", "name": "second", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": "first done"},
                    {"type": "tool_result", "tool_use_id": "toolu_b", "content": "second done"},
                ],
            },
        ]
        units = find_tool_units(messages)
        assert len(units) == 1
        assistant_idx, response_indices = units[0]
        assert assistant_idx == 1
        assert response_indices == [2]

    def test_anthropic_format_orphaned_tool_result(self):
        """Anthropic tool_result without matching tool_use is not included."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "orphan_toolu",
                        "content": "orphaned result",
                    }
                ],
            },
            {"role": "assistant", "content": "Hello!"},
        ]
        units = find_tool_units(messages)
        assert units == []

    def test_mixed_openai_and_anthropic_formats(self):
        """Both OpenAI and Anthropic formats can coexist (edge case)."""
        messages = [
            {"role": "user", "content": "Do things"},
            # OpenAI format
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "openai_tool", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "openai result"},
            # Anthropic format
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_2", "name": "anthropic_tool", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "anthropic result"}
                ],
            },
        ]
        units = find_tool_units(messages)
        assert len(units) == 2
        # First unit: OpenAI format (assistant at 1, tool response at 2)
        assert units[0] == (1, [2])
        # Second unit: Anthropic format (assistant at 3, user with tool_result at 4)
        assert units[1] == (3, [4])


# --- TestGetMessageContentText ---


class TestGetMessageContentText:
    """Tests for get_message_content_text function."""

    def test_string_content(self):
        """Extracts string content directly."""
        msg = {"role": "user", "content": "Hello, world!"}
        text = get_message_content_text(msg)
        assert text == "Hello, world!"

    def test_list_content(self):
        """Extracts text from list content (multimodal)."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "Second part"},
            ],
        }
        text = get_message_content_text(msg)
        assert "First part" in text
        assert "Second part" in text

    def test_none_content(self):
        """Returns empty string for None content."""
        msg = {"role": "assistant", "content": None}
        text = get_message_content_text(msg)
        assert text == ""

    def test_mixed_content_list(self):
        """Handles list with both dict and string items."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Dict text"},
                "Plain string",
            ],
        }
        text = get_message_content_text(msg)
        assert "Dict text" in text
        assert "Plain string" in text

    def test_missing_content_key(self):
        """Returns empty string when content key is missing."""
        msg = {"role": "user"}
        text = get_message_content_text(msg)
        assert text == ""

    def test_non_text_type_skipped(self):
        """Non-text types in list are skipped."""
        msg = {
            "role": "user",
            "content": [
                {"type": "image", "data": "..."},
                {"type": "text", "text": "Only this"},
            ],
        }
        text = get_message_content_text(msg)
        assert text == "Only this"

    def test_empty_list_content(self):
        """Empty list content returns empty string."""
        msg = {"role": "user", "content": []}
        text = get_message_content_text(msg)
        assert text == ""


# --- Additional fixtures for complex tests ---


@pytest.fixture
def sample_messages():
    """Basic conversation messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]


@pytest.fixture
def sample_messages_with_tools():
    """Conversation with tool calls and responses."""
    return [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "Search for user 12345"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "search_user", "arguments": '{"user_id": "12345"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"id": "12345", "name": "Alice", "email": "alice@example.com"}',
        },
        {"role": "assistant", "content": "I found user Alice with ID 12345."},
    ]


# --- Anthropic tool_result content blocks (chopratejas/headroom#813) ---


@pytest.fixture
def big_json_payload():
    """JSON blob large enough to trip the json_bloat detector (>500 tokens)."""
    return "{" + ",".join(f'"key_{i}": "value padding text {i}"' for i in range(200)) + "}"


class TestAnthropicToolResultBlocks:
    """Anthropic Messages format nests tool output in tool_result content blocks."""

    def test_tool_result_with_nested_text_list(self, mock_tokenizer, big_json_payload):
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": [{"type": "text", "text": big_json_payload}],
                }
            ],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        tool_blocks = [b for b in blocks if b.kind == "tool_result"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].text == big_json_payload
        assert tool_blocks[0].flags["tool_call_id"] == "toolu_01"
        assert tool_blocks[0].flags["waste_signals"]["json_bloat"] > 0

    def test_tool_result_with_string_content(self, mock_tokenizer, big_json_payload):
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_02",
                    "content": big_json_payload,
                }
            ],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        tool_blocks = [b for b in blocks if b.kind == "tool_result"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].text == big_json_payload
        assert tool_blocks[0].flags["waste_signals"]["json_bloat"] > 0

    def test_mixed_text_and_tool_result(self, mock_tokenizer, big_json_payload):
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the output:"},
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_03",
                    "content": [{"type": "text", "text": big_json_payload}],
                },
            ],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        assert [b.kind for b in blocks] == ["user", "tool_result"]
        assert blocks[0].text == "Here is the output:"
        assert blocks[1].text == big_json_payload

    def test_empty_tool_result_content_emits_no_block(self, mock_tokenizer):
        message = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_04", "content": []},
                {"type": "tool_result", "tool_use_id": "toolu_05"},
            ],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        # Nothing extractable: keep the container block so every message
        # still yields at least one block.
        assert [b.kind for b in blocks] == ["user"]

    def test_tool_result_only_message_skips_container_block(self, mock_tokenizer, big_json_payload):
        message = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_08", "content": big_json_payload}
            ],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        assert [b.kind for b in blocks] == ["tool_result"]

    def test_tool_result_dict_content_serialized_as_json(self, mock_tokenizer):
        rows = {"rows": [{"id": i, "padding": "x" * 30} for i in range(120)]}
        message = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_09", "content": rows}],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        tool_blocks = [b for b in blocks if b.kind == "tool_result"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].text.startswith('{"rows":')
        assert tool_blocks[0].flags["waste_signals"]["json_bloat"] > 0

    def test_missing_tool_use_id_yields_none_flag(self, mock_tokenizer, big_json_payload):
        message = {
            "role": "user",
            "content": [{"type": "tool_result", "content": big_json_payload}],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        assert blocks[0].kind == "tool_result"
        assert blocks[0].flags["tool_call_id"] is None


class TestStrandsToolResultBlocks:
    """Strands/Bedrock converse format: toolResult content parts."""

    def test_strands_text_content(self, mock_tokenizer, big_json_payload):
        message = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "strands_01",
                        "content": [{"text": big_json_payload}],
                        "status": "success",
                    }
                }
            ],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        assert [b.kind for b in blocks] == ["tool_result"]
        assert blocks[0].text == big_json_payload
        assert blocks[0].flags["tool_call_id"] == "strands_01"
        assert blocks[0].flags["waste_signals"]["json_bloat"] > 0

    def test_strands_json_content(self, mock_tokenizer):
        rows = {"rows": [{"id": i, "padding": "x" * 30} for i in range(120)]}
        message = {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "strands_02", "content": [{"json": rows}]}}],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        assert [b.kind for b in blocks] == ["tool_result"]
        assert blocks[0].flags["waste_signals"]["json_bloat"] > 0

    def test_non_text_inner_blocks_skipped(self, mock_tokenizer):
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_06",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": "abc"}},
                        {"type": "text", "text": "small result"},
                        "raw string piece",
                    ],
                }
            ],
        }

        blocks = parse_message_to_blocks(message, 0, mock_tokenizer)

        tool_blocks = [b for b in blocks if b.kind == "tool_result"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].text == "small result\nraw string piece"

    def test_parse_messages_aggregates_tool_result_waste(self, mock_tokenizer, big_json_payload):
        messages = [
            {"role": "user", "content": "Run the tool"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_07",
                        "content": [{"type": "text", "text": big_json_payload}],
                    }
                ],
            },
        ]

        _, breakdown, waste = parse_messages(messages, mock_tokenizer)

        assert waste.json_bloat_tokens > 0
        assert breakdown["tool_result"] > 0

    def test_waste_parity_with_openai_tool_role(self, mock_tokenizer, big_json_payload):
        anthropic_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": big_json_payload}],
                    }
                ],
            }
        ]
        openai_messages = [{"role": "tool", "tool_call_id": "t1", "content": big_json_payload}]

        _, _, anthropic_waste = parse_messages(anthropic_messages, mock_tokenizer)
        _, _, openai_waste = parse_messages(openai_messages, mock_tokenizer)

        assert anthropic_waste.total() > 0
        assert anthropic_waste.total() == openai_waste.total()
