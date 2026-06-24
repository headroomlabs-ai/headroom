"""Tests for Strands SDK content block tokenization (#111).

Strands SDK sends content blocks without a "type" field:
  {"text": "..."} instead of {"type": "text", "text": "..."}
  {"toolUse": {...}} instead of {"type": "tool_use", ...}
  {"toolResult": {...}} instead of {"type": "tool_result", ...}

The tokenizer must count these correctly.
"""

from headroom.tokenizers import get_tokenizer


def _get_counter():
    return get_tokenizer("claude-sonnet-4-6")


class TestStrandsTextBlocks:
    """Strands text blocks: {"text": "..."} without "type" field."""

    def test_strands_text_matches_anthropic_text(self):
        """Strands {"text": ...} should count same as Anthropic {"type": "text", "text": ...}."""
        t = _get_counter()
        text = "Hello world this is a test message " * 50

        anthropic = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        strands = [{"role": "user", "content": [{"text": text}]}]

        a = t.count_messages(anthropic)
        s = t.count_messages(strands)
        assert a == s, f"Anthropic={a}, Strands={s}"

    def test_strands_text_matches_plain_string(self):
        """Strands text block should count same as plain string content."""
        t = _get_counter()
        text = "Some question " * 1000

        plain = [{"role": "user", "content": text}]
        strands = [{"role": "user", "content": [{"text": text}]}]

        p = t.count_messages(plain)
        s = t.count_messages(strands)
        assert p == s, f"Plain={p}, Strands={s}"

    def test_strands_multiple_text_blocks(self):
        """Multiple Strands text blocks should all be counted."""
        t = _get_counter()

        msg = [
            {
                "role": "user",
                "content": [
                    {"text": "First block " * 100},
                    {"text": "Second block " * 100},
                ],
            }
        ]
        count = t.count_messages(msg)

        # Should be roughly 2x a single block
        single = [{"role": "user", "content": [{"text": "First block " * 100}]}]
        single_count = t.count_messages(single)

        assert count > single_count * 1.5, f"Multiple blocks={count}, single={single_count}"

    def test_strands_system_message(self):
        """System message with Strands text blocks."""
        t = _get_counter()
        text = "You are a helpful assistant. " * 200

        strands = [{"role": "system", "content": [{"text": text}]}]
        plain = [{"role": "system", "content": text}]

        s = t.count_messages(strands)
        p = t.count_messages(plain)
        assert s == p, f"Strands={s}, Plain={p}"


class TestStrandsToolBlocks:
    """Strands tool blocks: {"toolUse": {...}} and {"toolResult": {...}}."""

    def test_strands_tool_use_counted(self):
        """Strands toolUse block should be counted, not zero."""
        t = _get_counter()

        msg = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "read_file",
                            "input": {"path": "/src/main.py"},
                        }
                    }
                ],
            }
        ]
        count = t.count_messages(msg)
        # Should include the tool name and input, not just message overhead
        assert count > 15, f"toolUse count too low: {count}"

    def test_strands_tool_result_counted(self):
        """Strands toolResult with nested text content should be counted."""
        t = _get_counter()
        big_content = "File contents here. " * 500

        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [{"text": big_content}],
                        }
                    }
                ],
            }
        ]
        count = t.count_messages(msg)

        # Should reflect the size of the content, not just overhead
        plain_count = t.count_messages([{"role": "user", "content": big_content}])
        assert count > plain_count * 0.5, (
            f"toolResult count={count} should be close to plain={plain_count}"
        )

    def test_strands_tool_result_string_content(self):
        """Strands toolResult with string content."""
        t = _get_counter()

        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": "Simple string result " * 100,
                        }
                    }
                ],
            }
        ]
        count = t.count_messages(msg)
        assert count > 50, f"toolResult string count too low: {count}"


class TestMixedFormats:
    """Messages mixing Anthropic and Strands formats."""

    def test_mixed_conversation(self):
        """Full conversation with mixed Strands and Anthropic blocks."""
        t = _get_counter()
        messages = [
            # Strands system
            {"role": "system", "content": [{"text": "You are helpful. " * 50}]},
            # Strands user
            {"role": "user", "content": [{"text": "Fix the bug in auth.py"}]},
            # Strands assistant with toolUse
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "read_file",
                            "input": {"path": "auth.py"},
                        }
                    }
                ],
            },
            # Strands tool result
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [{"text": "def authenticate():\n    pass\n" * 100}],
                        }
                    }
                ],
            },
            # Anthropic-style text (for comparison)
            {"role": "assistant", "content": [{"type": "text", "text": "I found the issue."}]},
        ]
        count = t.count_messages(messages)
        # Should be substantial — the tool result alone is ~700 tokens
        assert count > 500, f"Mixed conversation count too low: {count}"


class TestStrandsReasoningContent:
    """Strands reasoning blocks: {"reasoningContent": {"reasoningText": {"text": "..."}}}."""

    def test_reasoning_text_counted_as_text(self):
        """reasoningContent text should be counted with count_text, not estimated."""
        t = _get_counter()
        reasoning = "Let me think step by step about this problem. " * 100

        # Strands format
        msg_strands = [
            {
                "role": "assistant",
                "content": [{"reasoningContent": {"reasoningText": {"text": reasoning}}}],
            }
        ]

        # Equivalent plain text for comparison
        msg_plain = [{"role": "assistant", "content": reasoning}]

        s = t.count_messages(msg_strands)
        p = t.count_messages(msg_plain)
        assert s == p, f"Reasoning={s} should equal plain text={p}"

    def test_reasoning_plus_text_both_counted(self):
        """Message with both reasoning and text blocks."""
        t = _get_counter()
        msg = [
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "thinking " * 200}}},
                    {"text": "Here is my answer " * 50},
                ],
            }
        ]
        count = t.count_messages(msg)
        # Should be substantial — both blocks counted
        assert count > 200, f"Combined reasoning+text too low: {count}"


class TestStrandsMediaContent:
    """Strands image, document, video blocks."""

    def test_image_not_zero(self):
        """Image block should have nonzero token count."""
        t = _get_counter()
        msg = [
            {
                "role": "user",
                "content": [{"image": {"format": "png", "source": {"bytes": b"x" * 50000}}}],
            }
        ]
        count = t.count_messages(msg)
        assert count > 100, f"Image count too low: {count}"

    def test_document_not_zero(self):
        """Document block should have nonzero token count."""
        t = _get_counter()
        msg = [
            {
                "role": "user",
                "content": [
                    {
                        "document": {
                            "format": "pdf",
                            "name": "report.pdf",
                            "source": {"bytes": b"x" * 30000},
                        }
                    }
                ],
            }
        ]
        count = t.count_messages(msg)
        assert count > 1000, f"Document count too low: {count}"

    def test_video_not_zero(self):
        """Video block should have nonzero token count."""
        t = _get_counter()
        msg = [
            {
                "role": "user",
                "content": [{"video": {"format": "mp4", "source": {"bytes": b"x" * 300000}}}],
            }
        ]
        count = t.count_messages(msg)
        assert count > 1000, f"Video count too low: {count}"


class TestStrandsFullConversation:
    """End-to-end conversation with all Strands content types."""

    def test_agent_conversation_with_reasoning_and_tools(self):
        """Realistic Strands agent conversation."""
        t = _get_counter()
        messages = [
            {"role": "user", "content": [{"text": "Analyze this code and fix the bug"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "Let me examine the code carefully. " * 50}
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "read_file",
                            "input": {"path": "main.py"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [
                                {
                                    "text": "def process():\n    data = fetch()\n    return transform(data)\n"
                                    * 50
                                }
                            ],
                        }
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "The bug is in the transform function. " * 30}
                        }
                    },
                    {"text": "I found the issue. The transform function doesn't handle None."},
                ],
            },
        ]
        count = t.count_messages(messages)
        # Reasoning + tool result + text = should be substantial
        assert count > 500, f"Full conversation too low: {count}"

        # Verify reasoning contributes meaningfully
        no_reasoning = [
            {"role": "user", "content": [{"text": "Analyze this code"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "read_file",
                            "input": {"path": "main.py"},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [
                                {
                                    "text": "def process():\n    data = fetch()\n    return transform(data)\n"
                                    * 50
                                }
                            ],
                        }
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"text": "I found the issue."},
                ],
            },
        ]
        count_no_reasoning = t.count_messages(no_reasoning)
        assert count > count_no_reasoning + 100, (
            f"Reasoning should add significant tokens: with={count}, without={count_no_reasoning}"
        )
