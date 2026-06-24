"""Integration tests: verify tags survive the full compress() pipeline."""

from __future__ import annotations

import os

import pytest


class TestCompressPreservesTags:
    """End-to-end tests: compress() with tagged content."""

    def test_system_reminder_survives(self):
        """<system-reminder> tags in tool output survive compression."""
        from headroom import compress

        messages = [
            {"role": "user", "content": "What are the rules?"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "Here is a very long and verbose explanation of the system rules "
                    "that contains a lot of unnecessary filler words and repetitive "
                    "phrasing that could be compressed significantly. "
                    "<system-reminder>You must always validate input before processing. "
                    "Never skip authentication checks.</system-reminder> "
                    "The rest of this text is also quite verbose and contains additional "
                    "unnecessary details that add tokens without adding information value "
                    "to the overall response that the language model needs to generate."
                ),
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        output = str(result.messages[-1].get("content", ""))

        assert "<system-reminder>" in output
        assert "</system-reminder>" in output
        assert "validate input" in output
        assert "authentication checks" in output

    def test_tool_call_tags_survive(self):
        """<tool_call> tags survive compression."""
        from headroom import compress

        messages = [
            {"role": "user", "content": "Search for results"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "Processing the search request with extensive verbose output "
                    "that includes many redundant descriptions and unnecessary detail. "
                    '<tool_call>{"name": "search", "args": {"query": "test"}}</tool_call> '
                    "Additional verbose context that repeats information already stated "
                    "in the previous paragraphs about the search functionality and its "
                    "various capabilities and features that are not directly relevant."
                ),
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        output = str(result.messages[-1].get("content", ""))

        assert "<tool_call>" in output
        assert "</tool_call>" in output
        assert '"name": "search"' in output

    def test_thinking_tags_survive(self):
        """<thinking> tags survive compression."""
        from headroom import compress

        messages = [
            {"role": "user", "content": "Analyze this data"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "The analysis produced extensive results with much verbose "
                    "explanatory text that describes methodology in great detail. "
                    "<thinking>Step 1: Parse input. Step 2: Validate schema. "
                    "Step 3: Run inference.</thinking> "
                    "Further verbose explanation of the analytical process and its "
                    "various stages and intermediate results and observations that "
                    "could be expressed much more concisely without losing meaning."
                ),
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        output = str(result.messages[-1].get("content", ""))

        assert "<thinking>" in output
        assert "</thinking>" in output
        assert "Step 1: Parse input" in output

    def test_html_tags_still_compressible(self):
        """Standard HTML tags are NOT protected — they're just text to the compressor."""
        from headroom.transforms.tag_protector import protect_tags

        html_text = "<div>Some content</div> <span>More content</span>"
        cleaned, protected = protect_tags(html_text)

        # HTML tags should NOT be protected
        assert protected == []
        assert "<div>" in cleaned

    def test_multiple_custom_tags_in_messages(self):
        """Multiple different custom tags all survive."""
        from headroom import compress

        messages = [
            {"role": "user", "content": "What should I do?"},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "Very verbose introductory text with many unnecessary words. "
                    "<system-reminder>Rule 1: Always validate</system-reminder> "
                    "More verbose middle text repeating previous information. "
                    "<context>session_id=abc-123</context> "
                    "Additional verbose concluding text with redundant information. "
                    "<IMPORTANT>Never expose API keys</IMPORTANT> "
                    "Final paragraph of verbose text."
                ),
            },
        ]
        result = compress(messages, model="claude-sonnet-4-5-20250929")
        output = str(result.messages[-1].get("content", ""))

        assert "<system-reminder>" in output
        assert "<context>" in output
        assert "<IMPORTANT>" in output


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
@pytest.mark.slow
class TestRealAPIWithTags:
    """Real API integration: verify tags survive compression end-to-end."""

    def test_tags_survive_compression_for_api(self):
        """Compress content with custom tags, verify tags + content intact before API call.

        This tests the compression pipeline, not Claude's behavior.
        We verify the compressed output still contains the protected tags
        and their content — which is what matters for tool/workflow correctness.
        """
        from headroom import compress

        # Tool output with workflow tags (realistic: tags appear in tool results,
        # not user messages)
        messages = [
            {"role": "user", "content": "Show me the deployment configuration"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_config_001",
                        "type": "function",
                        "function": {
                            "name": "get_config",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_config_001",
                "content": (
                    "Here is a very long and detailed configuration document with "
                    "extensive verbose descriptions of various system parameters and "
                    "their default values and recommended settings for production "
                    "deployment scenarios across different cloud providers and regions. "
                    "<deployment-config region='us-east-1' tier='production'>"
                    "max_connections=500, timeout_ms=3000, retry_count=3"
                    "</deployment-config> "
                    "Additional verbose documentation about system architecture and "
                    "deployment patterns and scaling strategies and monitoring setup "
                    "and alerting configuration and incident response procedures and "
                    "backup strategies and disaster recovery planning guidelines."
                ),
            },
        ]

        result = compress(messages, model="claude-sonnet-4-5-20250929")

        # Verify the custom tags survived compression
        tool_content = str(result.messages[-1].get("content", ""))
        assert "<deployment-config" in tool_content, "Opening tag was stripped"
        assert "</deployment-config>" in tool_content, "Closing tag was stripped"
        assert "max_connections=500" in tool_content, "Tag content was stripped"
        assert "timeout_ms=3000" in tool_content, "Tag content was stripped"
