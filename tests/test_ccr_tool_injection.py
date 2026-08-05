"""Tests for CCR tool injection and MCP integration."""

import json

from headroom.ccr import (
    CCR_TOOL_NAME,
    CCRToolInjector,
    create_ccr_tool_definition,
    create_system_instructions,
    parse_tool_call,
)


class TestCCRToolDefinition:
    """Test tool definition creation for different providers."""

    def test_anthropic_format(self):
        """Anthropic tool definition has correct format."""
        tool = create_ccr_tool_definition("anthropic")

        assert tool["name"] == CCR_TOOL_NAME
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "hash" in tool["input_schema"]["properties"]
        # Retrieval is by hash only — no query/search parameter.
        assert "query" not in tool["input_schema"]["properties"]
        assert tool["input_schema"]["required"] == ["hash"]

    def test_openai_format(self):
        """OpenAI tool definition has correct format."""
        tool = create_ccr_tool_definition("openai")

        assert tool["type"] == "function"
        assert tool["function"]["name"] == CCR_TOOL_NAME
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["required"] == ["hash"]

    def test_google_format(self):
        """Google tool definition has correct format."""
        tool = create_ccr_tool_definition("google")

        assert tool["name"] == CCR_TOOL_NAME
        assert "parameters" in tool
        assert tool["parameters"]["required"] == ["hash"]


class TestCCRToolInjector:
    """Test CCRToolInjector functionality."""

    def test_scan_for_markers_finds_hash(self):
        """Scanner detects compression markers in messages."""
        messages = [
            {"role": "user", "content": "Find errors"},
            {
                "role": "tool",
                "content": '[{"id": 1}]\n[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]',
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 1
        assert "abc123def456abc123def456" in hashes
        assert injector.has_compressed_content

    def test_scan_detects_read_lifecycle_stale_marker(self):
        """A read_lifecycle STALE marker carries a retrievable CCR hash via the
        'Retrieve original: hash=' phrase but never says 'compressed', so the
        other patterns miss it. The injector must still detect it, or the
        retrieve tool is not offered and the marker is unredeemable (#1006)."""
        ccr_hash = "a1b2c3d4e5f6a1b2c3d4e5f6"  # 24 hex chars (SHA-256[:24])
        messages = [
            {
                "role": "tool",
                "content": (
                    "[Read content stale: app.py was modified after this read — "
                    f"re-read the file for current content. Retrieve original: hash={ccr_hash}]"
                ),
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert ccr_hash in hashes
        assert injector.has_compressed_content

    def test_scan_for_markers_multiple_hashes(self):
        """Scanner finds multiple distinct hashes."""
        messages = [
            {
                "role": "tool",
                "content": "[50 items compressed to 5. Retrieve more: hash=aaa111111111aaa111111111]",
            },
            {
                "role": "tool",
                "content": "[200 items compressed to 20. Retrieve more: hash=bbb222222222bbb222222222]",
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 2
        assert "aaa111111111aaa111111111" in hashes
        assert "bbb222222222bbb222222222" in hashes

    def test_scan_no_duplicates(self):
        """Scanner deduplicates repeated hashes."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=aabbcc123456aabbcc123456]",
            },
            {
                "role": "assistant",
                "content": "I see [100 items compressed to 10. Retrieve more: hash=aabbcc123456aabbcc123456]",
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 1

    def test_scan_anthropic_content_blocks(self):
        """Scanner handles Anthropic's content block format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Find errors"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "[100 items compressed to 10. Retrieve more: hash=b10cf0a2b3c4b10cf0a2b3c4]",
                    },
                ],
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert "b10cf0a2b3c4b10cf0a2b3c4" in hashes

    def test_inject_tool_when_compression_detected(self):
        """Tool is injected when compression markers are found."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
            },
        ]

        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(None)

        assert was_injected
        assert len(tools) == 1
        assert tools[0]["name"] == CCR_TOOL_NAME

    def test_inject_tool_adds_to_existing(self):
        """CCR tool is added to existing tools list."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=e1e2e3f4f5f6e1e2e3f4f5f6]",
            },
        ]
        existing_tools = [{"name": "other_tool", "input_schema": {}}]

        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(existing_tools)

        assert was_injected
        assert len(tools) == 2
        assert tools[0]["name"] == "other_tool"
        assert tools[1]["name"] == CCR_TOOL_NAME

    def test_skip_injection_if_tool_present_anthropic(self):
        """Injection skipped if tool already present (Anthropic format)."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=aac123456789aac123456789]",
            },
        ]
        # Tool already present (e.g., from MCP)
        existing_tools = [{"name": CCR_TOOL_NAME, "input_schema": {}}]

        injector = CCRToolInjector(provider="anthropic")
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(existing_tools)

        assert not was_injected
        assert len(tools) == 1  # Not duplicated

    def test_skip_injection_if_tool_present_openai(self):
        """Injection skipped if tool already present (OpenAI format)."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=bbc456789012bbc456789012]",
            },
        ]
        # OpenAI format tool already present
        existing_tools = [
            {"type": "function", "function": {"name": CCR_TOOL_NAME, "parameters": {}}}
        ]

        injector = CCRToolInjector(provider="openai")
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(existing_tools)

        assert not was_injected
        assert len(tools) == 1

    def test_no_injection_without_compression(self):
        """No injection when no compression markers found."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": '{"result": "ok"}'},
        ]

        injector = CCRToolInjector()
        injector.scan_for_markers(messages)
        tools, was_injected = injector.inject_tool_definition(None)

        assert not was_injected
        assert tools == []

    def test_inject_system_instructions(self):
        """System instructions are injected when compression detected."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
            },
        ]

        injector = CCRToolInjector(inject_system_instructions=True)
        injector.scan_for_markers(messages)
        updated = injector.inject_into_system_message(messages)

        assert "Compressed Context Available" in updated[0]["content"]
        assert "abc123def456abc123def456" in updated[0]["content"]

    def test_process_request_full_flow(self):
        """process_request handles complete injection flow."""
        messages = [
            {"role": "system", "content": "Assistant"},
            {"role": "user", "content": "Search for errors"},
            {
                "role": "tool",
                "content": "[500 items compressed to 25. Retrieve more: hash=f011f10abcdef011f10abcde]",
            },
        ]

        injector = CCRToolInjector(
            provider="anthropic",
            inject_tool=True,
            inject_system_instructions=True,
        )
        updated_messages, updated_tools, was_injected = injector.process_request(messages, None)

        assert was_injected
        assert updated_tools is not None
        assert len(updated_tools) == 1
        assert updated_tools[0]["name"] == CCR_TOOL_NAME
        assert "Compressed Context Available" in updated_messages[0]["content"]


class TestParseToolCall:
    """Test parsing of tool calls from LLM responses."""

    def test_parse_anthropic_format(self):
        """Parse Anthropic tool call format."""
        tool_call = {
            "id": "toolu_123",
            "name": CCR_TOOL_NAME,
            "input": {"hash": "abc123def456abc123def456"},
        }

        hash_key = parse_tool_call(tool_call, "anthropic")

        assert hash_key == "abc123def456abc123def456"

    def test_parse_openai_format(self):
        """Parse OpenAI tool call format."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": CCR_TOOL_NAME,
                "arguments": json.dumps({"hash": "def456abc123def456abc123"}),
            },
        }

        hash_key = parse_tool_call(tool_call, "openai")

        assert hash_key == "def456abc123def456abc123"

    def test_parse_null_function_returns_none_without_crashing(self):
        """A tool call with an explicit {"function": null} / {"functionCall": null}
        must return None, not raise AttributeError."""
        assert parse_tool_call({"id": "c1", "function": None}, "openai") is None
        assert parse_tool_call({"functionCall": None}, "google") is None

    def test_parse_normalises_uppercase_hash_to_lowercase(self):
        """An uppercase hash echoed by the model must be lowercased so it
        matches the store (which keys entries by a lowercase hash)."""
        tool_call = {
            "id": "toolu_123",
            "name": CCR_TOOL_NAME,
            "input": {"hash": "ABC123DEF456ABC123DEF456"},
        }

        hash_key = parse_tool_call(tool_call, "anthropic")

        assert hash_key == "abc123def456abc123def456"

    def test_parse_non_ccr_tool(self):
        """Returns None for non-CCR tool calls."""
        tool_call = {
            "name": "other_tool",
            "input": {"param": "value"},
        }

        hash_key = parse_tool_call(tool_call, "anthropic")

        assert hash_key is None

    def test_parse_malformed_openai_args(self):
        """Handles malformed JSON in OpenAI arguments."""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": CCR_TOOL_NAME,
                "arguments": "not valid json",
            },
        }

        hash_key = parse_tool_call(tool_call, "openai")

        assert hash_key is None

    def test_parse_openai_non_object_arguments_returns_none(self):
        """OpenAI arguments that decode to a non-object (array/string/number)
        must return None, not crash on `.get`."""
        for args in ("[]", '"abc"', "123"):
            tool_call = {"function": {"name": CCR_TOOL_NAME, "arguments": args}}
            assert parse_tool_call(tool_call, "openai") is None

    def test_parse_openai_null_arguments_returns_none(self):
        """A null `arguments` value (json.loads(None) -> TypeError) is handled."""
        tool_call = {"function": {"name": CCR_TOOL_NAME, "arguments": None}}
        assert parse_tool_call(tool_call, "openai") is None

    def test_parse_anthropic_non_dict_input_returns_none(self):
        """A non-dict Anthropic `input` must return None, not crash."""
        tool_call = {"name": CCR_TOOL_NAME, "input": ["not", "a", "dict"]}
        assert parse_tool_call(tool_call, "anthropic") is None


class TestHashSecurityValidation:
    """Test hash validation security measures.

    CCR hashes are 12 hex chars (SmartCrusher) or 24 hex chars (legacy
    bracket markers / compression_store). Any other length or non-hex input
    is rejected to prevent hash spoofing with malformed hashes.
    """

    def test_rejects_short_hash(self):
        """Rejects hash that's too short (potential spoofing attack)."""
        tool_call = {
            "name": CCR_TOOL_NAME,
            "input": {"hash": "abc123"},  # Only 6 chars
        }

        hash_key = parse_tool_call(tool_call, "anthropic")
        assert hash_key is None  # Rejected

    def test_rejects_long_hash(self):
        """Rejects hash that's too long."""
        tool_call = {
            "name": CCR_TOOL_NAME,
            "input": {"hash": "abc123def456abc123def456abc123"},  # 30 chars
        }

        hash_key = parse_tool_call(tool_call, "anthropic")
        assert hash_key is None  # Rejected

    def test_rejects_non_hex_characters(self):
        """Rejects hash with non-hex characters."""
        tool_call = {
            "name": CCR_TOOL_NAME,
            "input": {"hash": "abc123xyz456abc123xyz456"},  # Contains xyz
        }

        hash_key = parse_tool_call(tool_call, "anthropic")
        assert hash_key is None  # Rejected

    def test_accepts_valid_24_char_hash(self):
        """Accepts properly formatted 24-char hex hash."""
        tool_call = {
            "name": CCR_TOOL_NAME,
            "input": {"hash": "abc123def456abc123def456"},
        }

        hash_key = parse_tool_call(tool_call, "anthropic")
        assert hash_key == "abc123def456abc123def456"

    def test_accepts_uppercase_hex(self):
        """Accepts uppercase hex characters (normalized to lowercase internally)."""
        tool_call = {
            "name": CCR_TOOL_NAME,
            "input": {"hash": "ABC123DEF456ABC123DEF456"},
        }

        hash_key = parse_tool_call(tool_call, "anthropic")
        # Note: validation accepts uppercase since we use .lower() for hex check
        assert hash_key == "abc123def456abc123def456"


class TestSmartCrusherCcrMarkers:
    """Regression tests for issue #1095.

    SmartCrusher emits 12-hex-char hashes inside ``<<ccr:HASH ...>>`` markers
    (the row-drop summary and the opaque-blob form). The injector must detect
    those markers and ``parse_tool_call`` must accept the 12-char hashes —
    previously both only recognized the 24-char legacy bracket markers.
    """

    def test_scan_detects_row_drop_marker(self):
        """Detects ``<<ccr:HASH N_rows_offloaded>>`` (12-char hash)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "x",
                        "content": '{"kept": 12, "ccr": "<<ccr:e21a26620105 988_rows_offloaded>>"}',
                    }
                ],
            }
        ]

        injector = CCRToolInjector(provider="anthropic")
        hashes = injector.scan_for_markers(messages)

        assert hashes == ["e21a26620105"]
        assert injector.has_compressed_content

    def test_scan_detects_opaque_blob_marker(self):
        """Detects the ``<<ccr:HASH,KIND,SIZE>>`` opaque-blob form."""
        messages = [
            {"role": "tool", "content": "<<ccr:deadbeefdead,string,2.3KB>>"},
        ]

        hashes = CCRToolInjector().scan_for_markers(messages)

        assert hashes == ["deadbeefdead"]

    def test_legacy_bracket_marker_still_detected(self):
        """The 24-char legacy bracket marker keeps working alongside the new one."""
        messages = [
            {
                "role": "tool",
                "content": "[100 items compressed to 10. Retrieve more: hash=abc123def456abc123def456]",
            },
        ]

        hashes = CCRToolInjector().scan_for_markers(messages)

        assert hashes == ["abc123def456abc123def456"]

    def test_parse_tool_call_accepts_12_char_hash(self):
        """``parse_tool_call`` accepts a 12-char SmartCrusher hash."""
        tool_call = {
            "name": CCR_TOOL_NAME,
            "input": {"hash": "e21a26620105"},
        }

        hash_key = parse_tool_call(tool_call, "anthropic")

        assert hash_key == "e21a26620105"

    def test_parse_tool_call_still_accepts_24_char_hash(self):
        """24-char legacy hashes remain valid (regression guard)."""
        tool_call = {
            "name": CCR_TOOL_NAME,
            "input": {"hash": "abc123def456abc123def456"},
        }

        hash_key = parse_tool_call(tool_call, "anthropic")

        assert hash_key == "abc123def456abc123def456"


class TestSystemInstructions:
    """Test system instruction generation."""

    def test_create_instructions_single_hash(self):
        """Instructions include single hash."""
        instructions = create_system_instructions(["hash123"])

        assert "hash123" in instructions
        assert CCR_TOOL_NAME in instructions
        assert "Compressed Context Available" in instructions

    def test_create_instructions_multiple_hashes(self):
        """Instructions include multiple hashes."""
        hashes = ["hash1", "hash2", "hash3"]
        instructions = create_system_instructions(hashes)

        for h in hashes:
            assert h in instructions

    def test_create_instructions_truncates_many_hashes(self):
        """Instructions truncate when many hashes present."""
        hashes = [f"hash{i}" for i in range(10)]
        instructions = create_system_instructions(hashes)

        # First 5 should be present, rest truncated
        assert "hash0" in instructions
        assert "hash4" in instructions
        assert "..." in instructions


class TestAlternativeMarkerFormats:
    """Test CCR marker detection for different compressor formats.

    Different compressors use slightly different marker formats:
    - SmartCrusher: [N items compressed to M. Retrieve more: hash=xxx]
    - TextCompressor: [N lines compressed to M. Retrieve more: hash=xxx]
    - LogCompressor: [N lines compressed to M. Retrieve more: hash=xxx]
    - SearchCompressor: [N matches compressed to M. Retrieve more: hash=xxx]
    - Kompress: [N items compressed to M. Retrieve more: hash=xxx]

    The CCRToolInjector should detect all these formats.
    """

    def test_textcompressor_format(self):
        """Detects TextCompressor marker format (lines)."""
        messages = [
            {
                "role": "assistant",
                "content": "Build output:\n[500 lines compressed to 50. Retrieve more: hash=aabbccddeeff001122334455]",
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 1
        assert "aabbccddeeff001122334455" in hashes

    def test_searchcompressor_format(self):
        """Detects SearchCompressor marker format (matches)."""
        messages = [
            {
                "role": "assistant",
                "content": "Search results:\n[100 matches compressed to 10. Retrieve more: hash=112233445566778899001122]",
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 1
        assert "112233445566778899001122" in hashes

    def test_mixed_compressor_formats(self):
        """Detects multiple marker formats in same conversation."""
        messages = [
            {
                "role": "assistant",
                "content": "Search results:\n[50 matches compressed to 5. Retrieve more: hash=aaaa11111111aaaa11111111]",
            },
            {
                "role": "assistant",
                "content": "Build logs:\n[200 lines compressed to 20. Retrieve more: hash=bbbb22222222bbbb22222222]",
            },
            {
                "role": "assistant",
                "content": "Database:\n[1000 items compressed to 100. Retrieve more: hash=cccc33333333cccc33333333]",
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 3
        assert "aaaa11111111aaaa11111111" in hashes
        assert "bbbb22222222bbbb22222222" in hashes
        assert "cccc33333333cccc33333333" in hashes

    def test_generic_compressed_marker(self):
        """Detects generic compression markers via fallback pattern."""
        messages = [
            {
                "role": "assistant",
                "content": "Data:\n[Content compressed for efficiency. hash=fedcba9876543210fedcba98]",
            },
        ]

        injector = CCRToolInjector()
        hashes = injector.scan_for_markers(messages)

        assert len(hashes) == 1
        assert "fedcba9876543210fedcba98" in hashes
