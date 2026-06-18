"""Tests for Google multimodal content preservation in the proxy.

Tests verify that:
1. _has_non_text_parts correctly detects non-text parts (images, files, function calls/responses)
2. _gemini_contents_to_messages returns preserved indices correctly
3. The preservation flow works end-to-end with real Gemini format structures

Uses REAL Google Gemini API format structures without any mocking.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from headroom.proxy.server import HeadroomProxy, ProxyConfig


@pytest.fixture
def proxy():
    """Create a minimal HeadroomProxy instance for testing helper methods."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    return HeadroomProxy(config)


# =============================================================================
# Test data: Real Google Gemini API format structures
# =============================================================================

# Text-only content
TEXT_ONLY_CONTENT = {"role": "user", "parts": [{"text": "Hello, world!"}]}

# Content with inline image (base64 encoded)
IMAGE_INLINE_CONTENT = {
    "role": "user",
    "parts": [
        {"text": "What's in this image?"},
        {"inlineData": {"mimeType": "image/jpeg", "data": "base64encodedimagedata..."}},
    ],
}

# Content with only inline image (no text)
IMAGE_ONLY_CONTENT = {
    "role": "user",
    "parts": [
        {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"}},
    ],
}

# Content with file reference (Google Cloud Storage)
FILE_DATA_CONTENT = {
    "role": "user",
    "parts": [
        {"text": "Summarize this document"},
        {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/document.pdf"}},
    ],
}

# Content with function call (model response)
FUNCTION_CALL_CONTENT = {
    "role": "model",
    "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}}],
}

# Content with function call and text
FUNCTION_CALL_WITH_TEXT_CONTENT = {
    "role": "model",
    "parts": [
        {"text": "Let me check the weather for you."},
        {"functionCall": {"name": "get_weather", "args": {"location": "San Francisco"}}},
    ],
}

# Content with function response (user provides)
FUNCTION_RESPONSE_CONTENT = {
    "role": "user",
    "parts": [
        {
            "functionResponse": {
                "name": "get_weather",
                "response": {"temperature": 72, "condition": "sunny"},
            }
        }
    ],
}

# Content with multiple images
MULTI_IMAGE_CONTENT = {
    "role": "user",
    "parts": [
        {"text": "Compare these two images"},
        {"inlineData": {"mimeType": "image/jpeg", "data": "firstimagebase64..."}},
        {"inlineData": {"mimeType": "image/jpeg", "data": "secondimagebase64..."}},
    ],
}

# Model response with only text
MODEL_TEXT_CONTENT = {
    "role": "model",
    "parts": [{"text": "Hello! How can I help you today?"}],
}

# Empty parts list
EMPTY_PARTS_CONTENT = {"role": "user", "parts": []}

# Content with mixed media types
MIXED_MEDIA_CONTENT = {
    "role": "user",
    "parts": [
        {"text": "Analyze this image and document"},
        {"inlineData": {"mimeType": "image/png", "data": "imagedata..."}},
        {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://bucket/file.pdf"}},
    ],
}


# =============================================================================
# Tests for _has_non_text_parts
# =============================================================================


class TestHasNonTextParts:
    """Test _has_non_text_parts correctly detects non-text content types."""

    def test_text_only_returns_false(self, proxy):
        """Content with only text parts returns False."""
        assert proxy._has_non_text_parts(TEXT_ONLY_CONTENT) is False

    def test_model_text_only_returns_false(self, proxy):
        """Model response with only text returns False."""
        assert proxy._has_non_text_parts(MODEL_TEXT_CONTENT) is False

    def test_empty_parts_returns_false(self, proxy):
        """Content with empty parts list returns False."""
        assert proxy._has_non_text_parts(EMPTY_PARTS_CONTENT) is False

    def test_inline_data_returns_true(self, proxy):
        """Content with inlineData (images) returns True."""
        assert proxy._has_non_text_parts(IMAGE_INLINE_CONTENT) is True

    def test_inline_data_only_returns_true(self, proxy):
        """Content with only inlineData (no text) returns True."""
        assert proxy._has_non_text_parts(IMAGE_ONLY_CONTENT) is True

    def test_file_data_returns_true(self, proxy):
        """Content with fileData returns True."""
        assert proxy._has_non_text_parts(FILE_DATA_CONTENT) is True

    def test_function_call_returns_true(self, proxy):
        """Content with functionCall returns True."""
        assert proxy._has_non_text_parts(FUNCTION_CALL_CONTENT) is True

    def test_function_call_with_text_returns_true(self, proxy):
        """Content with functionCall and text returns True."""
        assert proxy._has_non_text_parts(FUNCTION_CALL_WITH_TEXT_CONTENT) is True

    def test_function_response_returns_true(self, proxy):
        """Content with functionResponse returns True."""
        assert proxy._has_non_text_parts(FUNCTION_RESPONSE_CONTENT) is True

    def test_multiple_images_returns_true(self, proxy):
        """Content with multiple images returns True."""
        assert proxy._has_non_text_parts(MULTI_IMAGE_CONTENT) is True

    def test_mixed_media_returns_true(self, proxy):
        """Content with mixed media types returns True."""
        assert proxy._has_non_text_parts(MIXED_MEDIA_CONTENT) is True

    @pytest.mark.parametrize(
        "non_text_key",
        ["inlineData", "fileData", "functionCall", "functionResponse"],
    )
    def test_each_non_text_key_detected(self, proxy, non_text_key):
        """Each non-text part type is correctly detected."""
        content = {"role": "user", "parts": [{non_text_key: {"dummy": "data"}}]}
        assert proxy._has_non_text_parts(content) is True

    def test_content_without_parts_key(self, proxy):
        """Content missing 'parts' key returns False (graceful handling)."""
        content = {"role": "user"}
        assert proxy._has_non_text_parts(content) is False


# =============================================================================
# Tests for _gemini_contents_to_messages preserved indices
# =============================================================================


class TestGeminiContentsToMessagesPreservedIndices:
    """Test _gemini_contents_to_messages returns correct preserved indices."""

    def test_pure_text_returns_empty_set(self, proxy):
        """Pure text content returns empty preserved_indices set."""
        contents = [
            TEXT_ONLY_CONTENT,
            MODEL_TEXT_CONTENT,
            {"role": "user", "parts": [{"text": "Another question"}]},
        ]
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == set()
        assert len(messages) == 3

    def test_single_image_content_preserves_index(self, proxy):
        """Single content with image preserves its index."""
        contents = [IMAGE_INLINE_CONTENT]
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == {0}
        assert len(messages) == 1

    def test_image_at_beginning_preserves_correct_index(self, proxy):
        """Image at beginning of conversation preserves index 0."""
        contents = [
            IMAGE_INLINE_CONTENT,  # index 0 - has image
            MODEL_TEXT_CONTENT,  # index 1 - text only
            {"role": "user", "parts": [{"text": "Follow up"}]},  # index 2 - text only
        ]
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == {0}
        assert len(messages) == 3

    def test_image_at_middle_preserves_correct_index(self, proxy):
        """Image in middle of conversation preserves correct index."""
        contents = [
            TEXT_ONLY_CONTENT,  # index 0 - text only
            IMAGE_INLINE_CONTENT,  # index 1 - has image
            MODEL_TEXT_CONTENT,  # index 2 - text only
        ]
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == {1}
        assert len(messages) == 3

    def test_image_at_end_preserves_correct_index(self, proxy):
        """Image at end of conversation preserves correct index."""
        contents = [
            TEXT_ONLY_CONTENT,  # index 0 - text only
            MODEL_TEXT_CONTENT,  # index 1 - text only
            IMAGE_INLINE_CONTENT,  # index 2 - has image
        ]
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == {2}
        assert len(messages) == 3

    def test_multiple_images_preserves_all_indices(self, proxy):
        """Multiple contents with images preserve all their indices."""
        contents = [
            IMAGE_INLINE_CONTENT,  # index 0 - has image
            MODEL_TEXT_CONTENT,  # index 1 - text only
            FILE_DATA_CONTENT,  # index 2 - has file
            {"role": "model", "parts": [{"text": "Response"}]},  # index 3 - text only
            MULTI_IMAGE_CONTENT,  # index 4 - has multiple images
        ]
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == {0, 2, 4}
        assert len(messages) == 5

    def test_function_call_preserves_index(self, proxy):
        """Content with function call preserves its index."""
        contents = [
            TEXT_ONLY_CONTENT,  # index 0
            FUNCTION_CALL_CONTENT,  # index 1 - has function call
            FUNCTION_RESPONSE_CONTENT,  # index 2 - has function response
        ]
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == {1, 2}

    def test_all_non_text_preserves_all(self, proxy):
        """Conversation with all non-text content preserves all indices."""
        contents = [
            IMAGE_INLINE_CONTENT,  # index 0
            FUNCTION_CALL_CONTENT,  # index 1
            FUNCTION_RESPONSE_CONTENT,  # index 2
            FILE_DATA_CONTENT,  # index 3
        ]
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == {0, 1, 2, 3}

    def test_with_system_instruction(self, proxy):
        """System instruction does not affect content indexing."""
        contents = [
            TEXT_ONLY_CONTENT,  # index 0
            IMAGE_INLINE_CONTENT,  # index 1
        ]
        system_instruction = {"parts": [{"text": "You are a helpful assistant."}]}

        messages, preserved_indices = proxy._gemini_contents_to_messages(
            contents, system_instruction
        )

        # preserved_indices should reference content indices, not message indices
        assert preserved_indices == {1}
        # Messages should include system + 2 content messages
        assert len(messages) == 3
        assert messages[0]["role"] == "system"

    def test_empty_contents_returns_empty_set(self, proxy):
        """Empty contents list returns empty preserved_indices."""
        messages, preserved_indices = proxy._gemini_contents_to_messages([])

        assert preserved_indices == set()
        assert messages == []


# =============================================================================
# Tests for message conversion correctness
# =============================================================================


class TestGeminiContentsToMessagesConversion:
    """Test that _gemini_contents_to_messages correctly converts content."""

    def test_role_mapping_user(self, proxy):
        """User role is preserved."""
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        messages, _ = proxy._gemini_contents_to_messages(contents)

        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_role_mapping_model_to_assistant(self, proxy):
        """Model role is mapped to assistant."""
        contents = [{"role": "model", "parts": [{"text": "Hi there"}]}]
        messages, _ = proxy._gemini_contents_to_messages(contents)

        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi there"

    def test_multiple_text_parts_joined(self, proxy):
        """Multiple text parts in one content are joined."""
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": "First part."},
                    {"text": "Second part."},
                ],
            }
        ]
        messages, _ = proxy._gemini_contents_to_messages(contents)

        assert messages[0]["content"] == "First part.\nSecond part."

    def test_text_extracted_from_mixed_content(self, proxy):
        """Text is extracted from content with mixed parts."""
        contents = [IMAGE_INLINE_CONTENT]  # Has text + inlineData
        messages, _ = proxy._gemini_contents_to_messages(contents)

        assert messages[0]["content"] == "What's in this image?"

    def test_content_with_only_non_text_creates_empty_message(self, proxy):
        """Content with only non-text parts creates no message (no text to extract)."""
        contents = [FUNCTION_CALL_CONTENT]  # Has only functionCall, no text
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        # The index should still be preserved
        assert preserved_indices == {0}
        # But no message is created since there's no text
        assert messages == []

    def test_system_instruction_becomes_system_message(self, proxy):
        """System instruction is converted to system message."""
        contents = [TEXT_ONLY_CONTENT]
        system_instruction = {"parts": [{"text": "Be concise."}]}

        messages, _ = proxy._gemini_contents_to_messages(contents, system_instruction)

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise."
        assert messages[1]["role"] == "user"


# =============================================================================
# Tests for realistic conversation flows
# =============================================================================


class TestRealisticConversationFlows:
    """Test preservation with realistic conversation patterns."""

    def test_image_analysis_conversation(self, proxy):
        """Realistic image analysis conversation preserves image content."""
        contents = [
            # User sends an image for analysis
            {
                "role": "user",
                "parts": [
                    {"text": "What objects can you see in this photo?"},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": "base64encodedphoto...",
                        }
                    },
                ],
            },
            # Model responds with analysis
            {
                "role": "model",
                "parts": [
                    {
                        "text": "I can see a cat sitting on a windowsill. "
                        "The window overlooks a garden with flowers."
                    }
                ],
            },
            # User asks follow-up
            {
                "role": "user",
                "parts": [{"text": "What color is the cat?"}],
            },
        ]

        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        # Only the first content (with image) should be preserved
        assert preserved_indices == {0}
        assert len(messages) == 3

    def test_function_calling_conversation(self, proxy):
        """Realistic function calling conversation preserves function content."""
        contents = [
            # User asks about weather
            {"role": "user", "parts": [{"text": "What's the weather in Paris?"}]},
            # Model calls weather function
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
            },
            # User provides function response
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"temp_c": 18, "condition": "partly cloudy"},
                        }
                    }
                ],
            },
            # Model provides final answer
            {
                "role": "model",
                "parts": [{"text": "The weather in Paris is 18C and partly cloudy."}],
            },
            # User asks another question
            {"role": "user", "parts": [{"text": "Should I bring an umbrella?"}]},
        ]

        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        # Function call (index 1) and function response (index 2) should be preserved
        assert preserved_indices == {1, 2}

    def test_multi_modal_document_analysis(self, proxy):
        """Multi-modal document analysis with images and files."""
        contents = [
            # User provides document
            {
                "role": "user",
                "parts": [
                    {"text": "Please review this contract"},
                    {
                        "fileData": {
                            "mimeType": "application/pdf",
                            "fileUri": "gs://contracts/agreement.pdf",
                        }
                    },
                ],
            },
            # Model asks for clarification
            {
                "role": "model",
                "parts": [
                    {
                        "text": "I've reviewed the contract. Do you want me to highlight specific sections?"
                    }
                ],
            },
            # User provides screenshot of specific section
            {
                "role": "user",
                "parts": [
                    {"text": "Yes, please explain this clause:"},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": "screenshotbase64...",
                        }
                    },
                ],
            },
            # Model explains
            {
                "role": "model",
                "parts": [{"text": "This clause specifies the termination conditions..."}],
            },
        ]

        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        # First content (PDF) and third content (screenshot) should be preserved
        assert preserved_indices == {0, 2}
        assert len(messages) == 4

    def test_conversation_with_no_preservation_needed(self, proxy):
        """Pure text conversation needs no preservation."""
        contents = [
            {"role": "user", "parts": [{"text": "What is machine learning?"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "text": "Machine learning is a subset of AI that enables "
                        "computers to learn from data."
                    }
                ],
            },
            {"role": "user", "parts": [{"text": "Can you give an example?"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "Sure! Email spam filters use machine learning to classify messages."}
                ],
            },
            {"role": "user", "parts": [{"text": "Thanks!"}]},
        ]

        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)

        assert preserved_indices == set()
        assert len(messages) == 5


# =============================================================================
# Parametrized tests for comprehensive coverage
# =============================================================================


class TestParametrizedNonTextDetection:
    """Parametrized tests for non-text part detection."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            (TEXT_ONLY_CONTENT, False),
            (MODEL_TEXT_CONTENT, False),
            (EMPTY_PARTS_CONTENT, False),
            (IMAGE_INLINE_CONTENT, True),
            (IMAGE_ONLY_CONTENT, True),
            (FILE_DATA_CONTENT, True),
            (FUNCTION_CALL_CONTENT, True),
            (FUNCTION_CALL_WITH_TEXT_CONTENT, True),
            (FUNCTION_RESPONSE_CONTENT, True),
            (MULTI_IMAGE_CONTENT, True),
            (MIXED_MEDIA_CONTENT, True),
        ],
        ids=[
            "text_only",
            "model_text",
            "empty_parts",
            "image_inline",
            "image_only",
            "file_data",
            "function_call",
            "function_call_with_text",
            "function_response",
            "multi_image",
            "mixed_media",
        ],
    )
    def test_non_text_detection(self, proxy, content, expected):
        """Parametrized test for _has_non_text_parts."""
        assert proxy._has_non_text_parts(content) is expected


class TestParametrizedPreservation:
    """Parametrized tests for index preservation."""

    @pytest.mark.parametrize(
        "contents,expected_indices",
        [
            # Single text
            ([TEXT_ONLY_CONTENT], set()),
            # Single image
            ([IMAGE_INLINE_CONTENT], {0}),
            # Text then image
            ([TEXT_ONLY_CONTENT, IMAGE_INLINE_CONTENT], {1}),
            # Image then text
            ([IMAGE_INLINE_CONTENT, TEXT_ONLY_CONTENT], {0}),
            # All images
            ([IMAGE_INLINE_CONTENT, FILE_DATA_CONTENT], {0, 1}),
            # Mixed throughout
            (
                [TEXT_ONLY_CONTENT, IMAGE_INLINE_CONTENT, MODEL_TEXT_CONTENT, FILE_DATA_CONTENT],
                {1, 3},
            ),
            # Function call sequence
            (
                [TEXT_ONLY_CONTENT, FUNCTION_CALL_CONTENT, FUNCTION_RESPONSE_CONTENT],
                {1, 2},
            ),
        ],
        ids=[
            "single_text",
            "single_image",
            "text_then_image",
            "image_then_text",
            "all_images",
            "mixed_throughout",
            "function_call_sequence",
        ],
    )
    def test_preserved_indices(self, proxy, contents, expected_indices):
        """Parametrized test for preserved indices."""
        _, preserved_indices = proxy._gemini_contents_to_messages(contents)
        assert preserved_indices == expected_indices


# =============================================================================
# Tests for _rebuild_gemini_contents
# =============================================================================


class TestRebuildGeminiContents:
    """_rebuild_gemini_contents must re-insert preserved entries at their original positions."""

    def _round_trip(self, proxy, contents):
        """Simulate the full compression round-trip for a given contents list.

        Mimics what the handler does: convert → strip system msg → convert back → rebuild.
        """
        messages, preserved_indices = proxy._gemini_contents_to_messages(contents)
        preserved_contents = {idx: contents[idx] for idx in preserved_indices}
        optimized_contents, _ = proxy._messages_to_gemini_contents(messages)
        return proxy._rebuild_gemini_contents(
            contents, preserved_indices, preserved_contents, optimized_contents
        )

    def test_text_only_unchanged(self, proxy):
        """Text-only round-trip should produce identical contents."""
        contents = [TEXT_ONLY_CONTENT, MODEL_TEXT_CONTENT]
        result = self._round_trip(proxy, contents)
        assert len(result) == 2
        assert result[0]["parts"][0]["text"] == "Hello, world!"
        assert result[1]["parts"][0]["text"] == "Hello! How can I help you today?"

    def test_function_call_sequence_preserved(self, proxy):
        """functionCall and functionResponse entries must survive and appear at correct positions."""
        contents = [
            TEXT_ONLY_CONTENT,  # idx 0: text
            FUNCTION_CALL_CONTENT,  # idx 1: functionCall only — no text → preserved
            FUNCTION_RESPONSE_CONTENT,  # idx 2: functionResponse only — no text → preserved
            MODEL_TEXT_CONTENT,  # idx 3: text
        ]
        result = self._round_trip(proxy, contents)

        assert len(result) == 4, f"Expected 4 entries, got {len(result)}: {result}"
        # Position 0: original text
        assert result[0]["parts"][0].get("text") == "Hello, world!"
        # Position 1: functionCall preserved exactly
        assert "functionCall" in result[1]["parts"][0], "functionCall missing at position 1"
        assert result[1]["parts"][0]["functionCall"]["name"] == "get_weather"
        # Position 2: functionResponse preserved exactly
        assert "functionResponse" in result[2]["parts"][0], "functionResponse missing at position 2"
        # Position 3: text preserved
        assert result[3]["parts"][0].get("text") == "Hello! How can I help you today?"

    def test_function_call_at_start(self, proxy):
        """Preserved entry at idx=0 must not overwrite idx=0 of optimized_contents."""
        contents = [
            FUNCTION_CALL_CONTENT,  # idx 0: no text → preserved
            TEXT_ONLY_CONTENT,  # idx 1: text
        ]
        result = self._round_trip(proxy, contents)

        assert len(result) == 2
        assert "functionCall" in result[0]["parts"][0]
        assert result[1]["parts"][0].get("text") == "Hello, world!"

    def test_hybrid_entry_uses_original(self, proxy):
        """Entry with both text and functionCall keeps the original (with functionCall intact)."""
        contents = [
            TEXT_ONLY_CONTENT,
            FUNCTION_CALL_WITH_TEXT_CONTENT,  # idx 1: has both text and functionCall → preserved
            MODEL_TEXT_CONTENT,
        ]
        result = self._round_trip(proxy, contents)

        assert len(result) == 3
        # Hybrid entry must come back as the original (functionCall retained)
        hybrid = result[1]
        part_keys = {k for p in hybrid["parts"] for k in p}
        assert "functionCall" in part_keys, "functionCall lost from hybrid entry"
