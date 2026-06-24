"""Comprehensive tests for the image compression feature.

Tests ImageCompressor class and TrainedRouter for:
- Image detection in various provider formats
- Query extraction
- Compression routing
- Provider-specific compression
- Edge cases
- Token estimation
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import base64
import builtins
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Import from PIL for creating test images
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

torch = pytest.importorskip("torch")

from headroom.image.compressor import (  # noqa: E402
    CompressionResult,
    ImageCompressor,
    Technique,
    compress_images,
    get_compressor,
)
from headroom.image.trained_router import (  # noqa: E402
    ImageSignals,
    RouteDecision,
    TrainedRouter,
)
from headroom.image.trained_router import (  # noqa: E402
    Technique as RouterTechnique,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_test_image_bytes():
    """Create a small test image as bytes."""
    if not HAS_PIL:
        pytest.skip("PIL not available")

    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def large_test_image_bytes():
    """Create a larger test image as bytes (1024x1024)."""
    if not HAS_PIL:
        pytest.skip("PIL not available")

    # Create a 1024x1024 image with some pattern
    img = Image.new("RGB", (1024, 1024), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def small_image_base64(small_test_image_bytes):
    """Base64 encoded small test image."""
    return base64.b64encode(small_test_image_bytes).decode("utf-8")


@pytest.fixture
def large_image_base64(large_test_image_bytes):
    """Base64 encoded large test image."""
    return base64.b64encode(large_test_image_bytes).decode("utf-8")


@pytest.fixture
def openai_messages_with_image(small_image_base64):
    """Sample OpenAI format messages with image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{small_image_base64}",
                        "detail": "auto",
                    },
                },
            ],
        }
    ]


@pytest.fixture
def anthropic_messages_with_image(small_image_base64):
    """Sample Anthropic format messages with image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": small_image_base64,
                    },
                },
            ],
        }
    ]


@pytest.fixture
def google_messages_with_image(small_image_base64):
    """Sample Google format messages with image."""
    return [
        {
            "role": "user",
            "content": [
                {"text": "What do you see?"},
                {"inlineData": {"mimeType": "image/png", "data": small_image_base64}},
            ],
        }
    ]


@pytest.fixture
def text_only_messages():
    """Messages without any images."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture
def compressor():
    """Get an ImageCompressor instance."""
    return ImageCompressor()


@pytest.fixture
def mock_route_decision_full_low():
    """Mock RouteDecision for FULL_LOW."""
    return RouteDecision(
        technique=RouterTechnique.FULL_LOW,
        confidence=0.9,
        reason="General query about image contents",
        image_signals=None,
        query_prediction="full_low",
        query_confidence=0.9,
    )


@pytest.fixture
def mock_route_decision_preserve():
    """Mock RouteDecision for PRESERVE."""
    return RouteDecision(
        technique=RouterTechnique.PRESERVE,
        confidence=0.95,
        reason="Query requires fine detail analysis",
        image_signals=None,
        query_prediction="preserve",
        query_confidence=0.95,
    )


@pytest.fixture
def mock_route_decision_transcode():
    """Mock RouteDecision for TRANSCODE."""
    return RouteDecision(
        technique=RouterTechnique.TRANSCODE,
        confidence=0.88,
        reason="Query asks to read text from image",
        image_signals=None,
        query_prediction="transcode",
        query_confidence=0.88,
    )


@pytest.fixture
def mock_route_decision_crop():
    """Mock RouteDecision for CROP."""
    return RouteDecision(
        technique=RouterTechnique.CROP,
        confidence=0.85,
        reason="Query asks about specific region",
        image_signals=None,
        query_prediction="crop",
        query_confidence=0.85,
    )


def create_mock_router(route_decision):
    """Create a mock router that returns the given decision."""
    mock_router = MagicMock()
    mock_router.classify.return_value = route_decision
    return mock_router


# ============================================================================
# Test ImageCompressor class - Image detection
# ============================================================================


class TestImageDetection:
    """Tests for image detection in various formats."""

    def test_has_images_openai_format(self, compressor, openai_messages_with_image):
        """Detect images in OpenAI format."""
        assert compressor.has_images(openai_messages_with_image) is True

    def test_has_images_anthropic_format(self, compressor, anthropic_messages_with_image):
        """Detect images in Anthropic format."""
        assert compressor.has_images(anthropic_messages_with_image) is True

    def test_has_images_google_format(self, compressor, google_messages_with_image):
        """Detect images in Google format."""
        assert compressor.has_images(google_messages_with_image) is True

    def test_has_images_no_images(self, compressor, text_only_messages):
        """Returns False when no images in messages."""
        assert compressor.has_images(text_only_messages) is False

    def test_has_images_empty_messages(self, compressor):
        """Handles empty message list."""
        assert compressor.has_images([]) is False

    def test_has_images_string_content(self, compressor):
        """Handles messages with plain string content."""
        messages = [{"role": "user", "content": "Just text, no images"}]
        assert compressor.has_images(messages) is False

    def test_has_images_mixed_content(self, compressor, small_image_base64):
        """Detect images in messages with mixed content."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Now look at this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{small_image_base64}"},
                    },
                ],
            },
        ]
        assert compressor.has_images(messages) is True


# ============================================================================
# Test ImageCompressor class - Query extraction
# ============================================================================


class TestQueryExtraction:
    """Tests for extracting text query from messages."""

    def test_extract_query_from_openai_format(self, compressor, openai_messages_with_image):
        """Extracts text query from OpenAI format messages."""
        query = compressor._extract_query(openai_messages_with_image)
        assert query == "What is in this image?"

    def test_extract_query_from_anthropic_format(self, compressor, anthropic_messages_with_image):
        """Extracts text query from Anthropic format messages."""
        query = compressor._extract_query(anthropic_messages_with_image)
        assert query == "Describe this image"

    def test_extract_query_empty_string_when_no_text(self, compressor, small_image_base64):
        """Returns empty string when no text in user message."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{small_image_base64}"},
                    }
                ],
            }
        ]
        query = compressor._extract_query(messages)
        assert query == ""

    def test_extract_query_from_plain_text_message(self, compressor):
        """Extracts query from plain text user message."""
        messages = [{"role": "user", "content": "What is this?"}]
        query = compressor._extract_query(messages)
        assert query == "What is this?"

    def test_extract_query_uses_last_user_message(self, compressor):
        """Extracts query from the most recent user message."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        query = compressor._extract_query(messages)
        assert query == "Second question"


# ============================================================================
# Test ImageCompressor class - Image data extraction
# ============================================================================


class TestImageDataExtraction:
    """Tests for extracting base64 image data from messages."""

    def test_extract_image_data_openai_format(
        self, compressor, openai_messages_with_image, small_test_image_bytes
    ):
        """Extracts base64 image data from OpenAI format."""
        data = compressor._extract_image_data(openai_messages_with_image)
        assert data is not None
        assert isinstance(data, bytes)
        # Verify it's valid image data
        assert data == small_test_image_bytes

    def test_extract_image_data_anthropic_format(
        self, compressor, anthropic_messages_with_image, small_test_image_bytes
    ):
        """Extracts base64 image data from Anthropic format."""
        data = compressor._extract_image_data(anthropic_messages_with_image)
        assert data is not None
        assert isinstance(data, bytes)
        assert data == small_test_image_bytes

    def test_extract_image_data_google_format(
        self, compressor, google_messages_with_image, small_test_image_bytes
    ):
        """Extracts base64 image data from Google format."""
        data = compressor._extract_image_data(google_messages_with_image)
        assert data is not None
        assert isinstance(data, bytes)
        assert data == small_test_image_bytes

    def test_extract_image_data_returns_none_for_text_only(self, compressor, text_only_messages):
        """Returns None when no images in messages."""
        data = compressor._extract_image_data(text_only_messages)
        assert data is None

    def test_extract_image_data_returns_first_image(self, compressor, small_image_base64):
        """Extracts the first image when multiple images present."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{small_image_base64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,SECOND_IMAGE_DATA"},
                    },
                ],
            }
        ]
        data = compressor._extract_image_data(messages)
        assert data is not None


# ============================================================================
# Test Compression routing
# ============================================================================


class TestCompressionRouting:
    """Tests for compression technique routing based on query."""

    def test_compress_general_query(
        self, compressor, openai_messages_with_image, mock_route_decision_full_low
    ):
        """'What is this?' query routes to full_low technique."""
        mock_router = create_mock_router(mock_route_decision_full_low)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            result = compressor.compress(openai_messages_with_image, "openai")

            # Verify the router was called
            mock_router.classify.assert_called_once()

            # For FULL_LOW, OpenAI should get detail="low"
            content = result[0]["content"]
            for item in content:
                if item.get("type") == "image_url":
                    assert item["image_url"].get("detail") == "low"

    def test_compress_detail_query(
        self, compressor, small_image_base64, mock_route_decision_preserve
    ):
        """'Count the whiskers' query routes to preserve technique."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Count the whiskers on the cat"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{small_image_base64}"},
                    },
                ],
            }
        ]
        mock_router = create_mock_router(mock_route_decision_preserve)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            compressor.compress(messages, "openai")
            mock_router.classify.assert_called_once()

    def test_compress_text_query(
        self, compressor, small_image_base64, mock_route_decision_transcode
    ):
        """'Read the text' query routes to transcode technique."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read the text in this document"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{small_image_base64}"},
                    },
                ],
            }
        ]
        mock_router = create_mock_router(mock_route_decision_transcode)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            compressor.compress(messages, "openai")
            mock_router.classify.assert_called_once()

    def test_compress_region_query(self, compressor, small_image_base64, mock_route_decision_crop):
        """'What's in the corner?' query routes to crop technique."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in the top-left corner?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{small_image_base64}"},
                    },
                ],
            }
        ]
        mock_router = create_mock_router(mock_route_decision_crop)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            compressor.compress(messages, "openai")
            mock_router.classify.assert_called_once()


# ============================================================================
# Test Provider-specific compression
# ============================================================================


class TestProviderSpecificCompression:
    """Tests for provider-specific image compression."""

    def test_openai_detail_low(
        self, compressor, openai_messages_with_image, mock_route_decision_full_low
    ):
        """OpenAI: sets detail='low' for full_low technique."""
        mock_router = create_mock_router(mock_route_decision_full_low)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            result = compressor.compress(openai_messages_with_image, "openai")

            # Find the image item and check detail
            for item in result[0]["content"]:
                if item.get("type") == "image_url":
                    assert item["image_url"]["detail"] == "low"

    def test_openai_detail_preserved(
        self, compressor, small_image_base64, mock_route_decision_preserve
    ):
        """OpenAI: preserves original detail setting for preserve technique."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze fine details"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{small_image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
        mock_router = create_mock_router(mock_route_decision_preserve)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            result = compressor.compress(messages, "openai")

            # For preserve, the image should remain unchanged
            for item in result[0]["content"]:
                if item.get("type") == "image_url":
                    # Should keep original high detail
                    detail = item["image_url"].get("detail")
                    assert detail == "high"

    def test_anthropic_format(
        self, compressor, anthropic_messages_with_image, mock_route_decision_full_low
    ):
        """Handles Anthropic image format correctly."""
        mock_router = create_mock_router(mock_route_decision_full_low)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            result = compressor.compress(anthropic_messages_with_image, "anthropic")

            # Should return valid messages (may or may not transform Anthropic format)
            assert isinstance(result, list)
            assert len(result) > 0


# ============================================================================
# Test Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_images_passthrough(self, compressor, text_only_messages):
        """Returns messages unchanged if no images present."""
        result = compressor.compress(text_only_messages, "openai")
        assert result == text_only_messages

    def test_empty_messages(self, compressor):
        """Handles empty message list gracefully."""
        result = compressor.compress([], "openai")
        assert result == []

    def test_router_failure_fallback(self, compressor, openai_messages_with_image):
        """Falls back to preserve technique on router error."""
        mock_router = MagicMock()
        mock_router.classify.side_effect = Exception("Router failed")

        with patch.object(compressor, "_get_router", return_value=mock_router):
            # Should not raise, should fall back gracefully
            result = compressor.compress(openai_messages_with_image, "openai")

            # Messages should be returned (either original or with preserve)
            assert isinstance(result, list)
            assert len(result) > 0

    def test_invalid_base64_data(self, compressor, mock_route_decision_preserve):
        """Handles invalid base64 data gracefully."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,bm90X3ZhbGlkX2ltYWdlX2RhdGE="},
                    },
                ],
            }
        ]

        # Use a mock router to avoid actual model loading
        mock_router = create_mock_router(mock_route_decision_preserve)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            # Should not raise
            result = compressor.compress(messages, "openai")
            assert isinstance(result, list)

    def test_url_image_not_base64(self, compressor):
        """Handles URL-based images (not base64)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        # URL images should just pass through since we can't extract data
        result = compressor.compress(messages, "openai")
        assert isinstance(result, list)
        # Should return original messages since no base64 data to extract
        assert result == messages

    def test_none_content(self, compressor):
        """Handles messages with None content."""
        messages = [{"role": "user", "content": None}]

        result = compressor.compress(messages, "openai")
        assert result == messages

    def test_missing_content_key(self, compressor):
        """Handles messages missing content key."""
        messages = [{"role": "user"}]

        result = compressor.compress(messages, "openai")
        assert result == messages


# ============================================================================
# Test Token estimation
# ============================================================================


class TestTokenEstimation:
    """Tests for image token estimation."""

    def test_estimate_tokens_small_image(self, compressor, small_test_image_bytes):
        """Estimates tokens for a small image correctly."""
        # Pass actual image bytes, not base64
        # 100x100 image with low detail = 85 tokens
        tokens = compressor._estimate_tokens(small_test_image_bytes, "low")
        assert tokens == 85

    def test_estimate_tokens_large_image(self, compressor, large_test_image_bytes):
        """Estimates tokens for a large image correctly."""
        # 1024x1024 image with high detail
        # tiles_x = ceil(1024/512) = 2
        # tiles_y = ceil(1024/512) = 2
        # tokens = 85 * 2 * 2 + 170 = 510
        tokens = compressor._estimate_tokens(large_test_image_bytes, "high")
        assert tokens == 510

    def test_estimate_tokens_low_detail_constant(self, compressor, large_test_image_bytes):
        """Low detail always returns 85 tokens regardless of size."""
        tokens = compressor._estimate_tokens(large_test_image_bytes, "low")
        assert tokens == 85

    def test_savings_calculation(self):
        """CompressionResult calculates savings percentage correctly."""
        result = CompressionResult(
            technique=Technique.FULL_LOW, original_tokens=1000, compressed_tokens=85, confidence=0.9
        )

        # (1000 - 85) / 1000 * 100 = 91.5%
        assert result.savings_percent == pytest.approx(91.5, rel=0.01)

    def test_savings_zero_original_tokens(self):
        """Handles zero original tokens without division error."""
        result = CompressionResult(
            technique=Technique.PRESERVE, original_tokens=0, compressed_tokens=0, confidence=1.0
        )

        assert result.savings_percent == 0.0

    def test_estimate_tokens_invalid_data(self, compressor):
        """Returns default token count for invalid image data."""
        # Pass invalid bytes that can't be opened as image
        tokens = compressor._estimate_tokens(b"invalid_image_data", "high")
        # Should return a default value (765 based on the code)
        assert tokens == 765


# ============================================================================
# Test TrainedRouter (mocked)
# ============================================================================


class TestTrainedRouterMocked:
    """Tests for TrainedRouter with mocked model loading."""

    def test_router_technique_enum_values(self):
        """Verify Technique enum has expected values."""
        assert RouterTechnique.FULL_LOW.value == "full_low"
        assert RouterTechnique.PRESERVE.value == "preserve"
        assert RouterTechnique.TRANSCODE.value == "transcode"
        assert RouterTechnique.CROP.value == "crop"

    def test_route_decision_dataclass(self):
        """Verify RouteDecision dataclass structure."""
        decision = RouteDecision(
            technique=RouterTechnique.FULL_LOW,
            confidence=0.9,
            reason="Test reason",
            image_signals=None,
            query_prediction="full_low",
            query_confidence=0.9,
        )

        assert decision.technique == RouterTechnique.FULL_LOW
        assert decision.confidence == 0.9
        assert decision.reason == "Test reason"

    def test_image_signals_dataclass(self):
        """Verify ImageSignals dataclass structure."""
        signals = ImageSignals(has_text=0.8, is_document=0.6, is_complex=0.3, has_small_details=0.2)

        assert signals.has_text == 0.8
        assert signals.is_document == 0.6
        assert signals.is_complex == 0.3
        assert signals.has_small_details == 0.2

    def test_router_is_available_with_models(self):
        """Router reports available when models can load."""
        router = TrainedRouter()

        # Mock _load_models to not actually load (models loaded via MLModelRegistry now)
        with patch.object(router, "_load_models"):
            assert router.is_available() is True

    def test_router_is_available_false_on_error(self):
        """Router reports not available when models fail to load."""
        router = TrainedRouter(model_path="/nonexistent/path")

        # This should return False since the model path doesn't exist
        # and loading will fail
        with patch.object(router, "_load_models", side_effect=Exception("Model not found")):
            assert router.is_available() is False

    def test_release_models_unloads_registry_entries(self):
        """Releasing router state should drop shared registry keys too."""
        router = TrainedRouter()
        router._classifier_key = "technique_router:demo"
        router._siglip_key = "siglip:demo"
        router._classifier = object()
        router._tokenizer = object()
        router._siglip_model = object()
        router._siglip_processor = object()
        router._text_embeddings = object()

        with patch("headroom.models.ml_models.MLModelRegistry.unload_many") as unload_many:
            router.release_models()

        unload_many.assert_called_once_with(["technique_router:demo", "siglip:demo"])
        assert router._classifier is None
        assert router._tokenizer is None
        assert router._siglip_model is None
        assert router._siglip_processor is None
        assert router._text_embeddings is None


# ============================================================================
# Test Convenience functions
# ============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_compressor_returns_instance(self):
        """get_compressor returns an ImageCompressor instance."""
        compressor = get_compressor()
        assert isinstance(compressor, ImageCompressor)

    def test_get_compressor_returns_fresh_instance(self):
        """get_compressor returns a fresh caller-owned instance."""
        compressor1 = get_compressor()
        compressor2 = get_compressor()
        assert compressor1 is not compressor2

    def test_compress_images_function(self, text_only_messages):
        """compress_images convenience function works."""
        result = compress_images(text_only_messages, "openai")
        assert result == text_only_messages

    def test_compress_images_function_closes_temporary_compressor(self, text_only_messages):
        """compress_images should always close the one-shot compressor."""
        with (
            patch.object(ImageCompressor, "compress", return_value=text_only_messages),
            patch.object(ImageCompressor, "close") as close,
        ):
            result = compress_images(text_only_messages, "openai")

        assert result == text_only_messages
        close.assert_called_once()


# ============================================================================
# Integration tests (with mocked router)
# ============================================================================


class TestIntegration:
    """Integration tests with mocked router."""

    def test_full_compression_flow_openai(self, small_image_base64, mock_route_decision_full_low):
        """Test complete compression flow for OpenAI format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{small_image_base64}",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

        compressor = ImageCompressor()
        mock_router = create_mock_router(mock_route_decision_full_low)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            result = compressor.compress(messages, "openai")

            # Verify structure
            assert len(result) == 1
            assert result[0]["role"] == "user"
            assert isinstance(result[0]["content"], list)

            # Verify image was processed
            has_image = False
            for item in result[0]["content"]:
                if item.get("type") == "image_url":
                    has_image = True
                    assert item["image_url"]["detail"] == "low"
            assert has_image

    def test_full_compression_flow_anthropic(
        self, small_image_base64, mock_route_decision_full_low
    ):
        """Test complete compression flow for Anthropic format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": small_image_base64,
                        },
                    },
                ],
            }
        ]

        compressor = ImageCompressor()
        mock_router = create_mock_router(mock_route_decision_full_low)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            result = compressor.compress(messages, "anthropic")

            # Should return valid messages
            assert len(result) == 1
            assert result[0]["role"] == "user"

    def test_multiple_images_in_message(self, small_image_base64, mock_route_decision_full_low):
        """Test compression with multiple images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{small_image_base64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{small_image_base64}"},
                    },
                ],
            }
        ]

        compressor = ImageCompressor()
        mock_router = create_mock_router(mock_route_decision_full_low)

        with patch.object(compressor, "_get_router", return_value=mock_router):
            result = compressor.compress(messages, "openai")

            # Both images should be processed
            image_count = 0
            for item in result[0]["content"]:
                if item.get("type") == "image_url":
                    image_count += 1
                    assert item["image_url"]["detail"] == "low"
            assert image_count == 2


# ============================================================================
# ContentRouter Integration Tests
# ============================================================================


class TestContentRouterIntegration:
    """Test ImageCompressor integration with ContentRouter."""

    def test_content_router_loads_image_compressor(self):
        """Verify ContentRouter can load ImageCompressor (not None)."""
        from headroom.transforms.content_router import ContentRouter

        router = ContentRouter()
        compressor = router._get_image_optimizer()

        try:
            # This should NOT be None - if it is, the import failed silently
            assert compressor is not None, (
                "ContentRouter._get_image_optimizer() returned None. "
                "This means ImageCompressor import failed silently!"
            )
        finally:
            if compressor is not None:
                compressor.close()

    def test_content_router_compressor_is_image_compressor(self):
        """Verify ContentRouter uses ImageCompressor (not old ImageOptimizer)."""
        from headroom.image import ImageCompressor
        from headroom.transforms.content_router import ContentRouter

        router = ContentRouter()
        compressor = router._get_image_optimizer()

        try:
            assert isinstance(compressor, ImageCompressor), (
                f"Expected ImageCompressor, got {type(compressor).__name__}"
            )
        finally:
            if compressor is not None:
                compressor.close()

    def test_content_router_compressor_is_fresh_per_call(self):
        """ContentRouter should not share image compressors across callers."""
        from headroom.transforms.content_router import ContentRouter

        router = ContentRouter()
        first = router._get_image_optimizer()
        second = router._get_image_optimizer()

        try:
            assert first is not second
        finally:
            if first is not None:
                first.close()
            if second is not None:
                second.close()

    def test_content_router_optimize_images_works(self):
        """Test optimize_images_in_messages returns valid result."""
        from unittest.mock import MagicMock

        from headroom.transforms.content_router import ContentRouter

        router = ContentRouter()
        tokenizer = MagicMock()

        # Simple message without images
        messages = [{"role": "user", "content": "Hello"}]
        result, metrics = router.optimize_images_in_messages(messages, tokenizer, provider="openai")

        assert result == messages
        assert "images_optimized" in metrics
        assert metrics["tokens_saved"] == 0

    def test_content_router_returns_metrics_and_closes_after_compression(self):
        """Image optimization should report savings and close the compressor."""
        from headroom.transforms.content_router import ContentRouter

        router = ContentRouter()
        tokenizer = MagicMock()
        optimized = [{"role": "user", "content": "optimized"}]
        technique = SimpleNamespace(value="full_low")
        fake = MagicMock()
        fake.has_images.return_value = True
        fake.compress.return_value = optimized
        fake.last_result = SimpleNamespace(
            original_tokens=1000,
            compressed_tokens=85,
            technique=technique,
            confidence=0.9,
        )

        with patch.object(router, "_get_image_optimizer", return_value=fake):
            result, metrics = router.optimize_images_in_messages(
                [{"role": "user", "content": "with image"}],
                tokenizer,
                provider="openai",
            )

        assert result == optimized
        assert metrics == {
            "images_optimized": True,
            "tokens_before": 1000,
            "tokens_after": 85,
            "tokens_saved": 915,
            "technique": "full_low",
            "confidence": 0.9,
        }
        fake.close.assert_called_once()

    def test_content_router_returns_basic_metrics_when_compression_has_no_result(self):
        """Missing compressor result should still close and return neutral metrics."""
        from headroom.transforms.content_router import ContentRouter

        router = ContentRouter()
        tokenizer = MagicMock()
        optimized = [{"role": "user", "content": "optimized"}]
        fake = MagicMock()
        fake.has_images.return_value = True
        fake.compress.return_value = optimized
        fake.last_result = None

        with patch.object(router, "_get_image_optimizer", return_value=fake):
            result, metrics = router.optimize_images_in_messages(
                [{"role": "user", "content": "with image"}],
                tokenizer,
                provider="openai",
            )

        assert result == optimized
        assert metrics == {"images_optimized": 0, "tokens_saved": 0}
        fake.close.assert_called_once()

    def test_content_router_image_optimizer_returns_none_when_image_stack_missing(self):
        """Import failures should disable image optimization without raising."""
        from headroom.transforms.content_router import ContentRouter

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
            if name == "image" and fromlist == ("ImageCompressor",) and level == 2:
                raise ImportError("image extras unavailable")
            return real_import(name, globals, locals, fromlist, level)

        router = ContentRouter()
        with patch.object(builtins, "__import__", side_effect=fake_import):
            compressor = router._get_image_optimizer()

        assert compressor is None

    def test_content_router_releases_image_optimizer_after_use(self):
        """ContentRouter should drop the compressor after each optimization pass."""
        from headroom.transforms.content_router import ContentRouter

        router = ContentRouter()
        tokenizer = MagicMock()
        fake = MagicMock()
        fake.has_images.return_value = False

        with patch.object(router, "_get_image_optimizer", return_value=fake):
            result, metrics = router.optimize_images_in_messages(
                [{"role": "user", "content": "Hello"}],
                tokenizer,
                provider="openai",
            )

        assert result == [{"role": "user", "content": "Hello"}]
        assert metrics["tokens_saved"] == 0
        fake.close.assert_called_once()
