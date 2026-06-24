"""Tests for UniversalCompressor."""

import json

import pytest

from headroom.compression.detector import ContentType
from headroom.compression.handlers.base import NoOpHandler
from headroom.compression.universal import (
    CompressionResult,
    UniversalCompressor,
    UniversalCompressorConfig,
    compress,
)


class TestUniversalCompressorConfig:
    """Tests for UniversalCompressorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UniversalCompressorConfig()

        assert config.use_magika is True
        assert config.use_kompress is True
        assert config.use_entropy_preservation is True
        assert config.entropy_threshold == 0.85
        assert config.min_content_length == 100
        assert config.compression_ratio_target == 0.3

    def test_custom_config(self):
        """Test custom configuration."""
        config = UniversalCompressorConfig(
            use_magika=False,
            compression_ratio_target=0.5,
        )

        assert config.use_magika is False
        assert config.compression_ratio_target == 0.5


class TestCompressionResult:
    """Tests for CompressionResult."""

    def test_tokens_saved(self):
        """Test tokens_saved calculation."""
        result = CompressionResult(
            compressed="short",
            original="much longer original content",
            compression_ratio=0.5,
            tokens_before=100,
            tokens_after=50,
            content_type=ContentType.TEXT,
            detection_confidence=0.9,
            handler_used="test",
            preservation_ratio=0.5,
        )

        assert result.tokens_saved == 50

    def test_savings_percentage(self):
        """Test savings_percentage calculation."""
        result = CompressionResult(
            compressed="short",
            original="longer",
            compression_ratio=0.5,
            tokens_before=100,
            tokens_after=25,
            content_type=ContentType.TEXT,
            detection_confidence=0.9,
            handler_used="test",
            preservation_ratio=0.5,
        )

        assert result.savings_percentage == 75.0

    def test_zero_tokens_before(self):
        """Test handling of zero tokens_before."""
        result = CompressionResult(
            compressed="",
            original="",
            compression_ratio=1.0,
            tokens_before=0,
            tokens_after=0,
            content_type=ContentType.UNKNOWN,
            detection_confidence=0.0,
            handler_used="none",
            preservation_ratio=1.0,
        )

        assert result.savings_percentage == 0.0


class TestUniversalCompressor:
    """Tests for UniversalCompressor."""

    @pytest.fixture
    def compressor(self):
        """Create compressor with fallback detector (no Magika required)."""
        config = UniversalCompressorConfig(
            use_magika=False,  # Use fallback detector
            use_kompress=False,  # Use simple compression
            ccr_enabled=False,  # Skip CCR
        )
        return UniversalCompressor(config=config)

    def test_compress_short_content_unchanged(self, compressor):
        """Test that short content is not compressed."""
        content = "short"
        result = compressor.compress(content)

        assert result.compressed == content
        assert result.compression_ratio == 1.0
        assert "skipped" in result.metadata

    def test_compress_empty_content(self, compressor):
        """Test handling of empty content."""
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.content_type == ContentType.UNKNOWN

    def test_compress_json_content(self, compressor):
        """Test compression of JSON content."""
        content = json.dumps(
            {"users": [{"id": i, "name": f"User {i}", "bio": "x" * 100} for i in range(10)]}
        )

        result = compressor.compress(content)

        assert result.content_type == ContentType.JSON
        assert result.handler_used == "json"
        # Compression should reduce size
        assert len(result.compressed) < len(content)

    def test_compress_code_content(self, compressor):
        """Test compression of code content."""
        content = (
            '''
def hello_world():
    """Say hello to the world."""
    message = "Hello, World!"
    print(message)
    return message

def another_function():
    """Another function with a long body."""
    x = 1
    y = 2
    z = x + y
    '''
            + "result = z * " * 50
            + """
    return result
"""
        )
        result = compressor.compress(content)

        assert result.content_type == ContentType.CODE
        assert result.handler_used == "code"

    def test_compress_plain_text(self, compressor):
        """Test compression of plain text."""
        content = "This is plain text without any special structure. " * 20

        result = compressor.compress(content)

        assert result.content_type == ContentType.TEXT

    def test_compress_with_override_type(self, compressor):
        """Test compression with overridden content type."""
        content = '{"key": "value"}' + " " * 100  # Pad to meet min length

        result = compressor.compress(content, content_type=ContentType.TEXT)

        # Should use TEXT even though it looks like JSON
        assert result.content_type == ContentType.TEXT

    def test_compression_result_has_metadata(self, compressor):
        """Test that result includes metadata."""
        content = json.dumps({"items": [{"id": i} for i in range(20)]})

        result = compressor.compress(content)

        assert "detection" in result.metadata
        assert "handler" in result.metadata

    def test_register_custom_handler(self, compressor):
        """Test registering a custom handler."""
        custom_handler = NoOpHandler()
        compressor.register_handler(ContentType.JSON, custom_handler)

        content = '{"key": "value"}' + " " * 100

        result = compressor.compress(content)

        # Should use our custom handler
        assert result.handler_used == "noop"

    def test_get_handler(self, compressor):
        """Test getting handler for content type."""
        json_handler = compressor.get_handler(ContentType.JSON)
        assert json_handler is not None
        assert json_handler.name == "json"

        unknown_handler = compressor.get_handler(ContentType.UNKNOWN)
        assert unknown_handler.name == "noop"


class TestUniversalCompressorBatch:
    """Tests for batch compression."""

    @pytest.fixture
    def compressor(self):
        """Create compressor with fallback detector."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_compress_batch_empty(self, compressor):
        """Test batch compression with empty list."""
        results = compressor.compress_batch([])
        assert results == []

    def test_compress_batch_mixed_content(self, compressor):
        """Test batch compression with mixed content types."""
        contents = [
            json.dumps({"id": 1, "data": "x" * 100}),
            "def foo(): pass\n" * 10,
            "Plain text content " * 10,
        ]

        results = compressor.compress_batch(contents)

        assert len(results) == 3
        assert results[0].content_type == ContentType.JSON
        assert results[1].content_type == ContentType.CODE
        assert results[2].content_type == ContentType.TEXT


class TestCompressFunction:
    """Tests for the convenience compress function."""

    def test_compress_function(self):
        """Test one-off compression function."""
        content = json.dumps({"items": [{"id": i} for i in range(20)]})

        result = compress(content)

        assert isinstance(result, CompressionResult)
        assert result.content_type == ContentType.JSON


class TestStructurePreservation:
    """Integration tests for structure preservation."""

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_json_keys_preserved(self, compressor):
        """Test that JSON keys are visible after compression."""
        data = {
            "user_id": "12345",
            "user_name": "Alice",
            "user_email": "alice@example.com",
            "user_bio": "A very long biography that goes on and on " * 10,
        }
        content = json.dumps(data)

        result = compressor.compress(content)

        # All keys should be visible in compressed output
        for key in data.keys():
            assert key in result.compressed, f"Key {key} should be in compressed output"

    def test_code_signatures_preserved(self, compressor):
        """Test that code signatures are visible after compression."""
        content = (
            '''
def calculate_total(items, tax_rate=0.1):
    """Calculate total with tax."""
    subtotal = sum(item.price for item in items)
    tax = subtotal * tax_rate
    total = subtotal + tax
    '''
            + "# padding " * 50
            + '''
    return total

class ShoppingCart:
    """Shopping cart implementation."""

    def __init__(self):
        self.items = []
        '''
            + "# more padding " * 30
            + '''

    def add_item(self, item):
        """Add item to cart."""
        self.items.append(item)
'''
        )
        result = compressor.compress(content)

        # Function and class names should be visible
        assert "calculate_total" in result.compressed
        assert "ShoppingCart" in result.compressed
        assert "add_item" in result.compressed

    def test_compression_reduces_tokens(self, compressor):
        """Test that compression actually reduces token count."""
        # Large content that should be compressible
        data = {
            "results": [
                {
                    "id": i,
                    "title": f"Result {i}",
                    "description": f"This is a detailed description for result {i}. " * 5,
                }
                for i in range(50)
            ]
        }
        content = json.dumps(data)

        result = compressor.compress(content)

        # Should achieve some compression
        assert result.tokens_after < result.tokens_before
        assert result.compression_ratio < 1.0
