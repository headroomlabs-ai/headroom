"""Tests for image token compression pipeline.

Tests tile-boundary optimization, ONNX technique routing,
and the full compression pipeline across providers.
"""

from __future__ import annotations

import base64
import io

import pytest

# Tile optimizer is pure math — always available
from headroom.image.tile_optimizer import (
    estimate_anthropic_tokens,
    estimate_openai_tokens,
    find_optimal_anthropic_dimensions,
    find_optimal_openai_dimensions,
    optimize_images_in_messages,
)

# Tests that create images need Pillow (optional dependency)
_HAS_PIL = False
try:
    from PIL import Image as _Image  # noqa: F401

    _HAS_PIL = True
except ImportError:
    pass

needs_pillow = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")

# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_openai_low_detail(self):
        assert estimate_openai_tokens(1920, 1080, "low") == 85

    def test_openai_high_detail_single_tile(self):
        assert estimate_openai_tokens(512, 512) == 85 + 170  # 1 tile

    def test_openai_high_detail_multiple_tiles(self):
        # 768x768 → ceil(768/512) * ceil(768/512) = 2*2 = 4 tiles
        tokens = estimate_openai_tokens(768, 768)
        assert tokens == 85 + 170 * 4  # 765

    def test_openai_scales_large_images(self):
        # 4000x3000 → scaled to fit 2048 then shortest to 768
        # Tokens should be finite and reasonable
        tokens = estimate_openai_tokens(4000, 3000)
        assert 200 < tokens < 2000

    def test_anthropic_formula(self):
        # (1024 * 768) / 750 = 1048
        tokens = estimate_anthropic_tokens(1024, 768)
        assert tokens == (1024 * 768) // 750

    def test_anthropic_caps_at_1568(self):
        # 3000x2000 → scaled to 1568 max edge
        tokens = estimate_anthropic_tokens(3000, 2000)
        # After scaling: 1568 * 1045 → tokens = (1568*1045)//750
        assert tokens < 2200  # Capped

    def test_anthropic_caps_at_1_15mp(self):
        # 1568x1568 = 2.46MP > 1.15MP → further scaled
        tokens = estimate_anthropic_tokens(1568, 1568)
        assert tokens <= 1534  # 1.15M / 750


# ---------------------------------------------------------------------------
# Tile optimization tests
# ---------------------------------------------------------------------------


class TestTileOptimization:
    def test_full_hd_saves_tokens(self):
        """1920x1080 → should reduce tile count."""
        opt_w, opt_h = find_optimal_openai_dimensions(1920, 1080)
        before = estimate_openai_tokens(1920, 1080)
        after = estimate_openai_tokens(opt_w, opt_h)
        assert after < before
        assert before - after >= 340  # Significant savings

    def test_already_optimal_no_change(self):
        """512x512 is already on tile boundary."""
        opt_w, opt_h = find_optimal_openai_dimensions(512, 512)
        assert (opt_w, opt_h) == (512, 512)

    def test_just_over_boundary(self):
        """770x770 → should snap to 512x512."""
        opt_w, opt_h = find_optimal_openai_dimensions(770, 770)
        before = estimate_openai_tokens(770, 770)
        after = estimate_openai_tokens(opt_w, opt_h)
        assert after < before
        assert after == 255  # 1 tile

    def test_anthropic_caps_oversized(self):
        """3000x2000 → capped to 1568 max edge."""
        opt_w, opt_h = find_optimal_anthropic_dimensions(3000, 2000)
        assert max(opt_w, opt_h) <= 1568

    def test_anthropic_no_change_if_small(self):
        """800x600 → no change needed."""
        opt_w, opt_h = find_optimal_anthropic_dimensions(800, 600)
        assert (opt_w, opt_h) == (800, 600)


# ---------------------------------------------------------------------------
# Message-level optimization tests
# ---------------------------------------------------------------------------


def _make_openai_image_message(width: int, height: int) -> list[dict]:
    """Create an OpenAI-format message with a test image."""
    from PIL import Image

    img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }
    ]


def _make_anthropic_image_message(width: int, height: int) -> list[dict]:
    """Create an Anthropic-format message with a test image."""
    from PIL import Image

    img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
            ],
        }
    ]


@needs_pillow
class TestMessageOptimization:
    def test_openai_message_optimized(self):
        """OpenAI message with large image gets tile-optimized."""
        msgs = _make_openai_image_message(1920, 1080)
        optimized, results = optimize_images_in_messages(msgs, "openai")
        assert len(results) == 1
        assert results[0].tokens_saved > 0
        assert results[0].resized

    def test_anthropic_oversized_no_token_change(self):
        """Anthropic oversized image: provider would resize anyway, so no token savings.

        Anthropic's formula is (w*h)/750 after their internal resize. Pre-resizing
        to their limits doesn't change the token count — it only saves upload bandwidth.
        The optimizer correctly returns no results (no token savings to report).
        """
        msgs = _make_anthropic_image_message(3000, 2000)
        optimized, results = optimize_images_in_messages(msgs, "anthropic")
        # No token savings — Anthropic would resize internally anyway
        assert len(results) == 0

    def test_no_image_no_change(self):
        """Message without images passes through unchanged."""
        msgs = [{"role": "user", "content": "Hello"}]
        optimized, results = optimize_images_in_messages(msgs, "openai")
        assert len(results) == 0
        assert optimized == msgs

    def test_text_content_preserved(self):
        """Text content alongside image is preserved."""
        msgs = _make_openai_image_message(1920, 1080)
        optimized, results = optimize_images_in_messages(msgs, "openai")
        text_blocks = [
            b for b in optimized[0]["content"] if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "What is this?"

    def test_small_image_not_resized(self):
        """Image already at optimal size is not changed."""
        msgs = _make_openai_image_message(512, 512)
        optimized, results = optimize_images_in_messages(msgs, "openai")
        assert len(results) == 0  # No optimization needed


# ---------------------------------------------------------------------------
# ONNX Router tests (if available)
# ---------------------------------------------------------------------------


class TestOnnxRouter:
    @pytest.fixture(autouse=True)
    def _check_onnx(self):
        try:
            import onnxruntime  # noqa: F401
            from tokenizers import Tokenizer  # noqa: F401
        except ImportError:
            pytest.skip("onnxruntime or tokenizers not installed")

    def test_query_classification(self):
        """ONNX router classifies queries into techniques."""
        from headroom.image.onnx_router import OnnxTechniqueRouter, Technique

        router = OnnxTechniqueRouter(use_siglip=False)

        tech, conf = router.classify_query("What does the error message say?")
        assert tech == Technique.TRANSCODE
        assert conf > 0.5

        tech, conf = router.classify_query("What's in the top left corner?")
        assert tech == Technique.CROP
        assert conf > 0.5

    def test_preserve_for_detail_queries(self):
        """Queries needing detail should route to PRESERVE or FULL_LOW."""
        from headroom.image.onnx_router import OnnxTechniqueRouter, Technique

        router = OnnxTechniqueRouter(use_siglip=False)

        tech, _ = router.classify_query("Count every item in this image carefully")
        assert tech in (Technique.PRESERVE, Technique.FULL_LOW)

    def test_full_classify_with_image(self):
        """Full classification with query + image analysis."""
        from headroom.image.onnx_router import OnnxTechniqueRouter

        router = OnnxTechniqueRouter(use_siglip=True)

        # Create a simple test image
        from PIL import Image

        img = Image.new("RGB", (224, 224), "white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        decision = router.classify(buf.getvalue(), "Read the text")
        assert decision.technique is not None
        assert decision.confidence > 0
        assert decision.image_signals is not None


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


@needs_pillow
class TestFullPipeline:
    def test_compressor_with_openai_image(self):
        """Full compressor pipeline on OpenAI format."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)
        msgs = _make_openai_image_message(1920, 1080)

        result = compressor.compress(msgs, provider="openai")
        # Should have processed the image (tile opt at minimum)
        assert result is not None
        assert len(result) == 1

    def test_compressor_no_images(self):
        """Compressor is no-op when no images present."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)
        msgs = [{"role": "user", "content": "Hello, no images here"}]

        result = compressor.compress(msgs, provider="openai")
        assert result == msgs

    def test_has_images_openai(self):
        """Detects images in OpenAI format."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor()
        msgs = _make_openai_image_message(100, 100)
        assert compressor.has_images(msgs)

    def test_has_images_anthropic(self):
        """Detects images in Anthropic format."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor()
        msgs = _make_anthropic_image_message(100, 100)
        assert compressor.has_images(msgs)

    def test_no_images_detected(self):
        """No false positives on text-only messages."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor()
        msgs = [{"role": "user", "content": "Just text"}]
        assert not compressor.has_images(msgs)


# ---------------------------------------------------------------------------
# OCR routing tests
# ---------------------------------------------------------------------------


@needs_pillow
class TestOcrRouting:
    @pytest.fixture(autouse=True)
    def _check_ocr(self):
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        except ImportError:
            pytest.skip("rapidocr-onnxruntime not installed")

    def _make_text_image(self, lines: list[str], width: int = 800, height: int = 400) -> bytes:
        """Create a PNG image with text content."""
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        y = 30
        for line in lines:
            draw.text((30, y), line, fill="black")
            y += 40
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_ocr_extracts_text(self):
        """OCR should extract text from a text-heavy image."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)
        image_data = self._make_text_image(
            [
                "Error: connection refused",
                "at localhost:5432",
            ]
        )
        text = compressor._ocr_extract(image_data)
        assert text is not None
        assert len(text) > 10
        # Should contain key words (OCR may have minor errors)
        assert "connection" in text.lower() or "error" in text.lower()

    def test_ocr_returns_none_for_blank_image(self):
        """OCR should return None for a blank image (no text)."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)
        from PIL import Image

        img = Image.new("RGB", (200, 200), "blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        text = compressor._ocr_extract(buf.getvalue())
        assert text is None  # No text detected

    def test_ocr_confidence_threshold(self):
        """Low-confidence OCR should return None (fallback to image)."""
        from headroom.image import ImageCompressor

        compressor = ImageCompressor(use_siglip=False)
        # Very noisy image — OCR should have low confidence
        import numpy as np
        from PIL import Image

        noise = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        text = compressor._ocr_extract(buf.getvalue(), min_confidence=0.95)
        # Noisy image: either None (no text) or low confidence → None
        # Either outcome is correct — we don't want to OCR noise
        assert text is None or len(text) < 10

    def test_transcode_replaces_image_with_text(self):
        """Full pipeline: transcode technique should replace image with OCR text."""
        from headroom.image import ImageCompressor
        from headroom.image.trained_router import Technique

        compressor = ImageCompressor(use_siglip=False)

        # Create message with text-heavy image
        image_data = self._make_text_image(
            [
                "Traceback (most recent call last):",
                "  File server.py line 42",
                "psycopg2.OperationalError",
            ]
        )
        b64 = base64.b64encode(image_data).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What does the error say?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]

        # Apply transcode directly
        result = compressor._apply_compression(messages, Technique.TRANSCODE, "openai")

        # The image block should be replaced with a text block
        content = result[0]["content"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]

        # Should have at least 2 text blocks (original query + OCR output)
        assert len(text_blocks) >= 2
        # One should contain OCR output
        ocr_blocks = [b for b in text_blocks if "[OCR from image]" in b.get("text", "")]
        assert len(ocr_blocks) >= 1
