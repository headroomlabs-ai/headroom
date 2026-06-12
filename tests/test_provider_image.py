"""Provider-owned image policy tests."""

from __future__ import annotations

import base64

from headroom.providers.image import (
    decode_image_block,
    estimate_low_detail_tokens,
    rewrite_low_detail_image_block,
    rewrite_resized_image_block,
    tile_optimization_plan,
)


def test_decode_openai_image_block() -> None:
    encoded = base64.b64encode(b"image-bytes").decode()

    block = decode_image_block(
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
    )

    assert block is not None
    assert block.provider == "openai"
    assert block.media_type == "image/png"
    assert block.image_bytes == b"image-bytes"


def test_decode_anthropic_image_block() -> None:
    encoded = base64.b64encode(b"anthropic-bytes").decode()

    block = decode_image_block(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encoded,
            },
        }
    )

    assert block is not None
    assert block.provider == "anthropic"
    assert block.media_type == "image/jpeg"
    assert block.image_bytes == b"anthropic-bytes"


def test_rewrite_low_detail_openai_sets_detail_without_reencoding() -> None:
    item = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}

    rewritten = rewrite_low_detail_image_block(item, "openai")

    assert rewritten["image_url"]["url"] == "data:image/png;base64,abc"
    assert rewritten["image_url"]["detail"] == "low"
    assert estimate_low_detail_tokens(rewritten) == 85


def test_rewrite_resized_google_image_block() -> None:
    rewritten = rewrite_resized_image_block(
        {},
        "google",
        resized_data=b"new-bytes",
        media_type="image/jpeg",
    )

    assert rewritten == {
        "inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(b"new-bytes").decode(),
        }
    }


def test_openai_tile_plan_uses_requested_provider_result() -> None:
    plan = tile_optimization_plan(
        block_provider="openai",
        requested_provider="anthropic",
        width=1024,
        height=1024,
        estimate_openai_tokens=lambda width, height, detail: width + height,
        estimate_anthropic_tokens=lambda width, height: width * height,
        find_optimal_openai_dimensions=lambda width, height: (512, 512),
        find_optimal_anthropic_dimensions=lambda width, height: (768, 768),
    )

    assert plan is not None
    assert plan.result_provider == "anthropic"
    assert plan.tokens_before == 2048
    assert plan.tokens_after == 1536
    assert (plan.optimized_width, plan.optimized_height) == (768, 768)
