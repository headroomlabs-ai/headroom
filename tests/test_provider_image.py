"""Provider-owned image policy tests."""

from __future__ import annotations

import base64

from headroom.providers.image import (
    decode_image_block,
    estimate_anthropic_tokens,
    estimate_low_detail_tokens,
    estimate_openai_tokens,
    find_optimal_anthropic_dimensions,
    find_optimal_openai_dimensions,
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
    )

    assert plan is not None
    assert plan.result_provider == "anthropic"
    assert plan.tokens_before == estimate_openai_tokens(1024, 1024)
    assert plan.tokens_after == estimate_openai_tokens(
        plan.optimized_width,
        plan.optimized_height,
    )
    assert (plan.optimized_width, plan.optimized_height) == find_optimal_anthropic_dimensions(
        1024,
        1024,
    )


def test_provider_image_formula_helpers_are_authoritative() -> None:
    assert estimate_openai_tokens(512, 512) == 255
    assert estimate_openai_tokens(1920, 1080, "low") == 85
    assert estimate_anthropic_tokens(1024, 768) == 1048
    assert find_optimal_openai_dimensions(512, 512) == (512, 512)
    assert find_optimal_anthropic_dimensions(800, 600) == (800, 600)


def test_provider_image_openai_formula_scales_large_images() -> None:
    # Max-dimension and short-side scaling should match the public OpenAI tile formula.
    assert estimate_openai_tokens(4000, 3000) == 765
    assert estimate_openai_tokens(768, 768) == 765

    opt_w, opt_h = find_optimal_openai_dimensions(770, 770)

    assert (opt_w, opt_h) == (512, 512)
    assert estimate_openai_tokens(opt_w, opt_h) < estimate_openai_tokens(770, 770)


def test_provider_image_anthropic_formula_scales_to_limits() -> None:
    assert find_optimal_anthropic_dimensions(3000, 2000) == (1313, 875)

    opt_w, opt_h = find_optimal_anthropic_dimensions(1568, 1568)

    assert opt_w * opt_h <= 1_150_000
    assert estimate_anthropic_tokens(3000, 2000) == max(1, (1313 * 875) // 750)


def test_tile_plan_handles_anthropic_and_unsupported_blocks() -> None:
    anthropic = tile_optimization_plan(
        block_provider="anthropic",
        requested_provider="openai",
        width=3000,
        height=2000,
    )
    unsupported = tile_optimization_plan(
        block_provider="google",
        requested_provider="google",
        width=3000,
        height=2000,
    )

    assert anthropic is not None
    assert anthropic.result_provider == "anthropic"
    assert anthropic.tokens_after == estimate_anthropic_tokens(
        anthropic.optimized_width,
        anthropic.optimized_height,
    )
    assert unsupported is None
