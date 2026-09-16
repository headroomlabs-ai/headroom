"""Provider-owned image block parsing and rewrite helpers."""

from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass
from typing import Any

OPENAI_LOW_DETAIL_IMAGE_TOKENS = 85
OPENAI_TILE_TOKENS = 170
OPENAI_TILE_SIZE = 512
OPENAI_MAX_DIMENSION = 2048
OPENAI_SHORT_SIDE_LIMIT = 768
OPENAI_MIN_PIXEL_RETENTION = 0.4

ANTHROPIC_PIXELS_PER_TOKEN = 750
ANTHROPIC_MAX_EDGE = 1568
ANTHROPIC_MAX_PIXELS = 1_150_000


@dataclass(frozen=True)
class ProviderImageBlock:
    """Decoded provider-native image content block."""

    provider: str
    image_bytes: bytes
    media_type: str | None = None


@dataclass(frozen=True)
class ProviderImageTilePlan:
    """Provider-specific tile optimization plan."""

    result_provider: str
    tokens_before: int
    tokens_after: int
    optimized_width: int
    optimized_height: int


def decode_image_block(
    item: dict[str, Any], provider: str | None = None
) -> ProviderImageBlock | None:
    """Decode a provider-native image content block."""
    if item.get("type") == "image_url" and (provider is None or provider == "openai"):
        url = item.get("image_url", {}).get("url", "")
        if not isinstance(url, str) or not url.startswith("data:"):
            return None
        match = re.match(r"data:(image/[^;]+);base64,(.+)", url)
        if not match:
            return None
        return ProviderImageBlock(
            provider="openai",
            media_type=match.group(1),
            image_bytes=base64.b64decode(match.group(2)),
        )

    if item.get("type") == "image" and (provider is None or provider == "anthropic"):
        source = item.get("source", {})
        if not isinstance(source, dict) or source.get("type") != "base64":
            return None
        return ProviderImageBlock(
            provider="anthropic",
            media_type=source.get("media_type"),
            image_bytes=base64.b64decode(source.get("data", "")),
        )

    if "inlineData" in item and (provider is None or provider == "google"):
        inline_data = item.get("inlineData", {})
        if not isinstance(inline_data, dict):
            return None
        return ProviderImageBlock(
            provider="google",
            media_type=inline_data.get("mimeType"),
            image_bytes=base64.b64decode(inline_data.get("data", "")),
        )

    return None


def rewrite_low_detail_image_block(
    item: dict[str, Any],
    provider: str,
    *,
    resized_data: bytes | None = None,
    media_type: str | None = None,
) -> dict[str, Any]:
    """Return provider-native low-detail/resized image block."""
    if provider == "openai":
        return {
            "type": "image_url",
            "image_url": {
                **item.get("image_url", {}),
                "detail": "low",
            },
        }

    if provider == "anthropic" and resized_data is not None:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(resized_data).decode(),
            },
        }

    if provider == "google" and resized_data is not None:
        return {
            "inlineData": {
                "mimeType": media_type,
                "data": base64.b64encode(resized_data).decode(),
            }
        }

    return item


def rewrite_resized_image_block(
    item: dict[str, Any],
    provider: str,
    *,
    resized_data: bytes,
    media_type: str,
) -> dict[str, Any]:
    """Return provider-native image block with resized bytes."""
    b64 = base64.b64encode(resized_data).decode()
    if provider == "openai":
        return {
            "type": "image_url",
            "image_url": {
                **item.get("image_url", {}),
                "url": f"data:{media_type};base64,{b64}",
            },
        }

    if provider == "anthropic":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }

    if provider == "google":
        return {"inlineData": {"mimeType": media_type, "data": b64}}

    return item


def low_detail_resize_dimension(provider: str) -> int | None:
    """Return provider resize dimension for low-detail image compression."""
    return {
        "anthropic": 512,
        "google": 768,
    }.get(provider)


def estimate_low_detail_tokens(item: dict[str, Any]) -> int | None:
    """Return provider-native low-detail token estimate for an image block."""
    if item.get("type") == "image_url":
        detail = item.get("image_url", {}).get("detail", "high")
        if detail == "low":
            return OPENAI_LOW_DETAIL_IMAGE_TOKENS

    return None


def _scale_openai_dimensions(width: int, height: int) -> tuple[int, int]:
    """Apply OpenAI's documented pre-tokenization image scaling."""
    max_dim = max(width, height)
    if max_dim > OPENAI_MAX_DIMENSION:
        scale = OPENAI_MAX_DIMENSION / max_dim
        width = int(width * scale)
        height = int(height * scale)

    min_dim = min(width, height)
    if min_dim > OPENAI_SHORT_SIDE_LIMIT:
        scale = OPENAI_SHORT_SIDE_LIMIT / min_dim
        width = int(width * scale)
        height = int(height * scale)

    return width, height


def estimate_openai_tokens(width: int, height: int, detail: str = "high") -> int:
    """Return OpenAI GPT-4o-style vision token estimate."""
    if detail == "low":
        return OPENAI_LOW_DETAIL_IMAGE_TOKENS

    width, height = _scale_openai_dimensions(width, height)
    tiles = math.ceil(width / OPENAI_TILE_SIZE) * math.ceil(height / OPENAI_TILE_SIZE)
    return OPENAI_LOW_DETAIL_IMAGE_TOKENS + OPENAI_TILE_TOKENS * tiles


def estimate_anthropic_tokens(width: int, height: int) -> int:
    """Return Anthropic Claude vision token estimate."""
    width, height = find_optimal_anthropic_dimensions(width, height)
    return max(1, (width * height) // ANTHROPIC_PIXELS_PER_TOKEN)


def find_optimal_openai_dimensions(width: int, height: int) -> tuple[int, int]:
    """Find dimensions that minimize OpenAI tile count while preserving quality."""
    width, height = _scale_openai_dimensions(width, height)
    current_tiles = math.ceil(width / OPENAI_TILE_SIZE) * math.ceil(height / OPENAI_TILE_SIZE)
    best_w, best_h = width, height
    best_tiles = current_tiles

    for target_cols in range(1, math.ceil(width / OPENAI_TILE_SIZE) + 1):
        for target_rows in range(1, math.ceil(height / OPENAI_TILE_SIZE) + 1):
            tiles = target_cols * target_rows
            if tiles >= current_tiles:
                continue

            target_width = target_cols * OPENAI_TILE_SIZE
            target_height = target_rows * OPENAI_TILE_SIZE
            scale = min(target_width / width, target_height / height)
            optimized_width = int(width * scale)
            optimized_height = int(height * scale)

            if (
                optimized_width * optimized_height >= width * height * OPENAI_MIN_PIXEL_RETENTION
                and tiles < best_tiles
            ):
                best_w, best_h = optimized_width, optimized_height
                best_tiles = tiles

    return best_w, best_h


def find_optimal_anthropic_dimensions(width: int, height: int) -> tuple[int, int]:
    """Pre-resize to Anthropic image limits."""
    max_edge = max(width, height)
    if max_edge > ANTHROPIC_MAX_EDGE:
        scale = ANTHROPIC_MAX_EDGE / max_edge
        width = int(width * scale)
        height = int(height * scale)

    total = width * height
    if total > ANTHROPIC_MAX_PIXELS:
        scale = math.sqrt(ANTHROPIC_MAX_PIXELS / total)
        width = int(width * scale)
        height = int(height * scale)

    return width, height


def tile_optimization_plan(
    *,
    block_provider: str,
    requested_provider: str,
    width: int,
    height: int,
) -> ProviderImageTilePlan | None:
    """Return provider-specific tile optimization math for an image block."""
    if block_provider == "openai":
        tokens_before = estimate_openai_tokens(width, height, "high")
        if requested_provider == "openai":
            opt_w, opt_h = find_optimal_openai_dimensions(width, height)
        else:
            opt_w, opt_h = find_optimal_anthropic_dimensions(width, height)
        tokens_after = estimate_openai_tokens(opt_w, opt_h, "high")
        return ProviderImageTilePlan(
            result_provider=requested_provider,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            optimized_width=opt_w,
            optimized_height=opt_h,
        )

    if block_provider == "anthropic":
        tokens_before = estimate_anthropic_tokens(width, height)
        opt_w, opt_h = find_optimal_anthropic_dimensions(width, height)
        tokens_after = estimate_anthropic_tokens(opt_w, opt_h)
        return ProviderImageTilePlan(
            result_provider="anthropic",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            optimized_width=opt_w,
            optimized_height=opt_h,
        )

    return None
