"""Provider-owned image block parsing and rewrite helpers."""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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
            return 85

    return None


def tile_optimization_plan(
    *,
    block_provider: str,
    requested_provider: str,
    width: int,
    height: int,
    estimate_openai_tokens: Callable[[int, int, str], int],
    estimate_anthropic_tokens: Callable[[int, int], int],
    find_optimal_openai_dimensions: Callable[[int, int], tuple[int, int]],
    find_optimal_anthropic_dimensions: Callable[[int, int], tuple[int, int]],
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
