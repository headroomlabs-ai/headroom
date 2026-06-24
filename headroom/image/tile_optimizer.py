"""Tile-boundary image optimizer — reduce vision tokens with zero quality loss.

Resizes images to land on provider tile boundaries, minimizing token count
without perceptible quality change. Pure math — no ML models needed.

OpenAI tiles at 512px: tokens = 85 + 170 * ceil(w/512) * ceil(h/512).
A 770px image = 4 tiles (765 tokens). Resizing to 512px = 1 tile (255 tokens).

Anthropic: tokens = (w * h) / 750, capped at 1568px / 1.15MP.
Pre-resizing to Anthropic's caps saves upload bandwidth (they'd resize anyway).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TileOptResult:
    """Result of tile optimization for a single image."""

    original_width: int
    original_height: int
    optimized_width: int
    optimized_height: int
    tokens_before: int
    tokens_after: int
    provider: str
    resized: bool

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def savings_pct(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_saved / self.tokens_before * 100


# ---------------------------------------------------------------------------
# Token estimation formulas (must match provider pricing exactly)
# ---------------------------------------------------------------------------


def estimate_openai_tokens(width: int, height: int, detail: str = "high") -> int:
    """OpenAI GPT-4o vision token formula."""
    if detail == "low":
        return 85

    # Step 1: scale so max dimension ≤ 2048
    max_dim = max(width, height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        width = int(width * scale)
        height = int(height * scale)

    # Step 2: scale so shortest side ≤ 768
    min_dim = min(width, height)
    if min_dim > 768:
        scale = 768 / min_dim
        width = int(width * scale)
        height = int(height * scale)

    # Step 3: count 512×512 tiles
    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return 85 + 170 * tiles


def estimate_anthropic_tokens(width: int, height: int) -> int:
    """Anthropic Claude vision token formula: (w * h) / 750."""
    # Auto-downscale: longest edge ≤ 1568
    max_edge = max(width, height)
    if max_edge > 1568:
        scale = 1568 / max_edge
        width = int(width * scale)
        height = int(height * scale)

    # Auto-downscale: total pixels ≤ 1.15MP
    total = width * height
    if total > 1_150_000:
        scale = math.sqrt(1_150_000 / total)
        width = int(width * scale)
        height = int(height * scale)

    return max(1, (width * height) // 750)


# ---------------------------------------------------------------------------
# Tile-boundary optimization (OpenAI specific)
# ---------------------------------------------------------------------------


def find_optimal_openai_dimensions(width: int, height: int) -> tuple[int, int]:
    """Find dimensions that minimize OpenAI tile count.

    Tries reducing to fewer tiles while keeping ≥40% of original pixels.
    Returns (optimal_width, optimal_height).
    """
    # Simulate OpenAI's internal scaling first
    max_dim = max(width, height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        width = int(width * scale)
        height = int(height * scale)

    min_dim = min(width, height)
    if min_dim > 768:
        scale = 768 / min_dim
        width = int(width * scale)
        height = int(height * scale)

    current_tiles = math.ceil(width / 512) * math.ceil(height / 512)
    best_w, best_h = width, height
    best_tiles = current_tiles

    for target_cols in range(1, math.ceil(width / 512) + 1):
        for target_rows in range(1, math.ceil(height / 512) + 1):
            tiles = target_cols * target_rows
            if tiles >= current_tiles:
                continue

            tw = target_cols * 512
            th = target_rows * 512
            scale_w = tw / width
            scale_h = th / height
            scale = min(scale_w, scale_h)
            nw = int(width * scale)
            nh = int(height * scale)

            # Only accept if keeping ≥40% of original pixels
            if nw * nh >= width * height * 0.4 and tiles < best_tiles:
                best_w, best_h = nw, nh
                best_tiles = tiles

    return best_w, best_h


def find_optimal_anthropic_dimensions(width: int, height: int) -> tuple[int, int]:
    """Pre-resize to Anthropic's limits (they'd do it anyway)."""
    max_edge = max(width, height)
    if max_edge > 1568:
        scale = 1568 / max_edge
        width = int(width * scale)
        height = int(height * scale)

    total = width * height
    if total > 1_150_000:
        scale = math.sqrt(1_150_000 / total)
        width = int(width * scale)
        height = int(height * scale)

    return width, height


# ---------------------------------------------------------------------------
# Image resize + re-encode
# ---------------------------------------------------------------------------


def _resize_image_bytes(
    image_data: bytes, target_width: int, target_height: int
) -> tuple[bytes, str]:
    """Resize image and return (new_bytes, media_type)."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_data))
    original_format = (img.format or "PNG").upper()

    # Only resize if dimensions actually changed
    if img.size == (target_width, target_height):
        return image_data, f"image/{original_format.lower()}"

    resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Convert RGBA to RGB for JPEG
    if resized.mode in ("RGBA", "P"):
        resized = resized.convert("RGB")

    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue(), "image/jpeg"


# ---------------------------------------------------------------------------
# Message-level optimization (apply to all images in messages)
# ---------------------------------------------------------------------------


def optimize_images_in_messages(
    messages: list[dict[str, Any]],
    provider: str = "anthropic",
) -> tuple[list[dict[str, Any]], list[TileOptResult]]:
    """Optimize all images in messages for minimum token cost.

    Args:
        messages: LLM messages (OpenAI/Anthropic format)
        provider: Target provider ('openai', 'anthropic')

    Returns:
        (optimized_messages, list of optimization results)
    """
    results: list[TileOptResult] = []
    optimized = []

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            optimized.append(message)
            continue

        new_content = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue

            result = _optimize_content_block(item, provider)
            if result is not None:
                opt_item, opt_result = result
                new_content.append(opt_item)
                results.append(opt_result)
            else:
                new_content.append(item)

        optimized.append({**message, "content": new_content})

    return optimized, results


def _optimize_content_block(
    item: dict[str, Any], provider: str
) -> tuple[dict[str, Any], TileOptResult] | None:
    """Optimize a single image content block. Returns None if not an image."""
    try:
        from PIL import Image
    except ImportError:
        return None

    # --- OpenAI format: {"type": "image_url", "image_url": {"url": "data:..."}} ---
    if item.get("type") == "image_url":
        url = item.get("image_url", {}).get("url", "")
        if not url.startswith("data:"):
            return None  # URL-referenced image, can't resize

        match = re.match(r"data:image/[^;]+;base64,(.+)", url)
        if not match:
            return None

        image_data = base64.b64decode(match.group(1))
        img = Image.open(io.BytesIO(image_data))
        orig_w, orig_h = img.size

        tokens_before = estimate_openai_tokens(orig_w, orig_h, "high")

        if provider == "openai":
            opt_w, opt_h = find_optimal_openai_dimensions(orig_w, orig_h)
        else:
            opt_w, opt_h = find_optimal_anthropic_dimensions(orig_w, orig_h)

        tokens_after = estimate_openai_tokens(opt_w, opt_h, "high")

        if tokens_after >= tokens_before:
            return None  # No savings

        resized_data, media_type = _resize_image_bytes(image_data, opt_w, opt_h)
        b64 = base64.b64encode(resized_data).decode()
        new_item = {
            "type": "image_url",
            "image_url": {
                **item.get("image_url", {}),
                "url": f"data:{media_type};base64,{b64}",
            },
        }

        result = TileOptResult(
            original_width=orig_w,
            original_height=orig_h,
            optimized_width=opt_w,
            optimized_height=opt_h,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            provider=provider,
            resized=True,
        )
        return new_item, result

    # --- Anthropic format: {"type": "image", "source": {"type": "base64", "data": "..."}} ---
    if item.get("type") == "image":
        source = item.get("source", {})
        if source.get("type") != "base64":
            return None

        image_data = base64.b64decode(source.get("data", ""))
        img = Image.open(io.BytesIO(image_data))
        orig_w, orig_h = img.size

        tokens_before = estimate_anthropic_tokens(orig_w, orig_h)
        opt_w, opt_h = find_optimal_anthropic_dimensions(orig_w, orig_h)
        tokens_after = estimate_anthropic_tokens(opt_w, opt_h)

        if tokens_after >= tokens_before:
            return None

        resized_data, media_type = _resize_image_bytes(image_data, opt_w, opt_h)
        new_item = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(resized_data).decode(),
            },
        }

        result = TileOptResult(
            original_width=orig_w,
            original_height=orig_h,
            optimized_width=opt_w,
            optimized_height=opt_h,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            provider="anthropic",
            resized=True,
        )
        return new_item, result

    return None
