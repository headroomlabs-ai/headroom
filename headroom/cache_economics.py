"""Lightweight provider cache fallbacks shared by all pricing surfaces.

Published per-model catalog rates always take precedence. These ratios are
estimates for priced models whose catalog row has no cache rates. xAI uses
a 0.16 read ratio and no write premium; known model rates override it.
"""

from typing import Any

CACHE_ECONOMICS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "label": "Explicit breakpoints, 5-min TTL",
    },
    "openai": {
        "read_multiplier": 0.5,
        "write_multiplier": 1.0,
        "label": "Automatic, no TTL control",
    },
    "gemini": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.0,
        "label": "Explicit cachedContent, configurable TTL",
    },
    "bedrock": {
        "read_multiplier": 0.1,
        "write_multiplier": 1.25,
        "label": "Same as Anthropic (Bedrock)",
    },
    "xai": {"read_multiplier": 0.16, "write_multiplier": 1.0, "label": "Automatic, no TTL control"},
}
