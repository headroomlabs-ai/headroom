"""Cloud Code provider helpers."""

from .runtime import (
    ANTIGRAVITY_DAILY_API_URL,
    cloudcode_passthrough_target,
    is_antigravity_user_agent,
    normalize_cloudcode_passthrough_path,
)

# Server-created ASGI state, never an HTTP header supplied by a caller.
AGY_DISPATCH_SCOPE_KEY = "headroom.agy_dispatch"

__all__ = [
    "AGY_DISPATCH_SCOPE_KEY",
    "ANTIGRAVITY_DAILY_API_URL",
    "cloudcode_passthrough_target",
    "is_antigravity_user_agent",
    "normalize_cloudcode_passthrough_path",
]
