"""Cloud Code provider helpers."""

from .runtime import normalize_cloudcode_passthrough_path

# Server-created ASGI state, never an HTTP header supplied by a caller.
AGY_DISPATCH_SCOPE_KEY = "headroom.agy_dispatch"

__all__ = ["AGY_DISPATCH_SCOPE_KEY", "normalize_cloudcode_passthrough_path"]
