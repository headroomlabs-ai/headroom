"""Runtime helpers for Claude-facing integrations."""

from __future__ import annotations

from urllib.parse import urlparse

DEFAULT_API_URL = "https://api.anthropic.com"

# GH #746: Claude Code stops deferring MCP/system tool schemas (materializing
# every one into its context window) when ANTHROPIC_BASE_URL is a custom host
# and ENABLE_TOOL_SEARCH is unset. Every place that points Claude Code at the
# proxy must keep deferral on, so the env key and its default live here as the
# single source of truth shared by `wrap`, `init`, and `install`.
TOOL_SEARCH_ENV = "ENABLE_TOOL_SEARCH"
TOOL_SEARCH_DEFAULT = "true"
REMOTE_CONTROL_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
REMOTE_CONTROL_FEATURE = "Remote Control"
REMOTE_CONTROL_DISABLED_MESSAGE = (
    f"{REMOTE_CONTROL_FEATURE}: "
    "Claude Code may hide the Remote Control menu while "
    f"{REMOTE_CONTROL_BASE_URL_ENV} points at a custom endpoint "
    "({source}); "
    "launch Claude without Headroom for sessions that need this feature."
)


def remote_control_gate_message(source: str) -> str:
    """Return the shared Remote Control compatibility message for Claude warning paths."""
    source_clean = source.strip() or "this endpoint"
    return REMOTE_CONTROL_DISABLED_MESSAGE.format(source=source_clean)


def is_custom_anthropic_base_url(value: str | None) -> bool:
    """Return whether ANTHROPIC_BASE_URL is custom from Claude's Remote Control gate view."""
    raw = (value or "").strip()
    if not raw:
        return False
    host = (urlparse(raw).hostname or "").strip().lower()
    return host not in {"", "api.anthropic.com"}


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by Claude integrations."""
    return f"http://127.0.0.1:{port}"
