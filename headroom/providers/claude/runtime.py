"""Runtime helpers for Claude-facing integrations."""

from __future__ import annotations

DEFAULT_API_URL = "https://api.anthropic.com"

# GH #746: Claude Code stops deferring MCP/system tool schemas (materializing
# every one into its context window) when ANTHROPIC_BASE_URL is a custom host
# and ENABLE_TOOL_SEARCH is unset. Every place that points Claude Code at the
# proxy must keep deferral on, so the env key and its default live here as the
# single source of truth shared by `wrap`, `init`, and `install`.
TOOL_SEARCH_ENV = "ENABLE_TOOL_SEARCH"
TOOL_SEARCH_DEFAULT = "true"


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by Claude integrations."""
    return f"http://127.0.0.1:{port}"
