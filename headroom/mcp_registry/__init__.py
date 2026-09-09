"""MCP server registration for Claude Code.

The MCP protocol is universal but each agent's *registration* mechanism is
not — Claude Code uses its own CLI + ``~/.claude/.claude.json``. This module
provides the registrar that installs headroom's MCP server
(``headroom mcp serve``) into Claude Code.
"""

from __future__ import annotations

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec
from .claude import ClaudeConfigMutationError, ClaudeRegistrar
from .display import any_succeeded, format_result, format_results
from .install import (
    CLAUDE_SERENA_CONTEXT,
    DEFAULT_PROXY_URL,
    build_headroom_spec,
    build_serena_spec,
    get_all_registrars,
    install_everywhere,
)
from .server_json import build_server_json, render_server_json

__all__ = [
    "DEFAULT_PROXY_URL",
    "CLAUDE_SERENA_CONTEXT",
    "ClaudeConfigMutationError",
    "ClaudeRegistrar",
    "MCPRegistrar",
    "RegisterResult",
    "RegisterStatus",
    "ServerSpec",
    "any_succeeded",
    "build_headroom_spec",
    "build_serena_spec",
    "build_server_json",
    "format_result",
    "format_results",
    "get_all_registrars",
    "install_everywhere",
    "render_server_json",
]
