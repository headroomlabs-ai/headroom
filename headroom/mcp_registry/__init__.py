"""Generic MCP server registration across coding agents.

The MCP protocol is universal but each agent's *registration* mechanism is
not — Claude Code uses its own CLI + ``~/.claude/.claude.json``, Cursor
writes ``~/.cursor/mcp.json``, Codex patches a TOML file, and so on. This
module provides a uniform interface so headroom can install its MCP server
(``headroom mcp serve``) into every detected agent.

Wave 1 ships :class:`ClaudeRegistrar`. Other registrars (Cursor, Codex,
Continue, Cline, Windsurf, Goose, OpenHands) are added in subsequent waves
without changing the calling code.
"""

from __future__ import annotations

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec
from .claude import ClaudeRegistrar
from .codex import CodexRegistrar
from .display import any_succeeded, format_result, format_results
from .grok import GrokRegistrar
from .install import (
    DEFAULT_PROXY_URL,
    build_headroom_spec,
    build_serena_spec,
    get_all_registrars,
    install_everywhere,
)
from .opencode import OpencodeRegistrar
from .server_json import build_server_json, render_server_json

__all__ = [
    "DEFAULT_PROXY_URL",
    "ClaudeRegistrar",
    "CodexRegistrar",
    "GrokRegistrar",
    "MCPRegistrar",
    "OpencodeRegistrar",
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
