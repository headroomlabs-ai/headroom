"""Contract tests against the REAL MCP SDK (no stub).

Every other MCP test in this suite goes through ``tests/_mcp_stub.py``, which
replaces the SDK in ``sys.modules``. That is fast and dependency-free, but it
means a breaking change in the real SDK is invisible: the ``mcp`` 2.0.0 release
removed the ``Server.list_tools()``/``call_tool()`` decorators and every stubbed
test stayed green while ``headroom mcp serve`` crashed on startup for anyone who
installed it (#2642).

These tests import the SDK for real so that class of breakage fails here first.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

# ``mcp`` is an optional extra; skip rather than fail where it isn't installed.
mcp_types = pytest.importorskip("mcp.types", reason="mcp SDK not installed")
mcp_server = pytest.importorskip("mcp.server", reason="mcp SDK not installed")

from headroom.ccr.mcp_server import (  # noqa: E402 - must follow importorskip
    CCR_TOOL_NAME,
    COMPRESS_TOOL_NAME,
    STATS_TOOL_NAME,
    HeadroomMCPServer,
    _request_ctx,
)


def _server() -> HeadroomMCPServer:
    return HeadroomMCPServer(proxy_url="http://127.0.0.1:8787")


def _client_ctx(name: str) -> SimpleNamespace:
    """Minimal stand-in for ``ServerRequestContext`` for client attribution."""
    return SimpleNamespace(
        session=SimpleNamespace(
            client_params=SimpleNamespace(clientInfo=SimpleNamespace(name=name))
        )
    )


def test_sdk_registers_handlers_via_constructor_not_decorators() -> None:
    """The canary: 2.x dropped the decorator surface headroom used to rely on.

    If a future SDK reintroduces or re-removes either shape, this fails loudly
    instead of surfacing as a startup crash in front of a user.
    """
    assert not hasattr(mcp_server.Server, "list_tools")
    assert not hasattr(mcp_server.Server, "call_tool")
    # ``request_context`` moved to a per-call handler argument.
    assert not hasattr(mcp_server.Server, "request_context")


def test_server_constructs_and_advertises_tools_capability() -> None:
    srv = _server()
    opts = srv.server.create_initialization_options()
    assert opts.capabilities.tools is not None


def test_list_tools_returns_typed_result() -> None:
    result = asyncio.run(_server()._on_list_tools(None, None))
    assert isinstance(result, mcp_types.ListToolsResult)
    names = {t.name for t in result.tools}
    assert {COMPRESS_TOOL_NAME, CCR_TOOL_NAME, STATS_TOOL_NAME} <= names


def test_call_tool_returns_typed_result_and_dispatches_on_params_name() -> None:
    params = mcp_types.CallToolRequestParams(name="nope", arguments={})
    result = asyncio.run(_server()._on_call_tool(None, params))
    assert isinstance(result, mcp_types.CallToolResult)
    assert "Unknown tool: nope" in result.content[0].text


def test_call_tool_tolerates_omitted_arguments() -> None:
    """2.x types ``arguments`` as optional; the ``_handle_*`` helpers call ``.get()``."""
    params = mcp_types.CallToolRequestParams(name="nope")
    result = asyncio.run(_server()._on_call_tool(None, params))
    assert isinstance(result, mcp_types.CallToolResult)


def test_current_client_resolves_from_request_context() -> None:
    """Replaces the removed ``Server.request_context`` lookup.

    ``_current_client`` swallows all exceptions and returns "unknown", so a broken
    port degrades silently into mis-attributed savings rather than crashing. Assert
    the name actually resolves, not merely that the call survives.
    """
    srv = _server()
    token = _request_ctx.set(_client_ctx("claude-code"))
    try:
        assert srv._current_client() == "claude-code"
    finally:
        _request_ctx.reset(token)
    assert srv._current_client() == "unknown"


def test_request_context_does_not_leak_between_calls() -> None:
    """A ContextVar left set would mis-attribute the next call's client."""
    srv = _server()
    params = mcp_types.CallToolRequestParams(name="nope", arguments={})
    asyncio.run(srv._on_call_tool(_client_ctx("claude-code"), params))
    assert _request_ctx.get(None) is None
