"""Contract tests for the Headroom Streamable HTTP MCP transport."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

# MCP SDK 2 clients take an httpx2 client (a dependency of the SDK itself).
import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from headroom.ccr.mcp_http import (
    create_streamable_http_app,
    create_streamable_http_session_manager,
)
from headroom.ccr.mcp_server import create_ccr_mcp_server

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_streamable_http_initialize_and_list_tools() -> None:
    server = create_ccr_mcp_server()
    session_manager = create_streamable_http_session_manager(server)
    app = create_streamable_http_app(session_manager, path="/mcp")
    transport = httpx2.ASGITransport(app=app)
    # SDK 2's client no longer yields a session-id getter; read the header instead.
    session_ids: list[str] = []

    async def record_session_id(response: httpx2.Response) -> None:
        if session_id := response.headers.get("mcp-session-id"):
            session_ids.append(session_id)

    async with session_manager.run():
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:8788",
            event_hooks={"response": [record_session_id]},
        ) as http_client:
            async with streamable_http_client(
                "http://127.0.0.1:8788/mcp",
                http_client=http_client,
                terminate_on_close=False,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    initialize_result = await session.initialize()
                    list_tools_result = await session.list_tools()

    tool_names = [tool.name for tool in list_tools_result.tools]
    assert initialize_result.protocol_version
    assert session_ids
    assert "headroom_compress" in tool_names
    assert "headroom_retrieve" in tool_names
    assert "headroom_stats" in tool_names


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}


async def _post_initialize(app, base_url: str, headers: dict[str, str]) -> httpx2.Response:
    async with httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url=base_url
    ) as http_client:
        return await http_client.post(
            "/mcp",
            json=_INITIALIZE,
            headers={"Accept": "application/json, text/event-stream", **headers},
        )


@pytest.mark.parametrize(
    ("base_url", "headers", "status"),
    [
        # A page whose hostname now resolves to 127.0.0.1: Host names that page.
        ("http://page.example:8788", {"Origin": "http://page.example:8788"}, 421),
        # Right Host, but the request comes from another page's script.
        ("http://127.0.0.1:8788", {"Origin": "https://page.example"}, 403),
    ],
)
async def test_streamable_http_refuses_foreign_host_or_origin(
    base_url: str, headers: dict[str, str], status: int
) -> None:
    session_manager = create_streamable_http_session_manager(
        create_ccr_mcp_server(), host="127.0.0.1", port=8788
    )
    app = create_streamable_http_app(session_manager, path="/mcp")

    async with session_manager.run():
        response = await _post_initialize(app, base_url, headers)

    assert response.status_code == status
    assert "mcp-session-id" not in response.headers


async def test_streamable_http_accepts_configured_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_MCP_ALLOWED_HOSTS", "mcp.internal:8788")
    session_manager = create_streamable_http_session_manager(
        create_ccr_mcp_server(), host="0.0.0.0", port=8788
    )
    app = create_streamable_http_app(session_manager, path="/mcp")

    async with session_manager.run():
        configured = await _post_initialize(app, "http://mcp.internal:8788", {})
        loopback = await _post_initialize(app, "http://localhost:8788", {})
        other = await _post_initialize(app, "http://other.internal:8788", {})

    assert configured.status_code == 200, configured.text
    assert loopback.status_code == 200, loopback.text
    assert other.status_code == 421


@pytest.mark.parametrize(
    ("host", "port", "base_url"),
    [
        ("127.0.0.1", 80, "http://localhost"),
        ("[::1]", 8788, "http://[::1]:8788"),
        ("127.0.0.2", 8788, "http://127.0.0.2:8788"),
    ],
)
async def test_streamable_http_accepts_loopback_as_clients_address_it(
    host: str, port: int, base_url: str
) -> None:
    session_manager = create_streamable_http_session_manager(
        create_ccr_mcp_server(), host=host, port=port
    )
    app = create_streamable_http_app(session_manager, path="/mcp")

    async with session_manager.run():
        response = await _post_initialize(app, base_url, {"Origin": base_url})

    assert response.status_code == 200, response.text
