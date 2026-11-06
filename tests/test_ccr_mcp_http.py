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
            base_url="http://testserver",
            event_hooks={"response": [record_session_id]},
        ) as http_client:
            async with streamable_http_client(
                "http://testserver/mcp",
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
