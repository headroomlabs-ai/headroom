"""Exercise the installed Enola CLI over stdio MCP; optional extra required."""

from datetime import timedelta

import pytest


@pytest.mark.asyncio
async def test_enola_snapshot_navigation_and_delta(tmp_path, monkeypatch):
    mcp = pytest.importorskip("mcp")
    from mcp.client.stdio import stdio_client

    from headroom.mcp_registry.enola import build_enola_spec

    monkeypatch.delenv("ENOLA_PATH", raising=False)
    try:
        spec = build_enola_spec()
    except ValueError:
        pytest.skip("install headroom-ai[enola] to run the Enola MCP integration")

    source = tmp_path / "app.py"
    source.write_text("def greet(name):\n    return 'Hello ' + name\n")
    params = mcp.StdioServerParameters(
        command=spec.command, args=list(spec.args), env=spec.env, cwd=str(tmp_path)
    )
    async with stdio_client(params) as (reader, writer):
        async with mcp.ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {
                "generate_snapshot",
                "explore",
                "show_symbol",
                "set_baseline",
                "diff_snapshot",
            } <= names
            snapshot = await session.call_tool("generate_snapshot", {"repo_path": str(tmp_path)})
            assert not snapshot.model_dump(by_alias=True).get("isError"), snapshot
            overview = await session.call_tool("explore", {"focus": "app.py"})
            assert not overview.model_dump(by_alias=True).get("isError"), overview
            assert "greet" in str(overview.content)
            symbol = await session.call_tool("show_symbol", {"name": "greet"})
            assert not symbol.model_dump(by_alias=True).get("isError"), symbol
            assert "Hello" in str(symbol.content)
            baseline = await session.call_tool("set_baseline", {})
            assert not baseline.model_dump(by_alias=True).get("isError"), baseline
            source.write_text(
                source.read_text() + "\ndef farewell(name):\n    return 'Bye ' + name\n"
            )
            refreshed = await session.call_tool("generate_snapshot", {"repo_path": str(tmp_path)})
            assert not refreshed.model_dump(by_alias=True).get("isError"), refreshed
            delta = await session.call_tool("diff_snapshot", {})
            assert not delta.model_dump(by_alias=True).get("isError"), delta
            assert "farewell" in str(delta.content)
