"""Exercise the installed Enola CLI over stdio MCP; optional extra required."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude", "codex", "grok", "opencode"])
async def test_enola_from_wrapped_agent_configuration(tmp_path, monkeypatch, agent):
    mcp = pytest.importorskip("mcp")
    from mcp.client.stdio import stdio_client

    from headroom.cli import wrap
    from headroom.cli.main import main
    from headroom.mcp_registry import get_all_registrars
    from headroom.mcp_registry.enola import find_enola_binary

    monkeypatch.delenv("ENOLA_PATH", raising=False)
    try:
        find_enola_binary()
    except ValueError:
        pytest.skip("install headroom-ai[enola] to run the Enola MCP integration")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    for name in (
        "CODEX_HOME",
        "GROK_HOME",
        "OPENCODE_CONFIG",
        "OPENCODE_HOME",
        "XDG_CONFIG_HOME",
        "CLAUDE_CONFIG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".claude").mkdir()
    (tmp_path / "app.py").write_text("def greet(name):\n    return 'Hello ' + name\n")
    args = ["wrap", agent, "--code-memory", "enola", "--no-mcp", "--no-proxy"]
    if agent == "claude":
        # Claude's prepare-only returns before setup. Exercise its normal wrap
        # path with only the proxy and native CLI launch stubbed; MCP config
        # writes and the subsequent Enola subprocess remain real.
        with monkeypatch.context() as launch:
            launch.setattr(wrap, "_require_binary", lambda *a, **k: "claude")
            launch.setattr(wrap, "_ensure_proxy", lambda *a, **k: (None, 8787))
            launch.setattr(wrap, "_make_cleanup", lambda *a, **k: lambda: None)
            launch.setattr(wrap.signal, "signal", lambda *a, **k: None)
            launch.setattr(
                wrap.subprocess,
                "run",
                lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
            )
            result = CliRunner().invoke(main, args)
    else:
        result = CliRunner().invoke(main, [*args, "--prepare-only"])
    assert result.exit_code == 0, result.output
    registrar = next(r for r in get_all_registrars() if r.name == agent)
    spec = registrar.get_server("enola")
    assert spec is not None, result.output
    params = mcp.StdioServerParameters(
        command=spec.command, args=list(spec.args), env=spec.env, cwd=str(tmp_path)
    )
    async with stdio_client(params) as (reader, writer):
        async with mcp.ClientSession(reader, writer, read_timeout_seconds=60) as session:
            await session.initialize()
            snapshot = await session.call_tool("generate_snapshot", {"repo_path": str(tmp_path)})
            assert not snapshot.model_dump(by_alias=True).get("isError"), snapshot
            symbol = await session.call_tool("show_symbol", {"name": "greet"})
            assert not symbol.model_dump(by_alias=True).get("isError"), symbol
            assert "Hello" in str(symbol.content)
    print(
        f"{agent}: saved Enola configuration -> MCP initialize -> snapshot -> greet source verified"
    )


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
        async with mcp.ClientSession(reader, writer, read_timeout_seconds=60) as session:
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
