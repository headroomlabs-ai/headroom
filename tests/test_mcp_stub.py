from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest

from tests import _mcp_stub as mcp_stub


def test_build_mcp_sdk_stub_exposes_minimum_sdk_contract() -> None:
    modules = mcp_stub._build_mcp_sdk_stub()

    # MCP 2.x: handlers are constructor callbacks, not decorators.
    async def on_list_tools(ctx, params=None):  # pragma: no cover - identity only
        return None

    async def on_call_tool(ctx, params):  # pragma: no cover - identity only
        return None

    server = modules["mcp.server"].Server(
        "headroom", on_list_tools=on_list_tools, on_call_tool=on_call_tool
    )
    assert server.name == "headroom"
    assert server.handlers == {"on_list_tools": on_list_tools, "on_call_tool": on_call_tool}
    assert server.create_initialization_options() == {}

    # The 1.x decorator surface must be gone: keeping it would let a caller still
    # on decorators pass its tests while the real 2.x SDK rejects it.
    assert not hasattr(server, "list_tools")
    assert not hasattr(server, "call_tool")

    tool = modules["mcp.types"].Tool(name="search")
    text = modules["mcp.types"].TextContent(text="payload")
    assert tool.kwargs == {"name": "search"}
    assert text.kwargs == {"text": "payload"}

    tools_result = modules["mcp.types"].ListToolsResult(tools=[tool])
    call_result = modules["mcp.types"].CallToolResult(content=[text])
    assert tools_result.tools == [tool]
    assert call_result.content == [text]

    with pytest.raises(RuntimeError, match="should not run"):
        asyncio.run(modules["mcp.server.stdio"].stdio_server())


def test_import_module_with_mcp_stub_imports_target_and_cleans_up(monkeypatch) -> None:
    original_mcp = ModuleType("mcp")
    monkeypatch.setitem(sys.modules, "mcp", original_mcp)
    for name in mcp_stub._MCP_MODULE_NAMES[1:]:
        sys.modules.pop(name, None)

    imported = ModuleType("fake_target")

    def fake_import_module(module_name: str) -> ModuleType:
        assert module_name == "fake_target"
        for name in mcp_stub._MCP_MODULE_NAMES:
            assert name in sys.modules
        return imported

    monkeypatch.setattr(mcp_stub.importlib, "import_module", fake_import_module)

    result = mcp_stub.import_module_with_mcp_stub("fake_target")

    assert result is imported
    assert sys.modules["mcp"] is original_mcp
    for name in mcp_stub._MCP_MODULE_NAMES[1:]:
        assert name not in sys.modules


def test_import_module_with_mcp_stub_reimports_target_and_restores_originals(monkeypatch) -> None:
    original_modules = {}
    for name in mcp_stub._MCP_MODULE_NAMES:
        module = ModuleType(f"original::{name}")
        original_modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    existing = ModuleType("fake_target")
    monkeypatch.setitem(sys.modules, "fake_target", existing)
    imported = ModuleType("fake_target")

    def fake_import_module(module_name: str) -> ModuleType:
        assert module_name == "fake_target"
        assert "fake_target" not in sys.modules
        for name in mcp_stub._MCP_MODULE_NAMES:
            assert sys.modules[name] is not original_modules[name]
        return imported

    monkeypatch.setattr(mcp_stub.importlib, "import_module", fake_import_module)

    result = mcp_stub.import_module_with_mcp_stub("fake_target")

    assert result is imported
    assert sys.modules["fake_target"] is existing
    for name, module in original_modules.items():
        assert sys.modules[name] is module


def test_import_module_with_mcp_stub_cleans_up_dotted_target_attribute(monkeypatch) -> None:
    parent = ModuleType("fakepkg")
    monkeypatch.setitem(sys.modules, "fakepkg", parent)
    for name in mcp_stub._MCP_MODULE_NAMES:
        sys.modules.pop(name, None)

    imported = ModuleType("fakepkg.fake_target")

    def fake_import_module(module_name: str) -> ModuleType:
        assert module_name == "fakepkg.fake_target"
        parent.fake_target = imported
        return imported

    monkeypatch.setattr(mcp_stub.importlib, "import_module", fake_import_module)

    result = mcp_stub.import_module_with_mcp_stub("fakepkg.fake_target")

    assert result is imported
    assert not hasattr(parent, "fake_target")
