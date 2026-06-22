from __future__ import annotations

import importlib
import sys


def ensure_mcp_sdk_stub() -> None:
    if "mcp" in sys.modules:
        return

    mcp_module = type(sys)("mcp")
    mcp_server_module = type(sys)("mcp.server")
    mcp_stdio_module = type(sys)("mcp.server.stdio")
    mcp_types_module = type(sys)("mcp.types")

    class DummyServer:
        def __init__(self, name: str) -> None:
            self.name = name

        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            return lambda fn: fn

        def create_initialization_options(self):
            return {}

    class DummyTool:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class DummyTextContent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    async def dummy_stdio_server():
        raise RuntimeError("stdio_server should not run in unit tests")

    mcp_server_module.Server = DummyServer
    mcp_stdio_module.stdio_server = dummy_stdio_server
    mcp_types_module.TextContent = DummyTextContent
    mcp_types_module.Tool = DummyTool

    sys.modules["mcp"] = mcp_module
    sys.modules["mcp.server"] = mcp_server_module
    sys.modules["mcp.server.stdio"] = mcp_stdio_module
    sys.modules["mcp.types"] = mcp_types_module


def import_module_with_mcp_stub(module_name: str):
    ensure_mcp_sdk_stub()
    return importlib.import_module(module_name)
