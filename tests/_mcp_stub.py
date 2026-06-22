from __future__ import annotations

import importlib
import sys
from types import ModuleType

_MCP_MODULE_NAMES = (
    "mcp",
    "mcp.server",
    "mcp.server.stdio",
    "mcp.types",
)


def _build_mcp_sdk_stub() -> dict[str, ModuleType]:
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

    return {
        "mcp": mcp_module,
        "mcp.server": mcp_server_module,
        "mcp.server.stdio": mcp_stdio_module,
        "mcp.types": mcp_types_module,
    }


def import_module_with_mcp_stub(module_name: str):
    original_target_module = sys.modules.get(module_name)
    original_modules = {name: sys.modules.get(name) for name in _MCP_MODULE_NAMES}
    stub_modules = _build_mcp_sdk_stub()

    sys.modules.pop(module_name, None)
    for name, module in stub_modules.items():
        sys.modules[name] = module

    try:
        return importlib.import_module(module_name)
    finally:
        if original_target_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_target_module
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
