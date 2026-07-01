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
            self.list_tools_handler = None
            self.call_tool_handler = None

        def list_tools(self):
            def decorator(fn):
                self.list_tools_handler = fn
                return fn

            return decorator

        def call_tool(self):
            def decorator(fn):
                self.call_tool_handler = fn
                return fn

            return decorator

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
    parent_name, _, child_name = module_name.rpartition(".")
    original_parent_module = sys.modules.get(parent_name) if parent_name else None
    original_parent_attr_exists = bool(
        original_parent_module and child_name and hasattr(original_parent_module, child_name)
    )
    original_parent_attr = (
        getattr(original_parent_module, child_name) if original_parent_attr_exists else None
    )
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
        if child_name:
            current_parent_module = sys.modules.get(parent_name) or original_parent_module
            if current_parent_module is not None:
                if original_parent_attr_exists:
                    setattr(current_parent_module, child_name, original_parent_attr)
                else:
                    try:
                        delattr(current_parent_module, child_name)
                    except AttributeError:
                        pass
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module
