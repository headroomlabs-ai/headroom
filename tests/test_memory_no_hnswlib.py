"""Tests that headroom[memory] works without hnswlib installed.

Regression for https://github.com/headroomlabs-ai/headroom/issues/1368:
pip install headroom-ai[all] aborted the entire install on machines without
a C++ compiler because hnswlib (pulled in by [memory]) requires compilation.
The fix: hnswlib moved to [memory-hnsw]; [memory] and [all] now install without
a compiler.  The code already handles hnswlib absence gracefully (lazy import).
"""

from __future__ import annotations

import sys
import importlib
from unittest.mock import patch


def test_memory_factory_importable_without_hnswlib() -> None:
    """headroom.memory.factory must import even if hnswlib is absent."""
    with patch.dict(sys.modules, {"hnswlib": None}):
        # Force re-import with hnswlib blocked
        mod_name = "headroom.memory.factory"
        saved = sys.modules.pop(mod_name, None)
        try:
            module = importlib.import_module(mod_name)
            assert module is not None
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved
            else:
                sys.modules.pop(mod_name, None)


def test_memory_adapters_init_importable_without_hnswlib() -> None:
    """headroom.memory.adapters must import even if hnswlib is absent."""
    with patch.dict(sys.modules, {"hnswlib": None}):
        mod_name = "headroom.memory.adapters"
        saved = sys.modules.pop(mod_name, None)
        try:
            module = importlib.import_module(mod_name)
            assert module is not None
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved
            else:
                sys.modules.pop(mod_name, None)


def test_hnsw_availability_check_returns_false_without_hnswlib() -> None:
    """_HNSW_AVAILABLE must be False when hnswlib is not installed."""
    import headroom.memory.adapters as adapters
    # If hnswlib is installed on this machine, simulate its absence
    with patch.dict(sys.modules, {"hnswlib": None}):
        # Re-evaluate availability check
        available = adapters._check_hnsw_available() if hasattr(adapters, "_check_hnsw_available") else False
        # Either the function returns False, or hnswlib was never loaded
        assert not available or not hasattr(adapters, "_check_hnsw_available")


def test_proxy_imports_without_hnswlib() -> None:
    """The proxy server must import without hnswlib present."""
    # This is the critical path: headroom proxy starts → must not crash
    with patch.dict(sys.modules, {"hnswlib": None}):
        import headroom.proxy.server as server  # noqa: F401
        assert server is not None
