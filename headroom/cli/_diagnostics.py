"""Import-light diagnostics for the Headroom CLI."""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import TypedDict


class _CapabilityProbe(TypedDict):
    feature: str
    probe_modules: tuple[str, ...]
    install: str


def python_runtime() -> dict[str, object]:
    """Return details about the active Python runtime."""

    version_info = tuple(sys.version_info[:3])
    # README states "Requires Python 3.10+" with no upper bound. Only treat
    # an older interpreter as a hard blocker; newer ones are supported.
    supported = version_info[:2] >= (3, 10)
    note = None
    if not supported:
        note = "Python 3.10+ is required"

    result: dict[str, object] = {
        "version": platform.python_version(),
        "version_info": version_info,
        "supported": supported,
        "supported_range": "3.10+",
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    if note is not None:
        result["note"] = note
    return result


def native_core() -> dict[str, object]:
    """Probe the Rust extension without importing proxy server modules."""

    try:
        spec = importlib.util.find_spec("headroom._core")
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {"present": False, "detail": f"missing: {type(exc).__name__}: {exc}"}

    if spec is None:
        return {
            "present": False,
            "detail": (
                "missing: reinstall the built wheel with `make build-wheel && "
                "pip install --force-reinstall target/wheels/headroom_*.whl`"
            ),
        }

    try:
        core = __import__("headroom._core", fromlist=["hello"])
        hello = getattr(core, "hello", None)
        if hello is None:
            return {"present": False, "detail": "broken: headroom._core.hello() is missing"}
        marker = hello()
    except Exception as exc:  # ImportError, PyO3 init failures, or marker call failures
        return {"present": False, "detail": f"broken: {type(exc).__name__}: {exc}"}

    if marker != "headroom-core":
        return {
            "present": False,
            "detail": f"broken: unexpected marker {marker!r}; expected 'headroom-core'",
        }
    return {"present": True, "detail": "loaded: headroom-core"}


_CAPABILITY_PROBES: tuple[_CapabilityProbe, ...] = (
    {
        "feature": "proxy",
        # run_server() needs both FastAPI and uvicorn; FastAPI does not depend
        # on uvicorn, so probe both to avoid advertising an unusable surface.
        "probe_modules": ("fastapi", "uvicorn"),
        "install": "pip install 'headroom-ai[proxy]'",
    },
    {
        "feature": "mcp",
        "probe_modules": ("mcp",),
        "install": "pip install 'headroom-ai[mcp]'",
    },
    {
        "feature": "ml",
        "probe_modules": ("torch", "transformers", "huggingface_hub"),
        "install": "pip install 'headroom-ai[ml]'",
    },
    {
        "feature": "code",
        "probe_modules": ("tree_sitter_language_pack",),
        "install": "pip install 'headroom-ai[code]'",
    },
    {
        "feature": "memory",
        "probe_modules": ("hnswlib", "sqlite_vec", "sentence_transformers"),
        "install": "pip install 'headroom-ai[memory]'",
    },
    {
        "feature": "memory-stack",
        "probe_modules": ("mem0", "qdrant_client", "neo4j"),
        "install": "pip install 'headroom-ai[memory-stack]'",
    },
)


def _module_available(module_name: str) -> bool:
    """Return whether an optional module is installed without importing it."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def capabilities() -> list[dict[str, object]]:
    """Return optional feature readiness without importing optional modules."""

    results: list[dict[str, object]] = []
    for capability in _CAPABILITY_PROBES:
        probes = list(capability["probe_modules"])
        results.append(
            {
                "feature": capability["feature"],
                "available": all(_module_available(module) for module in probes),
                "probe_modules": probes,
                "install": capability["install"],
            }
        )
    return results


def headroom_version() -> str:
    """Return the installed Headroom version."""

    try:
        from headroom.cli.main import get_version

        return get_version()
    except ImportError:
        try:
            from headroom._version import __version__

            return __version__
        except ImportError:
            return "unknown"


def capability_summary(caps: list[dict[str, object]]) -> list[str]:
    """Return runnable CLI surfaces from the installed extras."""

    available = {str(cap["feature"]): bool(cap["available"]) for cap in caps}
    surfaces = ["library", "wrap"]
    if available.get("proxy"):
        surfaces.append("proxy")
    if available.get("mcp"):
        surfaces.append("mcp")
    return surfaces
