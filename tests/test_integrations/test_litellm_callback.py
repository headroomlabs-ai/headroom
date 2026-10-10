"""Tests for headroom.integrations.litellm_callback."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

_INTEGRATIONS = Path(__file__).resolve().parents[2] / "headroom" / "integrations"
_BACKEND = "headroom.integrations._compress_backend"


def _load_file(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _INTEGRATIONS / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_callback() -> type:
    # Load the module files directly to avoid triggering headroom/integrations/__init__.py,
    # which pulls in langchain and the native .so extension. litellm_callback imports
    # _compress_backend by its dotted name, so register a file-loaded copy for that one
    # import; with the name already in sys.modules, the integrations package is not
    # imported. Both files still import the top-level headroom package and its
    # stdlib-only helpers (headroom.offline, headroom.log_safety); the subprocess test
    # below checks that neither langchain nor headroom._core is loaded.
    saved = sys.modules.get(_BACKEND)
    sys.modules[_BACKEND] = _load_file(_BACKEND, "_compress_backend.py")
    try:
        mod = _load_file("headroom.integrations.litellm_callback", "litellm_callback.py")
    finally:
        if saved is None:
            del sys.modules[_BACKEND]
        else:
            sys.modules[_BACKEND] = saved
    return mod.HeadroomCallback  # type: ignore[no-any-return]


HeadroomCallback = _import_callback()


class TestHeadroomCallbackPostCallSuccessHook:
    """async_post_call_success_hook must exist and return response unchanged."""

    def test_method_exists(self) -> None:
        cb = HeadroomCallback()
        assert hasattr(cb, "async_post_call_success_hook"), (
            "HeadroomCallback must define async_post_call_success_hook "
            "for LiteLLM proxy compatibility"
        )

    def test_method_is_coroutine(self) -> None:
        cb = HeadroomCallback()
        assert inspect.iscoroutinefunction(cb.async_post_call_success_hook)

    @pytest.mark.asyncio
    async def test_returns_response_unchanged(self) -> None:
        cb = HeadroomCallback()
        sentinel = object()
        result = await cb.async_post_call_success_hook(
            data={},
            user_api_key_dict=None,
            response=sentinel,
        )
        assert result is sentinel


class TestHeadroomCallbackClientLifecycle:
    """Cloud client cleanup must be explicit and safe to repeat."""

    @pytest.mark.asyncio
    async def test_aclose_closes_and_clears_initialized_client(self) -> None:
        cb = HeadroomCallback(api_key="hdr_test")
        client = MagicMock()
        client.aclose = AsyncMock()
        cb._client = client

        await cb.aclose()

        client.aclose.assert_awaited_once_with()
        assert cb._client is None

        await cb.aclose()
        client.aclose.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_aclose_without_initialized_client_is_a_noop(self) -> None:
        cb = HeadroomCallback(api_key="hdr_test")

        await cb.aclose()

        assert cb._client is None


def test_direct_load_does_not_import_the_integrations_package() -> None:
    """The loader above must keep this file isolated from LangChain, the native
    extension and the integrations package; check it in a fresh interpreter."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('t', {str(Path(__file__))!r})\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "bad = [m for m in sys.modules if m in ('headroom.integrations', 'headroom._core')"
        " or m.startswith('langchain')]\n"
        "print(bad)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
