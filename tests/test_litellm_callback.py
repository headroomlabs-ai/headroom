"""Tests for HeadroomCallback LiteLLM integration.

Regression for #1114: HeadroomCallback did not inherit CustomLogger, so any
hook LiteLLM added post-1.89.x (e.g. async_post_call_success_hook) raised
AttributeError and crashed the LiteLLM proxy.
"""

from __future__ import annotations

import asyncio

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.integrations.litellm_callback import HeadroomCallback  # noqa: E402


class TestHeadroomCallbackCustomLoggerInheritance:
    def test_instantiates_without_error(self) -> None:
        cb = HeadroomCallback()
        assert cb is not None

    def test_has_async_post_call_success_hook(self) -> None:
        """Regression: AttributeError: 'HeadroomCallback' has no attr 'async_post_call_success_hook'."""
        cb = HeadroomCallback()
        assert hasattr(cb, "async_post_call_success_hook"), (
            "async_post_call_success_hook must exist (added in litellm 1.89.x)"
        )

    def test_async_post_call_success_hook_is_callable(self) -> None:
        """LiteLLM must be able to await the hook without exception."""
        cb = HeadroomCallback()
        hook = cb.async_post_call_success_hook
        assert callable(hook)

    def test_async_post_call_success_hook_does_not_raise(self) -> None:
        """Calling the hook (no-op from CustomLogger) must not raise."""
        cb = HeadroomCallback()

        async def _run() -> None:
            await cb.async_post_call_success_hook(
                data={"model": "gpt-4o", "messages": []},
                user_api_key_dict={},
                response=None,
            )

        asyncio.run(_run())

    def test_all_current_litellm_async_hooks_present(self) -> None:
        """HeadroomCallback must expose every async hook CustomLogger defines."""
        from litellm.integrations.custom_logger import CustomLogger

        cb = HeadroomCallback()
        missing = [
            name
            for name in dir(CustomLogger)
            if name.startswith("async_") and not hasattr(cb, name)
        ]
        assert not missing, f"Missing CustomLogger hooks: {missing}"

    def test_async_pre_call_hook_still_works(self) -> None:
        """Inheritance must not break the existing compression hook."""
        cb = HeadroomCallback()
        assert hasattr(cb, "async_pre_call_hook")
        assert callable(cb.async_pre_call_hook)

    def test_total_tokens_saved_property(self) -> None:
        cb = HeadroomCallback()
        assert cb.total_tokens_saved == 0
