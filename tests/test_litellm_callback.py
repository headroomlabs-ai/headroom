"""Tests for HeadroomCallback LiteLLM integration.

Regression for #1114: HeadroomCallback did not inherit CustomLogger, so any
hook LiteLLM added post-1.89.x (e.g. async_post_call_success_hook) raised
AttributeError and crashed the LiteLLM proxy.
"""

from __future__ import annotations

import asyncio

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


class TestAnthropicMessagesRouteCompression:
    """LiteLLM's proxy maps /v1/messages to the "anthropic_messages" call
    type (and its async twin), so compression must fire there too (#3503)."""

    @staticmethod
    def _anthropic_payload() -> dict:
        import json

        big = json.dumps(
            [
                {"id": i, "status": "active", "name": f"item_{i}", "value": i * 17}
                for i in range(200)
            ]
        )
        return {
            "model": "claude-sonnet-4-5",
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "What are the top items?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "list_items",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": big,
                        }
                    ],
                },
            ],
        }

    @staticmethod
    def _run(call_type: str, data: dict):
        import asyncio

        cb = HeadroomCallback()
        out = asyncio.run(
            cb.async_pre_call_hook(user_api_key_dict={}, cache=None, data=data, call_type=call_type)
        )
        return cb, out

    def test_anthropic_messages_route_compresses(self) -> None:
        cb, out = self._run("anthropic_messages", self._anthropic_payload())
        assert cb.total_tokens_saved > 0
        assert out["messages"][2]["content"][0]["type"] == "tool_result"

    def test_aanthropic_messages_route_compresses(self) -> None:
        cb, out = self._run("aanthropic_messages", self._anthropic_payload())
        assert cb.total_tokens_saved > 0
        assert out["messages"][2]["content"][0]["type"] == "tool_result"

    def test_completion_route_still_compresses(self) -> None:
        big = self._anthropic_payload()["messages"][2]["content"][0]["content"]
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What are the top items?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "list_items", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": big, "tool_call_id": "call_1"},
            ],
        }
        cb, _ = self._run("completion", payload)
        assert cb.total_tokens_saved > 0

    def test_non_chat_routes_still_skip(self) -> None:
        data = self._anthropic_payload()
        _, out = self._run("embeddings", data)
        assert out["messages"] == data["messages"]
