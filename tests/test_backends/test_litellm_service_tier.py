"""Both Anthropic translation paths forward the caller's service tier unchanged."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from tests._dotenv import importorskip_no_env_leak

litellm = importorskip_no_env_leak("litellm")

from headroom.backends import litellm as backend_module  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_format", ["anthropic", "openai"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tier", ["absent", None, "default", "flex", "priority"])
async def test_anthropic_service_tier_is_forwarded_in_both_paths(
    monkeypatch, streaming, tier, wire_format
) -> None:
    backend = backend_module.LiteLLMBackend(provider="openrouter")
    body = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hello"}]}
    if tier != "absent":
        body["service_tier"] = tier
    original = deepcopy(body)

    async def chunks():
        for chunk in []:
            yield chunk

    response = (
        chunks()
        if streaming
        else litellm.ModelResponse(
            model="claude-sonnet-4-5",
            choices=[{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
    )
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(backend_module, "acompletion", completion)
    if streaming:
        method = (
            backend.stream_message if wire_format == "anthropic" else backend.stream_openai_message
        )
        events = [event async for event in method(body, {})]
        assert events
        if wire_format == "anthropic":
            assert events[-1].event_type == "message_stop"
    else:
        method = backend.send_message if wire_format == "anthropic" else backend.send_openai_message
        assert (await method(body, {})).status_code == 200
    completion.assert_awaited_once()
    kwargs = completion.call_args.kwargs
    if tier == "absent":
        assert "service_tier" not in kwargs
    else:
        assert kwargs["service_tier"] == tier
    assert "service_tier" not in kwargs.get("extra_body", {})
    assert body == original
