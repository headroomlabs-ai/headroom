"""OpenAI-format requests carry Anthropic's ``thinking`` config natively."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402  (must follow importorskip)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("provider", "model", "native"),
    [("anthropic", "claude-sonnet-5-5", True), ("openai", "gpt-4o", False)],
)
def test_thinking_is_a_kwarg_only_for_anthropic_family_targets(stream, provider, model, native):
    thinking = {"type": "between_tools"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": thinking,
        "vendor_field": 1,
    }

    async def call(backend: LiteLLMBackend) -> None:
        if not stream:
            await backend.send_openai_message(body, {})
            return
        async for _ in backend.stream_openai_message(body, {}):
            pass

    with patch(
        "headroom.backends.litellm.acompletion",
        new_callable=AsyncMock,
        side_effect=RuntimeError("stop before the network"),
    ) as mock_acomp:
        asyncio.run(call(LiteLLMBackend(provider=provider)))

    sent = mock_acomp.call_args.kwargs
    if native:
        assert sent["thinking"] == thinking
        assert sent["extra_body"] == {"vendor_field": 1}
    else:
        assert "thinking" not in sent
        assert sent["extra_body"] == {"thinking": thinking, "vendor_field": 1}
