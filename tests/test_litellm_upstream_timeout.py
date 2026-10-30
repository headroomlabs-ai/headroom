"""Every upstream call must be bounded.

There was no timeout in this backend at all. Observed 2026-08-07 under load:
four agent workers blocked on ESTABLISHED connections for 36+ minutes while
the proxy answered /readyz in 0.11s. No error, no retry, no log line -- the
caller simply stops, forever, and that is indistinguishable from slow work.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from headroom.backends.litellm import (
    DEFAULT_UPSTREAM_TIMEOUT,
    UPSTREAM_TIMEOUT_ENV,
    LiteLLMBackend,
    _upstream_timeout,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["send_message", "stream_message", "send_openai_message", "stream_openai_message"]
)
@pytest.mark.parametrize("env_value", ["42.5", "", "0", "-1", "nonsense", "None"])
async def test_every_acompletion_call_is_bounded(monkeypatch, method, env_value):
    """Every send path supplies its timeout at the SDK boundary."""
    monkeypatch.setenv(UPSTREAM_TIMEOUT_ENV, env_value)
    backend = LiteLLMBackend(provider="openrouter")
    body = {"model": "qwen3", "messages": [{"role": "user", "content": "hello"}]}

    async def empty_stream():
        for chunk in ():
            yield chunk

    response = SimpleNamespace(
        id="resp_timeout",
        created=123456,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="ok", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
    )
    streaming = method.startswith("stream")
    completion = AsyncMock(return_value=empty_stream() if streaming else response)
    monkeypatch.setattr("headroom.backends.litellm.acompletion", completion)

    result = getattr(backend, method)(body, {})
    if streaming:
        chunks = [chunk async for chunk in result]
        assert chunks
    else:
        assert (await result).status_code == 200

    completion.assert_awaited_once()
    expected = 42.5 if env_value == "42.5" else DEFAULT_UPSTREAM_TIMEOUT
    assert completion.await_args.kwargs["timeout"] == pytest.approx(expected)


def test_a_junk_env_value_cannot_disable_the_timeout(monkeypatch):
    """`0` means 'no timeout' to httpx, i.e. exactly the bug. So does junk."""
    for bad in ("", "0", "-1", "nonsense", "None"):
        monkeypatch.setenv(UPSTREAM_TIMEOUT_ENV, bad)
        assert _upstream_timeout() == DEFAULT_UPSTREAM_TIMEOUT, bad


def test_an_operator_can_still_tune_it(monkeypatch):
    monkeypatch.setenv(UPSTREAM_TIMEOUT_ENV, "42.5")
    assert _upstream_timeout() == pytest.approx(42.5)


def test_the_default_is_generous_enough_for_real_work():
    """Streaming: litellm expands a float across all httpx phases, so this is
    the max gap BETWEEN CHUNKS, not a cap on total generation. A steady long
    answer is never cut off."""
    assert 60.0 <= DEFAULT_UPSTREAM_TIMEOUT <= 1800.0
