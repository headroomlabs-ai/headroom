"""The OpenAI chat and responses handlers check ``x-headroom-base-url`` off the event loop.

``is_safe_upstream_url`` waits on a DNS lookup. Called from an ``async def``
handler it holds the event-loop thread for that wait, so one slow-resolving
host delays every other in-flight request.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.test_grok_shared_proxy_call_sites import _run_chat, _run_responses
from tests.test_openai_codex_routing import _DummyTokenizer

_HEADERS = {
    "Authorization": "Bearer sk-test",
    "User-Agent": "codex-tui/0.146.0",
    "x-headroom-base-url": "https://gateway.example/v1",
}


@pytest.fixture
def guard_calls(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record, for each guard call, whether it ran on the event-loop thread."""
    monkeypatch.delenv("HEADROOM_ALLOWED_BASE_URLS", raising=False)
    calls: list[bool] = []

    def _recording_guard(url: str, *, scope: object = None) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            calls.append(False)
        else:
            calls.append(True)
        return True

    monkeypatch.setattr("headroom.proxy.upstream_guard.is_safe_upstream_url", _recording_guard)
    monkeypatch.setattr("headroom.proxy.handlers.openai.is_safe_upstream_url", _recording_guard)
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    return calls


@pytest.mark.parametrize("run", [_run_chat, _run_responses], ids=("chat", "responses"))
def test_the_override_is_checked_off_the_event_loop(run, guard_calls: list[bool]) -> None:  # noqa: ANN001
    outbound = run(_HEADERS, model="gpt-4o")

    assert outbound.url.startswith("https://gateway.example/v1")
    assert guard_calls == [False], "the guard ran on the event-loop thread"
