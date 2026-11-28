"""The OpenAI chat and responses handlers check ``x-headroom-base-url`` off the event loop.

``is_safe_upstream_url`` waits on a DNS lookup. Called from an ``async def``
handler it holds the event-loop thread for that wait, so one slow-resolving
host delays every other in-flight request.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Iterator

import pytest

from headroom.proxy.upstream_guard import is_safe_upstream_url
from tests.test_grok_shared_proxy_call_sites import _run_chat, _run_responses
from tests.test_openai_codex_routing import _DummyTokenizer

_HEADERS = {
    "Authorization": "Bearer sk-test",
    "User-Agent": "codex-tui/0.146.0",
    "x-headroom-base-url": "https://gateway.example/v1",
}


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Resolve the override publicly; fail if the blocking guard is called on the loop."""
    monkeypatch.delenv("HEADROOM_ALLOWED_BASE_URLS", raising=False)
    seen: list[str] = []
    real_getaddrinfo = socket.getaddrinfo

    def _lookup(host: object, *args: object, **kwargs: object) -> list:
        if host != "gateway.example":
            return real_getaddrinfo(host, *args, **kwargs)  # type: ignore[arg-type]
        seen.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("20.10.10.10", 0))]

    def _blocking_guard(url: str, *, scope: object = None) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return is_safe_upstream_url(url, scope=scope)  # type: ignore[arg-type]
        raise AssertionError("the blocking guard ran on the event-loop thread")

    monkeypatch.setattr(socket, "getaddrinfo", _lookup)
    monkeypatch.setattr("headroom.proxy.handlers.openai.is_safe_upstream_url", _blocking_guard)
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    yield seen


@pytest.mark.parametrize("run", [_run_chat, _run_responses], ids=("chat", "responses"))
def test_the_override_is_checked_off_the_event_loop(run, lookups: list[str]) -> None:  # noqa: ANN001
    outbound = run(_HEADERS, model="gpt-4o")

    assert outbound.url.startswith("https://gateway.example/v1")
    assert lookups == ["gateway.example"]
