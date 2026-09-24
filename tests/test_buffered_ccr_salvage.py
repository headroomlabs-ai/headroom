"""A successful upstream turn must never become a synthesized error (#3088).

The buffered CCR path flips a streaming turn to ``stream: false`` so retrieval
can be resolved server-side. Everything it does *after* the provider answers —
retrieval, memory tool calls, turn hooks, usage accounting, caching, SSE
resynthesis — is post-processing layered on a turn that already succeeded and
was already billed.

When one of those steps raised, the whole turn surfaced to the client as:

    event: error
    data: {"type":"error","error":{"type":"api_error", ...}}

In the reported capture the provider had returned a complete 69,351-byte answer
in 1.9s; the client received 1,841 bytes of keepalives and that error. The
answer was paid for and thrown away, and no traceback was logged, so the real
defect stayed invisible.

These tests pin the two halves of the fix: relay the upstream's own answer
rather than inventing a failure, and refuse to relay a response the client
cannot safely consume.
"""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.backends import InMemoryBackend  # noqa: E402
from headroom.cache.compression_store import (  # noqa: E402
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.tool_injection import create_ccr_tool_definition  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_event_streaming=False,
        ccr_context_tracking=False,
        image_optimize=False,
        # Commit immediately, so a failure is exercised on the committed path
        # too — the shape the report was filed against.
        buffered_ccr_grace_seconds=5.0,
    )


@pytest.fixture(autouse=True)
def _store():
    reset_compression_store()
    get_compression_store(backend=InMemoryBackend())
    try:
        yield
    finally:
        reset_compression_store()


def _marker() -> str:
    return get_compression_store().store(
        original=json.dumps({"earlier": "tool output"}),
        compressed="{}",
        original_item_count=1,
    )


def _upstream(content: list[dict], stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_upstream",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 295,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def _body() -> dict:
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 512,
        "stream": True,
        "tools": [create_ccr_tool_definition("anthropic")],
        "messages": [{"role": "user", "content": f"go (earlier output at <<ccr:{_marker()}>>)"}],
    }


def _headers() -> dict[str, str]:
    return {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}


def _run(upstream: dict, *, break_post_processing: bool):
    """Drive one buffered turn, optionally exploding after the upstream answers."""
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(200, json=upstream)

        proxy._retry_request = _fake_retry  # type: ignore[assignment]

        if break_post_processing:
            # Stand in for any of the post-upstream steps failing. The point is
            # that the provider already answered; what broke is ours.
            real = proxy._record_request_outcome

            async def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
                raise RuntimeError("post-processing exploded")

            proxy._record_request_outcome = _boom  # type: ignore[assignment]
            assert real is not None

        return client.post("/v1/messages", json=_body(), headers=_headers())


# --------------------------------------------------------------------------- #
# The reported failure
# --------------------------------------------------------------------------- #
def test_a_successful_turn_survives_post_processing_blowing_up() -> None:
    """The whole point: the client gets the answer the provider produced."""
    upstream = _upstream(
        [
            {"type": "thinking", "thinking": "reasoning", "signature": "sig-1"},
            {"type": "text", "text": "here is the answer"},
        ]
    )

    resp = _run(upstream, break_post_processing=True)

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    # The provider's content reaches the client...
    assert "here is the answer" in resp.text
    assert "message_start" in resp.text
    # ...and no invented failure does.
    assert "api_error" not in resp.text


def test_a_client_tool_call_is_salvaged_too() -> None:
    """The captured failure was a `bash` tool_use turn with no retrieve call."""
    upstream = _upstream(
        [
            {"type": "thinking", "thinking": "plan", "signature": "sig-2"},
            {
                "type": "tool_use",
                "id": "toolu_bash",
                "name": "bash",
                "input": {"command": "ls"},
            },
        ],
        stop_reason="tool_use",
    )

    resp = _run(upstream, break_post_processing=True)

    assert resp.status_code == 200, resp.text
    assert "toolu_bash" in resp.text
    assert "api_error" not in resp.text


def test_the_healthy_path_is_untouched() -> None:
    """Salvage must not change a turn that never failed."""
    upstream = _upstream([{"type": "text", "text": "ordinary answer"}])

    resp = _run(upstream, break_post_processing=False)

    assert resp.status_code == 200, resp.text
    assert "ordinary answer" in resp.text
    assert "api_error" not in resp.text


# --------------------------------------------------------------------------- #
# What must never be salvaged
# --------------------------------------------------------------------------- #
def test_an_unresolved_retrieve_call_is_not_relayed() -> None:
    """Failing closed here is deliberate and stays that way.

    The buffered path exists to resolve ``headroom_retrieve`` server-side. A
    response still carrying one is precisely the case the handler already fails
    closed on — relaying it would hand the client a tool call it is not expected
    to service and a marker nobody expanded.
    """
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        unresolved = _upstream(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_ccr",
                    "name": "headroom_retrieve",
                    "input": {"hash_key": "deadbeefcafe"},
                }
            ],
            stop_reason="tool_use",
        )
        assert proxy._can_salvage_buffered_upstream(unresolved) is False

        # An ordinary turn is salvageable, so the guard is not simply off.
        assert (
            proxy._can_salvage_buffered_upstream(_upstream([{"type": "text", "text": "hi"}]))
            is True
        )


@pytest.mark.parametrize("bad", [None, "not-a-dict", 42, []])
def test_a_non_dict_response_is_never_salvaged(bad) -> None:  # type: ignore[no-untyped-def]
    app = create_app(_config())
    with TestClient(app) as client:
        assert client.app.state.proxy._can_salvage_buffered_upstream(bad) is False
