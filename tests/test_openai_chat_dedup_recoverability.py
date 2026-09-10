"""OpenAI chat-completions: cross-turn dedup pointers are recoverability-gated.

The fold rewrites a repeated tool-output span to a bare ``[↑NL same as msg M]``
pointer naming Headroom's internal message index. On the STREAMING chat path
(``wrap copilot``) the CCR retrieval tool cannot be injected — the path cannot
intercept tool calls — and OpenAI-compatible clients never show the model
numbered messages, so the pointer is unresolvable: models read it as deleted
content and retry-loop. The chat handler therefore threads
``cross_turn_dedup_recoverable=_should_inject_openai_chat_ccr_tool(...)`` into
the router: streaming requests keep the repeated bytes verbatim, while the
buffered (non-streaming) path — where the retrieval tool IS injectable — keeps
folding.

These tests drive the real ``/v1/chat/completions`` handler through a TestClient
with dedup force-enabled (``HEADROOM_DEDUPE=1``) and capture the exact upstream
request body, the same evidence the proxy logs showed when the bug bit.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

_SPAN = "\n".join(f"    result_{i} = compute_overdraft(business_id={i})" for i in range(12))


def _messages() -> list[dict]:
    """Two identical multi-line tool outputs — the re-read dedup folds."""
    return [
        {"role": "user", "content": "fix the overdraft bug"},
        {"role": "assistant", "content": "cat merge.py"},
        {"role": "tool", "tool_call_id": "call_1", "content": f"$ cat merge.py\n{_SPAN}\n# end"},
        {"role": "assistant", "content": "sed -n range"},
        {"role": "tool", "tool_call_id": "call_2", "content": f"$ cat merge.py\n{_SPAN}\n# end"},
    ]


def _config() -> ProxyConfig:
    return ProxyConfig(optimize=True, cache_enabled=False, rate_limit_enabled=False)


def _post(client: TestClient, *, stream: bool):
    return client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": _messages(), "stream": stream},
        headers={"Authorization": "******"},
    )


def _sent_text(body: dict) -> str:
    """Concatenate the upstream message contents (parsed, so newlines are real)."""
    return "\n".join(str(m.get("content", "")) for m in body["messages"])


def test_streaming_chat_keeps_verbatim_bytes_no_dedup_pointer(monkeypatch):
    """The bug: a streaming chat request with a repeated span got a bare
    ``[↑NL same as msg M]`` pointer the model cannot resolve. Now the upstream
    body must carry the repeated bytes verbatim."""
    monkeypatch.setenv("HEADROOM_DEDUPE", "1")  # before create_app: router reads env at init
    captured: list[dict] = []

    async def fake_stream(url, headers, body, *args, **kwargs):
        captured.append(body)
        return StreamingResponse(iter([b"data: {}\n\n"]), media_type="text/event-stream")

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._stream_response = fake_stream
        resp = _post(client, stream=True)

    assert resp.status_code == 200, resp.text
    assert captured, "streaming upstream send was not captured"
    sent = _sent_text(captured[0])
    assert "[↑" not in sent  # no unresolvable pointer on the streaming path
    assert sent.count(_SPAN) == 2  # both copies forwarded byte-verbatim


def test_lossless_buffered_chat_also_skips_the_fold(monkeypatch):
    """Coupling lock: --lossless forces ccr_inject_tool=False (server.py), so
    the recoverability predicate is False for buffered chat too and the fold
    is skipped there as well (no retrieval tool exists to redeem anything in
    no-CCR mode). Bytes stay verbatim; the conservative direction is intended."""
    monkeypatch.setenv("HEADROOM_DEDUPE", "1")
    captured: list[dict] = []

    async def fake_retry(method, url, headers, body, *args, **kwargs):
        captured.append(body)
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        }
        return httpx.Response(200, json=payload, headers={"content-type": "application/json"})

    config = ProxyConfig(
        optimize=True, lossless=True, cache_enabled=False, rate_limit_enabled=False
    )
    app = create_app(config)
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        resp = _post(client, stream=False)

    assert resp.status_code == 200, resp.text
    assert captured, "buffered upstream send was not captured"
    sent = _sent_text(captured[0])
    assert "[↑" not in sent  # no retrieval tool in lossless mode -> no bare pointer
    assert sent.count(_SPAN) == 2  # both copies forwarded byte-verbatim


def test_buffered_chat_still_folds_repeated_tool_output(monkeypatch):
    """The recoverable counterpart: non-streaming chat can inject the CCR
    retrieval tool, so the in-context pointer stays resolvable and the
    repeated span still folds (today's behavior, unchanged)."""
    monkeypatch.setenv("HEADROOM_DEDUPE", "1")
    captured: list[dict] = []

    async def fake_retry(method, url, headers, body, *args, **kwargs):
        captured.append(body)
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        }
        return httpx.Response(200, json=payload, headers={"content-type": "application/json"})

    app = create_app(_config())
    with TestClient(app) as client:
        client.app.state.proxy._retry_request = fake_retry
        resp = _post(client, stream=False)

    assert resp.status_code == 200, resp.text
    assert captured, "buffered upstream send was not captured"
    sent = _sent_text(captured[0])
    assert "[↑" in sent  # fold still fires where the pointer resolves
    assert sent.count(_SPAN) == 1  # earliest copy stays as the in-context original
