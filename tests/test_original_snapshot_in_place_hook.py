"""The recorded original messages stay pre-hook when a hook mutates in place.

O1 aliases ``original_client_messages`` to the live ``messages`` list to skip a
deepcopy per request. That is only safe while nothing mutates the list before
the pipeline's own copy; ``config.hooks.pre_compress`` receives the live list,
so a hook that edits it in place (and returns the same list) used to rewrite the
recorded original too. With hooks or extensions configured the snapshot is an
independently owned copy; with neither, the alias stays (no env switch).

Observed at the handlers' own consumers: ``compute_session_id`` derives the
session from the ORIGINAL client messages after the hook has run, and the
upstream body carries the forwarded (hooked) messages.
"""

from __future__ import annotations

import copy
import json

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.hooks import CompressionHooks  # noqa: E402
from headroom.proxy.helpers import snapshot_original_messages  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

ORIGINAL = "the original user text"
HOOKED = "HOOKED IN PLACE"


class _InPlaceHooks(CompressionHooks):
    """Edits the live list in place and returns the same list."""

    def pre_compress(self, messages, ctx):
        for m in messages:
            if m.get("role") == "user":
                m["content"] = HOOKED
        return messages


def _texts(messages) -> list[str]:
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out.extend(b.get("text", "") for b in c if isinstance(b, dict))
    return out


def _config(**over) -> ProxyConfig:
    base = {
        "optimize": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
        "hooks": _InPlaceHooks(),
    }
    base.update(over)
    return ProxyConfig(**base)


def _record_session_messages(proxy) -> list:
    seen: list = []
    store = proxy.session_tracker_store
    real = store.compute_session_id

    def spy(request, model, messages, *a, **k):
        seen.append(copy.deepcopy(messages))
        return real(request, model, messages, *a, **k)

    store.compute_session_id = spy
    return seen


def test_openai_default_cache_path_keeps_the_original_pre_hook() -> None:
    forwarded: list = []
    with TestClient(create_app(_config(cache_enabled=True))) as client:
        proxy = client.app.state.proxy
        seen = _record_session_messages(proxy)

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            forwarded.append(copy.deepcopy(body))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
            )

        proxy._retry_request = _fake_retry
        r = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer sk-test"},
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": ORIGINAL}]},
        )
        assert r.status_code == 200, r.text

    assert seen, "compute_session_id was not reached"
    assert ORIGINAL in _texts(seen[-1]) and HOOKED not in _texts(seen[-1])
    assert forwarded and HOOKED in _texts(forwarded[-1]["messages"])


def test_anthropic_token_mode_keeps_the_original_pre_hook() -> None:
    forwarded: list = []
    with TestClient(create_app(_config(mode="token", cache_enabled=False))) as client:
        proxy = client.app.state.proxy
        seen = _record_session_messages(proxy)

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            forwarded.append(copy.deepcopy(body))
            return httpx.Response(
                200,
                content=json.dumps(
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 5, "output_tokens": 1},
                    }
                ).encode(),
                headers={"content-type": "application/json"},
            )

        proxy._retry_request = _fake_retry
        r = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": ORIGINAL}],
            },
        )
        assert r.status_code == 200, r.text

    assert seen, "compute_session_id was not reached"
    assert ORIGINAL in _texts(seen[-1]) and HOOKED not in _texts(seen[-1])
    assert forwarded and HOOKED in _texts(forwarded[-1]["messages"])


class _Ext:
    enabled = True


class _NoExt:
    enabled = False


def test_the_snapshot_aliases_only_when_nothing_can_mutate() -> None:
    msgs = [{"role": "user", "content": ORIGINAL}]
    assert snapshot_original_messages(msgs) is msgs
    assert snapshot_original_messages(msgs, extensions=_NoExt()) is msgs
    for kw in ({"hooks": _InPlaceHooks()}, {"extensions": _Ext()}):
        snap = snapshot_original_messages(msgs, **kw)
        assert snap == msgs and snap is not msgs and snap[0] is not msgs[0]
