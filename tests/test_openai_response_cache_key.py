"""OpenAI chat response cache must key on the looked-up messages, not the
``pre_compress``-mutated ones (parity with #2124 / #327).

Companion to ``test_proxy_openai_cache_key_integration.py``: same real-cache +
upstream-call-counting idiom, applied to the ``pre_compress`` mutation hazard
instead of a missing ``cache_key_fields`` entry. ``cache.get`` runs before the
``pre_compress`` hook reassigns ``messages``; caching the response under the live
(mutated) ``messages`` would store it under a key the next lookup can't produce,
so an identical repeat would never hit. Driving the real ``SemanticCache`` and
counting upstream calls proves the actual cache hit — a get/set-argument check
cannot, because it never runs ``_compute_key``.

The drift only fires when a message-rewriting ``pre_compress`` hook is configured
(a non-default deployment extension point); OSS default hooks are no-ops.
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.hooks import CompressionHooks  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


class _MutatingHooks(CompressionHooks):
    """A deployment-provided ``pre_compress`` hook that rewrites history (the
    cross-turn dedup / memory injection / redaction the hook exists for). It
    returns a NEW list — the documented contract (``hooks.py`` "Modify and
    return") and what the handler relies on — reproducing the
    get -> mutate -> set key drift.
    """

    def pre_compress(self, messages, ctx):
        return [dict(m, content="MUTATED") for m in messages]


def _content(response: httpx.Response) -> str:
    return response.json()["choices"][0]["message"]["content"]


def test_openai_chat_cache_hits_repeat_request_despite_pre_compress_mutation() -> None:
    """An identical repeat request must be served from cache, not re-sent
    upstream, even when a ``pre_compress`` hook rewrites ``messages`` between the
    cache lookup and the cache store.
    """
    calls = {"n": 0}

    config = ProxyConfig(
        optimize=False,
        cache_enabled=True,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        hooks=_MutatingHooks(),
    )
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            calls["n"] += 1
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"resp-{calls['n']}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
            )

        proxy._retry_request = _fake_retry
        headers = {"authorization": "Bearer sk-test"}
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}

        # First request: cache miss -> upstream call 1, response cached.
        r1 = client.post("/v1/chat/completions", headers=headers, json=body)
        assert r1.status_code == 200
        assert _content(r1) == "resp-1"
        assert calls["n"] == 1

        # Identical repeat: must be served from cache under the looked-up key,
        # NOT re-sent upstream. Pre-fix the response was stored under the
        # hook-mutated key, so this lookup missed and calls climbed to 2.
        r2 = client.post("/v1/chat/completions", headers=headers, json=body)
        assert r2.status_code == 200
        assert _content(r2) == "resp-1"
        assert calls["n"] == 1
