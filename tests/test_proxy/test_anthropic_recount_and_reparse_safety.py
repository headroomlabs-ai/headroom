"""Two request-path safety nets in ``handle_anthropic_messages``.

1. #2810 — the consistency re-count runs ``count_messages`` twice. Both passes
   are CPU-bound real BPE (since #2543) and used to run directly on the event
   loop, stalling every other in-flight request on the process (~1s on a 2.3 MB
   body). They must run off the loop.
2. #2768 — the byte-faithful forwarder's verification re-parse of the original
   body is best-effort, but ``MemoryError`` is not a ``ValueError``, so on
   1M-context payloads it escaped and aborted an otherwise-fine request. The
   block must never be able to fail the request.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

MESSAGES = "/v1/messages"
MODEL = "claude-sonnet-4-6"

# Only ever present in the PRE-compression snapshot, never in the outbound body.
# Long enough to clear the handler's min-token floors.
SENTINEL = "presnapshot-sentinel " * 500


def _config(**overrides) -> ProxyConfig:
    base = {
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "mode": "token",
    }
    base.update(overrides)
    return ProxyConfig(**base)


def _upstream_200() -> MagicMock:
    payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "model": MODEL,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.content = json.dumps(payload).encode()
    resp.text = json.dumps(payload)
    resp.json.return_value = payload
    return resp


def test_consistency_recount_runs_off_the_event_loop(monkeypatch):
    """No ``count_messages`` pass over the pre-compression snapshot may run on
    the loop thread. The snapshot is identified by SENTINEL, which the pipeline
    strips, so this pins the re-count specifically: the already-offloaded count
    at request start also sees the sentinel and passes either way, while the
    two re-count passes ran inline before #2810 and would fail here.
    """
    import headroom.tokenizers as tokenizers_mod

    seen: list[bool] = []  # one entry per snapshot count: True == ran on the loop

    # Patch the class, not the cached instance, so pytest restores it for us.
    tokenizer_cls = type(tokenizers_mod.get_tokenizer(MODEL))
    real_count = tokenizer_cls.count_messages

    def counting(self, messages):  # noqa: ANN001, ANN202
        if SENTINEL in json.dumps(messages, default=str):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                seen.append(False)  # worker thread — no running loop here
            else:
                seen.append(True)  # blocking the event loop
        return real_count(self, messages)

    monkeypatch.setattr(tokenizer_cls, "count_messages", counting)

    def stripping_apply(**kwargs):  # noqa: ANN003, ANN202
        """Return genuinely-changed messages with the sentinel removed."""
        from types import SimpleNamespace

        compressed = [{**m, "content": "compressed"} for m in kwargs["messages"]]
        return SimpleNamespace(
            messages=compressed,
            transforms_applied=["test_strip"],
            timing={},
            tokens_before=100,
            tokens_after=80,
            waste_signals=None,
        )

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=stripping_apply)
        proxy._retry_request = AsyncMock(return_value=_upstream_200())
        r = client.post(
            MESSAGES,
            json={
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": SENTINEL}],
            },
        )

    assert r.status_code == 200, r.text
    assert seen, "no count_messages pass saw the snapshot; test is not exercising #2810"
    assert not any(seen), f"{sum(seen)}/{len(seen)} snapshot counts blocked the event loop"


def test_memoryerror_in_verification_reparse_does_not_abort_the_request(monkeypatch):
    """A ``MemoryError`` from the best-effort original-body re-parse must be
    swallowed (the safe fallback marks the body mutated, forcing canonical
    re-serialization) rather than escaping and killing the request.
    """
    import headroom.proxy.handlers.anthropic as anthropic_mod

    real_loads = json.loads
    raised = {"n": 0}

    def exploding_loads(s, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        # Only the verification re-parse passes the raw original body bytes.
        if isinstance(s, (bytes, bytearray)) and b"reparse-bomb" in s:
            raised["n"] += 1
            raise MemoryError("simulated re-parse spike")
        return real_loads(s, *args, **kwargs)

    monkeypatch.setattr(anthropic_mod.json, "loads", exploding_loads)

    app = create_app(_config(optimize=False))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = AsyncMock(return_value=_upstream_200())
        r = client.post(
            MESSAGES,
            json={
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "reparse-bomb"}],
            },
        )

    assert raised["n"] > 0, "the verification re-parse never ran; test is not exercising #2768"
    assert r.status_code == 200, r.text
