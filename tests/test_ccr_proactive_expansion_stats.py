"""Proxy CCR *proactive* expansion re-injections count as retrieval drawback.

Proactive expansion re-injects original content into the billed request body
without the model calling ``headroom_retrieve``. These tests pin the third
retrieval channel (added alongside the MCP-tool and reactive-handler paths):

* the live handler counters (``record_proactive_retrievals``),
* the durable ledger event (payload + N*overhead, matching the reactive split),
* the ``cost.py`` net fold (overhead charged once, via ``retrievals_total``).

Sync tests (no pytest-asyncio needed).
"""

from __future__ import annotations

import asyncio
import json
import threading

from fastapi.testclient import TestClient

import headroom.ccr.mcp_server as mcp
from headroom.ccr import CCR_RETRIEVAL_OVERHEAD_TOKENS
from headroom.ccr.response_handler import CCRResponseHandler
from headroom.proxy.outcome import (
    RequestOutcome,
    consume_pending_proactive_retrieval,
    emit_request_outcome,
    set_pending_proactive_retrieval,
)
from headroom.proxy.server import ProxyConfig, create_app

# ---- live handler counters -------------------------------------------------


def test_record_proactive_retrievals_bumps_live_stats():
    handler = CCRResponseHandler()
    handler.record_proactive_retrievals(3, 900)
    stats = handler.get_stats()
    assert stats["total_retrievals"] == 3
    assert stats["tokens_retrieved"] == 900


def test_record_proactive_retrievals_noop_on_nonpositive():
    handler = CCRResponseHandler()
    handler.record_proactive_retrievals(0, 0)
    handler.record_proactive_retrievals(-5, -100)
    stats = handler.get_stats()
    assert stats["total_retrievals"] == 0
    assert stats["tokens_retrieved"] == 0


def test_record_proactive_retrievals_thread_safe():
    handler = CCRResponseHandler()

    def worker():
        for _ in range(1000):
            handler.record_proactive_retrievals(1, 10)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = handler.get_stats()
    assert stats["total_retrievals"] == 8_000
    assert stats["tokens_retrieved"] == 80_000


# ---- durable + live: the extracted recorder wires both channels ------------


class _FakeProxy:
    """Minimal stand-in exposing only what the recorder touches."""

    def __init__(self, handler):
        self.ccr_response_handler = handler


def test_record_proactive_retrieval_drawback_dual_channel(monkeypatch, tmp_path):
    """Drives the REAL recorder used by handlers/anthropic.py. Live counters get
    payload ONLY (cost.py adds overhead from the count); the durable ledger gets
    payload + N*overhead -- the same split as the reactive path, so cost.py
    aggregation stays consistent across channels."""
    from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

    ledger = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(ledger))
    handler = CCRResponseHandler()

    n = 2
    payload = 3_000
    AnthropicHandlerMixin._record_proactive_retrieval_drawback(_FakeProxy(handler), n, payload)

    stats = handler.get_stats()
    assert stats["total_retrievals"] == n
    assert stats["tokens_retrieved"] == payload  # payload only in the live channel

    event = json.loads(ledger.read_text().splitlines()[0])
    assert event["kind"] == "retrieve"
    assert event["source"] == "proxy"
    assert event["tokens_retrieved"] == payload + n * CCR_RETRIEVAL_OVERHEAD_TOKENS


def test_record_proactive_retrieval_drawback_noop(monkeypatch, tmp_path):
    """No re-injection -> no live bump, no ledger event."""
    from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

    ledger = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(ledger))
    handler = CCRResponseHandler()

    AnthropicHandlerMixin._record_proactive_retrieval_drawback(_FakeProxy(handler), 0, 0)

    assert handler.get_stats()["total_retrievals"] == 0
    assert not ledger.exists()


# ---- cost.py /stats net fold (overhead charged once) -----------------------


def test_stats_net_debits_proactive_expansion(tmp_path, monkeypatch):
    """Live proactive counters flow through the same ``cost.py`` aggregation as
    the reactive handler: net drops by payload + retrievals*overhead, overhead
    counted exactly once (via ``retrievals_total``)."""
    monkeypatch.setattr(mcp, "SHARED_STATS_FILE", tmp_path / "empty.jsonl", raising=False)
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "proxy_savings.json"))

    with TestClient(
        create_app(ProxyConfig(cache_enabled=False, rate_limit_enabled=False, log_requests=False))
    ) as client:
        proxy = client.app.state.proxy
        assert proxy.ccr_response_handler is not None
        asyncio.run(
            proxy.metrics.record_request(
                provider="anthropic",
                model="claude-opus-4-6",
                input_tokens=10_000,
                output_tokens=200,
                tokens_saved=5_000,
                latency_ms=1.0,
            )
        )
        # Two proactive re-injections, 3_000 payload tokens (payload only —
        # cost.py adds overhead from the count).
        proxy.ccr_response_handler.record_proactive_retrievals(2, 3_000)
        payload = client.get("/stats").json()

    comp = payload["compression"]
    assert comp["tokens_retrieved"] == 3_000
    assert comp["retrievals_total"] == 2
    assert comp["net_tokens_saved"] == 5_000 - 3_000 - 2 * CCR_RETRIEVAL_OVERHEAD_TOKENS


# ---- deferral: booked on success, skipped on upstream failure --------------


def test_pending_proactive_retrieval_contextvar_roundtrip():
    assert consume_pending_proactive_retrieval() is None
    set_pending_proactive_retrieval((5, 1234))
    assert consume_pending_proactive_retrieval() == (5, 1234)
    # consume clears -> a second read is empty (no leak into the next request)
    assert consume_pending_proactive_retrieval() is None


class _SpyMetrics:
    def __init__(self):
        self.failed = False

    async def record_request(self, **kwargs):
        pass

    async def record_failed(self, **kwargs):
        self.failed = True


class _SpyHandler:
    """Minimal outcome sink: records proactive-drawback calls, no-ops the rest."""

    def __init__(self):
        self.metrics = _SpyMetrics()
        self.drawback_calls = []

    def _record_proactive_retrieval_drawback(self, count, payload):
        self.drawback_calls.append((count, payload))


def _outcome(status=200):
    return RequestOutcome(
        request_id="r1",
        provider="anthropic",
        model="claude-opus-4-6",
        original_tokens=10_000,
        optimized_tokens=5_000,
        output_tokens=200,
        tokens_saved=5_000,
        attempted_input_tokens=5_000,
        status_code=status,
    )


def test_emit_books_proactive_drawback_on_success():
    handler = _SpyHandler()

    async def _run():
        # set + emit + consume must share one context (the request task), so
        # asyncio.run wraps all three -- mirrors the production request task.
        set_pending_proactive_retrieval((2, 3_000))
        await emit_request_outcome(handler, _outcome(status=200))
        return consume_pending_proactive_retrieval()

    leftover = asyncio.run(_run())
    assert handler.drawback_calls == [(2, 3_000)]
    assert leftover is None  # emit consumed it within the request context


def test_emit_skips_proactive_drawback_on_upstream_failure():
    handler = _SpyHandler()

    async def _run():
        set_pending_proactive_retrieval((2, 3_000))
        await emit_request_outcome(handler, _outcome(status=529))
        return consume_pending_proactive_retrieval()

    leftover = asyncio.run(_run())
    assert handler.drawback_calls == []  # 5xx must not book a retrieval
    assert handler.metrics.failed is True
    assert leftover is None  # consumed even on failure -> no cross-request leak
