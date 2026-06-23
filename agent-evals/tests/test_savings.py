"""Unit tests for agent_evals.metrics.savings.

No network, no subprocess, no API keys: ``fetch_run_savings`` is exercised against an
``httpx.MockTransport`` and the response hook against a hand-built ``httpx.Response``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_evals.metrics.savings import (
    HEADER_CACHED,
    HEADER_COMPRESSION_FAILED,
    HEADER_MODEL,
    HEADER_TOKENS_AFTER,
    HEADER_TOKENS_BEFORE,
    HEADER_TOKENS_SAVED,
    HEADER_TRANSFORMS,
    SavingsStore,
    fetch_run_savings,
    make_response_hook,
    parse_savings_headers,
)
from agent_evals.models import Pricing


def _headers(
    *,
    before: int,
    after: int,
    transforms: str | None = None,
    cached: bool = False,
    failed: bool = False,
    model: str = "claude-sonnet-4-6",
) -> dict[str, str]:
    """A realistic per-response header dict, mirroring what the proxy emits."""

    h: dict[str, str] = {
        HEADER_TOKENS_BEFORE: str(before),
        HEADER_TOKENS_AFTER: str(after),
        HEADER_TOKENS_SAVED: str(before - after),
        HEADER_MODEL: model,
    }
    if transforms is not None:
        h[HEADER_TRANSFORMS] = transforms
    if cached:
        h[HEADER_CACHED] = "true"
    if failed:
        h[HEADER_COMPRESSION_FAILED] = "true"
    return h


# --- parse_savings_headers -----------------------------------------------------------------


def test_parse_headers_full(pricing: Pricing) -> None:
    headers = _headers(
        before=1000, after=600, transforms="code_compressor,smart_crusher", cached=True
    )
    s = parse_savings_headers(headers, pricing)

    assert s is not None
    assert s.tokens_before == 1000
    assert s.tokens_after == 600
    assert s.tokens_saved == 400
    assert s.savings_percent == pytest.approx(40.0)
    assert s.ratio == pytest.approx(0.6)
    assert s.transforms == ["code_compressor", "smart_crusher"]
    assert s.cached is True
    assert s.compression_failed is False
    assert s.source == "headers"
    # cost derived from pricing fixture (input_usd_per_1m=2.0)
    assert s.cost_usd_before == pytest.approx(1000 / 1_000_000 * 2.0)
    assert s.cost_usd_after == pytest.approx(600 / 1_000_000 * 2.0)
    assert s.cost_usd_saved == pytest.approx(400 / 1_000_000 * 2.0)


def test_parse_headers_missing_token_headers_returns_none(pricing: Pricing) -> None:
    # Only the model header — no token headers => no optimization to attribute.
    assert parse_savings_headers({HEADER_MODEL: "claude-sonnet-4-6"}, pricing) is None
    # tokens-after present but tokens-before absent => still None (both required).
    assert parse_savings_headers({HEADER_TOKENS_AFTER: "100"}, pricing) is None


def test_parse_headers_mixed_case_keys(pricing: Pricing) -> None:
    headers = {
        "X-Headroom-Tokens-Before": "800",
        "X-HEADROOM-TOKENS-AFTER": "200",
        "X-Headroom-Cached": "TRUE",
    }
    s = parse_savings_headers(headers, pricing)

    assert s is not None
    assert s.tokens_before == 800
    assert s.tokens_after == 200
    assert s.tokens_saved == 600
    assert s.cached is True


def test_parse_headers_malformed_token_value_returns_none(pricing: Pricing) -> None:
    headers = {HEADER_TOKENS_BEFORE: "not-an-int", HEADER_TOKENS_AFTER: "200"}
    assert parse_savings_headers(headers, pricing) is None


def test_parse_headers_no_transforms_header_empty_list(pricing: Pricing) -> None:
    s = parse_savings_headers(_headers(before=500, after=500), pricing)
    assert s is not None
    assert s.transforms == []
    assert s.tokens_saved == 0
    assert s.savings_percent == pytest.approx(0.0)


def test_parse_headers_transforms_dedup_and_strip(pricing: Pricing) -> None:
    s = parse_savings_headers(
        _headers(before=100, after=50, transforms=" a , b , a ,, c "), pricing
    )
    assert s is not None
    assert s.transforms == ["a", "b", "c"]


# --- SavingsStore --------------------------------------------------------------------------


def test_store_aggregate_sums_three_requests(pricing: Pricing) -> None:
    store = SavingsStore()
    for before, after, tf in [
        (1000, 600, "code_compressor"),
        (2000, 1500, "smart_crusher"),
        (500, 400, "code_compressor"),
    ]:
        s = parse_savings_headers(_headers(before=before, after=after, transforms=tf), pricing)
        assert s is not None
        store.add("task-1", s)

    agg = store.aggregate("task-1", pricing)
    assert agg is not None
    assert agg.tokens_before == 3500
    assert agg.tokens_after == 2500
    assert agg.tokens_saved == 1000
    # ratio re-derived from summed counts, not averaged
    assert agg.ratio == pytest.approx(2500 / 3500)
    assert agg.savings_percent == pytest.approx(1000 / 3500 * 100.0)
    # transforms unioned, order preserved, deduped
    assert agg.transforms == ["code_compressor", "smart_crusher"]


def test_store_aggregate_unknown_task_is_none(pricing: Pricing) -> None:
    assert SavingsStore().aggregate("nope", pricing) is None


def test_store_aggregate_or_flags_and_latency(pricing: Pricing) -> None:
    store = SavingsStore()
    s1 = parse_savings_headers(_headers(before=100, after=80), pricing, added_latency_ms=5.0)
    s2 = parse_savings_headers(
        _headers(before=200, after=150, cached=True, failed=True), pricing, added_latency_ms=7.5
    )
    assert s1 is not None and s2 is not None
    store.add("t", s1)
    store.add("t", s2)

    agg = store.aggregate("t", pricing)
    assert agg is not None
    assert agg.cached is True
    assert agg.compression_failed is True
    assert agg.added_latency_ms == pytest.approx(12.5)


def test_store_get_and_task_ids(pricing: Pricing) -> None:
    store = SavingsStore()
    s = parse_savings_headers(_headers(before=10, after=5), pricing)
    assert s is not None
    store.add("a", s)
    store.add("b", s)
    assert set(store.task_ids()) == {"a", "b"}
    assert len(store.get("a")) == 1
    # returned list is a snapshot copy — mutating it must not affect the store
    store.get("a").clear()
    assert len(store.get("a")) == 1


# --- response hook -------------------------------------------------------------------------


def test_response_hook_records_for_active_task(pricing: Pricing) -> None:
    store = SavingsStore()
    hook = make_response_hook(store, lambda: "task-7", pricing)

    resp = httpx.Response(
        status_code=200,
        headers=_headers(before=1000, after=400, transforms="smart_crusher", cached=True),
        request=httpx.Request("POST", "https://example.test/v1/messages"),
    )
    hook(resp)

    agg = store.aggregate("task-7", pricing)
    assert agg is not None
    assert agg.tokens_before == 1000
    assert agg.tokens_after == 400
    assert agg.transforms == ["smart_crusher"]
    assert agg.cached is True


def test_response_hook_noop_when_no_active_task(pricing: Pricing) -> None:
    store = SavingsStore()
    hook = make_response_hook(store, lambda: None, pricing)

    resp = httpx.Response(
        status_code=200,
        headers=_headers(before=1000, after=400),
        request=httpx.Request("POST", "https://example.test/v1/messages"),
    )
    hook(resp)

    assert store.task_ids() == []


def test_response_hook_noop_when_no_headroom_headers(pricing: Pricing) -> None:
    store = SavingsStore()
    hook = make_response_hook(store, lambda: "task-9", pricing)

    resp = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://example.test/v1/messages"),
    )
    hook(resp)

    assert store.get("task-9") == []


# --- fetch_run_savings ---------------------------------------------------------------------


def _stats_payload(
    *,
    cache_read_tokens: int = 12_345,
    busts_avoided: int = 7,
    tokens_preserved: int | None = 98_765,
    include_prefix_freeze: bool = True,
) -> dict:
    """A trimmed but structurally-faithful /stats payload."""

    prefix_cache: dict = {
        "by_provider": {},
        "totals": {"cache_read_tokens": cache_read_tokens, "requests": 3},
    }
    if include_prefix_freeze:
        pf: dict = {"busts_avoided": busts_avoided}
        if tokens_preserved is not None:
            pf["tokens_preserved"] = tokens_preserved
        prefix_cache["prefix_freeze"] = pf
    return {"prefix_cache": prefix_cache, "cost": {}, "compression": {}}


def _mock_client(payload: dict, *, stats_path: str = "/stats") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == stats_path
        return httpx.Response(
            200, content=json.dumps(payload), headers={"content-type": "application/json"}
        )

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://proxy.test")


def test_fetch_run_savings_full() -> None:
    with _mock_client(_stats_payload()) as client:
        run = fetch_run_savings("http://proxy.test/stats", client)
    assert run.cache_read_tokens == 12_345
    assert run.prefix_freeze_busts_avoided == 7
    assert run.prefix_freeze_tokens_preserved == 98_765


def test_fetch_run_savings_missing_tokens_preserved_is_none() -> None:
    payload = _stats_payload(tokens_preserved=None)
    with _mock_client(payload) as client:
        run = fetch_run_savings("http://proxy.test/stats", client)
    assert run.cache_read_tokens == 12_345
    assert run.prefix_freeze_busts_avoided == 7
    assert run.prefix_freeze_tokens_preserved is None


def test_fetch_run_savings_missing_prefix_freeze_block() -> None:
    payload = _stats_payload(include_prefix_freeze=False)
    with _mock_client(payload) as client:
        run = fetch_run_savings("http://proxy.test/stats", client)
    assert run.cache_read_tokens == 12_345
    assert run.prefix_freeze_busts_avoided == 0
    assert run.prefix_freeze_tokens_preserved is None


def test_fetch_run_savings_empty_payload_all_defaults() -> None:
    with _mock_client({}) as client:
        run = fetch_run_savings("http://proxy.test/stats", client)
    assert run.cache_read_tokens == 0
    assert run.prefix_freeze_busts_avoided == 0
    assert run.prefix_freeze_tokens_preserved is None


def test_fetch_run_savings_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://proxy.test"
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            fetch_run_savings("http://proxy.test/stats", client)
