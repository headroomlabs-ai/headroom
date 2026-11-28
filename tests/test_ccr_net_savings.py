"""Tests for CCR retrieval drawback accounting: net savings in the durable ledger.

Covers the Phase 2 net-savings layer: retrieve events debit gross savings,
legacy (kind-less) events stay compress, and net is never clamped at zero.
"""

from __future__ import annotations

import json

from headroom import savings_ledger as L


def _events_env(monkeypatch, tmp_path):
    path = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(path))
    return path


def test_mixed_kind_aggregation_nets_out(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    assert L.record_savings_event(tokens_before=1000, tokens_after=200, model="gpt-4o", client="c")
    assert L.record_savings_event(
        tokens_before=0,
        tokens_after=0,
        kind="retrieve",
        tokens_retrieved=300,
        model="gpt-4o",
        client="c",
    )
    life = L.aggregate_savings().lifetime
    assert life["tokens_saved"] == 800
    assert life["tokens_retrieved"] == 300
    assert life["net_tokens_saved"] == 500
    assert life["calls"] == 1  # only the compress event is a "call"
    assert life["retrievals"] == 1


def test_legacy_events_without_kind_are_compress(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    legacy = {
        "v": L.SCHEMA_VERSION,
        "ts": L._utc_now().isoformat(),
        "before": 500,
        "after": 100,
        "saved": 400,
        "cost_usd": 0.01,
        "model": "m",
        "client": "c",
        "source": "mcp",
        "pid": 1,
    }
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    life = L.aggregate_savings().lifetime
    assert life["tokens_saved"] == 400
    assert life["tokens_retrieved"] == 0
    assert life["net_tokens_saved"] == 400  # no retrieve → net == gross


def test_negative_net_is_unclamped(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=900, model="m", client="c")  # saved 100
    L.record_savings_event(
        tokens_before=0,
        tokens_after=0,
        kind="retrieve",
        tokens_retrieved=500,
        model="m",
        client="c",
    )
    life = L.aggregate_savings().lifetime
    assert life["tokens_saved"] == 100
    assert life["tokens_retrieved"] == 500
    assert life["net_tokens_saved"] == -400  # honest: retrievals outweigh savings


def test_retrieve_event_zero_tokens_not_written(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    assert (
        L.record_savings_event(tokens_before=0, tokens_after=0, kind="retrieve", tokens_retrieved=0)
        is False
    )
    assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_retrieve_event_persists_kind_and_is_priced(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    assert L.record_savings_event(
        tokens_before=0,
        tokens_after=0,
        kind="retrieve",
        tokens_retrieved=1000,
        model="gpt-4o",
        client="c",
    )
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["kind"] == "retrieve"
    assert event["tokens_retrieved"] == 1000
    assert event["saved"] == 0
    assert event["cost_usd"] > 0  # retrieval drawback priced from retrieved tokens


def test_compress_event_stays_backward_compatible(monkeypatch, tmp_path):
    path = _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=200, model="gpt-4o", client="c")
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["kind"] == "compress"
    assert event["saved"] == 800
    assert event["tokens_retrieved"] == 0


def test_net_cost_usd_reflects_drawback(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=200, model="gpt-4o", client="c")
    L.record_savings_event(
        tokens_before=0,
        tokens_after=0,
        kind="retrieve",
        tokens_retrieved=400,
        model="gpt-4o",
        client="c",
    )
    life = L.aggregate_savings().lifetime
    # net_cost_usd = compression cost_usd - retrieval_cost_usd; both priced at the
    # input rate, so with saved(800) > retrieved(400) the net cost stays positive.
    assert life["retrieval_cost_usd"] > 0
    assert life["net_cost_usd"] == round(life["cost_usd"] - life["retrieval_cost_usd"], 6)


def test_by_model_bucket_carries_retrieval_fields(monkeypatch, tmp_path):
    _events_env(monkeypatch, tmp_path)
    L.record_savings_event(tokens_before=1000, tokens_after=200, model="gpt-4o", client="c")
    L.record_savings_event(
        tokens_before=0,
        tokens_after=0,
        kind="retrieve",
        tokens_retrieved=300,
        model="gpt-4o",
        client="c",
    )
    rows = {r["model"]: r for r in L.aggregate_savings().by_model}
    assert rows["gpt-4o"]["tokens_retrieved"] == 300
    assert rows["gpt-4o"]["net_tokens_saved"] == 500
