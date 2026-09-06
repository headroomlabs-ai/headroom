"""Cache-aware counterfactual for compression savings (realized input rate).

`_estimate_compression_savings_usd` priced every saved token at the model's
full list input rate, while `_estimate_input_cost_usd` right beside it prices
the same request's real spend on the cache read/write/uncached split.
Compression re-applies the same removals on every warm-prefix turn (clients
resend the original transcript), so on cache-heavy traffic most saved-token
instances would have billed at the provider's cache-read discount, not list —
flat list pricing overstates the layer roughly 7x on a 94%-read mix. Saved
tokens now price at the request's realized blended input rate whenever the
breakdown is available; behavior without a breakdown is unchanged.
"""

from __future__ import annotations

import json
import types

import pytest

from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import (
    _estimate_compression_savings_usd,
    estimate_request_savings_usd,
)

LIST = 10.0 / 1_000_000
CACHE_READ = 1.0 / 1_000_000
CACHE_WRITE = 12.5 / 1_000_000


def _fake_litellm() -> types.SimpleNamespace:
    # cost_per_token succeeding makes _resolve_litellm_model return the name as-is.
    return types.SimpleNamespace(
        model_cost={
            "m": {
                "input_cost_per_token": LIST,
                "cache_read_input_token_cost": CACHE_READ,
                "cache_creation_input_token_cost": CACHE_WRITE,
            }
        },
        cost_per_token=lambda **_kw: (0.0, 0.0),
    )


@pytest.fixture(autouse=True)
def fake_litellm(monkeypatch):
    monkeypatch.setattr(st, "_get_litellm_module", _fake_litellm)


def test_warm_mix_prices_at_realized_rate():
    # 94% cache reads / 4% writes / 2% uncached — the mix compression actually
    # re-applies removals into on long Claude Code sessions.
    read, write, uncached = 940_000, 40_000, 20_000
    saved = 100_000
    blended_rate = (read * CACHE_READ + write * CACHE_WRITE + uncached * LIST) / (
        read + write + uncached
    )
    got = _estimate_compression_savings_usd(
        "m",
        saved,
        cache_read_tokens=read,
        cache_write_tokens=write,
        uncached_input_tokens=uncached,
    )
    assert got == pytest.approx(saved * blended_rate)
    # The whole point: an order of magnitude below flat list on this mix.
    assert got < saved * LIST * 0.2


def test_cold_mix_prices_at_list():
    # A request with no cache traffic keeps first-ingest removals at list.
    saved = 100_000
    got = _estimate_compression_savings_usd("m", saved, uncached_input_tokens=500_000)
    assert got == pytest.approx(saved * LIST)


def test_no_breakdown_keeps_flat_list_pricing():
    # Callers without the split (legacy checkpoints) are byte-for-byte unchanged.
    assert _estimate_compression_savings_usd("m", 100_000) == pytest.approx(100_000 * LIST)


def test_estimate_request_savings_usd_threads_split():
    read, write, uncached = 900_000, 50_000, 50_000
    blended_rate = (read * CACHE_READ + write * CACHE_WRITE + uncached * LIST) / (
        read + write + uncached
    )
    priced = estimate_request_savings_usd(
        "m",
        compression_tokens_saved=10_000,
        cache_read_tokens=read,
        cache_write_tokens=write,
        uncached_input_tokens=uncached,
    )
    assert priced["compression"] == pytest.approx(10_000 * blended_rate)
    # Without the split the compression key keeps its historical list pricing.
    flat = estimate_request_savings_usd("m", compression_tokens_saved=10_000)
    assert flat["compression"] == pytest.approx(10_000 * LIST)


def test_record_request_fallback_uses_realized_rate(tmp_path):
    tracker = st.SavingsTracker(path=tmp_path / "savings.json")
    tracker.record_request(
        model="m",
        input_tokens=1_000_000,
        tokens_saved=100_000,
        cache_read_tokens=940_000,
        cache_write_tokens=40_000,
        uncached_input_tokens=20_000,
    )
    blended_rate = (940_000 * CACHE_READ + 40_000 * CACHE_WRITE + 20_000 * LIST) / 1_000_000
    got = tracker.snapshot()["lifetime"]["compression_savings_usd"]
    assert got == pytest.approx(100_000 * blended_rate, rel=1e-4)


def _legacy_ledger(path, *, list_priced: float = 10.0) -> None:
    """A ledger written before the pricing fix: dollars at list, no marker."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "lifetime": {
                    "requests": 100,
                    "tokens_saved": 1_000_000,
                    "compression_savings_usd": list_priced,
                    "cache_read_tokens": 900_000,
                    "cache_savings_usd": 0.5,
                    # Realized input spend: $1/M, i.e. a tenth of the $10/M list
                    # rate the saved tokens were priced at.
                    "total_input_tokens": 1_000_000,
                    "total_input_cost_usd": 1.0,
                },
                "history": [
                    {
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "total_tokens_saved": 400_000,
                        "compression_savings_usd": 4.0,
                        "total_input_tokens": 400_000,
                        "total_input_cost_usd": 0.4,
                    },
                    {
                        "timestamp": "2026-01-02T00:00:00+00:00",
                        "total_tokens_saved": 1_000_000,
                        "compression_savings_usd": list_priced,
                        "total_input_tokens": 1_000_000,
                        "total_input_cost_usd": 1.0,
                    },
                ],
                "by_model": {
                    "m": {
                        "requests": 100,
                        "tokens_saved": 1_000_000,
                        "compression_savings_usd": list_priced,
                        "total_input_tokens": 1_000_000,
                        "total_input_cost_usd": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_legacy_ledger_is_restated_on_the_realized_basis_at_load(tmp_path):
    """Without this the lifetime headline mixes list-priced and realized-rate
    dollars with no way to tell them apart, and only converges as old entries
    age out — i.e. the reported bug stays substantially unfixed on upgrade."""
    path = tmp_path / "proxy_savings.json"
    _legacy_ledger(path)

    tracker = st.SavingsTracker(path=str(path))
    snapshot = tracker.snapshot()

    assert snapshot["pricing_basis"] == "realized"
    # 1M saved tokens at the realized $1/M, not the $10/M list rate.
    assert snapshot["lifetime"]["compression_savings_usd"] == pytest.approx(1.0)
    assert snapshot["by_model"]["m"]["compression_savings_usd"] == pytest.approx(1.0)
    # Each history point restates at its own realized rate, so the cumulative
    # series stays monotonic and the derived deltas stay non-negative.
    assert [p["compression_savings_usd"] for p in snapshot["history"]] == [
        pytest.approx(0.4),
        pytest.approx(1.0),
    ]
    daily = tracker.history_response()["series"]["daily"]
    assert [d["compression_savings_usd_delta"] for d in daily] == [
        pytest.approx(0.4),
        pytest.approx(0.6),
    ]


def test_restatement_is_stamped_and_not_repeated(tmp_path):
    """The marker rides out with the next save (same deferral as the schema
    migrations beside it), so an install is restated once — and because the
    restatement recomputes from tokens it is idempotent even if it ran twice."""
    path = tmp_path / "proxy_savings.json"
    _legacy_ledger(path)

    tracker = st.SavingsTracker(path=str(path))
    tracker.record_request(model="m", input_tokens=10, tokens_saved=0)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["pricing_basis"] == "realized"
    assert persisted["lifetime"]["compression_savings_usd"] == pytest.approx(1.0)

    again = st.SavingsTracker(path=str(path)).snapshot()
    assert again["lifetime"]["compression_savings_usd"] == pytest.approx(1.0)


def test_unpriced_legacy_ledger_waits_instead_of_freezing_list_prices(tmp_path):
    """No realized rate yet (state predates input-cost tracking): leave the
    dollars alone and retry next load rather than stamping list prices as
    'realized' forever."""
    path = tmp_path / "proxy_savings.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "lifetime": {
                    "requests": 10,
                    "tokens_saved": 1_000_000,
                    "compression_savings_usd": 10.0,
                    "total_input_tokens": 0,
                    "total_input_cost_usd": 0.0,
                },
                "history": [],
            }
        ),
        encoding="utf-8",
    )

    snapshot = st.SavingsTracker(path=str(path)).snapshot()
    assert snapshot["pricing_basis"] == "legacy"
    assert snapshot["lifetime"]["compression_savings_usd"] == pytest.approx(10.0)


def test_fresh_install_is_marked_realized(tmp_path):
    snapshot = st.SavingsTracker(path=str(tmp_path / "proxy_savings.json")).snapshot()
    assert snapshot["pricing_basis"] == "realized"
