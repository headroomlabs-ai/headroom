"""A negative compression-savings total must survive a restart, not reload as 0."""

from __future__ import annotations

import json

import pytest

from headroom.proxy.persistent_metrics import PersistentMetricsState
from headroom.proxy.savings_tracker import SavingsTracker


def _losing_request(tracker: SavingsTracker) -> None:
    tracker.record_request(
        model="claude-sonnet-4",
        tokens_saved=100,
        input_tokens=1000,
        estimated_savings_usd={
            "compression": -0.5,
            "tool_schema": 0.0,
            "compression_list": -0.5,
            "tool_schema_list": 0.0,
            "basis": "cache_aware",
        },
    )
    tracker.flush()


def test_negative_savings_survive_reload(tmp_path) -> None:
    path = str(tmp_path / "savings.json")
    live = SavingsTracker(path=path)
    _losing_request(live)
    snap = live.snapshot()
    assert snap["lifetime"]["compression_savings_usd"] == pytest.approx(-0.5)
    assert snap["display_session"]["compression_savings_usd"] == pytest.approx(-0.5)

    reloaded = SavingsTracker(path=path).snapshot()
    assert reloaded["lifetime"]["compression_savings_usd"] == pytest.approx(-0.5)
    assert reloaded["display_session"]["compression_savings_usd"] == pytest.approx(-0.5)
    assert reloaded["history"][-1]["compression_savings_usd"] == pytest.approx(-0.5)


def test_lifetime_metrics_keep_negative_compression_savings() -> None:
    state = PersistentMetricsState()
    state.record_request(
        provider="anthropic",
        model="claude-sonnet-4",
        stack=None,
        compression_savings_usd=-0.5,
    )
    cost = state.to_dict()["cost"]
    assert cost["compression_savings_usd"] == pytest.approx(-0.5)
    assert cost["compression_savings_list_usd"] == pytest.approx(-0.5)

    reloaded = PersistentMetricsState(state.to_dict()).to_dict()["cost"]
    assert reloaded["compression_savings_usd"] == pytest.approx(-0.5)
    assert reloaded["compression_savings_list_usd"] == pytest.approx(-0.5)


def test_negative_list_total_survives_reload_and_next_request(tmp_path) -> None:
    path = str(tmp_path / "savings.json")
    live = SavingsTracker(path=path)
    _losing_request(live)

    reloaded = SavingsTracker(path=path)
    assert reloaded.snapshot()["lifetime"]["compression_savings_list_usd"] == pytest.approx(-0.5)
    reloaded.record_request(
        model="claude-sonnet-4",
        tokens_saved=100,
        input_tokens=1000,
        estimated_savings_usd={
            "compression": 0.1,
            "tool_schema": 0.0,
            "compression_list": 0.1,
            "tool_schema_list": 0.0,
            "basis": "cache_aware",
        },
    )
    lifetime = reloaded.snapshot()["lifetime"]
    assert lifetime["compression_savings_usd"] == pytest.approx(-0.4)
    assert lifetime["compression_savings_list_usd"] == pytest.approx(-0.4)


def test_loss_after_last_checkpoint_is_not_overwritten_by_history(tmp_path) -> None:
    path = tmp_path / "savings.json"
    live = SavingsTracker(path=str(path))
    live.record_request(
        model="claude-sonnet-4",
        tokens_saved=100,
        input_tokens=1000,
        estimated_savings_usd={
            "compression": 0.5,
            "tool_schema": 0.0,
            "compression_list": 0.5,
            "tool_schema_list": 0.0,
            "basis": "cache_aware",
        },
    )
    # No tokens saved and no cache reads: no history checkpoint is appended.
    live.record_request(
        model="claude-sonnet-4",
        tokens_saved=0,
        input_tokens=1000,
        estimated_savings_usd={
            "compression": -1.0,
            "tool_schema": 0.0,
            "compression_list": -1.0,
            "tool_schema_list": 0.0,
            "basis": "cache_aware",
        },
    )
    live.flush()
    saved = json.loads(path.read_text())
    assert saved["lifetime"]["compression_savings_usd"] == pytest.approx(-0.5)
    assert saved["history"][-1]["compression_savings_usd"] == pytest.approx(0.5)

    reloaded = SavingsTracker(path=str(path)).snapshot()
    assert reloaded["lifetime"]["compression_savings_usd"] == pytest.approx(-0.5)


@pytest.mark.parametrize("stale_usd", [0.0, 1.0])
def test_newer_negative_checkpoint_wins_over_stale_lifetime_dollars(tmp_path, stale_usd) -> None:
    path = tmp_path / "savings.json"
    _losing_request(SavingsTracker(path=str(path)))
    saved = json.loads(path.read_text())
    assert saved["history"][-1]["total_tokens_saved"] == 100
    assert saved["history"][-1]["compression_savings_usd"] == pytest.approx(-0.5)
    # Stale lifetime block: behind the checkpoint on tokens, dollars older/higher.
    saved["lifetime"]["tokens_saved"] = 0
    saved["lifetime"]["compression_savings_usd"] = stale_usd
    path.write_text(json.dumps(saved))

    lifetime = SavingsTracker(path=str(path)).snapshot()["lifetime"]
    assert lifetime["tokens_saved"] == 100
    assert lifetime["compression_savings_usd"] == pytest.approx(-0.5)
