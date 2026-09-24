"""A losing deployment must report the loss, and say so out loud.

Two independent ways a deployment can be net-negative, and both were silent:

1. A single request whose rewrite costs more than it removes. The priced
   buckets were floored at zero, so the request reported ``$0.00`` saved
   instead of a loss, and the loss was gone -- not merely hidden, but
   discarded before anything downstream could add it up.
2. Cumulative prompt-cache busts overtaking cumulative compression savings.
   Both numbers were recorded, in different places, and never compared.

The dashboard has long rendered "Net negative" for case 2, which is no help to
anyone running the proxy headless. Both cases now log at WARNING.
"""

from __future__ import annotations

import logging

import pytest

from headroom.proxy import savings_tracker as st
from headroom.proxy.savings_tracker import SavingsTracker


@pytest.fixture
def tracker(tmp_path) -> SavingsTracker:
    return SavingsTracker(path=str(tmp_path / "savings.json"))


def _priced(compression: float, tool_schema: float = 0.0) -> dict[str, object]:
    return {
        "compression": compression,
        "tool_schema": tool_schema,
        "compression_list": compression,
        "tool_schema_list": tool_schema,
        "basis": "cache_aware",
    }


# --------------------------------------------------------------------------
# per-request: the number itself
# --------------------------------------------------------------------------


def test_a_losing_request_is_recorded_as_a_loss(tracker: SavingsTracker) -> None:
    """The headline must be able to go below zero."""
    tracker.record_request(
        model="claude-sonnet-4",
        tokens_saved=100,
        input_tokens=1000,
        estimated_savings_usd=_priced(-0.25),
    )

    assert tracker._negative_savings_usd == pytest.approx(-0.25)
    assert tracker._negative_savings_count == 1


def test_a_winning_request_is_untouched(tracker: SavingsTracker) -> None:
    tracker.record_request(
        model="claude-sonnet-4",
        tokens_saved=100,
        input_tokens=1000,
        estimated_savings_usd=_priced(0.25),
    )

    assert tracker._negative_savings_count == 0


def test_buckets_net_against_each_other(tracker: SavingsTracker) -> None:
    """A loss in one bucket is not warned about if the request still wins."""
    tracker.record_request(
        model="claude-sonnet-4",
        tokens_saved=100,
        input_tokens=1000,
        estimated_savings_usd=_priced(-0.10, tool_schema=0.40),
    )

    assert tracker._negative_savings_count == 0


# --------------------------------------------------------------------------
# per-request: the warning
# --------------------------------------------------------------------------


def test_first_loss_warns_immediately(
    tracker: SavingsTracker, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=st.logger.name):
        tracker.record_request(
            model="claude-sonnet-4",
            tokens_saved=100,
            input_tokens=1000,
            estimated_savings_usd=_priced(-0.25),
        )

    assert "event=savings_negative" in caplog.text
    assert "model=claude-sonnet-4" in caplog.text


def test_repeat_losses_are_throttled_but_still_counted(
    tracker: SavingsTracker,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One line per interval, carrying the total -- not one line per request."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(st, "_monotonic", lambda: clock["t"])

    with caplog.at_level(logging.WARNING, logger=st.logger.name):
        for _ in range(5):
            tracker.record_request(
                model="claude-sonnet-4",
                tokens_saved=100,
                input_tokens=1000,
                estimated_savings_usd=_priced(-0.20),
            )

    warnings = [r for r in caplog.records if "event=savings_negative" in r.getMessage()]
    assert len(warnings) == 1
    # The suppressed four are still in the running total.
    assert tracker._negative_savings_count == 5
    assert tracker._negative_savings_usd == pytest.approx(-1.0)


def test_warning_resumes_after_the_interval(
    tracker: SavingsTracker,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr(st, "_monotonic", lambda: clock["t"])

    with caplog.at_level(logging.WARNING, logger=st.logger.name):
        tracker.record_request(
            model="m",
            tokens_saved=100,
            input_tokens=1000,
            estimated_savings_usd=_priced(-0.20),
        )
        clock["t"] += st.NEGATIVE_SAVINGS_WARN_INTERVAL_S + 1
        tracker.record_request(
            model="m",
            tokens_saved=100,
            input_tokens=1000,
            estimated_savings_usd=_priced(-0.20),
        )

    warnings = [r for r in caplog.records if "event=savings_negative" in r.getMessage()]
    assert len(warnings) == 2
    # The second line reports the cumulative figure, not just its own request.
    assert "negative_requests=2" in warnings[1].getMessage()


def test_the_other_buckets_keep_their_floor(tracker: SavingsTracker) -> None:
    """Output shaping and provider cache cannot go negative by construction."""
    tracker.record_request(
        model="m",
        tokens_saved=100,
        input_tokens=1000,
        estimated_savings_usd={
            **_priced(0.10),
            "output_shaping": -5.0,
            "provider_cache": -5.0,
        },
    )

    assert tracker._negative_savings_count == 0


# ---------------------------------------------------------------------------
# Removing the floor at the point of measurement is not enough on its own. The
# loss then has to survive every place it is added up, stored and read back,
# and it did not: three separate clamps downstream re-floored it, so the
# warning fired while every number an operator would actually look at still
# read zero.
# ---------------------------------------------------------------------------


def _losing_turn(tracker: SavingsTracker, usd: float = -0.50, project: str = "acme") -> None:
    tracker.record_request(
        model="claude-opus-4-6",
        provider="anthropic",
        input_tokens=20_000,
        tokens_saved=-5_000,
        project=project,
        estimated_savings_usd={
            "compression": usd,
            "tool_schema": 0.0,
            "compression_list": usd,
            "tool_schema_list": 0.0,
            "basis": "mix",
        },
    )


def test_the_loss_reaches_the_per_model_rollup(tmp_path) -> None:
    """``by_model`` is where an operator finds WHICH model is losing money.

    It floored the delta at zero, so the one breakdown that could answer that
    question reported a clean zero for a turn that lost 50 cents.
    """
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))

    _losing_turn(tracker)

    by_model = tracker._state["by_model"]
    assert [e["compression_savings_usd"] for e in by_model.values()] == [-0.5]


def test_the_loss_reaches_the_per_project_rollup(tmp_path) -> None:
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))

    _losing_turn(tracker)

    projects = tracker._state["projects"]
    assert [e["compression_savings_usd"] for e in projects.values()] == [-0.5]


def test_losses_accumulate_instead_of_resetting_each_turn(tmp_path) -> None:
    """The list-priced column floored its own running total before adding to it.

    Net effect: it only ever showed the MOST RECENT losing turn, so a
    deployment three turns into a sustained loss looked like it had lost once.
    """
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))

    for _ in range(3):
        _losing_turn(tracker)

    lifetime = tracker._state["lifetime"]
    assert lifetime["compression_savings_usd"] == pytest.approx(-1.5)
    assert lifetime["compression_savings_list_usd"] == pytest.approx(-1.5)


def test_a_loss_survives_a_restart(tmp_path) -> None:
    """The load path floored too, so a restart erased whatever was preserved."""
    path = str(tmp_path / "s.json")
    tracker = SavingsTracker(path=path)
    _losing_turn(tracker)

    reloaded = SavingsTracker(path=path)

    by_model = reloaded._state["by_model"]
    assert [e["compression_savings_usd"] for e in by_model.values()] == [-0.5]


def test_the_public_snapshot_shows_the_loss(tmp_path) -> None:
    """What /stats serves must agree with what the warning said."""
    tracker = SavingsTracker(path=str(tmp_path / "s.json"))

    _losing_turn(tracker)
    lifetime = tracker.snapshot()["lifetime"]

    assert lifetime["compression_savings_usd"] == pytest.approx(-0.5)
    assert lifetime["compression_savings_list_usd"] == pytest.approx(-0.5)
