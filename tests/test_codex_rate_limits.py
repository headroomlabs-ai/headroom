"""Unit tests for headroom.subscription.codex_rate_limits."""

from __future__ import annotations

import asyncio
import time

import headroom.subscription.codex_rate_limits as crl
from headroom.subscription.codex_rate_limits import (
    CodexRateLimitState,
    CodexRateLimitWindow,
    _build_usage_headers,
    maybe_schedule_usage_poll,
    parse_codex_rate_limits,
    parse_codex_usage_payload,
)

# A faithful GET /wham/usage body (shape captured from a live Plus account).
USAGE_PAYLOAD = {
    "plan_type": "plus",
    "rate_limit": {
        "allowed": True,
        "limit_reached": False,
        "primary_window": {
            "used_percent": 23,
            "limit_window_seconds": 18000,
            "reset_after_seconds": 12266,
            "reset_at": 1781276043,
        },
        "secondary_window": {
            "used_percent": 6,
            "limit_window_seconds": 604800,
            "reset_after_seconds": 359170,
            "reset_at": 1781622947,
        },
    },
    "additional_rate_limits": None,
    "credits": {
        "has_credits": False,
        "unlimited": False,
        "balance": "0",
    },
    "rate_limit_reached_type": None,
    "promo": None,
}

# ---------------------------------------------------------------------------
# CodexRateLimitWindow helpers
# ---------------------------------------------------------------------------


class TestCodexRateLimitWindow:
    def test_window_label_minutes(self):
        w = CodexRateLimitWindow(used_percent=10.0, window_minutes=45)
        assert w.window_label == "45m"

    def test_window_label_hours(self):
        w = CodexRateLimitWindow(used_percent=10.0, window_minutes=60)
        assert w.window_label == "1h"

    def test_window_label_hours_with_minutes(self):
        w = CodexRateLimitWindow(used_percent=10.0, window_minutes=90)
        assert w.window_label == "1h30m"

    def test_window_label_unknown(self):
        w = CodexRateLimitWindow(used_percent=10.0, window_minutes=None)
        assert w.window_label == "unknown"

    def test_seconds_until_reset_future(self):
        future = int(time.time()) + 3600
        w = CodexRateLimitWindow(used_percent=10.0, resets_at=future)
        secs = w.seconds_until_reset
        assert secs is not None
        assert 3590 <= secs <= 3600

    def test_seconds_until_reset_past(self):
        past = int(time.time()) - 100
        w = CodexRateLimitWindow(used_percent=10.0, resets_at=past)
        assert w.seconds_until_reset == 0

    def test_seconds_until_reset_none(self):
        w = CodexRateLimitWindow(used_percent=10.0, resets_at=None)
        assert w.seconds_until_reset is None

    def test_to_dict_keys(self):
        w = CodexRateLimitWindow(used_percent=42.5, window_minutes=60, resets_at=9999999)
        d = w.to_dict()
        assert set(d.keys()) == {
            "used_percent",
            "window_minutes",
            "window_label",
            "resets_at",
            "seconds_until_reset",
        }
        assert d["used_percent"] == 42.5
        assert d["window_label"] == "1h"


# ---------------------------------------------------------------------------
# parse_codex_rate_limits
# ---------------------------------------------------------------------------


class TestParseCodexRateLimits:
    def test_returns_none_for_empty_headers(self):
        assert parse_codex_rate_limits({}) is None

    def test_returns_none_for_non_codex_headers(self):
        headers = {"content-type": "application/json", "x-request-id": "abc"}
        assert parse_codex_rate_limits(headers) is None

    def test_parses_primary_window(self):
        headers = {
            "x-codex-primary-used-percent": "35.5",
            "x-codex-primary-window-minutes": "60",
            "x-codex-primary-reset-at": "1704069000",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.limit_id == "codex"
        assert snap.primary is not None
        assert snap.primary.used_percent == 35.5
        assert snap.primary.window_minutes == 60
        assert snap.primary.resets_at == 1704069000
        assert snap.secondary is None

    def test_parses_secondary_window(self):
        headers = {
            "x-codex-primary-used-percent": "10.0",
            "x-codex-secondary-used-percent": "80.0",
            "x-codex-secondary-window-minutes": "1440",
            "x-codex-secondary-reset-at": "1704100000",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.secondary is not None
        assert snap.secondary.used_percent == 80.0
        assert snap.secondary.window_minutes == 1440

    def test_parses_credits(self):
        headers = {
            "x-codex-primary-used-percent": "5.0",
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "false",
            "x-codex-credits-balance": "$12.50",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.credits is not None
        assert snap.credits.has_credits is True
        assert snap.credits.unlimited is False
        assert snap.credits.balance == "$12.50"

    def test_parses_unlimited_credits(self):
        headers = {
            "x-codex-primary-used-percent": "0.0",
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "true",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.credits is not None
        assert snap.credits.unlimited is True
        assert snap.credits.balance is None

    def test_parses_limit_name(self):
        headers = {
            "x-codex-primary-used-percent": "20.0",
            "x-codex-limit-name": "gpt-5.2-codex-sonic",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.limit_name == "gpt-5.2-codex-sonic"

    def test_parses_promo_message(self):
        headers = {
            "x-codex-primary-used-percent": "50.0",
            "x-codex-promo-message": "Try our new model!",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.promo_message == "Try our new model!"

    def test_only_credits_header_triggers_parse(self):
        headers = {
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "false",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        assert snap.primary is None
        assert snap.credits is not None

    def test_invalid_float_ignored(self):
        headers = {"x-codex-primary-used-percent": "not_a_number"}
        assert parse_codex_rate_limits(headers) is None

    def test_to_dict_structure(self):
        headers = {
            "x-codex-primary-used-percent": "42.0",
            "x-codex-primary-window-minutes": "60",
        }
        snap = parse_codex_rate_limits(headers)
        assert snap is not None
        d = snap.to_dict()
        assert "limit_id" in d
        assert "primary" in d
        assert "secondary" in d
        assert "credits" in d
        assert "captured_at" in d


# ---------------------------------------------------------------------------
# CodexRateLimitState
# ---------------------------------------------------------------------------


class TestCodexRateLimitState:
    def test_initial_state_is_none(self):
        state = CodexRateLimitState()
        assert state.latest is None
        assert state.get_stats() is None

    def test_update_from_headers_stores_snapshot(self):
        state = CodexRateLimitState()
        headers = {
            "x-codex-primary-used-percent": "55.0",
            "x-codex-primary-window-minutes": "60",
        }
        state.update_from_headers(headers)
        snap = state.latest
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 55.0

    def test_update_from_empty_headers_is_noop(self):
        state = CodexRateLimitState()
        state.update_from_headers({})
        assert state.latest is None

    def test_update_from_non_codex_headers_is_noop(self):
        state = CodexRateLimitState()
        state.update_from_headers({"content-type": "application/json"})
        assert state.latest is None

    def test_get_stats_returns_dict_when_data_present(self):
        state = CodexRateLimitState()
        state.update_from_headers({"x-codex-primary-used-percent": "10.0"})
        stats = state.get_stats()
        assert stats is not None
        assert isinstance(stats, dict)
        assert stats["limit_id"] == "codex"

    def test_update_overwrites_previous_snapshot(self):
        state = CodexRateLimitState()
        state.update_from_headers({"x-codex-primary-used-percent": "10.0"})
        state.update_from_headers({"x-codex-primary-used-percent": "90.0"})
        snap = state.latest
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 90.0


# ---------------------------------------------------------------------------
# parse_codex_usage_payload (GET /wham/usage)
# ---------------------------------------------------------------------------


class TestParseCodexUsagePayload:
    def test_parses_full_payload(self):
        snap = parse_codex_usage_payload(USAGE_PAYLOAD)
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 23.0
        assert snap.primary.window_minutes == 300  # 18000s rounded up
        assert snap.primary.resets_at == 1781276043
        assert snap.secondary is not None
        assert snap.secondary.used_percent == 6.0
        assert snap.secondary.window_minutes == 10080  # 604800s

    def test_window_minutes_rounds_up(self):
        snap = parse_codex_usage_payload(
            {"rate_limit": {"primary_window": {"used_percent": 1, "limit_window_seconds": 61}}}
        )
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.window_minutes == 2

    def test_no_credits_balance_suppressed(self):
        # has_credits False -> balance must not surface as "0".
        snap = parse_codex_usage_payload(USAGE_PAYLOAD)
        assert snap is not None
        assert snap.credits is not None
        assert snap.credits.has_credits is False
        assert snap.credits.balance is None

    def test_credits_balance_kept_when_has_credits(self):
        payload = {
            "rate_limit": {"primary_window": {"used_percent": 5}},
            "credits": {"has_credits": True, "unlimited": False, "balance": "$5.00"},
        }
        snap = parse_codex_usage_payload(payload)
        assert snap is not None
        assert snap.credits is not None
        assert snap.credits.balance == "$5.00"

    def test_promo_object_message(self):
        payload = {
            "rate_limit": {"primary_window": {"used_percent": 5}},
            "promo": {"message": "Hello"},
        }
        snap = parse_codex_usage_payload(payload)
        assert snap is not None
        assert snap.promo_message == "Hello"

    def test_returns_none_for_empty(self):
        assert parse_codex_usage_payload({}) is None
        assert parse_codex_usage_payload(None) is None
        assert parse_codex_usage_payload({"rate_limit": {}}) is None

    def test_missing_used_percent_window_skipped(self):
        snap = parse_codex_usage_payload(
            {"rate_limit": {"primary_window": {"limit_window_seconds": 60}}}
        )
        assert snap is None

    def test_update_from_usage_payload_stores(self):
        state = CodexRateLimitState()
        assert state.update_from_usage_payload(USAGE_PAYLOAD) is True
        snap = state.latest
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 23.0

    def test_update_from_usage_payload_noop_returns_false(self):
        state = CodexRateLimitState()
        assert state.update_from_usage_payload({}) is False
        assert state.latest is None


# ---------------------------------------------------------------------------
# Usage poll: header gating + throttle
# ---------------------------------------------------------------------------


class TestUsagePollGating:
    def test_build_headers_requires_account_id(self):
        assert _build_usage_headers({"authorization": "Bearer abc.def.ghi"}) is None

    def test_build_headers_requires_bearer(self):
        assert _build_usage_headers({"chatgpt-account-id": "acct"}) is None
        assert (
            _build_usage_headers({"authorization": "sk-live", "chatgpt-account-id": "acct"}) is None
        )

    def test_build_headers_happy_path(self):
        headers = _build_usage_headers(
            {
                "Authorization": "Bearer abc.def.ghi",
                "ChatGPT-Account-Id": "acct-1",
                "User-Agent": "codex_exec/0.139.0",
                "originator": "codex_exec",
            }
        )
        assert headers is not None
        assert headers["Authorization"] == "Bearer abc.def.ghi"
        assert headers["ChatGPT-Account-Id"] == "acct-1"
        assert headers["User-Agent"] == "codex_exec/0.139.0"
        assert headers["originator"] == "codex_exec"

    def test_try_begin_poll_throttles(self):
        state = CodexRateLimitState()
        assert state._try_begin_poll(60.0) is True
        # Second immediate attempt is throttled (within interval).
        assert state._try_begin_poll(60.0) is False
        state._end_poll()
        # Still throttled by time even after the in-flight flag clears.
        assert state._try_begin_poll(60.0) is False
        # A zero interval always allows once the in-flight flag is clear.
        assert state._try_begin_poll(0.0) is True

    def test_maybe_schedule_returns_false_without_loop(self):
        # No running event loop -> cannot schedule.
        assert (
            maybe_schedule_usage_poll(
                {"authorization": "Bearer a.b.c", "chatgpt-account-id": "acct"}
            )
            is False
        )

    def test_maybe_schedule_skips_non_codex(self):
        async def run():
            return maybe_schedule_usage_poll({"authorization": "Bearer a.b.c"})

        assert asyncio.run(run()) is False

    def test_maybe_schedule_creates_task_and_throttles(self, monkeypatch):
        # Replace the network fetch with a fast no-op coroutine.
        calls: list[str] = []

        async def fake_fetch(url, headers):  # noqa: ANN001
            calls.append(url)
            crl.get_codex_rate_limit_state()._end_poll()

        monkeypatch.setattr(crl, "_fetch_and_store_usage", fake_fetch)
        # Reset the singleton's throttle so this test is deterministic.
        monkeypatch.setattr(crl, "_state", None)
        monkeypatch.setattr(crl, "_state_lock", crl.Lock())

        async def run():
            req = {"authorization": "Bearer a.b.c", "chatgpt-account-id": "acct"}
            first = maybe_schedule_usage_poll(req, min_interval_s=60.0)
            second = maybe_schedule_usage_poll(req, min_interval_s=60.0)
            # Let the scheduled task run.
            await asyncio.sleep(0)
            return first, second

        first, second = asyncio.run(run())
        assert first is True
        assert second is False  # throttled
        assert calls == [crl.CODEX_USAGE_URL]
