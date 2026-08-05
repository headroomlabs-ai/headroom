"""UsageReporter must only advance its delta baseline after a confirmed 200.

Snapshotting on a failed send permanently drops that window's usage from the
delta-based usage report (billing/quota under-count)."""

from __future__ import annotations

from types import SimpleNamespace

import anyio

from headroom.telemetry.reporter import UsageReporter


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def json(self) -> dict:
        return {}


def _make_reporter(post_outcome) -> UsageReporter:
    r = object.__new__(UsageReporter)
    ct = SimpleNamespace(
        _tokens_saved_by_model={"m": 100},
        _tokens_sent_by_model={"m": 400},
        _requests_by_model={"m": 5},
    )
    r._proxy = SimpleNamespace(cost_tracker=ct)
    r._last_report_time = None
    r._last_tokens_saved_by_model = {}
    r._last_tokens_sent_by_model = {}
    r._last_requests_by_model = {}
    r._license_key = "k"
    r._cloud_url = "https://cloud.example"
    r._license_info = None

    class _Client:
        async def post(self, *args, **kwargs):
            if isinstance(post_outcome, Exception):
                raise post_outcome
            return _Resp(post_outcome)

    async def _get_client():
        return _Client()

    r._get_client = _get_client
    return r


def test_baseline_advances_only_on_success():
    r = _make_reporter(200)
    anyio.run(r._report_usage)
    # Success -> baseline rebased to the current cumulative counters.
    assert r._last_tokens_saved_by_model == {"m": 100}
    assert r._last_requests_by_model == {"m": 5}


def test_baseline_not_advanced_on_non_200():
    r = _make_reporter(500)
    anyio.run(r._report_usage)
    # Failed send -> baseline untouched so the window is retried next report.
    assert r._last_tokens_saved_by_model == {}
    assert r._last_requests_by_model == {}


def test_baseline_not_advanced_on_exception():
    r = _make_reporter(RuntimeError("network down"))
    anyio.run(r._report_usage)
    assert r._last_tokens_saved_by_model == {}
    assert r._last_requests_by_model == {}
