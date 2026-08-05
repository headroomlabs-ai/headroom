"""An exhausted-5xx outcome (e.g. a 529 Overloaded surfaced after retry
exhaustion) must be counted as a failed request, not fed into the
savings/cost success funnel where it would inflate the save-rate.

Companion to the retry-exhaustion change that returns the real upstream 5xx
instead of collapsing it to a 502.
"""

import asyncio

from headroom.proxy.outcome import RequestOutcome, emit_request_outcome


class _Metrics:
    def __init__(self):
        self.failed = []
        self.requested = []

    async def record_failed(self, provider):
        self.failed.append(provider)

    async def record_request(self, **kwargs):
        self.requested.append(kwargs)


class _Handler:
    # Deliberately exposes ONLY .metrics. If emit_request_outcome runs past the
    # >=500 guard it will AttributeError on cost_tracker/logger, failing the
    # test — which is exactly the contract we want to lock in for 5xx.
    def __init__(self):
        self.metrics = _Metrics()


def _outcome(status_code):
    return RequestOutcome(
        request_id="req-1",
        provider="anthropic",
        model="claude-opus-4-8",
        original_tokens=0,
        optimized_tokens=0,
        output_tokens=0,
        tokens_saved=0,
        attempted_input_tokens=0,
        status_code=status_code,
    )


def test_529_recorded_as_failed_and_skips_success_funnel():
    handler = _Handler()
    asyncio.run(emit_request_outcome(handler, _outcome(529)))
    assert handler.metrics.failed == ["anthropic"]  # counted as failed
    assert handler.metrics.requested == []  # NOT counted as a served request


def test_503_recorded_as_failed():
    handler = _Handler()
    asyncio.run(emit_request_outcome(handler, _outcome(503)))
    assert handler.metrics.failed == ["anthropic"]
    assert handler.metrics.requested == []
