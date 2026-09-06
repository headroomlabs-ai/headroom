"""A rejected outcome must never reach the savings/cost success funnel.

An exhausted 5xx (e.g. a 529 Overloaded surfaced after retry exhaustion) is the
original case. A 4xx belongs there too: the proxy served an error, but the
provider generated and billed nothing, so compression on that turn saved
nothing. Feeding it in let rejected turns inflate the save-rate — on a real
session with 143 of 300 turns returning 429, 46.5% of the headline
`total_saved` was compression on turns Anthropic refused.

429 is counted as rate-limited rather than generically failed, because that is
the one 4xx a user acts on.

Companion to the retry-exhaustion change that returns the real upstream 5xx
instead of collapsing it to a 502.
"""

import asyncio

import pytest

from headroom.proxy.outcome import RequestOutcome, emit_request_outcome


class _Metrics:
    def __init__(self):
        self.failed = []
        self.rate_limited = []
        self.requested = []

    async def record_failed(self, provider):
        self.failed.append(provider)

    async def record_rate_limited(self, provider):
        self.rate_limited.append(provider)

    async def record_request(self, **kwargs):
        self.requested.append(kwargs)


class _Handler:
    # Deliberately exposes ONLY .metrics. If emit_request_outcome runs past the
    # >=500 guard it will AttributeError on cost_tracker/logger, failing the
    # test — which is exactly the contract we want to lock in for 5xx.
    def __init__(self):
        self.metrics = _Metrics()


def _outcome(status_code, tokens_saved=0):
    return RequestOutcome(
        request_id="req-1",
        provider="anthropic",
        model="claude-opus-4-8",
        original_tokens=tokens_saved,
        optimized_tokens=0,
        output_tokens=0,
        tokens_saved=tokens_saved,
        attempted_input_tokens=tokens_saved,
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


def test_429_recorded_as_rate_limited_and_skips_success_funnel():
    """The production case: an upstream rate limit saved nothing."""
    handler = _Handler()
    asyncio.run(emit_request_outcome(handler, _outcome(429, tokens_saved=6380)))
    assert handler.metrics.rate_limited == ["anthropic"]
    assert handler.metrics.failed == []  # a rate limit is not a generic failure
    assert handler.metrics.requested == []  # its 6,380 "saved" tokens are not savings


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_other_4xx_recorded_as_failed_and_skips_success_funnel(status):
    handler = _Handler()
    asyncio.run(emit_request_outcome(handler, _outcome(status, tokens_saved=1000)))
    assert handler.metrics.failed == ["anthropic"]
    assert handler.metrics.requested == []


def test_200_still_reaches_the_success_funnel():
    """Guard against over-correcting: a served turn must still be counted."""
    handler = _Handler()
    asyncio.run(emit_request_outcome(handler, _outcome(200, tokens_saved=1000)))
    assert handler.metrics.failed == []
    assert handler.metrics.rate_limited == []
    assert len(handler.metrics.requested) == 1
    assert handler.metrics.requested[0]["tokens_saved"] == 1000
