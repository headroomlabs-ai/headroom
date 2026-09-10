"""The write timeout must be independent of the read timeout.

`write` used to inherit `request_timeout_seconds`, so pushing request bytes got
the same budget as waiting for a model to answer. That left the write phase
effectively unbounded against a dead peer: when an upstream stops draining, the
send blocks until the OS abandons retransmission (~180-220s on macOS), which is
*under* the 300s it inherited — so no timeout fired, the request hung, the retry
hung again, and a single incident blocked the client for minutes (#3259).

Separating them is what lets the existing retry logic fail over to a fresh
connection instead of stalling behind a socket whose peer is gone.
"""

from __future__ import annotations

import pytest

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import _provider_httpx_client_options


def _timeout(config: ProxyConfig):
    _http2, kwargs = _provider_httpx_client_options(config, verify=True)
    return kwargs["timeout"]


def test_write_does_not_inherit_the_read_budget() -> None:
    """The regression itself: a long read budget must not extend the send."""
    timeout = _timeout(ProxyConfig(request_timeout_seconds=300))

    assert timeout.read == 300
    assert timeout.write != 300
    assert timeout.write == ProxyConfig().write_timeout_seconds


def test_write_timeout_defaults_below_the_os_retransmit_ceiling() -> None:
    """A default above ~180s would never fire before the OS gave up anyway.

    The whole failure mode is that the inherited 300s sat above the point where
    macOS abandons retransmission, so the timeout was unreachable in practice.
    """
    assert ProxyConfig().write_timeout_seconds < 180


def test_write_timeout_default_can_carry_a_large_body() -> None:
    """The bound covers the whole upload, so it has to fit a real request.

    httpx hands a bytes body to the transport as one write, so on HTTP/1.1 the
    entire body is sent inside a single timer -- this is NOT a per-chunk budget.
    Measured against a peer draining a 16MB body at ~1MB/s, WriteTimeout fires
    at exactly the configured bound even though the peer is healthy. #3259's
    reporter sends 7-15MB bodies, so a default that cannot carry 15MB over a
    modest uplink would turn this fix into an outage for them.
    """
    budget = ProxyConfig().write_timeout_seconds
    largest_reported_body_bytes = 15 * 1024 * 1024
    slow_uplink_bytes_per_second = 125 * 1024  # ~1 Mbps

    assert largest_reported_body_bytes / slow_uplink_bytes_per_second <= budget


def test_write_timeout_is_configurable() -> None:
    timeout = _timeout(ProxyConfig(write_timeout_seconds=15))

    assert timeout.write == 15


def test_other_phases_are_unchanged() -> None:
    """Only `write` moves; connect/read/pool keep the values they always had."""
    config = ProxyConfig(
        request_timeout_seconds=300,
        connect_timeout_seconds=10,
    )

    timeout = _timeout(config)

    assert timeout.connect == 10
    assert timeout.read == 300
    assert timeout.pool == 10


@pytest.mark.parametrize("buffered_read", [600, 900])
def test_buffered_anthropic_turn_keeps_its_long_read_but_bounded_write(
    buffered_read: int,
) -> None:
    """A buffered turn waits longer for the answer, not longer to send.

    This path sets its own timeout, so it needs the split applied too —
    otherwise the one path most likely to carry a large body keeps the
    unbounded write.
    """
    from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

    class _Handler(AnthropicHandlerMixin):
        def __init__(self, config: ProxyConfig) -> None:
            self.config = config

    handler = _Handler(
        ProxyConfig(
            anthropic_buffered_request_timeout_seconds=buffered_read,
            request_timeout_seconds=300,
            write_timeout_seconds=60,
        )
    )

    timeout = handler._anthropic_buffered_request_timeout()

    assert timeout.read == buffered_read
    assert timeout.write == 60
