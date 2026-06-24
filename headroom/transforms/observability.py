"""Observability protocol for compression events.

A single `CompressionObserver` interface that any transform can call
after a real compression event. Concrete observers — Prometheus, OTel,
structured logs — implement this; transforms only see the protocol.

The motivating regression: `ContentRouter._record_to_toin` skipped
SmartCrusher on the assumption SmartCrusher recorded its own TOIN
events (it did when SmartCrusher was Python; it stopped when the Rust
port took over). The disconnect was invisible for three weeks because
no metric distinguished compression events by strategy. This module
exists so the next regression of that shape alerts on day 1: if
SmartCrusher events drop to zero in production, the Prometheus
counter shows it immediately.

Design choices, called out for posterity:

- **No fallback observer.** Callers pass `None` or pass a real
  observer. There is no "default no-op" instance — that would let a
  caller silently disable observability by forgetting to pass one,
  and we just spent a PR fixing exactly that class of bug. Be
  explicit.
- **No observer registry.** A single observer per transform instance.
  If you need multi-fanout, compose at the call site (one wrapper
  observer that forwards to N children) — but the trivial pattern
  doesn't need a registry baked in.
- **No batching.** Each compression event is one call. Volume is
  bounded by the number of routing decisions per request — small.
  Batching would only matter if observers had to round-trip to a
  remote system; production observers (Prometheus) are in-process
  counter increments, which are cheaper than the protocol dispatch.
- **Strategy as a string.** The router and crusher both already
  serialize their strategy as the enum's `.value` tag. Passing the
  string keeps observers from importing `CompressionStrategy` and
  lets non-router callers (e.g. SmartCrusher in legacy mode) emit
  the same shape without round-tripping through the enum.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CompressionObserver(Protocol):
    """Receive one notification per real compression event.

    Implementations should be cheap — this lives on the proxy hot path,
    one call per routing decision per request. A Prometheus-counter
    increment is the right order of magnitude.

    Args:
        strategy: Lowercase tag identifying the compression strategy
            that ran. Matches `CompressionStrategy.<NAME>.value` for
            ContentRouter; SmartCrusher's legacy direct-call path
            passes the literal `"smart_crusher"`.
        original_tokens: Token count of the input the strategy
            received.
        compressed_tokens: Token count of the output the strategy
            produced. Equal to `original_tokens` for passthrough;
            less when compression saved tokens.

    Implementations MUST NOT raise. If the observer needs to fail-
    over (Prometheus client misconfigured, OTel exporter offline)
    handle that internally — bubbling exceptions out of an observer
    would break the compression that just succeeded, which is the
    opposite of what observability should do. (See the audit
    in `RUST_DEV.md`: any silent regression is bad, but a noisy
    observer that breaks compression is worse.)
    """

    def record_compression(
        self,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
    ) -> None: ...
