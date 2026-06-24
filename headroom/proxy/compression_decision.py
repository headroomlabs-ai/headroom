"""``CompressionDecision``: the canonical value type for "should this
request be compressed?"

This is the input-side analog of :class:`headroom.proxy.outcome.RequestOutcome`.
Pre-this-PR, four handler files computed the same conjunction inline at
five different sites with subtle drift:

* ``handlers/anthropic.py:890`` — full ``not _bypass and _license_ok``
* ``handlers/openai.py:1406`` — full ``not _bypass and _license_ok``
* ``handlers/gemini.py:327`` (``handle_gemini_generate_content``) —
  **missing** ``not _bypass``
* ``handlers/gemini.py:630`` (``handle_google_cloudcode_stream``) —
  **missing** ``not _bypass``
* ``handlers/gemini.py:860`` (``handle_gemini_count_tokens``) —
  **missing** ``not _bypass`` AND ``_license_ok``

The Gemini divergence was a real bug — explicit
``x-headroom-bypass: true`` requests were silently ignored on every
Gemini path. Consolidating the decision into one factory makes that
divergence structurally impossible.

The same factory also surfaces every constituent boolean so the
dashboard can answer "what did the decision actually see?" without
re-deriving it (the analog of ``RequestOutcome``'s observability
fields).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from headroom.proxy.helpers import _headroom_bypass_enabled


@dataclass(frozen=True)
class CompressionDecision:
    """Immutable, value-equal snapshot of the input-side decision.

    Construction policy: use :meth:`decide`. Direct construction is
    legal but unusual — tests use it; handlers always go through
    ``decide``. The constituent observability booleans (``bypass_header_set``
    etc.) MUST match the inputs ``decide`` saw; the factory enforces
    that invariant, and the dataclass being frozen means downstream
    code can't violate it.
    """

    should_compress: bool
    # When ``should_compress`` is False, this is the canonical reason
    # surfaced in logs and (later) in the RequestOutcome tags so the
    # dashboard can slice passthrough traffic by cause. One of:
    #   * ``"bypass_header"``      — user set x-headroom-bypass or
    #                                x-headroom-mode=passthrough
    #   * ``"compression_disabled"`` — operator set config.optimize=False
    #   * ``"no_messages"``         — empty / missing messages on body
    #   * ``"license_denied"``      — usage reporter said no
    # When ``should_compress`` is True, this is ``None``.
    passthrough_reason: str | None

    # ── Observability: every constituent boolean exposed ──────────
    # These let dashboards / debug logs answer "why this decision?"
    # without re-running ``decide``. Populated even when the decision
    # was "compress" — useful for spotting near-misses ("license was
    # off but bypass also wasn't set, so we compressed anyway").
    bypass_header_set: bool
    config_optimize_enabled: bool
    license_allows: bool
    has_messages: bool

    @classmethod
    def decide(
        cls,
        *,
        headers: Any,
        config: Any,
        usage_reporter: Any | None,
        messages: Sequence[Any] | None,
    ) -> CompressionDecision:
        """Compute the canonical decision for one request.

        Precedence (highest first):

          1. ``bypass_header`` — user's explicit "do not touch my bytes"
             signal, which is a contract assertion about prefix-cache
             stability. Operators MUST honour this above all else;
             ignoring it would silently break the user's cache.
          2. ``compression_disabled`` — operator-level kill switch
             (``config.optimize=False``). Honoured next so the operator
             can run the proxy in pure-observability mode.
          3. ``no_messages`` — nothing to compress; surfaced before
             license because license-denial on an empty request would
             be misleading.
          4. ``license_denied`` — commercial gating. Only meaningful
             when there's something to compress, which is why it comes
             last.

        Parameters
        ----------
        headers
            Inbound request headers. Accepts any object with a
            ``.get(key)`` method (dict, starlette Headers, MutableMapping).
            Both ``x-headroom-bypass: true`` and
            ``x-headroom-mode: passthrough`` trigger bypass — semantics
            mirrored from :func:`headroom.proxy.helpers._headroom_bypass_enabled`.
        config
            ``HeadroomConfig``-shaped object; only ``optimize: bool``
            is read.
        usage_reporter
            Commercial gate. May be ``None`` (no licensing system
            configured) — that case is equivalent to ``should_compress=True``
            (license_allows). Otherwise must have a ``.should_compress``
            attribute.
        messages
            Messages list from the request body. ``None`` and ``[]`` are
            both "no messages" — equivalent in the decision.
        """
        bypass = _headroom_bypass_enabled(headers)
        config_ok = bool(getattr(config, "optimize", False))
        license_ok = usage_reporter.should_compress if usage_reporter is not None else True
        has_msgs = bool(messages)

        # Precedence: bypass > config > no_messages > license
        if bypass:
            reason: str | None = "bypass_header"
            should = False
        elif not config_ok:
            reason = "compression_disabled"
            should = False
        elif not has_msgs:
            reason = "no_messages"
            should = False
        elif not license_ok:
            reason = "license_denied"
            should = False
        else:
            reason = None
            should = True

        return cls(
            should_compress=should,
            passthrough_reason=reason,
            bypass_header_set=bypass,
            config_optimize_enabled=config_ok,
            license_allows=license_ok,
            has_messages=has_msgs,
        )

    def apply_to_tags(self, tags: dict[str, str]) -> None:
        """Stamp the passthrough reason into a tags dict for downstream
        observability.

        Mutates ``tags`` in place. No-op when ``should_compress=True``
        (compressing requests don't carry a ``passthrough_reason`` tag —
        absence vs presence is itself the signal).

        Handler call pattern, after ``CompressionDecision.decide(...)``::

            tags = self._extract_tags(headers)
            _decision = CompressionDecision.decide(...)
            _decision.apply_to_tags(tags)
            # ... tags now carries passthrough_reason if applicable;
            # every downstream RequestOutcome(tags=tags, ...) inherits
            # it for free, which flows through emit_request_outcome()
            # → RequestLog.tags → dashboard.
        """
        if self.passthrough_reason is not None:
            tags["passthrough_reason"] = self.passthrough_reason
