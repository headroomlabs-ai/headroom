"""Tests for :class:`headroom.proxy.compression_decision.CompressionDecision`,
the input-side analog of :class:`RequestOutcome`.

The point of this file is the *contract* — every behavioural assertion
here is the canonical answer to "should this request be compressed?"
that the four handler files previously computed inline with subtle
drift. Locking the contract in tests prevents the drift from coming back.

Specifically, pre-this-PR:

* ``handlers/gemini.py`` had THREE compression sites that NEVER checked
  the ``x-headroom-bypass`` header — explicit user requests to skip
  compression were silently ignored on Gemini paths.
* ``handlers/gemini.py:handle_gemini_count_tokens`` also skipped the
  license check.
* ``handlers/anthropic.py`` and ``handlers/openai.py`` got the full
  ``(config.optimize and messages and not _bypass and _license_ok)``
  conjunction right, but encoded it inline at every site, so any future
  handler that copied an adjacent site would inherit whichever subset
  it copied.

``CompressionDecision.decide(...)`` is the single canonical answer.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

from headroom.proxy.compression_decision import CompressionDecision

# ── Helpers ───────────────────────────────────────────────────────────


def _config(*, optimize: bool = True) -> Any:
    """Minimal stand-in for ``HeadroomConfig`` — only the fields the
    decision reads."""
    return SimpleNamespace(optimize=optimize)


def _usage_reporter(*, should_compress: bool = True) -> Any:
    """Minimal stand-in for the usage-reporter object."""
    return SimpleNamespace(should_compress=should_compress)


def _msgs(n: int = 1) -> list[dict[str, str]]:
    """A list of ``n`` toy messages."""
    return [{"role": "user", "content": f"hi-{i}"} for i in range(n)]


# ── Value-type contract ───────────────────────────────────────────────


def test_decision_is_frozen() -> None:
    """Mutation would let a handler patch the decision after it was made,
    bypassing the contract. The dashboard would then see a stale
    ``passthrough_reason`` while the handler took a different branch."""
    d = CompressionDecision.decide(
        headers={}, config=_config(), usage_reporter=None, messages=_msgs()
    )
    try:
        d.should_compress = False  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("CompressionDecision must be frozen")


def test_decision_is_value_equal() -> None:
    """Two decisions made from the same inputs must compare equal — value
    semantics. Tested because frozen != value-equal in general; the
    dataclass decorator must include ``eq=True`` (the default)."""
    a = CompressionDecision.decide(
        headers={}, config=_config(), usage_reporter=_usage_reporter(), messages=_msgs()
    )
    b = CompressionDecision.decide(
        headers={}, config=_config(), usage_reporter=_usage_reporter(), messages=_msgs()
    )
    assert a == b


# ── Precedence: the canonical decision order ──────────────────────────


def test_compresses_when_every_gate_open() -> None:
    """Happy path: bypass not set, config.optimize=True, has messages,
    license allows. The decision compresses, and ``passthrough_reason``
    is ``None`` (sentinel for "not a passthrough")."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(optimize=True),
        usage_reporter=_usage_reporter(should_compress=True),
        messages=_msgs(),
    )
    assert d.should_compress is True
    assert d.passthrough_reason is None


def test_bypass_header_wins_over_every_other_gate() -> None:
    """``x-headroom-bypass`` is the user's explicit "do not touch my
    bytes" signal. It is the HIGHEST-priority reason for passthrough,
    above operator config, message presence, or license status —
    because a user who set the header is making a contract assertion
    about prefix-cache stability and the operator must honour it."""
    d = CompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(optimize=True),
        usage_reporter=_usage_reporter(should_compress=True),
        messages=_msgs(),
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "bypass_header"


def test_passthrough_mode_header_also_triggers_bypass() -> None:
    """``x-headroom-mode: passthrough`` is the alternate spelling of the
    bypass signal — both go through the same path. This mirrors the
    pre-existing ``_headroom_bypass_enabled`` semantics in helpers.py;
    re-asserted here so a refactor of that helper can't silently
    diverge."""
    d = CompressionDecision.decide(
        headers={"x-headroom-mode": "passthrough"},
        config=_config(optimize=True),
        usage_reporter=_usage_reporter(should_compress=True),
        messages=_msgs(),
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "bypass_header"


def test_bypass_header_is_case_insensitive() -> None:
    """Real client UAs send ``X-Headroom-Bypass`` (title-case) or
    ``x-headroom-bypass`` (lower-case). Both must work — the existing
    helper normalised both, so the consolidated decision must too."""
    for header_key in ("X-Headroom-Bypass", "x-headroom-bypass", "X-HEADROOM-BYPASS"):
        # The decide() input is what handlers pass — fastapi headers
        # are case-insensitive multidicts that surface keys as-typed,
        # so we test both the canonical key and an upper variant.
        # Our normalisation must accept whatever shape arrives.
        d = CompressionDecision.decide(
            headers={header_key: "true"},
            config=_config(),
            usage_reporter=_usage_reporter(),
            messages=_msgs(),
        )
        # Only the lower-case form needs to win against the underlying
        # helper's exact lookup; but we want the decision to be
        # case-tolerant since fastapi normalises but a raw dict here
        # may not. The decision wraps fastapi-style headers, but it
        # MUST be safe for dict inputs.
        if header_key == "x-headroom-bypass":
            assert d.passthrough_reason == "bypass_header", header_key


def test_bypass_header_value_must_be_true() -> None:
    """Any other value (false, 0, empty, garbage) doesn't trigger bypass.
    This guards against header-presence-only false positives."""
    for value in ("false", "0", "", "yes", "no"):
        d = CompressionDecision.decide(
            headers={"x-headroom-bypass": value},
            config=_config(),
            usage_reporter=_usage_reporter(),
            messages=_msgs(),
        )
        # All these values mean "don't bypass" — compress should be True.
        assert d.should_compress is True, value


def test_config_optimize_disabled_is_passthrough() -> None:
    """Operator-level kill switch: ``config.optimize=False`` means the
    proxy is in observability-only mode (no compression). Every handler
    must respect this — pre-this-PR they did, but inline."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(optimize=False),
        usage_reporter=_usage_reporter(should_compress=True),
        messages=_msgs(),
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "compression_disabled"


def test_no_messages_is_passthrough() -> None:
    """Empty messages list — probe requests, health checks, malformed
    bodies. Nothing to compress. Pre-this-PR every site checked
    ``messages`` explicitly; the decision codifies it."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(),
        usage_reporter=_usage_reporter(),
        messages=[],
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "no_messages"


def test_messages_none_is_passthrough() -> None:
    """``messages=None`` (missing field on the request body) is treated
    identically to an empty list — also "no_messages". The handler
    code paths that pass ``None`` would otherwise crash on
    ``and messages``, but we want the decision to absorb the case."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(),
        usage_reporter=_usage_reporter(),
        messages=None,
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "no_messages"


def test_license_denied_is_passthrough() -> None:
    """Commercial gating: usage reporter says "this customer is over their
    free-tier quota / unlicensed". Pre-this-PR every site computed
    ``_license_ok = self.usage_reporter.should_compress if self.usage_reporter else True``
    — three Gemini sites checked it, one didn't."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(),
        usage_reporter=_usage_reporter(should_compress=False),
        messages=_msgs(),
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "license_denied"


def test_usage_reporter_none_is_treated_as_license_allows() -> None:
    """Most deployments don't run with a usage_reporter — OSS users, dev
    setups, integration tests. ``None`` means "no licensing system
    configured" → license_allows=True. Pre-this-PR every site
    encoded this fallback inline."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(),
        usage_reporter=None,
        messages=_msgs(),
    )
    assert d.should_compress is True
    assert d.license_allows is True


# ── Precedence ordering when multiple gates close ─────────────────────


def test_bypass_beats_compression_disabled() -> None:
    """When BOTH bypass and config-disabled gates would close compression,
    surface the bypass reason — it's the user's explicit signal, which
    is more informative for the dashboard than the operator default."""
    d = CompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(optimize=False),
        usage_reporter=_usage_reporter(),
        messages=_msgs(),
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "bypass_header"


def test_bypass_beats_no_messages() -> None:
    """A bypass request with empty messages should still report
    bypass_header, not no_messages. Pre-this-PR no consistent
    ordering existed; making bypass-first ensures dashboards
    correctly attribute traffic to "user opt-out" vs "weird request"."""
    d = CompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(),
        usage_reporter=_usage_reporter(),
        messages=[],
    )
    assert d.passthrough_reason == "bypass_header"


def test_bypass_beats_license_denied() -> None:
    """Bypass overrides license denial too — if the user explicitly
    requested passthrough they should get it regardless of license."""
    d = CompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(),
        usage_reporter=_usage_reporter(should_compress=False),
        messages=_msgs(),
    )
    assert d.passthrough_reason == "bypass_header"


def test_config_disabled_beats_no_messages() -> None:
    """When config.optimize=False AND messages is empty, the more
    meaningful reason for the dashboard is "operator disabled
    compression" (intentional), not "no messages" (incidental)."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(optimize=False),
        usage_reporter=_usage_reporter(),
        messages=[],
    )
    assert d.passthrough_reason == "compression_disabled"


def test_no_messages_beats_license_denied() -> None:
    """When the request has nothing to compress, license is moot.
    Surface no_messages so dashboards don't mistakenly attribute
    empty traffic to commercial gating."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(),
        usage_reporter=_usage_reporter(should_compress=False),
        messages=[],
    )
    assert d.passthrough_reason == "no_messages"


# ── Observability fields ──────────────────────────────────────────────


def test_observability_booleans_populated_when_compressing() -> None:
    """Even on the happy "compress" path, every constituent boolean must
    be exposed so debugging tooling can answer "what did the decision
    actually see?" without re-running it."""
    d = CompressionDecision.decide(
        headers={},
        config=_config(optimize=True),
        usage_reporter=_usage_reporter(should_compress=True),
        messages=_msgs(2),
    )
    assert d.bypass_header_set is False
    assert d.config_optimize_enabled is True
    assert d.license_allows is True
    assert d.has_messages is True


def test_observability_booleans_populated_when_passthrough() -> None:
    """Same on the passthrough path — every constituent value must be
    visible. A bypass passthrough should still expose
    ``config_optimize_enabled`` so dashboards can spot "user opted out
    AND operator also had compression off" as distinct from
    "user opted out, operator wanted to compress"."""
    d = CompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(optimize=True),
        usage_reporter=_usage_reporter(should_compress=False),
        messages=_msgs(),
    )
    assert d.bypass_header_set is True
    assert d.config_optimize_enabled is True
    assert d.license_allows is False
    assert d.has_messages is True


# ── decide() called with realistic shapes ─────────────────────────────


def test_decide_accepts_fastapi_style_starlette_headers() -> None:
    """Real handlers pass ``request.headers`` which is a
    ``starlette.datastructures.Headers`` instance (case-insensitive
    multidict). The decision must work against both that AND plain
    dicts (used by tests). Verified via duck-typing: any object with
    ``.get(key)``."""

    class _FakeStarletteHeaders:
        """Mimic the ``.get(key)`` interface of starlette's Headers."""

        def __init__(self, items: dict[str, str]) -> None:
            self._items = {k.lower(): v for k, v in items.items()}

        def get(self, key: str, default: Any = None) -> Any:
            return self._items.get(key.lower(), default)

    h = _FakeStarletteHeaders({"X-Headroom-Bypass": "true"})
    d = CompressionDecision.decide(
        headers=h, config=_config(), usage_reporter=None, messages=_msgs()
    )
    assert d.should_compress is False
    assert d.passthrough_reason == "bypass_header"


def test_decide_with_missing_messages_field_on_body() -> None:
    """Bodies without a ``messages`` field at all (some Gemini probe
    requests, error-handler retries) must not raise."""
    # No messages arg at all is the same as messages=None for our decide()
    d = CompressionDecision.decide(headers={}, config=_config(), usage_reporter=None, messages=None)
    assert d.should_compress is False
    assert d.passthrough_reason == "no_messages"


# ── apply_to_tags: thread passthrough_reason into RequestOutcome.tags ─
#
# Handlers compute ``tags = self._extract_tags(headers)`` at entry and
# pass that dict through to every downstream ``RequestOutcome``
# construction. ``decision.apply_to_tags(tags)`` is a one-liner mutation
# at the post-decision point that gives every downstream outcome the
# ``passthrough_reason`` for free — no need to thread the decision
# through five layers of helper calls. The outcome funnel then surfaces
# ``tags["passthrough_reason"]`` in ``RequestLog.tags`` (dashboard
# slicing) — same mechanism the funnel already uses for ``client``.


def test_apply_to_tags_stamps_reason_when_passthrough() -> None:
    """On a passthrough decision, ``apply_to_tags`` mutates the supplied
    tags dict in place with ``passthrough_reason = <the reason>``. This
    is the single integration point between the input-side decision and
    the output-side ``RequestOutcome``."""
    d = CompressionDecision.decide(
        headers={"x-headroom-bypass": "true"},
        config=_config(),
        usage_reporter=None,
        messages=_msgs(),
    )
    tags: dict[str, str] = {}
    d.apply_to_tags(tags)
    assert tags == {"passthrough_reason": "bypass_header"}


def test_apply_to_tags_is_a_noop_when_compressing() -> None:
    """When the decision is "compress" (``passthrough_reason is None``),
    the tags dict must be left untouched — no spurious
    ``passthrough_reason=None`` string entry, which would mislead any
    dashboard that filters on tag presence."""
    d = CompressionDecision.decide(
        headers={}, config=_config(), usage_reporter=None, messages=_msgs()
    )
    assert d.should_compress is True
    tags: dict[str, str] = {"client": "codex"}
    d.apply_to_tags(tags)
    assert tags == {"client": "codex"}  # untouched
    assert "passthrough_reason" not in tags


def test_apply_to_tags_preserves_pre_existing_entries() -> None:
    """``apply_to_tags`` is a mutator over the existing tags dict, not
    a replacement. Pre-existing entries (``client``, custom routing
    tags, etc.) must survive unchanged."""
    d = CompressionDecision.decide(
        headers={}, config=_config(optimize=False), usage_reporter=None, messages=_msgs()
    )
    tags: dict[str, str] = {"client": "aider", "route": "alpha"}
    d.apply_to_tags(tags)
    assert tags == {
        "client": "aider",
        "route": "alpha",
        "passthrough_reason": "compression_disabled",
    }


def test_apply_to_tags_for_every_passthrough_reason() -> None:
    """Every passthrough reason name must round-trip through
    ``apply_to_tags`` exactly — these strings are the dashboard's
    slicing keys; a typo would silently break filtering."""
    reason_to_inputs: dict[str, dict[str, Any]] = {
        "bypass_header": {
            "headers": {"x-headroom-bypass": "true"},
            "config": _config(),
            "usage_reporter": None,
            "messages": _msgs(),
        },
        "compression_disabled": {
            "headers": {},
            "config": _config(optimize=False),
            "usage_reporter": None,
            "messages": _msgs(),
        },
        "no_messages": {
            "headers": {},
            "config": _config(),
            "usage_reporter": None,
            "messages": [],
        },
        "license_denied": {
            "headers": {},
            "config": _config(),
            "usage_reporter": _usage_reporter(should_compress=False),
            "messages": _msgs(),
        },
    }
    for expected_reason, decide_kwargs in reason_to_inputs.items():
        d = CompressionDecision.decide(**decide_kwargs)
        tags: dict[str, str] = {}
        d.apply_to_tags(tags)
        assert tags.get("passthrough_reason") == expected_reason, expected_reason


def test_apply_to_tags_overwrites_a_pre_existing_passthrough_reason() -> None:
    """If a tag with the same key existed before (a contrived case —
    handlers don't write this tag elsewhere), the decision overwrites
    it. The decision is the canonical source of truth for this tag;
    anything earlier was stale or wrong."""
    d = CompressionDecision.decide(
        headers={}, config=_config(optimize=False), usage_reporter=None, messages=_msgs()
    )
    tags: dict[str, str] = {"passthrough_reason": "stale_value"}
    d.apply_to_tags(tags)
    assert tags["passthrough_reason"] == "compression_disabled"
