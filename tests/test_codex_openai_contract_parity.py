"""Parity test: the committed Codex<->OpenAI contract schema vs the live code.

The schema at ``tests/parity/fixtures/codex_openai_contracts/`` enshrines the
OpenAI interaction expectations the Codex usage-header fix depends on (PR #577):
the ``x-codex-*`` header family, and the WS-101 forward allow/deny rule.

Rather than validate golden instances against the schema (which would only check
the instances, and would pull in ``jsonschema`` as a new dep), this test binds
the schema to the *live code* in both directions, so drift in either the schema
or the parser/filter fails CI:

  - every ``x-codex-*`` header the schema declares is actually consumed by
    ``parse_codex_rate_limits`` (rename/removal upstream -> this test fails ->
    update schema + parser together);
  - ``_extract_codex_handshake_headers`` forwards exactly the ``x-codex-*``
    subset the schema's allow/deny ``$def`` permits, and never ``set-cookie`` /
    ``authorization`` (the security half of the contract).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from headroom.proxy.handlers.openai import _extract_codex_handshake_headers
from headroom.subscription.codex_rate_limits import parse_codex_rate_limits

_SCHEMA_PATH = (
    Path(__file__).parent
    / "parity"
    / "fixtures"
    / "codex_openai_contracts"
    / "codex-openai-interaction.schema.json"
)


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def _declared_codex_header_names(schema: dict) -> set[str]:
    """Every ``x-codex-*`` header name declared across the header ``$defs``."""
    names: set[str] = set()
    for def_name in (
        "OpenAICodexWindowHeaders",
        "OpenAICodexCreditsHeaders",
        "OpenAICodexMetaHeaders",
    ):
        props = schema["$defs"][def_name].get("properties", {})
        names.update(k for k in props if k.lower().startswith("x-codex-"))
    return names


class _FakeHeaders:
    def __init__(self, items: list[tuple[str, str]]):
        self._items = items

    def raw_items(self):
        return list(self._items)


def _fake_upstream(items: list[tuple[str, str]]):
    return SimpleNamespace(response=SimpleNamespace(headers=_FakeHeaders(items)))


# A valid wire value for every declared header (string-typed, as on the wire).
_VALID_VALUES = {
    "x-codex-primary-used-percent": "42",
    "x-codex-primary-window-minutes": "300",
    "x-codex-primary-reset-at": "1900000000",
    "x-codex-secondary-used-percent": "7",
    "x-codex-secondary-window-minutes": "10080",
    "x-codex-secondary-reset-at": "1900000000",
    "x-codex-credits-has-credits": "true",
    "x-codex-credits-unlimited": "false",
    "x-codex-credits-balance": "$5.00",
    "x-codex-limit-name": "gpt-5.2-codex-sonic",
    "x-codex-promo-message": "hello",
}


def test_schema_is_wellformed_and_has_expected_defs():
    """Guard against accidental corruption/deletion of the committed schema."""
    schema = _load_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    for required_def in (
        "OpenAICodexRateLimitHeaders",
        "WSUpstreamHandshakeResponse",
        "StreamingUpstreamResponseHeaders",
        "ClientForwardedHandshakeHeaders",
        "ClientForwardedStreamingHeaders",
        "WSClientRequestFrame",
        "WSRelayEvent",
        "HTTPFallbackRequestBody",
        "CodexRateLimitStatsOutput",
    ):
        assert required_def in schema["$defs"], f"missing $def: {required_def}"


def test_every_declared_test_value_covers_the_schema():
    """The test's fixture values must cover exactly the declared header set,
    so a header added to the schema without a value here is caught here rather
    than silently skipped by the parity assertion below."""
    declared = _declared_codex_header_names(_load_schema())
    assert set(_VALID_VALUES) == declared, (
        f"_VALID_VALUES drifted from the schema header set; "
        f"missing={declared - set(_VALID_VALUES)} extra={set(_VALID_VALUES) - declared}"
    )


def test_declared_headers_are_consumed_by_parser():
    """Every x-codex-* header the schema declares is actually parsed into the
    snapshot. Catches an upstream rename/removal or a schema/parser divergence."""
    declared = _declared_codex_header_names(_load_schema())
    headers = {name: _VALID_VALUES[name] for name in declared}
    snap = parse_codex_rate_limits(headers)
    assert snap is not None

    # Each schema group must materialize from its declared headers.
    assert snap.primary is not None, "primary window not parsed"
    assert snap.primary.used_percent == 42.0
    assert snap.primary.window_minutes == 300
    assert snap.primary.resets_at == 1900000000
    assert snap.secondary is not None, "secondary window not parsed"
    assert snap.secondary.used_percent == 7.0
    assert snap.credits is not None, "credits not parsed"
    assert snap.credits.has_credits is True
    assert snap.credits.balance == "$5.00"
    assert snap.limit_name == "gpt-5.2-codex-sonic"
    assert snap.promo_message == "hello"


def test_non_codex_headers_do_not_form_a_snapshot():
    """A response with no recognized x-codex-* headers yields no snapshot,
    matching the schema's gate (snapshot iff a window/credits/promo is present)."""
    assert parse_codex_rate_limits({"content-type": "text/event-stream"}) is None
    assert parse_codex_rate_limits({"x-codex-unknown-future-field": "1"}) is None


def test_handshake_forward_obeys_allow_deny_contract():
    """_extract_codex_handshake_headers forwards exactly the x-codex-* subset
    and never set-cookie / authorization (ClientForwardedHandshakeHeaders)."""
    declared = _declared_codex_header_names(_load_schema())
    upstream_items = [(name, _VALID_VALUES[name]) for name in sorted(declared)]
    upstream_items += [
        ("set-cookie", "session=should-not-forward"),
        ("authorization", "Bearer upstream-secret"),
        ("content-type", "application/json"),
    ]

    forwarded = _extract_codex_handshake_headers(_fake_upstream(upstream_items))
    forwarded_names = {name.lower() for name, _ in forwarded}

    assert forwarded_names == declared, (
        f"forwarded set != declared x-codex set; "
        f"missing={declared - forwarded_names} extra={forwarded_names - declared}"
    )
    assert "set-cookie" not in forwarded_names
    assert "authorization" not in forwarded_names
    assert "content-type" not in forwarded_names
