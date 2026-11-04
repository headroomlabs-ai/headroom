"""Unit tests for ``headroom.proxy.tenant_key.resolve_tenant_key``.

Phase F PR-F3 — covers the two resolution paths (header / global), the structured-log requirement, and the sanitization edge
cases (empty, too-long, unicode, control chars, hyphens, mixed
ASCII).

The handler-integration tests (``request.state`` plumbing) live with
the proxy handler tests; this file pins the pure resolver contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.proxy.tenant_key import (
    DEFAULT_TENANT_KEY_HEADER,
    GLOBAL_TENANT_KEY,
    SOURCE_GLOBAL,
    SOURCE_HEADER,
    TENANT_KEY_HEADER_ENV_VAR,
    get_current_tenant_key,
    resolve_tenant_key,
    set_request_tenant_key,
)


def _request(
    headers: dict[str, str] | None = None, auth_mode: str | None = None
) -> SimpleNamespace:
    """Build a minimal Starlette-shaped Request fixture.

    SimpleNamespace + a plain dict for `headers` is sufficient because
    ``resolve_tenant_key`` only touches ``request.headers.get`` /
    ``.items`` and ``request.state.auth_mode``.
    """
    state = SimpleNamespace(auth_mode=auth_mode)
    return SimpleNamespace(headers=headers or {}, state=state)


# ── Header path ───────────────────────────────────────────────────────


def test_header_path_returns_header_source() -> None:
    """Header present + sanitized non-empty ⇒ header source."""
    req = _request(headers={"X-Headroom-Tenant-ID": "cust_abc123"})
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "cust_abc123"
    assert source == SOURCE_HEADER


def test_header_path_lowercase_header_lookup() -> None:
    """Header lookup is case-insensitive (Starlette norm)."""
    req = _request(headers={"x-headroom-tenant-id": "tenant42"})
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "tenant42"
    assert source == SOURCE_HEADER


def test_header_path_custom_header_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``HEADROOM_TENANT_KEY_HEADER`` overrides the default name."""
    monkeypatch.setenv(TENANT_KEY_HEADER_ENV_VAR, "X-Custom-Tenant")
    req = _request(headers={"X-Custom-Tenant": "company9"})
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "company9"
    assert source == SOURCE_HEADER


def test_header_path_default_name_is_x_headroom_tenant_id() -> None:
    """Default header name is documented as ``X-Headroom-Tenant-ID``."""
    assert DEFAULT_TENANT_KEY_HEADER == "X-Headroom-Tenant-ID"


def test_header_path_falls_through_when_empty_after_sanitization() -> None:
    """Pure-non-allowed header (e.g. all unicode) ⇒ fall through, NOT header source."""
    # All-emoji header sanitizes to empty string — neither header
    # so we drop to the global fallback.
    req = _request(headers={"X-Headroom-Tenant-ID": "🦀🚀"})
    tenant_key, source = resolve_tenant_key(req)
    assert source == SOURCE_GLOBAL
    assert tenant_key == GLOBAL_TENANT_KEY


# ── Hash path ─────────────────────────────────────────────────────────


def test_bearer_without_header_uses_global_namespace() -> None:
    req = _request(
        headers={"authorization": "Bearer sk-ant-api03-abc123def"},
        auth_mode="payg",
    )

    tenant_key, source = resolve_tenant_key(req)

    assert tenant_key == GLOBAL_TENANT_KEY
    assert source == SOURCE_GLOBAL


def test_auth_mode_without_header_does_not_partition_tenants() -> None:
    bearer = "Bearer same-tok-aaabbbccc"

    payg = resolve_tenant_key(_request(headers={"authorization": bearer}, auth_mode="payg"))
    oauth = resolve_tenant_key(_request(headers={"authorization": bearer}, auth_mode="oauth"))

    assert payg == (GLOBAL_TENANT_KEY, SOURCE_GLOBAL)
    assert oauth == (GLOBAL_TENANT_KEY, SOURCE_GLOBAL)


def test_global_fallback_when_no_signals() -> None:
    """No header, no auth, no key ⇒ literal ``"global"``."""
    req = _request()
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == GLOBAL_TENANT_KEY == "global"
    assert source == SOURCE_GLOBAL


def _capture_tenant_resolution_logs(monkeypatch: pytest.MonkeyPatch) -> list[SimpleNamespace]:
    records: list[SimpleNamespace] = []

    def _info(_message: str, *, extra: dict[str, object] | None = None) -> None:
        records.append(SimpleNamespace(**(extra or {})))

    monkeypatch.setattr("headroom.proxy.tenant_key.logger.info", _info)
    return records


def test_global_fallback_emits_structured_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``"global"`` source MUST log (no silent fallback)."""
    req = _request()
    matching = _capture_tenant_resolution_logs(monkeypatch)
    resolve_tenant_key(req)

    assert len(matching) == 1
    assert matching[0].source == "global"  # type: ignore[attr-defined]
    assert matching[0].tenant_key == "global"  # type: ignore[attr-defined]


def test_header_source_emits_structured_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Header path also emits the structured log."""
    req = _request(headers={"X-Headroom-Tenant-ID": "tenant1"})
    matching = _capture_tenant_resolution_logs(monkeypatch)
    resolve_tenant_key(req)
    assert len(matching) == 1
    assert matching[0].source == "header"  # type: ignore[attr-defined]


def test_bearer_without_header_emits_global_structured_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bearer auth without a tenant header still emits the global log."""
    req = _request(
        headers={"authorization": "Bearer sk-ant-api03-abc123def"},
        auth_mode="payg",
    )
    matching = _capture_tenant_resolution_logs(monkeypatch)
    resolve_tenant_key(req)
    assert len(matching) == 1
    assert matching[0].source == "global"  # type: ignore[attr-defined]


# ── Sanitization edges ────────────────────────────────────────────────


def test_sanitization_allows_alphanumeric() -> None:
    """ASCII alphanumerics survive untouched."""
    req = _request(headers={"X-Headroom-Tenant-ID": "Cust42TenantA"})
    tenant_key, _ = resolve_tenant_key(req)
    assert tenant_key == "Cust42TenantA"


def test_sanitization_allows_hyphens_and_underscores() -> None:
    """Hyphens and underscores survive (common in SaaS tenant IDs)."""
    req = _request(headers={"X-Headroom-Tenant-ID": "cust_a-b_c-9"})
    tenant_key, _ = resolve_tenant_key(req)
    assert tenant_key == "cust_a-b_c-9"


def test_sanitization_drops_control_chars() -> None:
    """Control chars (``\\x00`` / ``\\x1f``) get stripped."""
    req = _request(headers={"X-Headroom-Tenant-ID": "ten\x00ant\x1f1"})
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "tenant1"
    assert source == SOURCE_HEADER


def test_sanitization_drops_tab_and_newline() -> None:
    """Whitespace (``\\t`` / ``\\n``) gets stripped."""
    req = _request(headers={"X-Headroom-Tenant-ID": "tenant\t\n1"})
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "tenant1"
    assert source == SOURCE_HEADER


def test_sanitization_drops_punctuation() -> None:
    """`|`, `:`, `.`, `,`, `/` stripped (TOIN store-key separator safety)."""
    req = _request(headers={"X-Headroom-Tenant-ID": "ten|ant:1.0/x"})
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "tenant10x"
    assert source == SOURCE_HEADER


def test_sanitization_drops_unicode_falls_through_when_pure_unicode() -> None:
    """Pure-unicode header (CJK / emoji) sanitizes to empty ⇒ falls through."""
    req = _request(headers={"X-Headroom-Tenant-ID": "测试テスト"})
    _, source = resolve_tenant_key(req)
    # No header source — sanitized to empty, dropped to global.
    assert source == SOURCE_GLOBAL


def test_sanitization_truncates_at_64_chars() -> None:
    """Headers > 64 ASCII alphanum chars are TRUNCATED, not rejected."""
    raw = "a" * 100
    req = _request(headers={"X-Headroom-Tenant-ID": raw})
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "a" * 64
    assert len(tenant_key) == 64
    assert source == SOURCE_HEADER


def test_sanitization_empty_header_falls_through() -> None:
    """Empty header value ⇒ fall through (header treated as absent)."""
    req = _request(headers={"X-Headroom-Tenant-ID": ""})
    _, source = resolve_tenant_key(req)
    assert source == SOURCE_GLOBAL


def test_sanitization_whitespace_only_falls_through() -> None:
    """Whitespace-only header ⇒ sanitizes to empty ⇒ falls through."""
    req = _request(headers={"X-Headroom-Tenant-ID": "  \t\n  "})
    _, source = resolve_tenant_key(req)
    assert source == SOURCE_GLOBAL


def test_sanitization_mixed_unicode_and_ascii_preserves_ascii() -> None:
    """``cust_我_42`` sanitizes to ``cust__42`` (ASCII chars survive)."""
    req = _request(headers={"X-Headroom-Tenant-ID": "cust_我_42"})
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "cust__42"
    assert source == SOURCE_HEADER


# ── ContextVar surface ────────────────────────────────────────────────


def test_get_current_tenant_key_defaults_to_global() -> None:
    """Outside a request context, ``get_current_tenant_key`` ⇒ ``"global"``."""
    set_request_tenant_key(None)  # Make sure we're cleared.
    assert get_current_tenant_key() == GLOBAL_TENANT_KEY


def test_set_request_tenant_key_round_trips() -> None:
    """``set`` then ``get`` returns the value."""
    set_request_tenant_key("tenant_abc")
    try:
        assert get_current_tenant_key() == "tenant_abc"
    finally:
        set_request_tenant_key(None)


def test_set_request_tenant_key_none_resets_to_global() -> None:
    """Clearing returns to the global default."""
    set_request_tenant_key("tenant_abc")
    set_request_tenant_key(None)
    assert get_current_tenant_key() == GLOBAL_TENANT_KEY


# ── Header precedence over auth headers ───────────────────────────────


def test_header_takes_precedence_over_auth_header() -> None:
    """When both a tenant header and bearer auth are present, the tenant header wins."""
    req = _request(
        headers={
            "X-Headroom-Tenant-ID": "cust_explicit",
            "authorization": "Bearer sk-ant-api03-aaaaa",
        },
        auth_mode="payg",
    )
    tenant_key, source = resolve_tenant_key(req)
    assert tenant_key == "cust_explicit"
    assert source == SOURCE_HEADER
