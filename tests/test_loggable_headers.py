"""Header logging is an allow-list: anything not on it is dropped, not redacted."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.helpers import capture_codex_wire_debug  # noqa: E402
from headroom.proxy.loggable_headers import (  # noqa: E402
    LOGGABLE_HEADER_NAMES,
    loggable_headers,
)
from headroom.proxy.models import ProxyConfig  # noqa: E402
from headroom.proxy.server import create_app  # noqa: E402
from headroom.proxy.wire_debug_redaction_policy import should_redact_key  # noqa: E402

# Header name -> value that must never reach a log line or a dump file.
DROPPED = {
    "Authorization": "Bearer sk-ant-FAKEBEARER-0123456789",
    "x-api-key": "sk-ant-FAKEAPIKEY-0123456789",
    "cookie": "sessionid=FAKECOOKIE",
    "x-headroom-proxy-token": "FAKEPROXYTOKEN",
    "x-internal-ops-key": "FAKEOPSKEY",
    "x-some-new-header": "secretvalue",
}
KEPT = {
    "anthropic-version": "2023-06-01",
    "x-app": "cli",
    "x-stainless-lang": "js",
    "x-headroom-user": "alice",
}
# Allow-listed names whose value the second layer (secret policy) still masks.
POLICY_MASKED_ALLOWLIST = {"x-claude-code-session-id"}


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )


def _assert_only_allow_listed(text: str, kept: dict[str, str] = KEPT) -> None:
    for name, value in DROPPED.items():
        assert name.lower() not in text.lower(), name
        assert value not in text, name
    for name, value in kept.items():
        assert name in text and value in text, name


def test_inbound_log_line_holds_only_allow_listed_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO), TestClient(create_app(_config())) as client:
        client.get("/stats", headers={**DROPPED, **KEPT})
    (line,) = [r.getMessage() for r in caplog.records if "proxy_inbound_request " in r.getMessage()]
    _assert_only_allow_listed(line)
    logged = json.loads(line.split("headers=", 1)[1])
    assert logged["auth_present"] is True
    # TestClient's own headers (host, accept, user-agent, ...) are allow-listed.
    assert logged["dropped_headers"] == len(DROPPED)


def test_counts_and_auth_flag() -> None:
    out = loggable_headers({**DROPPED, **KEPT})
    assert out["auth_present"] is True
    assert out["dropped_headers"] == len(DROPPED)
    assert loggable_headers({"host": "h"})["auth_present"] is False
    assert loggable_headers({"X-API-KEY": "k"}) == {"auth_present": True, "dropped_headers": 1}


def test_upstream_error_dump_holds_only_allow_listed_headers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom import paths

    dump_dir = tmp_path / "debug_400"
    monkeypatch.setattr(paths, "debug_400_dir", lambda: dump_dir)
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "1")
    error = httpx.Response(
        400,
        content=json.dumps({"error": {"type": "invalid_request_error", "message": "bad"}}).encode(),
        headers={"content-type": "application/json"},
    )
    body = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    with patch("headroom.proxy.server.AnyLLMBackend"):
        with TestClient(create_app(_config())) as client:
            client.app.state.proxy._retry_request = AsyncMock(return_value=error)
            client.post("/v1/messages", headers={**DROPPED, **KEPT}, json=body)

    (dump,) = dump_dir.glob("*.json")
    text = dump.read_text(encoding="utf-8")
    # The dump holds the upstream-bound headers: x-headroom-* is stripped before.
    upstream_kept = {k: v for k, v in KEPT.items() if not k.startswith("x-headroom-")}
    _assert_only_allow_listed(text, upstream_kept)
    assert json.loads(text)["headers"]["auth_present"] is True


def test_codex_wire_debug_file_holds_only_allow_listed_headers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADROOM_CODEX_WIRE_DEBUG", "1")
    monkeypatch.setenv("HEADROOM_CODEX_WIRE_DEBUG_DIR", str(tmp_path))
    path = capture_codex_wire_debug(
        "request", transport="http", direction="outbound", headers={**DROPPED, **KEPT}
    )
    assert path is not None
    _assert_only_allow_listed(path.read_text(encoding="utf-8"))


def test_codex_wire_debug_keeps_upstream_retry_and_rate_limit_headers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADROOM_CODEX_WIRE_DEBUG", "1")
    monkeypatch.setenv("HEADROOM_CODEX_WIRE_DEBUG_DIR", str(tmp_path))
    response_headers = {
        "retry-after": "7",
        "retry-after-ms": "7000",
        "x-ratelimit-remaining-tokens": "0",
        "anthropic-ratelimit-tokens-reset": "2026-11-22T10:00:00Z",
    }
    path = capture_codex_wire_debug(
        "http_stream_upstream_response_headers",
        transport="http_sse",
        direction="upstream_to_headroom",
        headers={**response_headers, "set-cookie": "FAKECOOKIE"},
        status_code=429,
    )
    assert path is not None
    headers = json.loads(path.read_text(encoding="utf-8"))["headers"]
    assert {k: headers.get(k) for k in response_headers} == response_headers
    assert "FAKECOOKIE" not in json.dumps(headers)


def test_allow_list_names_no_secret_header() -> None:
    flagged = {name for name in LOGGABLE_HEADER_NAMES if should_redact_key(name)}
    # Only names the policy masks on purpose; anything new here is a secret.
    assert flagged <= POLICY_MASKED_ALLOWLIST, flagged - POLICY_MASKED_ALLOWLIST
    assert not {"authorization", "x-api-key", "cookie", "proxy-authorization"} & (
        LOGGABLE_HEADER_NAMES
    )
