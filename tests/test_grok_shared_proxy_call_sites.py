"""Call-site regressions for Grok routing and OpenAI-header isolation.

The tests assert the outbound URL and headers together. A routing-only unit test
would miss operator credentials leaking after a Grok request selects xAI, while a
header-only test could pass if a handler simply stopped merging extras everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import anyio
import pytest

from headroom.cache.prefix_tracker import SessionTrackerStore
from tests.test_openai_codex_routing import (
    _build_request,
    _DummyOpenAIHandler,
    _DummyTokenizer,
    _jwt,
)


@pytest.fixture(autouse=True)
def _allow_client_chosen_test_upstreams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowlist the client-chosen upstreams these tests name.

    ``x-headroom-base-url`` is validated by the SSRF guard before it is honored,
    and in allowlist mode only these hosts pass. Naming them keeps the header
    policy under test here rather than DNS: ``litellm.internal`` is deliberately
    unresolvable, and the ``api.x.ai`` cases would otherwise depend on a live
    lookup. The trailing-dot form is a distinct host to the guard, which matches
    on the parsed hostname without normalizing it.

    This is the SSRF allowlist, not the designated-upstream list that governs
    which hosts may receive operator secrets, so the header assertions below
    still exercise the real policy.
    """
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "litellm.internal,api.x.ai,api.x.ai.")


class _MinimalConfig(SimpleNamespace):
    """Treat unset optional chat features as disabled."""

    def __getattr__(self, name: str) -> None:
        return None


class _ChatHandler(_DummyOpenAIHandler):
    """Add the state read by the full chat-completions path."""

    anthropic_backend: Any
    ccr_response_handler: Any

    def __init__(self) -> None:
        super().__init__()
        self.config = _MinimalConfig(**vars(self.config))
        self.cache = None
        self.session_tracker_store = cast(SimpleNamespace, SessionTrackerStore())


_GROK_HEADERS = {
    "Authorization": "Bearer xai-redacted",
    "x-xai-token-auth": "xai-grok-cli",
    "User-Agent": "grok-pager/0.2.117 grok-shell/0.2.117 (macos; aarch64)",
}
_OPENAI_CLIENT_HEADERS = {
    "Authorization": "Bearer sk-test",
    "User-Agent": "codex-tui/0.146.0",
}
_OPERATOR_EXTRAS = {
    "X-Operator-Sentinel": "must-not-leak",
    "Authorization": "Bearer operator-openai-key",
}
_SENTINEL_ONLY_EXTRAS = {"X-Operator-Sentinel": "must-not-leak"}
_SENTINEL_VALUE = "must-not-leak"
_OPERATOR_KEY = "Bearer operator-openai-key"
_CLIENT_XAI_KEY = "Bearer xai-redacted"


@pytest.fixture(autouse=True)
def _stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())


class _Outbound(NamedTuple):
    url: str
    headers: dict[str, str]


def _lower_unique(headers: dict[str, str]) -> dict[str, str]:
    """Normalize names and fail if duplicate case variants would go on wire."""
    lowered = {key.lower(): value for key, value in headers.items()}
    assert len(lowered) == len(headers), f"duplicate header names on wire: {sorted(headers)}"
    return lowered


def _outbound(captured: tuple[str, str, dict, dict] | None) -> _Outbound:
    assert captured is not None
    _method, url, headers, _body = captured
    return _Outbound(url=url, headers=_lower_unique(headers))


def _run_chat(
    request_headers: dict[str, str],
    *,
    model: str = "grok-4",
    extras: dict[str, str] = _OPERATOR_EXTRAS,
    openai_api_url: str | None = None,
) -> _Outbound:
    handler = _ChatHandler()
    handler.config.openai_extra_headers = dict(extras)
    if openai_api_url is not None:
        handler.OPENAI_API_URL = openai_api_url
    request = _build_request(
        {"model": model, "messages": [{"role": "user", "content": "hello"}]},
        request_headers,
        path="/v1/chat/completions",
    )
    anyio.run(handler.handle_openai_chat, request)
    return _outbound(handler.captured_request)


def _run_responses(
    request_headers: dict[str, str],
    *,
    model: str = "grok-4",
    extras: dict[str, str] = _OPERATOR_EXTRAS,
    openai_api_url: str | None = None,
) -> _Outbound:
    handler = _DummyOpenAIHandler()
    handler.config.openai_extra_headers = dict(extras)
    if openai_api_url is not None:
        handler.OPENAI_API_URL = openai_api_url
    request = _build_request({"model": model, "input": "hello"}, request_headers)
    anyio.run(handler.handle_openai_responses, request)
    return _outbound(handler.captured_request)


_Runner = Callable[..., _Outbound]


@pytest.mark.parametrize(
    ("run", "expected_url"),
    [
        (_run_chat, "https://api.x.ai/v1/chat/completions"),
        (_run_responses, "https://api.x.ai/v1/responses"),
    ],
    ids=("chat", "responses"),
)
def test_grok_call_sites_withhold_operator_extras_from_xai(run: _Runner, expected_url: str) -> None:
    outbound = run(_GROK_HEADERS)

    assert outbound.url == expected_url
    assert "x-operator-sentinel" not in outbound.headers
    assert outbound.headers["authorization"] == _CLIENT_XAI_KEY


@pytest.mark.parametrize(
    ("run", "expected_url"),
    [
        (_run_chat, "https://api.openai.com/v1/chat/completions"),
        (_run_responses, "https://api.openai.com/v1/responses"),
    ],
    ids=("chat", "responses"),
)
def test_ordinary_openai_call_sites_still_merge_operator_extras(
    run: _Runner, expected_url: str
) -> None:
    outbound = run(_OPENAI_CLIENT_HEADERS, model="gpt-4o-mini")

    assert outbound.url == expected_url
    assert outbound.headers["x-operator-sentinel"] == _SENTINEL_VALUE
    assert outbound.headers["authorization"] == _OPERATOR_KEY


@pytest.mark.parametrize(
    ("run", "expected_url"),
    [
        (_run_chat, "https://litellm.internal/v1/chat/completions"),
        (_run_responses, "https://litellm.internal/v1/responses"),
    ],
    ids=("chat", "responses"),
)
def test_configured_gateway_outranks_grok_and_keeps_operator_extras(
    run: _Runner, expected_url: str
) -> None:
    outbound = run(_GROK_HEADERS, openai_api_url="https://litellm.internal")

    assert outbound.url == expected_url
    assert outbound.headers["x-operator-sentinel"] == _SENTINEL_VALUE
    assert outbound.headers["authorization"] == _OPERATOR_KEY


@pytest.mark.parametrize(
    ("run", "base_url", "expected_url", "withhold"),
    [
        (
            _run_chat,
            "https://litellm.internal",
            "https://litellm.internal/v1/chat/completions",
            True,
        ),
        (
            _run_chat,
            "https://api.x.ai/v1",
            "https://api.x.ai/v1/v1/chat/completions",
            True,
        ),
        (
            _run_responses,
            "https://api.x.ai:8443",
            "https://api.x.ai:8443/v1/responses",
            True,
        ),
        (
            _run_responses,
            "https://api.x.ai.",
            "https://api.x.ai./v1/responses",
            True,
        ),
    ],
    ids=("untrusted-non-xai", "xai-subpath", "xai-port", "xai-trailing-dot"),
)
def test_custom_base_url_applies_hostname_scoped_header_policy(
    run: _Runner,
    base_url: str,
    expected_url: str,
    withhold: bool,
) -> None:
    outbound = run(
        {"Authorization": _CLIENT_XAI_KEY, "x-headroom-base-url": base_url},
    )

    assert outbound.url == expected_url
    if withhold:
        assert "x-operator-sentinel" not in outbound.headers
        assert outbound.headers["authorization"] == _CLIENT_XAI_KEY
    else:
        assert outbound.headers["x-operator-sentinel"] == _SENTINEL_VALUE
        assert outbound.headers["authorization"] == _OPERATOR_KEY


def test_configured_xai_target_withholds_ambient_openai_extras() -> None:
    outbound = _run_responses(_GROK_HEADERS, openai_api_url="https://api.x.ai")

    assert outbound.url == "https://api.x.ai/v1/responses"
    assert "x-operator-sentinel" not in outbound.headers
    assert outbound.headers["authorization"] == _CLIENT_XAI_KEY


def test_chatgpt_auth_wins_routing_while_xai_candidate_withholds_extras() -> None:
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-from-jwt"}})
    outbound = _run_responses(
        {**_GROK_HEADERS, "Authorization": f"Bearer {token}"},
        model="gpt-5.4",
        extras=_SENTINEL_ONLY_EXTRAS,
    )

    assert outbound.url == "https://chatgpt.com/backend-api/codex/responses"
    assert outbound.headers["chatgpt-account-id"] == "acct-from-jwt"
    assert "x-operator-sentinel" not in outbound.headers


class _RecordingBackend:
    name = "litellm"

    def __init__(self) -> None:
        self.captured_headers: dict[str, str] | None = None

    async def send_openai_message(self, body: dict, headers: dict) -> SimpleNamespace:
        self.captured_headers = dict(headers)
        return SimpleNamespace(
            status_code=200,
            error=None,
            body={
                "id": "chatcmpl-backend-1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )


def test_configured_chat_backend_retains_its_operator_extras() -> None:
    handler = _ChatHandler()
    handler.config.openai_extra_headers = dict(_OPERATOR_EXTRAS)
    backend = _RecordingBackend()
    handler.anthropic_backend = backend
    handler.ccr_response_handler = None
    request = _build_request(
        {"model": "grok-4", "messages": [{"role": "user", "content": "hello"}]},
        _GROK_HEADERS,
        path="/v1/chat/completions",
    )

    response = anyio.run(handler.handle_openai_chat, request)

    assert response.status_code == 200
    assert backend.captured_headers is not None
    headers = _lower_unique(backend.captured_headers)
    assert headers["x-operator-sentinel"] == _SENTINEL_VALUE
    assert headers["authorization"] == _OPERATOR_KEY
    assert handler.captured_request is None
