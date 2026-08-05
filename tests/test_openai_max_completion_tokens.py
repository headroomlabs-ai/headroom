"""OpenAI chat-path compatibility shim: max_tokens -> max_completion_tokens.

GPT-5 / o-series chat models reject the legacy ``max_tokens`` and require
``max_completion_tokens`` ("Unsupported parameter: 'max_tokens' is not supported
with this model. Use 'max_completion_tokens' instead."). openai-compatible
clients (opencode, older SDKs) still send ``max_tokens``, so the proxy — which
already owns the outbound request body — translates it.
"""

from __future__ import annotations

from headroom.proxy.handlers.openai import _normalize_openai_max_tokens


def test_renames_legacy_max_tokens():
    body = {"model": "gpt-5.3-chat-latest", "max_tokens": 256, "messages": []}
    _normalize_openai_max_tokens(body, backend_owns_translation=False)
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 256


def test_backend_owned_translation_preserves_max_tokens():
    body = {"model": "claude-sonnet-4-6", "max_tokens": 32, "messages": []}
    _normalize_openai_max_tokens(body, backend_owns_translation=True)
    assert body == {"model": "claude-sonnet-4-6", "max_tokens": 32, "messages": []}


def test_preserves_existing_max_completion_tokens_and_drops_legacy():
    body = {"max_tokens": 256, "max_completion_tokens": 100}
    _normalize_openai_max_tokens(body)
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 100  # explicit value wins


def test_noop_when_only_max_completion_tokens():
    body = {"max_completion_tokens": 128}
    _normalize_openai_max_tokens(body)
    assert body == {"max_completion_tokens": 128}


def test_noop_when_neither_present():
    body = {"model": "gpt-4o", "messages": []}
    _normalize_openai_max_tokens(body)
    assert "max_completion_tokens" not in body
    assert "max_tokens" not in body


def test_null_max_tokens_is_dropped_without_setting_completion():
    body = {"max_tokens": None}
    _normalize_openai_max_tokens(body)
    assert "max_tokens" not in body
    assert body.get("max_completion_tokens") is None


def test_non_dict_is_safe():
    _normalize_openai_max_tokens(None)  # type: ignore[arg-type]
    _normalize_openai_max_tokens("nope")  # type: ignore[arg-type]
