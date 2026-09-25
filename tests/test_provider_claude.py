from __future__ import annotations

import pytest

from headroom.providers.claude import (
    CONTEXT_1M_SUFFIX,
    DEFAULT_1M_MODEL,
    DEFAULT_API_URL,
    HEADROOM_1M_MODEL_ENV,
    proxy_base_url,
    resolve_1m_model,
)


def test_claude_runtime_exposes_default_api_and_local_proxy_url() -> None:
    # Arrange / Act / Assert
    assert DEFAULT_API_URL == "https://api.anthropic.com"
    assert proxy_base_url(4321) == "http://127.0.0.1:4321"


@pytest.mark.parametrize(
    ("current", "fallback", "expected"),
    [
        (" claude-sonnet-5 ", None, "claude-sonnet-5[1m]"),
        ("claude-opus-5[1m]", "ignored", "claude-opus-5[1m]"),
        (None, " claude-opus-9 ", "claude-opus-9[1m]"),
        ("  ", None, f"{DEFAULT_1M_MODEL}{CONTEXT_1M_SUFFIX}"),
    ],
)
def test_resolve_1m_model_contract_is_provider_owned(
    monkeypatch: pytest.MonkeyPatch,
    current: str | None,
    fallback: str | None,
    expected: str,
) -> None:
    if fallback is None:
        monkeypatch.delenv(HEADROOM_1M_MODEL_ENV, raising=False)
    else:
        monkeypatch.setenv(HEADROOM_1M_MODEL_ENV, fallback)

    assert resolve_1m_model(current) == expected
    assert resolve_1m_model(expected) == expected
