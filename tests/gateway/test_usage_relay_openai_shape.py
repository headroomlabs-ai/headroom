"""``/v1/usage`` accepts OpenAI-shaped usage (spec section 4).

Invariant: a relay built from Kong's ``ai-proxy`` log statistics
(``prompt_tokens_details.cached_tokens`` or top-level ``cached_tokens``) maps to
``cache_read_input_tokens`` and is applied; the pre-existing behaviour for
Anthropic-shaped usage and for a bare ``{"prompt_tokens": N}`` (400) is unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.gateway.conftest import compress
from tests.gateway.samples import big_tool_history


def _open_session(client, session_id: str) -> None:
    resp = compress(
        client,
        {"model": "gpt-4o", "messages": big_tool_history(), "config": {"session_id": session_id}},
    )
    assert resp.status_code == 200, resp.text


def _usage(client, session_id: str, usage: dict[str, Any]):
    return client.post("/v1/usage", json={"session_id": session_id, "usage": usage})


def test_prompt_tokens_details_cached_tokens_is_applied(headroom_client) -> None:
    _open_session(headroom_client, "oai-details")
    resp = _usage(
        headroom_client,
        "oai-details",
        {
            "prompt_tokens": 60_000,
            "completion_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 50_000},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["applied"] is True
    assert data["frozen_message_count"] >= 1


def test_top_level_cached_tokens_is_applied(headroom_client) -> None:
    _open_session(headroom_client, "oai-top")
    resp = _usage(headroom_client, "oai-top", {"prompt_tokens": 60_000, "cached_tokens": 50_000})
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is True


def test_openai_shape_matches_anthropic_shape_frozen_count(headroom_client) -> None:
    """The mapping is exact: the OpenAI shape freezes what the equivalent
    Anthropic read-only relay freezes."""
    _open_session(headroom_client, "oai-eq-a")
    _open_session(headroom_client, "oai-eq-b")
    a = _usage(headroom_client, "oai-eq-a", {"cache_read_input_tokens": 50_000})
    b = _usage(headroom_client, "oai-eq-b", {"prompt_tokens_details": {"cached_tokens": 50_000}})
    assert a.status_code == b.status_code == 200
    assert a.json()["frozen_message_count"] == b.json()["frozen_message_count"]


def test_cached_tokens_zero_alone_is_no_cache_signal(headroom_client) -> None:
    """OpenAI has no write signal, so ``cached_tokens: 0`` on its own must map to
    the existing read-only-zero rule: accepted, not applied."""
    _open_session(headroom_client, "oai-zero")
    resp = _usage(
        headroom_client,
        "oai-zero",
        {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is False
    assert resp.json()["reason"] == "no_cache_signal"


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens_details": {"cached_tokens": -1}},
        {"cached_tokens": True},
        {"prompt_tokens_details": {"cached_tokens": "5"}},
    ],
    ids=["negative", "bool", "string"],
)
def test_invalid_openai_cache_values_are_400(headroom_client, usage) -> None:
    _open_session(headroom_client, "oai-bad")
    resp = _usage(headroom_client, "oai-bad", usage)
    assert resp.status_code == 400, resp.text


# --- unchanged behaviour (passes today) --------------------------------------


def test_bare_prompt_tokens_is_still_400(headroom_client) -> None:
    _open_session(headroom_client, "oai-bare")
    resp = _usage(headroom_client, "oai-bare", {"prompt_tokens": 12345})
    assert resp.status_code == 400
    assert "cache" in resp.json()["error"]["message"]


def test_anthropic_shape_unchanged(headroom_client) -> None:
    _open_session(headroom_client, "ant-ok")
    resp = _usage(
        headroom_client,
        "ant-ok",
        {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 50_000},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is True


def test_unknown_session_still_404(headroom_client) -> None:
    resp = _usage(headroom_client, "never-seen", {"prompt_tokens_details": {"cached_tokens": 10}})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "unknown_session"
