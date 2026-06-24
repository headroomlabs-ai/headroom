"""Cache-stat surfacing for `LiteLLMBackend.send_openai_message`.

LiteLLM normalizes prompt-cache statistics onto its `Usage` object from
multiple upstream dialects:

* Anthropic / Bedrock-Claude → top-level attrs `cache_read_input_tokens`
  and `cache_creation_input_tokens` (also mirrored into
  `prompt_tokens_details.cached_tokens` / `cache_creation_tokens`).
* OpenAI prompt-caching → only `prompt_tokens_details.cached_tokens`.

Before the fix, `send_openai_message` flattened only
`prompt_tokens / completion_tokens / total_tokens` into the response dict
and silently dropped all cache stats on the floor — breaking
`PrefixCacheTracker.update_from_response` for the entire backend-routed
path (it always saw zero cache hits, so live-zone-only compression never
engaged).

These tests pin the contract for the three relevant shapes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402  (must follow importorskip)


class _FakeUsage:
    """Stand-in for `litellm.types.utils.Usage`.

    `MagicMock` auto-creates attributes on access, which would defeat the
    point of the "no cache fields → no keys added" test. A plain object
    with only the attributes we explicitly set keeps `getattr(..., 0)`
    honest.
    """

    def __init__(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
        prompt_tokens_details: Any | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        if cache_read_input_tokens is not None:
            self.cache_read_input_tokens = cache_read_input_tokens
        if cache_creation_input_tokens is not None:
            self.cache_creation_input_tokens = cache_creation_input_tokens
        if prompt_tokens_details is not None:
            self.prompt_tokens_details = prompt_tokens_details


class _FakePromptTokensDetails:
    """OpenAI-style nested cache shape stand-in."""

    def __init__(
        self,
        *,
        cached_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
    ) -> None:
        if cached_tokens is not None:
            self.cached_tokens = cached_tokens
        if cache_creation_tokens is not None:
            self.cache_creation_tokens = cache_creation_tokens


def _make_response(usage: _FakeUsage) -> MagicMock:
    """Build a minimal `ModelResponse`-shaped mock with the given usage."""
    response = MagicMock()
    response.id = "chatcmpl-test"
    response.created = 1_700_000_000
    response.choices = [
        MagicMock(
            index=0,
            message=MagicMock(role="assistant", content="hi", tool_calls=None),
            finish_reason="stop",
        )
    ]
    response.usage = usage
    return response


def _make_backend() -> LiteLLMBackend:
    # Patch the inference-profile fetch so `__init__` doesn't try to talk to AWS.
    with patch("headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value={}):
        return LiteLLMBackend(provider="openrouter")


def _request_body() -> dict[str, Any]:
    return {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
    }


# =============================================================================
# 1. Anthropic-style (top-level cache_read_input_tokens / cache_creation_input_tokens)
# =============================================================================


async def test_anthropic_style_cache_fields_surface_in_usage_block() -> None:
    """Bedrock-Claude / Anthropic responses set the top-level dialect.

    LiteLLM mirrors them into `prompt_tokens_details` too. Our extractor
    must prefer the explicit top-level values (cache_read=1500, cache_write=200)
    and also expose the OpenAI nested shape so single-dialect callers
    don't have to branch.
    """
    usage = _FakeUsage(
        prompt_tokens=2000,
        completion_tokens=100,
        total_tokens=2100,
        cache_read_input_tokens=1500,
        cache_creation_input_tokens=200,
        prompt_tokens_details=_FakePromptTokensDetails(
            cached_tokens=1500,
            cache_creation_tokens=200,
        ),
    )
    response = _make_response(usage)

    backend = _make_backend()
    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = response
        result = await backend.send_openai_message(_request_body(), {})

    body_usage = result.body["usage"]
    assert body_usage["prompt_tokens"] == 2000
    assert body_usage["completion_tokens"] == 100
    assert body_usage["total_tokens"] == 2100
    assert body_usage["cache_read_input_tokens"] == 1500
    assert body_usage["cache_creation_input_tokens"] == 200
    assert body_usage["prompt_tokens_details"] == {"cached_tokens": 1500}


# =============================================================================
# 2. OpenAI-style only (prompt_tokens_details.cached_tokens, no top-level)
# =============================================================================


async def test_openai_nested_cache_fields_surface_when_top_level_absent() -> None:
    """OpenAI prompt-caching responses only populate the nested dialect.

    With no top-level `cache_read_input_tokens` attribute on the Usage
    object, we must fall back to `prompt_tokens_details.cached_tokens`
    and mirror it into the Anthropic-style top-level keys for downstream
    consumers.
    """
    usage = _FakeUsage(
        prompt_tokens=1200,
        completion_tokens=50,
        total_tokens=1250,
        prompt_tokens_details=_FakePromptTokensDetails(cached_tokens=800),
    )
    response = _make_response(usage)

    backend = _make_backend()
    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = response
        result = await backend.send_openai_message(_request_body(), {})

    body_usage = result.body["usage"]
    assert body_usage["prompt_tokens"] == 1200
    assert body_usage["completion_tokens"] == 50
    assert body_usage["total_tokens"] == 1250
    assert body_usage["cache_read_input_tokens"] == 800
    assert body_usage["cache_creation_input_tokens"] == 0
    assert body_usage["prompt_tokens_details"] == {"cached_tokens": 800}


# =============================================================================
# 3. Cold start — no cache fields anywhere → keep usage_block shape stable
# =============================================================================


async def test_no_cache_fields_means_no_cache_keys_in_usage_block() -> None:
    """Cold-start path: no cache attributes at all on the Usage object.

    We must NOT inject `cache_read_input_tokens`, `cache_creation_input_tokens`,
    or `prompt_tokens_details` into `usage_block` — keep the dict shape
    identical to the pre-fix behaviour so callers that key off presence
    (rather than value) don't accidentally start seeing 0 as "we have
    cache data, the model just didn't cache".
    """
    usage = _FakeUsage(
        prompt_tokens=500,
        completion_tokens=25,
        total_tokens=525,
    )
    response = _make_response(usage)

    backend = _make_backend()
    with patch("headroom.backends.litellm.acompletion", new_callable=AsyncMock) as mock_acomp:
        mock_acomp.return_value = response
        result = await backend.send_openai_message(_request_body(), {})

    body_usage = result.body["usage"]
    assert body_usage == {
        "prompt_tokens": 500,
        "completion_tokens": 25,
        "total_tokens": 525,
    }
    assert "cache_read_input_tokens" not in body_usage
    assert "cache_creation_input_tokens" not in body_usage
    assert "prompt_tokens_details" not in body_usage
