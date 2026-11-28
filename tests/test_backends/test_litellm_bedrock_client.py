"""Tests for the ``bedrock_client_factory`` hook on ``LiteLLMBackend``.

When a factory is supplied, the backend eagerly builds a
``bedrock-runtime`` boto3 client at init time, then passes it to
``litellm.acompletion(..., aws_bedrock_client=client)`` on every call.
This is what enables pluggable session implementations (e.g.
``boto3-refresh-session``) to refresh STS credentials transparently
without proxy restarts.

These tests cover the three contract points:

1. ``None`` factory (the default) — no client is built, no
   ``aws_bedrock_client`` is sent to litellm.
2. Factory returns a boto3-shaped client — it is reused for every
   call and forwarded to litellm.
3. Factory returns ``None`` — falls back to default behavior.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402  (must follow importorskip)


def _fake_bedrock_client() -> MagicMock:
    """Return a stand-in for a boto3 bedrock-runtime client.

    The two methods the backend introspects for validation:
    ``converse`` and ``invoke_model``.
    """
    client = MagicMock(name="boto3_bedrock_runtime_client")
    client.converse = MagicMock()
    client.invoke_model = MagicMock()
    return client


def test_default_factory_is_none_no_client_built() -> None:
    """Without a factory, no boto3 client is constructed and the
    backend keeps the default ``None`` slot so litellm builds its
    own from the process environment."""
    backend = LiteLLMBackend(provider="bedrock", region="us-east-1")
    assert backend._bedrock_client is None
    assert backend._bedrock_client_factory is None


def test_factory_returning_client_is_reused_across_calls() -> None:
    """A factory returning a boto3 client is invoked once at init
    time, and the resulting client is forwarded to every litellm
    call via the ``aws_bedrock_client`` kwarg."""
    fake_client = _fake_bedrock_client()
    factory_calls: list[str | None] = []

    def factory(region: str | None) -> Any:
        factory_calls.append(region)
        return fake_client

    backend = LiteLLMBackend(
        provider="bedrock", region="ap-northeast-1", bedrock_client_factory=factory
    )
    # Factory invoked exactly once at init
    assert factory_calls == ["ap-northeast-1"]
    assert backend._bedrock_client is fake_client

    # Drive a non-streaming call and verify litellm received the client
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi", tool_calls=None))],
            usage=MagicMock(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )

    body = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
    }
    with patch("headroom.backends.litellm.acompletion", new=fake_acompletion):
        # ``send_message`` builds OpenAI-shaped responses from Anthropic
        # inputs; the test only cares about the kwargs litellm sees.
        import asyncio

        asyncio.run(backend.send_message(body, {}))

    assert captured.get("aws_bedrock_client") is fake_client
    # Factory still invoked exactly once (call reuse)
    assert factory_calls == ["ap-northeast-1"]


def test_factory_returning_none_falls_back_to_default() -> None:
    """If the factory returns ``None`` (or ``False``), the backend
    skips client construction and stays on the env-based path."""

    def factory(_region: str | None) -> Any:
        return None

    backend = LiteLLMBackend(provider="bedrock", region="us-west-2", bedrock_client_factory=factory)
    assert backend._bedrock_client is None


def test_factory_returning_non_bedrock_client_raises() -> None:
    """Defensive: refuse factories that return an object that does
    not look like a ``bedrock-runtime`` client."""

    def factory(_region: str | None) -> Any:
        return object()  # no converse / no invoke_model attrs

    import pytest

    with pytest.raises(TypeError, match="bedrock_client_factory"):
        LiteLLMBackend(provider="bedrock", region="us-west-2", bedrock_client_factory=factory)
