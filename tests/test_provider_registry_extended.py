from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from headroom.providers.registry import (
    ProviderApiTargets,
    ProxyProviderRuntime,
    call_client_transport,
    create_proxy_backend,
    format_backend_status,
)


class DummyStorage:
    def __init__(self) -> None:
        self.saved: list[Any] = []

    def save(self, metrics: Any) -> None:
        self.saved.append(metrics)


class DummyClient:
    def __init__(self) -> None:
        self._storage = DummyStorage()
        self._wrapped_stream: tuple[Any, Any] | None = None
        self._original = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._openai_create)),
            messages=SimpleNamespace(create=self._anthropic_create, stream=self._anthropic_stream),
        )
        self.openai_calls: list[dict[str, Any]] = []
        self.anthropic_calls: list[dict[str, Any]] = []

    def _openai_create(self, **kwargs: Any) -> Any:
        self.openai_calls.append(kwargs)
        if kwargs["stream"]:
            return iter(["chunk-1", "chunk-2"])
        return SimpleNamespace(
            usage=SimpleNamespace(
                completion_tokens=7,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            )
        )

    def _anthropic_create(self, **kwargs: Any) -> Any:
        self.anthropic_calls.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(
                output_tokens=5,
                cache_read_input_tokens=2,
            )
        )

    def _anthropic_stream(self, **kwargs: Any) -> Any:
        self.anthropic_calls.append(kwargs)
        return "anthropic-stream"

    def _wrap_stream(self, stream: Any, metrics: Any) -> Any:
        self._wrapped_stream = (stream, metrics)
        return ("wrapped", stream)


def test_proxy_provider_runtime_selects_targets_and_providers() -> None:
    runtime = ProxyProviderRuntime(
        api_targets=ProviderApiTargets(
            anthropic="https://anthropic.example",
            openai="https://openai.example",
            gemini="https://gemini.example",
            cloudcode="https://cloudcode.example",
        ),
        pipeline_providers={
            "anthropic": SimpleNamespace(name="anthropic"),
            "openai": SimpleNamespace(name="openai"),
        },
    )

    assert runtime.api_target("anthropic") == "https://anthropic.example"
    assert runtime.pipeline_provider("openai").name == "openai"
    assert runtime.model_metadata_provider({"Authorization": "Bearer sk-ant-api03-test"}) == (
        "anthropic"
    )
    assert runtime.select_passthrough_base_url({"x-goog-api-key": "test"}) == (
        "https://gemini.example"
    )
    assert (
        runtime.select_passthrough_base_url(
            {"api-key": "azure-key", "x-headroom-base-url": "https://azure.example/openai/"}
        )
        == "https://azure.example/openai"
    )
    assert runtime.select_passthrough_base_url({}) == "https://openai.example"


def test_create_proxy_backend_uses_injected_backend_types() -> None:
    logger = logging.getLogger("test")

    anyllm = create_proxy_backend(
        backend="anyllm",
        anyllm_provider="groq",
        bedrock_region=None,
        logger=logger,
        anyllm_backend_cls=lambda provider, api_base: {
            "kind": "anyllm",
            "provider": provider,
            "api_base": api_base,
        },
    )
    litellm = create_proxy_backend(
        backend="bedrock",
        anyllm_provider="ignored",
        bedrock_region="us-east-1",
        logger=logger,
        litellm_backend_cls=lambda provider, region, **_: {
            "kind": "litellm",
            "provider": provider,
            "region": region,
        },
    )

    assert anyllm == {"kind": "anyllm", "provider": "groq", "api_base": None}
    assert litellm == {"kind": "litellm", "provider": "bedrock", "region": "us-east-1"}


def test_create_proxy_backend_passes_openai_api_url_to_anyllm() -> None:
    """Regression for #942: --openai-api-url must reach the any-llm backend."""
    logger = logging.getLogger("test")

    anyllm = create_proxy_backend(
        backend="anyllm",
        anyllm_provider="openai",
        bedrock_region=None,
        logger=logger,
        openai_api_url="https://custom-provider.example/v1",
        anyllm_backend_cls=lambda provider, api_base: {
            "provider": provider,
            "api_base": api_base,
        },
    )

    assert anyllm == {
        "provider": "openai",
        "api_base": "https://custom-provider.example/v1",
    }


def test_create_proxy_backend_handles_missing_or_direct_backends(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test")

    direct = create_proxy_backend(
        backend="anthropic",
        anyllm_provider="ignored",
        bedrock_region=None,
        logger=logger,
    )

    with caplog.at_level(logging.WARNING):
        missing = create_proxy_backend(
            backend="anyllm",
            anyllm_provider="groq",
            bedrock_region=None,
            logger=logger,
            anyllm_backend_cls=lambda provider, api_base: (_ for _ in ()).throw(
                ImportError("missing")
            ),
        )

    assert direct is None
    assert missing is None
    assert "any-llm backend not available" in caplog.text


def test_format_backend_status_uses_litellm_provider_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "headroom.backends.litellm.get_provider_config",
        lambda provider: SimpleNamespace(
            display_name=provider.upper(),
            uses_region=(provider == "bedrock"),
        ),
    )

    assert (
        format_backend_status(
            backend="litellm-bedrock",
            anyllm_provider="ignored",
            bedrock_region="us-west-2",
        )
        == "BEDROCK via LiteLLM (region=us-west-2)"
    )
    assert (
        format_backend_status(
            backend="litellm-openai",
            anyllm_provider="ignored",
            bedrock_region=None,
        )
        == "OPENAI via LiteLLM"
    )


def test_call_client_transport_covers_openai_and_anthropic_paths() -> None:
    client = DummyClient()
    openai_metrics = SimpleNamespace(tokens_output=0, cached_tokens=0)
    anthropic_metrics = SimpleNamespace(tokens_output=0, cached_tokens=0)

    openai_response = call_client_transport(
        "openai",
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        metrics=openai_metrics,
        temperature=0,
    )
    openai_stream = call_client_transport(
        "openai",
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        metrics=openai_metrics,
    )
    anthropic_response = call_client_transport(
        "anthropic",
        client,
        model="claude-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        metrics=anthropic_metrics,
        max_tokens=32,
    )
    anthropic_stream = call_client_transport(
        "anthropic",
        client,
        model="claude-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        metrics=anthropic_metrics,
        max_tokens=32,
    )

    assert openai_response.usage.completion_tokens == 7
    assert openai_metrics.tokens_output == 7
    assert openai_metrics.cached_tokens == 3
    assert openai_stream == ("wrapped", client._wrapped_stream[0])
    assert anthropic_response.usage.output_tokens == 5
    assert anthropic_metrics.tokens_output == 5
    assert anthropic_metrics.cached_tokens == 2
    assert anthropic_stream == "anthropic-stream"
    assert len(client._storage.saved) == 3


def test_call_client_transport_rejects_unknown_api_style() -> None:
    with pytest.raises(ValueError, match="Unsupported api_style"):
        call_client_transport(
            "unknown",
            DummyClient(),
            model="gpt-4o",
            messages=[],
            stream=False,
            metrics=SimpleNamespace(),
        )
