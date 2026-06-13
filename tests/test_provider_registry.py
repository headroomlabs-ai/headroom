from __future__ import annotations

import logging

from headroom.providers.registry import (
    ProviderApiOverrides,
    build_proxy_provider_runtime,
    create_proxy_backend,
    format_backend_status,
    resolve_api_overrides,
    resolve_api_targets,
)
from headroom.proxy.models import ProxyConfig


def test_resolve_api_overrides_prefers_explicit_values_over_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_TARGET_API_URL", "https://env.anthropic.example/v1")
    monkeypatch.setenv("OPENAI_TARGET_API_URL", "https://env.openai.example/v1")
    monkeypatch.setenv("VERTEX_TARGET_API_URL", "https://env-vertex-aiplatform.example/v1")

    overrides = resolve_api_overrides(
        anthropic_api_url="https://cli.anthropic.example/v1",
        openai_api_url=None,
        gemini_api_url=None,
        cloudcode_api_url=None,
        vertex_api_url="https://cli-vertex-aiplatform.example/v1",
    )

    assert overrides == ProviderApiOverrides(
        anthropic="https://cli.anthropic.example/v1",
        openai="https://env.openai.example/v1",
        gemini=None,
        cloudcode=None,
        vertex="https://cli-vertex-aiplatform.example/v1",
    )


def test_resolve_api_targets_normalizes_trailing_v1() -> None:
    targets = resolve_api_targets(
        ProviderApiOverrides(
            anthropic="https://anthropic.example/v1/",
            openai="https://openai.example/v1",
            gemini="https://gemini.example/v1",
            cloudcode="https://cloudcode.example/v1/",
            vertex="https://vertex.example/v1/",
        )
    )

    assert targets.anthropic == "https://anthropic.example"
    assert targets.openai == "https://openai.example"
    assert targets.gemini == "https://gemini.example"
    assert targets.cloudcode == "https://cloudcode.example"
    assert targets.vertex == "https://vertex.example"


def test_proxy_config_exposes_provider_api_overrides() -> None:
    config = ProxyConfig(
        anthropic_api_url="https://anthropic.example",
        openai_api_url="https://openai.example",
        gemini_api_url=None,
        cloudcode_api_url="https://cloudcode.example",
        vertex_api_url="https://vertex.example",
    )

    assert config.provider_api_overrides == ProviderApiOverrides(
        anthropic="https://anthropic.example",
        openai="https://openai.example",
        gemini=None,
        cloudcode="https://cloudcode.example",
        vertex="https://vertex.example",
    )


def test_format_backend_status_for_anyllm() -> None:
    assert (
        format_backend_status(
            backend="anyllm",
            anyllm_provider="groq",
            bedrock_region="us-central1",
        )
        == "Groq via any-llm"
    )


def test_format_backend_status_for_anthropic_direct() -> None:
    assert (
        format_backend_status(
            backend="anthropic",
            anyllm_provider="ignored",
            bedrock_region=None,
        )
        == "ANTHROPIC (direct API)"
    )


def test_proxy_provider_runtime_routes_model_metadata_and_passthrough() -> None:
    runtime = build_proxy_provider_runtime(ProxyConfig())

    assert runtime.model_metadata_provider({"x-api-key": "test"}) == "anthropic"
    assert runtime.model_metadata_provider({}) == "openai"
    assert (
        runtime.select_passthrough_base_url({"x-api-key": "test"}) == runtime.api_targets.anthropic
    )
    assert (
        runtime.select_passthrough_base_url({"x-goog-api-key": "test"})
        == runtime.api_targets.gemini
    )
    assert runtime.select_passthrough_base_url({"api-key": "azure", "x-headroom-base-url": ""}) == (
        runtime.api_targets.openai
    )


def test_create_proxy_backend_handles_missing_litellm_backend(caplog) -> None:
    logger = logging.getLogger("test")

    with caplog.at_level(logging.WARNING):
        missing = create_proxy_backend(
            backend="bedrock",
            anyllm_provider="ignored",
            bedrock_region="us-east-1",
            logger=logger,
            litellm_backend_cls=lambda provider, region: (_ for _ in ()).throw(
                ImportError("missing")
            ),
        )

    assert missing is None
    assert "LiteLLM backend not available" in caplog.text


def test_proxy_provider_runtime_loaders_cache_backend_types(monkeypatch) -> None:
    import headroom.providers.registry as registry

    anyllm_loads = 0
    litellm_loads = 0

    class FakeAnyLLMBackend:
        pass

    class FakeLiteLLMBackend:
        pass

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal anyllm_loads, litellm_loads
        if name == "headroom.backends.anyllm":
            anyllm_loads += 1
            return type("Module", (), {"AnyLLMBackend": FakeAnyLLMBackend})()
        if name == "headroom.backends.litellm":
            litellm_loads += 1
            return type("Module", (), {"LiteLLMBackend": FakeLiteLLMBackend})()
        raise AssertionError(name)

    monkeypatch.setattr(registry, "AnyLLMBackendType", None)
    monkeypatch.setattr(registry, "LiteLLMBackendType", None)
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert registry._load_anyllm_backend() is FakeAnyLLMBackend
    assert registry._load_anyllm_backend() is FakeAnyLLMBackend
    assert registry._load_litellm_backend() is FakeLiteLLMBackend
    assert registry._load_litellm_backend() is FakeLiteLLMBackend
    assert anyllm_loads == 1
    assert litellm_loads == 1


def test_proxy_provider_runtime_transport_helpers_handle_missing_usage() -> None:
    import headroom.providers.registry as registry

    class Storage:
        def __init__(self) -> None:
            self.saved = []

        def save(self, metrics) -> None:
            self.saved.append(metrics)

    client = type(
        "Client",
        (),
        {
            "_storage": Storage(),
            "_original": type(
                "Original",
                (),
                {
                    "chat": type(
                        "Chat",
                        (),
                        {
                            "completions": type(
                                "Completions",
                                (),
                                {
                                    "create": staticmethod(
                                        lambda **kwargs: type("Resp", (), {"usage": None})()
                                    )
                                },
                            )()
                        },
                    )(),
                    "messages": type(
                        "Messages",
                        (),
                        {
                            "create": staticmethod(
                                lambda **kwargs: type("Resp", (), {"usage": None})()
                            )
                        },
                    )(),
                },
            )(),
        },
    )()
    openai_metrics = type("Metrics", (), {"tokens_output": 0, "cached_tokens": 0})()
    anthropic_metrics = type("Metrics", (), {"tokens_output": 0, "cached_tokens": 0})()

    registry._call_openai_transport(
        client,
        model="gpt-4o",
        messages=[],
        stream=False,
        metrics=openai_metrics,
    )
    registry._call_anthropic_transport(
        client,
        model="claude",
        messages=[],
        stream=False,
        metrics=anthropic_metrics,
    )

    assert openai_metrics.tokens_output == 0
    assert openai_metrics.cached_tokens == 0
    assert anthropic_metrics.tokens_output == 0
    assert anthropic_metrics.cached_tokens == 0
    assert len(client._storage.saved) == 2


def test_proxy_provider_runtime_transport_helpers_handle_usage_without_optional_cache_fields() -> (
    None
):
    import headroom.providers.registry as registry

    class Storage:
        def __init__(self) -> None:
            self.saved = []

        def save(self, metrics) -> None:
            self.saved.append(metrics)

    client = type(
        "Client",
        (),
        {
            "_storage": Storage(),
            "_original": type(
                "Original",
                (),
                {
                    "chat": type(
                        "Chat",
                        (),
                        {
                            "completions": type(
                                "Completions",
                                (),
                                {
                                    "create": staticmethod(
                                        lambda **kwargs: type(
                                            "Resp",
                                            (),
                                            {
                                                "usage": type(
                                                    "Usage",
                                                    (),
                                                    {"completion_tokens": 7},
                                                )()
                                            },
                                        )()
                                    )
                                },
                            )()
                        },
                    )(),
                    "messages": type(
                        "Messages",
                        (),
                        {
                            "create": staticmethod(
                                lambda **kwargs: type(
                                    "Resp",
                                    (),
                                    {
                                        "usage": type(
                                            "Usage",
                                            (),
                                            {"output_tokens": 5},
                                        )()
                                    },
                                )()
                            )
                        },
                    )(),
                },
            )(),
        },
    )()
    openai_metrics = type("Metrics", (), {"tokens_output": 0, "cached_tokens": 0})()
    anthropic_metrics = type("Metrics", (), {"tokens_output": 0, "cached_tokens": 0})()

    registry._call_openai_transport(
        client,
        model="gpt-4o",
        messages=[],
        stream=False,
        metrics=openai_metrics,
    )
    registry._call_anthropic_transport(
        client,
        model="claude",
        messages=[],
        stream=False,
        metrics=anthropic_metrics,
    )

    assert openai_metrics.tokens_output == 7
    assert openai_metrics.cached_tokens == 0
    assert anthropic_metrics.tokens_output == 5
    assert anthropic_metrics.cached_tokens == 0
    assert len(client._storage.saved) == 2


def test_proxy_provider_runtime_openai_transport_handles_prompt_details_without_cached_tokens() -> (
    None
):
    import headroom.providers.registry as registry

    class Storage:
        def __init__(self) -> None:
            self.saved = []

        def save(self, metrics) -> None:
            self.saved.append(metrics)

    client = type(
        "Client",
        (),
        {
            "_storage": Storage(),
            "_original": type(
                "Original",
                (),
                {
                    "chat": type(
                        "Chat",
                        (),
                        {
                            "completions": type(
                                "Completions",
                                (),
                                {
                                    "create": staticmethod(
                                        lambda **kwargs: type(
                                            "Resp",
                                            (),
                                            {
                                                "usage": type(
                                                    "Usage",
                                                    (),
                                                    {
                                                        "completion_tokens": 9,
                                                        "prompt_tokens_details": type(
                                                            "Details",
                                                            (),
                                                            {},
                                                        )(),
                                                    },
                                                )()
                                            },
                                        )()
                                    )
                                },
                            )()
                        },
                    )()
                },
            )(),
        },
    )()
    metrics = type("Metrics", (), {"tokens_output": 0, "cached_tokens": 0})()

    registry._call_openai_transport(
        client,
        model="gpt-4o",
        messages=[],
        stream=False,
        metrics=metrics,
    )

    assert metrics.tokens_output == 9
    assert metrics.cached_tokens == 0
    assert len(client._storage.saved) == 1
