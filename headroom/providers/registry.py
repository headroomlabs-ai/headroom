"""Provider runtime registry and transport helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from headroom.providers.claude import DEFAULT_API_URL as DEFAULT_ANTHROPIC_API_URL
from headroom.providers.codex import DEFAULT_API_URL as DEFAULT_OPENAI_API_URL
from headroom.providers.gemini import DEFAULT_API_URL as DEFAULT_GEMINI_API_URL

DEFAULT_CLOUDCODE_API_URL = "https://cloudcode-pa.googleapis.com"
DEFAULT_VERTEX_API_URL = "https://us-central1-aiplatform.googleapis.com"

if TYPE_CHECKING:
    from headroom.backends.base import Backend
    from headroom.providers.base import Provider

AnyLLMBackendType: Any = None
LiteLLMBackendType: Any = None


@dataclass(frozen=True)
class ProviderApiOverrides:
    """Optional upstream API URL overrides configured for the proxy."""

    anthropic: str | None = None
    openai: str | None = None
    gemini: str | None = None
    cloudcode: str | None = None
    vertex: str | None = None


@dataclass(frozen=True)
class ProviderApiTargets:
    """Resolved upstream API targets after provider normalization."""

    anthropic: str = DEFAULT_ANTHROPIC_API_URL
    openai: str = DEFAULT_OPENAI_API_URL
    gemini: str = DEFAULT_GEMINI_API_URL
    cloudcode: str = DEFAULT_CLOUDCODE_API_URL
    vertex: str = DEFAULT_VERTEX_API_URL


@dataclass(frozen=True)
class ProxyProviderRuntime:
    """Provider runtime state used by the proxy server."""

    api_targets: ProviderApiTargets
    pipeline_providers: dict[str, Provider]

    def api_target(self, provider_name: str) -> str:
        """Return the resolved upstream target for a provider."""
        return {
            "anthropic": self.api_targets.anthropic,
            "openai": self.api_targets.openai,
            "gemini": self.api_targets.gemini,
            "cloudcode": self.api_targets.cloudcode,
            "vertex": self.api_targets.vertex,
        }[provider_name]

    def pipeline_provider(self, provider_name: str) -> Provider:
        """Return the pipeline provider instance for a provider."""
        return self.pipeline_providers[provider_name]

    def model_metadata_provider(self, headers: Mapping[str, str]) -> str:
        """Resolve the upstream provider that should serve OpenAI-style model metadata."""
        return "anthropic" if _is_anthropic_auth(headers) else "openai"

    def select_passthrough_base_url(self, headers: Mapping[str, str]) -> str:
        """Resolve the upstream base URL for catch-all passthrough requests."""
        if _is_anthropic_auth(headers):
            return self.api_targets.anthropic
        if headers.get("x-goog-api-key"):
            return self.api_targets.gemini
        if headers.get("api-key"):
            azure_base = headers.get("x-headroom-base-url", "")
            if azure_base:
                return azure_base.rstrip("/")
        return self.api_targets.openai


def _normalize_api_url(url: str | None, *, default: str) -> str:
    if not url:
        return default

    normalized = url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def resolve_api_overrides(
    *,
    anthropic_api_url: str | None,
    openai_api_url: str | None,
    gemini_api_url: str | None,
    cloudcode_api_url: str | None,
    vertex_api_url: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProviderApiOverrides:
    """Resolve provider API URL overrides from CLI/config inputs and environment."""
    env = environ or os.environ
    return ProviderApiOverrides(
        anthropic=anthropic_api_url
        or env.get("ANTHROPIC_TARGET_API_URL")
        or env.get("ANTHROPIC_FOUNDRY_BASE_URL"),
        openai=openai_api_url or env.get("OPENAI_TARGET_API_URL"),
        gemini=gemini_api_url or env.get("GEMINI_TARGET_API_URL"),
        cloudcode=cloudcode_api_url or env.get("CLOUDCODE_TARGET_API_URL"),
        vertex=vertex_api_url or env.get("VERTEX_TARGET_API_URL"),
    )


def resolve_api_targets(overrides: ProviderApiOverrides) -> ProviderApiTargets:
    """Resolve normalized upstream provider targets from configured overrides."""
    return ProviderApiTargets(
        anthropic=_normalize_api_url(overrides.anthropic, default=DEFAULT_ANTHROPIC_API_URL),
        openai=_normalize_api_url(overrides.openai, default=DEFAULT_OPENAI_API_URL),
        gemini=_normalize_api_url(overrides.gemini, default=DEFAULT_GEMINI_API_URL),
        cloudcode=_normalize_api_url(overrides.cloudcode, default=DEFAULT_CLOUDCODE_API_URL),
        vertex=_normalize_api_url(overrides.vertex, default=DEFAULT_VERTEX_API_URL),
    )


def build_proxy_provider_runtime(config: Any) -> ProxyProviderRuntime:
    """Build provider runtime objects and resolved targets for the proxy."""
    from headroom.providers.anthropic import AnthropicProvider
    from headroom.providers.openai import OpenAIProvider

    api_targets = resolve_api_targets(config.provider_api_overrides)
    return ProxyProviderRuntime(
        api_targets=api_targets,
        pipeline_providers={
            # warn=False: the proxy pipeline provider intentionally uses tiktoken
            # approximation (no Anthropic client available at this layer).
            "anthropic": AnthropicProvider(warn=False),
            "openai": OpenAIProvider(),
        },
    )


def create_proxy_backend(
    *,
    backend: str,
    anyllm_provider: str,
    bedrock_region: str | None,
    logger: logging.Logger,
    anyllm_backend_cls: Any | None = None,
    litellm_backend_cls: Any | None = None,
) -> Backend | None:
    """Create the optional translated backend for Anthropic proxy requests."""
    if backend == "anthropic":
        return None

    if backend == "anyllm" or backend.startswith("anyllm-"):
        provider = anyllm_provider
        try:
            backend_cls = anyllm_backend_cls or _load_anyllm_backend()
            instance = cast("Backend", backend_cls(provider=provider))
            logger.info("any-llm backend enabled (provider=%s)", provider)
            return instance
        except ImportError as exc:
            logger.warning("any-llm backend not available: %s", exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to initialize any-llm backend: %s", exc)
            return None

    normalized_backend = backend if backend.startswith("litellm-") else f"litellm-{backend}"
    provider = normalized_backend.replace("litellm-", "")
    try:
        backend_cls = litellm_backend_cls or _load_litellm_backend()
        instance = cast("Backend", backend_cls(provider=provider, region=bedrock_region))
        logger.info("LiteLLM backend enabled (provider=%s, region=%s)", provider, bedrock_region)
        return instance
    except ImportError as exc:
        logger.warning("LiteLLM backend not available: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to initialize LiteLLM backend: %s", exc)
        return None


def format_backend_status(*, backend: str, anyllm_provider: str, bedrock_region: str | None) -> str:
    """Build the human-readable backend status string shown in CLI/server output."""
    if backend == "anthropic":
        return "ANTHROPIC (direct API)"
    if backend == "anyllm" or backend.startswith("anyllm-"):
        return f"{anyllm_provider.title()} via any-llm"

    from headroom.backends.litellm import get_provider_config

    provider = backend.replace("litellm-", "")
    provider_config = get_provider_config(provider)
    if provider_config.uses_region:
        return f"{provider_config.display_name} via LiteLLM (region={bedrock_region})"
    return f"{provider_config.display_name} via LiteLLM"


def call_client_transport(
    api_style: str,
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
    metrics: Any,
    **kwargs: Any,
) -> Any:
    """Dispatch the SDK request to the provider-specific transport handler."""
    try:
        transport = _CLIENT_TRANSPORTS[api_style]
    except KeyError as exc:
        raise ValueError(f"Unsupported api_style: {api_style}") from exc

    return transport(
        client,
        model=model,
        messages=messages,
        stream=stream,
        metrics=metrics,
        **kwargs,
    )


def _load_anyllm_backend() -> Any:
    global AnyLLMBackendType
    if AnyLLMBackendType is None:
        from headroom.backends.anyllm import AnyLLMBackend

        AnyLLMBackendType = AnyLLMBackend
    return AnyLLMBackendType


def _load_litellm_backend() -> Any:
    global LiteLLMBackendType
    if LiteLLMBackendType is None:
        from headroom.backends.litellm import LiteLLMBackend

        LiteLLMBackendType = LiteLLMBackend
    return LiteLLMBackendType


def _call_openai_transport(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
    metrics: Any,
    **kwargs: Any,
) -> Any:
    if stream:
        response = client._original.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        return client._wrap_stream(response, metrics)

    response = client._original.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        **kwargs,
    )

    if hasattr(response, "usage") and response.usage:
        metrics.tokens_output = response.usage.completion_tokens
        if hasattr(response.usage, "prompt_tokens_details"):
            details = response.usage.prompt_tokens_details
            if hasattr(details, "cached_tokens"):
                metrics.cached_tokens = details.cached_tokens

    client._storage.save(metrics)
    return response


def _call_anthropic_transport(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
    metrics: Any,
    **kwargs: Any,
) -> Any:
    if stream:
        stream_manager = client._original.messages.stream(
            model=model,
            messages=messages,
            **kwargs,
        )
        client._storage.save(metrics)
        return stream_manager

    response = client._original.messages.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    if hasattr(response, "usage") and response.usage:
        metrics.tokens_output = response.usage.output_tokens
        if hasattr(response.usage, "cache_read_input_tokens"):
            metrics.cached_tokens = response.usage.cache_read_input_tokens

    client._storage.save(metrics)
    return response


_ClientTransport = Callable[..., Any]
_CLIENT_TRANSPORTS: dict[str, _ClientTransport] = {
    "anthropic": _call_anthropic_transport,
    "openai": _call_openai_transport,
}


def _is_anthropic_auth(headers: Mapping[str, str]) -> bool:
    authorization = headers.get("authorization") or headers.get("Authorization") or ""
    user_agent = headers.get("user-agent") or headers.get("User-Agent") or ""
    return bool(
        headers.get("x-api-key")
        or headers.get("anthropic-version")
        or authorization.startswith("Bearer sk-ant-")
        or _is_claude_code_client(user_agent)
    )


def _is_claude_code_client(user_agent: str) -> bool:
    """Return True for Claude Code/Claude CLI requests using Anthropic gateway auth."""
    normalized = user_agent.lower()
    return "claude-code/" in normalized or "claude-cli/" in normalized
