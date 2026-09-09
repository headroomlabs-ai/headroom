"""Provider runtime registry and transport helpers (Anthropic-focused build)."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from headroom.providers.claude import DEFAULT_API_URL as DEFAULT_ANTHROPIC_API_URL
from headroom.proxy.upstream_guard import is_safe_upstream_url

#: Upstream used for OpenAI-format pipeline provider objects (tokenizer choice,
#: /v1/models metadata). Claude Code never sends traffic here.
DEFAULT_OPENAI_API_URL = "https://api.openai.com"

if TYPE_CHECKING:
    from headroom.providers.base import Provider


@dataclass(frozen=True)
class ProviderApiOverrides:
    """Optional upstream API URL overrides configured for the proxy."""

    anthropic: str | None = None
    openai: str | None = None


@dataclass(frozen=True)
class ProviderApiTargets:
    """Resolved upstream API targets after provider normalization."""

    anthropic: str = DEFAULT_ANTHROPIC_API_URL
    openai: str = DEFAULT_OPENAI_API_URL


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
        if headers.get("api-key"):
            azure_base = headers.get("x-headroom-base-url", "")
            # Same SSRF guard as `proxy_targets.select_passthrough_base_url`;
            # both resolve a caller-named upstream (CVE-2026-77775).
            if azure_base and is_safe_upstream_url(azure_base):
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
    openai_api_url: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProviderApiOverrides:
    """Resolve provider API URL overrides from CLI/config inputs and environment."""
    env = environ or os.environ
    return ProviderApiOverrides(
        anthropic=anthropic_api_url or env.get("ANTHROPIC_TARGET_API_URL"),
        openai=openai_api_url or env.get("OPENAI_TARGET_API_URL"),
    )


def resolve_extra_headers(
    cli_value: str | None,
    env_var: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Resolve extra headers to merge into (and override) forwarded provider requests.

    Accepts a JSON object string from CLI or env (CLI wins). Returns ``None`` if unset.
    Raises ``ValueError`` on invalid JSON or a non-string-keyed/valued object.
    """
    env = environ or os.environ
    raw = cli_value or env.get(env_var)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{env_var} must be a JSON object of header name/value strings") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ValueError(f"{env_var} must be a JSON object of header name/value strings")
    return parsed or None


def resolve_api_targets(overrides: ProviderApiOverrides) -> ProviderApiTargets:
    """Resolve normalized upstream provider targets from configured overrides."""
    return ProviderApiTargets(
        anthropic=_normalize_api_url(overrides.anthropic, default=DEFAULT_ANTHROPIC_API_URL),
        openai=_normalize_api_url(overrides.openai, default=DEFAULT_OPENAI_API_URL),
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
    logger: logging.Logger,
    **_ignored: Any,
) -> None:
    """Translated backends (LiteLLM / any-llm / Bedrock) were removed from this build.

    Only the direct Anthropic upstream is supported. Any other ``backend`` value
    logs a warning and falls back to direct Anthropic.
    """
    if backend != "anthropic":
        logger.warning(
            "backend %r is not available in this build; using direct Anthropic upstream",
            backend,
        )
    return None


def format_backend_status(*, backend: str, **_ignored: Any) -> str:
    """Build the human-readable backend status string shown in CLI/server output."""
    if backend == "anthropic":
        return "ANTHROPIC (direct API)"
    return f"{backend} (unavailable in this build — direct Anthropic used)"


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
