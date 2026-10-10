"""Provider runtime registry and transport helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlparse

from headroom.providers.claude import DEFAULT_API_URL as DEFAULT_ANTHROPIC_API_URL
from headroom.providers.codex import DEFAULT_API_URL as DEFAULT_OPENAI_API_URL
from headroom.providers.gemini import DEFAULT_API_URL as DEFAULT_GEMINI_API_URL
from headroom.proxy.upstream_guard import is_safe_upstream_url

DEFAULT_CLOUDCODE_API_URL = "https://cloudcode-pa.googleapis.com"
DEFAULT_VERTEX_API_URL = "https://us-central1-aiplatform.googleapis.com"

if TYPE_CHECKING:
    from headroom.backends.base import Backend
    from headroom.providers.base import Provider

AnyLLMBackendType: Any = None
LiteLLMBackendType: Any = None

_UPSTREAM_ROUTES_ENV = "HEADROOM_UPSTREAM_ROUTES"
_route_logger = logging.getLogger("headroom.proxy.routes")


@dataclass(frozen=True)
class PassthroughAuth:
    """Forward the inbound Authorization / x-api-key headers verbatim."""


@dataclass(frozen=True)
class BearerAuth:
    """Replace inbound auth with a bearer token sourced at request time."""

    env_var: str


AuthMode = PassthroughAuth | BearerAuth


class UpstreamAuthUnavailable(Exception):
    """A matched BearerAuth route has no token in its configured env var.

    Raised by the proxy's ``resolve_upstream`` so call sites fail closed
    (return 502 / a WS error event) *before* contacting the upstream rather
    than forwarding the request with inbound auth stripped and no
    replacement ``Authorization`` header. Carries the offending ``env_var``
    and the request ``model`` for logging; the message is intentionally
    generic so it can be surfaced to clients without leaking the env-var
    name.
    """

    def __init__(self, env_var: str, model: str | None) -> None:
        self.env_var = env_var
        self.model = model
        super().__init__(
            f"upstream route auth env var {env_var!r} is empty (model={model!r}); failing closed"
        )


class UpstreamRoutesConfigError(ValueError):
    """A non-empty HEADROOM_UPSTREAM_ROUTES value is unsafe or malformed."""


@dataclass(frozen=True)
class UpstreamRoute:
    """One entry in the HEADROOM_UPSTREAM_ROUTES table.

    ``model_prefix`` is matched as a case-insensitive prefix of the request's
    ``model`` field; ``"*"`` is the catch-all fallback. ``upstream`` overrides
    the base URL; if omitted, the route's ``protocol`` slot selects the default
    ``api_targets`` URL. ``auth`` selects passthrough vs. an env-sourced bearer.
    """

    model_prefix: str
    upstream: str | None = None
    auth: AuthMode = field(default_factory=PassthroughAuth)
    protocol: Literal["openai", "anthropic"] = "openai"


@dataclass(frozen=True)
class UpstreamResolution:
    """The result of resolve_upstream(): where to send the request and how."""

    base_url: str
    auth: AuthMode


def _parse_auth(spec: str) -> AuthMode:
    if spec == "passthrough":
        return PassthroughAuth()
    if spec.startswith("env:"):
        env_var = spec[len("env:") :]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_var):
            return BearerAuth(env_var=env_var)
        raise UpstreamRoutesConfigError("route auth env name is empty or invalid")
    raise UpstreamRoutesConfigError("route auth must be 'passthrough' or 'env:VARNAME'")


def parse_upstream_routes(env: Mapping[str, str]) -> tuple[UpstreamRoute, ...]:
    """Parse HEADROOM_UPSTREAM_ROUTES (JSON array) into an immutable tuple.

    Empty / unset returns an empty tuple (legacy single-upstream behavior).
    Invalid non-empty configuration fails closed during proxy startup so an
    operator typo cannot silently send credentials through the legacy route.
    """
    raw = env.get(_UPSTREAM_ROUTES_ENV, "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UpstreamRoutesConfigError(
            f"HEADROOM_UPSTREAM_ROUTES contains invalid JSON at position {exc.pos}"
        ) from exc
    if not isinstance(parsed, list):
        raise UpstreamRoutesConfigError("HEADROOM_UPSTREAM_ROUTES must be a JSON array")
    if not parsed:
        raise UpstreamRoutesConfigError("HEADROOM_UPSTREAM_ROUTES must contain at least one route")
    routes: list[UpstreamRoute] = []
    for idx, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise UpstreamRoutesConfigError(f"HEADROOM_UPSTREAM_ROUTES[{idx}] must be an object")
        raw_prefix = entry.get("model_prefix")
        prefix = raw_prefix.strip() if isinstance(raw_prefix, str) else ""
        if not prefix:
            raise UpstreamRoutesConfigError(
                f"HEADROOM_UPSTREAM_ROUTES[{idx}].model_prefix must be a non-empty string"
            )
        protocol = entry.get("protocol", "openai")
        if protocol not in ("openai", "anthropic"):
            raise UpstreamRoutesConfigError(
                f"HEADROOM_UPSTREAM_ROUTES[{idx}].protocol must be 'openai' or 'anthropic'"
            )
        upstream = entry.get("upstream")
        if upstream is not None:
            if not isinstance(upstream, str):
                raise UpstreamRoutesConfigError(
                    f"HEADROOM_UPSTREAM_ROUTES[{idx}].upstream must be an HTTP(S) URL"
                )
            upstream = upstream.rstrip("/") or None
            parsed_upstream = urlparse(upstream or "")
            if (
                parsed_upstream.scheme not in ("http", "https")
                or not parsed_upstream.netloc
                or parsed_upstream.username is not None
                or parsed_upstream.password is not None
                or parsed_upstream.query
                or parsed_upstream.fragment
            ):
                raise UpstreamRoutesConfigError(
                    f"HEADROOM_UPSTREAM_ROUTES[{idx}].upstream must be an HTTP(S) base URL without embedded credentials, query, or fragment"
                )
        auth_spec = entry.get("auth", "passthrough")
        if not isinstance(auth_spec, str):
            raise UpstreamRoutesConfigError(
                f"HEADROOM_UPSTREAM_ROUTES[{idx}].auth must be a string"
            )
        try:
            auth = _parse_auth(auth_spec)
        except UpstreamRoutesConfigError as exc:
            raise UpstreamRoutesConfigError(f"HEADROOM_UPSTREAM_ROUTES[{idx}]: {exc}") from exc
        routes.append(
            UpstreamRoute(
                model_prefix=prefix.lower(),
                upstream=upstream,
                auth=auth,
                protocol=protocol,  # type: ignore[arg-type]
            )
        )
    return tuple(routes)


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
    upstream_routes: tuple[UpstreamRoute, ...] = ()

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

    def select_passthrough_base_url(
        self, headers: Mapping[str, str], path: str | None = None
    ) -> str:
        """Resolve the upstream base URL for catch-all passthrough requests."""
        if path is not None:
            from headroom.providers.proxy_targets import is_anthropic_hello_path

            if is_anthropic_hello_path(path):
                return self.api_targets.anthropic
        if _is_anthropic_auth(headers):
            return self.api_targets.anthropic
        if headers.get("x-goog-api-key"):
            return self.api_targets.gemini
        if headers.get("api-key"):
            azure_base = headers.get("x-headroom-base-url", "")
            # Same SSRF guard as `proxy_targets.select_passthrough_base_url`;
            # both resolve a caller-named upstream (CVE-2026-77775).
            if azure_base and is_safe_upstream_url(azure_base):
                return azure_base.rstrip("/")
        return self.api_targets.openai

    def match_upstream_route(self, model: str | None) -> UpstreamRoute | None:
        """Return the first configured route matching this request model."""
        lowered = (model or "").lower()
        for route in self.upstream_routes:
            if route.model_prefix == "*" or lowered.startswith(route.model_prefix):
                return route
        return None

    def resolve_upstream(
        self,
        *,
        protocol: str,
        model: str | None,
        headers: Mapping[str, str],
        default_url: str | None = None,
    ) -> UpstreamResolution:
        """Resolve the per-request upstream base URL and auth mode.

        With no ``upstream_routes`` configured, returns the legacy resolution
        (``select_passthrough_base_url`` + passthrough auth) so behavior is
        byte-identical to a build without this feature.
        """
        if not self.upstream_routes:
            return UpstreamResolution(
                base_url=default_url or self.select_passthrough_base_url(headers),
                auth=PassthroughAuth(),
            )
        matched = self.match_upstream_route(model)
        if matched is None:
            return UpstreamResolution(
                base_url=default_url or self.select_passthrough_base_url(headers),
                auth=PassthroughAuth(),
            )
        if matched.upstream:
            base_url = matched.upstream
        else:
            try:
                base_url = self.api_target(matched.protocol)
            except KeyError:
                base_url = self.api_target(protocol)
        return UpstreamResolution(base_url=base_url, auth=matched.auth)


def _normalize_api_url(url: str | None, *, default: str) -> str:
    if not url:
        return default

    normalized = url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized


def _log_backend_init_failure(
    logger: logging.Logger,
    *,
    backend: str,
    provider: str,
    exc: Exception,
) -> None:
    logger.error(
        "backend initialization failed: backend=%s provider=%s error=%s",
        backend,
        provider,
        exc,
    )


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
    from headroom.copilot_auth import is_copilot_upstream_url

    openai = _normalize_api_url(overrides.openai, default=DEFAULT_OPENAI_API_URL)

    # GitHub Copilot serves BOTH its OpenAI surface (``/chat/completions``,
    # ``/responses``) and its Anthropic surface (``/v1/messages``, for Claude
    # models) from the same host. When the OpenAI target is a Copilot host
    # (``wrap copilot --subscription`` / ``wrap vscode`` both point it there so
    # GPT models work) but no Anthropic target was set, Claude-model requests
    # fell back to ``DEFAULT_ANTHROPIC_API_URL`` (api.anthropic.com) and 401'd
    # with the Copilot bearer — "Invalid bearer token" (#3247). Default the
    # Anthropic target to the same Copilot host so those requests reach the
    # surface that actually serves them. An explicit ``ANTHROPIC_TARGET_API_URL``
    # still wins (only a ``None`` override is filled in here).
    anthropic_override = overrides.anthropic
    if anthropic_override is None and is_copilot_upstream_url(openai):
        anthropic_override = openai

    return ProviderApiTargets(
        anthropic=_normalize_api_url(anthropic_override, default=DEFAULT_ANTHROPIC_API_URL),
        openai=openai,
        gemini=_normalize_api_url(overrides.gemini, default=DEFAULT_GEMINI_API_URL),
        cloudcode=_normalize_api_url(overrides.cloudcode, default=DEFAULT_CLOUDCODE_API_URL),
        vertex=_normalize_api_url(overrides.vertex, default=DEFAULT_VERTEX_API_URL),
    )


def build_proxy_provider_runtime(config: Any) -> ProxyProviderRuntime:
    """Build provider runtime objects and resolved targets for the proxy."""
    from headroom.providers.anthropic import AnthropicProvider
    from headroom.providers.openai import OpenAIProvider

    api_targets = resolve_api_targets(config.provider_api_overrides)
    upstream_routes = parse_upstream_routes(os.environ)
    if upstream_routes:
        _route_logger.info("HEADROOM_UPSTREAM_ROUTES: %d route(s) loaded", len(upstream_routes))
    return ProxyProviderRuntime(
        api_targets=api_targets,
        pipeline_providers={
            # warn=False: the proxy pipeline provider intentionally uses tiktoken
            # approximation (no Anthropic client available at this layer).
            "anthropic": AnthropicProvider(warn=False),
            "openai": OpenAIProvider(),
        },
        upstream_routes=upstream_routes,
    )


def create_proxy_backend(
    *,
    backend: str,
    anyllm_provider: str,
    bedrock_region: str | None,
    bedrock_profile: str | None = None,
    logger: logging.Logger,
    openai_api_url: str | None = None,
    anyllm_backend_cls: Any | None = None,
    litellm_backend_cls: Any | None = None,
) -> Backend | None:
    """Create the optional translated backend for Anthropic proxy requests."""
    if backend == "anthropic":
        return None

    if backend == "anyllm" or backend.startswith("anyllm-"):
        provider = anyllm_provider
        backend_name = "anyllm" if backend == "anyllm" else backend
        try:
            backend_cls = anyllm_backend_cls or _load_anyllm_backend()
            instance = cast("Backend", backend_cls(provider=provider, api_base=openai_api_url))
            logger.info("any-llm backend enabled (provider=%s)", provider)
            return instance
        except ImportError as exc:
            logger.warning("any-llm backend not available: %s", exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            _log_backend_init_failure(
                logger,
                backend=backend_name,
                provider=provider,
                exc=exc,
            )
            return None

    normalized_backend = backend if backend.startswith("litellm-") else f"litellm-{backend}"
    provider = normalized_backend.replace("litellm-", "")
    # `litellm-vertex` is the name in our docs/help, but LiteLLM (and our
    # provider registry) keys Google Vertex on `vertex_ai`. Without this alias
    # the provider falls through to a generic pass-through: wrong model prefix
    # (`vertex/…` instead of `vertex_ai/…`), region dropped, auth mishandled.
    if provider in ("vertex", "google-vertex", "googlevertex"):
        provider = "vertex_ai"
    try:
        backend_cls = litellm_backend_cls or _load_litellm_backend()
        instance = cast(
            "Backend",
            backend_cls(provider=provider, region=bedrock_region, profile_name=bedrock_profile),
        )
        logger.info("LiteLLM backend enabled (provider=%s, region=%s)", provider, bedrock_region)
        return instance
    except ImportError as exc:
        logger.warning("LiteLLM backend not available: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_backend_init_failure(
            logger,
            backend=normalized_backend,
            provider=provider,
            exc=exc,
        )
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
