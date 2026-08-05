"""Declarative route specifications for provider passthrough endpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderPassthroughRoute:
    """A direct provider passthrough route handled by ``HeadroomProxy.handle_passthrough``."""

    method: str
    path: str
    provider_name: str
    sub_path: str


@dataclass(frozen=True, slots=True)
class ProviderHandlerRoute:
    """A route that delegates directly to a named proxy handler."""

    method: str
    path: str
    handler_name: str
    path_param: str | None = None


ANTHROPIC_PASSTHROUGH_ROUTES: tuple[ProviderPassthroughRoute, ...] = (
    ProviderPassthroughRoute("POST", "/v1/messages/count_tokens", "anthropic", "count_tokens"),
)


OPENAI_PASSTHROUGH_ROUTES: tuple[ProviderPassthroughRoute, ...] = (
    ProviderPassthroughRoute("POST", "/v1/embeddings", "openai", "embeddings"),
    ProviderPassthroughRoute("POST", "/v1/moderations", "openai", "moderations"),
    ProviderPassthroughRoute("POST", "/v1/audio/transcriptions", "openai", "audio/transcriptions"),
    ProviderPassthroughRoute("POST", "/v1/audio/speech", "openai", "audio/speech"),
)


GEMINI_PASSTHROUGH_ROUTES: tuple[ProviderPassthroughRoute, ...] = (
    ProviderPassthroughRoute("GET", "/v1beta/models", "gemini", "models"),
    ProviderPassthroughRoute("GET", "/v1beta/models/{model_name}", "gemini", "models"),
    ProviderPassthroughRoute(
        "POST", "/v1beta/models/{model}:embedContent", "gemini", "embedContent"
    ),
    ProviderPassthroughRoute(
        "POST",
        "/v1beta/models/{model}:batchEmbedContents",
        "gemini",
        "batchEmbedContents",
    ),
    ProviderPassthroughRoute("POST", "/v1beta/cachedContents", "gemini", "cachedContents"),
    ProviderPassthroughRoute("GET", "/v1beta/cachedContents", "gemini", "cachedContents"),
    ProviderPassthroughRoute(
        "GET",
        "/v1beta/cachedContents/{cache_id}",
        "gemini",
        "cachedContents",
    ),
    ProviderPassthroughRoute(
        "DELETE",
        "/v1beta/cachedContents/{cache_id}",
        "gemini",
        "cachedContents",
    ),
)


PROVIDER_PASSTHROUGH_ROUTES: tuple[ProviderPassthroughRoute, ...] = (
    *ANTHROPIC_PASSTHROUGH_ROUTES,
    *OPENAI_PASSTHROUGH_ROUTES,
    *GEMINI_PASSTHROUGH_ROUTES,
)


ANTHROPIC_HANDLER_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    ProviderHandlerRoute("POST", "/v1/messages", "handle_anthropic_messages"),
)


ANTHROPIC_BATCH_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    ProviderHandlerRoute("POST", "/v1/messages/batches", "handle_anthropic_batch_create"),
    ProviderHandlerRoute("GET", "/v1/messages/batches", "handle_anthropic_batch_passthrough"),
    ProviderHandlerRoute(
        "GET",
        "/v1/messages/batches/{batch_id}",
        "handle_anthropic_batch_passthrough",
        "batch_id",
    ),
    ProviderHandlerRoute(
        "GET",
        "/v1/messages/batches/{batch_id}/results",
        "handle_anthropic_batch_results",
        "batch_id",
    ),
    ProviderHandlerRoute(
        "POST",
        "/v1/messages/batches/{batch_id}/cancel",
        "handle_anthropic_batch_passthrough",
        "batch_id",
    ),
)


OPENAI_HANDLER_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    ProviderHandlerRoute("POST", "/v1/chat/completions", "handle_openai_chat"),
)


OPENAI_BATCH_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    ProviderHandlerRoute("POST", "/v1/batches", "handle_batch_create"),
    ProviderHandlerRoute("GET", "/v1/batches", "handle_batch_list"),
    ProviderHandlerRoute("GET", "/v1/batches/{batch_id}", "handle_batch_get", "batch_id"),
    ProviderHandlerRoute(
        "POST",
        "/v1/batches/{batch_id}/cancel",
        "handle_batch_cancel",
        "batch_id",
    ),
)


GEMINI_HANDLER_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    ProviderHandlerRoute(
        "POST",
        "/v1beta/models/{model}:generateContent",
        "handle_gemini_generate_content",
        "model",
    ),
    ProviderHandlerRoute(
        "POST",
        "/v1beta/models/{model}:streamGenerateContent",
        "handle_gemini_stream_generate_content",
        "model",
    ),
    ProviderHandlerRoute(
        "POST",
        "/v1beta/models/{model}:countTokens",
        "handle_gemini_count_tokens",
        "model",
    ),
)


GEMINI_BATCH_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    ProviderHandlerRoute(
        "POST",
        "/v1beta/models/{model}:batchGenerateContent",
        "handle_google_batch_create",
        "model",
    ),
    ProviderHandlerRoute(
        "GET",
        "/v1beta/batches/{batch_name}",
        "handle_google_batch_results",
        "batch_name",
    ),
    ProviderHandlerRoute(
        "POST",
        "/v1beta/batches/{batch_name}:cancel",
        "handle_google_batch_passthrough",
        "batch_name",
    ),
    ProviderHandlerRoute(
        "DELETE",
        "/v1beta/batches/{batch_name}",
        "handle_google_batch_passthrough",
        "batch_name",
    ),
)


CLOUDCODE_HANDLER_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    ProviderHandlerRoute(
        "POST",
        "/v1internal:streamGenerateContent",
        "handle_google_cloudcode_stream",
    ),
    ProviderHandlerRoute(
        "POST",
        "/v1/v1internal:streamGenerateContent",
        "handle_google_cloudcode_stream",
    ),
)


PROVIDER_HANDLER_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    *ANTHROPIC_HANDLER_ROUTES,
    *ANTHROPIC_BATCH_ROUTES,
    *OPENAI_HANDLER_ROUTES,
    *OPENAI_BATCH_ROUTES,
    *GEMINI_HANDLER_ROUTES,
    *GEMINI_BATCH_ROUTES,
    *CLOUDCODE_HANDLER_ROUTES,
)
