from __future__ import annotations

from headroom.providers.route_specs import (
    ANTHROPIC_BATCH_ROUTES,
    ANTHROPIC_HANDLER_ROUTES,
    ANTHROPIC_PASSTHROUGH_ROUTES,
    CLOUDCODE_HANDLER_ROUTES,
    GEMINI_BATCH_ROUTES,
    GEMINI_HANDLER_ROUTES,
    GEMINI_PASSTHROUGH_ROUTES,
    OPENAI_BATCH_ROUTES,
    OPENAI_HANDLER_ROUTES,
    OPENAI_PASSTHROUGH_ROUTES,
    PROVIDER_HANDLER_ROUTES,
    PROVIDER_PASSTHROUGH_ROUTES,
    ProviderHandlerRoute,
    ProviderPassthroughRoute,
)


def test_provider_passthrough_route_specs_are_unique() -> None:
    route_keys = {(spec.method, spec.path) for spec in PROVIDER_PASSTHROUGH_ROUTES}

    assert len(route_keys) == len(PROVIDER_PASSTHROUGH_ROUTES)


def test_provider_handler_route_specs_are_unique() -> None:
    route_keys = {(spec.method, spec.path) for spec in PROVIDER_HANDLER_ROUTES}

    assert len(route_keys) == len(PROVIDER_HANDLER_ROUTES)


def test_openai_passthrough_routes_model_endpoint_intent() -> None:
    assert OPENAI_PASSTHROUGH_ROUTES == (
        ProviderPassthroughRoute("POST", "/v1/embeddings", "openai", "embeddings"),
        ProviderPassthroughRoute("POST", "/v1/moderations", "openai", "moderations"),
        ProviderPassthroughRoute(
            "POST",
            "/v1/audio/transcriptions",
            "openai",
            "audio/transcriptions",
        ),
        ProviderPassthroughRoute("POST", "/v1/audio/speech", "openai", "audio/speech"),
    )


def test_anthropic_passthrough_routes_model_endpoint_intent() -> None:
    assert ANTHROPIC_PASSTHROUGH_ROUTES == (
        ProviderPassthroughRoute(
            "POST",
            "/v1/messages/count_tokens",
            "anthropic",
            "count_tokens",
        ),
    )


def test_gemini_passthrough_routes_model_endpoint_intent() -> None:
    assert (
        ProviderPassthroughRoute(
            "POST",
            "/v1beta/models/{model}:batchEmbedContents",
            "gemini",
            "batchEmbedContents",
        )
        in GEMINI_PASSTHROUGH_ROUTES
    )
    assert (
        ProviderPassthroughRoute(
            "DELETE",
            "/v1beta/cachedContents/{cache_id}",
            "gemini",
            "cachedContents",
        )
        in GEMINI_PASSTHROUGH_ROUTES
    )


def test_direct_handler_routes_model_endpoint_intent() -> None:
    assert ANTHROPIC_HANDLER_ROUTES == (
        ProviderHandlerRoute("POST", "/v1/messages", "handle_anthropic_messages"),
    )
    assert OPENAI_HANDLER_ROUTES == (
        ProviderHandlerRoute("POST", "/v1/chat/completions", "handle_openai_chat"),
    )
    assert GEMINI_HANDLER_ROUTES == (
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
    assert CLOUDCODE_HANDLER_ROUTES == (
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


def test_batch_handler_routes_model_endpoint_intent() -> None:
    assert ANTHROPIC_BATCH_ROUTES[0] == ProviderHandlerRoute(
        "POST",
        "/v1/messages/batches",
        "handle_anthropic_batch_create",
    )
    assert (
        ProviderHandlerRoute(
            "GET",
            "/v1/messages/batches/{batch_id}/results",
            "handle_anthropic_batch_results",
            "batch_id",
        )
        in ANTHROPIC_BATCH_ROUTES
    )
    assert OPENAI_BATCH_ROUTES == (
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
    assert (
        ProviderHandlerRoute(
            "POST",
            "/v1beta/models/{model}:batchGenerateContent",
            "handle_google_batch_create",
            "model",
        )
        in GEMINI_BATCH_ROUTES
    )
    assert (
        ProviderHandlerRoute(
            "DELETE",
            "/v1beta/batches/{batch_name}",
            "handle_google_batch_passthrough",
            "batch_name",
        )
        in GEMINI_BATCH_ROUTES
    )
