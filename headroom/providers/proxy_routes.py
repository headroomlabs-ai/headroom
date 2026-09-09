# mypy: disable-error-code=no-untyped-def
"""Provider-specific proxy route registration (Anthropic only)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from headroom.providers.model_metadata import (
    MODEL_METADATA_LIST_ENDPOINT,
    handle_model_metadata_endpoint,
    model_metadata_get_endpoint,
)
from headroom.providers.proxy_targets import (
    api_target as _api_target,
)
from headroom.providers.proxy_targets import (
    select_passthrough_base_url as _select_passthrough_base_url,
)
from headroom.providers.route_specs import (
    PROVIDER_HANDLER_ROUTES,
    PROVIDER_PASSTHROUGH_ROUTES,
    ProviderHandlerRoute,
    ProviderPassthroughRoute,
)
from headroom.proxy.passthrough import (
    custom_base_passthrough_telemetry as _custom_base_passthrough_telemetry,
)
from headroom.proxy.upstream_guard import is_safe_upstream_url_async

logger = logging.getLogger("headroom.proxy.routes")


def _register_provider_passthrough_route(
    app: FastAPI,
    proxy: Any,
    spec: ProviderPassthroughRoute,
) -> None:
    async def provider_passthrough(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, spec.provider_name),
            spec.sub_path,
            spec.provider_name,
        )

    provider_passthrough.__name__ = (
        f"{spec.provider_name}_{spec.sub_path.replace('/', '_')}_{spec.method.lower()}_passthrough"
    )
    app.api_route(spec.path, methods=[spec.method])(provider_passthrough)


def _register_provider_passthrough_routes(app: FastAPI, proxy: Any) -> None:
    for spec in PROVIDER_PASSTHROUGH_ROUTES:
        _register_provider_passthrough_route(app, proxy, spec)


def _register_provider_handler_route(app: FastAPI, proxy: Any, spec: ProviderHandlerRoute) -> None:
    async def provider_handler(request: Request):
        handler = getattr(proxy, spec.handler_name)
        return await handler(request)

    provider_handler.__name__ = (
        spec.handler_name.replace("handle_", "")
        + "_"
        + spec.method.lower()
        + "_"
        + spec.path.strip("/").replace("/", "_").replace("-", "_")
    )
    app.api_route(spec.path, methods=[spec.method])(provider_handler)


def _register_provider_handler_routes(app: FastAPI, proxy: Any) -> None:
    for spec in PROVIDER_HANDLER_ROUTES:
        _register_provider_handler_route(app, proxy, spec)


def register_provider_routes(app: FastAPI, proxy: Any) -> None:
    """Register provider-specific proxy endpoints."""

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        # Honor the per-request upstream override so clients that speak the
        # Anthropic Messages wire format but authenticate against a
        # non-Anthropic gateway route correctly.
        custom_base = request.headers.get("x-headroom-base-url", "").strip()
        if custom_base:
            if not await is_safe_upstream_url_async(custom_base):
                logger.warning("rejecting unsafe x-headroom-base-url: %r", custom_base)
                raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
            return await proxy.handle_anthropic_messages(
                request, upstream_base_url=custom_base.rstrip("/")
            )
        return await proxy.handle_anthropic_messages(request)

    _register_provider_handler_routes(app, proxy)

    @app.get("/v1/models")
    async def list_models(request: Request):
        provider_name = proxy.provider_runtime.model_metadata_provider(dict(request.headers))
        return await handle_model_metadata_endpoint(
            proxy,
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT,
            provider_api_base_url=_api_target(proxy, provider_name),
            provider_name=provider_name,
        )

    @app.get("/v1/models/{model_id}")
    async def get_model(request: Request, model_id: str):
        provider_name = proxy.provider_runtime.model_metadata_provider(dict(request.headers))
        return await handle_model_metadata_endpoint(
            proxy,
            request,
            endpoint=model_metadata_get_endpoint(model_id),
            provider_api_base_url=_api_target(proxy, provider_name),
            provider_name=provider_name,
        )

    _register_provider_passthrough_routes(app, proxy)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
    async def passthrough(request: Request, path: str):
        custom_base = request.headers.get("x-headroom-base-url")
        if custom_base:
            if not await is_safe_upstream_url_async(custom_base):
                logger.warning("rejecting unsafe x-headroom-base-url: %r", custom_base)
                raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
            base_url = custom_base.rstrip("/")
            endpoint_name, provider_name = _custom_base_passthrough_telemetry(
                request.method,
                path,
                base_url,
            )
            return await proxy.handle_passthrough(
                request,
                base_url,
                endpoint_name,
                provider_name,
            )

        return await proxy.handle_passthrough(
            request,
            _select_passthrough_base_url(proxy, dict(request.headers), request.url.path),
        )
