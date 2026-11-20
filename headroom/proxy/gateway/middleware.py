"""FastAPI integration for gateway authentication."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from headroom.proxy.gateway.auth import validate_gateway_browser_request
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.runtime import GatewayRuntime

if TYPE_CHECKING:
    from headroom.proxy.gateway.config import GatewayConfigSnapshot


_UNPRIVILEGED_LOCAL_PATHS = frozenset({"/livez", "/readyz"})
_DISABLED_CONTROL_PATHS = frozenset(
    {
        "/admin/runtime-env",
        "/settings",
        "/settings/apply",
        "/settings/schema",
        "/dashboard/settings",
    }
)


def install_gateway_auth_middleware(
    app: FastAPI,
    snapshot: GatewayConfigSnapshot,
    environ: Mapping[str, str],
    config_path: Path | None = None,
) -> None:
    """Install mandatory gateway auth while leaving readiness locally observable."""

    runtime = GatewayRuntime(snapshot, environ=environ, config_path=config_path)
    app.state.gateway_runtime = runtime
    from headroom.proxy.gateway.control import install_gateway_controls

    install_gateway_controls(app, runtime)

    @app.middleware("http")
    async def gateway_authentication(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path in _DISABLED_CONTROL_PATHS:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        if request.url.path in _UNPRIVILEGED_LOCAL_PATHS:
            return await call_next(request)
        try:
            validate_gateway_browser_request(request.headers)
            generation = runtime.capture()
            request.state.gateway_generation = generation
            request.state.gateway_principal = generation.authenticator.authenticate(
                request.headers,
                query_string=request.scope.get("query_string", b""),
            )
        except GatewayPublicError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "type": "gateway_error",
                        "code": exc.code,
                        "message": exc.message,
                    }
                },
            )
        return await call_next(request)
