"""Redacted gateway control-plane value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from headroom.proxy.gateway.runtime import GatewayRuntime


@dataclass(frozen=True, slots=True)
class RedactedGatewayStatus:
    service: str
    profile: str
    generation: int
    ready: bool
    route_count: int
    credential_count: int
    config_digest: str = ""
    catalog_revision: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "service": self.service,
            "profile": self.profile,
            "generation": self.generation,
            "ready": self.ready,
            "route_count": self.route_count,
            "credential_count": self.credential_count,
            "config_digest": self.config_digest,
            "catalog_revision": self.catalog_revision,
        }


def install_gateway_controls(app: FastAPI, runtime: GatewayRuntime) -> None:
    @app.api_route("/admin/gateway/{operation:path}", methods=["GET", "POST"])
    async def control(request: Request, operation: str) -> JSONResponse:
        principal = request.state.gateway_principal
        client_host = request.client.host if request.client else ""
        if client_host not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse(
                status_code=403, content={"error": {"code": "gateway_loopback_required"}}
            )
        if "admin" not in principal.scopes:
            return JSONResponse(
                status_code=403, content={"error": {"code": "gateway_scope_forbidden"}}
            )
        if operation == "status" and request.method == "GET":
            return JSONResponse(
                {**runtime.status().as_dict(), "accounts": runtime.account_status()}
            )
        if request.method != "POST" or operation not in {"reload", "revoke", "catalog/refresh"}:
            return JSONResponse(
                status_code=404, content={"error": {"code": "gateway_control_unavailable"}}
            )
        body = await request.body()
        if len(body) > 4096:
            return JSONResponse(
                status_code=400, content={"error": {"code": "gateway_request_invalid"}}
            )
        try:
            payload = await request.json() if body else {}
            if not isinstance(payload, dict):
                raise ValueError("invalid control input")
            if operation != "revoke" and payload:
                raise ValueError("control does not accept input")
            if operation == "catalog/refresh":
                return JSONResponse(await runtime.refresh_catalog())
            if operation == "reload":
                result = await runtime.reload()
            else:
                if set(payload) - {"principal_id", "route_id", "account_id"} or any(
                    not isinstance(v, str) for v in payload.values()
                ):
                    raise ValueError("invalid revocation selector")
                result = await runtime.revoke(**payload)
            return JSONResponse(
                status_code=200 if result.applied else 409,
                content={
                    "applied": result.applied,
                    "generation": result.generation,
                    "error": result.error,
                },
            )
        except ValueError:
            return JSONResponse(
                status_code=400, content={"error": {"code": "gateway_request_invalid"}}
            )
