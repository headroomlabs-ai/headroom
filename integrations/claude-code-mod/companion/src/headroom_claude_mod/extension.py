"""Opt-in local conversation metrics, inspection and compression controls."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool

from . import __version__
from .telemetry import IncompatibleLogger, belongs, inspect, record, snapshot, summarize

PREFIX = "/headroom-mod/v1"
WINDOWS = {"all": None, "15m": 900, "1h": 3600, "24h": 86400}


class CompressionControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool


def timestamp(value: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def require_exact_origin(request: Request) -> None:
    """Captured text is readable only by this origin, not other local sites."""
    origin = request.headers.get("origin")
    if origin is None:
        return  # Native clients do not send a browser Origin header.
    try:
        parsed = urlsplit(origin)
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        target_port = request.url.port or (443 if request.url.scheme == "https" else 80)
        allowed = (
            parsed.scheme in {"http", "https"}
            and parsed.username is None
            and parsed.password is None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and (parsed.scheme, parsed.hostname, port)
            == (request.url.scheme, request.url.hostname, target_port)
        )
    except ValueError:
        allowed = False
    if not allowed:
        raise HTTPException(status_code=403, detail="cross-origin request rejected")


def install(app: Any, config: Any) -> None:
    # Reuse Headroom's actual peer + Host/DNS-rebinding + same-origin guards.
    from headroom.proxy.loopback_guard import require_loopback, require_same_origin

    proxy = app.state.proxy
    # Fail loudly during extension install rather than serve made-up zeros.
    snapshot(proxy.logger)
    epoch = str(uuid.uuid4())
    controls: dict[str, dict[str, Any]] = {}

    def control(sid: str) -> dict[str, Any]:
        if sid not in controls:
            # Never evict a paused conversation and silently re-enable compression.
            if len(controls) >= 256:
                raise HTTPException(409, "Conversation control capacity reached; restart the proxy")
            controls[sid] = {"enabled": True, "reset_at": None}
        return controls[sid]

    class ConversationCompressionMiddleware:
        def __init__(self, app: Any) -> None:
            self.app = app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if (
                scope["type"] == "http"
                and scope.get("method") == "POST"
                and scope.get("path", "").endswith("/v1/messages")
            ):
                headers = scope.get("headers", [])
                sessions = [v for k, v in headers if k.lower() == b"x-headroom-mod-session"]
                try:
                    sid = str(uuid.UUID(sessions[0].decode("ascii"))) if len(sessions) == 1 else ""
                except (ValueError, UnicodeError):
                    sid = ""
                if controls.get(sid, {}).get("enabled") is False:
                    scope = {
                        **scope,
                        "headers": [(k, v) for k, v in headers if k.lower() != b"x-headroom-bypass"]
                        + [(b"x-headroom-bypass", b"true")],
                    }
            await self.app(scope, receive, send)

    app.add_middleware(ConversationCompressionMiddleware)
    router = APIRouter(
        prefix=PREFIX,
        dependencies=[
            Depends(require_loopback),
            Depends(require_same_origin),
            Depends(require_exact_origin),
        ],
    )

    def response(payload: dict[str, Any]) -> JSONResponse:
        return JSONResponse(
            payload, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        )

    def entries() -> tuple[Any, ...]:
        try:
            return snapshot(proxy.logger)
        except IncompatibleLogger as exc:
            raise HTTPException(503, "Headroom request-log adapter is incompatible") from exc

    @router.get("/health")
    async def health() -> JSONResponse:
        entries()
        try:
            headroom_version = version("headroom-ai")
        except PackageNotFoundError:
            headroom_version = "source/unknown"
        return response(
            {
                "schema_version": 1,
                "service": "headroom-claude-mod",
                "version": __version__,
                "headroom_version": headroom_version,
                "epoch": epoch,
                "log_full_messages": bool(config.log_full_messages),
                "read_only": False,
                "session_controls": True,
            }
        )

    @router.get("/sessions/{session_id}")
    async def session(
        session_id: uuid.UUID,
        limit: int = Query(100, ge=1, le=100),
        window: Literal["all", "15m", "1h", "24h"] = "all",
    ) -> JSONResponse:
        sid = str(session_id)
        state = controls.get(sid, {"enabled": True, "reset_at": None})
        cutoff = state["reset_at"]
        if WINDOWS[window] is not None:
            cutoff = max(cutoff or 0, time.time() - WINDOWS[window])
        logs = entries()
        unique: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(logs):
            if not belongs(entry, sid):
                continue
            key = getattr(entry, "request_id", None) or f"unidentified-{index}"
            unique.pop(key, None)
            unique[key] = record(entry)
        rows = list(unique.values())
        if cutoff is not None:
            rows = [
                r for r in rows if (ts := timestamp(r["timestamp"])) is not None and ts > cutoff
            ]
        # Retention is proxy-global: other conversations can evict this one's history.
        capacity = proxy.logger._logs.maxlen
        payload = {
            "schema_version": 1,
            "session_id": sid,
            "epoch": epoch,
            "compression_enabled": state["enabled"],
            "window": window,
            "reset_at": state["reset_at"],
            "scope": "conversation_and_inherited_children",
            "basis": "retained_request_totals_not_unique_context_or_lifetime",
            "retention": {
                "capacity": capacity,
                "proxy_records": len(logs),
                "window_full": len(logs) >= capacity,
                "message_window": getattr(proxy.logger, "MESSAGE_WINDOW", None),
            },
            "log_full_messages": bool(config.log_full_messages),
            "totals": summarize(rows),
            "latest": rows[-1] if rows else None,
            "requests": list(reversed(rows[-100:])),
        }
        return response({**payload, "requests": payload["requests"][:limit]})

    def control_response(sid: str, state: dict[str, Any]) -> JSONResponse:
        return response(
            {
                "schema_version": 1,
                "session_id": sid,
                "epoch": epoch,
                "compression_enabled": state["enabled"],
                "reset_at": state["reset_at"],
            }
        )

    @router.post("/sessions/{session_id}/compression")
    async def compression(session_id: uuid.UUID, body: CompressionControl) -> JSONResponse:
        sid = str(session_id)
        state = control(sid)
        state["enabled"] = body.enabled
        return control_response(sid, state)

    @router.post("/sessions/{session_id}/reset")
    async def reset(session_id: uuid.UUID) -> JSONResponse:
        sid = str(session_id)
        state = control(sid)
        state["reset_at"] = time.time()
        return control_response(sid, state)

    @router.get("/sessions/{session_id}/requests/{request_id}")
    async def request_detail(
        session_id: uuid.UUID,
        request_id: str,
        side: Literal["original", "compressed", "diff"] = "compressed",
        message: int = Query(0, ge=0, le=100_000),
        page: int = Query(0, ge=0, le=100_000),
    ) -> JSONResponse:
        if len(request_id) > 128:
            raise HTTPException(404, "Request not retained in this conversation")
        # Filter before touching a body; don't call the global feed and post-filter.
        entry = next(
            (
                e
                for e in reversed(entries())
                if belongs(e, str(session_id)) and getattr(e, "request_id", None) == request_id
            ),
            None,
        )
        if entry is None:
            raise HTTPException(404, "Request not retained in this conversation")
        result = (
            inspect(entry, side=side, message=message, page=page)
            if config.log_full_messages
            else {
                "available": False,
                "reason": "Start Headroom with --log-messages to enable inspection. Capture is never enabled by the mod.",
            }
        )
        return response(
            {
                "schema_version": 1,
                "session_id": str(session_id),
                "epoch": epoch,
                "request_id": request_id,
                **result,
            }
        )

    app.include_router(router)
