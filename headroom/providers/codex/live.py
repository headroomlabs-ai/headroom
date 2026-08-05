"""Transparent Codex Live WebSocket transport."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Mapping
from typing import Any, cast

from fastapi import WebSocket

from headroom.copilot_auth import apply_copilot_api_auth, build_copilot_upstream_url
from headroom.providers.codex.endpoints import codex_backend_ws_url
from headroom.providers.codex.runtime import resolve_codex_routing
from headroom.proxy.handlers.openai import _is_allowed_websocket_origin
from headroom.proxy.helpers import _strip_internal_headers, merge_extra_headers
from headroom.proxy.ws_headers import WS_HOP_BY_HOP_HEADERS

logger = logging.getLogger("headroom.providers.codex.live")

CODEX_LIVE_ROUTE_PATHS: tuple[str, ...] = (
    "/v1/live",
    "/v1/codex/live",
    "/backend-api/live",
    "/backend-api/codex/live",
)
CODEX_LIVE_WS_PATH_ENV = "HEADROOM_CODEX_LIVE_WS_PATH"
DEFAULT_CODEX_LIVE_WS_PATH = "/live"


def codex_live_ws_path() -> str:
    """Return the correctable, derived Codex backend Live path."""
    configured = os.environ.get(CODEX_LIVE_WS_PATH_ENV, "").strip()
    path = configured or DEFAULT_CODEX_LIVE_WS_PATH
    return path if path.startswith("/") else f"/{path}"


def codex_live_websocket_url(
    *,
    subscription: bool,
    base_url: str,
    path: str = "/v1/live",
    query: str = "",
) -> str:
    """Build the Live WebSocket URL from the selected auth mode."""
    if subscription:
        url = codex_backend_ws_url(codex_live_ws_path())
    else:
        ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
        url = build_copilot_upstream_url(ws_base, path)
    return f"{url}?{query}" if query else url


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    forwarded = {
        key: value for key, value in headers.items() if key.lower() not in WS_HOP_BY_HOP_HEADERS
    }
    return _strip_internal_headers(forwarded)


def _ensure_live_authorization(headers: Mapping[str, str]) -> dict[str, str]:
    """Use the configured OpenAI key when a client omitted authorization."""
    if any(key.lower() == "authorization" for key in headers):
        return dict(headers)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return dict(headers)
    return {**headers, "Authorization": f"Bearer {api_key}"}


_NON_WIRE_CLOSE_CODES = {1004, 1005, 1006, 1015}


def _close_info(source: object, default_code: int) -> tuple[int, str]:
    received = getattr(source, "rcvd", None)
    if isinstance(source, Mapping):
        code = source.get("code")
        reason = source.get("reason")
    else:
        code = getattr(source, "close_code", None)
        reason = getattr(source, "close_reason", None)
    raw_code: object = code or getattr(received, "code", None)
    try:
        code = int(str(raw_code))
    except (TypeError, ValueError):
        code = default_code
    if code < 1000 or code > 4999 or code in _NON_WIRE_CLOSE_CODES:
        code = default_code

    reason = reason or getattr(received, "reason", None) or ""
    return code, str(reason)[:120]


async def handle_codex_live_websocket(
    websocket: WebSocket,
    proxy: object,
    openai_base_url: str,
    inbound_path: str,
) -> None:
    """Relay Live text and binary frames without parsing or transforming them."""
    client_headers = dict(websocket.headers)
    if not _is_allowed_websocket_origin(client_headers):
        await websocket.close(code=1008, reason="origin not allowed")
        return

    try:
        import websockets
    except ImportError:
        logger.exception("Codex Live relay unavailable because websockets is not installed")
        await websocket.accept()
        await websocket.close(
            code=1011, reason="websockets package not installed; pip install websockets"
        )
        return

    forwarded_headers = _forward_headers(client_headers)
    decision = resolve_codex_routing(forwarded_headers)
    forwarded_headers = decision.headers
    websocket_url = getattr(websocket, "url", None)
    query = str(getattr(websocket_url, "query", "") or "")
    upstream_url = codex_live_websocket_url(
        subscription=decision.is_chatgpt_auth,
        base_url=openai_base_url,
        path=inbound_path,
        query=query,
    )
    forwarded_headers = await apply_copilot_api_auth(forwarded_headers, url=upstream_url)
    config = getattr(proxy, "config", None)
    forwarded_headers = merge_extra_headers(
        forwarded_headers,
        getattr(config, "openai_extra_headers", None),
    )
    if not any(key.lower() == "authorization" for key in forwarded_headers):
        if os.environ.get("OPENAI_API_KEY", "").strip():
            forwarded_headers = _ensure_live_authorization(forwarded_headers)
            logger.debug("Codex Live injected Authorization from OPENAI_API_KEY")
        else:
            logger.warning("Codex Live has no Authorization header or OPENAI_API_KEY")

    raw_protocols = next(
        (value for key, value in client_headers.items() if key.lower() == "sec-websocket-protocol"),
        "",
    )
    subprotocols = [value.strip() for value in raw_protocols.split(",") if value.strip()]
    upstream = None
    relay_tasks: set[asyncio.Task[None]] = set()
    accepted = False
    client_disconnected = False
    close_code = 1000
    close_reason = ""
    try:
        try:
            upstream = await websockets.connect(
                upstream_url,
                additional_headers=forwarded_headers,
                # websockets types wire protocol tokens as a nominal wrapper.
                subprotocols=cast(Any, subprotocols or None),
                ssl=True if upstream_url.startswith("wss://") else None,
                open_timeout=max(30, getattr(config, "connect_timeout_seconds", 10) * 3),
                close_timeout=10,
                ping_interval=20,
                ping_timeout=None,
                max_size=None,
            )
        except Exception:
            logger.exception("Codex Live upstream handshake failed url=%s", upstream_url)
            await websocket.close(code=1011, reason="upstream connection failed")
            return

        selected_subprotocol = getattr(upstream, "subprotocol", None)
        if selected_subprotocol not in subprotocols:
            selected_subprotocol = None
        await websocket.accept(subprotocol=selected_subprotocol)
        accepted = True

        async def client_to_upstream() -> None:
            nonlocal client_disconnected, close_code, close_reason
            while True:
                message = await websocket.receive()
                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    client_disconnected = True
                    close_code, close_reason = _close_info(message, 1000)
                    with contextlib.suppress(Exception):
                        await upstream.close(code=close_code, reason=close_reason)
                    return
                if message_type != "websocket.receive":
                    continue
                if message.get("text") is not None:
                    await upstream.send(message["text"])
                elif message.get("bytes") is not None:
                    await upstream.send(message["bytes"])

        async def upstream_to_client() -> None:
            nonlocal close_code, close_reason
            try:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                close_code, close_reason = _close_info(exc, 1011)
                logger.warning(
                    "Codex Live upstream relay failed code=%s reason=%s error=%s",
                    close_code,
                    close_reason,
                    type(exc).__name__,
                )
            else:
                close_code, close_reason = _close_info(upstream, 1000)

        relay_tasks = {
            asyncio.create_task(client_to_upstream()),
            asyncio.create_task(upstream_to_client()),
        }
        done, _ = await asyncio.wait(
            relay_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            with contextlib.suppress(asyncio.CancelledError):
                error = task.exception()
                if error is not None:
                    close_code, close_reason = _close_info(error, 1011)
                    logger.warning(
                        "Codex Live relay task failed task=%s code=%s reason=%s error=%s",
                        task.get_name(),
                        close_code,
                        close_reason,
                        type(error).__name__,
                    )
    except asyncio.CancelledError:
        raise
    except Exception:
        close_code, close_reason = 1011, "relay failed"
        logger.exception("Codex Live relay failed url=%s", upstream_url)
    finally:
        pending = [task for task in relay_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if upstream is not None:
            with contextlib.suppress(Exception):
                await upstream.close()
        if accepted and not client_disconnected:
            with contextlib.suppress(Exception):
                await websocket.close(code=close_code, reason=close_reason)
