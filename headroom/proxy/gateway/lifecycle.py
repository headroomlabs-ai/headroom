"""Gateway server shutdown starts with runtime drain, before Uvicorn closes peers."""

from __future__ import annotations

import socket

import uvicorn

from headroom.proxy.gateway.runtime import GatewayRuntime


class GatewayServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, *, runtime: GatewayRuntime) -> None:
        super().__init__(config)
        self.runtime = runtime

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        # Uvicorn normally closes WebSockets before the ASGI lifespan shutdown,
        # and waits for HTTP requests before it reaches that lifespan. Start the
        # gateway's own drain here so its configured deadlines govern both.
        await self.runtime.shutdown()
        await super().shutdown(sockets)
