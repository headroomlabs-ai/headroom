"""Watchdog for the embedding server sidecar process.

EmbeddingServerWatchdog spawns the embedding server as a separate
multiprocessing.Process (spawn context, same as uvicorn workers) and
monitors it. On unexpected death it restarts with exponential backoff.
After too many crashes it gives up and allows workers to degrade
gracefully (memory features disabled).
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import time
from typing import Any, cast

logger = logging.getLogger(__name__)

# How long to wait for a server process to die cleanly before SIGKILL
_STOP_TIMEOUT_SECONDS = 10.0

# Crash window: consecutive crashes within this many seconds count toward
# the max_restarts counter.
_CRASH_WINDOW_SECONDS = 60.0


def _server_target(socket_path: str, kwargs: dict[str, Any]) -> None:
    """Entry point for the server subprocess.

    Runs in the spawned process. Import happens here (not at module level)
    so the parent process doesn't pay for the ONNX model load.
    """
    from headroom.memory.adapters.embedding_server import run_server

    run_server(socket_path=socket_path, **kwargs)


class EmbeddingServerWatchdog:
    """Spawns and monitors the embedding server sidecar process.

    Usage:
        watchdog = EmbeddingServerWatchdog("/tmp/headroom-embed-8787.sock")
        await watchdog.start()
        # ... proxy runs ...
        await watchdog.stop()
    """

    def __init__(
        self,
        socket_path: str,
        restart_delay: float = 1.0,
        max_restarts: int = 10,
        **server_kwargs: Any,
    ) -> None:
        """Initialize the watchdog.

        Args:
            socket_path: Unix socket path for the server.
            restart_delay: Initial delay before restart attempt (seconds).
                Doubles on each consecutive crash, capped at 30s.
            max_restarts: Maximum consecutive restarts within _CRASH_WINDOW_SECONDS
                before giving up.
            **server_kwargs: Extra kwargs forwarded to run_server()
                (e.g. max_elements=50_000, embed_threads=4).
        """
        self._socket_path = socket_path
        self._restart_delay = restart_delay
        self._max_restarts = max_restarts
        self._server_kwargs = server_kwargs

        self._process: multiprocessing.Process | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._started = False

        # Track crash history for exponential backoff
        self._crash_times: list[float] = []
        self._consecutive_crashes: int = 0

    @property
    def socket_path(self) -> str:
        return self._socket_path

    @property
    def pid(self) -> int | None:
        if self._process is not None and self._process.is_alive():
            return self._process.pid
        return None

    async def start(self) -> None:
        """Start the server process and begin monitoring."""
        self._started = True
        self._spawn_process()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "event=embedding_watchdog_started socket=%s pid=%s",
            self._socket_path,
            self.pid,
        )

    def _spawn_process(self) -> None:
        """Create and start a new server process using spawn context."""
        ctx = multiprocessing.get_context("spawn")
        self._process = cast(
            multiprocessing.Process,
            ctx.Process(
                target=_server_target,
                args=(self._socket_path, self._server_kwargs),
                daemon=True,
                name="headroom-embed-server",
            ),
        )
        self._process.start()
        logger.info(
            "event=embedding_server_process_started pid=%d socket=%s",
            self._process.pid or -1,
            self._socket_path,
        )

    async def _monitor_loop(self) -> None:
        """Background task that detects server crashes and restarts it."""
        while self._started:
            await asyncio.sleep(1.0)  # Poll every second

            process = self._process
            if process is None:
                continue

            if process.is_alive():
                # Process is running — nothing to do
                continue

            # Process died
            exit_code = process.exitcode
            now = time.monotonic()

            # Prune crash history outside the window
            self._crash_times = [t for t in self._crash_times if now - t < _CRASH_WINDOW_SECONDS]
            self._crash_times.append(now)
            self._consecutive_crashes = len(self._crash_times)

            logger.warning(
                "event=embedding_server_died exit_code=%s consecutive_crashes=%d max_restarts=%d",
                exit_code,
                self._consecutive_crashes,
                self._max_restarts,
            )

            # Exit code 3 = permanent dependency error (e.g. missing hnswlib).
            # Retrying will never help — give up immediately.
            if exit_code == 3:
                logger.error(
                    "event=embedding_server_giving_up "
                    "socket=%s reason=missing_dependency -- "
                    "install with: uv sync --extra memory  OR  pip install hnswlib",
                    self._socket_path,
                )
                self._started = False
                return

            if self._consecutive_crashes > self._max_restarts:
                logger.error(
                    "event=embedding_server_giving_up "
                    "socket=%s consecutive_crashes=%d -- "
                    "memory features will be unavailable",
                    self._socket_path,
                    self._consecutive_crashes,
                )
                self._started = False
                return

            # Exponential backoff: delay * 2^(consecutive-1), capped at 30s
            backoff = min(
                self._restart_delay * (2 ** (self._consecutive_crashes - 1)),
                30.0,
            )
            logger.info("event=embedding_server_restarting delay=%.1fs", backoff)
            await asyncio.sleep(backoff)

            if self._started:
                self._spawn_process()

    async def stop(self) -> None:
        """Stop the server process and the monitor task."""
        self._started = False

        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        await self._stop_process()
        logger.info("event=embedding_watchdog_stopped socket=%s", self._socket_path)

    async def _stop_process(self) -> None:
        """Send SIGTERM, wait up to _STOP_TIMEOUT_SECONDS, then SIGKILL."""
        process = self._process
        if process is None or not process.is_alive():
            return

        process.terminate()  # SIGTERM

        # Wait for clean exit
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, process.join, _STOP_TIMEOUT_SECONDS),
                timeout=_STOP_TIMEOUT_SECONDS + 1.0,
            )
        except asyncio.TimeoutError:
            pass

        if process.is_alive():
            logger.warning(
                "event=embedding_server_kill pid=%d (did not exit in %.0fs)",
                process.pid or -1,
                _STOP_TIMEOUT_SECONDS,
            )
            process.kill()  # SIGKILL
            await asyncio.sleep(0.2)

        self._process = None

    async def is_healthy(self) -> bool:
        """Ping the server socket. Returns True if the server is responsive."""
        try:
            from headroom.memory.adapters.remote import RemoteEmbedder

            embedder = RemoteEmbedder(self._socket_path, connect_timeout=2.0, request_timeout=2.0)
            result = await embedder.ping()
            await embedder.close()
            return result
        except Exception:
            return False

    async def wait_until_healthy(self, timeout: float = 10.0) -> bool:
        """Poll is_healthy() until it returns True or timeout expires.

        Returns True if healthy within timeout, False otherwise.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.is_healthy():
                return True
            await asyncio.sleep(0.2)
        return False
