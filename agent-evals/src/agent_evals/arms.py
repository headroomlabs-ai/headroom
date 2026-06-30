"""Arm runtime: turn an :class:`ArmSpec` into a live proxy + ``base_url``.

This module owns the lifecycle of one experiment arm. ``HeadroomArm`` is an async context
manager that — depending on the spec — either resolves a provider-default ``base_url`` (A0
direct, no proxy) or spawns a ``headroom proxy`` subprocess, waits for ``/readyz``, and tears
it down on exit.

Everything that varies between deployments (command, ports, ready path, timeouts, header/url
shapes) comes from :class:`Settings`/:class:`ProxyLaunchConfig` or is passed as a parameter —
nothing about the proxy invocation is hardcoded in logic here. The proxy flag names are verified
against ``headroom/cli/proxy.py`` (``--port``, ``--no-optimize``, ``--mode token``, and ablation
flags such as ``--disable-kompress`` / ``--no-read-lifecycle`` / ``--code-aware``).
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import IO

import httpx

from .config import Settings
from .logging import get_logger
from .models import ArmSpec, Provider, ProxyMode, TaskSavings

logger = get_logger("arms")

# Mode flags. Verified against headroom/cli/proxy.py:
#   --no-optimize  -> passthrough (optimize disabled) for ProxyMode.OFF (A1).
#   --mode token   -> compression enabled for ProxyMode.TOKEN (B).
# These two literals are the *protocol* the headroom CLI exposes; they are not tunable knobs,
# so they live here as the single source of truth for the off/token mapping. The command head
# (``headroom proxy``) and everything else come from Settings.
_PASSTHROUGH_FLAG = "--no-optimize"
_MODE_OPTION = "--mode"


def build_proxy_command(spec: ArmSpec, settings: Settings, port: int) -> list[str]:
    """Build the ``headroom proxy`` argv for an arm. PURE — no I/O.

    Layout: ``settings.proxy.headroom_cmd`` + ``--port <port>`` + mode flag + ``spec.proxy_flags``.

    Raises ``ValueError`` if called for a direct (``proxy_mode is None``) arm — A0 never launches
    a proxy, so building a command for it is a programming error, not a silent no-op.
    """

    if spec.proxy_mode is None:
        raise ValueError(
            f"build_proxy_command called for arm {spec.name.value!r} with proxy_mode=None "
            "(A0 direct launches no proxy)"
        )

    cmd: list[str] = list(settings.proxy.headroom_cmd)
    cmd += ["--port", str(port)]

    if spec.proxy_mode is ProxyMode.OFF:
        cmd.append(_PASSTHROUGH_FLAG)
    elif spec.proxy_mode is ProxyMode.TOKEN:
        cmd += [_MODE_OPTION, ProxyMode.TOKEN.value]
    else:  # pragma: no cover - exhaustive guard against new ProxyMode members
        raise ValueError(f"unsupported proxy_mode: {spec.proxy_mode!r}")

    cmd += list(spec.proxy_flags)
    return cmd


def build_arm_env(spec: ArmSpec, settings: Settings, base_url: str) -> dict[str, str]:
    """Build the provider-specific base-url env vars a harness must export. PURE — no I/O.

    ``base_url`` is the already-correctly-formed root for the arm (provider default, or the
    proxy's ``http://127.0.0.1:<port>``). Per the headroom CLI usage contract:

    * Anthropic clients point ``ANTHROPIC_BASE_URL`` at the root (the SDK appends ``/v1/messages``),
      so the Anthropic base_url carries NO ``/v1`` suffix.
    * OpenAI clients point ``OPENAI_BASE_URL`` (and the ``OPENAI_API_BASE`` alias) at ``<root>/v1``
      (the SDK appends ``/chat/completions``), so the OpenAI base_url MUST end in ``/v1``.

    For OpenAI, the provider-default ``settings.openai_base_url`` already includes ``/v1``; a proxy
    ``base_url`` does not, so we ensure exactly one ``/v1`` suffix here.
    """

    if spec.provider is Provider.ANTHROPIC:
        return {"ANTHROPIC_BASE_URL": base_url}

    if spec.provider is Provider.OPENAI:
        openai_url = (
            base_url if base_url.rstrip("/").endswith("/v1") else f"{base_url.rstrip('/')}/v1"
        )
        return {"OPENAI_BASE_URL": openai_url, "OPENAI_API_BASE": openai_url}

    raise ValueError(f"unsupported provider: {spec.provider!r}")  # pragma: no cover


def allocate_port(settings: Settings) -> int:
    """Find a free TCP port on 127.0.0.1 within the configured range.

    Scans ``[port_range_start, port_range_end]`` inclusive, binding ``127.0.0.1:<port>`` to probe
    availability and releasing it immediately. There is an unavoidable TOCTOU window between this
    bind and the proxy's own bind; that is acceptable for a single-host eval runner. Raises
    ``RuntimeError`` if no port in the range is free.
    """

    start = settings.proxy.port_range_start
    end = settings.proxy.port_range_end
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"no free port found in range [{start}, {end}] on 127.0.0.1 — "
        "all ports busy or range misconfigured"
    )


class ArmHandle:
    """A live arm handle: the ``base_url`` + ``env`` a harness uses, plus savings capture.

    Implements the :class:`agent_evals.protocols.ArmHandle` protocol structurally. Savings capture
    is delegated to an injected provider (the savings module supplies the real implementation);
    when none is injected, ``capture_savings`` returns ``None`` (no fabricated data).
    """

    def __init__(
        self,
        base_url: str,
        env: dict[str, str],
        savings_provider: Callable[[str], TaskSavings | None] | None = None,
    ) -> None:
        self.base_url = base_url
        self.env = env
        self._savings_provider = savings_provider

    def capture_savings(self, task_id: str) -> TaskSavings | None:
        """Return Layer-1 savings for ``task_id`` via the injected provider, or None."""

        if self._savings_provider is None:
            return None
        return self._savings_provider(task_id)


class HeadroomArm:
    """Async context manager that provisions an :class:`ArmHandle` for one arm.

    Implements the :class:`agent_evals.protocols.Arm` protocol. For an A0 direct spec
    (``proxy_mode is None``) no subprocess is launched and the provider-default base_url from
    settings is used. Otherwise a ``headroom proxy`` subprocess is spawned, probed at ``/readyz``,
    and torn down on exit.
    """

    def __init__(
        self,
        spec: ArmSpec,
        settings: Settings,
        run_dir: Path,
        savings_provider: Callable[[str], TaskSavings | None] | None = None,
    ) -> None:
        self.spec = spec
        self.settings = settings
        self.run_dir = run_dir
        self._savings_provider = savings_provider

        # Live-process state (None for A0 direct, or before/after the proxy runs).
        self._process: asyncio.subprocess.Process | None = None
        self._log_file: IO[bytes] | None = None
        self._port: int | None = None

    # -- provider-default base_url -------------------------------------------------------------

    def _provider_default_base_url(self) -> str:
        if self.spec.provider is Provider.ANTHROPIC:
            return self.settings.anthropic_base_url
        if self.spec.provider is Provider.OPENAI:
            return self.settings.openai_base_url
        raise ValueError(f"unsupported provider: {self.spec.provider!r}")  # pragma: no cover

    # -- async context manager -----------------------------------------------------------------

    async def __aenter__(self) -> ArmHandle:
        if self.spec.proxy_mode is None:
            base_url = self._provider_default_base_url()
            env = build_arm_env(self.spec, self.settings, base_url)
            logger.info(
                "arm direct (no proxy)",
                extra={
                    "fields": {
                        "arm": self.spec.name.value,
                        "provider": self.spec.provider.value,
                        "base_url": base_url,
                    }
                },
            )
            return ArmHandle(base_url, env, self._savings_provider)

        return await self._launch_proxy()

    async def _launch_proxy(self) -> ArmHandle:
        port = allocate_port(self.settings)
        self._port = port
        command = build_proxy_command(self.spec, self.settings, port)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.run_dir / f"proxy-{self.spec.name.value}-{port}.log"
        # Binary append handle; the child writes its own stdout/stderr here.
        self._log_file = log_path.open("ab")

        logger.info(
            "spawning proxy",
            extra={
                "fields": {
                    "arm": self.spec.name.value,
                    "port": port,
                    "command": command,
                    "log_path": str(log_path),
                }
            },
        )

        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdout=self._log_file,
            stderr=self._log_file,
        )

        base_url = f"http://127.0.0.1:{port}"
        try:
            await self._wait_for_ready(base_url, log_path)
        except Exception:
            # Ready probe failed (timeout or crash) — tear the child down before re-raising so we
            # never leak a half-started proxy.
            await self._terminate_process()
            self._close_log()
            raise

        env = build_arm_env(self.spec, self.settings, base_url)
        logger.info(
            "proxy ready",
            extra={
                "fields": {
                    "arm": self.spec.name.value,
                    "port": port,
                    "base_url": base_url,
                }
            },
        )
        return ArmHandle(base_url, env, self._savings_provider)

    async def _wait_for_ready(self, base_url: str, log_path: Path) -> None:
        readyz_url = f"{base_url}{self.settings.proxy.readyz_path}"
        timeout_s = self.settings.proxy.readyz_timeout_s
        poll_s = self.settings.proxy.poll_interval_s
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s

        async with httpx.AsyncClient() as client:
            while True:
                # Bail early if the child already exited — no point polling a dead proxy.
                returncode = getattr(self._process, "returncode", None)
                if returncode is not None:
                    raise RuntimeError(
                        f"proxy for arm {self.spec.name.value!r} exited with code {returncode} "
                        f"before becoming ready.\n{self._log_tail(log_path)}"
                    )
                try:
                    resp = await client.get(readyz_url, timeout=poll_s)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass  # not up yet — keep polling until the deadline

                if loop.time() >= deadline:
                    raise RuntimeError(
                        f"proxy for arm {self.spec.name.value!r} did not become ready at "
                        f"{readyz_url} within {timeout_s}s.\n{self._log_tail(log_path)}"
                    )
                await asyncio.sleep(poll_s)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        tb: TracebackType | None = None,
    ) -> None:
        await self._terminate_process()
        self._close_log()

    # -- teardown helpers ----------------------------------------------------------------------

    async def _terminate_process(self) -> None:
        """SIGTERM the proxy, await it briefly, then SIGKILL if it ignores us. Idempotent."""

        proc = self._process
        if proc is None:
            return
        self._process = None

        if getattr(proc, "returncode", None) is not None:
            return  # already exited

        terminate = getattr(proc, "terminate", None)
        if callable(terminate):
            with contextlib.suppress(ProcessLookupError):
                terminate()

        wait = getattr(proc, "wait", None)
        if not callable(wait):
            return
        try:
            await asyncio.wait_for(wait(), timeout=self._term_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "proxy ignored SIGTERM; killing",
                extra={"fields": {"arm": self.spec.name.value, "port": self._port}},
            )
            kill = getattr(proc, "kill", None)
            if callable(kill):
                with contextlib.suppress(ProcessLookupError):
                    kill()
            with contextlib.suppress(Exception):
                await wait()

    def _close_log(self) -> None:
        if self._log_file is not None:
            with contextlib.suppress(Exception):
                self._log_file.close()
            self._log_file = None

    @property
    def _term_timeout(self) -> float:
        # Reuse the ready timeout as the graceful-shutdown budget; both are governed by the same
        # proxy-launch config surface, so teardown stays configurable too.
        return self.settings.proxy.readyz_timeout_s

    @staticmethod
    def _log_tail(log_path: Path, max_chars: int = 4000) -> str:
        """Best-effort tail of the proxy log, for inclusion in failure messages."""

        try:
            data = log_path.read_bytes()
        except OSError:
            return f"(proxy log at {log_path} unreadable)"
        text = data.decode("utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[-max_chars:]
        return f"--- proxy log tail ({log_path}) ---\n{text}"
