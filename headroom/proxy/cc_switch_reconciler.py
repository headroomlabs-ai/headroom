"""cc-switch reconciler: keep Headroom in the request path without fighting cc-switch.

Background
----------
cc-switch (https://github.com/farion1231/cc-switch) in its default *direct
injection* mode rewrites the **entire** ``~/.claude/settings.json`` every time
the user switches provider (atomic overwrite -- see
``src-tauri/src/services/provider/live.rs``). It writes the selected provider's
real endpoint + token, e.g.::

    {"env": {"ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
              "ANTHROPIC_AUTH_TOKEN": "sk-..."}}

Claude Code re-reads that file, so a running session immediately follows the
switch. The problem: that overwrite blows away any ``ANTHROPIC_BASE_URL`` that
points at Headroom.

What this does
--------------
A lightweight in-process watcher (poll-based, robust against atomic renames):

1. Detects cc-switch's overwrite.
2. **Captures** the real provider endpoint and sets it as Headroom's upstream
   (runtime, no restart -- ``HeadroomProxy.ANTHROPIC_API_URL`` is a class attr
   read per request).
3. **Rewrites only** ``env.ANTHROPIC_BASE_URL`` back to Headroom's local URL,
   leaving the token / model / everything else untouched.

Result: ``Claude -> Headroom (compress) -> selected provider``. cc-switch never
knows; it is the *trigger*, not a coordination partner. The token rides in the
request (Claude -> Headroom -> upstream, passed through verbatim); Headroom
never reads or stores it.

Safety
------
- Loop-safe: once ``base_url`` already equals Headroom's URL, it is left alone.
- Official / empty env (``{"env": {}}``, what cc-switch writes for "Claude
  Official") is **left direct** by default -- subscription OAuth through a custom
  base URL is fragile, so v1 does not route it through Headroom. Set
  ``HEADROOM_CC_SWITCH_ROUTE_OFFICIAL=1`` to route official through Headroom too.
- Gated entirely behind ``HEADROOM_CC_SWITCH_RECONCILE=1`` -- off by default, so
  it never affects users who do not opt in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.3


def _settings_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
    return Path(base).expanduser() / "settings.json"


def reconciler_enabled() -> bool:
    return os.environ.get("HEADROOM_CC_SWITCH_RECONCILE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _route_official() -> bool:
    return os.environ.get("HEADROOM_CC_SWITCH_ROUTE_OFFICIAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class CCSwitchReconciler:
    """Polls Claude settings.json and keeps Headroom in the path (see module docstring)."""

    def __init__(
        self,
        *,
        proxy_url: str,
        default_upstream: str,
        set_upstream: Callable[[str], None],
        path: Path | None = None,
    ) -> None:
        self.proxy_url = proxy_url.rstrip("/")
        self.default_upstream = default_upstream
        self._set_upstream = set_upstream
        self.path = path or _settings_path()
        self.current_upstream: str | None = None
        self._task: asyncio.Task | None = None
        self._last_mtime_ns: int | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        logger.info("cc-switch reconciler: watching %s -> proxy %s", self.path, self.proxy_url)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - watcher must never die
                logger.debug("cc-switch reconciler tick error: %s", exc)
            await asyncio.sleep(_POLL_INTERVAL_S)

    # Synchronous core (also directly unit-testable).
    def tick(self) -> bool:
        """One reconcile pass. Returns True if it rewrote settings.json."""
        try:
            mtime_ns = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return False
        if mtime_ns == self._last_mtime_ns:
            return False

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Transient read/parse failure (e.g. caught mid atomic-replace, or
            # cc-switch wrote partial JSON). Do NOT consume this mtime — leave
            # _last_mtime_ns untouched so the next tick retries instead of
            # treating the broken state as already-processed.
            return False
        # Read succeeded: now it is safe to mark this mtime processed.
        self._last_mtime_ns = mtime_ns
        if not isinstance(data, dict):
            return False
        env = data.get("env")
        env = dict(env) if isinstance(env, dict) else {}
        url = env.get("ANTHROPIC_BASE_URL")
        # A non-string base_url (number/list from a hand-edited file) would
        # raise on .rstrip() below and spam the watcher loop; treat as empty.
        if not isinstance(url, str):
            url = ""

        # Empty / official: cc-switch wrote {"env": {}} (Claude Official, OAuth).
        if not url:
            if _route_official():
                self.current_upstream = self.default_upstream
                self._set_upstream(self.default_upstream)
                env["ANTHROPIC_BASE_URL"] = self.proxy_url
                data["env"] = env
                self._atomic_write(data)
                logger.info("cc-switch reconciler: official -> route via Headroom")
                return True
            return False  # leave official direct (default, safe for OAuth)

        # Already pointing at us: nothing to do (loop guard).
        if url.rstrip("/") == self.proxy_url:
            return False

        # Third-party / custom endpoint: capture it as upstream, point Claude at us.
        self.current_upstream = url
        self._set_upstream(url)
        env["ANTHROPIC_BASE_URL"] = self.proxy_url
        data["env"] = env
        self._atomic_write(data)
        logger.info(
            "cc-switch reconciler: captured upstream=%s, base_url -> %s", url, self.proxy_url
        )
        return True

    def _atomic_write(self, data: dict) -> None:
        # Per-process temp name: multiple Headroom processes reconciling the
        # same settings.json must not clobber each other's temp file.
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.hrtmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        # Skip the mtime bump caused by our own write so we don't re-process it.
        try:
            self._last_mtime_ns = self.path.stat().st_mtime_ns
        except OSError:
            pass
