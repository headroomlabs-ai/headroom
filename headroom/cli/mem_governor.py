"""Memory-pressure governor for ``headroom wrap`` (root-cause OOM guard).

Every wrapped agent (``headroom wrap claude``) crosses one ``wrap`` process
before it spawns the heavy ``claude`` + MCP stack (e.g. serena + rust-analyzer,
often several GB each). On a busy multi-agent box the *Nth* launch is what tips
total RAM past the limit and triggers a **global** OOM — which the kernel can
only resolve by killing some victim, in practice the desktop compositor,
freezing the whole session.

``wrap`` is the single choke point every agent passes through, so it is the
natural place to throttle. Before launching the child, :func:`gate_launch`
checks how much memory is actually available and, when the box is already low:

* ``wait``   (default) — poll until memory recovers, then proceed; if it never
  recovers within the budget, proceed anyway (fail-open) with a loud warning.
* ``refuse`` — exit immediately with ``EX_TEMPFAIL`` (75) so a supervisor or
  queue worker can retry later instead of OOMing the host.
* ``off``    — disabled.

Fail-open by construction: if available memory cannot be measured (e.g. a
non-Linux host without ``psutil``) the gate never blocks a launch.

Configuration (env):
    HEADROOM_WRAP_MEM_POLICY        wait | refuse | off          (default: wait)
    HEADROOM_WRAP_MEM_MIN_MB        floor of available MiB        (default: 2048)
    HEADROOM_WRAP_MEM_WAIT_SECONDS  max wait in ``wait`` mode     (default: 120)
    HEADROOM_WRAP_MEM_POLL_SECONDS  poll interval while waiting   (default: 3)
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping

__all__ = ["gate_launch", "available_memory_mb", "EX_TEMPFAIL"]

# sysexits.h: a temporary failure; signals "retry later" to a supervisor.
EX_TEMPFAIL = 75

_POLICY_ENV = "HEADROOM_WRAP_MEM_POLICY"
_MIN_MB_ENV = "HEADROOM_WRAP_MEM_MIN_MB"
_WAIT_ENV = "HEADROOM_WRAP_MEM_WAIT_SECONDS"
_POLL_ENV = "HEADROOM_WRAP_MEM_POLL_SECONDS"

_DEFAULT_MIN_MB = 2048
_DEFAULT_WAIT_SECONDS = 120.0
_DEFAULT_POLL_SECONDS = 3.0


def available_memory_mb() -> int | None:
    """Best-effort available system memory in MiB, or ``None`` if unknown.

    Prefers Linux ``/proc/meminfo`` ``MemAvailable`` (the kernel's own estimate
    of what can be allocated without swapping), falling back to ``psutil`` where
    present. Returns ``None`` when neither is available so callers fail open.
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    # Format: "MemAvailable:   12345678 kB"
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil  # type: ignore[import-untyped]  # optional dependency

        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:  # noqa: BLE001 -- any failure -> unmeasured -> fail open
        return None


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def gate_launch(
    echo: Callable[[str], None] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    read_available_mb: Callable[[], int | None] = available_memory_mb,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    """Throttle a wrapped-agent launch on system memory pressure.

    Returns a short outcome tag for telemetry/tests: ``off``, ``unmeasured``,
    ``ok`` (enough headroom up front), ``recovered`` (waited, memory freed), or
    ``proceeded`` (waited out the budget and launched anyway, fail-open). In
    ``refuse`` mode it raises ``SystemExit(EX_TEMPFAIL)`` instead of returning
    when memory is below the floor.

    All side effects (clock, sleep, memory read) are injectable so the wait loop
    is deterministic under test.
    """
    env = os.environ if env is None else env
    say = echo if echo is not None else (lambda _msg: None)

    policy = (env.get(_POLICY_ENV) or "wait").strip().lower()
    if policy == "off":
        return "off"
    if policy not in ("wait", "refuse"):
        policy = "wait"

    min_mb = _env_int(env, _MIN_MB_ENV, _DEFAULT_MIN_MB)
    if min_mb <= 0:
        return "off"

    avail = read_available_mb()
    if avail is None:
        return "unmeasured"  # fail-open: cannot measure -> never block
    if avail >= min_mb:
        return "ok"

    if policy == "refuse":
        say(
            f"  Memory governor: {avail} MiB available < {min_mb} MiB floor; "
            f"refusing launch ({_POLICY_ENV}=refuse). Retry when memory frees."
        )
        raise SystemExit(EX_TEMPFAIL)

    # policy == "wait": poll until memory recovers or the budget elapses.
    wait_s = _env_float(env, _WAIT_ENV, _DEFAULT_WAIT_SECONDS)
    poll_s = _env_float(env, _POLL_ENV, _DEFAULT_POLL_SECONDS)
    if poll_s <= 0:
        poll_s = _DEFAULT_POLL_SECONDS
    say(
        f"  Memory governor: {avail} MiB available < {min_mb} MiB floor; "
        f"waiting up to {wait_s:.0f}s for memory to free "
        f"(set {_POLICY_ENV}=off to disable)..."
    )
    deadline = monotonic() + wait_s
    while monotonic() < deadline:
        sleep(poll_s)
        avail = read_available_mb()
        if avail is None:
            return "unmeasured"
        if avail >= min_mb:
            say(f"  Memory governor: {avail} MiB available; proceeding.")
            return "recovered"
    # Budget exhausted: proceed anyway (fail-open) but make the risk visible.
    say(
        f"  Memory governor: still low ({avail} MiB) after {wait_s:.0f}s; "
        "proceeding anyway (fail-open). Consider fewer concurrent agents."
    )
    return "proceeded"
