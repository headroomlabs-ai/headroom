"""Pure serializers for the loopback-only /debug/* introspection endpoints.

Unit 5 of the Codex-proxy resilience plan. These helpers transform live
runtime state (asyncio tasks, WS session registry, warmup registry) into
JSON-serializable dicts that ``/debug/tasks``, ``/debug/ws-sessions`` and
``/debug/warmup`` return.

Design constraints
------------------
* **No state mutation.** Every helper is a pure read of the current
  state. Calling any of them N times in a row must not change registry
  contents or task counts.
* **No blocking I/O.** The helpers only call ``asyncio.all_tasks()``,
  ``task.get_name()`` and attribute reads on already-materialized
  registry state.
* **No privacy leaks.** Task serialization deliberately excludes
  ``cr_frame.f_locals``, coro ``locals()``, request bodies and anything
  that could accidentally carry user data. Only the task *name* and the
  coroutine's *qualname* (the static code symbol) are exposed.

Age tracking for generic tasks
------------------------------
:mod:`asyncio.Task` does not record a creation time natively. For tasks
that belong to a tracked WS session, the WS registry holds a
``started_at`` we can correlate by task name (the handler names relay
tasks ``codex-ws-c2u-<sid>`` / ``codex-ws-u2c-<sid>``). For every other
task we report ``age_seconds=None`` rather than faking it — the plan
explicitly prefers this minimal approach over invasive wrapper
instrumentation.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from headroom.proxy.ws_session_registry import WebSocketSessionRegistry

__all__ = [
    "collect_tasks",
]


# Task-name prefixes emitted by the Codex WS handler for its relay tasks.
# See ``headroom/proxy/handlers/openai.py`` ``handle_openai_responses_ws``.
_CODEX_WS_RELAY_PREFIXES: tuple[str, ...] = (
    "codex-ws-c2u-",
    "codex-ws-u2c-",
)


def _coro_qualname(task: asyncio.Task[Any]) -> str | None:
    """Return the coroutine qualname for ``task`` without touching locals.

    The qualname is a static code symbol (``Class.method``) and never
    carries request data. We intentionally do not touch ``cr_frame`` or
    any mutable coroutine state beyond ``cr_code.co_qualname``.
    """
    try:
        coro = task.get_coro()
    except Exception:
        return None
    if coro is None:
        return None
    code = getattr(coro, "cr_code", None)
    if code is None:
        return None
    qualname = getattr(code, "co_qualname", None)
    if qualname is None:
        qualname = getattr(code, "co_name", None)
    return qualname if isinstance(qualname, str) else None


def _stack_depth(task: asyncio.Task[Any]) -> int | None:
    """Return a short stack-depth summary for ``task``.

    Uses :meth:`asyncio.Task.get_stack` which returns frame objects only
    up to a bounded limit; we report the *count*, never the frame
    contents. Frame locals are never inspected.
    """
    try:
        stack = task.get_stack(limit=32)
    except Exception:
        return None
    return len(stack)


def _age_for_named_task(
    task_name: str,
    ws_registry: WebSocketSessionRegistry | None,
) -> float | None:
    """Resolve an age for tasks named by the WS handler.

    Codex relay task names embed the session id after a known prefix.
    When the WS registry is present and holds that session, we can
    derive an age from the session's ``started_at``. For everything
    else we return ``None`` — the plan prefers a truthful ``null`` to
    an invented age.
    """
    if ws_registry is None:
        return None
    for prefix in _CODEX_WS_RELAY_PREFIXES:
        if task_name.startswith(prefix):
            session_id = task_name[len(prefix) :]
            handle = ws_registry.get(session_id)
            if handle is None:
                return None
            return max(0.0, time.perf_counter() - handle.started_at)
    return None


def collect_tasks(
    ws_registry: WebSocketSessionRegistry | None = None,
    *,
    with_stack_depth: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate ``asyncio.all_tasks()`` for /debug/tasks.

    Each entry carries: ``name``, ``coro_qualname``, ``age_seconds``
    (``None`` unless the task is a tracked WS relay), ``stack_depth``,
    and ``done``. Sorted by age descending with ``None`` ages sorted
    after known ages. System noise (``None`` tasks, tasks with no
    coroutine) is filtered out.

    ``stack_depth`` is only computed when ``with_stack_depth=True``
    because :meth:`asyncio.Task.get_stack` walks coroutine frames and
    can noticeably stall the event loop during a storm with 50+ relay
    tasks. The default returns ``stack_depth=None``; callers that need
    it (a human debugging one snapshot) can pass ``with_stack_depth=True``.
    """
    try:
        tasks = asyncio.all_tasks()
    except RuntimeError:
        # No running loop — e.g. called outside of an event loop.
        return []

    entries: list[dict[str, Any]] = []
    for task in tasks:
        if task is None:
            continue
        try:
            name = task.get_name()
        except Exception:
            name = None
        qualname = _coro_qualname(task)
        if qualname is None and name is None:
            # No stable identity — skip rather than emit a blank row.
            continue
        age = _age_for_named_task(name or "", ws_registry)
        entry: dict[str, Any] = {
            "name": name,
            "coro_qualname": qualname,
            "age_seconds": age,
            "stack_depth": _stack_depth(task) if with_stack_depth else None,
            "done": bool(task.done()),
        }
        entries.append(entry)

    # Sort by age descending; None ages sort last (treat as -inf for desc).
    def _sort_key(entry: dict[str, Any]) -> tuple[int, float]:
        age = entry.get("age_seconds")
        if age is None:
            return (1, 0.0)
        return (0, -float(age))

    entries.sort(key=_sort_key)
    return entries
