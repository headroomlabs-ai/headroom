"""Third-party proxy extension point.

External packages hook into the Headroom proxy at startup by declaring an
entry point in the ``headroom.proxy_extension`` group in their ``pyproject.toml``:

    [project.entry-points."headroom.proxy_extension"]
    my_extension = "my_pkg.extension:install"

Each ``install`` callable is invoked with the FastAPI ``app`` and the
``ProxyConfig`` at app creation time, and is free to:

  * register ASGI middleware (``app.add_middleware(...)``)
  * add routes or health endpoints
  * mutate config
  * raise on an environment or auth failure to disable *itself* (the proxy logs
    the failure and starts without that extension)

OSS makes no assumptions about what extensions do. The interface is
deliberately minimal; extensions own the complexity behind it.

Where extension middleware sits
-------------------------------

``install_all`` runs **before** the proxy registers its inbound security gate
and its request-body ceiling, and Starlette builds the stack so that the last
layer registered is the outermost. Middleware an extension adds therefore
always runs *inside* both, and an extension author can rely on two things
without re-implementing them:

  * a request that reaches extension middleware has already passed
    ``HEADROOM_PROXY_TOKEN`` (when one is configured) — an extension never sees,
    answers, buffers, or rewrites headers on an unauthenticated request, on
    either transport (HTTP or WebSocket);
  * a request body that reaches extension middleware is already bounded by
    :data:`headroom.proxy.helpers.MAX_REQUEST_BODY_SIZE` — an extension that
    buffers the body cannot be made to hold more than that.

The flip side is a constraint: extension middleware cannot act as the
proxy's authentication layer, because the proxy's own gate runs first. An
extension that needs its own inbound auth applies it in addition to, not
instead of, the proxy token. ``install_all`` records every middleware entry an
extension adds on ``app.state.extension_middleware`` so the ordering can be
asserted by tests rather than assumed.

Reporting what an extension saved, and what it cost
---------------------------------------------------

An extension that changes the bill should say so, or the operator sees a
different total with nothing to attribute it to. Two calls, both taking the
ASGI ``scope`` so they work from middleware — which runs outside the request
handler and has no other way in::

    from headroom.proxy.savings_attribution import (
        record_scope_savings, record_scope_timing,
    )

    record_scope_savings(scope, "my_extension", tokens=1200, usd=0.004)
    record_scope_timing(scope, "my_extension", elapsed_ms)

``record_scope_savings`` takes ``tokens``, ``usd``, or both, so an extension
that saves money WITHOUT saving tokens — routing a request to a cheaper model,
say — can report a real number instead of a token count nobody saved. Pass
``realized=False`` for a projection rather than a measured amount; the two are
kept apart everywhere they surface. Savings land on ``/stats`` under
``savings.by_source``, on the dashboard as their own card, and in Prometheus as
``headroom_savings_attributed_usd_total{source=...}``. **Attribution only** —
these rows explain the headline total, they are never added to it.

``record_scope_timing`` is the other half of the trade: an extension's own
latency, which is otherwise invisible because ``overhead_ms`` is measured
inside the handler that the extension wraps. It lands in ``/stats`` under
``pipeline_timing``, in the dashboard's Performance panel, and in
``headroom_transform_timing_ms_*``, namespaced ``ext:<name>`` so it can never
collide with a built-in transform.

Both are bounded (32 sources, 16 stages), never raise, and never change a
response — telemetry from a plugin must not be able to break the request it is
describing.

**Extensions are opt-in.** Discovery enumerates every registered extension,
but ``install_all`` only invokes those explicitly enabled by the operator.
This protects users from silent behavior changes when a package they didn't
audit gets installed in the same environment (e.g., as a transitive dep).

Enabling extensions:

  * CLI:  ``headroom proxy --proxy-extension myorg_ext,mypkg``
  * Env:  ``HEADROOM_PROXY_EXTENSIONS=myorg_ext,mypkg``
  * Wildcard: ``--proxy-extension '*'`` enables every discovered extension
    (use only when you trust everything in your environment).

Stability contract: this module is load-bearing for any third-party extensions.
Changes to the signature of ``install(app, config)`` or the entry-point group
name require a deprecation cycle.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "headroom.proxy_extension"
ENV_VAR = "HEADROOM_PROXY_EXTENSIONS"

ProxyExtension = Callable[[Any, Any], None]
"""Signature: ``install(app: FastAPI, config: ProxyConfig) -> None``."""


def discover() -> Iterator[tuple[str, ProxyExtension]]:
    """Yield ``(name, install_callable)`` pairs for every registered extension.

    Entry-point load failures are logged and skipped — a broken third-party
    package must not prevent the proxy from starting. An extension that fails
    its environment/auth check raises from ``install()``; ``install_all`` logs
    and skips it, so it is disabled rather than aborting startup.
    """
    try:
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 — importlib.metadata can raise varied types
        log.warning(
            "proxy extensions: entry-point enumeration failed (%s), so no proxy "
            "extensions will be installed",
            describe_exception(exc),
        )
        return
    for entry in entries:
        try:
            install = entry.load()
        except Exception as exc:  # noqa: BLE001
            log.warning("proxy extension %r failed to load: %s", entry.name, exc)
            continue
        yield entry.name, install


def _resolve_enabled(enabled: Iterable[str] | None) -> set[str]:
    """Resolve the set of enabled extension names.

    Precedence: explicit ``enabled`` argument > ``HEADROOM_PROXY_EXTENSIONS``
    env var > empty (no extensions). Empty strings and whitespace are
    stripped. The literal ``*`` enables all discovered extensions.
    """
    raw: Iterable[str]
    if enabled is not None:
        raw = enabled
    else:
        raw = (os.environ.get(ENV_VAR) or "").split(",")
    out: set[str] = set()
    for n in raw:
        n = n.strip()
        if n:
            out.add(n)
    return out


def install_all(
    app: Any,
    config: Any,
    enabled: Iterable[str] | None = None,
) -> list[str]:
    """Run only the explicitly-enabled extensions' ``install(app, config)``.

    Discovery still runs so we can log the universe of available extensions,
    but only those whose entry-point ``name`` is in ``enabled`` are invoked.
    The literal ``"*"`` in ``enabled`` is a wildcard that enables every
    discovered extension.

    Returns the names of successfully installed extensions. If an extension
    raises inside ``install()`` it is logged, skipped, and recorded as failed —
    a single broken extension is disabled rather than aborting proxy startup for
    every other extension.
    """
    enabled_set = _resolve_enabled(enabled)
    discovered = list(discover())
    discovered_names = [n for n, _ in discovered]

    if not enabled_set:
        if discovered_names:
            log.info(
                "proxy extensions discovered but disabled (opt-in): %s. "
                "Enable with --proxy-extension <name> or %s=<name1,name2>.",
                ",".join(discovered_names),
                ENV_VAR,
            )
        return []

    wildcard = "*" in enabled_set
    installed: list[str] = []
    failed: list[str] = []
    # Middleware entries added by extensions, in registration order. Starlette
    # prepends on ``add_middleware``, so the entries an install() adds are
    # whatever is new at the FRONT of ``app.user_middleware`` afterwards. Kept
    # on ``app.state`` so the ordering contract in the module docstring is
    # something the test suite can check against the real app.
    middleware_list = getattr(app, "user_middleware", None)
    added_middleware: list[Any] = []
    for name, install in discovered:
        if not wildcard and name not in enabled_set:
            continue
        before = len(middleware_list) if isinstance(middleware_list, list) else 0
        try:
            install(app, config)
        except Exception as exc:  # noqa: BLE001 — one bad extension must not brick the proxy
            # A failing extension disables *itself* and the proxy keeps running
            # without it — covers environment/auth failures and compatibility
            # errors (e.g. a plugin built against a core API this version lacks).
            log.warning(
                "proxy extension %r failed to install and was skipped: %s",
                name,
                exc,
                exc_info=True,
            )
            failed.append(name)
            continue
        installed.append(name)
        if isinstance(middleware_list, list):
            added_middleware.extend(middleware_list[: len(middleware_list) - before])
        log.info("proxy extension installed: %s", name)

    state = getattr(app, "state", None)
    if state is not None:
        try:
            state.extension_middleware = added_middleware
        except Exception:  # noqa: BLE001 — a bare test double may not accept attributes
            log.debug("could not record extension middleware on app.state", exc_info=True)

    if failed:
        skipped = ",".join(sorted(failed))
        log.warning("proxy extensions skipped due to install errors: %s", skipped)
        # The startup banner lists enabled extensions *before* install runs, so a
        # skip would otherwise only appear if logging is configured to show this
        # logger. Surface it on the console unconditionally.
        print(
            f"[headroom] proxy extensions SKIPPED: {skipped} "
            f"(install failed — running without them; see logs)",
            file=sys.stderr,
        )

    # Warn about names the user asked for that weren't found.
    if not wildcard:
        missing = enabled_set - set(discovered_names)
        if missing:
            log.warning(
                "proxy extensions requested but not found: %s (available: %s)",
                ",".join(sorted(missing)),
                ",".join(discovered_names) or "<none>",
            )
    return installed
