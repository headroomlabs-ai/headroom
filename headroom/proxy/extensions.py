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
  * raise on license / environment failure to abort startup

OSS makes no assumptions about what extensions do. The interface is
deliberately minimal; extensions own the complexity behind it.

**Extensions are opt-in.** Discovery enumerates every registered extension,
but ``install_all`` only invokes those explicitly enabled by the operator.
This protects users from silent behavior changes when a package they didn't
audit gets installed in the same environment (e.g., as a transitive dep).

Enabling extensions:

  * CLI:  ``headroom proxy --proxy-extension shield_enterprise,mypkg``
  * Env:  ``HEADROOM_PROXY_EXTENSIONS=shield_enterprise,mypkg``
  * Wildcard: ``--proxy-extension '*'`` enables every discovered extension
    (use only when you trust everything in your environment).

Stability contract: this module is load-bearing for the Enterprise build and
any third-party extensions. Changes to the signature of ``install(app, config)``
or the entry-point group name require a deprecation cycle.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "headroom.proxy_extension"
ENV_VAR = "HEADROOM_PROXY_EXTENSIONS"

ProxyExtension = Callable[[Any, Any], None]
"""Signature: ``install(app: FastAPI, config: ProxyConfig) -> None``."""


def discover() -> Iterator[tuple[str, ProxyExtension]]:
    """Yield ``(name, install_callable)`` pairs for every registered extension.

    Entry-point load failures are logged and skipped — a broken third-party
    package must not prevent the proxy from starting. An extension that wants
    to fail-closed can raise from its ``install()``.
    """
    try:
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 — importlib.metadata can raise varied types
        log.debug("proxy extensions: entry-point enumeration failed: %s", exc)
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
    raises inside ``install()``, the exception propagates — this is the
    documented fail-closed signal (e.g., a license check failing should
    abort startup rather than silently run without protection).
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
    for name, install in discovered:
        if not wildcard and name not in enabled_set:
            continue
        install(app, config)
        installed.append(name)
        log.info("proxy extension installed: %s", name)

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
