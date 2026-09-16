"""Third-party CLI subcommand extension point.

External packages add subcommands to the ``headroom`` CLI by declaring an entry
point in the ``headroom.cli_extension`` group in their ``pyproject.toml``:

    [project.entry-points."headroom.cli_extension"]
    my_commands = "my_pkg.cli:register"

Each ``register`` callable is invoked with the root ``click.Group`` and is free
to attach commands or groups to it::

    def register(main: click.Group) -> None:
        main.add_command(my_command)

Unlike :mod:`headroom.proxy.extensions`, this seam is **not** opt-in gated.
Installing the package is the opt-in: adding a subcommand cannot silently
change what an existing command does, so requiring an extra env var before
``headroom <name>`` even appears in ``--help`` would buy no safety.

What *would* be a silent behavior change is shadowing a built-in command, so
that is refused: a registrant may only add names the core CLI does not already
define. Everything else is the extension's business — OSS makes no assumptions
about what the commands do.

Stability contract: this module is load-bearing for any third-party CLI
extensions. Changes to the signature of ``register(main)`` or the entry-point
group name require a deprecation cycle.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Any

import click

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "headroom.cli_extension"

CliExtension = Callable[[Any], None]
"""Signature: ``register(main: click.Group) -> None``."""


def register_all(main: click.Group) -> list[str]:
    """Invoke every registered ``register(main)`` and return the names installed.

    Failures are isolated: a broken third-party package is logged and skipped so
    the rest of the CLI still works. Registrants that try to shadow an existing
    command are rolled back to the built-in and reported, so a stale plugin can
    never quietly take over ``headroom proxy``.
    """
    try:
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 — importlib.metadata can raise varied types
        log.debug("cli extensions: entry-point enumeration failed: %s", exc)
        return []

    installed: list[str] = []
    for entry in entries:
        before = dict(main.commands)
        try:
            register = entry.load()
            register(main)
        except Exception as exc:  # noqa: BLE001 — one bad plugin must not brick the CLI
            log.warning("cli extension %r failed to register and was skipped: %s", entry.name, exc)
            main.commands = before
            continue

        shadowed = [n for n, cmd in before.items() if main.commands.get(n) is not cmd]
        if shadowed:
            log.warning(
                "cli extension %r tried to override built-in command(s) %s and was skipped",
                entry.name,
                ",".join(sorted(shadowed)),
            )
            main.commands = before
            continue

        installed.append(entry.name)
        log.debug("cli extension registered: %s", entry.name)

    return installed
