"""Live (per-request) env knobs and a hot-reload override store.

Most Headroom settings are read once at proxy startup into ``Config`` and are
visible in ``/health``. A second, smaller class of environment variables is
read *live* — on every request (the output-shaper family) or captured at module
import (the ast-grep read-rewrite threshold). The proxy reads these from its own
process environment, so a *reused* proxy — one ``headroom wrap`` attaches to
rather than starting fresh — never sees values a user exports afterwards. The
fix without this module would be to restart the proxy, which is disruptive
(cold-start of the ML stack, dropped in-flight requests, lost compression
caches).

This module is the single source of truth for that class of knob and provides a
process-global override store. ``headroom wrap`` pushes the values it would
otherwise only be able to apply by restarting (``POST /admin/runtime-env``), and
the proxy applies them in memory with no restart. Readers call :func:`getenv`
instead of ``os.environ.get`` so an override wins over the launch-time
environment; with no override set, behaviour is byte-for-byte identical to
reading the environment directly.

Scope rule: a variable belongs here only if the proxy reads it *after* startup
(or captures it at import) AND it is not already reflected in the ``/health``
``config`` block that ``wrap`` compares for reuse. Startup-captured settings
(``HEADROOM_TARGET_RATIO`` etc.) do not belong here — a fresh proxy already
gets them and they ride the existing config channel.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import overload


@dataclass(frozen=True)
class Knob:
    """One reuse-invalidating live env var.

    ``env`` is the environment variable name (also the key used in ``/health``
    and in the hot-reload payload). ``kind`` is advisory metadata for the
    ``/health`` surface and validation — readers still parse the raw string
    exactly as they did when reading the environment directly, so a knob's
    parsing/clamping semantics live with its reader, not here.
    """

    env: str
    kind: str  # "bool" | "int" | "float" | "str"
    summary: str


# The registry. Adding a knob here is all it takes to make a live env var
# hot-reloadable and visible in /health. Keep this list to genuinely live knobs
# (see the scope rule in the module docstring).
RUNTIME_ENV_KNOBS: tuple[Knob, ...] = (
    Knob("HEADROOM_OUTPUT_SHAPER", "bool", "Master switch for output-token shaping."),
    Knob(
        "HEADROOM_VERBOSITY_LEVEL", "int", "Verbosity steering level 0-4 (unset = learned/default)."
    ),
    Knob("HEADROOM_EFFORT_ROUTER", "bool", "Lower effort on mechanical tool-result continuations."),
    Knob("HEADROOM_MECHANICAL_EFFORT", "str", "Effort value used on mechanical continuations."),
    Knob("HEADROOM_VERBOSITY_AUTOTUNE", "bool", "Use the AIMD verbosity controller state."),
    Knob(
        "HEADROOM_OUTPUT_HOLDOUT",
        "float",
        "Fraction of conversations held out for A/B measurement.",
    ),
    Knob(
        "HEADROOM_INTERCEPT_READ_MIN_CHARS",
        "int",
        "Min tool-output chars before the ast-grep read rewrite.",
    ),
)

_KNOBS_BY_ENV: dict[str, Knob] = {k.env: k for k in RUNTIME_ENV_KNOBS}

# Process-global override store. Writes take the lock; reads are a plain
# ``dict.get`` (atomic in CPython) so the per-request hot path stays lock-free.
_lock = threading.Lock()
_overrides: dict[str, str] = {}


@overload
def getenv(name: str, default: str) -> str: ...


@overload
def getenv(name: str, default: None = ...) -> str | None: ...


def getenv(name: str, default: str | None = None) -> str | None:
    """Return the live value for ``name``: hot-reload override, else environment.

    Drop-in for ``os.environ.get`` at the reader site. When no override has been
    pushed, this is exactly ``os.environ.get(name, default)``. Overloaded like
    ``os.environ.get`` so a string default yields ``str`` — callers can ``.lower()``
    or ``int(...)`` the result without a None-check.
    """
    override = _overrides.get(name)
    if override is not None:
        return override
    return os.environ.get(name, default)


def set_overrides(values: dict[str, object]) -> dict[str, str]:
    """Apply hot-reload overrides for known knobs. Returns what was applied.

    Unknown keys and non-string values are ignored (the endpoint is loopback-only
    but we still never trust the body blindly). Storing the raw string preserves
    each reader's own parsing/clamping semantics.
    """
    applied: dict[str, str] = {}
    with _lock:
        for key, value in values.items():
            if key not in _KNOBS_BY_ENV or not isinstance(value, str):
                continue
            _overrides[key] = value
            applied[key] = value
    return applied


def clear_overrides() -> None:
    """Drop all overrides (used by tests and to reset state)."""
    with _lock:
        _overrides.clear()


def explicit_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Knobs *explicitly* set (non-empty) in ``environ`` — the wrap push payload.

    Only explicitly-set knobs are pushed so a session never clobbers another
    session's setting with a default it never asked for. To force a knob back to
    a default on a shared proxy, set it explicitly (e.g. ``HEADROOM_OUTPUT_SHAPER=0``).
    """
    src = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for knob in RUNTIME_ENV_KNOBS:
        raw = src.get(knob.env)
        if raw is not None and raw.strip() != "":
            out[knob.env] = raw
    return out


def effective_runtime_env() -> dict[str, str | None]:
    """The live value of every knob (override-or-environment) for ``/health``.

    ``None`` means the knob is unset, so the reader will fall back to its own
    default. This is what the proxy will actually use on the next request.
    """
    return {knob.env: getenv(knob.env) for knob in RUNTIME_ENV_KNOBS}
