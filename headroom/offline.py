"""Air-gap / no-egress master switch (``HEADROOM_OFFLINE``).

THE POLICY, in one sentence: with ``HEADROOM_OFFLINE`` set, Headroom refuses
every connection **Headroom itself decides to make to a destination Headroom
itself chose**, and keeps working for the two kinds of traffic an air-gapped
deployment exists to carry.

Refused (this is not a sample — it is the rule, and the meta-test enforces it):
the telemetry beacon, the update check, the license/usage reporter, OTLP and
Langfuse export, HuggingFace/Kompress/fastembed model downloads, release-binary
and codebase-memory-mcp downloads, eval dataset downloads and the provider SDK
clients the eval harness drives, GitHub Copilot device-flow auth and token
exchange, the Anthropic / Codex / Copilot subscription pollers, the OpenAI
embedders, and both Headroom Cloud compression integrations.

Permitted, explicitly and by name — an air-gapped deployment needs these and
none of them is Headroom phoning home:

``user-traffic``
    The proxy forwarding the caller's own request to the upstream the operator
    configured. Forwarding a request the user's client made, to the endpoint
    the user pointed Headroom at, is the proxy's entire job. An air-gapped
    install points it at an on-prem model endpoint.
``operator-endpoint``
    Headroom dialling an address that comes **entirely** from operator
    configuration and whose default is loopback — today that is exactly one
    thing, the Ollama embedder (``http://localhost:11434``). No hard-coded
    internet host may hide behind this category.
``loopback``
    Hard-coded ``127.0.0.1`` traffic to the operator's own local proxy: the
    readiness and health probes, ``headroom doctor``, the MCP sidecar.
``gated``
    A path that an ``is_offline()`` check upstream already makes unreachable,
    so the connection is never even built.

Those four are the complete list of exceptions. There is no "known violation"
category: a path that dials out under ``HEADROOM_OFFLINE`` for any other
reason is a bug, not an entry in a table.

:func:`guard_egress` is the chokepoint: a path that is about to open an
outbound connection calls it and gets a loud :class:`OfflineEgressBlocked`
instead of a socket. Prefer it over a bare ``if is_offline(): return`` at any
call site that actually dials out — the meta-test in
``tests/test_offline_egress_chokepoint.py`` enumerates the outbound clients in
``headroom/`` and ``crates/`` and requires each **site** to have a guard that
dominates it or to be counted in an allowlist under one of the four categories
above, so the guard is what keeps a newly-added egress path from silently
escaping the air-gap.

A refusal must reach a human as a sentence, never as a traceback and never as
a silent degradation. Interactive paths get that from the CLI boundary in
``headroom/cli/main.py``, which turns the refusal into a Click error; model
loaders translate it into their own "model unavailable"; background pollers
call :func:`note_refusal` and stop.

Kept at the top level (depends only on the stdlib) so any layer — telemetry,
proxy, model code — can import it without creating a package cycle.
"""

from __future__ import annotations

import logging
import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

# Whitespace trimmed off the raw value before matching. Enumerated rather than
# left to ``str.strip()``'s default, because that default is Python's
# ``str.isspace()`` — which includes U+001C-U+001F (the ASCII file/group/record
# /unit separators) — while Rust's ``str::trim()`` is the Unicode White_Space
# property, which does not. ``HEADROOM_OFFLINE=$'\x1c1'`` therefore read as
# offline to Python and online to Rust: one process air-gapped, the other not,
# from one environment variable. Neither side's default is more right than the
# other, so both now trim exactly this set. Mirrored by ``TRIM_CHARS`` in
# ``crates/headroom-core/src/offline.rs``; the parity test compares them.
#
# Case folding needs no such treatment: Python's ``str.lower()`` and Rust's
# ``to_ascii_lowercase()`` differ only on non-ASCII input, and no non-ASCII
# character lowercases into any character of "1"/"true"/"yes"/"on".
_TRIM_CHARS = " \t\n\r\x0b\x0c"

OFFLINE_ENV = "HEADROOM_OFFLINE"


def is_offline() -> bool:
    """Return True when ``HEADROOM_OFFLINE`` selects fully-offline operation."""
    return os.environ.get(OFFLINE_ENV, "").strip(_TRIM_CHARS).lower() in _TRUE_VALUES


def apply_offline_env() -> None:
    """Force HuggingFace/Transformers offline so model code uses only locally
    cached artifacts and never reaches the Hub.

    Idempotent and uses ``setdefault`` so an explicit operator override (e.g.
    ``HF_HUB_OFFLINE=0``) still wins. Call once early in startup.
    """
    if is_offline():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class OfflineEgressBlocked(BaseException):
    """Raised by :func:`guard_egress` when ``HEADROOM_OFFLINE`` is in force.

    A distinct, named type so callers can tell "the operator air-gapped this
    box" apart from "the network was flaky". That distinction matters because
    several Headroom egress paths deliberately fail OPEN on network errors
    (remote Kompress passes content through verbatim, the license reporter
    falls back to a cached grant). Fail-open is right for a flaky endpoint and
    WRONG for a policy refusal: swallowing this turns the air-gap switch back
    into a suggestion.

    **It derives from BaseException, not Exception, on purpose.** The first
    version of this class was a ``RuntimeError`` whose docstring asked every
    broad handler to re-raise it. Nothing did — there were zero re-raise sites
    in the package — and the refusal was swallowed by four reachable
    ``except Exception`` blocks on the way out of a single ``/v1/messages``
    request, which returned 200 with the content uncompressed while the guard
    had fired twice. A convention that has to be re-applied at every future
    ``except Exception`` in a 100k-line codebase is not a guarantee; the type
    hierarchy is. This is the same reason ``KeyboardInterrupt`` and
    ``SystemExit`` sit outside ``Exception``: a policy decision is control
    flow, not a runtime error to degrade around.

    Consequences to know about:

    * ``except Exception`` no longer catches it anywhere — including in
      third-party code (httpx, the OTEL SDK, Starlette's error middleware).
    * ``except BaseException`` still does. Those are rare, they are enumerated
      by ``tests/test_offline_egress_chokepoint.py``'s broad-handler sweep, and
      each must either re-raise or carry a written reason.
    * Reaching this at request time means the operator changed the environment
      under a running proxy; the ordinary contradictory-configuration case is
      caught at startup by ``headroom/proxy/server.py``'s offline preflight,
      which exits 78 with an explanation instead.
    * Where a refusal SHOULD become a degradation, the boundary that owns the
      degradation says so explicitly. Fetching public model weights is not data
      leaving the box, so each optional model loader catches this and re-raises
      its own "model unavailable" error with the switch named — see
      ``kompress_compressor._hf_artifact``, ``image/onnx_router._hf_artifact``
      and the ONNX embedder in ``memory/adapters/embedders.py``. Translating is
      fine; inheriting a translation from whichever ``except Exception`` happens
      to be in the stack is what this type prevents.
    """

    def __init__(self, purpose: str, destination: str | None = None) -> None:
        self.purpose = purpose
        self.destination = destination
        where = f" to {destination}" if destination else ""
        super().__init__(
            f"{OFFLINE_ENV} is set: refusing outbound network access for "
            f"{purpose}{where}. Unset {OFFLINE_ENV}, or turn off the feature "
            f"that needs this connection."
        )


def guard_egress(purpose: str, destination: str | None = None) -> None:
    """The single chokepoint every Headroom-initiated egress path must call.

    Raise :class:`OfflineEgressBlocked` when ``HEADROOM_OFFLINE`` selects
    offline operation; return silently otherwise.

    Call it BEFORE the socket exists — before constructing the client, not
    just before the request — so a pooled/keep-alive connection is never even
    opened. ``purpose`` and ``destination`` land verbatim in the message, so
    an operator who trips this learns which feature to turn off.

    Why raise instead of returning a no-op result: a silent skip is
    indistinguishable from success at the call site, so a future refactor can
    quietly reintroduce egress and nothing fails. A loud, named exception is
    what makes the meta-test in ``tests/test_offline_egress_chokepoint.py``
    able to assert "every egress path is behind this or allowlisted".

    Deliberately NOT exempted:

    * Loopback / in-cluster destinations. The guard cannot reliably tell an
      in-cluster collector from an internet host (DNS, proxies and sidecars
      all blur it), and ``HEADROOM_OFFLINE`` is documented as "no outbound
      traffic". Paths that only ever talk to ``127.0.0.1`` — the readiness
      probes, the proxy's own ``/livez`` check — simply do not call the guard
      and are allowlisted by name in the meta-test instead.
    * The proxy's own request forwarding to the caller's configured upstream.
      That is the caller's traffic, not Headroom phoning home; an air-gapped
      deployment points it at an on-prem endpoint and still needs it to work.
    """
    if is_offline():
        raise OfflineEgressBlocked(purpose, destination)


# Purposes already reported, so a poller that runs every 30s says it once at
# WARNING and then keeps quiet. Module-level rather than per-object because
# several of these pollers are recreated per poll.
_REPORTED_REFUSALS: set[str] = set()


def note_refusal(blocked: OfflineEgressBlocked, log: logging.Logger) -> str:
    """Report an air-gap refusal on a background path and return its message.

    Background pollers and other fire-and-forget tasks cannot let a
    ``BaseException`` escape — an unhandled task exception is a traceback in
    the log, not an explanation, and asyncio may not surface it until the
    task is garbage collected. They also must not swallow the refusal
    silently, because "the subscription window stopped updating" with no
    stated cause is the exact confusion this switch was supposed to end.

    So: catch :class:`OfflineEgressBlocked` specifically, call this, and stop.
    The first refusal for a given purpose is logged at WARNING with the switch
    named; later ones drop to DEBUG so a 30-second poller does not flood the
    log with a decision the operator already made.
    """
    message = str(blocked)
    if blocked.purpose in _REPORTED_REFUSALS:
        log.debug("%s", message)
    else:
        _REPORTED_REFUSALS.add(blocked.purpose)
        log.warning("%s", message)
    return message
