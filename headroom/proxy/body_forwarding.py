"""Byte-faithful outbound request body forwarding policy.

This module owns the small algebra used by Python proxy forwarders to decide
which bytes leave Headroom:

* unmutated body with original bytes -> byte-for-byte passthrough
* signed thinking blocks with original bytes -> byte-for-byte passthrough
* mutated body or missing original bytes -> canonical JSON bytes
* explicit rollback mode -> legacy httpx-style JSON bytes
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from headroom.proxy import python_forwarder_mode_policy

_PYTHON_FORWARDER_MODE_ENV = python_forwarder_mode_policy.PYTHON_FORWARDER_MODE_ENV

PythonForwarderMode = python_forwarder_mode_policy.PythonForwarderMode
OutboundBodySource = Literal["passthrough", "canonical", "legacy"]

_PYTHON_FORWARDER_MODE_DEFAULT = python_forwarder_mode_policy.PYTHON_FORWARDER_MODE_DEFAULT


@dataclass(frozen=True, slots=True)
class OutboundBody:
    """Concrete outbound body bytes plus their provenance."""

    content: bytes
    source: OutboundBodySource
    #: True when byte-faithful passthrough won over a mutated body, so every
    #: edit the handler made to ``body`` was discarded before the wire. Callers
    #: that gated a downstream decision on their own mutation (for example
    #: flipping ``stream`` to False to buffer a reply) MUST consult this — the
    #: request upstream actually sees is the client's original one.
    dropped_mutations: bool = False
    #: The ``BodyMutationTracker`` reasons discarded alongside it, when the
    #: caller supplied them. Empty when the caller passed no reason list.
    dropped_mutation_reasons: tuple[str, ...] = ()


def get_python_forwarder_mode() -> PythonForwarderMode:
    """Return the active Python-forwarder mode.

    Read at request time. Unknown values raise loudly per the no-silent-
    fallback build constraint. The ``legacy_json_kwarg`` value is an
    explicit operator opt-in for emergency rollback, not a fallback.
    """
    return python_forwarder_mode_policy.resolve_python_forwarder_mode(
        os.environ.get(_PYTHON_FORWARDER_MODE_ENV)
    )


def serialize_body_canonical(body: dict[str, Any]) -> bytes:
    """Re-serialize a request body deterministically with cache-stable formatting.

    ``ensure_ascii=False`` keeps the bytes compact and cache-stable, but it also
    means a lone surrogate anywhere in the body raises here. That is reachable
    input, not a hypothetical: ``"\\ud800"`` is valid JSON, ``json.loads``
    accepts it happily, and a tool result carrying truncated UTF-16 or sliced
    binary produces one. Both forwarders resolve outbound bytes *outside* their
    connection-retry loop, so the exception escapes as a 500 with no retry.

    #3124 made that newly load-bearing: mutated thinking-bearing bodies used to
    return the client's bytes verbatim and never reached this function at all,
    so the largest, most tool-result-heavy population in Claude Code traffic now
    depends on it not raising.

    The escaped form is the right degradation -- it encodes the identical parsed
    values, so upstream reconstructs exactly the same request, and every mutation
    still reaches the wire (important: the caller's ``stream`` flip rides on
    these bytes). Only the byte-level encoding differs, costing one cache miss on
    a request that would otherwise have failed outright.
    """
    try:
        return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(body, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def has_signed_thinking_blocks(body: dict[str, Any]) -> bool:
    """Return whether message history contains Anthropic signed thinking blocks."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {
                "thinking",
                "redacted_thinking",
            }:
                return True
    return False


#: A body carrying a ``thinking`` or ``redacted_thinking`` block contains this
#: substring, because it appears in the block's own ``type`` value. Scanning for
#: it is orders of magnitude cheaper than parsing, and request bodies here reach
#: 9.3 MB in real agent traffic -- so this prescreen keeps the common case (no
#: thinking anywhere, ~2 of every 3 requests) from paying a full JSON parse it
#: cannot learn anything from.
#:
#: A false positive costs one parse we would have done anyway. A false negative
#: is not strictly impossible -- a ``type`` value whose ASCII letters are written
#: as JSON unicode escapes still parses to ``thinking`` while containing no
#: literal match, and that is legal JSON no standard encoder emits
#: (``json.dumps`` does not escape ASCII even under
#: ``ensure_ascii=True``, and neither does any client we forward for) -- and its
#: consequences are asymmetric in our favour: when the mutated body still holds
#: the block, ``has_signed_thinking_blocks(body)`` sees it on the parsed dict and
#: we lock anyway. The single reachable gap needs an escaped ``type`` in the
#: original AND a transform that removed the block from the body, which is the
#: rare #3015 shape crossed with an encoder nobody uses. Documented rather than
#: defended, because closing it means re-parsing every large body to catch a case
#: no observed client can produce.
_THINKING_SUBSTRING = b"thinking"


def _parse_original_body(original_body_bytes: bytes | None) -> dict[str, Any] | None:
    """Parse the client's body once, or ``None`` when it cannot be used.

    Every caller here needs the same parsed document, and a 9.3 MB body parsed
    twice per decision (and twice again in the handler's earlier probe) is real
    latency on a stage that already has a 30s timeout whose expiry quarantines
    compression process-wide. Parse in one place and pass the result down.

    ``None`` means "cannot prove anything from the original", which every caller
    must treat as the conservative answer.
    """
    if original_body_bytes is None:
        return None
    if _THINKING_SUBSTRING not in original_body_bytes:
        return None
    try:
        original = json.loads(original_body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, MemoryError, RecursionError):
        return None
    return original if isinstance(original, dict) else None


def _original_body_has_signed_thinking_blocks(original_body_bytes: bytes | None) -> bool:
    original = _parse_original_body(original_body_bytes)
    return original is not None and has_signed_thinking_blocks(original)


#: Allow canonical re-serialization on a thinking-bearing request when every
#: thinking block is provably unchanged. **Default ON**; set this to ``0`` (or
#: ``false``/``no``/``off``) to restore the previous blanket lock.
#:
#: The lock this relaxes was added by #2254 after real upstream 400s
#: ("`thinking` blocks ... cannot be modified").
#:
#: What the signature actually covers is now measured, not assumed --
#: ``tests/test_thinking_signature_scope_live.py`` pins it against the live API
#: on sonnet-4-5, opus-4-5, sonnet-4-6, sonnet-5 and opus-5, with identical
#: results on all five. Anthropic accepts a replayed turn whose signed thinking
#: block is intact while we compress a ``tool_result``, rewrite sibling
#: ``text``/``tool_use`` blocks *inside the same assistant message*, rewrite
#: top-level ``system``/``tools``, or re-serialize the whole body with reordered
#: keys. It rejects exactly one thing: a forged ``signature`` ("Invalid
#: `signature` in `thinking` block"), which is the negative control proving the
#: endpoint validates signatures on this shape at all -- without it every
#: acceptance above would be vacuous.
#:
#: So the seal binds the block, not the request, and #2254's stated cause (a
#: plain canonical re-encode) is disproven directly: variant F changes the bytes
#: and is accepted. The 400s were real but were never traced to their true
#: trigger. This relaxation stays narrower than the evidence permits anyway --
#: it forwards edits ONLY when every thinking block is byte-identical to the
#: client's -- so the measurements above are headroom, not the safety margin.
#:
#: If Anthropic ever changes this, that live test fails loudly, and setting the
#: env var to ``0`` is a single-variable, no-deploy rollback to the blanket lock.
THINKING_PRESERVING_MUTATIONS_ENV = "HEADROOM_THINKING_PRESERVING_MUTATIONS"


def thinking_preserving_mutations_enabled() -> bool:
    """Whether to forward edits that provably left every thinking block intact.

    Defaults to enabled. Only an explicit falsey value restores the blanket lock,
    so an unset or unparseable variable keeps the documented default rather than
    silently reverting behaviour.
    """
    raw = os.environ.get(THINKING_PRESERVING_MUTATIONS_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def thinking_block_fingerprint(body: Any) -> list[tuple[int, int, str]]:
    """Positional, order-sensitive fingerprint of every thinking block.

    Each entry is ``(message_index, block_index, canonical_json_of_block)``, so
    the comparison catches a block whose text or ``signature`` changed, one that
    was added, removed, reordered, or moved between messages. Keys are sorted so
    a dict rebuilt in a different order is not mistaken for an edit -- the wire
    contract is over parsed values, not key order.
    """
    out: list[tuple[int, int, str]] = []
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return out
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") in {
                "thinking",
                "redacted_thinking",
            }:
                out.append(
                    (
                        message_index,
                        block_index,
                        json.dumps(block, sort_keys=True, ensure_ascii=False),
                    )
                )
    return out


def thinking_blocks_survived_mutation(
    body: dict[str, Any],
    original_body_bytes: bytes | None,
    original: dict[str, Any] | None = None,
) -> bool:
    """True when every thinking block is byte-equal to the one the client sent.

    This is the whole point of the relaxation. Anthropic signs the thinking
    block, not the request: the signature covers that block's own content, so
    edits to a ``tool_result`` twenty turns back, or to the top-level ``tools``
    and ``system`` fields (which are not even inside ``messages`` and therefore
    cannot be covered by any per-block signature), leave every seal intact.
    Treating the presence of a sealed block as a reason to freeze the entire
    body is a category error -- it protects bytes the signature says nothing
    about, and on Claude Code traffic that is nearly the whole request.

    That scope claim is measured against the live API, not inferred -- see
    ``tests/test_thinking_signature_scope_live.py``, which also pins the one
    thing Anthropic does reject (a forged ``signature``).

    Conservative by construction: any parse failure, or any detectable
    difference at all, returns False and the caller keeps today's passthrough.

    Pass ``original`` when the caller has already parsed the client body, so a
    multi-megabyte document is not re-parsed once per predicate.
    """
    if original is None:
        original = _parse_original_body(original_body_bytes)
    if original is None:
        return False
    try:
        return thinking_block_fingerprint(body) == thinking_block_fingerprint(original)
    except (TypeError, ValueError, RecursionError):
        # An unserializable block (or pathological nesting) means we cannot prove
        # the blocks are untouched, so we must not claim they are.
        return False


class BodyMutationTracker:
    """Records whether a request body was mutated and why."""

    __slots__ = ("_mutated", "_reasons")

    def __init__(self) -> None:
        self._mutated: bool = False
        self._reasons: list[str] = []

    def mark_mutated(self, reason: str) -> None:
        """Mark the body as mutated and record the stable aggregation reason."""
        if not reason:
            raise ValueError("BodyMutationTracker.mark_mutated: reason must be non-empty")
        self._mutated = True
        if reason not in self._reasons:
            self._reasons.append(reason)

    @property
    def mutated(self) -> bool:
        return self._mutated

    @property
    def reasons(self) -> list[str]:
        return list(self._reasons)


def select_outbound_body(
    *,
    body: dict[str, Any],
    original_body_bytes: bytes | None,
    body_mutated: bool,
    forwarder_mode: PythonForwarderMode | None = None,
    mutation_reasons: list[str] | None = None,
) -> OutboundBody:
    """Select the exact bytes to forward upstream.

    ``mutation_reasons`` is optional and only used for reporting: when the
    signed-thinking passthrough overrides a mutated body, the discarded reasons
    are echoed back on the result so the call site can log what never made it
    upstream instead of silently claiming the edit landed.
    """
    mode = forwarder_mode if forwarder_mode is not None else get_python_forwarder_mode()
    # Parse the client body at most once for the whole decision (bodies here
    # reach 9.3 MB in real traffic; this used to cost two full parses).
    original_parsed = _parse_original_body(original_body_bytes)
    if original_body_bytes is not None and (
        has_signed_thinking_blocks(body)
        or (original_parsed is not None and has_signed_thinking_blocks(original_parsed))
    ):
        # The lock is about the SEAL, not the box. When every thinking block is
        # provably identical to the one the client sent, no signature can have
        # been invalidated, so the remaining edits (compressed tool_result text,
        # compacted tool schemas, tool-search deferral) are safe to forward.
        # Without this, one thinking block anywhere in history froze the entire
        # request for the rest of the session -- on real Claude Code traffic that
        # discarded every computed compression on 34% of requests.
        if thinking_preserving_mutations_enabled() and thinking_blocks_survived_mutation(
            body, original_body_bytes, original=original_parsed
        ):
            if mode == "legacy_json_kwarg":
                content = json.dumps(body, separators=(", ", ": "), ensure_ascii=True).encode(
                    "utf-8"
                )
                return OutboundBody(content=content, source="legacy")
            if body_mutated:
                return OutboundBody(content=serialize_body_canonical(body), source="canonical")
            return OutboundBody(content=original_body_bytes, source="passthrough")
        return OutboundBody(
            content=original_body_bytes,
            source="passthrough",
            dropped_mutations=body_mutated,
            dropped_mutation_reasons=tuple(mutation_reasons or ()) if body_mutated else (),
        )

    if mode == "legacy_json_kwarg":
        content = json.dumps(body, separators=(", ", ": "), ensure_ascii=True).encode("utf-8")
        return OutboundBody(content=content, source="legacy")

    if body_mutated or original_body_bytes is None:
        return OutboundBody(content=serialize_body_canonical(body), source="canonical")
    return OutboundBody(content=original_body_bytes, source="passthrough")


def prepare_outbound_body_bytes(
    *,
    body: dict[str, Any],
    original_body_bytes: bytes | None,
    body_mutated: bool,
    forwarder_mode: PythonForwarderMode | None = None,
    mutation_reasons: list[str] | None = None,
) -> tuple[bytes, OutboundBodySource]:
    """Compatibility tuple wrapper around :func:`select_outbound_body`.

    Keeps the two-value shape its existing callers unpack. Call
    :func:`select_outbound_body` directly when you need the dropped-mutation
    reporting.
    """
    outbound = select_outbound_body(
        body=body,
        original_body_bytes=original_body_bytes,
        body_mutated=body_mutated,
        forwarder_mode=forwarder_mode,
        mutation_reasons=mutation_reasons,
    )
    return outbound.content, outbound.source


def outbound_body_is_client_bytes(
    *,
    body: dict[str, Any],
    original_body_bytes: bytes | None,
) -> bool:
    """Return whether the wire body will be the client's original bytes.

    A handler that changes ``body`` to steer its own upstream call — the
    ``stream`` flip that buys a buffered reply is the load-bearing case — has to
    know that the signed-thinking passthrough will throw that change away and
    send the client's bytes verbatim. Asking before acting is cheaper than
    discovering it from a reply in the wrong wire format.

    Mirrors the first branch of :func:`select_outbound_body`, INCLUDING the
    thinking-preserving relaxation. These two must agree exactly: the caller
    flips ``stream`` to False to buy a buffered reply, and that flip only lands
    if the mutated body actually reaches the wire. If this said "locked" while
    ``select_outbound_body`` shipped canonical bytes, we would ask upstream for a
    stream and then parse the reply as buffered JSON -- the 200-with-empty-body
    failure from #2952, in reverse.

    The forwarder mode is deliberately not consulted: the lock branch overrides
    it, and the relaxation only ever returns bytes derived from ``body``, so
    either way the caller's edits are on the wire.
    """
    if original_body_bytes is None:
        return False
    original_parsed = _parse_original_body(original_body_bytes)
    if not (
        has_signed_thinking_blocks(body)
        or (original_parsed is not None and has_signed_thinking_blocks(original_parsed))
    ):
        return False
    if thinking_preserving_mutations_enabled() and thinking_blocks_survived_mutation(
        body, original_body_bytes, original=original_parsed
    ):
        return False
    return True
