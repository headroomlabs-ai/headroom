"""Shared cache-safety helpers every edge builds on.

Rules encoded here (see the test protocol's "Cache-safety rules for the edge
layer", and the corrections layered on top after checking them against this
checkout):

1. Compress once, replay forever — deterministic given (content, config
   version). No edge here reads wall-clock time, randomness, or "turns left
   in session" as an input to what it produces, so message k is
   byte-identical across turns as long as config doesn't change mid-session.
2. Deterministic only — folding `config_version` into the stored hash
   (`explicit_hash=`) means a config change can never silently resolve to a
   stale entry stored under the same key.
3. Never touch `thinking` / `redacted_thinking` blocks. Enforced structurally,
   not just by convention: every edge in this package only ever reads
   assistant content (read-only, for the intent query) or mutates
   `tool_result` content inside user messages. `proxy/body_forwarding.py`'s
   `outbound_body_is_client_bytes()` silently reverts the whole request to
   the client's original bytes when a signed-thinking-block request's
   mutation doesn't "survive" — there is no error surfaced when that
   happens, so the only safe rule is to never give it a reason to fire.
4. Every elision is retrievable — original content is always stored in the
   real `CompressionStore` (`headroom/cache/compression_store.py`) before the
   marker referencing it goes out, using one of the two marker conventions
   the engine already recognizes (`content_router.py`'s
   `_ALREADY_COMPRESSED_MARKERS`), not invented syntax.
5. CCR entries expire after `default_ttl` (1800s by default). A marker that's
   still being replayed in message k on turn 40 needs its backing entry
   refreshed, not stored once and forgotten — `refresh_marker` below re-calls
   `store()` with the same `explicit_hash` every time the edge sees it again.
6. Fail open. A network call to a reranker gets a real wall-clock timeout
   (meaningful because it's I/O). CPU-only work (grep parsing, diffing) gets
   an input-size cap instead — a `signal`/thread-based timeout can't preempt
   synchronous CPU work, so bounding the input is what actually protects the
   request path.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from headroom.tokenizers.estimator import EstimatingTokenCounter

logger = logging.getLogger("headroom_edge_layer")

_TOKEN_ESTIMATOR = EstimatingTokenCounter()

# The two marker conventions this package emits, both already recognized by
# headroom/transforms/content_router.py's _is_already_compressed() and by
# headroom/ccr/tool_injection.py's marker scanner, so the engine's own
# retrieve-tool injection and "don't re-compress" guard both just work.
_UNCHANGED_MARKER = (
    "[Unchanged since message {origin_index}: {path}, lines {line_range}. "
    "Retrieve original: hash={hash_key}]"
)
_ELIDED_MARKER = "[{count} {unit} compressed to {kept}. Retrieve more: hash={hash_key}]"

_HEX24_RE = re.compile(r"^[0-9a-f]{24}$")


class SupportsStore(Protocol):
    def store(
        self,
        original: str,
        compressed: str,
        *,
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        query_context: str | None = None,
        ttl: int | None = None,
        explicit_hash: str | None = None,
    ) -> str: ...


def estimate_tokens(text: str) -> int:
    return max(1, _TOKEN_ESTIMATOR.count_text(text))


def config_scoped_hash(content: str, *, config_version: str, salt: str = "") -> str:
    """SHA-256[:24] of content + config_version (+ optional salt).

    This is what gets passed as `explicit_hash=` to `CompressionStore.store()`.
    Folding `config_version` in means a config change is a new key, never a
    silent collision with a stale entry stored under the old config.
    """
    digest = hashlib.sha256(f"{config_version}\x00{salt}\x00{content}".encode()).hexdigest()
    return digest[:24]


@dataclass
class SavingsLedgerEntry:
    """One row logged per edge invocation, per the protocol's ask that every
    edge "logs, per request, the bucket, tokens before and after, and the
    stored hash, so the savings ledger and the recall metric can read it."
    """

    edge: str
    bucket: str
    tool_name: str | None
    tokens_before: int
    tokens_after: int
    hash_key: str | None
    request_id: str


def log_savings(entry: SavingsLedgerEntry) -> None:
    logger.info(
        "edge=%s bucket=%s tool=%s tokens_before=%d tokens_after=%d hash=%s request_id=%s",
        entry.edge,
        entry.bucket,
        entry.tool_name,
        entry.tokens_before,
        entry.tokens_after,
        entry.hash_key,
        entry.request_id,
    )


def store_original(
    store: SupportsStore,
    original: str,
    compressed: str,
    *,
    config_version: str,
    salt: str = "",
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    query_context: str | None = None,
    ttl: int | None = None,
) -> str:
    """Store `original` under a config-scoped explicit hash, return the hash.

    Safe to call every time the same content is seen again (refreshes TTL in
    place — `CompressionStore.store()` overwrites an existing key rather than
    erroring or duplicating).
    """
    hash_key = config_scoped_hash(original, config_version=config_version, salt=salt)
    stored_key = store.store(
        original,
        compressed,
        original_tokens=estimate_tokens(original),
        compressed_tokens=estimate_tokens(compressed),
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        query_context=query_context,
        ttl=ttl,
        explicit_hash=hash_key,
    )
    assert stored_key == hash_key, "explicit_hash must round-trip verbatim"
    return hash_key


def format_unchanged_marker(*, origin_index: int, path: str, line_range: str, hash_key: str) -> str:
    assert _HEX24_RE.match(hash_key), f"hash_key must be 24 lowercase hex chars, got {hash_key!r}"
    return _UNCHANGED_MARKER.format(
        origin_index=origin_index, path=path, line_range=line_range, hash_key=hash_key
    )


def format_elided_marker(*, count: int, unit: str, kept: int, hash_key: str) -> str:
    assert _HEX24_RE.match(hash_key), f"hash_key must be 24 lowercase hex chars, got {hash_key!r}"
    return _ELIDED_MARKER.format(count=count, unit=unit, kept=kept, hash_key=hash_key)


def cap_input(text: str, *, max_chars: int) -> str | None:
    """Return None if `text` exceeds the CPU-side size cap.

    A wall-clock timeout can't interrupt synchronous CPU work (grep parsing,
    diffing) mid-computation in Python, so the real fail-open guard for those
    paths is refusing oversized input up front rather than timing the call.
    Callers pass the original text through unchanged when this returns None.
    """
    if len(text) > max_chars:
        logger.warning(
            "edge input exceeds max_input_chars_per_edge (%d > %d); passing through unchanged",
            len(text),
            max_chars,
        )
        return None
    return text


def already_compressed(text: str) -> bool:
    """Mirrors content_router.py's `_is_already_compressed` so an edge never
    re-touches (and thereby orphans the hash inside) content it or the engine
    already elided.
    """
    return any(
        marker in text for marker in ("Retrieve more: hash=", "Retrieve original: hash=", "<<ccr:")
    )
