"""Per-session memoized message token counts (perf review F2).

Token mode does up to 6 full-transcript ``count_messages`` calls per request,
two of them synchronous on the event loop. The frozen prefix never changes
across turns, so only the delta (new/changed messages) needs a real count —
everything else is a cache hit keyed by message content.

Correctness relies on an algebraic identity::

    tokenizer.count_messages(messages)
        == sum(tokenizer.count_message(m) for m in messages) + tokenizer.REPLY_OVERHEAD

which holds for ``BaseTokenizer``'s default per-message loop (and every
subclass that just delegates to it — ``EstimatingTokenCounter``,
``TiktokenCounter``). This is the ONLY tokenizer family the proxy uses for
Claude models: ``tokenizers/registry.py``'s ``_create_anthropic`` always
returns ``EstimatingTokenCounter``. The identity does NOT hold for tokenizers
that encode the whole conversation jointly (``HuggingFaceTokenizer``'s
``apply_chat_template`` path, ``MistralTokenizer``'s
``encode_chat_completion`` path) — do not memoize with those.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Protocol


class _MessageCountingTokenizer(Protocol):
    REPLY_OVERHEAD: int

    def count_message(self, message: dict[str, Any]) -> int: ...

    def count_messages(self, messages: list[dict[str, Any]]) -> int: ...


class TokenCountMemo:
    """LRU cache of per-message token counts, keyed by message content hash.

    Bound to one tokenizer instance at a time: counts computed under one
    tokenizer must never be served for another (different ratios/backends),
    so a tokenizer swap — e.g. the handler's count fail-open path downgrading
    to a fresh ``EstimatingTokenCounter`` mid-session — clears the memo and
    rebinds. The registry caches tokenizer instances per model, so on the
    normal path the binding is stable across a session's requests.

    The store is per-session and a session's requests run concurrently, so
    binding alone is not enough: request A can bind tokenizer A, release the
    lock, and still be mid-count when request B rebinds to tokenizer B (the
    fail-open path builds a *fresh* estimator per request — see
    ``proxy/token_counting._count_offloaded``). ``get``/``put`` therefore
    re-check tokenizer identity under the same lock that owns the binding, so
    an obsolete binding's reads miss and its writes are dropped rather than
    leaking counts across tokenizers. That makes each lookup/populate atomic
    with respect to tokenizer identity without holding the lock across the
    (slow) tokenizer call itself.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._counts: OrderedDict[str, list[tuple[str, int]]] = OrderedDict()
        # Strong reference on purpose: identity (`is`) comparison against a
        # weak/GC'd tokenizer could alias a new instance at the same address.
        self._bound_tokenizer: object | None = None

    def bind_or_reset(self, tokenizer: object) -> None:
        """Bind this memo to ``tokenizer``, clearing stale counts on a swap."""
        with self._lock:
            if self._bound_tokenizer is not tokenizer:
                if self._bound_tokenizer is not None:
                    self._counts.clear()
                self._bound_tokenizer = tokenizer

    @staticmethod
    def canonical_message(message: dict[str, Any]) -> str:
        """Return the exact serialized value verified on every cache hit."""
        try:
            return json.dumps(message, sort_keys=True, ensure_ascii=False)
        except Exception:
            return repr(message)

    @staticmethod
    def message_hash(message: dict[str, Any]) -> str:
        """Hash the whole message — role, content, tool_calls, function_call,
        name — every field ``Tokenizer.count_message`` sums over, so a memo
        hit guarantees the identical count. (Not reusing
        ``CompressionCache.content_hash``: that hashes tool_result CONTENT
        only, not the whole message.)
        """
        raw = TokenCountMemo.canonical_message(message)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str, canonical: str | None = None, *, tokenizer: object) -> int | None:
        """Read a count, but only for the tokenizer currently bound.

        ``tokenizer`` is required so no caller can silently opt out of the
        identity check: a stale binding must miss, not serve another
        tokenizer's numbers.
        """
        with self._lock:
            if self._bound_tokenizer is not tokenizer:
                return None
            entries = self._counts.get(key)
            if entries is None:
                return None
            self._counts.move_to_end(key)
            if canonical is None:
                return entries[0][1]
            return next((count for stored, count in entries if stored == canonical), None)

    def put(self, key: str, count: int, canonical: str | None = None, *, tokenizer: object) -> None:
        """Store a count, dropping writes from an obsolete tokenizer binding.

        A concurrent rebind (another request's tokenizer swap) already cleared
        the store for its own tokenizer; letting this write land would mark a
        stale count as belonging to the new binding.
        """
        with self._lock:
            if self._bound_tokenizer is not tokenizer:
                return
            canonical = key if canonical is None else canonical
            entries = self._counts.pop(key, [])
            entries = [(stored, value) for stored, value in entries if stored != canonical]
            entries.append((canonical, count))
            self._counts[key] = entries
            while len(self._counts) > self.max_entries:
                self._counts.popitem(last=False)

    def get_stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._counts)}


def count_messages_memoized(
    memo: TokenCountMemo,
    tokenizer: _MessageCountingTokenizer,
    messages: list[dict[str, Any]],
) -> int:
    """``tokenizer.count_messages(messages)``, memoizing each message's count.

    Only correct for additive tokenizers — see module docstring. Enforced at
    runtime via the ``ADDITIVE_COUNTS`` capability flag (``tokenizers/base``):
    non-additive or unflagged tokenizers get a plain (correct, unmemoized)
    ``count_messages`` call instead. The memo is also bound to the tokenizer
    instance — a swap (count fail-open downgrade) clears stale counts rather
    than serving numbers computed under a different tokenizer.

    Every ``get``/``put`` re-asserts that identity under the memo lock, so a
    concurrent rebind on the shared per-session memo degrades this call to
    uncached counting (all values still produced by ``tokenizer``) instead of
    trading counts with the other request's tokenizer.
    """
    if not getattr(tokenizer, "ADDITIVE_COUNTS", False):
        return tokenizer.count_messages(messages)
    memo.bind_or_reset(tokenizer)
    total = 0
    for msg in messages:
        canonical = TokenCountMemo.canonical_message(msg)
        key = TokenCountMemo.message_hash(msg)
        count = memo.get(key, canonical, tokenizer=tokenizer)
        if count is None:
            count = tokenizer.count_message(msg)
            memo.put(key, count, canonical, tokenizer=tokenizer)
        total += count
    return total + tokenizer.REPLY_OVERHEAD
