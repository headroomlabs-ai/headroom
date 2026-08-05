"""Privacy-preserving query identifiers for proxy logs."""

from __future__ import annotations

import hashlib

QUERY_LOG_HASH_BYTES = 8


def hash_query_for_log(query: str) -> str:
    """Stable short hash of a memory-context query, safe to log."""
    h = hashlib.blake2b(query.encode("utf-8", errors="replace"), digest_size=QUERY_LOG_HASH_BYTES)
    return h.hexdigest()
