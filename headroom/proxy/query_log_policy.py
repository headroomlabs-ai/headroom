"""Privacy-preserving query identifiers for proxy logs."""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl

QUERY_LOG_HASH_BYTES = 8


def hash_query_for_log(query: str) -> str:
    """Stable short hash of a memory-context query, safe to log."""
    h = hashlib.blake2b(query.encode("utf-8", errors="replace"), digest_size=QUERY_LOG_HASH_BYTES)
    return h.hexdigest()


def redact_query_string_for_log(query: str) -> str:
    """Keep a URL query's parameter names and drop the values.

    A value can be a credential (Google routes take ``?key=<api key>``), and no
    rule tells a credential from an ordinary value, so every value is dropped.
    """
    names = [name for name, _ in parse_qsl(query, keep_blank_values=True)]
    return "&".join(f"{name}=<redacted>" for name in names)
