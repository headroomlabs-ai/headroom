"""Privacy-preserving query identifiers for proxy logs."""

from __future__ import annotations

import hashlib
import logging
from urllib.parse import parse_qsl, quote

QUERY_LOG_HASH_BYTES = 8


def hash_query_for_log(query: str) -> str:
    """Stable short hash of a memory-context query, safe to log."""
    h = hashlib.blake2b(query.encode("utf-8", errors="replace"), digest_size=QUERY_LOG_HASH_BYTES)
    return h.hexdigest()


def redact_query_string_for_log(query: str) -> str:
    """Keep a URL query's parameter names and drop the values.

    A value can be a credential (Google routes take ``?key=<api key>``), and no
    rule tells a credential from an ordinary value, so every value is dropped.
    Names are percent-encoded, so they cannot carry a newline into the log.
    """
    names = [name for name, _ in parse_qsl(query, keep_blank_values=True)]
    # parse_qsl decodes the names; encode them again so a %0A in a name cannot
    # start a new log line.
    return "&".join(f"{quote(name, safe='')}=<redacted>" for name in names)


class AccessLogQueryFilter(logging.Filter):
    """Drop query values from uvicorn access-log records.

    uvicorn logs ``(client, method, full_path, http_version, status)`` with the
    raw query, so ``?key=<api key>`` would reach the log at info level.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) == 5 and isinstance(args[2], str):
            path, sep, query = args[2].partition("?")
            if sep:
                record.args = (*args[:2], f"{path}?{redact_query_string_for_log(query)}", *args[3:])
        return True


def install_access_log_query_filter() -> None:
    """Attach ``AccessLogQueryFilter`` to ``uvicorn.access`` once per process."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, AccessLogQueryFilter) for f in access.filters):
        access.addFilter(AccessLogQueryFilter())
