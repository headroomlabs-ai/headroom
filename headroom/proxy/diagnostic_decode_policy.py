"""Lossy byte decoding policy for diagnostics and logs."""

from __future__ import annotations

import codecs


def safe_decode_for_logging(raw: bytes, *, max_bytes: int | None = None) -> str:
    """Decode bytes to a string for log/diagnostic display only.

    Wire/protocol parsers should decode complete protocol frames strictly. This
    policy is for already-discarded diagnostics where replacement characters are
    preferable to failing the error-reporting path.
    """
    blob = raw[:max_bytes] if max_bytes is not None else raw
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    return decoder.decode(bytes(blob), final=True)
