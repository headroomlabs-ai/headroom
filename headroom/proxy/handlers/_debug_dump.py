"""Shared helpers for diagnostic dumps of upstream-error (>=400) requests.

The dump can contain cleartext prompt / tool / system content, so it is OFF by
default, is never written in stateless mode, and content is redacted unless the
operator explicitly opts in to full content. Used by both the Anthropic and the
OpenAI handlers so the gating stays consistent across providers.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from headroom.log_safety import content_logging_enabled, redact_url
from headroom.proxy.wire_debug_redaction_policy import WIRE_DEBUG_REDACTED, redact_for_wire_debug

logger = logging.getLogger(__name__)


def _debug_dump_mode(config: Any) -> str:
    """Return the diagnostic-dump mode for upstream (>=400) errors.

    - "off"      : nothing written (default, and forced in stateless mode)
    - "redacted" : structure, roles, and lengths only — content elided
                   (``HEADROOM_DEBUG_DUMP=1``/``true``/``on``/``redacted``)
    - "full"     : everything including content (``HEADROOM_DEBUG_DUMP=full``)
    """
    if getattr(config, "stateless", False):
        return "off"
    if content_logging_enabled():
        return "full"
    raw = os.environ.get("HEADROOM_DEBUG_DUMP", "").strip().lower()
    if raw in ("1", "true", "yes", "on", "redacted"):
        return "redacted"
    return "off"


def _redact_debug_value(value: Any, _max_len: int = 80) -> Any:
    """Recursively elide long strings (likely prompt/tool content) while keeping
    structure, roles, type tags, ids, and other short fields for debugging.

    Note: short strings (<= ``_max_len``) are preserved, so the redacted tier is
    not a guarantee against leaking very short sensitive values — it is a
    best-effort convenience. The default ("off") writes nothing at all.
    """
    if isinstance(value, str):
        return value if len(value) <= _max_len else f"<redacted: {len(value)} chars>"
    if isinstance(value, dict):
        return {k: _redact_debug_value(v, _max_len) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_debug_value(v, _max_len) for v in value]
    return value


def _decode_sent_body(body: Any) -> Any:
    """Decode the exact bytes sent upstream back to JSON for the dump."""
    if not isinstance(body, (bytes, bytearray)):
        return body
    try:
        return json.loads(body)
    except ValueError:
        return f"<{len(body)} bytes, not JSON>"


def redact_dump_headers(
    headers: Mapping[str, str], extra_secret_names: Iterable[str] = ()
) -> dict[str, Any]:
    """Return ``headers`` for a dump with every credential header fully masked.

    ``extra_secret_names`` are header names the operator configured to carry
    a credential (``*_extra_headers``), masked whatever their name looks like.
    """
    extra = {name.lower() for name in extra_secret_names}
    redacted = redact_for_wire_debug(dict(headers))
    return {k: WIRE_DEBUG_REDACTED if k.lower() in extra else v for k, v in redacted.items()}


def write_owner_only_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON readable by the owner only (dir 0700, file 0600).

    The modes passed to ``mkdir`` and ``os.open`` apply only when they create
    the path, so an existing directory or file is tightened as well, before any
    content is written.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    # No O_TRUNC: an existing file is emptied only once it is owner-only.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        if hasattr(os, "fchmod"):  # POSIX; Windows has no owner-only mode bits
            os.fchmod(fd, 0o600)
        os.ftruncate(fd, 0)
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def write_upstream_error_dump(
    config: Any,
    *,
    request_id: str,
    url: str,
    status: int,
    provider: str | None = None,
    model: str | None = None,
    body: Any = None,
    body_source: str | None = None,
    transforms: Any = None,
    stream: bool = False,
) -> Path | None:
    """Write a diagnostic dump of an erroring (>=400) upstream request.

    ``body`` may be the raw bytes that went on the wire; they are decoded here,
    after the off-by-default gate, so a disabled dump costs nothing.
    ``body_source`` records which body was forwarded (for example
    ``"passthrough"`` when the proxy's mutations were dropped), since that is
    often the whole explanation for a 400.

    Returns the path written, or ``None`` when the dump is disabled (the
    default, and always in stateless mode) or could not be written. Never
    raises: a diagnostic must not turn an upstream error into a proxy error.
    """
    mode = _debug_dump_mode(config)
    if mode == "off":
        return None
    try:
        from headroom import paths as _hr_paths

        dump_dir = _hr_paths.debug_400_dir()
        dump_path = dump_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request_id}.json"
        sent_body = _decode_sent_body(body)
        # The dump can hold prompt content: keep it readable by the owner only.
        write_owner_only_json(
            dump_path,
            {
                "request_id": request_id,
                "url": redact_url(url),
                "status": status,
                "provider": provider,
                "model": model,
                "stream": stream,
                "transforms": transforms,
                "body_source": body_source,
                "body": sent_body if mode == "full" else _redact_debug_value(sent_body),
            },
        )
        return dump_path
    except Exception:
        logger.debug("upstream error debug dump skipped", exc_info=True)
        return None
