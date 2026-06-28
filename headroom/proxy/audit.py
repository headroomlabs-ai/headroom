"""Lightweight audit log for administrative / state-mutating proxy actions.

Emits one structured JSON line per sensitive action (``/admin/*`` runtime
changes, cache clears, stats resets) to the dedicated ``headroom.audit``
logger. Operators capture this stream the same way they capture the proxy
log; routing/retention is theirs to configure.

Logger-only by design — it writes **no** dedicated file, so it is safe under
``HEADROOM_STATELESS`` (no filesystem writes) and respects whatever log sink
the deployment already uses.
"""

from __future__ import annotations

import json
import logging
from typing import Any

audit_logger = logging.getLogger("headroom.audit")

# Paths whose requests mutate runtime state or expose stored content and so
# warrant an audit trail. Matched by exact value or, for ``/admin/``, prefix.
_ADMIN_PREFIX = "/admin/"
_SENSITIVE_EXACT = frozenset({"/cache/clear", "/stats/reset"})


def is_auditable_path(path: str) -> bool:
    """Return True when requests to ``path`` should be audited."""
    return path.startswith(_ADMIN_PREFIX) or path in _SENSITIVE_EXACT


def _client_ip(request: Any) -> str | None:
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client is not None else None


def record_admin_action(
    *,
    request: Any,
    action: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> None:
    """Emit a structured audit event. Never raises (audit must not break a
    request); logs at WARNING on its own failure."""
    try:
        event: dict[str, Any] = {
            "event": "headroom_admin_audit",
            "action": action,
            "method": getattr(request, "method", None),
            "path": getattr(getattr(request, "url", None), "path", None),
            "source_ip": _client_ip(request),
            "status_code": status_code,
        }
        if details:
            event["details"] = details
        audit_logger.info(json.dumps(event, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001 — auditing must never break the request
        audit_logger.warning("audit event emission failed", exc_info=True)
