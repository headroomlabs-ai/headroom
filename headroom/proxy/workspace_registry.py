"""Server-side lookup of a wrap session's real workspace root.

``headroom wrap`` is the true, blocking parent of the coding-agent CLI it
launches, so its own ``os.getcwd()`` at launch time is authoritative for
that session. It records ``(session_token, cwd)`` into the same per-PID
marker file it already writes for proxy-client liveness tracking
(``paths.proxy_clients_dir(port)``), hardened to mode ``0700`` -- only the
invoking OS user can read or write there.

That's what makes ``session_token`` meaningfully different from the
``x-headroom-cwd`` header (PR #3192's review rejected trusting it directly):
the header's value was fully caller-chosen, so any loopback/SSRF-capable
caller could assert an arbitrary directory. A token only lets a caller
*reference* a root ``wrap`` already established over a channel an
HTTP-only attacker can't write to.
"""

from __future__ import annotations

import hmac
import json

from headroom import fsutil, paths
from headroom.proxy_client_liveness import marker_pid_reused, pid_alive


def resolve_registered_cwd(port: int, session_token: str | None) -> str | None:
    """Return the real cwd registered for `session_token` on `port`, or `None`.

    `None` if the token is missing/blank, no marker matches it, or the
    matching marker's process is no longer live -- never a guess. Read-only:
    doesn't prune stale markers itself, that's wrap.py's job.
    """
    if not session_token:
        return None
    token_bytes = session_token.encode("utf-8", "replace")
    d = paths.proxy_clients_dir(port)
    if not d.exists():
        return None
    for marker in d.glob("*.json"):
        try:
            pid = int(marker.stem)
        except ValueError:
            continue
        try:
            payload = json.loads(fsutil.read_text(marker))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("session_token")
        if not isinstance(candidate, str):
            continue
        # Constant-time compare, same precedent as HEADROOM_PROXY_TOKEN.
        if not hmac.compare_digest(candidate.encode("utf-8", "replace"), token_bytes):
            continue
        if not pid_alive(pid) or marker_pid_reused(marker, pid):
            return None  # matched but stale -- never guess
        cwd = payload.get("cwd")
        return cwd if isinstance(cwd, str) and cwd else None
    return None
