"""Helpers for Anthropic server-stored Threads (Claude Code via api.anthropic.com)."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any

_PINNED_KEYS = ("system", "tools")


def is_thread_continue(body: Any) -> bool:
    """True for a request body carrying ``thread: {"type": "continue", ...}``.

    A continue turn sends only the new delta: ``messages`` omits everything
    stored upstream, ``tools`` is usually absent and ``system`` may be just the
    billing header. Request-local history heuristics do not apply to it.
    """
    thread = body.get("thread") if isinstance(body, dict) else None
    return isinstance(thread, dict) and thread.get("type") == "continue"


def snapshot_thread_pinned(body: Any) -> dict[str, Any] | None:
    """Deep copy of ``system``/``tools`` (presence and value) on a continue turn, else None."""
    if not is_thread_continue(body):
        return None
    return {k: copy.deepcopy(body[k]) for k in _PINNED_KEYS if k in body}


def restore_thread_pinned(
    body: dict[str, Any],
    snapshot: dict[str, Any] | None,
    keep_tools: list[dict[str, Any]] | None = None,
) -> bool:
    """Put ``system``/``tools`` back exactly as snapshotted; True if anything had changed.

    The API 400s on any change to a stored thread's system/tools/cache_control,
    so this is the last word before the upstream send, whichever pass touched them.
    ``keep_tools`` (the sticky retrieve tool) are appended to the restored tools
    when the snapshot carries a tools list and does not already name them.
    """
    if snapshot is None:
        return False
    changed = False
    for k in _PINNED_KEYS:
        if k in snapshot:
            target = copy.deepcopy(snapshot[k])
            if k == "tools" and keep_tools and isinstance(target, list):
                names = {t.get("name") for t in target if isinstance(t, dict)}
                target += [copy.deepcopy(t) for t in keep_tools if t.get("name") not in names]
            changed = changed or k not in body or body[k] != target
            body[k] = target
        elif k in body:
            changed = True
            del body[k]
    return changed


def thread_scope(headers: Mapping[str, str], tenant_key: str, upstream: str) -> str:
    """Opaque partition key for thread pins and tool names: tenant + credential + upstream.

    ``tenant_key`` is the shared-proxy tenant key (``tenant_key.resolve_tenant_key``);
    it defaults to the literal ``global`` on a proxy with no tenant header, so the
    credential the caller presents and the upstream it targets are mixed in too.
    Only a SHA-256 digest is kept, never the credential itself.
    """
    parts = (
        "thread-scope",
        tenant_key,
        upstream,
        headers.get("x-api-key") or "",
        headers.get("authorization") or "",
    )
    return hashlib.sha256("\x00".join(parts).encode("utf-8", "ignore")).hexdigest()
