"""Reconcile Codex thread provider tags across the Headroom proxy boundary.

Codex stamps every thread with the ``model_provider`` it ran under and filters
its history/projects menu by the active provider set.  When Headroom rewrites
Codex's config to route through the custom ``headroom`` provider (see
:mod:`headroom.providers.codex.install`), threads created through Headroom are
tagged ``headroom`` while native threads keep ``openai`` -- so the two sets never
appear in the same menu, and connecting Headroom appears to "lose" history.

To keep the menu whole we retag threads to match whichever provider is active:
``openai -> headroom`` when Headroom is enabled, ``headroom -> openai`` when it is
reverted.  Only rows whose ``model_provider`` equals the source value are
touched, so third-party providers are left alone.

Every operation is best-effort: a missing store, a missing ``threads`` table, or
a store momentarily locked by a running Codex is logged and skipped -- never
raised -- so install/uninstall never fail on account of the history menu.  The
store is WAL-mode, so the update succeeds even while Codex is running; the short
busy timeout only covers a transient checkpoint lock.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

HEADROOM_PROVIDER = "headroom"
NATIVE_PROVIDER = "openai"

# Seconds to wait on a busy store before giving up (a running Codex only holds an
# exclusive lock briefly, during a WAL checkpoint).
_BUSY_TIMEOUT_S = 0.75


def _codex_state_db_paths(codex_home: Path) -> list[Path]:
    """Both known Codex state stores under ``codex_home`` (the ``.codex`` dir).

    The v148 desktop GUI reads ``<codex_home>/sqlite/state_5.sqlite``; the
    CLI/TUI uses ``<codex_home>/state_5.sqlite``.  Retag whichever exist.
    """
    return [codex_home / "sqlite" / "state_5.sqlite", codex_home / "state_5.sqlite"]


def _retag_one(path: Path, *, frm: str, to: str) -> int:
    """Retag a single store and return the number of rows moved.

    No-ops (returns 0) on a store whose schema lacks the ``threads`` table.
    """
    conn = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_S)
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'threads'"
        ).fetchone()
        if has_table is None:
            return 0
        cur = conn.execute(
            "UPDATE threads SET model_provider = ? WHERE model_provider = ?",
            (to, frm),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def retag_thread_providers(codex_home: Path, *, frm: str, to: str) -> None:
    """Best-effort retag of Codex thread provider tags across all known stores.

    ``codex_home`` is the Codex configuration directory (the parent of
    ``config.toml``); resolving from it keeps callers and tests pointed at one
    location rather than re-deriving ``~/.codex`` independently.
    """
    if frm == to:
        return
    for path in _codex_state_db_paths(codex_home):
        if not path.exists():
            continue
        try:
            moved = _retag_one(path, frm=frm, to=to)
        except sqlite3.Error as exc:
            logger.warning("codex thread retag %s->%s skipped for %s: %s", frm, to, path, exc)
            continue
        if moved:
            logger.info("codex thread retag %s->%s: %d thread(s) in %s", frm, to, moved, path)


def retag_to_headroom(codex_home: Path) -> None:
    """Pull existing native threads into the headroom-provider menu (on enable)."""
    retag_thread_providers(codex_home, frm=NATIVE_PROVIDER, to=HEADROOM_PROVIDER)


def retag_to_native(codex_home: Path) -> None:
    """Hand threads back to the native-provider menu (on revert)."""
    retag_thread_providers(codex_home, frm=HEADROOM_PROVIDER, to=NATIVE_PROVIDER)
