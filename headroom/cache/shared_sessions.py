"""Session-aware ``/v1/compress`` state shared by several Headroom replicas.

A compress session's state is its prefix tracker: the transcript it was last
given and the one it returned, which every later turn must replay byte for byte
so the provider's prompt cache holds. In memory, only the replica that served
the last turn has it, so a gateway running several replicas must pin each
session to one (``session_affinity``).

With ``HEADROOM_COMPRESS_SESSION_STORE=redis://...`` every replica also keeps
each session in Redis, and any replica can serve its next turn. Per session:

* ``<prefix>:session:{<digest>}:msgs``, a hash of message digest to message
  JSON. The original and returned transcripts mostly share messages, and each
  turn mostly repeats the last one's, so each message is stored once and a turn
  writes only the messages Redis does not have yet: what the turn added, and
  any message whose bytes changed (a moved cache marker, an edit, a compaction);
* ``<prefix>:session:{<digest>}:meta``, the tracker's counters and its
  transcripts as lists of digests (``m``), and a write token (``t``).

Before a turn a replica reads the token (one small ``HGET``). When another
replica has written since, it reads the metadata and only the messages it does
not already hold, and the turn seeds its compression cache from that tracker as
after a restart (:func:`~headroom.cache.prefix_tracker.seed_restored_compression_cache`).
After the turn it writes the new messages, then the metadata if the token is
still the one it read and every message it references is stored, checked in
one script (compare-and-set): a random token, so a lost key or a failover
cannot leave a replica believing stale state is current. A missing key
resets the replica, whose next write stores the session whole again. Fetched
messages are checked against their digests. Both keys expire with the tracker.

A Redis error never fails a turn (:mod:`headroom.cache.redis_client`): the
replica serves from its own memory, as without the store. CCR originals behind
the markers in these transcripts live in the shared CCR store, which this
setting selects too unless ``HEADROOM_CCR_BACKEND`` names another. Requires the
``redis`` extra.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from .redis_client import Guard, connect, key_prefix

if TYPE_CHECKING:
    from .prefix_tracker import PrefixCacheTracker, SessionTrackerStore

logger = logging.getLogger(__name__)

STORE_ENV = "HEADROOM_COMPRESS_SESSION_STORE"
#: Messages per HSET/HMGET, so no single command carries a whole transcript.
_CHUNK = 256

# Install ARGV[2] as the token and ARGV[3] as the metadata if the token is still
# ARGV[1] ("" when the key must not exist) and the messages hash (KEYS[2]) holds
# every message the metadata references (ARGV[5:], including those this write
# just sent: Redis can evict the hash between those writes and this script).
# 1 on success, 0 when another replica wrote first, 2 when the messages hash
# lost one.
_CAS = """
local t = redis.call('HGET', KEYS[1], 't') or ''
if t ~= ARGV[1] then return 0 end
for i = 5, #ARGV do
  if redis.call('HEXISTS', KEYS[2], ARGV[i]) == 0 then return 2 end
end
redis.call('HSET', KEYS[1], 't', ARGV[2], 'm', ARGV[3])
redis.call('PEXPIRE', KEYS[1], ARGV[4])
return 1
"""

_FAILED = object()


class SharedSessions:
    """Compress-session trackers in Redis, beside the in-memory store."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._cas = client.register_script(_CAS)
        self._guard = Guard("compress sessions")
        self._prefix = f"{key_prefix()}:session:"

    def fetch(
        self, store: SessionTrackerStore, key: str, local: PrefixCacheTracker | None
    ) -> PrefixCacheTracker | None:
        """The session's tracker when another replica wrote a newer one than
        ``local``, decoded but not installed; else ``None``."""
        meta_key, msgs_key = self._keys(key)
        token = self._guard.call("read", lambda: self._client.hget(meta_key, "t"), _FAILED)
        if token is _FAILED:
            return None
        if token is None:
            if local is not None:  # lost or expired: the next write stores it whole
                local.shared_token, local.shared_digests = "", frozenset()
            return None
        try:
            if local is not None and _text(token) == local.shared_token:
                return None
        except UnicodeDecodeError:
            return self._drop(local, meta_key, msgs_key)
        row = self._guard.call("read", lambda: self._client.hmget(meta_key, "t", "m"), None)
        if row is None:
            return None
        if row[0] is None:  # expired between the two reads
            return None
        try:
            token = _text(row[0])
        except UnicodeDecodeError:
            return self._drop(local, meta_key, msgs_key)
        try:
            meta = json.loads(_text(row[1]))
            digests = meta.pop("table")
            if not isinstance(digests, list) or not all(isinstance(d, str) for d in digests):
                raise ValueError("table is not a list of digests")
        except (TypeError, ValueError, KeyError, AttributeError, IndexError):
            return self._unusable(local, token, "metadata", meta_key)
        held = self._held(store, key, local)
        missing = [d for d in dict.fromkeys(digests) if d not in held]
        for start in range(0, len(missing), _CHUNK):
            part = missing[start : start + _CHUNK]
            values = self._guard.call(
                "read", functools.partial(self._client.hmget, msgs_key, part), None
            )
            if values is None:
                return None
            for digest, value in zip(part, values, strict=True):
                try:
                    text = None if value is None else _text(value)
                except UnicodeDecodeError:
                    text = None
                if text is None or _digest(text) != digest:
                    return self._unusable(local, token, "messages", msgs_key)
                held[digest] = text
        try:
            tracker = store.decode_shared(meta, [json.loads(held[d]) for d in digests])
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
            return self._unusable(local, token, "metadata", meta_key)
        if tracker is None:
            logger.info("compress sessions: shared state for %s has expired", meta_key)
            return None
        tracker.shared_token, tracker.shared_digests = token, frozenset(digests)
        return tracker

    @staticmethod
    def _unusable(
        local: PrefixCacheTracker | None, token: str, what: str, name: str
    ) -> PrefixCacheTracker | None:
        """Shared state this replica cannot use (evicted, corrupt or altered):
        keep its own, and claim the current token with no messages known to be
        stored, so its next write replaces the session whole."""
        logger.warning("compress sessions: unusable %s in %s; this replica rewrites it", what, name)
        if local is not None:
            local.shared_token, local.shared_digests = token, frozenset()
        return None

    def _drop(
        self, local: PrefixCacheTracker | None, meta_key: str, msgs_key: str
    ) -> PrefixCacheTracker | None:
        """A token that is not text cannot be claimed: delete the session, so
        the next write stores it whole."""
        logger.warning("compress sessions: corrupt token in %s; deleting the session", meta_key)
        self._guard.call("write", lambda: self._client.delete(meta_key, msgs_key), 0)
        if local is not None:
            local.shared_token, local.shared_digests = "", frozenset()
        return None

    def publish(self, store: SessionTrackerStore, key: str, tracker: PrefixCacheTracker) -> None:
        """Write the messages Redis lacks, then the metadata unless another
        replica wrote since ``tracker`` was read."""
        try:
            meta, table = store.encode_shared(key, tracker)
        except (TypeError, ValueError) as exc:  # messages that are not JSON
            logger.debug("compress sessions: tracker not shareable (%s)", exc)
            return
        digests = [_digest(m) for m in table]
        meta["table"] = digests
        meta_json = json.dumps(meta)
        ttl_ms = max(1000, int(tracker.retention_seconds() * 1000))
        meta_key, msgs_key = self._keys(key)
        referenced = sorted(set(digests))

        def write(known: frozenset[str]) -> tuple[int, str]:
            new = [(d, m) for d, m in zip(digests, table, strict=True) if d not in known]
            token = uuid.uuid4().hex
            pipe = self._client.pipeline(transaction=False)
            for start in range(0, len(new), _CHUNK):
                pipe.hset(msgs_key, mapping=dict(new[start : start + _CHUNK]))
            pipe.pexpire(msgs_key, ttl_ms)
            pipe.execute()
            args = [tracker.shared_token, token, meta_json, ttl_ms, *referenced]
            return int(self._cas(keys=[meta_key, msgs_key], args=args)), token

        result = self._guard.call("write", lambda: write(tracker.shared_digests), None)
        if result is not None and result[0] == 2:
            # The messages hash lost entries (evicted on its own): store it whole.
            result = self._guard.call("write", lambda: write(frozenset()), None)
        if result is None:
            return
        written, token = result
        if written == 1:
            tracker.shared_token, tracker.shared_digests = token, frozenset(digests)
            self._prune(msgs_key, set(digests))
        elif written == 2:
            # Evicted again during the whole rewrite: publish nothing, so no
            # replica takes a session it cannot replay; the next turn retries.
            logger.warning("compress sessions: messages of %s evicted while writing", meta_key)
        elif written == 0:
            # Two replicas ran turns of one session at once, which the caller's
            # contract excludes; the next turn takes the state that won.
            logger.warning("compress sessions: another replica wrote %s first", meta_key)

    def _prune(self, msgs_key: str, referenced: set[str]) -> None:
        """Drop messages no transcript references (moved markers, edits,
        compactions) once they outnumber the referenced ones."""
        size = self._guard.call("write", lambda: int(self._client.hlen(msgs_key)), 0)
        if size <= 2 * len(referenced):
            return

        def prune() -> None:
            stale = [k for k in map(_text, self._client.hkeys(msgs_key)) if k not in referenced]
            for start in range(0, len(stale), _CHUNK):
                self._client.hdel(msgs_key, *stale[start : start + _CHUNK])

        self._guard.call("write", prune, None)

    def _keys(self, key: str) -> tuple[str, str]:
        # One hash tag: the compare-and-set script names both keys, which Redis
        # Cluster allows only within one slot.
        base = f"{self._prefix}{{{hashlib.blake2b(key.encode(), digest_size=16).hexdigest()}}}"
        return f"{base}:meta", f"{base}:msgs"

    @staticmethod
    def _held(
        store: SessionTrackerStore, key: str, local: PrefixCacheTracker | None
    ) -> dict[str, str]:
        """The messages this replica already holds for the session, by digest."""
        if local is None:
            return {}
        try:
            _, table = store.encode_shared(key, local)
        except (TypeError, ValueError):
            return {}
        return {_digest(m): m for m in table}


def _digest(message_json: str) -> str:
    return hashlib.blake2b(message_json.encode(), digest_size=16).hexdigest()


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def open_shared_sessions() -> SharedSessions | None:
    """The shared store ``HEADROOM_COMPRESS_SESSION_STORE`` names, or ``None``.

    Raises :class:`~headroom.exceptions.ConfigurationError` when it is set but
    unusable, so a replica meant to share sessions never starts without it."""
    url = os.environ.get(STORE_ENV, "").strip()
    if not url:
        return None
    client = connect(url, setting=STORE_ENV)
    ccr = (os.environ.get("HEADROOM_CCR_BACKEND") or "").strip().lower()
    if ccr not in ("", "redis"):
        logger.warning(
            "%s is set but HEADROOM_CCR_BACKEND=%s: CCR markers in a session's "
            "transcript resolve only on the replica that made them",
            STORE_ENV,
            ccr,
        )
    return SharedSessions(client)
