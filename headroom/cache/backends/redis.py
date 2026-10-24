"""Redis storage backend for CompressionStore: one CCR store for every replica.

A ``<<ccr:hash>>`` marker points at an original kept by whichever replica
compressed it; behind a gateway that spreads requests across replicas, the
retrieval can land anywhere. With ``HEADROOM_CCR_BACKEND=redis`` and
``HEADROOM_REDIS_URL``, every replica reads and writes one store.

Each entry is one key, ``<prefix>:ccr:<hash>``, holding the entry as JSON and
expiring when the entry does (``created_at + ttl``). A Redis error is a miss
(see :mod:`headroom.cache.redis_client`): a retrieval then fails as it would for
an expired entry, and compression never waits on Redis.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, fields
from typing import TYPE_CHECKING, Any

from ..redis_client import Guard, connect, key_prefix

if TYPE_CHECKING:
    from ..compression_store import CompressionEntry


class RedisBackend:
    """CCR entries in Redis, shared by every replica that points at it."""

    #: Redis expires each entry with its TTL: CompressionStore neither caps nor
    #: sweeps this store (a per-replica cap would evict other replicas' entries).
    expires_entries = True

    def __init__(self, url: str, client: Any = None) -> None:
        self._client = client if client is not None else connect(url, setting="HEADROOM_REDIS_URL")
        self._prefix = f"{key_prefix()}:ccr:"
        self._guard = Guard("CCR store")

    def _key(self, hash_key: str) -> str:
        return self._prefix + hash_key

    def get(self, hash_key: str) -> CompressionEntry | None:
        raw = self._guard.call("read", lambda: self._client.get(self._key(hash_key)), None)
        return None if raw is None else _decode(raw)

    def set(self, hash_key: str, entry: CompressionEntry) -> bool:
        """Whether Redis took the entry (an error is logged, not raised)."""
        remaining_ms = int((entry.created_at + entry.ttl - time.time()) * 1000)
        if remaining_ms <= 0:
            return False
        blob = json.dumps(asdict(entry))
        return bool(
            self._guard.call(
                "write", lambda: self._client.set(self._key(hash_key), blob, px=remaining_ms), None
            )
        )

    def delete(self, hash_key: str) -> bool:
        return bool(self._guard.call("delete", lambda: self._client.delete(self._key(hash_key)), 0))

    def exists(self, hash_key: str) -> bool:
        return bool(self._guard.call("read", lambda: self._client.exists(self._key(hash_key)), 0))

    def clear(self) -> None:
        keys = self._scan()
        if keys:
            self._guard.call("delete", lambda: self._client.delete(*keys), 0)

    def count(self) -> int:
        return len(self._scan())

    def keys(self) -> list[str]:
        return [k[len(self._prefix) :] for k in self._scan()]

    def items(self) -> list[tuple[str, CompressionEntry]]:
        out = []
        for hash_key in self.keys():
            entry = self.get(hash_key)
            if entry is not None:
                out.append((hash_key, entry))
        return out

    def purge_expired(self, now: float | None = None) -> int:
        """Redis has already dropped them."""
        return 0

    def get_stats(self) -> dict[str, Any]:
        return {"backend_type": "redis", "entry_count": self.count()}

    def _scan(self) -> list[str]:
        def scan() -> list[str]:
            return [
                k.decode() if isinstance(k, bytes) else k
                for k in self._client.scan_iter(match=_glob_escape(self._prefix) + "*", count=500)
            ]

        return self._guard.call("scan", scan, [])


def _decode(raw: Any) -> CompressionEntry | None:
    """An entry from its JSON, or ``None`` when it is not one (a miss)."""
    from ..compression_store import CompressionEntry

    try:
        data = json.loads(raw)
        known = {f.name for f in fields(CompressionEntry)}
        entry = CompressionEntry(**{k: v for k, v in data.items() if k in known})
    except (ValueError, TypeError, AttributeError):
        return None
    texts_ok = isinstance(entry.original_content, str) and isinstance(entry.compressed_content, str)
    times_ok = isinstance(entry.created_at, (int, float)) and isinstance(entry.ttl, int)
    return entry if texts_ok and times_ok else None


def _glob_escape(text: str) -> str:
    """``text`` matched literally in a SCAN pattern."""
    return "".join("\\" + c if c in "*?[]\\" else c for c in text)
