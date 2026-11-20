"""Remote adapters for the embedding server sidecar.

Provides RemoteEmbedder and RemoteVectorIndex that connect to an
EmbeddingServer over a Unix domain socket, implementing the same
Embedder and VectorIndex protocols as the local adapters.

Protocol: length-prefixed JSON frames (4-byte uint32 LE + UTF-8 JSON body).
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from headroom.memory.ports import VectorFilter, VectorSearchResult

logger = logging.getLogger(__name__)

# How many times to retry on a broken connection before giving up
_MAX_RETRIES = 3
# Backoff between retries (seconds)
_RETRY_BACKOFF = 0.1
# Maximum payload size: 10 MB
_MAX_PAYLOAD_BYTES = 10 * 1024 * 1024


class EmbeddingServerUnavailable(RuntimeError):
    """Raised when the embedding server cannot be reached after retries."""


class _EmbeddingServerConnection:
    """Shared helper: manages a persistent asyncio Unix socket connection.

    Thread-safe via asyncio.Lock. Handles reconnect logic transparently.
    """

    def __init__(
        self,
        socket_path: str,
        connect_timeout: float = 5.0,
        request_timeout: float = 10.0,
    ) -> None:
        self.socket_path = socket_path
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create lock bound to the current event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _connect(self) -> None:
        """Open (or re-open) the Unix socket connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None

        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(self.socket_path),
            timeout=self.connect_timeout,
        )
        self._reader = reader
        self._writer = writer

    async def _ensure_connected(self) -> None:
        """Connect if not already connected."""
        if self._writer is None or self._writer.is_closing():
            await self._connect()

    @staticmethod
    def _read_frame_sync(data: bytes) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(data.decode("utf-8"))
        return result

    async def _read_frame(self) -> bytes:
        assert self._reader is not None
        header = await self._reader.readexactly(4)
        length = struct.unpack("<I", header)[0]
        if length > _MAX_PAYLOAD_BYTES:
            raise ValueError(f"Frame too large: {length} bytes")
        return await self._reader.readexactly(length)

    async def _write_frame(self, data: bytes) -> None:
        assert self._writer is not None
        header = struct.pack("<I", len(data))
        self._writer.write(header + data)
        await self._writer.drain()

    async def send_request(self, op: str, **kwargs: Any) -> dict[str, Any]:
        """Send a request and return the response. Retries on connection errors."""
        request_id = str(uuid.uuid4())
        payload = {"op": op, "id": request_id, **kwargs}
        raw = json.dumps(payload).encode("utf-8")

        lock = self._get_lock()
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                async with lock:
                    await self._ensure_connected()
                    await asyncio.wait_for(
                        self._write_frame(raw),
                        timeout=self.request_timeout,
                    )
                    response_bytes = await asyncio.wait_for(
                        self._read_frame(),
                        timeout=self.request_timeout,
                    )
                response: dict[str, Any] = json.loads(response_bytes.decode("utf-8"))
                if "error" in response:
                    raise RuntimeError(f"Server error: {response['error']}")
                return response

            except (ConnectionError, BrokenPipeError, asyncio.IncompleteReadError) as exc:
                last_exc = exc
                # Force reconnect on next attempt
                async with lock:
                    self._reader = None
                    self._writer = None
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_BACKOFF * (2**attempt))
            except (FileNotFoundError, OSError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_BACKOFF * (2**attempt))

        raise EmbeddingServerUnavailable(
            f"Cannot connect to embedding server at {self.socket_path} "
            f"after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def close(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None


# ---------------------------------------------------------------------------
# RemoteEmbedder
# ---------------------------------------------------------------------------


class RemoteEmbedder:
    """Embedder protocol implementation backed by the embedding server sidecar.

    Implements the same Embedder protocol as OnnxLocalEmbedder, but
    delegates all computation to the shared server process.
    """

    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        socket_path: str,
        connect_timeout: float = 5.0,
        request_timeout: float = 10.0,
    ) -> None:
        self._conn = _EmbeddingServerConnection(
            socket_path=socket_path,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )

    async def embed(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text."""
        response = await self._conn.send_request("embed", text=text)
        return np.array(response["embedding"], dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        response = await self._conn.send_request("embed_batch", texts=texts)
        return [np.array(e, dtype=np.float32) for e in response["embeddings"]]

    async def ping(self) -> bool:
        """Health check: returns True if the server is reachable."""
        try:
            response = await self._conn.send_request("ping")
            return response.get("status") == "ok"
        except (EmbeddingServerUnavailable, Exception):
            return False

    async def close(self) -> None:
        """Close the connection."""
        await self._conn.close()

    @property
    def dimension(self) -> int:
        return self.DEFAULT_DIMENSION

    @property
    def model_name(self) -> str:
        return "all-MiniLM-L6-v2-onnx-remote"

    @property
    def max_tokens(self) -> int:
        return 256


# ---------------------------------------------------------------------------
# RemoteVectorIndex
# ---------------------------------------------------------------------------


class RemoteVectorIndex:
    """VectorIndex protocol implementation backed by the embedding server sidecar.

    Delegates all index operations to the shared server process so all
    workers share a single HNSW index (no per-worker fragmentation).
    """

    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        socket_path: str,
        connect_timeout: float = 5.0,
        request_timeout: float = 10.0,
    ) -> None:
        self._conn = _EmbeddingServerConnection(
            socket_path=socket_path,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )

    async def index(self, memory: Any) -> None:
        """Index a memory's embedding."""
        from headroom.memory.models import Memory as _Memory

        assert isinstance(memory, _Memory), "memory must be a Memory object"

        embedding = memory.embedding
        memory_data: dict[str, Any] = {
            "id": memory.id,
            "content": memory.content,
            "user_id": memory.user_id,
            "session_id": memory.session_id,
            "agent_id": memory.agent_id,
            "importance": memory.importance,
            "entity_refs": memory.entity_refs,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
        }
        kwargs: dict[str, Any] = {"memory": memory_data}
        if embedding is not None:
            emb = np.asarray(embedding, dtype=np.float32)
            kwargs["embedding"] = emb.tolist()

        await self._conn.send_request("store", **kwargs)

    async def index_batch(self, memories: list[Any]) -> int:
        """Index multiple memories."""
        count = 0
        for memory in memories:
            if memory.embedding is not None:
                await self.index(memory)
                count += 1
        return count

    async def remove(self, memory_id: str) -> bool:
        """Remove a memory from the vector index."""
        response = await self._conn.send_request("delete", memory_id=memory_id)
        return response.get("status") == "deleted"

    async def remove_batch(self, memory_ids: list[str]) -> int:
        """Remove multiple memories from the vector index."""
        count = 0
        for mid in memory_ids:
            if await self.remove(mid):
                count += 1
        return count

    async def search(self, filter: VectorFilter) -> list[VectorSearchResult]:
        """Search for similar memories using vector similarity."""
        from headroom.memory.models import Memory as _Memory
        from headroom.memory.ports import VectorSearchResult

        kwargs: dict[str, Any] = {
            "top_k": filter.top_k,
            "min_similarity": filter.min_similarity,
        }
        if filter.user_id:
            kwargs["user_id"] = filter.user_id
        if filter.session_id:
            kwargs["session_id"] = filter.session_id
        if filter.agent_id:
            kwargs["agent_id"] = filter.agent_id

        if filter.query_vector is not None:
            kwargs["query_embedding"] = np.asarray(filter.query_vector, dtype=np.float32).tolist()
        elif filter.query_text:
            kwargs["query_text"] = filter.query_text
        else:
            raise ValueError("Either query_vector or query_text must be provided")

        response = await self._conn.send_request("search", **kwargs)
        results: list[VectorSearchResult] = []
        for item in response.get("results", []):
            memory = _Memory(
                id=item["memory_id"],
                content=item.get("content", ""),
                user_id=item.get("user_id", ""),
                created_at=datetime.now(timezone.utc),
                valid_from=datetime.now(timezone.utc),
            )
            results.append(
                VectorSearchResult(
                    memory=memory,
                    similarity=float(item["similarity"]),
                    rank=int(item["rank"]),
                )
            )
        return results

    async def update_embedding(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Report unsupported rather than claiming an update that never happened."""
        return False

    async def stats(self) -> dict[str, Any]:
        """Get index statistics from the server."""
        return await self._conn.send_request("stats")

    async def close(self) -> None:
        """Close the connection."""
        await self._conn.close()

    @property
    def dimension(self) -> int:
        return self.DEFAULT_DIMENSION

    @property
    def size(self) -> int:
        # Can't easily get this synchronously; return 0 as sentinel
        return 0
