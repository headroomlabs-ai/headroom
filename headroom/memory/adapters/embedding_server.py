"""Embedding server sidecar for the Headroom proxy.

A standalone asyncio server process that:
- Loads OnnxLocalEmbedder once on startup (saves ~602 MB across 8 workers)
- Owns a single HNSWVectorIndex (shared truth for all workers)
- Listens on a Unix domain socket
- Handles embed/embed_batch/search/store/delete/stats/ping requests
- Uses a ThreadPoolExecutor for CPU-bound ONNX inference
- Supports adaptive micro-batching (queues embed requests for up to 5ms)

Protocol: length-prefixed JSON frames
  - 4-byte uint32 LE length
  - UTF-8 JSON body

Entry points:
  serve(socket_path, ...) - async server coroutine
  run_server(socket_path, ...) - synchronous wrapper
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import signal
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Maximum payload size: 10 MB
MAX_PAYLOAD_BYTES = 10 * 1024 * 1024

# Adaptive micro-batching window: queue embed requests for up to this many ms
MICRO_BATCH_WINDOW_MS = 5.0

# How many pending embed requests to accumulate before forcing a flush
MICRO_BATCH_MAX_SIZE = 64


@dataclass
class _PendingEmbed:
    """A single embed request waiting in the micro-batch queue."""

    request_id: str
    text: str
    future: asyncio.Future[np.ndarray] = field(default_factory=asyncio.Future)


class EmbeddingServer:
    """Core server logic: loads embedder + HNSW index, handles requests."""

    def __init__(
        self,
        socket_path: str,
        max_elements: int = 10_000,
        embed_threads: int = 4,
    ) -> None:
        self.socket_path = socket_path
        self.max_elements = max_elements
        self.embed_threads = embed_threads

        self._embedder: Any = None
        self._index: Any = None
        self._executor: ThreadPoolExecutor | None = None
        self._lock = Lock()

        # Stats counters
        self._total_requests: int = 0
        self._total_embed_calls: int = 0
        self._total_latency_ms: float = 0.0
        self._start_time: float = time.monotonic()
        # Per-op breakdown
        self._op_counts: dict[str, int] = collections.defaultdict(int)
        # Rolling latency window for percentile calculation (last 1000 requests)
        self._latency_window: collections.deque[float] = collections.deque(maxlen=1000)
        # Batch efficiency tracking
        self._total_batches: int = 0
        self._total_batched_items: int = 0

        # Micro-batching
        self._batch_queue: asyncio.Queue[_PendingEmbed] = asyncio.Queue()
        self._batch_task: asyncio.Task[None] | None = None

        # Shutdown
        self._shutdown_event: asyncio.Event = asyncio.Event()

    def _load_resources(self) -> None:
        """Load embedder and HNSW index (runs in thread pool on startup)."""
        from headroom.memory.adapters.embedders import OnnxLocalEmbedder
        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        # Check hnswlib before loading the 86 MB ONNX model so a missing
        # dependency fails fast. Exit code 3 signals a permanent (non-retryable)
        # error to the watchdog.
        try:
            import hnswlib as _hnswlib  # noqa: F401
        except ImportError:
            logger.error(
                "event=embedding_server_missing_dep dep=hnswlib "
                "fix='uv sync --extra memory'  OR  pip install hnswlib"
            )
            import sys

            sys.exit(3)

        logger.info("event=embedding_server_loading")
        self._embedder = OnnxLocalEmbedder()
        # Force-load the model now so the first request is fast
        # Use a dummy embed to trigger ONNX model download + JIT
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._embedder.embed("warmup"))
        finally:
            loop.close()

        self._index = HNSWVectorIndex(
            dimension=384,
            max_elements=self.max_elements,
            max_entries=self.max_elements,
        )
        logger.info(
            "event=embedding_server_loaded model=all-MiniLM-L6-v2-onnx max_elements=%d",
            self.max_elements,
        )

    async def start(self) -> None:
        """Initialize resources."""
        self._executor = ThreadPoolExecutor(max_workers=self.embed_threads)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_resources)
        # Start micro-batch processor
        self._batch_task = asyncio.create_task(self._batch_processor())

    async def stop(self) -> None:
        """Shutdown gracefully."""
        self._shutdown_event.set()
        if self._batch_task is not None:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        logger.info("event=embedding_server_stopped")

    # -------------------------------------------------------------------------
    # Micro-batch processor
    # -------------------------------------------------------------------------

    async def _batch_processor(self) -> None:
        """Drain the embed queue in micro-batches."""
        while not self._shutdown_event.is_set():
            # Wait for at least one item
            try:
                first = await asyncio.wait_for(
                    self._batch_queue.get(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            batch: list[_PendingEmbed] = [first]

            # Collect more items within the micro-batch window
            deadline = time.monotonic() + MICRO_BATCH_WINDOW_MS / 1000.0
            while len(batch) < MICRO_BATCH_MAX_SIZE:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=remaining,
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    break

            # Run batch through embedder in thread pool
            texts = [item.text for item in batch]
            try:
                loop = asyncio.get_running_loop()
                embeddings: list[np.ndarray] = await loop.run_in_executor(
                    self._executor,
                    self._sync_embed_batch,
                    texts,
                )
                for item, emb in zip(batch, embeddings):
                    if not item.future.done():
                        item.future.set_result(emb)
                self._total_embed_calls += len(batch)
                self._total_batches += 1
                self._total_batched_items += len(batch)
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

    def _sync_embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Synchronous batch embed (runs in ThreadPoolExecutor)."""
        assert self._embedder is not None
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._embedder.embed_batch(texts))
        finally:
            loop.close()

    async def _embed_via_batch_queue(self, text: str, request_id: str) -> np.ndarray:
        """Queue a single embed request and await the batch result."""
        pending = _PendingEmbed(request_id=request_id, text=text)
        await self._batch_queue.put(pending)
        result = await pending.future
        return result

    # -------------------------------------------------------------------------
    # Request handlers
    # -------------------------------------------------------------------------

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a request to the appropriate handler."""
        op = request.get("op", "")
        req_id = request.get("id", "")
        t0 = time.monotonic()

        try:
            if op == "ping":
                result = {"status": "ok", "id": req_id}
            elif op == "embed":
                result = await self._handle_embed(request)
            elif op == "embed_batch":
                result = await self._handle_embed_batch(request)
            elif op == "search":
                result = await self._handle_search(request)
            elif op == "store":
                result = await self._handle_store(request)
            elif op == "delete":
                result = await self._handle_delete(request)
            elif op == "stats":
                result = self._handle_stats()
            else:
                result = {"error": f"unknown op: {op}"}

            result["id"] = req_id
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._total_requests += 1
            self._total_latency_ms += elapsed_ms
            self._op_counts[op] += 1
            self._latency_window.append(elapsed_ms)
            logger.debug(
                "event=embedding_server_request op=%s id=%s latency_ms=%.2f",
                op,
                req_id,
                elapsed_ms,
            )
            return result

        except Exception as exc:
            logger.exception("event=embedding_server_error op=%s id=%s: %s", op, req_id, exc)
            return {"id": req_id, "error": str(exc)}

    async def _handle_embed(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request.get("text", "")
        req_id = request.get("id", "")
        embedding = await self._embed_via_batch_queue(text, req_id)
        return {"embedding": embedding.tolist()}

    async def _handle_embed_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        texts: list[str] = request.get("texts", [])
        if not texts:
            return {"embeddings": []}
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            self._executor,
            self._sync_embed_batch,
            texts,
        )
        self._total_embed_calls += len(texts)
        return {"embeddings": [e.tolist() for e in embeddings]}

    async def _handle_search(self, request: dict[str, Any]) -> dict[str, Any]:
        from headroom.memory.ports import VectorFilter

        query_embedding_data = request.get("query_embedding")
        query_text = request.get("query_text")

        if query_embedding_data is not None:
            query_vector = np.array(query_embedding_data, dtype=np.float32)
        elif query_text:
            req_id = request.get("id", "")
            query_vector = await self._embed_via_batch_queue(query_text, req_id + "_q")
        else:
            return {"error": "query_embedding or query_text required"}

        top_k = request.get("top_k", 10)
        min_similarity = request.get("min_similarity", 0.0)
        user_id = request.get("user_id")
        session_id = request.get("session_id")
        agent_id = request.get("agent_id")

        vf = VectorFilter(
            query_vector=query_vector,
            top_k=top_k,
            min_similarity=min_similarity,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
        )
        assert self._index is not None
        results = await self._index.search(vf)
        return {
            "results": [
                {
                    "memory_id": r.memory.id,
                    "similarity": r.similarity,
                    "rank": r.rank,
                    "content": r.memory.content,
                    "user_id": r.memory.user_id,
                }
                for r in results
            ]
        }

    async def _handle_store(self, request: dict[str, Any]) -> dict[str, Any]:
        from headroom.memory.models import Memory

        memory_data = request.get("memory", {})
        embedding_data = request.get("embedding")

        if embedding_data is not None:
            embedding = np.array(embedding_data, dtype=np.float32)
        else:
            # Embed the content
            content = memory_data.get("content", "")
            req_id = request.get("id", "")
            embedding = await self._embed_via_batch_queue(content, req_id + "_s")

        # Reconstruct a Memory object from the dict
        # Only populate the fields we actually need for indexing
        created_at_str = memory_data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
        )

        memory = Memory(
            id=memory_data["id"],
            content=memory_data.get("content", ""),
            user_id=memory_data.get("user_id", ""),
            session_id=memory_data.get("session_id"),
            agent_id=memory_data.get("agent_id"),
            embedding=embedding,
            importance=memory_data.get("importance", 0.5),
            entity_refs=memory_data.get("entity_refs", []),
            created_at=created_at,
        )

        assert self._index is not None
        await self._index.index(memory)
        return {"status": "stored", "memory_id": memory.id}

    async def _handle_delete(self, request: dict[str, Any]) -> dict[str, Any]:
        memory_id = request.get("memory_id", "")
        assert self._index is not None
        removed = await self._index.remove(memory_id)
        return {"status": "deleted" if removed else "not_found", "memory_id": memory_id}

    def _handle_stats(self) -> dict[str, Any]:
        uptime_s = time.monotonic() - self._start_time
        avg_latency_ms = (
            self._total_latency_ms / self._total_requests if self._total_requests > 0 else 0.0
        )
        index_size = self._index.size if self._index is not None else 0
        avg_batch_size = (
            self._total_batched_items / self._total_batches if self._total_batches > 0 else 0.0
        )

        # Compute p50/p95/p99 from rolling window (sorted ascending)
        latency_sorted = sorted(self._latency_window)
        n = len(latency_sorted)

        def _pct(p: float) -> float:
            if n == 0:
                return 0.0
            idx = int(p / 100.0 * n)
            return round(latency_sorted[min(idx, n - 1)], 3)

        return {
            "index_size": index_size,
            "total_requests": self._total_requests,
            "total_embed_calls": self._total_embed_calls,
            "latency_ms": {
                "avg": round(avg_latency_ms, 3),
                "p50": _pct(50),
                "p95": _pct(95),
                "p99": _pct(99),
                "window_size": n,
            },
            "batch": {
                "total_batches": self._total_batches,
                "avg_batch_size": round(avg_batch_size, 2),
            },
            "ops": dict(self._op_counts),
            "uptime_seconds": round(uptime_s, 1),
            "embed_threads": self.embed_threads,
        }


# ---------------------------------------------------------------------------
# Socket protocol helpers
# ---------------------------------------------------------------------------


async def _read_frame(reader: asyncio.StreamReader) -> bytes | None:
    """Read a length-prefixed frame. Returns None on EOF."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    length = struct.unpack("<I", header)[0]
    if length > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Frame too large: {length} bytes (max {MAX_PAYLOAD_BYTES})")
    data = await reader.readexactly(length)
    return data


async def _write_frame(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write a length-prefixed frame."""
    header = struct.pack("<I", len(data))
    writer.write(header + data)
    await writer.drain()


# ---------------------------------------------------------------------------
# Main server coroutine
# ---------------------------------------------------------------------------


async def serve(
    socket_path: str,
    max_elements: int = 10_000,
    embed_threads: int = 4,
) -> None:
    """Run the embedding server.

    Args:
        socket_path: Path for the Unix domain socket.
        max_elements: Initial HNSW capacity (auto-resizes).
        embed_threads: Thread pool size for ONNX inference.
    """
    server_obj = EmbeddingServer(
        socket_path=socket_path,
        max_elements=max_elements,
        embed_threads=embed_threads,
    )

    await server_obj.start()

    # Remove stale socket file if it exists
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    # Ensure the socket directory exists
    socket_dir = os.path.dirname(socket_path)
    if socket_dir:
        os.makedirs(socket_dir, exist_ok=True)

    # Set up graceful shutdown on SIGTERM
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_sigterm() -> None:
        logger.info("event=embedding_server_sigterm_received")
        stop_event.set()

    loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
    loop.add_signal_handler(signal.SIGINT, _handle_sigterm)

    async def _client_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<unix>")
        logger.debug("event=embedding_server_client_connected peer=%s", peer)
        try:
            while True:
                raw = await _read_frame(reader)
                if raw is None:
                    break
                try:
                    request = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    response = {"error": f"invalid JSON: {exc}"}
                    await _write_frame(writer, json.dumps(response).encode("utf-8"))
                    continue

                response = await server_obj.handle_request(request)
                await _write_frame(writer, json.dumps(response).encode("utf-8"))
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        except Exception as exc:
            logger.warning("event=embedding_server_client_error: %s", exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    unix_server = await asyncio.start_unix_server(
        _client_handler,
        path=socket_path,
    )

    # Set restrictive permissions on socket (0600 = owner rw only)
    os.chmod(socket_path, 0o600)

    logger.info(
        "event=embedding_server_started socket=%s max_elements=%d embed_threads=%d",
        socket_path,
        max_elements,
        embed_threads,
    )

    # Wait until shutdown is signaled
    async with unix_server:
        await stop_event.wait()

    logger.info("event=embedding_server_draining")
    await server_obj.stop()

    # Clean up socket file
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass


def run_server(
    socket_path: str,
    max_elements: int = 10_000,
    embed_threads: int = 4,
) -> None:
    """Synchronous wrapper — runs serve() until shutdown.

    Intended for use as the target of multiprocessing.Process.
    """
    asyncio.run(
        serve(
            socket_path=socket_path,
            max_elements=max_elements,
            embed_threads=embed_threads,
        )
    )
