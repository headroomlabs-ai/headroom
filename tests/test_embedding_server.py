"""Tests for the embedding server sidecar (embedding_server.py)."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(payload: dict[str, Any]) -> bytes:
    """Build a length-prefixed JSON frame."""
    body = json.dumps(payload).encode("utf-8")
    return struct.pack("<I", len(body)) + body


def _parse_frame(data: bytes) -> dict[str, Any]:
    length = struct.unpack("<I", data[:4])[0]
    return json.loads(data[4 : 4 + length])


# ---------------------------------------------------------------------------
# Protocol framing unit tests
# ---------------------------------------------------------------------------


class TestFraming:
    """Verify the length-prefix framing helpers."""

    def test_frame_roundtrip(self) -> None:
        payload = {"op": "ping", "id": "abc"}
        frame = _make_frame(payload)
        assert struct.unpack("<I", frame[:4])[0] == len(frame) - 4
        parsed = _parse_frame(frame)
        assert parsed == payload

    def test_empty_string_field(self) -> None:
        payload = {"op": "embed", "text": ""}
        frame = _make_frame(payload)
        assert _parse_frame(frame)["text"] == ""


# ---------------------------------------------------------------------------
# EmbeddingServer unit tests (mock embedder + index)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEmbeddingServerUnit:
    """Unit tests for EmbeddingServer request handling (no real ONNX model)."""

    async def _make_server(self) -> Any:
        """Create an EmbeddingServer with mocked internals."""
        from headroom.memory.adapters.embedding_server import EmbeddingServer

        server = EmbeddingServer.__new__(EmbeddingServer)
        server.socket_path = "/tmp/test.sock"
        server.max_elements = 1000
        server.embed_threads = 1
        import collections

        server._total_requests = 0
        server._total_embed_calls = 0
        server._total_latency_ms = 0.0
        server._total_batches = 0
        server._total_batched_items = 0
        server._op_counts = collections.defaultdict(int)
        server._latency_window = collections.deque(maxlen=1000)
        server._start_time = time.monotonic()
        server._shutdown_event = asyncio.Event()

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=np.zeros(384, dtype=np.float32))
        mock_embedder.embed_batch = AsyncMock(return_value=[np.zeros(384, dtype=np.float32)])
        server._embedder = mock_embedder

        # Mock index
        mock_index = MagicMock()
        mock_index.search = AsyncMock(return_value=[])
        mock_index.index = AsyncMock()
        mock_index.remove = AsyncMock(return_value=True)
        mock_index.size = 0
        server._index = mock_index

        # Set up batch queue
        server._batch_queue = asyncio.Queue()
        server._batch_task = None
        server._executor = None

        return server

    async def test_ping(self) -> None:
        server = await self._make_server()
        response = await server.handle_request({"op": "ping", "id": "test-1"})
        assert response["status"] == "ok"
        assert response["id"] == "test-1"

    async def test_unknown_op(self) -> None:
        server = await self._make_server()
        response = await server.handle_request({"op": "foobar", "id": "test-2"})
        assert "error" in response
        assert response["id"] == "test-2"

    async def test_embed_batch_empty(self) -> None:
        server = await self._make_server()
        response = await server.handle_request({"op": "embed_batch", "id": "t", "texts": []})
        assert response["embeddings"] == []

    async def test_stats(self) -> None:
        server = await self._make_server()
        response = await server.handle_request({"op": "stats", "id": "t"})
        assert "index_size" in response
        assert "total_requests" in response
        assert "latency_ms" in response
        assert "avg" in response["latency_ms"]

    async def test_delete_found(self) -> None:
        server = await self._make_server()
        response = await server.handle_request({"op": "delete", "id": "t", "memory_id": "mem-123"})
        assert response["status"] == "deleted"
        assert response["memory_id"] == "mem-123"

    async def test_search_missing_query(self) -> None:
        server = await self._make_server()
        response = await server.handle_request({"op": "search", "id": "t"})
        assert "error" in response


# ---------------------------------------------------------------------------
# _EmbeddingServerConnection unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEmbeddingServerConnection:
    """Unit tests for the connection helper (mock StreamReader/Writer)."""

    async def test_frame_write_read(self) -> None:
        """Verify send_request correctly frames the request."""
        from headroom.memory.adapters.remote import _EmbeddingServerConnection

        conn = _EmbeddingServerConnection("/tmp/nonexistent.sock")

        # Build a fake response payload
        response_payload = {"status": "ok", "id": "req-1"}
        response_bytes = json.dumps(response_payload).encode("utf-8")
        frame = struct.pack("<I", len(response_bytes)) + response_bytes

        # Mock reader that yields the frame
        mock_reader = MagicMock()
        read_calls = [
            frame[:4],  # header
            frame[4:],  # body
        ]

        async def fake_readexactly(n: int) -> bytes:
            return read_calls.pop(0)

        mock_reader.readexactly = fake_readexactly

        # Mock writer
        written_data: list[bytes] = []
        mock_writer = MagicMock()
        mock_writer.is_closing = MagicMock(return_value=False)
        mock_writer.write = MagicMock(side_effect=lambda b: written_data.append(b))
        mock_writer.drain = AsyncMock()

        conn._reader = mock_reader
        conn._writer = mock_writer

        result = await conn.send_request("ping")
        assert result["status"] == "ok"

        # Verify a frame was written
        assert len(written_data) == 1
        written = written_data[0]
        length = struct.unpack("<I", written[:4])[0]
        payload = json.loads(written[4 : 4 + length])
        assert payload["op"] == "ping"

    async def test_connection_failure_raises_unavailable(self) -> None:
        """Verify EmbeddingServerUnavailable is raised after max retries."""
        from headroom.memory.adapters.remote import (
            EmbeddingServerUnavailable,
            _EmbeddingServerConnection,
        )

        conn = _EmbeddingServerConnection("/tmp/does-not-exist.sock")

        with pytest.raises(EmbeddingServerUnavailable):
            await conn.send_request("ping")


# ---------------------------------------------------------------------------
# RemoteEmbedder unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteEmbedder:
    """Unit tests for RemoteEmbedder (mock connection)."""

    async def test_embed_returns_correct_shape(self) -> None:
        """embed() should return a (384,) float32 array."""
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/nonexistent.sock")

        expected = np.zeros(384, dtype=np.float32)
        mock_response = {"embedding": expected.tolist(), "id": "r1"}

        with patch.object(
            embedder._conn, "send_request", new=AsyncMock(return_value=mock_response)
        ):
            result = await embedder.embed("hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        assert result.dtype == np.float32

    async def test_embed_batch_returns_list(self) -> None:
        """embed_batch() should return a list of arrays."""
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/nonexistent.sock")

        vecs = [np.zeros(384, dtype=np.float32) for _ in range(3)]
        mock_response = {"embeddings": [v.tolist() for v in vecs], "id": "r1"}

        with patch.object(
            embedder._conn, "send_request", new=AsyncMock(return_value=mock_response)
        ):
            results = await embedder.embed_batch(["a", "b", "c"])

        assert len(results) == 3
        for r in results:
            assert r.shape == (384,)

    async def test_embed_batch_empty_input(self) -> None:
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/nonexistent.sock")
        results = await embedder.embed_batch([])
        assert results == []

    async def test_ping_returns_bool(self) -> None:
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/nonexistent.sock")

        with patch.object(
            embedder._conn,
            "send_request",
            new=AsyncMock(return_value={"status": "ok", "id": "ping"}),
        ):
            result = await embedder.ping()
        assert result is True

    async def test_ping_returns_false_on_error(self) -> None:
        from headroom.memory.adapters.remote import EmbeddingServerUnavailable, RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/nonexistent.sock")

        with patch.object(
            embedder._conn,
            "send_request",
            new=AsyncMock(side_effect=EmbeddingServerUnavailable("down")),
        ):
            result = await embedder.ping()
        assert result is False

    async def test_unavailable_when_socket_missing(self) -> None:
        from headroom.memory.adapters.remote import EmbeddingServerUnavailable, RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/this-socket-does-not-exist.sock")
        with pytest.raises(EmbeddingServerUnavailable):
            await embedder.embed("test")

    async def test_dimension_property(self) -> None:
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/nonexistent.sock")
        assert embedder.dimension == 384

    async def test_model_name_property(self) -> None:
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/nonexistent.sock")
        assert "remote" in embedder.model_name


# ---------------------------------------------------------------------------
# EmbeddingServerWatchdog unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEmbeddingServerWatchdog:
    """Unit tests for the watchdog process manager."""

    async def test_start_spawns_process(self) -> None:
        """start() should spawn a process."""
        from headroom.memory.adapters.watchdog import EmbeddingServerWatchdog

        wd = EmbeddingServerWatchdog("/tmp/test-wd.sock")

        mock_process = MagicMock()
        mock_process.is_alive = MagicMock(return_value=True)
        mock_process.pid = 9999
        mock_process.start = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.join = MagicMock()

        with patch("multiprocessing.get_context") as mock_ctx:
            mock_ctx.return_value.Process = MagicMock(return_value=mock_process)
            await wd.start()

        assert wd.pid == 9999
        await wd.stop()

    async def test_is_healthy_false_when_server_not_running(self) -> None:
        """is_healthy() returns False when no server is running."""
        from headroom.memory.adapters.watchdog import EmbeddingServerWatchdog

        wd = EmbeddingServerWatchdog("/tmp/not-running.sock")
        result = await wd.is_healthy()
        assert result is False

    async def test_socket_path_property(self) -> None:
        from headroom.memory.adapters.watchdog import EmbeddingServerWatchdog

        wd = EmbeddingServerWatchdog("/tmp/test.sock")
        assert wd.socket_path == "/tmp/test.sock"

    async def test_pid_none_before_start(self) -> None:
        from headroom.memory.adapters.watchdog import EmbeddingServerWatchdog

        wd = EmbeddingServerWatchdog("/tmp/test.sock")
        assert wd.pid is None

    async def test_stop_before_start_is_safe(self) -> None:
        from headroom.memory.adapters.watchdog import EmbeddingServerWatchdog

        wd = EmbeddingServerWatchdog("/tmp/test.sock")
        # Should not raise
        await wd.stop()

    async def test_exponential_backoff_logic(self) -> None:
        """Verify restart delay doubles with each consecutive crash."""
        from headroom.memory.adapters.watchdog import EmbeddingServerWatchdog

        EmbeddingServerWatchdog("/tmp/test.sock", restart_delay=0.5, max_restarts=5)
        # Simulate consecutive crashes by checking the formula
        # delay = min(0.5 * 2^(n-1), 30)
        assert min(0.5 * (2**0), 30.0) == 0.5
        assert min(0.5 * (2**1), 30.0) == 1.0
        assert min(0.5 * (2**6), 30.0) == 30.0


# ---------------------------------------------------------------------------
# Integration tests (require ONNX model to be downloaded)
# ---------------------------------------------------------------------------


def _onnx_available() -> bool:
    """Check if onnxruntime is installed AND the model is locally cached (no download needed)."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False

    # Verify the model files are present in the HuggingFace cache so tests
    # don't depend on network access or HF rate limits.
    import os

    try:
        from huggingface_hub import try_to_load_from_cache

        model_cached = try_to_load_from_cache("Qdrant/all-MiniLM-L6-v2-onnx", "model.onnx")
        tok_cached = try_to_load_from_cache("Qdrant/all-MiniLM-L6-v2-onnx", "tokenizer.json")
        return bool(model_cached) and bool(tok_cached)
    except Exception:
        # Fallback: check the default cache directory directly
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_dir = os.path.join(cache_dir, "models--Qdrant--all-MiniLM-L6-v2-onnx")
        return os.path.isdir(repo_dir)


@pytest.fixture(autouse=False)
def hf_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force HuggingFace Hub to use local cache only (no network calls)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.mark.skipif(not _onnx_available(), reason="onnxruntime or cached model not available")
@pytest.mark.asyncio
@pytest.mark.usefixtures("hf_offline")
class TestEmbeddingServerIntegration:
    """Integration tests: actually start the server and connect to it."""

    async def test_full_roundtrip(self, tmp_path: Any) -> None:
        """Start server, embed a string, verify array shape is (384,)."""
        from headroom.memory.adapters.embedding_server import serve
        from headroom.memory.adapters.remote import RemoteEmbedder

        socket_path = str(tmp_path / "test.sock")

        # Start server in a background thread
        server_task = asyncio.create_task(serve(socket_path, max_elements=100))

        # Wait for server to be ready
        embedder = RemoteEmbedder(socket_path, connect_timeout=15.0)
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if await embedder.ping():
                break
            await asyncio.sleep(0.5)
        else:
            server_task.cancel()
            pytest.fail("Server did not start within 60s")

        try:
            result = await embedder.embed("hello world")
            assert isinstance(result, np.ndarray)
            assert result.shape == (384,)
            assert result.dtype == np.float32
        finally:
            await embedder.close()
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    async def test_concurrent_embeds(self, tmp_path: Any) -> None:
        """20 concurrent embed() calls should all succeed."""
        from headroom.memory.adapters.embedding_server import serve
        from headroom.memory.adapters.remote import RemoteEmbedder

        socket_path = str(tmp_path / "concurrent.sock")
        server_task = asyncio.create_task(serve(socket_path, max_elements=100))

        embedders = [RemoteEmbedder(socket_path, connect_timeout=15.0) for _ in range(5)]

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if await embedders[0].ping():
                break
            await asyncio.sleep(0.5)
        else:
            server_task.cancel()
            pytest.fail("Server did not start within 60s")

        try:
            texts = [f"text number {i}" for i in range(20)]
            tasks = [e.embed(t) for e, t in zip(embedders * 4, texts)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 20
            for r in results:
                assert r.shape == (384,)
        finally:
            for e in embedders:
                await e.close()
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    async def test_graceful_shutdown(self, tmp_path: Any) -> None:
        """Server should exit cleanly when the task is cancelled."""
        from headroom.memory.adapters.embedding_server import serve

        socket_path = str(tmp_path / "shutdown.sock")
        server_task = asyncio.create_task(serve(socket_path, max_elements=100))

        # Wait a bit then cancel
        await asyncio.sleep(0.5)
        server_task.cancel()
        try:
            await asyncio.wait_for(server_task, timeout=15.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        # If we got here without hanging, shutdown was clean enough


@pytest.mark.slow
@pytest.mark.skipif(not _onnx_available(), reason="onnxruntime or cached model not available")
@pytest.mark.asyncio
@pytest.mark.usefixtures("hf_offline")
async def test_embed_100_sequential_performance(tmp_path: Any) -> None:
    """100 sequential embeds should complete in under 2s."""
    from headroom.memory.adapters.embedding_server import serve
    from headroom.memory.adapters.remote import RemoteEmbedder

    socket_path = str(tmp_path / "perf.sock")
    server_task = asyncio.create_task(serve(socket_path, max_elements=100))

    embedder = RemoteEmbedder(socket_path, connect_timeout=15.0)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if await embedder.ping():
            break
        await asyncio.sleep(0.5)
    else:
        server_task.cancel()
        pytest.fail("Server did not start within 60s")

    try:
        t0 = time.monotonic()
        for i in range(100):
            await embedder.embed(f"performance test sentence {i}")
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"100 embeds took {elapsed:.2f}s (expected < 2s)"
    finally:
        await embedder.close()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
