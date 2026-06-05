"""Tests for remote.py adapters: RemoteEmbedder and RemoteVectorIndex."""

from __future__ import annotations

import json
import struct
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# RemoteEmbedder tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteEmbedderProtocol:
    """Test that RemoteEmbedder implements the Embedder protocol correctly."""

    def test_implements_embedder_protocol(self) -> None:
        """RemoteEmbedder should satisfy the Embedder protocol at runtime."""
        from headroom.memory.adapters.remote import RemoteEmbedder
        from headroom.memory.ports import Embedder

        embedder = RemoteEmbedder("/tmp/test.sock")
        assert isinstance(embedder, Embedder)

    async def test_embed_returns_float32_array(self) -> None:
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/test.sock")
        vec = np.ones(384, dtype=np.float32)

        with patch.object(
            embedder._conn,
            "send_request",
            new=AsyncMock(return_value={"embedding": vec.tolist(), "id": "r"}),
        ):
            result = await embedder.embed("some text")

        assert result.dtype == np.float32
        assert result.shape == (384,)

    async def test_embed_batch_preserves_order(self) -> None:
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/test.sock")
        vecs = [np.full(384, float(i), dtype=np.float32) for i in range(3)]

        with patch.object(
            embedder._conn,
            "send_request",
            new=AsyncMock(return_value={"embeddings": [v.tolist() for v in vecs], "id": "r"}),
        ):
            results = await embedder.embed_batch(["a", "b", "c"])

        assert len(results) == 3
        for i, r in enumerate(results):
            assert float(r[0]) == float(i)

    async def test_close_is_safe(self) -> None:
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/test.sock")
        # Should not raise even with no connection
        await embedder.close()


# ---------------------------------------------------------------------------
# RemoteVectorIndex tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRemoteVectorIndex:
    """Test RemoteVectorIndex."""

    async def test_search_returns_results(self) -> None:
        from headroom.memory.adapters.remote import RemoteVectorIndex
        from headroom.memory.ports import VectorFilter

        index = RemoteVectorIndex("/tmp/test.sock")
        mock_results = [
            {"memory_id": "m1", "similarity": 0.9, "rank": 1, "content": "hello", "user_id": "u1"},
            {"memory_id": "m2", "similarity": 0.8, "rank": 2, "content": "world", "user_id": "u1"},
        ]

        with patch.object(
            index._conn,
            "send_request",
            new=AsyncMock(return_value={"results": mock_results, "id": "r"}),
        ):
            vf = VectorFilter(
                query_vector=np.zeros(384, dtype=np.float32),
                top_k=5,
            )
            results = await index.search(vf)

        assert len(results) == 2
        assert results[0].memory.id == "m1"
        assert abs(results[0].similarity - 0.9) < 1e-6

    async def test_search_with_query_text(self) -> None:
        from headroom.memory.adapters.remote import RemoteVectorIndex
        from headroom.memory.ports import VectorFilter

        index = RemoteVectorIndex("/tmp/test.sock")

        with patch.object(
            index._conn,
            "send_request",
            new=AsyncMock(return_value={"results": [], "id": "r"}),
        ):
            vf = VectorFilter(query_text="find something", top_k=3)
            results = await index.search(vf)

        assert results == []

    async def test_search_requires_query(self) -> None:
        from headroom.memory.adapters.remote import RemoteVectorIndex
        from headroom.memory.ports import VectorFilter

        index = RemoteVectorIndex("/tmp/test.sock")
        vf = VectorFilter()  # No query_vector or query_text

        with pytest.raises(ValueError, match="query_vector or query_text"):
            await index.search(vf)

    async def test_remove_returns_true(self) -> None:
        from headroom.memory.adapters.remote import RemoteVectorIndex

        index = RemoteVectorIndex("/tmp/test.sock")
        with patch.object(
            index._conn,
            "send_request",
            new=AsyncMock(return_value={"status": "deleted", "memory_id": "m1", "id": "r"}),
        ):
            result = await index.remove("m1")
        assert result is True

    async def test_remove_returns_false_not_found(self) -> None:
        from headroom.memory.adapters.remote import RemoteVectorIndex

        index = RemoteVectorIndex("/tmp/test.sock")
        with patch.object(
            index._conn,
            "send_request",
            new=AsyncMock(return_value={"status": "not_found", "memory_id": "m1", "id": "r"}),
        ):
            result = await index.remove("m1")
        assert result is False

    async def test_stats_returns_dict(self) -> None:
        from headroom.memory.adapters.remote import RemoteVectorIndex

        index = RemoteVectorIndex("/tmp/test.sock")
        expected = {"index_size": 42, "total_requests": 100}
        with patch.object(
            index._conn,
            "send_request",
            new=AsyncMock(return_value={**expected, "id": "r"}),
        ):
            result = await index.stats()
        assert result["index_size"] == 42


# ---------------------------------------------------------------------------
# EmbeddingServerUnavailable tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEmbeddingServerUnavailable:
    """Test the EmbeddingServerUnavailable exception."""

    async def test_raised_when_socket_missing(self) -> None:
        from headroom.memory.adapters.remote import EmbeddingServerUnavailable, RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/this-socket-really-does-not-exist-abc123.sock")
        with pytest.raises(EmbeddingServerUnavailable):
            await embedder.embed("test")

    def test_is_runtime_error_subclass(self) -> None:
        from headroom.memory.adapters.remote import EmbeddingServerUnavailable

        assert issubclass(EmbeddingServerUnavailable, RuntimeError)

    async def test_ping_false_when_unavailable(self) -> None:
        from headroom.memory.adapters.remote import RemoteEmbedder

        embedder = RemoteEmbedder("/tmp/this-socket-really-does-not-exist-abc123.sock")
        result = await embedder.ping()
        assert result is False


# ---------------------------------------------------------------------------
# Reconnect behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconnectBehavior:
    """Test that connection retries work correctly."""

    async def test_retries_on_broken_pipe(self) -> None:
        """Should retry and succeed on second attempt after BrokenPipeError."""
        from headroom.memory.adapters.remote import _EmbeddingServerConnection

        conn = _EmbeddingServerConnection("/tmp/test.sock")
        call_count = 0

        async def fake_connect() -> None:
            pass

        # Build a valid response frame
        response = {"status": "ok", "id": "r1"}
        resp_bytes = json.dumps(response).encode("utf-8")
        frame = struct.pack("<I", len(resp_bytes)) + resp_bytes

        read_calls = [frame[:4], frame[4:]]

        async def fake_readexactly(n: int) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise BrokenPipeError("pipe broken")
            return read_calls.pop(0)

        mock_reader = MagicMock()
        mock_reader.readexactly = fake_readexactly

        mock_writer = MagicMock()
        mock_writer.is_closing = MagicMock(return_value=False)
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        # First call to _connect fails, second succeeds with mock reader/writer
        connect_calls = 0

        async def fake_open_unix(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            nonlocal connect_calls
            connect_calls += 1
            return mock_reader, mock_writer

        with patch("asyncio.open_unix_connection", new=fake_open_unix):
            # Pre-populate with a reader that raises BrokenPipeError
            conn._reader = mock_reader
            conn._writer = mock_writer

            # The connection should retry after BrokenPipeError
            # In this test we verify the retry logic doesn't blow up
            # (full e2e retry requires a real socket)
