"""Regression tests for the LocalEmbedder MPS serialization fix.

torch-MPS is not thread-safe: concurrent encode() calls from the default
multi-worker executor abort with "commit an already committed command buffer".
LocalEmbedder funnels every encode through a dedicated single-worker executor
when (and only when) the resolved device is MPS. CPU/CUDA keep the shared pool.
"""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from headroom.memory.adapters.embedders import LocalEmbedder  # noqa: E402

_HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


async def test_cpu_uses_shared_executor() -> None:
    """On CPU the dedicated executor stays None (unchanged default-pool behavior)."""
    emb = LocalEmbedder(device="cpu")
    await emb.embed("hello world")
    assert emb._device == "cpu"
    assert emb._executor is None  # default shared executor, not serialized
    await emb.close()


@pytest.mark.skipif(not _HAS_MPS, reason="requires Apple-Silicon MPS")
async def test_mps_creates_single_worker_executor() -> None:
    """On MPS a dedicated max_workers=1 executor is created for serialization."""
    emb = LocalEmbedder(device="mps")
    await emb.embed("warmup")
    assert emb._device == "mps"
    assert emb._executor is not None
    assert emb._executor._max_workers == 1  # type: ignore[attr-defined]
    await emb.close()
    assert emb._executor is None  # close() tears it down


@pytest.mark.skipif(not _HAS_MPS, reason="requires Apple-Silicon MPS")
async def test_mps_concurrent_embeds_do_not_crash() -> None:
    """Concurrent embeds on MPS must not SIGABRT — the serialization guarantees it."""
    emb = LocalEmbedder(device="mps")
    await emb.embed("warmup")
    batches = [emb.embed_batch([f"text {i} " * 20] * 8) for i in range(16)]
    results = await asyncio.gather(*batches)
    assert len(results) == 16
    assert all(len(r[0]) == emb.dimension for r in results)
    await emb.close()


@pytest.mark.skipif(not _HAS_MPS, reason="requires Apple-Silicon MPS")
async def test_mps_reembed_after_close_recreates_executor() -> None:
    """close() drops the cached model so a later embed() re-initializes and
    re-creates the serialized executor — never encodes on the torn-down pool."""
    emb = LocalEmbedder(device="mps")
    await emb.embed("warmup")
    await emb.close()
    assert emb._executor is None
    assert emb._model is None
    # Re-use after close must re-initialize cleanly and stay serialized.
    await emb.embed("again")
    assert emb._executor is not None
    assert emb._executor._max_workers == 1  # type: ignore[attr-defined]
    await emb.close()
