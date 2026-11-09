"""Regression tests for the LocalEmbedder MPS serialization fix.

torch-MPS is not thread-safe: concurrent encode() calls from the default
multi-worker executor abort with "commit an already committed command buffer".
LocalEmbedder funnels every MPS encode through one process-owned queue, including
calls from separate instances that share a model. CPU keeps its capped executor.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from headroom.memory.adapters.embedders import LocalEmbedder  # noqa: E402
from headroom.models.ml_models import MLModelRegistry  # noqa: E402

_HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


class _ConcurrencyProbeModel:
    """Observe scheduling without allowing concurrent calls into Metal."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts: str | list[str], **kwargs: object) -> np.ndarray:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.02)
            shape = (384,) if isinstance(texts, str) else (len(texts), 384)
            return np.ones(shape, dtype=np.float32)
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def probe_model(monkeypatch: pytest.MonkeyPatch) -> _ConcurrencyProbeModel:
    model = _ConcurrencyProbeModel()
    monkeypatch.setattr(MLModelRegistry, "get_sentence_transformer", lambda *args: model)
    return model


async def test_mps_serializes_across_instances(probe_model: _ConcurrencyProbeModel) -> None:
    """Each instance must use the same queue when the registry shares a model."""
    embedders = [LocalEmbedder(device="mps") for _ in range(4)]
    try:
        for embedder in embedders:
            await embedder.embed("warmup")
        results = await asyncio.gather(
            *(
                embedders[i % 4].embed_batch(["first", "second"])
                if i % 2
                else embedders[i % 4].embed("text")
                for i in range(16)
            )
        )
        assert all(np.isfinite(np.asarray(result)).all() for result in results)
        assert probe_model.peak == 1, "MPS encodes overlapped across instances"
    finally:
        for embedder in embedders:
            await embedder.close()


async def test_mps_close_preserves_other_instances_and_reopening(
    probe_model: _ConcurrencyProbeModel,
) -> None:
    """Closing one client must not stop the shared queue or a later new client."""
    first, second = LocalEmbedder(device="mps"), LocalEmbedder(device="mps")
    try:
        before = await first.embed("text")
        await second.embed("warmup")
        await first.close()
        np.testing.assert_allclose(await second.embed("text"), before)
        np.testing.assert_allclose(await first.embed("text"), before)
        await second.close()
        await first.close()
        np.testing.assert_allclose(await second.embed("text"), before)
    finally:
        await first.close()
        await second.close()


async def test_cpu_uses_dedicated_thread_capped_executor() -> None:
    """On CPU a dedicated, size-limited executor is used so encodes run with a
    bounded thread pool instead of oversubscribing BLAS/OMP threads (issue #198)."""
    emb = LocalEmbedder(device="cpu")
    await emb.embed("hello world")
    assert emb._device == "cpu"
    assert emb._executor is not None  # dedicated capped pool, not the shared default
    assert emb._executor._max_workers >= 1  # type: ignore[attr-defined]
    await emb.close()
    assert emb._executor is None  # close() tears it down


@pytest.mark.skipif(not _HAS_MPS, reason="requires Apple-Silicon MPS")
async def test_mps_creates_single_worker_executor() -> None:
    """On MPS the shared queue uses max_workers=1 for serialization."""
    emb = LocalEmbedder(device="mps")
    await emb.embed("warmup")
    assert emb._device == "mps"
    assert emb._executor is not None
    assert emb._executor._max_workers == 1  # type: ignore[attr-defined]
    await emb.close()
    assert emb._executor is None  # close() releases this instance's reference


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
async def test_mps_reembed_after_close_reinitializes() -> None:
    """close() drops the cached model so a later embed() re-initializes and
    re-acquires the shared executor instead of using a torn-down pool."""
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
