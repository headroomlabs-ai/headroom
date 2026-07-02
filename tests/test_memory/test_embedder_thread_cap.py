"""Regression tests for the LocalEmbedder CPU thread cap (issue #198).

Under concurrent load the torch/sentence-transformers embedder oversubscribes
BLAS/OpenMP threads (≈ ``os.cpu_count()`` per ``encode()``), starving the
asyncio event loop and spiking ``/livez`` latency. ``LocalEmbedder`` now runs
CPU encodes on a dedicated, size-limited executor whose workers each pin their
torch/BLAS/OpenMP thread pool, bounding total embedding threads to
``HEADROOM_EMBED_CONCURRENCY x HEADROOM_EMBED_NUM_THREADS``. The ONNX embedder
already caps its threads; this brings the torch path to parity.
"""

from __future__ import annotations

import os

import pytest

from headroom.memory.adapters import embedders
from headroom.memory.adapters.embedders import (
    _BLAS_THREAD_ENV_VARS,
    _init_cpu_embed_worker,
    _resolve_embed_concurrency,
    _resolve_embed_thread_cap,
)

# ---------------------------------------------------------------------------
# Env resolution (no torch required)
# ---------------------------------------------------------------------------


def test_thread_cap_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_EMBED_NUM_THREADS", raising=False)
    assert _resolve_embed_thread_cap() == 1


def test_thread_cap_reads_positive_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_EMBED_NUM_THREADS", "3")
    assert _resolve_embed_thread_cap() == 3


def test_thread_cap_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_EMBED_NUM_THREADS", "not-a-number")
    assert _resolve_embed_thread_cap() == 1


def test_thread_cap_non_positive_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_EMBED_NUM_THREADS", "0")
    assert _resolve_embed_thread_cap() == 1


def test_concurrency_default_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_EMBED_CONCURRENCY", raising=False)
    value = _resolve_embed_concurrency()
    assert 1 <= value <= 4
    assert value <= (os.cpu_count() or 1)


def test_concurrency_reads_positive_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_EMBED_CONCURRENCY", "7")
    assert _resolve_embed_concurrency() == 7


# ---------------------------------------------------------------------------
# Worker initializer env application (no torch required)
# ---------------------------------------------------------------------------


def test_worker_init_sets_blas_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _BLAS_THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HEADROOM_EMBED_NUM_THREADS", "2")

    _init_cpu_embed_worker()

    for var in _BLAS_THREAD_ENV_VARS:
        assert os.environ[var] == "2", var


def test_worker_init_does_not_override_operator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit operator setting must win over our default (setdefault)."""
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("HEADROOM_EMBED_NUM_THREADS", "1")

    _init_cpu_embed_worker()

    assert os.environ["OMP_NUM_THREADS"] == "8"


# ---------------------------------------------------------------------------
# Behavioral: real CPU load path bounds every encode worker's thread pool
# ---------------------------------------------------------------------------


async def test_cpu_embed_workers_are_thread_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU encodes run on a dedicated, size-limited executor and every worker
    pins its torch intra-op thread pool to the configured cap."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")

    monkeypatch.setenv("HEADROOM_EMBED_NUM_THREADS", "1")
    monkeypatch.setenv("HEADROOM_EMBED_CONCURRENCY", "2")

    emb = embedders.LocalEmbedder(device="cpu")
    await emb.embed("hello world")

    assert emb._device == "cpu"
    assert emb._executor is not None
    assert emb._executor._max_workers == 2  # type: ignore[attr-defined]

    # Probe the actual encode workers: each was pinned to 1 intra-op thread.
    futures = [emb._executor.submit(torch.get_num_threads) for _ in range(4)]
    assert [f.result() for f in futures] == [1, 1, 1, 1]

    await emb.close()
    assert emb._executor is None  # close() tears the executor down
