from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Any

logger = logging.getLogger("headroom.proxy")

_IMAGE_POOL_LOCK = threading.Lock()
_IMAGE_POOL: ProcessPoolExecutor | None = None

# Per-worker-process cached compressor. The image pool is a persistent
# single-worker ProcessPoolExecutor, so building a fresh ImageCompressor (and,
# inside it, a fresh ONNX router loading native ort.InferenceSession models) on
# every call accumulated native memory in the worker and grew RSS to 1+ GB over
# a day (#2513). Load the models once per worker and reuse them.
_WORKER_COMPRESSOR: Any = None


def _get_worker_compressor() -> Any:
    global _WORKER_COMPRESSOR
    if _WORKER_COMPRESSOR is None:
        from headroom.image import ImageCompressor

        instance = ImageCompressor()
        # Shared across calls in this worker: don't let a per-call close() unload
        # the models the next call reuses.
        instance._is_singleton = True
        _WORKER_COMPRESSOR = instance
    return _WORKER_COMPRESSOR


def _compress_messages_worker(
    messages: list[dict[str, Any]],
    provider: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    compressor = _get_worker_compressor()
    compressed = compressor.compress(messages, provider=provider)
    if compressor.last_result is None:
        return compressed, None
    return compressed, {
        "technique": compressor.last_result.technique.value,
        "original_tokens": compressor.last_result.original_tokens,
        "compressed_tokens": compressor.last_result.compressed_tokens,
        "confidence": compressor.last_result.confidence,
        "savings_percent": compressor.last_result.savings_percent,
    }


def _success_worker(
    messages: list[dict[str, Any]],
    provider: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    compressed = [dict(message) for message in messages]
    compressed[-1] = {**compressed[-1], "content": f"compressed:{provider}"}
    return compressed, {
        "technique": "preserve",
        "original_tokens": 100,
        "compressed_tokens": 60,
        "confidence": 1.0,
        "savings_percent": 40.0,
    }


def _raise_worker(
    messages: list[dict[str, Any]],
    provider: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    raise RuntimeError(f"boom:{provider}")


def _sleep_worker(
    messages: list[dict[str, Any]],
    provider: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    time.sleep(0.2)
    return messages, None


def _abort_worker(
    messages: list[dict[str, Any]],
    provider: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    os.abort()


_IMAGE_WORKER: Callable[
    [list[dict[str, Any]], str],
    tuple[list[dict[str, Any]], dict[str, Any] | None],
] = _compress_messages_worker


def _image_pool() -> ProcessPoolExecutor:
    global _IMAGE_POOL
    with _IMAGE_POOL_LOCK:
        if _IMAGE_POOL is None:
            _IMAGE_POOL = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return _IMAGE_POOL


def _reset_image_pool() -> None:
    global _IMAGE_POOL
    with _IMAGE_POOL_LOCK:
        pool = _IMAGE_POOL
        _IMAGE_POOL = None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


async def run_image_compression_isolated(
    messages: list[dict[str, Any]],
    provider: str,
    *,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(_image_pool(), _IMAGE_WORKER, messages, provider)
        return await asyncio.wait_for(future, timeout=timeout)
    except BrokenProcessPool:
        logger.warning("Image compression worker crashed; forwarding original image payload")
        _reset_image_pool()
        return messages, None
    except TimeoutError:
        logger.warning("Image compression worker timed out; forwarding original image payload")
        _reset_image_pool()
        return messages, None
    except Exception as exc:
        logger.warning(
            "Image compression worker failed (%s); forwarding original image payload",
            type(exc).__name__,
        )
        _reset_image_pool()
        return messages, None
