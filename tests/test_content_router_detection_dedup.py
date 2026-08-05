"""Regression guards for the content-detection dedup on the router hot path.

``ContentRouter.compress`` used to run the native ``_detect_content`` twice on
identical content: once for (default-off) debug logging, then again inside
``_determine_strategy``.  The native Rust/Magika pass is the router's hottest
per-message cost, so ``compress`` now runs it once and threads the result into
``_determine_strategy``.  These tests pin the call count at one and prove the
threaded result routes identically to the recomputed one.
"""

from __future__ import annotations

import pytest

import headroom.transforms.content_router as content_router_module
from headroom.transforms.content_detector import ContentType, DetectionResult
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
)


@pytest.fixture(autouse=True)
def _reset_detect_module_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # The native-detector circuit breaker (#575) is process-wide; keep it from
    # leaking across tests (mirrors tests/test_transforms_content_router.py).
    monkeypatch.setattr(content_router_module, "_detect_native_unhealthy", False)
    monkeypatch.setattr(content_router_module, "_detect_backend_warned", False)
    monkeypatch.setattr(content_router_module, "_detect_panic_warned", False)


def _count_detects(monkeypatch: pytest.MonkeyPatch, result: DetectionResult) -> dict[str, int]:
    """Replace the module-level ``_detect_content`` with a deterministic counter.

    Both call sites (``compress`` and ``_determine_strategy``) resolve the name
    from the module namespace at call time, so a single patch counts every call.
    """
    calls = {"n": 0}

    def _counting(content: str) -> DetectionResult:
        calls["n"] += 1
        return result

    monkeypatch.setattr(content_router_module, "_detect_content", _counting)
    return calls


def test_compress_runs_detection_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # Before the dedup this was 2 — compress ran _detect_content for debug
    # logging, then _determine_strategy ran it again on identical bytes. The
    # fix threads the single result through, so it must now be exactly 1.
    router = ContentRouter(ContentRouterConfig(prefer_code_aware_for_code=False))
    calls = _count_detects(monkeypatch, DetectionResult(ContentType.PLAIN_TEXT, 1.0, {}))
    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: False)
    # Stub the downstream compressor — this test isolates detection, not output.
    monkeypatch.setattr(
        router,
        "_compress_pure",
        lambda *a, **k: RouterCompressionResult(
            compressed="x", original="x", strategy_used=CompressionStrategy.TEXT
        ),
    )

    router.compress("a representative plain-text blob that is comfortably non-empty")

    assert calls["n"] == 1


@pytest.mark.parametrize(
    ("mixed", "detected"),
    [
        (False, ContentType.SOURCE_CODE),
        (False, ContentType.JSON_ARRAY),
        (False, ContentType.BUILD_OUTPUT),
        (False, ContentType.PLAIN_TEXT),
        (True, ContentType.SOURCE_CODE),  # mixed regex hit, native says code -> trust native
        (True, ContentType.PLAIN_TEXT),  # genuinely mixed
    ],
)
def test_determine_strategy_threaded_matches_recomputed(
    monkeypatch: pytest.MonkeyPatch, mixed: bool, detected: ContentType
) -> None:
    # Threading precomputed (mixed, detection) must pick the SAME strategy as
    # letting _determine_strategy recompute them: the dedup changes cost, not
    # routing.
    router = ContentRouter(ContentRouterConfig(prefer_code_aware_for_code=False))
    detection = DetectionResult(detected, 1.0, {})
    monkeypatch.setattr(content_router_module, "is_mixed_content", lambda content: mixed)
    monkeypatch.setattr(content_router_module, "_detect_content", lambda content: detection)

    recomputed = router._determine_strategy("payload")
    threaded = router._determine_strategy("payload", mixed=mixed, detection=detection)

    assert threaded is recomputed
