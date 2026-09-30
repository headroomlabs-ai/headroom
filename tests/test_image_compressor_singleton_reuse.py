"""Image models must be loaded once and reused, not rebuilt per request (#2513).

Every image request built a new ImageCompressor and a new OnnxTechniqueRouter,
each loading native ort.InferenceSession models that grew worker RSS to 1+ GB
over a day. These tests pin the caching / singleton behavior with the heavy
model construction mocked out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from headroom.image.compressor import ImageCompressor


def test_onnx_router_is_built_once_and_cached() -> None:
    compressor = ImageCompressor()
    fake_router = MagicMock(name="OnnxTechniqueRouter")

    with patch("headroom.image.onnx_router.OnnxTechniqueRouter", return_value=fake_router) as ctor:
        first = compressor._get_onnx_router()
        second = compressor._get_onnx_router()

    assert first is second is fake_router
    assert ctor.call_count == 1


def test_close_is_a_noop_on_a_singleton_instance() -> None:
    compressor = ImageCompressor()
    compressor._is_singleton = True
    router = MagicMock()
    compressor._router = router
    compressor._onnx_router = MagicMock()

    compressor.close()

    # Models stay loaded so the next request reuses them.
    assert compressor._router is router
    assert compressor._onnx_router is not None
    router.release_models.assert_not_called()


def test_close_releases_models_on_a_non_singleton_instance() -> None:
    compressor = ImageCompressor()
    router = MagicMock()
    compressor._router = router
    compressor._onnx_router = MagicMock()

    compressor.close()

    router.release_models.assert_called_once()
    assert compressor._router is None
    assert compressor._onnx_router is None


def test_get_image_compressor_returns_a_shared_singleton() -> None:
    import headroom.proxy.helpers as helpers

    helpers._image_compressor_available = None
    helpers._image_compressor_instance = None
    try:
        a = helpers._get_image_compressor()
        b = helpers._get_image_compressor()
        assert a is not None
        assert a is b
        assert a._is_singleton is True
    finally:
        helpers._image_compressor_available = None
        helpers._image_compressor_instance = None


def test_worker_compressor_is_reused_across_calls() -> None:
    import headroom.proxy.image_isolation as iso

    iso._WORKER_COMPRESSOR = None
    try:
        a = iso._get_worker_compressor()
        b = iso._get_worker_compressor()
        assert a is b
        assert a._is_singleton is True
    finally:
        iso._WORKER_COMPRESSOR = None
