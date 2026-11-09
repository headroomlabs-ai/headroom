"""code_aware's unknown-language fallback must honour the Kompress switches.

Regression tests for #4719: ``CodeAwareCompressor._fallback_compress`` used to
instantiate Kompress directly, ignoring ``ContentRouterConfig.enable_kompress``
(HEADROOM_COMPRESSORS / HEADROOM_DISABLE_KOMPRESS), loading torch + the model
(~1 GB RSS) even with Kompress disabled. ``_is_onnx_available`` also imported
``transformers`` (which pulls in torch when installed) despite documenting
itself as lightweight.
"""

from __future__ import annotations

import builtins
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _clear_onnx_probe_cache():
    from headroom.transforms import kompress_compressor

    kompress_compressor._is_onnx_available.cache_clear()
    yield
    kompress_compressor._is_onnx_available.cache_clear()


def _unknown_language_code() -> str:
    """Code ``detect_language`` cannot identify -> CodeLanguage.UNKNOWN."""
    return "\n".join(
        f"step {i}: load register r{i % 8} <- mem[{i * 4}]; jump-if-zero r{i % 8} label_{i}"
        for i in range(80)
    )


def test_router_wires_enable_kompress_into_code_compressor(monkeypatch: pytest.MonkeyPatch) -> None:
    """One switch covers both Kompress paths: router dispatch and code_aware fallback."""
    from headroom.transforms import code_compressor
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    monkeypatch.setattr(code_compressor, "_check_tree_sitter_available", lambda: True)

    router = ContentRouter(ContentRouterConfig(enable_kompress=False))
    compressor = router._get_code_compressor()
    assert compressor is not None, "expected a code compressor to be built"
    assert compressor.config.fallback_to_kompress is False

    # Positive control: the default keeps the fallback enabled.
    router_on = ContentRouter(ContentRouterConfig(enable_kompress=True))
    assert router_on._get_code_compressor().config.fallback_to_kompress is True


def test_unknown_language_never_consults_kompress_when_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fallback_to_kompress=False must pass unknown code through untouched."""
    from headroom.transforms import code_compressor, kompress_compressor
    from headroom.transforms.code_compressor import (
        CodeAwareCompressor,
        CodeCompressorConfig,
    )

    def _forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("Kompress must not be consulted when the fallback is disabled")

    monkeypatch.setattr(kompress_compressor, "is_kompress_available", _forbidden)
    monkeypatch.setattr(
        kompress_compressor,
        "KompressCompressor",
        lambda *a, **k: (_ for _ in ()).throw(_forbidden()),
    )

    compressor = CodeAwareCompressor(
        CodeCompressorConfig(fallback_to_kompress=False, enable_ccr=False)
    )
    code = _unknown_language_code()
    assert code_compressor.detect_language(code)[0] == code_compressor.CodeLanguage.UNKNOWN

    result = compressor.compress(code)
    assert result.compression_ratio == 1.0
    assert result.compressed == code
    assert result.language == code_compressor.CodeLanguage.UNKNOWN


def test_onnx_probe_does_not_import_transformers_or_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The availability probe stays lightweight: no torch, no transformers import."""
    import importlib.machinery

    from headroom.transforms import kompress_compressor

    # Stub the ML stack so the probe's imports succeed trivially; the probe must
    # not even attempt them. __spec__ is set so importlib.util.find_spec returns
    # a spec instead of raising ValueError for a spec-less sys.modules entry.
    for name in ("onnxruntime", "transformers"):
        stub = types.ModuleType(name)
        stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, stub)

    attempted: list[str] = []
    real_import = builtins.__import__

    def _tracking_import(name, *args, **kwargs):
        if name.split(".")[0] in ("transformers", "torch"):
            attempted.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _tracking_import)

    kompress_compressor._is_onnx_available()

    assert attempted == [], f"probe imported the ML stack: {attempted}"


def test_onnx_probe_result_unchanged_with_ml_stack_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe still reports availability when onnxruntime+transformers are importable."""
    import importlib.machinery

    from headroom.transforms import kompress_compressor

    for name in ("onnxruntime", "transformers"):
        stub = types.ModuleType(name)
        stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, stub)

    assert kompress_compressor._is_onnx_available() is True
