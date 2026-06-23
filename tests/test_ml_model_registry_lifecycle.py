from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from headroom.models.ml_models import MLModelRegistry


@pytest.fixture(autouse=True)
def reset_ml_model_registry():
    MLModelRegistry.reset()
    yield
    MLModelRegistry.reset()


def test_unload_many_removes_requested_keys_once(monkeypatch) -> None:
    MLModelRegistry.reset()
    registry = MLModelRegistry.get()
    kept_model = object()
    registry._models.update(
        {
            "technique_router:demo": object(),
            "siglip:demo": object(),
            "sentence_transformer:keep": kept_model,
        }
    )
    release = Mock()
    monkeypatch.setattr(MLModelRegistry, "_release_runtime_memory", release)

    removed = MLModelRegistry.unload_many(["missing", "technique_router:demo", "siglip:demo"])

    assert removed == ["technique_router:demo", "siglip:demo"]
    assert registry._models == {"sentence_transformer:keep": kept_model}
    release.assert_called_once_with()


def test_unload_many_skips_runtime_cleanup_when_nothing_removed(monkeypatch) -> None:
    MLModelRegistry.reset()
    registry = MLModelRegistry.get()
    registry._models["sentence_transformer:keep"] = object()
    release = Mock()
    monkeypatch.setattr(MLModelRegistry, "_release_runtime_memory", release)

    removed = MLModelRegistry.unload_many(["missing"])

    assert removed == []
    assert "sentence_transformer:keep" in registry._models
    release.assert_not_called()


def test_unload_prefix_removes_only_matching_models(monkeypatch) -> None:
    MLModelRegistry.reset()
    registry = MLModelRegistry.get()
    kept_model = object()
    registry._models.update(
        {
            "siglip:a": object(),
            "siglip:b": object(),
            "technique_router:keep": kept_model,
        }
    )
    release = Mock()
    monkeypatch.setattr(MLModelRegistry, "_release_runtime_memory", release)

    removed = MLModelRegistry.unload_prefix("siglip:")

    assert removed == ["siglip:a", "siglip:b"]
    assert registry._models == {"technique_router:keep": kept_model}
    release.assert_called_once_with()


def test_unload_delegates_to_unload_many(monkeypatch) -> None:
    unload_many = Mock(return_value=["siglip:demo"])
    monkeypatch.setattr(MLModelRegistry, "unload_many", unload_many)

    assert MLModelRegistry.unload("siglip:demo") is True

    unload_many.assert_called_once_with(["siglip:demo"])


def test_release_runtime_memory_handles_missing_torch(monkeypatch) -> None:
    collect = Mock()
    monkeypatch.setattr("headroom.models.ml_models.gc.collect", collect)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == "torch":
            raise ImportError("torch unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    MLModelRegistry._release_runtime_memory()

    collect.assert_called_once_with()


def test_release_runtime_memory_clears_available_torch_caches(monkeypatch) -> None:
    collect = Mock()
    cuda = SimpleNamespace(is_available=Mock(return_value=True), empty_cache=Mock())
    mps = SimpleNamespace(empty_cache=Mock())
    fake_torch = SimpleNamespace(cuda=cuda, mps=mps)
    monkeypatch.setattr("headroom.models.ml_models.gc.collect", collect)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    MLModelRegistry._release_runtime_memory()

    collect.assert_called_once_with()
    cuda.is_available.assert_called_once_with()
    cuda.empty_cache.assert_called_once_with()
    mps.empty_cache.assert_called_once_with()
