"""Tests for EXTERNAL memory backends (entry-point plugins).

Three extension groups let packages register memory backends via
setuptools entry points:
  - headroom.memory_store
  - headroom.memory_vector
  - headroom.memory_text

A package registers a callable under one of these groups; the factory
loads it when the corresponding backend enum is EXTERNAL.

These tests do not require hnswlib and must remain independent of it.
"""

from __future__ import annotations

import pytest

from headroom.memory.config import (
    MemoryConfig,
    StoreBackend,
    TextBackend,
    VectorBackend,
)
from headroom.memory.factory import (
    _create_store,
    _create_text_index,
    _create_vector_index,
)


class _FakeEntryPoint:
    """Minimal stand-in for importlib.metadata.EntryPoint used in tests."""

    def __init__(self, name: str, target):
        self.name = name
        self._target = target

    def load(self):
        return self._target


def _patch_entry_points(monkeypatch, expected_group: str, name: str, target):
    """Patch headroom.memory.factory.entry_points to return our fake EP."""
    from headroom.memory import factory as factory_mod

    def fake_entry_points(*, group: str):
        if group == expected_group:
            return [_FakeEntryPoint(name, target)]
        return []

    monkeypatch.setattr(factory_mod, "entry_points", fake_entry_points)


class TestExternalStoreBackend:
    """EXTERNAL store backend loads via entry_points(group='headroom.memory_store')."""

    def test_loads_external_store(self, monkeypatch):
        sentinel = object()

        def make_store(config):
            assert isinstance(config, MemoryConfig)
            return sentinel

        _patch_entry_points(monkeypatch, "headroom.memory_store", "myvec", make_store)

        config = MemoryConfig(
            store_backend=StoreBackend.EXTERNAL,
            store_backend_name="myvec",
        )
        assert _create_store(config) is sentinel

    def test_external_without_name_raises(self):
        config = MemoryConfig(store_backend=StoreBackend.EXTERNAL)
        with pytest.raises(ValueError, match="store_backend_name is required"):
            _create_store(config)

    def test_external_unknown_name_raises(self, monkeypatch):
        from headroom.memory import factory as factory_mod

        monkeypatch.setattr(factory_mod, "entry_points", lambda *, group: [])

        config = MemoryConfig(
            store_backend=StoreBackend.EXTERNAL,
            store_backend_name="nonexistent",
        )
        with pytest.raises(ValueError, match="No entry point .* 'nonexistent'"):
            _create_store(config)


class TestExternalVectorBackend:
    """EXTERNAL vector backend loads via entry_points(group='headroom.memory_vector')."""

    def test_loads_external_vector(self, monkeypatch):
        sentinel = object()
        _patch_entry_points(monkeypatch, "headroom.memory_vector", "myvec", lambda cfg: sentinel)

        config = MemoryConfig(
            vector_backend=VectorBackend.EXTERNAL,
            vector_backend_name="myvec",
        )
        assert _create_vector_index(config) is sentinel

    def test_external_without_name_raises(self):
        config = MemoryConfig(vector_backend=VectorBackend.EXTERNAL)
        with pytest.raises(ValueError, match="vector_backend_name is required"):
            _create_vector_index(config)


class TestExternalTextBackend:
    """EXTERNAL text backend loads via entry_points(group='headroom.memory_text')."""

    def test_loads_external_text(self, monkeypatch):
        sentinel = object()
        _patch_entry_points(monkeypatch, "headroom.memory_text", "mytext", lambda cfg: sentinel)

        config = MemoryConfig(
            text_backend=TextBackend.EXTERNAL,
            text_backend_name="mytext",
        )
        assert _create_text_index(config) is sentinel

    def test_external_without_name_raises(self):
        config = MemoryConfig(text_backend=TextBackend.EXTERNAL)
        with pytest.raises(ValueError, match="text_backend_name is required"):
            _create_text_index(config)
