"""The embedder cache must not serve an embedder bound to the wrong server.

Kept out of ``test_factory.py`` (which skips wholesale without hnswlib) because
these cases only construct the lightweight Ollama embedder and need no vector
index.
"""

from __future__ import annotations

from headroom.memory.config import EmbedderBackend, MemoryConfig
from headroom.memory.factory import _create_embedder, _reset_embedder_cache_for_tests


def test_ollama_embedder_cache_keys_on_base_url():
    """Two configs that share backend + model but differ in ollama_base_url must
    not share a cached embedder — the second would otherwise get an embedder
    bound to the first server."""
    _reset_embedder_cache_for_tests()
    try:
        cfg1 = MemoryConfig(
            embedder_backend=EmbedderBackend.OLLAMA,
            embedder_model="nomic-embed-text",
            ollama_base_url="http://gpu1:11434",
        )
        cfg2 = MemoryConfig(
            embedder_backend=EmbedderBackend.OLLAMA,
            embedder_model="nomic-embed-text",
            ollama_base_url="http://gpu2:11434",
        )

        e1 = _create_embedder(cfg1)
        e2 = _create_embedder(cfg2)

        assert e1 is not e2
        assert e1._base_url == "http://gpu1:11434"
        assert e2._base_url == "http://gpu2:11434"
    finally:
        _reset_embedder_cache_for_tests()


def test_ollama_embedder_cache_reuses_same_base_url():
    """Same backend + model + base_url still hits the cache (one model load)."""
    _reset_embedder_cache_for_tests()
    try:
        cfg = MemoryConfig(
            embedder_backend=EmbedderBackend.OLLAMA,
            embedder_model="nomic-embed-text",
            ollama_base_url="http://gpu1:11434",
        )
        assert _create_embedder(cfg) is _create_embedder(cfg)
    finally:
        _reset_embedder_cache_for_tests()
