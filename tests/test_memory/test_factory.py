"""Tests for memory factory module.

Tests component creation via the factory functions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from headroom.memory.config import (
    EmbedderBackend,
    MemoryConfig,
    StoreBackend,
    TextBackend,
    VectorBackend,
)
from headroom.memory.factory import (
    _create_cache,
    _create_embedder,
    _create_store,
    _create_text_index,
    _create_vector_index,
    create_memory_system,
)

# Check if hnswlib is available (most factory tests require it)
try:
    from headroom.memory.adapters.hnsw import _check_hnswlib_available

    HNSW_AVAILABLE = _check_hnswlib_available()
except ImportError:
    HNSW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HNSW_AVAILABLE, reason="hnswlib not available")

# =============================================================================
# Test MemoryConfig
# =============================================================================


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = MemoryConfig()

        assert config.store_backend == StoreBackend.SQLITE
        assert config.vector_backend == VectorBackend.AUTO
        assert config.text_backend == TextBackend.FTS5
        assert config.embedder_backend == EmbedderBackend.LOCAL
        assert config.cache_enabled is True
        assert config.cache_max_size == 1000
        assert config.auto_bubble is True
        assert config.bubble_threshold == 0.7

    def test_config_with_custom_path(self):
        """Config should accept custom db path."""
        config = MemoryConfig(db_path=Path("/custom/path.db"))
        assert config.db_path == Path("/custom/path.db")

    def test_config_converts_string_path(self):
        """String paths should be converted to Path objects."""
        config = MemoryConfig(db_path="/custom/path.db")  # type: ignore[arg-type]
        assert isinstance(config.db_path, Path)
        assert config.db_path == Path("/custom/path.db")

    def test_config_validates_vector_dimension(self):
        """Vector dimension must be positive."""
        with pytest.raises(ValueError, match="vector_dimension must be positive"):
            MemoryConfig(vector_dimension=0)

        with pytest.raises(ValueError, match="vector_dimension must be positive"):
            MemoryConfig(vector_dimension=-1)

    def test_config_validates_hnsw_params(self):
        """HNSW parameters must be positive."""
        with pytest.raises(ValueError, match="hnsw_ef_construction must be positive"):
            MemoryConfig(hnsw_ef_construction=0)

        with pytest.raises(ValueError, match="hnsw_m must be positive"):
            MemoryConfig(hnsw_m=0)

        with pytest.raises(ValueError, match="hnsw_ef_search must be positive"):
            MemoryConfig(hnsw_ef_search=0)

    def test_config_validates_cache_size(self):
        """Cache max size must be positive."""
        with pytest.raises(ValueError, match="cache_max_size must be positive"):
            MemoryConfig(cache_max_size=0)

    def test_config_requires_openai_key_for_openai_backend(self):
        """OpenAI backend requires API key."""
        with pytest.raises(ValueError, match="openai_api_key is required"):
            MemoryConfig(embedder_backend=EmbedderBackend.OPENAI)

    def test_config_accepts_openai_key(self):
        """Config should accept OpenAI API key."""
        config = MemoryConfig(embedder_backend=EmbedderBackend.OPENAI, openai_api_key="sk-test-key")
        assert config.openai_api_key == "sk-test-key"

    def test_config_custom_hnsw_params(self):
        """Config should accept custom HNSW parameters."""
        config = MemoryConfig(
            vector_dimension=768, hnsw_ef_construction=400, hnsw_m=32, hnsw_ef_search=100
        )
        assert config.vector_dimension == 768
        assert config.hnsw_ef_construction == 400
        assert config.hnsw_m == 32
        assert config.hnsw_ef_search == 100

    def test_config_cache_disabled(self):
        """Config should allow disabling cache."""
        config = MemoryConfig(cache_enabled=False)
        assert config.cache_enabled is False

    def test_config_ollama_base_url(self):
        """Config should accept Ollama base URL."""
        config = MemoryConfig(
            embedder_backend=EmbedderBackend.OLLAMA, ollama_base_url="http://remote:11434"
        )
        assert config.ollama_base_url == "http://remote:11434"


# =============================================================================
# Test _create_store
# =============================================================================


class TestCreateStore:
    """Tests for _create_store factory function."""

    def test_creates_sqlite_store(self):
        """Should create SQLiteMemoryStore for SQLITE backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(
                store_backend=StoreBackend.SQLITE, db_path=Path(tmpdir) / "test.db"
            )
            store = _create_store(config)

            from headroom.memory.adapters.sqlite import SQLiteMemoryStore

            assert isinstance(store, SQLiteMemoryStore)

    def test_sqlite_store_with_custom_path(self):
        """SQLite store should use config db_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / "custom_memory.db"
            config = MemoryConfig(db_path=custom_path)
            store = _create_store(config)

            from headroom.memory.adapters.sqlite import SQLiteMemoryStore

            assert isinstance(store, SQLiteMemoryStore)

    def test_unknown_store_backend_raises(self):
        """Unknown store backend should raise ValueError."""
        config = MemoryConfig()
        # Manually set an invalid backend to bypass __post_init__
        config.store_backend = "invalid"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unknown store backend"):
            _create_store(config)


# =============================================================================
# Test _create_embedder
# =============================================================================


class TestCreateEmbedder:
    """Tests for _create_embedder factory function."""

    def test_creates_local_embedder(self):
        """Should create LocalEmbedder for LOCAL backend."""
        config = MemoryConfig(embedder_backend=EmbedderBackend.LOCAL)
        embedder = _create_embedder(config)

        from headroom.memory.adapters.embedders import LocalEmbedder

        assert isinstance(embedder, LocalEmbedder)

    def test_local_embedder_with_custom_model(self):
        """LocalEmbedder should use config embedder_model."""
        config = MemoryConfig(
            embedder_backend=EmbedderBackend.LOCAL, embedder_model="paraphrase-MiniLM-L6-v2"
        )
        embedder = _create_embedder(config)

        from headroom.memory.adapters.embedders import LocalEmbedder

        assert isinstance(embedder, LocalEmbedder)
        assert embedder.model_name == "paraphrase-MiniLM-L6-v2"

    def test_creates_openai_embedder(self):
        """Should create OpenAIEmbedder for OPENAI backend."""
        config = MemoryConfig(embedder_backend=EmbedderBackend.OPENAI, openai_api_key="sk-test-key")
        embedder = _create_embedder(config)

        from headroom.memory.adapters.embedders import OpenAIEmbedder

        assert isinstance(embedder, OpenAIEmbedder)

    def test_openai_embedder_requires_key(self):
        """OpenAI embedder should raise if no API key."""
        config = MemoryConfig(embedder_backend=EmbedderBackend.LOCAL)
        # Manually override to test the factory check
        config.embedder_backend = EmbedderBackend.OPENAI
        config.openai_api_key = None

        with pytest.raises(ValueError, match="openai_api_key is required"):
            _create_embedder(config)

    def test_creates_ollama_embedder(self):
        """Should create OllamaEmbedder for OLLAMA backend."""
        config = MemoryConfig(embedder_backend=EmbedderBackend.OLLAMA)
        embedder = _create_embedder(config)

        from headroom.memory.adapters.embedders import OllamaEmbedder

        assert isinstance(embedder, OllamaEmbedder)

    def test_ollama_embedder_with_custom_url(self):
        """OllamaEmbedder should use config base URL and model."""
        config = MemoryConfig(
            embedder_backend=EmbedderBackend.OLLAMA,
            ollama_base_url="http://custom:11434",
            embedder_model="custom-model",
        )
        embedder = _create_embedder(config)

        from headroom.memory.adapters.embedders import OllamaEmbedder

        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.model_name == "custom-model"

    def test_unknown_embedder_backend_raises(self):
        """Unknown embedder backend should raise ValueError."""
        config = MemoryConfig()
        # Manually set an invalid backend
        config.embedder_backend = "invalid"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unknown embedder backend"):
            _create_embedder(config)


# =============================================================================
# Test _create_vector_index
# =============================================================================


class TestCreateVectorIndex:
    """Tests for _create_vector_index factory function."""

    def test_creates_hnsw_index(self):
        """Should create HNSWVectorIndex for HNSW backend."""
        config = MemoryConfig(vector_backend=VectorBackend.HNSW)
        index = _create_vector_index(config)

        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        assert isinstance(index, HNSWVectorIndex)

    def test_hnsw_index_with_custom_params(self):
        """HNSW index should use config parameters."""
        config = MemoryConfig(
            vector_backend=VectorBackend.HNSW,
            vector_dimension=768,
            hnsw_ef_construction=400,
            hnsw_m=32,
            hnsw_ef_search=100,
        )
        index = _create_vector_index(config)

        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        assert isinstance(index, HNSWVectorIndex)
        assert index._dimension == 768
        assert index._ef_construction == 400
        assert index._m == 32
        assert index._ef_search == 100

    def test_unknown_vector_backend_raises(self):
        """Unknown vector backend should raise ValueError."""
        config = MemoryConfig()
        # Manually set an invalid backend
        config.vector_backend = "invalid"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unknown vector backend"):
            _create_vector_index(config)


# =============================================================================
# Test _create_text_index
# =============================================================================


class TestCreateTextIndex:
    """Tests for _create_text_index factory function."""

    def test_creates_fts5_index(self):
        """Should create FTS5TextIndex for FTS5 backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(text_backend=TextBackend.FTS5, db_path=Path(tmpdir) / "test.db")
            index = _create_text_index(config)

            from headroom.memory.adapters.fts5 import FTS5TextIndex

            assert isinstance(index, FTS5TextIndex)

    def test_fts5_index_uses_db_path(self):
        """FTS5 index should use config db_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / "custom_memory.db"
            config = MemoryConfig(db_path=custom_path)
            index = _create_text_index(config)

            from headroom.memory.adapters.fts5 import FTS5TextIndex

            assert isinstance(index, FTS5TextIndex)

    def test_unknown_text_backend_raises(self):
        """Unknown text backend should raise ValueError."""
        config = MemoryConfig()
        # Manually set an invalid backend
        config.text_backend = "invalid"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unknown text backend"):
            _create_text_index(config)


# =============================================================================
# Test _create_cache
# =============================================================================


class TestCreateCache:
    """Tests for _create_cache factory function."""

    def test_creates_lru_cache(self):
        """Should create LRUMemoryCache."""
        config = MemoryConfig(cache_max_size=500)
        cache = _create_cache(config)

        from headroom.memory.adapters.cache import LRUMemoryCache

        assert isinstance(cache, LRUMemoryCache)
        assert cache.max_size == 500

    def test_cache_with_default_size(self):
        """Cache should use default size from config."""
        config = MemoryConfig()  # Default cache_max_size=1000
        cache = _create_cache(config)

        from headroom.memory.adapters.cache import LRUMemoryCache

        assert isinstance(cache, LRUMemoryCache)
        assert cache.max_size == 1000

    def test_cache_with_custom_size(self):
        """Cache should use custom max_size from config."""
        config = MemoryConfig(cache_max_size=5000)
        cache = _create_cache(config)

        assert cache.max_size == 5000


# =============================================================================
# Test create_memory_system
# =============================================================================


class TestCreateMemorySystem:
    """Tests for create_memory_system factory function."""

    @pytest.mark.asyncio
    async def test_creates_all_components_with_default_config(self):
        """Should create all components with default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(db_path=Path(tmpdir) / "test.db")
            store, vector, text, embedder, cache = await create_memory_system(config)

            from headroom.memory.adapters.cache import LRUMemoryCache
            from headroom.memory.adapters.embedders import LocalEmbedder
            from headroom.memory.adapters.fts5 import FTS5TextIndex
            from headroom.memory.adapters.hnsw import HNSWVectorIndex
            from headroom.memory.adapters.sqlite import SQLiteMemoryStore
            from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

            assert isinstance(store, SQLiteMemoryStore)
            # Factory auto-selects best available: SQLiteVectorIndex (preferred) or HNSW (fallback)
            assert isinstance(vector, (SQLiteVectorIndex, HNSWVectorIndex))
            assert isinstance(text, FTS5TextIndex)
            assert isinstance(embedder, LocalEmbedder)
            assert isinstance(cache, LRUMemoryCache)

    @pytest.mark.asyncio
    async def test_creates_components_without_cache(self):
        """Should return None for cache when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(db_path=Path(tmpdir) / "test.db", cache_enabled=False)
            store, vector, text, embedder, cache = await create_memory_system(config)

            assert cache is None

    @pytest.mark.asyncio
    async def test_creates_components_with_none_config(self):
        """Should use default config when None is passed."""
        # This will use default path, which may create file in current dir
        # Just verify it doesn't raise
        store, vector, text, embedder, cache = await create_memory_system(None)

        from headroom.memory.adapters.cache import LRUMemoryCache
        from headroom.memory.adapters.embedders import LocalEmbedder
        from headroom.memory.adapters.fts5 import FTS5TextIndex
        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.adapters.sqlite import SQLiteMemoryStore
        from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

        assert isinstance(store, SQLiteMemoryStore)
        # Factory auto-selects best available: SQLiteVectorIndex (preferred) or HNSW (fallback)
        assert isinstance(vector, (SQLiteVectorIndex, HNSWVectorIndex))
        assert isinstance(text, FTS5TextIndex)
        assert isinstance(embedder, LocalEmbedder)
        assert isinstance(cache, LRUMemoryCache)

    @pytest.mark.asyncio
    async def test_creates_ollama_embedder(self):
        """Should create OllamaEmbedder when specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(
                db_path=Path(tmpdir) / "test.db", embedder_backend=EmbedderBackend.OLLAMA
            )
            store, vector, text, embedder, cache = await create_memory_system(config)

            from headroom.memory.adapters.embedders import OllamaEmbedder

            assert isinstance(embedder, OllamaEmbedder)

    @pytest.mark.asyncio
    async def test_creates_openai_embedder(self):
        """Should create OpenAIEmbedder when specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(
                db_path=Path(tmpdir) / "test.db",
                embedder_backend=EmbedderBackend.OPENAI,
                openai_api_key="sk-test-key",
            )
            store, vector, text, embedder, cache = await create_memory_system(config)

            from headroom.memory.adapters.embedders import OpenAIEmbedder

            assert isinstance(embedder, OpenAIEmbedder)

    @pytest.mark.asyncio
    async def test_custom_hnsw_params_propagate(self):
        """Custom HNSW params should propagate to vector index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(
                db_path=Path(tmpdir) / "test.db",
                vector_backend=VectorBackend.HNSW,  # Explicitly use HNSW to test params
                vector_dimension=512,
                hnsw_ef_construction=300,
                hnsw_m=24,
                hnsw_ef_search=75,
            )
            store, vector, text, embedder, cache = await create_memory_system(config)

            assert vector._dimension == 512
            assert vector._ef_construction == 300
            assert vector._m == 24
            assert vector._ef_search == 75

    @pytest.mark.asyncio
    async def test_returns_tuple_of_five(self):
        """Should return exactly 5 components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(db_path=Path(tmpdir) / "test.db")
            result = await create_memory_system(config)

            assert isinstance(result, tuple)
            assert len(result) == 5


# =============================================================================
# Test Backend Enums
# =============================================================================


class TestBackendEnums:
    """Tests for backend enum values."""

    def test_store_backend_values(self):
        """StoreBackend should have expected values."""
        assert StoreBackend.SQLITE.value == "sqlite"

    def test_vector_backend_values(self):
        """VectorBackend should have expected values."""
        assert VectorBackend.HNSW.value == "hnsw"

    def test_text_backend_values(self):
        """TextBackend should have expected values."""
        assert TextBackend.FTS5.value == "fts5"

    def test_embedder_backend_values(self):
        """EmbedderBackend should have expected values."""
        assert EmbedderBackend.LOCAL.value == "local"
        assert EmbedderBackend.OPENAI.value == "openai"
        assert EmbedderBackend.OLLAMA.value == "ollama"


# =============================================================================
# Integration Tests
# =============================================================================


class TestFactoryIntegration:
    """Integration tests for factory module."""

    @pytest.mark.asyncio
    async def test_created_components_are_usable(self):
        """Created components should be functional."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MemoryConfig(db_path=Path(tmpdir) / "test.db")
            store, vector, text, embedder, cache = await create_memory_system(config)

            # Test store has expected interface (SQLiteMemoryStore uses 'save' not 'add')
            assert hasattr(store, "save")
            assert hasattr(store, "get")
            assert hasattr(store, "delete")
            assert hasattr(store, "query")

            # Test vector index has expected interface (HNSWVectorIndex uses 'index')
            assert hasattr(vector, "index")
            assert hasattr(vector, "index_batch")
            assert hasattr(vector, "search")
            assert hasattr(vector, "remove")

            # Test text index has expected interface (FTS5TextIndex uses 'index')
            assert hasattr(text, "index")
            assert hasattr(text, "index_batch")
            assert hasattr(text, "search")

            # Test embedder has expected interface
            assert hasattr(embedder, "embed")
            assert hasattr(embedder, "embed_batch")
            assert hasattr(embedder, "dimension")

            # Test cache has expected interface
            assert cache is not None
            assert hasattr(cache, "get")
            assert hasattr(cache, "put")
            assert hasattr(cache, "invalidate")

    @pytest.mark.asyncio
    async def test_embedder_dimension_property(self):
        """Created embedder should report correct dimension."""
        config = MemoryConfig(embedder_backend=EmbedderBackend.LOCAL)
        embedder = _create_embedder(config)

        # LocalEmbedder reports dimension before model is loaded
        assert embedder.dimension == 384  # all-MiniLM-L6-v2 default

    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Created cache should support basic operations."""
        config = MemoryConfig(cache_max_size=10)
        cache = _create_cache(config)

        # Cache should be empty initially
        assert cache.size == 0
        assert cache.max_size == 10

        # Test stats
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["utilization"] == 0.0

    def test_config_with_all_custom_options(self):
        """Config should accept all custom options together."""
        config = MemoryConfig(
            store_backend=StoreBackend.SQLITE,
            db_path=Path("/custom/path.db"),
            vector_backend=VectorBackend.HNSW,
            vector_dimension=1024,
            hnsw_ef_construction=500,
            hnsw_m=48,
            hnsw_ef_search=200,
            text_backend=TextBackend.FTS5,
            embedder_backend=EmbedderBackend.OLLAMA,
            embedder_model="custom-model",
            ollama_base_url="http://custom:11434",
            cache_enabled=True,
            cache_max_size=5000,
            auto_bubble=False,
            bubble_threshold=0.5,
        )

        assert config.store_backend == StoreBackend.SQLITE
        assert config.db_path == Path("/custom/path.db")
        assert config.vector_dimension == 1024
        assert config.hnsw_ef_construction == 500
        assert config.hnsw_m == 48
        assert config.hnsw_ef_search == 200
        assert config.embedder_backend == EmbedderBackend.OLLAMA
        assert config.embedder_model == "custom-model"
        assert config.ollama_base_url == "http://custom:11434"
        assert config.cache_enabled is True
        assert config.cache_max_size == 5000
        assert config.auto_bubble is False
        assert config.bubble_threshold == 0.5
