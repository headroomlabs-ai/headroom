"""Tests for the cognee memory backend and its env-var resolution.

Covers:
- ``cognee_env`` readers (dataset/system_root/data_root/search_type/auto_cognify)
- ``CogneeConfig`` defaults and env pickup via ``field(default_factory=...)``
- ``save_memory`` mapping to ``cognee.add`` (node_set tags, facts) and the
  ``cognee.cognify`` trigger per config
- ``search_memories`` mapping to ``cognee.search`` (node_name scoping, top_k,
  GRAPH_COMPLETION only_context) and result -> MemorySearchResult conversion
- Durable tombstone/registry semantics for update/delete (restart + multiple
  live instances sharing one metadata DB)
- Process-wide immutable cognee configuration (conflicting roots fail closed)
- ImportError guard message when cognee is not installed
- ``supports_graph`` / ``supports_vector_search`` capability flags

cognee is NOT a test dependency: a fake module is injected into
``sys.modules`` before the backend's lazy import runs.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import types
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest

from headroom.memory import cognee_env
from headroom.memory.backends import cognee as cognee_backend_module
from headroom.memory.backends.cognee import (
    CogneeBackend,
    CogneeConfig,
    CogneeDeletionUnverifiedError,
    _resolve_metadata_db_path,
)
from headroom.memory.cognee_env import (
    DEFAULT_COGNEE_DATASET,
    DEFAULT_COGNEE_SEARCH_TYPE,
    cognee_env_auto_cognify,
    cognee_env_data_root,
    cognee_env_dataset,
    cognee_env_metadata_db,
    cognee_env_search_type,
    cognee_env_system_root,
)
from headroom.memory.ports import MemorySearchResult

# All HEADROOM_COGNEE_* vars are cleared before every test so the host
# environment cannot leak into unit tests.
_COGNEE_ENV_VARS = (
    "HEADROOM_COGNEE_DATASET",
    "HEADROOM_COGNEE_SYSTEM_ROOT",
    "HEADROOM_COGNEE_DATA_ROOT",
    "HEADROOM_COGNEE_SEARCH_TYPE",
    "HEADROOM_COGNEE_AUTO_COGNIFY",
    "HEADROOM_COGNEE_METADATA_DB",
)


@pytest.fixture(autouse=True)
def _cognee_test_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Any:
    """Isolate every test: clean env, tmp metadata DB, fresh process state.

    The metadata DB env var is pointed at a per-test tmp file so no test can
    write to ``~/.headroom``. The module-level cognee import/config state is
    reset before and after each test (cognee configuration is process-wide
    and immutable; tests simulate fresh processes).
    """
    for var in _COGNEE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HEADROOM_COGNEE_METADATA_DB", str(tmp_path / "cognee_meta.db"))
    cognee_backend_module._reset_process_state_for_testing()
    yield
    cognee_backend_module._reset_process_state_for_testing()


# =============================================================================
# Fake cognee module
# =============================================================================


class _FakeSearchType(str, Enum):
    """Stand-in for cognee.SearchType (a str enum in the real package)."""

    CHUNKS = "CHUNKS"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    SUMMARIES = "SUMMARIES"


def _make_fake_cognee(
    search_results: list[Any] | None = None,
    cognify_error: Exception | None = None,
) -> tuple[types.ModuleType, SimpleNamespace]:
    """Build a fake ``cognee`` module plus a call recorder."""
    calls = SimpleNamespace(add=[], cognify=[], search=[], system_root=[], data_root=[])
    module = types.ModuleType("cognee")

    async def add(data, dataset_name=None, node_set=None, **kwargs):
        calls.add.append({"data": data, "dataset_name": dataset_name, "node_set": node_set})

    async def cognify(datasets=None, run_in_background=None, **kwargs):
        calls.cognify.append({"datasets": datasets, "run_in_background": run_in_background})
        if cognify_error is not None:
            raise cognify_error

    async def search(**kwargs):
        calls.search.append(kwargs)
        return list(search_results or [])

    module.SearchType = _FakeSearchType
    module.add = add
    module.cognify = cognify
    module.search = search
    module.config = SimpleNamespace(
        system_root_directory=calls.system_root.append,
        data_root_directory=calls.data_root.append,
    )
    return module, calls


def _fake_hit(payload: Any, dataset_name: str = "headroom_memories") -> SimpleNamespace:
    """Build a fake cognee SearchResult (.search_result/.dataset_id/.dataset_name)."""
    return SimpleNamespace(search_result=payload, dataset_id="ds-1", dataset_name=dataset_name)


@pytest.fixture
def fake_cognee(monkeypatch: pytest.MonkeyPatch):
    """Inject a default fake cognee module; returns its call recorder.

    Tests that need custom search results or a failing cognify build their
    own module via ``_make_fake_cognee`` and ``monkeypatch.setitem``.
    """
    module, calls = _make_fake_cognee()
    monkeypatch.setitem(sys.modules, "cognee", module)
    return calls


# =============================================================================
# cognee_env readers
# =============================================================================


class TestCogneeEnvReaders:
    def test_dataset_defaults_when_unset(self) -> None:
        assert cognee_env_dataset() == DEFAULT_COGNEE_DATASET

    def test_dataset_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "my_dataset")
        assert cognee_env_dataset() == "my_dataset"

    def test_dataset_trims_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "  padded  ")
        assert cognee_env_dataset() == "padded"

    def test_dataset_empty_string_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "")
        assert cognee_env_dataset() == DEFAULT_COGNEE_DATASET

    def test_system_root_none_when_unset(self) -> None:
        assert cognee_env_system_root() is None

    def test_system_root_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_SYSTEM_ROOT", "/var/cognee/system")
        assert cognee_env_system_root() == "/var/cognee/system"

    def test_data_root_none_when_unset(self) -> None:
        assert cognee_env_data_root() is None

    def test_data_root_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATA_ROOT", "/var/cognee/data")
        assert cognee_env_data_root() == "/var/cognee/data"

    def test_search_type_defaults_when_unset(self) -> None:
        assert cognee_env_search_type() == DEFAULT_COGNEE_SEARCH_TYPE == "CHUNKS"

    def test_search_type_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_SEARCH_TYPE", "GRAPH_COMPLETION")
        assert cognee_env_search_type() == "GRAPH_COMPLETION"

    def test_auto_cognify_defaults_true(self) -> None:
        assert cognee_env_auto_cognify() is True

    @pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "y", "on"])
    def test_auto_cognify_truthy_values(self, monkeypatch: pytest.MonkeyPatch, truthy: str) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", truthy)
        assert cognee_env_auto_cognify() is True

    @pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "no", "n", "off"])
    def test_auto_cognify_falsy_values(self, monkeypatch: pytest.MonkeyPatch, falsy: str) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", falsy)
        assert cognee_env_auto_cognify() is False

    def test_auto_cognify_rejects_garbage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", "maybe")
        with pytest.raises(ValueError, match="Invalid boolean value"):
            cognee_env_auto_cognify()

    def test_metadata_db_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_COGNEE_METADATA_DB", raising=False)
        assert cognee_env_metadata_db() is None

    def test_metadata_db_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_METADATA_DB", "/var/cognee/meta.db")
        assert cognee_env_metadata_db() == "/var/cognee/meta.db"

    def test_module_exports_readers(self) -> None:
        assert callable(cognee_env.cognee_env_dataset)
        assert callable(cognee_env.cognee_env_system_root)
        assert callable(cognee_env.cognee_env_data_root)
        assert callable(cognee_env.cognee_env_search_type)
        assert callable(cognee_env.cognee_env_auto_cognify)
        assert callable(cognee_env.cognee_env_metadata_db)


# =============================================================================
# CogneeConfig
# =============================================================================


class TestCogneeConfig:
    def test_defaults_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_COGNEE_METADATA_DB", raising=False)
        cfg = CogneeConfig()
        assert cfg.dataset_name == "headroom_memories"
        assert cfg.system_root is None
        assert cfg.data_root is None
        assert cfg.search_type == "CHUNKS"
        assert cfg.auto_cognify is True
        assert cfg.background_cognify is True
        assert cfg.metadata_db_path is None

    def test_defaults_read_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "env_dataset")
        monkeypatch.setenv("HEADROOM_COGNEE_SYSTEM_ROOT", "/sys/root")
        monkeypatch.setenv("HEADROOM_COGNEE_DATA_ROOT", "/data/root")
        monkeypatch.setenv("HEADROOM_COGNEE_SEARCH_TYPE", "GRAPH_COMPLETION")
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", "false")
        monkeypatch.setenv("HEADROOM_COGNEE_METADATA_DB", "/var/cognee/meta.db")

        cfg = CogneeConfig()
        assert cfg.dataset_name == "env_dataset"
        assert cfg.system_root == "/sys/root"
        assert cfg.data_root == "/data/root"
        assert cfg.search_type == "GRAPH_COMPLETION"
        assert cfg.auto_cognify is False
        assert cfg.metadata_db_path == "/var/cognee/meta.db"

    def test_explicit_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "env_dataset")
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", "false")

        cfg = CogneeConfig(dataset_name="explicit_dataset", auto_cognify=True)
        assert cfg.dataset_name == "explicit_dataset"
        assert cfg.auto_cognify is True


class TestMetadataDbPathResolution:
    """Default location of the durable metadata DB (see CogneeConfig docs)."""

    def test_explicit_path_wins(self) -> None:
        cfg = CogneeConfig(
            metadata_db_path="/explicit/meta.db", data_root="/data", system_root="/sys"
        )
        assert str(_resolve_metadata_db_path(cfg)) == "/explicit/meta.db"

    def test_defaults_under_data_root(self) -> None:
        cfg = CogneeConfig(metadata_db_path=None, data_root="/data", system_root="/sys")
        assert str(_resolve_metadata_db_path(cfg)) == "/data/headroom_cognee_meta.db"

    def test_falls_back_to_system_root(self) -> None:
        cfg = CogneeConfig(metadata_db_path=None, data_root=None, system_root="/sys")
        assert str(_resolve_metadata_db_path(cfg)) == "/sys/headroom_cognee_meta.db"

    def test_falls_back_to_workspace_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
        cfg = CogneeConfig(metadata_db_path=None)
        assert _resolve_metadata_db_path(cfg) == tmp_path / "ws" / "headroom_cognee_meta.db"


# =============================================================================
# Lazy import / ImportError guard
# =============================================================================


class TestImportGuard:
    def test_construction_does_not_import_cognee(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Poison the import: construction must still succeed (lazy import).
        monkeypatch.setitem(sys.modules, "cognee", None)
        backend = CogneeBackend(CogneeConfig())
        assert backend is not None

    async def test_save_raises_install_hint_when_cognee_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "cognee", None)
        backend = CogneeBackend(CogneeConfig())
        with pytest.raises(ImportError, match=r'pip install "headroom-ai\[cognee\]"'):
            await backend.save_memory(content="x", user_id="alice", importance=0.5)

    async def test_search_raises_install_hint_when_cognee_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "cognee", None)
        backend = CogneeBackend(CogneeConfig())
        with pytest.raises(ImportError, match=r'pip install "headroom-ai\[cognee\]"'):
            await backend.search_memories(query="x", user_id="alice")


# =============================================================================
# Initialization (directory isolation)
# =============================================================================


class TestInitialization:
    async def test_applies_root_directories_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, calls = _make_fake_cognee()
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(system_root="/sys/x", data_root="/data/y"))
        await backend.ensure_initialized()

        assert calls.system_root == ["/sys/x"]
        assert calls.data_root == ["/data/y"]

    async def test_skips_root_directories_when_unset(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        await backend.ensure_initialized()

        assert fake_cognee.system_root == []
        assert fake_cognee.data_root == []

    async def test_close_resets_and_allows_reinit(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        await backend.ensure_initialized()
        await backend.close()
        assert backend._initialized is False

        # Reusable after close.
        await backend.save_memory(content="again", user_id="alice", importance=0.5)
        assert len(fake_cognee.add) == 1


# =============================================================================
# save_memory
# =============================================================================


class TestSaveMemory:
    async def test_maps_content_and_scoping_to_add_node_set(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        memory = await backend.save_memory(
            content="Alice prefers Python",
            user_id="alice",
            importance=0.8,
            entities=["python", "ml"],
            session_id="s1",
        )

        assert len(fake_cognee.add) == 1
        call = fake_cognee.add[0]
        assert call["data"] == "Alice prefers Python"
        assert call["dataset_name"] == "ds"
        assert call["node_set"] == ["user:alice", "session:s1", "entity:python", "entity:ml"]

        assert memory.content == "Alice prefers Python"
        assert memory.user_id == "alice"
        assert memory.session_id == "s1"
        assert memory.importance == 0.8
        assert memory.entity_refs == ["python", "ml"]
        assert memory.id

    async def test_node_set_omits_session_and_entities_when_absent(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        await backend.save_memory(content="c", user_id="bob", importance=0.5)
        assert fake_cognee.add[0]["node_set"] == ["user:bob"]

    async def test_facts_are_added_alongside_content(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(
            content="Alice works at Netflix using Python",
            user_id="alice",
            importance=0.5,
            facts=["Alice works at Netflix", "Alice uses Python"],
        )
        assert fake_cognee.add[0]["data"] == [
            "Alice works at Netflix using Python",
            "Alice works at Netflix",
            "Alice uses Python",
        ]
        assert memory.metadata["_fact_count"] == 2

    async def test_triggers_cognify_by_default_in_background(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        await backend.save_memory(content="c", user_id="alice", importance=0.5)

        assert fake_cognee.cognify == [{"datasets": ["ds"], "run_in_background": True}]

    async def test_no_cognify_when_auto_cognify_disabled(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig(auto_cognify=False))
        await backend.save_memory(content="c", user_id="alice", importance=0.5)
        assert fake_cognee.cognify == []

    async def test_foreground_cognify_when_background_disabled(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig(background_cognify=False))
        await backend.save_memory(content="c", user_id="alice", importance=0.5)
        assert fake_cognee.cognify[0]["run_in_background"] is False

    async def test_cognify_failure_does_not_lose_save(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, calls = _make_fake_cognee(cognify_error=RuntimeError("LLM down"))
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="kept", user_id="alice", importance=0.5)

        assert len(calls.add) == 1
        assert memory.content == "kept"
        fetched = await backend.get_memory(memory.id)
        assert fetched is not None
        assert fetched.id == memory.id
        assert fetched.content == "kept"

    async def test_relationships_and_extractions_recorded_in_metadata(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        memory = await backend.save_memory(
            content="Alice works at Netflix",
            user_id="alice",
            importance=0.5,
            relationships=[{"source": "Alice", "relationship": "works_at", "target": "Netflix"}],
            extracted_entities=[{"entity": "Alice", "entity_type": "person"}],
            extracted_relationships=[{"source": "Alice", "target": "Netflix"}],
            metadata={"origin": "test"},
        )
        assert memory.metadata["origin"] == "test"
        assert memory.metadata["_cognee_dataset"] == "ds"
        assert memory.metadata["_cognee_node_set"] == ["user:alice"]
        assert memory.metadata["relationships"] == [
            {"source": "Alice", "relationship": "works_at", "target": "Netflix"}
        ]
        assert memory.metadata["extracted_entities"] == [
            {"entity": "Alice", "entity_type": "person"}
        ]
        assert memory.metadata["extracted_relationships"] == [
            {"source": "Alice", "target": "Netflix"}
        ]


# =============================================================================
# search_memories
# =============================================================================


class TestSearchMemories:
    async def test_passes_query_scoping_and_top_k_to_cognee(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, calls = _make_fake_cognee(search_results=[])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        await backend.search_memories(
            query="python preferences",
            user_id="alice",
            entities=["python"],
            top_k=7,
            session_id="s1",
        )

        assert len(calls.search) == 1
        kwargs = calls.search[0]
        assert kwargs["query_text"] == "python preferences"
        assert kwargs["query_type"] is _FakeSearchType.CHUNKS
        assert kwargs["datasets"] == ["ds"]
        assert kwargs["top_k"] == 7
        assert kwargs["node_name"] == ["user:alice", "session:s1", "entity:python"]
        # AND semantics: every tag must match. cognee's default is OR, which
        # would leak other users' memories sharing a (global) entity tag.
        assert kwargs["node_name_filter_operator"] == "AND"
        assert "only_context" not in kwargs

    async def test_maps_results_to_memory_search_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, _ = _make_fake_cognee(
            search_results=[_fake_hit(["first chunk", "second chunk"], dataset_name="ds")]
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        results = await backend.search_memories(query="q", user_id="alice", session_id="s1")

        assert len(results) == 2
        assert all(isinstance(r, MemorySearchResult) for r in results)
        assert [r.memory.content for r in results] == ["first chunk", "second chunk"]
        assert all(r.memory.user_id == "alice" for r in results)
        assert all(r.memory.session_id == "s1" for r in results)
        assert all(r.memory.metadata["_cognee_dataset_name"] == "ds" for r in results)
        # Rank-based scores, descending, compressed into (0.5, 1.0] so the
        # proxy's default min_similarity floor (0.3) never filters them.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 1.0
        assert all(0.5 < s <= 1.0 for s in scores)

    async def test_unwraps_cognee_15_dict_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cognee >=1.5 returns plain dicts; the payload must be unwrapped, not str()'d."""
        hit = {
            "dataset_id": "ds-1",
            "dataset_name": "ds",
            "dataset_tenant_id": None,
            "search_result": [{"id": "n-1", "type": "IndexSchema", "text": "dict-shaped chunk"}],
        }
        module, _ = _make_fake_cognee(search_results=[hit])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        results = await backend.search_memories(query="q", user_id="alice")

        assert [r.memory.content for r in results] == ["dict-shaped chunk"]
        assert results[0].memory.metadata["_cognee_dataset_name"] == "ds"
        assert results[0].memory.metadata["_cognee_dataset_id"] == "ds-1"

    async def test_dict_results_resolve_saved_canonical_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 1.5-style dict result for saved content returns the saved memory's ID."""
        content = "Vasilije prefers uv over poetry"
        hit = {"dataset_name": "ds", "search_result": [{"text": content}]}
        module, _ = _make_fake_cognee(search_results=[hit])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        saved = await backend.save_memory(content=content, user_id="alice", importance=0.5)
        results = await backend.search_memories(query="q", user_id="alice")
        assert results[0].memory.id == saved.id

    async def test_returns_empty_when_store_never_cognified(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cognee raises NoDataError before any cognify has run; that is an empty result."""

        class NoDataError(Exception):
            pass

        async def search(**kwargs):
            raise NoDataError("No data found in the system, please add data first.")

        module, _ = _make_fake_cognee()
        module.search = search
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        assert await backend.search_memories(query="q", user_id="alice") == []

    async def test_other_search_errors_still_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the empty-store signal is swallowed; real failures propagate."""

        async def search(**kwargs):
            raise RuntimeError("boom")

        module, _ = _make_fake_cognee()
        module.search = search
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        with pytest.raises(RuntimeError, match="boom"):
            await backend.search_memories(query="q", user_id="alice")

    async def test_result_ids_are_stable_and_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Search-result IDs are content-derived, repeatable, and resolvable."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["a fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        first = await backend.search_memories(query="q", user_id="alice")
        second = await backend.search_memories(query="q", user_id="alice")

        assert first[0].memory.id == second[0].memory.id
        registered = await backend.get_memory(first[0].memory.id)
        assert registered is not None
        assert registered.content == first[0].memory.content

        # Same content under a different user gets a different ID.
        other = await backend.search_memories(query="q", user_id="bob")
        assert other[0].memory.id != first[0].memory.id

    async def test_saved_memory_id_matches_search_result_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A saved memory surfaces from search with its original ID."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["remember me"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="remember me", user_id="alice", importance=0.9)
        results = await backend.search_memories(query="q", user_id="alice")

        assert results[0].memory.id == memory.id
        # The registered (saved) memory is reused, keeping its importance.
        assert results[0].memory.importance == 0.9

    async def test_search_result_id_roundtrips_to_update_and_delete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IDs surfaced by search work with update_memory / delete_memory."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["fact one", "fact two"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        results = await backend.search_memories(query="q", user_id="alice")

        updated = await backend.update_memory(results[0].memory.id, "fact one, revised")
        assert updated.content == "fact one, revised"

        assert await backend.delete_memory(results[1].memory.id) is True

    async def test_extracts_text_from_dict_payloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, _ = _make_fake_cognee(
            search_results=[_fake_hit([{"text": "from text key"}, {"chunk": "from chunk key"}])]
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        results = await backend.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in results] == ["from text key", "from chunk key"]

    async def test_truncates_to_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, _ = _make_fake_cognee(search_results=[_fake_hit([f"chunk {i}" for i in range(5)])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        results = await backend.search_memories(query="q", user_id="alice", top_k=2)
        assert len(results) == 2

    async def test_empty_results(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        results = await backend.search_memories(query="q", user_id="alice")
        assert results == []

    async def test_graph_completion_sets_only_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, calls = _make_fake_cognee(search_results=[])
        monkeypatch.setitem(sys.modules, "cognee", module)

        # Lowercase on purpose: resolution is case-insensitive.
        backend = CogneeBackend(CogneeConfig(search_type="graph_completion"))
        await backend.search_memories(query="q", user_id="alice")

        kwargs = calls.search[0]
        assert kwargs["query_type"] is _FakeSearchType.GRAPH_COMPLETION
        assert kwargs["only_context"] is True

    async def test_invalid_search_type_raises(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig(search_type="BOGUS"))
        with pytest.raises(ValueError, match="Invalid cognee search type"):
            await backend.search_memories(query="q", user_id="alice")


# =============================================================================
# update / delete / get (tombstone semantics)
# =============================================================================


class TestUpdateDeleteGet:
    async def test_get_memory_roundtrip(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="c", user_id="alice", importance=0.5)
        fetched = await backend.get_memory(memory.id)
        assert fetched is not None
        assert fetched.id == memory.id
        assert fetched.content == "c"
        assert fetched.user_id == "alice"
        assert fetched.importance == 0.5
        assert await backend.get_memory("nonexistent") is None

    async def test_update_tombstones_and_readds(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        memory = await backend.save_memory(
            content="old fact", user_id="alice", importance=0.7, session_id="s1"
        )

        updated = await backend.update_memory(memory.id, "new fact", reason="correction")

        assert updated.id == memory.id
        assert updated.content == "new fact"
        assert updated.importance == 0.7
        assert updated.metadata["update_reason"] == "correction"
        # Re-added with the same scoping tags.
        assert len(fake_cognee.add) == 2
        assert fake_cognee.add[1]["data"] == "new fact"
        assert fake_cognee.add[1]["node_set"] == fake_cognee.add[0]["node_set"]
        fetched = await backend.get_memory(memory.id)
        assert fetched is not None
        assert fetched.id == memory.id
        assert fetched.content == "new fact"

    async def test_update_unknown_id_raises(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        with pytest.raises(ValueError, match="Memory not found"):
            await backend.update_memory("missing-id", "new content")

    async def test_update_wrong_user_raises(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="c", user_id="alice", importance=0.5)
        with pytest.raises(ValueError, match="other users"):
            await backend.update_memory(memory.id, "new", user_id="bob")

    async def test_delete_tombstones_memory(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="c", user_id="alice", importance=0.5)
        assert await backend.delete_memory(memory.id) is True
        assert await backend.get_memory(memory.id) is None

    async def test_delete_unknown_id_returns_false(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        assert await backend.delete_memory("missing-id") is False

    async def test_delete_wrong_user_returns_false(self, fake_cognee) -> None:
        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="c", user_id="alice", importance=0.5)
        assert await backend.delete_memory(memory.id, user_id="bob") is False
        fetched = await backend.get_memory(memory.id)
        assert fetched is not None
        assert fetched.id == memory.id

    async def test_deleted_content_filtered_from_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["stale fact", "fresh fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="stale fact", user_id="alice", importance=0.5)
        await backend.delete_memory(memory.id)

        results = await backend.search_memories(query="fact", user_id="alice")
        assert [r.memory.content for r in results] == ["fresh fact"]

    async def test_superseded_content_filtered_after_update(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["old fact", "new fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="old fact", user_id="alice", importance=0.5)
        await backend.update_memory(memory.id, "new fact")

        results = await backend.search_memories(query="fact", user_id="alice")
        assert [r.memory.content for r in results] == ["new fact"]

    async def test_tombstones_are_scoped_per_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deleting one user's memory never hides identical content of another user."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["shared fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        alice_memory = await backend.save_memory(
            content="shared fact", user_id="alice", importance=0.5
        )
        await backend.save_memory(content="shared fact", user_id="bob", importance=0.5)
        await backend.delete_memory(alice_memory.id)

        assert await backend.search_memories(query="q", user_id="alice") == []
        bob_results = await backend.search_memories(query="q", user_id="bob")
        assert [r.memory.content for r in bob_results] == ["shared fact"]

    async def test_tombstone_filters_chunks_of_deleted_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunks that are substrings of a deleted memory are filtered too."""
        long_content = "First sentence of a long memory. Second sentence with more detail."
        module, _ = _make_fake_cognee(
            search_results=[_fake_hit(["Second sentence with more detail.", "unrelated fact"])]
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content=long_content, user_id="alice", importance=0.5)
        await backend.delete_memory(memory.id)

        results = await backend.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in results] == ["unrelated fact"]

    async def test_facts_tombstoned_alongside_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-extracted facts (saved as separate items) are tombstoned on delete."""
        module, _ = _make_fake_cognee(
            search_results=[_fake_hit(["Alice works at Netflix", "kept fact"])]
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(
            content="Alice works at Netflix using Python",
            user_id="alice",
            importance=0.5,
            facts=["Alice works at Netflix"],
        )
        await backend.delete_memory(memory.id)

        results = await backend.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in results] == ["kept fact"]

    async def test_survivors_keep_top_ranks_after_tombstone_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ranks (and scores) are computed after tombstone filtering."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["deleted", "survivor"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(content="deleted", user_id="alice", importance=0.5)
        await backend.delete_memory(memory.id)

        results = await backend.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in results] == ["survivor"]
        assert results[0].score == 1.0


# =============================================================================
# Best-effort hard delete via cognee.datasets
# =============================================================================


def _attach_fake_datasets_api(
    module: types.ModuleType,
    dataset_name: str,
    data_items: list[SimpleNamespace],
    delete_error: Exception | None = None,
    fail_once_on: str | None = None,
) -> SimpleNamespace:
    """Attach a fake ``cognee.datasets`` namespace; returns a call recorder.

    ``delete_data`` really removes the item from the live item list (like
    cognee's API), so the backend's non-mutating discovery and post-delete
    verification passes see an honest picture. The live list is exposed as
    ``calls.live_items`` so tests can simulate items (re)appearing.
    ``fail_once_on`` makes ``delete_data`` raise exactly once for that
    data_id, then succeed — an interrupted deletion.
    """
    calls = SimpleNamespace(deleted=[], live_items=list(data_items))
    dataset = SimpleNamespace(name=dataset_name, id="dataset-uuid")
    pending_one_shot: set[str] = {fail_once_on} if fail_once_on else set()

    class _Datasets:
        @staticmethod
        async def list_datasets():
            return [dataset]

        @staticmethod
        async def list_data(dataset_id):
            assert dataset_id == "dataset-uuid"
            return list(calls.live_items)

        @staticmethod
        async def delete_data(dataset_id=None, data_id=None):
            if delete_error is not None:
                raise delete_error
            if data_id in pending_one_shot:
                pending_one_shot.discard(data_id)
                raise RuntimeError(f"transient delete failure for {data_id}")
            calls.deleted.append({"dataset_id": dataset_id, "data_id": data_id})
            calls.live_items[:] = [item for item in calls.live_items if item.id != data_id]

    module.datasets = _Datasets
    return calls


def _md5(content: str) -> str:
    import hashlib

    return hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()


class TestBestEffortHardDelete:
    async def test_delete_hard_deletes_matching_data_by_content_hash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, _ = _make_fake_cognee()
        data_items = [
            SimpleNamespace(id="data-1", content_hash=_md5("doomed fact"), raw_content_hash=None),
            SimpleNamespace(id="data-2", content_hash=_md5("other fact"), raw_content_hash=None),
        ]
        calls = _attach_fake_datasets_api(module, "ds", data_items)
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        memory = await backend.save_memory(content="doomed fact", user_id="alice", importance=0.5)
        assert await backend.delete_memory(memory.id) is True

        assert calls.deleted == [{"dataset_id": "dataset-uuid", "data_id": "data-1"}]

    async def test_hard_delete_failure_does_not_break_delete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The durable tombstone wins even when cognee's delete API fails."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["doomed fact"])])
        data_items = [
            SimpleNamespace(id="data-1", content_hash=_md5("doomed fact"), raw_content_hash=None),
        ]
        _attach_fake_datasets_api(module, "ds", data_items, delete_error=RuntimeError("boom"))
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        memory = await backend.save_memory(content="doomed fact", user_id="alice", importance=0.5)
        assert await backend.delete_memory(memory.id) is True
        assert await backend.search_memories(query="q", user_id="alice") == []

    async def test_hard_delete_skips_other_datasets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Data items in unrelated datasets are never touched."""
        module, _ = _make_fake_cognee()
        data_items = [
            SimpleNamespace(id="data-1", content_hash=_md5("doomed fact"), raw_content_hash=None),
        ]
        calls = _attach_fake_datasets_api(module, "someone_elses_dataset", data_items)
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        memory = await backend.save_memory(content="doomed fact", user_id="alice", importance=0.5)
        assert await backend.delete_memory(memory.id) is True
        assert calls.deleted == []

    async def test_unavailable_cognee_is_unproven_but_chunks_delete_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No cognee => removal unproven; CHUNKS deletes still succeed via tombstones."""
        module, _ = _make_fake_cognee()
        monkeypatch.setitem(sys.modules, "cognee", module)
        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        memory = await backend.save_memory(content="doomed fact", user_id="alice", importance=0.5)

        async def broken_init() -> None:
            raise RuntimeError("cognee is down")

        monkeypatch.setattr(backend, "_ensure_initialized", broken_init)
        assert await backend._try_hard_delete(["doomed fact"]) is False
        # Under CHUNKS the tombstone is authoritative, so the public delete
        # still succeeds — and its results filter provably excludes the text.
        assert await backend.delete_memory(memory.id) is True

    async def test_unmatched_content_is_unproven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'Nothing matched' is not proof of removal (chunk-registered rows)."""
        module, _ = _make_fake_cognee()
        _attach_fake_datasets_api(module, "ds", data_items=[])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        assert await backend._try_hard_delete(["never stored verbatim"]) is False

    async def test_matched_and_verified_is_proven(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, _ = _make_fake_cognee()
        data_items = [
            SimpleNamespace(id="data-1", content_hash=_md5("doomed"), raw_content_hash=None),
        ]
        _attach_fake_datasets_api(module, "ds", data_items)
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))
        assert await backend._try_hard_delete(["doomed"]) is True


# =============================================================================
# Deletion contract under synthesized search types (fail closed)
# =============================================================================


class TestSynthesizedModeDeletionContract:
    """The public API must never both report a successful delete and later
    surface information derived from the deleted memory.

    Under non-``CHUNKS`` search types cognee synthesizes result text, so a
    text-matched tombstone cannot enforce deletion; success there requires
    the underlying hard delete to be proven, otherwise the operation raises.
    """

    async def test_unverified_delete_fails_closed_instead_of_resurfacing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hard delete fails + a synthesized derivation exists => no success report."""
        original = "Vasilije prefers uv over poetry for Python projects"
        # Graph-synthesized text derived from the memory, sharing no verbatim
        # substring with it — exactly what tombstones cannot catch.
        synthesized = "The user is known to favor uv-style tooling."
        module, _ = _make_fake_cognee(search_results=[_fake_hit([synthesized])])
        _attach_fake_datasets_api(
            module,
            "ds",
            [SimpleNamespace(id="data-1", content_hash=_md5(original), raw_content_hash=None)],
            delete_error=RuntimeError("delete_data is down"),
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds", search_type="GRAPH_COMPLETION"))
        memory = await backend.save_memory(content=original, user_id="alice", importance=0.5)

        with pytest.raises(CogneeDeletionUnverifiedError):
            await backend.delete_memory(memory.id)

        # The synthesized derivation may still surface — but deletion was
        # never reported as successful, and the memory remains accounted for
        # in the registry so the delete can be retried.
        results = await backend.search_memories(query="tooling preference", user_id="alice")
        assert [r.memory.content for r in results] == [synthesized]
        assert await backend.get_memory(memory.id) is not None

    async def test_verified_delete_succeeds_in_synthesized_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, _ = _make_fake_cognee(search_results=[])
        _attach_fake_datasets_api(
            module,
            "ds",
            [SimpleNamespace(id="data-1", content_hash=_md5("doomed"), raw_content_hash=None)],
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds", search_type="GRAPH_COMPLETION"))
        memory = await backend.save_memory(content="doomed", user_id="alice", importance=0.5)
        assert await backend.delete_memory(memory.id) is True
        assert await backend.get_memory(memory.id) is None

    async def test_unverified_update_raises_without_mutation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, calls = _make_fake_cognee()
        _attach_fake_datasets_api(module, "ds", data_items=[])  # nothing matches: unproven
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds", search_type="GRAPH_COMPLETION"))
        memory = await backend.save_memory(content="original", user_id="alice", importance=0.5)
        adds_before = len(calls.add)

        with pytest.raises(CogneeDeletionUnverifiedError):
            await backend.update_memory(memory.id, "new content", user_id="alice")

        assert len(calls.add) == adds_before  # no new content was added
        fetched = await backend.get_memory(memory.id)
        assert fetched is not None
        assert fetched.content == "original"

    async def test_partial_match_deletes_nothing_and_stays_retryable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Discovery is non-mutating: a partial hash match must not delete the
        matched subset, or the unmatched remainder becomes permanently
        unprovable (absence is deliberately not proof) and the memory can
        never be deleted through this API again."""
        content = "the main content"
        fact = "an extracted fact"
        # Only the main content has a stored data item; the fact hash matches
        # nothing (content + facts need not map to separate cognee items).
        module, _ = _make_fake_cognee()
        calls = _attach_fake_datasets_api(
            module,
            "ds",
            [SimpleNamespace(id="data-1", content_hash=_md5(content), raw_content_hash=None)],
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds", search_type="GRAPH_COMPLETION"))
        memory = await backend.save_memory(
            content=content, user_id="alice", importance=0.5, facts=[fact]
        )

        with pytest.raises(CogneeDeletionUnverifiedError):
            await backend.delete_memory(memory.id)
        assert calls.deleted == []  # nothing was deleted on the failed attempt
        assert await backend.get_memory(memory.id) is not None

        # Once the fact's data item exists too, the SAME delete succeeds:
        # the earlier failure left the state fully retryable.
        calls.live_items.append(
            SimpleNamespace(id="data-2", content_hash=_md5(fact), raw_content_hash=None)
        )
        assert await backend.delete_memory(memory.id) is True
        assert {d["data_id"] for d in calls.deleted} == {"data-1", "data-2"}
        assert await backend.get_memory(memory.id) is None

    async def test_interrupted_deletion_is_retryable_via_ledger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure BETWEEN item deletions must not strand the memory: the
        durable hard-delete ledger remembers hashes already removed, so a
        retry can still assemble the full proof."""
        content = "the main content"
        fact = "an extracted fact"
        module, _ = _make_fake_cognee()
        calls = _attach_fake_datasets_api(
            module,
            "ds",
            [
                SimpleNamespace(id="data-1", content_hash=_md5(content), raw_content_hash=None),
                SimpleNamespace(id="data-2", content_hash=_md5(fact), raw_content_hash=None),
            ],
            fail_once_on="data-2",
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds", search_type="GRAPH_COMPLETION"))
        memory = await backend.save_memory(
            content=content, user_id="alice", importance=0.5, facts=[fact]
        )

        # First attempt: data-1 is deleted, then data-2's deletion fails.
        with pytest.raises(CogneeDeletionUnverifiedError):
            await backend.delete_memory(memory.id)
        assert await backend.get_memory(memory.id) is not None

        # Retry: data-1's hash is proven via the ledger (it no longer has an
        # item to match), data-2 is deleted now — full proof, delete succeeds.
        assert await backend.delete_memory(memory.id) is True
        assert await backend.get_memory(memory.id) is None

    async def test_resave_clears_ledger_so_stale_proof_cannot_vouch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-saving content invalidates its old 'provably removed' record."""
        module, _ = _make_fake_cognee()
        calls = _attach_fake_datasets_api(
            module,
            "ds",
            [SimpleNamespace(id="data-1", content_hash=_md5("doomed"), raw_content_hash=None)],
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds", search_type="GRAPH_COMPLETION"))
        memory = await backend.save_memory(content="doomed", user_id="alice", importance=0.5)
        assert await backend.delete_memory(memory.id) is True  # proven, ledger records it

        # Same content saved again — but the fake cognee has NO data item for
        # it, so its removal cannot be proven. The stale ledger entry from the
        # first delete must not vouch for it.
        memory2 = await backend.save_memory(content="doomed", user_id="alice", importance=0.5)
        with pytest.raises(CogneeDeletionUnverifiedError):
            await backend.delete_memory(memory2.id)

    async def test_chunks_mode_unverified_delete_succeeds_and_filters_fragments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counterpart: under CHUNKS the tombstone is authoritative, so an
        unproven hard delete still succeeds — and even a chunk FRAGMENT of
        the deleted content is provably filtered from results."""
        content = "Vasilije prefers uv over poetry for Python projects"
        fragment = "prefers uv over poetry"  # verbatim chunk of the content
        module, _ = _make_fake_cognee(search_results=[_fake_hit([fragment])])
        _attach_fake_datasets_api(module, "ds", data_items=[], delete_error=RuntimeError("down"))
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(dataset_name="ds"))  # CHUNKS default
        memory = await backend.save_memory(content=content, user_id="alice", importance=0.5)
        assert await backend.delete_memory(memory.id) is True
        assert await backend.search_memories(query="q", user_id="alice") == []


# =============================================================================
# Edge paths: import guard details, text extraction fallbacks, fact handling
# =============================================================================


class TestEdgePaths:
    async def test_import_error_when_search_type_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cognee module without SearchType is unusable and fails like a missing install."""
        module, _ = _make_fake_cognee()
        del module.SearchType
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        with pytest.raises(ImportError):
            await backend.ensure_initialized()

    def test_extract_text_fallbacks(self) -> None:
        backend = CogneeBackend(CogneeConfig())
        assert backend._extract_text("plain") == "plain"
        assert backend._extract_text({"text": "keyed"}) == "keyed"
        # Dict without any known text key falls back to its repr.
        assert backend._extract_text({"weird": 1}) == str({"weird": 1})
        # Non-str scalars stringify; None becomes empty (and is dropped later).
        assert backend._extract_text(42) == "42"
        assert backend._extract_text(None) == ""

    async def test_search_drops_items_without_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["", None, "real chunk"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        results = await backend.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in results] == ["real chunk"]

    async def test_update_tombstones_old_facts_without_cognify(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Update tombstones the old content AND its facts; auto_cognify=False skips enrichment."""
        module, calls = _make_fake_cognee(
            search_results=[_fake_hit(["old content", "old fact", "new content"])]
        )
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig(auto_cognify=False))
        memory = await backend.save_memory(
            content="old content",
            user_id="alice",
            importance=0.5,
            facts=["old fact", ""],  # falsey fact entries are ignored
        )
        await backend.update_memory(memory.id, "new content", user_id="alice")

        assert calls.cognify == []  # auto_cognify off: never called on save or update
        results = await backend.search_memories(query="q", user_id="alice")
        # Old content and its fact are tombstoned; only the new content survives.
        assert [r.memory.content for r in results] == ["new content"]

    async def test_delete_tombstones_facts_ignoring_falsey_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["doomed", "doomed fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)

        backend = CogneeBackend(CogneeConfig())
        memory = await backend.save_memory(
            content="doomed",
            user_id="alice",
            importance=0.5,
            facts=["doomed fact", "", None],  # type: ignore[list-item]
        )
        assert await backend.delete_memory(memory.id) is True
        assert await backend.search_memories(query="q", user_id="alice") == []


# =============================================================================
# Durable state: restarts and multiple live instances share one metadata DB
# =============================================================================


class TestDurableState:
    def _backend(self, db_path: str) -> CogneeBackend:
        return CogneeBackend(CogneeConfig(metadata_db_path=db_path))

    async def test_delete_survives_restart(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """A new instance (simulated restart) still sees the deletion."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["doomed fact", "kept fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)
        db_path = str(tmp_path / "shared_meta.db")

        instance_a = self._backend(db_path)
        memory = await instance_a.save_memory(
            content="doomed fact", user_id="alice", importance=0.5
        )
        assert await instance_a.delete_memory(memory.id) is True

        # Simulate a proxy restart: fresh process-wide state, same DB file.
        cognee_backend_module._reset_process_state_for_testing()
        instance_b = self._backend(db_path)
        results = await instance_b.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in results] == ["kept fact"]
        assert await instance_b.get_memory(memory.id) is None

    async def test_two_live_instances_share_deletions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A second concurrent instance cannot resurrect deleted content."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["doomed fact", "kept fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)
        db_path = str(tmp_path / "shared_meta.db")

        instance_a = self._backend(db_path)
        instance_b = self._backend(db_path)

        memory = await instance_a.save_memory(
            content="doomed fact", user_id="alice", importance=0.5
        )
        before = await instance_b.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in before] == ["doomed fact", "kept fact"]

        assert await instance_a.delete_memory(memory.id) is True
        after = await instance_b.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in after] == ["kept fact"]

    async def test_update_roundtrip_across_instances_keeps_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """save -> update -> (new instance) search/get/delete, all on the old ID."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["old fact", "new fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)
        db_path = str(tmp_path / "shared_meta.db")

        instance_a = self._backend(db_path)
        saved = await instance_a.save_memory(content="old fact", user_id="alice", importance=0.8)
        old_id = saved.id
        updated = await instance_a.update_memory(old_id, "new fact")
        assert updated.id == old_id

        # Simulate a restart / second worker.
        cognee_backend_module._reset_process_state_for_testing()
        instance_b = self._backend(db_path)

        results = await instance_b.search_memories(query="q", user_id="alice")
        assert [r.memory.content for r in results] == ["new fact"]
        assert results[0].memory.id == old_id

        fetched = await instance_b.get_memory(old_id)
        assert fetched is not None
        assert fetched.content == "new fact"
        assert fetched.importance == 0.8

        assert await instance_b.delete_memory(old_id) is True
        assert await instance_b.search_memories(query="q", user_id="alice") == []
        assert await instance_b.get_memory(old_id) is None

    async def test_update_keeps_id_stable_within_one_instance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Search after update returns the old ID; no duplicate registry row."""
        module, _ = _make_fake_cognee(search_results=[_fake_hit(["old fact", "new fact"])])
        monkeypatch.setitem(sys.modules, "cognee", module)
        db_path = str(tmp_path / "meta.db")

        backend = self._backend(db_path)
        saved = await backend.save_memory(content="old fact", user_id="alice", importance=0.5)
        await backend.update_memory(saved.id, "new fact")

        results = await backend.search_memories(query="q", user_id="alice")
        assert [r.memory.id for r in results] == [saved.id]
        assert results[0].memory.content == "new fact"

        # The durable registry holds exactly one row for alice (updated in
        # place) — the search did not mint a second object for the new
        # content under a fresh content-derived ID.
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT id, content FROM memories WHERE user_id = ?", ("alice",)
            ).fetchall()
        finally:
            conn.close()
        assert rows == [(saved.id, "new fact")]


# =============================================================================
# Process-wide immutable cognee configuration
# =============================================================================


class TestProcessWideConfiguration:
    async def test_conflicting_roots_fail_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A second instance with different roots must not reconfigure cognee."""
        module, calls = _make_fake_cognee()
        monkeypatch.setitem(sys.modules, "cognee", module)

        first = CogneeBackend(CogneeConfig(system_root="/sys/x", data_root="/data/x"))
        await first.ensure_initialized()

        second = CogneeBackend(CogneeConfig(system_root="/sys/y", data_root="/data/y"))
        with pytest.raises(RuntimeError, match="process-wide and immutable"):
            await second.ensure_initialized()

        # The first tenant's configuration was never overwritten.
        assert calls.system_root == ["/sys/x"]
        assert calls.data_root == ["/data/x"]

    async def test_disables_cognee_session_caching_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cognee >=1.5 session memory can replay deleted content; headroom opts out."""
        module, _ = _make_fake_cognee()
        monkeypatch.setitem(sys.modules, "cognee", module)
        monkeypatch.delenv("CACHING", raising=False)

        backend = CogneeBackend(CogneeConfig())
        await backend.ensure_initialized()
        assert os.environ["CACHING"] == "false"

    async def test_respects_operator_caching_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicitly exported CACHING value wins over the opt-out default."""
        module, _ = _make_fake_cognee()
        monkeypatch.setitem(sys.modules, "cognee", module)
        monkeypatch.setenv("CACHING", "true")

        backend = CogneeBackend(CogneeConfig())
        await backend.ensure_initialized()
        assert os.environ["CACHING"] == "true"

    async def test_same_roots_second_instance_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same roots reuse the configured module without reconfiguring."""
        module, calls = _make_fake_cognee()
        monkeypatch.setitem(sys.modules, "cognee", module)

        first = CogneeBackend(CogneeConfig(system_root="/sys/x", data_root="/data/x"))
        await first.ensure_initialized()
        second = CogneeBackend(CogneeConfig(system_root="/sys/x", data_root="/data/x"))
        await second.ensure_initialized()

        # Configured exactly once, shared by both instances.
        assert calls.system_root == ["/sys/x"]
        assert calls.data_root == ["/data/x"]
        assert second._cognee is first._cognee

    async def test_concurrent_instances_with_different_roots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent init with different roots: exactly one wins, one fails."""
        module, calls = _make_fake_cognee()
        monkeypatch.setitem(sys.modules, "cognee", module)

        instance_a = CogneeBackend(CogneeConfig(system_root="/sys/a", data_root="/data/a"))
        instance_b = CogneeBackend(CogneeConfig(system_root="/sys/b", data_root="/data/b"))

        outcomes = await asyncio.gather(
            instance_a.ensure_initialized(),
            instance_b.ensure_initialized(),
            return_exceptions=True,
        )

        errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "process-wide and immutable" in str(errors[0])
        # Exactly one configuration was applied (never both, never a mix).
        assert calls.system_root in (["/sys/a"], ["/sys/b"])
        assert calls.data_root in (["/data/a"], ["/data/b"])
        assert (calls.system_root, calls.data_root) in (
            (["/sys/a"], ["/data/a"]),
            (["/sys/b"], ["/data/b"]),
        )


# =============================================================================
# Environment isolation around the deferred import
# =============================================================================


class TestImportEnvIsolation:
    async def test_import_side_effects_on_env_are_reverted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """cognee's import-time env mutations (dotenv override) are undone.

        A real module is executed (written to a temp dir, not pre-seeded in
        sys.modules) so import side effects actually run. Pre-existing env
        vars it overwrites must be restored; vars it newly sets are kept.
        """
        fake_src = """
import os
from enum import Enum

# Simulate dotenv.load_dotenv(override=True) side effects.
os.environ["HEADROOM_TEST_PREEXISTING"] = "overwritten-by-dotenv"
os.environ["HEADROOM_TEST_NEW_FROM_DOTENV"] = "added-by-dotenv"


class SearchType(str, Enum):
    CHUNKS = "CHUNKS"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"


async def add(data, dataset_name=None, node_set=None, **kwargs):
    pass


async def cognify(datasets=None, run_in_background=None, **kwargs):
    pass


async def search(**kwargs):
    return []


class config:
    @staticmethod
    def system_root_directory(path):
        pass

    @staticmethod
    def data_root_directory(path):
        pass
"""
        (tmp_path / "cognee.py").write_text(fake_src)
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setenv("HEADROOM_TEST_PREEXISTING", "original")
        monkeypatch.delenv("HEADROOM_TEST_NEW_FROM_DOTENV", raising=False)
        sys.modules.pop("cognee", None)

        try:
            backend = CogneeBackend(CogneeConfig())
            await backend.ensure_initialized()

            # Pre-existing value restored (env-over-.env precedence kept).
            assert os.environ["HEADROOM_TEST_PREEXISTING"] == "original"
            # Newly added key kept (cognee's own .env config still works).
            assert os.environ["HEADROOM_TEST_NEW_FROM_DOTENV"] == "added-by-dotenv"

            # The snapshot/restore runs exactly once per process: a second
            # instance initializing with the same roots reuses the already
            # imported module and never re-runs the import side effects.
            os.environ["HEADROOM_TEST_PREEXISTING"] = "changed-after-first-init"
            os.environ.pop("HEADROOM_TEST_NEW_FROM_DOTENV", None)

            second = CogneeBackend(CogneeConfig())
            await second.ensure_initialized()

            assert os.environ["HEADROOM_TEST_PREEXISTING"] == "changed-after-first-init"
            assert "HEADROOM_TEST_NEW_FROM_DOTENV" not in os.environ
            assert second._cognee is backend._cognee
        finally:
            sys.modules.pop("cognee", None)
            os.environ.pop("HEADROOM_TEST_NEW_FROM_DOTENV", None)


# =============================================================================
# Capability flags / package exports
# =============================================================================


class TestCapabilities:
    def test_supports_graph(self) -> None:
        assert CogneeBackend(CogneeConfig()).supports_graph is True

    def test_supports_vector_search(self) -> None:
        assert CogneeBackend(CogneeConfig()).supports_vector_search is True

    def test_lazy_exports_from_backends_package(self) -> None:
        from headroom.memory import backends

        assert backends.CogneeBackend is CogneeBackend
        assert backends.CogneeConfig is CogneeConfig
        assert "CogneeBackend" in backends.__all__
        assert "CogneeConfig" in backends.__all__
