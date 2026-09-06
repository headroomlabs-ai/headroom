"""Proxy-level wiring tests for the cognee memory backend.

Covers:
- ``MemoryConfig`` (proxy handler) accepts ``backend="cognee"`` and resolves
  ``cognee_*`` fields from ``HEADROOM_COGNEE_*`` env vars
- ``ProxyConfig`` accepts ``memory_backend="cognee"`` and resolves
  ``memory_cognee_*`` fields (incl. the fail-soft auto_cognify default)
- ``MemoryHandler._init_backend_locked`` selects and configures
  ``CogneeBackend`` for ``backend="cognee"`` (cognee module faked)
- ImportError guard message when cognee is not installed

cognee is NOT a test dependency: a fake module is injected into
``sys.modules`` before the backend's lazy import runs.
"""

from __future__ import annotations

import sys
import types
from enum import Enum
from types import SimpleNamespace

import pytest

from headroom.memory.backends import cognee as cognee_backend_module
from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler
from headroom.proxy.models import ProxyConfig

_COGNEE_ENV_VARS = (
    "HEADROOM_COGNEE_DATASET",
    "HEADROOM_COGNEE_SYSTEM_ROOT",
    "HEADROOM_COGNEE_DATA_ROOT",
    "HEADROOM_COGNEE_SEARCH_TYPE",
    "HEADROOM_COGNEE_AUTO_COGNIFY",
    "HEADROOM_COGNEE_METADATA_DB",
)


@pytest.fixture(autouse=True)
def _cognee_test_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Clean env, per-test tmp metadata DB, fresh process-wide cognee state."""
    for var in _COGNEE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HEADROOM_COGNEE_METADATA_DB", str(tmp_path / "cognee_meta.db"))
    cognee_backend_module._reset_process_state_for_testing()
    yield
    cognee_backend_module._reset_process_state_for_testing()


def _install_fake_cognee(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Inject a minimal fake ``cognee`` module; returns a call recorder."""
    calls = SimpleNamespace(add=[], cognify=[], search=[], system_root=[], data_root=[])
    module = types.ModuleType("cognee")

    class _SearchType(str, Enum):
        CHUNKS = "CHUNKS"
        GRAPH_COMPLETION = "GRAPH_COMPLETION"

    async def add(data, dataset_name=None, node_set=None, **kwargs):
        calls.add.append({"data": data, "dataset_name": dataset_name, "node_set": node_set})

    async def cognify(datasets=None, run_in_background=None, **kwargs):
        calls.cognify.append({"datasets": datasets, "run_in_background": run_in_background})

    async def search(**kwargs):
        calls.search.append(kwargs)
        return []

    module.SearchType = _SearchType
    module.add = add
    module.cognify = cognify
    module.search = search
    module.config = SimpleNamespace(
        system_root_directory=calls.system_root.append,
        data_root_directory=calls.data_root.append,
    )
    monkeypatch.setitem(sys.modules, "cognee", module)
    return calls


# =============================================================================
# MemoryConfig (proxy handler)
# =============================================================================


class TestProxyMemoryConfig:
    def test_accepts_cognee_backend(self) -> None:
        cfg = MemoryConfig(enabled=True, backend="cognee")
        assert cfg.backend == "cognee"

    def test_cognee_defaults_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_COGNEE_METADATA_DB", raising=False)
        cfg = MemoryConfig(backend="cognee")
        assert cfg.cognee_dataset == "headroom_memories"
        assert cfg.cognee_system_root is None
        assert cfg.cognee_data_root is None
        assert cfg.cognee_search_type == "CHUNKS"
        assert cfg.cognee_auto_cognify is True
        assert cfg.cognee_metadata_db_path is None

    def test_cognee_fields_read_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "proxy_env_ds")
        monkeypatch.setenv("HEADROOM_COGNEE_SYSTEM_ROOT", "/sys/root")
        monkeypatch.setenv("HEADROOM_COGNEE_DATA_ROOT", "/data/root")
        monkeypatch.setenv("HEADROOM_COGNEE_SEARCH_TYPE", "GRAPH_COMPLETION")
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", "false")
        monkeypatch.setenv("HEADROOM_COGNEE_METADATA_DB", "/var/cognee/meta.db")

        cfg = MemoryConfig(backend="cognee")
        assert cfg.cognee_dataset == "proxy_env_ds"
        assert cfg.cognee_system_root == "/sys/root"
        assert cfg.cognee_data_root == "/data/root"
        assert cfg.cognee_search_type == "GRAPH_COMPLETION"
        assert cfg.cognee_auto_cognify is False
        assert cfg.cognee_metadata_db_path == "/var/cognee/meta.db"

    def test_explicit_values_win_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "env_ds")
        cfg = MemoryConfig(backend="cognee", cognee_dataset="explicit_ds")
        assert cfg.cognee_dataset == "explicit_ds"

    def test_strict_bool_resolver_raises_on_garbage_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MemoryConfig uses the strict resolver (matches qdrant_env_port usage)."""
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", "maybe")
        with pytest.raises(ValueError, match="Invalid boolean value"):
            MemoryConfig(backend="cognee")


# =============================================================================
# ProxyConfig
# =============================================================================


class TestProxyConfigCogneeFields:
    def test_accepts_cognee_memory_backend(self) -> None:
        cfg = ProxyConfig(memory_backend="cognee")
        assert cfg.memory_backend == "cognee"

    def test_cognee_defaults_when_no_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HEADROOM_COGNEE_METADATA_DB", raising=False)
        cfg = ProxyConfig()
        assert cfg.memory_backend == "local"
        assert cfg.memory_cognee_dataset == "headroom_memories"
        assert cfg.memory_cognee_system_root is None
        assert cfg.memory_cognee_data_root is None
        assert cfg.memory_cognee_search_type == "CHUNKS"
        assert cfg.memory_cognee_auto_cognify is True
        assert cfg.memory_cognee_metadata_db_path is None

    def test_cognee_fields_read_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "shared_ds")
        monkeypatch.setenv("HEADROOM_COGNEE_SYSTEM_ROOT", "/sys/px")
        monkeypatch.setenv("HEADROOM_COGNEE_DATA_ROOT", "/data/px")
        monkeypatch.setenv("HEADROOM_COGNEE_SEARCH_TYPE", "GRAPH_COMPLETION")
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", "no")
        monkeypatch.setenv("HEADROOM_COGNEE_METADATA_DB", "/var/cognee/meta.db")

        cfg = ProxyConfig(memory_backend="cognee")
        assert cfg.memory_cognee_dataset == "shared_ds"
        assert cfg.memory_cognee_system_root == "/sys/px"
        assert cfg.memory_cognee_data_root == "/data/px"
        assert cfg.memory_cognee_search_type == "GRAPH_COMPLETION"
        assert cfg.memory_cognee_auto_cognify is False
        assert cfg.memory_cognee_metadata_db_path == "/var/cognee/meta.db"

    def test_auto_cognify_fail_soft_on_garbage_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo'd env var must not crash unrelated ProxyConfig construction."""
        monkeypatch.setenv("HEADROOM_COGNEE_AUTO_COGNIFY", "maybe")
        cfg = ProxyConfig()  # must not raise
        assert cfg.memory_cognee_auto_cognify is True


# =============================================================================
# MemoryHandler backend selection
# =============================================================================


class TestMemoryHandlerCogneeSelection:
    async def test_init_backend_selects_cognee(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from headroom.memory.backends.cognee import CogneeBackend

        _install_fake_cognee(monkeypatch)
        metadata_db = str(tmp_path / "proxy_cognee_meta.db")

        config = MemoryConfig(
            enabled=True,
            backend="cognee",
            cognee_dataset="proxy_ds",
            cognee_search_type="GRAPH_COMPLETION",
            cognee_auto_cognify=False,
            cognee_metadata_db_path=metadata_db,
        )
        handler = MemoryHandler(config)
        await handler._ensure_initialized()

        assert handler._initialized is True
        assert isinstance(handler._backend, CogneeBackend)
        # MemoryConfig values are threaded through to CogneeConfig.
        assert handler._backend._config.dataset_name == "proxy_ds"
        assert handler._backend._config.search_type == "GRAPH_COMPLETION"
        assert handler._backend._config.auto_cognify is False
        assert handler._backend._config.metadata_db_path == metadata_db

    async def test_init_backend_applies_isolation_roots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _install_fake_cognee(monkeypatch)

        config = MemoryConfig(
            enabled=True,
            backend="cognee",
            cognee_system_root="/sys/isolated",
            cognee_data_root="/data/isolated",
        )
        handler = MemoryHandler(config)
        await handler._ensure_initialized()

        # ensure_initialized() ran eagerly and applied directory isolation.
        assert calls.system_root == ["/sys/isolated"]
        assert calls.data_root == ["/data/isolated"]

    async def test_init_backend_raises_install_hint_when_cognee_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Poison the import so the lazy `import cognee` raises ImportError.
        monkeypatch.setitem(sys.modules, "cognee", None)

        config = MemoryConfig(enabled=True, backend="cognee")
        handler = MemoryHandler(config)
        with pytest.raises(ImportError, match=r"headroom-ai\[cognee\]"):
            await handler._init_backend_locked()

    async def test_disabled_memory_never_touches_cognee(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "cognee", None)

        config = MemoryConfig(enabled=False, backend="cognee")
        handler = MemoryHandler(config)
        await handler._ensure_initialized()  # must not raise
        assert handler._backend is None


# =============================================================================
# Per-project scoping (GH #462) with the cognee backend
# =============================================================================


class TestCogneeProjectScoping:
    async def _initialized_handler(self, monkeypatch: pytest.MonkeyPatch) -> MemoryHandler:
        _install_fake_cognee(monkeypatch)
        handler = MemoryHandler(MemoryConfig(enabled=True, backend="cognee"))
        await handler._ensure_initialized()
        return handler

    async def test_scope_router_is_built_for_cognee(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = await self._initialized_handler(monkeypatch)
        assert handler._router is not None

    async def test_effective_user_id_composes_project_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default PROJECT storage mode partitions cognee via user::project_key."""
        from headroom.memory.storage_router import RequestContext

        handler = await self._initialized_handler(monkeypatch)
        ctx = RequestContext(
            headers={"x-headroom-project-id": "projA"},
            system_prompt="",
            base_user_id="alice",
        )
        backend, scope, effective_user_id = handler._resolve_for_request("alice", ctx)

        assert backend is handler._backend
        assert scope is not None
        # Current project scoping appends a collision-resistant digest to the
        # human-readable project ID; Cognee must consume that canonical key
        # instead of rebuilding the older unhashed form.
        assert scope.project_key.startswith("projA-")
        assert effective_user_id == f"alice::{scope.project_key}"

    async def test_unresolved_project_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No project signal in PROJECT mode -> sentinel scope, bare user id.

        The fail-closed guard in search_and_format_context checks
        ``scope.mode is PROJECT and scope.project_key is None`` and skips
        injection instead of pooling into a global bucket.
        """
        from headroom.memory.storage_router import MemoryStorageMode, RequestContext

        handler = await self._initialized_handler(monkeypatch)
        ctx = RequestContext(headers={}, system_prompt="", base_user_id="alice")
        _backend, scope, effective_user_id = handler._resolve_for_request("alice", ctx)

        assert scope is not None
        assert scope.mode is MemoryStorageMode.PROJECT
        assert scope.project_key is None
        assert effective_user_id == "alice"


# =============================================================================
# CLI: --memory-backend cognee and --memory-cognee-* flags reach ProxyConfig
# =============================================================================


class TestProxyCLICogneeFlags:
    def _invoke(self, args: list[str]):
        from unittest.mock import patch

        from click.testing import CliRunner

        from headroom.cli.main import main

        captured: dict[str, object] = {}

        def fake_run_server(config, **kwargs):
            captured["config"] = config
            raise SystemExit(0)

        with patch("headroom.proxy.server.run_server", side_effect=fake_run_server):
            result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, result.output
        assert "config" in captured, result.output
        return captured["config"]

    def test_cognee_flags_reach_proxy_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = self._invoke(
            [
                "proxy",
                "--memory",
                "--memory-backend",
                "cognee",
                "--memory-cognee-dataset",
                "cli_ds",
                "--memory-cognee-system-root",
                "/tmp/cognee-sys",
                "--memory-cognee-data-root",
                "/tmp/cognee-data",
                "--memory-cognee-search-type",
                "GRAPH_COMPLETION",
                "--no-memory-cognee-auto-cognify",
                "--memory-cognee-metadata-db",
                "/tmp/cognee-meta.db",
            ]
        )
        assert config.memory_backend == "cognee"
        assert config.memory_cognee_dataset == "cli_ds"
        assert config.memory_cognee_system_root == "/tmp/cognee-sys"
        assert config.memory_cognee_data_root == "/tmp/cognee-data"
        assert config.memory_cognee_search_type == "GRAPH_COMPLETION"
        assert config.memory_cognee_auto_cognify is False
        assert config.memory_cognee_metadata_db_path == "/tmp/cognee-meta.db"

    def test_omitted_cognee_flags_fall_back_to_env_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without CLI flags, ProxyConfig resolves HEADROOM_COGNEE_* env defaults."""
        monkeypatch.delenv("HEADROOM_MEMORY_BACKEND", raising=False)
        monkeypatch.setenv("HEADROOM_COGNEE_DATASET", "env_ds")

        config = self._invoke(["proxy", "--memory"])
        assert config.memory_backend == "local"
        assert config.memory_cognee_dataset == "env_ds"
        assert config.memory_cognee_system_root is None
        assert config.memory_cognee_auto_cognify is True
