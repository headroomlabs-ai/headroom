from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from headroom.proxy.capabilities import DetachedModeError, build_capability_report
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from headroom import paths
    from headroom.cache import compression_store
    from headroom.telemetry import toin

    monkeypatch.setattr(paths, "_PROCESS_STATELESS", False)
    monkeypatch.setattr(compression_store, "_compression_store", None)
    monkeypatch.setattr(toin, "_toin_instance", None)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("HEADROOM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.delenv("HEADROOM_DETACHED_PROFILE", raising=False)
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.delenv("HEADROOM_TOIN_BACKEND", raising=False)
    monkeypatch.delenv("HEADROOM_CCR_BACKEND", raising=False)


def _minimal_config(**overrides: Any) -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        subscription_tracking_enabled=False,
        **overrides,
    )


def test_capability_report_marks_stateless_as_detached() -> None:
    report = build_capability_report(_minimal_config(stateless=True))

    payload = report.to_dict()

    assert payload["detached"] is True
    assert payload["profile"] == "lenient"
    assert payload["local_state"]["available"] is False
    features = {item["feature"]: item for item in payload["features"]}
    assert features["proxy_request_handling"]["state"] == "full"
    assert features["compression"]["state"] == "disabled"
    assert features["dashboard_live_data"]["state"] == "degraded"
    assert features["session_aggregation"]["state"] == "disabled"


def test_strict_detached_profile_refuses_enabled_memory_without_state() -> None:
    config = _minimal_config(
        stateless=True,
        detached_profile="strict",
        memory_enabled=True,
    )

    with pytest.raises(DetachedModeError) as exc:
        create_app(config)

    violations = [item["feature"] for item in exc.value.report.to_dict()["strict_violations"]]
    assert violations == ["memory"]


def test_embedded_startup_honors_saved_detached_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom import settings_store

    monkeypatch.setattr(settings_store, "load", lambda: {"detached_profile": "silent"})

    app = create_app()

    assert app.state.capabilities["profile"] == "silent"


def test_capabilities_endpoint_health_stats_and_metrics_share_report() -> None:
    app = create_app(_minimal_config(stateless=True))

    with TestClient(app) as client:
        capabilities = client.get("/capabilities")
        health = client.get("/health")
        stats = client.get("/stats")
        metrics = client.get("/metrics")

    assert capabilities.status_code == 200
    capability_payload = capabilities.json()
    assert capability_payload["detached"] is True
    assert capability_payload["local_state"]["available"] is False
    assert "workspace_dir" not in capability_payload["local_state"]
    assert health.json()["capabilities"] == capability_payload
    assert stats.json()["capabilities"] == capability_payload
    assert 'headroom_feature_enabled{feature="proxy_request_handling"' in metrics.text
    assert 'headroom_feature_enabled{feature="session_aggregation"' in metrics.text


def test_loopback_capabilities_include_operator_workspace_path() -> None:
    app = create_app(_minimal_config(stateless=True))

    with TestClient(
        app,
        client=("127.0.0.1", 50000),
        headers={"host": "127.0.0.1"},
    ) as client:
        capabilities = client.get("/capabilities").json()
        health = client.get("/health").json()
        stats = client.get("/stats").json()

    expected = str(Path(os.environ["HEADROOM_WORKSPACE_DIR"]))
    assert capabilities["local_state"]["workspace_dir"] == expected
    assert health["capabilities"]["local_state"]["workspace_dir"] == expected
    assert stats["capabilities"]["local_state"]["workspace_dir"] == expected


def test_remote_health_never_exposes_workspace_path() -> None:
    app = create_app(_minimal_config(stateless=True, host="0.0.0.0"))

    with TestClient(
        app,
        client=("203.0.113.10", 50000),
        headers={"host": "proxy.example.test"},
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert str(Path(os.environ["HEADROOM_WORKSPACE_DIR"])) not in response.text
    assert "workspace_dir" not in response.json()["capabilities"]["local_state"]


def test_remote_capabilities_redact_workspace_paths_in_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = str(Path(os.environ["HEADROOM_WORKSPACE_DIR"]))
    import headroom.proxy.capabilities as capabilities_module

    monkeypatch.setattr(
        capabilities_module,
        "_probe_local_state",
        lambda _config: (False, f"PermissionError: {workspace}/private-file"),
    )
    app = create_app(_minimal_config(host="0.0.0.0"))

    with TestClient(
        app,
        client=("203.0.113.10", 50000),
        headers={"host": "proxy.example.test"},
    ) as client:
        response = client.get("/capabilities")

    assert workspace not in response.text
    assert "<workspace>/private-file" in response.text


def test_capabilities_do_not_expose_backend_url_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "redis://user:toin-secret@cache.internal:6379/0")
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "https://token:ccr-secret@ccr.internal/store")

    payload = build_capability_report(_minimal_config(stateless=True)).to_dict(
        include_workspace_dir=False
    )
    serialized = str(payload)
    features = {item["feature"]: item for item in payload["features"]}

    assert "toin-secret" not in serialized
    assert "ccr-secret" not in serialized
    assert "cache.internal" not in serialized
    assert "ccr.internal" not in serialized
    assert features["toin_tagging"]["backend"] == "memory"
    assert features["ccr_retrieval"]["backend"] == "memory"


def test_stateless_startup_skips_file_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_file_logging() -> None:
        calls.append("file_logging")
        raise AssertionError("stateless startup must not install file logging")

    import headroom.proxy.server as server

    monkeypatch.setattr(server, "_setup_file_logging", fail_file_logging)

    app = create_app(_minimal_config(stateless=True))
    with TestClient(app) as client:
        response = client.get("/capabilities")

    assert response.status_code == 200
    assert calls == []


@pytest.mark.parametrize("backend", ["sqlite", "qdrant-neo4j"])
def test_strict_stateless_refuses_memory_for_every_backend(backend: str) -> None:
    with pytest.raises(DetachedModeError) as exc:
        create_app(
            _minimal_config(
                stateless=True,
                detached_profile="strict",
                memory_enabled=True,
                memory_backend=backend,
            )
        )
    assert [item.feature for item in exc.value.report.strict_violations] == ["memory"]


def test_lenient_stateless_memory_report_matches_runtime() -> None:
    app = create_app(
        _minimal_config(stateless=True, memory_enabled=True, memory_backend="qdrant-neo4j")
    )
    memory = next(f for f in app.state.capabilities["features"] if f["feature"] == "memory")
    assert memory["state"] == "disabled"
    assert memory["enabled"] is False
    assert app.state.proxy.memory_handler is None


def _ccr_feature(config: ProxyConfig) -> dict[str, Any]:
    report = build_capability_report(config)
    return next(f.to_dict() for f in report.features if f.feature == "ccr_retrieval")


def test_ccr_report_uses_initialized_sqlite_default() -> None:
    from headroom.cache.compression_store import get_compression_store

    feature = _ccr_feature(_minimal_config())
    assert feature["backend"] == "sqlite"
    assert feature["state"] == "full"
    assert get_compression_store().get_stats()["backend"]["backend_type"] == "sqlite"


def test_ccr_report_respects_explicit_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    feature = _ccr_feature(_minimal_config())
    assert feature["backend"] == "memory"
    assert feature["state"] == "degraded"


@pytest.mark.parametrize("failure", ["missing", "load", "factory"])
def test_ccr_report_does_not_promote_failed_remote_adapter(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import importlib.metadata

    class EntryPoint:
        name = "redis"

        def load(self):
            if failure == "load":
                raise ImportError("adapter unavailable")

            def factory(**kwargs):
                raise OSError("backend unavailable")

            return factory

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: [] if failure == "missing" else [EntryPoint()],
    )
    feature = _ccr_feature(_minimal_config())
    assert feature["backend"] == "memory"
    assert feature["state"] == "degraded"


def test_ccr_report_reflects_sqlite_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from headroom.cache.backends import sqlite

    # Patch initialization rather than replacing the type: backend identification
    # must still distinguish a fallback InMemoryBackend from SQLiteBackend.
    def fail_init(self, *args, **kwargs):
        raise OSError("read-only database")

    monkeypatch.setattr(sqlite.SQLiteBackend, "__init__", fail_init)
    feature = _ccr_feature(_minimal_config())
    assert feature["backend"] == "memory"
    assert feature["state"] == "degraded"


def test_resolving_stateless_ccr_does_not_create_workspace() -> None:
    feature = _ccr_feature(_minimal_config(stateless=True))
    assert feature["backend"] == "memory"
    assert not Path(os.environ["HEADROOM_WORKSPACE_DIR"]).exists()


def test_ccr_report_uses_successfully_initialized_custom_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata

    from headroom.cache.backends.memory import InMemoryBackend

    class Adapter:
        def __init__(self):
            self.storage = InMemoryBackend()

        def __getattr__(self, name):
            return getattr(self.storage, name)

    class EntryPoint:
        name = "redis"

        def load(self):
            return lambda **kwargs: Adapter()

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "redis")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: [EntryPoint()])
    feature = _ccr_feature(_minimal_config())
    assert feature["backend"] == "custom"
    assert feature["state"] == "full"


def test_ccr_report_describes_existing_store_even_if_env_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.cache.compression_store import get_compression_store

    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    store = get_compression_store()
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "sqlite")
    feature = _ccr_feature(_minimal_config())
    assert feature["backend"] == "memory"
    assert get_compression_store() is store


@pytest.mark.parametrize("stateless", [False, True])
def test_unavailable_local_state_disables_runtime_learning(
    monkeypatch: pytest.MonkeyPatch, stateless: bool
) -> None:
    import headroom.proxy.capabilities as capabilities_module

    monkeypatch.setattr(
        capabilities_module, "_probe_local_state", lambda config: (False, "read-only workspace")
    )
    app = create_app(
        _minimal_config(
            stateless=stateless, traffic_learning_enabled=True, detached_profile="lenient"
        )
    )
    feature = next(f for f in app.state.capabilities["features"] if f["feature"] == "learn_plugins")
    assert feature["enabled"] is False
    assert app.state.proxy.traffic_learner is None


@pytest.mark.parametrize("failure", ["missing", "load", "factory"])
def test_toin_report_uses_actual_fallback_backend(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import importlib.metadata

    from headroom.telemetry.toin import get_toin

    class EntryPoint:
        name = "redis"

        def load(self):
            if failure == "load":
                raise ImportError("adapter unavailable")

            def factory(**kwargs):
                raise OSError("backend unavailable")

            return factory

    monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "redis")
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: (
            [EntryPoint()]
            if kwargs.get("group") == "headroom.toin_backend" and failure != "missing"
            else []
        ),
    )
    app = create_app(_minimal_config(stateless=True))
    feature = next(f for f in app.state.capabilities["features"] if f["feature"] == "toin_tagging")
    assert get_toin()._backend is None
    assert feature["backend"] == "memory"
    assert feature["state"] == "degraded"
    assert feature["enabled"] is True  # observations still accumulate in memory


def test_toin_report_keeps_the_initialized_custom_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    from headroom.telemetry.toin import get_toin

    class Adapter:
        def load(self):
            return {}

        def save(self, data):
            pass

    adapter = Adapter()
    calls = []

    class EntryPoint:
        name = "redis"

        def load(self):
            def factory(**kwargs):
                calls.append("initialized")
                return adapter

            return factory

    monkeypatch.setenv("HEADROOM_TOIN_BACKEND", "redis")
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: [EntryPoint()] if kwargs.get("group") == "headroom.toin_backend" else [],
    )
    app = create_app(_minimal_config(stateless=True))
    feature = next(f for f in app.state.capabilities["features"] if f["feature"] == "toin_tagging")
    assert calls == ["initialized"]
    assert get_toin()._backend is adapter
    assert feature["backend"] == "custom"
    assert feature["state"] == "full"


def test_stateful_learning_remains_enabled_and_strict_stateless_refuses_it() -> None:
    app = create_app(_minimal_config(traffic_learning_enabled=True))
    assert app.state.proxy.traffic_learner is not None
    with pytest.raises(DetachedModeError) as exc:
        create_app(
            _minimal_config(
                stateless=True, traffic_learning_enabled=True, detached_profile="strict"
            )
        )
    assert [f.feature for f in exc.value.report.strict_violations] == ["learn_plugins"]
