import pytest
from fastapi.testclient import TestClient

from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app
from headroom.transforms import kompress_compressor


class _ReadyCompressor:
    def __init__(self, backend="onnx", ready=True, error=None):
        self.backend = backend
        self.ready = ready
        self.error = error
        self.calls = []

    def is_ready(self):
        self.calls.append("is_ready")
        if self.error:
            raise self.error
        return self.ready

    def ready_backend(self):
        self.calls.append("ready_backend")
        return self.backend


def _health_app(monkeypatch, compressor=None, *, disabled=False, **config_kwargs):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            disable_kompress=disabled,
            **config_kwargs,
        )
    )
    app.state.ready = True
    proxy = app.state.proxy
    proxy.http_client = object()
    router = proxy.anthropic_pipeline.transforms[-1]
    if compressor is not None:
        router._kompress = compressor
    return app, proxy


def test_readyz_promotes_deferred_kompress_after_runtime_load(monkeypatch):
    compressor = _ReadyCompressor()
    app, proxy = _health_app(monkeypatch, compressor)
    proxy.warmup.kompress.info["source_status"] = "deferred"

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": True,
        "status": "healthy",
        "optional": True,
        "backend": "onnx",
    }
    # The promotion must also clear the startup marker, otherwise the slot
    # serializes as loaded-but-deferred in /debug/warmup.
    assert proxy.warmup.kompress.info["source_status"] == "runtime"


@pytest.mark.parametrize("attached", [False, True])
def test_readyz_promotes_kompress_from_module_cache(monkeypatch, attached):
    model = object()
    monkeypatch.setattr(
        kompress_compressor,
        "_kompress_cache",
        {kompress_compressor.HF_MODEL_ID: (model, object(), "onnx")},
    )
    compressor = _ReadyCompressor(ready=False) if attached else None
    app, proxy = _health_app(monkeypatch, compressor)
    proxy.warmup.kompress.info["source_status"] = "deferred"

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": True,
        "status": "healthy",
        "optional": True,
        "backend": "onnx",
    }
    assert proxy.warmup.kompress.handle is model
    if compressor is not None:
        assert compressor.calls == ["is_ready"]


def test_readyz_promotes_remote_kompress_backend(monkeypatch):
    compressor = _ReadyCompressor(backend="remote")
    app, proxy = _health_app(monkeypatch)
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = None
    router._kompress_remote = compressor

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"]["backend"] == "remote"
    assert payload["checks"]["kompress"]["ready"] is True


def test_readyz_never_starts_kompress_loading(monkeypatch):
    compressor = _ReadyCompressor()
    app, proxy = _health_app(monkeypatch, compressor)
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = compressor

    TestClient(app).get("/readyz")

    assert compressor.calls == ["is_ready", "ready_backend"]


def test_readyz_kompress_inspection_failure_fails_open(monkeypatch):
    compressor = _ReadyCompressor(error=RuntimeError("inspection failed"))
    app, proxy = _health_app(monkeypatch, compressor)
    proxy.warmup.kompress.mark_loaded(handle=object(), backend="onnx")

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"]["ready"] is True
    assert payload["checks"]["kompress"]["backend"] == "onnx"


def test_readyz_disabled_kompress_skips_inspection(monkeypatch):
    compressor = _ReadyCompressor()
    app, proxy = _health_app(monkeypatch, compressor, disabled=True)
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = compressor

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": False,
        "ready": True,
        "status": "disabled",
        "optional": True,
        "backend": None,
    }
    assert compressor.calls == []


def test_readyz_per_provider_kompress_override_reenables_health(monkeypatch):
    compressor = _ReadyCompressor()
    app, proxy = _health_app(
        monkeypatch,
        disabled=True,
        disable_kompress_anthropic=False,
    )
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = compressor

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": True,
        "status": "healthy",
        "optional": True,
        "backend": "onnx",
    }
    assert compressor.calls == ["is_ready", "ready_backend"]


