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


def test_readyz_excludes_kompress_from_aggregate_readiness(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
        )
    )
    app.state.ready = True
    proxy = app.state.proxy
    proxy.http_client = object()
    proxy.warmup.kompress.mark_error("model not cached")

    client = TestClient(app)
    response = client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["status"] == "healthy"
    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": False,
        "status": "degraded",
        "optional": True,
        "backend": None,
        "detail": "model not cached",
    }


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
        "detail": "loaded",
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
        "detail": "loaded",
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


def test_readyz_keeps_pending_kompress_unloaded(monkeypatch):
    compressor = _ReadyCompressor(backend="onnx", ready=False)
    app, proxy = _health_app(monkeypatch, compressor)
    router = proxy.anthropic_pipeline.transforms[-1]
    router._kompress = compressor

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": False,
        "status": "degraded",
        "optional": True,
        "backend": None,
        "detail": "not installed",
    }
    assert compressor.calls == ["is_ready"]


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
        "detail": "not installed",
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
        "detail": "loaded",
    }
    assert compressor.calls == ["is_ready", "ready_backend"]


def test_readyz_never_calls_lazy_kompress_getters(monkeypatch):
    app, proxy = _health_app(monkeypatch)
    router = proxy.anthropic_pipeline.transforms[-1]

    def _boom():
        raise AssertionError("health should not instantiate kompress")

    router._get_kompress = _boom
    router._get_remote_kompress = _boom

    payload = TestClient(app).get("/readyz").json()

    assert payload["checks"]["kompress"] == {
        "enabled": True,
        "ready": False,
        "status": "degraded",
        "optional": True,
        "backend": None,
        "detail": "not installed",
    }


@pytest.mark.parametrize(
    ("slot_status", "compressor", "disabled", "expected"),
    [
        (
            "null",
            None,
            False,
            {
                "enabled": True,
                "ready": False,
                "status": "degraded",
                "optional": True,
                "backend": None,
                "detail": "not installed",
            },
        ),
        (
            "null",
            _ReadyCompressor(),
            False,
            {
                "enabled": True,
                "ready": True,
                "status": "healthy",
                "optional": True,
                "backend": "onnx",
                "detail": "loaded",
            },
        ),
        (
            "null",
            _ReadyCompressor(backend="remote"),
            False,
            {
                "enabled": True,
                "ready": True,
                "status": "healthy",
                "optional": True,
                "backend": "remote",
                "detail": "loaded",
            },
        ),
        (
            "error",
            _ReadyCompressor(),
            False,
            {
                "enabled": True,
                "ready": True,
                "status": "healthy",
                "optional": True,
                "backend": "onnx",
                "detail": "loaded",
            },
        ),
        (
            "loaded",
            _ReadyCompressor(),
            False,
            {
                "enabled": True,
                "ready": True,
                "status": "healthy",
                "optional": True,
                "backend": "existing",
                "detail": "loaded",
            },
        ),
        (
            "null",
            _ReadyCompressor(),
            True,
            {
                "enabled": False,
                "ready": True,
                "status": "disabled",
                "optional": True,
                "backend": None,
                "detail": "not installed",
            },
        ),
    ],
)
def test_readyz_kompress_state_matrix(monkeypatch, slot_status, compressor, disabled, expected):
    app, proxy = _health_app(monkeypatch, compressor, disabled=disabled)
    if slot_status == "error":
        proxy.warmup.kompress.mark_error("not cached")
    elif slot_status == "loaded":
        proxy.warmup.kompress.mark_loaded(handle=object(), backend=expected["backend"])
    if compressor is not None and expected["backend"] == "existing":
        compressor.backend = "new"

    payload = TestClient(app).get("/readyz").json()["checks"]["kompress"]

    assert payload == expected


# /health and /readyz are auth-exempt, so nothing they serialize may carry raw
# exception text: loader failures embed absolute paths, model ids and URLs.
_SENTINEL_DETAIL = "/opt/secret-path/model.onnx token=SENTINEL-LEAK-9f3a"


@pytest.mark.parametrize("endpoint", ["/health", "/readyz"])
def test_public_health_does_not_leak_warm_failure_text(monkeypatch, endpoint):
    app, proxy = _health_app(monkeypatch)
    # What the background warm thread would have written pre-#2799 review.
    proxy.warmup.kompress.info["detail"] = f"warm failed: {_SENTINEL_DETAIL}"

    response = TestClient(app).get(endpoint)

    assert response.json()["checks"]["kompress"]["detail"] == "unavailable"
    assert "SENTINEL-LEAK-9f3a" not in response.text
    assert "secret-path" not in response.text


@pytest.mark.parametrize("endpoint", ["/health", "/readyz"])
def test_public_health_does_not_leak_slot_error_text(monkeypatch, endpoint):
    app, proxy = _health_app(monkeypatch)
    proxy.warmup.kompress.mark_error(f"native load failed: {_SENTINEL_DETAIL}")

    response = TestClient(app).get(endpoint)

    assert response.json()["checks"]["kompress"]["detail"] == "unavailable"
    assert "SENTINEL-LEAK-9f3a" not in response.text
    assert "secret-path" not in response.text
