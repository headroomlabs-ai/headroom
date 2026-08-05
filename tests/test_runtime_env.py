"""Tests for the live runtime-env registry, override store, hot-reload endpoint,
and the wrap-side push that keeps a reused proxy in sync without a restart.
"""

from __future__ import annotations

import pytest

from headroom.proxy import runtime_env as rt

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_runtime_env(monkeypatch):
    """Each test starts with no overrides and no knob env vars set."""
    for knob in rt.RUNTIME_ENV_KNOBS:
        monkeypatch.delenv(knob.env, raising=False)
    rt.clear_overrides()
    yield
    rt.clear_overrides()


# ---------------------------------------------------------------------------
# Registry + override store
# ---------------------------------------------------------------------------


def test_getenv_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    assert rt.getenv("HEADROOM_OUTPUT_SHAPER") == "1"
    assert rt.getenv("HEADROOM_VERBOSITY_LEVEL", "2") == "2"  # unset -> default
    assert rt.getenv("HEADROOM_VERBOSITY_LEVEL") is None


def test_getenv_override_wins_over_environment(monkeypatch):
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "0")
    rt.set_overrides({"HEADROOM_OUTPUT_SHAPER": "1"})
    assert rt.getenv("HEADROOM_OUTPUT_SHAPER") == "1"


def test_set_overrides_ignores_unknown_keys_and_non_strings():
    applied = rt.set_overrides(
        {
            "HEADROOM_OUTPUT_SHAPER": "1",
            "NOT_A_KNOB": "x",
            "HEADROOM_VERBOSITY_LEVEL": 3,  # non-string ignored
        }
    )
    assert applied == {"HEADROOM_OUTPUT_SHAPER": "1"}
    assert rt.getenv("NOT_A_KNOB") is None
    # The rejected non-string did not become an override.
    assert rt.getenv("HEADROOM_VERBOSITY_LEVEL") is None


def test_explicit_env_returns_only_explicitly_set_knobs():
    environ = {
        "HEADROOM_OUTPUT_SHAPER": "1",
        "HEADROOM_MECHANICAL_EFFORT": "low",
        "HEADROOM_VERBOSITY_LEVEL": "   ",  # blank -> not "explicitly set"
        "PATH": "/usr/bin",  # not a knob
    }
    assert rt.explicit_env(environ) == {
        "HEADROOM_OUTPUT_SHAPER": "1",
        "HEADROOM_MECHANICAL_EFFORT": "low",
    }


def test_effective_runtime_env_reports_override_or_none(monkeypatch):
    monkeypatch.setenv("HEADROOM_EFFORT_ROUTER", "0")
    rt.set_overrides({"HEADROOM_OUTPUT_SHAPER": "1"})
    eff = rt.effective_runtime_env()
    assert eff["HEADROOM_OUTPUT_SHAPER"] == "1"  # from override
    assert eff["HEADROOM_EFFORT_ROUTER"] == "0"  # from env
    assert eff["HEADROOM_VERBOSITY_LEVEL"] is None  # unset
    # Every registered knob is reported.
    assert set(eff) == {knob.env for knob in rt.RUNTIME_ENV_KNOBS}


def test_clear_overrides_resets(monkeypatch):
    rt.set_overrides({"HEADROOM_OUTPUT_SHAPER": "1"})
    rt.clear_overrides()
    assert rt.getenv("HEADROOM_OUTPUT_SHAPER") is None


# ---------------------------------------------------------------------------
# Overrides reach the live readers (the whole point)
# ---------------------------------------------------------------------------


def test_override_enables_output_shaper_without_env():
    from headroom.proxy.output_shaper import OutputShaperSettings

    assert OutputShaperSettings.from_env().enabled is False
    rt.set_overrides({"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "3"})
    settings = OutputShaperSettings.from_env()
    assert settings.enabled is True
    assert settings.verbosity_level == 3


def test_override_changes_astgrep_threshold_without_env():
    from headroom.proxy.interceptors import astgrep

    assert astgrep._min_chars_to_rewrite() == 500
    rt.set_overrides({"HEADROOM_INTERCEPT_READ_MIN_CHARS": "999"})
    assert astgrep._min_chars_to_rewrite() == 999
    # Bad value falls back to the documented default rather than raising.
    rt.set_overrides({"HEADROOM_INTERCEPT_READ_MIN_CHARS": "not-an-int"})
    assert astgrep._min_chars_to_rewrite() == 500


# ---------------------------------------------------------------------------
# /health surface + /admin/runtime-env hot-reload endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def loopback_client(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as c:
        yield c


def test_health_exposes_runtime_env(loopback_client):
    config = loopback_client.get("/health").json()["config"]
    assert "runtime_env" in config
    assert set(config["runtime_env"]) == {knob.env for knob in rt.RUNTIME_ENV_KNOBS}
    assert config["runtime_env"]["HEADROOM_OUTPUT_SHAPER"] is None


def test_admin_runtime_env_applies_and_reflects_in_health(loopback_client):
    resp = loopback_client.post(
        "/admin/runtime-env",
        json={"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "3", "BOGUS": "x"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] == {"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "3"}
    assert body["runtime_env"]["HEADROOM_OUTPUT_SHAPER"] == "1"
    # And it is observable on the live /health surface.
    health = loopback_client.get("/health").json()["config"]["runtime_env"]
    assert health["HEADROOM_OUTPUT_SHAPER"] == "1"
    assert health["HEADROOM_VERBOSITY_LEVEL"] == "3"


def test_admin_runtime_env_rejects_non_object(loopback_client):
    resp = loopback_client.post("/admin/runtime-env", json=["not", "a", "dict"])
    assert resp.status_code == 400


def test_admin_runtime_env_is_loopback_only():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app, base_url="http://127.0.0.1", client=("10.0.0.1", 54321)) as external:
        resp = external.post("/admin/runtime-env", json={"HEADROOM_OUTPUT_SHAPER": "1"})
    assert resp.status_code == 404  # invisible to non-loopback callers
    assert rt.getenv("HEADROOM_OUTPUT_SHAPER") is None  # nothing applied


# ---------------------------------------------------------------------------
# wrap-side push
# ---------------------------------------------------------------------------


def test_push_runtime_env_posts_explicit_env(monkeypatch):
    import urllib.request

    from headroom.cli import wrap

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_VERBOSITY_LEVEL", "3")

    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = request.data
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    wrap._push_runtime_env(8787, no_proxy=False)

    assert captured["url"] == "http://127.0.0.1:8787/admin/runtime-env"
    import json

    assert json.loads(captured["body"]) == {
        "HEADROOM_OUTPUT_SHAPER": "1",
        "HEADROOM_VERBOSITY_LEVEL": "3",
    }


def test_push_runtime_env_noop_when_nothing_set(monkeypatch):
    import urllib.request

    from headroom.cli import wrap

    def boom(*a, **k):  # must never be called
        raise AssertionError("should not POST when nothing is explicitly set")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    wrap._push_runtime_env(8787, no_proxy=False)  # no env set -> no-op


def test_push_runtime_env_noop_when_no_proxy(monkeypatch):
    import urllib.request

    from headroom.cli import wrap

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no POST"))
    )
    wrap._push_runtime_env(8787, no_proxy=True)  # --no-proxy -> no-op


def test_push_runtime_env_swallows_unreachable_proxy(monkeypatch):
    import urllib.request

    from headroom.cli import wrap

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")

    def refused(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    # Best-effort: an unreachable / old proxy must not raise.
    wrap._push_runtime_env(8787, no_proxy=False)
