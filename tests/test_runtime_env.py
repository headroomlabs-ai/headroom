"""Tests for the live runtime-env registry, override store, hot-reload endpoint,
and the wrap-side push that keeps a reused proxy in sync without a restart.
"""

from __future__ import annotations

import pytest

from headroom.proxy import helpers as proxy_helpers
from headroom.proxy import runtime_env as rt

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.rollout import resolve_rollout  # noqa: E402


def _json_object_of_size(total_bytes: int) -> bytes:
    """Build ``{"BOGUS":"...padding..."}`` at an exact byte length.

    ``BOGUS`` is not a registered knob, so a request built with this is
    expected to be *accepted* (200, ``applied == {}``) once it clears the
    size gate -- it isolates the size check from override semantics.
    """
    prefix, suffix = b'{"BOGUS":"', b'"}'
    pad_len = total_bytes - len(prefix) - len(suffix)
    assert pad_len >= 0, f"{total_bytes} bytes is too small for the JSON skeleton"
    return prefix + b"x" * pad_len + suffix


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
        "HEADROOM_VERBOSITY_LEVEL": "   ",  # blank -> not "explicitly set"
        "PATH": "/usr/bin",  # not a knob
    }
    assert rt.explicit_env(environ) == {
        "HEADROOM_OUTPUT_SHAPER": "1",
    }


def test_effective_runtime_env_reports_override_or_none(monkeypatch):
    rt.set_overrides({"HEADROOM_OUTPUT_SHAPER": "1"})
    eff = rt.effective_runtime_env()
    assert eff["HEADROOM_OUTPUT_SHAPER"] == "1"  # from override
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


@pytest.mark.parametrize(
    ("rollout", "expected_enabled", "expected_reason"),
    [
        (resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": "beta"}), True, "legacy_alias"),
        # Was ``(resolve_rollout({}), False, "blocked_by_channel")`` while
        # ``proxy_output_shaper`` was BETA: on the default channel the admin
        # POST could not enable it. The feature is now STABLE and on by
        # default, so the same POST is honoured. The escalation-refusal
        # property this case used to cover cannot be reproduced through this
        # endpoint any more — ``/admin/runtime-env`` re-resolves exactly one
        # rollout alias, ``HEADROOM_OUTPUT_SHAPER`` (see server.py), so there
        # is no second, still-gated feature to point it at. Channel gating
        # itself stays covered in test_rollout.py.
        (resolve_rollout({}), True, "legacy_alias"),
        (
            resolve_rollout(
                {
                    "HEADROOM_ROLLOUT_CHANNEL": "beta",
                    "HEADROOM_DISABLE_FEATURES": "proxy_output_shaper",
                }
            ),
            False,
            "disabled",
        ),
    ],
)
def test_admin_runtime_env_reresolves_running_rollout_without_weakening_policy(
    rollout, expected_enabled, expected_reason
):
    app = create_app(
        ProxyConfig(
            rollout=rollout,
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        before = client.get("/stats?cached=1").json()["rollout"]
        response = client.post("/admin/runtime-env", json={"HEADROOM_OUTPUT_SHAPER": "1"})
        after = client.get("/stats?cached=1").json()["rollout"]

    decision = next(item for item in after["features"] if item["name"] == "proxy_output_shaper")
    assert response.status_code == 200
    assert response.json()["rollout"] == after
    assert decision["enabled"] is expected_enabled
    assert decision["decision"] == expected_reason
    assert after["snapshot_digest"] != before["snapshot_digest"]


def test_admin_runtime_env_rejects_non_object(loopback_client):
    resp = loopback_client.post("/admin/runtime-env", json=["not", "a", "dict"])
    assert resp.status_code == 400


def test_admin_runtime_env_rejects_process_local_update_with_multiple_workers(monkeypatch):
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    rollout = resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": "beta"})
    config = ProxyConfig(
        worker_processes=2,
        rollout=rollout,
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    before_digest = rollout.snapshot_digest

    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        response = client.post("/admin/runtime-env", json={"HEADROOM_OUTPUT_SHAPER": "1"})
        after = client.get("/stats").json()["rollout"]

    assert response.status_code == 409
    assert response.json()["worker_processes"] == 2
    assert "restart" in response.json()["error"]
    assert rt.getenv("HEADROOM_OUTPUT_SHAPER") is None
    assert after["snapshot_digest"] == before_digest


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
# request body size limit (#3479)
#
# ``MAX_REQUEST_BODY_SIZE`` lives in ``headroom.proxy.helpers`` -- server.py
# only re-exports the name -- so patching ``server.MAX_REQUEST_BODY_SIZE``
# would not affect the check the reader actually runs. Patch the helpers
# module directly.
# ---------------------------------------------------------------------------


def test_admin_runtime_env_rejects_oversized_body(loopback_client, monkeypatch):
    monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 128)
    resp = loopback_client.post(
        "/admin/runtime-env", json={"HEADROOM_OUTPUT_SHAPER": "x" * 1000}
    )
    assert resp.status_code == 413
    assert rt.getenv("HEADROOM_OUTPUT_SHAPER") is None  # rejected before applying


def test_admin_runtime_env_accepts_body_at_the_cap(loopback_client, monkeypatch):
    monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 128)
    body = _json_object_of_size(128)
    resp = loopback_client.post(
        "/admin/runtime-env", content=body, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] == {}


def test_admin_runtime_env_rejects_body_one_byte_over_the_cap(loopback_client, monkeypatch):
    monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 128)
    body = _json_object_of_size(129)
    resp = loopback_client.post(
        "/admin/runtime-env", content=body, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 413


async def test_admin_runtime_env_rejects_oversized_chunked_body(monkeypatch):
    """An oversized body with no trustworthy Content-Length must still be refused.

    The request body is handed to httpx as an async generator, which httpx
    sends without a Content-Length header -- the exact case a Content-Length
    pre-check alone cannot catch. This proves the streaming read itself
    enforces the cap.
    """
    import httpx

    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setattr(proxy_helpers, "MAX_REQUEST_BODY_SIZE", 128)
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)

    async def body_stream():
        yield b'{"BOGUS":"'
        yield b"x" * 1000
        yield b'"}'

    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        resp = await client.post(
            "/admin/runtime-env",
            content=body_stream(),
            headers={"content-type": "application/json"},
        )

    assert "content-length" not in resp.request.headers
    assert resp.status_code == 413
    assert rt.getenv("HEADROOM_OUTPUT_SHAPER") is None


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
