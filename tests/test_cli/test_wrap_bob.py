"""Tests for `headroom wrap bob` (IBM Bob CLI) and the registry seams it uses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

import headroom.cli.wrap as wrap_mod
from headroom.cli.wrap import _warn_proxy_mode_mismatch, wrap
from headroom.providers.route_specs import OPENAI_HANDLER_ROUTES
from headroom.providers.wrap_registry import (
    WRAP_TARGETS,
    bob_preflight,
    build_launch_env,
    resolve_origin_passthrough_url,
    strip_origin_passthrough_response_keys,
)

BOB = WRAP_TARGETS["bob"]
BASE = "https://api.us-east.bob.ibm.com/inference"


def test_env_is_bare_origin_with_project_prefix():
    # Bob appends /inference/v1/... itself; a /v1 base would double the prefix.
    env, display = build_launch_env(BOB, 8788, environ={}, project="myproj")
    assert env["BOB_GATEWAY_URL"] == "http://127.0.0.1:8788/p/myproj"
    assert display == ["BOB_GATEWAY_URL=http://127.0.0.1:8788/p/myproj"]


def test_inference_chat_route_reaches_openai_handler():
    routes = [r for r in OPENAI_HANDLER_ROUTES if r.path == "/inference/v1/chat/completions"]
    assert [(r.method, r.handler_name) for r in routes] == [("POST", "handle_openai_chat")]


class TestLaunch:
    @pytest.fixture(autouse=True)
    def _isolated_bob_home(self, monkeypatch, tmp_path):
        # The preflight reads ~/.bob/settings/settings.json; never the developer's.
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

    @staticmethod
    def _invoke(monkeypatch, expect_exit: int = 0):
        monkeypatch.setattr(wrap_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        captured: dict = {}

        def fake_launch_tool(**kwargs):
            captured.update(kwargs, mode=os.environ.get("HEADROOM_MODE"))

        monkeypatch.setattr(wrap_mod, "_launch_tool", fake_launch_tool)
        result = CliRunner().invoke(wrap, ["bob", "--", "run", "fix it"])
        assert result.exit_code == expect_exit, result.output
        return captured if expect_exit == 0 else result.output

    def test_hands_proxy_the_inference_upstream(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_MODE", raising=False)
        captured = self._invoke(monkeypatch)
        # /inference/v1: the proxy strips /v1 and handle_openai_chat re-appends
        # /v1/chat/completions, composing back into the path IBM serves.
        assert captured["openai_api_url"] == "https://api.us-east.bob.ibm.com/inference/v1"
        assert captured["args"] == ("run", "fix it")

    def test_default_mode_fills_unset_headroom_mode(self, monkeypatch):
        # setenv-then-delenv registers restoration even though the command
        # writes os.environ itself.
        monkeypatch.setenv("HEADROOM_MODE", "sentinel")
        monkeypatch.delenv("HEADROOM_MODE")
        assert self._invoke(monkeypatch)["mode"] == "token"

    def test_explicit_headroom_mode_wins(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_MODE", "cache")
        assert self._invoke(monkeypatch)["mode"] == "cache"

    def test_saved_gateway_url_aborts_before_launch(self, monkeypatch, tmp_path):
        # bobshell re-resolves settings.gatewayUrl over BOB_GATEWAY_URL at startup,
        # so launching would run Bob uncompressed behind a banner saying otherwise.
        settings = tmp_path / ".bob" / "settings" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({"gatewayUrl": "https://api.eu-de.bob.ibm.com"}))
        output = self._invoke(monkeypatch, expect_exit=1)
        # Exact ClickException text; a substring check on the URL reads to
        # CodeQL as URL sanitization.
        expected = bob_preflight({"BOB_GATEWAY_URL": "http://127.0.0.1:8787"}, settings)
        assert output.strip() == f"Error: {expected}"


class TestBobPreflight:
    def _settings(self, tmp_path, payload) -> Path:
        path = tmp_path / "settings.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return path

    ENV = {"BOB_GATEWAY_URL": "http://127.0.0.1:8787/p/myproj"}

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"gatewayUrl": ""},
            {"gatewayUrl": None},
            {"gatewayUrl": "http://127.0.0.1:8787/p/myproj/"},  # already the proxy
            "not json",
        ],
    )
    def test_passes(self, tmp_path, payload):
        assert bob_preflight(self.ENV, self._settings(tmp_path, payload)) is None

    def test_passes_when_no_settings_file(self, tmp_path):
        assert bob_preflight(self.ENV, tmp_path / "missing.json") is None

    def test_fails_on_foreign_gateway(self, tmp_path):
        path = self._settings(tmp_path, {"gatewayUrl": "https://api.eu-de.bob.ibm.com"})
        message = bob_preflight(self.ENV, path)
        assert message == (
            "Bob's saved gatewayUrl (https://api.eu-de.bob.ibm.com) overrides "
            "BOB_GATEWAY_URL, so Bob would bypass the Headroom proxy. Remove the "
            f"gatewayUrl entry from {path} (or set it to the proxy URL shown by this "
            "wrap) and retry. If your organisation enforces a GatewayUrl policy, Bob "
            "cannot be wrapped."
        )


class TestOriginPassthrough:
    """Bob builds full gateway paths itself; the catch-all must not re-prefix
    them. Regression for the 403 loop: base .../inference + inbound
    /inference/v1/model/info doubled the prefix, and /admin/v1/profile was
    misrooted under /inference."""

    @pytest.mark.parametrize(
        "path",
        [
            "/inference/v1/model/info",
            "/inference/v1/embeddings",
            "/admin/v1/profile",
            "/admin/v1/teams/t1/users/u1",
            "/rag/v1/search",  # IBM docs tools (search_ibm_docs)
            "/metrics-forwarder/v1/codeagent/core/metrics",
        ],
    )
    def test_declared_paths_are_origin_rooted(self, path):
        assert (
            resolve_origin_passthrough_url(BASE, path) == f"https://api.us-east.bob.ibm.com{path}"
        )

    @pytest.mark.parametrize(
        ("base", "path"),
        [
            (BASE, "/v1/embeddings"),
            ("https://api.openai.com/v1", "/inference/v1/model/info"),
            (None, "/inference/v1/model/info"),
        ],
    )
    def test_everything_else_falls_back(self, base, path):
        assert resolve_origin_passthrough_url(base, path) is None


class TestResponseStrip:
    """Bob 2.0.1 rewrites its gateway host from region_domain in the proxied
    /admin/v1/profile response while keeping the proxy's port; stripping the
    key keeps it on its configured gateway URL (the proxy)."""

    def test_strips_region_domain_from_profile(self):
        body = json.dumps(
            {"profiles": [{"id": "p1", "region": "us-east", "region_domain": "us-east.x"}]}
        ).encode()
        out = strip_origin_passthrough_response_keys(BASE, "/admin/v1/profile", body)
        assert out is not None
        assert json.loads(out) == {"profiles": [{"id": "p1", "region": "us-east"}]}

    @pytest.mark.parametrize(
        ("base", "path", "body"),
        [
            (BASE, "/admin/v1/profile", b'{"id": "p1"}'),  # key absent
            (BASE, "/inference/v1/model/info", b'{"region_domain": "x"}'),  # undeclared path
            ("https://api.openai.com/v1", "/admin/v1/profile", b'{"region_domain": "x"}'),
            (BASE, "/admin/v1/profile", b"<html>403"),  # not JSON
        ],
    )
    def test_none_when_nothing_to_strip(self, base, path, body):
        assert strip_origin_passthrough_response_keys(base, path, body) is None


class TestModeMismatchWarning:
    @staticmethod
    def _warnings(monkeypatch, running_config, requested=None) -> list[str]:
        if requested is None:
            monkeypatch.delenv("HEADROOM_MODE", raising=False)
        else:
            monkeypatch.setenv("HEADROOM_MODE", requested)
        lines: list[str] = []
        monkeypatch.setattr("click.echo", lines.append)
        _warn_proxy_mode_mismatch(running_config)
        return lines

    def test_warns_on_mismatch(self, monkeypatch):
        (line,) = self._warnings(monkeypatch, {"mode": "cache"}, requested="token")
        assert "'token' mode" in line and "'cache' mode" in line

    @pytest.mark.parametrize(
        ("running_config", "requested"),
        [
            ({"mode": "cache"}, "cache"),
            ({"mode": "cache"}, "cost_savings"),  # alias of cache
            ({"mode": "token"}, None),  # nothing requested
            ({}, "token"),  # pre-upgrade proxy without the field
            (None, "token"),  # config unavailable
        ],
    )
    def test_silent_otherwise(self, monkeypatch, running_config, requested):
        assert self._warnings(monkeypatch, running_config, requested) == []


def test_passthrough_handler_roots_profile_at_origin_and_strips_region_domain():
    import asyncio
    from types import SimpleNamespace

    import httpx

    from headroom.proxy.handlers.openai import OpenAIHandlerMixin

    class _Upstream:
        calls: list[str] = []

        async def request(self, **kwargs):
            self.calls.append(kwargs["url"])
            return httpx.Response(
                200,
                request=httpx.Request(kwargs["method"], kwargs["url"]),
                json={"id": "p1", "region_domain": "us-east.bob.ibm.com"},
                # Validators/integrity metadata describe the unfiltered bytes.
                headers={
                    "ETag": '"upstream-v1"',
                    "Last-Modified": "Mon, 28 Sep 2026 00:00:00 GMT",
                    "Cache-Control": "max-age=60",
                    "Content-Digest": "sha-256=:dW5maWx0ZXJlZA==:",
                    "Digest": "SHA-256=dW5maWx0ZXJlZA==",
                    "X-Request-Id": "req-1",
                },
            )

    class _ProfileRequest:
        method = "GET"
        headers: dict[str, str] = {}
        url = SimpleNamespace(path="/admin/v1/profile", query="")

        async def body(self) -> bytes:
            return b""

    handler = object.__new__(OpenAIHandlerMixin)
    handler.http_client = _Upstream()

    response = asyncio.run(handler.handle_passthrough(_ProfileRequest(), BASE))

    assert handler.http_client.calls == ["https://api.us-east.bob.ibm.com/admin/v1/profile"]
    assert response.status_code == 200
    assert json.loads(response.body) == {"id": "p1"}
    forwarded = {k.lower() for k in response.headers}
    stale = {"etag", "last-modified", "cache-control", "content-digest", "digest"}
    assert not forwarded & stale, "filtered body must not carry the upstream's validators"
    assert response.headers["x-request-id"] == "req-1"
    assert response.headers["content-type"] == "application/json"


class TestModeWarningOnEveryReusePath:
    """Bob requests token mode; every path that hands it an existing proxy warns
    when that proxy runs a different mode, not only the ordinary reuse return."""

    @pytest.fixture
    def cache_mode_proxy(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_MODE", "token")
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_foreign_listener", lambda _p: False)
        monkeypatch.setattr(
            wrap_mod, "_query_proxy_health", lambda _p: {"config": {"mode": "cache"}}
        )
        lines: list[str] = []
        monkeypatch.setattr("click.echo", lambda msg="", *a, **k: lines.append(str(msg)))
        return lines

    def test_no_proxy_warns(self, cache_mode_proxy):
        assert wrap_mod._ensure_proxy_unlocked(8787, True) == (None, 8787)
        assert any("'token' mode" in line for line in cache_mode_proxy)

    def test_recovered_persistent_proxy_warns(self, monkeypatch, cache_mode_proxy):
        from types import SimpleNamespace

        import headroom.install.health as install_health

        manifest = SimpleNamespace(profile="p", health_url="http://127.0.0.1:8787/readyz")
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: manifest)
        monkeypatch.setattr(install_health, "probe_ready", lambda _url: False)
        monkeypatch.setattr(wrap_mod, "_recover_persistent_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_routing_mismatches", lambda *_a, **_k: [])

        assert wrap_mod._ensure_proxy_unlocked(8787, False) == (None, 8787)
        assert any("'token' mode" in line for line in cache_mode_proxy)
