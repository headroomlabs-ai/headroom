"""Tests for `headroom wrap bob` (IBM Bob CLI) and the registry seams it uses."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

import headroom.cli.wrap as wrap_mod
from headroom.cli.wrap import _warn_proxy_mode_mismatch, wrap
from headroom.providers.route_specs import OPENAI_HANDLER_ROUTES
from headroom.providers.wrap_registry import (
    WRAP_TARGETS,
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
    @staticmethod
    def _invoke(monkeypatch) -> dict:
        monkeypatch.setattr(wrap_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        captured: dict = {}

        def fake_launch_tool(**kwargs):
            captured.update(kwargs, mode=os.environ.get("HEADROOM_MODE"))

        monkeypatch.setattr(wrap_mod, "_launch_tool", fake_launch_tool)
        result = CliRunner().invoke(wrap, ["bob", "--", "run", "fix it"])
        assert result.exit_code == 0, result.output
        return captured

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


class TestOriginPassthrough:
    """Bob builds full gateway paths itself; the catch-all must not re-prefix
    them. Regression for the 403 loop: base .../inference + inbound
    /inference/v1/model/info doubled the prefix, and /admin/v1/profile was
    misrooted under /inference."""

    @pytest.mark.parametrize("path", ["/inference/v1/model/info", "/admin/v1/profile"])
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
