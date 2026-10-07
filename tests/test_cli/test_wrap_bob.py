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


class TestSavedGatewayGuard:
    """bobshell re-resolves settings.gatewayUrl over BOB_GATEWAY_URL at startup,
    so launching would run Bob uncompressed behind a banner saying otherwise.
    The guard must judge the URL Bob actually receives, i.e. after _ensure_proxy
    may have fallen back to another port."""

    @pytest.fixture
    def saved(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        settings = tmp_path / ".bob" / "settings" / "settings.json"
        settings.parent.mkdir(parents=True)

        def _write(gateway_url: str) -> Path:
            settings.write_text(json.dumps({"gatewayUrl": gateway_url}))
            return settings

        return _write

    @staticmethod
    def _launch(monkeypatch, tmp_path, *, actual_port: int):
        """Run `wrap bob` for real up to the child spawn, with the proxy stubbed
        to come up on ``actual_port`` (8787 requested)."""
        import sys
        from unittest.mock import patch

        monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: sys.executable)
        monkeypatch.setattr(wrap_mod, "_project_name_from_cwd", lambda: "proj")
        marker = tmp_path / "child.txt"
        child = (
            "import os; from pathlib import Path; "
            f"Path({str(marker)!r}).write_text(os.environ['BOB_GATEWAY_URL'])"
        )
        with (
            patch.object(wrap_mod, "_make_cleanup", return_value=lambda: None),
            patch.object(wrap_mod.signal, "signal"),
            patch.object(wrap_mod, "_register_proxy_client"),
            patch.object(wrap_mod, "_unregister_proxy_client"),
            patch.object(wrap_mod, "_push_runtime_env"),
            patch.object(wrap_mod, "_ensure_proxy", return_value=(None, actual_port)),
            patch.object(wrap_mod, "_configure_quiet_cli_env", return_value=[]),
        ):
            result = CliRunner().invoke(wrap, ["bob", "--port", "8787", "--", "-c", child])
        return result, marker

    def test_foreign_gateway_aborts(self, monkeypatch, tmp_path, saved):
        settings = saved("https://api.eu-de.bob.ibm.com")
        result, marker = self._launch(monkeypatch, tmp_path, actual_port=8787)
        assert result.exit_code == 1
        # Exact ClickException text; a substring check on the URL reads to
        # CodeQL as URL sanitization.
        expected = bob_preflight({"BOB_GATEWAY_URL": "http://127.0.0.1:8787"}, settings)
        assert result.output.strip().endswith(f"Error: {expected}")
        assert not marker.exists(), "Bob must not be spawned"

    def test_saved_requested_port_aborts_after_fallback(self, monkeypatch, tmp_path, saved):
        # Saved URL matches the requested port, but the proxy came up on 8899:
        # Bob would ignore BOB_GATEWAY_URL and talk to a port with no proxy.
        saved("http://127.0.0.1:8787/p/proj")
        result, marker = self._launch(monkeypatch, tmp_path, actual_port=8899)
        assert result.exit_code == 1 and "overrides BOB_GATEWAY_URL" in result.output
        assert not marker.exists()

    def test_saved_fallback_port_launches(self, monkeypatch, tmp_path, saved):
        # The inverse: saved URL already names the port the proxy ended up on.
        saved("http://127.0.0.1:8899/p/proj")
        result, marker = self._launch(monkeypatch, tmp_path, actual_port=8899)
        assert result.exit_code == 0, result.output
        assert marker.read_text() == "http://127.0.0.1:8899/p/proj"


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
            {"gatewayUrl": "http://127.0.0.1:8787/p/another-project"},  # attribution only
            {"gatewayUrl": "http://localhost:8787"},  # same proxy, spelled differently
            "not json",
        ],
    )
    def test_passes(self, tmp_path, payload):
        assert bob_preflight(self.ENV, self._settings(tmp_path, payload)) is None

    def test_passes_when_no_settings_file(self, tmp_path):
        assert bob_preflight(self.ENV, tmp_path / "missing.json") is None

    def test_fails_on_same_host_other_port(self, tmp_path):
        path = self._settings(tmp_path, {"gatewayUrl": "http://127.0.0.1:9999"})
        assert bob_preflight(self.ENV, path) is not None

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

    def test_other_ibm_regions_get_the_same_rules(self):
        # IBM runs one gateway per region; --openai-api-url selects it.
        base = "https://api.eu-de.bob.ibm.com/inference"
        assert (
            resolve_origin_passthrough_url(base, "/admin/v1/profile")
            == "https://api.eu-de.bob.ibm.com/admin/v1/profile"
        )
        body = b'{"instances":[{"teams":[{"id":"t","region_domain":"eu-de.bob.ibm.com"}]}]}'
        assert json.loads(
            strip_origin_passthrough_response_keys(base, "/admin/v1/profile", body)
        ) == {"instances": [{"teams": [{"id": "t"}]}]}
        # The web-login host is not a gateway.
        assert resolve_origin_passthrough_url("https://bob.ibm.com", "/admin/v1/profile") is None

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


class TestStartupCompressionMismatchWarning:
    @pytest.fixture(autouse=True)
    def _session_env(self, monkeypatch):
        for name in ("HEADROOM_MODE", "HEADROOM_MIN_TOKENS", "HEADROOM_EXCLUDE_TOOLS"):
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.parametrize(
        ("name", "value", "config"),
        [
            ("HEADROOM_MIN_TOKENS", "2000", {"min_tokens_to_crush": 120}),
            ("HEADROOM_MIN_TOKENS", "0", {"min_tokens_to_crush": 120}),
            ("HEADROOM_EXCLUDE_TOOLS", "Bash,Read", {"exclude_tools": []}),
            ("HEADROOM_EXCLUDE_TOOLS", "", {"exclude_tools": ["Bash"]}),
        ],
    )
    def test_warns_when_reuse_ignores_explicit_compression_settings(
        self, monkeypatch, capsys, name, value, config
    ):
        monkeypatch.setenv(name, value)
        _warn_proxy_mode_mismatch(config)
        output = capsys.readouterr().out
        assert name in output
        assert "Restart" in output and "--port" in output

    @pytest.mark.parametrize(
        ("env", "config"),
        [
            ({}, {"min_tokens_to_crush": 120, "exclude_tools": ["Bash"]}),
            ({"HEADROOM_MIN_TOKENS": "120"}, {"min_tokens_to_crush": 120}),
            ({"HEADROOM_MIN_TOKENS": "invalid"}, {"min_tokens_to_crush": 120}),
            ({"HEADROOM_MIN_TOKENS": "2000"}, {}),
            (
                {"HEADROOM_EXCLUDE_TOOLS": " Read, BASH, bash "},
                {"exclude_tools": ["Bash", "bash"]},
            ),
            ({"HEADROOM_EXCLUDE_TOOLS": "Read"}, {"exclude_tools": []}),
            ({"HEADROOM_EXCLUDE_TOOLS": "Bash"}, {}),
            ({"HEADROOM_EXCLUDE_TOOLS": "Bash"}, {"exclude_tools": None}),
        ],
    )
    def test_does_not_warn_for_matching_unrequested_or_unknown_settings(
        self, monkeypatch, capsys, env, config
    ):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        _warn_proxy_mode_mismatch(config)
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        ("name", "value"),
        [("HEADROOM_MIN_TOKENS", "2000"), ("HEADROOM_EXCLUDE_TOOLS", "Bash")],
    )
    def test_no_start_reuse_warns_without_an_explicit_mode(self, monkeypatch, capsys, name, value):
        monkeypatch.setenv(name, value)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: True)
        monkeypatch.setattr(
            wrap_mod,
            "_query_proxy_health",
            lambda _port: {"config": {"min_tokens_to_crush": 120, "exclude_tools": []}},
        )

        wrap_mod._ensure_proxy_unlocked(18795, True)

        assert name in capsys.readouterr().out


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

    def test_stale_version_left_running_for_attached_clients_warns(
        self, monkeypatch, cache_mode_proxy
    ):
        # Shared proxy on an older Headroom with other wrappers attached is
        # deliberately left running; Bob still ends up on its cache mode.
        monkeypatch.setattr(
            wrap_mod,
            "_query_proxy_health",
            lambda _p: {"version": "0.0.1", "config": {"mode": "cache"}},
        )
        monkeypatch.setattr(wrap_mod, "_live_proxy_clients", lambda *_a, **_k: ["other-wrapper"])
        monkeypatch.setattr(wrap_mod, "_proxy_routing_mismatches", lambda *_a, **_k: [])
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: None)

        assert wrap_mod._ensure_proxy_unlocked(8787, False) == (None, 8787)
        assert any("'token' mode" in line for line in cache_mode_proxy)

    def test_persistent_restart_warns_with_the_restarted_config(
        self, monkeypatch, cache_mode_proxy
    ):
        from types import SimpleNamespace

        import headroom.install.health as install_health

        # Dormant persistent deployment: recovery brings it up without
        # --memory, so wrap restarts it from the manifest, in the manifest's
        # (cache) mode.
        manifest = SimpleNamespace(profile="p", health_url="http://127.0.0.1:8787/readyz")
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: manifest)
        monkeypatch.setattr(install_health, "probe_ready", lambda _url: False)
        monkeypatch.setattr(wrap_mod, "_recover_persistent_proxy", lambda _p: True)
        monkeypatch.setattr(wrap_mod, "_proxy_routing_mismatches", lambda *_a, **_k: [])
        restarted: list[int] = []
        monkeypatch.setattr(
            wrap_mod, "_restart_persistent_proxy", lambda _m, p: restarted.append(p) or True
        )
        monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda _p: {"mode": "cache"})

        assert wrap_mod._ensure_proxy_unlocked(8787, False, memory=True) == (None, 8787)
        assert restarted == [8787]
        assert any("'token' mode" in line for line in cache_mode_proxy)

    def test_feature_gap_without_pid_reused_as_is_warns(self, monkeypatch, cache_mode_proxy):
        # Proxy lacks --memory and exposes no PID, so wrap reuses it unchanged.
        monkeypatch.setattr(wrap_mod, "_proxy_needs_version_restart", lambda _p: False)
        monkeypatch.setattr(wrap_mod, "_live_proxy_clients", lambda *_a, **_k: [])
        monkeypatch.setattr(wrap_mod, "_proxy_routing_mismatches", lambda *_a, **_k: [])
        monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: None)

        assert wrap_mod._ensure_proxy_unlocked(8787, False, memory=True) == (None, 8787)
        assert any("Cannot restart automatically" in line for line in cache_mode_proxy)
        assert any("'token' mode" in line for line in cache_mode_proxy)
