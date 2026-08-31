"""Tests for the declarative WrapTarget registry.

The registry replaces hand-written per-tool wrap commands for tools whose
integration is fully described by data: binary name, env keys pointed at the
proxy, and upstream defaults. These tests pin two things:

1. ``build_launch_env`` produces byte-identical env/display output to the
   hand-written command bodies it replaced (goose, openhands, openclaude).
2. The generated click commands exist under ``headroom wrap`` with the same
   names and option surface as before.
"""

from __future__ import annotations

from click.testing import CliRunner

from headroom.cli.wrap import wrap
from headroom.providers.wrap_registry import (
    WRAP_TARGETS,
    EnvVar,
    WrapTarget,
    build_launch_env,
    get_wrap_target,
)


class TestBuildLaunchEnv:
    def test_goose_env_matches_legacy_body(self):
        target = get_wrap_target("goose")
        env, display = build_launch_env(target, 8787, environ={})
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8787/v1"
        assert env["OPENAI_API_BASE"] == "http://127.0.0.1:8787/v1"
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
        # Legacy goose displayed OPENAI_BASE_URL and ANTHROPIC_BASE_URL only.
        assert display == [
            "OPENAI_BASE_URL=http://127.0.0.1:8787/v1",
            "ANTHROPIC_BASE_URL=http://127.0.0.1:8787",
        ]

    def test_openhands_env_matches_legacy_body(self):
        target = get_wrap_target("openhands")
        env, display = build_launch_env(target, 9000, environ={})
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"
        assert env["OPENAI_API_BASE"] == "http://127.0.0.1:9000/v1"
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
        assert env["LLM_BASE_URL"] == "http://127.0.0.1:9000/v1"
        assert display == [
            "OPENAI_BASE_URL=http://127.0.0.1:9000/v1",
            "ANTHROPIC_BASE_URL=http://127.0.0.1:9000",
            "LLM_BASE_URL=http://127.0.0.1:9000/v1",
        ]

    def test_openclaude_env_matches_legacy_aider_builder(self):
        target = get_wrap_target("openclaude")
        env, display = build_launch_env(target, 8787, environ={}, project="myproj")
        assert env["OPENAI_API_BASE"] == "http://127.0.0.1:8787/p/myproj/v1"
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787/p/myproj"
        assert display == [
            "OPENAI_API_BASE=http://127.0.0.1:8787/p/myproj/v1",
            "ANTHROPIC_BASE_URL=http://127.0.0.1:8787/p/myproj",
        ]

    def test_project_prefix_skipped_when_target_opts_out(self):
        target = get_wrap_target("goose")
        env, _ = build_launch_env(target, 8787, environ={}, project="myproj")
        # Legacy goose never encoded the project prefix.
        assert "/p/" not in env["OPENAI_BASE_URL"]

    def test_environ_is_copied_not_mutated(self):
        target = get_wrap_target("goose")
        source = {"PATH": "/bin"}
        env, _ = build_launch_env(target, 8787, environ=source)
        assert source == {"PATH": "/bin"}
        assert env["PATH"] == "/bin"


class TestBobTarget:
    def test_bob_env_is_bare_origin_with_project_prefix(self):
        target = get_wrap_target("bob")
        env, display = build_launch_env(target, 8788, environ={}, project="myproj")
        # Bare origin: Bob appends /inference/v1/... itself; a /v1 base would
        # produce a doubled prefix.
        assert env["BOB_GATEWAY_URL"] == "http://127.0.0.1:8788/p/myproj"
        assert display == ["BOB_GATEWAY_URL=http://127.0.0.1:8788/p/myproj"]

    def test_bob_upstream_carries_inference_suffix(self):
        target = get_wrap_target("bob")
        assert target.openai_api_url == "https://api.us-east.bob.ibm.com/inference/v1"

    def test_bob_declares_inference_chat_route(self):
        target = get_wrap_target("bob")
        assert "/inference/v1/chat/completions" in target.extra_chat_routes

    def test_registry_chat_routes_reach_openai_handler_routes(self):
        from headroom.providers.route_specs import OPENAI_HANDLER_ROUTES

        inference = [r for r in OPENAI_HANDLER_ROUTES if r.path == "/inference/v1/chat/completions"]
        assert len(inference) == 1
        assert inference[0].method == "POST"
        assert inference[0].handler_name == "handle_openai_chat"


class TestDefaultMode:
    """Registry-preferred HEADROOM_MODE: fills the gap when unset, never wins
    over an explicit user value."""

    def test_bob_declares_token_default_mode(self):
        assert get_wrap_target("bob").default_mode == "token"

    def test_other_targets_have_no_default_mode(self):
        for name in ("goose", "openhands", "openclaude"):
            assert get_wrap_target(name).default_mode is None

    @staticmethod
    def _invoke_bob(monkeypatch):
        import headroom.cli.wrap as wrap_mod

        monkeypatch.setattr(wrap_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
        captured: dict[str, str | None] = {}

        def fake_launch_tool(**kwargs):
            import os

            captured["mode"] = os.environ.get("HEADROOM_MODE")

        monkeypatch.setattr(wrap_mod, "_launch_tool", fake_launch_tool)
        result = CliRunner().invoke(wrap, ["bob"])
        assert result.exit_code == 0, result.output
        return captured

    def test_registry_default_fills_unset_headroom_mode(self, monkeypatch):
        # setenv-then-delenv registers restoration of the pre-test state even
        # though the command under test writes os.environ itself.
        monkeypatch.setenv("HEADROOM_MODE", "sentinel")
        monkeypatch.delenv("HEADROOM_MODE")
        captured = self._invoke_bob(monkeypatch)
        assert captured["mode"] == "token"

    def test_explicit_headroom_mode_wins_over_registry_default(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_MODE", "cache")
        captured = self._invoke_bob(monkeypatch)
        assert captured["mode"] == "cache"


class TestRegistry:
    def test_pilot_targets_registered(self):
        for name in ("goose", "openhands", "openclaude", "bob"):
            assert name in WRAP_TARGETS

    def test_get_unknown_target_raises(self):
        try:
            get_wrap_target("no-such-tool")
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError")

    def test_targets_are_immutable(self):
        target = get_wrap_target("goose")
        assert isinstance(target, WrapTarget)
        try:
            target.name = "other"  # type: ignore[misc]
        except AttributeError:
            pass
        else:
            raise AssertionError("expected frozen dataclass")

    def test_env_vars_have_valid_styles(self):
        for target in WRAP_TARGETS.values():
            for var in target.env_vars:
                assert isinstance(var, EnvVar)
                assert var.style in {"openai_v1", "anthropic", "bare_origin"}


class TestGeneratedCommands:
    def test_commands_exist_under_wrap_group(self):
        for name in ("goose", "openhands", "openclaude", "bob"):
            assert name in wrap.commands, f"wrap {name} missing"

    def test_generated_command_option_surface(self):
        params = {p.name for p in wrap.commands["goose"].params}
        for expected in (
            "port",
            "no_proxy",
            "learn",
            "memory",
            "backend",
            "anyllm_provider",
            "region",
            "verbose",
            "prepare_only",
            "code_graph",
        ):
            assert expected in params, f"goose lost option {expected}"

    def test_prepare_only_exits_cleanly(self):
        runner = CliRunner()
        for name in ("goose", "openhands", "openclaude", "bob"):
            result = runner.invoke(wrap, [name, "--prepare-only"])
            assert result.exit_code == 0, f"{name} --prepare-only failed: {result.output}"

    def test_help_mentions_env_keys(self):
        runner = CliRunner()
        result = runner.invoke(wrap, ["goose", "--help"])
        assert result.exit_code == 0
        assert "OPENAI_BASE_URL" in result.output


class TestOriginPassthrough:
    """Bob builds full gateway paths itself; catch-all must not re-prefix them.

    Regression for the 403 poll loop: base .../inference + inbound
    /inference/v1/model/info composed into a doubled prefix, and
    /admin/v1/profile was misrooted under /inference (#3360).
    """

    BASE = "https://api.us-east.bob.ibm.com/inference"

    def test_inference_path_is_origin_rooted_not_doubled(self):
        from headroom.providers.wrap_registry import resolve_origin_passthrough_url

        url = resolve_origin_passthrough_url(self.BASE, "/inference/v1/model/info")
        assert url == "https://api.us-east.bob.ibm.com/inference/v1/model/info"

    def test_admin_path_is_origin_rooted(self):
        from headroom.providers.wrap_registry import resolve_origin_passthrough_url

        url = resolve_origin_passthrough_url(self.BASE, "/admin/v1/profile")
        assert url == "https://api.us-east.bob.ibm.com/admin/v1/profile"

    def test_non_matching_path_falls_back(self):
        from headroom.providers.wrap_registry import resolve_origin_passthrough_url

        assert resolve_origin_passthrough_url(self.BASE, "/v1/embeddings") is None

    def test_non_matching_host_falls_back(self):
        from headroom.providers.wrap_registry import resolve_origin_passthrough_url

        assert (
            resolve_origin_passthrough_url("https://api.openai.com/v1", "/inference/v1/model/info")
            is None
        )

    def test_none_base_url_falls_back(self):
        from headroom.providers.wrap_registry import resolve_origin_passthrough_url

        assert resolve_origin_passthrough_url(None, "/inference/v1/model/info") is None


class TestOriginPassthroughResponseStrip:
    """Bob 2.0.1 rewrites its gateway host from region_domain in the proxied
    /admin/v1/profile response while keeping the proxy's port, pointing every
    later request at api.<region>:<proxy-port> (unreachable). The proxy strips
    the key so bob keeps using its configured gateway URL."""

    BASE = "https://api.us-east.bob.ibm.com/inference"

    def test_strips_region_domain_from_profile(self):
        import json

        from headroom.providers.wrap_registry import strip_origin_passthrough_response_keys

        body = json.dumps(
            {
                "profiles": [
                    {"id": "p1", "region": "us-east", "region_domain": "us-east.bob.ibm.com"}
                ]
            }
        ).encode()
        out = strip_origin_passthrough_response_keys(self.BASE, "/admin/v1/profile", body)
        assert out is not None
        payload = json.loads(out)
        assert "region_domain" not in payload["profiles"][0]
        assert payload["profiles"][0]["region"] == "us-east"

    def test_none_when_key_absent(self):
        from headroom.providers.wrap_registry import strip_origin_passthrough_response_keys

        assert (
            strip_origin_passthrough_response_keys(self.BASE, "/admin/v1/profile", b'{"id": "p1"}')
            is None
        )

    def test_none_for_undeclared_path(self):
        from headroom.providers.wrap_registry import strip_origin_passthrough_response_keys

        body = b'{"region_domain": "us-east.bob.ibm.com"}'
        assert (
            strip_origin_passthrough_response_keys(self.BASE, "/inference/v1/model/info", body)
            is None
        )

    def test_none_for_other_host(self):
        from headroom.providers.wrap_registry import strip_origin_passthrough_response_keys

        body = b'{"region_domain": "us-east.bob.ibm.com"}'
        assert (
            strip_origin_passthrough_response_keys(
                "https://api.openai.com/v1", "/admin/v1/profile", body
            )
            is None
        )

    def test_none_for_non_json_body(self):
        from headroom.providers.wrap_registry import strip_origin_passthrough_response_keys

        assert (
            strip_origin_passthrough_response_keys(self.BASE, "/admin/v1/profile", b"<html>403")
            is None
        )
