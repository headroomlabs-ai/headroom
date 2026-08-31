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

        inference = [
            r for r in OPENAI_HANDLER_ROUTES if r.path == "/inference/v1/chat/completions"
        ]
        assert len(inference) == 1
        assert inference[0].method == "POST"
        assert inference[0].handler_name == "handle_openai_chat"


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
