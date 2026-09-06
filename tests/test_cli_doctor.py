"""Tests for `headroom doctor`."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from click.testing import CliRunner

import headroom.cli.doctor as doctor_mod
from headroom.cli.doctor import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    check_budget,
    check_claude_desktop,
    check_claude_remote_control_gate,
    check_claude_routing,
    check_codex_routing,
    check_deployments,
    check_proxy_liveness,
    check_savings,
    check_shell_env,
    check_version_drift,
)
from headroom.cli.main import main
from headroom.providers.claude.runtime import remote_control_gate_message

LIVEZ_OK = {
    "service": "headroom-proxy",
    "status": "healthy",
    "alive": True,
    "version": "0.26.0",
    "uptime_seconds": 260135.0,
}

STATS_OK = {
    "persistent_savings": {
        "lifetime": {"tokens_saved": 17_583_102, "compression_savings_usd": 7.81701},
        "display_session": {"last_activity_at": "2026-06-12T12:00:00Z"},
    },
    "cost": {"budget_limit_usd": 10.0, "budget_period": "daily"},
}


class TestProxyLiveness:
    def test_down_is_fail_with_hint(self):
        result = check_proxy_liveness(None, "http://127.0.0.1:8787")
        assert result.status == FAIL
        assert "headroom proxy" in (result.hint or "")

    def test_up_mentions_version_and_uptime(self):
        result = check_proxy_liveness(LIVEZ_OK, "http://127.0.0.1:8787")
        assert result.status == PASS
        assert "v0.26.0" in result.summary
        assert "3d" in result.summary

    def test_up_leaves_source_label_unprefixed(self):
        livez = {**LIVEZ_OK, "version": "source-build+sha.abcdef123456"}
        result = check_proxy_liveness(livez, "http://127.0.0.1:8787")
        assert result.status == PASS
        assert "source-build+sha.abcdef123456" in result.summary
        assert "vsource-build" not in result.summary


class TestVersionDrift:
    def test_match_passes(self):
        assert check_version_drift(LIVEZ_OK, "0.26.0").status == PASS

    def test_mismatch_warns_with_restart_hint(self):
        result = check_version_drift(LIVEZ_OK, "0.27.0")
        assert result.status == WARN
        assert "drift" in result.summary
        assert "restart" in (result.hint or "")

    def test_proxy_down_skips(self):
        assert check_version_drift(None, "0.26.0").status == SKIP

    def test_unknown_version_warns(self):
        assert check_version_drift({"version": "unknown"}, "0.26.0").status == WARN
        assert check_version_drift(LIVEZ_OK, "unknown").status == WARN

    @pytest.mark.parametrize(
        ("running", "installed"),
        [
            ("source-build+g6266a1d774b5", "0.26.0"),
            ("source-build+sha.abcdef123456", "0.26.0"),
            ("6266a1d", "0.26.0"),
            ("0.26.0+gabcdef0", "0.26.0"),
            ("0.26.0", "source-build+sha.abcdef123456"),
        ],
    )
    def test_non_release_version_labels_skip_drift_comparison(self, running, installed):
        result = check_version_drift({"version": running}, installed)
        assert result.status == SKIP
        assert "drift" not in result.summary


class TestClaudeRouting:
    def test_missing_file_warns(self, tmp_path):
        result = check_claude_routing(tmp_path / "settings.json", 8787)
        assert result.status == WARN
        assert "wrap claude" in (result.hint or "")

    def test_malformed_json_warns(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("{not json", encoding="utf-8")
        assert check_claude_routing(path, 8787).status == WARN

    @pytest.mark.parametrize("body", ["[]", "null", "42", '"a string"'])
    def test_non_object_json_warns(self, tmp_path, body):
        # Valid JSON that isn't an object parses cleanly, so it slips past the
        # JSONDecodeError guard; the later payload.get("env") must not crash the
        # diagnostic command that is being run precisely because the config is
        # suspect.
        path = tmp_path / "settings.json"
        path.write_text(body, encoding="utf-8")
        assert check_claude_routing(path, 8787).status == WARN

    def test_no_env_key_warns(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"env": {}}), encoding="utf-8")
        assert check_claude_routing(path, 8787).status == WARN

    def test_correct_url_passes(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        assert check_claude_routing(path, 8787).status == PASS

    def test_port_mismatch_warns(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8788"}}),
            encoding="utf-8",
        )
        result = check_claude_routing(path, 8787)
        assert result.status == WARN
        assert "8788" in result.summary

    def test_non_headroom_url_warns(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gateway.corp.example/v1"}}),
            encoding="utf-8",
        )
        result = check_claude_routing(path, 8787)
        assert result.status == WARN
        assert "gateway.corp.example" in result.summary


class TestClaudeDesktop:
    def test_no_desktop_dir_produces_no_row(self, tmp_path):
        # #2925: absent Desktop -> no row, so it never contradicts a routed CLI.
        assert check_claude_desktop(tmp_path / "Claude") is None

    def test_desktop_present_warns_about_bypass(self, tmp_path):
        desktop = tmp_path / "Claude"
        desktop.mkdir()
        result = check_claude_desktop(desktop)
        assert result is not None
        assert result.name == "claude desktop"
        assert result.status == WARN
        assert "bypass" in result.summary
        assert "#869" in (result.hint or "")

    def test_doctor_appends_desktop_row_when_present(self, tmp_path, monkeypatch):
        # Integration: the entrypoint surfaces the Desktop row when detected.
        desktop = tmp_path / "Claude"
        desktop.mkdir()
        monkeypatch.setattr(doctor_mod, "claude_desktop_config_dir", lambda: desktop)
        monkeypatch.setattr(doctor_mod, "probe_json", lambda *a, **k: None)
        monkeypatch.setattr(doctor_mod, "list_manifests", lambda: [])
        result = CliRunner().invoke(main, ["doctor", "--json"])
        payload = json.loads(result.output)
        rows = {c["name"]: c for c in payload["checks"]}
        assert "claude desktop" in rows
        assert rows["claude desktop"]["status"] == WARN

    def test_doctor_omits_desktop_row_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(doctor_mod, "claude_desktop_config_dir", lambda: tmp_path / "Claude")
        monkeypatch.setattr(doctor_mod, "probe_json", lambda *a, **k: None)
        monkeypatch.setattr(doctor_mod, "list_manifests", lambda: [])
        result = CliRunner().invoke(main, ["doctor", "--json"])
        payload = json.loads(result.output)
        assert "claude desktop" not in {c["name"] for c in payload["checks"]}


class TestClaudeRemoteControlGate:
    def test_settings_custom_base_warns(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        result = check_claude_remote_control_gate(path, {})
        assert result is not None
        assert result.status == WARN
        assert remote_control_gate_message("ANTHROPIC_BASE_URL from settings") in result.summary

    def test_shell_env_custom_base_warns(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("{}", encoding="utf-8")
        result = check_claude_remote_control_gate(
            path, {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}
        )
        assert result is not None
        assert result.status == WARN
        assert remote_control_gate_message("ANTHROPIC_BASE_URL in shell") in result.summary

    def test_no_custom_base_no_warning(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}}),
            encoding="utf-8",
        )
        assert check_claude_remote_control_gate(path, {}) is None

    @pytest.mark.parametrize("body", ["[]", "null", "42"])
    def test_non_object_settings_does_not_crash(self, tmp_path, body):
        # A valid-but-non-object settings file parses past the JSONDecodeError
        # guard; payload.get("env") must not raise AttributeError. The shell env
        # still drives the gate, so a custom base there still warns.
        path = tmp_path / "settings.json"
        path.write_text(body, encoding="utf-8")
        result = check_claude_remote_control_gate(
            path, {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}
        )
        assert result is not None
        assert result.status == WARN

    def test_api_key_auth_suppresses_warning(self, tmp_path):
        # Issue #1779: a PAYG / API-key session never had Remote Control, so the
        # gate warning must not fire even behind a custom base URL.
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        assert check_claude_remote_control_gate(path, {"ANTHROPIC_API_KEY": "sk-ant-api-x"}) is None

    def test_settings_api_key_suppresses_warning(self, tmp_path):
        # An API key configured in settings.json (not just the shell) also means
        # a non-subscription session — stay silent.
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "env": {
                        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787",
                        "ANTHROPIC_API_KEY": "sk-ant-api-x",
                    }
                }
            ),
            encoding="utf-8",
        )
        assert check_claude_remote_control_gate(path, {}) is None

    def test_version_resolver_not_called_without_custom_base(self, tmp_path):
        # The `claude --version` subprocess is expensive (Node CLI cold start);
        # the check must not invoke the resolver when no custom base URL exists.
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}}),
            encoding="utf-8",
        )

        def boom() -> tuple[int, int, int]:
            raise AssertionError("resolver must not run when no custom base URL")

        assert check_claude_remote_control_gate(path, {}, version_resolver=boom) is None

    def test_version_resolver_not_called_for_api_key_auth(self, tmp_path):
        # PAYG sessions are suppressed before version matters — no subprocess.
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )

        def boom() -> tuple[int, int, int]:
            raise AssertionError("resolver must not run for API-key auth")

        assert (
            check_claude_remote_control_gate(
                path, {"ANTHROPIC_API_KEY": "sk-ant-api-x"}, version_resolver=boom
            )
            is None
        )

    def test_version_resolver_called_once_and_honored(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        calls: list[int] = []

        def resolver() -> tuple[int, int, int]:
            calls.append(1)
            return (2, 1, 196)

        # Shell env ALSO custom so both loop sources are live — still one call.
        result = check_claude_remote_control_gate(
            path,
            {"ANTHROPIC_BASE_URL": "http://127.0.0.1:9999"},
            version_resolver=resolver,
        )
        assert result is not None
        assert "2.1.196" in result.summary
        assert calls == [1]

    def test_version_resolver_pre_gate_version_suppresses(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        assert (
            check_claude_remote_control_gate(path, {}, version_resolver=lambda: (2, 1, 195)) is None
        )

    def test_malformed_settings_base_url_does_not_crash(self, tmp_path):
        # Issue #1779: settings.json is user-edited; a typo'd IPv6 literal made
        # urlparse raise ValueError("Invalid IPv6 URL") and crashed doctor.
        # Malformed values degrade to "no host" and the check stays silent —
        # check_claude_routing separately flags unusable URLs.
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://[::1:8787"}}),
            encoding="utf-8",
        )
        assert check_claude_remote_control_gate(path, {}) is None

    def test_malformed_shell_base_url_does_not_crash(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("{}", encoding="utf-8")
        assert check_claude_remote_control_gate(path, {"ANTHROPIC_BASE_URL": "http://["}) is None

    def test_pre_gate_version_suppresses_warning(self, tmp_path):
        # Older Claude Code does not gate RC on the base URL — no false alarm.
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        assert check_claude_remote_control_gate(path, {}, version=(2, 1, 195)) is None

    def test_gated_version_warns_with_exact_version(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        result = check_claude_remote_control_gate(path, {}, version=(2, 1, 196))
        assert result is not None
        assert result.status == WARN
        assert "2.1.196" in result.summary
        assert "disables" in result.summary
        # Sibling gates are co-reported in the hint (#746 / #1158).
        assert "#746" in (result.hint or "")
        assert "#1158" in (result.hint or "")

    def test_settings_check_still_routes(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        result = check_claude_routing(path, 8787)
        assert result.status == PASS


class TestClaudeRoutingScope:
    """Project-scoped routing must not read as "not routed" (#3205).

    `headroom init claude` without --global writes
    `.claude/settings.local.json`. Reading only `~/.claude/settings.json`
    reported not-routed for sessions that were genuinely routed and actively
    compressing, which sent one team hand-checking `ps eww` on every session.
    """

    @staticmethod
    def _settings(path, base_url):  # noqa: ANN001, ANN205
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"env": {"ANTHROPIC_BASE_URL": base_url}} if base_url else {"env": {}}
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_project_local_settings_count_as_routed(self, tmp_path):
        user = tmp_path / "user" / "settings.json"
        project = self._settings(
            tmp_path / "proj" / ".claude" / "settings.local.json", "http://127.0.0.1:8787"
        )

        result = check_claude_routing(user, 8787, [project])

        assert result.status == PASS
        assert "settings.local.json" in result.summary or "settings.local.json" in str(result)

    def test_project_settings_json_counts_as_routed(self, tmp_path):
        user = tmp_path / "user" / "settings.json"
        project = self._settings(
            tmp_path / "proj" / ".claude" / "settings.json", "http://127.0.0.1:8787"
        )

        assert check_claude_routing(user, 8787, [project]).status == PASS

    def test_project_scope_takes_precedence_over_user_scope(self, tmp_path):
        """Claude layers project over user, so the reported port follows suit."""
        user = self._settings(tmp_path / "user" / "settings.json", "http://127.0.0.1:9999")
        project = self._settings(
            tmp_path / "proj" / ".claude" / "settings.local.json", "http://127.0.0.1:8787"
        )

        assert check_claude_routing(user, 8787, [project]).status == PASS

    def test_falls_back_to_user_scope_when_project_has_no_base_url(self, tmp_path):
        user = self._settings(tmp_path / "user" / "settings.json", "http://127.0.0.1:8787")
        project = self._settings(tmp_path / "proj" / ".claude" / "settings.local.json", "")

        assert check_claude_routing(user, 8787, [project]).status == PASS

    def test_still_warns_when_nothing_routes(self, tmp_path):
        user = self._settings(tmp_path / "user" / "settings.json", "")
        project = self._settings(tmp_path / "proj" / ".claude" / "settings.local.json", "")

        assert check_claude_routing(user, 8787, [project]).status == WARN

    def test_missing_project_file_is_skipped_not_fatal(self, tmp_path):
        user = self._settings(tmp_path / "user" / "settings.json", "http://127.0.0.1:8787")
        absent = tmp_path / "proj" / ".claude" / "settings.local.json"

        assert check_claude_routing(user, 8787, [absent]).status == PASS

    def test_unparseable_project_file_surfaces_rather_than_reporting_not_routed(self, tmp_path):
        project = tmp_path / "proj" / ".claude" / "settings.local.json"
        project.parent.mkdir(parents=True, exist_ok=True)
        project.write_text("{not json", encoding="utf-8")
        user = tmp_path / "user" / "settings.json"

        result = check_claude_routing(user, 8787, [project])

        assert result.status == WARN
        assert "could not parse" in result.summary

    def test_no_project_paths_preserves_original_behaviour(self, tmp_path):
        user = self._settings(tmp_path / "user" / "settings.json", "http://127.0.0.1:8787")

        assert check_claude_routing(user, 8787).status == PASS


class TestCodexRouting:
    def test_missing_file_warns(self, tmp_path):
        assert check_codex_routing(tmp_path / "config.toml", 8787).status == WARN

    def test_marker_block_right_port_passes(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            'model_provider = "headroom"\n'
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n',
            encoding="utf-8",
        )
        assert check_codex_routing(path, 8787).status == PASS

    def test_preserved_provider_id_right_port_passes(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            'model_provider = "codex-lb"\n'
            "[model_providers.codex-lb]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n',
            encoding="utf-8",
        )
        result = check_codex_routing(path, 8787)
        assert result.status == PASS
        assert result.hint is None

    def test_preserved_provider_id_port_mismatch_warns(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            'model_provider = "codex-lb"\n'
            "[model_providers.codex-lb]\n"
            'base_url = "http://localhost:9999/v1"\n',
            encoding="utf-8",
        )
        result = check_codex_routing(path, 8787)
        assert result.status == WARN
        assert "9999" in result.summary

    def test_active_provider_takes_precedence_over_headroom_block(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            'model_provider = "corp"\n'
            "[model_providers.corp]\n"
            'base_url = "https://gateway.corp.example/v1"\n'
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n',
            encoding="utf-8",
        )
        result = check_codex_routing(path, 8787)
        assert result.status == WARN
        assert "gateway.corp.example" in result.summary

    def test_port_mismatch_warns(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            '[model_providers.headroom]\nbase_url = "http://127.0.0.1:9999/v1"\n',
            encoding="utf-8",
        )
        result = check_codex_routing(path, 8787)
        assert result.status == WARN
        assert "9999" in result.summary

    def test_no_marker_warns(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('model = "gpt-5"\n', encoding="utf-8")
        assert check_codex_routing(path, 8787).status == WARN

    def test_garbage_bytes_warn_not_crash(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_bytes(b"\xff\xfe garbage \x00")
        assert check_codex_routing(path, 8787).status == WARN

    # -- requires_openai_auth (#3206) ------------------------------------
    # Codex attaches no Authorization header to a custom provider unless the
    # block carries requires_openai_auth. A ChatGPT-OAuth user then 401s on
    # every request with "Missing bearer" while doctor reported green -- the
    # reason one report went 15h before anyone could see the cause.

    @staticmethod
    def _routed(tmp_path, *, requires_auth: bool):
        path = tmp_path / "config.toml"
        block = (
            "[model_providers.headroom]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n'
            "supports_websockets = true\n"
        )
        if requires_auth:
            block += "requires_openai_auth = true\n"
        path.write_text(block, encoding="utf-8")
        return path

    @staticmethod
    def _chatgpt_auth(tmp_path):
        (tmp_path / "auth.json").write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")

    def test_chatgpt_auth_without_requires_openai_auth_warns(self, tmp_path):
        path = self._routed(tmp_path, requires_auth=False)
        self._chatgpt_auth(tmp_path)

        result = check_codex_routing(path, 8787)

        assert result.status == WARN
        assert "Authorization" in result.summary

    def test_chatgpt_auth_with_requires_openai_auth_passes(self, tmp_path):
        path = self._routed(tmp_path, requires_auth=True)
        self._chatgpt_auth(tmp_path)

        assert check_codex_routing(path, 8787).status == PASS

    def test_preserved_provider_id_without_requires_openai_auth_warns(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            'model_provider = "codex-lb"\n'
            "[model_providers.codex-lb]\n"
            'base_url = "http://127.0.0.1:8787/v1"\n',
            encoding="utf-8",
        )
        self._chatgpt_auth(tmp_path)

        result = check_codex_routing(path, 8787)
        assert result.status == WARN
        assert "Authorization" in result.summary

    def test_api_key_user_without_requires_openai_auth_still_passes(self, tmp_path):
        """API-key users must not be nagged -- the flag would break them (#406)."""
        path = self._routed(tmp_path, requires_auth=False)
        (tmp_path / "auth.json").write_text('{"OPENAI_API_KEY": "sk-test"}', encoding="utf-8")

        assert check_codex_routing(path, 8787).status == PASS

    def test_no_auth_json_does_not_warn(self, tmp_path):
        path = self._routed(tmp_path, requires_auth=False)

        assert check_codex_routing(path, 8787).status == PASS


class TestShellEnv:
    def test_unset_warns(self):
        result = check_shell_env({}, 8787)
        assert result.status == WARN
        assert "bypasses" in result.summary

    def test_matching_anthropic_url_passes(self):
        env = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}
        assert check_shell_env(env, 8787).status == PASS

    def test_localhost_also_passes(self):
        env = {"OPENAI_BASE_URL": "http://localhost:8787/v1"}
        assert check_shell_env(env, 8787).status == PASS

    def test_other_url_warns(self):
        env = {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}
        assert check_shell_env(env, 8787).status == WARN

    def test_ollama_launch_url_names_the_collision(self):
        # `ollama launch claude` points Claude Code at Ollama's :11434, which
        # outranks the persistent Headroom route (issue #2199). The diagnostic
        # must name Ollama, not tell the user to re-probe port 11434.
        env = {"ANTHROPIC_BASE_URL": "http://127.0.0.1:11434"}
        result = check_shell_env(env, 8787)
        assert result.status == WARN
        assert "Ollama" in result.summary
        assert "#2199" in (result.hint or "")
        assert "--port 11434" not in (result.hint or "")


class TestSavings:
    def test_from_stats_passes_with_totals(self, tmp_path):
        result = check_savings(STATS_OK, tmp_path / "missing.json")
        assert result.status == PASS
        assert "17,583,102" in result.summary
        assert "$7.82" in result.summary

    def test_falls_back_to_file_when_proxy_down(self, tmp_path):
        savings_file = tmp_path / "proxy_savings.json"
        savings_file.write_text(
            json.dumps(
                {
                    "lifetime": {"tokens_saved": 500, "compression_savings_usd": 0.01},
                    "display_session": {"last_activity_at": "2026-06-12T11:00:00Z"},
                }
            ),
            encoding="utf-8",
        )
        result = check_savings(None, savings_file)
        assert result.status == PASS
        assert "500" in result.summary
        assert str(savings_file) in result.summary

    def test_no_data_warns(self, tmp_path):
        assert check_savings(None, tmp_path / "missing.json").status == WARN

    def test_zero_tokens_warns(self, tmp_path):
        stats = {"persistent_savings": {"lifetime": {"tokens_saved": 0}}}
        assert check_savings(stats, tmp_path / "missing.json").status == WARN


class TestBudget:
    def test_proxy_down_skips(self):
        assert check_budget(None).status == SKIP

    def test_cost_tracking_disabled_warns(self):
        assert check_budget({"cost": None}).status == WARN

    def test_old_proxy_without_keys_warns(self):
        result = check_budget({"cost": {"savings_usd": 1.0}})
        assert result.status == WARN
        assert "older version" in result.summary

    def test_unset_budget_warns_with_hint(self):
        result = check_budget({"cost": {"budget_limit_usd": None}})
        assert result.status == WARN
        assert "--budget" in (result.hint or "")

    def test_configured_budget_passes(self):
        result = check_budget(STATS_OK)
        assert result.status == PASS
        assert "$10.0/daily" in result.summary

    def test_estimated_basis_share_is_reported_without_warning(self):
        """#2713: spend booked from a token estimate is surfaced, not warned on.

        A provider that never reports a usage breakdown would otherwise sit at a
        permanent WARN, so this stays informational.
        """
        result = check_budget(
            {
                "cost": {
                    "budget_limit_usd": 10.0,
                    "budget_period": "daily",
                    "budget_estimated_basis": "count",
                    "budget_basis": {"estimated_usd": 1.24, "estimated_pct": 62.3},
                }
            }
        )
        assert result.status == PASS
        assert "62% of period spend ($1.2400)" in result.summary
        assert "Headroom token estimates" in result.summary

    def test_all_measured_spend_adds_no_note(self):
        result = check_budget(
            {
                "cost": {
                    "budget_limit_usd": 10.0,
                    "budget_period": "daily",
                    "budget_estimated_basis": "count",
                    "budget_basis": {"estimated_usd": 0.0, "estimated_pct": 0.0},
                }
            }
        )
        assert result.summary == "$10.0/daily budget enforced"

    def test_non_default_basis_policy_is_named(self):
        result = check_budget(
            {
                "cost": {
                    "budget_limit_usd": 10.0,
                    "budget_period": "daily",
                    "budget_estimated_basis": "block",
                }
            }
        )
        assert "estimated-basis policy: block" in result.summary

    def test_missing_basis_fields_degrade_quietly(self):
        """`doctor` must still work against a proxy predating these fields."""
        result = check_budget({"cost": {"budget_limit_usd": 10.0, "budget_period": "daily"}})
        assert result.status == PASS
        assert result.summary == "$10.0/daily budget enforced"

        malformed = check_budget(
            {"cost": {"budget_limit_usd": 10.0, "budget_period": "daily", "budget_basis": "nope"}}
        )
        assert malformed.status == PASS


@dataclass
class _FakeManifest:
    profile: str
    health_url: str


class TestDeployments:
    def test_no_manifests_omits_section(self):
        assert check_deployments([]) is None

    def test_all_healthy_passes(self):
        manifests = [_FakeManifest("default", "http://127.0.0.1:8787/readyz")]
        result = check_deployments(manifests, probe=lambda url: {"ready": True})
        assert result is not None and result.status == PASS

    def test_unhealthy_fails_naming_profile(self):
        manifests = [_FakeManifest("prod", "http://127.0.0.1:9999/readyz")]
        result = check_deployments(manifests, probe=lambda url: None)
        assert result is not None and result.status == FAIL
        assert "prod" in result.summary


class TestDoctorCommand:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def isolated(self, tmp_path, monkeypatch):
        """Point all filesystem/network surfaces at controlled fakes."""
        monkeypatch.setattr(doctor_mod, "claude_settings_path", lambda: tmp_path / "settings.json")
        monkeypatch.setattr(doctor_mod, "codex_config_path", lambda: tmp_path / "config.toml")
        monkeypatch.setattr(doctor_mod, "savings_path", lambda: tmp_path / "savings.json")
        monkeypatch.setattr(doctor_mod, "list_manifests", lambda: [])
        for var in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_FOUNDRY",
            "CLAUDE_CODE_USE_VERTEX",
            "OPENAI_BASE_URL",
            "HEADROOM_PORT",
        ):
            monkeypatch.delenv(var, raising=False)
        return tmp_path

    def _probe(self, livez, stats):
        def fake_probe(url, timeout=2.0):
            if url.endswith("/livez"):
                return livez
            if url.endswith("/stats"):
                return stats
            return None

        return fake_probe

    def test_proxy_down_exits_2(self, runner, isolated, monkeypatch):
        monkeypatch.setattr(doctor_mod, "probe_json", self._probe(None, None))
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 2
        assert "not reachable" in result.output

    def test_conflicting_claude_auth_is_a_redacted_failure(self, runner, isolated, monkeypatch):
        settings = isolated / "settings.json"
        settings.write_text('{"env":{"ANTHROPIC_AUTH_TOKEN":"token-value"}}', encoding="utf-8")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "api-value")
        monkeypatch.setattr(doctor_mod, "probe_json", self._probe(None, None))

        result = runner.invoke(main, ["doctor", "--json"])

        assert result.exit_code == 2
        payload = json.loads(result.output)
        auth = next(check for check in payload["checks"] if check["name"] == "claude auth")
        assert auth["status"] == "fail"
        assert "shell environment" in auth["summary"]
        assert str(settings) in auth["summary"]
        assert "api-value" not in result.output
        assert "token-value" not in result.output

    def test_warnings_only_exits_1(self, runner, isolated, monkeypatch):
        monkeypatch.setattr(doctor_mod, "probe_json", self._probe(LIVEZ_OK, STATS_OK))
        monkeypatch.setattr(doctor_mod, "get_version", lambda: "0.26.0")
        # proxy healthy, but clients unwrapped + shell env unset -> warns
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1

    def test_remote_control_warning_exits_1(self, runner, isolated, monkeypatch):
        monkeypatch.setattr(doctor_mod, "probe_json", self._probe(LIVEZ_OK, STATS_OK))
        monkeypatch.setattr(doctor_mod, "get_version", lambda: "0.26.0")
        monkeypatch.setattr(doctor_mod, "detect_claude_code_version", lambda: None)
        (isolated / "settings.json").write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        (isolated / "config.toml").write_text(
            '[model_providers.headroom]\nbase_url = "http://127.0.0.1:8787/v1"\n',
            encoding="utf-8",
        )
        result = runner.invoke(
            main, ["doctor"], env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}
        )
        assert result.exit_code == 1, result.output
        assert "Remote Control" in result.output

    def test_json_output_parses(self, runner, isolated, monkeypatch):
        monkeypatch.setattr(doctor_mod, "probe_json", self._probe(LIVEZ_OK, STATS_OK))
        result = runner.invoke(main, ["doctor", "--json"])
        payload = json.loads(result.output)
        assert payload["port"] == 8787
        assert {c["name"] for c in payload["checks"]} >= {"proxy", "version", "budget"}
        assert all(c["status"] in ("pass", "warn", "fail", "skip") for c in payload["checks"])

    def test_port_option_changes_probe_url(self, runner, isolated, monkeypatch):
        seen: list[str] = []

        def recording_probe(url, timeout=2.0):
            seen.append(url)
            return None

        monkeypatch.setattr(doctor_mod, "probe_json", recording_probe)
        runner.invoke(main, ["doctor", "--port", "9999"])
        assert "http://127.0.0.1:9999/livez" in seen

    def test_port_env_var_respected(self, runner, isolated, monkeypatch):
        seen: list[str] = []

        def recording_probe(url, timeout=2.0):
            seen.append(url)
            return None

        monkeypatch.setattr(doctor_mod, "probe_json", recording_probe)
        runner.invoke(main, ["doctor"], env={"HEADROOM_PORT": "9999"})
        assert "http://127.0.0.1:9999/livez" in seen


class TestCostTrackerBudgetKeys:
    def test_stats_exposes_budget_config(self):
        from headroom.proxy.cost import CostTracker

        stats = CostTracker(budget_limit_usd=5.0, budget_period="monthly").stats()
        assert stats["budget_limit_usd"] == 5.0
        assert stats["budget_period"] == "monthly"

    def test_stats_budget_none_when_unset(self):
        from headroom.proxy.cost import CostTracker

        assert CostTracker().stats()["budget_limit_usd"] is None
