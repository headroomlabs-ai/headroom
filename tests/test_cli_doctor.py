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


class TestClaudeRouting:
    def test_missing_file_warns(self, tmp_path):
        result = check_claude_routing(tmp_path / "settings.json", 8787)
        assert result.status == WARN
        assert "wrap claude" in (result.hint or "")

    def test_malformed_json_warns(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("{not json", encoding="utf-8")
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

    def test_settings_check_still_routes(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}),
            encoding="utf-8",
        )
        result = check_claude_routing(path, 8787)
        assert result.status == PASS


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
        for var in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "HEADROOM_PORT"):
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

    def test_warnings_only_exits_1(self, runner, isolated, monkeypatch):
        monkeypatch.setattr(doctor_mod, "probe_json", self._probe(LIVEZ_OK, STATS_OK))
        monkeypatch.setattr(doctor_mod, "get_version", lambda: "0.26.0")
        # proxy healthy, but clients unwrapped + shell env unset -> warns
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 1

    def test_remote_control_warning_exits_1(self, runner, isolated, monkeypatch):
        monkeypatch.setattr(doctor_mod, "probe_json", self._probe(LIVEZ_OK, STATS_OK))
        monkeypatch.setattr(doctor_mod, "get_version", lambda: "0.26.0")
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
