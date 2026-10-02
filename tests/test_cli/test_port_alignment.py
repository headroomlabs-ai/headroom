"""CODEX_HOME resolution, live-proxy port discovery, and agent/proxy alignment.

Every test isolates HOME/CODEX_HOME/cwd under ``tmp_path`` and injects fake
probes, so no real ``~/.codex``, ``~/.headroom`` or listening proxy is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest
from click.testing import CliRunner

import headroom.cli.doctor as doctor_mod
import headroom.cli.init as init_mod
import headroom.cli.port_discovery as pd
import headroom.cli.wrap as wrap_mod
from headroom.cli.main import main
from headroom.install import paths as install_paths


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake home + cwd; discovery enabled; HEADROOM_WORKSPACE_DIR sandboxed."""
    fake_home = tmp_path / "home"
    project = tmp_path / "project"
    fake_home.mkdir()
    project.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("HEADROOM_PORT", raising=False)
    monkeypatch.delenv("HEADROOM_PORT_DISCOVERY", raising=False)
    monkeypatch.chdir(project)
    return fake_home


def _manifest(profile: str, port: int, targets: list[str] | None = None) -> Any:
    return SimpleNamespace(profile=profile, port=port, targets=targets or [])


# ---------------------------------------------------------------------------
# 1. CODEX_HOME
# ---------------------------------------------------------------------------


class TestCodexHome:
    def test_defaults_to_dot_codex(self, home: Path) -> None:
        assert install_paths.codex_home_dir() == home / ".codex"
        assert install_paths.codex_config_path() == home / ".codex" / "config.toml"
        assert install_paths.codex_hooks_path() == home / ".codex" / "hooks.json"

    def test_honors_codex_home(self, home: Path, tmp_path: Path, monkeypatch) -> None:
        custom = tmp_path / "custom-codex"
        monkeypatch.setenv("CODEX_HOME", str(custom))
        assert install_paths.codex_config_path() == custom / "config.toml"
        assert install_paths.codex_hooks_path() == custom / "hooks.json"
        # Every consumer resolves through the one helper.
        assert wrap_mod._codex_home_dir() == custom
        from headroom.mcp_registry.codex import CodexRegistrar

        assert CodexRegistrar()._config_file == custom / "config.toml"

    def test_blank_codex_home_falls_back(self, home: Path, monkeypatch) -> None:
        monkeypatch.setenv("CODEX_HOME", "   ")
        assert install_paths.codex_home_dir() == home / ".codex"

    def test_init_global_codex_paths_follow_codex_home(
        self, home: Path, tmp_path: Path, monkeypatch
    ) -> None:
        custom = tmp_path / "custom-codex"
        monkeypatch.setenv("CODEX_HOME", str(custom))
        assert init_mod._codex_scope_path(True) == custom / "config.toml"
        assert init_mod._codex_hooks_path(True) == custom / "hooks.json"
        # Project scope stays in the project.
        assert init_mod._codex_scope_path(False) == Path.cwd() / ".codex" / "config.toml"
        assert init_mod._codex_hooks_path(False) == Path.cwd() / ".codex" / "hooks.json"

    def test_doctor_reads_codex_home_config(self, home: Path, tmp_path: Path, monkeypatch) -> None:
        custom = tmp_path / "custom-codex"
        custom.mkdir()
        (custom / "config.toml").write_text(
            'model_provider = "headroom"\n[model_providers.headroom]\n'
            'base_url = "http://127.0.0.1:8787/v1"\nrequires_openai_auth = true\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CODEX_HOME", str(custom))
        monkeypatch.setattr(doctor_mod, "probe_json", lambda *a, **k: None)
        monkeypatch.setattr(doctor_mod, "list_manifests", lambda: [])
        monkeypatch.setattr(doctor_mod, "claude_settings_path", lambda: home / "settings.json")
        monkeypatch.setattr(doctor_mod, "savings_path", lambda: home / "savings.json")
        result = CliRunner().invoke(main, ["doctor", "--json"])
        codex = next(c for c in json.loads(result.output)["checks"] if c["name"] == "codex")
        assert codex["status"] == "pass", codex
        assert str(custom / "config.toml") in codex["summary"]

    def test_doctor_prefers_project_codex_config(self, tmp_path: Path) -> None:
        user = tmp_path / "user.toml"
        user.write_text(
            '[model_providers.headroom]\nbase_url = "http://127.0.0.1:9999/v1"\n',
            encoding="utf-8",
        )
        project = tmp_path / "project.toml"
        project.write_text(
            '[model_providers.headroom]\nbase_url = "http://127.0.0.1:8787/v1"\n',
            encoding="utf-8",
        )
        result = doctor_mod.check_codex_routing(user, 8787, [project])
        assert result.status == doctor_mod.PASS
        assert str(project) in result.summary

    def test_doctor_missing_config_names_actual_path(self, tmp_path: Path) -> None:
        result = doctor_mod.check_codex_routing(tmp_path / "nope.toml", 8787, [])
        assert result.status == doctor_mod.WARN
        assert str(tmp_path / "nope.toml") in result.summary


# ---------------------------------------------------------------------------
# 2. Default port: HEADROOM_PORT + live-proxy discovery
# ---------------------------------------------------------------------------


class TestPortDiscovery:
    def test_default_port_reads_env(self) -> None:
        assert pd.default_port({}) == 8787
        assert pd.default_port({"HEADROOM_PORT": "9100"}) == 9100
        assert pd.default_port({"HEADROOM_PORT": "not-a-port"}) == 8787
        assert pd.default_port({"HEADROOM_PORT": "70000"}) == 8787

    def test_candidates_include_env_manifests_and_wrap_marker(self, tmp_path: Path) -> None:
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / ".headroom_wrap_marker.json").write_text(
            json.dumps({"port": 9300}), encoding="utf-8"
        )
        ports = pd.candidate_ports(
            8787,
            environ={"HEADROOM_PORT": "9100"},
            manifests=[_manifest("a", 9200), _manifest("b", 8787)],
            cwd=tmp_path,
        )
        assert ports == [8787, 9100, 9200, 9300]

    def test_candidates_are_bounded(self, tmp_path: Path) -> None:
        manifests = [_manifest(str(i), 10000 + i) for i in range(50)]
        ports = pd.candidate_ports(8787, environ={}, manifests=manifests, cwd=tmp_path)
        assert len(ports) == pd._MAX_CANDIDATES

    def test_finds_live_proxy_on_other_port(self, tmp_path: Path) -> None:
        live = pd.find_live_proxy_elsewhere(
            8787,
            environ={},
            manifests=[_manifest("p", 9200)],
            cwd=tmp_path,
            probe=lambda port: port == 9200,
        )
        assert live == 9200

    def test_requested_port_live_means_nothing_to_reconcile(self, tmp_path: Path) -> None:
        live = pd.find_live_proxy_elsewhere(
            8787,
            environ={},
            manifests=[_manifest("p", 9200)],
            cwd=tmp_path,
            probe=lambda port: True,
        )
        assert live is None

    def test_discovery_can_be_disabled(self, tmp_path: Path) -> None:
        live = pd.find_live_proxy_elsewhere(
            8787,
            environ={"HEADROOM_PORT_DISCOVERY": "0"},
            manifests=[_manifest("p", 9200)],
            cwd=tmp_path,
            probe=lambda port: port == 9200,
        )
        assert live is None

    def test_probe_errors_never_raise(self, tmp_path: Path) -> None:
        def boom(port: int) -> bool:
            raise RuntimeError("nope")

        assert (
            pd.find_live_proxy_elsewhere(
                8787, environ={}, manifests=[_manifest("p", 9200)], cwd=tmp_path, probe=boom
            )
            is None
        )

    def test_livez_identity_required(self) -> None:
        assert pd.is_headroom_livez({"service": "headroom-proxy", "alive": True})
        assert pd.is_headroom_livez({"alive": True, "version": "1.0"})  # legacy shape
        assert not pd.is_headroom_livez({"service": "something-else", "alive": True})
        assert not pd.is_headroom_livez({"status": "ok"})
        assert not pd.is_headroom_livez(None)

    def test_non_interactive_keeps_requested_and_warns(self, tmp_path: Path, capsys) -> None:
        port = pd.reconcile_default_port(
            8787,
            interactive=False,
            environ={},
            manifests=[_manifest("p", 9200)],
            cwd=tmp_path,
            probe=lambda p: p == 9200,
        )
        assert port == 8787
        err = capsys.readouterr().err
        assert "port 9200" in err
        assert "--port 9200" in err

    @pytest.mark.parametrize(("answer", "expected"), [("c", 9200), ("k", 8787)])
    def test_interactive_prompt_connect_or_keep(
        self, tmp_path: Path, monkeypatch, answer: str, expected: int
    ) -> None:
        monkeypatch.setattr(pd.click, "prompt", lambda *a, **k: answer)
        port = pd.reconcile_default_port(
            8787,
            interactive=True,
            environ={},
            manifests=[_manifest("p", 9200)],
            cwd=tmp_path,
            probe=lambda p: p == 9200,
        )
        assert port == expected

    def test_interactive_prompt_abort(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(pd.click, "prompt", lambda *a, **k: "a")
        with pytest.raises(click.Abort):
            pd.reconcile_default_port(
                8787,
                interactive=True,
                environ={},
                manifests=[_manifest("p", 9200)],
                cwd=tmp_path,
                probe=lambda p: p == 9200,
            )

    def test_no_live_proxy_is_silent(self, tmp_path: Path, capsys) -> None:
        port = pd.reconcile_default_port(
            8787, interactive=False, environ={}, manifests=[], cwd=tmp_path, probe=lambda p: False
        )
        assert port == 8787
        assert capsys.readouterr().err == ""


class TestProxyPortOption:
    @pytest.fixture
    def cmd(self) -> click.Command:
        @click.command()
        @pd.proxy_port_option()
        def _cmd(port: int) -> None:
            click.echo(f"port={port}")

        return _cmd

    @pytest.fixture
    def reconcile_calls(self, monkeypatch) -> list[int]:
        calls: list[int] = []

        def fake(requested: int, **kwargs: Any) -> int:
            calls.append(requested)
            return 9200

        monkeypatch.setattr(pd, "reconcile_default_port", fake)
        return calls

    def test_default_triggers_reconcile(self, cmd, reconcile_calls) -> None:
        result = CliRunner().invoke(cmd, [])
        assert result.exit_code == 0, result.output
        assert "port=9200" in result.output
        assert reconcile_calls == [8787]

    def test_explicit_port_is_honored_without_discovery(self, cmd, reconcile_calls) -> None:
        result = CliRunner().invoke(cmd, ["--port", "9500"])
        assert "port=9500" in result.output
        assert reconcile_calls == []

    def test_headroom_port_env_is_honored_without_discovery(self, cmd, reconcile_calls) -> None:
        result = CliRunner().invoke(cmd, [], env={"HEADROOM_PORT": "9400"})
        assert "port=9400" in result.output
        assert reconcile_calls == []

    def test_wrap_subcommands_use_shared_option(self) -> None:
        # Every wrap subcommand with a port option resolves it through the callback.
        wrap_group = main.commands["wrap"]
        assert isinstance(wrap_group, click.Group)
        checked = 0
        for name, command in wrap_group.commands.items():
            for param in command.params:
                if param.name in {"port", "proxy_port"}:
                    assert param.callback is pd.port_option_callback, name
                    assert param.envvar == "HEADROOM_PORT", name
                    checked += 1
        assert checked >= 15


class TestCodexProviderPortWarning:
    def test_warns_when_repointing(self, tmp_path: Path, capsys) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            '[model_providers.headroom]\nname = "x"\nbase_url = "http://127.0.0.1:9200/v1"\n',
            encoding="utf-8",
        )
        assert pd.warn_codex_provider_port_change(config, 8787) == 9200
        err = capsys.readouterr().err
        assert "9200" in err and "8787" in err and "--port 9200" in err

    def test_silent_when_same_port_or_absent(self, tmp_path: Path, capsys) -> None:
        config = tmp_path / "config.toml"
        assert pd.warn_codex_provider_port_change(config, 8787) is None
        config.write_text(
            '[model_providers.headroom]\nbase_url = "http://127.0.0.1:8787/v1"\n',
            encoding="utf-8",
        )
        assert pd.warn_codex_provider_port_change(config, 8787) is None
        assert capsys.readouterr().err == ""

    def test_init_provider_write_warns(self, tmp_path: Path, capsys, monkeypatch) -> None:
        monkeypatch.setattr(init_mod, "retag_to_headroom", lambda _path: None)
        config = tmp_path / "config.toml"
        config.write_text(
            '[model_providers.headroom]\nbase_url = "http://127.0.0.1:9200/v1"\n',
            encoding="utf-8",
        )
        init_mod._ensure_codex_provider(config, 8787)
        assert "port 9200" in capsys.readouterr().err
        assert 'base_url = "http://127.0.0.1:8787/v1"' in config.read_text(encoding="utf-8")


class TestInitPortResolution:
    def _ctx(self, args: list[str]) -> click.Context:
        group = main.commands["init"]
        return group.make_context("init", list(args), resilient_parsing=False)

    def test_explicit_port_wins(self, monkeypatch) -> None:
        monkeypatch.setattr(init_mod, "reconcile_default_port", lambda p, **k: 1)
        ctx = self._ctx(["--port", "9500"])
        assert init_mod._resolve_init_port(ctx, 9500) == 9500

    def test_env_port_used_when_no_flag(self, monkeypatch) -> None:
        monkeypatch.setattr(init_mod, "reconcile_default_port", lambda p, **k: 1)
        monkeypatch.setenv("HEADROOM_PORT", "9400")
        ctx = self._ctx([])
        assert init_mod._resolve_init_port(ctx, 8787) == 9400

    def test_default_reconciles_with_live_proxy(self, monkeypatch) -> None:
        monkeypatch.setattr(init_mod, "reconcile_default_port", lambda p, **k: 9200)
        ctx = self._ctx([])
        assert init_mod._resolve_init_port(ctx, 8787) == 9200

    def test_hook_subcommand_skips_discovery(self, monkeypatch) -> None:
        def fail(*a: Any, **k: Any) -> int:
            raise AssertionError("hook must not run discovery")

        monkeypatch.setattr(init_mod, "reconcile_default_port", fail)
        ctx = self._ctx([])
        ctx.invoked_subcommand = "hook"
        assert init_mod._resolve_init_port(ctx, 8787) == 8787


# ---------------------------------------------------------------------------
# 3. init hook ensure alignment warning
# ---------------------------------------------------------------------------


class TestHookAlignment:
    @pytest.fixture
    def claude_settings(self, home: Path, monkeypatch) -> Path:
        path = home / ".claude" / "settings.json"
        path.parent.mkdir(parents=True)
        monkeypatch.setattr(init_mod, "claude_settings_path", lambda: path)
        monkeypatch.setattr(init_mod, "load_manifest", lambda profile: _manifest(profile, 8787))
        monkeypatch.setattr(init_mod, "_ensure_profile_running", lambda profile: None)
        return path

    def _route_claude(self, path: Path, port: int) -> None:
        path.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}}),
            encoding="utf-8",
        )

    def test_aligned_returns_none(self, claude_settings: Path) -> None:
        self._route_claude(claude_settings, 8787)
        assert init_mod.alignment_warning("claude", "init-user", probe=lambda p: True) is None

    def test_mismatch_to_dead_port_suggests_managed_port(self, claude_settings: Path) -> None:
        self._route_claude(claude_settings, 9200)
        message = init_mod.alignment_warning("claude", "init-user", probe=lambda p: False)
        assert message is not None
        assert "9200" in message and "8787" in message
        assert "headroom init -g --port 8787 claude" in message

    def test_mismatch_to_live_port_suggests_live_port(self, claude_settings: Path) -> None:
        self._route_claude(claude_settings, 9200)
        message = init_mod.alignment_warning("claude", "init-user", probe=lambda p: p == 9200)
        assert message is not None
        assert "headroom init -g --port 9200 claude" in message

    def test_codex_mismatch_uses_codex_home(self, home: Path, tmp_path: Path, monkeypatch) -> None:
        custom = tmp_path / "custom-codex"
        custom.mkdir()
        (custom / "config.toml").write_text(
            'model_provider = "headroom"\n[model_providers.headroom]\n'
            'base_url = "http://127.0.0.1:9200/v1"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("CODEX_HOME", str(custom))
        monkeypatch.setattr(init_mod, "load_manifest", lambda profile: _manifest(profile, 8787))
        message = init_mod.alignment_warning("codex", "init-user", probe=lambda p: False)
        assert message is not None and "--port 8787 codex" in message

    def test_session_start_emits_hook_json(self, claude_settings: Path) -> None:
        self._route_claude(claude_settings, 9200)
        result = CliRunner().invoke(
            main,
            [
                "init",
                "hook",
                "ensure",
                "--profile",
                "init-user",
                "--marker",
                "headroom-init-claude",
            ],
            input=json.dumps({"hook_event_name": "SessionStart"}),
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output.strip().splitlines()[-1])
        assert "9200" in payload["systemMessage"]
        assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert payload["hookSpecificOutput"]["additionalContext"] == payload["systemMessage"]

    def test_pre_tool_use_stays_silent(self, claude_settings: Path) -> None:
        self._route_claude(claude_settings, 9200)
        result = CliRunner().invoke(
            main,
            [
                "init",
                "hook",
                "ensure",
                "--profile",
                "init-user",
                "--marker",
                "headroom-init-claude",
            ],
            input=json.dumps({"hook_event_name": "PreToolUse"}),
        )
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_never_raises(self, claude_settings: Path, monkeypatch) -> None:
        def boom(*a: Any, **k: Any) -> str:
            raise RuntimeError("broken")

        monkeypatch.setattr(init_mod, "alignment_warning", boom)
        result = CliRunner().invoke(
            main,
            ["init", "hook", "ensure", "--profile", "init-user", "--marker", "headroom-init-codex"],
            input=json.dumps({"hook_event_name": "SessionStart"}),
        )
        assert result.exit_code == 0
        assert result.output.strip() == ""


# ---------------------------------------------------------------------------
# 4. Doctor: readiness, hooks, live proxy elsewhere
# ---------------------------------------------------------------------------


class TestDoctorAdditions:
    def test_readiness(self) -> None:
        live = {"service": "headroom-proxy"}
        base = "http://127.0.0.1:8787"
        assert doctor_mod.check_proxy_readiness(None, None, base).status == doctor_mod.SKIP
        assert doctor_mod.check_proxy_readiness(live, None, base).status == doctor_mod.WARN
        assert (
            doctor_mod.check_proxy_readiness(live, {"ready": True}, base).status == doctor_mod.PASS
        )

    def _hooks_file(self, path: Path, command: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {
                                "matcher": "startup",
                                "hooks": [{"type": "command", "command": command}],
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_hooks_present(self, tmp_path: Path) -> None:
        exe = tmp_path / "headroom-bin"
        exe.write_text("", encoding="utf-8")
        hooks = tmp_path / "hooks.json"
        self._hooks_file(hooks, f"{exe.as_posix()} init hook ensure --marker headroom-init-codex")
        result = doctor_mod.check_installed_hooks(
            [("codex", "user", hooks, "headroom init -g codex")]
        )
        assert result.status == doctor_mod.PASS, result

    def test_hooks_missing(self, tmp_path: Path) -> None:
        result = doctor_mod.check_installed_hooks(
            [("claude", "user", tmp_path / "settings.json", "headroom init -g claude")]
        )
        assert result.status == doctor_mod.WARN
        assert "headroom init -g claude" in (result.hint or "")

    def test_hook_program_missing(self, tmp_path: Path) -> None:
        hooks = tmp_path / "settings.json"
        gone = (tmp_path / "gone" / "headroom").as_posix()
        self._hooks_file(hooks, f"{gone} init hook ensure --marker headroom-init-claude")
        result = doctor_mod.check_installed_hooks(
            [("claude", "user", hooks, "headroom init -g claude")]
        )
        assert result.status == doctor_mod.WARN
        assert "not found" in result.summary

    def test_no_init_means_skip(self) -> None:
        assert doctor_mod.check_installed_hooks([]).status == doctor_mod.SKIP

    def test_expected_hooks_from_manifests(self, home: Path, tmp_path: Path, monkeypatch) -> None:
        custom = tmp_path / "custom-codex"
        monkeypatch.setenv("CODEX_HOME", str(custom))
        monkeypatch.setattr(doctor_mod, "claude_settings_path", lambda: home / "settings.json")
        local = init_mod._local_profile(Path.cwd())
        expected = doctor_mod.expected_init_hooks(
            [
                _manifest("init-user", 9200, ["claude", "codex"]),
                _manifest(local, 8787, ["claude"]),
                _manifest("someone-else", 8787, ["claude"]),
            ],
            cwd=Path.cwd(),
        )
        assert ("codex", "user", custom / "hooks.json", "headroom init -g --port 9200 codex") in (
            expected
        )
        assert (
            "claude",
            "project",
            Path.cwd() / ".claude" / "settings.local.json",
            "headroom init claude",
        ) in expected
        assert len(expected) == 3

    def test_doctor_names_live_proxy_on_other_port(self, home: Path, monkeypatch) -> None:
        def fake_probe(url: str, timeout: float = 2.0) -> dict[str, Any] | None:
            if url == "http://127.0.0.1:9200/livez":
                return {"service": "headroom-proxy", "alive": True, "version": "1"}
            return None

        monkeypatch.setattr(doctor_mod, "probe_json", fake_probe)
        monkeypatch.setattr(doctor_mod, "list_manifests", lambda: [_manifest("prod", 9200)])
        monkeypatch.setattr(doctor_mod, "check_deployments", lambda manifests: None)
        monkeypatch.setattr(doctor_mod, "claude_settings_path", lambda: home / "settings.json")
        monkeypatch.setattr(doctor_mod, "savings_path", lambda: home / "savings.json")
        result = CliRunner().invoke(main, ["doctor", "--json"])
        payload = json.loads(result.output)
        assert payload["exit_code"] == 2  # still a failure at the probed port
        proxy = next(c for c in payload["checks"] if c["name"] == "proxy")
        assert "port 9200" in proxy["summary"]
        assert "headroom doctor --port 9200" in proxy["hint"]


# ---------------------------------------------------------------------------
# wrap selfheal: warn (never repoint) when a live proxy exists elsewhere
# ---------------------------------------------------------------------------


class TestSelfhealWarning:
    def test_warns_with_live_port(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(pd, "find_live_proxy_elsewhere", lambda requested, **k: 9200)
        wrap_mod._warn_live_proxy_after_selfheal({"port": 8787})
        err = capsys.readouterr().err
        assert "headroom wrap claude --port 9200" in err

    def test_silent_without_live_proxy(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(pd, "find_live_proxy_elsewhere", lambda requested, **k: None)
        wrap_mod._warn_live_proxy_after_selfheal({"port": 8787})
        assert capsys.readouterr().err == ""
