"""Enola discovery and ownership-safe code-memory switching."""

import hashlib
import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from headroom.cli import wrap
from headroom.cli.main import main
from headroom.mcp_registry import build_serena_spec, enola
from headroom.mcp_registry.base import RegisterResult, RegisterStatus, ServerSpec
from headroom.mcp_registry.claude import SCOPE_LOCAL, SCOPE_USER, ClaudeRegistrar
from headroom.mcp_registry.ledger import (
    headroom_installed_matching,
    record_install,
    spec_fingerprint,
)


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_binary_discovery_precedence(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    installed = executable(scripts / ("enola.exe" if enola.os.name == "nt" else "enola"))
    on_path = executable(tmp_path / "path" / "enola")
    override = executable(tmp_path / "override" / "enola")
    monkeypatch.delenv("ENOLA_PATH", raising=False)
    monkeypatch.setattr(enola.sysconfig, "get_path", lambda name: str(scripts))
    monkeypatch.setattr(enola.shutil, "which", lambda name: str(on_path))
    assert enola.find_enola_binary() == str(installed.resolve())
    installed.unlink()
    assert enola.find_enola_binary() == str(on_path.resolve())
    monkeypatch.setenv("ENOLA_PATH", str(override))
    assert enola.find_enola_binary() == str(override.resolve())
    override.unlink()
    with pytest.raises(ValueError, match="ENOLA_PATH"):
        enola.find_enola_binary()


def test_missing_binary_has_install_hint(monkeypatch, tmp_path):
    monkeypatch.delenv("ENOLA_PATH", raising=False)
    monkeypatch.setattr(enola.sysconfig, "get_path", lambda name: str(tmp_path))
    monkeypatch.setattr(enola.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match=r"headroom-ai\[enola\]"):
        enola.build_enola_spec()


@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for name in ("GROK_HOME", "CODEX_HOME", "OPENCODE_CONFIG", "OPENCODE_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "enola")
    monkeypatch.delenv("HEADROOM_SERENA_INSTRUCTIONS", raising=False)
    monkeypatch.delenv("HEADROOM_CODE_MEMORY_INSTRUCTIONS", raising=False)
    monkeypatch.setenv("ENOLA_PATH", str(executable(tmp_path / "bin" / "enola")))
    (tmp_path / ".claude").mkdir()
    return ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, project_dir=tmp_path, scope=SCOPE_LOCAL
    )


@pytest.mark.parametrize("edited_global", [False, True])
def test_user_scope_reconciliation_checks_the_user_entry_not_its_project_shadow(
    project, tmp_path, edited_global
):
    registrar = ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, project_dir=tmp_path, scope=SCOPE_USER
    )
    old = ServerSpec("enola", "/old/enola")
    install_owned(registrar, old)
    shadow = old if edited_global else ServerSpec("enola", "/project/enola")
    assert project.register_server(shadow).ok
    if edited_global:
        customized = ServerSpec("enola", "/user/enola", env={"CUSTOM": "keep"})
        assert registrar.register_server(customized, force=True).ok
        assert not wrap._setup_enola_mcp(registrar)
        assert registrar.get_server("enola", scope=SCOPE_USER) == customized
    else:
        assert wrap._setup_enola_mcp(registrar)
        assert registrar.get_server("enola", scope=SCOPE_USER) == enola.build_enola_spec()
    assert project.get_server("enola", scope=SCOPE_LOCAL) == shadow


def test_none_preserves_user_managed_enola_guidance(project, monkeypatch):
    assert project.register_server(ServerSpec("enola", "/custom/enola")).ok
    original = "# User guidance\n\n" + wrap.ENOLA_INSTRUCTIONS_BLOCK
    Path("CLAUDE.md").write_text(original, encoding="utf-8", newline="\n")
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "none")
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert Path("CLAUDE.md").read_bytes() == original.encode()


def test_failed_removal_preserves_enola_guidance(project, monkeypatch):
    install_owned(project, enola.build_enola_spec())
    original = "# User guidance\n\n" + wrap.ENOLA_INSTRUCTIONS_BLOCK
    Path("CLAUDE.md").write_text(original, encoding="utf-8", newline="\n")
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "none")
    monkeypatch.setattr(project, "unregister_server", lambda *a, **k: False)
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert Path("CLAUDE.md").read_bytes() == original.encode()


def test_shared_guidance_survives_until_the_last_agent_removes_enola(project, tmp_path):
    from headroom.mcp_registry import CodexRegistrar, OpencodeRegistrar

    codex = CodexRegistrar(home_dir=tmp_path)
    opencode = OpencodeRegistrar(config_path=tmp_path / ".config/opencode/opencode.json")
    spec = enola.build_enola_spec()
    for registrar in (codex, opencode):
        assert registrar.register_server(spec).ok
        record_install(registrar.name, spec)
    original = "# Shared guidance\n\n" + wrap.ENOLA_INSTRUCTIONS_BLOCK
    Path("AGENTS.md").write_text(original, encoding="utf-8", newline="\n")
    assert wrap._disable_enola_mcp(codex) == "removed"
    assert Path("AGENTS.md").read_bytes() == original.encode()
    assert wrap._disable_enola_mcp(opencode) == "removed"
    assert Path("AGENTS.md").read_bytes() == b"# Shared guidance"


@pytest.mark.parametrize("was_owned", [False, True])
def test_disabled_opencode_enola_does_not_replace_serena(project, tmp_path, was_owned):
    from headroom.mcp_registry import OpencodeRegistrar

    path = tmp_path / ".config/opencode/opencode.json"
    registrar = OpencodeRegistrar(config_path=path)
    serena = build_serena_spec("agent")
    assert registrar.register_server(serena).ok
    record_install(registrar.name, serena)
    spec = enola.build_enola_spec()
    assert registrar.register_server(spec).ok
    if was_owned:
        record_install(registrar.name, spec)
    data = json.loads(path.read_text())
    data["mcp"]["enola"]["enabled"] = False
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    wrap._setup_coding_compressor(registrar, serena_context="agent", force=True)
    assert registrar.get_server("serena") == serena
    assert registrar.get_server("enola").enabled is False
    assert not headroom_installed_matching(registrar.name, registrar.get_server("enola"))


def test_enabled_specs_keep_legacy_ledger_fingerprints():
    spec = ServerSpec("enola", "/bin/enola", env={"SETTING": "value"})
    legacy = {"name": spec.name, "command": spec.command, "args": [], "env": spec.env}
    expected = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert spec_fingerprint(spec) == expected
    assert (
        spec_fingerprint(ServerSpec("enola", "/bin/enola", env=spec.env, enabled=False)) != expected
    )


@pytest.mark.windows_newline
def test_enola_guidance_is_utf8_lf_on_disk(project, monkeypatch):
    path = Path("CLAUDE.md")
    path.write_bytes("# Café\r\n".encode())
    monkeypatch.setenv("HEADROOM_CODE_MEMORY_INSTRUCTIONS", "1")
    wrap._inject_enola_instructions(path)
    assert b"\r" not in path.read_bytes()
    assert "Café".encode() in path.read_bytes()
    wrap._remove_unused_code_memory_instructions(project, "enola", wrap.ENOLA_INSTRUCTIONS_BLOCK)
    assert path.read_bytes() == "# Café\n".encode()


@pytest.mark.parametrize("customized", [False, True])
def test_opencode_unwrap_restores_routing_but_preserves_customized_enola(
    project, tmp_path, customized
):
    config = tmp_path / ".config/opencode/opencode.json"
    config.parent.mkdir(parents=True)
    original = '{"model":"openai/gpt-4o","theme":"dark"}\n'
    config.write_text(original, encoding="utf-8", newline="\n")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "wrap",
            "opencode",
            "--prepare-only",
            "--code-memory",
            "enola",
            "--code-memory-instructions",
            "--no-mcp",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(config.read_text())
    if customized:
        data["mcp"]["enola"]["environment"]["CUSTOM"] = "keep"
        data["mcp"]["enola"]["enabled"] = False
        data["mcp"]["enola"]["timeout"] = 123456
        config.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    saved = data["mcp"]["enola"]
    result = runner.invoke(main, ["unwrap", "opencode", "--no-stop-proxy"])
    assert result.exit_code == 0, result.output
    if customized:
        restored = json.loads(config.read_text())
        assert restored["mcp"]["enola"] == saved
        assert restored["model"] == "openai/gpt-4o"
        assert wrap.ENOLA_INSTRUCTIONS_BLOCK in Path("AGENTS.md").read_text()
    else:
        assert config.read_bytes() == original.encode()
        assert wrap.ENOLA_INSTRUCTIONS_BLOCK not in Path("AGENTS.md").read_text()


def install_owned(registrar, spec, *, scope=None):
    assert registrar.register_server(spec).ok
    record_install(
        registrar.name, spec, ownership_key=registrar.ownership_key(spec.name, scope=scope)
    )


def test_opencode_restore_with_invalid_backup_preserves_customized_entry(project, tmp_path):
    config = tmp_path / "opencode.json"
    original = '{"mcp":{"enola":{"command":["/custom/enola"],"enabled":false}}}\n'
    config.write_text(original, encoding="utf-8", newline="\n")
    backup = tmp_path / "opencode.json.headroom-backup"
    backup.write_text("invalid json", encoding="utf-8")
    with pytest.raises(click.ClickException, match="invalid OpenCode backup"):
        wrap._restore_opencode_backup(config, backup)
    assert config.read_bytes() == original.encode()
    assert backup.exists()


@pytest.mark.parametrize("prepare_only", [False, True])
def test_fresh_grok_registers_enola_before_launch(project, tmp_path, monkeypatch, prepare_only):
    from headroom.mcp_registry import GrokRegistrar

    launches = []
    monkeypatch.setattr(wrap, "_require_binary", lambda *a, **k: "/bin/grok")
    monkeypatch.setattr(wrap, "_launch_tool", lambda **kwargs: launches.append(kwargs))
    args = ["wrap", "grok", "--code-memory", "enola", "--no-mcp"]
    if prepare_only:
        args.append("--prepare-only")
    assert not (tmp_path / ".grok").exists()
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert GrokRegistrar(home_dir=tmp_path).get_server("enola") == enola.build_enola_spec()
    assert bool(launches) is not prepare_only


def test_missing_grok_binary_does_not_initialize_config(project, tmp_path, monkeypatch):
    monkeypatch.setattr(wrap.shutil, "which", lambda name: None)
    result = CliRunner().invoke(main, ["wrap", "grok", "--code-memory", "enola", "--no-mcp"])
    assert result.exit_code == 1, result.output
    assert not (tmp_path / ".grok").exists()


def test_switch_to_enola_is_project_scoped_and_idempotent(project):
    install_owned(project, build_serena_spec("claude-code"))
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    spec = project.get_server("enola", scope=SCOPE_LOCAL)
    assert spec == enola.build_enola_spec()
    assert spec.env == {"ENOLA_NO_UPDATE_CHECK": "1", "ENOLA_NO_PROMPTS": "1"}
    assert project.get_server("enola", scope=SCOPE_USER) is None
    assert project.get_server("serena") is None
    assert headroom_installed_matching("claude", spec, ownership_key=project.ownership_key("enola"))
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("enola") == spec
    assert not Path("CLAUDE.md").exists()


def test_failed_discovery_keeps_serena(project, monkeypatch):
    spec = build_serena_spec("claude-code")
    install_owned(project, spec)
    monkeypatch.setenv("ENOLA_PATH", "/missing/enola")
    with pytest.raises(click.ClickException, match="ENOLA_PATH"):
        wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("serena") == spec


def test_conflicting_user_enola_keeps_serena(project):
    serena = build_serena_spec("claude-code")
    install_owned(project, serena)
    user = ServerSpec("enola", "/custom/enola")
    assert project.register_server(user).ok
    wrap._setup_coding_compressor(project, serena_context="claude-code", force=True)
    assert project.get_server("enola") == user
    assert project.get_server("serena") == serena


def test_owned_enola_drift_is_updated(project):
    install_owned(project, ServerSpec("enola", "/old/enola"))
    assert wrap._setup_enola_mcp(project)
    assert project.get_server("enola") == enola.build_enola_spec()


@pytest.mark.parametrize(
    "forced_status", [RegisterStatus.REGISTERED, RegisterStatus.ALREADY, RegisterStatus.FAILED]
)
def test_owned_update_tracks_success_and_preserves_ledger_on_failure(
    project, monkeypatch, forced_status
):
    old = ServerSpec("enola", "/old/enola")
    install_owned(project, old)
    original_register = project.register_server

    def register(spec, *, force=False):
        if not force:
            return original_register(spec)
        if forced_status != RegisterStatus.FAILED:
            assert original_register(spec, force=True).ok
        return RegisterResult(forced_status)

    monkeypatch.setattr(project, "register_server", register)
    success = forced_status != RegisterStatus.FAILED
    assert wrap._setup_enola_mcp(project) == success
    expected = enola.build_enola_spec() if success else old
    assert project.get_server("enola") == expected
    assert headroom_installed_matching(
        project.name, expected, ownership_key=project.ownership_key("enola")
    )
    if success:
        assert wrap._disable_enola_mcp(project) == "removed"


def test_relocated_binary_is_reconciled_and_remains_owned(project, monkeypatch, tmp_path):
    old = enola.build_enola_spec()
    install_owned(project, old)
    relocated = executable(tmp_path / "new-environment" / "bin" / "enola")
    monkeypatch.setenv("ENOLA_PATH", str(relocated))
    assert wrap._setup_enola_mcp(project)
    assert project.get_server("enola").command == str(relocated.resolve())
    assert wrap._disable_enola_mcp(project) == "removed"


def test_user_managed_serena_survives_switch(project):
    user = ServerSpec("serena", "/custom/serena")
    assert project.register_server(user).ok
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("enola") is not None
    assert project.get_server("serena") == user


def test_owned_global_enola_retires_when_scoped_to_project(project, tmp_path):
    global_registrar = ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, project_dir=tmp_path, scope=SCOPE_USER
    )
    old = ServerSpec("enola", "/old/enola")
    install_owned(global_registrar, old)
    assert wrap._setup_enola_mcp(project)
    assert project.get_server("enola", scope=SCOPE_USER) is None
    assert not headroom_installed_matching("claude", old)
    assert project.get_server("enola", scope=SCOPE_LOCAL) is not None


def test_user_global_enola_survives_project_setup(project, tmp_path):
    global_registrar = ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, project_dir=tmp_path, scope=SCOPE_USER
    )
    user = ServerSpec("enola", "/custom/enola")
    assert global_registrar.register_server(user).ok
    assert wrap._setup_enola_mcp(project)
    assert project.get_server("enola", scope=SCOPE_USER) == user


def test_cleanup_preserves_user_global_entry_and_clears_local_ownership(project, tmp_path):
    global_registrar = ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, project_dir=tmp_path, scope=SCOPE_USER
    )
    user = ServerSpec("enola", "/custom/enola")
    assert global_registrar.register_server(user).ok
    owned = enola.build_enola_spec()
    install_owned(project, owned)
    assert wrap._disable_enola_mcp(project) == "removed"
    assert project.get_server("enola", scope=SCOPE_LOCAL) is None
    assert project.get_server("enola", scope=SCOPE_USER) == user
    assert not headroom_installed_matching(
        "claude", owned, ownership_key=project.ownership_key("enola")
    )


def test_matching_user_enola_is_never_adopted(project, monkeypatch):
    user = enola.build_enola_spec()
    assert project.register_server(user).ok
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "none")
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("enola") == user


def test_switch_back_only_removes_enola_after_serena_success(project, monkeypatch):
    install_owned(project, enola.build_enola_spec())
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "serena")
    monkeypatch.setattr(wrap, "_setup_serena_mcp", lambda *a, **k: False)
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("enola") is not None
    monkeypatch.setattr(wrap, "_setup_serena_mcp", lambda *a, **k: True)
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("enola") is None


def test_serena_selection_preserves_user_enola_and_its_guidance(project, monkeypatch):
    user = ServerSpec("enola", "/custom/enola")
    assert project.register_server(user).ok
    original = "# My project\n\n" + wrap.ENOLA_INSTRUCTIONS_BLOCK
    Path("CLAUDE.md").write_text(original)
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "serena")
    monkeypatch.setattr(wrap, "_setup_serena_mcp", lambda *a, **k: True)
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("enola") == user
    assert Path("CLAUDE.md").read_text() == original


def test_serena_selection_removes_owned_enola_and_its_guidance(project, monkeypatch):
    install_owned(project, enola.build_enola_spec())
    original = "# My project\n"
    Path("CLAUDE.md").write_text(original + "\n\n" + wrap.ENOLA_INSTRUCTIONS_BLOCK)
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "serena")
    monkeypatch.setattr(wrap, "_setup_serena_mcp", lambda *a, **k: True)
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("enola") is None
    assert Path("CLAUDE.md").read_text() == original


@pytest.mark.parametrize(
    ("flag", "expected_env", "other_env"),
    [
        (
            "--code-memory-instructions",
            "HEADROOM_CODE_MEMORY_INSTRUCTIONS",
            "HEADROOM_SERENA_INSTRUCTIONS",
        ),
        (
            "--serena-instructions",
            "HEADROOM_SERENA_INSTRUCTIONS",
            "HEADROOM_CODE_MEMORY_INSTRUCTIONS",
        ),
    ],
)
def test_instruction_flags_set_their_own_environment_variable(
    project, flag, expected_env, other_env
):
    import os

    @click.command()
    @wrap._code_memory_instructions_option
    @wrap._serena_instructions_option
    def command():
        assert os.environ[expected_env] == "1"
        assert other_env not in os.environ
        assert wrap._serena_instructions_opt_in()

    result = CliRunner().invoke(command, [flag])
    assert result.exit_code == 0, result.exception


def test_none_cleans_owned_backends_and_intact_guidance(project, monkeypatch):
    install_owned(project, build_serena_spec("claude-code"))
    install_owned(project, enola.build_enola_spec())
    original = "# My project\nKeep this text.\n"
    Path("CLAUDE.md").write_text(
        original + "\n\n" + wrap.SERENA_INSTRUCTIONS_BLOCK + "\n\n" + wrap.ENOLA_INSTRUCTIONS_BLOCK
    )
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "none")
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert project.get_server("enola") is None
    assert project.get_server("serena") is None
    assert Path("CLAUDE.md").read_text() == original


def test_instructions_opt_in_replaces_generated_serena_block(project, monkeypatch):
    original = "# User guidance\n"
    Path("CLAUDE.md").write_text(original + "\n\n" + wrap.SERENA_INSTRUCTIONS_BLOCK)
    monkeypatch.setenv("HEADROOM_CODE_MEMORY_INSTRUCTIONS", "1")
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    wrap._setup_coding_compressor(project, serena_context="claude-code")
    assert Path("CLAUDE.md").read_text() == original + "\n\n" + wrap.ENOLA_INSTRUCTIONS_BLOCK


def test_modified_guidance_is_preserved(project):
    changed = wrap.ENOLA_INSTRUCTIONS_BLOCK.replace("Code Navigation", "Custom Navigation")
    Path("CLAUDE.md").write_text(changed)
    wrap._disable_enola_mcp(project)
    assert Path("CLAUDE.md").read_text() == changed


@pytest.mark.parametrize("keep_mcp", [False, True])
def test_unwrap_claude_cleans_owned_enola(project, monkeypatch, keep_mcp):
    import headroom.mcp_registry as registry

    install_owned(project, enola.build_enola_spec())
    monkeypatch.setattr(registry, "ClaudeRegistrar", lambda **kwargs: project)
    monkeypatch.setattr(wrap, "_stop_local_proxy_for_unwrap", lambda *a, **k: "not_running")
    monkeypatch.setattr(wrap, "_unwrap_claude_route_cleanup", lambda *a, **k: {})
    args = ["unwrap", "claude", "--no-stop-proxy"]
    if keep_mcp:
        args.append("--keep-mcp")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert (project.get_server("enola") is not None) == keep_mcp, result.output
    assert ("Removed Headroom-installed Enola" in result.output) != keep_mcp


def test_strands_selects_enola_without_serena(project):
    pytest.importorskip("strands")
    from headroom.integrations.strands.bundle import HeadroomBundle

    bundle = HeadroomBundle(code_memory="enola", enable_headroom_mcp=False)
    assert bundle.tools == [bundle.enola_mcp]
    assert bundle.enola_mcp is not None
    assert bundle.serena_mcp is None
    assert HeadroomBundle(enable_headroom_mcp=False, enable_serena_mcp=False).tools == []
    with pytest.raises(ValueError, match="code_memory"):
        HeadroomBundle(code_memory="unknown")
