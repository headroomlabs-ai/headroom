"""Enola discovery and ownership-safe code-memory switching."""

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from headroom.cli import wrap
from headroom.cli.main import main
from headroom.mcp_registry import build_serena_spec, enola
from headroom.mcp_registry.base import RegisterResult, RegisterStatus, ServerSpec
from headroom.mcp_registry.claude import SCOPE_LOCAL, SCOPE_USER, ClaudeRegistrar
from headroom.mcp_registry.ledger import headroom_installed_matching, record_install


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
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    monkeypatch.setenv("HEADROOM_CODE_MEMORY", "enola")
    monkeypatch.delenv("HEADROOM_SERENA_INSTRUCTIONS", raising=False)
    monkeypatch.delenv("HEADROOM_CODE_MEMORY_INSTRUCTIONS", raising=False)
    monkeypatch.setenv("ENOLA_PATH", str(executable(tmp_path / "bin" / "enola")))
    (tmp_path / ".claude").mkdir()
    return ClaudeRegistrar(
        claude_cli=None, home_dir=tmp_path, project_dir=tmp_path, scope=SCOPE_LOCAL
    )


def install_owned(registrar, spec, *, scope=None):
    assert registrar.register_server(spec).ok
    record_install(
        registrar.name, spec, ownership_key=registrar.ownership_key(spec.name, scope=scope)
    )


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
