"""The retired rtk / lean-ctx integrations must be uninstalled, not just unshipped.

Deleting the integration code does nothing for a machine that already ran the
old default — the Claude ``PreToolUse`` hook, the vendored binaries, the MCP
registration and the injected hint-file guidance are all durable on disk. These
tests pin the two properties that make the cleanup safe to run unattended on
every ``wrap``: it removes everything Headroom put there, and it touches nothing
else.
"""

from __future__ import annotations

import json

import pytest

from headroom import context_tool_cleanup, paths


@pytest.fixture
def home(monkeypatch, tmp_path):
    """Point HOME, cwd and Headroom's bin dir at a scratch tree."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(paths, "bin_dir", lambda: tmp_path / ".headroom" / "bin")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return tmp_path


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_removes_hooks_for_both_tools_but_keeps_user_hooks(home):
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps(
            {
                "permissions": {"allow": ["Bash"]},
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {"type": "command", "command": "~/.claude/hooks/rtk-rewrite.sh"}
                            ]
                        },
                        {"hooks": [{"type": "command", "command": "lean-ctx hook rewrite"}]},
                        {"hooks": [{"type": "command", "command": "my-own-linter --check"}]},
                    ],
                    "SessionStart": [{"hooks": [{"type": "command", "command": "echo hi"}]}],
                },
            }
        ),
    )

    report = context_tool_cleanup.purge_context_tool_artifacts()

    payload = json.loads(settings.read_text())
    commands = [
        item["command"] for entry in payload["hooks"]["PreToolUse"] for item in entry["hooks"]
    ]
    assert commands == ["my-own-linter --check"]
    # Unrelated events and unrelated top-level keys survive untouched.
    assert payload["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "echo hi"
    assert payload["permissions"] == {"allow": ["Bash"]}
    assert any("hook" in line for line in report)


def test_removes_binaries_hook_scripts_and_backups(home):
    bin_dir = home / ".headroom" / "bin"
    rtk = _write(bin_dir / "rtk", "binary")
    lean = _write(bin_dir / "lean-ctx", "binary")
    script = _write(home / ".claude" / "hooks" / "lean-ctx-rewrite.sh", "#!/bin/sh\n")
    backup = _write(home / ".claude" / "hooks" / "lean-ctx-rewrite.sh.lean-ctx.bak", "#!/bin/sh\n")

    context_tool_cleanup.purge_context_tool_artifacts()

    assert not rtk.exists()
    assert not lean.exists()
    assert not script.exists()
    assert not backup.exists()


def test_leaves_a_users_own_binary_on_path_alone(home):
    """A real file in ~/.local/bin is not ours to reclaim — only our symlink is."""
    own = _write(home / ".local" / "bin" / "lean-ctx", "my own build")
    managed = _write(home / ".headroom" / "bin" / "rtk", "binary")
    link = home / ".local" / "bin" / "rtk"
    link.symlink_to(managed)

    context_tool_cleanup.purge_context_tool_artifacts()

    assert own.exists() and own.read_text() == "my own build"
    assert not link.exists()


def test_removes_mcp_entry_and_preserves_siblings(home):
    config = _write(
        home / ".claude.json",
        json.dumps(
            {
                "projects": {"/some/path": {"history": []}},
                "mcpServers": {
                    "lean-ctx": {"command": "lean-ctx", "args": ["mcp"]},
                    "headroom": {"command": "headroom", "args": ["mcp"]},
                },
            }
        ),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    payload = json.loads(config.read_text())
    assert list(payload["mcpServers"]) == ["headroom"]
    assert payload["projects"] == {"/some/path": {"history": []}}


def test_strips_guidance_fence_but_keeps_surrounding_prose(home):
    agents = _write(
        home / "project" / "AGENTS.md",
        "# My project\n\nMy own notes.\n\n"
        "<!-- headroom:rtk-instructions -->\nAlways prefix with rtk.\n"
        "<!-- /headroom:rtk-instructions -->\n",
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    content = agents.read_text()
    assert "rtk" not in content
    assert "My own notes." in content
    assert content.startswith("# My project")


def test_skips_malformed_json_instead_of_clobbering_it(home):
    settings = _write(home / ".claude" / "settings.json", '{"permissions": {oops')

    report = context_tool_cleanup.purge_context_tool_artifacts()

    assert settings.read_text() == '{"permissions": {oops'
    assert any("skipped" in line for line in report)


def test_is_idempotent(home):
    _write(home / ".headroom" / "bin" / "rtk", "binary")
    _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": "rtk rewrite"}]}]}}),
    )

    assert context_tool_cleanup.purge_context_tool_artifacts()
    # Steady state after the first run: nothing left to report.
    assert context_tool_cleanup.purge_context_tool_artifacts() == []


def test_no_op_on_a_clean_machine(home):
    assert context_tool_cleanup.purge_context_tool_artifacts() == []


def test_purge_reports_on_stderr_so_json_stdout_stays_parseable(home):
    """`wrap openclaw --prepare-only` emits machine-readable JSON as its whole contract.

    The purge runs from the `wrap` group callback, i.e. before that JSON is
    written. Reporting on stdout prepended a human line to it and broke every
    ``json.loads(stdout)`` consumer — but only on the single run that actually
    had something to remove, so a clean CI machine never caught it.
    """
    from click.testing import CliRunner

    from headroom.cli.main import main

    _write(home / ".headroom" / "bin" / "rtk", "binary")

    result = CliRunner().invoke(
        main, ["wrap", "openclaw", "--prepare-only", "--gateway-provider-id", "codex"]
    )

    assert result.exit_code == 0, result.output
    # Whole of stdout must still parse — no cleanup preamble.
    assert json.loads(result.stdout)["enabled"] is True
    assert "Retired CLI context tool cleanup" in result.stderr


def test_help_does_not_purge(home, monkeypatch):
    """`--help` must stay read-only — reading help should not delete files."""
    from click.testing import CliRunner

    from headroom.cli.main import main

    binary = _write(home / ".headroom" / "bin" / "rtk", "binary")
    monkeypatch.setattr("sys.argv", ["headroom", "wrap", "codex", "--help"])

    result = CliRunner().invoke(main, ["wrap", "codex", "--help"])

    assert result.exit_code == 0
    assert binary.exists(), "--help performed filesystem cleanup"


def test_selfheal_does_not_purge(home, monkeypatch):
    """`wrap selfheal` runs from a SessionStart hook — no config surgery there.

    It fires on every new conversation, where rewriting ~/.claude.json would race
    Claude Code's own writer.
    """
    from click.testing import CliRunner

    from headroom.cli.main import main

    binary = _write(home / ".headroom" / "bin" / "rtk", "binary")
    monkeypatch.setattr("sys.argv", ["headroom", "wrap", "selfheal"])

    CliRunner().invoke(main, ["wrap", "selfheal", "--marker", "headroom-wrap-selfheal"])

    assert binary.exists(), "selfheal performed filesystem cleanup"
