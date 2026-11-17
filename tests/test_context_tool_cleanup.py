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
import os
import sys

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
    bin_dir = paths.bin_dir()
    hooks_dir = home / ".claude" / "hooks"
    # Managed: a script whose body execs the Headroom-installed binary.
    managed_script = _write(
        hooks_dir / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {bin_dir / "lean-ctx"} "$@"\n',
    )
    # User-owned: same marker-matching filename, but the body execs the
    # user's own install — no path inside bin_dir anywhere.
    user_script = _write(
        hooks_dir / "lean-ctx-redirect.sh",
        '#!/bin/sh\nexec /usr/bin/lean-ctx "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps(
            {
                "permissions": {"allow": ["Bash"]},
                "hooks": {
                    "PreToolUse": [
                        {"hooks": [{"type": "command", "command": str(managed_script)}]},
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"{bin_dir / 'lean-ctx'} hook rewrite",
                                }
                            ]
                        },
                        {"hooks": [{"type": "command", "command": str(user_script)}]},
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
    assert commands == [str(user_script), "my-own-linter --check"]
    # Unrelated events and unrelated top-level keys survive untouched.
    assert payload["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "echo hi"
    assert payload["permissions"] == {"allow": ["Bash"]}
    assert any("hook" in line for line in report)


def test_removes_binaries_hook_scripts_and_backups(home):
    bin_dir = paths.bin_dir()
    hooks_dir = home / ".claude" / "hooks"
    rtk = _write(bin_dir / "rtk", "binary")
    lean = _write(bin_dir / "lean-ctx", "binary")
    script = _write(
        hooks_dir / "lean-ctx-rewrite.sh", f'#!/bin/sh\nexec {bin_dir / "lean-ctx"} "$@"\n'
    )
    backup = _write(
        hooks_dir / "lean-ctx-rewrite.sh.lean-ctx.bak",
        f'#!/bin/sh\nexec {bin_dir / "lean-ctx"} "$@"\n',
    )
    managed_rtk_script = _write(
        hooks_dir / "rtk-rewrite.sh", f'#!/bin/sh\nexec {bin_dir / "rtk"} "$@"\n'
    )
    managed_rtk_digest = _write(hooks_dir / ".rtk-hook.sha256", "deadbeef\n")

    context_tool_cleanup.purge_context_tool_artifacts()

    assert not rtk.exists()
    assert not lean.exists()
    assert not script.exists()
    assert not backup.exists()
    # .rtk-hook.sha256 follows rtk-rewrite.sh's classification: both managed,
    # both removed.
    assert not managed_rtk_script.exists()
    assert not managed_rtk_digest.exists()


def test_leaves_a_users_own_rtk_digest_alone(home):
    """The digest is a hex hash, so it can only follow the script it authenticates."""
    hooks_dir = home / ".claude" / "hooks"
    script = _write(hooks_dir / "rtk-rewrite.sh", '#!/bin/sh\nexec /usr/bin/rtk "$@"\n')
    digest = _write(hooks_dir / ".rtk-hook.sha256", "cafef00d\n")

    context_tool_cleanup.purge_context_tool_artifacts()

    assert script.exists()
    assert digest.exists()


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
    bin_dir = paths.bin_dir()
    config = _write(
        home / ".claude.json",
        json.dumps(
            {
                "projects": {"/some/path": {"history": []}},
                "mcpServers": {
                    "lean-ctx": {"command": str(bin_dir / "lean-ctx"), "args": ["mcp"]},
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
    """Re-running the purge body reports nothing new.

    Normally the completion stamp stops a second run, but a workspace the
    stamp cannot be written to falls back to running every time — so the body
    itself has to stay idempotent. Removing the stamp between runs is what
    that machine does.
    """
    bin_dir = paths.bin_dir()
    _write(bin_dir / "rtk", "binary")
    script = _write(
        home / ".claude" / "hooks" / "rtk-rewrite.sh", f'#!/bin/sh\nexec {bin_dir / "rtk"} "$@"\n'
    )
    _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    assert context_tool_cleanup.purge_context_tool_artifacts()
    (bin_dir.parent / ".context-tools-purged").unlink()
    # Steady state after the first run: nothing left to report.
    assert context_tool_cleanup.purge_context_tool_artifacts() == []


def test_no_op_on_a_clean_machine(home):
    assert context_tool_cleanup.purge_context_tool_artifacts() == []


def test_a_completed_purge_never_runs_again(home):
    """Machine-global artifacts stay stamped-done: no per-launch re-audit.

    A tool installed under the *global* half (hooks, binaries) after the
    migration is not a leftover, so a later run must leave it alone without
    even looking — this is what stops `headroom wrap` from re-litigating
    machine-global state on every launch, forever. Project- and config-
    directory-scoped guidance is a different story: see
    `test_a_completed_purge_still_cleans_a_different_project`.
    """
    bin_dir = paths.bin_dir()
    bin_dir.parent.mkdir(parents=True)

    assert context_tool_cleanup.purge_context_tool_artifacts() == []
    assert (bin_dir.parent / ".context-tools-purged").exists()

    # Artifacts that would otherwise be removed, installed after the migration.
    binary = _write(bin_dir / "lean-ctx", "binary")
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {bin_dir / "lean-ctx"} "$@"\n',
    )

    assert context_tool_cleanup.purge_context_tool_artifacts() == []
    assert binary.exists()
    assert script.exists()


def test_a_completed_purge_still_cleans_a_different_project(home, monkeypatch):
    """The one-time stamp is machine-global; project guidance is not.

    A completed run in project A must not leave project B's fenced guidance in
    place forever — the stamp only ever covered a snapshot of ``Path.cwd()``.
    """
    paths.bin_dir().parent.mkdir(parents=True)
    assert context_tool_cleanup.purge_context_tool_artifacts() == []
    assert (paths.bin_dir().parent / ".context-tools-purged").exists()

    project_b = home / "project-b"
    project_b.mkdir()
    agents = _write(
        project_b / "AGENTS.md",
        "# Project B\n\n<!-- headroom:rtk-instructions -->\nAlways prefix with rtk.\n"
        "<!-- /headroom:rtk-instructions -->\n",
    )
    monkeypatch.chdir(project_b)

    report = context_tool_cleanup.purge_context_tool_artifacts()

    assert "rtk" not in agents.read_text()
    assert any(str(agents) in line for line in report)


def test_a_completed_purge_still_inspects_a_repointed_codex_home(home, monkeypatch):
    """``CODEX_HOME`` can point somewhere new after the stamp; that target is not exempt."""
    paths.bin_dir().parent.mkdir(parents=True)
    assert context_tool_cleanup.purge_context_tool_artifacts() == []
    assert (paths.bin_dir().parent / ".context-tools-purged").exists()

    new_codex_home = home / "elsewhere-codex"
    new_codex_home.mkdir()
    agents = _write(
        new_codex_home / "AGENTS.md",
        "<!-- headroom:rtk-instructions -->\nAlways prefix with rtk.\n"
        "<!-- /headroom:rtk-instructions -->\n",
    )
    monkeypatch.setenv("CODEX_HOME", str(new_codex_home))

    report = context_tool_cleanup.purge_context_tool_artifacts()

    # Nothing but the fence was in the file, so it is removed outright.
    assert not agents.exists()
    assert any(str(agents) in line for line in report)


def test_a_scoped_deferral_does_not_withhold_the_global_stamp(home):
    """A scoped step's leftover must not re-run the whole global half forever.

    Only the invocation-scoped half is unprovable here (malformed Continue
    config); the machine-global half has nothing to defer, so its stamp must
    still be written — otherwise a permanently-broken ``.continue/config.json``
    would force every hook/binary/MCP step to be re-walked on every launch.
    """
    paths.bin_dir().parent.mkdir(parents=True)
    _write(home / "project" / ".continue" / "config.json", "{not json")

    report = context_tool_cleanup.purge_context_tool_artifacts()

    assert any(line.startswith("skipped ") for line in report)
    assert (paths.bin_dir().parent / ".context-tools-purged").exists()


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


def test_leaves_a_users_own_mcp_entry_alone(home):
    config = _write(
        home / ".claude.json",
        json.dumps({"mcpServers": {"lean-ctx": {"command": "lean-ctx", "args": ["mcp"]}}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    payload = json.loads(config.read_text())
    assert payload["mcpServers"] == {"lean-ctx": {"command": "lean-ctx", "args": ["mcp"]}}


def test_leaves_a_users_own_hook_script_and_hook_entry_alone(home):
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        '#!/bin/sh\nexec /usr/bin/lean-ctx "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert script.exists()
    payload = json.loads(settings.read_text())
    assert payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == str(script)


def test_removes_the_managed_hook_script_and_its_hook_entry(home):
    bin_dir = paths.bin_dir()
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {bin_dir / "lean-ctx"} "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert not script.exists()
    payload = json.loads(settings.read_text())
    assert "hooks" not in payload


def test_removes_a_hook_entry_pointing_at_the_managed_binary_directly(home):
    bin_dir = paths.bin_dir()
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"{bin_dir / 'lean-ctx'} hook rewrite",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    payload = json.loads(settings.read_text())
    assert "hooks" not in payload


def test_matches_a_managed_path_written_in_tilde_form(home):
    """The dangling-hook regression test: bin_dir is <tmp>/.headroom/bin, and the
    script references it in unexpanded tilde form — the guard must normalize
    both sides before comparing, or it wrongly treats this as unprovable and
    leaves a hook pointing at a script Headroom itself no longer manages.
    """
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        '#!/bin/sh\nexec ~/.headroom/bin/lean-ctx "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert not script.exists()
    payload = json.loads(settings.read_text())
    assert "hooks" not in payload


def test_leaves_a_hook_script_in_a_sibling_bin_named_directory_alone(home):
    """A directory that merely starts with the bin dir's name is not the bin dir.

    ``<workspace>/.headroom/binaries`` shares a prefix with
    ``<workspace>/.headroom/bin`` but is a different, user-owned directory —
    a naive substring match (no directory-boundary check) would treat the
    shared prefix as a reference to the managed bin dir and wrongly delete
    this script.
    """
    bin_dir = paths.bin_dir()
    sibling = bin_dir.parent / "binaries"
    own_binary = _write(sibling / "lean-ctx", "my own build")
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {own_binary} "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert script.exists()
    payload = json.loads(settings.read_text())
    assert payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == str(script)


def test_leaves_an_mcp_entry_in_a_sibling_bin_named_directory_alone(home):
    """Same sibling-directory hazard as above, for the MCP-entry command check."""
    bin_dir = paths.bin_dir()
    sibling_command = str(bin_dir.parent / "bin-backup" / "lean-ctx")
    config = _write(
        home / ".claude.json",
        json.dumps({"mcpServers": {"lean-ctx": {"command": sibling_command, "args": ["mcp"]}}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    payload = json.loads(config.read_text())
    assert payload["mcpServers"] == {"lean-ctx": {"command": sibling_command, "args": ["mcp"]}}


def test_leaves_a_parent_traversal_path_through_the_bin_dir_alone(home):
    """``bin/../evil`` contains the managed prefix as literal text but does not
    resolve inside it — the guard must collapse ``..`` before comparing, or a
    crafted (or coincidental) traversal path would be treated as Headroom's.
    """
    bin_dir = paths.bin_dir()
    evil_binary = _write(bin_dir.parent / "evil" / "lean-ctx", "not ours")
    traversal_command = f"{bin_dir}/../evil/lean-ctx"
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {traversal_command} "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert script.exists()
    assert evil_binary.exists()
    payload = json.loads(settings.read_text())
    assert payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == str(script)


def test_leaves_a_hook_script_whose_body_has_the_bin_dir_as_a_path_segment_alone(home):
    """``/prefix<bin_dir>/lean-ctx`` contains the managed prefix as literal text
    but at a position with no boundary before it — the run of characters
    immediately preceding the match is a filename character (``x``), not
    whitespace or a :data:`_PATH_BOUNDARY_CHARS` character, so this names a
    different, user-owned directory that only happens to end in the managed
    path's tail. Only checking the trailing boundary (the pre-fix behavior)
    would misclassify this as Headroom's and delete the user's script.
    """
    bin_dir = paths.bin_dir()
    lookalike = f"/prefix{bin_dir}/lean-ctx"
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {lookalike} "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert script.exists()
    payload = json.loads(settings.read_text())
    assert payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == str(script)


def test_leaves_an_mcp_entry_whose_command_has_the_bin_dir_as_a_path_segment_alone(home):
    """Same leading-boundary hazard as above, for the MCP-entry command check."""
    bin_dir = paths.bin_dir()
    lookalike_command = f"/prefix{bin_dir}/lean-ctx"
    config = _write(
        home / ".claude.json",
        json.dumps({"mcpServers": {"lean-ctx": {"command": lookalike_command, "args": ["mcp"]}}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    payload = json.loads(config.read_text())
    assert payload["mcpServers"] == {"lean-ctx": {"command": lookalike_command, "args": ["mcp"]}}


def test_leaves_a_hook_entry_whose_command_has_the_bin_dir_as_a_path_segment_alone(home):
    """Same hazard for a hook entry whose ``command`` names the managed
    directory directly (no script indirection). The command still carries a
    ``_HOOK_COMMAND_MARKERS`` token (``lean-ctx hook``) so it reaches the
    provenance guard rather than being filtered out earlier.
    """
    bin_dir = paths.bin_dir()
    lookalike_command = f"/prefix{bin_dir}/lean-ctx hook rewrite"
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [{"hooks": [{"type": "command", "command": lookalike_command}]}]
                }
            }
        ),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    payload = json.loads(settings.read_text())
    assert payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == lookalike_command


def test_removes_a_hook_script_whose_body_has_a_lookalike_prefix_before_a_genuine_managed_path(
    home,
):
    """A body can contain both a rejected lookalike occurrence and a later
    genuine, boundary-correct occurrence of the managed path. Rejecting the
    first must not short-circuit the scan (``continue``, not
    ``return False``) or the genuine occurrence right after it would never
    be seen.
    """
    bin_dir = paths.bin_dir()
    lookalike = f"/prefix{bin_dir}/lean-ctx-fake"
    genuine = str(bin_dir / "lean-ctx")
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {lookalike} --check\nexec {genuine} "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert not script.exists()
    payload = json.loads(settings.read_text())
    assert "hooks" not in payload


def test_removes_a_hook_script_whose_body_names_the_managed_dir_only_via_quoting_or_delimiters(
    home,
):
    """Real shell scripts quote paths and join them with `=`/`:`/`()`, not bare
    whitespace. The guard must not miss the managed directory just because it
    sits inside a quoted string, a `VAR=` assignment, a `PATH=...:` join, or a
    subshell — a naive whitespace-token split would sever every one of these
    (and the quote/paren characters would still be glued onto the token).
    """
    bin_dir = paths.bin_dir()
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        "#!/bin/sh\n"
        f'exec "{bin_dir}/lean-ctx" "$@"\n'
        f"# or: exec '{bin_dir}/lean-ctx'\n"
        f'export PATH="{bin_dir}:$PATH"\n'
        f"BIN={bin_dir}/lean-ctx\n"
        f"({bin_dir}/lean-ctx)\n",
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert not script.exists()
    payload = json.loads(settings.read_text())
    assert "hooks" not in payload


def test_matches_a_managed_path_when_home_contains_a_space(home, monkeypatch):
    """A ``$HOME`` with a space is real, and yields a bin dir with a literal
    space inside it. Splitting a script's body on whitespace before searching
    would sever the path at that space and miss it entirely.
    """
    bin_dir = home / "space here" / ".headroom" / "bin"
    monkeypatch.setattr(paths, "bin_dir", lambda: bin_dir)
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {bin_dir}/lean-ctx "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert not script.exists()
    payload = json.loads(settings.read_text())
    assert "hooks" not in payload


def test_leaves_a_same_named_hook_script_in_a_different_directory_alone(home):
    """A hook entry's command must name the exact managed script in
    ``~/.claude/hooks`` — not merely share a basename with one. A user's own
    ``~/mytools/lean-ctx-rewrite.sh`` must never inherit the classification of
    Headroom's ``~/.claude/hooks/lean-ctx-rewrite.sh`` just because the
    filename matches — but a wrapper invocation (``bash <script>``), a quoted
    command, or a redundant ``./`` segment naming the *managed* script by
    absolute path must still be recognised, or the entry survives while step
    2 deletes the very script it names (the same class of stale, silently
    no-op hook the rtk case documents as an accepted limitation, not one to
    introduce here).

    A *relative* command (``.claude/hooks/lean-ctx-rewrite.sh``) must survive
    even though it shares wording with the managed script's home-relative
    form: a relative hook command is resolved by the harness against the
    project's cwd, never against home, so it names a project-local script
    this purge never inspects — treating it as a home-relative reference
    would delete a different file than the one the guard just proved nothing
    about.
    """
    bin_dir = paths.bin_dir()
    hooks_dir = home / ".claude" / "hooks"
    # The managed script that gives "lean-ctx-rewrite.sh" a True verdict.
    managed_script = _write(
        hooks_dir / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {bin_dir / "lean-ctx"} "$@"\n',
    )
    # The user's own script: same basename, different directory, own binary.
    user_script = _write(
        home / "mytools" / "lean-ctx-rewrite.sh",
        '#!/bin/sh\nexec /usr/bin/lean-ctx "$@"\n',
    )
    relative_command = ".claude/hooks/lean-ctx-rewrite.sh"
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {"hooks": [{"command": f"bash {managed_script}"}]},
                        {"hooks": [{"command": f'"{managed_script}"'}]},
                        {"hooks": [{"command": f"{hooks_dir}/./lean-ctx-rewrite.sh"}]},
                        {"hooks": [{"command": relative_command}]},
                        {"hooks": [{"command": str(user_script)}]},
                    ]
                }
            }
        ),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    assert user_script.exists()
    payload = json.loads(settings.read_text())
    commands = [
        item["command"] for entry in payload["hooks"]["PreToolUse"] for item in entry["hooks"]
    ]
    assert commands == [relative_command, str(user_script)]


def test_reports_an_unreadable_hook_script_instead_of_guessing(home):
    if sys.platform.startswith("win") or os.geteuid() == 0:
        pytest.skip("chmod 0o000 does not deny access on Windows or when running as root")

    bin_dir = paths.bin_dir()
    script = _write(
        home / ".claude" / "hooks" / "lean-ctx-rewrite.sh",
        f'#!/bin/sh\nexec {bin_dir / "lean-ctx"} "$@"\n',
    )
    settings = _write(
        home / ".claude" / "settings.json",
        json.dumps({"hooks": {"PreToolUse": [{"hooks": [{"command": str(script)}]}]}}),
    )
    script.chmod(0o000)

    try:
        report = context_tool_cleanup.purge_context_tool_artifacts()
    finally:
        if script.exists():
            script.chmod(0o644)

    # Unprovable, so kept — not deleted on a guess — but named in the report.
    assert script.exists()
    assert any(str(script) in line for line in report)
    payload = json.loads(settings.read_text())
    assert payload["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == str(script)
    # A run that could not decide is not the completed migration: leaving the
    # stamp off is what gets this script looked at again once it is readable.
    assert not (paths.bin_dir().parent / ".context-tools-purged").exists()


def test_an_unparseable_config_defers_the_migration_stamp(home):
    """A config the user must fix by hand still holds a leftover.

    Stamping the migration complete here would retire the only reminder they
    get, and the entry would never be cleaned once the typo is fixed.
    """
    _write(home / ".claude" / "settings.json", "{not json")

    report = context_tool_cleanup.purge_context_tool_artifacts()

    assert any(line.startswith("skipped ") for line in report)
    assert not (paths.bin_dir().parent / ".context-tools-purged").exists()


def test_leaves_an_mcp_entry_without_a_command_alone(home):
    config = _write(
        home / ".claude.json",
        json.dumps(
            {
                "mcpServers": {
                    "lean-ctx": {"args": ["mcp"]},
                    "rtk": "not-a-dict",
                    "lean_ctx": {"command": ["lean-ctx", "mcp"]},
                }
            }
        ),
    )

    context_tool_cleanup.purge_context_tool_artifacts()

    payload = json.loads(config.read_text())
    assert set(payload["mcpServers"]) == {"lean-ctx", "rtk", "lean_ctx"}
