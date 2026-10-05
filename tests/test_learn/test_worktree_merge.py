"""A linked git worktree's sessions belong to its repo's learn project.

Claude Code files each working directory's sessions under its own
``~/.claude/projects`` folder, so a worktree (a Conductor workspace,
``.claude/worktrees/*``) used to surface as its own project.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from headroom.learn.plugins.claude import ClaudeCodePlugin, _main_worktree_root
from headroom.memory.traffic_learner import (
    ExtractedPattern,
    PatternCategory,
    _project_for_pattern,
)


def _repo_with_worktree(base: Path) -> tuple[Path, Path]:
    main, worktree = base / "repo", base / "workspaces" / "san-salvador"
    gitdir = main / ".git" / "worktrees" / "san-salvador"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n")
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n")
    return main, worktree


def _folder(claude_dir: Path, cwd: Path) -> Path:
    return claude_dir / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def _write_session(claude_dir: Path, cwd: Path, name: str) -> None:
    call = {"type": "tool_use", "id": "u1", "name": "Bash", "input": {"command": "ls"}}
    result = {"type": "tool_result", "tool_use_id": "u1", "content": "ok"}
    lines = [
        {"type": "assistant", "cwd": str(cwd), "message": {"content": [call]}},
        {"type": "user", "cwd": str(cwd), "message": {"content": [result]}},
    ]
    folder = _folder(claude_dir, cwd)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def test_main_worktree_root_resolves_linked_worktrees_only(tmp_path: Path) -> None:
    base = tmp_path.resolve()
    main, worktree = _repo_with_worktree(base)
    (worktree / "src").mkdir()
    submodule, sub_gitdir = main / "vendor" / "lib", main / ".git" / "modules" / "lib"
    submodule.mkdir(parents=True)
    sub_gitdir.mkdir(parents=True)
    (submodule / ".git").write_text("gitdir: ../../.git/modules/lib\n")

    assert _main_worktree_root(worktree) == main
    assert _main_worktree_root(worktree / "src") == main
    assert _main_worktree_root(main) is None
    assert _main_worktree_root(submodule) is None
    assert _main_worktree_root(base) is None


def test_relative_path_never_resolves_through_the_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    _, worktree = _repo_with_worktree(tmp_path.resolve())
    monkeypatch.chdir(worktree)

    assert _main_worktree_root(Path("C:\\Users\\dev\\work")) is None
    assert _main_worktree_root(Path("src")) is None


def test_worktree_sessions_merge_into_the_main_checkout(tmp_path: Path) -> None:
    base = tmp_path.resolve()
    main, worktree = _repo_with_worktree(base)
    claude_dir = base / "claude"
    _write_session(claude_dir, main, "a")
    _write_session(claude_dir, worktree, "b")
    plugin = ClaudeCodePlugin(claude_dir=claude_dir)

    [project] = plugin.discover_projects()

    assert project.project_path == main
    assert project.data_path == _folder(claude_dir, main)
    assert project.worktree_paths == [worktree]
    assert project.extra_data_paths == [_folder(claude_dir, worktree)]
    assert len(plugin.scan_project(project)) == 2


def test_worktree_only_repo_gets_the_main_checkouts_memory_folder(tmp_path: Path) -> None:
    # Conductor shape: every session runs in a workspace, none in the repo.
    base = tmp_path.resolve()
    main, worktree = _repo_with_worktree(base)
    claude_dir = base / "claude"
    _write_session(claude_dir, worktree, "a")
    plugin = ClaudeCodePlugin(claude_dir=claude_dir)

    [project] = plugin.discover_projects()

    assert project.project_path == main
    assert project.name == "repo"
    assert project.data_path == _folder(claude_dir, main)
    assert project.extra_data_paths == [_folder(claude_dir, worktree)]
    assert len(plugin.scan_project(project)) == 1


def test_projects_outside_worktrees_are_untouched(tmp_path: Path) -> None:
    base = tmp_path.resolve()
    one, two = base / "one", base / "two"
    one.mkdir()
    two.mkdir()
    claude_dir = base / "claude"
    _write_session(claude_dir, one, "a")
    _write_session(claude_dir, two, "b")

    projects = ClaudeCodePlugin(claude_dir=claude_dir).discover_projects()

    assert sorted(p.project_path for p in projects) == [one, two]
    assert all(not p.worktree_paths and not p.extra_data_paths for p in projects)


def test_traffic_pattern_in_a_worktree_routes_to_its_repo(tmp_path: Path) -> None:
    base = tmp_path.resolve()
    main, worktree = _repo_with_worktree(base)
    claude_dir = base / "claude"
    _write_session(claude_dir, main, "a")
    _write_session(claude_dir, worktree, "b")
    [project] = ClaudeCodePlugin(claude_dir=claude_dir).discover_projects()
    pattern = ExtractedPattern(
        category=PatternCategory.PREFERENCE,
        content=f"Edit `{worktree}/src/app.py` rather than the generated copy.",
        importance=0.5,
    )

    assert _project_for_pattern(pattern, [project]) is project


def test_a_session_started_in_a_worktree_subfolder_records_the_checkout(tmp_path: Path) -> None:
    """Selecting the worktree itself must find a project whose only session
    started in one of its subfolders."""
    base = tmp_path.resolve()
    main, worktree = _repo_with_worktree(base)
    (worktree / "src").mkdir()
    claude_dir = base / "claude"
    _write_session(claude_dir, worktree / "src", "a")

    [project] = ClaudeCodePlugin(claude_dir=claude_dir).discover_projects()

    assert project.project_path == main
    assert project.worktree_paths == [worktree / "src", worktree]
