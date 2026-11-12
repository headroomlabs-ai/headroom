"""Git-repo project keys for --memory-project-key=git (GH #4712).

The fails-on-main repro: with project storage (the default) two sessions
in one repository got different memory DBs whenever they started in
different directories — the main checkout vs a linked worktree, or the
repo root vs a subdirectory. Under the ``git`` key strategy both resolve
to one shared key derived from the repository's common git dir, while
``cwd`` (the default) keeps today's per-directory keys untouched.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from headroom.memory.storage_router import (
    ProjectKeyStrategy,
    ProjectResolver,
    RequestContext,
    _git_common_dir,
)
from tests.test_memory_storage_router import _ctx


def _git(tmp_path: Path, *argv: str, cwd: Path | None = None) -> None:
    # CI runners have no global git identity; commit would fail without these.
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    subprocess.run(
        ["git", *argv],
        cwd=str(cwd or tmp_path),
        check=True,
        capture_output=True,
        env=env,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> dict[str, Path]:
    """A real repo with a linked worktree, a nested repo, and a subdir."""
    main = tmp_path / "acme"
    main.mkdir()
    _git(main, "init", "-q", ".")
    (main / "README.md").write_text("x", encoding="utf-8")
    _git(main, "add", ".")
    _git(main, "commit", "-qm", "init")
    # Make the default branch name deterministic across git configurations.
    _git(main, "branch", "-M", "main")
    (main / "src").mkdir()
    (main / "src" / "app.py").write_text("print('hi')", encoding="utf-8")

    worktree = tmp_path / "acme-feature"
    _git(main, "worktree", "add", "-q", str(worktree), "-b", "feature")

    nested = main / "vendor-tool"
    nested.mkdir()
    _git(nested, "init", "-q", ".")

    _git_common_dir.cache_clear()
    return {"main": main, "worktree": worktree, "sub": main / "src", "nested": nested}


GIT_STRATEGY = ProjectKeyStrategy(mode="git")
CWD_STRATEGY = ProjectKeyStrategy(mode="cwd")


class TestGitKeyStrategy:
    def test_repo_root_worktree_and_subdir_share_one_key(self, repo: dict[str, Path]) -> None:
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        keys = set()
        for label in ("main", "worktree", "sub"):
            resolved = r.resolve(
                _ctx(headers={"x-headroom-cwd": str(repo[label])})  # type: ignore[misc]
            )
            assert resolved is not None
            keys.add(resolved[0])
        assert len(keys) == 1, "one repository must map to exactly one memory project"

    def test_git_strategy_default_is_cwd_unchanged(self, repo: dict[str, Path]) -> None:
        # The default resolver (no strategy) keeps today's per-directory keys:
        # this is the behaviour that fails on main and must stay for `cwd`.
        r_default = ProjectResolver()
        k_main = r_default.resolve(_ctx(headers={"x-headroom-cwd": str(repo["main"])}))  # type: ignore[misc]
        k_wt = r_default.resolve(_ctx(headers={"x-headroom-cwd": str(repo["worktree"])}))  # type: ignore[misc]
        assert k_main is not None and k_wt is not None
        assert k_main[0] != k_wt[0], "cwd strategy must keep worktrees distinct"

    def test_explicit_cwd_strategy_keeps_worktrees_distinct(self, repo: dict[str, Path]) -> None:
        r = ProjectResolver(key_strategy=CWD_STRATEGY)
        k_main = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["main"])}))  # type: ignore[misc]
        k_wt = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["worktree"])}))  # type: ignore[misc]
        assert k_main is not None and k_wt is not None
        assert k_main[0] != k_wt[0]

    def test_distinct_repos_keep_distinct_keys(self, repo: dict[str, Path], tmp_path: Path) -> None:
        other = tmp_path / "other-repo"
        other.mkdir()
        _git(other, "init", "-q", ".")
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        k1 = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["main"])}))  # type: ignore[misc]
        k2 = r.resolve(_ctx(headers={"x-headroom-cwd": str(other)}))  # type: ignore[misc]
        assert k1 is not None and k2 is not None
        assert k1[0] != k2[0], "repo isolation from #462 must survive the git strategy"

    def test_nested_repo_gets_own_key(self, repo: dict[str, Path]) -> None:
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        k_outer = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["main"])}))  # type: ignore[misc]
        k_nested = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["nested"])}))  # type: ignore[misc]
        assert k_outer is not None and k_nested is not None
        assert k_outer[0] != k_nested[0], "nested repos key themselves, as git itself does"

    def test_non_git_directory_falls_back_to_cwd_key(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain-dir"
        plain.mkdir()
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        via_git = r.resolve(_ctx(headers={"x-headroom-cwd": str(plain)}))  # type: ignore[misc]
        r_cwd = ProjectResolver(key_strategy=CWD_STRATEGY)
        plain_cwd = r_cwd.resolve(_ctx(headers={"x-headroom-cwd": str(plain)}))  # type: ignore[misc]
        assert via_git is not None and plain_cwd is not None
        assert via_git[0] == plain_cwd[0], "non-git dirs keep the cwd key"

    def test_nonexistent_remote_path_falls_back_to_cwd_key(self) -> None:
        remote = "/mnt/remote-machine/projects/acme"
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        via_git = r.resolve(_ctx(headers={"x-headroom-cwd": remote}))  # type: ignore[misc]
        r_cwd = ProjectResolver(key_strategy=CWD_STRATEGY)
        plain_cwd = r_cwd.resolve(_ctx(headers={"x-headroom-cwd": remote}))  # type: ignore[misc]
        assert via_git is not None and plain_cwd is not None
        assert via_git[0] == plain_cwd[0], "proxy-host-absent paths keep the cwd key"

    def test_display_name_is_repo_basename_for_worktree(self, repo: dict[str, Path]) -> None:
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        resolved = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["worktree"])}))  # type: ignore[misc]
        assert resolved is not None
        assert resolved[1] == "acme", "worktree should display the readable repo name"

    def test_tier_precedence_unchanged_explicit_id_wins(self, repo: dict[str, Path]) -> None:
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        resolved = r.resolve(
            _ctx(
                headers={
                    "x-headroom-project-id": "billing-svc",
                    "x-headroom-cwd": str(repo["main"]),
                }
            )
        )
        assert resolved is not None
        assert resolved[0].startswith("billing-svc-")
        assert resolved[1] == "billing-svc"

    def test_context_key_strategy_beats_resolver_default(self, repo: dict[str, Path]) -> None:
        # Handler call sites set the strategy on the RequestContext, not the
        # resolver; the per-request value must win over the constructor's.
        r_cwd_default = ProjectResolver()  # cwd default
        resolved = r_cwd_default.resolve(
            RequestContext(
                headers={"x-headroom-cwd": str(repo["worktree"])},
                system_prompt="",
                base_user_id="u",
                key_strategy=GIT_STRATEGY,
            )
        )
        assert resolved is not None
        r_git = ProjectResolver(key_strategy=GIT_STRATEGY)
        expected = r_git.resolve(_ctx(headers={"x-headroom-cwd": str(repo["worktree"])}))  # type: ignore[misc]
        assert expected is not None
        assert resolved[0] == expected[0]

    def test_strategy_cache_reuses_git_lookup(self, repo: dict[str, Path]) -> None:
        _git_common_dir.cache_clear()
        ProjectResolver(key_strategy=GIT_STRATEGY).resolve(
            _ctx(headers={"x-headroom-cwd": str(repo["main"])})  # type: ignore[misc]
        )
        info = _git_common_dir.cache_info()
        assert info.hits >= 0  # cache is active
        # Second resolution must hit the cache rather than re-walking the FS.
        ProjectResolver(key_strategy=GIT_STRATEGY).resolve(
            _ctx(headers={"x-headroom-cwd": str(repo["main"])})  # type: ignore[misc]
        )
        assert _git_common_dir.cache_info().hits > info.hits

    def test_invalid_strategy_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            ProjectKeyStrategy(mode="bogus")

    def test_system_prompt_tier_uses_strategy(self, repo: dict[str, Path]) -> None:
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        prompt = f"You have been invoked in the following environment:\n - Primary working directory: {repo['sub']}\n"
        via_prompt = r.resolve(_ctx(system_prompt=prompt))
        via_header = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["main"])}))  # type: ignore[misc]
        assert via_prompt is not None and via_header is not None
        assert via_prompt[0] == via_header[0], "tier-4 cwd must key like tier-2 under git"

    def test_subdir_reported_as_system_prompt_matches_header_from_root(
        self, repo: dict[str, Path]
    ) -> None:
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        prompt = f"Working directory: {repo['sub']}\n"
        via_prompt = r.resolve(_ctx(system_prompt=prompt))
        via_header = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["main"])}))  # type: ignore[misc]
        assert via_prompt is not None and via_header is not None
        assert via_prompt[0] == via_header[0]

    def test_project_root_override_tier_uses_strategy(self, repo: dict[str, Path]) -> None:
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        via_override = r.resolve(_ctx(project_root_override=str(repo["worktree"])))
        via_header = r.resolve(_ctx(headers={"x-headroom-cwd": str(repo["main"])}))  # type: ignore[misc]
        assert via_override is not None and via_header is not None
        assert via_override[0] == via_header[0], "CLI override must key like a header cwd under git"


class TestGitCommonDirHelper:
    def test_main_checkout_common_dir(self, repo: dict[str, Path]) -> None:
        resolved = str((repo["main"] / "src").resolve())
        common = _git_common_dir(resolved)
        assert common is not None
        assert Path(common).name == ".git"
        assert Path(common).parent == repo["main"].resolve()

    def test_no_git_returns_none(self, tmp_path: Path) -> None:
        plain = tmp_path / "no-repo"
        plain.mkdir()
        assert _git_common_dir(str(plain)) is None

    def test_traversal_stops_at_filesystem_root(self, tmp_path: Path) -> None:
        deep = tmp_path / "no-repo" / "a" / "b" / "c"
        deep.mkdir(parents=True)
        assert _git_common_dir(str(deep)) is None

    def test_git_file_without_gitdir_line_returns_none(self, tmp_path: Path) -> None:
        # A malformed ``.git`` file (no ``gitdir:`` line) must fail closed
        # to the cwd key, not raise or guess.
        work = tmp_path / "broken-gitfile"
        work.mkdir()
        (work / ".git").write_text("garbage\n", encoding="utf-8")
        assert _git_common_dir(str(work)) is None

    def test_git_file_empty_gitdir_target_returns_none(self, tmp_path: Path) -> None:
        work = tmp_path / "empty-gitdir"
        work.mkdir()
        (work / ".git").write_text("gitdir:\n", encoding="utf-8")
        assert _git_common_dir(str(work)) is None

    def test_git_file_pointing_at_missing_dir_returns_none(self, tmp_path: Path) -> None:
        work = tmp_path / "dangling-gdir"
        work.mkdir()
        target = tmp_path / "does-not-exist"
        (work / ".git").write_text(f"gitdir: {target}\n", encoding="utf-8")
        assert _git_common_dir(str(work)) is None

    def test_worktree_git_file_resolves_through_commondir(self, repo: dict[str, Path]) -> None:
        # The linked worktree's ``.git`` *file* must resolve through the
        # gitdir's ``commondir`` back to the main checkout's common dir.
        main_common = _git_common_dir(str(repo["main"].resolve()))
        assert main_common is not None
        wt_common = _git_common_dir(str(repo["worktree"].resolve()))
        assert wt_common == main_common

    def test_gitdir_without_commondir_file_keys_on_gitdir(self, tmp_path: Path) -> None:
        # A ``.git`` file pointing at a gitdir that has no ``commondir``
        # (the shape a submodule checkout has) keys on that gitdir and
        # labels with the parent directory of the gitdir's target.
        gitdir = tmp_path / "sub-src" / ".git"
        gitdir.mkdir(parents=True)
        work = tmp_path / "sub-checkout"
        work.mkdir()
        (work / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        resolved = r.resolve(_ctx(headers={"x-headroom-cwd": str(work)}))  # type: ignore[misc]
        assert resolved is not None
        assert resolved[1] == "sub-src"

    def test_missing_file_entries_fail_closed(self, tmp_path: Path) -> None:
        # ``.git`` dir with an empty ``commondir`` file: fall through to
        # the directory itself (it IS the common dir in that case).
        work = tmp_path / "dirrepo"
        dot_git = work / ".git"
        dot_git.mkdir(parents=True)
        (dot_git / "commondir").write_text("  \n", encoding="utf-8")
        common = _git_common_dir(str(work))
        assert common is not None
        assert Path(common) == dot_git.resolve()

    def test_main_worktree_basename_fallbacks(self, tmp_path: Path) -> None:
        # ``modules/<name>`` labels as the module name; a bare-ish common dir
        # without a ``.git`` basename labels as its parent directory.
        from headroom.memory.storage_router import _main_worktree_basename

        assert _main_worktree_basename(str(tmp_path / "r" / ".git" / "modules" / "mlib")) == "mlib"
        assert _main_worktree_basename(str(tmp_path / "weird" / "odd")) == "weird"

    def test_nested_submodule_labels_on_its_own_name(self, tmp_path: Path) -> None:
        # Review feedback on #4795: a submodule at a nested work-tree path
        # (``vendor/lib``) has common dir ``.git/modules/vendor/lib``. The
        # label must be the submodule itself (``lib``), not ``vendor``.
        from headroom.memory.storage_router import _main_worktree_basename

        common = tmp_path / "r" / ".git" / "modules" / "vendor" / "lib"
        assert _main_worktree_basename(str(common)) == "lib"
        deep = tmp_path / "r" / ".git" / "modules" / "a" / "b" / "c"
        assert _main_worktree_basename(str(deep)) == "c"

    def test_nested_submodule_key_is_distinct_and_shared_with_checkout(
        self, tmp_path: Path
    ) -> None:
        # The key comes from the common dir, so a submodule checkout and its
        # nested gitdir address one project — and stay distinct from the
        # outer repo that hosts the ``.git/modules`` tree.
        outer = tmp_path / "r"
        (outer / ".git").mkdir(parents=True)
        nested_gitdir = outer / ".git" / "modules" / "vendor" / "lib"
        nested_gitdir.mkdir(parents=True)
        sub_worktree = outer / "vendor" / "lib"
        sub_worktree.mkdir(parents=True)
        (sub_worktree / ".git").write_text(f"gitdir: {nested_gitdir}\n", encoding="utf-8")

        r = ProjectResolver(key_strategy=GIT_STRATEGY)
        via_sub = r.resolve(_ctx(headers={"x-headroom-cwd": str(sub_worktree)}))  # type: ignore[misc]
        via_outer = r.resolve(_ctx(headers={"x-headroom-cwd": str(outer)}))  # type: ignore[misc]
        assert via_sub is not None and via_outer is not None
        assert via_sub[1] == "lib"
        assert via_sub[0] != via_outer[0], "a submodule keeps its own memory DB"
