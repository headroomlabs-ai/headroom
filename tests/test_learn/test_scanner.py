"""Tests for _decode_project_path and _greedy_path_decode (issue #47).

Directory names that contain dots (e.g. ``GitHub.nosync``) or multiple
hyphens (e.g. ``my-cool-project``) were silently dropped because
_greedy_path_decode only tried joining two consecutive tokens with a hyphen,
making it impossible to reconstruct names formed from three or more tokens.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest

from headroom.learn.scanner import ClaudeCodeScanner, _decode_project_path, _greedy_path_decode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dirs(base: Path, *rel_paths: str) -> None:
    """Create one or more relative directory paths under *base*."""
    for rel in rel_paths:
        (base / rel).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# _greedy_path_decode
# ---------------------------------------------------------------------------


class TestGreedyPathDecode:
    """Unit tests for _greedy_path_decode."""

    def test_simple_directory(self, tmp_path: Path) -> None:
        _make_dirs(tmp_path, "headroom")
        result = _greedy_path_decode(tmp_path, ["headroom"])
        assert result == tmp_path / "headroom"

    def test_single_hyphen_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains one literal hyphen."""
        _make_dirs(tmp_path, "my-project")
        result = _greedy_path_decode(tmp_path, ["my", "project"])
        assert result == tmp_path / "my-project"

    def test_multiple_hyphens_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains multiple literal hyphens (the regression case)."""
        _make_dirs(tmp_path, "my-cool-project")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project"])
        assert result == tmp_path / "my-cool-project"

    def test_dot_only_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains a dot but no hyphen (e.g. GitHub.nosync)."""
        _make_dirs(tmp_path, "GitHub.nosync")
        result = _greedy_path_decode(tmp_path, ["GitHub.nosync"])
        assert result == tmp_path / "GitHub.nosync"

    def test_dot_and_single_hyphen_in_dirname(self, tmp_path: Path) -> None:
        """Directory name has both a dot and a single hyphen (e.g. my-project.nosync)."""
        _make_dirs(tmp_path, "my-project.nosync")
        result = _greedy_path_decode(tmp_path, ["my", "project.nosync"])
        assert result == tmp_path / "my-project.nosync"

    def test_dot_and_multiple_hyphens_in_dirname(self, tmp_path: Path) -> None:
        """Directory name has a dot and multiple hyphens (e.g. my-cool-project.nosync).

        This was the primary regression: the old code only joined pairs, so it
        could never reconstruct a three-token hyphenated name.
        """
        _make_dirs(tmp_path, "my-cool-project.nosync")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project.nosync"])
        assert result == tmp_path / "my-cool-project.nosync"

    def test_dot_dir_containing_hyphenated_subdir(self, tmp_path: Path) -> None:
        """Path like GitHub.nosync/my-project — dot parent + hyphen child."""
        _make_dirs(tmp_path, "GitHub.nosync/my-project")
        result = _greedy_path_decode(tmp_path, ["GitHub.nosync", "my", "project"])
        assert result == tmp_path / "GitHub.nosync" / "my-project"

    def test_dot_dir_with_multi_hyphen_subdir(self, tmp_path: Path) -> None:
        """Path like GitHub.nosync/my-cool-app — dot parent + multi-hyphen child."""
        _make_dirs(tmp_path, "GitHub.nosync/my-cool-app")
        result = _greedy_path_decode(tmp_path, ["GitHub.nosync", "my", "cool", "app"])
        assert result == tmp_path / "GitHub.nosync" / "my-cool-app"

    def test_multi_hyphen_dot_dir_containing_subproject(self, tmp_path: Path) -> None:
        """Path like my-cool-project.nosync/headroom — hardest combination."""
        _make_dirs(tmp_path, "my-cool-project.nosync/headroom")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project.nosync", "headroom"])
        assert result == tmp_path / "my-cool-project.nosync" / "headroom"

    def test_dot_flattened_into_separate_tokens(self, tmp_path: Path) -> None:
        """Flattened encoding like GitHub-nosync should map back to GitHub.nosync."""
        _make_dirs(tmp_path, "GitHub.nosync/thebest")
        result = _greedy_path_decode(tmp_path, ["GitHub", "nosync", "thebest"])
        assert result == tmp_path / "GitHub.nosync" / "thebest"

    def test_hybrid_hyphen_and_dot_flattening(self, tmp_path: Path) -> None:
        """Flattened encoding should reconstruct mixed separators in one component."""
        _make_dirs(tmp_path, "my-cool-project.nosync/headroom")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project", "nosync", "headroom"])
        assert result == tmp_path / "my-cool-project.nosync" / "headroom"

    # ---- Space tests (issue #997) ----

    def test_single_space_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains a space (e.g. 'Claude Projects')."""
        _make_dirs(tmp_path, "Claude Projects")
        result = _greedy_path_decode(tmp_path, ["Claude", "Projects"])
        assert result == tmp_path / "Claude Projects"

    def test_multiple_spaces_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains multiple spaces (e.g. 'Claude Code Projects')."""
        _make_dirs(tmp_path, "Claude Code Projects")
        result = _greedy_path_decode(tmp_path, ["Claude", "Code", "Projects"])
        assert result == tmp_path / "Claude Code Projects"

    def test_space_nested_path(self, tmp_path: Path) -> None:
        """Nested path like Desktop/'Claude Code Projects' should decode correctly."""
        _make_dirs(tmp_path, "Desktop/Claude Code Projects")
        result = _greedy_path_decode(tmp_path, ["Desktop", "Claude", "Code", "Projects"])
        assert result == tmp_path / "Desktop" / "Claude Code Projects"

    # ---- Underscore tests (issue #159) ----

    def test_single_underscore_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains one literal underscore (e.g. my_project)."""
        _make_dirs(tmp_path, "my_project")
        result = _greedy_path_decode(tmp_path, ["my", "project"])
        assert result == tmp_path / "my_project"

    def test_multiple_underscores_in_dirname(self, tmp_path: Path) -> None:
        """Directory name contains multiple underscores (e.g. my_cool_project)."""
        _make_dirs(tmp_path, "my_cool_project")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project"])
        assert result == tmp_path / "my_cool_project"

    def test_underscore_nested_path(self, tmp_path: Path) -> None:
        """Nested path like org/my_project should decode correctly."""
        _make_dirs(tmp_path, "org/my_project")
        result = _greedy_path_decode(tmp_path, ["org", "my", "project"])
        assert result == tmp_path / "org" / "my_project"

    def test_mixed_underscore_and_hyphen_in_dirname(self, tmp_path: Path) -> None:
        """Directory with both hyphens and underscores (e.g. my-cool_project)."""
        _make_dirs(tmp_path, "my-cool_project")
        result = _greedy_path_decode(tmp_path, ["my", "cool", "project"])
        assert result == tmp_path / "my-cool_project"

    def test_underscore_dir_containing_hyphen_subdir(self, tmp_path: Path) -> None:
        """Path like my_app/sub-module — underscore parent + hyphen child."""
        _make_dirs(tmp_path, "my_app/sub-module")
        result = _greedy_path_decode(tmp_path, ["my", "app", "sub", "module"])
        assert result == tmp_path / "my_app" / "sub-module"

    def test_nonexistent_path_returns_none(self, tmp_path: Path) -> None:
        result = _greedy_path_decode(tmp_path, ["does", "not", "exist"])
        assert result is None

    def test_empty_parts_returns_base_when_exists(self, tmp_path: Path) -> None:
        result = _greedy_path_decode(tmp_path, [])
        assert result == tmp_path

    def test_empty_parts_returns_none_when_not_exists(self, tmp_path: Path) -> None:
        result = _greedy_path_decode(tmp_path / "missing", [])
        assert result is None


# ---------------------------------------------------------------------------
# _decode_project_path
# ---------------------------------------------------------------------------


class TestDecodeProjectPath:
    """Integration-level tests for _decode_project_path.

    Note: _decode_project_path's greedy branch only activates for paths whose
    first component is ``Users`` (the common macOS home prefix).  Tests that
    exercise the greedy decoder therefore synthesise an encoded name rooted at
    ``/Users/<username>/…`` inside a real temporary directory created under
    that prefix.  When the temp directory does not exist under ``/Users`` the
    tests fall back to ``/tmp`` and rely only on the fast simple-replace path.
    """

    def test_returns_none_for_non_absolute_encoded_name(self) -> None:
        assert _decode_project_path("Users-foo-bar") is None

    def test_simple_replace_finds_dot_path(self, users_tmp: Path) -> None:
        """Simple replace-all works when no dir names contain hyphens.

        The encoded name maps directly to the real path because every ``-`` is
        a path separator; dots in directory names are preserved unchanged.
        """
        project = users_tmp / "GitHub.nosync" / "headroom"
        project.mkdir(parents=True)
        # Build the encoded name exactly as Claude Code does (/  →  -)
        encoded = "-" + str(project)[1:].replace("/", "-")
        result = _decode_project_path(encoded)
        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            assert result is None or result == project

    # ------------------------------------------------------------------
    # Greedy-decoder tests — require a /Users-rooted path to activate.
    # We try to create a temp dir under the real /Users tree; if that is
    # not writable we skip rather than fail (CI typically runs as a real
    # macOS user whose home IS under /Users).
    # ------------------------------------------------------------------

    @pytest.fixture()
    def users_tmp(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Return a temporary directory whose path starts with /Users/…

        On macOS the system temp dir is under /private/var, so we create a
        disposable directory directly inside the real user's home instead.
        Falls back to tmp_path so tests still run on non-macOS platforms
        (where the greedy branch isn't reached but no crash occurs either).
        """

        home = Path.home()
        if str(home).startswith("/Users/"):
            base = home / ".pytest_headroom_tmp"
            try:
                base.mkdir(exist_ok=True)
            except PermissionError:
                pytest.skip("Cannot create /Users-rooted temp dir in this environment")
            # Use a sub-directory unique to this test invocation
            unique = base / uuid4().hex
            try:
                unique.mkdir()
            except PermissionError:
                pytest.skip("Cannot create /Users-rooted temp dir in this environment")
            yield unique
            import shutil

            shutil.rmtree(unique, ignore_errors=True)
        else:
            yield tmp_path

    def test_dot_and_hyphen_in_dirname_via_greedy(self, users_tmp: Path) -> None:
        """GitHub.nosync/my-project — dot parent + hyphenated child (issue #47).

        Simple replace-all gives ``…/GitHub.nosync/my/project`` which does not
        exist, so the greedy decoder must reconstruct ``my-project``.
        """
        project = users_tmp / "GitHub.nosync" / "my-project"
        project.mkdir(parents=True)

        encoded = "-" + str(project)[1:].replace("/", "-")
        result = _decode_project_path(encoded)

        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            # Greedy branch not reached outside /Users; just confirm no crash
            assert result is None or result == project

    def test_multi_hyphen_dot_dirname_via_greedy(self, users_tmp: Path) -> None:
        """my-cool-project.nosync/app — primary regression from issue #47.

        Three tokens joined by hyphens form the parent dir name; the old code
        only tried pairs and therefore could never reconstruct this component.
        """
        project = users_tmp / "my-cool-project.nosync" / "app"
        project.mkdir(parents=True)

        encoded = "-" + str(project)[1:].replace("/", "-")
        result = _decode_project_path(encoded)

        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            assert result is None or result == project

    def test_flattened_dot_dirname_via_greedy(self, users_tmp: Path) -> None:
        """GitHub.nosync/thebest should decode from GitHub-nosync-thebest."""
        project = users_tmp / "GitHub.nosync" / "thebest"
        project.mkdir(parents=True)

        encoded = "-" + str(project)[1:].replace("/", "-").replace(".", "-")
        result = _decode_project_path(encoded)

        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            assert result is None or result == project

    def test_underscore_dirname_via_greedy(self, users_tmp: Path) -> None:
        """my_project — underscore in directory name (issue #159).

        Claude Code encodes /Users/foo/org/my_project as
        -Users-foo-org-my-project.  Simple replace gives
        …/org/my/project which does not exist, so the greedy decoder
        must reconstruct my_project from tokens ['my', 'project'].
        """
        project = users_tmp / "org" / "my_project"
        project.mkdir(parents=True)

        encoded = "-" + str(project)[1:].replace("/", "-")
        result = _decode_project_path(encoded)

        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            assert result is None or result == project

    def test_multi_underscore_dirname_via_greedy(self, users_tmp: Path) -> None:
        """my_cool_project — multiple underscores (issue #159)."""
        project = users_tmp / "my_cool_project"
        project.mkdir(parents=True)

        encoded = "-" + str(project)[1:].replace("/", "-")
        result = _decode_project_path(encoded)

        if str(users_tmp).startswith("/Users/"):
            assert result == project
        else:
            assert result is None or result == project

    def test_windows_drive_letter_pattern(self) -> None:
        """Encoded name -C-MQ2-macros should detect Windows drive letter."""
        import sys

        result = _decode_project_path("-C-MQ2-macros")
        if sys.platform == "win32":
            # On Windows: tries C:\\MQ2\\macros, may or may not exist
            assert result is None or str(result).startswith("C:")
        else:
            # On Unix: drive detection runs but path doesn't exist → falls through
            # Then Unix paths tried → also don't exist → returns None
            assert result is None

    def test_windows_users_path(self) -> None:
        """Encoded name -C-Users-foo-project detects drive letter."""
        result = _decode_project_path("-C-Users-foo-project")
        assert result is not None
        assert str(result).startswith("C:")
        assert "Users" in str(result)

    def test_windows_username_with_dot_stays_single_component(self) -> None:
        """Windows profile names like john.doe must not decode as john/doe."""
        result = _decode_project_path("-C-Users-john.doe-work")

        assert result is not None
        rendered = str(result)
        assert rendered.startswith("C:")
        assert "john.doe" in rendered
        assert "john\\doe" not in rendered
        assert "john/doe" not in rendered

    def test_windows_path_with_spaces_decoded_via_greedy(self) -> None:
        """Spaces in Windows dir names must not split into separate components (#997).

        Claude Code encodes 'C:\\Users\\user\\Desktop\\Claude Code Projects' as
        '-C-Users-user-Desktop-Claude-Code-Projects'. The greedy decoder must
        reconstruct 'Claude Code Projects' as a single directory.
        """
        import sys
        import tempfile

        if sys.platform != "win32":
            pytest.skip("greedy Windows-path decode requires real Windows filesystem")

        with tempfile.TemporaryDirectory() as td:
            space_dir = Path(td) / "Claude Code Projects"
            space_dir.mkdir()

            drive = Path(td).drive[0]
            rest = str(Path(td))[3:]  # strip 'C:\\'
            rest_parts = rest.replace("\\", "-").replace(" ", "-")
            encoded = f"-{drive}-{rest_parts}-Claude-Code-Projects"

            result = _decode_project_path(encoded)
            assert result is not None
            assert result == space_dir

    def test_discover_windows_project_uses_leaf_name(self, tmp_path: Path) -> None:
        """A syntactic Windows path decoded on Unix should still display the project leaf."""
        claude_dir = tmp_path / ".claude"
        project_dir = claude_dir / "projects" / "-C-Users-john.doe-work"
        project_dir.mkdir(parents=True)
        (project_dir / "session.jsonl").write_text("{}\n")

        projects = ClaudeCodeScanner(claude_dir=claude_dir).discover_projects()

        assert len(projects) == 1
        assert projects[0].name == "work"
        assert str(projects[0].project_path).startswith("C:")

    def test_discover_project_prefers_session_cwd(self, tmp_path: Path, monkeypatch) -> None:
        """Session JSONL cwd should override ambiguous path decoding."""
        claude_dir = tmp_path / ".claude"
        project_dir = claude_dir / "projects" / "-C-work-vibe-remote"
        project_dir.mkdir(parents=True)

        actual = tmp_path / "work" / "vibe-remote"
        actual.mkdir(parents=True)
        (project_dir / "session.jsonl").write_text(
            json.dumps({"cwd": str(actual)}) + "\n", encoding="utf-8"
        )

        monkeypatch.setattr(
            "headroom.learn.plugins.claude._decode_project_path", lambda name: tmp_path / "wrong"
        )

        projects = ClaudeCodeScanner(claude_dir=claude_dir).discover_projects()

        assert len(projects) == 1
        assert projects[0].project_path == actual
        assert projects[0].name == "vibe-remote"

    def test_home_dir_username_stays_single_component(self) -> None:
        """A home-directory name must survive decoding as one component.

        Claude Code flattens ``/``, ``.``, ``-`` and ``_`` all to ``-`` when
        escaping, so a project under ``/Users/first.last`` (or
        ``/home/first.last``) is stored as ``-Users-first-last-…``. The decoder
        used to consume only the first token after ``Users``/``home`` as the
        home directory and walk from ``/Users/first`` (which does not exist), so
        it bailed out and callers fell back to the literal
        ``/Users/first/last`` — causing ``headroom learn --apply`` to fail with
        ``PermissionError: '/Users/first'`` for usernames such as
        ``first.last``. This is the Unix counterpart of
        ``test_windows_username_with_dot_stays_single_component``.

        Rooted at the real home so it exercises the ``Users``/``home`` branch on
        both macOS (``/Users/…``) and Linux (``/home/…``); skipped when the home
        directory is neither rooted there nor writable.
        """
        import shutil

        home = Path.home()
        if len(home.parts) < 3 or home.parts[1] not in ("Users", "home"):
            pytest.skip("decoder branch only activates under /Users or /home")

        base = home / f"pytest_headroom_{uuid4().hex}"
        try:
            base.mkdir()
        except (PermissionError, OSError):
            pytest.skip("home directory is not writable")

        try:
            project = base / "my.project"
            project.mkdir()

            # Flatten separators exactly as Claude Code does when escaping.
            encoded = "-" + str(project)[1:].replace("/", "-").replace(".", "-").replace("_", "-")
            result = _decode_project_path(encoded)
        finally:
            shutil.rmtree(base, ignore_errors=True)

        assert result == project
        # The home component is reconstructed whole, never split on a separator.
        assert home.name in result.parts
