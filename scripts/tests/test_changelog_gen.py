"""Tests for changelog-gen.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent

# Load changelog_gen module from scripts directory (filename has hyphen)
_spec = importlib.util.spec_from_file_location(
    "changelog_gen", ROOT / "scripts" / "changelog-gen.py"
)
if _spec is None:
    raise ImportError("Could not load changelog_gen module")
_changelog_gen = importlib.util.module_from_spec(_spec)
if _spec.loader is None:
    raise ImportError("Could not load changelog_gen module")
_spec.loader.exec_module(_changelog_gen)

COMMIT_PATTERN = _changelog_gen.COMMIT_PATTERN
BREAKING_CHANGE_PATTERN = _changelog_gen.BREAKING_CHANGE_PATTERN
FIELD_SEP = _changelog_gen.FIELD_SEP
RECORD_SEP = _changelog_gen.RECORD_SEP
ParsedCommit = _changelog_gen.ParsedCommit
generate_changelog = _changelog_gen.generate_changelog
iter_commit_entries = _changelog_gen.iter_commit_entries
parse_commits = _changelog_gen.parse_commits


def make_log_entry(subject: str, commit_hash: str, body: str = "") -> str:
    return f"{subject}{FIELD_SEP}{body}{FIELD_SEP}{commit_hash}{RECORD_SEP}"


class TestParseCommits:
    """Tests for parse_commits function."""

    def test_parses_feat_commit(self) -> None:
        log_output = make_log_entry("feat(core): add feature", "abc1234")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "feat"
        assert commits[0].scope == "core"
        assert commits[0].message == "add feature"
        assert commits[0].hash == "abc1234"
        assert commits[0].breaking is False

    def test_parses_fix_commit(self) -> None:
        log_output = make_log_entry("fix(ui): fix bug", "def5678")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "fix"
        assert commits[0].scope == "ui"
        assert commits[0].message == "fix bug"
        assert commits[0].hash == "def5678"

    def test_parses_ci_commit(self) -> None:
        log_output = make_log_entry("ci: update github actions", "xyz789")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "ci"
        assert commits[0].scope is None
        assert commits[0].message == "update github actions"
        assert commits[0].hash == "xyz789"

    def test_parses_chore_commit(self) -> None:
        log_output = make_log_entry("chore: cleanup", "xyz999")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "chore"
        assert commits[0].scope is None

    def test_parses_perf_commit(self) -> None:
        log_output = make_log_entry("perf(dashboard): improve performance", "abc111")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "perf"
        assert commits[0].scope == "dashboard"

    def test_parses_refactor_commit(self) -> None:
        log_output = make_log_entry("refactor(api): refactor endpoint", "abc222")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "refactor"
        assert commits[0].scope == "api"

    def test_parses_docs_commit(self) -> None:
        log_output = make_log_entry("docs: update readme", "abc333")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "docs"

    def test_parses_style_commit(self) -> None:
        log_output = make_log_entry("style: format code", "abc444")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "style"

    def test_parses_test_commit(self) -> None:
        log_output = make_log_entry("test: add tests for feature", "abc555")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "test"

    def test_detects_breaking_change_in_body(self) -> None:
        log_output = make_log_entry(
            "feat(core): add feature",
            "abc666",
            "BREAKING CHANGE: api changed",
        )
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].breaking is True

    def test_detects_breaking_change_exclamation(self) -> None:
        log_output = make_log_entry("feat(core)!: api changed", "abc777")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].breaking is True

    def test_no_scope_no_problem(self) -> None:
        log_output = make_log_entry("feat: simple feature", "abc888")
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].scope is None
        assert commits[0].message == "simple feature"

    def test_uses_pr_title_from_merge_commit_body(self) -> None:
        log_output = make_log_entry(
            "Merge pull request #173 from JerrettDavis/fix/pipeline-permissions-and-docs",
            "73f6673",
            "fix: repair release and docs pipelines",
        )
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "fix"
        assert commits[0].message == "repair release and docs pipelines"

    def test_falls_back_to_other_changes_for_non_conventional_merge(self) -> None:
        log_output = make_log_entry(
            "Merge pull request #186 from skorokithakis/patch-1",
            "1e80ee3",
            "Add support for custom Anthropic API URL",
        )
        commits = parse_commits(log_output)
        assert len(commits) == 1
        assert commits[0].type == "other"
        assert commits[0].message == "Add support for custom Anthropic API URL"

    def test_iter_commit_entries_parses_real_git_log_delimiters(self) -> None:
        log_output = make_log_entry(
            "fix: patch release flow", "abc1234", "BREAKING CHANGE: no"
        ) + make_log_entry("docs: update readme", "def5678")
        assert iter_commit_entries(log_output) == [
            ("fix: patch release flow", "BREAKING CHANGE: no", "abc1234"),
            ("docs: update readme", "", "def5678"),
        ]

    def test_iter_commit_entries_keeps_field_separator_inside_body(self) -> None:
        body = f"line one{FIELD_SEP}line two"
        log_output = make_log_entry("fix: patch release flow", "abc1234", body)
        assert iter_commit_entries(log_output) == [
            ("fix: patch release flow", body, "abc1234"),
        ]


class TestGenerateChangelog:
    """Tests for generate_changelog function."""

    def test_generates_version_header(self) -> None:
        commits = [
            ParsedCommit(type="feat", scope=None, breaking=False, message="test", hash="abc123")
        ]
        result = generate_changelog("0.6.0", commits)
        assert "## [0.6.0]" in result

    def test_includes_date(self) -> None:
        commits = []
        result = generate_changelog("0.6.0", commits)
        import re

        date_match = re.search(r"\d{4}-\d{2}-\d{2}", result)
        assert date_match is not None

    def test_groups_by_type(self) -> None:
        commits = [
            ParsedCommit(
                type="feat", scope=None, breaking=False, message="add feature", hash="abc123"
            ),
            ParsedCommit(type="fix", scope=None, breaking=False, message="fix bug", hash="def456"),
        ]
        result = generate_changelog("0.6.0", commits)
        assert "### Features" in result
        assert "### Bug Fixes" in result
        assert "- add feature (abc123)" in result
        assert "- fix bug (def456)" in result

    def test_includes_scope_in_bullet(self) -> None:
        commits = [
            ParsedCommit(
                type="feat", scope="core", breaking=False, message="add feature", hash="abc123"
            )
        ]
        result = generate_changelog("0.6.0", commits)
        assert "- **core**: add feature (abc123)" in result

    def test_breaking_change_section_when_present(self) -> None:
        commits = [
            ParsedCommit(
                type="feat", scope="core", breaking=True, message="api changed", hash="abc123"
            )
        ]
        result = generate_changelog("0.6.0", commits)
        assert "### Breaking Changes" in result
        assert "**core**" in result

    def test_no_breaking_change_section_when_none(self) -> None:
        commits = [
            ParsedCommit(
                type="feat", scope=None, breaking=False, message="add feature", hash="abc123"
            )
        ]
        result = generate_changelog("0.6.0", commits)
        assert "Breaking Changes" not in result

    def test_includes_other_changes_section(self) -> None:
        commits = [
            ParsedCommit(
                type="other",
                scope=None,
                breaking=False,
                message="Add support for custom Anthropic API URL",
                hash="abc123",
            )
        ]
        result = generate_changelog("0.6.0", commits)
        assert "### Other Changes" in result
        assert "- Add support for custom Anthropic API URL (abc123)" in result


class TestIntegrationWithMock:
    """Integration tests with mocked subprocess.run."""

    def test_full_flow_with_mocked_git(self) -> None:
        log_output = (
            make_log_entry("feat(core): add new feature", "abc1234")
            + make_log_entry("fix(ui): fix bug", "def5678")
            + make_log_entry("ci: update github actions", "xyz789")
            + make_log_entry(
                "feat(outer): breaking change", "bbb111", "BREAKING CHANGE: this is breaking"
            )
            + make_log_entry("chore: cleanup", "yyy999")
        )
        commits = parse_commits(log_output)

        assert len(commits) == 5
        assert any(c.type == "feat" and c.scope == "core" for c in commits)
        assert any(c.type == "fix" and c.scope == "ui" for c in commits)
        assert any(c.type == "ci" for c in commits)
        assert any(c.type == "feat" and c.breaking for c in commits)
        assert any(c.type == "chore" for c in commits)

        changelog = generate_changelog("0.7.0", commits)
        assert "## [0.7.0]" in changelog
        assert "### Features" in changelog
        assert "### Bug Fixes" in changelog
        assert "### CI/CD" in changelog
        assert "### Breaking Changes" in changelog


class TestCommitPattern:
    """Tests for the commit regex pattern."""

    def test_feat_with_scope(self) -> None:
        match = COMMIT_PATTERN.match("feat(core): add feature")
        assert match is not None
        assert match.group(1) == "feat"
        assert match.group(2) == "(core)"
        assert match.group(4) == "add feature"

    def test_fix_without_scope(self) -> None:
        match = COMMIT_PATTERN.match("fix: fix bug")
        assert match is not None
        assert match.group(1) == "fix"
        assert match.group(2) is None
        assert match.group(4) == "fix bug"

    def test_with_exclamation(self) -> None:
        match = COMMIT_PATTERN.match("feat(core)!: api changed")
        assert match is not None
        assert match.group(3) == "!"

    def test_without_exclamation(self) -> None:
        match = COMMIT_PATTERN.match("feat(core): add feature")
        assert match is not None
        assert match.group(3) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
