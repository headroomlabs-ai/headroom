"""Tests for headroom.ignore — the central compress/learn/mutate/memory ignore policy.

Covers issue #1150: preventing Headroom from compressing, learning from,
indexing, or mutating generated agent-harness files (CLAUDE.md, AGENTS.md,
.github/copilot-instructions.md, .cursorrules, ANTIGRAVITY.md, ...) that are
projections of a canonical source of truth (e.g. cARL's .github/carl/).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.config import HeadroomConfig, IgnoreConfig
from headroom.ignore import IgnorePolicy, is_path_ignored


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestNoIgnoreConfig:
    """1. No ignore config means existing behavior is unchanged."""

    def test_no_headroomignore_no_config_nothing_ignored(self, tmp_path: Path) -> None:
        policy = IgnorePolicy.load(tmp_path)
        assert policy.active_rules() == []
        for behavior in ("compress", "learn", "mutate", "memory"):
            assert not policy.is_ignored("CLAUDE.md", behavior)
            assert not policy.is_ignored("src/main.py", behavior)

    def test_empty_headroomignore_file_ignores_nothing(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "\n# just comments\n\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.active_rules() == []
        assert not policy.is_ignored("CLAUDE.md", "mutate")


class TestHeadroomIgnoreFile:
    """2 & 3. .headroomignore exact-file and directory/glob matches."""

    def test_exact_file_match(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "CLAUDE.md\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored("CLAUDE.md", "mutate")
        assert policy.is_ignored("CLAUDE.md", "learn")
        assert policy.is_ignored("CLAUDE.md", "compress")
        assert policy.is_ignored("CLAUDE.md", "memory")
        assert not policy.is_ignored("README.md", "mutate")

    def test_directory_trailing_slash_match(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", ".github/carl/\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored(".github/carl/policy.md", "mutate")
        assert policy.is_ignored(".github/carl/nested/deep.md", "mutate")
        assert policy.is_ignored(".github/carl", "mutate")
        assert not policy.is_ignored(".github/other.md", "mutate")

    def test_directory_globstar_match(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", ".github/carl/**\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored(".github/carl/policy.md", "mutate")
        assert policy.is_ignored(".github/carl/nested/deep.md", "mutate")
        assert not policy.is_ignored(".github/other.md", "mutate")

    def test_comments_and_blank_lines_ignored(self, tmp_path: Path) -> None:
        _write(
            tmp_path / ".headroomignore",
            "# comment\n\nCLAUDE.md\n   \n# another\nAGENTS.md\n",
        )
        policy = IgnorePolicy.load(tmp_path)
        assert len(policy.active_rules()) == 2
        assert policy.is_ignored("CLAUDE.md", "mutate")
        assert policy.is_ignored("AGENTS.md", "mutate")


class TestBehaviorScopedRules:
    """4. Behavior-scoped ignore rules (config ignore.compress/learn/mutate/memory)."""

    def test_scoped_rule_only_applies_to_its_behavior(self, tmp_path: Path) -> None:
        config = IgnoreConfig(compress=[".github/carl/**"])
        policy = IgnorePolicy.load(tmp_path, config)
        assert policy.is_ignored(".github/carl/x.md", "compress")
        assert not policy.is_ignored(".github/carl/x.md", "mutate")
        assert not policy.is_ignored(".github/carl/x.md", "learn")
        assert not policy.is_ignored(".github/carl/x.md", "memory")

    def test_multiple_scopes_independent(self, tmp_path: Path) -> None:
        config = IgnoreConfig(
            learn=["CLAUDE.md", "AGENTS.md"],
            mutate=["CLAUDE.md", "AGENTS.md", ".cursorrules"],
        )
        policy = IgnorePolicy.load(tmp_path, config)
        assert policy.is_ignored("CLAUDE.md", "learn")
        assert policy.is_ignored("CLAUDE.md", "mutate")
        assert not policy.is_ignored("CLAUDE.md", "compress")
        assert policy.is_ignored(".cursorrules", "mutate")
        assert not policy.is_ignored(".cursorrules", "learn")

    def test_headroom_config_carries_ignore_config(self) -> None:
        cfg = HeadroomConfig(ignore=IgnoreConfig(mutate=["CLAUDE.md"]))
        assert cfg.ignore.mutate == ["CLAUDE.md"]
        assert cfg.ignore.paths == []


class TestGlobalIgnoreForMutate:
    """5. Global ignore rules (`paths`) applying to mutation at minimum."""

    def test_global_paths_rule_blocks_mutate(self, tmp_path: Path) -> None:
        config = IgnoreConfig(paths=["CLAUDE.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        assert policy.is_ignored("CLAUDE.md", "mutate")
        assert policy.is_ignored("CLAUDE.md", "compress")
        assert policy.is_ignored("CLAUDE.md", "learn")
        assert policy.is_ignored("CLAUDE.md", "memory")

    def test_headroomignore_and_config_are_merged(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "CLAUDE.md\n")
        config = IgnoreConfig(mutate=["AGENTS.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        assert policy.is_ignored("CLAUDE.md", "mutate")  # from file
        assert policy.is_ignored("AGENTS.md", "mutate")  # from config
        assert not policy.is_ignored("AGENTS.md", "compress")  # scoped


class TestRootRelativeMatching:
    """6. Root-relative matching for nested files."""

    def test_rooted_pattern_matches_only_at_that_path(self, tmp_path: Path) -> None:
        config = IgnoreConfig(paths=[".github/copilot-instructions.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        assert policy.is_ignored(".github/copilot-instructions.md", "mutate")
        # A different file with the same name in a different directory
        # must not match a rooted (slash-containing) pattern.
        assert not policy.is_ignored("other/copilot-instructions.md", "mutate")

    def test_absolute_path_under_root_is_relativized(self, tmp_path: Path) -> None:
        config = IgnoreConfig(paths=["CLAUDE.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        nested_abs = tmp_path / "CLAUDE.md"
        assert policy.is_ignored(nested_abs, "mutate")

    def test_bare_name_matches_any_depth(self, tmp_path: Path) -> None:
        config = IgnoreConfig(paths=["CLAUDE.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        deep_abs = tmp_path / "sub" / "dir" / "CLAUDE.md"
        assert policy.is_ignored(deep_abs, "mutate")
        assert policy.is_ignored("sub/dir/CLAUDE.md", "mutate")

    def test_absolute_path_outside_root_falls_back_gracefully(self, tmp_path: Path) -> None:
        config = IgnoreConfig(paths=["CLAUDE.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        outside = Path("/some/other/tree/CLAUDE.md")
        # Bare-name rule still matches by basename even if the path can't be
        # relativized against root.
        assert policy.is_ignored(outside, "mutate")


class TestDiagnostics:
    """7. Diagnostics/debug output includes active ignore rules."""

    def test_active_rules_lists_all_sources(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "CLAUDE.md\n")
        config = IgnoreConfig(mutate=["AGENTS.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        rules = policy.active_rules()
        assert len(rules) == 2
        patterns = {r.pattern for r in rules}
        assert patterns == {"CLAUDE.md", "AGENTS.md"}

    def test_describe_returns_human_readable_lines(self, tmp_path: Path) -> None:
        config = IgnoreConfig(mutate=["AGENTS.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        lines = policy.describe()
        assert len(lines) == 1
        assert "AGENTS.md" in lines[0]
        assert "mutate" in lines[0]
        assert "config:ignore.mutate" in lines[0]

    def test_doctor_check_reports_active_rules(self, tmp_path: Path, monkeypatch) -> None:
        from headroom.cli.doctor import check_ignore_rules

        _write(tmp_path / ".headroomignore", "CLAUDE.md\nAGENTS.md\n")
        result = check_ignore_rules(tmp_path)
        assert result.status == "pass"
        assert "2 active rule(s)" in result.summary
        assert "CLAUDE.md" in (result.hint or "")

    def test_doctor_check_no_rules(self, tmp_path: Path) -> None:
        from headroom.cli.doctor import check_ignore_rules

        result = check_ignore_rules(tmp_path)
        assert result.status == "pass"
        assert "no ignore rules" in result.summary


class TestMalformedConfig:
    """Malformed config entries are surfaced, not silently swallowed."""

    def test_non_string_entry_logged_and_skipped(self, tmp_path: Path, caplog) -> None:
        config = IgnoreConfig(mutate=[123, "CLAUDE.md"])  # type: ignore[list-item]
        with caplog.at_level("WARNING"):
            policy = IgnorePolicy.load(tmp_path, config)
        assert policy.is_ignored("CLAUDE.md", "mutate")
        assert any("malformed" in record.message for record in caplog.records)
        # Not just logged — surfaced on the policy so `headroom doctor` can
        # report it too.
        assert any("malformed" in w for w in policy.warnings)

    def test_unreadable_headroomignore_surfaces_warning(self, tmp_path: Path) -> None:
        ignore_path = _write(tmp_path / ".headroomignore", "CLAUDE.md\n")
        # Simulate an unreadable file by pointing at a directory instead.
        ignore_path.unlink()
        ignore_path.mkdir()
        policy = IgnorePolicy.load(tmp_path)
        assert policy.warnings
        assert "could not read" in policy.warnings[0]


class TestCarlFixture:
    """8. Representative cARL-style fixture."""

    def test_carl_fixture(self, tmp_path: Path) -> None:
        _write(
            tmp_path / ".headroomignore",
            "\n".join(
                [
                    "# cARL canonical runtime and governance artefacts",
                    ".github/carl/**",
                    "",
                    "# cARL-generated harness projections",
                    ".github/copilot-instructions.md",
                    "CLAUDE.md",
                    "AGENTS.md",
                    ".cursorrules",
                    "ANTIGRAVITY.md",
                    "",
                ]
            ),
        )
        # Also create the fixture files on disk to mirror a real repo layout.
        _write(tmp_path / ".github" / "carl" / "governance.md", "canonical")
        _write(tmp_path / ".github" / "copilot-instructions.md", "generated")
        _write(tmp_path / "CLAUDE.md", "generated")
        _write(tmp_path / "AGENTS.md", "generated")
        _write(tmp_path / "src" / "main.py", "print('hi')")

        policy = IgnorePolicy.load(tmp_path)

        for behavior in ("compress", "learn", "mutate", "memory"):
            assert policy.is_ignored(".github/carl/governance.md", behavior)
            assert policy.is_ignored(".github/copilot-instructions.md", behavior)
            assert policy.is_ignored("CLAUDE.md", behavior)
            assert policy.is_ignored("AGENTS.md", behavior)
            assert policy.is_ignored(".cursorrules", behavior)
            assert policy.is_ignored("ANTIGRAVITY.md", behavior)
            # Unrelated source files remain untouched.
            assert not policy.is_ignored("src/main.py", behavior)
            assert not policy.is_ignored("README.md", behavior)


class TestConvenienceFunction:
    def test_is_path_ignored_one_shot(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "CLAUDE.md\n")
        assert is_path_ignored("CLAUDE.md", "mutate", root=tmp_path)
        assert not is_path_ignored("README.md", "mutate", root=tmp_path)


class TestGlobSemantics:
    """9. Tricky glob cases — pins down the *limited* gitignore-like syntax.

    These document actual current behavior (including known deviations from
    real gitignore, e.g. a single "*" crossing "/" boundaries) so future
    changes to matching semantics are visible as test diffs, not surprises.
    """

    def test_github_star_matches_direct_and_nested_children(self, tmp_path: Path) -> None:
        # Known limitation: unlike real gitignore, ".github/*" is NOT limited
        # to direct children — fnmatch's "*" has no path-segment concept.
        _write(tmp_path / ".headroomignore", ".github/*\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored(".github/copilot-instructions.md", "mutate")
        assert policy.is_ignored(".github/workflows/ci.yml", "mutate")
        assert not policy.is_ignored("src/.github/x.md", "mutate")

    def test_github_globstar_matches_any_depth(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", ".github/**\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored(".github/workflows/ci.yml", "mutate")
        assert policy.is_ignored(".github/carl/x/y.md", "mutate")
        assert not policy.is_ignored("other/.github/x.md", "mutate")

    def test_github_carl_globstar_covers_directory_itself(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", ".github/carl/**\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored(".github/carl", "mutate")
        assert policy.is_ignored(".github/carl/x.md", "mutate")
        assert not policy.is_ignored(".github/other.md", "mutate")

    def test_bare_directory_slash_matches_any_depth(self, tmp_path: Path) -> None:
        # "dir/" with no other "/" matches like a plain gitignore directory
        # entry: at any depth, not just at the repository root.
        _write(tmp_path / ".headroomignore", "dir/\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored("dir/x.py", "mutate")
        assert policy.is_ignored("dir", "mutate")
        assert policy.is_ignored("nested/dir/x.py", "mutate")
        assert not policy.is_ignored("otherdir/x.py", "mutate")

    def test_star_md_matches_any_depth(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "*.md\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored("CLAUDE.md", "mutate")
        assert policy.is_ignored("nested/CLAUDE.md", "mutate")
        assert not policy.is_ignored("nested/CLAUDE.txt", "mutate")

    def test_nested_bare_filename_matches_at_any_depth_not_just_listed_depth(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / ".headroomignore", "nested/CLAUDE.md\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored("nested/CLAUDE.md", "mutate")
        # Rooted (contains "/"): does NOT match at a different depth.
        assert not policy.is_ignored("other/nested/CLAUDE.md", "mutate")

    def test_rooted_leading_slash_matches_only_at_repo_root(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "/CLAUDE.md\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored("CLAUDE.md", "mutate")
        assert not policy.is_ignored("nested/CLAUDE.md", "mutate")

    def test_rooted_directory_leading_slash(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "/dir/\n")
        policy = IgnorePolicy.load(tmp_path)
        assert policy.is_ignored("dir/x.py", "mutate")
        assert not policy.is_ignored("nested/dir/x.py", "mutate")

    def test_negation_patterns_unsupported_and_warned(self, tmp_path: Path) -> None:
        _write(tmp_path / ".headroomignore", "!README.md\n")
        policy = IgnorePolicy.load(tmp_path)
        # The negation line is skipped entirely (not loaded as a rule at
        # all — in particular NOT matched as a literal path "!README.md")
        # and reported as a warning instead.
        assert policy.active_rules() == []
        assert not policy.is_ignored("!README.md", "mutate")
        assert any("negation" in w for w in policy.warnings)

    def test_negation_pattern_in_config_unsupported_and_warned(self, tmp_path: Path) -> None:
        config = IgnoreConfig(mutate=["!CLAUDE.md"])
        policy = IgnorePolicy.load(tmp_path, config)
        assert policy.active_rules() == []
        assert any("negation" in w for w in policy.warnings)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
