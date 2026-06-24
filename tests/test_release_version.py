"""Tests for release version normalization and bumping."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from headroom.release_version import (
    CommitInfo,
    classify_commit_bump,
    compute_release_version,
    determine_bump_level,
    find_latest_release_tag,
    get_canonical_version,
    list_release_commits,
    normalize_release_tag,
    parse_release_tag,
)

ROOT = Path(__file__).resolve().parent.parent


def test_normalize_release_tag_preserves_three_part_tag() -> None:
    assert str(normalize_release_tag("v0.5.20")) == "0.5.20"


def test_normalize_release_tag_collapses_four_part_tag() -> None:
    assert str(normalize_release_tag("v0.5.25.2")) == "0.5.25"


def test_compute_patch_release_from_four_part_history() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=["v0.5.20", "v0.5.25.1", "v0.5.25.2"],
    )

    assert info.version == "0.5.26"
    assert info.npm_version == "0.5.26"
    assert info.previous_tag == "v0.5.25.2"
    assert info.bump == "patch"


def test_compute_minor_release_from_four_part_history() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="minor",
        tags=["v0.5.20", "v0.5.25.1", "v0.5.25.2"],
    )

    assert info.version == "0.6.0"
    assert info.npm_version == "0.6.0"
    assert info.previous_tag == "v0.5.25.2"
    assert info.bump == "minor"


def test_compute_patch_release_from_canonical_without_tags() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=[],
    )

    assert info.version == "0.5.26"
    assert info.npm_version == "0.5.26"
    assert info.previous_tag == ""


def test_manual_version_override_uses_single_semver() -> None:
    info = compute_release_version(
        canonical_version="0.5.25",
        level="patch",
        tags=["v0.5.25.2"],
        manual_version="0.6.0",
    )

    assert info.version == "0.6.0"
    assert info.npm_version == "0.6.0"
    assert info.previous_tag == ""
    assert info.bump == "manual"


def test_manual_version_override_rejects_legacy_four_part_version() -> None:
    with pytest.raises(ValueError, match="Invalid semantic version"):
        compute_release_version(
            canonical_version="0.5.25",
            level="patch",
            tags=["v0.5.25.2"],
            manual_version="0.5.25.3",
        )


def test_find_latest_release_tag_prefers_highest_normalized_version() -> None:
    assert find_latest_release_tag(["v0.5.25.2", "v0.5.27", "not-a-tag"]) == "v0.5.27"


def test_find_latest_release_tag_prefers_higher_legacy_height_with_same_base() -> None:
    assert find_latest_release_tag(["v0.5.25.2", "v0.5.25.3", "v0.5.25"]) == "v0.5.25.3"


def test_parse_release_tag_preserves_legacy_height_for_sorting() -> None:
    tag = parse_release_tag("v0.5.25.3")
    assert str(tag.version) == "0.5.25"
    assert tag.legacy_height == 3


def test_classify_commit_bump_treats_breaking_change_as_major() -> None:
    assert (
        classify_commit_bump(
            CommitInfo(subject="fix(api)!: change response shape", body=""),
        )
        == "major"
    )


def test_determine_bump_level_uses_greatest_commit_level() -> None:
    commits = [
        CommitInfo(subject="fix: patch one", body=""),
        CommitInfo(subject="feat: add capability", body=""),
        CommitInfo(subject="chore: maintenance", body=""),
    ]

    assert determine_bump_level(commits) == "minor"


def test_determine_bump_level_prefers_major_over_minor_and_patch() -> None:
    commits = [
        CommitInfo(subject="fix: patch one", body=""),
        CommitInfo(subject="feat: add capability", body=""),
        CommitInfo(
            subject="docs: update migration guide",
            body="BREAKING CHANGE: the API changed",
        ),
    ]

    assert determine_bump_level(commits) == "major"


def test_list_release_commits_parses_empty_body_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock()
    run.return_value = Mock(
        stdout="feat: add capability\x1f\x1efix: patch bug\x1fbody text\x1e",
    )
    monkeypatch.setattr("headroom.release_version.subprocess.run", run)

    commits = list_release_commits(ROOT, "")

    assert commits == [
        CommitInfo(subject="feat: add capability", body=""),
        CommitInfo(subject="fix: patch bug", body="body text"),
    ]


def test_release_version_script_runs_directly_without_importing_headroom_package(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "github-output.txt"
    env = os.environ.copy()
    env["GITHUB_OUTPUT"] = str(output_path)
    env["LEVEL"] = "patch"
    env["MANUAL_VER"] = "0.6.0"

    result = subprocess.run(
        [sys.executable, str(ROOT / "headroom" / "release_version.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    canonical_version = get_canonical_version(ROOT)
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "version=0.6.0",
        "npm_version=0.6.0",
        f"canonical={canonical_version}",
        "height=0",
        "bump=manual",
        "previous_tag=",
    ]
