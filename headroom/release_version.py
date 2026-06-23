"""Release version helpers for the GitHub Actions release workflow."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
RELEASE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?$")
CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(feat|fix|ci|chore|perf|refactor|docs|style|test)(\(.+\))?(!)?:\s*(.+)$"
)
BREAKING_CHANGE_RE = re.compile(r"^BREAKING CHANGE:\s*(.+)$", re.MULTILINE)
FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"
GIT_LOG_FORMAT = "%s%x1f%b%x1e"
BUMP_PRIORITY = {"patch": 0, "minor": 1, "major": 2}


@dataclass(frozen=True, order=True)
class SemVer:
    """Semantic version tuple with simple bump helpers."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> SemVer:
        match = SEMVER_RE.match(value)
        if not match:
            raise ValueError(f"Invalid semantic version: {value}")
        return cls(*(int(part) for part in match.groups()))

    def bump(self, level: str) -> SemVer:
        if level == "major":
            return SemVer(self.major + 1, 0, 0)
        if level == "minor":
            return SemVer(self.major, self.minor + 1, 0)
        if level == "patch":
            return SemVer(self.major, self.minor, self.patch + 1)
        raise ValueError(f"Unsupported bump level: {level}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class ReleaseVersionInfo:
    """Workflow outputs for release version calculation."""

    version: str
    npm_version: str
    canonical: str
    height: str
    bump: str
    previous_tag: str

    def as_outputs(self) -> dict[str, str]:
        return {
            "version": self.version,
            "npm_version": self.npm_version,
            "canonical": self.canonical,
            "height": self.height,
            "bump": self.bump,
            "previous_tag": self.previous_tag,
        }


@dataclass(frozen=True, order=True)
class ReleaseTag:
    """Parsed release tag metadata used for sorting and normalization."""

    version: SemVer
    legacy_height: int = -1
    raw: str = ""


@dataclass(frozen=True)
class CommitInfo:
    """Commit subject/body pair used for bump detection."""

    subject: str
    body: str = ""


def parse_release_tag(tag: str) -> ReleaseTag:
    """Parse a release tag, preserving legacy fourth-component ordering."""

    match = RELEASE_TAG_RE.match(tag)
    if not match:
        raise ValueError(f"Invalid release tag: {tag}")
    major, minor, patch, extra = match.groups()
    return ReleaseTag(
        version=SemVer(int(major), int(minor), int(patch)),
        legacy_height=int(extra) if extra is not None else -1,
        raw=tag,
    )


def normalize_release_tag(tag: str) -> SemVer:
    """Collapse historic 4-part release tags into their base semantic version."""

    return parse_release_tag(tag).version


def find_latest_release_tag(tags: Sequence[str]) -> str | None:
    """Return the latest release tag after normalizing legacy 4-part tags."""

    candidates: list[ReleaseTag] = []
    for tag in tags:
        if RELEASE_TAG_RE.match(tag):
            candidates.append(parse_release_tag(tag))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0].raw


def _merge_summary(subject: str, body: str) -> str:
    """Return the first meaningful body line for merge commits."""

    if not subject.startswith("Merge "):
        return ""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def classify_commit_bump(commit: CommitInfo) -> str:
    """Classify one commit using conventional commit semantics."""

    merge_summary = _merge_summary(commit.subject, commit.body)
    candidates = [commit.subject]
    if merge_summary:
        candidates.insert(0, merge_summary)

    has_breaking_change = bool(BREAKING_CHANGE_RE.search(commit.body))
    for candidate in candidates:
        match = CONVENTIONAL_COMMIT_RE.match(candidate)
        if not match:
            continue
        if has_breaking_change or bool(match.group(3)):
            return "major"
        if match.group(1) == "feat":
            return "minor"
        return "patch"

    if has_breaking_change:
        return "major"
    return "patch"


def determine_bump_level(commits: Sequence[CommitInfo]) -> str:
    """Return the highest required bump across a commit range."""

    level = "patch"
    for commit in commits:
        candidate = classify_commit_bump(commit)
        if BUMP_PRIORITY[candidate] > BUMP_PRIORITY[level]:
            level = candidate
    return level


def compute_release_version(
    canonical_version: str,
    level: str,
    tags: Sequence[str],
    manual_version: str = "",
) -> ReleaseVersionInfo:
    """Compute the next release version from the canonical version and existing tags."""

    if manual_version:
        manual = str(SemVer.parse(manual_version))
        return ReleaseVersionInfo(
            version=manual,
            npm_version=manual,
            canonical=canonical_version,
            height="0",
            bump="manual",
            previous_tag="",
        )

    canonical = SemVer.parse(canonical_version)
    previous_tag = find_latest_release_tag(tags)
    current = canonical
    if previous_tag is not None:
        current = max(current, normalize_release_tag(previous_tag))

    next_version = str(current.bump(level))
    return ReleaseVersionInfo(
        version=next_version,
        npm_version=next_version,
        canonical=canonical_version,
        height="0",
        bump=level,
        previous_tag=previous_tag or "",
    )


def get_canonical_version(root: Path) -> str:
    """Read the canonical project version from pyproject.toml."""

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    with open(root / "pyproject.toml", "rb") as file:
        project = tomllib.load(file)["project"]
    return str(project["version"])


def list_release_tags(root: Path) -> list[str]:
    """List release tags from the local Git checkout."""

    result = subprocess.run(
        ["git", "tag", "-l", "v*"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [tag.strip() for tag in result.stdout.splitlines() if tag.strip()]


def list_release_commits(root: Path, previous_tag: str) -> list[CommitInfo]:
    """List commit subject/body pairs since the previous release tag."""

    cmd = ["git", "log", "--first-parent", f"--pretty=format:{GIT_LOG_FORMAT}"]
    if previous_tag:
        cmd.append(f"{previous_tag}..HEAD")
    else:
        cmd.append("HEAD")

    result = subprocess.run(
        cmd,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    commits: list[CommitInfo] = []
    for raw_entry in result.stdout.split(RECORD_SEP):
        if not raw_entry or FIELD_SEP not in raw_entry:
            continue
        subject, body = raw_entry.split(FIELD_SEP, 1)
        commits.append(CommitInfo(subject=subject.strip(), body=body.strip()))
    return commits


def commit_height_since(root: Path, previous_tag: str) -> str:
    """Count commits since the previous release tag for changelog/debug outputs."""

    if not previous_tag:
        return "0"

    result = subprocess.run(
        ["git", "rev-list", f"{previous_tag}..HEAD", "--count"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "0"


def write_github_outputs(info: ReleaseVersionInfo, output_path: str) -> None:
    """Append workflow outputs to the GitHub Actions output file."""

    with open(output_path, "a", encoding="utf-8") as output_file:
        for key, value in info.as_outputs().items():
            output_file.write(f"{key}={value}\n")


def main() -> None:
    root = Path.cwd()
    manual_version = os.environ.get("MANUAL_VER", "").strip()
    manual_raw = os.environ.get("MANUAL_VER") or os.environ.get("LEVEL") or "patch"
    manual_match = re.fullmatch(
        r"v?(\d+\.\d+\.\d+(?:[abrc]\d+)?)",
        manual_raw.strip(),
    )
    if manual_match:
        version = manual_match.group(1)
        info = ReleaseVersionInfo(
            version=version,
            npm_version=version,
            canonical=get_canonical_version(root),
            bump="manual",
            height="0",
            previous_tag="",
        )
        output_path = os.environ.get("GITHUB_OUTPUT")
        if output_path:
            write_github_outputs(info, output_path)
        print(f"version={info.version}")
        print(f"npm_version={info.npm_version}")
        print(f"height={info.height}")
        return
    tags = list_release_tags(root)
    previous_tag = find_latest_release_tag(tags) or ""
    level = os.environ.get("LEVEL", "").strip()
    manual_match = re.fullmatch(r"v?(\d+\.\d+\.\d+(?:[abrc]\d+)?)", level.strip())
    if manual_match:
        version = manual_match.group(1)
        info = ReleaseVersionInfo(
            version=version,
            npm_version=version,
            canonical=get_canonical_version(root),
            bump="manual",
            height="0",
            previous_tag="",
        )
        output_path = os.environ.get("GITHUB_OUTPUT")
        if output_path:
            write_github_outputs(info, output_path)
        print(f"version={info.version}")
        print(f"npm_version={info.npm_version}")
        print(f"height={info.height}")
        return
    if not level:
        level = determine_bump_level(list_release_commits(root, previous_tag))

    info = compute_release_version(
        canonical_version=get_canonical_version(root),
        level=level,
        tags=tags,
        manual_version=manual_version,
    )
    info = replace(info, height=commit_height_since(root, info.previous_tag))

    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if output_path:
        write_github_outputs(info, output_path)
        return

    for key, value in info.as_outputs().items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
