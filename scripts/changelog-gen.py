#!/usr/bin/env python3
"""Generate changelog from conventional commits."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).parent.parent

COMMIT_PATTERN = re.compile(
    r"^(feat|fix|ci|chore|perf|refactor|docs|style|test)(\(.+\))?(!)?:\s*(.+)$"
)
BREAKING_CHANGE_PATTERN = re.compile(r"^BREAKING CHANGE:\s*(.+)$", re.MULTILINE)
COMMIT_ENTRY_PATTERN = re.compile(r"^(.+?)(?:\n(.+))?\|(\w+)$", re.MULTILINE)
FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"
GIT_LOG_FORMAT = "%s%x1f%b%x1f%h%x1e"

TYPE_LABELS: dict[str, str] = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "ci": "CI/CD",
    "chore": "Chores",
    "perf": "Performance",
    "refactor": "Refactors",
    "docs": "Documentation",
    "style": "Styles",
    "test": "Tests",
    "other": "Other Changes",
}


class ParsedCommit(NamedTuple):
    type: str
    scope: str | None
    breaking: bool
    message: str
    hash: str


def iter_commit_entries(log_output: str) -> list[tuple[str, str, str]]:
    """Split raw git log output into (subject, body, hash) tuples."""

    if not log_output.strip():
        return []

    if RECORD_SEP in log_output and FIELD_SEP in log_output:
        entries: list[tuple[str, str, str]] = []
        for raw_entry in log_output.split(RECORD_SEP):
            if not raw_entry:
                continue
            if FIELD_SEP not in raw_entry:
                continue
            subject, body_and_hash = raw_entry.split(FIELD_SEP, 1)
            if FIELD_SEP not in body_and_hash:
                continue
            body, commit_hash = body_and_hash.rsplit(FIELD_SEP, 1)
            entries.append((subject.strip(), body.strip(), commit_hash.strip()))
        return entries

    return [
        (
            match.group(1).strip(),
            (match.group(2) or "").strip(),
            match.group(3).strip(),
        )
        for match in COMMIT_ENTRY_PATTERN.finditer(log_output)
    ]


def get_merge_summary(subject: str, body: str) -> str:
    """Return the first meaningful summary line for a merge commit."""

    if not subject.startswith("Merge "):
        return ""

    for line in body.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def parse_commits(log_output: str) -> list[ParsedCommit]:
    """Parse git log output into structured commits."""

    commits: list[ParsedCommit] = []

    for subject, body, commit_hash in iter_commit_entries(log_output):
        is_breaking = bool(BREAKING_CHANGE_PATTERN.search(body))
        merge_summary = get_merge_summary(subject, body)
        candidates = [subject]
        if merge_summary:
            candidates.insert(0, merge_summary)

        for candidate in candidates:
            commit_match = COMMIT_PATTERN.match(candidate)
            if not commit_match:
                continue

            scope = commit_match.group(2)
            if scope:
                scope = scope[1:-1]
            commits.append(
                ParsedCommit(
                    type=commit_match.group(1),
                    scope=scope,
                    breaking=is_breaking or bool(commit_match.group(3)),
                    message=commit_match.group(4),
                    hash=commit_hash,
                )
            )
            break
        else:
            fallback_message = merge_summary or subject
            if not fallback_message or fallback_message.startswith("Merge "):
                continue
            commits.append(
                ParsedCommit(
                    type="other",
                    scope=None,
                    breaking=is_breaking,
                    message=fallback_message,
                    hash=commit_hash,
                )
            )

    return commits


def generate_changelog(version: str, commits: list[ParsedCommit]) -> str:
    """Generate markdown changelog from parsed commits."""
    today = date.today().isoformat()
    lines = [f"## [{version}] - {today}", ""]

    # Collect breaking changes
    breaking_commits = [c for c in commits if c.breaking]
    if breaking_commits:
        lines.append("### Breaking Changes")
        for commit in breaking_commits:
            if commit.scope:
                lines.append(f"- **{commit.scope}**: {commit.message} ({commit.hash})")
            else:
                lines.append(f"- {commit.message} ({commit.hash})")
        lines.append("")

    # Group by type
    by_type: dict[str, list[ParsedCommit]] = {}
    for commit in commits:
        by_type.setdefault(commit.type, []).append(commit)

    for commit_type, label in TYPE_LABELS.items():
        type_commits = by_type.get(commit_type, [])
        if not type_commits:
            continue
        lines.append(f"### {label}")
        for commit in type_commits:
            if commit.scope:
                lines.append(f"- **{commit.scope}**: {commit.message} ({commit.hash})")
            else:
                lines.append(f"- {commit.message} ({commit.hash})")
        lines.append("")

    return "\n".join(lines) + "\n"


def run_git_log(since: str | None, cwd: Path) -> str:
    """Run git log command and return output."""
    cmd = ["git", "log", "--first-parent", f"--pretty=format:{GIT_LOG_FORMAT}"]
    if since:
        cmd.append(f"{since}..HEAD")
    else:
        cmd.append("HEAD")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate changelog from conventional commits")
    parser.add_argument("--version", required=True, help="Version number (e.g., 0.6.0)")
    parser.add_argument("--since", help="Starting tag (exclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing")
    args = parser.parse_args()

    log_output = run_git_log(args.since, ROOT)
    commits = parse_commits(log_output)
    changelog = generate_changelog(args.version, commits)

    if args.dry_run:
        print(changelog)
    else:
        output_path = ROOT / ".changelog.md"
        output_path.write_text(changelog, encoding="utf-8")
        print(f"Changelog written to {output_path}")


if __name__ == "__main__":
    main()
