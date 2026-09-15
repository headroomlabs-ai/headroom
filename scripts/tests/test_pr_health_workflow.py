"""Tests for the PR governance workflow contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


def test_incomplete_pr_template_is_reported_without_failing_job() -> None:
    workflow = Path(".github/workflows/pr-health.yml").read_text(encoding="utf-8")

    assert "Fetch current PR body" in workflow
    assert "--body-file .pr-body.md" in workflow
    assert "Report incomplete PR body" in workflow
    assert "PR template validation found missing fields" in workflow
    assert "Fail when the PR body is incomplete" not in workflow
    assert 'echo "PR template validation failed' not in workflow


def test_ready_for_review_label_is_removed_when_changes_are_requested() -> None:
    workflow = Path(".github/workflows/pr-health.yml").read_text(encoding="utf-8")

    assert "reviewDecision" in workflow
    assert 'review_decision="$(jq -r \'.reviewDecision // ""\'' in workflow
    assert '$review_decision" == "CHANGES_REQUESTED"' in workflow


def test_merge_state_unknown_does_not_clear_conflict_or_rebase_labels() -> None:
    workflow = Path(".github/workflows/pr-health.yml").read_text(encoding="utf-8")

    assert 'elif [[ "$merge_state" != "UNKNOWN" ]]; then' in workflow
    assert 'gh pr edit "$pr" --repo "$REPO" --remove-label "status: needs rebase"' in workflow
    assert 'gh pr edit "$pr" --repo "$REPO" --remove-label "status: has conflicts"' in workflow


def test_rebase_label_uses_ref_comparison_instead_of_merge_state() -> None:
    workflow = Path(".github/workflows/pr-health.yml").read_text(encoding="utf-8")

    assert 'if [[ "$merge_state" == "BEHIND" ]]; then' not in workflow
    assert 'gh api "repos/$REPO/compare/$base_ref...$head_oid"' in workflow
    assert "--field drift" in workflow
    assert 'if [[ "$drift" == "stale" ]]; then' in workflow


def test_unknown_drift_does_not_clear_the_rebase_label() -> None:
    workflow = Path(".github/workflows/pr-health.yml").read_text(encoding="utf-8")

    assert 'elif [[ "$drift" == "current" ]]; then' in workflow


def _base_files_snippet(workflow: str) -> str:
    """The shell block that asks GitHub which files the base branch moved."""
    start = workflow.index("            # Files the base branch changed")
    end = workflow.index('            drift="$(python3', start)
    return dedent(workflow[start:end])


def test_failed_base_file_comparison_keeps_an_unknown_sentinel() -> None:
    workflow = Path(".github/workflows/pr-health.yml").read_text(encoding="utf-8")

    assert "echo '[]'" not in workflow
    assert "base_files=''" in workflow
    assert "|| base_files=''" in workflow


def test_failed_base_file_comparison_does_not_clear_the_rebase_label() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")

    workflow = Path(".github/workflows/pr-health.yml").read_text(encoding="utf-8")
    script = "\n".join(
        [
            "set -euo pipefail",
            # Stands in for a throttled API call that writes partial output and fails.
            "gh() { printf 'partia'; return 1; }",
            'REPO="headroomlabs-ai/headroom"',
            'base_ref="main"',
            'behind_by="13"',
            'merge_base="0123456789abcdef0123456789abcdef01234567"',
            _base_files_snippet(workflow),
            'printf "%s" "$base_files"',
        ]
    )
    result = subprocess.run([bash, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""

    drift = subprocess.run(
        [
            sys.executable,
            ".github/scripts/pr-health-labels.py",
            "--state-json",
            json.dumps(
                {"mergeStateStatus": "CLEAN", "files": [{"path": "headroom/proxy/server.py"}]}
            ),
            "--field",
            "drift",
            "--behind-by",
            "13",
            "--base-files",
            result.stdout,
        ],
        capture_output=True,
        text=True,
    )

    assert drift.returncode == 0, drift.stderr
    assert drift.stdout.strip() == "unknown"
    # An unknown verdict matches neither branch, so no gh pr edit runs for the label.
    start = workflow.index('            if [[ "$drift" == "stale" ]]; then')
    dispatch = workflow[start : workflow.index("\n            fi\n", start)]
    assert 'elif [[ "$drift" == "current" ]]; then' in dispatch
    assert "else" not in dispatch
