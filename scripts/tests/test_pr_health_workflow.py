"""Tests for the PR governance workflow contract."""

from __future__ import annotations

from pathlib import Path


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
