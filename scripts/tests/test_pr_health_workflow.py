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
