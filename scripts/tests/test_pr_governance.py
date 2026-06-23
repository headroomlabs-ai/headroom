"""Tests for pr-governance.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).parent.parent / "pr-governance.py"
    spec = importlib.util.spec_from_file_location("pr_governance", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _event(body: str, *, draft: bool = False, login: str = "octocat") -> dict[str, object]:
    return {
        "pull_request": {
            "number": 42,
            "draft": draft,
            "body": body,
            "user": {"login": login},
        }
    }


VALID_BODY = """## Description

Add a required PR-governance gate for template validation and review readiness.

Closes #123

## Type of Change

- [x] New feature (non-breaking change that adds functionality)
- [ ] Documentation update

## Changes Made

- Added a workflow-backed PR template validator.
- Added local commit message linting in the commit-msg hook.

## Testing

- [x] Unit tests pass (`pytest`)
- [x] Manual testing performed

### Test Output

```text
pytest scripts/tests/test_pr_governance.py -q
```

## Real Behavior Proof

- Environment: Ubuntu runner, Python 3.12
- Exact command / steps: Open a PR, remove the ready checkbox, re-run the workflow.
- Observed result: The governance check fails and the PR gets a needs-author-action label.
- Not tested: Automatic Copilot review rulesets in repository settings.

## Review Readiness

- [x] I have performed a self-review
- [x] This PR is ready for human review

## Additional Notes

- Maintainers can optionally enable Copilot code review from repository rulesets.
"""


def test_validate_pull_request_marks_ready_pr_valid() -> None:
    module = _load_module()

    report = module.validate_pull_request(_event(VALID_BODY))

    assert report.valid is True
    assert report.ready_for_review is True
    assert report.needs_author_action is False
    assert report.problems == []
    assert report.labels_to_add == [module.READY_LABEL]
    assert module.AUTHOR_ACTION_LABEL in report.labels_to_remove


def test_validate_pull_request_accepts_crlf_test_output_code_block() -> None:
    module = _load_module()
    body = VALID_BODY.replace("\n", "\r\n")

    report = module.validate_pull_request(_event(body))

    assert report.valid is True
    assert report.problems == []


def test_validate_pull_request_allows_draft_without_ready_checkboxes() -> None:
    module = _load_module()
    body = VALID_BODY.replace(
        "- [x] I have performed a self-review", "- [ ] I have performed a self-review"
    )
    body = body.replace(
        "- [x] This PR is ready for human review",
        "- [ ] This PR is ready for human review",
    )

    report = module.validate_pull_request(_event(body, draft=True))

    assert report.valid is True
    assert report.ready_for_review is False
    assert report.needs_author_action is False
    assert report.labels_to_add == []
    assert module.READY_LABEL in report.labels_to_remove


def test_validate_pull_request_fails_on_missing_required_content() -> None:
    module = _load_module()
    body = """## Description

Fixes #123

## Type of Change

- [ ] New feature (non-breaking change that adds functionality)

## Changes Made

- Change 1

## Testing

### Test Output

```text
# Paste relevant command output or artifact links here
```

## Real Behavior Proof

- Environment:
- Exact command / steps:
- Observed result:
- Not tested:

## Review Readiness

- [ ] I have performed a self-review
- [ ] This PR is ready for human review
"""

    report = module.validate_pull_request(_event(body))

    assert report.valid is False
    assert report.needs_author_action is True
    assert module.AUTHOR_ACTION_LABEL in report.labels_to_add
    assert any("Description" in problem for problem in report.problems)
    assert any("Type of Change" in problem for problem in report.problems)
    assert any("Test Output" in problem for problem in report.problems)
    assert any("Real Behavior Proof" in problem for problem in report.problems)


def test_validate_pull_request_skips_bot_authored_prs() -> None:
    module = _load_module()

    report = module.validate_pull_request(_event("", login="dependabot[bot]"))

    assert report.valid is True
    assert report.is_bot_pr is True
    assert report.needs_author_action is False
    assert report.labels_to_add == []
