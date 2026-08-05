"""Tests for pr-health-labels.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).parents[2] / ".github" / "scripts" / "pr-health-labels.py"
    spec = importlib.util.spec_from_file_location("pr_health_labels", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_state_ignores_historical_failures_when_latest_attempt_passed() -> None:
    module = _load_module()
    payload = {
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "workflowName": "PR Governance",
                "name": "template",
                "conclusion": "FAILURE",
                "startedAt": "2026-06-14T13:38:35Z",
                "completedAt": "2026-06-14T13:38:46Z",
            },
            {
                "__typename": "CheckRun",
                "workflowName": "PR Governance",
                "name": "template",
                "conclusion": "SUCCESS",
                "startedAt": "2026-06-14T13:43:26Z",
                "completedAt": "2026-06-14T13:43:35Z",
            },
            {
                "__typename": "CheckRun",
                "workflowName": "PR Governance",
                "name": "label",
                "conclusion": "CANCELLED",
                "startedAt": "2026-06-14T13:43:18Z",
                "completedAt": "2026-06-14T13:43:24Z",
            },
            {
                "__typename": "CheckRun",
                "workflowName": "PR Governance",
                "name": "label",
                "conclusion": "SUCCESS",
                "startedAt": "2026-06-14T13:43:26Z",
                "completedAt": "2026-06-14T13:43:35Z",
            },
            {
                "__typename": "CheckRun",
                "workflowName": "",
                "name": "GitGuardian Security Checks",
                "conclusion": "SUCCESS",
                "startedAt": "2026-06-14T13:38:32Z",
                "completedAt": "2026-06-14T13:39:04Z",
            },
        ]
    }

    assert module.check_state(payload) == "passing"


def test_check_state_fails_when_latest_attempt_failed() -> None:
    module = _load_module()
    payload = {
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "workflowName": "PR Governance",
                "name": "template",
                "conclusion": "SUCCESS",
                "startedAt": "2026-06-14T13:38:35Z",
                "completedAt": "2026-06-14T13:38:46Z",
            },
            {
                "__typename": "CheckRun",
                "workflowName": "PR Governance",
                "name": "template",
                "conclusion": "FAILURE",
                "startedAt": "2026-06-14T13:43:26Z",
                "completedAt": "2026-06-14T13:43:35Z",
            },
        ]
    }

    assert module.check_state(payload) == "failing"
