"""Tests for pr-health-labels.py."""

from __future__ import annotations

import importlib.util
import json
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


def test_drift_state_flags_branch_behind_on_files_it_also_changes() -> None:
    module = _load_module()
    payload = {
        "mergeStateStatus": "CLEAN",
        "files": [{"path": "headroom/proxy/server.py"}],
    }

    base_files = ["headroom/proxy/server.py", "docs/index.md"]

    assert module.drift_state(payload, 13, base_files) == "stale"


def test_drift_state_ignores_base_movement_on_unrelated_files() -> None:
    module = _load_module()
    payload = {
        "mergeStateStatus": "CLEAN",
        "files": [{"path": "headroom/proxy/server.py"}],
    }

    assert module.drift_state(payload, 13, ["docs/index.md"]) == "current"


def test_drift_state_is_current_when_the_branch_is_up_to_date() -> None:
    module = _load_module()
    payload = {
        "mergeStateStatus": "CLEAN",
        "files": [{"path": "headroom/proxy/server.py"}],
    }

    assert module.drift_state(payload, 0, []) == "current"


def test_drift_state_is_unknown_when_the_comparison_failed() -> None:
    module = _load_module()
    payload = {
        "mergeStateStatus": "UNKNOWN",
        "files": [{"path": "headroom/proxy/server.py"}],
    }

    assert module.drift_state(payload, None, []) == "unknown"


def test_drift_state_still_trusts_a_behind_merge_state() -> None:
    module = _load_module()
    payload = {"mergeStateStatus": "BEHIND", "files": []}

    assert module.drift_state(payload, None, []) == "stale"


def test_parse_behind_by_reads_an_empty_comparison_as_unknown() -> None:
    module = _load_module()

    assert module.parse_behind_by("13") == 13
    assert module.parse_behind_by("") is None
    assert module.parse_behind_by("null") is None


def test_main_prints_the_drift_state() -> None:
    module = _load_module()
    state_json = json.dumps(
        {"mergeStateStatus": "CLEAN", "files": [{"path": "headroom/proxy/server.py"}]}
    )

    assert (
        module.main(
            [
                "--state-json",
                state_json,
                "--field",
                "drift",
                "--behind-by",
                "13",
                "--base-files",
                json.dumps(["headroom/proxy/server.py"]),
            ]
        )
        == 0
    )


def test_drift_state_is_unknown_when_the_base_file_comparison_failed() -> None:
    module = _load_module()
    payload = {
        "mergeStateStatus": "CLEAN",
        "files": [{"path": "headroom/proxy/server.py"}],
    }

    assert module.drift_state(payload, 13, None) == "unknown"


def test_drift_state_is_current_when_the_base_moved_no_files() -> None:
    module = _load_module()
    payload = {
        "mergeStateStatus": "CLEAN",
        "files": [{"path": "headroom/proxy/server.py"}],
    }

    assert module.drift_state(payload, 13, []) == "current"


def test_parse_base_files_reads_a_failed_comparison_as_unknown() -> None:
    module = _load_module()

    assert module.parse_base_files("") is None
    assert module.parse_base_files("   ") is None
    assert module.parse_base_files("not json") is None
    assert module.parse_base_files('{"files": []}') is None
    assert module.parse_base_files("[]") == []
    assert module.parse_base_files('["headroom/proxy/server.py"]') == ["headroom/proxy/server.py"]


def test_main_prints_unknown_when_the_base_file_comparison_failed(capsys) -> None:
    module = _load_module()
    state_json = json.dumps(
        {"mergeStateStatus": "CLEAN", "files": [{"path": "headroom/proxy/server.py"}]}
    )

    assert (
        module.main(
            [
                "--state-json",
                state_json,
                "--field",
                "drift",
                "--behind-by",
                "13",
                "--base-files",
                "",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "unknown"
