"""scripts/codeql_triage.sh: one triage issue per ISO week, however often it runs.

The script runs for real against a fake ``gh`` on PATH that keeps alerts and
issues in a JSON file, so re-runs, later dispatches in the same week, a closed
issue with ticked boxes, and the ISO year boundary are all exercised end to end.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "codeql_triage.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("bash") and shutil.which("jq")),
    reason="needs bash and jq (present on the Linux runner that executes the workflow)",
)

FAKE_GH = r'''#!/usr/bin/env python3
"""Just enough of `gh` for scripts/codeql_triage.sh, backed by $FAKE_GH_STATE."""
import json, os, sys

path = os.environ["FAKE_GH_STATE"]
state = json.load(open(path))
args = sys.argv[1:]

def save():
    json.dump(state, open(path, "w"))

def opt(name):
    return args[args.index(name) + 1]

if args[0] == "api":
    target = next(a for a in args[1:] if a.startswith("repos/"))
    if "/code-scanning/alerts" in target:
        print(json.dumps(state["alerts"]))
    elif "/issues?" in target:
        print(json.dumps(state["issues"]))
    elif target.endswith("/comments"):
        number = int(target.split("/")[-2])
        issue = next(i for i in state["issues"] if i["number"] == number)
        print(json.dumps([{"body": c} for c in issue["comments"]]))
    else:
        number = int(target.rsplit("/", 1)[1])
        print(json.dumps(next(i for i in state["issues"] if i["number"] == number)))
elif args[:2] == ["issue", "create"]:
    number = len(state["issues"]) + 1
    state["issues"].append({
        "number": number,
        "title": opt("--title"),
        "body": open(opt("--body-file")).read(),
        "state": "open",
        "labels": [opt("--label")],
        "comments": [],
    })
    save()
    print(f"https://github.com/o/r/issues/{number}")
elif args[:2] == ["issue", "comment"]:
    number = int(args[2])
    issue = next(i for i in state["issues"] if i["number"] == number)
    issue["comments"].append(open(opt("--body-file")).read())
    save()
else:
    sys.exit(f"fake gh: unsupported {args}")
'''


def _alert(number: int, severity: str = "high") -> dict:
    return {
        "number": number,
        "html_url": f"https://github.com/o/r/security/code-scanning/{number}",
        "rule": {"id": f"py/rule-{number}", "security_severity_level": severity},
        "most_recent_instance": {"location": {"path": "headroom/x.py", "start_line": number}},
        "created_at": "2026-10-01T00:00:00Z",
    }


@pytest.fixture
def harness(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    state_path = tmp_path / "state.json"
    workdir = tmp_path / "work"
    workdir.mkdir()

    def set_alerts(*alerts):
        state = json.loads(state_path.read_text()) if state_path.exists() else {"issues": []}
        state["alerts"] = list(alerts)
        state_path.write_text(json.dumps(state))

    def state():
        return json.loads(state_path.read_text())

    def edit_issue(number, **changes):
        s = state()
        next(i for i in s["issues"] if i["number"] == number).update(changes)
        state_path.write_text(json.dumps(s))

    def run(today: str) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GH_STATE": str(state_path),
            "REPO": "o/r",
            "TRIAGE_ASSIGNEE": "owner",
            "TRIAGE_TODAY": today,
        }
        result = subprocess.run(
            ["bash", str(SCRIPT)], cwd=workdir, env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        return result

    set_alerts()
    return set_alerts, state, edit_issue, run


def test_first_run_opens_one_issue_keyed_to_the_iso_week(harness):
    set_alerts, state, _, run = harness
    set_alerts(_alert(1), _alert(2, "critical"), _alert(3, "medium"))

    run("2026-10-07")  # a Wednesday

    (issue,) = state()["issues"]
    assert issue["title"] == "CodeQL triage: week of 2026-10-05 (2 critical/high open)"
    assert "<!-- codeql-triage:2026-W41 -->" in issue["body"]
    assert "[#1]" in issue["body"] and "[#2]" in issue["body"]
    assert "[#3]" not in issue["body"]  # medium is out of scope


def test_rerun_same_day_creates_nothing(harness):
    set_alerts, state, _, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")
    result = run("2026-10-05")

    (issue,) = state()["issues"]
    assert issue["comments"] == []
    assert "Nothing to do" in result.stdout


def test_later_dispatch_reuses_a_closed_issue_and_keeps_its_ticks(harness):
    set_alerts, state, edit_issue, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")  # Monday's scheduled run

    ticked = state()["issues"][0]["body"].replace("- [ ]", "- [x]")
    edit_issue(1, body=ticked, state="closed")  # #1 fixed, box ticked
    set_alerts(_alert(7))  # a new high alert midweek

    run("2026-10-09")  # Friday's manual dispatch

    (issue,) = state()["issues"]
    assert issue["state"] == "closed"
    assert issue["body"] == ticked
    (comment,) = issue["comments"]
    assert "[#7]" in comment and "[#1]" not in comment

    run("2026-10-11")  # Sunday: same ISO week, nothing new since the comment
    (issue,) = state()["issues"]
    assert len(issue["comments"]) == 1  # #7 is not reported twice


def test_next_week_opens_a_new_issue(harness):
    set_alerts, state, _, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")
    run("2026-10-12")

    titles = [i["title"] for i in state()["issues"]]
    assert titles == [
        "CodeQL triage: week of 2026-10-05 (1 critical/high open)",
        "CodeQL triage: week of 2026-10-12 (1 critical/high open)",
    ]


def test_iso_year_boundary_shares_one_week(harness):
    set_alerts, state, _, run = harness
    set_alerts(_alert(1))
    run("2026-12-28")  # Monday of 2026-W53
    run("2027-01-01")  # Friday, same ISO week in a new calendar year

    (issue,) = state()["issues"]
    assert "<!-- codeql-triage:2026-W53 -->" in issue["body"]
    assert issue["title"].startswith("CodeQL triage: week of 2026-12-28")


def test_no_alerts_still_records_the_review(harness):
    _, state, _, run = harness
    run("2026-10-05")
    (issue,) = state()["issues"]
    assert "(0 critical/high open)" in issue["title"]
    assert "None. Close this issue to record the review." in issue["body"]


def test_an_alert_reopened_after_its_box_was_ticked_is_reported_again(harness):
    set_alerts, state, edit_issue, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")
    ticked = state()["issues"][0]["body"].replace("- [ ]", "- [x]")
    edit_issue(1, body=ticked, state="closed")  # dismissed and ticked

    # Reopened later the same week: open again, and its only mention is ticked.
    run("2026-10-08")
    (issue,) = state()["issues"]
    (comment,) = issue["comments"]
    assert "[#1]" in comment
    assert issue["body"] == ticked and issue["state"] == "closed"

    run("2026-10-09")  # listed unticked in that comment now: not repeated
    assert len(state()["issues"][0]["comments"]) == 1
