"""scripts/codeql_triage.sh: one triage issue per ISO week, however often it runs.

The script runs for real against a fake ``gh`` on PATH that keeps alerts and
issues in a JSON file, so re-runs, later dispatches in the same week, a closed
issue with ticked boxes, and the ISO year boundary are all exercised end to end.
"""

from __future__ import annotations

import json
import os
import re
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
now = os.environ["TRIAGE_TODAY"] + "T12:00:00Z"
BOT = {"login": "github-actions[bot]"}

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
        if os.environ.get("FAKE_GH_FAIL_COMMENTS"):
            sys.exit("fake gh: HTTP 502")
        number = int(target.split("/")[-2])
        issue = next(i for i in state["issues"] if i["number"] == number)
        print(json.dumps(issue["comments"]))
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
        "user": BOT,
        "created_at": now,
        "comments": [],
    })
    save()
    print(f"https://github.com/o/r/issues/{number}")
elif args[:2] == ["issue", "comment"]:
    number = int(args[2])
    issue = next(i for i in state["issues"] if i["number"] == number)
    issue["comments"].append(
        {"body": open(opt("--body-file")).read(), "user": BOT, "created_at": now}
    )
    save()
else:
    sys.exit(f"fake gh: unsupported {args}")
'''


def _alert(
    number: int,
    severity: str = "high",
    created_at: str = "2026-10-01T00:00:00Z",
    updated_at: str | None = None,
) -> dict:
    return {
        "number": number,
        "html_url": f"https://github.com/o/r/security/code-scanning/{number}",
        "rule": {"id": f"py/rule-{number}", "security_severity_level": severity},
        "most_recent_instance": {"location": {"path": "headroom/x.py", "start_line": number}},
        "created_at": created_at,
        "updated_at": updated_at or created_at,
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

    def run(
        today: str, check: bool = True, fail_comments: bool = False
    ) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GH_STATE": str(state_path),
            "REPO": "o/r",
            "TRIAGE_ASSIGNEE": "owner",
            "TRIAGE_TODAY": today,
        }
        if fail_comments:
            env["FAKE_GH_FAIL_COMMENTS"] = "1"
        result = subprocess.run(
            ["bash", str(SCRIPT)], cwd=workdir, env=env, capture_output=True, text=True
        )
        if check:
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
    assert "Critical: 1, high: 1." in issue["body"]  # medium is out of scope
    assert "https://github.com/o/r/security/code-scanning" in issue["body"]


def test_public_issue_and_comments_carry_only_counts(harness):
    set_alerts, state, _, run = harness
    set_alerts(_alert(11, "critical"), _alert(12))
    run("2026-10-05")
    set_alerts(_alert(11, "critical"), _alert(12), _alert(13))
    run("2026-10-06")  # a new alert midweek goes in a comment

    (issue,) = state()["issues"]
    (comment,) = issue["comments"]
    public_text = issue["title"] + issue["body"] + comment["body"]
    assert "py/rule-" not in public_text
    assert "headroom/x.py" not in public_text
    for number in (11, 12, 13):
        assert f"#{number}" not in public_text
        assert f"code-scanning/{number}" not in public_text
    assert "critical 1, high 2" in comment["body"]
    # The hidden state markers hold counts only, nothing derived from alert ids.
    markers = re.findall(r"<!-- codeql-triage-state:(.*?) -->", public_text)
    assert markers == ["c1-h1", "c1-h2"]


def test_rerun_same_day_creates_nothing(harness):
    set_alerts, state, _, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")
    result = run("2026-10-05")

    (issue,) = state()["issues"]
    assert issue["comments"] == []
    assert "Nothing to do" in result.stdout


def test_later_dispatch_reuses_a_closed_issue(harness):
    set_alerts, state, edit_issue, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")  # Monday's scheduled run

    body = state()["issues"][0]["body"]
    edit_issue(1, state="closed")  # #1 fixed, issue closed
    set_alerts(_alert(7, created_at="2026-10-08T09:00:00Z"))  # new midweek, same counts

    run("2026-10-09")  # Friday's manual dispatch

    (issue,) = state()["issues"]
    assert issue["state"] == "closed"
    assert issue["body"] == body
    (comment,) = issue["comments"]
    assert "critical 0, high 1" in comment["body"]

    run("2026-10-11")  # Sunday: same ISO week, nothing new since the comment
    (issue,) = state()["issues"]
    assert len(issue["comments"]) == 1  # not reported twice


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


def test_an_alert_reopened_after_it_was_closed_is_reported_again(harness):
    set_alerts, state, edit_issue, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")
    set_alerts()  # dismissed
    run("2026-10-06")
    edit_issue(1, state="closed")

    set_alerts(_alert(1))  # reopened later the same week
    run("2026-10-08")
    (issue,) = state()["issues"]
    assert len(issue["comments"]) == 2
    assert "critical 0, high 1" in issue["comments"][-1]["body"]
    assert issue["state"] == "closed"

    run("2026-10-09")  # unchanged since that comment: not repeated
    assert len(state()["issues"][0]["comments"]) == 2


def test_a_marker_in_someone_elses_comment_is_ignored(harness):
    set_alerts, state, edit_issue, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")
    set_alerts()
    run("2026-10-06")  # bot records c0-h0
    issue = state()["issues"][0]
    forged = {
        "body": "<!-- codeql-triage-state:c0-h1 -->",
        "user": {"login": "someone"},
        "created_at": "2026-10-07T00:00:00Z",
    }
    edit_issue(1, comments=issue["comments"] + [forged])

    set_alerts(_alert(1))  # reopened: matches only the forged marker
    run("2026-10-08")
    bodies = [c["body"] for c in state()["issues"][0]["comments"]]
    assert len(bodies) == 3
    assert "critical 0, high 1" in bodies[-1]


def test_a_failed_comment_read_stops_the_run(harness):
    set_alerts, state, _, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")
    set_alerts(_alert(1), _alert(2, created_at="2026-10-06T09:00:00Z"))

    result = run("2026-10-06", check=False, fail_comments=True)

    assert result.returncode != 0
    assert state()["issues"][0]["comments"] == []


def test_one_alert_closing_while_another_reopens_is_reported(harness):
    set_alerts, state, _, run = harness
    set_alerts(_alert(1))
    run("2026-10-05")  # records c0-h1

    # #1 fixed, older #2 reopened: same counts, nothing newly created.
    set_alerts(_alert(2, created_at="2026-09-01T00:00:00Z", updated_at="2026-10-06T08:00:00Z"))
    run("2026-10-06")

    (comment,) = state()["issues"][0]["comments"]
    assert "critical 0, high 1" in comment["body"]


def test_an_issue_opened_by_someone_else_is_not_trusted(harness):
    set_alerts, state, edit_issue, run = harness
    set_alerts(_alert(1))
    run("2026-09-28")  # issue 1, an earlier week
    # Turn it into a lookalike for this week, written by someone else, whose
    # recorded counts match the open alerts.
    edit_issue(
        1,
        user={"login": "someone"},
        body="<!-- codeql-triage:2026-W41 -->\n<!-- codeql-triage-state:c0-h1 -->",
        created_at="2026-10-05T13:00:00Z",
    )

    run("2026-10-06")

    issues = state()["issues"]
    assert issues[0]["comments"] == []
    assert len(issues) == 2  # the bot opened its own issue for the week
    assert issues[1]["user"]["login"] == "github-actions[bot]"
    assert "<!-- codeql-triage:2026-W41 -->" in issues[1]["body"]
