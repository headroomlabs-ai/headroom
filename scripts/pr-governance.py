#!/usr/bin/env python3
"""Validate Headroom PR template compliance for GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

COMMENT_MARKER = "<!-- headroom-pr-governance -->"
READY_LABEL = "status: ready for review"
AUTHOR_ACTION_LABEL = "status: needs author action"

REQUIRED_SECTIONS = (
    "Description",
    "Type of Change",
    "Changes Made",
    "Testing",
    "Real Behavior Proof",
    "Review Readiness",
)
PROOF_FIELDS = (
    "Environment",
    "Exact command / steps",
    "Observed result",
    "Not tested",
)

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
CHECKBOX_RE = re.compile(r"^- \[(?P<checked>[ xX])\] (?P<label>.+)$", re.MULTILINE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```(?:[\w.+-]+)?\n(?P<content>.*?)```", re.DOTALL)


@dataclass(slots=True)
class GovernanceReport:
    """Serializable PR governance result."""

    comment_marker: str
    valid: bool
    is_draft: bool
    is_bot_pr: bool
    ready_for_review: bool
    needs_author_action: bool
    problems: list[str] = field(default_factory=list)
    labels_to_add: list[str] = field(default_factory=list)
    labels_to_remove: list[str] = field(default_factory=list)
    comment_markdown: str = ""
    summary_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_event(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_sections(body: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[match.group(1).strip()] = body[start:end].strip()
    return sections


def strip_html_comments(text: str) -> str:
    return HTML_COMMENT_RE.sub("", text).strip()


def non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in strip_html_comments(text).splitlines() if line.strip()]


def checked_items(section: str) -> list[str]:
    return [
        match.group("label").strip()
        for match in CHECKBOX_RE.finditer(section)
        if match.group("checked").lower() == "x"
    ]


def has_descriptive_text(section: str) -> bool:
    ignored_prefixes = ("closes #", "fixes #", "resolves #", "related to #")
    for line in non_empty_lines(section):
        lowered = line.lower()
        if line.startswith("#"):
            continue
        if lowered.startswith(ignored_prefixes):
            continue
        if len(line) >= 10:
            return True
    return False


def has_non_placeholder_bullets(section: str) -> bool:
    placeholders = {"change 1", "change 2", "change 3"}
    for line in non_empty_lines(section):
        if not line.startswith("- "):
            continue
        bullet = line[2:].strip().lower()
        if bullet and bullet not in placeholders:
            return True
    return False


def has_test_output(section: str) -> bool:
    for match in CODE_BLOCK_RE.finditer(section):
        content = strip_html_comments(match.group("content")).strip()
        if not content:
            continue
        if "paste relevant command output or artifact links here" in content.lower():
            continue
        return True
    return False


def proof_field_values(section: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in non_empty_lines(section):
        if not line.startswith("- ") or ":" not in line:
            continue
        label, value = line[2:].split(":", 1)
        values[label.strip()] = value.strip()
    return values


def normalize_checkbox_map(items: list[str]) -> set[str]:
    return {item.lower() for item in items}


def validate_pull_request(event: dict[str, Any]) -> GovernanceReport:
    pull_request = event["pull_request"]
    author = pull_request["user"]["login"]
    is_draft = bool(pull_request.get("draft", False))
    is_bot_pr = author.endswith("[bot]")
    body = pull_request.get("body") or ""
    # Normalize Windows line endings so regex patterns expecting \n
    # (particularly the code-block fence regex) match correctly.
    body = body.replace("\r\n", "\n")

    if is_bot_pr:
        summary = "### PR governance\n\nBot-authored PR detected; template enforcement is skipped."
        return GovernanceReport(
            comment_marker=COMMENT_MARKER,
            valid=True,
            is_draft=is_draft,
            is_bot_pr=True,
            ready_for_review=False,
            needs_author_action=False,
            comment_markdown=summary,
            summary_markdown=summary,
        )

    sections = extract_sections(body)
    problems: list[str] = []

    for section_name in REQUIRED_SECTIONS:
        if section_name not in sections:
            problems.append(f"Missing required section `{section_name}`.")

    description = sections.get("Description", "")
    if description and not has_descriptive_text(description):
        problems.append("Fill in `Description` with a real summary of the change.")

    changes_made = sections.get("Changes Made", "")
    if changes_made and not has_non_placeholder_bullets(changes_made):
        problems.append(
            "Replace the placeholder bullets in `Changes Made` with the actual changes."
        )

    type_of_change_checked = checked_items(sections.get("Type of Change", ""))
    if sections.get("Type of Change") and not type_of_change_checked:
        problems.append("Check at least one box in `Type of Change`.")

    testing_section = sections.get("Testing", "")
    testing_checked = checked_items(testing_section)
    if testing_section and not testing_checked:
        problems.append("Check at least one verification item in `Testing`.")
    if testing_section and not has_test_output(testing_section):
        problems.append("Paste real command output or artifact links in `Testing` → `Test Output`.")

    proof_section = sections.get("Real Behavior Proof", "")
    proof_values = proof_field_values(proof_section)
    for field_name in PROOF_FIELDS:
        if proof_section and not proof_values.get(field_name):
            problems.append(f"Fill in `Real Behavior Proof` → `{field_name}`.")

    readiness_checked = normalize_checkbox_map(checked_items(sections.get("Review Readiness", "")))
    has_self_review = "i have performed a self-review" in readiness_checked
    has_ready_checkbox = "this pr is ready for human review" in readiness_checked
    if not is_draft:
        if not has_self_review:
            problems.append(
                "Check `I have performed a self-review` before requesting human review."
            )
        if not has_ready_checkbox:
            problems.append(
                "Check `This PR is ready for human review` or convert the PR back to draft."
            )

    valid = not problems
    ready_for_review = valid and not is_draft and has_ready_checkbox and has_self_review
    needs_author_action = not valid

    if valid and ready_for_review:
        status_lines = [
            "### PR governance",
            "",
            "This PR follows the template and is marked ready for human review.",
        ]
    elif valid:
        status_lines = [
            "### PR governance",
            "",
            "This draft PR follows the template so far. Keep it in draft until it is ready for human review.",
        ]
    else:
        status_lines = [
            "### PR governance",
            "",
            "This PR does not yet satisfy the required template fields:",
            "",
            *[f"- {problem}" for problem in problems],
            "",
            "Please update the PR body, or move the PR back to draft while it is still in progress.",
        ]

    labels_to_add: list[str] = []
    labels_to_remove: list[str] = []
    if needs_author_action:
        labels_to_add.append(AUTHOR_ACTION_LABEL)
        labels_to_remove.append(READY_LABEL)
    else:
        labels_to_remove.append(AUTHOR_ACTION_LABEL)
        if ready_for_review:
            labels_to_add.append(READY_LABEL)
        else:
            labels_to_remove.append(READY_LABEL)

    comment_markdown = "\n".join(status_lines)
    return GovernanceReport(
        comment_marker=COMMENT_MARKER,
        valid=valid,
        is_draft=is_draft,
        is_bot_pr=False,
        ready_for_review=ready_for_review,
        needs_author_action=needs_author_action,
        problems=problems,
        labels_to_add=labels_to_add,
        labels_to_remove=labels_to_remove,
        comment_markdown=comment_markdown,
        summary_markdown=comment_markdown,
    )


def validate_pull_request_body(event: dict[str, Any], body: str | None = None) -> GovernanceReport:
    """Validate a PR event, optionally replacing the event payload body.

    GitHub reruns use the original event payload. That makes a governance rerun
    keep validating an old PR body even after maintainers fix the live body.
    The workflow fetches the current body via the API and passes it here so the
    check reflects what reviewers see on the PR page.
    """
    if body is None:
        return validate_pull_request(event)

    event_copy = dict(event)
    pull_request = dict(event["pull_request"])
    pull_request["body"] = body
    event_copy["pull_request"] = pull_request
    return validate_pull_request(event_copy)


def emit_outputs(report: GovernanceReport) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    lines = [
        f"valid={str(report.valid).lower()}",
        f"ready_for_review={str(report.ready_for_review).lower()}",
        f"needs_author_action={str(report.needs_author_action).lower()}",
        f"is_bot_pr={str(report.is_bot_pr).lower()}",
    ]
    if not output_path:
        for line in lines:
            print(line)
        return

    with Path(output_path).open("a", encoding="utf-8") as output_file:
        for line in lines:
            output_file.write(f"{line}\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event", type=Path, required=True, help="Path to the GitHub event payload JSON."
    )
    parser.add_argument(
        "--body-file",
        type=Path,
        help=(
            "Optional file containing the current PR body. Use this in GitHub Actions "
            "so reruns validate the live PR body instead of the stale event payload."
        ),
    )
    parser.add_argument("--report", type=Path, required=True, help="Path to write the JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    body_override = (
        args.body_file.read_text(encoding="utf-8") if args.body_file is not None else None
    )
    report = validate_pull_request_body(load_event(args.event), body_override)
    args.report.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    emit_outputs(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
