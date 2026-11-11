"""Repair newline-damaged merged release-please bodies before release detection.

Windows text-mode writes can turn existing CRLF into CRCRLF. release-please
normalizes CRLF once and then looks for an exact ``---`` line, so the remaining
CR hides its envelope and a successful run silently skips the pending release.
This tool only repairs merged, pending release PRs on the configured base.
It never changes labels, tags, releases, release notes, or package versions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from typing import Any
from urllib.parse import urlencode


def github_json(endpoint: str, **payload: Any) -> Any:
    command = ["gh", "api", endpoint]
    if payload:
        command.extend(["--method", "PATCH", "--input", "-"])
    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


def normalize_body(body: str) -> str:
    """Collapse CR runs before LF without altering note content or blank lines."""
    return re.sub(r"\r+\n", "\n", body)


def repair_release(repo: str, pr: dict[str, Any], *, apply: bool = True) -> bool:
    if (
        not pr.get("merged_at")
        or pr.get("base", {}).get("ref") != "main"
        or pr.get("user", {}).get("type") != "Bot"
        or (pr.get("head", {}).get("repo") or {}).get("full_name") != repo
        or not pr.get("head", {}).get("ref", "").startswith("release-please--branches--main--")
        or "autorelease: pending" not in {label["name"] for label in pr.get("labels", [])}
    ):
        return False
    body = pr.get("body") or ""
    # Normal CRLF already works. Do not rewrite unrelated historical formats.
    if not re.search(r"\r{2,}\n", body):
        return False
    normalized = normalize_body(body)
    # Match the structural boundary release-please needs, not an arbitrary body
    # containing a version somewhere in its maintainer comments.
    lines = normalized.strip().split("\n")
    delimiters = [i for i, line in enumerate(lines) if line == "---"]
    if len(delimiters) < 2 or not re.search(
        r"^#{2,} \[?\d+\.\d+\.\d+", "\n".join(lines[delimiters[0] + 1 : delimiters[-1]]), re.M
    ):
        raise ValueError(f"PR #{pr['number']} lacks a parseable release-please envelope")
    if body == normalized:
        return False
    if apply:
        github_json(f"repos/{repo}/pulls/{pr['number']}", body=normalized)
    print(f"{'Normalized' if apply else 'Would normalize'} release PR #{pr['number']}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), required=False)
    parser.add_argument("--apply", action="store_true", help="Write only normalized PR body text")
    args = parser.parse_args()
    if not args.repo or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("--repo must be owner/repository")
    page = 1
    while True:
        query = urlencode(
            {"state": "closed", "labels": "autorelease: pending", "per_page": 100, "page": page}
        )
        issues = github_json(f"repos/{args.repo}/issues?{query}")
        for issue in issues:
            if "pull_request" in issue:
                pr = github_json(f"repos/{args.repo}/pulls/{issue['number']}")
                repair_release(args.repo, pr, apply=args.apply)
        if len(issues) < 100:
            return
        page += 1


if __name__ == "__main__":
    main()
