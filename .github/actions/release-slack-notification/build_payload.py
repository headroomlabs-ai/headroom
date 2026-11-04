"""Build the Slack payload for a completed GitHub Release asset publication."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import urlparse


def _nonempty(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _slack_text(value: str) -> str:
    """Escape characters Slack reserves in mrkdwn text."""
    return html.escape(value, quote=False)


def build_payload(
    *, release_name: str, tag: str, url: str, repository: str, actor: str
) -> dict[str, object]:
    tag = _nonempty(tag, "tag")
    url = _nonempty(url, "url")
    repository = _nonempty(repository, "repository")
    actor = _nonempty(actor, "actor")
    release_name = release_name.strip() or tag

    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.netloc != "github.com":
        raise ValueError("url must be an https://github.com release URL")

    header = f"{release_name} is ready"
    if len(header) > 150:
        raise ValueError("release name is too long for a Slack header block")

    return {
        "text": (
            f"{_slack_text(release_name)} ({_slack_text(tag)}) "
            f"GitHub Release assets are ready: {_slack_text(url)}"
        ),
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": header, "emoji": True},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Tag*\n`{_slack_text(tag)}`"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Repository*\n`{_slack_text(repository)}`",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "GitHub Release assets are ready. "
                        f"<{_slack_text(url)}|View release notes and downloads>"
                    ),
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"Published by `{_slack_text(actor)}` after the release asset job "
                            "completed."
                        ),
                    }
                ],
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_payload(
        release_name=args.release_name,
        tag=args.tag,
        url=args.url,
        repository=args.repository,
        actor=args.actor,
    )
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
