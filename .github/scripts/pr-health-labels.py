#!/usr/bin/env python3
"""Helpers for PR health maintenance labels."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

FAILING_STATES = {"FAILURE", "TIMED_OUT", "ACTION_REQUIRED", "CANCELLED", "ERROR"}


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _check_key(check: dict[str, Any]) -> tuple[str, str]:
    workflow = str(check.get("workflowName") or check.get("workflow") or "")
    name = str(check.get("name") or check.get("context") or "")
    return workflow, name


def _check_time(check: dict[str, Any]) -> datetime:
    return max(
        _parse_timestamp(check.get("startedAt")),
        _parse_timestamp(check.get("completedAt")),
    )


def _state(check: dict[str, Any]) -> str:
    return str(check.get("conclusion") or check.get("state") or "").upper()


def current_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for check in payload.get("statusCheckRollup") or []:
        if not isinstance(check, dict):
            continue
        key = _check_key(check)
        if not any(key):
            continue
        previous = latest_by_key.get(key)
        if previous is None or _check_time(check) >= _check_time(previous):
            latest_by_key[key] = check
    return list(latest_by_key.values())


def check_state(payload: dict[str, Any]) -> str:
    for check in current_checks(payload):
        if _state(check) in FAILING_STATES:
            return "failing"
    return "passing"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-json", required=True, help="JSON from gh pr view")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    print(check_state(json.loads(args.state_json)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
