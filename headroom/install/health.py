"""Health helpers for persistent deployments."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def probe_json(url: str, timeout: float = 2.0) -> dict[str, Any] | None:
    """Return a JSON payload from the URL when reachable."""

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def probe_ready(url: str, timeout: float = 2.0) -> bool:
    """Return True when the ready endpoint reports readiness."""

    payload = probe_json(url, timeout=timeout)
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("ready", False) or payload.get("status") == "healthy")
