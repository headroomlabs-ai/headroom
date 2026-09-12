"""Runtime helpers for CodeBuddy-facing integrations."""

from __future__ import annotations

DEFAULT_API_URL = "https://tencent.sso.copilot.tencent.com"


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by CodeBuddy integrations."""
    return f"http://127.0.0.1:{port}/v2"
