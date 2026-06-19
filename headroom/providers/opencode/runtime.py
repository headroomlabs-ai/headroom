"""Runtime helpers for OpenCode (anomalyco/opencode) integrations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from headroom.proxy.savings_tracker import sanitize_project_name


def proxy_base_url(port: int) -> str:
    # Both @ai-sdk/openai and @ai-sdk/anthropic append the bare resource path
    # ('/chat/completions', '/messages') to a baseURL that MUST end in /v1.
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    env = dict(environ or os.environ)
    base_url = proxy_base_url(port)
    options: dict[str, Any] = {"baseURL": base_url}
    name = sanitize_project_name(project)
    if name:
        options = {**options, "headers": {"X-Headroom-Project": name}}
    config = {
        "provider": {"openai": {"options": options}, "anthropic": {"options": options}},
        "autoupdate": False,
    }
    content = json.dumps(config)
    env["OPENCODE_CONFIG_CONTENT"] = content
    return env, [f"OPENCODE_CONFIG_CONTENT={content}"]
