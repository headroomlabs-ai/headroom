"""Runtime helpers for Grok Build CLI integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_API_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-build-0.1"
HEADROOM_MODEL_ALIAS = "headroom-grok-proxy"


def resolve_launch_model(requested_model: str | None) -> str:
    """Return the upstream model id the Headroom Grok alias should target."""

    candidate = (requested_model or "").strip()
    if candidate == HEADROOM_MODEL_ALIAS:
        return DEFAULT_MODEL
    return candidate or DEFAULT_MODEL


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Return the Grok launch environment.

    Grok's documented routing surface is ``~/.grok/config.toml`` custom models,
    not an env-only base-url override, so this helper is intentionally a no-op.
    """

    env = dict(environ or os.environ)
    _ = port, project
    return env, []
