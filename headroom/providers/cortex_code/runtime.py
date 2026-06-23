"""Runtime helpers for Cortex Code (CoCo) integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping

from headroom.proxy.project_context import with_project_prefix

SNOWFLAKE_ACCOUNT_ENV = "SNOWFLAKE_ACCOUNT"
SNOWFLAKE_HOST_ENV = "SNOWFLAKE_HOST"
_FALLBACK_API_URL = "https://app.snowflake.com"


def default_api_url(environ: Mapping[str, str] | None = None) -> str:
    """Return the upstream Snowflake Cortex API URL.

    Reads SNOWFLAKE_HOST first, then SNOWFLAKE_ACCOUNT, and constructs
    a ``https://<host>.snowflakecomputing.com`` base URL.  Falls back to
    ``https://app.snowflake.com`` when neither variable is set.
    """
    env = environ or os.environ
    host = env.get(SNOWFLAKE_HOST_ENV) or env.get(SNOWFLAKE_ACCOUNT_ENV, "")
    if host:
        if host.startswith("https://"):
            return host
        if ".snowflakecomputing.com" in host:
            return f"https://{host}"
        return f"https://{host}.snowflakecomputing.com"
    return _FALLBACK_API_URL


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL for OpenAI-compatible Cortex requests."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build the environment variables that redirect Cortex Code through the proxy.

    Returns a ``(env_dict, printed_lines)`` tuple.  ``env_dict`` is a copy of
    *environ* with ``OPENAI_BASE_URL`` set to the local proxy endpoint.
    ``printed_lines`` is the ``KEY=VALUE`` form shown to the user on launch.
    """
    env = dict(environ or os.environ)
    base_url = with_project_prefix(proxy_base_url(port), project)
    env["OPENAI_BASE_URL"] = base_url
    return env, [f"OPENAI_BASE_URL={base_url}"]
