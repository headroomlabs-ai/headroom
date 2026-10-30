"""Runtime helpers for DeepSeek Harness (dsh) integrations."""

from __future__ import annotations

import os
import shlex
import shutil
from collections.abc import Mapping

DEFAULT_API_URL = "https://api.deepseek.com"


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenAI-compatible integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for dsh through the local proxy.

    Sets ``DEEPSEEK_BASE_URL`` to the proxy and leaves ``DEEPSEEK_API_KEY``
    (and every other variable) as-is so dsh resolves its own key and the proxy
    forwards the bearer token verbatim.
    """
    env = dict(environ or os.environ)
    base_url = proxy_base_url(port)
    env["DEEPSEEK_BASE_URL"] = base_url
    return env, [f"DEEPSEEK_BASE_URL={base_url}"]


def resolve_dsh_command(
    *,
    profile: str = "web",
    command: str | None = None,
    task_args: tuple[str, ...] = (),
) -> list[str]:
    """Resolve the dsh launch argv.

    Precedence for the launcher: explicit ``--command``, then ``dsh`` on ``PATH``,
    then ``pnpm dsh``. ``profile`` is ``"web"`` (default) or ``"headless"``.
    """
    if command:
        argv = shlex.split(command)
    else:
        dsh_bin = shutil.which("dsh")
        if dsh_bin:
            argv = [dsh_bin]
        elif shutil.which("pnpm"):
            argv = ["pnpm", "dsh"]
        else:
            raise RuntimeError(
                "dsh was not found in PATH. Install it with `npm i -g @deepseek-ai/dsh` "
                "or pass `--command`."
            )
    if profile == "headless":
        argv.extend(["--profile", "headless", *task_args])
    else:
        argv.extend(["web", *task_args])
    return argv
