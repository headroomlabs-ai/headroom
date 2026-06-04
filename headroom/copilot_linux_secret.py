"""Linux secret-service lookup helpers for GitHub Copilot CLI auth."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def read_copilot_oauth_token(*, host: str = "github.com") -> str | None:
    """Return a Copilot CLI OAuth token from Linux Secret Service, if available."""

    if not sys.platform.startswith("linux"):
        return None

    secret_tool = os.environ.get("GITHUB_COPILOT_SECRET_TOOL", "secret-tool").strip()
    if not secret_tool:
        return None

    normalized_host = host.strip().lower() or "github.com"
    login = _read_copilot_config_login()
    for command in _candidate_secret_tool_commands(secret_tool, normalized_host, login):
        token = _run_secret_tool_lookup(command)
        if token:
            return token

    return None


def _read_copilot_config_login() -> str | None:
    path = Path(os.environ.get("COPILOT_HOME", str(Path.home() / ".copilot"))) / "config.json"
    try:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("//")
        ]
        payload = json.loads("\n".join(lines))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    user = payload.get("lastLoggedInUser")
    if not isinstance(user, dict):
        return None
    login = user.get("login")
    return login.strip() if isinstance(login, str) and login.strip() else None


def _candidate_secret_tool_commands(
    secret_tool: str,
    host: str,
    login: str | None,
) -> list[list[str]]:
    commands: list[list[str]] = []
    accounts = [
        value
        for value in (
            f"https://{host}:{login}" if login else None,
            f"{host}:{login}" if login else None,
            login,
            f"https://{host}",
            host,
        )
        if value
    ]
    service_names = ("copilot-cli", "GitHub Copilot CLI", "github-copilot", "copilot")

    for service in service_names:
        commands.append([secret_tool, "lookup", "service", service])
        commands.append([secret_tool, "lookup", "application", service])
        for account in accounts:
            commands.append([secret_tool, "lookup", "service", service, "account", account])
            commands.append([secret_tool, "lookup", "application", service, "account", account])
            commands.append([secret_tool, "lookup", "service", service, "username", account])

    return commands


def _run_secret_tool_lookup(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5,
        )
    except OSError as exc:
        logger.debug("Unable to invoke secret-tool for Copilot auth discovery: %s", exc)
        return None
    except subprocess.TimeoutExpired:
        logger.debug("secret-tool lookup timed out for Copilot auth")
        return None

    if result.returncode != 0:
        return None
    token = result.stdout.strip()
    return token or None
