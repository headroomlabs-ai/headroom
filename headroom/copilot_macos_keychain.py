"""macOS Keychain lookup helpers for GitHub Copilot CLI auth."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def read_copilot_oauth_token(*, host: str = "github.com") -> str | None:
    """Return a Copilot CLI OAuth token from macOS Keychain, if available."""

    if sys.platform != "darwin":
        return None

    normalized_host = host.strip().lower() or "github.com"
    login = _read_copilot_config_login()
    services = _split_env_list("GITHUB_COPILOT_KEYCHAIN_SERVICE") or [
        "GitHub Copilot",
        "GitHub Copilot CLI",
        "github-copilot",
        "copilot",
        "copilot-cli",
        "GitHub CLI",
        "github.com",
        f"https://{normalized_host}",
        normalized_host,
    ]
    accounts = _split_env_list("GITHUB_COPILOT_KEYCHAIN_ACCOUNT") or [
        value
        for value in (
            f"https://{normalized_host}:{login}" if login else None,
            f"{normalized_host}:{login}" if login else None,
            login,
            os.environ.get("USER"),
            os.environ.get("USERNAME"),
            normalized_host,
            f"https://{normalized_host}",
        )
        if value
    ]

    for command in _candidate_security_commands(normalized_host, services, accounts):
        token = _run_security_lookup(command)
        if token:
            return token

    return None


def _read_copilot_config_login() -> str | None:
    """Return the last logged-in Copilot CLI username from ~/.copilot/config.json."""

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


def _split_env_list(name: str) -> list[str]:
    return [part.strip() for part in os.environ.get(name, "").split(",") if part.strip()]


def _candidate_security_commands(
    host: str,
    services: list[str],
    accounts: list[str],
) -> list[list[str]]:
    commands: list[list[str]] = []
    for service in services:
        commands.append(["security", "find-generic-password", "-s", service, "-w"])
        for account in accounts:
            commands.append(
                ["security", "find-generic-password", "-s", service, "-a", account, "-w"]
            )
    for server in (host, f"https://{host}"):
        commands.append(["security", "find-internet-password", "-s", server, "-w"])
        for account in accounts:
            commands.append(
                ["security", "find-internet-password", "-s", server, "-a", account, "-w"]
            )
    return commands


def _run_security_lookup(command: list[str]) -> str | None:
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
        logger.debug("Unable to invoke macOS Keychain lookup for Copilot auth: %s", exc)
        return None
    except subprocess.TimeoutExpired:
        logger.debug("macOS Keychain lookup timed out for Copilot auth")
        return None

    if result.returncode != 0:
        return None
    token = result.stdout.strip()
    return token or None
