"""Transparent VS Code Copilot proxy configuration helpers."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import click

from headroom import fsutil
from headroom.proxy.project_context import with_project_prefix

_MARKER_START = "// --- Headroom Copilot proxy ---"
_MARKER_END = "// --- end Headroom Copilot proxy ---"
_PROXY_KEY = "github.copilot.advanced.debug.overrideProxyUrl"
_AUTH_KEY = "github.copilot.advanced.debug.overrideAuthType"


def _read_settings(path: Path) -> str:
    """Read UTF-8 settings without normalizing Windows line endings or a BOM."""
    with path.open(encoding="utf-8", newline="") as settings_file:
        return settings_file.read()


def vscode_user_dir(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    """Return stable VS Code's user-data directory on macOS, Windows, or Linux."""
    env = environ if environ is not None else os.environ
    current_platform = platform or sys.platform
    home = Path(env.get("HOME") or env.get("USERPROFILE") or Path.home())
    if current_platform == "darwin":
        return home / "Library" / "Application Support" / "Code" / "User"
    if current_platform == "win32":
        appdata = env.get("APPDATA")
        if not appdata:
            raise click.ClickException("APPDATA is not set; pass --settings-file explicitly.")
        return Path(appdata) / "Code" / "User"
    return Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "Code" / "User"


def vscode_settings_path(
    *, platform: str | None = None, environ: Mapping[str, str] | None = None
) -> Path:
    return vscode_user_dir(platform=platform, environ=environ) / "settings.json"


def vscode_proxy_url(port: int, project: str | None = None) -> str:
    """Return the transparent Copilot endpoint override for a Headroom proxy."""
    return str(with_project_prefix(f"http://127.0.0.1:{port}", project))


def _strip_jsonc_comments(value: str) -> str:
    """Strip JSONC comments while respecting quoted strings."""
    result: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(value):
        char = value[index]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if value.startswith("//", index):
            newline = value.find("\n", index)
            if newline < 0:
                break
            result.append("\n")
            index = newline + 1
            continue
        if value.startswith("/*", index):
            end = value.find("*/", index + 2)
            if end < 0:
                raise click.ClickException(
                    "VS Code settings contain an unterminated block comment."
                )
            result.extend("\n" for char in value[index : end + 2] if char == "\n")
            index = end + 2
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _validate_settings(raw: str, path: Path) -> None:
    candidate = _strip_jsonc_comments(raw)
    if candidate.startswith("\ufeff"):
        candidate = candidate[1:]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Could not safely parse {path}: {exc}. Headroom did not overwrite it."
        ) from exc
    if not isinstance(parsed, dict):
        raise click.ClickException(f"{path} must contain a JSON object; refusing to overwrite it.")


def _managed_block(proxy_url: str, *, owns_preceding_comma: bool, line_sep: str) -> str:
    marker = _MARKER_START + (" (comma-added)" if owns_preceding_comma else "")
    return (
        f"\t{marker}{line_sep}"
        f"\t{json.dumps(_PROXY_KEY)}: {json.dumps(proxy_url)},{line_sep}"
        f'\t{json.dumps(_AUTH_KEY)}: "token"{line_sep}'
        f"\t{_MARKER_END}"
    )


def remove_vscode_proxy_settings(path: Path) -> bool:
    """Remove Headroom's marker-owned settings without reformatting user content."""
    if not path.exists():
        return False
    raw = _read_settings(path)
    start_count = raw.count(_MARKER_START)
    end_count = raw.count(_MARKER_END)
    start = raw.find(_MARKER_START)
    end = raw.find(_MARKER_END)
    if start < 0 and end < 0:
        return False
    if start_count != 1 or end_count != 1 or end < start:
        raise click.ClickException(f"Incomplete Headroom marker block in {path}; refusing to edit.")
    line_start = raw.rfind("\n", 0, start) + 1
    line_end = raw.find("\n", end)
    line_end = len(raw) if line_end < 0 else line_end + 1
    prefix = raw[:line_start]
    suffix = raw[line_end:]
    # The installer owns a preceding comma only when it had to add one.
    marker_line = raw[line_start : raw.find("\n", start)]
    trimmed = prefix.rstrip()
    if "(comma-added)" in marker_line and trimmed.endswith(","):
        prefix = trimmed[:-1] + prefix[len(trimmed) :]
    updated = prefix + suffix
    _validate_settings(updated, path)
    fsutil.write_text(path, updated)
    return True


def configure_vscode_proxy_settings(path: Path, proxy_url: str) -> str:
    """Add/update transparent routing while preserving all non-Headroom text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _read_settings(path) if path.exists() else "{}\n"
    _validate_settings(raw, path)
    had_managed_block = _MARKER_START in raw or _MARKER_END in raw
    if had_managed_block:
        remove_vscode_proxy_settings(path)
        raw = _read_settings(path)
    elif _PROXY_KEY in raw or _AUTH_KEY in raw:
        raise click.ClickException(
            f"{path} already configures a Copilot endpoint override outside Headroom's "
            "managed block; refusing to replace it. Remove it or use --no-configure."
        )

    close = raw.rfind("}")
    if close < 0:
        raise click.ClickException(f"Could not locate the root object in {path}.")
    before = raw[:close].rstrip()
    after = raw[close:]
    inner = _strip_jsonc_comments(before).rstrip()
    needs_comma = not inner.endswith("{") and not inner.endswith(",")
    separator = "," if needs_comma else ""
    line_sep = "\r\n" if "\r\n" in raw else "\n"
    newline = "" if before.endswith(("\n", "\r")) else line_sep
    updated = (
        before
        + separator
        + newline
        + _managed_block(proxy_url, owns_preceding_comma=needs_comma, line_sep=line_sep)
        + line_sep
        + after
    )
    _validate_settings(updated, path)
    fsutil.write_text(path, updated)
    return "updated" if had_managed_block else "added"
