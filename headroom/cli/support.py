"""Support diagnostics CLI."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from headroom import paths

from .main import main

_SECRET_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"), r"\1<redacted>"),
    (
        re.compile(r"(?i)(api[_-]?key|token|secret|password)([\"']?\s*[:=]\s*[\"']?)[^\"'\s,;}]+"),
        r"\1\2<redacted>",
    ),
    (re.compile(r"\b(sk-(?:ant-|proj-)?[A-Za-z0-9._-]{8,})\b"), "<redacted>"),
    (re.compile(r"\b(ghp_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,})\b"), "<redacted>"),
)
_DIAGNOSTIC_MARKERS = (
    " PERF ",
    "content_router:",
    "Transform ",
    "Pipeline complete:",
    "TOIN:",
)
_ENV_KEYS = (
    "HEADROOM_WORKSPACE_DIR",
    "HEADROOM_CONFIG_DIR",
    "HEADROOM_SAVINGS_PATH",
    "HEADROOM_TOIN_PATH",
    "HEADROOM_SUBSCRIPTION_STATE_PATH",
    "HEADROOM_HOST",
    "HEADROOM_PORT",
    "HEADROOM_MODE",
    "HEADROOM_BACKEND",
    "HEADROOM_LOG_LEVEL",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def _default_output_path() -> Path:
    return Path.cwd() / f"headroom-support-{_utc_stamp()}.zip"


def _redact_line(line: str) -> str:
    redacted = line
    for pattern, replacement in _SECRET_REPLACEMENTS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _write_json(zf: zipfile.ZipFile, arcname: str, payload: Any) -> None:
    zf.writestr(
        arcname,
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
    )


def _write_text(zf: zipfile.ZipFile, arcname: str, payload: str) -> None:
    zf.writestr(arcname, payload if payload.endswith("\n") else payload + "\n")


def _file_info(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _log_files() -> list[Path]:
    log_dir = paths.log_dir()
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob("proxy.log*"), key=lambda p: p.stat().st_mtime)


def _tail_log_lines(*, max_lines: int, diagnostic_only: bool) -> list[str]:
    kept: deque[str] = deque(maxlen=max_lines)
    for log_file in _log_files():
        try:
            with log_file.open(encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    line = raw.rstrip("\r\n")
                    if diagnostic_only and not any(
                        marker in line for marker in _DIAGNOSTIC_MARKERS
                    ):
                        continue
                    kept.append(_redact_line(line))
        except OSError:
            continue
    return list(kept)


def _write_file_if_present(
    zf: zipfile.ZipFile,
    path: Path,
    arcname: str,
    *,
    max_bytes: int = 2_000_000,
) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) > max_bytes:
        _write_text(
            zf,
            f"{arcname}.omitted.txt",
            f"{path} is {len(data)} bytes; omitted because it exceeds {max_bytes} bytes.",
        )
        return False
    zf.writestr(arcname, data)
    return True


def _perf_report(hours: float) -> str:
    from headroom.perf import analyzer

    original_log_dir = analyzer.LOG_DIR
    analyzer.LOG_DIR = paths.log_dir()
    try:
        return analyzer.format_report(analyzer.parse_log_files(last_n_hours=hours))
    finally:
        analyzer.LOG_DIR = original_log_dir


def _manifest(*, hours: float, max_lines: int, include_full_log_tail: bool) -> dict[str, Any]:
    try:
        from headroom._version import __version__
    except ImportError:
        __version__ = "unknown"

    selected_env = {
        key: _redact_line(value) for key in _ENV_KEYS if (value := os.environ.get(key, "").strip())
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "headroom_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "options": {
            "hours": hours,
            "max_lines": max_lines,
            "include_full_log_tail": include_full_log_tail,
        },
        "paths": {
            "workspace_dir": str(paths.workspace_dir()),
            "config_dir": str(paths.config_dir()),
            "log_dir": str(paths.log_dir()),
            "proxy_log": _file_info(paths.proxy_log_path()),
            "savings": _file_info(paths.savings_path()),
            "session_stats": _file_info(paths.session_stats_path()),
        },
        "selected_environment": selected_env,
        "log_files": [_file_info(path) for path in _log_files()],
    }


def create_support_bundle(
    *,
    output: Path | None = None,
    hours: float = 168.0,
    max_lines: int = 2000,
    include_full_log_tail: bool = False,
) -> Path:
    """Create a local diagnostics zip for issue reports."""

    output_path = (output or _default_output_path()).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_json(
            zf,
            "manifest.json",
            _manifest(
                hours=hours, max_lines=max_lines, include_full_log_tail=include_full_log_tail
            ),
        )
        _write_text(zf, "perf-report.txt", _perf_report(hours))
        _write_file_if_present(zf, paths.savings_path(), "state/proxy_savings.json")

        session_lines = _tail_file(paths.session_stats_path(), max_lines=max_lines)
        if session_lines:
            _write_text(zf, "state/session_stats_tail.jsonl", "\n".join(session_lines))

        diagnostic_lines = _tail_log_lines(max_lines=max_lines, diagnostic_only=True)
        if diagnostic_lines:
            _write_text(zf, "logs/proxy-diagnostics-tail.txt", "\n".join(diagnostic_lines))

        if include_full_log_tail:
            full_lines = _tail_log_lines(max_lines=max_lines, diagnostic_only=False)
            if full_lines:
                _write_text(zf, "logs/proxy-full-tail.redacted.txt", "\n".join(full_lines))
        else:
            _write_text(
                zf,
                "logs/full-log-tail-not-included.txt",
                "Full proxy log tail is not included by default. Re-run with "
                "`headroom support bundle --include-full-log-tail` if a maintainer asks for it.",
            )

    return output_path


def _tail_file(path: Path, *, max_lines: int) -> list[str]:
    kept: deque[str] = deque(maxlen=max_lines)
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                kept.append(_redact_line(raw.rstrip("\r\n")))
    except OSError:
        return []
    return list(kept)


@main.group("support")
def support_group() -> None:
    """Collect diagnostics for maintainers."""


@support_group.command("bundle")
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    default=None,
    help="Where to write the zip file. Defaults to ./headroom-support-<timestamp>.zip.",
)
@click.option(
    "--hours",
    type=float,
    default=168.0,
    show_default=True,
    help="Performance-report window to include.",
)
@click.option(
    "--max-lines",
    type=click.IntRange(1, 50_000),
    default=2000,
    show_default=True,
    help="Maximum tail lines to include per bundled text artifact.",
)
@click.option(
    "--include-full-log-tail",
    is_flag=True,
    help="Also include a redacted tail of proxy.log. Off by default because logs can contain prompts.",
)
def bundle(
    output_path: Path | None, hours: float, max_lines: int, include_full_log_tail: bool
) -> None:
    """Create a local zip with logs and stats for an issue report."""

    bundle_path = create_support_bundle(
        output=output_path,
        hours=hours,
        max_lines=max_lines,
        include_full_log_tail=include_full_log_tail,
    )
    click.echo(f"Created support bundle: {bundle_path}")
    if not include_full_log_tail:
        click.echo("Full proxy logs were not included. Use --include-full-log-tail only if needed.")
