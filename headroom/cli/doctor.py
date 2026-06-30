"""`headroom doctor` — diagnose whether the local Headroom setup is working.

Headroom's failure mode is silent: when a client is not routed through the
proxy (or the proxy runs stale code), everything still works — you just
stop saving tokens. This command correlates the state nothing else
reconciles: the proxy process, per-client wrap configs, the current shell
environment, savings flow, and budget configuration.

Exit codes: 0 = all checks pass, 1 = warnings only, 2 = any failure.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from headroom.install.health import probe_json
from headroom.install.paths import claude_settings_path, codex_config_path
from headroom.install.state import list_manifests
from headroom.paths import savings_path
from headroom.providers.claude import (
    REMOTE_CONTROL_BASE_URL_ENV,
    is_custom_anthropic_base_url,
    remote_control_gate_message,
)

from .main import get_version, main

PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

_LOOPBACK_URL_RE = re.compile(r"https?://(?:127\.0\.0\.1|localhost):(\d+)")
_CODEX_BASE_URL_RE = re.compile(r'base_url\s*=\s*"https?://(?:127\.0\.0\.1|localhost):(\d+)')


@dataclass
class CheckResult:
    """One diagnostic outcome."""

    name: str
    status: str  # pass | warn | fail | skip
    summary: str
    hint: str | None = None


def _format_uptime(seconds: float) -> str:
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _format_since(iso_ts: str) -> str | None:
    try:
        then = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    delta = datetime.now(then.tzinfo) - then
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def check_proxy_liveness(livez: dict[str, Any] | None, base_url: str) -> CheckResult:
    """Is the proxy process up and answering /livez?"""
    if livez is None:
        return CheckResult(
            name="proxy",
            status=FAIL,
            summary=f"not reachable at {base_url}",
            hint="start it with: headroom proxy",
        )
    version = livez.get("version", "unknown")
    uptime = livez.get("uptime_seconds")
    uptime_text = f"up {_format_uptime(uptime)}" if isinstance(uptime, (int, float)) else "up"
    return CheckResult(
        name="proxy",
        status=PASS,
        summary=f"running at {base_url} ({uptime_text}, v{version})",
    )


def check_version_drift(livez: dict[str, Any] | None, installed: str) -> CheckResult:
    """Does the running proxy match the installed package version?"""
    if livez is None:
        return CheckResult(name="version", status=SKIP, summary="proxy not reachable")
    running = str(livez.get("version") or "unknown")
    if "unknown" in (running, installed):
        return CheckResult(
            name="version",
            status=WARN,
            summary=f"cannot compare versions (proxy {running}, installed {installed})",
        )
    if running != installed:
        return CheckResult(
            name="version",
            status=WARN,
            summary=f"version drift: proxy {running}, installed {installed}",
            hint="restart the proxy to pick up new code: headroom proxy",
        )
    return CheckResult(name="version", status=PASS, summary=f"proxy matches installed v{installed}")


def check_claude_routing(settings_path: Path, port: int) -> CheckResult:
    """Is Claude Code configured to route through the proxy?"""
    name = "claude"
    if not settings_path.exists():
        return CheckResult(
            name=name,
            status=WARN,
            summary="not routed (no ~/.claude/settings.json)",
            hint="wrap it: headroom wrap claude",
        )
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return CheckResult(
            name=name,
            status=WARN,
            summary=f"could not parse {settings_path}: {exc}",
        )
    base_url = ""
    env_block = payload.get("env")
    if isinstance(env_block, dict):
        base_url = str(env_block.get("ANTHROPIC_BASE_URL", "") or "")
    if not base_url:
        return CheckResult(
            name=name,
            status=WARN,
            summary="not routed (no ANTHROPIC_BASE_URL in settings env)",
            hint="wrap it: headroom wrap claude",
        )
    return _classify_routing_url(name, base_url, port, source=str(settings_path))


def check_claude_remote_control_gate(
    settings_path: Path, environ: Mapping[str, str]
) -> CheckResult | None:
    """Warn once when Claude custom-base routing hides Remote Control."""
    name = "claude remote control"
    settings_base_url = ""
    if settings_path.exists():
        try:
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
            env_block = payload.get("env")
            if isinstance(env_block, dict):
                settings_base_url = str(env_block.get("ANTHROPIC_BASE_URL", "") or "")
        except (OSError, ValueError):
            settings_base_url = ""
    if is_custom_anthropic_base_url(settings_base_url):
        remote_message = remote_control_gate_message(f"{REMOTE_CONTROL_BASE_URL_ENV} from settings")
        return CheckResult(
            name=name,
            status=WARN,
            summary=remote_message,
            hint=remote_message,
        )

    env_base_url = environ.get("ANTHROPIC_BASE_URL", "")
    if is_custom_anthropic_base_url(env_base_url):
        remote_message = remote_control_gate_message(f"{REMOTE_CONTROL_BASE_URL_ENV} in shell")
        return CheckResult(
            name=name,
            status=WARN,
            summary=remote_message,
            hint=remote_message,
        )
    return None


def check_codex_routing(config_path: Path, port: int) -> CheckResult:
    """Is Codex configured to route through the proxy?

    Detection keys on the ``[model_providers.headroom]`` section, which both
    writers emit (install's persistent block and wrap's auto-injected block).
    Substring matching keeps malformed TOML a WARN instead of a crash.
    """
    name = "codex"
    if not config_path.exists():
        return CheckResult(
            name=name,
            status=WARN,
            summary="not routed (no ~/.codex/config.toml)",
            hint="wrap it: headroom wrap codex",
        )
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return CheckResult(name=name, status=WARN, summary=f"could not read {config_path}: {exc}")
    if "[model_providers.headroom]" not in text:
        return CheckResult(
            name=name,
            status=WARN,
            summary="not routed (no Headroom provider in config.toml)",
            hint="wrap it: headroom wrap codex",
        )
    match = _CODEX_BASE_URL_RE.search(text)
    if match and int(match.group(1)) != port:
        return CheckResult(
            name=name,
            status=WARN,
            summary=f"routed to port {match.group(1)}, but doctor probed port {port}",
            hint=f"re-run with: headroom doctor --port {match.group(1)}",
        )
    return CheckResult(name=name, status=PASS, summary=f"routed ({config_path})")


def check_shell_env(environ: Mapping[str, str], port: int) -> CheckResult:
    """Is the *current shell* pointed at the proxy for ad-hoc runs?"""
    name = "shell env"
    for var in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"):
        value = environ.get(var, "")
        if value:
            return _classify_routing_url(name, value, port, source=var)
    return CheckResult(
        name=name,
        status=WARN,
        summary="ANTHROPIC_BASE_URL / OPENAI_BASE_URL unset — this shell bypasses the proxy",
        hint=f"export ANTHROPIC_BASE_URL=http://127.0.0.1:{port} (or launch via headroom wrap)",
    )


def _classify_routing_url(name: str, url: str, port: int, *, source: str) -> CheckResult:
    match = _LOOPBACK_URL_RE.match(url.strip())
    if match is None:
        return CheckResult(
            name=name,
            status=WARN,
            summary=f"points at {url}, not the local Headroom proxy ({source})",
        )
    found_port = int(match.group(1))
    if found_port != port:
        return CheckResult(
            name=name,
            status=WARN,
            summary=f"routed to port {found_port}, but doctor probed port {port} ({source})",
            hint=f"re-run with: headroom doctor --port {found_port}",
        )
    return CheckResult(name=name, status=PASS, summary=f"routed via {source}")


def check_savings(stats: dict[str, Any] | None, savings_file: Path) -> CheckResult:
    """Are savings actually flowing? Lifetime totals + last activity."""
    name = "savings"
    payload: dict[str, Any] | None = None
    source = "proxy /stats"
    if stats is not None and isinstance(stats.get("persistent_savings"), dict):
        payload = stats["persistent_savings"]
    elif savings_file.exists():
        source = str(savings_file)
        try:
            payload = json.loads(savings_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return CheckResult(
                name=name, status=WARN, summary=f"could not read savings file {savings_file}"
            )
    if payload is None:
        return CheckResult(
            name=name,
            status=WARN,
            summary="no savings recorded yet",
            hint="route a client through the proxy and make a request",
        )

    lifetime = payload.get("lifetime") or {}
    tokens = lifetime.get("tokens_saved", 0) or 0
    usd = lifetime.get("compression_savings_usd", 0.0) or 0.0
    if not tokens:
        return CheckResult(
            name=name,
            status=WARN,
            summary="no tokens saved yet",
            hint="route a client through the proxy and make a request",
        )

    session = payload.get("display_session") or {}
    freshness = None
    last_activity = session.get("last_activity_at")
    if isinstance(last_activity, str):
        freshness = _format_since(last_activity)
    summary = f"{tokens:,} tokens / ${usd:,.2f} saved lifetime"
    if freshness:
        summary += f" — last request {freshness}"
    return CheckResult(name=name, status=PASS, summary=f"{summary} ({source})")


def check_budget(stats: dict[str, Any] | None) -> CheckResult:
    """Is a spend budget configured on the proxy?"""
    name = "budget"
    if stats is None:
        return CheckResult(name=name, status=SKIP, summary="proxy not reachable")
    cost = stats.get("cost")
    if not isinstance(cost, dict):
        return CheckResult(name=name, status=WARN, summary="cost tracking disabled (--no-cost)")
    if "budget_limit_usd" not in cost:
        return CheckResult(
            name=name,
            status=WARN,
            summary="proxy does not report budget config (older version?)",
            hint="restart the proxy on the current version",
        )
    limit = cost.get("budget_limit_usd")
    if limit is None:
        return CheckResult(
            name=name,
            status=WARN,
            summary="no budget configured — spend is unlimited",
            hint="set one: headroom proxy --budget 10 (env: HEADROOM_BUDGET)",
        )
    period = cost.get("budget_period", "daily")
    return CheckResult(name=name, status=PASS, summary=f"${limit}/{period} budget enforced")


def check_deployments(manifests: list[Any], probe: Any = probe_json) -> CheckResult | None:
    """Probe persistent deployment health URLs. None when no deployments."""
    if not manifests:
        return None
    down = []
    for manifest in manifests:
        payload = probe(manifest.health_url)
        ready = bool(payload and (payload.get("ready") or payload.get("status") == "healthy"))
        if not ready:
            down.append(manifest.profile)
    if down:
        return CheckResult(
            name="deployments",
            status=FAIL,
            summary=f"{len(down)} of {len(manifests)} deployment(s) down: {', '.join(down)}",
            hint="inspect with: headroom install status --profile <name>",
        )
    return CheckResult(
        name="deployments",
        status=PASS,
        summary=f"{len(manifests)} deployment(s) healthy",
    )


_STATUS_STYLE = {PASS: "green", WARN: "yellow", FAIL: "red", SKIP: "dim"}
_STATUS_GLYPH = {PASS: "✓", WARN: "⚠", FAIL: "✗", SKIP: "·"}


def _render(checks: list[CheckResult], port: int, installed: str) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    console.print(f"[bold]Headroom Doctor[/bold] [dim]v{installed} · port {port}[/dim]\n")
    table = Table(show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("summary")
    for check in checks:
        style = _STATUS_STYLE.get(check.status, "white")
        glyph = _STATUS_GLYPH.get(check.status, "?")
        table.add_row(
            check.name,
            f"[{style}]{glyph} {check.status}[/{style}]",
            escape(check.summary),
        )
    console.print(table)
    for check in checks:
        if check.hint:
            console.print(f"[dim]{check.name}:[/dim] {escape(check.hint)}")

    fails = sum(1 for c in checks if c.status == FAIL)
    warns = sum(1 for c in checks if c.status == WARN)
    if fails or warns:
        console.print(f"\n[bold]{fails} failure(s), {warns} warning(s)[/bold]")
    else:
        console.print("\n[green bold]all checks passed[/green bold]")


@main.command()
@click.option(
    "--port",
    "-p",
    default=8787,
    type=click.IntRange(1, 65535),
    envvar="HEADROOM_PORT",
    help="Proxy port to check (default: 8787, env: HEADROOM_PORT)",
)
@click.option("--json", "emit_json", is_flag=True, help="Emit JSON instead of formatted output.")
def doctor(port: int, emit_json: bool) -> None:
    """Check that the Headroom proxy and client routing are working.

    \b
    Exit codes:
        0  everything healthy
        1  warnings only (working, but not optimally wired)
        2  at least one failure (proxy down / deployment down)
    """
    base_url = f"http://127.0.0.1:{port}"
    livez = probe_json(f"{base_url}/livez")
    stats = probe_json(f"{base_url}/stats", timeout=5.0) if livez else None
    installed = get_version()

    checks = [
        check_proxy_liveness(livez, base_url),
        check_version_drift(livez, installed),
        check_claude_routing(claude_settings_path(), port),
        check_codex_routing(codex_config_path(), port),
        check_shell_env(os.environ, port),
        check_savings(stats, savings_path()),
        check_budget(stats),
    ]
    remote_control_gate_check = check_claude_remote_control_gate(claude_settings_path(), os.environ)
    if remote_control_gate_check is not None:
        checks.append(remote_control_gate_check)
    deployments = check_deployments(list_manifests())
    if deployments is not None:
        checks.append(deployments)

    if any(c.status == FAIL for c in checks):
        exit_code = 2
    elif any(c.status == WARN for c in checks):
        exit_code = 1
    else:
        exit_code = 0

    if emit_json:
        click.echo(
            json.dumps(
                {
                    "port": port,
                    "installed_version": installed,
                    "exit_code": exit_code,
                    "checks": [asdict(c) for c in checks],
                },
                indent=2,
            )
        )
    else:
        _render(checks, port, installed)
    raise SystemExit(exit_code)
