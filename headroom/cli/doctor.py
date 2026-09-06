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
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from headroom._version import format_version_label, normalize_release_version
from headroom.install.health import probe_json
from headroom.install.models import DeploymentManifest
from headroom.install.paths import claude_settings_path, codex_config_path
from headroom.install.state import list_manifests
from headroom.paths import savings_path
from headroom.providers.claude import (
    REMOTE_CONTROL_BASE_URL_ENV,
    REMOTE_CONTROL_SIBLING_GATE_NOTE,
    claude_auth_conflict_message,
    claude_auth_conflict_sources,
    detect_claude_code_version,
    is_custom_anthropic_base_url,
    remote_control_applies_to_auth,
    remote_control_gate_active,
    remote_control_gate_message,
)

from .main import get_version, main
from .wrap import _read_wrap_marker, _wrap_marker_is_stale

PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

DEFAULT_PROXY_PORT = 8787

_LOOPBACK_URL_RE = re.compile(r"https?://(?:127\.0\.0\.1|localhost):(\d+)")
_CODEX_BASE_URL_RE = re.compile(r'base_url\s*=\s*"https?://(?:127\.0\.0\.1|localhost):(\d+)')

# Ollama's fixed default port. `ollama launch claude` writes
# ``ANTHROPIC_BASE_URL=http://127.0.0.1:11434`` into the launched Claude Code
# child, which outranks the persistent-install env block and silently bypasses
# the Headroom proxy (issue #2199). Recognized so the routing diagnostic names
# the collision instead of telling the user to re-probe port 11434.
_OLLAMA_DEFAULT_PORT = 11434


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
    uptime_text = f"up {_format_uptime(uptime)}" if isinstance(uptime, int | float) else "up"
    return CheckResult(
        name="proxy",
        status=PASS,
        summary=f"running at {base_url} ({uptime_text}, {format_version_label(version)})",
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
    running_release = normalize_release_version(running)
    installed_release = normalize_release_version(installed)
    if running_release is None or installed_release is None:
        return CheckResult(
            name="version",
            status=SKIP,
            summary=f"source/non-release version label (proxy {running}, installed {installed})",
        )
    if running_release != installed_release:
        return CheckResult(
            name="version",
            status=WARN,
            summary=f"version drift: proxy {running}, installed {installed}",
            hint="restart the proxy to pick up new code: headroom proxy",
        )
    return CheckResult(
        name="version",
        status=PASS,
        summary=f"proxy matches installed {format_version_label(installed)}",
    )


def _claude_base_url_in(path: Path) -> tuple[str, CheckResult | None]:
    """Read ``env.ANTHROPIC_BASE_URL`` from one Claude settings file.

    Returns ``(base_url, error)``. A parse problem comes back as a WARN so the
    caller surfaces it verbatim instead of skipping the file and reporting the
    misleading "not routed".
    """
    name = "claude"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return "", CheckResult(name=name, status=WARN, summary=f"could not parse {path}: {exc}")
    # `json.loads` succeeds on valid non-object JSON (e.g. `[]`, `null`, `42`),
    # which a hand-edited or reset settings file can contain. `.get` on a
    # non-dict raises AttributeError, and it is not one of the caught parse
    # errors above, so it would crash the very command run to diagnose the
    # broken config. Treat a non-object like an unparseable file.
    if not isinstance(payload, dict):
        return "", CheckResult(
            name=name,
            status=WARN,
            summary=f"could not parse {path}: not a JSON object",
        )
    env_block = payload.get("env")
    if isinstance(env_block, dict):
        return str(env_block.get("ANTHROPIC_BASE_URL", "") or ""), None
    return "", None


def check_claude_routing(
    settings_path: Path,
    port: int,
    project_settings_paths: Sequence[Path] | None = None,
) -> CheckResult:
    """Is Claude Code configured to route through the proxy?

    Claude Code layers project settings over user settings, and `headroom init
    claude` without --global writes the project-scoped
    ``.claude/settings.local.json``. Reading only ``~/.claude/settings.json``
    reported "not routed" for sessions that demonstrably were -- confirmed by
    `ps eww` on the live process and by active compression on it (#3205).
    Candidates are consulted in Claude's own precedence order, and the summary
    names the file that supplied the routing so the scope is never ambiguous.
    """
    name = "claude"
    candidates = [*(project_settings_paths or []), settings_path]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return CheckResult(
            name=name,
            status=WARN,
            summary="not routed (no ~/.claude/settings.json)",
            hint="wrap it: headroom wrap claude",
        )
    first_error: CheckResult | None = None
    for candidate in existing:
        base_url, error = _claude_base_url_in(candidate)
        if error is not None:
            first_error = first_error or error
            continue
        if base_url:
            return _classify_routing_url(name, base_url, port, source=str(candidate))
    if first_error is not None:
        return first_error
    return CheckResult(
        name=name,
        status=WARN,
        summary="not routed (no ANTHROPIC_BASE_URL in settings env)",
        hint="wrap it: headroom wrap claude",
    )


def check_claude_auth_conflict(
    settings_path: Path,
    project_settings_path: Path,
    project_local_settings_path: Path,
    environ: Mapping[str, str],
) -> CheckResult | None:
    """Report contradictory effective Claude credentials without their values."""

    def settings_env(path: Path) -> dict[str, object]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        env = payload.get("env") if isinstance(payload, dict) else None
        return dict(env) if isinstance(env, dict) else {}

    conflict = claude_auth_conflict_sources(
        (str(settings_path), settings_env(settings_path)),
        (str(project_settings_path), settings_env(project_settings_path)),
        (str(project_local_settings_path), settings_env(project_local_settings_path)),
        ("shell environment", environ),
    )
    if conflict is None:
        return None
    return CheckResult(
        name="claude auth",
        status=FAIL,
        summary=claude_auth_conflict_message(conflict),
    )


def claude_desktop_config_dir() -> Path:
    """Return Claude Desktop's per-user config directory for this platform.

    Claude Desktop (``com.anthropic.claudefordesktop``) stores its config here,
    distinct from Claude Code CLI's ``~/.claude``. Directory existence is used as
    a proxy for "Desktop is installed / has been run" (#2925).
    """
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Claude"
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "Claude"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else home / ".config"
    return base / "Claude"


def check_claude_desktop(config_dir: Path) -> CheckResult | None:
    """Surface that Claude Desktop agent sessions bypass the proxy (#2925 / #869).

    Claude Desktop unconditionally overwrites ``ANTHROPIC_BASE_URL`` when it
    spawns agent sessions, so a correctly-wrapped ``~/.claude/settings.json``
    (which the ``claude`` check verifies for the terminal CLI) does not route
    Desktop traffic. Without this, ``doctor`` passes on the settings value alone
    and never hints that Desktop sessions are unrouted.

    Reported as its own per-surface row -- like ``wrap_marker`` and ``shell env``
    -- and only when Desktop is detected, so it never contradicts a genuinely
    routed CLI. Returns ``None`` when Desktop is absent (no row).
    """
    if not config_dir.exists():
        return None
    return CheckResult(
        name="claude desktop",
        status=WARN,
        summary="agent sessions bypass the proxy (Desktop overwrites ANTHROPIC_BASE_URL)",
        hint=(
            "Desktop routing is not supported yet (see #869); use the terminal "
            "Claude Code CLI for proxy-routed sessions."
        ),
    )


def check_claude_remote_control_gate(
    settings_path: Path,
    environ: Mapping[str, str],
    *,
    version: tuple[int, int, int] | None = None,
    version_resolver: Callable[[], tuple[int, int, int] | None] | None = None,
) -> CheckResult | None:
    """Warn once when Claude custom-base routing hides Remote Control (issue #1779).

    Fires only for a session that could ever have had Remote Control — a
    subscription auth mode (not API-key/cloud IAM) on a Claude Code build at/after
    the gate version, or an unknown version. Auth signals are read from the shell
    ``environ`` overlaid on the settings-file ``env`` block, so an API key
    configured in either place suppresses the warning.

    ``version`` is the detected Claude Code version (``None`` = unknown); tests
    pass it directly so the check stays pure. ``version_resolver`` lets the
    ``doctor`` entrypoint defer the ``claude --version`` subprocess until the
    cheap gates (custom base URL + subscription auth) have passed — most doctor
    runs never pay it. An explicit ``version`` wins over the resolver; the
    resolver is called at most once.
    """
    name = "claude remote control"
    settings_env: dict[str, object] = {}
    settings_base_url = ""
    if settings_path.exists():
        try:
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
            # Valid non-object JSON (`[]`, `null`, ...) parses fine but has no
            # `.get`, and AttributeError is not caught below; guard for dict-ness
            # so a malformed settings file can't crash the gate check.
            env_block = payload.get("env") if isinstance(payload, dict) else None
            if isinstance(env_block, dict):
                settings_env = env_block
                settings_base_url = str(env_block.get("ANTHROPIC_BASE_URL", "") or "")
        except (OSError, ValueError):
            settings_env = {}
            settings_base_url = ""

    # Shell env wins over settings env, matching Claude Code's own precedence.
    effective_env: dict[str, object] = {**settings_env, **dict(environ)}
    env_base_url = environ.get("ANTHROPIC_BASE_URL", "")

    resolved_version = version
    version_resolved = version is not None or version_resolver is None

    for base_url, source in (
        (settings_base_url, "from settings"),
        (env_base_url, "in shell"),
    ):
        # Cheap gates first so the version subprocess only runs when a warning
        # is actually plausible for this environment.
        if not is_custom_anthropic_base_url(base_url):
            continue
        if not remote_control_applies_to_auth(effective_env):
            return None
        if not version_resolved and version_resolver is not None:
            resolved_version = version_resolver()
            version_resolved = True
        if remote_control_gate_active(base_url, effective_env, resolved_version):
            remote_message = remote_control_gate_message(
                f"{REMOTE_CONTROL_BASE_URL_ENV} {source}", version=resolved_version
            )
            return CheckResult(
                name=name,
                status=WARN,
                summary=remote_message,
                hint=REMOTE_CONTROL_SIBLING_GATE_NOTE,
            )
    return None


def check_wrap_marker_staleness(settings_path: Path) -> CheckResult:
    """Flag a project-local ANTHROPIC_BASE_URL left by a crashed wrap session.

    A crashed ``headroom wrap claude`` (SIGKILL, OOM, reboot) can leave
    ``.claude/settings.local.json`` pointing at a dead proxy port, hanging
    every subsequent bare ``claude`` invocation in the project (issue #1768).
    This checks the project-local settings file — separate from the global
    ``~/.claude/settings.json`` :func:`check_claude_routing` inspects.
    """
    name = "wrap_marker"
    marker = _read_wrap_marker(settings_path)
    if marker is None:
        return CheckResult(name=name, status=SKIP, summary="no wrap marker found")
    if not _wrap_marker_is_stale(marker):
        return CheckResult(
            name=name, status=PASS, summary=f"live wrap session (pid {marker.get('pid')})"
        )
    return CheckResult(
        name=name,
        status=WARN,
        summary=(
            f"stale ANTHROPIC_BASE_URL from crashed wrap session "
            f"(pid {marker.get('pid')}, port {marker.get('port')}) — "
            "run `headroom unwrap claude` to clean it up"
        ),
    )


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
    # Routed, but Codex may still attach no credentials. A ChatGPT-OAuth user
    # needs `requires_openai_auth = true` in the provider block or Codex sends
    # no Authorization header at all and every request 401s with "Missing
    # bearer" (#3206). That failure is invisible from here -- the proxy is up,
    # the block is present -- so this check is the only place it can surface.
    if _codex_block_missing_openai_auth(text, config_path):
        return CheckResult(
            name=name,
            status=WARN,
            summary="routed, but Codex will send no Authorization (missing requires_openai_auth)",
            hint="re-run: headroom wrap codex (or headroom init codex) to rewrite the block",
        )
    return CheckResult(name=name, status=PASS, summary=f"routed ({config_path})")


def _codex_block_missing_openai_auth(text: str, config_path: Path) -> bool:
    """ChatGPT-OAuth Codex routed without ``requires_openai_auth`` (#3206)."""
    start = text.find("[model_providers.headroom]")
    if start == -1:
        return False
    rest = text[start + len("[model_providers.headroom]") :]
    end = rest.find("\n[")
    block = rest if end == -1 else rest[:end]
    if "requires_openai_auth" in block:
        return False
    try:
        from headroom.providers.codex.install import codex_uses_chatgpt_auth

        return codex_uses_chatgpt_auth(config_path.parent / "auth.json")
    except Exception:  # pragma: no cover - never let a doctor check crash
        return False


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
        if found_port == _OLLAMA_DEFAULT_PORT:
            # Not a mis-probed Headroom port — this is Ollama's endpoint, so
            # `headroom doctor --port 11434` would only chase a red herring.
            return CheckResult(
                name=name,
                status=WARN,
                summary=(
                    f"points at Ollama ({url}), not the Headroom proxy ({source}) — "
                    "`ollama launch claude` bypasses the persistent Headroom route"
                ),
                hint=(
                    "both claim ANTHROPIC_BASE_URL; run Ollama-backed sessions "
                    "through Headroom by chaining the proxy at its Ollama upstream "
                    "(see issue #2199)"
                ),
            )
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
    cache_reads = lifetime.get("cache_read_tokens", 0) or 0
    if not tokens and not cache_reads:
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
    if cache_reads:
        cache_usd = lifetime.get("cache_savings_usd", 0.0) or 0.0
        summary += f"; {cache_reads:,} cache-read tokens / ${cache_usd:,.2f} cache savings"
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
    summary = f"${limit}/{period} budget enforced"
    return CheckResult(name=name, status=PASS, summary=summary + _estimated_basis_note(cost))


def _estimated_basis_note(cost: dict[str, Any]) -> str:
    """Describe how much of the period's spend was booked from a token estimate.

    Informational, never a WARN: a provider that simply never reports a usage
    breakdown would otherwise sit at a permanent warning. Every read is
    defensive so `doctor` still works against a proxy predating these fields.
    """
    note = ""

    basis = cost.get("budget_basis")
    if isinstance(basis, dict):
        estimated_usd = basis.get("estimated_usd")
        estimated_pct = basis.get("estimated_pct")
        if isinstance(estimated_usd, (int, float)) and estimated_usd > 0:
            pct = f"{estimated_pct:.0f}% " if isinstance(estimated_pct, (int, float)) else ""
            note += (
                f" — {pct}of period spend (${estimated_usd:.4f}) "
                "booked from Headroom token estimates"
            )

    # Reported independently of the breakdown: a non-default policy changes how
    # the budget is enforced and should surface even if the split is missing.
    policy = cost.get("budget_estimated_basis")
    if isinstance(policy, str) and policy and policy != "count":
        note += f" — estimated-basis policy: {policy}"
    return note


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
    console.print(
        f"[bold]Headroom Doctor[/bold] [dim]{format_version_label(installed)} · port {port}[/dim]\n"
    )
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


def resolve_probe_port(
    requested: int | None,
    manifests: Sequence[DeploymentManifest],
) -> int:
    """Resolve the port `doctor` should probe.

    Precedence: an explicit ``--port`` (or ``HEADROOM_PORT``) wins, then the
    port a managed deployment recorded, then the built-in default.

    The manifest tier matters because ``headroom deploy --port N`` writes N
    into the deployment's own environment, not into the shell `doctor` later
    runs in. Without it a healthy non-default deployment is reported as
    unreachable, and the remediation hints point at the wrong port.

    A deployment is only consulted when it is unambiguous: the ``default``
    profile, or a lone profile under another name. With several named profiles
    and no explicit request there is no non-arbitrary choice, so the built-in
    default is used and the per-deployment section reports each one instead.

    Args:
        requested: Port supplied on the command line or via the environment.
        manifests: Deployment manifests to consider.

    Returns:
        The port to probe.
    """
    if requested is not None:
        return requested
    default_profile = next((m for m in manifests if m.profile == "default"), None)
    if default_profile is not None:
        return default_profile.port
    if len(manifests) == 1:
        return manifests[0].port
    return DEFAULT_PROXY_PORT


@main.command()
@click.option(
    "--port",
    "-p",
    default=None,
    type=click.IntRange(1, 65535),
    envvar="HEADROOM_PORT",
    help=(
        "Proxy port to check (env: HEADROOM_PORT). Defaults to the port of a "
        f"single managed deployment, else {DEFAULT_PROXY_PORT}."
    ),
)
@click.option("--json", "emit_json", is_flag=True, help="Emit JSON instead of formatted output.")
def doctor(port: int | None, emit_json: bool) -> None:
    """Check that the Headroom proxy and client routing are working.

    \b
    Exit codes:
        0  everything healthy
        1  warnings only (working, but not optimally wired)
        2  at least one failure (proxy down / deployment down)
    """
    manifests = list_manifests()
    port = resolve_probe_port(port, manifests)
    base_url = f"http://127.0.0.1:{port}"
    livez = probe_json(f"{base_url}/livez")
    stats = probe_json(f"{base_url}/stats", timeout=5.0) if livez else None
    installed = get_version()

    project_claude_settings = Path.cwd() / ".claude" / "settings.json"
    project_local_claude_settings = Path.cwd() / ".claude" / "settings.local.json"
    checks = [
        check_proxy_liveness(livez, base_url),
        check_version_drift(livez, installed),
        check_claude_routing(
            claude_settings_path(),
            port,
            [project_local_claude_settings, project_claude_settings],
        ),
        check_wrap_marker_staleness(project_local_claude_settings),
        check_codex_routing(codex_config_path(), port),
        check_shell_env(os.environ, port),
        check_savings(stats, savings_path()),
        check_budget(stats),
    ]
    auth_conflict_check = check_claude_auth_conflict(
        claude_settings_path(),
        project_claude_settings,
        project_local_claude_settings,
        os.environ,
    )
    if auth_conflict_check is not None:
        checks.append(auth_conflict_check)
    # Lazy resolver: `claude --version` is a Node CLI subprocess (seconds of
    # cold start, 10s worst-case timeout) — only pay for it when the RC gate
    # is actually plausible (custom base URL + subscription auth).
    remote_control_gate_check = check_claude_remote_control_gate(
        claude_settings_path(), os.environ, version_resolver=detect_claude_code_version
    )
    if remote_control_gate_check is not None:
        checks.append(remote_control_gate_check)
    desktop_check = check_claude_desktop(claude_desktop_config_dir())
    if desktop_check is not None:
        checks.append(desktop_check)
    deployments = check_deployments(manifests)
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
