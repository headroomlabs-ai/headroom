"""Resolve which local port the Headroom proxy is (or should be) on.

``wrap``, ``init`` and ``doctor`` all default to port 8787. When a proxy is
already running elsewhere (``HEADROOM_PORT``, a persistent deployment on a
custom port, a live wrap session), silently defaulting to 8787 either starts a
second proxy or points an agent at a dead port. These helpers make the default
port follow ``HEADROOM_PORT`` and cheaply discover a live Headroom proxy on the
other ports Headroom itself recorded, so the CLI can warn -- or, on a TTY, ask
-- instead of guessing.

Discovery is deliberately bounded: loopback only, a handful of candidate ports
(the requested/default port, ``HEADROOM_PORT``, 8787, live wrap-client markers,
deployment-manifest ports and the project's wrap marker), probed concurrently
with a sub-second timeout.
It never scans port ranges. ``HEADROOM_PORT_DISCOVERY=0`` disables it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import click

from headroom import fsutil, paths
from headroom._subprocess import identity_mismatch, pid_alive, proc_identity

DEFAULT_PROXY_PORT = 8787
PORT_ENV = "HEADROOM_PORT"
DISCOVERY_ENV = "HEADROOM_PORT_DISCOVERY"
_PROBE_TIMEOUT_SECONDS = 0.5
_MAX_CANDIDATES = 8
_MAX_CLIENT_MARKER_CANDIDATES = 3
_HEADROOM_SERVICE = "headroom-proxy"
_LOOPBACK_PORT_RE = re.compile(r"^https?://(?:127\.0\.0\.1|localhost|\[::1\]):(\d+)")

Probe = Callable[[int], bool]


def _valid_port(value: Any) -> int | None:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return None
    return port if 1 <= port <= 65535 else None


def env_port(environ: Mapping[str, str] | None = None) -> int | None:
    """Return ``HEADROOM_PORT`` as a port number, or None when unset/invalid."""
    env = os.environ if environ is None else environ
    raw = (env.get(PORT_ENV) or "").strip()
    return _valid_port(raw) if raw else None


def default_port(environ: Mapping[str, str] | None = None) -> int:
    """The port a command should use absent ``--port``: ``HEADROOM_PORT`` or 8787."""
    return env_port(environ) or DEFAULT_PROXY_PORT


def discovery_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether live-proxy discovery may run (``HEADROOM_PORT_DISCOVERY=0`` disables)."""
    env = os.environ if environ is None else environ
    return (env.get(DISCOVERY_ENV) or "").strip().lower() not in {"0", "false", "no", "off"}


def loopback_port(url: str) -> int | None:
    """Port of a loopback ``http(s)://127.0.0.1|localhost:<port>`` URL, else None."""
    match = _LOOPBACK_PORT_RE.match((url or "").strip())
    return _valid_port(match.group(1)) if match else None


def _manifest_ports(manifests: Iterable[Any] | None) -> list[int]:
    if manifests is None:
        try:
            from headroom.install.state import list_manifests

            manifests = list_manifests()
        except Exception:  # noqa: BLE001 - discovery is best-effort
            return []
    ports: list[int] = []
    for manifest in manifests:
        port = _valid_port(getattr(manifest, "port", None))
        if port is not None:
            ports.append(port)
    return ports


def _wrap_marker_ports(cwd: Path | None) -> list[int]:
    """Ports recorded by ``wrap claude``'s project marker (read-only, no import of wrap)."""
    marker = (cwd or Path.cwd()) / ".claude" / ".headroom_wrap_marker.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    port = _valid_port(payload.get("port")) if isinstance(payload, dict) else None
    return [port] if port is not None else []


def _marker_pid_reused(marker: Path, pid: int) -> bool:
    try:
        payload = json.loads(fsutil.read_text(marker))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return identity_mismatch(
        payload.get("start_src"),
        payload.get("start_time"),
        pid,
        identity_fn=proc_identity,
    )


def live_client_pids(port: int) -> list[int]:
    """Live wrap-client PIDs for ``port``, pruning dead and recycled markers."""
    clients_dir = paths.proxy_clients_dir(port)
    if not clients_dir.exists():
        return []
    live: list[int] = []
    for marker in clients_dir.glob("*.json"):
        try:
            pid = int(marker.stem)
        except ValueError:
            continue
        if not pid_alive(pid) or _marker_pid_reused(marker, pid):
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        live.append(pid)
    return live


def _marker_started_at(port: int, pid: int) -> float:
    marker = paths.proxy_clients_dir(port) / f"{pid}.json"
    try:
        payload = json.loads(fsutil.read_text(marker))
    except (OSError, ValueError):
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    started_at = payload.get("started_at")
    return float(started_at) if isinstance(started_at, int | float) else 0.0


def live_client_proxy_ports() -> list[tuple[int, float]]:
    """Return live-client ports, newest marker first as ``(port, started_at)``."""
    clients_dir = paths.workspace_dir() / "clients"
    try:
        entries = list(clients_dir.iterdir())
    except OSError:
        return []
    candidates: list[tuple[int, float]] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        port = _valid_port(entry.name)
        if port is None:
            continue
        pids = live_client_pids(port)
        if pids:
            candidates.append((port, max(_marker_started_at(port, pid) for pid in pids)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def candidate_ports(
    requested: int,
    *,
    environ: Mapping[str, str] | None = None,
    manifests: Iterable[Any] | None = None,
    cwd: Path | None = None,
    extra: Sequence[int] = (),
) -> list[int]:
    """Ordered, de-duplicated, bounded list of ports worth probing."""
    ordered: list[int] = [requested]
    env_value = env_port(environ)
    if env_value is not None:
        ordered.append(env_value)
    ordered.append(DEFAULT_PROXY_PORT)
    ordered.extend(extra)
    ordered.extend(_manifest_ports(manifests))
    ordered.extend(_wrap_marker_ports(cwd))
    # Bounded and last so stale wrap sessions can't evict recorded deployments.
    ordered.extend(port for port, _ in live_client_proxy_ports()[:_MAX_CLIENT_MARKER_CANDIDATES])
    seen: list[int] = []
    for port in ordered:
        if port not in seen:
            seen.append(port)
    return seen[:_MAX_CANDIDATES]


def probe_headroom_proxy(port: int, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    """True when a Headroom proxy answers ``/livez`` on loopback ``port``.

    Requires the ``service: headroom-proxy`` identity (or the legacy
    ``alive``+``version`` shape) so an unrelated local server on the port is
    never mistaken for Headroom.
    """
    from headroom.install.health import probe_json

    return is_headroom_livez(probe_json(f"http://127.0.0.1:{port}/livez", timeout=timeout))


def is_headroom_livez(payload: Mapping[str, Any] | None) -> bool:
    """Whether a ``/livez`` payload identifies a Headroom proxy."""
    if not payload:
        return False
    if payload.get("service") == _HEADROOM_SERVICE:
        return True
    return "alive" in payload and "version" in payload and "service" not in payload


def live_ports(ports: Sequence[int], probe: Probe | None = None) -> list[int]:
    """Return the subset of ``ports`` with a live Headroom proxy, preserving order."""
    if not ports:
        return []
    check = probe or probe_headroom_proxy
    if len(ports) == 1:
        return [ports[0]] if check(ports[0]) else []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        results = list(pool.map(check, ports))
    return [port for port, alive in zip(ports, results, strict=True) if alive]


def find_live_proxy_elsewhere(
    requested: int,
    *,
    environ: Mapping[str, str] | None = None,
    manifests: Iterable[Any] | None = None,
    cwd: Path | None = None,
    probe: Probe | None = None,
    extra: Sequence[int] = (),
) -> int | None:
    """Port of a live Headroom proxy when ``requested`` has none, else None.

    Returns None when discovery is disabled, when ``requested`` itself is live
    (nothing to reconcile), or when no candidate answers. Never raises.
    """
    if not discovery_enabled(environ):
        return None
    try:
        candidates = candidate_ports(
            requested, environ=environ, manifests=manifests, cwd=cwd, extra=extra
        )
        alive = live_ports(candidates, probe=probe)
    except Exception:  # noqa: BLE001 - discovery must never break a command
        return None
    if requested in alive:
        return None
    return next((port for port in alive if port != requested), None)


def resolve_read_port(
    explicit: int | None, *, default: int = DEFAULT_PROXY_PORT
) -> tuple[int, str]:
    """Resolve a read-only command's proxy port, preferring the latest live wrap."""
    if explicit is not None:
        return explicit, "explicit"

    raw_env_port = (os.environ.get(PORT_ENV) or "").strip()
    if raw_env_port:
        try:
            parsed = int(raw_env_port)
        except ValueError:
            raise click.ClickException(
                f"{PORT_ENV} must be an integer, got {raw_env_port!r}"
            ) from None
        if not 1 <= parsed <= 65535:
            raise click.ClickException(f"{PORT_ENV} must be between 1 and 65535, got {parsed}")
        return parsed, "env"

    if not discovery_enabled():
        return default, "default"

    marker_ports = [port for port, _ in live_client_proxy_ports()[:_MAX_CLIENT_MARKER_CANDIDATES]]
    healthy_markers = live_ports(marker_ports, probe=proxy_is_healthy)
    if healthy_markers:
        return healthy_markers[0], "discovered"

    already_probed = set(marker_ports)

    def _probe_unchecked(port: int) -> bool:
        return port not in already_probed and proxy_is_healthy(port)

    live_port = find_live_proxy_elsewhere(default, probe=_probe_unchecked)
    if live_port is not None:
        return live_port, "discovered"
    return default, "default"


def proxy_is_healthy(port: int, timeout: float = 1.0) -> bool:
    """True if ``/health`` identifies a live Headroom proxy on ``port``."""
    from headroom.install.health import probe_json

    payload = probe_json(f"http://127.0.0.1:{port}/health", timeout=timeout)
    return payload is not None and payload.get("service") == _HEADROOM_SERVICE


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def reconcile_default_port(
    requested: int,
    *,
    option_name: str = "--port",
    interactive: bool | None = None,
    environ: Mapping[str, str] | None = None,
    manifests: Iterable[Any] | None = None,
    cwd: Path | None = None,
    probe: Probe | None = None,
) -> int:
    """Return the port to use when the user did not choose one explicitly.

    * No live proxy elsewhere (or ``requested`` is live): ``requested``.
    * A live Headroom proxy on another port and a TTY: ask whether to connect
      to it, keep ``requested``, or abort (``click.Abort``).
    * Non-interactive: keep ``requested`` -- never adopt a port silently -- and
      warn on stderr naming the live port and the flag that selects it.
    """
    live = find_live_proxy_elsewhere(
        requested, environ=environ, manifests=manifests, cwd=cwd, probe=probe
    )
    if live is None:
        return requested
    click.echo(
        f"headroom: no Headroom proxy on port {requested}, but one is running on port {live}.",
        err=True,
    )
    if interactive is None:
        interactive = _is_interactive()
    if not interactive:
        click.echo(
            f"headroom: continuing with port {requested}. To use the running proxy, pass "
            f"{option_name} {live} (or set {PORT_ENV}={live}).",
            err=True,
        )
        return requested
    choice = click.prompt(
        f"Connect to the running proxy on {live} [c], keep port {requested} [k], or abort [a]?",
        type=click.Choice(["c", "k", "a"], case_sensitive=False),
        default="c",
        err=True,
    )
    choice = choice.lower()
    if choice == "a":
        raise click.Abort()
    return live if choice == "c" else requested


def port_option_callback(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
    """Click callback: reconcile a defaulted ``--port`` with any live proxy.

    An explicit ``--port`` (command line) or ``HEADROOM_PORT`` (the option's
    envvar) is always honored as-is. Only a pure default triggers discovery.
    """
    from click.core import ParameterSource

    if value is None or param.name is None:
        return value
    if ctx.resilient_parsing:
        return value
    if ctx.get_parameter_source(param.name) is not ParameterSource.DEFAULT:
        return value
    option_name = next((opt for opt in param.opts if opt.startswith("--")), "--port")
    return reconcile_default_port(int(value), option_name=option_name)


def proxy_port_option(*param_decls: str, help_text: str | None = None) -> Callable[[Any], Any]:
    """``--port`` option shared by ``wrap`` subcommands.

    Defaults to ``HEADROOM_PORT`` (else 8787) and, when left at the default,
    detects a live Headroom proxy on another recorded port (see module docs).
    """
    decls = param_decls or ("--port", "-p")
    option: Callable[[Any], Any] = click.option(
        *decls,
        default=DEFAULT_PROXY_PORT,
        envvar=PORT_ENV,
        show_envvar=False,
        type=click.IntRange(1, 65535),
        callback=port_option_callback,
        help=help_text
        or (
            f"Proxy port (default: ${PORT_ENV}, else {DEFAULT_PROXY_PORT}; a live "
            "Headroom proxy on another port is detected and offered)"
        ),
    )
    return option


_CODEX_HEADROOM_TABLE_RE = re.compile(
    r"(?ms)^[ \t]*\[model_providers\.headroom\][^\n]*\n(?P<body>.*?)(?=^[ \t]*\[|\Z)"
)
_CODEX_BASE_URL_LINE_RE = re.compile(r'(?m)^[ \t]*base_url[ \t]*=[ \t]*"(?P<url>[^"\n]*)"')


def codex_headroom_provider_port(content: str) -> int | None:
    """Port in an existing ``[model_providers.headroom]`` loopback ``base_url``."""
    table = _CODEX_HEADROOM_TABLE_RE.search(content or "")
    if table is None:
        return None
    base = _CODEX_BASE_URL_LINE_RE.search(table.group("body"))
    return loopback_port(base.group("url")) if base else None


def warn_codex_provider_port_change(config_path: Path, new_port: int) -> int | None:
    """Warn (stderr) before a Codex headroom provider is repointed to a new port.

    Returns the previous port when it differs from ``new_port``; None otherwise.
    Never raises.
    """
    try:
        if not config_path.exists():
            return None
        previous = codex_headroom_provider_port(
            config_path.read_text(encoding="utf-8", errors="replace")
        )
    except OSError:
        return None
    if previous is None or previous == new_port:
        return None
    click.echo(
        f"headroom: warning: {config_path} routes Codex's [model_providers.headroom] to "
        f"port {previous}; repointing it to port {new_port}. Pass --port {previous} to keep "
        "the existing route.",
        err=True,
    )
    return previous
