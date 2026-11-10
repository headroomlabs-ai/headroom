"""Pure (GUI-free) logic for the Headroom system tray: stats polling and formatting."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_PORT = 8787


@dataclass(frozen=True)
class Snapshot:
    """What the tray shows: lifetime tokens and estimated dollars saved."""

    online: bool = False
    tokens_saved: int = 0
    usd_saved: float = 0.0
    requests: int = 0


def base_url(host: str, port: int) -> str:
    # A wildcard bind address is not connectable; talk to loopback instead.
    if host in ("", "0.0.0.0", "::", "[::]"):  # noqa: S104 — comparison, not a bind
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def dashboard_url(url: str) -> str:
    return f"{url}/dashboard"


def parse_stats(payload: dict[str, Any]) -> Snapshot:
    """Extract the tray figures from a ``/stats`` payload (missing keys -> 0)."""
    persistent = payload.get("persistent_savings")
    lifetime = persistent.get("lifetime") if isinstance(persistent, dict) else None
    if not isinstance(lifetime, dict):
        lifetime = {}

    def _num(key: str) -> float:
        value = lifetime.get(key)
        return float(value) if isinstance(value, (int, float)) else 0.0

    tokens = _num("tokens_saved") + _num("output_tokens_saved")
    usd = _num("compression_savings_usd") + _num("output_savings_usd")
    return Snapshot(
        online=True,
        tokens_saved=int(tokens),
        usd_saved=usd,
        requests=int(_num("requests")),
    )


def fetch_snapshot(url: str, *, timeout: float = 3.0) -> Snapshot:
    """Poll the local proxy. Never raises; an unreachable proxy yields ``online=False``."""
    try:
        req = urllib.request.Request(  # noqa: S310 — http URL built from local config
            f"{url}/stats?cached=1", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        return parse_stats(data) if isinstance(data, dict) else Snapshot()
    except Exception:
        return Snapshot()


def format_tokens(count: int) -> str:
    """Compact token count: 950, 12.3K, 4.56M, 1.20B."""
    n = max(int(count), 0)
    if n < 1_000:
        return str(n)
    for divisor, suffix in ((1_000, "K"), (1_000_000, "M"), (1_000_000_000, "B")):
        value = n / divisor
        digits = 2 if value < 10 else 1 if value < 100 else 0
        if round(value, digits) < 1000 or suffix == "B":
            return f"{value:.{digits}f}{suffix}"
    return str(n)


def format_usd(amount: float) -> str:
    amount = max(float(amount), 0.0)
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f}M"
    if amount >= 10_000:
        return f"${amount / 1_000:.1f}K"
    return f"${amount:,.2f}"


def stats_label(snap: Snapshot) -> str:
    if not snap.online:
        return "Proxy offline"
    return (
        f"\u25bc {format_tokens(snap.tokens_saved)} tokens  \u00b7  "
        f"~{format_usd(snap.usd_saved)} saved"
    )


def tooltip(snap: Snapshot, update: str | None = None) -> str:
    if not snap.online:
        text = "Headroom - proxy offline"
    else:
        text = (
            f"Headroom - {format_tokens(snap.tokens_saved)} tokens, "
            f"~{format_usd(snap.usd_saved)} saved"
        )
    if update:
        text += f" - update {update} available"
    return text[:127]  # Windows tray tooltips are capped at 127 chars


def available_update() -> str | None:
    """Latest version when newer than the installed one (cache only, no network)."""
    try:
        from packaging.version import Version

        from headroom.update_check import (
            _in_docker,
            _is_source_checkout,
            installed_version,
            is_update_check_enabled,
            read_cache,
        )

        if not is_update_check_enabled() or _is_source_checkout() or _in_docker():
            return None
        cache = read_cache() or {}
        latest, current = cache.get("latest_version"), installed_version()
        if isinstance(latest, str) and current and Version(latest) > Version(current):
            return latest
    except Exception:
        pass
    return None
