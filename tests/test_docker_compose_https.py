"""The HTTPS compose file keeps Headroom behind Caddy and behind its proxy token."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

DOCKER = Path(__file__).resolve().parents[1] / "docker"


def _services() -> dict:
    compose = yaml.safe_load((DOCKER / "docker-compose.https.yml").read_text(encoding="utf-8"))
    return compose["services"]


def test_only_caddy_is_published_and_headroom_stays_on_the_compose_network() -> None:
    services = _services()
    assert "ports" not in services["headroom"]
    assert services["headroom"]["expose"] == ["8787"]
    assert [p.split(":")[-1] for p in services["caddy"]["ports"]] == ["80", "443"]


def test_headroom_requires_the_proxy_token_and_never_acknowledges_an_open_bind() -> None:
    env = _services()["headroom"]["environment"]
    assert env["HEADROOM_PROXY_TOKEN"].startswith("${HEADROOM_PROXY_TOKEN:?")
    assert "HEADROOM_ALLOW_UNAUTHENTICATED_BIND" not in env


def test_caddy_forwards_to_the_headroom_service_not_loopback() -> None:
    """Headroom skips the token for loopback peers, so Caddy must reach it by
    service name from its own container, never through 127.0.0.1."""
    caddyfile = (DOCKER / "Caddyfile").read_text(encoding="utf-8")
    targets = re.findall(r"^\s*reverse_proxy\s+(\S+)", caddyfile, re.MULTILINE)
    assert targets == ["headroom:8787"]
    assert "network_mode" not in _services()["caddy"]
