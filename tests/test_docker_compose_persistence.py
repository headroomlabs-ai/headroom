"""Regression checks for Docker Compose persistence wiring."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_top_level_compose_publishes_only_to_loopback() -> None:
    """Every published port must name an explicit loopback host IP.

    None of the three services authenticates by default: the proxy's /v1/*
    data plane is open without HEADROOM_PROXY_TOKEN, Qdrant has no API key,
    and NEO4J_AUTH falls back to a password published in the compose file. A
    bare "8787:8787" binds 0.0.0.0 on the host, so `docker compose up -d` on a
    shared network would expose all three.
    """
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    published = [
        (service, port)
        for service, spec in compose["services"].items()
        for port in spec.get("ports", [])
    ]
    assert published, "expected the compose file to publish at least one port"

    for service, port in published:
        # Short syntax is "HOST_IP:HOST_PORT:CONTAINER_PORT"; anything with
        # fewer than three segments is published on every interface.
        assert isinstance(port, str), f"{service}: expected short-syntax port, got {port!r}"
        segments = port.split(":")
        assert len(segments) == 3, f"{service}: port {port!r} publishes on all interfaces"
        assert segments[0] in {"127.0.0.1", "::1"}, (
            f"{service}: port {port!r} publishes on {segments[0]}, not loopback"
        )


def test_top_level_compose_pins_headroom_state_to_named_volume() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "- headroom_workspace:/home/nonroot/.headroom" in compose
    assert "- HOME=/home/nonroot" in compose
    assert "- HEADROOM_WORKSPACE_DIR=/home/nonroot/.headroom" in compose
    assert "- HEADROOM_CONFIG_DIR=/home/nonroot/.headroom/config" in compose


def test_top_level_compose_marks_source_build_version() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "HEADROOM_BUILD_VERSION: ${HEADROOM_BUILD_VERSION:-source-build}" in compose
    assert 'ARG HEADROOM_BUILD_VERSION=""' in dockerfile
    assert "ARG PYTHON_SITE_PACKAGES" in dockerfile
    assert "if not build_version:" in dockerfile
    assert "source-build+g{revision}" in dockerfile
    assert "source-build+sha256." in dockerfile
    assert "_build_info.py" in dockerfile
    assert "import headroom._version" not in dockerfile
    assert ".git/*" in dockerignore
    assert "!.git/HEAD" in dockerignore
    assert "!.git/refs/**" in dockerignore


def test_native_compose_keeps_host_publication_loopback_and_optional_token() -> None:
    compose = yaml.safe_load(
        (ROOT / "docker" / "docker-compose.native.yml").read_text(encoding="utf-8")
    )
    proxy = compose["services"]["proxy"]

    assert proxy["ports"] == ["127.0.0.1:${HEADROOM_PORT:-8787}:${HEADROOM_PORT:-8787}"]
    assert proxy["environment"]["HEADROOM_HOST"] == "0.0.0.0"
    assert proxy["environment"]["HEADROOM_PROXY_TOKEN"] == "${HEADROOM_PROXY_TOKEN:-}"
    assert proxy["command"] == ["--host", "0.0.0.0", "--port", "${HEADROOM_PORT:-8787}"]


def test_differential_capture_compose_keeps_proxy_internal_and_host_ports_loopback() -> None:
    compose = yaml.safe_load(
        (ROOT / "docker" / "differential-network-capture" / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
    )
    services = compose["services"]
    assert (
        services["headroom-proxy"]["environment"]["HEADROOM_PROXY_TOKEN"]
        == "${HEADROOM_PROXY_TOKEN:?Set HEADROOM_PROXY_TOKEN}"
    )
    assert (
        services["mitm-headroom-client"]["environment"]["CAPTURE_PROXY_TOKEN"]
        == "${HEADROOM_PROXY_TOKEN:?Set HEADROOM_PROXY_TOKEN}"
    )
    assert (
        services["mitm-headroom-client"]["environment"]["CAPTURE_PROXY_TOKEN_HOSTS"]
        == "headroom-proxy"
    )
    assert services["mitm-direct"]["ports"] == ["127.0.0.1:${DIRECT_MITM_PORT:-18080}:8080"]
    assert services["mitm-headroom-upstream"]["ports"] == [
        "127.0.0.1:${HEADROOM_MITM_PORT:-18081}:8080"
    ]
    assert "ports" not in services["mitm-headroom-client"]
    assert "HEADROOM_CLIENT_MITM_PORT" not in compose["services"]["mitm-headroom-client"].get(
        "environment", {}
    )


def test_published_dockerfile_stages_default_to_loopback() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.count("ENV HEADROOM_HOST=127.0.0.1") == 2
    assert dockerfile.count('CMD ["--port", "8787"]') == 2
