"""Regression checks for Docker Compose persistence wiring."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
