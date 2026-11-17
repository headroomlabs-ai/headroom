"""CI must build from base images pinned by digest, with exact install versions.

A tag such as ``python:3.11-slim`` can be re-pushed to point at a different
image; ``tag@sha256:...`` cannot. These Dockerfiles are the ones CI builds
(docker-native-e2e, init-e2e, wrap-e2e), so an unpinned base or a ranged
install there changes what CI runs without a commit in this repo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CI_DOCKERFILES = ("Dockerfile", "e2e/init/Dockerfile", "e2e/wrap/Dockerfile")
DIGEST = re.compile(r"@(sha256:[0-9a-f]{64}|\$\{[A-Z_]+_DIGEST\})")
FROM = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+(?i:AS)\s+(\S+))?", re.M)
ARG_DEFAULT = re.compile(r"^ARG\s+([A-Z_]+)=(\S+)", re.M)


def _resolve(image: str, args: dict[str, str]) -> str:
    match = re.fullmatch(r"\$\{([A-Z_]+)\}", image)
    return args.get(match.group(1), image) if match else image


@pytest.mark.parametrize("dockerfile", CI_DOCKERFILES)
def test_every_external_base_image_is_pinned_by_digest(dockerfile: str) -> None:
    text = (ROOT / dockerfile).read_text(encoding="utf-8")
    args = dict(ARG_DEFAULT.findall(text))
    stages: set[str] = set()
    unpinned = []
    for image, stage in FROM.findall(text):
        resolved = _resolve(image, args)
        if resolved not in stages and not DIGEST.search(resolved):
            unpinned.append(image)
        if stage:
            stages.add(stage)
    assert not unpinned, f"{dockerfile}: base images without a digest: {unpinned}"
    for name in re.findall(r"\$\{([A-Z_]+_DIGEST)\}", text):
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", args.get(name, "")), (
            f"{dockerfile}: ARG {name} needs a sha256 default"
        )


@pytest.mark.parametrize("dockerfile", CI_DOCKERFILES)
def test_installs_use_exact_versions(dockerfile: str) -> None:
    text = (ROOT / dockerfile).read_text(encoding="utf-8")
    assert "--upgrade pip" not in text, f"{dockerfile}: pin pip instead of upgrading it"
    assert "maturin>=" not in text, f"{dockerfile}: pin maturin to one version"
    assert "--constraints /tmp/constraints.txt" in text or "-c /tmp/constraints.txt" in text, (
        f"{dockerfile}: install the uv.lock dependency set as constraints"
    )


def test_bind_security_probe_image_is_pinned_by_digest() -> None:
    script = (ROOT / "e2e/docker-bind-security.sh").read_text(encoding="utf-8")
    probes = re.findall(r"curlimages/curl:\S+", script)
    assert probes
    assert all("@sha256:" in probe for probe in probes), probes
