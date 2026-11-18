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


CONSTRAINED_INSTALL = re.compile(r"pip install\b[^\n]*?(?:-c|--constraints) /tmp/constraints\.txt")
MATURIN_SPEC = re.compile(r"maturin\s*([<>=!~]=?)")
TOOL_VERSION = {
    "maturin": re.compile(r"maturin==(\d[\w.]*)|ARG MATURIN_VERSION=(\d[\w.]*)"),
    "uv": re.compile(r"\buv==(\d[\w.]*)()"),
}


COPIED_REQUIREMENTS = re.compile(r"^COPY\s+((?:e2e|docker)/\S+\.txt)\s", re.M)
HASHED_INSTALL = re.compile(
    r"pip install (?:--no-cache-dir )?--require-hashes --only-binary :all: -r (/tmp/\S+\.txt)"
)


def _install_text(dockerfile: str) -> str:
    """The Dockerfile plus the requirement files it copies in and installs."""
    text = (ROOT / dockerfile).read_text(encoding="utf-8")
    copied = [
        (ROOT / path).read_text(encoding="utf-8") for path in COPIED_REQUIREMENTS.findall(text)
    ]
    return "\n".join([text, *copied])


def _tool_versions(text: str, tool: str) -> set[str]:
    return {a or b for a, b in TOOL_VERSION[tool].findall(text)}


@pytest.mark.parametrize("dockerfile", CI_DOCKERFILES)
def test_installs_use_exact_versions(dockerfile: str) -> None:
    text = _install_text(dockerfile)
    assert "--upgrade pip" not in text, f"{dockerfile}: pin pip instead of upgrading it"
    specs = MATURIN_SPEC.findall(text)
    assert specs and set(specs) == {"=="}, (
        f"{dockerfile}: maturin must be pinned with ==, got {specs}"
    )
    assert CONSTRAINED_INSTALL.search(text), (
        f"{dockerfile}: install the uv.lock dependency set as constraints"
    )


def test_build_tools_are_the_same_version_in_every_ci_dockerfile() -> None:
    for tool in TOOL_VERSION:
        found = {name: _tool_versions(_install_text(name), tool) for name in CI_DOCKERFILES}
        assert all(len(versions) == 1 for versions in found.values()), (tool, found)
        assert len(set().union(*found.values())) == 1, (tool, found)


@pytest.mark.parametrize(
    ("dockerfile", "hashed"),
    [
        ("Dockerfile", {"/tmp/uv.txt"}),
        ("e2e/init/Dockerfile", {"/tmp/build-tools.txt", "/tmp/pip.txt"}),
        ("e2e/wrap/Dockerfile", {"/tmp/build-tools.txt", "/tmp/pip.txt"}),
    ],
)
def test_tool_installs_check_wheel_hashes(dockerfile: str, hashed: set[str]) -> None:
    """uv, maturin and pip run inside these images before anything else is
    verified, so they come from wheels whose sha256 is committed here."""
    text = (ROOT / dockerfile).read_text(encoding="utf-8")
    assert not re.search(r"pip install [^\n]*\b(?:uv|maturin|pip)==", text), (
        f"{dockerfile}: install uv, maturin and pip from a hashed requirements file"
    )
    copied = {f"/tmp/{Path(path).name}": path for path in COPIED_REQUIREMENTS.findall(text)}
    installed = HASHED_INSTALL.findall(text)
    assert hashed <= set(installed), installed
    for target in installed:
        lines = [
            line
            for line in (ROOT / copied[target]).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert lines, copied[target]
        for line in lines:
            assert re.fullmatch(r"[a-z0-9-]+==[\w.]+( --hash=sha256:[0-9a-f]{64})+", line), (
                f"{copied[target]}: {line!r} needs an exact version and its sha256"
            )


def test_bind_security_probe_image_is_pinned_by_digest() -> None:
    script = (ROOT / "e2e/docker-bind-security.sh").read_text(encoding="utf-8")
    probes = re.findall(r"curlimages/curl:\S+", script)
    assert probes
    assert all("@sha256:" in probe for probe in probes), probes
