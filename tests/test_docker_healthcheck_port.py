"""Regression checks for port-aware container startup and healthchecks (#2432).

`headroom proxy --port` is a click option with `envvar="HEADROOM_PORT"`, and an
explicit CLI argument beats the envvar. So a baked `--port` in CMD and a
hardcoded probe port in HEALTHCHECK are two halves of the same defect: the
first makes HEADROOM_PORT unable to move the listener, the second makes the
probe unable to follow it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGE_RE = re.compile(r"^FROM\s+\S+\s+AS\s+(\S+)\s*$", re.IGNORECASE)


def _stage_body(dockerfile: str, stage: str) -> str:
    """Return the lines belonging to a single named build stage."""

    body: list[str] = []
    collecting = False
    for line in dockerfile.splitlines():
        match = STAGE_RE.match(line.strip())
        if match:
            collecting = match.group(1) == stage
            continue
        if collecting:
            body.append(line)
    assert body, f"stage {stage!r} not found in Dockerfile"
    return "\n".join(body)


def _healthcheck_directives(dockerfile: str) -> list[str]:
    """Collect every HEALTHCHECK directive, joining backslash continuations."""

    directives: list[str] = []
    lines = dockerfile.splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("HEALTHCHECK"):
            continue
        parts = [line.rstrip("\\").strip()]
        cursor = index
        while lines[cursor].rstrip().endswith("\\") and cursor + 1 < len(lines):
            cursor += 1
            parts.append(lines[cursor].rstrip("\\").strip())
        directives.append(" ".join(parts))
    return directives


def test_dockerfile_defines_a_healthcheck_per_runtime_stage() -> None:
    directives = _healthcheck_directives((ROOT / "Dockerfile").read_text(encoding="utf-8"))

    assert len(directives) == 2


def test_no_healthcheck_hardcodes_the_default_port() -> None:
    """A hardcoded probe port reports a working non-8787 deployment as unhealthy."""

    directives = _healthcheck_directives((ROOT / "Dockerfile").read_text(encoding="utf-8"))

    offenders = [d for d in directives if "127.0.0.1:8787" in d]
    assert offenders == []


def test_every_healthcheck_resolves_the_port_from_the_environment() -> None:
    directives = _healthcheck_directives((ROOT / "Dockerfile").read_text(encoding="utf-8"))

    assert directives, "expected at least one HEALTHCHECK directive"
    for directive in directives:
        assert "HEADROOM_PORT" in directive


def test_distroless_stage_healthcheck_stays_shell_free() -> None:
    """The distroless stage has no shell, so ${VAR} expansion is unavailable."""

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    directives = _healthcheck_directives(_stage_body(dockerfile, "runtime-slim"))

    assert len(directives) == 1
    assert "${HEADROOM_PORT" not in directives[0]
    # `or` and not a get() default, so an empty value falls back like `:-` does.
    assert "(os.environ.get('HEADROOM_PORT') or '8787')" in directives[0]


def test_debian_stage_healthcheck_expands_the_port_variable() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    directives = _healthcheck_directives(_stage_body(dockerfile, "runtime-slim-base"))

    assert len(directives) == 1
    assert "${HEADROOM_PORT:-8787}" in directives[0]


def test_no_stage_bakes_a_port_into_cmd() -> None:
    """A baked `--port` outranks HEADROOM_PORT and pins the listener to 8787."""

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    cmds = [line for line in dockerfile.splitlines() if line.startswith("CMD ")]

    assert cmds, "expected at least one CMD directive"
    assert not any("--port" in cmd for cmd in cmds)
