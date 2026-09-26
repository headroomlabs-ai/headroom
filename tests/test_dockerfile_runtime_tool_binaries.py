"""Regression checks for external tool binaries in the published runtime images.

The runtime stages copy site-packages plus a hand-maintained whitelist of
executables out of the builder. A PyPI dependency that ships a binary installs
it through the wheel's ``.data/scripts`` scheme, so it lands in
``/usr/local/bin`` and never inside site-packages: the site-packages COPY
brings the dist-info across but leaves the executable behind.

Nothing fails at build time, so the gap stays invisible until a feature that
needs the tool refuses to start. That is how ``ast-grep`` shipped missing from
``ghcr.io/headroomlabs-ai/headroom`` while ``pip list`` still reported
``ast-grep-cli`` as installed, which broke
``headroom proxy --intercept-tool-results`` (issue #3649).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
TOOLS_REGISTRY = ROOT / "headroom" / "tools.json"

_FROM_RE = re.compile(r"^FROM\s+\S+\s+AS\s+(\S+)\s*$", re.IGNORECASE)
_COPY_FROM_BUILDER = "COPY --from=builder"
_SCRIPTS_DIR = "/usr/local/bin"


def _stages() -> dict[str, str]:
    """Return ``{stage name: stage body}`` for every ``FROM`` stage in the Dockerfile."""
    stages: dict[str, list[str]] = {}
    current: str | None = None
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        match = _FROM_RE.match(line)
        if match:
            current = match.group(1)
            stages[current] = []
        elif current is not None:
            stages[current].append(line)
    return {name: "\n".join(body) for name, body in stages.items()}


def _builder_backed_stages() -> dict[str, str]:
    """Return the stages that copy artifacts out of the builder stage.

    The final ``runtime`` stage is an alias of ``runtime-slim-base`` and
    inherits its filesystem, so it is not listed here and is covered
    transitively.
    """
    return {name: body for name, body in _stages().items() if _COPY_FROM_BUILDER in body}


def _pip_only_tools() -> dict[str, str]:
    """Return registry tools installed by the interpreter, mapped to binary name.

    Mirrors ``headroom.binaries._is_pypi_tool``: a tool ships with the
    interpreter rather than being fetched at runtime when its version is
    ``"pypi"`` or it declares no download assets. Those are exactly the tools
    the image has to provide, because ``binaries.resolve()`` has no download to
    fall back to for them.
    """
    registry = json.loads(TOOLS_REGISTRY.read_text(encoding="utf-8"))["tools"]
    return {
        name: entry.get("binary", name)
        for name, entry in registry.items()
        if entry.get("version") == "pypi" or not entry.get("assets")
    }


def test_registry_still_declares_a_pip_only_tool() -> None:
    """Guard the copy check below from going vacuous.

    If every registry tool ever gains download assets there is nothing left to
    copy, and the copy check would pass for the wrong reason.
    """
    assert _pip_only_tools(), (
        "expected at least one tool installed via PyPI; if they all became "
        "downloadable, update this test rather than deleting it"
    )


def test_builder_backed_stages_copy_every_pip_only_tool_binary() -> None:
    """A tool that cannot be downloaded at runtime must be copied out of the builder.

    Otherwise it resolves to nothing inside the image: ``binaries._path_lookup()``
    only checks ``PATH`` and ``sys.prefix/bin`` (which is ``/usr/bin`` for the
    ``python:slim`` system interpreter), while the wheel installs the
    executable into ``/usr/local/bin``.
    """
    pip_only = _pip_only_tools()
    stages = _builder_backed_stages()
    assert stages, f"expected at least one stage to copy from the builder in {DOCKERFILE.name}"

    for name, body in stages.items():
        copied = body.splitlines()
        for tool, binary in pip_only.items():
            expected = f"{_COPY_FROM_BUILDER} {_SCRIPTS_DIR}/{binary} {_SCRIPTS_DIR}/{binary}"
            assert expected in copied, (
                f"stage {name!r} does not copy the {tool} binary out of the builder; "
                f"expected a line `{expected}`. {DOCKERFILE.name} has to ship every tool "
                "that cannot be downloaded at runtime."
            )


def test_sg_alias_is_not_copied_into_the_runtime_stages() -> None:
    """The ast-grep wheel also installs an ``sg`` alias; copying it would shadow /usr/bin/sg.

    The Debian base images ship shadow-utils' ``set-group`` at ``/usr/bin/sg``
    and ``/usr/local/bin`` precedes it on ``PATH``, so an ``sg`` there would
    silently replace ``set-group`` for every shell in the container. headroom
    resolves the tool by the name ``ast-grep``, so the alias buys nothing.
    """
    for name, body in _builder_backed_stages().items():
        copied = re.search(rf"^COPY .*{_SCRIPTS_DIR}/sg\b", body, re.MULTILINE)
        assert copied is None, (
            f"stage {name!r} copies the ast-grep `sg` alias, which shadows "
            "/usr/bin/sg (shadow-utils set-group)"
        )
