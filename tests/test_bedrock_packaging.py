"""Packaging guard for the Bedrock optional dependency.

``headroom wrap claude`` auto-enables ``--bedrock-sign`` whenever
``CLAUDE_CODE_USE_BEDROCK=1`` is set, and the signer imports ``boto3`` on the
first signed request. ``boto3`` ships in the ``bedrock`` extra. These tests pin
the dependency story so the turnkey path can never silently ship without it:

1. The ``bedrock`` extra exists and declares ``boto3``.
2. The comprehensive ``all`` extra references ``bedrock`` — so
   ``pip install headroom-ai[all]`` includes the signer's dependency. This is
   the regression the reviewer flagged: ``all`` had previously omitted it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only hit on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


def _find_pyproject() -> Path | None:
    """Walk up from this file to locate the repo's pyproject.toml.

    Returns ``None`` when running from an installed wheel without the source
    tree (e.g. some CI smoke environments), so the test skips rather than fails.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


@pytest.fixture(scope="module")
def optional_dependencies() -> dict[str, list[str]]:
    pyproject = _find_pyproject()
    if pyproject is None:
        pytest.skip("pyproject.toml not available (installed without source tree)")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_bedrock_extra_declares_boto3(optional_dependencies: dict[str, list[str]]) -> None:
    assert "bedrock" in optional_dependencies, "the [bedrock] extra must exist"
    bedrock = optional_dependencies["bedrock"]
    assert any(req.replace(" ", "").lower().startswith("boto3") for req in bedrock), (
        f"[bedrock] must declare boto3, got: {bedrock}"
    )


def test_all_extra_includes_bedrock(optional_dependencies: dict[str, list[str]]) -> None:
    """``all`` must pull in ``bedrock`` so the comprehensive install ships boto3.

    Guards against silently dropping ``bedrock`` from the self-referential
    ``headroom-ai[...]`` aggregate again.
    """
    all_reqs = optional_dependencies["all"]
    referenced_extras: set[str] = set()
    for req in all_reqs:
        # e.g. "headroom-ai[proxy,code,...,bedrock]"
        if "[" in req and "]" in req:
            inner = req[req.index("[") + 1 : req.index("]")]
            referenced_extras.update(part.strip() for part in inner.split(","))
    assert "bedrock" in referenced_extras, (
        "the [all] extra must reference [bedrock] so headroom-ai[all] ships boto3; "
        f"referenced extras: {sorted(referenced_extras)}"
    )
