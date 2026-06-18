"""litellm must be optional on Python 3.14 (GH #956).

litellm's metadata pins requires-python <3.14, so a hard dependency makes
`pip install headroom-ai` unsatisfiable on 3.14. headroom only uses litellm for
model registry / pricing / non-core providers (all lazily imported and
ImportError-guarded), so every litellm requirement must carry a marker that
skips it on 3.14, and the core paths must degrade gracefully without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _litellm_requirements() -> list[Requirement]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specs = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    return [r for spec in specs if (r := Requirement(spec)).name == "litellm"]


def test_every_litellm_requirement_is_skipped_on_py314() -> None:
    reqs = _litellm_requirements()
    assert reqs, "expected litellm to be declared in pyproject"
    for r in reqs:
        assert r.marker is not None, f"{r}: litellm must carry a python_version marker (GH #956)"
        assert not r.marker.evaluate({"python_version": "3.14"}), (
            f"{r}: must be skipped on Python 3.14"
        )
        assert r.marker.evaluate({"python_version": "3.13"}), (
            f"{r}: must still install on Python 3.13"
        )


def test_proxy_cost_degrades_without_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    # With litellm absent (its state on 3.14), the proxy cost path must return
    # None rather than raise.
    from headroom.proxy import cost

    monkeypatch.setattr(cost, "LITELLM_AVAILABLE", False)
    monkeypatch.setattr(cost, "litellm", None)
    assert cost._get_litellm_module() is None
