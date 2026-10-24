"""Security overrides reach every install the project controls.

fastembed caps pillow <13 while the decoder fixes are only in 13, so uv.lock
takes 13 through [tool.uv] override-dependencies. A lock refresh, or a build
step that installs from package metadata, would silently drop it.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]


def _overrides() -> list[str]:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["uv"]["override-dependencies"]


def test_the_lock_takes_the_fixed_pillow_on_supported_pythons():
    with open(ROOT / "uv.lock", "rb") as f:
        versions = [p["version"] for p in tomllib.load(f)["package"] if p["name"] == "pillow"]
    assert any(int(v.split(".")[0]) >= 13 for v in versions), versions
    assert any(o.startswith("pillow>=13") for o in _overrides())


def test_the_image_build_applies_the_overrides():
    # `uv pip install` ignores [tool.uv] override-dependencies; the Dockerfile
    # must pass them, or the image installs pillow 12 under fastembed's cap.
    dockerfile = (ROOT / "Dockerfile").read_text()
    install = re.search(r"uv pip install --system (.*)", dockerfile)
    assert install and "--overrides" in install.group(1)
    assert "override-dependencies" in dockerfile
