"""Dependency contract between Rust ort API 24 and pip ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def test_shipping_ort_dependencies_require_api24_where_wheels_exist() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    optional = project["optional-dependencies"]

    for extra in ("proxy", "voice"):
        requirements = [
            Requirement(value)
            for value in optional[extra]
            if Requirement(value).name == "onnxruntime"
        ]
        assert len(requirements) == 2
        modern = next(
            req
            for req in requirements
            if req.marker and req.marker.evaluate({"python_version": "3.11"})
        )
        legacy = next(
            req
            for req in requirements
            if req.marker and req.marker.evaluate({"python_version": "3.10"})
        )
        assert modern.specifier.contains("1.24.0")
        assert not legacy.specifier.contains("1.24.0")
