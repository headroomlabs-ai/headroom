"""Dependency contract between Rust ort API 24 and pip ONNX Runtime."""

from __future__ import annotations

from pathlib import Path

import tomllib
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]


def test_shipping_ort_dependencies_match_supported_platforms() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    optional = project["optional-dependencies"]

    cases = [
        ("3.11", "darwin", "x86_64", SpecifierSet(">=1.16.0,<1.24.0")),
        ("3.13", "darwin", "x86_64", SpecifierSet(">=1.16.0,<1.24.0")),
        ("3.14", "darwin", "x86_64", None),
        ("3.15", "darwin", "x86_64", None),
        ("3.14", "darwin", "arm64", SpecifierSet(">=1.24.0")),
        ("3.14", "linux", "x86_64", SpecifierSet(">=1.24.0")),
        ("3.11", "darwin", "arm64", SpecifierSet(">=1.24.0")),
        ("3.11", "linux", "x86_64", SpecifierSet(">=1.24.0")),
        ("3.10", "darwin", "x86_64", SpecifierSet(">=1.16.0,<1.24.0")),
        ("3.10", "linux", "x86_64", SpecifierSet(">=1.16.0,<1.24.0")),
    ]

    for extra in ("proxy", "voice"):
        requirements = [
            Requirement(value)
            for value in optional[extra]
            if Requirement(value).name == "onnxruntime"
        ]
        assert len(requirements) == 2
        for python_version, sys_platform, platform_machine, expected in cases:
            environment = default_environment()
            environment.update(
                python_version=python_version,
                python_full_version=f"{python_version}.0",
                sys_platform=sys_platform,
                platform_machine=platform_machine,
                extra="",
            )
            selected = [
                req for req in requirements if req.marker and req.marker.evaluate(environment)
            ]
            if expected is None:
                assert not selected
            else:
                assert len(selected) == 1
                assert selected[0].specifier == expected


def test_intel_macos_python314_omits_ort_dependent_optional_backends() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    environment = default_environment()
    environment.update(
        python_version="3.14",
        python_full_version="3.14.0",
        sys_platform="darwin",
        platform_machine="x86_64",
        extra="",
    )
    ort_backends = {"onnxruntime", "magika", "fastembed", "rapidocr", "rapidocr-onnxruntime"}
    for extra in ("proxy", "voice", "relevance", "image"):
        selected = {
            req.name
            for raw in project["optional-dependencies"][extra]
            if (req := Requirement(raw)).marker is None or req.marker.evaluate(environment)
        }
        assert not selected & ort_backends, (extra, selected & ort_backends)


def test_all_extra_includes_proxy_and_voice() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    all_requirements = project["optional-dependencies"]["all"]

    assert any(
        requirement.name == "headroom-ai" and {"proxy", "voice"} <= set(requirement.extras)
        for raw_requirement in all_requirements
        if (requirement := Requirement(raw_requirement))
    )
