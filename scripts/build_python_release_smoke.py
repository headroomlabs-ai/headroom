#!/usr/bin/env python
"""Build and smoke-test local Python release artifacts.

This is the local companion to the release workflow's wheel/sdist gates. It
builds a wheel with maturin, builds an sdist, validates the sdist License-File
metadata, installs the wheel into a clean virtual environment, and imports the
native `headroom._core` extension from that installed wheel.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def load_project_version() -> str:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def quote_arg(value: str | os.PathLike[str]) -> str:
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:=\\-]+", text):
        return text
    return f'"{text.replace(chr(34), chr(34) * 2)}"'


def run(
    args: list[str | os.PathLike[str]],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> None:
    print("\n> " + " ".join(quote_arg(arg) for arg in args), flush=True)
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def ensure_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise SystemExit(f"output directory must be empty to avoid stale artifacts: {path}")


def build_artifacts(out_dir: Path, python_exe: str, profile: str, release: bool) -> None:
    env = os.environ.copy()
    env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")

    run([python_exe, "-m", "maturin", "--version"], env=env)

    build_args = [
        python_exe,
        "-m",
        "maturin",
        "build",
        "--out",
        out_dir,
        "--interpreter",
        python_exe,
    ]
    if release:
        build_args.append("--release")
    else:
        build_args.extend(["--profile", profile])
    run(build_args, env=env)

    run([python_exe, "-m", "maturin", "sdist", "--out", out_dir], env=env)


def find_one_artifact(out_dir: Path, pattern: str) -> Path:
    matches = sorted(out_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {out_dir}, found {len(matches)}")
    return matches[0]


def verify_wheel(wheel: Path, expected_version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        native_members = [
            name
            for name in names
            if name.startswith("headroom/_core") and Path(name).suffix.lower() in {".pyd", ".so"}
        ]
        if not native_members:
            raise SystemExit(f"{wheel.name} does not contain headroom/_core native extension")

        metadata_members = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_members) != 1:
            raise SystemExit(
                f"{wheel.name} should contain exactly one dist-info/METADATA, "
                f"found {len(metadata_members)}"
            )
        metadata = archive.read(metadata_members[0]).decode("utf-8")

    expected_line = f"Version: {expected_version}"
    if expected_line not in metadata.splitlines():
        raise SystemExit(f"{wheel.name} metadata missing {expected_line!r}")

    print(f"wheel metadata OK: {wheel.name} contains {native_members[0]}")


def verify_sdist_license_files(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(roots) != 1:
            raise SystemExit(f"expected one sdist root directory, found {sorted(roots)}")
        root = roots.pop()

        pkg_info_path = f"{root}/PKG-INFO"
        member = archive.getmember(pkg_info_path)
        fh = archive.extractfile(member)
        if fh is None:
            raise SystemExit(f"could not read {pkg_info_path} from {sdist.name}")
        pkg_info = fh.read().decode("utf-8")

    declared = []
    for line in pkg_info.splitlines():
        if not line.strip():
            break
        if line.startswith("License-File:"):
            declared.append(line.split(":", 1)[1].strip())

    if not declared:
        raise SystemExit(f"{sdist.name} declares no License-File entries")

    missing = [name for name in declared if f"{root}/{name}" not in names]
    if missing:
        raise SystemExit(
            f"{sdist.name} declares License-File entries missing from tarball: {missing}"
        )

    print(f"sdist License-File metadata OK: {declared}")


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def smoke_install_wheel(wheel: Path, python_exe: str, expected_version: str) -> None:
    with tempfile.TemporaryDirectory(prefix="headroom-python-smoke-") as tmp:
        venv_dir = Path(tmp) / "venv"
        run([python_exe, "-m", "venv", venv_dir])
        smoke_python = venv_python(venv_dir)

        run(
            [
                smoke_python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-warn-script-location",
                wheel,
            ]
        )

        smoke_code = f"""
import importlib.metadata as metadata
import headroom
from headroom._core import DiffCompressor, SmartCrusher, hello

version = metadata.version("headroom-ai")
assert version == {expected_version!r}, version
assert headroom.__version__ == {expected_version!r}, headroom.__version__
print(f"smoke-import OK: version={{version}} hello={{hello()}} diff={{DiffCompressor!r}} smart={{SmartCrusher!r}}")
"""
        import_cwd = Path(tmp) / "import-cwd"
        import_cwd.mkdir()
        run([smoke_python, "-c", smoke_code], cwd=import_cwd)


def parse_args() -> argparse.Namespace:
    version = load_project_version()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_out = ROOT / "release-assets-local" / f"python-{version}-{stamp}"

    parser = argparse.ArgumentParser(
        description="Build and smoke-test local Headroom Python release artifacts."
    )
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--profile",
        default="ci",
        help="Cargo profile for local wheel smoke builds; ignored with --release.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Use maturin --release instead of the faster local smoke profile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out.resolve()
    python_exe = shutil.which(args.python) or args.python
    expected_version = load_project_version()

    ensure_empty_dir(out_dir)
    build_artifacts(out_dir, python_exe, args.profile, args.release)

    wheel = find_one_artifact(out_dir, "headroom_ai-*.whl")
    sdist = find_one_artifact(out_dir, "headroom_ai-*.tar.gz")
    verify_wheel(wheel, expected_version)
    verify_sdist_license_files(sdist)
    smoke_install_wheel(wheel, python_exe, expected_version)

    print(f"\nBuilt and verified Python release artifacts in {out_dir}")


if __name__ == "__main__":
    main()
