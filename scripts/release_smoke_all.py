#!/usr/bin/env python
"""Run the local release artifact smoke suite.

This command is the one-stop local release gate for artifact packaging. It runs
the version preflight, then delegates to the npm and Python artifact smoke
builders so their detailed checks stay in one place.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
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


def run(args: list[str | os.PathLike[str]]) -> None:
    print("\n> " + " ".join(quote_arg(arg) for arg in args), flush=True)
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def ensure_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise SystemExit(f"output directory must be empty to avoid stale artifacts: {path}")


def parse_args() -> argparse.Namespace:
    version = load_project_version()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_out = ROOT / "release-assets-local" / f"all-{version}-{stamp}"

    parser = argparse.ArgumentParser(description="Run local npm and Python release smokes.")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--node", default=shutil.which("node") or "node")
    parser.add_argument(
        "--python-release",
        action="store_true",
        help="Run the Python smoke with maturin --release instead of the faster ci profile.",
    )
    parser.add_argument("--skip-npm", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out.resolve()
    version = load_project_version()

    if args.skip_npm and args.skip_python:
        raise SystemExit("nothing to run: --skip-npm and --skip-python were both set")

    ensure_empty_dir(out_dir)

    run([args.python, "scripts/verify-versions.py"])

    completed: list[tuple[str, Path]] = []
    if not args.skip_npm:
        npm_out = out_dir / "npm"
        run([args.node, "scripts/build_npm_release_assets.mjs", version, npm_out])
        completed.append(("npm", npm_out))

    if not args.skip_python:
        python_out = out_dir / "python"
        python_args: list[str | os.PathLike[str]] = [
            args.python,
            "scripts/build_python_release_smoke.py",
            "--out",
            python_out,
        ]
        if args.python_release:
            python_args.append("--release")
        run(python_args)
        completed.append(("python", python_out))

    print("\nLocal release smoke suite complete:")
    for name, path in completed:
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
