"""Verify and smoke-test one exact locally built Headroom wheel."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    wheel = args.wheel.resolve(strict=True)
    actual = _sha256(wheel)
    expected = args.expected_sha256.casefold()
    if actual != expected:
        print(
            f"wheel digest mismatch: expected {expected}, got {actual}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="headroom-wheel-verify-") as directory:
        root = Path(directory)
        environment_path = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment_path)
        python = environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [str(python), "-m", "pip", "install", f"{wheel}[proxy]"],
            cwd=root,
            env=environment,
            check=True,
        )
        subprocess.run(
            [str(python), "-I", "-m", "headroom.cli", "proxy", "--help"],
            cwd=root,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        probe = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "from headroom.proxy.gateway.runtime import GatewayRuntime; print('gateway-ok')",
            ],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        if probe.stdout.strip() != "gateway-ok":
            print("installed gateway import smoke failed", file=sys.stderr)
            return 3

    print(f"verified wheel sha256={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
