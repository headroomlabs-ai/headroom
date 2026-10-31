"""Build a distribution wheel using the companion's declared build-system requirements."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        str(root / "companion"),
        "-w",
        str(root / "dist"),
    ],
    check=True,
)
