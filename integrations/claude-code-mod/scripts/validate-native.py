"""Explicit release gate: missing native tools are a FAILURE, never a skipped pass."""

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
claude = shutil.which("claude")
if claude is None:
    raise SystemExit(
        "NOT VALIDATED: Claude Code is not installed. Native validation/test gate remains open."
    )
result = subprocess.run(
    [claude, "--version"], capture_output=True, text=True, timeout=15, check=True
)
match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
if not match or tuple(map(int, match.groups())) < (2, 1, 287):
    raise SystemExit("Claude Code 2.1.287 or newer is required.")
if importlib.util.find_spec("headroom") is None:
    raise SystemExit("NOT VALIDATED: run from the Python environment that contains real Headroom.")
plugin = str(root / "plugins" / "headroom-sidebar")
for argv in (
    [claude, "plugin", "validate", plugin],
    [claude, "plugin", "test", plugin],
    [sys.executable, "-m", "pytest", "-q", "tests/test_real_headroom.py"],
):
    subprocess.run(argv, cwd=root / "companion" if argv[0] == sys.executable else root, check=True)
print(
    "Native static/runtime contracts passed. Complete the real authenticated-session checklist in docs/VALIDATION.md."
)
