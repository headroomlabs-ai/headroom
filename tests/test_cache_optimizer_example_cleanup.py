"""The actual SDK loopback example must not leave a global scratch SQLite store."""

import os
import subprocess
import sys
from pathlib import Path


def test_cache_optimizer_example_keeps_ccr_in_memory() -> None:
    repo = Path(__file__).resolve().parents[1]
    probe = """
import runpy
import tempfile
from pathlib import Path

original = tempfile.TemporaryDirectory
class GuardedTemporaryDirectory(original):
    def __exit__(self, *args):
        databases = list(Path(self.name).rglob("ccr_store.db"))
        try:
            assert not databases, "Example opened a process-global scratch CCR SQLite database"
        finally:
            super().__exit__(*args)

tempfile.TemporaryDirectory = GuardedTemporaryDirectory
runpy.run_path("examples/cache_optimizer_config_local.py", run_name="__main__")
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(repo),
        "HEADROOM_CCR_BACKEND": "sqlite",
        "HEADROOM_COMPRESS_SESSION_STORE": "",
        "HEADROOM_OFFLINE": "1",
        "HEADROOM_BEACON": "off",
        "DO_NOT_TRACK": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
