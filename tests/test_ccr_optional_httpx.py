"""Core compression must not require httpx.

httpx is only declared under the [proxy]/[mcp]/[dev] extras, not the base
`headroom-ai` install. Before this fix, headroom.ccr unconditionally
imported batch_processor (which imports httpx at module level), and
SmartCrusher — documented as core, lightweight compression — reaches
CCR_TOOL_NAME through headroom.ccr's __init__, so a plain `pip install
headroom-ai` with no extras broke SmartCrusher/compress() entirely with an
uncaught ImportError.

Each check runs in a fresh subprocess with httpx import blocked at the
meta-path level, rather than monkeypatching this process's sys.modules:
mutating already-imported headroom modules in place corrupts identity-keyed
state (e.g. transform registries) for every test that runs afterwards in
the same session.
"""

from __future__ import annotations

import subprocess
import sys

_BLOCK_HTTPX = """
import sys

class _BlockHttpx:
    def find_spec(self, name, path, target=None):
        if name == "httpx" or name.startswith("httpx."):
            raise ImportError("No module named 'httpx' (blocked for this test)")
        return None

sys.meta_path.insert(0, _BlockHttpx())
"""


def _run_with_httpx_blocked(import_line: str) -> subprocess.CompletedProcess[str]:
    script = _BLOCK_HTTPX + "\n" + import_line
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_ccr_package_imports_without_httpx() -> None:
    result = _run_with_httpx_blocked(
        "import headroom.ccr as m\n"
        "assert m.BATCH_PROCESSING_AVAILABLE is False, m.BATCH_PROCESSING_AVAILABLE\n"
        "assert m.BatchResultProcessor is None\n"
        "assert m.CCR_TOOL_NAME\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_smart_crusher_imports_without_httpx() -> None:
    # Regression check: this raised ModuleNotFoundError: No module named
    # 'httpx' before the ccr/__init__.py fix, even on a plain
    # `pip install headroom-ai` with no extras.
    result = _run_with_httpx_blocked(
        "import headroom.transforms.smart_crusher as m\n"
        "assert m.SmartCrusher is not None\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
