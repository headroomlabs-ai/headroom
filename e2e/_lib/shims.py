"""Cross-platform agent binary shim factory for e2e tests.

A "shim" is a tiny executable with a given name (e.g. `claude`, `codex`) that
the harness drops into a temporary directory and prepends to PATH. It lets
tests drive `headroom init` without requiring a real Claude/Codex install.

Three behaviors are supported:

* ``noop``         — exits 0 with no output. Default.
* ``fail``         — exits 1 with a short stderr message.
* ``record-args``  — appends a JSON record of (tool, argv, cwd) to the file at
                     ``$HEADROOM_E2E_SHIM_LOG``, then exits 0. Useful for
                     asserting that `init claude` invoked
                     `claude plugin install` with the right arguments.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Literal

ShimBehavior = Literal["noop", "fail", "record-args"]

_NOOP_SH = """#!/usr/bin/env bash
exit 0
"""

_FAIL_SH = """#!/usr/bin/env bash
echo "${0##*/}: simulated failure" >&2
exit 1
"""

_RECORD_SH = """#!/usr/bin/env bash
tool="${0##*/}"
log="${HEADROOM_E2E_SHIM_LOG:-/dev/null}"
mkdir -p "$(dirname "$log")" 2>/dev/null || true
python3 - "$tool" "$log" "$@" <<'PY'
import json, os, sys
tool, log, *argv = sys.argv[1:]
record = {"tool": tool, "argv": argv, "cwd": os.getcwd()}
if log != "/dev/null":
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\\n")
print(f"{tool} shim executed")
PY
exit 0
"""

# Windows equivalents. Use `.cmd` so `shutil.which` and PATHEXT find them.
_NOOP_CMD = "@echo off\r\nexit /b 0\r\n"

_FAIL_CMD = "@echo off\r\necho %~n0: simulated failure 1>&2\r\nexit /b 1\r\n"

_RECORD_CMD = (
    "@echo off\r\n"
    "setlocal\r\n"
    'if "%HEADROOM_E2E_SHIM_LOG%"=="" set HEADROOM_E2E_SHIM_LOG=NUL\r\n'
    "python -c \"import json,os,sys; name=r'%~n0'; log=os.environ['HEADROOM_E2E_SHIM_LOG']; "
    "rec={'tool':name,'argv':sys.argv[1:],'cwd':os.getcwd()};\r\n"
    "open(log,'a',encoding='utf-8').write(json.dumps(rec)+chr(10)) if log!='NUL' else None;\r\n"
    "print(f'{name} shim executed')\" %*\r\n"
    "exit /b 0\r\n"
)


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform == "win32"


def make_shim(name: str, dir: Path, behavior: ShimBehavior = "noop") -> Path:
    """Create an executable shim named ``name`` inside ``dir``.

    Returns the absolute path to the created shim. On POSIX this is a ``.sh``
    file made executable and named without extension (so ``shutil.which(name)``
    finds it). On Windows this is a ``.cmd`` file — again, ``shutil.which``
    honours ``PATHEXT`` and will find it.
    """

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    if _is_windows():
        body = {"noop": _NOOP_CMD, "fail": _FAIL_CMD, "record-args": _RECORD_CMD}[behavior]
        path = dir / f"{name}.cmd"
        path.write_text(body, encoding="utf-8")
        return path

    body = {"noop": _NOOP_SH, "fail": _FAIL_SH, "record-args": _RECORD_SH}[behavior]
    path = dir / name
    path.write_text(body, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path
