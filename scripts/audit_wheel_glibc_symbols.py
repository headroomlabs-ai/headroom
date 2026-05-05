#!/usr/bin/env python3
"""Audit a manylinux Python wheel for glibc symbol references that exceed
the wheel's declared manylinux ABI floor.

This catches the bug class from issue #355: a wheel tagged
`manylinux_2_28_x86_64` whose `_core.so` references `__isoc23_strtoll`
(introduced in glibc 2.38) — the link succeeds because the manylinux
build host has 2.38+, but every end user with glibc < 2.38 sees an
`ImportError: undefined symbol` at `import headroom._core`.

How it works
------------

1. Parse the wheel filename to extract the manylinux tag
   (`manylinux_2_28` → glibc floor 2.28).
2. Run `objdump -T` (or `nm -D`) on every `.so` inside the wheel.
3. For every `UND` symbol, check whether it's allowed at the glibc
   floor. Allowed:
   - Symbols with a `GLIBC_x.y` version tag where `x.y <= floor`.
   - Symbols defined by us locally (i.e. NOT marked `UND`).
4. Reject any `UND` symbol that's:
   - Tagged with a GLIBC version > floor.
   - Or in the `__isoc23_*` family (no version tag, but introduced
     in glibc 2.38 — special-cased).
   - Or in any other known "introduced after floor" symbol list
     (extensible).

Exit 0 = wheel is portable. Exit 1 = wheel will break on some users.

Usage
-----

    scripts/audit_wheel_glibc_symbols.py path/to/headroom_ai-*.whl

Run on Linux only — `objdump` from binutils is the audit tool. macOS's
default `objdump` is llvm-objdump, also works on ELF.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Symbol families introduced after specific glibc versions, beyond what
# `auditwheel` already checks. Add here as new bug classes surface.
#
# Each entry is `(symbol_name_prefix, min_glibc_version_introduced, justification_url)`.
POST_FLOOR_SYMBOLS = [
    (
        "__isoc23_",
        (2, 38),
        "https://sourceware.org/glibc/wiki/Release/2.38",
    ),
]


def parse_manylinux_floor(wheel_filename: str) -> tuple[int, int] | None:
    """Extract glibc floor from a manylinux wheel filename.

    `headroom_ai-0.20.26-cp312-cp312-manylinux_2_28_x86_64.whl` → (2, 28).
    Returns None for non-manylinux wheels (macOS, Windows, sdist).
    """
    m = re.search(r"manylinux_(\d+)_(\d+)_", wheel_filename)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # `manylinux2014_x86_64` is the legacy alias for manylinux_2_17.
    if "manylinux2014_" in wheel_filename:
        return (2, 17)
    if "manylinux1_" in wheel_filename:
        return (2, 5)
    return None


def list_undef_symbols(so_path: Path) -> list[tuple[str, str]]:
    """Return [(symbol_name, glibc_version_or_empty), ...] for every
    UND (undefined) dynamic symbol in `so_path`.
    """
    objdump = shutil.which("objdump") or shutil.which("llvm-objdump")
    if not objdump:
        raise RuntimeError(
            "neither `objdump` nor `llvm-objdump` is on PATH; "
            "install binutils (Linux) or LLVM (macOS) to run this audit"
        )
    out = subprocess.run(
        [objdump, "-T", str(so_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found = []
    for line in out.splitlines():
        # `objdump -T` lines: `address SECTION ... NAME` where SECTION
        # contains `*UND*` for undefined references and a versioned
        # symbol name like `__isoc23_strtoll@GLIBC_2.38` or unversioned.
        if "*UND*" not in line:
            continue
        # The last whitespace-separated token is the (versioned) symbol.
        token = line.split()[-1]
        if "@" in token:
            name, _, ver = token.partition("@")
            # `@@` indicates the default version; strip the second `@`.
            ver = ver.lstrip("@")
        else:
            name, ver = token, ""
        found.append((name, ver))
    return found


def glibc_version_from_token(ver: str) -> tuple[int, int] | None:
    """`GLIBC_2.28` → (2, 28). Returns None for non-GLIBC tokens."""
    m = re.match(r"^GLIBC_(\d+)\.(\d+)", ver)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def audit_so(so_path: Path, floor: tuple[int, int]) -> list[str]:
    """Return a list of human-readable violation strings."""
    violations: list[str] = []
    for name, ver in list_undef_symbols(so_path):
        v = glibc_version_from_token(ver)
        if v is not None and v > floor:
            violations.append(
                f"  {name}@{ver} requires glibc {v[0]}.{v[1]} > floor {floor[0]}.{floor[1]}"
            )
            continue
        # Versionless symbols: cross-check the post-floor list.
        for prefix, introduced, url in POST_FLOOR_SYMBOLS:
            if name.startswith(prefix) and introduced > floor:
                violations.append(
                    f"  {name} (no version tag, introduced in glibc "
                    f"{introduced[0]}.{introduced[1]} > floor "
                    f"{floor[0]}.{floor[1]} — see {url})"
                )
                break
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="path to the .whl to audit")
    args = parser.parse_args()

    wheel: Path = args.wheel
    if not wheel.exists():
        print(f"ERROR: wheel not found: {wheel}", file=sys.stderr)
        return 1

    floor = parse_manylinux_floor(wheel.name)
    if floor is None:
        print(
            f"OK: {wheel.name} is not a manylinux wheel; nothing to audit "
            "(macOS / Windows / sdist run on a different runtime ABI)."
        )
        return 0

    print(f"Auditing {wheel.name} (glibc floor: {floor[0]}.{floor[1]})")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        with zipfile.ZipFile(wheel) as zf:
            so_members = [m for m in zf.namelist() if m.endswith(".so")]
            if not so_members:
                print(
                    f"WARN: {wheel.name} contains no .so files; "
                    "nothing to audit (pure-Python wheel?)"
                )
                return 0
            zf.extractall(td_path, members=so_members)

        all_violations: list[tuple[str, list[str]]] = []
        for so_rel in so_members:
            so_path = td_path / so_rel
            violations = audit_so(so_path, floor)
            if violations:
                all_violations.append((so_rel, violations))

    if not all_violations:
        print(f"OK: all .so files in {wheel.name} are within glibc {floor[0]}.{floor[1]}.")
        return 0

    print(f"\nFAIL: {wheel.name} references symbols above its glibc floor:")
    for so_name, viols in all_violations:
        print(f"\n  {so_name}:")
        for v in viols:
            print(f"  {v}")
    print(
        "\nThis wheel will fail to import on end-user systems with the "
        "older glibc. Fix the build (or add a compat shim) before "
        "publishing to PyPI. See issue #355 for the canonical example "
        "and `crates/headroom-py/glibc_compat.c` for the shim pattern."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
