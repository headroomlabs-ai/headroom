"""`headroom xray` — multi-CLI context-bloat diagnostics.

This is a thin Click wrapper around the `headroom-xray` Rust binary built
from `crates/headroom-xray/`. The binary itself shells out to CodeBurn
(MIT) via npx and adds the Headroom compression-opportunity footer.

Phase 1.0 requires source-built binary; Phase 1.1 will add download.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import click

from .main import main

_BIN_NAME = "headroom-xray.exe" if sys.platform == "win32" else "headroom-xray"


def _find_binary() -> Path | None:
    """Locate the headroom-xray binary.

    Order:
      1. ~/.headroom/bin/headroom-xray  (Phase 1.1 install target)
      2. System PATH                     (user installed via cargo install)
      3. <repo>/target/release/         (dev mode)
      4. <repo>/target/debug/           (dev mode)
    """
    # 1) Headroom-managed install
    home_bin = Path.home() / ".headroom" / "bin" / _BIN_NAME
    if home_bin.exists() and home_bin.is_file():
        return home_bin

    # 2) System PATH
    on_path = shutil.which("headroom-xray")
    if on_path:
        return Path(on_path)

    # 3) + 4) Dev-mode lookup, traversing upward from this file looking for Cargo.toml
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Cargo.toml").exists():
            for sub in ("release", "debug"):
                candidate = parent / "target" / sub / _BIN_NAME
                if candidate.exists() and candidate.is_file():
                    return candidate
            break

    return None


@main.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],  # forward --help to the binary
    },
)
@click.argument("xray_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def xray(ctx: click.Context, xray_args: tuple[str, ...]) -> None:
    """Multi-CLI context-bloat diagnostics (wraps CodeBurn + Headroom footer)."""
    binary = _find_binary()
    if binary is None:
        click.echo(
            "headroom xray: `headroom-xray` binary not found.\n\n"
            "Phase 1.0 ships for source-installed users. To build it:\n"
            "  cd <headroom repo>\n"
            "  cargo build --release -p headroom-xray\n"
            "  # or, to install to ~/.headroom/bin/:\n"
            "  cargo install --path crates/headroom-xray --root ~/.headroom\n\n"
            "(Pre-built binaries are coming in Phase 1.1.)",
            err=True,
        )
        ctx.exit(127)
        return

    # Exec the binary with all forwarded args.
    result = subprocess.run([str(binary), *xray_args], check=False)
    ctx.exit(result.returncode)
