"""CLI: passthrough subcommands for bundled tools and a `tools` management group.

Exposes:

    headroom sg …           ->  ast-grep (from the ast-grep-cli PyPI wheel)
    headroom diff A B …     ->  difftastic
    headroom loc [PATH] …   ->  scc
    headroom tools install  ->  pre-fetch all bundled binaries
    headroom tools doctor   ->  print a status table
    headroom tools list     ->  show the registry

The passthrough commands forward every argument, stdin, stdout, stderr, and
the exit code verbatim, so agents can invoke them via their existing shell
tool without any Headroom-specific protocol.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

import click

from headroom import binaries

from .main import main

_PASSTHROUGH_CTX = {
    "ignore_unknown_options": True,
    "allow_extra_args": True,
    "help_option_names": [],  # let the underlying tool handle --help
}


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def _exec_tool(tool: str, argv: Sequence[str]) -> None:
    try:
        path = binaries.resolve(tool)
    except binaries.PlatformNotSupported as e:
        click.secho(f"error: {e}", fg="red", err=True)
        sys.exit(2)
    except binaries.OfflineError as e:
        click.secho(
            f"error: {e}\nHint: run `headroom tools install` on a networked machine, "
            f"or pass --from <bundle.tar.gz>.",
            fg="red",
            err=True,
        )
        sys.exit(2)
    except (binaries.Sha256Mismatch, binaries.BinaryFetchError) as e:
        click.secho(f"error: {e}", fg="red", err=True)
        sys.exit(2)

    # Replace the current process on POSIX for correct signal handling and
    # fd/pty passthrough. NOTE: os.execv replaces the process image — atexit
    # handlers, context managers, and Python finalizers do NOT run. Anything
    # that needs to clean up on shell exit must be handled elsewhere (e.g.
    # the parent `headroom` process, not these thin passthroughs).
    cmd = [str(path), *argv]
    if not _is_windows():
        os.execv(cmd[0], cmd)  # never returns
    else:
        completed = subprocess.run(cmd, check=False)
        sys.exit(completed.returncode)


@main.command(
    "sg",
    context_settings=_PASSTHROUGH_CTX,
    short_help="Run ast-grep (AST-aware structural search/replace).",
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def sg_cmd(args: tuple[str, ...]) -> None:
    """Forward every argument to ast-grep."""
    _exec_tool("ast-grep", list(args))


@main.command(
    "diff",
    context_settings=_PASSTHROUGH_CTX,
    short_help="Run difftastic (structural diff).",
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def diff_cmd(args: tuple[str, ...]) -> None:
    """Forward every argument to difftastic (`difft`)."""
    _exec_tool("difft", list(args))


@main.command(
    "loc",
    context_settings=_PASSTHROUGH_CTX,
    short_help="Run scc (fast lines-of-code / repo-shape probe).",
    add_help_option=False,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def loc_cmd(args: tuple[str, ...]) -> None:
    """Forward every argument to scc."""
    _exec_tool("scc", list(args))


@main.group("tools")
def tools_group() -> None:
    """Manage bundled CLI tool binaries (ast-grep, difft, scc)."""


@tools_group.command("list")
def tools_list_cmd() -> None:
    """Print the tool registry (versions, platforms, cache dir)."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    plat = binaries.detect_platform()
    console.print(f"[dim]platform:[/dim] {plat.key()}")
    console.print(f"[dim]cache:[/dim] {binaries.cache_dir()}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("tool")
    table.add_column("version")
    table.add_column("source")
    table.add_column("platforms")
    reg = binaries._registry()  # noqa: SLF001 (intentional internal read)
    for name, entry in reg.get("tools", {}).items():
        platforms = ", ".join(sorted(entry.get("assets", {}).keys())) or "(pypi)"
        table.add_row(name, str(entry.get("version")), entry.get("source", ""), platforms)
    console.print(table)


@tools_group.command("doctor")
@click.option("--json", "emit_json", is_flag=True, help="Emit JSON instead of a table.")
def tools_doctor_cmd(emit_json: bool) -> None:
    """Check the status of every bundled tool."""
    rows = binaries.status()
    if emit_json:
        import json as _json

        click.echo(_json.dumps(rows, indent=2))
        broken = any(r["state"] in ("missing", "unsupported-platform") for r in rows)
        sys.exit(1 if broken else 0)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(show_header=True, header_style="bold")
    for col in ("tool", "state", "version", "platform", "path"):
        table.add_column(col)
    state_style = {
        "on-path": "green",
        "cached": "green",
        "missing": "yellow",
        "unsupported-platform": "red",
    }
    broken = False
    for r in rows:
        style = state_style.get(r["state"], "white")
        if r["state"] in ("missing", "unsupported-platform"):
            broken = True
        table.add_row(
            r["tool"],
            f"[{style}]{r['state']}[/{style}]",
            str(r.get("version")),
            r.get("platform", ""),
            r.get("path") or "-",
        )
    console.print(table)
    from rich.markup import escape as _escape

    for r in rows:
        if r.get("detail"):
            console.print(f"[dim]{r['tool']}:[/dim] {_escape(r['detail'])}")
    sys.exit(1 if broken else 0)


@tools_group.command("install")
@click.option(
    "--tool",
    "tools",
    multiple=True,
    help="Install only the named tool (repeatable). Default: all.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-fetch even if the binary is already cached.",
)
def tools_install_cmd(tools: tuple[str, ...], force: bool) -> None:
    """Pre-fetch all bundled tool binaries into the per-user cache."""
    reg = binaries._registry()  # noqa: SLF001
    selected = list(tools) if tools else list(reg.get("tools", {}).keys())
    exit_code = 0
    for name in selected:
        if name not in reg.get("tools", {}):
            click.secho(f"unknown tool {name!r}; skipping", fg="yellow", err=True)
            exit_code = 1
            continue
        if binaries._is_pypi_tool(name):  # noqa: SLF001
            on_path = binaries._path_lookup(name)  # noqa: SLF001
            if on_path:
                click.echo(f"{name}: on PATH at {on_path} (pypi wheel)")
            else:
                click.secho(
                    f"{name}: not on PATH — `pip install headroom-ai` should provide it",
                    fg="yellow",
                )
                exit_code = 1
            continue
        if force:
            plat = binaries.detect_platform()
            try:
                cached = binaries._cached_path(  # noqa: SLF001
                    name, reg["tools"][name]["version"], plat
                )
                if cached.exists():
                    cached.unlink()
            except OSError as e:
                click.secho(
                    f"{name}: failed to remove cached binary: {e}",
                    fg="yellow",
                    err=True,
                )
                exit_code = 1
        try:
            path = binaries.resolve(name)
            click.secho(f"{name}: installed → {path}", fg="green")
        except binaries.PlatformNotSupported as e:
            click.secho(f"{name}: {e}", fg="red")
            exit_code = 1
        except (binaries.BinaryFetchError, binaries.Sha256Mismatch, binaries.OfflineError) as e:
            click.secho(f"{name}: {e}", fg="red")
            exit_code = 1
    sys.exit(exit_code)
