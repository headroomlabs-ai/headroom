"""Top-level ``headroom uninstall`` — reverse a Headroom install in one step.

``wrap`` / ``mcp install`` / ``install agent`` each touch a different place
(Codex ``config.toml``, Claude MCP config, LaunchAgents/systemd/crontab, shell
rc env blocks, ``~/.headroom``).  Undoing them one command at a time is
error-prone, so this command detects what is present and reverses all of it,
reusing the existing per-tool commands rather than duplicating their logic.

The Python/Node package itself is left to the user's package manager; the
final ``pip`` / ``npm`` step is printed at the end.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click

from .main import main


def _codex_wrapped() -> bool:
    """Return True when ``~/.codex/config.toml`` carries a Headroom wrap."""
    from headroom.cli import wrap as wrap_mod

    config_file, backup_file = wrap_mod._codex_config_paths()
    if backup_file.exists():
        return True
    if not config_file.exists():
        return False
    content = config_file.read_text(encoding="utf-8", errors="ignore")
    return wrap_mod._CODEX_TOP_LEVEL_MARKER in content


def _claude_wrapped() -> bool:
    """Return True when any Claude-side Headroom artifact is still present."""
    if _claude_rtk_hook_present():
        return True

    try:
        from headroom.mcp_registry import ClaudeRegistrar
        from headroom.mcp_registry.ledger import headroom_installed_matching
    except Exception:
        return False

    registrar = ClaudeRegistrar()
    if not registrar.detect():
        return False
    if registrar.get_server("headroom") is not None:
        return True
    if registrar.get_server("codebase-memory-mcp") is not None:
        return True
    return headroom_installed_matching("claude", registrar.get_server("serena"))


def _claude_rtk_hook_present(settings_path: Path | None = None) -> bool:
    """Return True when Claude settings still contain the Headroom rtk hook."""
    path = settings_path or (Path.home() / ".claude" / "settings.json")
    if not path.exists():
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False

    hooks = payload.get("hooks")
    if not isinstance(hooks, dict):
        return False

    for entries in hooks.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            hook_items = entry.get("hooks")
            if not isinstance(hook_items, list):
                continue
            for item in hook_items:
                if not isinstance(item, dict):
                    continue
                if "rtk-rewrite" in str(item.get("command", "")).lower():
                    return True
    return False


def _openclaw_wrapped() -> bool:
    """Return True when the Headroom OpenClaw plugin entry exists."""
    openclaw_bin = shutil.which("openclaw")
    if not openclaw_bin:
        return False
    from headroom.cli import wrap as wrap_mod

    return (
        wrap_mod._read_openclaw_config_value(openclaw_bin, "plugins.entries.headroom") is not None
    )


def _deployment_profiles() -> list[str]:
    """Return the names of any persistent supervisor deployments."""
    try:
        from headroom.install.state import list_manifests
    except Exception:
        return []
    return [m.profile for m in list_manifests()]


def _remove_deployments(profiles: list[str]) -> None:
    from headroom.cli.install import _remove_deployment
    from headroom.install.state import load_manifest

    for profile in profiles:
        manifest = load_manifest(profile)
        if manifest is None:
            continue
        _remove_deployment(manifest)
        click.echo(f"  Removed persistent deployment '{profile}'.")


def _print_package_step() -> None:
    click.echo()
    click.echo("  The package itself is left to your package manager. To finish:")
    click.echo("    pip uninstall headroom-ai")
    click.echo("    npm uninstall -g headroom-ai   # if you installed the Node package")
    click.echo()


@main.command("uninstall")
@click.option(
    "--port", "-p", default=8787, type=int, help="Local proxy port to stop (default: 8787)."
)
@click.option("--keep-proxy", is_flag=True, help="Leave the local Headroom proxy running.")
@click.option(
    "--purge-state",
    is_flag=True,
    help="Also delete the ~/.headroom workspace (savings history, caches, telemetry).",
)
@click.option(
    "--dry-run", is_flag=True, help="Report what would be removed without changing anything."
)
@click.pass_context
def uninstall(
    ctx: click.Context,
    port: int,
    keep_proxy: bool,
    purge_state: bool,
    dry_run: bool,
) -> None:
    """Reverse a Headroom install in one step.

    \b
    Detects and undoes everything Headroom set up:
      * unwraps each wrapped tool (Codex, Claude, OpenClaw)
      * removes MCP server registrations (`mcp uninstall`)
      * tears down any persistent supervisor deployment
      * stops the local proxy

    \b
    The package itself is left to your package manager; the final
    `pip` / `npm` step is printed at the end. Use `--dry-run` to preview.
    """
    from headroom import paths
    from headroom.cli import wrap as wrap_mod

    click.echo()
    click.echo("  ╔═══════════════════════════════════════════════╗")
    click.echo("  ║              HEADROOM UNINSTALL               ║")
    click.echo("  ╚═══════════════════════════════════════════════╝")
    click.echo()

    codex = _codex_wrapped()
    claude = _claude_wrapped()
    openclaw = _openclaw_wrapped()
    profiles = _deployment_profiles()
    proxy_running = wrap_mod._check_proxy(port)
    workspace = paths.workspace_dir()

    if dry_run:
        click.echo("  Dry run — nothing will be changed.")
        click.echo()
        click.echo(f"  Codex wrap:          {'remove' if codex else 'not found'}")
        click.echo(f"  Claude wrap:         {'remove' if claude else 'not found'}")
        click.echo(f"  OpenClaw wrap:       {'remove' if openclaw else 'not found'}")
        click.echo("  MCP registrations:   remove (if present)")
        click.echo(f"  Persistent deploys:  {', '.join(profiles) if profiles else 'none'}")
        click.echo(
            f"  Local proxy (:{port}):  "
            f"{'stop' if proxy_running and not keep_proxy else ('keep' if keep_proxy else 'not running')}"
        )
        click.echo(
            f"  Workspace state:     "
            f"{('delete ' + str(workspace)) if (purge_state and workspace.exists()) else 'keep'}"
        )
        _print_package_step()
        return

    if codex:
        ctx.invoke(wrap_mod.unwrap_codex, port=port, no_stop_proxy=True)
    if claude:
        ctx.invoke(wrap_mod.unwrap_claude, port=port, no_stop_proxy=True)
    if openclaw:
        try:
            ctx.invoke(wrap_mod.unwrap_openclaw, proxy_port=port, no_stop_proxy=True)
        except click.ClickException as exc:
            click.echo(f"  Skipped OpenClaw unwrap: {exc.format_message()}")

    # Covers a standalone `headroom mcp install` that wasn't tied to a wrap.
    from headroom.cli.mcp import mcp_uninstall

    ctx.invoke(mcp_uninstall)

    if profiles:
        _remove_deployments(profiles)

    if not keep_proxy:
        wrap_mod._echo_unwrap_proxy_stop_status(wrap_mod._stop_local_proxy_for_unwrap(port), port)

    if purge_state:
        if workspace.exists():
            try:
                shutil.rmtree(workspace)
            except OSError as exc:
                raise click.ClickException(
                    f"could not delete workspace state at {workspace}: {exc}"
                ) from exc
            click.echo(f"  Deleted workspace state at {workspace}.")
        else:
            click.echo("  No workspace state to delete.")
    elif workspace.exists():
        click.echo(f"  Left workspace state at {workspace} (use --purge-state to delete).")

    click.echo()
    click.echo("✓ Headroom integrations removed.")
    _print_package_step()
