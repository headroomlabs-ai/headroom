"""MCP (Model Context Protocol) CLI commands for Claude Code integration.

Provides commands to configure and run the Headroom MCP server, enabling
Claude Code subscription users to use CCR (Compress-Cache-Retrieve) without
needing API key access.
"""

import click

from .main import main

DEFAULT_PROXY_URL = "http://127.0.0.1:8787"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8788
DEFAULT_HTTP_PATH = "/mcp"


@main.group()
def mcp() -> None:
    """MCP server for Claude Code integration.

    \b
    The MCP server exposes headroom_retrieve as a tool that Claude Code
    can use to retrieve compressed content. This enables CCR (Compress-
    Cache-Retrieve) for subscription users who don't have API access.

    \b
    Quick Start:
        headroom mcp install    # Configure Claude Code
        headroom proxy          # Start the proxy (in another terminal)
        ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude

    \b
    The MCP server provides on-demand tools (compress, retrieve, stats).
    For automatic compression of ALL traffic, also set ANTHROPIC_BASE_URL
    to route through the proxy.

    \b
    How it works:
        1. ANTHROPIC_BASE_URL routes all requests through the proxy
        2. The proxy compresses large tool outputs (file listings, search results)
        3. Claude sees compressed summaries with hash markers
        4. When Claude needs full details, it calls headroom_retrieve
        5. The MCP server fetches original content from the proxy

    \b
    Note on tool naming: MCP clients display tools as
    `mcp__<server>__<tool>`. Our server is named "headroom" and our
    tools are named headroom_retrieve / headroom_compress / etc., so
    Claude Code shows them as `mcp__headroom__headroom_retrieve`. The
    "headroom" doubling is normal MCP namespacing — not a bug. The
    proxy's compression markers (and any docs/prompts) reference the
    bare tool name `headroom_retrieve`.
    """
    pass


@mcp.command("install")
@click.option(
    "--proxy-url",
    default=DEFAULT_PROXY_URL,
    help=f"Headroom proxy URL (default: {DEFAULT_PROXY_URL})",
)
@click.option(
    "--agent",
    "agents",
    multiple=True,
    help="Restrict installation to specific agents (default: every detected agent).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing headroom config in case of mismatch.",
)
def mcp_install(proxy_url: str, agents: tuple[str, ...], force: bool) -> None:
    """Install the Headroom MCP server into every detected coding agent.

    \b
    By default this installs into every agent that has a registrar and is
    detected on this system (Antigravity IDE, Claude Code, Codex, Grok CLI,
    OpenCode today; Cursor / Continue / others added in subsequent releases).
    Pass ``--agent NAME`` one or more times to restrict the installation.

    \b
    Examples:
        headroom mcp install                            # every detected agent
        headroom mcp install --agent claude             # Claude Code only
        headroom mcp install --proxy-url http://localhost:9000
    """
    try:
        import mcp  # noqa: F401
    except ImportError:
        click.echo("Error: MCP SDK not installed.", err=True)
        click.echo("Install with: pip install 'headroom-ai[mcp]'", err=True)
        raise SystemExit(1) from None

    from headroom.mcp_registry import any_succeeded, format_results, install_everywhere

    results = install_everywhere(
        proxy_url=proxy_url,
        agents=list(agents) if agents else None,
        force=force,
    )

    if not results:
        click.echo("No agents matched the requested filter.")
        raise SystemExit(1)

    click.echo("Installing Headroom MCP server...")
    for line in format_results(
        results,
        verbose=True,
        overwrite_hint=f"headroom mcp install --proxy-url {proxy_url} --force",
    ):
        click.echo(line)

    if not any_succeeded(results):
        raise SystemExit(1)

    click.echo(
        f"\nNext steps:\n"
        f"  1. Start the Headroom proxy (if not running): headroom proxy\n"
        f"  2. Start your agent (e.g.) ANTHROPIC_BASE_URL={proxy_url} claude\n"
        f"  3. Restart any agent that was already running so it picks up the new MCP server.\n"
    )


@mcp.command("uninstall")
@click.option(
    "--agent",
    "agents",
    multiple=True,
    help="Restrict uninstallation to specific agents (default: every detected agent).",
)
def mcp_uninstall(agents: tuple[str, ...]) -> None:
    """Remove Headroom MCP server from detected agent configs.

    \b
    Removes headroom from every agent registrar known to Headroom. Other MCP
    servers are preserved. Pass ``--agent NAME`` one or more times to restrict
    the uninstallation.

    \b
    Examples:
        headroom mcp uninstall                       # every detected agent
        headroom mcp uninstall --agent antigravity   # Antigravity IDE only
    """
    from headroom.mcp_registry import get_all_registrars

    registrars = get_all_registrars()
    if agents:
        wanted = {agent.strip().lower() for agent in agents if agent.strip()}
        registrars = [registrar for registrar in registrars if registrar.name in wanted]
        if not registrars:
            click.echo("No agents matched the requested filter.")
            raise SystemExit(1)

    removed = False

    for registrar in registrars:
        if not registrar.detect():
            continue
        removed_names: list[str] = []
        for server_name in ("headroom", "codebase-memory-mcp"):
            if registrar.unregister_server(server_name):
                removed_names.append(server_name)
        if removed_names:
            click.echo(
                f"✓ {registrar.display_name}: removed {', '.join(removed_names)} MCP server(s)"
            )
            removed = True

    if not removed:
        click.echo("Headroom MCP is not configured. Nothing to uninstall.")


@mcp.command("reconcile")
@click.option("--adopt", is_flag=True, help="Replace only the Serena entry with Headroom's spec.")
def mcp_reconcile(adopt: bool) -> None:
    """Inspect or explicitly reconcile a user-managed Serena MCP entry."""
    from headroom.mcp_registry import (
        CLAUDE_SERENA_CONTEXT,
        ClaudeConfigMutationError,
        ClaudeRegistrar,
        RegisterStatus,
        build_serena_spec,
    )
    from headroom.mcp_registry.ledger import (
        LedgerMutationError,
        record_install,
        validate_ledger_for_mutation,
    )

    registrar = ClaudeRegistrar()
    if not registrar.detect():
        raise click.ClickException("claude is not detected")
    recommended = build_serena_spec(CLAUDE_SERENA_CONTEXT)
    observed = registrar.get_server("serena")

    if adopt:
        try:
            registrar.validate_configs_for_mutation()
            validate_ledger_for_mutation()
        except (ClaudeConfigMutationError, LedgerMutationError) as exc:
            raise click.ClickException(str(exc)) from exc
    if adopt:
        result = registrar.register_server(recommended, force=True)
        if result.status not in (RegisterStatus.REGISTERED, RegisterStatus.ALREADY):
            raise click.ClickException(result.detail or "could not adopt Serena configuration")
        record_install("claude", recommended)
        click.echo(
            "Adopted Headroom's Serena configuration for Claude; unrelated config preserved."
        )
        return

    click.echo("Serena reconciliation for Claude")
    click.echo(f"  observed: {'absent' if observed is None else 'present'}")
    click.echo(f"  recommendation: {recommended.command} {' '.join(recommended.args)}")
    if observed is not None and observed != recommended:
        click.echo("  action: use --adopt to replace it")
    # A Serena bundled by an enabled Claude Code plugin runs alongside the
    # entry above (``claude mcp list`` shows it as ``plugin:serena:serena``)
    # with none of Headroom's flags. It is a third-party plugin, so it is
    # never removed here — point at the command that disables it (#3570).
    for plugin_id, spec in registrar.get_plugin_servers("serena"):
        click.echo(
            f"  plugin: {plugin_id} also provides a Serena MCP server "
            f"({spec.command} {' '.join(spec.args)})"
        )
        click.echo(
            f"  action: disable it so only one Serena runs: claude plugin disable {plugin_id}"
        )


@mcp.command("status")
def mcp_status() -> None:
    """Check Headroom MCP configuration status.

    \b
    Shows whether headroom is configured in Claude Code and if
    the proxy is reachable.
    """
    click.echo("Headroom MCP Status")
    click.echo("=" * 40)

    # Check MCP SDK
    try:
        import mcp  # noqa: F401

        click.echo("MCP SDK:        ✓ Installed")
    except ImportError:
        click.echo("MCP SDK:        ✗ Not installed")
        click.echo("                pip install 'headroom-ai[mcp]'")

    from headroom.mcp_registry import get_all_registrars

    proxy_url = DEFAULT_PROXY_URL
    any_configured = False
    click.echo("Agent Config:")
    for registrar in get_all_registrars():
        if not registrar.detect():
            click.echo(f"  {registrar.display_name}: ✗ Not detected")
            continue
        spec = registrar.get_server("headroom")
        if spec is None:
            click.echo(f"  {registrar.display_name}: ✗ Not configured")
            continue
        any_configured = True
        proxy_url = spec.env.get("HEADROOM_PROXY_URL", proxy_url)
        click.echo(f"  {registrar.display_name}: ✓ Configured")

    if not any_configured:
        click.echo("                Run: headroom mcp install")
    click.echo(f"Proxy URL:      {proxy_url}")

    # Check proxy connectivity
    try:
        import httpx

        try:
            response = httpx.get(f"{proxy_url}/health", timeout=2.0)
            if response.status_code == 200:
                click.echo(f"Proxy Status:   ✓ Running at {proxy_url}")
            else:
                click.echo(f"Proxy Status:   ✗ Unhealthy (status {response.status_code})")
        except httpx.ConnectError:
            click.echo("Proxy Status:   ✗ Not running")
            click.echo("                Run: headroom proxy")
        except httpx.TimeoutException:
            click.echo("Proxy Status:   ✗ Timeout")
        except httpx.HTTPError as e:
            # Catch the rest (InvalidURL, UnsupportedProtocol, ProtocolError, …)
            # so a malformed configured HEADROOM_PROXY_URL can't crash `status`.
            click.echo(f"Proxy Status:   ✗ Unreachable ({proxy_url}: {e})")
    except ImportError:
        click.echo("Proxy Status:   ? (httpx not installed)")


@mcp.command("serve")
@click.option(
    "--proxy-url",
    default=None,
    envvar="HEADROOM_PROXY_URL",
    help=f"Headroom proxy URL (default: {DEFAULT_PROXY_URL})",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"], case_sensitive=False),
    default="stdio",
    show_default=True,
    help="Transport to use for headroom mcp serve",
)
@click.option(
    "--host",
    default=DEFAULT_HTTP_HOST,
    show_default=True,
    help="HTTP bind host",
)
@click.option(
    "--port",
    default=DEFAULT_HTTP_PORT,
    show_default=True,
    type=int,
    help="HTTP bind port",
)
@click.option(
    "--path",
    default=DEFAULT_HTTP_PATH,
    show_default=True,
    help="HTTP endpoint path",
)
@click.option(
    "--direct",
    is_flag=True,
    help="(Deprecated, ignored) Direct CompressionStore access is no longer supported",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
def mcp_serve(
    proxy_url: str | None,
    transport: str,
    host: str,
    port: int,
    path: str,
    direct: bool,
    debug: bool,
) -> None:
    """Start the MCP server (called by Claude Code).

    \b
    This command is typically invoked by Claude Code via the MCP config,
    not run directly. It starts the MCP server with stdio by default or
    Streamable HTTP when requested.

    \b
    For manual testing:
        headroom mcp serve --debug
        headroom mcp serve --transport http --host 127.0.0.1 --port 8788 --path /mcp
    """
    import asyncio
    import logging

    # Check for MCP SDK
    try:
        from headroom.ccr.mcp_server import create_ccr_mcp_server
    except ImportError as e:
        click.echo(f"Error: MCP dependencies not installed: {e}", err=True)
        click.echo("Install with: pip install 'headroom-ai[mcp]'", err=True)
        raise SystemExit(1) from None

    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # Minimal logging for MCP (stdout is used for protocol)
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
        )

    transport = transport.lower()

    # Use default if not specified
    effective_proxy_url = proxy_url or DEFAULT_PROXY_URL

    if direct:
        click.echo(
            "Warning: --direct is deprecated and ignored; MCP retrieval uses the proxy URL.",
            err=True,
        )

    server = create_ccr_mcp_server(proxy_url=effective_proxy_url)

    async def run() -> None:
        try:
            if transport == "http":
                await server.run_streamable_http(host=host, port=port, path=path, debug=debug)
            else:
                await server.run_stdio()
        finally:
            await server.cleanup()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
