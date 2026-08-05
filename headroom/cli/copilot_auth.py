"""GitHub Copilot authentication commands."""

from __future__ import annotations

import click

from headroom.cli.main import main
from headroom.copilot_auth import (
    default_oauth_domain,
    headroom_copilot_auth_path,
    poll_copilot_device_authorization,
    read_headroom_copilot_oauth_token,
    save_headroom_copilot_oauth_token,
    start_copilot_device_authorization,
    token_fingerprint,
)


@main.group("copilot-auth")
def copilot_auth() -> None:
    """Manage Headroom's GitHub Copilot OAuth token."""


@copilot_auth.command("login")
@click.option(
    "--domain",
    default=None,
    help=(
        "GitHub login domain. Defaults to GITHUB_COPILOT_ENTERPRISE_URL if set, "
        "otherwise github.com."
    ),
)
def login(domain: str | None) -> None:
    """Sign in with GitHub's Copilot OAuth device-code flow."""

    domain = domain or default_oauth_domain()

    try:
        device = start_copilot_device_authorization(domain=domain)
    except Exception as exc:
        raise click.ClickException(f"Unable to start GitHub device login: {exc}") from exc

    verification_uri = str(device.get("verification_uri") or "").strip()
    user_code = str(device.get("user_code") or "").strip()
    device_code = str(device.get("device_code") or "").strip()
    interval = int(device.get("interval") or 5)
    expires_in = int(device.get("expires_in") or 900)
    if not verification_uri or not user_code or not device_code:
        raise click.ClickException("GitHub device login returned an incomplete response.")

    click.echo("GitHub Copilot OAuth login")
    click.echo(f"  Open: {verification_uri}")
    click.echo(f"  Code: {user_code}")
    click.echo("  Waiting for authorization...")

    try:
        token = poll_copilot_device_authorization(
            device_code,
            domain=domain,
            interval=interval,
            expires_in=expires_in,
        )
    except Exception as exc:
        raise click.ClickException(f"GitHub device login failed: {exc}") from exc

    path = save_headroom_copilot_oauth_token(token, domain=domain)
    click.echo(f"  Saved: {path}")
    click.echo(f"  Token fingerprint: {token_fingerprint(token)}")


@copilot_auth.command("status")
def status() -> None:
    """Show whether Headroom has a saved Copilot OAuth token."""

    token = read_headroom_copilot_oauth_token()
    path = headroom_copilot_auth_path()
    click.echo(f"Auth file: {path}")
    if not token:
        click.echo("Status: not logged in")
        return
    click.echo("Status: logged in")
    click.echo(f"Token fingerprint: {token_fingerprint(token)}")
