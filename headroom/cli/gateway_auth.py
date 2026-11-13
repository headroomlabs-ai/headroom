"""Gateway provider identity discovery and activation commands."""

from __future__ import annotations

import click

from headroom.cli.main import main
from headroom.proxy.gateway.admission_catalog import ProviderAdmissionCatalog


@main.group("auth")
def auth() -> None:
    """Inspect gateway identity support without reading credentials."""


@auth.command("discover")
def discover() -> None:
    """List admitted identity metadata; never import or validate a credential."""
    click.echo("Admitted gateway identities:")
    for provider, identity_kind in ProviderAdmissionCatalog().discover():
        click.echo(f"  {provider} {identity_kind}")


@auth.command("status")
def status() -> None:
    """Report activation status without exposing secret-derived metadata."""
    click.echo("Gateway identity activation: configuration-managed")
    click.echo("Native subscription adapters: unavailable")


@auth.command("login")
@click.option("--provider", required=True)
@click.option("--identity-kind", required=True)
@click.option("--product-class", default="subscription", show_default=True)
def login(provider: str, identity_kind: str, product_class: str) -> None:
    """Begin an admitted interactive flow (none are enabled in gateway v1)."""
    decision = ProviderAdmissionCatalog().check(provider, identity_kind, product_class, "inference")
    if not decision.admitted:
        raise click.ClickException(f"Provider identity is not admitted: {decision.reason}")
    raise click.ClickException(
        "Interactive login is unavailable; configure an admitted API or workload identity"
    )
