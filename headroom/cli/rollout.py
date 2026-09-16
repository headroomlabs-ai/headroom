"""Runtime rollout diagnostics commands."""

from __future__ import annotations

import json
import os

import click

from headroom.rollout import RolloutConfigurationError, resolve_rollout

from .main import main


@main.group("rollout")
def rollout_group() -> None:
    """Inspect runtime feature-rollout policy (not package releases)."""


@rollout_group.command("status")
@click.option("--channel", envvar="HEADROOM_ROLLOUT_CHANNEL")
@click.option("--features", envvar="HEADROOM_FEATURES")
@click.option("--disable-features", envvar="HEADROOM_DISABLE_FEATURES")
@click.option(
    "--unsafe-allow-unstable-features",
    is_flag=True,
    envvar="HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES",
)
@click.option("--json", "json_output", is_flag=True, help="Emit the versioned JSON snapshot.")
def rollout_status(
    channel: str | None,
    features: str | None,
    disable_features: str | None,
    unsafe_allow_unstable_features: bool,
    json_output: bool,
) -> None:
    """Resolve and print the supplied runtime rollout configuration."""

    env = dict(os.environ)
    if channel is not None:
        env["HEADROOM_ROLLOUT_CHANNEL"] = channel
    if features is not None:
        env["HEADROOM_FEATURES"] = features
    if disable_features is not None:
        env["HEADROOM_DISABLE_FEATURES"] = disable_features
    if unsafe_allow_unstable_features:
        env["HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES"] = "1"
    try:
        snapshot = resolve_rollout(env, strict=True)
    except RolloutConfigurationError as exc:
        raise click.ClickException(str(exc)) from exc

    payload = snapshot.to_dict()
    if json_output:
        click.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return

    click.echo(f"Rollout channel: {snapshot.channel.value}")
    click.echo(f"Policy: {snapshot.policy_version} ({snapshot.registry_digest})")
    click.echo(f"Snapshot: {snapshot.snapshot_digest}")
    click.echo(f"Qualification eligible: {str(snapshot.qualification_eligible).lower()}")
    for decision in snapshot.decisions:
        click.echo(
            f"  {decision.name}: enabled={str(decision.enabled).lower()} "
            f"decision={decision.reason.value}"
        )
