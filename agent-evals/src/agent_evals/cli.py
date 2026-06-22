"""agent-evals CLI.

Phase 0 surfaces version + resolved config. The ``run`` command is wired in once the
orchestrator and benchmark adapters land (Phase 1/2).
"""

from __future__ import annotations

import click

from . import __version__
from .config import Settings
from .logging import configure_logging


@click.group()
def main() -> None:
    """End-to-end accuracy A/B for Headroom (coding-agent benchmarks WITH vs WITHOUT compression)."""


@main.command()
def version() -> None:
    """Print the agent-evals version."""

    click.echo(__version__)


@main.command(name="show-config")
def show_config() -> None:
    """Print the resolved settings (defaults + env) as JSON."""

    settings = Settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    click.echo(settings.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
