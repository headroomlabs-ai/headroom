"""CLI: verify a HuggingFace cache holds the pinned offline artifacts."""

from __future__ import annotations

from pathlib import Path

import click

from .main import main


@main.command(name="verify-hf-cache")
@click.argument(
    "cache",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--model",
    "model",
    default=None,
    help="Model whose counting tokenizer must be cached, e.g. Qwen/Qwen2.5-7B "
    "or the qwen2.5-7b alias.",
)
@click.option(
    "--tokenizer-repo",
    default=None,
    help="Explicit tokenizer repo id to check (instead of --model).",
)
@click.option(
    "--no-kompress",
    "require_kompress",
    is_flag=True,
    default=False,
    help="Check only the counting tokenizer, for deployments without Kompress.",
)
def verify_hf_cache(
    cache: Path | None,
    model: str | None,
    tokenizer_repo: str | None,
    require_kompress: bool,
) -> None:
    """Verify a HuggingFace cache holds the pinned artifacts an offline host needs.

    Run this after seeding a cache and before copying it to an air-gapped host.
    It checks the Kompress pair (one supported ONNX candidate plus the tokenizer
    it was exported against) and, with --model, the counting tokenizer for the
    model you serve — the two things whose absence degrades work silently.

    Exits non-zero and prints MISSING for each artifact the target would lack.
    """
    from ..hf_cache_verify import check

    if require_kompress and not model and not tokenizer_repo:
        # `--no-kompress` with no target asks for nothing, which would print OK
        # over an empty directory — a green check that verified no files.
        click.echo(
            "verify-hf-cache: --no-kompress requires --model or --tokenizer-repo "
            "(otherwise there is nothing to check).",
            err=True,
        )
        raise SystemExit(2)

    if cache is None:
        # Default to the cache this process would actually download into, so an
        # operator on the seeding host need not restate HF_HUB_CACHE.
        from huggingface_hub.constants import HF_HUB_CACHE

        cache = Path(HF_HUB_CACHE)

    if not cache.is_dir():
        click.echo(f"Not a cache directory: {cache}", err=True)
        raise SystemExit(1)
    missing = check(
        cache, model=model, tokenizer_repo=tokenizer_repo, require_kompress=not require_kompress
    )
    if not missing:
        click.echo(f"OK: pinned artifacts present in {cache}")
        return

    for entry in missing:
        click.echo(f"MISSING {entry}")
    click.echo(
        "\nSeed these before copying the cache to an air-gapped host. See "
        "'Offline upgrade path' in the configuration docs.",
        err=True,
    )
    raise SystemExit(1)
