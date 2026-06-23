"""Build the frozen :class:`RunManifest`.

All time/git/identity is injected at the edge (the CLI passes ``now`` and repo paths) so the
core stays reproducible — no ``datetime.now()`` or RNG in here.
"""

from __future__ import annotations

import subprocess
from datetime import datetime

from .config import Settings
from .models import ArmSpec, RunManifest


def git_sha(repo_path: str) -> str:
    """Return the HEAD sha of a repo, or ``"unknown"`` if it is not a usable git repo."""

    try:
        out = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def build_manifest(
    settings: Settings,
    *,
    now: datetime,
    arms: list[ArmSpec],
    benchmark: str,
    benchmark_ref: str,
    harness: str,
    harness_version: str,
    headroom_repo_path: str,
    agent_evals_repo_path: str,
    auth_mode: str = "payg",
    temperature: float = 0.0,
    seeds: list[int] | None = None,
    docker_digests: dict[str, str] | None = None,
) -> RunManifest:
    """Assemble a fully-pinned manifest. ``experiment_id`` is deterministic given ``now``."""

    experiment_id = f"{benchmark}-{now:%Y%m%dT%H%M%SZ}"
    return RunManifest(
        experiment_id=experiment_id,
        created_at=now,
        headroom_git_sha=git_sha(headroom_repo_path),
        agent_evals_git_sha=git_sha(agent_evals_repo_path),
        model_snapshot=settings.model_snapshot,
        provider=settings.provider,
        auth_mode=auth_mode,
        benchmark=benchmark,
        benchmark_ref=benchmark_ref,
        harness=harness,
        harness_version=harness_version,
        docker_digests=docker_digests or {},
        arms=arms,
        k_runs=settings.stats.k_runs,
        temperature=temperature,
        seeds=seeds if seeds is not None else list(range(settings.stats.k_runs)),
        alpha=settings.stats.alpha,
        margins={"ccr": settings.stats.margin_ccr_pp, "lossy": settings.stats.margin_lossy_pp},
        pricing=settings.pricing,
    )
