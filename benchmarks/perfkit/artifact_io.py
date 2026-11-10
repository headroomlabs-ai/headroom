"""Artifact I/O for perfkit (JSON + Markdown renderers).

Per ``contracts/json-artifact-schema.md`` §2 + §7:

* The canonical artifact lives at
  ``benchmarks/artifacts/<commit_sha>/<tier>/<run_id>.json``.
* The companion Markdown summary lives next to it as
  ``<run_id>.md``.
* Filenames are stable across machines because they are derived only from
  the artifact's own contents.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import (
    SCHEMA_VERSION,
    PerfArtifact,
)

DEFAULT_ARTIFACTS_DIR = Path("benchmarks/artifacts")


def artifact_path(
    artifacts_dir: Path | str,
    commit: str,
    tier: str,
    run_id: str,
) -> Path:
    """Return ``<artifacts_dir>/<commit>/<tier>/<run_id>.json``.

    ``commit`` is the short or full SHA; the directory uses the value as
    provided so the caller can choose which granularity to inspect.
    """

    p = Path(artifacts_dir) / commit / tier
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{run_id}.json"


def markdown_path(json_path: Path) -> Path:
    """Companion Markdown path: ``foo.json`` -> ``foo.md``."""

    return json_path.with_suffix(".md")


def write_artifact(
    artifact: PerfArtifact,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    """Serialize ``artifact`` to disk; return the resolved JSON path.

    The Markdown summary is written alongside the JSON so downstream tools
    (release notes, dashboards) can read a flat file without re-running the
    renderer.
    """

    json_path = artifact_path(artifacts_dir, artifact.commit, artifact.tier, artifact.run_id)
    payload = artifact.to_dict()
    # Enforce deterministic key ordering: sort_keys gives byte-stable diffs.
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    md = render_markdown(artifact)
    markdown_path(json_path).write_text(md)
    return json_path


def read_artifact(json_path: Path | str) -> dict[str, Any]:
    """Read a JSON artifact back; raises if the schema is wrong."""

    text = Path(json_path).read_text()
    payload = json.loads(text)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    return payload


# ---------------------------------------------------------------------------
# Markdown rendering (contracts/json-artifact-schema.md §7).
# ---------------------------------------------------------------------------


_SECTION_ORDER = (
    "Tier / Identity",
    "Identity join keys",
    "Harness overhead",
    "Cold/warm split",
    "Trends",
    "Verdicts",
    "Notes",
)


def render_markdown(artifact: PerfArtifact) -> str:
    """Render the artifact as a Markdown summary matching §7.

    The renderer is deliberately minimal: it produces a stable, copy-paste
    friendly summary suitable for release notes and CI annotations.
    """

    lines: list[str] = []
    push = lines.append

    push(f"# {artifact.tier} performance summary")
    push("")
    push(f"- schema_version: `{artifact.schema_version}`")
    push(f"- commit: `{artifact.commit}`")
    push(f"- run_id: `{artifact.run_id}`")
    push(f"- started_at: {artifact.started_at}")
    push(f"- finished_at: {artifact.finished_at}")
    push(f"- low_confidence: {'yes' if artifact.low_confidence else 'no'}")
    push(f"- attribution_id: `{artifact.attribution.stable_id}`")

    push("")
    push("## Tier / Identity")
    push("")
    push(f"- tier: **{artifact.tier}**")
    push(f"- request_path: `{artifact.attribution.request_path}`")
    if artifact.attribution.stage:
        push(f"- stage: `{artifact.attribution.stage}`")
    if artifact.attribution.transform:
        push(f"- transform: `{artifact.attribution.transform}`")
    if artifact.attribution.function:
        push(f"- function: `{artifact.attribution.function}`")

    push("")
    push("## Identity join keys")
    push("")
    push("| key | value |")
    push("| --- | --- |")
    push(f"| schema_version | `{artifact.schema_version}` |")
    push(f"| commit | `{artifact.commit}` |")
    push(f"| tier | `{artifact.tier}` |")
    push(f"| run_id | `{artifact.run_id}` |")
    push(f"| attribution_id | `{artifact.attribution.stable_id}` |")
    push(f"| fixture_hash | `{_lookup_fixture_hash(artifact)}` |")

    if artifact.harness_overhead is not None:
        push("")
        push("## Harness overhead")
        push("")
        ho = artifact.harness_overhead
        push(f"- driver_cpu_pct: `{ho.driver_cpu_pct:.2f}`")
        push(f"- proxy_overhead_p99_s: `{ho.proxy_overhead_p99_s * 1000:.3f} ms`")
        push(f"- share_of_p99: `{ho.share_of_p99 * 100:.2f}%`")
        push(f"- sample_count: `{ho.sample_count}`")

    if artifact.cold_warm_split is not None:
        push("")
        push("## Cold/warm split")
        push("")
        cw = artifact.cold_warm_split
        push(f"- cold_p99_s: `{cw.cold_p99_s * 1000:.3f} ms`")
        push(f"- warm_p99_s: `{cw.warm_p99_s * 1000:.3f} ms`")
        push(f"- cold_warm_ratio: `{cw.cold_warm_ratio:.2f}`")
        push(f"- method: `{cw.method}`")

    if artifact.trends:
        push("")
        push("## Trends")
        push("")
        push("| metric | slope | r² | above_threshold |")
        push("| --- | --- | --- | --- |")
        for t in artifact.trends:
            push(
                f"| {t.metric} | {t.slope:.6g} {t.unit} | {t.r_squared:.3f} | "
                f"{'yes' if t.above_threshold else 'no'} |"
            )

    if artifact.verdicts:
        push("")
        push("## Verdicts")
        push("")
        push("| tier | metric | baseline | candidate | Δ% | threshold% | verdict | attribution |")
        push("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for v in artifact.verdicts:
            push(
                f"| {v.tier} | {v.metric} | {v.baseline_value:.4g} | "
                f"{v.candidate_value:.4g} | {v.delta_pct:+.2f} | "
                f"{v.threshold_pct:.2f} | {v.verdict} | "
                f"`{v.attribution_id or ''}` |"
            )

    if artifact.notes:
        push("")
        push("## Notes")
        push("")
        for n in artifact.notes:
            push(f"- {n}")

    # The metrics block keeps its natural shape; downstream readers re-parse JSON.
    push("")
    push("<!-- metrics block retained as JSON in companion .json file -->")
    push("")
    return "\n".join(lines)


def _lookup_fixture_hash(artifact: PerfArtifact) -> str:
    """Best-effort: return the join key for the artifact's fixture.

    Tier-specific blocks carry different names; we look for the most common
    one and fall back to the empty string when the artifact has no fixture
    (e.g. pure CI reports).
    """

    metrics = artifact.metrics
    if hasattr(metrics, "per_transform"):
        first = next(iter(metrics.per_transform), None)
        if isinstance(first, Mapping) and "fixture_hash" in first:
            return str(first["fixture_hash"])
    if hasattr(metrics, "per_stage"):
        first = next(iter(metrics.per_stage), None)
        if isinstance(first, Mapping) and "fixture_hash" in first:
            return str(first["fixture_hash"])
    return ""


# ---------------------------------------------------------------------------
# Small utility — useful for tests + CLI to obtain a timezone-aware ISO 8601.
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """ISO-8601 UTC timestamp with second precision."""

    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
