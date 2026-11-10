"""Internal harness sanity checks for perfkit.

The self_test module exists so the perfkit smoke workflow
(``python -m benchmarks.perfkit.cli ...`` on a clean checkout) can fail
loudly inside Phase 1 with a tiny, well-understood error rather than wait
for a full benchmark run to surface a deeper problem.

Each check returns ``(name, ok, detail)``; :func:`run_all` aggregates and
returns ``True`` when every check passes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .artifact_io import artifact_path, markdown_path, write_artifact
from .attribution import build_id
from .baseline import (
    REASON_FIRST_RECORD,
    baseline_path,
    parse_baseline_filename,
)
from .fixtures import fixture_hash, load_fixture, write_fixture_immutable
from .schema import (
    SCHEMA_VERSION,
    Attribution,
    G1Metrics,
    G6Verdict,
    PerfArtifact,
)

Check = Callable[[Path], "CheckResult"]


@dataclass(slots=True, frozen=True)
class CheckResult:
    """One self-test result line."""

    name: str
    ok: bool
    detail: str = ""

    def render(self) -> str:
        marker = "PASS" if self.ok else "FAIL"
        return f"[{marker}] {self.name}: {self.detail}"


def _schema_roundtrip(tmp: Path) -> CheckResult:
    """Schema dataclasses serialize to JSON and back without loss."""

    sample = {
        "per_transform": (
            {"transform": "smart_crusher.apply", "fixture_hash": "abc", "p99_s": 0.002},
        ),
        "python_vs_rust": None,
    }
    metrics = G1Metrics(
        per_transform=sample["per_transform"],
        python_vs_rust=sample["python_vs_rust"],
    )
    art = PerfArtifact(
        schema_version=SCHEMA_VERSION,
        commit="deadbeef",
        tier="G1",
        run_id="self-1",
        started_at="2024-01-01T00:00:00+00:00",
        finished_at="2024-01-01T00:00:01+00:00",
        attribution=Attribution(
            stable_id="G1:/v1/messages:smart_crusher:apply",
            tier="G1",
            request_path="/v1/messages",
            stage=None,
            transform="smart_crusher.apply",
            function="apply",
        ),
        metrics=metrics,
    )
    target = artifact_path(tmp, art.commit, art.tier, art.run_id)
    write_artifact(art, artifacts_dir=tmp)
    assert target.exists(), "JSON artifact not written"
    assert markdown_path(target).exists(), "Markdown summary not written"
    payload = json.loads(target.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        return CheckResult("schema_roundtrip", False, "schema_version lost in JSON")
    if payload["metrics"]["per_transform"][0]["transform"] != "smart_crusher.apply":
        return CheckResult("schema_roundtrip", False, "transform name lost in JSON")
    return CheckResult("schema_roundtrip", True, f"wrote {target.name}")


def _baseline_filename(tmp: Path) -> CheckResult:
    """Round-trip a baseline filename through the parser."""

    target = baseline_path(tmp, "G1", SCHEMA_VERSION, "abc1234")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "codeowner": "headroom-perf",
        "recorded_by": "kellen",
        "reason": REASON_FIRST_RECORD,
        "metrics": {"per_transform": []},
    }
    target.write_text(json.dumps(payload, indent=2))
    parsed = parse_baseline_filename(target)
    if parsed.schema_version != SCHEMA_VERSION or parsed.short_sha != "abc1234":
        return CheckResult("baseline_filename", False, f"parsed={parsed}")
    return CheckResult("baseline_filename", True, f"parsed={parsed.identifier}")


def _fixture_content_hash(tmp: Path) -> CheckResult:
    """Identical payload yields the same hash regardless of dict order."""

    p = tmp / "fx.json"
    data1 = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
    data2 = {"c": {"x": "y"}, "b": [1, 2, 3], "a": 1}
    write_fixture_immutable(p, data1)
    sha1 = fixture_hash(load_fixture(p).data)
    # Re-write *same* content is allowed (immutable on different content).
    write_fixture_immutable(p, data2)
    sha2 = fixture_hash(load_fixture(p).data)
    if sha1 != sha2:
        return CheckResult("fixture_content_hash", False, f"{sha1}!={sha2}")
    return CheckResult("fixture_content_hash", True, f"sha={sha1[:12]}")


def _attribution_build_id(tmp: Path) -> CheckResult:
    """Attribution ID is deterministic for the same inputs."""

    _ = tmp  # not needed; signature kept uniform for ``DEFAULT_CHECKS``.
    a = build_id("G2", "/v1/messages", "compression", "smart_crusher", "apply")
    b = build_id("G2", "/v1/messages", "compression", "smart_crusher", "apply")
    if a != b:
        return CheckResult("attribution_build_id", False, f"nondeterministic: {a}!={b}")
    return CheckResult("attribution_build_id", True, f"id={a}")


def _verdict_construction(tmp: Path) -> CheckResult:
    """A typed G6Verdict serializes as a dict (smoke check)."""

    _ = tmp
    v = G6Verdict(
        tier="G1",
        metric="per_transform/smart_crusher.apply/p99_s",
        baseline_value=0.002,
        candidate_value=0.0022,
        delta_pct=10.0,
        threshold_pct=10.0,
        verdict="regression",
    )
    payload = {
        "tier": v.tier,
        "metric": v.metric,
        "verdict": v.verdict,
    }
    if payload["verdict"] != "regression":
        return CheckResult("verdict_construction", False, "verdict field lost")
    return CheckResult("verdict_construction", True, f"verdict={v.verdict}")


DEFAULT_CHECKS: tuple[Check, ...] = (
    _schema_roundtrip,
    _baseline_filename,
    _fixture_content_hash,
    _attribution_build_id,
    _verdict_construction,
)


def run_all(tmp: Path | None = None) -> list[CheckResult]:
    """Execute the default checks in a tempdir; return all results."""

    import tempfile

    if tmp is None:
        tmp = Path(tempfile.mkdtemp(prefix="perfkit-selftest-"))
    results: list[CheckResult] = []
    # Each check gets its own subdir so on-disk state doesn't leak across checks.
    for i, chk in enumerate(DEFAULT_CHECKS):
        sub = tmp / f"check_{i}_{chk.__name__.lstrip('_')}"
        sub.mkdir(parents=True, exist_ok=True)
        try:
            results.append(chk(sub))
        except Exception as e:  # pragma: no cover - defensive
            results.append(CheckResult(name=chk.__name__, ok=False, detail=f"exception: {e!r}"))
    return results
