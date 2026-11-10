"""Baseline file loader for perfkit.

Per ``contracts/baseline-file-format.md`` §2-§6 + §8:

* Pinned baseline filename convention:
  ``{tier}.schema-{schema_ver}.{hash12}.{short_sha}.json``
  where ``{hash12}`` is a 12-character content hash and ``{short_sha}`` is the
  short commit SHA the baseline was recorded at.

* Governance fields (``codeowner``, ``recorded_by``, ``reason``,
  ``recorded_at``) live at the top level.

* ``noise_floor_used`` and ``policy_limits_used`` are preserved byte-for-byte
  in the baseline so a replay against a different noise floor policy is
  detected by the comparator.

* No auto-bump: first record is explicit
  (``--update-baseline --reason first_record``), subsequent intentional shifts
  require ``--reason approved_intended_shift``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION

# Reasons accepted by ``--update-baseline``.
REASON_FIRST_RECORD = "first_record"
REASON_APPROVED_INTENDED_SHIFT = "approved_intended_shift"
REASON_SCHEMA_BUMP = "schema_bump"
REASON_FIXTURE_CHANGE = "fixture_change"
REASON_FLOOR_CHANGE = "floor_change"

_VALID_REASONS: frozenset[str] = frozenset(
    {
        REASON_FIRST_RECORD,
        REASON_APPROVED_INTENDED_SHIFT,
        REASON_SCHEMA_BUMP,
        REASON_FIXTURE_CHANGE,
        REASON_FLOOR_CHANGE,
    }
)

# Filename pattern: tier.schema-{ver}.{hash12}.{short_sha}.json
FILENAME_RE = re.compile(
    r"^(?P<tier>[A-Za-z0-9]+)\.schema-(?P<ver>[\d.]+)\.(?P<hash>[0-9a-f]{12})\.(?P<sha>[0-9a-f]{7,40})\.json$"
)


@dataclass(slots=True, frozen=True)
class BaselineRef:
    """Parsed baseline filename.

    A baseline is fully identified by its filename contents; we never read the
    JSON to resolve the canonical identity.
    """

    tier: str
    schema_version: str
    content_hash12: str
    short_sha: str
    path: Path

    @property
    def identifier(self) -> str:
        """Stable identifier used in logs and attribution chains."""

        return f"{self.tier}.{self.short_sha}.{self.content_hash12}"

    @property
    def filename(self) -> str:
        """Original filename as stored on disk."""

        return self.path.name


def baseline_path(
    baseline_dir: Path | str,
    tier: str,
    schema_version: str,
    short_sha: str,
    content_hash12: str = "0" * 12,
) -> Path:
    """Compose the canonical baseline filename + return its full path."""

    fname = f"{tier}.schema-{schema_version}.{content_hash12}.{short_sha}.json"
    return Path(baseline_dir) / fname


def parse_baseline_filename(path: Path | str) -> BaselineRef:
    """Parse a baseline filename; raise ``ValueError`` if it doesn't fit."""

    p = Path(path)
    m = FILENAME_RE.match(p.name)
    if not m:
        raise ValueError(f"baseline filename does not match convention: {p.name!r}")
    return BaselineRef(
        tier=m.group("tier"),
        schema_version=m.group("ver"),
        content_hash12=m.group("hash"),
        short_sha=m.group("sha"),
        path=p,
    )


def list_baselines(
    baseline_dir: Path | str,
    tier: str | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> list[Path]:
    """All baselines under ``baseline_dir`` matching the given tier + schema.

    ``tier=None`` returns every tier; the comparator calls this with ``tier=``
    to find candidates.
    """

    d = Path(baseline_dir)
    if not d.exists():
        return []
    out: list[Path] = []
    for p in sorted(d.glob("*.json")):
        try:
            ref = parse_baseline_filename(p)
        except ValueError:
            continue
        if ref.schema_version != schema_version:
            continue
        if tier is not None and ref.tier != tier:
            continue
        out.append(p)
    return out


def content_hash12(payload: Mapping[str, Any]) -> str:
    """Stable 12-hex-char content hash of a baseline payload.

    The hash ignores ``recorded_at`` so two records taken at different times
    but with identical evidence are equal (audit trails are not part of the
    evidence).
    """

    sanitized = _strip_audit(payload)
    blob = json.dumps(sanitized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _strip_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` without the volatile ``recorded_at``."""

    out = dict(payload)
    out.pop("recorded_at", None)
    return out


# ---------------------------------------------------------------------------
# First-record policy.
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class BaselinePolicyDecision:
    """What the comparator should do when looking up a baseline for a tier.

    A missing baseline is NOT an error — it is a ``first_record`` opportunity
    (FR-004 + baseline-file-format.md §8). The comparator surfaces this as
    the ``baseline_missing`` verdict and exits ``3`` on per-PR guards.
    """

    exists: bool
    path: Path | None
    reason_for_action: str | None  # one of REASON_*; populated when ``exists=False``

    @classmethod
    def missing(cls, *, reason: str) -> BaselinePolicyDecision:
        if reason not in _VALID_REASONS:
            raise ValueError(f"invalid baseline reason {reason!r}; valid: {sorted(_VALID_REASONS)}")
        return cls(exists=False, path=None, reason_for_action=reason)

    @classmethod
    def present(cls, path: Path) -> BaselinePolicyDecision:
        return cls(exists=True, path=path, reason_for_action=None)


def evaluate_first_record(
    baseline_dir: Path | str,
    tier: str,
    short_sha: str,
) -> BaselinePolicyDecision:
    """Return the policy for the ``(tier, short_sha)`` pair.

    * No baseline exists for that tier at any commit → ``first_record``.
    * Baselines exist, but none at this commit → ``fixture_change`` (the
      perf landscape around this tier has moved; rerecord).
    * A baseline at the exact commit exists → ``present``.
    """

    candidates = list_baselines(baseline_dir, tier=tier)
    if not candidates:
        return BaselinePolicyDecision.missing(reason=REASON_FIRST_RECORD)
    for c in candidates:
        ref = parse_baseline_filename(c)
        if ref.short_sha == short_sha:
            return BaselinePolicyDecision.present(c)
    return BaselinePolicyDecision.missing(reason=REASON_FIXTURE_CHANGE)


def assert_reason(reason: str) -> None:
    """Raise if ``reason`` is not a known update-baseline reason."""

    if reason not in _VALID_REASONS:
        raise ValueError(f"invalid --reason {reason!r}; valid: {sorted(_VALID_REASONS)}")
