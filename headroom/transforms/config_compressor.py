"""Structured-config compressor for YAML/TOML/INI tool output.

Config files (k8s manifests, CI pipelines, pyproject/Cargo manifests, INI
files) are high-frequency agent payloads with heavy structural repetition,
but they have no native compressor — magika tags them SOURCE_CODE and they
fall through to the lossy prose path. This module compresses them in two
lossless-first tiers:

- Tier 1 (reversible, allowed in no-CCR lossless mode): format-agnostic
  text-level compaction via `lossless_compaction` — identical-line runs and
  repeated multi-line stanzas collapse behind exact-inverse markers, with the
  round-trip self-verified before the result is adopted.
- Tier 2 (CCR-recoverable, default mode only): whole-line comments and blank
  lines are elided behind a summary line carrying a ``Retrieve original:
  hash=…`` marker; the full original is persisted to the CompressionStore
  first, so nothing is lost.
- Tier 3 (CCR-recoverable, default mode only): TOML files with an
  ``[[array-of-tables]]`` are parsed with the stdlib ``tomllib`` reference
  parser and bridged to SmartCrusher's lossless ``csv-schema`` renderer,
  which folds the repeated per-record keys into a single schema. Only TOML
  is bridged — its stdlib parser makes the records ground-truth; YAML/INI
  need non-stdlib or bespoke parsing and stay out of scope. Recovery again
  rides a stored-original + ``Retrieve original: hash=…`` marker.

No new compression algorithm and no new CCR plumbing live here — Tier 1 rides
`compact_lossless`, Tier 2/Tier 3 ride the production `CompressionStore`, and
Tier 3's fold rides `SmartCrusher`.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

from .content_detector import ContentType, detect_content_type
from .lossless_compaction import compact_lossless

logger = logging.getLogger(__name__)

# Whole-line comment prefixes per flavor. INI values can span indented
# continuation lines, so INI only elides column-0 comment lines and keeps
# blanks (configparser keeps blank lines inside multi-line values).
_COMMENT_RES = {
    "yaml": re.compile(r"^\s*#"),
    "toml": re.compile(r"^\s*#"),
    "ini": re.compile(r"^[#;]"),
}

# Content where whole-line elision is unsafe: a '#' line inside a YAML block
# scalar or a TOML multi-line string is data, not a comment. Detection is
# deliberately over-broad — when in doubt, Tier 2 stays off.
_YAML_BLOCK_SCALAR_RE = re.compile(r":\s*[|>][+-]?\d*\s*$", re.MULTILINE)
_TOML_MULTILINE_RE = re.compile(r'"""|\'\'\'')


def _load_toml(content: str) -> dict[str, Any] | None:
    """Parse TOML with the stdlib parser (or the tomli backport); None on error."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 only
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None
    try:
        return cast("dict[str, Any]", tomllib.loads(content))
    except (tomllib.TOMLDecodeError, ValueError):
        return None


def _json_default(value: Any) -> str:
    """Render TOML date/time values as ISO strings; bail on anything else."""
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    raise TypeError(f"unserializable config value: {type(value).__name__}")


@dataclass
class ConfigCompressorConfig:
    """Configuration for structured-config compression."""

    # Emit the CCR-marked comment/blank elision tier. The router wires this
    # to its ccr_inject_marker setting; lossless mode turns it off.
    enable_ccr: bool = True
    # Bridge TOML array-of-tables to SmartCrusher csv-schema (Tier 3). Rides
    # CCR for recovery, so it only runs when enable_ccr is also on.
    enable_schema_fold: bool = True
    # Only adopt a result that is strictly smaller than the original.
    min_savings_chars: int = 1


@dataclass
class ConfigCompressionResult:
    """Result of structured-config compression."""

    compressed: str
    original: str
    was_modified: bool
    flavor: str  # "yaml" | "toml" | "ini" | "unknown"
    lines_elided: int = 0
    ccr_hash: str | None = None
    strategy: str = "config"

    @property
    def compression_ratio(self) -> float:
        if not self.original:
            return 0.0
        return len(self.compressed) / len(self.original)


class ConfigCompressor:
    """Compresses YAML/TOML/INI text via reversible + CCR-recoverable tiers.

    Public surface mirrors the other content-type compressors so the router
    and tests treat it uniformly.
    """

    def __init__(self, config: ConfigCompressorConfig | None = None) -> None:
        self.config = config or ConfigCompressorConfig()

    def compress(
        self,
        content: str,
        context: str = "",
        bias: float = 1.0,
    ) -> ConfigCompressionResult:
        detection = detect_content_type(content)
        if detection.content_type is not ContentType.STRUCTURED_CONFIG:
            return ConfigCompressionResult(
                compressed=content,
                original=content,
                was_modified=False,
                flavor="unknown",
            )
        flavor = detection.metadata.get("flavor", "yaml")

        # Tier 3: schema fold (TOML array-of-tables → csv-schema). Computed
        # first so it can compete with the text tiers; it wins on lockfiles
        # and override-lists where repeated keys dominate.
        schema_fold: tuple[str, str] | None = None
        if self.config.enable_ccr and self.config.enable_schema_fold:
            schema_fold = self._schema_fold(content, flavor, context, bias)

        working = content
        lines_elided = 0
        ccr_hash: str | None = None

        # Tier 2: comment/blank elision behind a CCR marker. Persist first —
        # the elided lines are only droppable because the original is stored.
        if self.config.enable_ccr and self._elision_safe(content, flavor):
            stripped, elided = self._strip_comment_lines(content, flavor)
            if elided > 0:
                ccr_hash = self._store_original(content, stripped)
                if ccr_hash is not None:
                    marker = (
                        f"[{elided} comment/blank lines elided. Retrieve original: hash={ccr_hash}]"
                    )
                    working = stripped + ("" if stripped.endswith("\n") else "\n") + marker
                    lines_elided = elided

        # Tier 1: reversible run/stanza folding; self-verified round-trip.
        compressed = compact_lossless(working, "config")

        # Prefer the schema fold when it beats the text tiers.
        if schema_fold is not None and len(schema_fold[0]) < len(compressed):
            return ConfigCompressionResult(
                compressed=schema_fold[0],
                original=content,
                was_modified=True,
                flavor=flavor,
                ccr_hash=schema_fold[1],
                strategy="config_schema_fold",
            )

        savings = len(content) - len(compressed)
        if savings < self.config.min_savings_chars:
            return ConfigCompressionResult(
                compressed=content,
                original=content,
                was_modified=False,
                flavor=flavor,
            )

        return ConfigCompressionResult(
            compressed=compressed,
            original=content,
            was_modified=True,
            flavor=flavor,
            lines_elided=lines_elided,
            ccr_hash=ccr_hash,
        )

    def _schema_fold(
        self, content: str, flavor: str, context: str, bias: float
    ) -> tuple[str, str] | None:
        """Fold a TOML array-of-tables into SmartCrusher csv-schema.

        Returns ``(folded_text_with_marker, ccr_hash)`` when the fold is a
        strictly-smaller, faithful representation whose original is safely
        stored for CCR retrieval; otherwise ``None`` so the caller keeps the
        text tiers. Only TOML is bridged: ``tomllib`` is the reference parser,
        so the extracted records are ground-truth and the csv-schema renderer
        is itself lossless — the model reads a faithful, reformatted view.
        """
        if flavor != "toml" or "[[" not in content:
            return None
        data = _load_toml(content)
        if data is None:
            return None
        try:
            json_str = json.dumps(data, ensure_ascii=False, default=_json_default)
        except TypeError:
            return None  # a value we can't represent faithfully → don't fold

        from .smart_crusher import SmartCrusher

        crusher = SmartCrusher(with_compaction=True, compaction_format="csv-schema")
        result = crusher.crush(json_str, context, bias)
        # `passthrough` means SmartCrusher only re-canonicalized the JSON and
        # applied no schema fold, so there is nothing worth adopting.
        if not result.was_modified or result.strategy == "passthrough":
            return None

        ccr_hash = self._store_original(content, result.compressed)
        if ccr_hash is None:
            return None  # can't recover the original → never emit a lossy form
        marker = f"[config folded to schema. Retrieve original: hash={ccr_hash}]"
        folded = result.compressed + "\n" + marker
        if len(content) - len(folded) < self.config.min_savings_chars:
            return None
        return folded, ccr_hash

    @staticmethod
    def _elision_safe(content: str, flavor: str) -> bool:
        """False when a '#' line could be data (block scalars, multi-line strings)."""
        if flavor == "yaml":
            return not _YAML_BLOCK_SCALAR_RE.search(content)
        if flavor == "toml":
            return not _TOML_MULTILINE_RE.search(content)
        return True

    @staticmethod
    def _strip_comment_lines(content: str, flavor: str) -> tuple[str, int]:
        """Drop whole-line comments (and, outside INI, blank lines)."""
        comment_re = _COMMENT_RES.get(flavor, _COMMENT_RES["yaml"])
        keep_blanks = flavor == "ini"
        had_trailing = content.endswith("\n")
        lines = (content[:-1] if had_trailing else content).split("\n")
        kept: list[str] = []
        elided = 0
        for line in lines:
            if comment_re.match(line) or (not keep_blanks and not line.strip()):
                elided += 1
            else:
                kept.append(line)
        return "\n".join(kept) + ("\n" if had_trailing else ""), elided

    @staticmethod
    def _store_original(original: str, compressed: str) -> str | None:
        """Persist the original to the CompressionStore; returns its hash."""
        try:
            from ..cache.compression_store import get_compression_store
        except ImportError as e:  # pragma: no cover - store ships with headroom
            logger.warning("CCR store import failed; config elision skipped: %s", e)
            return None
        try:
            store: Any = get_compression_store()
            stored = store.store(original, compressed, compression_strategy="config")
            return str(stored) if stored else None
        except Exception as e:
            logger.warning("CCR store write failed; config elision skipped: %s", e)
            return None


__all__ = [
    "ConfigCompressor",
    "ConfigCompressorConfig",
    "ConfigCompressionResult",
]
