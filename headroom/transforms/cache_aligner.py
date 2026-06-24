"""Cache alignment detector for Headroom SDK.

PR-A2 / P2-23 fix: This module is now a **detector-only** transform.

The previous rewrite path (which strips dynamic content from the system
prompt and re-inserts it as a context block) violated invariant I2 — the
cache hot zone (system prompt) must never be mutated. That path has been
removed. ``CacheAligner`` now exclusively:

1. Detects volatile / dynamic content in the system prompt using
   structural parsers (no regex):
   - UUIDs via the stdlib ``uuid`` module
   - ISO 8601 timestamps via ``datetime.fromisoformat``
   - JWTs via shape-only structural checks (three dot-separated
     base64url segments with the expected size profile)
   - Hex hashes (MD5/SHA1/SHA256) via length + alphabet checks

2. Emits a customer-visible warning log line surfacing detected
   dynamic content so callers know their cache prefix is unstable.
   The prompt itself is never modified.

The transform's ``apply`` method is a no-op for messages — it only
populates ``warnings`` and ``cache_metrics`` for observability.
"""

from __future__ import annotations

import base64
import binascii
import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..config import CacheAlignerConfig, CachePrefixMetrics, TransformResult
from ..tokenizer import Tokenizer
from ..tokenizers import EstimatingTokenCounter
from ..utils import compute_short_hash, deep_copy_messages
from .base import Transform

logger = logging.getLogger(__name__)


# Length profile for hex hash detection. Kept as named constants — no magic
# numbers in production code (build constraint #2). MD5 = 32 hex chars,
# SHA1 = 40, SHA256 = 64.
_HEX_HASH_LENGTHS = frozenset({32, 40, 64})

# Canonical UUID (RFC 4122) with dashes is 36 chars. We deliberately do NOT
# accept the 32-char dashless form since it is structurally identical to an
# MD5 hex digest and would mis-classify a hash as a UUID.
_UUID_CANONICAL_LEN = 36

# JWT shape constraints. A JWT is exactly three base64url-encoded segments
# joined by ``.``. We do NOT verify the signature (we don't have the key,
# and we're only doing detection); we only check the shape.
_JWT_SEGMENT_COUNT = 3
_JWT_MIN_SEGMENT_BYTES = 4

# Token classification labels — keep stable so log consumers can filter.
_LABEL_UUID = "uuid"
_LABEL_ISO8601 = "iso8601"
_LABEL_JWT = "jwt"
_LABEL_HEX_HASH = "hex_hash"


@dataclass(frozen=True)
class VolatileFinding:
    """One detected piece of volatile content."""

    label: str
    sample: str  # Truncated, never full content


def _is_uuid(token: str) -> bool:
    """Return True if ``token`` parses as a canonical UUID.

    Accepts only the canonical 36-char form with dashes
    (``xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx``). The 32-char dashless form
    is structurally indistinguishable from an MD5 hex digest and would
    misclassify hashes; we treat that case as a hex hash instead.
    Defers to ``uuid.UUID`` for parsing — no regex.
    """
    if len(token) != _UUID_CANONICAL_LEN:
        return False
    if token.count("-") != 4:
        return False
    try:
        _uuid.UUID(token)
    except (ValueError, AttributeError):
        return False
    return True


def _is_iso8601(token: str) -> bool:
    """Return True if ``token`` parses as an ISO 8601 datetime.

    Uses ``datetime.fromisoformat`` (Python 3.11+ supports the full ISO
    spec including the ``Z`` suffix; on 3.10 the parser is stricter but
    handles the common forms we care about).
    """
    if len(token) < 8:
        return False
    if "T" not in token and "-" not in token:
        return False
    candidate = token[:-1] + "+00:00" if token.endswith("Z") else token
    try:
        datetime.fromisoformat(candidate)
    except (ValueError, TypeError):
        return False
    return True


def _is_jwt_shape(token: str) -> bool:
    """Return True if ``token`` has the shape of a JWT.

    A JWT is three base64url-encoded segments separated by ``.``. We only
    verify shape (segment count + each segment decodes); we never verify
    the signature.
    """
    if token.count(".") != _JWT_SEGMENT_COUNT - 1:
        return False
    segments = token.split(".")
    if len(segments) != _JWT_SEGMENT_COUNT:
        return False
    for seg in segments:
        if len(seg) < _JWT_MIN_SEGMENT_BYTES:
            return False
        # base64url decode requires padding to multiple of 4
        padded = seg + "=" * (-len(seg) % 4)
        try:
            base64.urlsafe_b64decode(padded.encode("ascii"))
        except (binascii.Error, ValueError, UnicodeEncodeError):
            return False
    return True


def _is_hex_hash(token: str) -> bool:
    """Return True if ``token`` looks like an MD5/SHA1/SHA256 hex digest.

    Length must be one of the known fixed sizes and every character must be
    a hex digit. We use ``str.isalnum``+``int(token, 16)`` rather than a
    regex; the former two are O(n) C-level checks.
    """
    if len(token) not in _HEX_HASH_LENGTHS:
        return False
    try:
        int(token, 16)
    except ValueError:
        return False
    return True


def _classify_token(token: str) -> str | None:
    """Return a label for ``token`` if it matches a volatile pattern.

    Order matters: more specific (longer / more constrained) checks first
    so we don't mis-classify a UUID-without-dashes as a hex hash.
    """
    # UUID: structurally distinct (dashes or 32 hex)
    if _is_uuid(token):
        return _LABEL_UUID
    # JWT: requires literal dots — cheapest discriminator
    if "." in token and _is_jwt_shape(token):
        return _LABEL_JWT
    # ISO 8601: requires ``T`` or ``-`` — cheap discriminator
    if _is_iso8601(token):
        return _LABEL_ISO8601
    # Hex hash: pure hex, fixed length
    if _is_hex_hash(token):
        return _LABEL_HEX_HASH
    return None


def _split_tokens(content: str) -> list[str]:
    """Split content into whitespace-delimited tokens for inspection.

    No regex. ``str.split`` (default) collapses consecutive whitespace and
    handles all standard whitespace classes. We then strip surrounding
    punctuation that commonly wraps an inline token (``,``, ``;``, ``)``,
    ``"``, etc.) so ``"Date:2024-01-15."`` yields the bare ``2024-01-15``.
    """
    if not content:
        return []
    tokens: list[str] = []
    for raw in content.split():
        cleaned = raw.strip(".,;:!?\"'()[]{}<>")
        if cleaned:
            tokens.append(cleaned)
    return tokens


def detect_volatile_content(content: str) -> list[VolatileFinding]:
    """Detect volatile/dynamic content in arbitrary text.

    Pure detection: no regex, no mutation. Returns one finding per token
    that matches any structural pattern. Callers can decide whether to
    emit a warning, alert, or ignore.
    """
    if not content:
        return []
    findings: list[VolatileFinding] = []
    for token in _split_tokens(content):
        label = _classify_token(token)
        if label is None:
            continue
        # Truncate the sample so we never log full secrets verbatim.
        sample = token if len(token) <= 16 else token[:8] + "..." + token[-4:]
        findings.append(VolatileFinding(label=label, sample=sample))
    return findings


class CacheAligner(Transform):
    """Detect volatile content in the system prompt and warn — never rewrite.

    P2-23 fix: this is now a **detector-only** transform. It NEVER mutates
    messages, never moves content, never normalizes whitespace. Callers
    that previously relied on the rewrite behavior must instead route
    memory / dynamic context to the live zone (latest user turn) per
    PR-A2.
    """

    name = "cache_aligner"

    def __init__(self, config: CacheAlignerConfig | None = None):
        """Initialize the detector-only cache aligner."""
        self.config = config or CacheAlignerConfig()
        # Track previous hash for cache hit detection (observability only).
        self._previous_prefix_hash: str | None = None

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Return True iff detection is enabled and a system message exists.

        Detection is cheap; we run it whenever ``enabled`` is set so the
        warning log line is emitted on every relevant turn.

        Phase F PR-F2.1 c4/5: when the request's
        :class:`~headroom.transforms.compression_policy.CompressionPolicy`
        is passed via ``kwargs["compression_policy"]`` and has
        ``cache_aligner_enabled=False`` (Subscription auth mode under
        the enforcement flag), this method returns ``False`` so the
        detector is skipped for that request. The hidden state
        ``self._previous_prefix_hash`` is NOT cleared on skip — the
        field is per-pipeline-instance, not per-request, so clearing
        it would race with concurrent PAYG requests on the same
        pipeline (which is the production shape).
        """
        if not self.config.enabled:
            return False
        policy = kwargs.get("compression_policy")
        if policy is not None and not policy.cache_aligner_enabled:
            return False
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    return True
        return False

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Detect volatile content; emit warnings; never mutate messages.

        Invariant: ``result.messages`` is byte-equal to the input
        ``messages`` (modulo a deep copy for downstream isolation). The
        prompt is never rewritten.
        """
        tokens_before = tokenizer.count_messages(messages)
        # Deep copy so callers receive a stable list they can further
        # transform without aliasing back into the input. The COPY is
        # not modified — invariant I2.
        result_messages = deep_copy_messages(messages)
        warnings: list[str] = []
        all_findings: list[VolatileFinding] = []
        frozen_message_count = kwargs.get("frozen_message_count", 0)

        for i, msg in enumerate(result_messages):
            if i < frozen_message_count:
                continue
            if msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                continue
            findings = detect_volatile_content(content)
            if findings:
                all_findings.extend(findings)

        if all_findings:
            counts: dict[str, int] = {}
            for f in all_findings:
                counts[f.label] = counts.get(f.label, 0) + 1
            counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            msg_text = (
                f"CacheAligner: detected volatile content in system prompt "
                f"({counts_str}); cache prefix unstable. "
                "Move dynamic values out of the system prompt to recover cache hits."
            )
            warnings.append(msg_text)
            logger.warning(msg_text)

        # Compute a stable hash of all system messages for observability.
        # This is just a hash of the (unchanged) bytes — no extraction.
        system_text = "\n---\n".join(
            (m.get("content") or "")
            for m in result_messages
            if m.get("role") == "system" and isinstance(m.get("content"), str)
        )
        stable_hash = compute_short_hash(system_text)
        prefix_bytes = len(system_text.encode("utf-8"))
        prefix_tokens_est = tokenizer.count_text(system_text)
        prefix_changed = (
            self._previous_prefix_hash is not None and self._previous_prefix_hash != stable_hash
        )
        previous_hash = self._previous_prefix_hash
        self._previous_prefix_hash = stable_hash

        cache_metrics = CachePrefixMetrics(
            stable_prefix_bytes=prefix_bytes,
            stable_prefix_tokens_est=prefix_tokens_est,
            stable_prefix_hash=stable_hash,
            prefix_changed=prefix_changed,
            previous_hash=previous_hash,
        )

        tokens_after = tokenizer.count_messages(result_messages)
        result = TransformResult(
            messages=result_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=[],  # Never applies a rewrite.
            warnings=warnings,
            cache_metrics=cache_metrics,
        )
        result.markers_inserted.append(f"stable_prefix_hash:{stable_hash}")
        return result

    def get_alignment_score(self, messages: list[dict[str, Any]]) -> float:
        """Compute cache alignment score (0-100).

        Higher score means fewer detected volatile patterns. Penalty is a
        flat 10 points per finding, clamped to [0, 100]. This is a
        coarse signal for dashboards — it does not change behavior.
        """
        score = 100.0
        for msg in messages:
            if msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                continue
            findings = detect_volatile_content(content)
            score -= len(findings) * 10
        return max(0.0, min(100.0, score))


def align_for_cache(
    messages: list[dict[str, Any]],
    config: CacheAlignerConfig | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Convenience wrapper that runs detection and returns the unchanged messages.

    Kept as a stable public API; the second tuple element is the stable
    prefix hash for callers that want to track cache prefix drift.
    """
    cfg = config or CacheAlignerConfig()
    aligner = CacheAligner(cfg)
    tokenizer = Tokenizer(EstimatingTokenCounter())  # type: ignore[arg-type]

    result = aligner.apply(messages, tokenizer)

    stable_hash = ""
    for marker in result.markers_inserted:
        if marker.startswith("stable_prefix_hash:"):
            stable_hash = marker.split(":", 1)[1]
            break

    return result.messages, stable_hash
