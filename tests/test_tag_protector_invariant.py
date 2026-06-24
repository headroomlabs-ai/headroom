"""Hotfix-A9 invariant tests for ``headroom.transforms.tag_protector``.

Pins the discard-wrap semantics end-to-end through the public Python
API. Mirrors the Rust ``proptest`` suite but at the language boundary
so a regression in the PyO3 bridge can't slip past Rust-side tests.

Three invariants:

1. **No orphan-tag injection.** ``restore_tags`` never adds bytes that
   weren't already in ``compressed`` *unless* it is substituting a
   placeholder it found. No appends, no prepends, no whitespace
   inserted outside placeholder substitutions.
2. **Idempotence on missing placeholders.** If every placeholder is
   absent from ``compressed``, ``restore_tags`` returns ``compressed``
   byte-for-byte unchanged.
3. **No introduced asymmetry.** ``restore_tags`` never INTRODUCES
   ``opens != closes`` skew. The orphan-append bug fixed by Hotfix-A9
   could turn a balanced input into an asymmetric output whenever a
   placeholder was lost; the discard-wrap path makes that impossible
   because every protected span is a balanced wrap (or self-closer).

The generator is a small deterministic random walk seeded with a
fixed seed so the suite is reproducible and CI-stable. We don't need
``hypothesis`` to get coverage of the relevant shapes (the alphabet
is `[a-z<>/]` plus a handful of literal tag fragments).
"""

from __future__ import annotations

import random

import pytest

from headroom.transforms.tag_protector import protect_tags, restore_tags

# ──────────────────────────────────────────────────────────────────────
# Helpers — independent open/close counters that don't share code with
# the implementation under test, so a parser bug can't mask itself.
# ──────────────────────────────────────────────────────────────────────


def _is_name_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def count_open_tags(s: str) -> int:
    """Count `<name…>` style opening tags. Excludes `</…>` closes and
    `<…/>` self-closers."""
    count = 0
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "<":
            i += 1
            continue
        if i + 1 < n and s[i + 1] == "/":
            i += 1
            continue
        if i + 1 >= n or not _is_name_start(s[i + 1]):
            i += 1
            continue
        # Walk to the matching `>`. Track self-closer marker.
        j = i + 1
        self_closing = False
        while j < n and s[j] != ">":
            if s[j] == "/":
                self_closing = True
            j += 1
        if j >= n:
            break
        if not self_closing:
            count += 1
        i = j + 1
    return count


def count_close_tags(s: str) -> int:
    """Count `</name>` style closing tags."""
    count = 0
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "<":
            i += 1
            continue
        if i + 1 >= n or s[i + 1] != "/":
            i += 1
            continue
        if i + 2 >= n or not _is_name_start(s[i + 2]):
            i += 1
            continue
        j = i + 2
        while j < n and s[j] != ">":
            j += 1
        if j >= n:
            break
        count += 1
        i = j + 1
    return count


# ──────────────────────────────────────────────────────────────────────
# Deterministic content generator. Yields a mix of:
#   - random ASCII letters/punct
#   - balanced custom-tag pairs (`<sys>x</sys>`, `<tool>y</tool>`, …)
#   - bare orphan opens / closes / self-closers
#   - HTML tags that should not be protected
# Seed is fixed so failures reproduce.
# ──────────────────────────────────────────────────────────────────────

_TAG_NAMES = [
    "sys",
    "tool",
    "thinking",
    "system-reminder",
    "EXTREMELY_IMPORTANT",
    "context",
]


def _gen_content(rng: random.Random, max_segments: int = 8) -> str:
    parts: list[str] = []
    n_segments = rng.randint(0, max_segments)
    for _ in range(n_segments):
        roll = rng.random()
        if roll < 0.30:
            # Balanced custom tag.
            name = rng.choice(_TAG_NAMES)
            body = "".join(rng.choice("abcde 123") for _ in range(rng.randint(0, 12)))
            parts.append(f"<{name}>{body}</{name}>")
        elif roll < 0.45:
            # Self-closing custom tag.
            name = rng.choice(_TAG_NAMES)
            parts.append(f"<{name}/>")
        elif roll < 0.55:
            # Orphan open (no close) — exercises the asymmetric-input case.
            name = rng.choice(_TAG_NAMES)
            parts.append(f"<{name}>")
        elif roll < 0.65:
            # Orphan close.
            name = rng.choice(_TAG_NAMES)
            parts.append(f"</{name}>")
        elif roll < 0.75:
            # HTML tag — should be passthrough.
            parts.append("<div>plain</div>")
        else:
            # Plain text segment.
            parts.append("".join(rng.choice("abcde <>/") for _ in range(rng.randint(0, 30))))
    return "".join(parts)


def _strip_placeholders(compressed: str, blocks: list[tuple[str, str]]) -> str:
    out = compressed
    for placeholder, _original in blocks:
        out = out.replace(placeholder, "")
    return out


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


# Number of generated cases. Kept moderate so the suite stays fast in
# CI; bug-shrinking is done by the Rust proptest suite which runs the
# same property at the lower layer.
_CASES = 256
_SEED = 0xA9_CAFE


@pytest.fixture(scope="module")
def rng() -> random.Random:
    return random.Random(_SEED)


def test_restore_idempotent_when_all_placeholders_lost(rng: random.Random) -> None:
    """If every placeholder is stripped before restore, the function
    returns the compressed text byte-for-byte unchanged."""
    for _ in range(_CASES):
        content = _gen_content(rng)
        _cleaned, blocks = protect_tags(content)
        if not blocks:
            continue
        # Drop every placeholder by simulating a compressor that ate
        # them. `compressed` carries no placeholders at all.
        compressed = _gen_content(rng)
        # If the random `compressed` happens to contain a placeholder
        # (vanishingly unlikely with our alphabet, but defensive),
        # skip — the property under test is the *no-placeholder* case.
        if any(p in compressed for (p, _o) in blocks):
            continue
        restored = restore_tags(compressed, blocks)
        assert restored == compressed, (
            f"discard-wrap path corrupted output: blocks={blocks}, "
            f"compressed={compressed!r}, restored={restored!r}"
        )


def test_restore_never_introduces_asymmetry(rng: random.Random) -> None:
    """`restore_tags` never INTRODUCES `opens != closes` skew. With
    every placeholder lost (worst case for the discard-wrap path),
    skew equals the cleaned-minus-placeholders baseline. With every
    placeholder present, skew equals the original input's skew."""
    for _ in range(_CASES):
        content = _gen_content(rng)
        cleaned, blocks = protect_tags(content)

        # Worst case: every placeholder lost. Restored output must
        # equal `cleaned` with placeholders stripped, and so must
        # carry exactly that asymmetry.
        stripped = _strip_placeholders(cleaned, blocks)
        baseline_skew = count_open_tags(stripped) - count_close_tags(stripped)
        restored_lost = restore_tags(stripped, blocks)
        lost_skew = count_open_tags(restored_lost) - count_close_tags(restored_lost)
        assert lost_skew == baseline_skew, (
            f"discard-wrap introduced asymmetry: baseline={baseline_skew}, "
            f"after_restore={lost_skew}, content={content!r}"
        )

        # Full restore: skew matches the original input.
        restored_full = restore_tags(cleaned, blocks)
        full_skew = count_open_tags(restored_full) - count_close_tags(restored_full)
        content_skew = count_open_tags(content) - count_close_tags(content)
        assert full_skew == content_skew, (
            f"full-restore drifted from input skew: input={content_skew}, "
            f"restored={full_skew}, content={content!r}"
        )


def test_restore_no_orphan_byte_injection(rng: random.Random) -> None:
    """`restore_tags` never adds bytes outside placeholder
    substitution. The restored length is bounded above by the
    compressed length plus the size delta of placeholders that were
    actually substituted; lost-placeholder originals contribute zero
    bytes."""
    for _ in range(_CASES):
        content = _gen_content(rng)
        cleaned, blocks = protect_tags(content)
        restored = restore_tags(cleaned, blocks)
        # Bytes added by substitution: for each placeholder that
        # appears in `cleaned`, replace it with `original` (delta =
        # len(original) - len(placeholder)).
        substituted_delta = sum(
            max(0, len(original) - len(placeholder))
            for placeholder, original in blocks
            if placeholder in cleaned
        )
        upper_bound = len(cleaned) + substituted_delta
        assert len(restored) <= upper_bound, (
            f"restored too long: restored={len(restored)} "
            f"upper_bound={upper_bound} cleaned={len(cleaned)} "
            f"content={content!r}"
        )


def test_restore_lost_real_world_tag_does_not_inject_orphan() -> None:
    """Anchor-case mirroring the production findings: when the
    placeholder for a `<system-reminder>` block is lost, the
    compressed text must NOT end with an orphan opening tag."""
    blocks = [
        (
            "{{HEADROOM_TAG_0}}",
            "<system-reminder>[Showing lines 1-50 of 1000 total lines]</system-reminder>",
        )
    ]
    compressed = "compressed body without any placeholder reference"
    restored = restore_tags(compressed, blocks)
    # The bug we are killing: opening tag at the END with no body / no close.
    assert not restored.endswith("<system-reminder>")
    assert "<system-reminder>" not in restored
    assert "</system-reminder>" not in restored
    assert restored == compressed
