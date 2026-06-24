"""Tests for the Python ``CompressionPolicy`` and its parity with Rust.

The Python module is a hand-mirror of
``headroom_core::compression_policy::CompressionPolicy``. These tests
pin both halves: that the per-mode values are right, and that the
Python and Rust sides agree on the field map. F2.2 will likely retire
the hand-mirror via PyO3 — until then, this file is the canary.

F2.2 extends the F2.1 surface with three tuning fields:
``volatile_token_threshold``, ``max_lossy_ratio``, ``toin_read_only``.
Per-mode value tests below mirror the Rust unit tests in
``crates/headroom-core/src/compression_policy.rs``.
"""

from __future__ import annotations

import pytest

from headroom.proxy.auth_mode import AuthMode
from headroom.transforms.compression_policy import (
    CompressionPolicy,
    policy_default_payg,
    policy_for_mode,
)


class TestCompressionPolicyForMode:
    """Per-mode field assertions. Mirrors the Rust unit tests in
    `crates/headroom-core/src/compression_policy.rs`.
    """

    def test_payg_is_aggressive(self):
        p = policy_for_mode(AuthMode.PAYG)
        assert p.live_zone_only is False, "PAYG can touch outside live zone"
        assert p.cache_aligner_enabled is True, "PAYG runs cache aligner"

    def test_payg_tuning_fields_aggressive(self):
        # F2.2: per-mode tuning fields. Values are the conservative
        # defaults pending bake telemetry (see PR body and module
        # docstring).
        p = policy_for_mode(AuthMode.PAYG)
        assert p.volatile_token_threshold == 128, (
            "PAYG volatile threshold is the relaxed default; F2.2-followup will tune"
        )
        assert p.max_lossy_ratio == pytest.approx(0.45), (
            "PAYG max_lossy_ratio caps lossy paths at 0.45; F2.2-followup will tune"
        )
        assert p.toin_read_only is False, (
            "PAYG keeps TOIN write-enabled — network effect feeds on PAYG traffic"
        )

    def test_oauth_matches_payg_today(self):
        # Canary: when F2.2-followup diverges OAuth from PAYG, this test
        # fails and forces a deliberate update on BOTH sides (Rust +
        # Python). Covers ALL fields (F2.1 + F2.2) so a future field-
        # level divergence trips the assertion just as loudly as a flag
        # flip.
        oauth = policy_for_mode(AuthMode.OAUTH)
        payg = policy_for_mode(AuthMode.PAYG)
        assert oauth == payg, (
            "F2.1+F2.2 ship OAuth=PAYG; F2.2-followup will diverge based on telemetry. "
            "If you are reading this assertion failure: also update "
            "crates/headroom-core/src/compression_policy.rs "
            "::oauth_matches_payg_today, otherwise the Rust + Python "
            "parities silently drift apart."
        )

    def test_subscription_disables_cache_aligner(self):
        p = policy_for_mode(AuthMode.SUBSCRIPTION)
        assert p.live_zone_only is True, "Subscription is live-zone-only"
        assert p.cache_aligner_enabled is False, (
            "Subscription MUST skip cache aligner — load-bearing for issues #327 / #388"
        )

    def test_subscription_tuning_fields_conservative(self):
        # F2.2: per-mode tuning fields. Subscription is the conservative
        # end — tighter threshold, lower lossy cap, TOIN read-only — so
        # cache prefixes stay stable and the learning pool isn't
        # mutated from cache-stability-sensitive traffic.
        p = policy_for_mode(AuthMode.SUBSCRIPTION)
        assert p.volatile_token_threshold == 32, (
            "Subscription volatile threshold flags content earlier (cache stability)"
        )
        assert p.max_lossy_ratio == pytest.approx(0.25), (
            "Subscription max_lossy_ratio caps lossy paths at 0.25 (conservative)"
        )
        assert p.toin_read_only is True, (
            "Subscription MUST be TOIN read-only — load-bearing for keeping the "
            "learning pool consistent across cache-sensitive traffic"
        )

    def test_max_lossy_ratio_in_unit_interval(self):
        # Defensive: every per-mode `max_lossy_ratio` MUST be in
        # ``[0.0, 1.0]`` because it expresses a fraction. A tune that
        # drifts outside the unit interval is a bug — catch it cheaply
        # here rather than at the eventual consumer site.
        for mode in (AuthMode.PAYG, AuthMode.OAUTH, AuthMode.SUBSCRIPTION):
            r = policy_for_mode(mode).max_lossy_ratio
            assert 0.0 <= r <= 1.0, f"max_lossy_ratio for {mode!r} = {r} is outside [0.0, 1.0]"


class TestPolicyDefaultPayg:
    """The constant used when the enforcement flag is disabled."""

    def test_default_payg_equals_for_mode_payg(self):
        assert policy_default_payg() == policy_for_mode(AuthMode.PAYG)


class TestImmutability:
    """The struct is `frozen=True`; mutation must raise."""

    def test_policy_is_frozen(self):
        p = policy_for_mode(AuthMode.PAYG)
        with pytest.raises((AttributeError, Exception)):
            # Attempting to mutate a frozen dataclass raises
            # FrozenInstanceError (subclass of AttributeError on
            # CPython 3.10+). Catch both for compatibility.
            p.live_zone_only = True  # type: ignore[misc]

    def test_f22_tuning_fields_also_frozen(self):
        # Each F2.2 field gets its own immutability assertion — a
        # future refactor that accidentally drops `frozen=True` on the
        # dataclass would silently allow per-request mutation. The
        # F2.1 test only covered ``live_zone_only``; explicit per-
        # field coverage prevents quiet regressions.
        p = policy_for_mode(AuthMode.PAYG)
        for attr_name in ("volatile_token_threshold", "max_lossy_ratio", "toin_read_only"):
            with pytest.raises((AttributeError, Exception)):
                setattr(p, attr_name, 0)  # type: ignore[misc]


class TestRustParityFieldMap:
    """The Python policy must have the same fields as the Rust struct.

    The canonical Rust struct lives at
    ``crates/headroom-core/src/compression_policy.rs``. When you add a
    field there for a future PR, add it here AND update this test.
    Otherwise the parity silently drifts.
    """

    def test_field_set_matches_rust(self):
        # Hard-coded set — when Rust grows fields, this test fails until
        # Python catches up. F2.2 added three: volatile_token_threshold,
        # max_lossy_ratio, toin_read_only.
        expected_fields = {
            "live_zone_only",
            "cache_aligner_enabled",
            "volatile_token_threshold",
            "max_lossy_ratio",
            "toin_read_only",
        }
        actual_fields = {f.name for f in CompressionPolicy.__dataclass_fields__.values()}
        assert actual_fields == expected_fields, (
            f"Python CompressionPolicy fields drifted from Rust. "
            f"Expected exactly {expected_fields}, got {actual_fields}. "
            f"Update both `headroom/transforms/compression_policy.py` "
            f"and `crates/headroom-core/src/compression_policy.rs` in "
            f"the same commit."
        )


class TestNetCostFormula:
    """Net-cost mutation formula (#856) — Rust parity.

    Scenario values are golden: the Rust unit tests in
    ``crates/headroom-core/src/compression_policy.rs`` assert the
    identical numbers, so a drift in either side trips the parity pair
    loudly.
    """

    def test_small_shave_deep_suffix_is_loss(self):
        # 2000*(1.25 + 0.1*9) - 1.0*1.15*52000 = 4300 - 59800 = -55500.
        p = policy_for_mode(AuthMode.PAYG)
        gain = p.net_mutation_gain(2_000, 50_000, 10.0, 1.0)
        assert abs(gain - (-55_500.0)) < 1.0
        assert not p.should_mutate_deep(2_000, 50_000, 10.0, 1.0)

    def test_big_shave_shallow_suffix_is_win(self):
        # 50000*(1.25 + 0.1*2) - 1.0*1.15*60000 = 72500 - 69000 = 3500.
        # Tight but positive — consistent with the 2.3-read break-even.
        p = policy_for_mode(AuthMode.PAYG)
        gain = p.net_mutation_gain(50_000, 10_000, 3.0, 1.0)
        assert abs(gain - 3_500.0) < 1.0
        assert p.should_mutate_deep(50_000, 10_000, 3.0, 1.0)

    def test_no_suffix_edit_profitable_with_reads_remaining(self):
        # S = 0: warm-case saving is the avoided rereads, dT*r*R —
        # positive whenever at least one read remains. At R=0 with a
        # warm cache the gain is exactly 0 (already written, never read
        # again): pointless rather than harmful.
        p = policy_for_mode(AuthMode.SUBSCRIPTION)
        assert p.should_mutate_deep(1, 0, 1.0, 1.0)
        assert p.should_mutate_deep(2_000, 0, 1.0, 1.0)
        assert abs(p.net_mutation_gain(2_000, 0, 0.0, 1.0)) < 1e-6

    def test_cold_cache_ignores_suffix(self):
        # P_alive = 0 (TTL lapsed): the idle-timer compaction window.
        p = policy_for_mode(AuthMode.PAYG)
        assert p.should_mutate_deep(2_000, 50_000, 0.0, 0.0)

    def test_clamps_out_of_range_inputs(self):
        p = policy_for_mode(AuthMode.PAYG)
        clamped = p.net_mutation_gain(2_000, 50_000, -5.0, 7.0)
        reference = p.net_mutation_gain(2_000, 50_000, 0.0, 1.0)
        assert abs(clamped - reference) < 1e-6

    def test_nan_inputs_guarded(self):
        # NaN reads -> 0, NaN p_alive -> 1 (same as Rust): the gain stays
        # finite instead of poisoning the mutate decision.
        import math

        p = policy_for_mode(AuthMode.PAYG)
        guarded = p.net_mutation_gain(2_000, 50_000, float("nan"), float("nan"))
        assert math.isfinite(guarded)
        reference = p.net_mutation_gain(2_000, 50_000, 0.0, 1.0)
        assert abs(guarded - reference) < 1e-6

    def test_negative_int_inputs_clamped(self):
        # Rust takes u32 — negative Python ints must not flip the sign of
        # the result; they clamp to 0.
        p = policy_for_mode(AuthMode.PAYG)
        assert p.net_mutation_gain(-2_000, -50_000, 5.0, 1.0) == p.net_mutation_gain(0, 0, 5.0, 1.0)
        assert p.break_even_reads(-5, 10_000) == 0.0
        assert p.net_mutation_gain(2_000, -1, 5.0, 1.0) == p.net_mutation_gain(2_000, 0, 5.0, 1.0)

    def test_break_even_reads_matches_research_anchor(self):
        # R = 11.5*S/dT, the #856 anchors exactly: 2K/50K -> 287.5;
        # 50K/10K -> 2.3; dT=0 -> 0.
        p = policy_for_mode(AuthMode.PAYG)
        assert abs(p.break_even_reads(2_000, 50_000) - 287.5) < 0.5
        assert abs(p.break_even_reads(50_000, 10_000) - 2.3) < 0.05
        assert p.break_even_reads(0, 10_000) == 0.0

    def test_constants_match_rust(self):
        from headroom.transforms.compression_policy import (
            CACHE_READ_MULTIPLIER,
            CACHE_WRITE_MULTIPLIER,
        )

        assert CACHE_WRITE_MULTIPLIER == 1.25
        assert CACHE_READ_MULTIPLIER == 0.1
