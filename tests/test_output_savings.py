"""Tests for headroom.proxy.output_savings — the counterfactual estimator."""

from __future__ import annotations

from headroom.proxy.output_savings import (
    BaselineModel,
    SavingsLedger,
    assign_arm,
    conversation_key_from_body,
    echo_ratio,
    input_bucket,
    model_family,
    stratum_key,
)

# ---------------------------------------------------------------------------
# stratification primitives
# ---------------------------------------------------------------------------


class TestStratification:
    def test_input_buckets_monotone(self):
        assert input_bucket(0) == "xs"
        assert input_bucket(1_999) == "xs"
        assert input_bucket(2_000) == "s"
        assert input_bucket(8_000) == "m"
        assert input_bucket(32_000) == "l"
        assert input_bucket(200_000) == "xl"

    def test_model_family_collapses_point_releases(self):
        assert model_family("claude-opus-4-8") == "opus"
        assert model_family("claude-opus-4-7") == "opus"
        assert model_family("claude-sonnet-4-6") == "sonnet"
        assert model_family("gpt-4o") == "gpt"
        assert model_family("something-weird") == "other"

    def test_stratum_key_is_most_to_least_specific(self):
        key = stratum_key(
            turn_kind="new_user_ask", input_tokens=5000, model="claude-opus-4-8", has_tools=True
        )
        assert key == "opus|new_user_ask|s|tools"

    def test_stratum_key_distinguishes_tools(self):
        a = stratum_key(turn_kind="x", input_tokens=100, model="m", has_tools=True)
        b = stratum_key(turn_kind="x", input_tokens=100, model="m", has_tools=False)
        assert a != b


# ---------------------------------------------------------------------------
# holdout arm assignment
# ---------------------------------------------------------------------------


class TestArmAssignment:
    def test_zero_holdout_always_treatment(self):
        assert assign_arm("anything", 0.0) == "treatment"

    def test_full_holdout_always_control(self):
        assert assign_arm("anything", 1.0) == "control"

    def test_assignment_is_stable_for_same_key(self):
        assert assign_arm("conv-123", 0.5) == assign_arm("conv-123", 0.5)

    def test_roughly_matches_fraction(self):
        keys = [f"conv-{i}" for i in range(4000)]
        control = sum(1 for k in keys if assign_arm(k, 0.1) == "control")
        # 10% holdout over 4000 keys — allow generous slack for hash noise.
        assert 250 < control < 550

    def test_conversation_key_stable_across_turns(self):
        first = {
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "build a cache"}],
        }
        later = {
            "model": "claude-opus-4-8",
            "messages": [
                {"role": "user", "content": "build a cache"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "tool_result", "content": "x"}]},
            ],
        }
        assert conversation_key_from_body(first) == conversation_key_from_body(later)

    def test_conversation_key_differs_by_first_message(self):
        a = {"model": "m", "messages": [{"role": "user", "content": "task A"}]}
        b = {"model": "m", "messages": [{"role": "user", "content": "task B"}]}
        assert conversation_key_from_body(a) != conversation_key_from_body(b)


# ---------------------------------------------------------------------------
# baseline model
# ---------------------------------------------------------------------------


class TestBaselineModel:
    def test_observe_and_lookup_exact(self):
        m = BaselineModel()
        for v in (100, 200, 300):
            m.observe("opus|new_user_ask|s|tools", v)
        mean, var, n = m.lookup("opus|new_user_ask|s|tools")
        assert mean == 200.0
        assert n == 3
        assert var > 0

    def test_lookup_backs_off_to_prefix(self):
        m = BaselineModel()
        m.observe("opus|new_user_ask|s|tools", 500)
        # Query a sibling stratum (different tools flag) — backs off on prefix.
        mean, _, n = m.lookup("opus|new_user_ask|s|notools")
        assert mean == 500.0
        assert n == 1

    def test_lookup_falls_back_to_global(self):
        m = BaselineModel()
        m.observe("opus|a|s|tools", 100)
        m.observe("sonnet|b|m|notools", 300)
        mean, _, n = m.lookup("gpt|totally|xl|tools")
        assert mean == 200.0  # global mean of 100 and 300
        assert n == 2

    def test_roundtrip_serialization(self):
        m = BaselineModel()
        for v in (10, 20, 30):
            m.observe("k|a|s|tools", v)
        m2 = BaselineModel.from_dict(m.to_dict())
        assert m2.lookup("k|a|s|tools") == m.lookup("k|a|s|tools")
        assert m2.total_samples == 3


# ---------------------------------------------------------------------------
# synthetic-control estimate
# ---------------------------------------------------------------------------


class TestEstimateFromBaseline:
    def _ledger_with_baseline(self, baseline_val: float, n: int = 50) -> SavingsLedger:
        ledger = SavingsLedger()
        for _ in range(n):
            ledger.baseline.observe("opus|new_user_ask|s|tools", baseline_val)
        return ledger

    def test_positive_savings_when_treatment_below_baseline(self):
        ledger = self._ledger_with_baseline(1000.0)
        for _ in range(20):
            ledger.record("treatment", "opus|new_user_ask|s|tools", 700)
        est = ledger.estimate_from_baseline()
        assert est.kind == "estimated"
        assert est.n_requests == 20
        # 20 requests * (1000 - 700) = 6000 tokens saved.
        assert abs(est.tokens_saved - 6000) < 1e-6
        assert abs(est.pct - 30.0) < 1e-6

    def test_signed_delta_not_clamped(self):
        # A treatment request LARGER than baseline must reduce the total, not
        # be clamped to zero (clamping would bias the estimate upward).
        ledger = self._ledger_with_baseline(1000.0)
        ledger.record("treatment", "opus|new_user_ask|s|tools", 700)
        ledger.record("treatment", "opus|new_user_ask|s|tools", 1400)
        est = ledger.estimate_from_baseline()
        # (1000-700) + (1000-1400) = 300 - 400 = -100
        assert abs(est.tokens_saved - (-100)) < 1e-6

    def test_zero_baseline_samples_yields_zero(self):
        ledger = SavingsLedger()
        ledger.record("treatment", "opus|x|s|tools", 500)
        est = ledger.estimate_from_baseline()
        # No baseline at all -> global is empty -> nothing contributes.
        assert est.n_requests == 0
        assert est.tokens_saved == 0.0

    def test_ci_band_brackets_point_estimate(self):
        ledger = SavingsLedger()
        for v in (900, 1000, 1100):
            for _ in range(20):
                ledger.baseline.observe("opus|new_user_ask|s|tools", v)
        for v in (600, 700, 800):
            for _ in range(20):
                ledger.record("treatment", "opus|new_user_ask|s|tools", v)
        est = ledger.estimate_from_baseline()
        assert est.ci_low_pct <= est.pct <= est.ci_high_pct
        assert est.ci_low_pct < est.ci_high_pct  # nonzero band given spread


# ---------------------------------------------------------------------------
# A/B measured estimate
# ---------------------------------------------------------------------------


class TestEstimateFromHoldout:
    def test_none_without_control_data(self):
        ledger = SavingsLedger()
        ledger.record("treatment", "opus|x|s|tools", 500)
        assert ledger.estimate_from_holdout() is None

    def test_measured_difference_of_means(self):
        ledger = SavingsLedger()
        for _ in range(30):
            ledger.record("control", "opus|new_user_ask|s|tools", 1000)
            ledger.record("treatment", "opus|new_user_ask|s|tools", 750)
        est = ledger.estimate_from_holdout()
        assert est is not None
        assert est.kind == "measured"
        # 30 * (1000 - 750) = 7500 saved; 25% of the 1000 baseline.
        assert abs(est.tokens_saved - 7500) < 1e-6
        assert abs(est.pct - 25.0) < 1e-6

    def test_only_strata_present_in_both_arms_contribute(self):
        ledger = SavingsLedger()
        for _ in range(10):
            ledger.record("control", "opus|a|s|tools", 1000)
            ledger.record("treatment", "opus|a|s|tools", 800)
        # Treatment-only stratum must not contribute (no control to compare).
        ledger.record("treatment", "opus|b|m|notools", 50)
        est = ledger.estimate_from_holdout()
        assert est is not None
        assert est.n_requests == 10

    def test_best_estimate_prefers_measured(self):
        ledger = SavingsLedger()
        for _ in range(10):
            ledger.baseline.observe("opus|a|s|tools", 1000)
            ledger.record("control", "opus|a|s|tools", 1000)
            ledger.record("treatment", "opus|a|s|tools", 900)
        assert ledger.best_estimate().kind == "measured"

    def test_best_estimate_falls_back_to_estimated(self):
        ledger = SavingsLedger()
        for _ in range(10):
            ledger.baseline.observe("opus|a|s|tools", 1000)
            ledger.record("treatment", "opus|a|s|tools", 900)
        assert ledger.best_estimate().kind == "estimated"


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


class TestLedgerPersistence:
    def test_roundtrip(self, tmp_path):
        ledger = SavingsLedger()
        ledger.baseline.observe("opus|a|s|tools", 1000)
        ledger.record("treatment", "opus|a|s|tools", 800)
        ledger.record("control", "opus|a|s|tools", 1000)
        path = tmp_path / "savings.json"
        ledger.save(path)
        loaded = SavingsLedger.load(path)
        assert loaded.estimate_from_baseline().tokens_saved == (
            ledger.estimate_from_baseline().tokens_saved
        )
        assert loaded.estimate_from_holdout() is not None

    def test_load_missing_returns_empty(self, tmp_path):
        ledger = SavingsLedger.load(tmp_path / "nope.json")
        assert ledger.baseline.total_samples == 0

    def test_load_corrupt_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        ledger = SavingsLedger.load(p)
        assert ledger.baseline.total_samples == 0


# ---------------------------------------------------------------------------
# echo ratio (direct waste signal)
# ---------------------------------------------------------------------------


class TestEchoRatio:
    def test_full_echo(self):
        ctx = "the quick brown fox jumps over the lazy dog every single time"
        assert echo_ratio(ctx, ctx, n=4) == 1.0

    def test_no_echo(self):
        out = "completely unrelated words appearing nowhere within the given source context here"
        ctx = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        assert echo_ratio(out, ctx, n=4) == 0.0

    def test_partial_echo_between_zero_and_one(self):
        ctx = "alpha beta gamma delta epsilon zeta eta theta"
        out = "alpha beta gamma delta brand new tokens here now"
        r = echo_ratio(out, ctx, n=4)
        assert 0.0 < r < 1.0

    def test_short_output_returns_zero(self):
        assert echo_ratio("a b", "a b c d e f g h", n=8) == 0.0
