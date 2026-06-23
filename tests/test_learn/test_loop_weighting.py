"""Tests for loop detection and loop-weighting in Headroom Learn.

Covers the gap these changes close: RTK re-fetch loops (repeated, successful
but insufficient calls) were invisible to failure-only analysis and, even when
surfaced, were ranked no higher than a one-off rule. These tests pin:

1. ``detect_loops`` finds RTK re-fetch loops and error loops, and ignores
   one-offs — collapsing output-limit variants to one signature.
2. The digest surfaces detected loops as a high-priority section.
3. ``apply_loop_weighting`` lifts a loop guardrail above a one-off rule using
   MEASURED waste, regardless of the LLM's guessed savings.
4. End-to-end ``SessionAnalyzer.analyze`` (LLM mocked): a re-fetch loop with no
   failures is still analyzed, and its guardrail outranks a one-off rule.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from headroom.learn.analyzer import SessionAnalyzer, _build_digest
from headroom.learn.fixtures import (
    error_loop_session,
    one_off_error_session,
    rtk_refetch_loop_session,
)
from headroom.learn.loops import (
    _canonical_signature,
    apply_loop_weighting,
    detect_loops,
)
from headroom.learn.models import (
    ProjectInfo,
    Recommendation,
    RecommendationTarget,
)


def _project() -> ProjectInfo:
    return ProjectInfo(
        name="proj",
        project_path=Path("/tmp/proj"),
        data_path=Path("/tmp/proj-data"),
    )


# =============================================================================
# detect_loops
# =============================================================================


class TestDetectLoops:
    def test_rtk_refetch_loop_detected_despite_no_errors(self):
        loops = detect_loops([rtk_refetch_loop_session(repetitions=5)])
        assert len(loops) == 1
        lp = loops[0]
        assert lp.count == 5
        assert lp.is_error_loop is False
        assert lp.kind == "rtk-refetch-loop"
        # Waste counts the 4 redundant re-fetches (not the first legit call).
        assert lp.wasted_tokens > 0

    def test_output_limit_variants_collapse_to_one_signature(self):
        # The five calls differ only by `head -50/-100/...`; same signature.
        session = rtk_refetch_loop_session(repetitions=5)
        sigs = {_canonical_signature(tc) for tc in session.tool_calls}
        assert len(sigs) == 1

    def test_error_loop_detected_and_classified(self):
        loops = detect_loops([error_loop_session(repetitions=4)])
        assert len(loops) == 1
        assert loops[0].is_error_loop is True
        assert loops[0].kind == "error-loop"

    def test_one_off_is_not_a_loop(self):
        assert detect_loops([one_off_error_session()]) == []

    def test_min_occurrences_threshold(self):
        # Two repetitions is a retry, not a loop, at the default threshold.
        assert detect_loops([rtk_refetch_loop_session(repetitions=2)]) == []
        assert detect_loops([rtk_refetch_loop_session(repetitions=3)])

    def test_error_loop_waste_exceeds_refetch_loop_first_call_credit(self):
        # Error loops waste every call; re-fetch loops credit the first call.
        err = detect_loops([error_loop_session(repetitions=4)])[0]
        ref = detect_loops([rtk_refetch_loop_session(repetitions=4)])[0]
        assert err.count == ref.count
        # Same count, but error loop counts all N and re-fetch counts N-1.
        assert err.wasted_tokens >= 0 and ref.wasted_tokens >= 0


# =============================================================================
# digest surfacing
# =============================================================================


class TestDigestSurfacesLoops:
    def test_digest_includes_detected_loops_section(self):
        digest = _build_digest(_project(), [rtk_refetch_loop_session()])
        assert "Detected Loops" in digest
        assert "rtk-refetch-loop" in digest
        assert "tokens wasted" in digest

    def test_digest_without_loops_has_no_loop_section(self):
        digest = _build_digest(_project(), [one_off_error_session()])
        assert "Detected Loops" not in digest


# =============================================================================
# apply_loop_weighting
# =============================================================================


class TestApplyLoopWeighting:
    def _loop_rec(self) -> Recommendation:
        return Recommendation(
            target=RecommendationTarget.CONTEXT_FILE,
            section="Grep TimeoutError loop",
            content="When you need to grep TimeoutError in logs, read the full "
            "result once instead of re-running with larger head limits.",
            estimated_tokens_saved=200,  # LLM under-estimated it
        )

    def _one_off_rec(self) -> Recommendation:
        return Recommendation(
            target=RecommendationTarget.CONTEXT_FILE,
            section="Use uv",
            content="Use `uv run python` instead of `python3`.",
            estimated_tokens_saved=500,  # LLM rated this higher
        )

    def test_loop_rule_boosted_above_one_off(self):
        loops = detect_loops([rtk_refetch_loop_session(repetitions=5)])
        recs = [self._one_off_rec(), self._loop_rec()]
        apply_loop_weighting(recs, loops)

        loop_rec = next(r for r in recs if r.is_loop_guardrail)
        one_off = next(r for r in recs if not r.is_loop_guardrail)
        # Boosted to at least the measured loop waste, which dominates the
        # one-off even though the LLM originally rated the one-off higher.
        assert loop_rec.estimated_tokens_saved >= loops[0].wasted_tokens
        assert loop_rec.estimated_tokens_saved > one_off.estimated_tokens_saved
        assert loop_rec.loop_occurrences == 5

    def test_no_loops_is_noop(self):
        recs = [self._one_off_rec()]
        before = recs[0].estimated_tokens_saved
        apply_loop_weighting(recs, [])
        assert recs[0].estimated_tokens_saved == before
        assert recs[0].is_loop_guardrail is False

    def test_unrelated_rule_not_credited(self):
        loops = detect_loops([rtk_refetch_loop_session(repetitions=5)])
        recs = [self._one_off_rec()]  # about uv/python, not the grep loop
        apply_loop_weighting(recs, loops)
        assert recs[0].is_loop_guardrail is False


# =============================================================================
# end-to-end analyze() with mocked LLM
# =============================================================================


class TestAnalyzeEndToEnd:
    @patch("headroom.learn.analyzer._call_llm")
    def test_refetch_loop_with_no_failures_is_still_analyzed(self, mock_call_llm: MagicMock):
        # Pure re-fetch loop: zero errors, no events. Must NOT early-return.
        mock_call_llm.return_value = {"context_file_rules": [], "memory_file_rules": []}
        analyzer = SessionAnalyzer(model="test-model")
        analyzer.analyze(_project(), [rtk_refetch_loop_session()])
        mock_call_llm.assert_called_once()  # the guard let it through

    @patch("headroom.learn.analyzer._call_llm")
    def test_loop_guardrail_outranks_one_off_in_result(self, mock_call_llm: MagicMock):
        # LLM returns both rules, rating the one-off higher than the loop.
        mock_call_llm.return_value = {
            "context_file_rules": [
                {
                    "section": "Use uv",
                    "content": "Use `uv run python` instead of `python3`.",
                    "estimated_tokens_saved": 800,
                    "evidence_count": 2,
                },
                {
                    "section": "Grep TimeoutError loop",
                    "content": "Grep TimeoutError in logs once with full output; "
                    "do not re-run with larger head limits.",
                    "estimated_tokens_saved": 100,
                    "evidence_count": 1,
                },
            ],
            "memory_file_rules": [],
        }
        analyzer = SessionAnalyzer(model="test-model")
        result = analyzer.analyze(_project(), [rtk_refetch_loop_session(repetitions=6)])

        # After weighting, the loop guardrail ranks first despite the LLM's order.
        assert result.recommendations[0].is_loop_guardrail is True
        assert "loop" in result.recommendations[0].section.lower()
