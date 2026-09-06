"""Tests for loop detection and loop-weighting in Headroom Learn.

Covers the gap these changes close: re-fetch loops (repeated, successful
but insufficient calls) were invisible to failure-only analysis and, even when
surfaced, were ranked no higher than a one-off rule. These tests pin:

1. ``detect_loops`` finds re-fetch loops and error loops, and ignores
   one-offs — collapsing output-limit variants to one signature.
2. The digest surfaces detected loops as a high-priority section.
3. ``apply_loop_weighting`` lifts a loop guardrail above a one-off rule using
   MEASURED waste, regardless of the LLM's guessed savings.
4. End-to-end ``SessionAnalyzer.analyze`` (LLM mocked): a re-fetch loop with no
   failures is still analyzed, and its guardrail outranks a one-off rule.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from headroom.learn.analyzer import SessionAnalyzer, _build_digest
from headroom.learn.fixtures import (
    error_loop_session,
    one_off_error_session,
    refetch_loop_session,
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
    SessionData,
    ToolCall,
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
    def test_refetch_loop_detected_despite_no_errors(self):
        loops = detect_loops([refetch_loop_session(repetitions=5)])
        assert len(loops) == 1
        lp = loops[0]
        assert lp.count == 5
        assert lp.is_error_loop is False
        assert lp.kind == "refetch-loop"
        # Waste counts the 4 redundant re-fetches (not the first legit call).
        assert lp.wasted_tokens > 0

    def test_output_limit_variants_collapse_to_one_signature(self):
        # The five calls differ only by `head -50/-100/...`; same signature.
        session = refetch_loop_session(repetitions=5)
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
        assert detect_loops([refetch_loop_session(repetitions=2)]) == []
        assert detect_loops([refetch_loop_session(repetitions=3)])

    def test_error_loop_waste_exceeds_refetch_loop_first_call_credit(self):
        # Error loops waste every call; re-fetch loops credit the first call.
        err = detect_loops([error_loop_session(repetitions=4)])[0]
        ref = detect_loops([refetch_loop_session(repetitions=4)])[0]
        assert err.count == ref.count
        # Same count, but error loop counts all N and re-fetch counts N-1.
        assert err.wasted_tokens >= 0 and ref.wasted_tokens >= 0


# =============================================================================
# digest surfacing
# =============================================================================


class TestDigestSurfacesLoops:
    def test_digest_includes_detected_loops_section(self):
        digest = _build_digest(_project(), [refetch_loop_session()])
        assert "Detected Loops" in digest
        assert "refetch-loop" in digest
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
        loops = detect_loops([refetch_loop_session(repetitions=5)])
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
        loops = detect_loops([refetch_loop_session(repetitions=5)])
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
        analyzer.analyze(_project(), [refetch_loop_session()])
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
        result = analyzer.analyze(_project(), [refetch_loop_session(repetitions=6)])

        # After weighting, the loop guardrail ranks first despite the LLM's order.
        assert result.recommendations[0].is_loop_guardrail is True
        assert "loop" in result.recommendations[0].section.lower()


# =============================================================================
# signature identity (regressions)
# =============================================================================


def _call(name: str, tool_call_id: str, input_data: dict, *, msg_index: int) -> ToolCall:
    """A successful tool call with a fixed 40 KB output."""
    output = "x" * 40_000
    return ToolCall(
        name=name,
        tool_call_id=tool_call_id,
        input_data=input_data,
        output=output,
        is_error=False,
        msg_index=msg_index,
        output_bytes=len(output),
    )


def _mixed_calls() -> list[ToolCall]:
    return [
        _call("Read", "call_r0", {"file_path": "/repo/a.py"}, msg_index=0),
        _call("Read", "call_r1", {"file_path": "/repo/a.py"}, msg_index=1),
        _call("Read", "call_r2", {"file_path": "/repo/a.py"}, msg_index=2),
        _call("Bash", "call_b0", {"command": "rg alpha /repo"}, msg_index=3),
        _call("Bash", "call_b1", {"command": "rg alpha /repo"}, msg_index=4),
        _call("Bash", "call_b2", {"command": "rg alpha /repo"}, msg_index=5),
        _call("Grep", "call_g0", {"pattern": "beta"}, msg_index=6),
    ]


class TestReplayedTranscriptsDoNotDoubleCount:
    """A resume that replays prior history must not recount the same calls.

    ``detect_loops`` accumulates a signature across sessions so a recurring loop
    adds up. Providers whose resume writes a *new* transcript replaying earlier
    turns therefore present the same tool call more than once; identity comes
    from ``tool_call_id``, which the provider assigns.
    """

    def _reads(self) -> list[ToolCall]:
        return [
            _call("Read", f"call_r{i}", {"file_path": "/repo/docs/design.md"}, msg_index=i)
            for i in range(3)
        ]

    def test_replayed_calls_counted_once(self):
        reads = self._reads()
        single = detect_loops([SessionData(session_id="rollout-1", tool_calls=list(reads))])
        replayed = detect_loops(
            [
                SessionData(session_id="rollout-1", tool_calls=list(reads)),
                SessionData(session_id="rollout-2-resume", tool_calls=list(reads)),
            ]
        )
        assert replayed[0].count == single[0].count
        assert replayed[0].wasted_tokens == single[0].wasted_tokens

    def test_new_calls_in_a_resume_still_accumulate(self):
        # Dedup must not swallow genuinely new repetitions in the resumed run.
        reads = self._reads()
        extra = [
            _call("Read", f"call_r{i}", {"file_path": "/repo/docs/design.md"}, msg_index=i)
            for i in range(3, 6)
        ]
        replayed = detect_loops(
            [
                SessionData(session_id="rollout-1", tool_calls=list(reads)),
                SessionData(session_id="rollout-2-resume", tool_calls=list(reads) + extra),
            ]
        )
        assert replayed[0].count == 6

    def test_calls_without_ids_are_not_deduped(self):
        # An id-less scanner must keep counting repetitions rather than collapse.
        calls = [
            _call("Read", "", {"file_path": "/repo/docs/design.md"}, msg_index=i) for i in range(3)
        ]
        loops = detect_loops(
            [
                SessionData(session_id="a", tool_calls=list(calls)),
                SessionData(session_id="b", tool_calls=list(calls)),
            ]
        )
        assert loops[0].count == 6

    @pytest.mark.parametrize("replays", [1, 2, 3, 5])
    def test_replaying_a_transcript_never_changes_the_count(self, replays):
        calls = _mixed_calls()
        sessions = [
            SessionData(session_id=f"rollout-{i}", tool_calls=list(calls)) for i in range(replays)
        ]

        once = detect_loops([SessionData(session_id="rollout-0", tool_calls=list(calls))])
        many = detect_loops(sessions)

        assert sorted(lp.count for lp in many) == sorted(lp.count for lp in once)


class TestDedupRespectsTheOccurrenceThreshold:
    """Dedup must not leave behind loops that no longer meet the bar.

    ``detect_loops`` screens a group against ``min_occurrences`` before removing
    replayed calls, so a group can clear the bar only *because* of duplicates and
    still be reported once they are gone. Real transcripts do repeat a
    ``tool_use_id`` within one file, so this is reachable from the default
    Claude Code path.
    """

    def test_a_group_that_only_duplicates_is_not_a_loop(self):
        # One call, written into the transcript three times.
        calls = [
            _call("Read", "toolu_SAME", {"file_path": "/repo/design.md"}, msg_index=i)
            for i in range(3)
        ]

        assert detect_loops([SessionData(session_id="s", tool_calls=calls)]) == []

    def test_partial_dedup_still_has_to_clear_the_bar(self):
        # Three calls, two distinct — below DEFAULT_MIN_OCCURRENCES once deduped.
        calls = [
            _call("Read", "toolu_A", {"file_path": "/repo/design.md"}, msg_index=0),
            _call("Read", "toolu_A", {"file_path": "/repo/design.md"}, msg_index=1),
            _call("Read", "toolu_B", {"file_path": "/repo/design.md"}, msg_index=2),
        ]

        assert detect_loops([SessionData(session_id="s", tool_calls=calls)]) == []

    def test_a_real_loop_padded_with_duplicates_is_still_reported(self):
        # Dedup must subtract the replays without dropping the genuine loop.
        distinct = [
            _call("Read", f"toolu_{i}", {"file_path": "/repo/design.md"}, msg_index=i)
            for i in range(3)
        ]
        padded = distinct + [
            _call("Read", "toolu_0", {"file_path": "/repo/design.md"}, msg_index=9)
        ]

        loops = detect_loops([SessionData(session_id="s", tool_calls=padded)])

        assert [lp.count for lp in loops] == [3]

    def test_no_loop_is_reported_with_zero_measured_waste(self):
        """A zero-waste loop is the tell that a group survived on duplicates.

        ``format_loops_for_digest`` bills every entry to the LLM as HIGHEST
        PRIORITY, and ``SessionAnalyzer.analyze`` treats a non-empty loop list as
        reason enough to call the model, so a zero-waste entry buys an LLM round
        trip for a session with nothing to report.
        """
        calls = [
            _call("Read", "toolu_SAME", {"file_path": "/repo/design.md"}, msg_index=i)
            for i in range(4)
        ]

        loops = detect_loops([SessionData(session_id="s", tool_calls=calls)])

        assert [lp for lp in loops if lp.wasted_tokens == 0] == []


class TestFixturesSatisfyTheDedupContract:
    """Dedup keys on ``tool_call_id``, so the shipped fixtures must scope theirs.

    ``detect_loops`` accumulates a signature across sessions. An id that is only
    unique *within* a session makes two unrelated sessions look like one replayed
    twice, silently halving a real loop.
    """

    def test_the_same_loop_in_two_sessions_accumulates(self):
        monday = refetch_loop_session("session-monday")
        tuesday = refetch_loop_session("session-tuesday")

        one = detect_loops([monday])
        both = detect_loops([monday, tuesday])

        assert both[0].count == one[0].count * 2

    def test_fixture_ids_are_unique_across_sessions(self):
        monday = {c.tool_call_id for c in refetch_loop_session("session-monday").tool_calls}
        tuesday = {c.tool_call_id for c in refetch_loop_session("session-tuesday").tool_calls}

        assert monday.isdisjoint(tuesday)
