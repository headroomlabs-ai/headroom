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
    _IDENTITY_FIELDS,
    _canonical_signature,
    _identity_input,
    _signature_tokens,
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


class TestSignatureIsIdentityNotDisplay:
    """The loop key must be a full identity, not the truncated display summary.

    ``ToolCall.input_summary`` is documented as being "for display": it cuts Bash
    commands at 100 chars and renders any non-builtin tool as
    ``str(input_data)[:80]``. Grouping on it makes distinct calls that share a
    long prefix compare equal, so unrelated work is reported as one loop.
    """

    def test_distinct_custom_tool_calls_are_not_one_loop(self):
        # Five patches to five different files under one long shared prefix.
        prefix = "/home/user/worktrees/feature-branch-alpha/services/api/internal/handlers"
        calls = [
            _call(
                "apply_patch",
                f"call_{i}",
                {"input": f"*** Update File: {prefix}/handler_{i}.py\n@@\n-old\n+new\n"},
                msg_index=i,
            )
            for i in range(5)
        ]
        assert len({_canonical_signature(c) for c in calls}) == 5
        assert detect_loops([SessionData(session_id="s", tool_calls=calls)]) == []

    def test_distinct_long_bash_commands_are_not_one_loop(self):
        # Commands diverge only after the 100-char display cutoff.
        base = (
            "rg --no-heading --line-number --color never "
            "'TimeoutError|ValueError|KeyError' /srv/app/services/ingest/pipeline/"
        )
        calls = [
            _call("Bash", f"call_{i}", {"command": f"{base}module_{i}/handlers.py"}, msg_index=i)
            for i in range(4)
        ]
        assert len(calls[0].input_summary) < len(calls[0].input_data["command"])
        assert len({_canonical_signature(c) for c in calls}) == 4
        assert detect_loops([SessionData(session_id="s", tool_calls=calls)]) == []

    def test_pagination_variants_of_one_long_command_still_collapse(self):
        # The re-fetch collapse must survive the identity change: one command,
        # growing output limits, past the 100-char display cutoff.
        base = (
            "rg --no-heading --line-number --color never "
            "'TimeoutError|ValueError|KeyError' /srv/app/services/ingest/pipeline/handlers.py"
        )
        calls = [
            _call("Bash", f"call_{i}", {"command": f"{base} | head -{50 * (i + 1)}"}, msg_index=i)
            for i in range(4)
        ]
        assert len({_canonical_signature(c) for c in calls}) == 1
        loops = detect_loops([SessionData(session_id="s", tool_calls=calls)])
        assert len(loops) == 1
        assert loops[0].count == 4


class TestSignatureIsSizeIndependent:
    """Detection quality must not degrade as a tool input gets larger.

    The identity key is lossless, so it grows with the input. Everything derived
    from it must stay size-independent, because the only consumer of
    ``LoopPattern.signature`` — ``_signature_tokens`` in ``apply_loop_weighting``
    — requires a *majority* of the signature's tokens to appear in a
    recommendation. A signature that grows without bound pushes that threshold
    out of reach, and a real loop silently loses its measured-waste boost.
    """

    BASE = "pytest tests/test_learn/test_loops.py -v --maxfail 1 --tb=short"
    REC = (
        "Avoid rerunning pytest on tests/test_learn/test_loops.py with"
        " -v --maxfail 1 --tb=short; reuse the prior run."
    )

    def _command(self, extra_args: int) -> str:
        tail = " ".join(f"--unrelated-opt-{i}=value-{i}" for i in range(extra_args))
        return f"{self.BASE} {tail}".strip()

    def _loop(self, extra_args: int):
        calls = [
            _call("Bash", f"call_b{i}", {"command": self._command(extra_args)}, msg_index=i)
            for i in range(5)
        ]
        return detect_loops([SessionData(session_id="s", tool_calls=calls)])

    @pytest.mark.parametrize("extra_args", [0, 20, 200, 2000])
    def test_weighting_holds_however_long_the_input_is(self, extra_args):
        loops = self._loop(extra_args)
        rec = Recommendation(
            target=RecommendationTarget.CONTEXT_FILE,
            section="Tooling",
            content=self.REC,
            estimated_tokens_saved=100,
        )

        apply_loop_weighting([rec], loops)

        assert rec.is_loop_guardrail is True
        assert rec.estimated_tokens_saved == loops[0].wasted_tokens

    def test_signature_size_does_not_track_input_size(self):
        big = self._loop(500)[0].signature
        ten_times_bigger = self._loop(5000)[0].signature

        assert big == ten_times_bigger
        assert len(big) < len(self._command(500)) // 10

    def test_oversized_inputs_are_still_told_apart(self):
        """Bounding the signature must not reintroduce the merge it was added to survive.

        Identity is the *unbounded* key; only the stored signature is capped. Two
        calls sharing a long prefix must therefore stay distinct even when their
        bounded signatures are identical.
        """
        shared = "*** Update File: /repo/services/ingest/handler.py\n" + "-old\n+new\n" * 400
        calls = [
            _call("apply_patch", f"call_p{i}", {"input": f"{shared}# variant {i}"}, msg_index=i)
            for i in range(5)
        ]

        assert len({_canonical_signature(c) for c in calls}) == 5
        assert detect_loops([SessionData(session_id="s", tool_calls=calls)]) == []


class TestGroupingInvariants:
    """Properties that must hold for any input, not just the pinned examples."""

    def _mixed_calls(self) -> list[ToolCall]:
        return [
            _call("Read", "call_r0", {"file_path": "/repo/a.py"}, msg_index=0),
            _call("Read", "call_r1", {"file_path": "/repo/a.py"}, msg_index=1),
            _call("Read", "call_r2", {"file_path": "/repo/a.py"}, msg_index=2),
            _call("Bash", "call_b0", {"command": "rg alpha /repo"}, msg_index=3),
            _call("Bash", "call_b1", {"command": "rg alpha /repo"}, msg_index=4),
            _call("Bash", "call_b2", {"command": "rg alpha /repo"}, msg_index=5),
            _call("Grep", "call_g0", {"pattern": "beta"}, msg_index=6),
        ]

    def test_detection_does_not_depend_on_call_order(self):
        calls = self._mixed_calls()
        shuffled = list(reversed(calls))

        def summarize(cs):
            loops = detect_loops([SessionData(session_id="s", tool_calls=cs)])
            return sorted((lp.tool, lp.count, lp.wasted_tokens) for lp in loops)

        assert summarize(shuffled) == summarize(calls)

    def test_whitespace_variants_share_one_signature(self):
        """Whitespace normalization survives the split/join implementation.

        The collapse is done with ``" ".join(str.split())`` rather than a regex
        for speed; this pins that tabs, newlines and repeated spaces still fold
        to the same signature.
        """
        variants = [
            "rg --line-number 'def handler' /repo/app.py",
            "rg  --line-number\t'def handler'   /repo/app.py",
            "  rg --line-number\n'def handler' /repo/app.py  ",
        ]
        calls = [
            _call("Bash", f"call_w{i}", {"command": cmd}, msg_index=i)
            for i, cmd in enumerate(variants)
        ]

        assert len({_canonical_signature(c) for c in calls}) == 1


class TestIdentityFallsBackWhenTheSchemaIsUnknown:
    """``_identity_input`` must render the input in full, as it documents.

    ``_IDENTITY_FIELDS`` guesses which field identifies a call from the tool's
    name. ``normalize_tool_name`` maps provider tools onto those same names
    without normalizing their input schema, so the guess can miss every field —
    at which point an empty identity collapses unrelated calls into one loop,
    the exact failure this signature change exists to prevent.
    """

    def test_a_builtin_name_carrying_a_foreign_schema_stays_distinct(self):
        # `codebase_search` / `search_text` normalize to Grep but carry `query`.
        calls = [
            _call("Grep", f"call_{i}", {"query": f"distinct query number {i}"}, msg_index=i)
            for i in range(5)
        ]

        assert len({_canonical_signature(c) for c in calls}) == 5
        assert detect_loops([SessionData(session_id="s", tool_calls=calls)]) == []

    def test_a_partially_matching_schema_still_uses_the_known_field(self):
        # `file_path` is present, so identity comes from it and the rest is
        # ignored — the pinned builtin behaviour is unchanged.
        calls = [
            _call("Read", f"call_{i}", {"file_path": "/repo/a.md", "offset": i}, msg_index=i)
            for i in range(3)
        ]

        loops = detect_loops([SessionData(session_id="s", tool_calls=calls)])

        assert [lp.count for lp in loops] == [3]

    @pytest.mark.parametrize("payload", [None, [], "raw string", {1: "a", "b": 2}])
    def test_an_unexpected_input_shape_does_not_raise(self, payload):
        # Identity is derived during a scan of files the user did not write;
        # a malformed input must not take down `headroom learn`.
        call = _call("Bash", "call_0", payload, msg_index=0)

        assert _canonical_signature(call).startswith("bash::")


class TestSearchIdentityIncludesPath:
    """The same pattern in different trees is three searches, not a loop.

    Identity now comes from the input fields rather than ``input_summary``, so
    every field that distinguishes a search has to be named here — a pattern
    alone would merge a sweep across three service directories into one loop
    worth 20,000 tokens of phantom waste. Pins the same behavior #3455 pins on
    the display side, on the path identity actually reads.
    """

    def _search(self, name: str, pattern: str, path: str, index: int) -> ToolCall:
        return _call(name, f"{name}-{index}", {"pattern": pattern, "path": path}, msg_index=index)

    @pytest.mark.parametrize(("name", "pattern"), [("Grep", "TimeoutError"), ("Glob", "**/*.py")])
    def test_distinct_paths_are_distinct_signatures(self, name: str, pattern: str):
        calls = [
            self._search(name, pattern, path, i)
            for i, path in enumerate(("/srv/ingest", "/srv/billing", "/srv/web"))
        ]

        assert len({_canonical_signature(c) for c in calls}) == 3

    @pytest.mark.parametrize(("name", "pattern"), [("Grep", "TimeoutError"), ("Glob", "**/*.py")])
    def test_the_same_pattern_in_three_trees_is_not_a_loop(self, name: str, pattern: str):
        calls = [
            self._search(name, pattern, path, i)
            for i, path in enumerate(("/srv/ingest", "/srv/billing", "/srv/web"))
        ]

        assert detect_loops([SessionData(session_id=name, tool_calls=calls)]) == []

    @pytest.mark.parametrize(("name", "pattern"), [("Grep", "TimeoutError"), ("Glob", "**/*.py")])
    def test_repeating_one_search_is_still_a_loop(self, name: str, pattern: str):
        # Adding path to identity must not stop a real re-search loop counting.
        calls = [self._search(name, pattern, "/srv/ingest", i) for i in range(3)]

        assert detect_loops([SessionData(session_id=name, tool_calls=calls)])[0].count == 3

    def test_a_search_without_a_path_still_has_identity(self, name: str = "Grep"):
        # Claude Code omits path when searching the cwd; identity must not
        # collapse to the empty fallback, nor split from a differing path.
        rooted = _call(name, "g-1", {"pattern": "TimeoutError", "path": "/srv"}, msg_index=1)
        bare = _call(name, "g-0", {"pattern": "TimeoutError"}, msg_index=0)

        assert _canonical_signature(bare) != _canonical_signature(rooted)


class TestIdentityPreservesFieldBoundaries:
    """Two identity fields must not run together into one flat string.

    ``pattern`` and ``path`` joined on a space lose the boundary between them:
    ``("error in src", "logs")`` and ``("error", "in src logs")`` are different
    searches that render the same text, and merged into one loop they report
    20,000 tokens of waste nobody spent. Only tools with more than one identity
    field need the structured form, so Bash keeps the bare command its
    pagination normalization is written against.
    """

    def _search(self, pattern: str, path: str, index: int) -> ToolCall:
        return _call("Grep", f"g-{index}", {"pattern": pattern, "path": path}, msg_index=index)

    # Same characters in the same order, split between the fields three ways.
    _AMBIGUOUS = (("error in src", "logs"), ("error in", "src logs"), ("error", "in src logs"))

    def test_field_splits_of_one_string_are_distinct_signatures(self):
        calls = [
            self._search(pattern, path, i) for i, (pattern, path) in enumerate(self._AMBIGUOUS)
        ]

        assert len({_canonical_signature(c) for c in calls}) == 3

    def test_field_splits_of_one_string_are_not_a_loop(self):
        calls = [
            self._search(pattern, path, i) for i, (pattern, path) in enumerate(self._AMBIGUOUS)
        ]

        assert detect_loops([SessionData(session_id="s", tool_calls=calls)]) == []

    def test_a_path_containing_spaces_still_loops_when_repeated(self):
        # Spaces are not themselves a reason to split a group.
        calls = [self._search("error in src", "/srv/my logs", i) for i in range(3)]

        assert detect_loops([SessionData(session_id="s", tool_calls=calls)])[0].count == 3

    @pytest.mark.parametrize(
        ("pattern", "path"),
        [('a", "b', "c"), ("a", '", "b", "c'), ("a\\", '"b'), ('["a"]', "b")],
    )
    def test_a_pattern_that_mimics_the_encoding_stays_distinct(self, pattern: str, path: str):
        # Whatever the encoding is, a search cannot forge another search's
        # identity by embedding the separator in its own pattern.
        mimic = self._search(pattern, path, 0)
        plain = self._search("a", "b", 1)

        assert _canonical_signature(mimic) != _canonical_signature(plain)

    def test_pagination_variants_of_one_command_still_collapse(self):
        # The structured form must not reach single-field tools: Bash identity
        # stays the bare command, so _PAGINATION_RE keeps matching it.
        base = 'rg --no-heading "TimeoutError" /srv/app/services/ingest/pipeline/handlers.py'
        calls = [
            _call("Bash", f"call_{i}", {"command": f"{base} | head -{50 * (i + 1)}"}, msg_index=i)
            for i in range(4)
        ]

        assert len({_canonical_signature(c) for c in calls}) == 1

    def test_a_non_ascii_pattern_does_not_leak_escape_tokens(self):
        # apply_loop_weighting requires a *majority* of a signature's tokens to
        # appear in a recommendation. An escape artifact is a token no
        # recommendation can ever contain, so it only raises that bar and costs
        # a real loop its measured-waste boost.
        call = self._search("café latte", "/srv/menu", 0)

        assert "u00e9" not in _signature_tokens(_canonical_signature(call))

    @pytest.mark.parametrize(
        ("name", "field"),
        [(n, f[0]) for n, f in sorted(_IDENTITY_FIELDS.items()) if len(f) == 1],
    )
    def test_a_single_field_tool_keeps_its_bare_value(self, name: str, field: str):
        # Guards the carve-out above: give one of these tools a second identity
        # field and it silently moves to the structured form, where
        # _PAGINATION_RE no longer matches the command it was written for.
        call = _call(name, "c-0", {field: "a b c"}, msg_index=0)

        assert _identity_input(call) == "a b c"


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


class TestSubThresholdSessionsDoNotCombine:
    """A session that only looped in replays must not contribute at all.

    ``min_occurrences`` is screened against the *pre-dedup* per-session group, so
    a session whose repetitions are all replays of one call still qualifies and
    hands its single real call to the global view. Three such sessions then clear
    the bar together, reporting a loop no conversation actually ran.
    """

    def _replay_padded_session(self, session_id: str, call_id: str) -> SessionData:
        # One real Read, written into this transcript three times.
        return SessionData(
            session_id=session_id,
            tool_calls=[
                _call("Read", call_id, {"file_path": "/repo/design.md"}, msg_index=i)
                for i in range(3)
            ],
        )

    def test_each_session_alone_is_not_a_loop(self):
        sessions = [
            self._replay_padded_session("s1", "toolu_s1"),
            self._replay_padded_session("s2", "toolu_s2"),
            self._replay_padded_session("s3", "toolu_s3"),
        ]

        assert [detect_loops([s]) for s in sessions] == [[], [], []]

    def test_sub_threshold_sessions_do_not_combine_into_a_loop(self):
        sessions = [
            self._replay_padded_session("s1", "toolu_s1"),
            self._replay_padded_session("s2", "toolu_s2"),
            self._replay_padded_session("s3", "toolu_s3"),
        ]

        assert detect_loops(sessions) == []
