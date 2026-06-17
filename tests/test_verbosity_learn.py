"""Tests for headroom.learn.verbosity — behavioral signal extraction."""

from __future__ import annotations

import json
from pathlib import Path

from headroom.learn.verbosity import (
    VerbositySignals,
    analyze,
    extract_signals,
    recommend_level,
)


def _write_session(tmp_path: Path, name: str, lines: list[dict]) -> Path:
    p = tmp_path / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(line) for line in lines))
    return p


def _assistant(
    text: str,
    *,
    ts: str,
    out_tokens: int = 100,
    model: str = "claude-opus-4-8",
    in_tokens: int = 5000,
) -> dict:
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "model": model,
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
        },
    }


def _user(text: str, *, ts: str) -> dict:
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}


def _tool_result(*, ts: str, content: str = "ok") -> dict:
    return {
        "type": "user",
        "timestamp": ts,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}],
        },
    }


LONG = " ".join(["word"] * 400)  # well above the long-output floor


class TestSignalExtraction:
    def test_interrupt_counted(self, tmp_path):
        p = _write_session(
            tmp_path,
            "s",
            [
                _user("do a thing", ts="2026-01-01T00:00:00Z"),
                _assistant(LONG, ts="2026-01-01T00:00:10Z"),
                _user("[Request interrupted by user]", ts="2026-01-01T00:00:12Z"),
            ],
        )
        sig, _ = extract_signals([p])
        assert sig.interrupts == 1
        assert sig.human_msgs == 1  # the initial ask

    def test_fast_skip_detected_length_adaptive(self, tmp_path):
        # 400-word answer needs ~96s to read; reply after 5s = fast skip.
        p = _write_session(
            tmp_path,
            "s",
            [
                _user("explain", ts="2026-01-01T00:00:00Z"),
                _assistant(LONG, ts="2026-01-01T00:00:00Z"),
                _user("ok next", ts="2026-01-01T00:00:05Z"),
            ],
        )
        sig, _ = extract_signals([p])
        assert sig.skip_eligible == 1
        assert sig.fast_skips == 1

    def test_slow_reply_is_not_a_skip(self, tmp_path):
        # Reply 120s after a 400-word answer (>read time) = read, not skipped.
        p = _write_session(
            tmp_path,
            "s",
            [
                _user("explain", ts="2026-01-01T00:00:00Z"),
                _assistant(LONG, ts="2026-01-01T00:00:00Z"),
                _user("ok next", ts="2026-01-01T00:02:00Z"),
            ],
        )
        sig, _ = extract_signals([p])
        assert sig.skip_eligible == 1
        assert sig.fast_skips == 0

    def test_short_answer_not_skip_eligible(self, tmp_path):
        p = _write_session(
            tmp_path,
            "s",
            [
                _user("hi", ts="2026-01-01T00:00:00Z"),
                _assistant("short reply", ts="2026-01-01T00:00:00Z"),
                _user("ok", ts="2026-01-01T00:00:01Z"),
            ],
        )
        sig, _ = extract_signals([p])
        assert sig.skip_eligible == 0

    def test_baseline_captures_output_tokens_by_stratum(self, tmp_path):
        p = _write_session(
            tmp_path,
            "s",
            [
                _user("task", ts="2026-01-01T00:00:00Z"),
                _assistant("a reply", ts="2026-01-01T00:00:01Z", out_tokens=420, in_tokens=5000),
            ],
        )
        _, baseline = extract_signals([p])
        assert baseline.total_samples == 1
        # new_user_ask, input bucket "s" (5000), opus, no tools in this session
        mean, _, n = baseline.lookup("opus|new_user_ask|s|notools")
        assert n == 1
        assert mean == 420.0

    def test_tool_result_makes_session_have_tools(self, tmp_path):
        p = _write_session(
            tmp_path,
            "s",
            [
                _user("task", ts="2026-01-01T00:00:00Z"),
                _assistant("reading", ts="2026-01-01T00:00:01Z", out_tokens=50),
                _tool_result(ts="2026-01-01T00:00:02Z"),
                _assistant("done", ts="2026-01-01T00:00:03Z", out_tokens=200),
            ],
        )
        _, baseline = extract_signals([p])
        # Every response in a tool-using session is stratified as has_tools.
        assert any("|tools" in k for k in baseline.strata)
        assert not any("|notools" in k for k in baseline.strata)

    def test_tool_result_reply_not_counted_as_human(self, tmp_path):
        p = _write_session(
            tmp_path,
            "s",
            [
                _user("task", ts="2026-01-01T00:00:00Z"),
                _assistant("reading", ts="2026-01-01T00:00:01Z"),
                _tool_result(ts="2026-01-01T00:00:02Z"),
            ],
        )
        sig, _ = extract_signals([p])
        assert sig.human_msgs == 1  # only the real ask, not the tool_result


class TestRecommendLevel:
    def _sig(self, *, human, interrupts, skip_eligible, fast_skips) -> VerbositySignals:
        s = VerbositySignals()
        s.human_msgs = human
        s.interrupts = interrupts
        s.skip_eligible = skip_eligible
        s.fast_skips = fast_skips
        return s

    def test_too_few_turns_defaults_l2_low(self):
        level, conf, _ = recommend_level(
            self._sig(human=3, interrupts=0, skip_eligible=0, fast_skips=0)
        )
        assert level == 2
        assert conf == "low"

    def test_low_pressure_user_gets_l1(self):
        # 100 turns, almost no interrupts/skips.
        s = self._sig(human=100, interrupts=1, skip_eligible=100, fast_skips=2)
        level, conf, _ = recommend_level(s)
        assert level == 1
        assert conf == "high"

    def test_moderate_pressure_gets_l2(self):
        s = self._sig(human=80, interrupts=8, skip_eligible=80, fast_skips=12)
        level, _, _ = recommend_level(s)
        assert level == 2

    def test_high_pressure_gets_l3(self):
        # Mirrors the real measured user: ~11% interrupt, ~26% skip.
        s = self._sig(human=200, interrupts=29, skip_eligible=119, fast_skips=31)
        level, conf, _ = recommend_level(s)
        assert level == 3
        assert conf == "high"


class TestAnalyze:
    def test_llm_judge_overrides_heuristic(self, tmp_path):
        p = _write_session(
            tmp_path,
            "s",
            [_user("x", ts="2026-01-01T00:00:00Z"), _assistant("y", ts="2026-01-01T00:00:01Z")]
            * 20,
        )

        def judge(signals_dict):
            return 4, "LLM says this user wants caveman mode"

        profile, _ = analyze([p], "/proj", llm_judge=judge)
        assert profile.level == 4
        assert profile.source == "llm"
        assert "caveman" in profile.rationale

    def test_llm_judge_failure_falls_back_to_heuristic(self, tmp_path):
        p = _write_session(
            tmp_path,
            "s",
            [_user("x", ts="2026-01-01T00:00:00Z"), _assistant("y", ts="2026-01-01T00:00:01Z")]
            * 20,
        )

        def bad_judge(signals_dict):
            raise RuntimeError("no api key")

        profile, _ = analyze([p], "/proj", llm_judge=bad_judge)
        assert profile.source == "heuristic"

    def test_profile_roundtrip(self, tmp_path):
        from headroom.learn.verbosity import VerbosityProfile

        prof = VerbosityProfile(
            project_path="/proj",
            level=3,
            confidence="high",
            source="heuristic",
            rationale="because",
            signals={"interrupt_rate": 0.11},
        )
        path = tmp_path / "verbosity.json"
        prof.save(path)
        loaded = VerbosityProfile.load(path)
        assert loaded is not None
        assert loaded.level == 3
        assert loaded.confidence == "high"

    def test_load_missing_returns_none(self, tmp_path):
        from headroom.learn.verbosity import VerbosityProfile

        assert VerbosityProfile.load(tmp_path / "nope.json") is None
