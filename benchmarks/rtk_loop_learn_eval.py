"""RTK-loop eval — does Headroom Learn catch a loop and write a guardrail that
would prevent it recurring?

This is the agentic eval for the loop-weighting work. It runs in two phases:

  Phase 1 — TRIGGER + LEARN
    Reproduce an RTK re-fetch loop (a grep whose RTK-truncated output forces the
    agent to re-run larger-limit variants), run it through ``SessionAnalyzer``,
    and SCORE the resulting guardrail:
      • produced        — a loop guardrail was emitted at all
      • ranked_first    — it outranks the one-off rules (the weighting works)
      • names_command   — the rule identifies the command that looped
      • prescribes_fix  — the rule says how to avoid it (fetch full output once)
      • weight_reflects — its savings estimate >= the MEASURED wasted tokens

  Phase 2 — GUARDRAIL HOLDS
    Inject that guardrail as a prior learned pattern, then feed a session where
    the agent FOLLOWED it (one full-output fetch, no loop). Re-run the analyzer
    and assert NO new loop guardrail is produced for that command — i.e. once
    the rule exists and is honored, the loop does not re-trigger and Learn does
    not need to relearn it.

Runs deterministically by default (a stubbed analyzer LLM so CI is hermetic).
With ``--real`` it drives the real analyzer LLM and scores the actually-generated
rule, using an API key (ANTHROPIC/OPENAI/GEMINI) or an installed CLI backend.

Usage:
    python benchmarks/rtk_loop_learn_eval.py                          # deterministic
    python benchmarks/rtk_loop_learn_eval.py --real                   # real LLM (API key)
    HEADROOM_LEARN_CLI=claude python benchmarks/rtk_loop_learn_eval.py --real  # via CLI
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from headroom.learn.analyzer import SessionAnalyzer  # noqa: E402
from headroom.learn.fixtures import rtk_refetch_loop_session  # noqa: E402
from headroom.learn.loops import detect_loops  # noqa: E402
from headroom.learn.models import (  # noqa: E402
    ProjectInfo,
    SessionData,
    ToolCall,
)

REPETITIONS = 6


# =============================================================================
# Deterministic LLM stub — stands in for the analyzer's _call_llm in CI.
# It mimics a competent model: emits the loop guardrail (under-estimating its
# savings, so the weighting layer has real work to do) plus a one-off rule the
# model would naively rank higher. In Phase 2 it emits NO loop rule, because a
# non-looping guarded session gives it nothing to relearn.
# =============================================================================


def _stub_llm_phase1(digest: str, model: str) -> dict:
    return {
        "context_file_rules": [
            {
                "section": "Use uv for Python",
                "content": "Use `uv run python` instead of `python3`.",
                "estimated_tokens_saved": 900,  # model rates the one-off high
                "evidence_count": 2,
            },
            {
                "section": "Avoid grep TimeoutError re-fetch loop",
                "content": (
                    "When searching logs for TimeoutError, capture the full "
                    "result once (grep into a file and read it) instead of "
                    "re-running grep with larger `head` limits."
                ),
                "estimated_tokens_saved": 150,  # simulated low estimate (stub value, not a real-model figure)
                "evidence_count": 1,
            },
        ],
        "memory_file_rules": [],
    }


def _stub_llm_phase2(digest: str, model: str) -> dict:
    # Guarded, non-looping session → nothing new to learn about the grep.
    return {"context_file_rules": [], "memory_file_rules": []}


# =============================================================================
# Scoring
# =============================================================================


@dataclass
class Scorecard:
    checks: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, passed: bool, note: str = "") -> None:
        self.checks[name] = passed
        if note:
            self.notes[name] = note

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def render(self) -> str:
        width = max(len(k) for k in self.checks)
        lines = []
        for name, ok in self.checks.items():
            mark = "PASS" if ok else "FAIL"
            note = f"  ({self.notes[name]})" if name in self.notes else ""
            lines.append(f"  [{mark}] {name.ljust(width)}{note}")
        return "\n".join(lines)


def _guarded_session() -> SessionData:
    """A session where the agent followed the guardrail: one full-output fetch,
    no re-fetch loop."""
    return SessionData(
        session_id="guarded",
        tool_calls=[
            ToolCall(
                name="Bash",
                tool_call_id="tc_0",
                input_data={"command": "grep -rn 'TimeoutError' logs/ > /tmp/hits.txt"},
                output="(wrote 1240 matches to /tmp/hits.txt)",
                is_error=False,
                msg_index=0,
                output_bytes=40,
            ),
            ToolCall(
                name="Read",
                tool_call_id="tc_1",
                input_data={"file_path": "/tmp/hits.txt"},
                output="logs/app.log:42: TimeoutError ...",
                is_error=False,
                msg_index=1,
                output_bytes=8000,
            ),
        ],
    )


def run_eval(*, use_real_llm: bool) -> Scorecard:
    project = ProjectInfo(
        name="rtk-loop-eval",
        project_path=Path("/tmp/rtk-loop-eval"),
        data_path=Path("/tmp/rtk-loop-eval-data"),
    )
    card = Scorecard()

    # ---- Phase 1: trigger + learn -----------------------------------------
    loop_session = rtk_refetch_loop_session(repetitions=REPETITIONS)
    loops = detect_loops([loop_session])
    measured_waste = loops[0].wasted_tokens if loops else 0
    card.add("loop_detected", bool(loops), f"{len(loops)} loop(s), ~{measured_waste:,} tok wasted")

    analyzer = SessionAnalyzer(model=None if use_real_llm else "stub")
    phase1_ctx = (
        nullcontext()
        if use_real_llm
        else patch("headroom.learn.analyzer._call_llm", _stub_llm_phase1)
    )
    with phase1_ctx:
        result = analyzer.analyze(project, [loop_session])

    recs = result.recommendations
    loop_recs = [r for r in recs if r.is_loop_guardrail]
    card.add("guardrail_produced", bool(loop_recs))

    top = recs[0] if recs else None
    card.add(
        "ranked_first",
        bool(top and top.is_loop_guardrail),
        "" if (top and top.is_loop_guardrail) else "loop rule did not rank #1",
    )

    guardrail = loop_recs[0] if loop_recs else None
    text = (guardrail.section + " " + guardrail.content).lower() if guardrail else ""
    # The rule must identify the LOOPING COMMAND (grep + its output-limit shape),
    # not the incidental search string — a good fix generalizes beyond it. (The
    # real-LLM run surfaced this: the model wrote a general "grepping logs / `head
    # -N` limits" rule and never echoed "TimeoutError", which an earlier
    # literal-match check wrongly failed.)
    card.add(
        "names_command",
        "grep" in text and any(k in text for k in ("head", "log", "limit")),
    )
    card.add(
        "prescribes_fix",
        any(k in text for k in ("full", "once", "into a file", "instead", "limit")),
    )
    card.add(
        "weight_reflects_waste",
        bool(guardrail and guardrail.estimated_tokens_saved >= measured_waste),
        ""
        if (guardrail and guardrail.estimated_tokens_saved >= measured_waste)
        else f"savings {getattr(guardrail, 'estimated_tokens_saved', 0)} < waste {measured_waste}",
    )

    # ---- Phase 2: guardrail holds -----------------------------------------
    # Inject the produced guardrail as a prior pattern via the project's
    # context file, then analyze a guarded (non-looping) session.
    held = True
    note = ""
    if guardrail:
        ctx_path = Path("/tmp/rtk-loop-eval-CLAUDE.md")
        ctx_path.write_text(
            "<!-- headroom:learn:start -->\n"
            f"### {guardrail.section}\n{guardrail.content}\n"
            "<!-- headroom:learn:end -->\n",
            encoding="utf-8",
        )
        project.context_file = ctx_path
        phase2_ctx = (
            nullcontext()
            if use_real_llm
            else patch("headroom.learn.analyzer._call_llm", _stub_llm_phase2)
        )
        with phase2_ctx:
            held_result = analyzer.analyze(project, [_guarded_session()])
        # No NEW loop guardrail should be needed for the (now-guarded) grep.
        new_loop_rules = [
            r
            for r in held_result.recommendations
            if r.is_loop_guardrail and "grep" in (r.section + r.content).lower()
        ]
        held = not new_loop_rules
        note = "" if held else f"{len(new_loop_rules)} new grep loop rule(s) re-emitted"
    else:
        held = False
        note = "no guardrail from phase 1 to test"
    card.add("guardrail_holds", held, note)

    return card


def _real_backend_available() -> bool:
    """True when the analyzer can reach a real LLM — API key or installed CLI."""
    import shutil

    if any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")):
        return True
    return any(shutil.which(cli) for cli in ("claude", "gemini", "codex"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Drive the real analyzer LLM — needs an API key (ANTHROPIC_API_KEY / "
        "OPENAI_API_KEY / GEMINI_API_KEY) or an installed CLI backend "
        "(claude / gemini / codex; force one with HEADROOM_LEARN_CLI=claude).",
    )
    args = parser.parse_args()

    if args.real and not _real_backend_available():
        print(
            "--real needs an LLM backend (API key or claude/gemini/codex CLI); "
            "falling back to deterministic mode.\n"
        )
        args.real = False

    mode = "REAL LLM" if args.real else "deterministic stub"
    print(f"RTK-loop eval — mode: {mode}\n")
    card = run_eval(use_real_llm=args.real)
    print(card.render())
    print()
    if card.passed:
        print("RESULT: PASS — loop caught, guardrail ranked first, and it holds.")
        return 0
    print("RESULT: FAIL — see failed checks above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
