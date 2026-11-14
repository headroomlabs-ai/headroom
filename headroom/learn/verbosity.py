"""Learn a user's preferred output verbosity from past sessions.

The premise (validated on real transcripts): users almost never *say* how terse
they want answers — explicit "be brief" feedback is near-zero — but they *show*
it behaviorally. They interrupt long answers, and they reply faster than a long
answer could possibly have been read. Those signals are mechanical to extract
from Claude Code's JSONL transcripts.

This module:

1. Parses transcripts into per-response records (output tokens, word count,
   timestamps, the preceding turn's structural kind).
2. Extracts behavioral signals — interrupt rate, fast-skip rate, long-output
   frequency, echo ratio — using length-adaptive thresholds (a "fast skip" is
   defined relative to how long the answer would take to *read*, not a fixed
   number of seconds; "long" is relative to the user's own median).
3. Recommends a verbosity level (heuristic prior; an optional LLM judgment pass
   can override it — see ``analyze``).
4. Builds the per-stratum output-token baseline that
   :mod:`headroom.proxy.output_savings` uses as its synthetic control — so the
   same pass that picks the level also establishes how to measure its effect.

The structural signals are *inputs* to the decision, not the decision itself —
which is why thresholds here are interpretable, length-adaptive, and (when an
LLM is available) advisory rather than final.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..proxy.output_savings import BaselineModel, echo_ratio, stratum_key
from ..proxy.output_shaper import classify_turn

logger = logging.getLogger(__name__)

# Average adult reading speed (words/min) for technical prose. Used to turn a
# response's length into "how long it would take to read", so a fast-skip is
# defined relative to answer length rather than a fixed wall-clock cutoff.
_READING_WPM = 250.0
# A reply arriving in less than this fraction of the read-time means the answer
# was almost certainly not read.
_SKIP_READ_FRACTION = 0.5
# Don't score skips on trivially short answers — there's nothing to skip.
_MIN_WORDS_FOR_SKIP = 150
# Floor for the per-user adaptive "long output" threshold (words).
_LONG_OUTPUT_FLOOR = 200
# Only sample the few preceding messages for echo context (bounded cost).
_ECHO_CONTEXT_LOOKBACK = 4

_INTERRUPT_MARKER = "[Request interrupted by user"


@dataclass
class _Response:
    """One assistant response, with the request features that produced it."""

    words: int
    output_tokens: int
    input_tokens: int
    model: str
    turn_kind: str
    has_tools: bool
    ts: float | None
    echo: float


@dataclass
class _HumanMsg:
    ts: float | None
    is_interrupt: bool


@dataclass
class VerbositySignals:
    """Behavioral signals aggregated across a project's sessions."""

    sessions: int = 0
    human_msgs: int = 0
    interrupts: int = 0
    asst_responses: int = 0
    asst_words: int = 0
    long_outputs: int = 0
    fast_skips: int = 0
    skip_eligible: int = 0
    mean_echo_ratio: float = 0.0

    @property
    def interrupt_rate(self) -> float:
        denom = self.human_msgs + self.interrupts
        return self.interrupts / denom if denom else 0.0

    @property
    def fast_skip_rate(self) -> float:
        return self.fast_skips / self.skip_eligible if self.skip_eligible else 0.0

    @property
    def long_output_rate(self) -> float:
        return self.long_outputs / self.asst_responses if self.asst_responses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "human_msgs": self.human_msgs,
            "interrupts": self.interrupts,
            "interrupt_rate": round(self.interrupt_rate, 4),
            "asst_responses": self.asst_responses,
            "long_outputs": self.long_outputs,
            "long_output_rate": round(self.long_output_rate, 4),
            "fast_skips": self.fast_skips,
            "skip_eligible": self.skip_eligible,
            "fast_skip_rate": round(self.fast_skip_rate, 4),
            "mean_echo_ratio": round(self.mean_echo_ratio, 4),
        }


@dataclass
class VerbosityProfile:
    """The learned recommendation for a project."""

    project_path: str
    level: int
    confidence: str  # "low" | "medium" | "high"
    source: str  # "heuristic" | "llm"
    rationale: str
    signals: dict[str, Any] = field(default_factory=dict)
    learned_at: str | None = None  # caller stamps (Date.now unavailable here)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_path": self.project_path,
            "verbosity_level": self.level,
            "confidence": self.confidence,
            "source": self.source,
            "rationale": self.rationale,
            "signals": self.signals,
            "learned_at": self.learned_at,
        }

    @classmethod
    def load(cls, path: Path) -> VerbosityProfile | None:
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        return cls(
            project_path=d.get("project_path", ""),
            level=int(d.get("verbosity_level", 2)),
            confidence=d.get("confidence", "low"),
            source=d.get("source", "heuristic"),
            rationale=d.get("rationale", ""),
            signals=d.get("signals", {}),
            learned_at=d.get("learned_at"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _parse_ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _assistant_words_and_text(content: Any) -> tuple[int, str]:
    if not isinstance(content, list):
        return 0, ""
    text = " ".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    )
    return len(text.split()), text


def _human_text(content: Any) -> str | None:
    """Return human-typed text, or None for tool results / slash commands / meta."""
    if isinstance(content, str):
        t = content
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        t = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        return None
    if "<command-name>" in t or "<local-command-stdout>" in t:
        return None
    return t


def _parse_session(path: Path) -> tuple[list[_Response], list[_HumanMsg], bool]:
    """Parse one transcript into responses + human messages + has_tools flag."""
    responses: list[_Response] = []
    humans: list[_HumanMsg] = []
    has_tools = False
    prior_messages: list[dict[str, Any]] = []
    recent_context: list[str] = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return [], [], False

    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ltype = d.get("type")
        msg = d.get("message", {}) if isinstance(d.get("message"), dict) else {}
        ts = _parse_ts(d.get("timestamp"))

        if ltype == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_use" for b in content
            ):
                has_tools = True
            words, text = _assistant_words_and_text(content)
            usage = msg.get("usage", {}) if isinstance(msg.get("usage"), dict) else {}
            in_tok = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )
            out_tok = usage.get("output_tokens", 0)
            if words > 0 or out_tok > 0:
                ctx = " ".join(recent_context[-_ECHO_CONTEXT_LOOKBACK:])
                responses.append(
                    _Response(
                        words=words,
                        output_tokens=out_tok,
                        input_tokens=in_tok,
                        model=str(msg.get("model", "")),
                        turn_kind=classify_turn(prior_messages).value,
                        has_tools=False,  # filled after the session scan
                        ts=ts,
                        echo=echo_ratio(text, ctx) if text and ctx else 0.0,
                    )
                )
            prior_messages.append({"role": "assistant", "content": content})

        elif ltype == "user":
            content = msg.get("content")
            prior_messages.append({"role": "user", "content": content})
            # Feed tool results / user text into echo context.
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        rc = b.get("content", "")
                        recent_context.append(rc if isinstance(rc, str) else str(rc))
                        has_tools = True
            human = _human_text(content)
            if human is None:
                continue
            if _INTERRUPT_MARKER in human:
                humans.append(_HumanMsg(ts=ts, is_interrupt=True))
            else:
                humans.append(_HumanMsg(ts=ts, is_interrupt=False))
                recent_context.append(human)

    # Backfill has_tools (a session-level property of the harness).
    for r in responses:
        r.has_tools = has_tools
    return responses, humans, has_tools


def _ordered_events(path: Path) -> list[tuple[float | None, str, _Response | _HumanMsg]]:
    """Re-read interleaving order so fast-skip can pair a human reply to the
    assistant response immediately before it. Kept simple: re-parse with a tag.
    """
    out: list[tuple[float | None, str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return out
    responses, humans, _ = _parse_session(path)
    # The two lists are already in file order; interleave by re-walking lines.
    ri = hi = 0
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ltype = d.get("type")
        if ltype == "assistant":
            # Mirror the _parse_session filter (a _Response is only created when
            # `words > 0 or out_tok > 0`). Without the same guard here, an empty
            # assistant line (e.g. a pure tool_use turn with no usage) consumes a
            # response slot that belongs to a later real response, desyncing the
            # two lists so a following human reply is paired with the wrong
            # (future-timestamped) response -> a negative gap -> a spurious
            # fast_skip.
            msg = d.get("message", {}) if isinstance(d.get("message"), dict) else {}
            words, _ = _assistant_words_and_text(msg.get("content", []))
            usage = msg.get("usage", {}) if isinstance(msg.get("usage"), dict) else {}
            out_tok = usage.get("output_tokens", 0)
            if (words > 0 or out_tok > 0) and ri < len(responses):
                out.append((responses[ri].ts, "assistant", responses[ri]))
                ri += 1
        elif ltype == "user":
            msg = d.get("message", {}) if isinstance(d.get("message"), dict) else {}
            text = _human_text(msg.get("content"))
            if text is None:
                continue
            if hi < len(humans):
                out.append((humans[hi].ts, "human", humans[hi]))
                hi += 1
    return out


def extract_signals(
    session_paths: list[Path],
) -> tuple[VerbositySignals, BaselineModel]:
    """Compute behavioral signals and the per-stratum output-token baseline."""
    sig = VerbositySignals()
    baseline = BaselineModel()
    all_response_words: list[int] = []

    # First pass: collect every response word-count to derive the per-user
    # adaptive "long" threshold (median), so "long" scales to the user.
    parsed: list[tuple[list[_Response], list[_HumanMsg]]] = []
    for p in session_paths:
        responses, humans, _ = _parse_session(p)
        if not responses and not humans:
            continue
        sig.sessions += 1
        parsed.append((responses, humans))
        all_response_words.extend(r.words for r in responses if r.words > 0)

    long_threshold = _LONG_OUTPUT_FLOOR
    if all_response_words:
        all_response_words.sort()
        median = all_response_words[len(all_response_words) // 2]
        long_threshold = max(_LONG_OUTPUT_FLOOR, median)

    echo_sum = 0.0
    echo_n = 0
    for responses, humans in parsed:
        for r in responses:
            sig.asst_responses += 1
            sig.asst_words += r.words
            if r.words >= long_threshold:
                sig.long_outputs += 1
            if r.echo > 0:
                echo_sum += r.echo
                echo_n += 1
            if r.output_tokens > 0:
                key = stratum_key(
                    turn_kind=r.turn_kind,
                    input_tokens=r.input_tokens,
                    model=r.model or "unknown",
                    has_tools=r.has_tools,
                )
                baseline.observe(key, r.output_tokens)
        for h in humans:
            if h.is_interrupt:
                sig.interrupts += 1
            else:
                sig.human_msgs += 1

    # Second pass: fast-skip pairing via interleaved order.
    for p in session_paths:
        events = _ordered_events(p)
        last_resp: _Response | None = None
        for ts, kind, obj in events:
            if kind == "assistant":
                last_resp = obj  # type: ignore[assignment]
            elif kind == "human":
                hm: _HumanMsg = obj  # type: ignore[assignment]
                if (
                    last_resp is not None
                    and not hm.is_interrupt
                    and last_resp.words >= _MIN_WORDS_FOR_SKIP
                    and ts is not None
                    and last_resp.ts is not None
                ):
                    sig.skip_eligible += 1
                    read_secs = last_resp.words / _READING_WPM * 60.0
                    if (ts - last_resp.ts) < _SKIP_READ_FRACTION * read_secs:
                        sig.fast_skips += 1
                last_resp = None

    sig.mean_echo_ratio = echo_sum / echo_n if echo_n else 0.0
    return sig, baseline


def recommend_level(sig: VerbositySignals) -> tuple[int, str, str]:
    """Heuristic prior mapping signals → (level, confidence, rationale).

    This is the prior; an LLM judgment pass (in :func:`analyze`) may override
    it. Bands are interpretable: the more a user interrupts and fast-skips, the
    less of the output they consume, so the terser we should make it.
    """
    if sig.human_msgs + sig.interrupts < 10:
        return 2, "low", "Too few human turns to calibrate; defaulting to L2."

    ir = sig.interrupt_rate
    fsr = sig.fast_skip_rate
    pressure = ir + fsr  # combined "too much output" pressure

    confidence = "high" if (sig.human_msgs + sig.interrupts) >= 60 else "medium"

    if pressure < 0.10:
        return (
            1,
            confidence,
            (
                f"Low push-back (interrupt {ir:.0%}, fast-skip {fsr:.0%}); user reads "
                "answers — light touch (L1)."
            ),
        )
    if pressure < 0.30:
        return (
            2,
            confidence,
            (
                f"Moderate push-back (interrupt {ir:.0%}, fast-skip {fsr:.0%}); "
                "drop ceremony and echo (L2)."
            ),
        )
    if pressure < 0.55:
        return (
            3,
            confidence,
            (
                f"High push-back (interrupt {ir:.0%}, fast-skip {fsr:.0%}); user "
                "rarely reads long answers — conclusions only (L3)."
            ),
        )
    return (
        3,
        confidence,
        (
            f"Very high push-back (interrupt {ir:.0%}, fast-skip {fsr:.0%}); capping "
            "at L3 rather than auto-applying caveman L4."
        ),
    )


def analyze(
    session_paths: list[Path],
    project_path: str,
    *,
    llm_judge: Any | None = None,
) -> tuple[VerbosityProfile, BaselineModel]:
    """Full analysis: signals → recommendation → profile, plus the baseline.

    ``llm_judge``, if given, is a callable ``(signals_dict) -> (level, rationale)``
    that overrides the heuristic. Kept injectable so the core stays LLM-free and
    testable; the CLI wires a real LLM call.
    """
    sig, baseline = extract_signals(session_paths)
    level, confidence, rationale = recommend_level(sig)
    source = "heuristic"
    if llm_judge is not None:
        try:
            verdict = llm_judge(sig.to_dict())
            if verdict is not None:
                level, rationale = verdict
                level = max(0, min(4, int(level)))
                source = "llm"
        except Exception as e:  # LLM is advisory — never fail the analysis
            logger.warning("verbosity LLM judge failed, using heuristic: %s", e)

    profile = VerbosityProfile(
        project_path=project_path,
        level=level,
        confidence=confidence,
        source=source,
        rationale=rationale,
        signals=sig.to_dict(),
    )
    return profile, baseline
