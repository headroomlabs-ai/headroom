"""Session analysis via LLM — replaces all regex/heuristic analysis.

Pipeline: Scanner (events) → Digest Builder → LLM → Recommendations

No regex patterns, no static lookback windows, no hardcoded heuristics.
A single LLM call understands the full conversation context and produces
structured recommendations for CLAUDE.md / MEMORY.md.

Supports any LLM provider via LiteLLM: Anthropic, OpenAI, Google, Bedrock,
Ollama, and 100+ others. Auto-detects the best available model from env vars.
Also supports CLI-based backends (claude, gemini, codex) for subscription
users without raw API keys.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import typing

from headroom._subprocess import Popen, run

from .loops import LoopPattern, apply_loop_weighting, detect_loops, format_loops_for_digest
from .models import (
    AnalysisResult,
    ProjectInfo,
    Recommendation,
    RecommendationTarget,
    SessionData,
    SessionEvent,
    ToolCall,
)
from .writer import extract_marker_block

logger = logging.getLogger(__name__)

# Default models by provider (checked in order)
_MODEL_DEFAULTS: list[tuple[str, str]] = [
    ("ANTHROPIC_API_KEY", "claude-sonnet-4-6"),
    ("OPENAI_API_KEY", "gpt-4o"),
    ("GEMINI_API_KEY", "gemini/gemini-flash-latest"),
]

_MAX_DIGEST_TOKENS = 80_000  # Budget for the digest (leave room for prompt + output)

# CLI tools to try when no API key is set (checked in order).
# Each entry: (binary_name, model_identifier, command_prefix). The claude-cli
# command uses stream-json output so the analyzer can detect progress and
# enforce an idle (rather than wall-clock-only) timeout — see _call_cli_llm.
_CLI_BACKENDS: list[tuple[str, str, list[str]]] = [
    ("claude", "claude-cli", ["claude", "-p", "--output-format", "stream-json", "--verbose"]),
    ("gemini", "gemini-cli", ["gemini", "-p"]),
    ("codex", "codex-cli", ["codex", "exec"]),
]

# Set of valid CLI model identifiers, derived from _CLI_BACKENDS.
_CLI_MODEL_IDS: set[str] = {model for _, model, _ in _CLI_BACKENDS}

_USER_PROMPT_PREFIX = "Analyze these coding agent sessions and return JSON recommendations:\n\n"  # Shared by _call_cli_llm and _call_llm
_MAX_SNIPPET_LEN = 2000  # Max chars of CLI output (stdout/stderr) in error messages
# Hard wall-clock cap for CLI backends (seconds). Override with
# HEADROOM_LEARN_CLI_TIMEOUT_SECS for slow networks or large digests.
_CLI_TIMEOUT = 300
# Idle cap (seconds) for streaming claude-cli: kill if no output arrives for
# this long. Lets us catch genuine hangs quickly while letting long-but-active
# analyses run to completion. Override with HEADROOM_LEARN_CLI_IDLE_TIMEOUT_SECS.
_CLI_IDLE_TIMEOUT = 60


def _resolve_timeout_secs(env_var: str, default: int) -> int:
    """Resolve a positive-integer timeout from *env_var* or fall back to *default*.

    Invalid or non-positive values are logged and ignored so a typo in env
    config can't accidentally disable the timeout.
    """
    raw = os.environ.get(env_var)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r — using default %ds", env_var, raw, default)
        return default
    if value <= 0:
        logger.warning(
            "Invalid %s=%r (must be positive) — using default %ds", env_var, raw, default
        )
        return default
    return value


def _detect_default_model() -> str:
    """Pick the best available model based on API keys, env config, or CLI tools.

    Priority order:
      1. API key present → use corresponding LiteLLM model
      2. HEADROOM_LEARN_CLI env var → use specified CLI backend
      3. Auto-detect installed CLI tools (claude > gemini > codex)
      4. Raise RuntimeError with setup instructions
    """
    # 1. API key detection (existing behavior)
    for env_var, model in _MODEL_DEFAULTS:
        if os.environ.get(env_var):
            return model

    # 2. Explicit CLI selection via environment variable
    cli_override = os.environ.get("HEADROOM_LEARN_CLI")
    if cli_override:
        for cli_name, model, _cmd in _CLI_BACKENDS:
            if cli_name == cli_override:
                logger.info("HEADROOM_LEARN_CLI=%s — using %s CLI backend", cli_override, cli_name)
                return model
        valid = ", ".join(name for name, _, _ in _CLI_BACKENDS)
        raise ValueError(
            f"HEADROOM_LEARN_CLI={cli_override!r} is not a supported CLI. Valid values: {valid}"
        )

    # 3. Auto-detect installed CLI tools
    for cli_name, model, _cmd in _CLI_BACKENDS:
        if shutil.which(cli_name):
            logger.info("No API key found — auto-detected %s CLI as LLM backend", cli_name)
            return model

    raise RuntimeError(
        "No LLM API key found. headroom learn needs one of:\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...   → uses claude-sonnet-4-6\n"
        "  export OPENAI_API_KEY=sk-...          → uses gpt-4o\n"
        "  export GEMINI_API_KEY=...             → uses gemini-flash-latest\n"
        "Or set HEADROOM_LEARN_CLI to a coding agent CLI (claude, gemini, codex).\n"
        "Or install one of those CLIs for auto-detection.\n"
        "Or specify a model directly: headroom learn --model <litellm-model-name>"
    )


class SessionAnalyzer:
    """Analyzes session data via LLM to produce actionable recommendations.

    Uses LiteLLM for provider-agnostic access to 100+ models.
    Auto-detects the best available model from environment API keys.
    """

    def __init__(self, model: str | None = None):
        self.model = model

    def analyze(self, project: ProjectInfo, sessions: list[SessionData]) -> AnalysisResult:
        """Analyze sessions and produce recommendations via LLM."""
        all_calls = [tc for s in sessions for tc in s.tool_calls]
        failed_calls = [tc for tc in all_calls if tc.is_error]

        result = AnalysisResult(
            project=project,
            total_sessions=len(sessions),
            total_calls=len(all_calls),
            total_failures=len(failed_calls),
        )

        # Detect loops up front: an RTK re-fetch loop has NO failed calls
        # (each truncated command succeeds), so it must be a first-class reason
        # to analyze — otherwise the guard below would skip the most expensive
        # waste pattern whenever a session has no failures and no events.
        loops = detect_loops(sessions)

        if not failed_calls and not loops and not any(s.events for s in sessions):
            return result

        # Build compact digest of all sessions, leading with detected loops.
        digest = _build_digest(project, sessions, loops=loops)

        # Resolve model (auto-detect if not specified)
        model = self.model or _detect_default_model()

        # Call LLM for analysis
        try:
            raw = _call_llm(digest, model)
            result.recommendations = _parse_llm_response(raw)
            # Weight loop guardrails above one-off rules using MEASURED waste.
            apply_loop_weighting(result.recommendations, loops)
            result.recommendations.sort(key=lambda r: r.estimated_tokens_saved, reverse=True)
        except Exception as e:
            logger.warning("LLM analysis failed: %s", e)
            # Return result with stats but no recommendations

        return result


# =============================================================================
# Digest Builder — compact text representation of session events
# =============================================================================


def _build_prior_patterns_section(project: ProjectInfo) -> str:
    """Format the current marker blocks from CLAUDE.md / MEMORY.md for the LLM.

    Returns "" when neither file exists nor contains a marker block. When at
    least one file has a block, returns a header + labeled raw blocks so the
    LLM can treat them as the starting baseline. See the "Prior Learned
    Patterns" rule in _SYSTEM_PROMPT for the contract with the model.
    """
    parts: list[tuple[str, str]] = []  # (label, block)
    candidates = (
        ("CLAUDE.md (CONTEXT_FILE, project-level stable facts)", project.context_file),
        ("MEMORY.md (MEMORY_FILE, session-level evolving preferences)", project.memory_file),
    )
    for label, path in candidates:
        if path is None or not path.exists():
            continue
        block = extract_marker_block(path.read_text(encoding="utf-8", errors="replace"))
        if block:
            parts.append((label, block))

    if not parts:
        return ""

    lines = [
        "=== Prior Learned Patterns ===",
        (
            f"These patterns are currently written to {project.name}'s context "
            f"files. They are your starting baseline — see the 'Prior Learned "
            f"Patterns' rule in the system prompt for how to integrate them."
        ),
        "",
    ]
    for label, block in parts:
        lines.append(f"--- From {label} ---")
        lines.append(block)
        lines.append("")
    return "\n".join(lines)


def _build_digest(
    project: ProjectInfo,
    sessions: list[SessionData],
    loops: list[LoopPattern] | None = None,
) -> str:
    """Build a token-efficient text digest of all session events.

    The digest includes:
    - Project context
    - Detected loops (highest priority) — repeated patterns + measured waste
    - Prior learned patterns (if any) from CLAUDE.md / MEMORY.md
    - Per-session summaries with condensed event streams
    - Error outputs (truncated), success indicators, user messages

    ``loops`` is computed by the caller (``SessionAnalyzer.analyze``) and passed
    in to avoid detecting twice; when omitted it is detected here so callers
    that build a digest directly still surface loops.
    """
    if loops is None:
        loops = detect_loops(sessions)

    lines: list[str] = []

    # Project header
    lines.append(f"Project: {project.name} ({project.project_path})")
    total_calls = sum(len(s.tool_calls) for s in sessions)
    total_failures = sum(s.failure_count for s in sessions)
    total_tokens_in = sum(s.total_input_tokens for s in sessions)
    total_tokens_out = sum(s.total_output_tokens for s in sessions)
    lines.append(
        f"Total: {len(sessions)} sessions, {total_calls} tool calls, "
        f"{total_failures} failures ({total_failures / total_calls:.1%})"
        if total_calls
        else f"Total: {len(sessions)} sessions, 0 tool calls"
    )
    if total_tokens_in:
        lines.append(f"Tokens used: {total_tokens_in:,} in / {total_tokens_out:,} out")
    lines.append("")

    # Detected loops first — the most expensive waste pattern, so the LLM sees
    # it before the (budget-truncatable) per-session event stream.
    loop_section = format_loops_for_digest(loops)
    if loop_section:
        lines.append(loop_section)

    # Prior learned patterns (if any) — gives the LLM the current baseline so
    # it can produce complete updated sections instead of condensed deltas.
    prior_section = _build_prior_patterns_section(project)
    if prior_section:
        lines.append(prior_section)

    # Budget tracking — stop adding events when we approach the limit
    # Rough estimate: 4 chars per token
    char_budget = _MAX_DIGEST_TOKENS * 4
    chars_used = sum(len(ln) for ln in lines)

    for session in sessions:
        if chars_used > char_budget:
            lines.append(
                f"... (remaining {len(sessions) - sessions.index(session)} sessions truncated)"
            )
            break

        session_header = (
            f"=== Session {session.session_id[:12]} "
            f"({len(session.tool_calls)} calls, {session.failure_count} failures"
        )
        if session.total_input_tokens:
            session_header += f", {session.total_input_tokens:,} input tokens"
        session_header += ") ==="
        lines.append(session_header)
        chars_used += len(session_header)

        # Use events if available (richer context), fall back to tool_calls
        if session.events:
            for event in session.events:
                if chars_used > char_budget:
                    lines.append("  ... (remaining events truncated)")
                    break
                event_line = _format_event(event)
                if event_line:
                    lines.append(event_line)
                    chars_used += len(event_line)
        else:
            for tc in session.tool_calls:
                if chars_used > char_budget:
                    lines.append("  ... (remaining calls truncated)")
                    break
                tc_line = _format_tool_call(tc)
                lines.append(tc_line)
                chars_used += len(tc_line)

        lines.append("")

    return "\n".join(lines)


def _format_event(event: SessionEvent) -> str | None:
    """Format a single event into a compact digest line."""

    if event.type == "tool_call" and event.tool_call:
        return _format_tool_call(event.tool_call)

    if event.type == "user_message" and event.text.strip():
        text = event.text.strip()[:300]
        return f'  [{event.msg_index}] USER: "{text}"'

    if event.type == "interruption":
        return f"  [{event.msg_index}] INTERRUPTED: {event.text[:150]}"

    if event.type == "agent_summary":
        return (
            f"  [{event.msg_index}] SUBAGENT: {event.agent_tool_count} tool calls, "
            f"{event.agent_tokens:,} tokens, {event.agent_duration_ms / 1000:.1f}s "
            f'— prompt: "{event.agent_prompt[:100]}"'
        )

    return None


def _format_tool_call(tc: ToolCall) -> str:
    """Format a single tool call into a compact digest line."""
    status = "ERROR" if tc.is_error else "OK"
    error_cat = f"({tc.error_category.value})" if tc.is_error else ""

    # Input summary
    input_str = tc.input_summary[:120]

    if tc.is_error:
        # Include truncated error output for failures
        output_preview = tc.output[:200].replace("\n", " ").strip()
        return f"  [{tc.msg_index}] {tc.name}: {input_str} → {status}{error_cat}: {output_preview}"
    else:
        # Just indicate success with size
        size = f"({tc.output_bytes} bytes)" if tc.output_bytes > 0 else ""
        return f"  [{tc.msg_index}] {tc.name}: {input_str} → {status} {size}"


# =============================================================================
# LLM Call — Sonnet 4.6 with structured output
# =============================================================================

_SYSTEM_PROMPT = """\
You are an expert at analyzing coding agent sessions to extract actionable patterns.

You will receive a digest of tool call sessions from a coding agent (Claude Code, Codex, etc.).
Your job is to identify patterns that, if documented, would PREVENT TOKEN WASTE in future sessions.

Focus on (in priority order):
1. **Loops (HIGHEST PRIORITY)** — patterns that REPEATED within a session. If the
   digest has a "Detected Loops" section, every loop there MUST get a guardrail
   rule, because loop waste scales with repetition. This includes RTK re-fetch
   loops: a command whose output was truncated, so the agent re-ran variants of
   it to fetch more. The fix names the command and prescribes getting the full
   output up front (e.g., "read the whole file" / "raise the output limit for X").
2. **Environment rules** — what runtime commands work vs fail (e.g., "use uv run python, not python3")
3. **File structure facts** — known large files, correct paths, search scopes
4. **User preferences** — things the user corrected, rejected, or explicitly requested
5. **Failure patterns** — repeated failures that could be prevented with upfront knowledge
6. **Workflow rules** — subagent guidance, command execution preferences
7. **Token waste hotspots** — patterns that waste the most tokens (re-reads, wrong paths, retries)

Rules:
- A loop in the "Detected Loops" section is sufficient evidence on its own — emit
  its guardrail even if it appears only once as a loop, and set its
  estimated_tokens_saved to at least the measured wasted tokens reported there.
- Only include patterns with CLEAR evidence from the data (2+ occurrences or explicit user direction)
- Every recommendation must be specific and actionable (not "be careful" but "use X instead of Y")
- Estimate tokens saved per recommendation (how many tokens would be saved per session if this rule existed)
- Separate stable project facts (CONTEXT_FILE) from evolving preferences (MEMORY_FILE)
- CONTEXT_FILE rules go in CLAUDE.md/AGENTS.md — they are project-level, stable facts
- MEMORY_FILE rules go in MEMORY.md — they are session-level, evolving preferences
- Keep recommendations concise — each should be 1-3 lines of markdown
- Do NOT produce tautological rules (e.g., "use python3 not python3")
- Do NOT produce rules about things that only happened once (transient errors)

Prior Learned Patterns:
- The input may contain a "Prior Learned Patterns" section showing what is
  already written to the project's CLAUDE.md / MEMORY.md. Treat those as the
  starting baseline for your analysis.
- When you re-emit a section heading that appears in the prior block, your
  output REPLACES that prior section wholesale — so your section must be the
  COMPLETE updated version:
    * Preserve prior bullets that remain accurate (copy them forward)
    * Revise bullets when new evidence refines them (merge, don't duplicate)
    * Drop a prior bullet only when contradicted by clear new evidence
- Sections from prior runs that you do NOT re-emit are preserved automatically
  by the writer, so focus only on sections where you have something to add or
  change. Do NOT re-emit a prior section just to echo it verbatim — that wastes
  output tokens without changing the outcome.
- Do NOT write bullets that reference prior siblings you are about to drop
  (e.g., "X is ALSO large — same rule as Y, Z") unless Y and Z are also present
  in your current output or preserved in the prior block.

Return ONLY valid JSON matching this schema — no other text:
{
  "context_file_rules": [
    {
      "section": "string — section heading (e.g., 'Environment', 'File Paths', 'Commands')",
      "content": "string — markdown content, 1-3 bullet points",
      "estimated_tokens_saved": "integer — tokens saved per session if rule existed",
      "evidence_count": "integer — number of occurrences supporting this rule"
    }
  ],
  "memory_file_rules": [
    {
      "section": "string — section heading",
      "content": "string — markdown content, 1-3 bullet points",
      "estimated_tokens_saved": "integer",
      "evidence_count": "integer"
    }
  ]
}
"""


def _strip_fenced_json(raw: str) -> dict:
    """Strip optional markdown fences and parse JSON.

    Handles both raw JSON and fenced code blocks (e.g. ``​`json ... ``​`).
    Only the first opening fence and last closing fence are removed, preserving
    any triple-backtick content that may appear inside the JSON payload.

    Args:
        raw: Raw text output from an LLM, possibly wrapped in markdown fences.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        json.JSONDecodeError: If the text is not valid JSON after stripping.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove the first line (opening fence, e.g. ```json)
        lines = lines[1:]
        # Remove the last line if it is a closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    result: dict = json.loads(text)
    return result


def _call_cli_llm(digest: str, model: str) -> dict:
    """Call a locally installed CLI tool as the LLM backend.

    Enables keyless usage for subscription-based CLI tools that handle
    their own OAuth authentication. The prompt is passed via stdin to avoid
    OS ``ARG_MAX`` limits and argument-injection risks.

    CLI invocations:
      claude-cli → claude -p --output-format stream-json --verbose (idle-timeout)
      gemini-cli → gemini -p (wall-clock timeout)
      codex-cli  → codex exec (wall-clock timeout)

    The claude-cli path streams JSON events, letting the analyzer kill genuine
    hangs while letting long-but-active analyses run to completion.

    Args:
        digest: Token-efficient session digest to analyze.
        model: CLI model identifier (e.g. ``claude-cli``).

    Returns:
        Parsed JSON recommendations from the CLI tool.

    Raises:
        ValueError: If *model* is not a known CLI backend.
        RuntimeError: If the CLI is not installed, exits non-zero, or times out.
    """
    cmd: list[str] | None = None
    for _name, model_name, cmd_parts in _CLI_BACKENDS:
        if model_name == model:
            cmd = cmd_parts
            break
    if cmd is None:
        raise ValueError(f"Unknown CLI model: {model}")

    prompt = _SYSTEM_PROMPT + "\n\n" + _USER_PROMPT_PREFIX + digest
    hard_cap = _resolve_timeout_secs("HEADROOM_LEARN_CLI_TIMEOUT_SECS", _CLI_TIMEOUT)

    if model == "claude-cli":
        idle_cap = _resolve_timeout_secs("HEADROOM_LEARN_CLI_IDLE_TIMEOUT_SECS", _CLI_IDLE_TIMEOUT)
        return _call_claude_cli_streaming(cmd, prompt, hard_cap=hard_cap, idle_cap=idle_cap)

    try:
        result = run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=hard_cap,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"`{cmd[0]}` not found in PATH. Install it or use a different backend "
            "with --model <litellm-model-name>."
        ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"`{' '.join(cmd)}` did not respond within {hard_cap}s. "
            "Check network connectivity, raise HEADROOM_LEARN_CLI_TIMEOUT_SECS, "
            "or try a different backend with --model <litellm-model-name>."
        ) from None

    if result.returncode != 0:
        stderr_snippet = (result.stderr or "")[:_MAX_SNIPPET_LEN]
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed (exit {result.returncode}):\n{stderr_snippet}"
        )

    # Log stderr warnings even on success (auth refreshes, deprecation notices).
    if result.stderr and result.stderr.strip():
        logger.debug("CLI stderr (exit 0): %s", result.stderr[:_MAX_SNIPPET_LEN])

    try:
        return _strip_fenced_json(result.stdout)
    except json.JSONDecodeError as exc:
        stdout_snippet = (result.stdout or "")[:_MAX_SNIPPET_LEN]
        raise RuntimeError(
            f"`{' '.join(cmd)}` returned unparseable output. "
            f"First {_MAX_SNIPPET_LEN} chars:\n{stdout_snippet}"
        ) from exc


def _call_claude_cli_streaming(
    cmd: list[str], prompt: str, *, hard_cap: int, idle_cap: int
) -> dict:
    """Run claude-cli with stream-json output and an idle-timeout watchdog.

    Each line of stdout is one JSON event from claude (system/assistant/user/
    result). Any line resets the idle deadline. The process is killed if no
    output arrives for *idle_cap* seconds, or if total elapsed exceeds
    *hard_cap* seconds. The final ``type:"result"`` event carries the assistant
    response, which is then parsed as JSON.

    Threads (rather than ``select``) drain stdout/stderr so the watchdog works
    on Windows too, where ``select`` does not support pipe handles.
    """
    # On Windows, raw scripts (e.g. `claude` with a bash shebang from an npm
    # global install) can't be executed directly by CreateProcess — they need
    # the `.cmd`/`.exe` shim. Try the bare name first; on FileNotFoundError fall
    # back to <name>.cmd (and <name>.exe) so npm-global CLIs work on Windows.
    candidates = [cmd]
    if os.name == "nt" and "/" not in cmd[0] and "\\" not in cmd[0]:
        base = cmd[0]
        for ext in (".cmd", ".exe", ".bat"):
            candidates.append([base + ext] + cmd[1:])

    proc = None
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            proc = Popen(
                candidate,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
            break
        except FileNotFoundError as exc:
            last_exc = exc
            continue

    if proc is None:
        raise RuntimeError(
            f"`{cmd[0]}` not found in PATH. Install it or use a different backend "
            "with --model <litellm-model-name>."
        ) from last_exc

    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    try:
        proc.stdin.write(prompt)
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:  # pragma: no cover — defensive, claude exits before stdin drain
            pass

    events: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def _pump(stream: typing.IO[str], tag: str) -> None:
        try:
            for line in stream:
                events.put((tag, line))
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("stream pump (%s) errored: %s", tag, exc)
        finally:
            events.put((tag, None))  # EOF marker

    threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True).start()

    start = time.monotonic()
    last_activity = start
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    final_result: str | None = None
    eofs = 0

    def _kill(reason: str) -> None:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except (
            subprocess.TimeoutExpired
        ):  # pragma: no cover — defensive, kill normally returns fast
            pass
        logger.debug("claude-cli killed: %s", reason)

    while eofs < 2:
        elapsed = time.monotonic() - start
        if elapsed > hard_cap:
            _kill(f"hard cap {hard_cap}s exceeded")
            raise RuntimeError(
                f"`{' '.join(cmd)}` exceeded the {hard_cap}s hard cap. "
                "Raise HEADROOM_LEARN_CLI_TIMEOUT_SECS for slower networks or "
                "larger digests, or try a different backend with "
                "--model <litellm-model-name>."
            )
        idle_elapsed = time.monotonic() - last_activity
        if idle_elapsed > idle_cap:
            _kill(f"idle cap {idle_cap}s exceeded")
            raise RuntimeError(
                f"`{' '.join(cmd)}` produced no output for {idle_cap}s. "
                "Check network connectivity, raise "
                "HEADROOM_LEARN_CLI_IDLE_TIMEOUT_SECS, or try a different "
                "backend with --model <litellm-model-name>."
            )

        # Block up to 1s waiting for the next event, then re-check deadlines.
        try:
            tag, line = events.get(timeout=1.0)
        except queue.Empty:
            continue

        if line is None:
            eofs += 1
            continue
        last_activity = time.monotonic()
        if tag == "stdout":
            stdout_lines.append(line)
            event = _parse_stream_event(line)
            if event is not None and event.get("type") == "result":
                # Last result event wins if multiple are emitted.
                result_text = event.get("result")
                if isinstance(result_text, str):
                    final_result = result_text
        else:
            stderr_lines.append(line)

    proc.wait()

    if proc.returncode != 0:
        stderr_blob = "".join(stderr_lines)[:_MAX_SNIPPET_LEN]
        raise RuntimeError(f"`{' '.join(cmd)}` failed (exit {proc.returncode}):\n{stderr_blob}")

    stderr_blob = "".join(stderr_lines)
    if stderr_blob.strip():
        logger.debug("CLI stderr (exit 0): %s", stderr_blob[:_MAX_SNIPPET_LEN])

    if final_result is None:
        stdout_snippet = "".join(stdout_lines)[:_MAX_SNIPPET_LEN]
        raise RuntimeError(
            f"`{' '.join(cmd)}` did not emit a final `result` event. "
            f"First {_MAX_SNIPPET_LEN} chars of stdout:\n{stdout_snippet}"
        )

    try:
        return _strip_fenced_json(final_result)
    except json.JSONDecodeError as exc:
        snippet = final_result[:_MAX_SNIPPET_LEN]
        raise RuntimeError(
            f"`{' '.join(cmd)}` returned unparseable output. "
            f"First {_MAX_SNIPPET_LEN} chars:\n{snippet}"
        ) from exc


def _parse_stream_event(line: str) -> dict | None:
    """Parse one line of claude-cli stream-json output, returning None on junk."""
    line = line.strip()
    if not line:
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _call_llm(digest: str, model: str) -> dict:
    """Call LLM with the session digest and return parsed JSON.

    Uses LiteLLM for provider-agnostic access. The model string determines
    the provider: "claude-*" → Anthropic, "gpt-*" → OpenAI, "gemini/*" → Google, etc.
    For CLI-based models (ending in "-cli"), delegates to ``_call_cli_llm``.
    """
    if model in _CLI_MODEL_IDS:
        return _call_cli_llm(digest, model)

    import litellm

    # Suppress LiteLLM's verbose logging
    litellm.suppress_debug_info = True

    # For Anthropic models, bypass ANTHROPIC_BASE_URL which may point to
    # the user's local headroom proxy
    api_base = None
    if model.startswith("claude"):
        api_base = "https://api.anthropic.com"

    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT_PREFIX + digest,
            },
        ],
        max_tokens=4096,
        api_base=api_base,
    )

    # Extract text from response
    text = response.choices[0].message.content or ""
    return _strip_fenced_json(text)


# =============================================================================
# Response Parser — LLM JSON → Recommendation list
# =============================================================================


def _parse_llm_response(raw: dict) -> list[Recommendation]:
    """Convert LLM structured output into Recommendation objects."""
    recommendations: list[Recommendation] = []

    for rule in raw.get("context_file_rules", []):
        if not isinstance(rule, dict):
            continue
        section = rule.get("section", "").strip()
        content = rule.get("content", "").strip()
        if not section or not content:
            continue
        recommendations.append(
            Recommendation(
                target=RecommendationTarget.CONTEXT_FILE,
                section=section,
                content=content,
                confidence=0.9,
                evidence_count=_safe_int(rule.get("evidence_count", 1)),
                estimated_tokens_saved=_safe_int(rule.get("estimated_tokens_saved", 0)),
            )
        )

    for rule in raw.get("memory_file_rules", []):
        if not isinstance(rule, dict):
            continue
        section = rule.get("section", "").strip()
        content = rule.get("content", "").strip()
        if not section or not content:
            continue
        recommendations.append(
            Recommendation(
                target=RecommendationTarget.MEMORY_FILE,
                section=section,
                content=content,
                confidence=0.7,
                evidence_count=_safe_int(rule.get("evidence_count", 1)),
                estimated_tokens_saved=_safe_int(rule.get("estimated_tokens_saved", 0)),
            )
        )

    # Sort by estimated token savings
    recommendations.sort(key=lambda r: r.estimated_tokens_saved, reverse=True)

    return recommendations


def _safe_int(val: object) -> int:
    """Safely convert a value to int."""
    if isinstance(val, int):
        return val
    if isinstance(val, (float, str)):
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0
    return 0


# =============================================================================
# Legacy compatibility alias
# =============================================================================


class FailureAnalyzer:
    """Legacy alias for SessionAnalyzer — used by existing CLI code."""

    def __init__(self) -> None:
        self._analyzer = SessionAnalyzer()

    def analyze(self, project: ProjectInfo, sessions: list[SessionData]) -> AnalysisResult:
        return self._analyzer.analyze(project, sessions)
