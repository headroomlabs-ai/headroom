"""Live Traffic Pattern Learner — extracts memories from proxy traffic.

Hooks into the proxy request/response pipeline to learn patterns without
any LLM calls. Rule-based extraction from traffic the proxy already sees:
- Error → Recovery patterns (tool fails → next success teaches right approach)
- Environment facts (commands that work/fail, paths, tool availability)
- Preference signals (repeated patterns, corrections)
- Architectural decisions (file references, dependency choices)

Usage:
    learner = TrafficLearner(memory_backend)
    await learner.on_request(messages, agent_type="claude")
    await learner.on_response(response, messages, agent_type="claude")

The learner is designed to be zero-config and zero-latency: it processes
patterns in the background and never blocks the proxy pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from headroom.learn.models import ProjectInfo
    from headroom.memory.backends.local import LocalBackend

logger = logging.getLogger(__name__)

# Minimum seconds between successive flush_to_file calls when driven by the
# dirty-flag worker. Prevents CLAUDE.md thrash during bursty traffic while
# still staying "near real-time" from the user's perspective.
FLUSH_DEBOUNCE_SECONDS = 10.0

# Absolute file-path heuristic for anchoring a pattern to a project root.
# Matches POSIX paths (starts with /) and common Windows drive paths.
_ABS_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[\w./\\@\-]+")

# Error-recovery refinement: the Learned: error recovery section is capped,
# decayed, and re-validated at render time. Other categories are untouched.
_ERROR_RECOVERY_SECTION_CAP = 15
_ERROR_RECOVERY_HALF_LIFE_DAYS = 5.0
_ERROR_RECOVERY_HARD_FLOOR_DAYS = 21

# Suffixes that vary between otherwise-identical Bash recoveries. Stripping
# them before hashing collapses near-duplicates.
_BASH_VOLATILE_SUFFIX_RE = re.compile(
    r"(?:\s*\|\s*(?:head|tail)\s+-n?\s*\d+"
    r"|\s+-A\s*\d+|\s+-B\s*\d+|\s+-C\s*\d+"
    r"|\s+2>&1|\s+2>/dev/null)+\s*$"
)


# =============================================================================
# Pattern Categories
# =============================================================================


class PatternCategory(str, Enum):
    """Categories of patterns extracted from traffic."""

    ERROR_RECOVERY = "error_recovery"  # Tool failed → next call succeeded
    ENVIRONMENT = "environment"  # Working commands, paths, tool availability
    PREFERENCE = "preference"  # Repeated choices, corrections
    ARCHITECTURE = "architecture"  # File structure, dependencies, conventions


class AgentType(str, Enum):
    """Supported coding agent types."""

    CLAUDE = "claude"
    CURSOR = "cursor"
    CODEX = "codex"
    AIDER = "aider"
    GEMINI = "gemini"
    UNKNOWN = "unknown"


# =============================================================================
# Extracted Pattern Model
# =============================================================================


@dataclass
class ExtractedPattern:
    """A pattern extracted from proxy traffic."""

    category: PatternCategory
    content: str  # Human-readable memory content
    importance: float  # 0.0 - 1.0
    evidence_count: int = 1  # How many times this pattern was observed
    entity_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.content_hash:
            key = _normalize_hash_key(self.category, self.content, self.metadata)
            self.content_hash = hashlib.sha256(key.encode()).hexdigest()[:16]


def _normalize_hash_key(
    category: PatternCategory,
    content: str,
    metadata: dict[str, Any],
) -> str:
    """Build the string that feeds the content hash.

    Error-recovery rows are collapsed on recovery intent, not literal text:
    trivial invocation differences (tail counts, pipe suffixes, full paths
    that share a basename) hash to the same key. Other categories hash the
    raw content for backwards compatibility.
    """
    if category is not PatternCategory.ERROR_RECOVERY:
        return content

    tool = metadata.get("tool")
    if tool == "Read":
        error_path = metadata.get("error_path", "")
        success_path = metadata.get("success_path", "")
        return (
            f"error_recovery|Read|{os.path.basename(error_path)}|{os.path.basename(success_path)}"
        )
    if tool == "Bash":
        failed = metadata.get("failed_cmd", "")
        success = metadata.get("success_cmd", "")
        return (
            f"error_recovery|Bash|"
            f"{_normalize_bash_for_hash(failed)}|{_normalize_bash_for_hash(success)}"
        )
    return content


def _normalize_bash_for_hash(cmd: str) -> str:
    """Strip volatile suffixes and truncate at the first pipe/chain boundary."""
    if not cmd:
        return ""
    # Drop paging, line-context flags, and redirections that vary between runs.
    trimmed = _BASH_VOLATILE_SUFFIX_RE.sub("", cmd).strip()
    # Cut at the first pipe or && so we hash the primary command, not the tail.
    for sep in (" | ", " && "):
        idx = trimmed.find(sep)
        if idx != -1:
            trimmed = trimmed[:idx].rstrip()
            break
    return trimmed


# =============================================================================
# Error Classification (reused from learn/scanner.py patterns)
# =============================================================================

_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"No such file or directory|ENOENT|FileNotFoundError|does not exist", re.I),
        "file_not_found",
    ),
    (re.compile(r"ModuleNotFoundError|ImportError|No module named", re.I), "module_not_found"),
    (re.compile(r"command not found", re.I), "command_not_found"),
    (re.compile(r"Permission denied|EACCES|EPERM|auto-denied", re.I), "permission_denied"),
    (re.compile(r"file is too large|too many lines|exceeds.*limit", re.I), "file_too_large"),
    (re.compile(r"SyntaxError|IndentationError", re.I), "syntax_error"),
    (re.compile(r"Traceback \(most recent|Exception:|Error:", re.I), "runtime_error"),
    (re.compile(r"timed? ?out|TimeoutError|deadline exceeded", re.I), "timeout"),
    (re.compile(r"exit code|non-zero|exited with", re.I), "exit_code"),
    (re.compile(r"BUILD FAILED|compilation error|compile error", re.I), "build_failure"),
]


def _classify_error(content: str) -> str | None:
    """Classify error content. Returns category or None if not an error."""
    snippet = content[:2000]
    for pattern, category in _ERROR_PATTERNS:
        if pattern.search(snippet):
            return category
    return None


def _is_error(content: str) -> bool:
    """Quick check if tool output looks like an error."""
    if not content or len(content) < 10:
        return False
    return _classify_error(content) is not None


# =============================================================================
# Tool Call Extractors
# =============================================================================

# Extract command from Bash tool calls
_COMMAND_RE = re.compile(r"^(?:source\s+\S+\s*&&\s*)?(.+)", re.I)

# Extract file paths
_FILE_PATH_RE = re.compile(r"(?:/[\w./-]+(?:\.\w+)?)")

# Extract package/module names from errors
_MODULE_RE = re.compile(r"No module named ['\"]?(\w[\w.]*)['\"]?")
_COMMAND_NF_RE = re.compile(r"(\w[\w-]*): command not found")


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance. Pure Python, no deps.

    Bounded use only — callers should keep input sizes reasonable
    (basenames, command strings) to avoid O(n*m) blowups.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        curr = [j] + [0] * len(a)
        for i, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[-1]


def _paths_related_as_typo(failed: str, success: str) -> bool:
    """Heuristic: are these two file paths plausibly the same target?

    Two paths are "related as typo recovery" if their basenames are
    identical or close in edit distance. Different basenames in the same
    directory (e.g. `state.rs` vs `lib.rs`) are NOT related — the matcher
    must reject them, otherwise unrelated reads get paired into bogus
    "File X does not exist, use Y" rules.
    """
    if not failed or not success or failed == success:
        return False
    a = failed.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    b = success.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not a or not b:
        return False
    if a == b:
        return True
    threshold = max(2, max(len(a), len(b)) // 3)
    return _levenshtein(a, b) <= threshold


# Tokens that occur in many unrelated commands and don't, by themselves,
# suggest two commands are related retries.
_COMMAND_NOISE_TOKENS = frozenset(
    {
        "head",
        "tail",
        "cat",
        "grep",
        "awk",
        "sed",
        "sort",
        "uniq",
        "wc",
        "xargs",
        "find",
    }
)


def _bash_first_binary(cmd: str) -> str | None:
    """Return the first binary name in a Bash command, or None.

    Strips a leading `source <venv> && ` prefix and skips over `VAR=value`
    environment-variable assignments before the binary. Used to gate
    command-recovery pairing: if two commands don't share a binary, they
    are not retries of each other.
    """
    s = cmd.strip()
    m = re.match(r"source\s+\S+\s*&&\s*(.*)", s, re.I)
    if m:
        s = m.group(1)
    for tok in s.split():
        if "=" in tok and tok.split("=", 1)[0].replace("_", "").isalnum():
            continue
        return tok
    return None


def _bash_binaries_match(a: str, b: str) -> bool:
    """Treat two binaries as 'the same tool' for recovery purposes.

    Equal strings, basename equality (`ruff` vs `.venv/bin/ruff`), and
    short prefix-style versions (`python` vs `python3`) all qualify.
    Different tools (`grep` vs `find`) do not.
    """
    if a == b:
        return True
    a_base = a.rsplit("/", 1)[-1]
    b_base = b.rsplit("/", 1)[-1]
    if a_base == b_base:
        return True
    if (a_base.startswith(b_base) or b_base.startswith(a_base)) and _levenshtein(
        a_base, b_base
    ) <= 2:
        return True
    return False


def _commands_related_as_retry(failed: str, success: str) -> bool:
    """Heuristic: is `success` plausibly a corrected retry of `failed`?

    Requires the same binary AND either:
      - low normalized edit distance (≤40% of max length), OR
      - at least one shared substantive token (length ≥ 5, not a flag,
        not a generic shell verb).

    The bar is conservative: noise like `grep <pattern A> <file A>` paired
    with `grep <pattern B> <file B>` shares the `grep` binary but no real
    arguments, and gets rejected. Genuine retries (extra flag, single
    arg edit) pass via the edit-distance path.
    """
    if not failed or not success or failed == success:
        return False
    bin_a = _bash_first_binary(failed)
    bin_b = _bash_first_binary(success)
    if not bin_a or not bin_b or not _bash_binaries_match(bin_a, bin_b):
        return False

    max_len = max(len(failed), len(success))
    if max_len > 0 and _levenshtein(failed, success) / max_len <= 0.40:
        return True

    def _substantive(cmd: str, binary: str) -> set[str]:
        out: set[str] = set()
        for tok in cmd.split():
            if len(tok) < 5 or tok.startswith("-") or tok == binary:
                continue
            if tok.lower() in _COMMAND_NOISE_TOKENS:
                continue
            out.add(tok)
        return out

    return bool(_substantive(failed, bin_a) & _substantive(success, bin_b))


_FILE_X_DOES_NOT_EXIST_RE = re.compile(
    r"^File `([^`]+)` does not exist\. The correct path is `([^`]+)`\.$"
)


def _drop_contradictions(patterns: list[ExtractedPattern]) -> list[ExtractedPattern]:
    """Remove A→B and B→A pairs from error_recovery patterns.

    When the matcher emits a "File X does not exist, use Y" rule and the
    inverse "File Y does not exist, use X" rule, both are likely the
    result of opposite-direction typos rather than a stable truth. Drop
    both rather than persisting contradictory advice.
    """
    forward: dict[tuple[str, str], int] = {}
    for idx, p in enumerate(patterns):
        if p.category != PatternCategory.ERROR_RECOVERY:
            continue
        m = _FILE_X_DOES_NOT_EXIST_RE.match(p.content)
        if not m:
            continue
        forward[(m.group(1), m.group(2))] = idx

    drop: set[int] = set()
    for (a, b), idx_ab in forward.items():
        idx_ba = forward.get((b, a))
        if idx_ba is not None:
            drop.add(idx_ab)
            drop.add(idx_ba)

    if not drop:
        return patterns
    return [p for i, p in enumerate(patterns) if i not in drop]


# =============================================================================
# Traffic Learner
# =============================================================================


class TrafficLearner:
    """Extracts learnable patterns from live proxy traffic.

    Operates entirely on rule-based heuristics — no LLM calls.
    Designed to be called from the proxy request/response path
    with minimal overhead (async, non-blocking).
    """

    def __init__(
        self,
        backend: LocalBackend | None = None,
        user_id: str = "default",
        agent_type: str = "unknown",
        max_history: int = 20,
        dedup_window: int = 100,
        min_evidence: int = 5,
    ) -> None:
        """Initialize the traffic learner.

        Args:
            backend: Memory backend to save patterns to. If None, patterns
                are accumulated but not persisted until a backend is set.
            user_id: Default user ID for saved memories.
            agent_type: Which coding agent is being wrapped (claude, codex, gemini, etc.).
                Used to determine the correct output file for flushing patterns.
            max_history: Number of recent tool calls to keep for pattern matching.
            dedup_window: Number of recent pattern hashes to track for dedup.
            min_evidence: Minimum times a pattern must be seen before saving.
        """
        self._backend = backend
        self._user_id = user_id
        self.agent_type = agent_type
        self._max_history = max_history
        self._min_evidence = min_evidence

        # Recent tool call history for error→recovery matching
        self._tool_history: list[dict[str, Any]] = []

        # Pattern accumulator: hash → (pattern, count)
        self._pattern_counts: dict[str, tuple[ExtractedPattern, int]] = {}

        # Dedup: hashes of patterns already saved to DB
        self._saved_hashes: set[str] = set()
        # content_hash → memory.id for persisted rows. Lets re-sightings
        # bump the existing row's evidence_count instead of creating a
        # duplicate row.
        self._persisted_ids: dict[str, str] = {}
        self._dedup_window = dedup_window

        # Stats
        self._patterns_extracted = 0
        self._patterns_saved = 0
        self._requests_processed = 0

        # Background save queue
        self._save_queue: asyncio.Queue[ExtractedPattern] = asyncio.Queue(maxsize=100)
        self._save_task: asyncio.Task[None] | None = None
        self._stopping = False

        # Dirty-flag debounced flush to CLAUDE.md / MEMORY.md. Set whenever
        # a pattern is accumulated; checked by _flush_worker.
        self._flush_dirty = False
        self._last_flush_at = 0.0
        self._flush_task: asyncio.Task[None] | None = None

        # Cached project roots discovered via the learn plugin registry.
        # Populated lazily in flush_to_file.
        self._project_roots_cache: list[ProjectInfo] | None = None

    # =========================================================================
    # Public API
    # =========================================================================

    def set_backend(self, backend: LocalBackend) -> None:
        """Set or update the memory backend."""
        self._backend = backend

    async def start(self) -> None:
        """Start the background save worker and flush worker."""
        # Hydrate persisted dedup state before workers spin up so cross-session
        # re-sightings bump existing rows instead of creating duplicates.
        await self._hydrate_persisted_state()
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._save_worker())
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_worker())

    async def stop(self) -> None:
        """Stop the background workers, drain the save queue, final flush."""
        self._stopping = True

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Drain any remaining patterns in the queue before cancelling
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

        # Drain any patterns left in the queue (worker may have been cancelled mid-flight)
        while not self._save_queue.empty():
            try:
                pattern = self._save_queue.get_nowait()
                if self._backend is not None:
                    await self._backend.save_memory(
                        content=pattern.content,
                        user_id=self._user_id,
                        importance=pattern.importance,
                        metadata={
                            "source": "traffic_learner",
                            "category": pattern.category.value,
                            "evidence_count": pattern.evidence_count,
                            **pattern.metadata,
                        },
                    )
                    self._patterns_saved += 1
            except Exception:
                break

        # Final flush on shutdown — bypass debounce.
        await self.flush_to_file()

    async def _flush_worker(self) -> None:
        """Background worker: call flush_to_file when dirty, rate-limited."""
        while True:
            try:
                await asyncio.sleep(2.0)
                if not self._flush_dirty:
                    continue
                if time.monotonic() - self._last_flush_at < FLUSH_DEBOUNCE_SECONDS:
                    continue
                # Reset before flushing so patterns accumulated during the
                # flush still trigger a follow-up.
                self._flush_dirty = False
                self._last_flush_at = time.monotonic()
                await self.flush_to_file()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Traffic learner flush worker iteration failed: %s", e)

    async def flush_to_file(self) -> None:
        """Flush patterns (persisted + in-memory) to agent-native context files.

        Buckets patterns by project via longest-matching file path in content
        or entity_refs, routes by category to CLAUDE.md vs MEMORY.md, and
        delegates the actual write to the learn plugin writer.

        Un-anchored patterns (no absolute path in content) are dropped in v1.
        """
        try:
            from headroom.learn.registry import auto_detect_plugins, get_plugin
        except Exception as e:
            logger.debug("Traffic learner flush: learn package unavailable (%s)", e)
            return

        # Resolve plugin: explicit agent_type wins, else first detected plugin.
        try:
            if self.agent_type and self.agent_type != "unknown":
                plugin = get_plugin(self.agent_type)
            else:
                detected = auto_detect_plugins()
                if not detected:
                    logger.debug("Traffic learner flush: no agent plugins detected")
                    return
                plugin = detected[0]
        except KeyError:
            logger.debug("No learn plugin for agent_type=%s", self.agent_type)
            return

        # Gather patterns: persisted rows + in-memory accumulator, deduped.
        patterns = self._collect_all_patterns()
        if not patterns:
            return

        # Evidence gate: require self._min_evidence corroborations to flush,
        # including at shutdown. One-shot singletons are noise, not signal.
        patterns = [p for p in patterns if p.evidence_count >= self._min_evidence]
        if not patterns:
            return

        # Drop A→B / B→A contradictions among error_recovery patterns.
        # Both directions appearing with enough evidence usually means
        # opposite-direction typos in different sessions, not stable advice.
        patterns = _drop_contradictions(patterns)
        if not patterns:
            return

        # Bucket patterns by project.
        if self._project_roots_cache is None:
            try:
                self._project_roots_cache = plugin.discover_projects()
            except Exception as e:
                logger.warning("discover_projects failed: %s", e)
                self._project_roots_cache = []

        roots = self._project_roots_cache
        if not roots:
            logger.debug("Traffic learner flush: no projects discovered, skipping")
            return

        by_project: dict[Path, list[ExtractedPattern]] = {}
        unanchored = 0
        for p in patterns:
            proj = _project_for_pattern(p, roots)
            if proj is None:
                unanchored += 1
                continue
            by_project.setdefault(proj.project_path, []).append(p)

        if unanchored:
            logger.debug("Traffic learner flush: dropped %d un-anchored pattern(s)", unanchored)

        writer = plugin.create_writer()
        project_by_path = {p.project_path: p for p in roots}

        for project_path, proj_patterns in by_project.items():
            project = project_by_path[project_path]
            recommendations = _patterns_to_recommendations(proj_patterns)
            if not recommendations:
                continue
            try:
                result = writer.write(recommendations, project, dry_run=False)
                if result.files_written:
                    logger.info(
                        "Traffic learner flushed %d pattern(s) to %s",
                        len(proj_patterns),
                        ", ".join(str(f) for f in result.files_written),
                    )
            except Exception as e:
                logger.warning("Traffic learner write failed for %s: %s", project_path, e)

    def _collect_all_patterns(self) -> list[ExtractedPattern]:
        """Merge persisted (memory.db) + in-memory patterns, deduped by content.

        Evidence counts are summed across duplicates.
        """
        by_hash: dict[str, ExtractedPattern] = {}
        now = datetime.now(timezone.utc)

        # Persisted rows from memory.db
        db_path = _resolve_backend_db_path(self._backend)
        if db_path is not None and db_path.exists():
            try:
                persisted = _load_persisted_patterns_from_sqlite(db_path)
            except Exception as e:
                logger.debug("Reading persisted traffic patterns failed: %s", e)
                persisted = []
            for p in persisted:
                if p.content_hash in by_hash:
                    by_hash[p.content_hash].evidence_count += p.evidence_count
                else:
                    by_hash[p.content_hash] = p

        # In-memory accumulator (patterns not yet persisted). Re-sightings in
        # this session bump last_seen_at to "now" on top of the persisted
        # timestamp so recency ranking reflects live activity.
        for pattern, count in self._pattern_counts.values():
            h = pattern.content_hash
            if h in by_hash:
                existing = by_hash[h]
                existing.evidence_count += count
                existing.last_seen_at = now
            else:
                by_hash[h] = ExtractedPattern(
                    category=pattern.category,
                    content=pattern.content,
                    importance=pattern.importance,
                    evidence_count=count,
                    entity_refs=list(pattern.entity_refs),
                    metadata=dict(pattern.metadata),
                    content_hash=pattern.content_hash,
                    first_seen_at=now,
                    last_seen_at=now,
                )

        return list(by_hash.values())

    def get_learned_patterns(self) -> list[ExtractedPattern]:
        """Return patterns from the in-memory accumulator.

        Retained for backwards compatibility. Reads only the accumulator;
        does not consult persisted rows. Use flush_to_file() for full data.
        """
        return [pattern for pattern, count in self._pattern_counts.values() if count >= 1]

    async def on_tool_result(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        is_error: bool,
        agent_type: str = "unknown",
    ) -> None:
        """Process a tool call result from proxy traffic.

        Called by the proxy after each tool_result block is processed.
        Non-blocking — patterns are queued for async persistence.

        Args:
            tool_name: Name of the tool (Bash, Read, Grep, etc.)
            tool_input: Tool input parameters
            tool_output: Tool output content
            is_error: Whether the tool call failed
            agent_type: Which agent is being proxied
        """
        self._requests_processed += 1

        entry = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output[:2000],  # Cap for memory
            "is_error": is_error,
            "error_category": _classify_error(tool_output) if is_error else None,
            "timestamp": time.time(),
            "agent_type": agent_type,
        }

        # Check for error→recovery pattern BEFORE adding to history
        if not is_error and self._tool_history:
            patterns = self._extract_error_recovery(entry)
            for pattern in patterns:
                await self._accumulate(pattern)

        # Extract environment patterns
        env_patterns = self._extract_environment(entry)
        for pattern in env_patterns:
            await self._accumulate(pattern)

        # Add to history (bounded)
        self._tool_history.append(entry)
        if len(self._tool_history) > self._max_history:
            self._tool_history.pop(0)

    async def on_messages(
        self,
        messages: list[dict[str, Any]],
        agent_type: str = "unknown",
    ) -> None:
        """Process message content for preference/architecture patterns.

        Called with the messages array from a proxy request.
        Extracts patterns from user corrections, assistant decisions, etc.

        Args:
            messages: The messages array from the API request
            agent_type: Which agent is being proxied
        """
        for msg in messages[-3:]:  # Only look at recent messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if not content:
                continue

            if role == "user":
                patterns = self._extract_preferences(content)
                for pattern in patterns:
                    await self._accumulate(pattern)

    def get_stats(self) -> dict[str, Any]:
        """Get learner statistics."""
        return {
            "requests_processed": self._requests_processed,
            "patterns_extracted": self._patterns_extracted,
            "patterns_saved": self._patterns_saved,
            "pending_patterns": len(self._pattern_counts),
            "history_size": len(self._tool_history),
        }

    # =========================================================================
    # Pattern Extraction
    # =========================================================================

    def _extract_error_recovery(self, success_entry: dict[str, Any]) -> list[ExtractedPattern]:
        """Extract error→recovery patterns.

        Looks backward in history for recent errors, then checks if the
        current successful call is a recovery (same tool, different params).
        """
        patterns: list[ExtractedPattern] = []
        tool_name = success_entry["tool_name"]

        # Look at recent history for matching errors
        for i in range(len(self._tool_history) - 1, max(-1, len(self._tool_history) - 6), -1):
            prev = self._tool_history[i]
            if not prev["is_error"]:
                continue

            # Same tool type — likely a retry with corrected params
            if prev["tool_name"] == tool_name:
                pattern = self._build_recovery_pattern(prev, success_entry)
                if pattern:
                    patterns.append(pattern)
                break  # Only match the most recent error

            # Bash → Bash with different command (common for env issues)
            if prev["tool_name"] == "Bash" and tool_name == "Bash":
                pattern = self._build_command_recovery(prev, success_entry)
                if pattern:
                    patterns.append(pattern)
                break

        return patterns

    def _build_recovery_pattern(
        self,
        error_entry: dict[str, Any],
        success_entry: dict[str, Any],
    ) -> ExtractedPattern | None:
        """Build a recovery pattern from an error→success pair."""
        tool = error_entry["tool_name"]
        error_cat = error_entry.get("error_category", "unknown")

        if tool == "Bash":
            return self._build_command_recovery(error_entry, success_entry)
        elif tool == "Read":
            error_path = error_entry["input"].get("file_path", "")
            success_path = success_entry["input"].get("file_path", "")
            # Reject pairs whose basenames don't look like typos of each
            # other — different files in the same directory are unrelated
            # reads, not a recovery, and emitting a rule is actively wrong.
            if not _paths_related_as_typo(error_path, success_path):
                return None
            content = f"File `{error_path}` does not exist. The correct path is `{success_path}`."
            return ExtractedPattern(
                category=PatternCategory.ERROR_RECOVERY,
                content=content,
                importance=0.7,
                entity_refs=[success_path],
                metadata={
                    "error_category": error_cat,
                    "tool": "Read",
                    "error_path": error_path,
                    "success_path": success_path,
                },
            )
        elif tool in ("Grep", "Glob"):
            error_pattern = error_entry["input"].get("pattern", "")
            success_pattern = success_entry["input"].get("pattern", "")
            if error_pattern != success_pattern:
                content = (
                    f"Search pattern `{error_pattern}` found no results. "
                    f"Use `{success_pattern}` instead."
                )
                return ExtractedPattern(
                    category=PatternCategory.ERROR_RECOVERY,
                    content=content,
                    importance=0.5,
                )
        return None

    def _build_command_recovery(
        self,
        error_entry: dict[str, Any],
        success_entry: dict[str, Any],
    ) -> ExtractedPattern | None:
        """Build a command recovery pattern from Bash error→success."""
        failed_cmd = error_entry["input"].get("command", "")
        success_cmd = success_entry["input"].get("command", "")
        error_cat = error_entry.get("error_category", "unknown")

        if not failed_cmd or not success_cmd or failed_cmd == success_cmd:
            return None
        # Require the two commands to look like the same operation retried.
        # Without this, any failed Bash followed by any Bash success in the
        # last 5 calls becomes a "use Y instead of X" rule, even when X and
        # Y are unrelated (e.g. two grep calls with different needles and
        # different files).
        if not _commands_related_as_retry(failed_cmd, success_cmd):
            return None

        # Determine importance based on error category
        importance = 0.7
        if error_cat == "command_not_found":
            importance = 0.85  # Environment setup is high-value
        elif error_cat == "module_not_found":
            importance = 0.8

        # Truncate long commands
        failed_short = failed_cmd[:200]
        success_short = success_cmd[:200]

        content = f"Command `{failed_short}` fails ({error_cat}). Use `{success_short}` instead."

        # Extract entity references
        entities: list[str] = []
        module_match = _MODULE_RE.search(error_entry["output"])
        if module_match:
            entities.append(module_match.group(1))
        cmd_match = _COMMAND_NF_RE.search(error_entry["output"])
        if cmd_match:
            entities.append(cmd_match.group(1))

        return ExtractedPattern(
            category=PatternCategory.ERROR_RECOVERY,
            content=content,
            importance=importance,
            entity_refs=entities,
            metadata={
                "error_category": error_cat,
                "tool": "Bash",
                "failed_cmd": failed_short,
                "success_cmd": success_short,
            },
        )

    def _extract_environment(self, entry: dict[str, Any]) -> list[ExtractedPattern]:
        """Extract environment facts from tool calls."""
        patterns: list[ExtractedPattern] = []

        if entry["tool_name"] != "Bash":
            return patterns

        cmd = entry["input"].get("command", "")
        output = entry["output"]

        # Successful commands reveal working environment patterns
        if not entry["is_error"]:
            # Python/venv activation patterns
            if "activate" in cmd and "source" in cmd:
                # Extract the venv path
                venv_match = re.search(r"source\s+(\S+/activate)", cmd)
                if venv_match:
                    venv_path = venv_match.group(1)
                    patterns.append(
                        ExtractedPattern(
                            category=PatternCategory.ENVIRONMENT,
                            content=f"Python virtual environment: `source {venv_path}` before running Python tools.",
                            importance=0.8,
                            entity_refs=[venv_path],
                            metadata={"type": "venv_activation"},
                        )
                    )

            # Detect working test commands
            if "pytest" in cmd and "PASSED" in output:
                patterns.append(
                    ExtractedPattern(
                        category=PatternCategory.ENVIRONMENT,
                        content=f"Working test command: `{cmd[:200]}`",
                        importance=0.6,
                        metadata={"type": "test_command"},
                    )
                )

        return patterns

    # ---------------------------------------------------------------
    # Preference detection (GH #464)
    # ---------------------------------------------------------------
    # The detector is regex-free on purpose. The previous regex-based
    # implementation matched scaffolding ("don't mention this
    # reminder…" injected by Claude Code's system-reminder blocks) and
    # produced mid-sentence truncations because ``.{10,100}`` captured
    # an arbitrary 100-char window with no boundary awareness. A
    # tokenized scanner is easier to reason about, doesn't suffer
    # catastrophic-backtracking edge cases, and lets us layer
    # boundary rules (sentence terminator / end-of-input / max-length)
    # without nesting more pattern syntax.

    # Each entry is a sequence of lowercase tokens that must appear
    # in order with whitespace / single-comma separation. ``max_chars``
    # caps how much content we'll capture after the last trigger
    # token. The "instead" trigger gets a tighter cap because in
    # practice its tail tends to be shorter and we want to be less
    # forgiving of long rambles after it.
    _PREFERENCE_TRIGGERS: ClassVar[tuple[tuple[tuple[str, ...], int], ...]] = (
        (("don't",), 98),
        (("dont",), 98),
        (("do", "not"), 98),
        (("stop",), 98),
        (("never",), 98),
        (("avoid",), 98),
        (("no", "use"), 98),
        (("no", "try"), 98),
        (("no", "do"), 98),
        (("instead",), 78),
    )

    # Characters that mark the end of the captured preference.
    _SENTENCE_TERMINATORS: ClassVar[frozenset[str]] = frozenset(".!?\n")

    # Characters allowed between the trigger and the start of the
    # capture (e.g. the comma in "No, use httpx").
    _PRE_CAPTURE_PUNCT: ClassVar[frozenset[str]] = frozenset(",;:")

    # Characters stripped from individual tokens before trigger
    # matching ("don't," → "don't"; "stop." → "stop"). Whitespace is
    # handled separately by the tokenizer.
    _TOKEN_STRIP_CHARS: ClassVar[str] = ",.;:!?\"'()[]{}"

    def _extract_preferences(self, user_text: str) -> list[ExtractedPattern]:
        """Extract preference signals from user messages.

        Defends against GH #464 noise sources:

        * Claude Code (and other agent harnesses) inject
          ``<system-reminder>…</system-reminder>`` blocks into user-role
          message bodies. Their content is *not* user-stated
          preferences ("don't mention this reminder", "use colgrep
          instead of Grep") but the old correction regexes matched
          them. We strip those blocks first so reminders never feed
          the learner.
        * The capture used to be a fixed-length window which produced
          mid-sentence truncations. The token-based scanner below
          ends each capture at a sentence terminator OR at
          end-of-input, and rejects anything that would require
          truncation past ``max_chars``.
        """

        cleaned = self._strip_system_reminders(user_text)[:500]
        correction = self._find_correction(cleaned)
        if correction is None:
            return []

        return [
            ExtractedPattern(
                category=PatternCategory.PREFERENCE,
                content=f"User preference: {correction}",
                importance=0.75,
                metadata={"type": "correction", "source_text": cleaned[:200]},
            )
        ]

    @classmethod
    def _find_correction(cls, text: str) -> str | None:
        """Return the captured preference content or ``None``.

        Walks the input once, tokenising on whitespace. At every
        token position we try each trigger sequence in priority
        order; the first satisfied trigger wins.
        """

        tokens = cls._tokenize(text)
        if not tokens:
            return None

        for trigger_idx in range(len(tokens)):
            for sequence, max_chars in cls._PREFERENCE_TRIGGERS:
                if not cls._matches_sequence(tokens, trigger_idx, sequence):
                    continue
                last_token_end = tokens[trigger_idx + len(sequence) - 1][2]
                captured = cls._capture_after(text, last_token_end, max_chars)
                if captured is None:
                    continue
                return captured
        return None

    @classmethod
    def _matches_sequence(
        cls,
        tokens: list[tuple[str, int, int]],
        start: int,
        sequence: tuple[str, ...],
    ) -> bool:
        if start + len(sequence) > len(tokens):
            return False
        for offset, expected in enumerate(sequence):
            actual_token = tokens[start + offset][0]
            normalised = actual_token.strip(cls._TOKEN_STRIP_CHARS)
            if normalised != expected:
                return False
        return True

    @classmethod
    def _capture_after(
        cls,
        text: str,
        capture_after_pos: int,
        max_chars: int,
    ) -> str | None:
        """Capture up to ``max_chars`` of content starting at the first
        non-whitespace, non-pre-punct character after ``capture_after_pos``.

        Returns ``None`` when the capture would have to truncate past
        ``max_chars`` without hitting a sentence terminator or
        end-of-input. Returns ``None`` for captures shorter than 10
        chars (those are noise — likely a stray trigger word with no
        real correction following it).
        """

        n = len(text)
        cap_start = capture_after_pos
        while cap_start < n and (
            text[cap_start].isspace() or text[cap_start] in cls._PRE_CAPTURE_PUNCT
        ):
            cap_start += 1

        cap_end = cap_start
        while (
            cap_end < n
            and (cap_end - cap_start) < max_chars
            and text[cap_end] not in cls._SENTENCE_TERMINATORS
        ):
            cap_end += 1

        length = cap_end - cap_start
        if length < 10:
            return None

        # If we hit ``max_chars`` without finding a terminator and the
        # text continues past us, this is a rambling fragment — reject.
        if length >= max_chars and cap_end < n and text[cap_end] not in cls._SENTENCE_TERMINATORS:
            return None

        captured = text[cap_start:cap_end].strip()
        captured = captured.rstrip("".join(cls._SENTENCE_TERMINATORS)).strip()
        return captured or None

    @staticmethod
    def _tokenize(text: str) -> list[tuple[str, int, int]]:
        """Whitespace-split tokenizer.

        Returns ``[(lower_token, start, end), …]``. Positions are byte
        offsets into the original string so callers can resume
        scanning from the end of a token. Tokens are lowercased once,
        up front, so trigger comparisons don't have to call
        ``.lower()`` per match.
        """

        out: list[tuple[str, int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            while i < n and text[i].isspace():
                i += 1
            start = i
            while i < n and not text[i].isspace():
                i += 1
            if i > start:
                out.append((text[start:i].lower(), start, i))
        return out

    @staticmethod
    def _strip_system_reminders(text: str) -> str:
        """Remove ``<system-reminder>…</system-reminder>`` blocks from text.

        Uses a literal scan (``str.find``) rather than a regex so the
        matcher cannot accidentally pick up unrelated ``<*>``-shaped
        content. Unclosed reminders (missing ``</system-reminder>``)
        are dropped to end-of-string — the agent harness writes
        balanced tags, and an unbalanced one is corrupt input we
        shouldn't index. Matching is case-insensitive on the tag name.
        """
        if not text or "<" not in text:
            return text

        open_tag = "<system-reminder"
        close_tag = "</system-reminder>"

        lower = text.lower()
        out: list[str] = []
        cursor = 0
        n = len(text)
        while cursor < n:
            start = lower.find(open_tag, cursor)
            if start < 0:
                out.append(text[cursor:])
                break
            out.append(text[cursor:start])
            tag_end = text.find(">", start)
            if tag_end < 0:
                break
            close_start = lower.find(close_tag, tag_end + 1)
            if close_start < 0:
                break
            cursor = close_start + len(close_tag)
        return "".join(out)

    # =========================================================================
    # Pattern Accumulation & Persistence
    # =========================================================================

    async def _accumulate(self, pattern: ExtractedPattern) -> None:
        """Accumulate a pattern, saving when evidence threshold is met."""
        self._patterns_extracted += 1
        self._flush_dirty = True
        h = pattern.content_hash

        # Already saved — bump the persisted row's evidence_count rather
        # than creating a duplicate.
        if h in self._saved_hashes:
            memory_id = self._persisted_ids.get(h)
            if memory_id is not None:
                await self._bump_persisted_evidence(memory_id)
            return

        # Accumulate evidence
        if h in self._pattern_counts:
            existing, count = self._pattern_counts[h]
            count += 1
            self._pattern_counts[h] = (existing, count)
        else:
            self._pattern_counts[h] = (pattern, 1)
            return  # First sighting — wait for more evidence

        # Check if evidence threshold met
        _, count = self._pattern_counts[h]
        if count >= self._min_evidence:
            # Ready to save
            del self._pattern_counts[h]
            self._saved_hashes.add(h)
            # Trim saved hashes to prevent unbounded growth
            if len(self._saved_hashes) > self._dedup_window:
                # Remove oldest (arbitrary, set is unordered, but prevents growth)
                self._saved_hashes.pop()

            # Persist the real accumulated count, not the dataclass default.
            pattern.evidence_count = count
            try:
                self._save_queue.put_nowait(pattern)
            except asyncio.QueueFull:
                logger.debug("Traffic learner save queue full, dropping pattern")

    async def _save_worker(self) -> None:
        """Background worker that persists patterns to memory backend."""
        while True:
            try:
                pattern = await self._save_queue.get()
                if self._backend is None:
                    continue

                now_iso = datetime.now(timezone.utc).isoformat()
                memory = await self._backend.save_memory(
                    content=pattern.content,
                    user_id=self._user_id,
                    importance=pattern.importance,
                    metadata={
                        "source": "traffic_learner",
                        "category": pattern.category.value,
                        "evidence_count": pattern.evidence_count,
                        "first_seen_at": now_iso,
                        "last_seen_at": now_iso,
                        **pattern.metadata,
                    },
                )
                self._patterns_saved += 1
                # Track id so future re-sightings bump this row.
                memory_id = getattr(memory, "id", None)
                if memory_id is not None:
                    self._persisted_ids[pattern.content_hash] = memory_id
                logger.debug(f"Traffic learner saved pattern: {pattern.content[:80]}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Traffic learner save failed: {e}")

    async def _hydrate_persisted_state(self) -> None:
        """Load existing traffic_learner rows into _saved_hashes / _persisted_ids.

        Runs once at start() so re-sightings across process restarts bump the
        existing row rather than inserting a duplicate. Read-only; if the DB
        is absent or unreadable we simply skip.
        """
        db_path = _resolve_backend_db_path(self._backend)
        if db_path is None or not db_path.exists():
            return

        def _read() -> list[tuple[str, str, str]]:
            uri = f"file:{db_path}?mode=ro"
            try:
                conn = sqlite3.connect(uri, uri=True)
            except sqlite3.OperationalError:
                return []
            try:
                rows = conn.execute(
                    "SELECT id, content, metadata FROM memories "
                    "WHERE json_extract(metadata, '$.source') = 'traffic_learner'"
                ).fetchall()
            except sqlite3.DatabaseError:
                return []
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            return [(row[0], row[1] or "", row[2] or "{}") for row in rows]

        try:
            rows = await asyncio.to_thread(_read)
        except Exception as e:
            logger.debug("Traffic learner hydrate failed: %s", e)
            return

        for memory_id, content, metadata_json in rows:
            if not content:
                continue
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except json.JSONDecodeError:
                metadata = {}
            category_value = metadata.get("category")
            try:
                category = PatternCategory(category_value) if category_value else None
            except ValueError:
                category = None
            if category is None:
                # Legacy row without category — fall back to literal hash.
                key = content
            else:
                key = _normalize_hash_key(category, content, metadata)
            h = hashlib.sha256(key.encode()).hexdigest()[:16]
            self._saved_hashes.add(h)
            # If multiple rows share the same content (legacy duplicates),
            # last-wins — we only need one id to target the bump.
            self._persisted_ids[h] = memory_id

    async def _bump_persisted_evidence(self, memory_id: str) -> None:
        """Atomically increment a persisted row's metadata.evidence_count."""
        db_path = _resolve_backend_db_path(self._backend)
        if db_path is None or not db_path.exists():
            return

        now_iso = datetime.now(timezone.utc).isoformat()

        def _bump() -> None:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    "UPDATE memories SET metadata = json_set("
                    "metadata, '$.evidence_count', "
                    "COALESCE(json_extract(metadata, '$.evidence_count'), 0) + 1, "
                    "'$.last_seen_at', ?"
                    ") WHERE id = ?",
                    (now_iso, memory_id),
                )
                conn.commit()
            finally:
                conn.close()

        try:
            await asyncio.to_thread(_bump)
        except Exception as e:
            logger.debug("Traffic learner evidence bump failed for %s: %s", memory_id, e)

    # =========================================================================
    # Convenience: Extract from Anthropic messages format
    # =========================================================================

    def extract_tool_results_from_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract tool_result blocks from Anthropic-format messages.

        Useful for processing the messages array to find tool calls and
        their results for pattern extraction.

        Returns list of dicts with: tool_name, input, output, is_error
        """
        results: list[dict[str, Any]] = []

        # Build tool_use_id → tool_use mapping
        tool_uses: dict[str, dict[str, Any]] = {}
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_uses[block.get("id", "")] = block

        # Find tool_results and match with tool_uses
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue

                tool_use_id = block.get("tool_use_id", "")
                tool_use = tool_uses.get(tool_use_id, {})

                # Extract output text
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    result_content = " ".join(
                        b.get("text", "")
                        for b in result_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )

                results.append(
                    {
                        "tool_name": tool_use.get("name", "unknown"),
                        "input": tool_use.get("input", {}),
                        "output": str(result_content),
                        "is_error": block.get("is_error", False) or _is_error(str(result_content)),
                    }
                )

        return results


# =============================================================================
# Module helpers: project routing, memory.db loading, recommendation build
# =============================================================================

# Category → file routing. Stable project facts go to CLAUDE.md; evolving
# preferences and error recovery tips go to MEMORY.md (which the user's
# auto-memory system already owns).
_CATEGORY_TO_TARGET: dict[PatternCategory, str] = {
    PatternCategory.ENVIRONMENT: "context_file",
    PatternCategory.ARCHITECTURE: "context_file",
    PatternCategory.PREFERENCE: "memory_file",
    PatternCategory.ERROR_RECOVERY: "memory_file",
}

_CATEGORY_SECTION_TITLE: dict[PatternCategory, str] = {
    PatternCategory.ENVIRONMENT: "Learned: environment",
    PatternCategory.ARCHITECTURE: "Learned: architecture",
    PatternCategory.PREFERENCE: "Learned: preference",
    PatternCategory.ERROR_RECOVERY: "Learned: error recovery",
}


def _project_for_pattern(pattern: ExtractedPattern, roots: list[ProjectInfo]) -> ProjectInfo | None:
    """Return the project whose root most specifically matches this pattern.

    We look for absolute paths in the pattern's content and entity_refs, then
    pick the longest project root that prefixes any of those paths. Returns
    None if the pattern mentions no paths under a known project.
    """
    if not roots:
        return None

    # Collect candidate absolute paths from content and entity_refs
    candidates: list[str] = []
    for match in _ABS_PATH_RE.findall(pattern.content or ""):
        candidates.append(match)
    for ref in pattern.entity_refs or []:
        if ref and (ref.startswith("/") or (len(ref) > 2 and ref[1] == ":")):
            candidates.append(ref)

    if not candidates:
        return None

    # Longest root first — most specific wins
    roots_sorted = sorted(roots, key=lambda p: len(str(p.project_path)), reverse=True)

    for cand in candidates:
        for root in roots_sorted:
            root_str = str(root.project_path).rstrip("/\\")
            if not root_str:
                continue
            if (
                cand == root_str
                or cand.startswith(root_str + "/")
                or cand.startswith(root_str + "\\")
            ):
                return root
    return None


def _resolve_backend_db_path(backend: Any) -> Path | None:
    """Best-effort lookup of the SQLite path used by the memory backend.

    Returns None if the backend is not a LocalBackend or its config is not
    accessible (e.g. mem0 remote backend).
    """
    if backend is None:
        return None
    cfg = getattr(backend, "_config", None)
    db_path = getattr(cfg, "db_path", None) if cfg is not None else None
    if not db_path:
        return None
    return Path(db_path)


def _load_persisted_patterns_from_sqlite(db_path: Path) -> list[ExtractedPattern]:
    """Read traffic_learner rows from memory.db, dedupe, return patterns.

    Uses a direct read-only SQLite connection — we don't go through the
    backend's vector search because we want all rows, not semantically
    similar ones, and the backend doesn't expose a "list by source" query.
    """
    uri = f"file:{db_path}?mode=ro"
    patterns: dict[str, ExtractedPattern] = {}
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return []
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT content, metadata, entity_refs, importance, created_at "
            "FROM memories "
            "WHERE json_extract(metadata, '$.source') = 'traffic_learner'"
        ).fetchall()
    except sqlite3.DatabaseError:
        conn.close()
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    for row in rows:
        content = row["content"] or ""
        if not content:
            continue
        try:
            meta = json.loads(row["metadata"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        try:
            entity_refs = json.loads(row["entity_refs"] or "[]") or []
        except json.JSONDecodeError:
            entity_refs = []

        cat_str = meta.get("category", "")
        try:
            category = PatternCategory(cat_str)
        except ValueError:
            continue  # Skip rows whose category we don't recognize

        evidence = int(meta.get("evidence_count", 1) or 1)
        try:
            importance = float(row["importance"]) if row["importance"] is not None else 0.5
        except (TypeError, ValueError):
            importance = 0.5

        first_seen = _parse_iso_timestamp(meta.get("first_seen_at")) or _parse_iso_timestamp(
            row["created_at"]
        )
        last_seen = _parse_iso_timestamp(meta.get("last_seen_at")) or first_seen

        key = _normalize_hash_key(category, content, meta)
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        if h in patterns:
            existing = patterns[h]
            existing.evidence_count += evidence
            if importance > existing.importance:
                existing.importance = importance
            if last_seen and (existing.last_seen_at is None or last_seen > existing.last_seen_at):
                existing.last_seen_at = last_seen
            if first_seen and (
                existing.first_seen_at is None or first_seen < existing.first_seen_at
            ):
                existing.first_seen_at = first_seen
        else:
            patterns[h] = ExtractedPattern(
                category=category,
                content=content,
                importance=importance,
                evidence_count=evidence,
                entity_refs=list(entity_refs),
                metadata=meta,
                content_hash=h,
                first_seen_at=first_seen,
                last_seen_at=last_seen,
            )

    return list(patterns.values())


def _parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp stored as TEXT. Returns None on any failure."""
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _patterns_to_recommendations(patterns: list[ExtractedPattern]) -> list:
    """Group patterns by category into one Recommendation per category.

    Returns a list of Recommendation objects ready for ContextWriter.write.
    """
    from headroom.learn.models import Recommendation, RecommendationTarget

    by_category: dict[PatternCategory, list[ExtractedPattern]] = {}
    for p in patterns:
        by_category.setdefault(p.category, []).append(p)

    recs: list[Recommendation] = []
    for category, items in by_category.items():
        target_str = _CATEGORY_TO_TARGET.get(category)
        if target_str is None:
            continue
        target = (
            RecommendationTarget.CONTEXT_FILE
            if target_str == "context_file"
            else RecommendationTarget.MEMORY_FILE
        )
        if category is PatternCategory.ERROR_RECOVERY:
            items = _refine_error_recovery(items)
        else:
            # Sort by evidence_count desc so the most-supported rules appear first.
            items.sort(key=lambda p: p.evidence_count, reverse=True)
        if not items:
            continue
        bullets = "\n".join(f"- {p.content}" for p in items)
        recs.append(
            Recommendation(
                target=target,
                section=_CATEGORY_SECTION_TITLE.get(category, f"Learned: {category.value}"),
                content=bullets,
                confidence=max((p.importance for p in items), default=0.5),
                evidence_count=sum(p.evidence_count for p in items),
            )
        )
    return recs


def _refine_error_recovery(patterns: list[ExtractedPattern]) -> list[ExtractedPattern]:
    """Apply the render-time pipeline for error_recovery patterns.

    Pipeline: hard-floor drop by last_seen_at, re-validate Read success
    paths against the filesystem, collapse ambiguous error_paths into a
    single "search first" hint, rank by recency-weighted evidence, and
    cap the section at _ERROR_RECOVERY_SECTION_CAP bullets.
    """
    now = datetime.now(timezone.utc)

    # 1. Hard floor — drop rows not re-observed in the last N days.
    alive: list[ExtractedPattern] = []
    for p in patterns:
        last_seen = p.last_seen_at or p.first_seen_at
        if last_seen is None:
            # No timestamp — treat as just-seen so it survives one render.
            alive.append(p)
            continue
        age_days = (now - last_seen).total_seconds() / 86400.0
        if age_days <= _ERROR_RECOVERY_HARD_FLOOR_DAYS:
            alive.append(p)

    # 2. Re-validate Read recoveries — drop if success_path no longer exists.
    validated: list[ExtractedPattern] = []
    for p in alive:
        if p.metadata.get("tool") == "Read":
            success_path = p.metadata.get("success_path")
            if success_path:
                try:
                    if not Path(success_path).exists():
                        continue
                except OSError:
                    # Path check failed (permissions, etc.) — keep the row
                    # rather than drop on a transient error.
                    pass
        validated.append(p)

    # 3. Collision-collapse — same error_path with >=2 distinct success_paths
    #    is an ambiguity signal, not N separate lessons. Replace the group
    #    with one synthesized "search first" bullet.
    read_groups: dict[str, list[ExtractedPattern]] = {}
    others: list[ExtractedPattern] = []
    for p in validated:
        if p.metadata.get("tool") == "Read" and p.metadata.get("error_path"):
            read_groups.setdefault(p.metadata["error_path"], []).append(p)
        else:
            others.append(p)

    collapsed: list[ExtractedPattern] = list(others)
    for error_path, group in read_groups.items():
        distinct_targets = {g.metadata.get("success_path") for g in group}
        distinct_targets.discard(None)
        if len(group) >= 2 and len(distinct_targets) >= 2:
            basename = os.path.basename(error_path) or error_path
            synth_content = (
                f"Path `{basename}` has been guessed wrong repeatedly — "
                f"use Glob/Grep to locate before reading."
            )
            max_last_seen = max(
                (g.last_seen_at for g in group if g.last_seen_at),
                default=now,
            )
            collapsed.append(
                ExtractedPattern(
                    category=PatternCategory.ERROR_RECOVERY,
                    content=synth_content,
                    importance=max(g.importance for g in group),
                    evidence_count=sum(g.evidence_count for g in group),
                    metadata={
                        "tool": "Read",
                        "error_path": error_path,
                        "collapsed": True,
                    },
                    last_seen_at=max_last_seen,
                    first_seen_at=min(
                        (g.first_seen_at for g in group if g.first_seen_at),
                        default=max_last_seen,
                    ),
                )
            )
        else:
            collapsed.extend(group)

    # 4. Recency-weighted score.
    def _score(p: ExtractedPattern) -> float:
        last_seen = p.last_seen_at or p.first_seen_at or now
        age_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
        decay = float(0.5 ** (age_days / _ERROR_RECOVERY_HALF_LIFE_DAYS))
        return float(p.evidence_count) * decay

    collapsed.sort(key=_score, reverse=True)

    # 5. Cap the section.
    return collapsed[:_ERROR_RECOVERY_SECTION_CAP]
