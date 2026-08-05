"""Data models for Headroom Learn — tool-agnostic abstractions.

These models normalize tool call data from ANY agent system (Claude Code, Cursor,
Codex, custom agents) into a common format that analyzers can work with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

# =============================================================================
# Error Classification
# =============================================================================


class ErrorCategory(str, Enum):
    """Classified error categories for tool call failures."""

    FILE_NOT_FOUND = "file_not_found"
    MODULE_NOT_FOUND = "module_not_found"
    COMMAND_NOT_FOUND = "command_not_found"
    PERMISSION_DENIED = "permission_denied"
    FILE_TOO_LARGE = "file_too_large"
    IS_DIRECTORY = "is_directory"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    NO_MATCHES = "no_matches"  # Grep/Glob found nothing
    USER_REJECTED = "user_rejected"
    SIBLING_ERROR = "sibling_error"  # Cascade from parallel call failure
    EXIT_CODE = "exit_code"
    CONNECTION_ERROR = "connection_error"
    BUILD_FAILURE = "build_failure"
    UNKNOWN = "unknown"


# =============================================================================
# Core Data Models (Tool-Agnostic)
# =============================================================================


@dataclass
class ToolCall:
    """A single tool call and its result — normalized from any agent system.

    This is the fundamental unit of analysis. Scanners produce these,
    analyzers consume them.
    """

    name: str  # Tool name ("Bash", "Read", "file_search", etc.)
    tool_call_id: str  # Unique ID linking call to result
    input_data: dict  # Tool input parameters
    output: str  # Result content (may be error message)
    is_error: bool  # Whether the call failed
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    msg_index: int = 0  # Position in conversation
    output_bytes: int = 0  # Size of output

    @property
    def input_summary(self) -> str:
        """Short summary of tool input for display."""
        if self.name in ("Bash", "bash"):
            cmd: str = self.input_data.get("command", "")
            return cmd[:100] + "..." if len(cmd) > 100 else cmd
        if self.name in ("Read", "read"):
            return str(self.input_data.get("file_path", "?"))
        if self.name in ("Grep", "grep"):
            return str(self.input_data.get("pattern", "?"))
        if self.name in ("Glob", "glob"):
            return str(self.input_data.get("pattern", "?"))
        if self.name in ("Edit", "edit", "Write", "write"):
            return str(self.input_data.get("file_path", "?"))
        return str(self.input_data)[:80]


@dataclass
class SessionEvent:
    """Any event in a session — tool calls, user messages, interruptions.

    Provides richer context than ToolCall alone, enabling
    user preference mining and conversation understanding.
    """

    type: str  # "tool_call", "user_message", "interruption", "agent_summary"
    msg_index: int
    timestamp: str | None = None

    # For tool_call type
    tool_call: ToolCall | None = None

    # For user_message type
    text: str = ""

    # For agent_summary type (subagent results)
    agent_id: str = ""
    agent_tool_count: int = 0
    agent_tokens: int = 0
    agent_duration_ms: int = 0
    agent_prompt: str = ""


@dataclass
class SessionData:
    """Normalized data from a single conversation session."""

    session_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    events: list[SessionEvent] = field(default_factory=list)
    timestamp: datetime | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    source: str = "main"  # "main" | "subagent" | "workflow" — where this transcript came from

    @property
    def failure_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.is_error)

    @property
    def failure_rate(self) -> float:
        if not self.tool_calls:
            return 0.0
        return self.failure_count / len(self.tool_calls)


@dataclass
class ProjectInfo:
    """Information about a project discovered by a scanner."""

    name: str  # Human-readable project name
    project_path: Path  # Actual project directory
    data_path: Path  # Where conversation logs are stored
    context_file: Path | None = None  # CLAUDE.md / .cursorrules / AGENTS.md
    memory_file: Path | None = None  # MEMORY.md or equivalent


# =============================================================================
# Analysis Output Models
# =============================================================================


class RecommendationTarget(str, Enum):
    """Where a recommendation should be written."""

    CONTEXT_FILE = "context_file"  # CLAUDE.md, .cursorrules, AGENTS.md
    MEMORY_FILE = "memory_file"  # MEMORY.md or equivalent


@dataclass
class Recommendation:
    """A concrete recommendation to write to a context/memory file."""

    target: RecommendationTarget
    section: str  # Section heading (e.g., "Environment", "Known Large Files")
    content: str  # Markdown content for the section
    confidence: float = 0.0  # 0-1, based on evidence strength
    evidence_count: int = 0  # Number of failures supporting this
    estimated_tokens_saved: int = 0  # Projected savings if recommendation is followed
    # Loop weighting (see headroom.learn.loops): set when this recommendation
    # guards against a detected repeated pattern. Loop guardrails are ranked
    # above one-off rules because their waste scales with repetition.
    is_loop_guardrail: bool = False
    loop_occurrences: int = 0  # Repetitions of the loop this rule guards against


@dataclass
class AnalysisResult:
    """Output of session analysis — stats + recommendations."""

    project: ProjectInfo
    total_sessions: int = 0
    total_calls: int = 0
    total_failures: int = 0
    recommendations: list[Recommendation] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        if not self.total_calls:
            return 0.0
        return self.total_failures / self.total_calls
