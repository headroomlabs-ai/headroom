"""Base class for headroom learn plugins.

Each coding agent (Claude Code, Codex, Gemini, Cursor, etc.) implements
a LearnPlugin that bundles scanning, writing, and detection into one unit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import ProjectInfo, SessionData
from .writer import ContextWriter


class ConversationScanner(ABC):
    """Base class for scanning conversation logs from any agent system.

    Subclasses implement log format parsing for specific tools (Claude Code,
    Cursor, Codex, etc.) and produce normalized ToolCall sequences.
    """

    @abstractmethod
    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects with conversation data."""
        ...

    @abstractmethod
    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        """Scan all sessions for a project, returning normalized tool calls."""
        ...


class LearnPlugin(ABC):
    """A self-contained learn plugin for a single coding agent.

    Bundles identity, detection, scanning, and writer creation.
    Plugins are discovered automatically from headroom.learn.plugins.*
    or via ``headroom.learn_plugin`` entry points for external packages.

    Example::

        class MyAgentPlugin(LearnPlugin, ConversationScanner):
            @property
            def name(self) -> str:
                return "myagent"

            @property
            def display_name(self) -> str:
                return "My Agent"

            def detect(self) -> bool:
                return Path("~/.myagent/sessions").expanduser().exists()

            def discover_projects(self) -> list[ProjectInfo]: ...
            def scan_project(
                self,
                project: ProjectInfo,
                max_workers: int = 1,
                include_subagents: bool = True,
            ) -> list[SessionData]: ...

            def create_writer(self) -> ContextWriter:
                from headroom.learn.writer import GeminiWriter
                return GeminiWriter()  # or a custom writer

        # Module-level instance for auto-discovery
        plugin = MyAgentPlugin()
    """

    # --- Identity ---

    @property
    @abstractmethod
    def name(self) -> str:
        """Short lowercase identifier used in CLI (e.g., 'claude', 'cursor')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g., 'Claude Code', 'Cursor')."""
        ...

    @property
    def description(self) -> str:
        """One-line description for --help output."""
        return f"{self.display_name} coding agent"

    # --- Detection ---

    @abstractmethod
    def detect(self) -> bool:
        """Return True if this agent has data on the current machine.

        Called during auto-detection. Must be cheap (stat checks only, no I/O).
        """
        ...

    # --- Scanning ---

    @abstractmethod
    def discover_projects(self) -> list[ProjectInfo]:
        """Discover all projects with conversation data for this agent."""
        ...

    @abstractmethod
    def scan_project(
        self, project: ProjectInfo, max_workers: int = 1, include_subagents: bool = True
    ) -> list[SessionData]:
        """Scan all sessions for a project, returning normalized data.

        Args:
            project: The project to scan.
            max_workers: Number of threads for parallel file scanning.
                         1 (default) = serial.  >1 = concurrent.
            include_subagents: Also scan nested subagent/workflow transcripts
                         where the agent system writes them (Claude Code).
                         Ignored by agents without a nested transcript layout.
        """
        ...

    # --- Writing ---

    @abstractmethod
    def create_writer(self) -> ContextWriter:
        """Return the appropriate ContextWriter for this agent."""
        ...
