"""Base class and shared utilities for agent-native memory writers.

Writers convert Headroom memory entries into agent-specific file formats.
The base class handles token budgeting, deduplication, marker management,
and priority ranking. Subclasses implement format-specific rendering.
"""

from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Marker delimiters for Headroom-managed sections (matches learn/writer.py)
MARKER_START = "<!-- headroom:memory:start -->"
MARKER_END = "<!-- headroom:memory:end -->"
MARKER_PATTERN = re.compile(
    re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
    re.DOTALL,
)


@dataclass
class MemoryEntry:
    """A memory entry to be written to an agent's file.

    Simplified view of HierarchicalMemory's Memory model,
    focused on what writers need.
    """

    content: str
    importance: float = 0.5
    category: str = ""  # error_recovery, environment, preference, architecture
    entity_refs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = 0.0
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: str = ""

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def score(self) -> float:
        """Combined score for ranking: importance × recency × access."""
        age_days = (time.time() - self.created_at) / 86400
        recency = 1.0 / (1.0 + age_days * 0.1)  # Decay over ~10 days
        access_boost = min(1.0, 0.5 + self.access_count * 0.1)
        return self.importance * recency * access_boost


@dataclass
class ExportResult:
    """Result of a memory export operation."""

    files_written: list[Path] = field(default_factory=list)
    content_by_file: dict[str, str] = field(default_factory=dict)  # path → content
    memories_exported: int = 0
    memories_skipped_dedup: int = 0
    memories_skipped_budget: int = 0
    dry_run: bool = True


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


class AgentWriter(ABC):
    """Base class for agent-native memory writers.

    Subclasses implement:
    - format_memories(): render memories in agent-specific format
    - default_paths(): where to write for this agent
    - agent_name: human-readable agent name
    """

    # Subclasses set these
    agent_name: str = "generic"
    default_token_budget: int = 3000  # Max tokens for the memory section

    def __init__(
        self,
        project_path: Path | None = None,
        token_budget: int | None = None,
    ) -> None:
        self._project_path = project_path or Path.cwd()
        self._token_budget = token_budget or self.default_token_budget

    def export(
        self,
        memories: list[MemoryEntry],
        output_path: Path | None = None,
        dry_run: bool = True,
    ) -> ExportResult:
        """Export memories to agent-native format.

        Args:
            memories: Memory entries to export.
            output_path: Override output path (uses default if None).
            dry_run: If True, don't write files.

        Returns:
            ExportResult with files written and stats.
        """
        result = ExportResult(dry_run=dry_run)

        if not memories:
            return result

        # Rank by combined score
        ranked = sorted(memories, key=lambda m: m.score, reverse=True)

        # Deduplicate by content hash
        seen_hashes: set[str] = set()
        unique: list[MemoryEntry] = []
        for m in ranked:
            h = m.content_hash
            if h in seen_hashes:
                result.memories_skipped_dedup += 1
                continue
            seen_hashes.add(h)
            unique.append(m)

        # Apply token budget
        budgeted: list[MemoryEntry] = []
        tokens_used = 0
        for m in unique:
            entry_tokens = _estimate_tokens(m.content) + 10  # overhead
            if tokens_used + entry_tokens > self._token_budget:
                result.memories_skipped_budget += 1
                continue
            tokens_used += entry_tokens
            budgeted.append(m)

        if not budgeted:
            return result

        # Format in agent-specific way
        formatted = self.format_memories(budgeted)

        # Wrap in markers
        section = f"{MARKER_START}\n{formatted}\n{MARKER_END}"

        # Determine output path
        target = output_path or self.default_path()

        # Merge into existing file
        full_content = _merge_section(target, section)

        result.files_written.append(target)
        result.content_by_file[str(target)] = full_content
        result.memories_exported = len(budgeted)

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(full_content, encoding="utf-8")

        return result

    @abstractmethod
    def format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format memories in agent-specific format.

        Args:
            memories: Ranked, deduped, budget-constrained memory entries.

        Returns:
            Formatted string ready to be wrapped in markers and written.
        """
        ...

    @abstractmethod
    def default_path(self) -> Path:
        """Default output path for this agent."""
        ...


def _merge_section(file_path: Path, section: str) -> str:
    """Merge a marker-delimited section into an existing file."""
    if file_path.exists():
        existing = file_path.read_text(encoding="utf-8")
        if MARKER_START in existing:
            return MARKER_PATTERN.sub(lambda _match: section, existing)
        return existing.rstrip() + "\n\n" + section + "\n"
    return section + "\n"
