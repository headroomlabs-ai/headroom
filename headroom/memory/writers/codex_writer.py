"""Codex CLI memory writer — exports to AGENTS.md / instructions.md.

OpenAI Codex CLI's memory system:
- AGENTS.md files walk up directory tree (like Claude's CLAUDE.md)
- ~/.codex/AGENTS.override.md for global overrides
- All layers are merged before injection
- Plain markdown format, no special frontmatter
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from headroom.memory.writers.base import AgentWriter, MemoryEntry


class CodexMemoryWriter(AgentWriter):
    """Writes memories to Codex's AGENTS.md format."""

    agent_name = "codex"
    default_token_budget = 3000

    def format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format as AGENTS.md section."""
        lines = [
            "## Headroom Learned Context",
            "",
        ]

        grouped: dict[str, list[MemoryEntry]] = defaultdict(list)
        for m in memories:
            cat = m.category or "General"
            heading = cat.replace("_", " ").title()
            grouped[heading].append(m)

        for heading, entries in grouped.items():
            lines.append(f"### {heading}")
            for entry in entries:
                lines.append(f"- {entry.content}")
            lines.append("")

        return "\n".join(lines)

    def default_path(self) -> Path:
        """Default: AGENTS.md in project root."""
        return self._project_path / "AGENTS.md"
