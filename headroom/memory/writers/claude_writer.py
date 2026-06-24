"""Claude Code memory writer — exports to MEMORY.md and per-topic files.

Claude Code's memory system:
- MEMORY.md index at ~/.claude/projects/<project>/memory/MEMORY.md
- First 200 lines always loaded into context
- Per-topic files in same directory, loaded on demand
- Format: markdown with ## headers and bullet points
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from headroom.memory.sync_adapters.claude_code import encode_claude_project_path
from headroom.memory.writers.base import AgentWriter, MemoryEntry


class ClaudeCodeMemoryWriter(AgentWriter):
    """Writes memories to Claude Code's MEMORY.md format."""

    agent_name = "claude"
    default_token_budget = 2000  # Claude loads first 200 lines (~2K tokens)

    def __init__(
        self,
        project_path: Path | None = None,
        token_budget: int | None = None,
        memory_dir: Path | None = None,
    ) -> None:
        super().__init__(project_path, token_budget)
        self._memory_dir = memory_dir

    def format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format as Claude Code MEMORY.md section."""
        lines = [
            "## Headroom Learned Context",
            "*Auto-maintained by Headroom — do not edit manually*",
            "",
        ]

        # Group by category
        grouped: dict[str, list[MemoryEntry]] = defaultdict(list)
        for m in memories:
            cat = m.category or "General"
            # Capitalize category for heading
            heading = cat.replace("_", " ").title()
            grouped[heading].append(m)

        for heading, entries in grouped.items():
            lines.append(f"### {heading}")
            for entry in entries:
                # Each memory as a bullet point
                lines.append(f"- {entry.content}")
            lines.append("")

        return "\n".join(lines)

    def default_path(self) -> Path:
        """Default: Claude Code project memory directory."""
        if self._memory_dir:
            return self._memory_dir / "MEMORY.md"

        # Try to find Claude Code project memory path
        project_path = self._project_path
        # Claude Code stores per-project memory at:
        # ~/.claude/projects/-<sanitized-path>/memory/MEMORY.md
        sanitized = encode_claude_project_path(project_path)
        claude_memory_dir = Path.home() / ".claude" / "projects" / sanitized / "memory"
        return claude_memory_dir / "MEMORY.md"

    def export_topics(
        self,
        memories: list[MemoryEntry],
        dry_run: bool = True,
    ) -> dict[str, str]:
        """Export high-importance memories to per-topic files.

        Claude Code loads topic files on demand, so we can put
        detailed memories here without consuming the 200-line budget.

        Returns:
            Dict of filename → content for topic files written.
        """
        topic_files: dict[str, str] = {}

        # Group by category
        grouped: dict[str, list[MemoryEntry]] = defaultdict(list)
        for m in memories:
            cat = m.category or "general"
            grouped[cat].append(m)

        memory_dir = self.default_path().parent

        for category, entries in grouped.items():
            if len(entries) < 2:
                continue  # Not enough for a dedicated topic file

            filename = f"headroom_{category}.md"
            lines = [
                "---",
                f"name: headroom-{category}",
                f"description: Headroom-learned {category.replace('_', ' ')} patterns",
                "type: reference",
                "---",
                "",
                f"# {category.replace('_', ' ').title()}",
                "",
            ]
            for entry in entries:
                lines.append(f"- {entry.content}")
            lines.append("")

            content = "\n".join(lines)
            topic_files[filename] = content

            if not dry_run:
                memory_dir.mkdir(parents=True, exist_ok=True)
                (memory_dir / filename).write_text(content, encoding="utf-8")

        return topic_files
