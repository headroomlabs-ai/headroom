"""Cursor memory writer — exports to .cursor/rules/*.mdc files.

Cursor's rules system:
- Files in .cursor/rules/ with .mdc extension
- YAML frontmatter with: description, globs (optional), alwaysApply (bool)
- Markdown body with instructions
- Rules with alwaysApply: true are always loaded
- Rules with globs are loaded when matching files are in context
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from headroom.memory.writers.base import (
    MARKER_END,
    MARKER_PATTERN,
    MARKER_START,
    AgentWriter,
    ExportResult,
    MemoryEntry,
    _estimate_tokens,
)


class CursorMemoryWriter(AgentWriter):
    """Writes memories to Cursor's .cursor/rules/*.mdc format."""

    agent_name = "cursor"
    default_token_budget = 3000

    def format_memories(self, memories: list[MemoryEntry]) -> str:
        """Format as Cursor .mdc content (body only, no frontmatter).

        Frontmatter is handled in export() since it's outside markers.
        """
        lines: list[str] = []

        grouped: dict[str, list[MemoryEntry]] = defaultdict(list)
        for m in memories:
            cat = m.category or "General"
            heading = cat.replace("_", " ").title()
            grouped[heading].append(m)

        for heading, entries in grouped.items():
            lines.append(f"## {heading}")
            lines.append("")
            for entry in entries:
                lines.append(f"- {entry.content}")
            lines.append("")

        return "\n".join(lines)

    def default_path(self) -> Path:
        """Default: .cursor/rules/headroom-memory.mdc in project root."""
        return self._project_path / ".cursor" / "rules" / "headroom-memory.mdc"

    def export(
        self,
        memories: list[MemoryEntry],
        output_path: Path | None = None,
        dry_run: bool = True,
    ) -> ExportResult:
        """Export with Cursor-specific .mdc frontmatter."""
        result = ExportResult(dry_run=dry_run)

        if not memories:
            return result

        # Rank, dedup, budget (reuse base logic)
        ranked = sorted(memories, key=lambda m: m.score, reverse=True)
        seen: set[str] = set()
        unique: list[MemoryEntry] = []
        for m in ranked:
            h = m.content_hash
            if h in seen:
                result.memories_skipped_dedup += 1
                continue
            seen.add(h)
            unique.append(m)

        budgeted: list[MemoryEntry] = []
        tokens_used = 0
        for m in unique:
            entry_tokens = _estimate_tokens(m.content) + 10
            if tokens_used + entry_tokens > self._token_budget:
                result.memories_skipped_budget += 1
                continue
            tokens_used += entry_tokens
            budgeted.append(m)

        if not budgeted:
            return result

        # Build full .mdc file content
        body = self.format_memories(budgeted)

        target = output_path or self.default_path()

        # If file exists and has our markers, only replace marker section
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if MARKER_START in existing:
                section = f"{MARKER_START}\n{body}\n{MARKER_END}"
                full_content = MARKER_PATTERN.sub(lambda _match: section, existing)
            else:
                # Append our section
                section = f"{MARKER_START}\n{body}\n{MARKER_END}"
                full_content = existing.rstrip() + "\n\n" + section + "\n"
        else:
            # Create new .mdc file with frontmatter
            full_content = (
                "---\n"
                "description: Headroom-learned patterns from proxy traffic\n"
                "alwaysApply: true\n"
                "---\n"
                "\n"
                "# Headroom Learned Context\n"
                "\n"
                f"{MARKER_START}\n{body}\n{MARKER_END}\n"
            )

        result.files_written.append(target)
        result.content_by_file[str(target)] = full_content
        result.memories_exported = len(budgeted)

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(full_content, encoding="utf-8")

        return result
