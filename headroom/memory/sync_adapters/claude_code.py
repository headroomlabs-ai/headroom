"""Claude Code memory sync adapter.

Reads/writes Claude Code's native memory format:
    ~/.claude/projects/<sanitized-path>/memory/
        MEMORY.md          — index file (first 200 lines always in context)
        user_role.md       — individual memory files with YAML frontmatter
        project_codename.md
        ...

Each .md file has:
    ---
    name: <title>
    description: <one-line summary>
    type: <user|project|reference|feedback>
    headroom_id: <uuid>          (added by sync for cross-reference)
    source_agent: <agent name>   (added by sync for lineage)
    ---
    <body content>
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from headroom.memory.sync import AgentMemory, AgentMemoryAdapter


def _sanitize_for_filename(text: str) -> str:
    """Convert text to a safe filename slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    slug = slug.strip("_")[:50]
    return slug or "memory"


def encode_claude_project_path(project_path: Path | str) -> str:
    """Encode a project path the way Claude Code names project directories.

    POSIX absolute paths naturally become ``-Users-me-repo``. Windows drive
    paths should become ``-C-Users-me-repo`` rather than ``C:-Users-me-repo``.
    """
    rendered = str(project_path)
    drive_match = re.match(r"^([A-Za-z]):[\\/](.*)$", rendered)
    if drive_match:
        drive, rest = drive_match.groups()
        rest = rest.replace("\\", "-").replace("/", "-")
        return f"-{drive.upper()}-{rest}" if rest else f"-{drive.upper()}"
    return rendered.replace("/", "-").replace("\\", "-")


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from a markdown file.

    Returns (frontmatter_dict, body).
    """
    if not content.startswith("---"):
        return {}, content

    end = content.find("---", 3)
    if end == -1:
        return {}, content

    fm_text = content[3:end].strip()
    body = content[end + 3 :].strip()

    fm: dict[str, str] = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip().strip('"').strip("'")

    return fm, body


def _build_frontmatter(fields: dict[str, str]) -> str:
    """Build YAML frontmatter block."""
    lines = ["---"]
    for key, value in fields.items():
        if value:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


class ClaudeCodeAdapter(AgentMemoryAdapter):
    """Sync adapter for Claude Code's native memory files."""

    agent_name = "claude"

    def __init__(self, memory_dir: Path | str) -> None:
        self._memory_dir = Path(memory_dir)

    async def read_memories(self) -> list[AgentMemory]:
        """Read all .md memory files (except MEMORY.md index)."""
        if not self._memory_dir.exists():
            return []

        memories: list[AgentMemory] = []
        for md_file in sorted(self._memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue  # Index file, not a memory

            try:
                content = md_file.read_text(encoding="utf-8")
            except OSError:
                continue

            fm, body = _parse_frontmatter(content)
            if not body.strip():
                continue

            memories.append(
                AgentMemory(
                    content=body.strip(),
                    category=fm.get("type", ""),
                    source_file=md_file.name,
                    metadata={
                        "name": fm.get("name", ""),
                        "description": fm.get("description", ""),
                        "headroom_id": fm.get("headroom_id", ""),
                        "source_agent": fm.get("source_agent", "claude"),
                    },
                )
            )

        return memories

    async def write_memories(self, memories: list[dict[str, Any]]) -> int:
        """Write memories as individual .md files with frontmatter.

        Also updates MEMORY.md index.
        """
        if not memories:
            return 0

        self._memory_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        new_index_entries: list[str] = []

        for mem in memories:
            content = mem["content"]
            category = mem.get("category", "project")
            headroom_id = mem.get("headroom_id", "")
            source_agent = mem.get("source_agent", "unknown")
            content_hash = mem.get("content_hash", "")

            # Generate filename from content
            first_line = content.split("\n")[0][:60].strip()
            slug = _sanitize_for_filename(first_line)
            filename = f"headroom_{slug}.md"

            # Skip if file already exists with same content
            target = self._memory_dir / filename
            if target.exists():
                existing_fm, existing_body = _parse_frontmatter(target.read_text(encoding="utf-8"))
                existing_hash = hashlib.sha256(existing_body.strip().encode()).hexdigest()[:16]
                if existing_hash == content_hash:
                    continue

            # Build description (first 100 chars)
            description = content.replace("\n", " ")[:100]

            # Write file
            fm = _build_frontmatter(
                {
                    "name": first_line[:60],
                    "description": description,
                    "type": category or "project",
                    "headroom_id": headroom_id,
                    "source_agent": source_agent,
                }
            )
            target.write_text(f"{fm}\n\n{content}\n", encoding="utf-8")
            written += 1

            # Track for MEMORY.md index
            new_index_entries.append(f"- [{first_line[:60]}]({filename}) — {description[:80]}")

        # Update MEMORY.md index
        if new_index_entries:
            self._update_memory_md_index(new_index_entries)

        return written

    def _update_memory_md_index(self, new_entries: list[str]) -> None:
        """Append new entries to MEMORY.md under a Headroom section."""
        memory_md = self._memory_dir / "MEMORY.md"

        section_marker = "## Headroom Shared Memory"
        new_section = f"\n{section_marker}\n" + "\n".join(new_entries) + "\n"

        if memory_md.exists():
            content = memory_md.read_text(encoding="utf-8")
            if section_marker in content:
                # Append to existing section (before next ## or end)
                idx = content.index(section_marker)
                # Find end of section (next ## or end of file)
                next_section = content.find("\n## ", idx + len(section_marker))
                if next_section == -1:
                    # Append at end
                    content = content.rstrip() + "\n" + "\n".join(new_entries) + "\n"
                else:
                    # Insert before next section
                    content = (
                        content[:next_section].rstrip()
                        + "\n"
                        + "\n".join(new_entries)
                        + "\n"
                        + content[next_section:]
                    )
            else:
                content = content.rstrip() + "\n" + new_section
        else:
            content = "# Memory\n" + new_section

        memory_md.write_text(content, encoding="utf-8")

    def fingerprint(self) -> str:
        """Hash of all .md filenames + contents for change detection."""
        if not self._memory_dir.exists():
            return "empty"

        hasher = hashlib.sha256()
        found = False
        for md_file in sorted(self._memory_dir.glob("*.md")):
            try:
                hasher.update(md_file.name.encode())
                hasher.update(b"\0")
                hasher.update(md_file.read_bytes())
                hasher.update(b"\0")
                found = True
            except OSError:
                continue

        if not found:
            return "empty"
        return hasher.hexdigest()[:16]


def get_claude_memory_dir(project_path: Path | None = None) -> Path:
    """Get the Claude Code memory directory for a project.

    Claude Code stores per-project memory at:
        ~/.claude/projects/-<sanitized-path>/memory/
    """
    project = project_path or Path.cwd()
    sanitized = encode_claude_project_path(project)
    return Path.home() / ".claude" / "projects" / sanitized / "memory"
