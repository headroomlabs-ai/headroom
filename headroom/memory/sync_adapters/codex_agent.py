"""Codex CLI memory sync adapter.

Syncs memories to/from a headroom-managed section in AGENTS.md.
Codex reads AGENTS.md automatically before every task.

Note: Codex primarily uses the MCP server for memory (memory_search/save).
This adapter provides supplementary context injection via AGENTS.md so
Codex has key memories even without explicit tool calls.

Format in AGENTS.md:
    <!-- headroom:memory:start -->
    <!-- headroom:memory:format:2 -->

    ## Headroom Shared Memory
    - fact 1
      continuation of fact 1
    - fact 2
    <!-- headroom:memory:end -->
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from headroom.memory.sync import AgentMemory, AgentMemoryAdapter

_MARKER_START = "<!-- headroom:memory:start -->"
_MARKER_END = "<!-- headroom:memory:end -->"
_MULTILINE_FORMAT = "<!-- headroom:memory:format:2 -->"
_MARKER_PATTERN = re.compile(
    re.escape(_MARKER_START) + r"(.*?)" + re.escape(_MARKER_END),
    re.DOTALL,
)


def _parse_facts(section: str) -> list[str]:
    """Read top-level bullets and the continuations of new-format bullets."""
    lines = section.split("\n")
    multiline = _MULTILINE_FORMAT in lines
    facts: list[list[str]] = []
    for line in lines:
        if not multiline:
            line = line.strip()
        if line.startswith("- "):
            facts.append([line[2:]])
        elif multiline and facts and (not line or line.startswith("  ")):
            facts[-1].append(line[2:])
    return [fact for parts in facts if (fact := "\n".join(parts).strip())]


def _normalize_note(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


class CodexAdapter(AgentMemoryAdapter):
    """Sync adapter for Codex's AGENTS.md."""

    agent_name = "codex"

    def __init__(self, agents_md_path: Path | str | None = None) -> None:
        self._path = Path(agents_md_path) if agents_md_path else Path.cwd() / "AGENTS.md"

    def memory_content_hash(self, content: str) -> str:
        """Match both the original note and its LF-normalized file form."""
        return super().memory_content_hash(_normalize_note(content))

    async def read_memories(self) -> list[AgentMemory]:
        """Read memories from the headroom section of AGENTS.md."""
        if not self._path.exists():
            return []

        content = self._path.read_text(encoding="utf-8")
        match = _MARKER_PATTERN.search(content)
        if not match:
            return []

        return [
            AgentMemory(content=fact, source_file=self._path.name)
            for fact in _parse_facts(match.group(1))
        ]

    async def write_memories(self, memories: list[dict[str, Any]]) -> int:
        """Merge memories into the headroom section of AGENTS.md.

        ``sync_export`` hands this adapter only the *delta* — memories the
        agent doesn't already have (see ``AgentMemoryAdapter`` contract; the
        sibling ClaudeCode adapter is additive for the same reason). So this
        must accumulate: rebuilding the section from just ``memories`` would
        erase every previously-synced fact on each run, thrashing the file
        between disjoint subsets and never converging.
        """
        if not memories:
            return 0

        existing_content = self._path.read_text(encoding="utf-8") if self._path.exists() else ""

        # The reader reconstructs the complete text, so two facts with the
        # same first line remain distinct.
        existing_match = _MARKER_PATTERN.search(existing_content)
        facts = dict.fromkeys(_parse_facts(existing_match.group(1))) if existing_match else {}

        added = 0
        for mem in memories:
            fact = _normalize_note(mem["content"])
            if fact and fact not in facts:
                facts[fact] = None
                added += 1

        if not added:
            return 0
        lines = [_MULTILINE_FORMAT, "", "## Headroom Shared Memory", ""]
        for fact in facts:
            first, *continuations = fact.split("\n")
            lines.append(f"- {first}")
            lines.extend(f"  {line}" if line else "" for line in continuations)
        lines.append("")
        section = f"{_MARKER_START}\n" + "\n".join(lines) + f"{_MARKER_END}"

        # Splice the section back in. Use a function replacement (not a string
        # template) so literal backslashes / \\u in a memory aren't treated as
        # regex escapes.
        if existing_content:
            if _MARKER_START in existing_content:
                content = _MARKER_PATTERN.sub(lambda _match: section, existing_content)
            else:
                content = existing_content.rstrip() + "\n\n" + section + "\n"
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            content = section + "\n"

        self._path.write_text(content, encoding="utf-8", newline="\n")
        return added

    def fingerprint(self) -> str:
        """Hash of AGENTS.md contents."""
        if not self._path.exists():
            return "empty"
        try:
            hasher = hashlib.sha256()
            # Recheck once after an upgrade so previously abbreviated notes
            # can be exported even when the database and file did not change.
            hasher.update(_MULTILINE_FORMAT.encode())
            hasher.update(b"\0")
            hasher.update(self._path.name.encode())
            hasher.update(b"\0")
            hasher.update(self._path.read_bytes())
            return hasher.hexdigest()[:16]
        except OSError:
            return "error"
