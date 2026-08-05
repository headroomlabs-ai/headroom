"""Codex CLI memory sync adapter.

Syncs memories to/from a headroom-managed section in AGENTS.md.
Codex reads AGENTS.md automatically before every task.

Note: Codex primarily uses the MCP server for memory (memory_search/save).
This adapter provides supplementary context injection via AGENTS.md so
Codex has key memories even without explicit tool calls.

Format in AGENTS.md:
    <!-- headroom:memory:start -->
    ## Headroom Shared Memory
    - fact 1
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
_MARKER_PATTERN = re.compile(
    re.escape(_MARKER_START) + r"(.*?)" + re.escape(_MARKER_END),
    re.DOTALL,
)


class CodexAdapter(AgentMemoryAdapter):
    """Sync adapter for Codex's AGENTS.md."""

    agent_name = "codex"

    def __init__(self, agents_md_path: Path | str | None = None) -> None:
        self._path = Path(agents_md_path) if agents_md_path else Path.cwd() / "AGENTS.md"

    async def read_memories(self) -> list[AgentMemory]:
        """Read memories from the headroom section of AGENTS.md."""
        if not self._path.exists():
            return []

        content = self._path.read_text(encoding="utf-8")
        match = _MARKER_PATTERN.search(content)
        if not match:
            return []

        section = match.group(1).strip()
        memories: list[AgentMemory] = []

        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                fact = line[2:].strip()
                if fact:
                    memories.append(
                        AgentMemory(
                            content=fact,
                            source_file=self._path.name,
                        )
                    )

        return memories

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

        # Facts already in the managed section — preserve them (dedup by the
        # rendered first-line, matching how read_memories reconstructs them).
        facts: list[str] = []
        seen: set[str] = set()
        existing_match = _MARKER_PATTERN.search(existing_content)
        if existing_match:
            for line in existing_match.group(1).split("\n"):
                stripped = line.strip()
                if stripped.startswith("- "):
                    fact = stripped[2:].strip()
                    if fact and fact not in seen:
                        seen.add(fact)
                        facts.append(fact)

        added = 0
        for mem in memories:
            fact = mem["content"].split("\n")[0].strip()  # First line only
            if fact and fact not in seen:
                seen.add(fact)
                facts.append(fact)
                added += 1

        lines = ["## Headroom Shared Memory", ""]
        lines.extend(f"- {fact}" for fact in facts)
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

        self._path.write_text(content, encoding="utf-8")
        return added

    def fingerprint(self) -> str:
        """Hash of AGENTS.md contents."""
        if not self._path.exists():
            return "empty"
        try:
            hasher = hashlib.sha256()
            hasher.update(self._path.name.encode())
            hasher.update(b"\0")
            hasher.update(self._path.read_bytes())
            return hasher.hexdigest()[:16]
        except OSError:
            return "error"
