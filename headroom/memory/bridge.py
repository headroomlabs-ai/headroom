"""Memory Bridge: bidirectional bridge between markdown files and Headroom memory.

Imports markdown memory files (Claude Code MEMORY.md, ChatGPT facts, etc.)
into Headroom's semantic memory system, and exports Headroom memories back
to organized markdown files. Supports bidirectional sync with hash-based
change detection.

Usage:
    from headroom.memory.bridge import MemoryBridge
    from headroom.memory.bridge_config import BridgeConfig
    from pathlib import Path

    config = BridgeConfig(
        md_paths=[Path("~/.claude/.../memory/MEMORY.md")],
        user_id="alice",
    )
    bridge = MemoryBridge(config, backend)

    # Import markdown -> Headroom
    stats = await bridge.import_from_markdown()

    # Export Headroom -> markdown
    markdown = await bridge.export_to_markdown()

    # Bidirectional sync
    stats = await bridge.sync()
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from headroom.memory.bridge_config import BridgeConfig, MarkdownFormat
from headroom.memory.bridge_parsers import (
    ParsedFile,
    ParsedSection,
    extract_relationships_from_section,
    parse_markdown,
)

if TYPE_CHECKING:
    from headroom.memory.backends.local import LocalBackend
    from headroom.memory.models import Memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ImportStats:
    """Statistics from an import operation."""

    files_processed: int = 0
    files_skipped_unchanged: int = 0
    sections_imported: int = 0
    sections_skipped_duplicate: int = 0
    sections_failed: int = 0
    entities_extracted: int = 0
    total_facts: int = 0


@dataclass
class SyncStats:
    """Statistics from a sync operation."""

    import_stats: ImportStats = field(default_factory=ImportStats)
    memories_exported: int = 0
    files_unchanged: int = 0
    files_updated: int = 0


# ---------------------------------------------------------------------------
# MemoryBridge
# ---------------------------------------------------------------------------


class MemoryBridge:
    """Bidirectional bridge between markdown memory files and Headroom's
    semantic memory system.

    Supports import (md -> Headroom), export (Headroom -> md), and
    bidirectional sync with hash-based change detection.
    """

    def __init__(
        self,
        config: BridgeConfig,
        backend: LocalBackend | Any,
    ) -> None:
        self._config = config
        self._backend = backend
        self._sync_state: dict[str, Any] = {}
        self._load_sync_state()

    # =========================================================================
    # Sync State Management
    # =========================================================================

    def _load_sync_state(self) -> None:
        """Load sync state from disk."""
        path = self._config.sync_state_path
        if path.exists():
            try:
                self._sync_state = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Bridge: Failed to load sync state from {path}: {e}")
                self._sync_state = {}
        if "version" not in self._sync_state:
            self._sync_state = {"version": 1, "last_sync": None, "files": {}}

    def _save_sync_state(self) -> None:
        """Persist sync state to disk."""
        path = self._config.sync_state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._sync_state, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"Bridge: Failed to save sync state to {path}: {e}")

    def _get_stored_file_hash(self, file_path: str) -> str | None:
        """Get the stored hash for a file, or None if not tracked."""
        files = self._sync_state.get("files", {})
        file_state = files.get(file_path, {})
        result = file_state.get("hash")
        return str(result) if result is not None else None

    def _get_stored_section_hashes(self, file_path: str) -> dict[str, str]:
        """Get stored section hash -> memory_id mapping for a file."""
        files = self._sync_state.get("files", {})
        file_state = files.get(file_path, {})
        sections: dict[str, str] = file_state.get("sections", {})
        return sections

    def _update_file_state(
        self,
        file_path: str,
        file_hash: str,
        section_mapping: dict[str, str],
    ) -> None:
        """Update sync state for a file."""
        if "files" not in self._sync_state:
            self._sync_state["files"] = {}
        self._sync_state["files"][file_path] = {
            "hash": file_hash,
            "last_imported": datetime.now(timezone.utc).isoformat(),
            "sections": section_mapping,
        }
        self._sync_state["last_sync"] = datetime.now(timezone.utc).isoformat()

    # =========================================================================
    # Import: Markdown -> Headroom
    # =========================================================================

    async def import_from_markdown(
        self,
        paths: list[Path] | None = None,
        user_id: str | None = None,
        force: bool = False,
    ) -> ImportStats:
        """Import markdown memory files into Headroom's vector store.

        Args:
            paths: Files to import (uses config.md_paths if None).
            user_id: User ID for imported memories (uses config.user_id if None).
            force: If True, import even if file hash hasn't changed.

        Returns:
            ImportStats with counts.
        """
        paths = paths or self._config.md_paths
        user_id = user_id or self._config.user_id
        total_stats = ImportStats()

        for path in paths:
            path = Path(path).expanduser()
            if not path.exists():
                logger.warning(f"Bridge: File not found: {path}")
                total_stats.files_skipped_unchanged += 1
                continue

            file_stats = await self._import_file(path, user_id, force)
            total_stats.files_processed += file_stats.files_processed
            total_stats.files_skipped_unchanged += file_stats.files_skipped_unchanged
            total_stats.sections_imported += file_stats.sections_imported
            total_stats.sections_skipped_duplicate += file_stats.sections_skipped_duplicate
            total_stats.sections_failed += file_stats.sections_failed
            total_stats.entities_extracted += file_stats.entities_extracted
            total_stats.total_facts += file_stats.total_facts

        self._save_sync_state()
        logger.info(
            f"Bridge: Import complete — {total_stats.sections_imported} sections imported, "
            f"{total_stats.sections_skipped_duplicate} duplicates skipped"
        )
        return total_stats

    async def _import_file(
        self,
        path: Path,
        user_id: str,
        force: bool = False,
    ) -> ImportStats:
        """Import a single markdown file."""
        stats = ImportStats()
        file_path_str = str(path)

        # Parse file
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error(f"Bridge: Failed to read {path}: {e}")
            stats.sections_failed += 1
            return stats

        parsed = self._parse_content(content, file_path_str)

        # Check if file has changed
        stored_hash = self._get_stored_file_hash(file_path_str)
        if not force and stored_hash == parsed.file_hash:
            logger.debug(f"Bridge: File unchanged, skipping: {path}")
            stats.files_skipped_unchanged += 1
            return stats

        stats.files_processed = 1
        stored_sections = self._get_stored_section_hashes(file_path_str)
        new_section_mapping: dict[str, str] = {}

        for section in parsed.sections:
            if not section.content.strip():
                continue

            # Skip sections that haven't changed (by content hash)
            if not force and section.content_hash in stored_sections:
                memory_id = stored_sections[section.content_hash]
                new_section_mapping[section.content_hash] = memory_id
                stats.sections_skipped_duplicate += 1
                continue

            try:
                imported_id = await self._import_section(section, user_id, file_path_str)
                if imported_id:
                    new_section_mapping[section.content_hash] = imported_id
                    stats.sections_imported += 1
                    stats.entities_extracted += len(section.entities)
                    stats.total_facts += len(section.facts)
                else:
                    stats.sections_skipped_duplicate += 1
            except Exception as e:
                logger.error(f"Bridge: Failed to import section '{section.heading}': {e}")
                stats.sections_failed += 1

        self._update_file_state(file_path_str, parsed.file_hash, new_section_mapping)
        return stats

    async def _import_section(
        self,
        section: ParsedSection,
        user_id: str,
        file_path: str,
    ) -> str | None:
        """Import a single parsed section into Headroom.

        Returns the memory ID if imported, None if skipped as duplicate.
        """
        # Check for semantic duplicates
        if await self._check_duplicate(section.content, user_id):
            return None

        # Compute importance from heading level
        importance = self._config.heading_importance_map.get(
            section.heading_level, self._config.default_importance
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "source": self._config.source_tag,
            "source_file": file_path,
            "content_hash": section.content_hash,
            "imported_at": datetime.now(timezone.utc).isoformat(),
        }
        if section.heading:
            metadata["section_heading"] = section.heading

        # Extract entities and relationships
        entities = section.entities if self._config.extract_entities else None
        relationships = None
        if self._config.extract_entities and section.entities:
            relationships = extract_relationships_from_section(section)

        # Store via backend
        # If we have individual facts and chunk_by_section is True, store as facts
        if self._config.chunk_by_section and section.facts:
            memory = await self._backend.save_memory(
                content=section.content,
                user_id=user_id,
                importance=importance,
                entities=entities,
                relationships=relationships if relationships else None,
                metadata=metadata,
                facts=section.facts,
            )
        else:
            memory = await self._backend.save_memory(
                content=section.content,
                user_id=user_id,
                importance=importance,
                entities=entities,
                relationships=relationships if relationships else None,
                metadata=metadata,
            )

        return memory.id

    def _parse_content(self, content: str, file_path: str) -> ParsedFile:
        """Parse markdown content using configured or auto-detected format."""
        fmt = self._config.md_format
        if fmt == MarkdownFormat.AUTO:
            return parse_markdown(content, file_path, format=None)
        return parse_markdown(content, file_path, format=fmt.value)

    async def _check_duplicate(self, content: str, user_id: str) -> bool:
        """Check if similar content already exists in memory.

        Uses semantic search with high similarity threshold.
        """
        threshold = self._config.dedup_similarity_threshold
        try:
            results = await self._backend.search_memories(
                query=content[:500],  # Limit query length
                user_id=user_id,
                top_k=3,
                min_similarity=threshold,
            )
            return len(results) > 0
        except Exception:
            # If search fails, don't block import
            return False

    # =========================================================================
    # Export: Headroom -> Markdown
    # =========================================================================

    async def export_to_markdown(
        self,
        path: Path | None = None,
        user_id: str | None = None,
        format: MarkdownFormat | None = None,
        top_k: int = 200,
    ) -> str:
        """Export Headroom memories to a markdown file.

        Args:
            path: Output file path (uses config.export_path if None).
            user_id: User to export (uses config.user_id if None).
            format: Output format (uses config.export_format if None).
            top_k: Maximum memories to export.

        Returns:
            The generated markdown string.
        """
        path = path or self._config.export_path
        user_id = user_id or self._config.user_id
        format = format or self._config.export_format

        memories = await self._fetch_all_memories(user_id, top_k)
        if not memories:
            markdown = "# Memories\n\nNo memories stored yet.\n"
        elif format == MarkdownFormat.CHATGPT:
            markdown = self._format_chatgpt_style(memories)
        elif format == MarkdownFormat.CLAUDE_CODE:
            grouped = self._group_memories_by_topic(memories)
            markdown = self._format_claude_code_style(grouped)
        else:
            grouped = self._group_memories_by_topic(memories)
            markdown = self._format_generic_style(grouped)

        if path:
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(markdown, encoding="utf-8")
            logger.info(f"Bridge: Exported {len(memories)} memories to {path}")

            # Update sync state to avoid re-importing our own export
            import hashlib

            file_hash = hashlib.sha256(markdown.encode()).hexdigest()
            self._update_file_state(str(path), file_hash, {})
            self._save_sync_state()

        return markdown

    async def _fetch_all_memories(
        self,
        user_id: str,
        top_k: int = 200,
    ) -> list[Memory]:
        """Fetch all memories for a user, sorted by importance then recency."""
        if hasattr(self._backend, "get_user_memories"):
            memories = await self._backend.get_user_memories(user_id, limit=top_k)
        else:
            # Fallback: search with broad query
            results = await self._backend.search_memories(
                query="*",
                user_id=user_id,
                top_k=top_k,
            )
            memories = [r.memory for r in results]

        # Sort: importance descending, then recency descending
        memories.sort(key=lambda m: (-m.importance, -m.created_at.timestamp()))
        return memories

    def _group_memories_by_topic(
        self,
        memories: list[Memory],
    ) -> dict[str, list[Memory]]:
        """Group memories by topic using metadata and entity clustering."""
        groups: dict[str, list[Memory]] = defaultdict(list)

        for memory in memories:
            metadata = memory.metadata or {}

            # Priority 1: section_heading from import
            heading = metadata.get("section_heading")
            if heading:
                groups[heading].append(memory)
                continue

            # Priority 2: topic from metadata
            topic = metadata.get("topic")
            if topic:
                groups[topic].append(memory)
                continue

            # Priority 3: most common entity
            if memory.entity_refs:
                groups[memory.entity_refs[0]].append(memory)
                continue

            # Fallback
            groups["General"].append(memory)

        return dict(groups)

    def _format_claude_code_style(
        self,
        grouped: dict[str, list[Memory]],
    ) -> str:
        """Format memories as Claude Code MEMORY.md style."""
        lines = ["# Memory\n"]

        for topic, memories in grouped.items():
            lines.append(f"## {topic}")
            for memory in memories:
                content = memory.content.strip()
                # If content has multiple lines, use first line as the bullet
                first_line = content.split("\n")[0].strip()
                if first_line.startswith("- "):
                    lines.append(first_line)
                else:
                    lines.append(f"- {first_line}")
            lines.append("")

        return "\n".join(lines)

    def _format_chatgpt_style(
        self,
        memories: list[Memory],
    ) -> str:
        """Format as ChatGPT flat fact list."""
        lines = []
        for memory in memories:
            content = memory.content.strip()
            first_line = content.split("\n")[0].strip()
            if first_line.startswith("- "):
                first_line = first_line[2:]
            lines.append(first_line)
        return "\n".join(lines) + "\n"

    def _format_generic_style(
        self,
        grouped: dict[str, list[Memory]],
    ) -> str:
        """Format as generic structured markdown."""
        lines = ["# Memories\n"]

        for topic, memories in grouped.items():
            lines.append(f"## {topic}")
            for memory in memories:
                content = memory.content.strip()
                first_line = content.split("\n")[0].strip()
                if first_line.startswith("- "):
                    lines.append(first_line)
                else:
                    lines.append(f"- {first_line}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Sync: Bidirectional
    # =========================================================================

    async def sync(
        self,
        paths: list[Path] | None = None,
        user_id: str | None = None,
    ) -> SyncStats:
        """Bidirectional sync between markdown files and Headroom.

        1. Import new/changed sections from md files.
        2. Export new organic Headroom memories back to md files.
        """
        paths = paths or self._config.md_paths
        user_id = user_id or self._config.user_id
        stats = SyncStats()

        # Phase 1: Import from markdown
        stats.import_stats = await self.import_from_markdown(paths=paths, user_id=user_id)

        # Phase 2: Export new organic memories to markdown
        last_sync_str = self._sync_state.get("last_sync")
        since = None
        if last_sync_str:
            try:
                since = datetime.fromisoformat(last_sync_str)
            except (ValueError, TypeError):
                pass

        new_memories = await self._get_new_organic_memories(user_id, since)

        if new_memories and paths:
            # Append to the first configured file
            primary_path = Path(paths[0]).expanduser()
            count = await self._append_to_markdown(primary_path, new_memories)
            stats.memories_exported = count
            if count > 0:
                stats.files_updated = 1

        self._sync_state["last_sync"] = datetime.now(timezone.utc).isoformat()
        self._save_sync_state()

        logger.info(
            f"Bridge: Sync complete — imported {stats.import_stats.sections_imported}, "
            f"exported {stats.memories_exported}"
        )
        return stats

    async def _get_new_organic_memories(
        self,
        user_id: str,
        since: datetime | None = None,
    ) -> list[Memory]:
        """Get memories created since last sync that didn't come from bridge import.

        Filters out memories with metadata source == source_tag.
        """
        if hasattr(self._backend, "get_user_memories"):
            all_memories = await self._backend.get_user_memories(user_id, limit=500)
        else:
            results = await self._backend.search_memories(query="*", user_id=user_id, top_k=500)
            all_memories = [r.memory for r in results]

        organic: list[Memory] = []
        for memory in all_memories:
            # Skip memories created by the bridge itself
            metadata = memory.metadata or {}
            if metadata.get("source") == self._config.source_tag:
                continue

            # Skip memories from before last sync
            if since:
                # Handle timezone-naive vs timezone-aware comparison
                mem_time = memory.created_at
                cmp_time = since
                if mem_time.tzinfo is None and cmp_time.tzinfo is not None:
                    mem_time = mem_time.replace(tzinfo=timezone.utc)
                elif mem_time.tzinfo is not None and cmp_time.tzinfo is None:
                    cmp_time = cmp_time.replace(tzinfo=timezone.utc)
                if mem_time < cmp_time:
                    continue

            organic.append(memory)

        return organic

    async def _append_to_markdown(
        self,
        path: Path,
        memories: list[Memory],
    ) -> int:
        """Append new memories to an existing markdown file.

        Reads the file, appends memories under appropriate sections,
        writes back. Returns count of memories appended.
        """
        if not memories:
            return 0

        # Read existing content
        existing_content = ""
        if path.exists():
            try:
                existing_content = path.read_text(encoding="utf-8")
            except OSError:
                pass

        # Group new memories by topic
        grouped = self._group_memories_by_topic(memories)
        lines_to_append: list[str] = []

        for topic, topic_memories in grouped.items():
            # Check if section already exists in the file
            section_exists = f"## {topic}" in existing_content

            if not section_exists:
                lines_to_append.append(f"\n## {topic}")

            for memory in topic_memories:
                content = memory.content.strip()
                first_line = content.split("\n")[0].strip()
                if first_line.startswith("- "):
                    lines_to_append.append(first_line)
                else:
                    lines_to_append.append(f"- {first_line}")

        if not lines_to_append:
            return 0

        # Append to file
        append_text = "\n".join(lines_to_append) + "\n"
        new_content = existing_content.rstrip() + "\n" + append_text

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            logger.error(f"Bridge: Failed to write {path}: {e}")
            return 0

        # Update file hash in sync state
        import hashlib

        file_hash = hashlib.sha256(new_content.encode()).hexdigest()
        stored_sections = self._get_stored_section_hashes(str(path))
        self._update_file_state(str(path), file_hash, stored_sections)

        return len(memories)
