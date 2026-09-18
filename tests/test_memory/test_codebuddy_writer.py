"""Tests for CodeBuddy memory writer."""

from __future__ import annotations

import time
from pathlib import Path

from headroom.memory.writers.base import MemoryEntry
from headroom.memory.writers.codebuddy_writer import CodeBuddyMemoryWriter


def _make_entries(count: int = 5) -> list[MemoryEntry]:
    entries = []
    categories = ["error_recovery", "environment", "preference", "architecture"]
    for i in range(count):
        entries.append(
            MemoryEntry(
                content=f"Test memory entry {i}: use pytest not unittest",
                importance=0.5 + (i % 3) * 0.15,
                category=categories[i % len(categories)],
                entity_refs=[f"/path/to/file{i}.py"],
                created_at=time.time() - i * 3600,
                access_count=max(0, 3 - i),
            )
        )
    return entries


class TestCodeBuddyWriter:
    def test_format_memories(self):
        writer = CodeBuddyMemoryWriter()
        entries = _make_entries(3)
        formatted = writer.format_memories(entries)

        assert "## Headroom Learned Context" in formatted
        assert "Auto-maintained by Headroom" in formatted
        assert "Test memory entry" in formatted
        assert "### " in formatted  # category headers

    def test_format_memories_groups_by_category(self):
        writer = CodeBuddyMemoryWriter()
        entries = [
            MemoryEntry(content="Use ruff", importance=0.9, category="preference"),
            MemoryEntry(content="Default port 8787", importance=0.8, category="environment"),
            MemoryEntry(content="Always lint first", importance=0.7, category="preference"),
        ]
        formatted = writer.format_memories(entries)
        # Both preference entries should appear
        assert "Use ruff" in formatted
        assert "Always lint first" in formatted
        assert "Default port 8787" in formatted

    def test_default_path_uses_memory_dir(self, tmp_path: Path):
        writer = CodeBuddyMemoryWriter(memory_dir=tmp_path)
        p = writer.default_path()
        assert p == tmp_path / "MEMORY.md"

    def test_default_path_encodes_project(self):
        writer = CodeBuddyMemoryWriter(project_path=Path("/Users/me/repo"))
        p = writer.default_path()
        assert "codebuddy" in str(p)
        assert "MEMORY.md" in str(p)

    def test_export_topics_dry_run(self, tmp_path: Path):
        writer = CodeBuddyMemoryWriter(memory_dir=tmp_path)
        entries = [
            MemoryEntry(content="Use ruff", importance=0.9, category="preference"),
            MemoryEntry(content="Always lint first", importance=0.7, category="preference"),
            MemoryEntry(content="Default port 8787", importance=0.8, category="environment"),
        ]
        result = writer.export_topics(entries, dry_run=True)

        assert "headroom_preference.md" in result
        assert "Use ruff" in result["headroom_preference.md"]
        assert "Always lint first" in result["headroom_preference.md"]
        # environment only has 1 entry, should not export
        assert "headroom_environment.md" not in result

    def test_export_topics_writes_files(self, tmp_path: Path):
        writer = CodeBuddyMemoryWriter(memory_dir=tmp_path)
        entries = [
            MemoryEntry(content="Use ruff", importance=0.9, category="preference"),
            MemoryEntry(content="Always lint first", importance=0.7, category="preference"),
        ]
        result = writer.export_topics(entries, dry_run=False)

        assert "headroom_preference.md" in result
        written = (tmp_path / "headroom_preference.md").read_text()
        assert "Use ruff" in written
        assert "---" in written  # YAML frontmatter

    def test_export_topics_single_entry_not_exported(self, tmp_path: Path):
        writer = CodeBuddyMemoryWriter(memory_dir=tmp_path)
        entries = [
            MemoryEntry(content="Only one", importance=0.9, category="security"),
        ]
        result = writer.export_topics(entries, dry_run=True)
        assert result == {}

    def test_export_topics_creates_parent_dir(self, tmp_path: Path):
        memory_dir = tmp_path / "nested" / "memory"
        writer = CodeBuddyMemoryWriter(memory_dir=memory_dir)
        entries = [
            MemoryEntry(content="Use ruff", importance=0.9, category="preference"),
            MemoryEntry(content="Always lint", importance=0.7, category="preference"),
        ]
        writer.export_topics(entries, dry_run=False)
        assert (memory_dir / "headroom_preference.md").exists()
