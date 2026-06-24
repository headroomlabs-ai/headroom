"""Tests for agent-native memory writers."""

from __future__ import annotations

import time
from pathlib import Path

from headroom.memory.writers.base import (
    MARKER_END,
    MARKER_START,
    MemoryEntry,
    _estimate_tokens,
    _merge_section,
)
from headroom.memory.writers.claude_writer import ClaudeCodeMemoryWriter
from headroom.memory.writers.codex_writer import CodexMemoryWriter
from headroom.memory.writers.cursor_writer import CursorMemoryWriter
from headroom.memory.writers.generic_writer import GenericMemoryWriter

# =============================================================================
# Test Data
# =============================================================================


def _make_entries(count: int = 5) -> list[MemoryEntry]:
    """Create test memory entries."""
    entries = []
    categories = ["error_recovery", "environment", "preference", "architecture"]
    for i in range(count):
        entries.append(
            MemoryEntry(
                content=f"Test memory entry {i}: use pytest not unittest",
                importance=0.5 + (i % 3) * 0.15,
                category=categories[i % len(categories)],
                entity_refs=[f"/path/to/file{i}.py"],
                created_at=time.time() - i * 3600,  # Each an hour older
                access_count=max(0, 3 - i),
            )
        )
    return entries


# =============================================================================
# Base / Shared Tests
# =============================================================================


class TestMemoryEntry:
    def test_score_calculation(self):
        entry = MemoryEntry(
            content="test",
            importance=0.9,
            created_at=time.time(),  # Just now
            access_count=5,
        )
        assert entry.score > 0.5  # High importance + recent + accessed

    def test_score_decays_with_age(self):
        recent = MemoryEntry(content="new", importance=0.9, created_at=time.time())
        old = MemoryEntry(content="old", importance=0.9, created_at=time.time() - 30 * 86400)
        assert recent.score > old.score

    def test_content_hash(self):
        e1 = MemoryEntry(content="same content")
        e2 = MemoryEntry(content="same content")
        assert e1.content_hash == e2.content_hash

    def test_different_hash(self):
        e1 = MemoryEntry(content="content A")
        e2 = MemoryEntry(content="content B")
        assert e1.content_hash != e2.content_hash


class TestTokenEstimate:
    def test_rough_estimate(self):
        assert _estimate_tokens("hello world") > 0
        assert _estimate_tokens("a" * 400) == 100  # ~4 chars per token


class TestMergeSection:
    def test_new_file(self, tmp_path: Path):
        nonexistent = tmp_path / "new.md"
        result = _merge_section(nonexistent, "new section")
        assert result == "new section\n"

    def test_append_to_existing(self, tmp_path: Path):
        existing = tmp_path / "existing.md"
        existing.write_text("# Existing Content\n\nSome stuff here.")
        result = _merge_section(existing, "new section")
        assert "# Existing Content" in result
        assert "new section" in result

    def test_replace_existing_markers(self, tmp_path: Path):
        existing = tmp_path / "marked.md"
        existing.write_text(f"# Header\n\n{MARKER_START}\nold content\n{MARKER_END}\n\n# Footer")
        result = _merge_section(existing, f"{MARKER_START}\nnew content\n{MARKER_END}")
        assert "new content" in result
        assert "old content" not in result
        assert "# Header" in result
        assert "# Footer" in result

    def test_replace_existing_markers_handles_literal_backslashes(self, tmp_path: Path):
        existing = tmp_path / "marked.md"
        existing.write_text(f"# Header\n\n{MARKER_START}\nold content\n{MARKER_END}\n")
        section = f"{MARKER_START}\n- Keep C:\\Users\\john.doe\\repo and literal \\u\n{MARKER_END}"

        result = _merge_section(existing, section)

        assert r"C:\Users\john.doe\repo" in result
        assert r"literal \u" in result
        assert "old content" not in result


# =============================================================================
# Claude Code Writer Tests
# =============================================================================


class TestClaudeCodeWriter:
    def test_format_memories(self):
        writer = ClaudeCodeMemoryWriter()
        entries = _make_entries(3)
        formatted = writer.format_memories(entries)

        assert "## Headroom Learned Context" in formatted
        assert "Auto-maintained by Headroom" in formatted
        assert "Test memory entry" in formatted

    def test_export_dry_run(self, tmp_path: Path):
        writer = ClaudeCodeMemoryWriter(
            project_path=tmp_path,
            memory_dir=tmp_path / "memory",
        )
        entries = _make_entries(3)
        result = writer.export(entries, dry_run=True)

        assert result.dry_run is True
        assert result.memories_exported == 3
        assert len(result.files_written) == 1
        assert not (tmp_path / "memory" / "MEMORY.md").exists()

    def test_export_writes_file(self, tmp_path: Path):
        writer = ClaudeCodeMemoryWriter(
            project_path=tmp_path,
            memory_dir=tmp_path / "memory",
        )
        entries = _make_entries(3)
        result = writer.export(entries, dry_run=False)

        assert result.memories_exported == 3
        written = (tmp_path / "memory" / "MEMORY.md").read_text()
        assert MARKER_START in written
        assert MARKER_END in written
        assert "Test memory entry" in written

    def test_budget_limits_output(self, tmp_path: Path):
        writer = ClaudeCodeMemoryWriter(
            project_path=tmp_path,
            memory_dir=tmp_path / "memory",
            token_budget=50,  # Very small budget
        )
        entries = _make_entries(10)
        result = writer.export(entries, dry_run=True)

        assert result.memories_skipped_budget > 0
        assert result.memories_exported < 10

    def test_dedup(self, tmp_path: Path):
        writer = ClaudeCodeMemoryWriter(
            project_path=tmp_path,
            memory_dir=tmp_path / "memory",
        )
        entries = [
            MemoryEntry(content="duplicate content", importance=0.5),
            MemoryEntry(content="duplicate content", importance=0.8),  # Same content
            MemoryEntry(content="unique content", importance=0.6),
        ]
        result = writer.export(entries, dry_run=True)

        assert result.memories_skipped_dedup == 1
        assert result.memories_exported == 2

    def test_export_topics(self, tmp_path: Path):
        writer = ClaudeCodeMemoryWriter(
            project_path=tmp_path,
            memory_dir=tmp_path / "memory",
        )
        entries = _make_entries(6)
        topics = writer.export_topics(entries, dry_run=True)

        # Should produce topic files for categories with 2+ entries
        assert len(topics) > 0
        for filename, content in topics.items():
            assert filename.startswith("headroom_")
            assert "---" in content  # YAML frontmatter

    def test_default_path_encodes_windows_user_with_dot(self):
        writer = ClaudeCodeMemoryWriter(project_path=Path(r"C:\Users\john.doe\work"))

        rendered = str(writer.default_path())
        assert "-C-Users-john.doe-work" in rendered
        assert "john-doe" not in rendered
        assert rendered.endswith("MEMORY.md")


# =============================================================================
# Cursor Writer Tests
# =============================================================================


class TestCursorWriter:
    def test_format_memories(self):
        writer = CursorMemoryWriter()
        entries = _make_entries(3)
        formatted = writer.format_memories(entries)
        assert "##" in formatted
        assert "Test memory entry" in formatted

    def test_creates_mdc_with_frontmatter(self, tmp_path: Path):
        writer = CursorMemoryWriter(project_path=tmp_path)
        entries = _make_entries(3)
        writer.export(entries, dry_run=False)

        mdc_path = tmp_path / ".cursor" / "rules" / "headroom-memory.mdc"
        assert mdc_path.exists()
        content = mdc_path.read_text()
        assert "---" in content
        assert "alwaysApply: true" in content
        assert "description:" in content
        assert MARKER_START in content

    def test_updates_existing_mdc(self, tmp_path: Path):
        mdc_dir = tmp_path / ".cursor" / "rules"
        mdc_dir.mkdir(parents=True)
        mdc_file = mdc_dir / "headroom-memory.mdc"
        mdc_file.write_text(
            "---\ndescription: test\nalwaysApply: true\n---\n\n"
            f"# Header\n\n{MARKER_START}\nold stuff\n{MARKER_END}\n"
        )

        writer = CursorMemoryWriter(project_path=tmp_path)
        entries = _make_entries(2)
        writer.export(entries, dry_run=False)

        content = mdc_file.read_text()
        assert "old stuff" not in content
        assert "Test memory entry" in content
        assert "alwaysApply: true" in content  # Preserved


# =============================================================================
# Codex Writer Tests
# =============================================================================


class TestCodexWriter:
    def test_format_memories(self):
        writer = CodexMemoryWriter()
        entries = _make_entries(3)
        formatted = writer.format_memories(entries)
        assert "## Headroom Learned Context" in formatted

    def test_default_path(self, tmp_path: Path):
        writer = CodexMemoryWriter(project_path=tmp_path)
        assert writer.default_path() == tmp_path / "AGENTS.md"

    def test_export(self, tmp_path: Path):
        writer = CodexMemoryWriter(project_path=tmp_path)
        entries = _make_entries(3)
        writer.export(entries, dry_run=False)

        agents_md = tmp_path / "AGENTS.md"
        assert agents_md.exists()
        content = agents_md.read_text()
        assert MARKER_START in content


# =============================================================================
# Generic Writer Tests
# =============================================================================


class TestGenericWriter:
    def test_custom_filename(self, tmp_path: Path):
        writer = GenericMemoryWriter(project_path=tmp_path, filename="GEMINI.md")
        assert writer.default_path() == tmp_path / "GEMINI.md"

    def test_export(self, tmp_path: Path):
        writer = GenericMemoryWriter(project_path=tmp_path)
        entries = _make_entries(3)
        result = writer.export(entries, dry_run=False)

        assert (tmp_path / "HEADROOM_MEMORY.md").exists()
        assert result.memories_exported == 3
