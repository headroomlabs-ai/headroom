"""Tests for the Memory Bridge (markdown <-> Headroom bidirectional sync).

Parser tests are pure functions (no backend needed).
Bridge tests use a temp LocalBackend with a temporary database.

Run with: pytest tests/test_memory_bridge.py -v
"""

from __future__ import annotations

import functools
import json
import os
import uuid

import pytest

from headroom.memory.bridge_config import BridgeConfig, MarkdownFormat
from headroom.memory.bridge_parsers import (
    ParsedSection,
    detect_format,
    extract_entities_from_text,
    extract_relationships_from_section,
    parse_chatgpt_facts,
    parse_claude_code_memory,
    parse_generic_markdown,
    parse_markdown,
)
from tests._skip_helpers import external_model_skip_reason

# Sample content for testing
CLAUDE_CODE_MEMORY = """\
# Project Memory

## Project Overview
- **Headroom**: Context optimization layer for LLM applications
- **Repos**: OSS at ~/claude-projects/headroom

## Key Architecture
- 186 Python files, 34 packages, 100K+ lines
- 6 compression algorithms: SmartCrusher, CacheAligner, ContentRouter

## Competitors
- Direct: Compresr (YC W26), Token Company
- Gateways: Portkey, Helicone, LiteLLM
"""

CHATGPT_FACTS = """\
User prefers Python over JavaScript
User works at Netflix
User likes dark mode
- User has a cat named Luna
"""

GENERIC_MARKDOWN = """\
# Notes

## Architecture
The system uses FastAPI for the proxy layer.
- SQLite for storage
- HNSW for vector search

## TODO
- Add caching layer
- Improve error handling
"""


def skip_offline_model_failures(func):
    """Skip bridge integration tests when the local embedder cannot start offline."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            reason = external_model_skip_reason(exc)
            if reason is not None:
                pytest.skip(reason)
            raise

    return wrapper


def decorate_async_test_methods(cls):
    """Wrap every async test method on a class with the offline-model skip helper."""
    for name, value in vars(cls).items():
        if name.startswith("test_"):
            setattr(cls, name, skip_offline_model_failures(value))
    return cls


def skip_if_offline_bridge_import_failed(stats) -> None:
    """Skip bridge assertions when every section failed only because the model cache is offline."""
    if (
        os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        and stats.sections_imported == 0
        and stats.sections_failed > 0
    ):
        pytest.skip("Skipped because required Hugging Face model files are unavailable offline")


# =============================================================================
# Parser Tests (pure functions, no backend)
# =============================================================================


class TestClaudeCodeParser:
    def test_parse_sections(self):
        parsed = parse_claude_code_memory(CLAUDE_CODE_MEMORY)
        # H1 + 3 H2 sections
        assert len(parsed.sections) >= 3
        assert parsed.format == "claude_code"

    def test_heading_levels(self):
        parsed = parse_claude_code_memory(CLAUDE_CODE_MEMORY)
        headings = {s.heading: s.heading_level for s in parsed.sections if s.heading}
        assert headings.get("Project Overview") == 2
        assert headings.get("Key Architecture") == 2
        assert headings.get("Competitors") == 2

    def test_bullets_become_facts(self):
        parsed = parse_claude_code_memory(CLAUDE_CODE_MEMORY)
        overview = next(s for s in parsed.sections if s.heading == "Project Overview")
        assert len(overview.facts) == 2
        assert any("Headroom" in f for f in overview.facts)
        assert any("Repos" in f for f in overview.facts)

    def test_bold_text_extracted_as_entities(self):
        parsed = parse_claude_code_memory(CLAUDE_CODE_MEMORY)
        overview = next(s for s in parsed.sections if s.heading == "Project Overview")
        assert "Headroom" in overview.entities
        assert "Repos" in overview.entities

    def test_content_hash_computed(self):
        parsed = parse_claude_code_memory(CLAUDE_CODE_MEMORY)
        for section in parsed.sections:
            if section.content:
                assert section.content_hash
                assert len(section.content_hash) == 64  # SHA-256

    def test_content_hash_deterministic(self):
        parsed1 = parse_claude_code_memory(CLAUDE_CODE_MEMORY)
        parsed2 = parse_claude_code_memory(CLAUDE_CODE_MEMORY)
        for s1, s2 in zip(parsed1.sections, parsed2.sections):
            assert s1.content_hash == s2.content_hash

    def test_file_hash_computed(self):
        parsed = parse_claude_code_memory(CLAUDE_CODE_MEMORY)
        assert parsed.file_hash
        assert len(parsed.file_hash) == 64


class TestChatGPTParser:
    def test_parse_flat_facts(self):
        parsed = parse_chatgpt_facts(CHATGPT_FACTS)
        assert parsed.format == "chatgpt"
        assert len(parsed.sections) == 1
        assert len(parsed.sections[0].facts) == 4

    def test_bullet_prefix_stripped(self):
        parsed = parse_chatgpt_facts(CHATGPT_FACTS)
        facts = parsed.sections[0].facts
        assert "User has a cat named Luna" in facts

    def test_empty_lines_skipped(self):
        content = "Fact 1\n\n\nFact 2\n\n"
        parsed = parse_chatgpt_facts(content)
        assert len(parsed.sections[0].facts) == 2

    def test_empty_content(self):
        parsed = parse_chatgpt_facts("")
        assert len(parsed.sections) == 0


class TestGenericParser:
    def test_parse_multi_level_headers(self):
        parsed = parse_generic_markdown(GENERIC_MARKDOWN)
        assert parsed.format == "generic"
        headings = [s.heading for s in parsed.sections if s.heading]
        assert "Architecture" in headings
        assert "TODO" in headings

    def test_non_bullet_lines_are_facts(self):
        parsed = parse_generic_markdown(GENERIC_MARKDOWN)
        arch = next(s for s in parsed.sections if s.heading == "Architecture")
        # "The system uses FastAPI..." and bullets should all be facts
        assert len(arch.facts) >= 3


class TestFormatDetection:
    def test_detect_claude_code(self):
        assert detect_format(CLAUDE_CODE_MEMORY) == "claude_code"

    def test_detect_chatgpt(self):
        assert detect_format(CHATGPT_FACTS) == "chatgpt"

    def test_detect_generic(self):
        content = "Some long paragraph without headers or bullet points that goes on and on describing things in great detail.\nAnother very long line that describes more things in this generic format."
        assert detect_format(content) in ("generic", "chatgpt")

    def test_empty_content(self):
        assert detect_format("") == "generic"


class TestAutoParser:
    def test_auto_parses_claude_code(self):
        parsed = parse_markdown(CLAUDE_CODE_MEMORY)
        assert parsed.format == "claude_code"

    def test_auto_parses_chatgpt(self):
        parsed = parse_markdown(CHATGPT_FACTS)
        assert parsed.format == "chatgpt"

    def test_force_format(self):
        parsed = parse_markdown(CLAUDE_CODE_MEMORY, format="generic")
        assert parsed.format == "generic"


class TestEntityExtraction:
    def test_bold_text(self):
        entities = extract_entities_from_text("I use **Python** and **FastAPI**")
        assert "Python" in entities
        assert "FastAPI" in entities

    def test_camel_case(self):
        entities = extract_entities_from_text("Using SmartCrusher and CacheAligner")
        assert "SmartCrusher" in entities
        assert "CacheAligner" in entities

    def test_no_false_positives_on_stop_words(self):
        entities = extract_entities_from_text("The system is very important and useful")
        # "The" and other stop words should not appear
        assert "The" not in entities

    def test_all_caps(self):
        entities = extract_entities_from_text("Using HNSW and SQLite")
        assert "HNSW" in entities


class TestRelationshipExtraction:
    def test_bold_colon_pattern(self):
        section = ParsedSection(
            heading="Test",
            heading_level=2,
            content="- **Headroom**: Context optimization layer",
            facts=["**Headroom**: Context optimization layer"],
        )
        rels = extract_relationships_from_section(section)
        assert len(rels) >= 1
        assert rels[0]["source"] == "Headroom"
        assert rels[0]["relationship"] == "is"

    def test_verb_patterns(self):
        section = ParsedSection(
            heading="Test",
            heading_level=2,
            content="Headroom uses SQLite for storage",
            facts=["Headroom uses SQLite for storage"],
        )
        rels = extract_relationships_from_section(section)
        uses_rels = [r for r in rels if r["relationship"] == "uses"]
        assert len(uses_rels) >= 1


# =============================================================================
# Bridge Tests (require backend)
# =============================================================================


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def user_id():
    """Unique user ID for test isolation."""
    return f"test_bridge_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def bridge_config(tmp_dir):
    """Create a BridgeConfig with test paths."""
    return BridgeConfig(
        user_id="test_user",
        sync_state_path=tmp_dir / "bridge_state.json",
        dedup_similarity_threshold=0.95,
    )


@pytest.fixture
async def backend(tmp_dir):
    """Create a LocalBackend with temp database."""
    from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

    config = LocalBackendConfig(db_path=str(tmp_dir / "test_memory.db"))
    backend = LocalBackend(config)
    await backend._ensure_initialized()
    yield backend
    await backend.close()


@pytest.fixture
def bridge(bridge_config, backend):
    """Create a MemoryBridge."""
    from headroom.memory.bridge import MemoryBridge

    return MemoryBridge(bridge_config, backend)


@decorate_async_test_methods
class TestMemoryBridgeImport:
    @pytest.mark.asyncio
    async def test_import_claude_code_memory(self, bridge, tmp_dir, backend):
        """Import a MEMORY.md file and verify memories are stored."""
        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text(CLAUDE_CODE_MEMORY, encoding="utf-8")

        stats = await bridge.import_from_markdown(paths=[md_path], user_id="test_user")
        skip_if_offline_bridge_import_failed(stats)

        assert stats.files_processed == 1
        assert stats.sections_imported > 0
        assert stats.total_facts > 0

        # Verify memories exist in backend
        memories = await backend.get_user_memories("test_user", limit=100)
        assert len(memories) > 0

    @pytest.mark.asyncio
    async def test_import_skips_unchanged_file(self, bridge, tmp_dir):
        """Second import of same file should skip (hash unchanged)."""
        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text(CLAUDE_CODE_MEMORY, encoding="utf-8")

        stats1 = await bridge.import_from_markdown(paths=[md_path], user_id="test_user")
        skip_if_offline_bridge_import_failed(stats1)
        assert stats1.sections_imported > 0

        stats2 = await bridge.import_from_markdown(paths=[md_path], user_id="test_user")
        assert stats2.files_skipped_unchanged == 1
        assert stats2.sections_imported == 0

    @pytest.mark.asyncio
    async def test_import_detects_changes(self, bridge, tmp_dir):
        """Modified file should re-import changed sections."""
        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text(CLAUDE_CODE_MEMORY, encoding="utf-8")

        await bridge.import_from_markdown(paths=[md_path], user_id="test_user")

        # Modify file
        modified = CLAUDE_CODE_MEMORY + "\n## New Section\n- Brand new fact\n"
        md_path.write_text(modified, encoding="utf-8")

        stats = await bridge.import_from_markdown(paths=[md_path], user_id="test_user")
        skip_if_offline_bridge_import_failed(stats)
        assert stats.files_processed == 1
        assert stats.sections_imported >= 1  # At least the new section

    @pytest.mark.asyncio
    async def test_import_force(self, bridge, tmp_dir):
        """Force import should re-import even if unchanged."""
        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text(CLAUDE_CODE_MEMORY, encoding="utf-8")

        await bridge.import_from_markdown(paths=[md_path], user_id="test_user")

        stats = await bridge.import_from_markdown(paths=[md_path], user_id="test_user", force=True)
        # Force should process the file, though sections may be deduped by semantic search
        assert stats.files_processed == 1

    @pytest.mark.asyncio
    async def test_import_chatgpt_facts(self, bridge, tmp_dir, backend):
        """Import ChatGPT-style facts."""
        md_path = tmp_dir / "chatgpt.txt"
        md_path.write_text(CHATGPT_FACTS, encoding="utf-8")

        bridge._config.md_format = MarkdownFormat.CHATGPT
        stats = await bridge.import_from_markdown(paths=[md_path], user_id="test_user")
        skip_if_offline_bridge_import_failed(stats)
        assert stats.sections_imported > 0

    @pytest.mark.asyncio
    async def test_import_missing_file(self, bridge, tmp_dir):
        """Missing file should be skipped gracefully."""
        from pathlib import Path

        stats = await bridge.import_from_markdown(
            paths=[Path(tmp_dir / "nonexistent.md")], user_id="test_user"
        )
        assert stats.files_processed == 0

    @pytest.mark.asyncio
    async def test_metadata_preserved(self, bridge, tmp_dir, backend):
        """Imported memories should have bridge metadata."""
        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text(CLAUDE_CODE_MEMORY, encoding="utf-8")

        await bridge.import_from_markdown(paths=[md_path], user_id="test_user")

        memories = await backend.get_user_memories("test_user", limit=100)
        for memory in memories:
            metadata = memory.metadata or {}
            assert metadata.get("source") == "memory_bridge"
            assert "source_file" in metadata


@decorate_async_test_methods
class TestMemoryBridgeExport:
    @pytest.mark.asyncio
    async def test_export_claude_code_style(self, bridge, tmp_dir, backend):
        """Export memories as Claude Code style markdown."""
        # Add some memories
        await backend.save_memory(
            content="Headroom is a context optimization layer",
            user_id="test_user",
            importance=0.8,
            metadata={"section_heading": "Overview"},
        )
        await backend.save_memory(
            content="Uses SQLite for storage",
            user_id="test_user",
            importance=0.7,
            metadata={"section_heading": "Architecture"},
        )

        export_path = tmp_dir / "export.md"
        markdown = await bridge.export_to_markdown(
            path=export_path,
            user_id="test_user",
            format=MarkdownFormat.CLAUDE_CODE,
        )

        assert "# Memory" in markdown
        assert "## Overview" in markdown
        assert "## Architecture" in markdown
        assert "Headroom" in markdown
        assert export_path.exists()

    @pytest.mark.asyncio
    async def test_export_chatgpt_style(self, bridge, backend):
        """Export as flat facts."""
        await backend.save_memory(
            content="User prefers Python",
            user_id="test_user",
            importance=0.7,
        )

        markdown = await bridge.export_to_markdown(
            user_id="test_user",
            format=MarkdownFormat.CHATGPT,
        )

        assert "User prefers Python" in markdown
        # Should NOT have headers
        assert "## " not in markdown

    @pytest.mark.asyncio
    async def test_export_empty(self, bridge):
        """Export with no memories should produce placeholder."""
        markdown = await bridge.export_to_markdown(user_id="nonexistent_user")
        assert "No memories" in markdown


@decorate_async_test_methods
class TestMemoryBridgeSync:
    @pytest.mark.asyncio
    async def test_sync_imports_and_exports(self, bridge, tmp_dir, backend):
        """Full sync: import from file, add organic memory, sync exports it."""
        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text("## Facts\n- User likes Python\n", encoding="utf-8")
        bridge._config.md_paths = [md_path]

        # First sync: imports from file
        stats = await bridge.sync(user_id="test_user")
        skip_if_offline_bridge_import_failed(stats.import_stats)
        assert stats.import_stats.sections_imported > 0

        # Add an organic memory (not from bridge)
        await backend.save_memory(
            content="User also likes Rust",
            user_id="test_user",
            importance=0.7,
            metadata={},  # No source tag = organic
        )

        # Second sync: should export the organic memory
        stats2 = await bridge.sync(user_id="test_user")
        assert stats2.memories_exported >= 1

        # Verify the file now contains the new memory
        updated_content = md_path.read_text(encoding="utf-8")
        assert "Rust" in updated_content

    @pytest.mark.asyncio
    async def test_source_tag_prevents_reexport(self, bridge, tmp_dir, backend):
        """Memories imported via bridge should not be re-exported."""
        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text("## Facts\n- Imported fact\n", encoding="utf-8")
        bridge._config.md_paths = [md_path]

        # Import
        await bridge.sync(user_id="test_user")

        # Sync again - nothing should be exported (all memories have source tag)
        stats = await bridge.sync(user_id="test_user")
        assert stats.memories_exported == 0


@decorate_async_test_methods
class TestSyncStatePersistence:
    @pytest.mark.asyncio
    async def test_state_saved_and_loaded(self, tmp_dir, backend):
        """Sync state should persist across bridge instances."""
        from headroom.memory.bridge import MemoryBridge

        state_path = tmp_dir / "state.json"
        config = BridgeConfig(
            user_id="test_user",
            sync_state_path=state_path,
        )

        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text(CLAUDE_CODE_MEMORY, encoding="utf-8")

        # First bridge instance: import
        bridge1 = MemoryBridge(config, backend)
        await bridge1.import_from_markdown(paths=[md_path], user_id="test_user")

        # Verify state file exists
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert "files" in state
        assert str(md_path) in state["files"]

        # Second bridge instance: should detect unchanged file
        bridge2 = MemoryBridge(config, backend)
        stats = await bridge2.import_from_markdown(paths=[md_path], user_id="test_user")
        assert stats.files_skipped_unchanged == 1


@decorate_async_test_methods
class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_import_export_preserves_facts(self, bridge, tmp_dir, backend):
        """Import a MEMORY.md, export it, verify all facts are present."""
        md_path = tmp_dir / "MEMORY.md"
        md_path.write_text(CLAUDE_CODE_MEMORY, encoding="utf-8")

        # Import
        await bridge.import_from_markdown(paths=[md_path], user_id="test_user")

        # Export
        export_path = tmp_dir / "exported.md"
        markdown = await bridge.export_to_markdown(
            path=export_path,
            user_id="test_user",
            format=MarkdownFormat.CLAUDE_CODE,
        )

        # Key facts should survive the round trip
        assert "Headroom" in markdown
        assert "compression" in markdown.lower() or "SmartCrusher" in markdown
        assert "Compresr" in markdown or "Portkey" in markdown
