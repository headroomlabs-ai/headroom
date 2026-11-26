"""Claude memory metadata must remain strings for a YAML consumer."""

from pathlib import Path

import pytest
import yaml

from headroom.memory.sync_adapters.claude_code import (
    ClaudeCodeAdapter,
    _build_frontmatter,
    _parse_frontmatter,
)


@pytest.mark.parametrize(
    "value",
    [
        "Deployment: use blue-green",
        "# keep this note",
        "[alpha, beta]",
        "true",
        "null",
        "2026-11-25",
        'Say "hello" # literally',
        '"quoted title"',
        r"Use C:\work\notes",
        "保留 Unicode: 项目说明",
        "first\nsecond",
        "first\x85second",
        "first\u2028second\u2029third",
        "terminal\x7foutput",
        "terminal\x81output\x9f",
        "first\ufffesecond\uffff",
    ],
)
def test_frontmatter_values_are_yaml_strings(value: str) -> None:
    fields = {"name": value, "type": "project"}
    rendered = _build_frontmatter(fields)
    assert yaml.safe_load(rendered.split("---", 2)[1]) == fields
    metadata, body = _parse_frontmatter(f"{rendered}\n\nOriginal body")
    assert metadata == fields
    assert body == "Original body"


@pytest.mark.asyncio
async def test_written_memory_is_readable_as_yaml_and_round_trips(tmp_path: Path) -> None:
    adapter = ClaudeCodeAdapter(tmp_path)
    content = 'Deployment: use "blue-green" # production\nKeep the rollback pool available.'
    assert await adapter.write_memories([{"content": content, "headroom_id": "memory-1"}]) == 1
    written = next(tmp_path.glob("headroom_*.md"))
    raw = written.read_text(encoding="utf-8")
    metadata = yaml.safe_load(raw.split("---", 2)[1])
    assert metadata["name"] == content.split("\n")[0]
    assert metadata["description"] == content.replace("\n", " ")[:100]
    assert metadata["headroom_id"] == "memory-1"
    loaded = await adapter.read_memories()
    assert loaded[0].content == content
    assert loaded[0].metadata["name"] == metadata["name"]


@pytest.mark.asyncio
async def test_numeric_source_agent_keeps_the_batch_exportable(tmp_path: Path) -> None:
    adapter = ClaudeCodeAdapter(tmp_path)
    assert (
        await adapter.write_memories(
            [
                {"content": "First memory", "headroom_id": "first", "source_agent": 42},
                {"content": "Second memory", "headroom_id": "second", "source_agent": "codex"},
            ]
        )
        == 2
    )
    loaded = await adapter.read_memories()
    assert len(loaded) == 2
    first = next(memory for memory in loaded if memory.metadata["headroom_id"] == "first")
    assert first.metadata["source_agent"] == "42"
    raw = next(tmp_path.glob("headroom_first_memory.md")).read_text(encoding="utf-8")
    assert yaml.safe_load(raw.split("---", 2)[1])["source_agent"] == "42"


def test_unmarked_legacy_path_keeps_literal_backslashes() -> None:
    content = '---\nname: "C:\\temp\\notes"\ndescription: "C:\\temp\\notes"\n---\n\nBody'
    metadata, body = _parse_frontmatter(content)
    assert metadata == {"name": r"C:\temp\notes", "description": r"C:\temp\notes"}
    assert body == "Body"


def test_marked_export_preserves_quoted_paths() -> None:
    fields = {"name": r'"C:\temp\notes"', "description": r"C:\temp\notes"}
    rendered = _build_frontmatter(fields)
    assert yaml.safe_load(rendered.split("---", 2)[1]) == fields
    assert _parse_frontmatter(rendered + "\n\nBody") == (fields, "Body")


def test_falsey_metadata_keeps_legacy_omission() -> None:
    rendered = _build_frontmatter({"name": "memory", "source_agent": 0, "empty": ""})
    assert yaml.safe_load(rendered.split("---", 2)[1]) == {"name": "memory"}
    assert _parse_frontmatter(rendered + "\n\nBody") == ({"name": "memory"}, "Body")
