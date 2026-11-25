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
