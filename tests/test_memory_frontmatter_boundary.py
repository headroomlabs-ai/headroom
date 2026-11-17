"""A horizontal separator in metadata must not end YAML frontmatter."""

import hashlib

import pytest

from headroom.memory.sync_adapters.claude_code import ClaudeCodeAdapter, _parse_frontmatter


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_embedded_dashes_are_not_a_closing_delimiter(newline):
    text = newline.join(["---", "name: foo---bar", "type: project", "---", "", "Original body"])
    metadata, body = _parse_frontmatter(text)
    assert metadata == {"name": "foo---bar", "type": "project"}
    assert body == "Original body"


@pytest.mark.parametrize("opening", ["---not-a-header", "----"])
def test_non_delimiter_opening_preserves_the_original_body(opening):
    text = f"{opening}\nname: text\n---\nbody"
    assert _parse_frontmatter(text) == ({}, text)


def test_closing_delimiter_must_occupy_the_line():
    text = "---\nname: value\n---suffix\ntype: project\n---\nBody\n---\nTail"
    metadata, body = _parse_frontmatter(text)
    assert metadata == {"name": "value", "type": "project"}
    assert body == "Body\n---\nTail"


def test_missing_delimiter_preserves_input():
    text = "---\nname: foo---bar\nbody without closing delimiter"
    assert _parse_frontmatter(text) == ({}, text)


@pytest.mark.asyncio
async def test_real_adapter_round_trips_dashes_in_generated_metadata(tmp_path):
    adapter = ClaudeCodeAdapter(tmp_path / "memory")
    content = "Use foo---bar as the separator\nKeep the second line intact."
    original_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    assert await adapter.write_memories([{"content": content, "content_hash": original_hash}]) == 1
    reread = await adapter.read_memories()
    assert len(reread) == 1
    assert reread[0].content == content
    assert reread[0].content_hash == original_hash
    assert await adapter.write_memories([{"content": content, "content_hash": original_hash}]) == 0
