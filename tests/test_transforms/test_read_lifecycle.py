"""Tests for ReadLifecycleManager - event-driven Read lifecycle management.

Tests covering:
- Disabled by default (backward compatibility)
- Stale detection (file edited after Read)
- Superseded detection (file re-Read)
- Fresh Reads untouched
- Multiple files and complex chains
- OpenAI and Anthropic message formats
- CCR store integration
- Size gating
"""

import json

from headroom.config import ReadLifecycleConfig
from headroom.transforms.read_lifecycle import (
    ReadLifecycleManager,
)

# =============================================================================
# Helpers
# =============================================================================


def make_openai_read(tool_call_id: str, file_path: str) -> dict:
    """Create an OpenAI-format assistant message with a Read tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": json.dumps({"file_path": file_path}),
                },
            }
        ],
    }


def make_openai_edit(tool_call_id: str, file_path: str) -> dict:
    """Create an OpenAI-format assistant message with an Edit tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "Edit",
                    "arguments": json.dumps(
                        {
                            "file_path": file_path,
                            "old_string": "old",
                            "new_string": "new",
                        }
                    ),
                },
            }
        ],
    }


def make_openai_write(tool_call_id: str, file_path: str) -> dict:
    """Create an OpenAI-format assistant message with a Write tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "Write",
                    "arguments": json.dumps({"file_path": file_path, "content": "new content"}),
                },
            }
        ],
    }


def make_openai_tool_result(tool_call_id: str, content: str) -> dict:
    """Create an OpenAI-format tool result message."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def make_anthropic_read(tool_call_id: str, file_path: str) -> dict:
    """Create an Anthropic-format assistant message with a Read tool call."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_call_id,
                "name": "Read",
                "input": {"file_path": file_path},
            }
        ],
    }


def make_anthropic_edit(tool_call_id: str, file_path: str) -> dict:
    """Create an Anthropic-format assistant message with an Edit tool call."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_call_id,
                "name": "Edit",
                "input": {
                    "file_path": file_path,
                    "old_string": "old",
                    "new_string": "new",
                },
            }
        ],
    }


def make_anthropic_tool_result(tool_call_id: str, content: str) -> dict:
    """Create an Anthropic-format user message with a tool_result block."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content,
            }
        ],
    }


LARGE_CONTENT = "x" * 2000  # Well above min_size_bytes
SMALL_CONTENT = "tiny"  # Below min_size_bytes


# =============================================================================
# Tests
# =============================================================================


class TestReadLifecycleDisabled:
    """Verify backward compatibility when disabled."""

    def test_disabled_when_explicitly_off(self):
        """Explicitly disabled config: no changes to messages."""
        config = ReadLifecycleConfig(enabled=False)
        assert config.enabled is False

        mgr = ReadLifecycleManager(config)
        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
        ]

        result = mgr.apply(messages)
        assert result.messages is messages  # Same object, not copied
        assert result.reads_total == 0
        assert result.transforms_applied == []

    def test_enabled_by_default(self):
        """Default config has lifecycle enabled."""
        config = ReadLifecycleConfig()
        assert config.enabled is True


class TestStaleDetection:
    """Read outputs become stale when the file is subsequently edited."""

    def test_read_then_edit_makes_stale(self):
        """Read(A) → Edit(A): Read becomes stale."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "edit success"),
        ]

        result = mgr.apply(messages)
        assert result.reads_stale == 1
        assert result.reads_fresh == 0
        # Read content should be replaced with marker
        tool_result = result.messages[1]
        assert "stale" in tool_result["content"].lower()
        assert "/src/app.py" in tool_result["content"]
        assert "hash=" in tool_result["content"]

    def test_write_makes_read_stale(self):
        """Read(A) → Write(A): Read becomes stale."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_write("w1", "/src/app.py"),
            make_openai_tool_result("w1", "write success"),
        ]

        result = mgr.apply(messages)
        assert result.reads_stale == 1
        assert "stale" in result.messages[1]["content"].lower()

    def test_edit_different_file_not_stale(self):
        """Read(A) → Edit(B): Read(A) stays fresh."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/other.py"),
            make_openai_tool_result("e1", "edit success"),
        ]

        result = mgr.apply(messages)
        assert result.reads_stale == 0
        assert result.reads_fresh == 1
        assert result.messages[1]["content"] == LARGE_CONTENT

    def test_multiple_reads_all_stale(self):
        """Read(A) × 3 → Edit(A): all 3 Reads become stale."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_read("r2", "/src/app.py"),
            make_openai_tool_result("r2", LARGE_CONTENT + "_v2"),
            make_openai_read("r3", "/src/app.py"),
            make_openai_tool_result("r3", LARGE_CONTENT + "_v3"),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "edit success"),
        ]

        result = mgr.apply(messages)
        # All 3 reads are stale (edit happened after all of them)
        assert result.reads_stale == 3
        assert result.reads_fresh == 0

    def test_compress_stale_disabled(self):
        """compress_stale=False: stale Reads are not replaced."""
        config = ReadLifecycleConfig(enabled=True, compress_stale=False)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "edit success"),
        ]

        result = mgr.apply(messages)
        # With compress_stale=False but compress_superseded=True,
        # Read is superseded by nothing (only one read), and not stale → fresh
        assert result.reads_fresh == 1
        assert result.messages[1]["content"] == LARGE_CONTENT


class TestSupersededDetection:
    """Read outputs become superseded when the same file is re-Read."""

    def test_reread_makes_superseded(self):
        """Read(A) → Read(A): first Read becomes superseded."""
        config = ReadLifecycleConfig(enabled=True, compress_superseded=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_read("r2", "/src/app.py"),
            make_openai_tool_result("r2", LARGE_CONTENT + "_updated"),
        ]

        result = mgr.apply(messages)
        assert result.reads_superseded == 1
        assert result.reads_fresh == 1
        # First read replaced, second read untouched
        assert "superseded" in result.messages[1]["content"].lower()
        assert result.messages[3]["content"] == LARGE_CONTENT + "_updated"

    def test_compress_superseded_disabled(self):
        """compress_superseded=False: superseded Reads not replaced."""
        config = ReadLifecycleConfig(enabled=True, compress_superseded=False)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_read("r2", "/src/app.py"),
            make_openai_tool_result("r2", LARGE_CONTENT + "_updated"),
        ]

        result = mgr.apply(messages)
        # Both reads are fresh (superseded detection disabled)
        assert result.reads_fresh == 2
        assert result.messages[1]["content"] == LARGE_CONTENT


class TestFreshReads:
    """Fresh Reads must never be modified."""

    def test_single_read_stays_fresh(self):
        """One Read, no Edit: stays fresh."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
        ]

        result = mgr.apply(messages)
        assert result.reads_fresh == 1
        assert result.reads_stale == 0
        assert result.reads_superseded == 0
        assert result.messages[1]["content"] == LARGE_CONTENT

    def test_read_edit_read_chain(self):
        """Read(A) → Edit(A) → Read(A): first stale, second fresh."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "edit success"),
            make_openai_read("r2", "/src/app.py"),
            make_openai_tool_result("r2", LARGE_CONTENT + "_v2"),
        ]

        result = mgr.apply(messages)
        # First read: stale (edit happened after) AND superseded (re-read after)
        # → classified as stale (stale takes priority)
        assert result.reads_stale == 1
        # Second read: fresh (latest, no edit after)
        assert result.reads_fresh == 1
        assert "stale" in result.messages[1]["content"].lower()
        assert result.messages[5]["content"] == LARGE_CONTENT + "_v2"


class TestMultipleFiles:
    """Lifecycle management across multiple files."""

    def test_independent_files(self):
        """Read(A) → Edit(A) → Read(B): A stale, B fresh."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "edit success"),
            make_openai_read("r2", "/src/utils.py"),
            make_openai_tool_result("r2", LARGE_CONTENT + "_utils"),
        ]

        result = mgr.apply(messages)
        assert result.reads_stale == 1
        assert result.reads_fresh == 1
        assert "stale" in result.messages[1]["content"].lower()
        assert result.messages[5]["content"] == LARGE_CONTENT + "_utils"


class TestSizeGating:
    """Small Read outputs should be skipped."""

    def test_small_read_not_replaced(self):
        """Read output below min_size_bytes: not replaced even if stale."""
        config = ReadLifecycleConfig(enabled=True, min_size_bytes=512)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", SMALL_CONTENT),  # 4 bytes
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "edit success"),
        ]

        result = mgr.apply(messages)
        # Stale but too small to replace
        assert result.messages[1]["content"] == SMALL_CONTENT


class TestAnthropicFormat:
    """Lifecycle works with Anthropic message format."""

    def test_anthropic_stale_read(self):
        """Anthropic format: Read(A) → Edit(A): Read becomes stale."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_anthropic_read("r1", "/src/app.py"),
            make_anthropic_tool_result("r1", LARGE_CONTENT),
            make_anthropic_edit("e1", "/src/app.py"),
            make_anthropic_tool_result("e1", "edit success"),
        ]

        result = mgr.apply(messages)
        assert result.reads_stale == 1
        # Check the tool_result block inside the user message was replaced
        user_msg = result.messages[1]
        tool_result_block = user_msg["content"][0]
        assert "stale" in tool_result_block["content"].lower()
        assert "hash=" in tool_result_block["content"]

    def test_anthropic_fresh_read(self):
        """Anthropic format: single Read stays fresh."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_anthropic_read("r1", "/src/app.py"),
            make_anthropic_tool_result("r1", LARGE_CONTENT),
        ]

        result = mgr.apply(messages)
        assert result.reads_fresh == 1
        user_msg = result.messages[1]
        assert user_msg["content"][0]["content"] == LARGE_CONTENT


class TestCCRStoreIntegration:
    """Lifecycle manager stores originals in CCR."""

    def test_original_stored_in_ccr(self):
        """When a Read is replaced, original content is stored in CCR."""

        class MockStore:
            def __init__(self):
                self.stored = []

            def store(self, **kwargs):
                self.stored.append(kwargs)
                return "mock_hash_1234567890ab"

        mock_store = MockStore()
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config, compression_store=mock_store)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "edit success"),
        ]

        result = mgr.apply(messages)
        assert len(mock_store.stored) == 1
        assert mock_store.stored[0]["original"] == LARGE_CONTENT
        assert mock_store.stored[0]["tool_name"] == "Read"
        assert "mock_hash_1234567890ab" in result.messages[1]["content"]
        assert result.ccr_hashes == ["mock_hash_1234567890ab"]

    def test_no_store_uses_content_hash(self):
        """Without CCR store, marker uses content-derived hash."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config, compression_store=None)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "edit success"),
        ]

        result = mgr.apply(messages)
        assert "hash=" in result.messages[1]["content"]


class TestTransformTracking:
    """Lifecycle transforms are tracked correctly."""

    def test_transforms_recorded(self):
        """Each replacement generates a transform entry."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_read("r2", "/src/app.py"),
            make_openai_tool_result("r2", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "done"),
        ]

        result = mgr.apply(messages)
        stale_transforms = [t for t in result.transforms_applied if "stale" in t]
        assert len(stale_transforms) == 2  # Both reads are stale

    def test_transform_tag_includes_file_path_openai(self):
        """OpenAI-format tag shape is ``read_lifecycle:<state>:<file_path>``."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)
        messages = [
            make_openai_read("r1", "/src/app.py"),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "done"),
        ]

        result = mgr.apply(messages)
        assert "read_lifecycle:stale:/src/app.py" in result.transforms_applied

    def test_transform_tag_includes_file_path_anthropic(self):
        """Anthropic-format tag shape matches OpenAI tag shape."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "r1",
                        "name": "Read",
                        "input": {"file_path": "/src/notes.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "r1", "content": LARGE_CONTENT}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "e1",
                        "name": "Edit",
                        "input": {
                            "file_path": "/src/notes.md",
                            "old_string": "old",
                            "new_string": "new",
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "e1", "content": "done"}],
            },
        ]

        result = mgr.apply(messages)
        assert "read_lifecycle:stale:/src/notes.md" in result.transforms_applied

    def test_transform_tag_preserves_colons_in_path(self):
        """Paths containing ``:`` survive — consumers must bound their split."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)
        weird_path = "/tmp/has:colon/file.py"
        messages = [
            make_openai_read("r1", weird_path),
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", weird_path),
            make_openai_tool_result("e1", "done"),
        ]

        result = mgr.apply(messages)
        tag = next(t for t in result.transforms_applied if t.startswith("read_lifecycle:stale"))
        assert tag.split(":", 2) == ["read_lifecycle", "stale", weird_path]


class TestNoFilePathHandling:
    """Reads without parseable file_path should be left alone."""

    def test_read_without_file_path(self):
        """Read with no file_path in arguments: treated as unknown, not matched."""
        config = ReadLifecycleConfig(enabled=True)
        mgr = ReadLifecycleManager(config)

        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "r1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": "{}"},
                    }
                ],
            },
            make_openai_tool_result("r1", LARGE_CONTENT),
            make_openai_edit("e1", "/src/app.py"),
            make_openai_tool_result("e1", "done"),
        ]

        result = mgr.apply(messages)
        # Can't match file_path, so Read is not classified at all
        assert result.reads_total == 0
        assert result.messages[1]["content"] == LARGE_CONTENT
