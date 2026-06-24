"""Tests for text-based compressors (coding task support).

Tests content detection, search compressor, and log compressor.
"""

from headroom.transforms import (
    ContentType,
    LogCompressor,
    LogCompressorConfig,
    SearchCompressor,
    SearchCompressorConfig,
    detect_content_type,
)


class TestContentDetector:
    """Tests for content type detection."""

    def test_detect_json_array_of_dicts(self):
        """JSON arrays of dicts are detected correctly."""
        content = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
        result = detect_content_type(content)
        assert result.content_type == ContentType.JSON_ARRAY
        assert result.confidence >= 0.8
        assert result.metadata.get("is_dict_array") is True

    def test_detect_json_array_non_dict(self):
        """JSON arrays of non-dicts are detected."""
        content = "[1, 2, 3, 4, 5]"
        result = detect_content_type(content)
        assert result.content_type == ContentType.JSON_ARRAY
        assert result.metadata.get("is_dict_array") is False

    def test_detect_search_results(self):
        """grep-style search results are detected."""
        content = """src/main.py:42:def process_data(items):
src/main.py:43:    \"\"\"Process items.\"\"\"
src/utils.py:15:def validate(data):
src/utils.py:16:    return data is not None
src/models.py:100:class DataProcessor:
"""
        result = detect_content_type(content)
        assert result.content_type == ContentType.SEARCH_RESULTS
        assert result.confidence >= 0.6

    def test_detect_build_output(self):
        """Build/test output is detected."""
        content = """
============================= test session starts ==============================
platform darwin -- Python 3.11.0
collected 15 items

tests/test_foo.py::test_basic PASSED
tests/test_foo.py::test_edge_case FAILED
tests/test_bar.py::test_another ERROR

=================================== FAILURES ===================================
tests/test_foo.py::test_edge_case - AssertionError: expected 5, got 3

=========================== short test summary info ============================
FAILED tests/test_foo.py::test_edge_case
ERROR tests/test_bar.py::test_another
========================= 1 failed, 1 passed, 1 error =========================
"""
        result = detect_content_type(content)
        assert result.content_type == ContentType.BUILD_OUTPUT
        assert result.confidence >= 0.5

    def test_detect_git_diff(self):
        """Git diff format is detected."""
        content = """diff --git a/src/main.py b/src/main.py
index abc123..def456 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@ def process():
-    old_line = True
+    new_line = True
     unchanged = "same"
"""
        result = detect_content_type(content)
        assert result.content_type == ContentType.GIT_DIFF
        assert result.confidence >= 0.7

    def test_detect_python_code(self):
        """Python source code is detected."""
        content = """
import json
from typing import Any

def process_data(items: list[dict]) -> dict[str, Any]:
    \"\"\"Process a list of items.

    Args:
        items: List of dictionaries to process.

    Returns:
        Processed result dictionary.
    \"\"\"
    result = {}
    for item in items:
        key = item.get("id")
        result[key] = item
    return result

class DataProcessor:
    def __init__(self):
        self.cache = {}

    async def async_process(self, data):
        return await self._do_process(data)
"""
        result = detect_content_type(content)
        assert result.content_type == ContentType.SOURCE_CODE
        assert result.metadata.get("language") == "python"

    def test_detect_plain_text(self):
        """Plain text falls back correctly."""
        content = """This is just some random text
that doesn't match any specific pattern.
It's just prose, really.
Nothing special about it."""
        result = detect_content_type(content)
        assert result.content_type == ContentType.PLAIN_TEXT


class TestSearchCompressor:
    """Tests for search results compression."""

    def test_compress_search_results(self):
        """Search results are compressed."""
        content = "\n".join([f"src/file{i}.py:{i * 10}:def function_{i}():" for i in range(100)])

        compressor = SearchCompressor()
        result = compressor.compress(content, context="find function_50")

        assert result.original_match_count == 100
        assert result.compressed_match_count < 100
        assert "function_" in result.compressed

    def test_keeps_first_and_last(self):
        """First and last matches are preserved."""
        content = "\n".join([f"src/file.py:{i}:line {i}" for i in range(1, 101)])

        compressor = SearchCompressor(
            config=SearchCompressorConfig(
                always_keep_first=True,
                always_keep_last=True,
            )
        )
        result = compressor.compress(content)

        assert "src/file.py:1:line 1" in result.compressed
        assert "src/file.py:100:line 100" in result.compressed

    def test_prioritizes_errors(self):
        """Error lines are prioritized."""
        lines = [f"src/file.py:{i}:normal line" for i in range(1, 50)]
        lines.append("src/file.py:50:ERROR: something failed")
        lines.extend([f"src/file.py:{i}:normal line" for i in range(51, 100)])
        content = "\n".join(lines)

        compressor = SearchCompressor()
        result = compressor.compress(content)

        assert "ERROR: something failed" in result.compressed

    def test_small_results_unchanged(self):
        """Small search results pass through unchanged."""
        content = "src/file.py:1:def foo():\nsrc/file.py:2:    pass"

        compressor = SearchCompressor()
        result = compressor.compress(content)

        assert result.compression_ratio == 1.0
        assert result.compressed == content


class TestLogCompressor:
    """Tests for log/build output compression."""

    def test_compress_pytest_output(self):
        """pytest output is compressed."""
        lines = ["=" * 40 + " test session starts " + "=" * 40]
        lines.append("collected 100 items")
        lines.extend([f"tests/test_{i}.py::test_case_{i} PASSED" for i in range(95)])
        lines.extend(
            [
                "tests/test_fail.py::test_case_fail FAILED",
                "",
                "=" * 40 + " FAILURES " + "=" * 40,
                "tests/test_fail.py::test_case_fail",
                "AssertionError: expected True, got False",
                "",
                "=" * 40 + " short test summary " + "=" * 40,
                "FAILED tests/test_fail.py::test_case_fail",
                "1 failed, 95 passed",
            ]
        )
        content = "\n".join(lines)

        compressor = LogCompressor()
        result = compressor.compress(content)

        # Should keep failures and summary
        assert "FAILED" in result.compressed
        assert "AssertionError" in result.compressed
        assert result.compression_ratio < 0.5

    def test_keeps_errors_and_stack_traces(self):
        """Errors and stack traces are preserved."""
        content = """
INFO: Starting process
INFO: Loading data
INFO: Processing item 1
INFO: Processing item 2
ERROR: Failed to process item 3
Traceback (most recent call last):
  File "main.py", line 42, in process
    result = compute(data)
  File "utils.py", line 15, in compute
    return data / 0
ZeroDivisionError: division by zero
INFO: Continuing with remaining items
INFO: Done
"""
        compressor = LogCompressor()
        result = compressor.compress(content)

        assert "ERROR: Failed to process" in result.compressed
        assert "Traceback" in result.compressed
        assert "ZeroDivisionError" in result.compressed

    def test_small_logs_unchanged(self):
        """Small logs pass through unchanged."""
        content = "INFO: Starting\nINFO: Done"

        compressor = LogCompressor(config=LogCompressorConfig(min_lines_for_ccr=100))
        result = compressor.compress(content)

        assert result.compression_ratio == 1.0


class TestSmartCrusherTextIntegration:
    """Tests for SmartCrusher behavior with different content types.

    NOTE: SmartCrusher is designed for JSON compression only.
    Plain text content (search results, logs, etc.) passes through UNCHANGED.

    Text compression utilities (SearchCompressor, LogCompressor) are
    available as standalone tools for applications to use explicitly.
    This is intentional - text compression is opt-in, not automatic.
    """

    @staticmethod
    def _get_tokenizer():
        """Get a tokenizer for tests using OpenAI provider."""
        from headroom.providers import OpenAIProvider
        from headroom.tokenizer import Tokenizer

        provider = OpenAIProvider()
        token_counter = provider.get_token_counter("gpt-4o")
        return Tokenizer(token_counter, "gpt-4o")

    def test_smart_crusher_passes_through_search_results_unchanged(self):
        """SmartCrusher passes non-JSON search results through unchanged.

        Applications should use SearchCompressor directly if compression is needed.
        """
        from headroom.transforms import SmartCrusher, SmartCrusherConfig

        # Create search results content
        search_results = "\n".join([f"src/file{i}.py:{i}:def function_{i}():" for i in range(100)])

        messages = [
            {"role": "user", "content": "Find all function definitions"},
            {"role": "tool", "content": search_results},
        ]

        crusher = SmartCrusher(config=SmartCrusherConfig(min_tokens_to_crush=10))
        tokenizer = self._get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Non-JSON passes through UNCHANGED - this is correct behavior
        tool_content = result.messages[1]["content"]
        assert tool_content == search_results

    def test_smart_crusher_passes_through_log_output_unchanged(self):
        """SmartCrusher passes non-JSON log output through unchanged.

        Applications should use LogCompressor directly if compression is needed.
        """
        from headroom.transforms import SmartCrusher, SmartCrusherConfig

        # Create log content
        lines = ["INFO: Processing item " + str(i) for i in range(100)]
        lines.append("ERROR: Critical failure at item 50")
        lines.append("Traceback (most recent call last):")
        lines.append('  File "main.py", line 100, in process')
        lines.append("RuntimeError: something broke")
        log_content = "\n".join(lines)

        messages = [
            {"role": "user", "content": "Run the tests"},
            {"role": "tool", "content": log_content},
        ]

        crusher = SmartCrusher(config=SmartCrusherConfig(min_tokens_to_crush=10))
        tokenizer = self._get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Non-JSON passes through UNCHANGED - this is correct behavior
        tool_content = result.messages[1]["content"]
        assert tool_content == log_content

    def test_search_compressor_available_as_standalone(self):
        """SearchCompressor is available for explicit use by applications."""
        # Create search results content
        search_results = "\n".join([f"src/file{i}.py:{i}:def function_{i}():" for i in range(100)])

        # Application explicitly chooses to compress
        compressor = SearchCompressor()
        result = compressor.compress(search_results, context="find function_50")

        # Compression happens when explicitly requested
        assert result.original_match_count == 100
        assert result.compressed_match_count < 100
        assert "function_" in result.compressed

    def test_log_compressor_available_as_standalone(self):
        """LogCompressor is available for explicit use by applications."""
        # Create log content
        lines = ["INFO: Processing item " + str(i) for i in range(100)]
        lines.append("ERROR: Critical failure at item 50")
        lines.append("Traceback (most recent call last):")
        lines.append('  File "main.py", line 100, in process')
        lines.append("RuntimeError: something broke")
        log_content = "\n".join(lines)

        # Application explicitly chooses to compress
        compressor = LogCompressor()
        result = compressor.compress(log_content)

        # Compression happens when explicitly requested, errors preserved
        assert "ERROR: Critical failure" in result.compressed
        assert "RuntimeError" in result.compressed
        assert result.compression_ratio < 1.0  # Some compression occurred

    def test_smart_crusher_json_still_works(self):
        """SmartCrusher still handles JSON correctly.

        Asserts the legacy lossy + JSON-shape behavior: output is a
        JSON-parseable array. The PR4 lossless default substitutes a
        CSV+schema STRING for tabular arrays, which doesn't round-trip
        as a JSON array — that's tested separately in
        `test_smart_crusher_lossless_default.py`.
        """
        import json
        import re

        from headroom.transforms import SmartCrusher, SmartCrusherConfig

        # Create JSON array content with larger items to trigger compression
        items = [
            {
                "id": i,
                "name": f"Item {i}",
                "value": i * 10,
                "description": f"This is item number {i}",
            }
            for i in range(500)
        ]
        json_content = json.dumps(items)

        messages = [
            {"role": "user", "content": "Get all items"},
            {"role": "tool", "content": json_content},
        ]

        # Use without_compaction to exercise the legacy lossy + JSON-shape
        # path. Lossless default would substitute a non-JSON string.
        crusher = SmartCrusher(
            config=SmartCrusherConfig(min_tokens_to_crush=10, min_items_to_analyze=5),
            with_compaction=False,
        )
        tokenizer = self._get_tokenizer()

        result = crusher.apply(messages, tokenizer)

        # Check JSON compression happened (character-level, not necessarily item count)
        tool_content = result.messages[1]["content"]
        # SmartCrusher compresses JSON (may reduce chars via field pruning, etc.)
        # or adds a digest marker - either way it processes the JSON
        assert len(tool_content) <= len(json_content) + 100  # Allow for digest marker

        # Extract JSON part (may have headroom digest marker appended)
        base_content = re.split(r"\n<headroom:", tool_content)[0]
        parsed = json.loads(base_content)
        assert isinstance(parsed, list)
        assert len(parsed) > 0  # JSON is still valid and contains items
