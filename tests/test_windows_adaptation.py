"""Tests for Windows adaptation: content detection, compression thresholds, tool exclusion.

Validates the changes in fix/windows-adaptation:
- Bash/Grep/Glob removed from DEFAULT_EXCLUDE_TOOLS
- File listing detection for Glob output
- Compression thresholds (min_ratio, min_tokens, timeout)
"""

import json
import pytest

from headroom.config import DEFAULT_EXCLUDE_TOOLS
from headroom.transforms.content_detector import detect_content_type, ContentType
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.search_compressor import SearchCompressor, SearchCompressorConfig
from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig


# ── Fixtures ──────────────────────────────────────────────────────────────

BASH_OUTPUT = "\n".join([
    "Compiling project v0.1.0",
    "warning: unused variable `x` in src/main.rs:10",
    "warning: deprecated function in src/lib.rs:25",
    "error[E0308]: mismatched types in src/parser.rs:42",
    "  --> src/parser.rs:42:15",
    "42 |     let x: i32 = \"hello\";",
    "error: aborting due to 1 previous error",
    "Build FAILED with 1 error, 2 warnings",
] * 20)

GREP_OUTPUT = "\n".join([
    f"src/auth/login.py:{10+i}:    def handle_login(self, request): # TODO"
    for i in range(80)
])

GLOB_OUTPUT = "\n".join(
    [f"src/components/Component{i:03d}.tsx" for i in range(60)]
    + [f"src/utils/util_{i}.ts" for i in range(40)]
)

JSON_ARRAY = json.dumps([{"id": i, "name": f"item_{i}", "value": i * 10} for i in range(50)])

SMALL_MESSAGE = "This is a short message."


# ── Tool Exclusion Tests ──────────────────────────────────────────────────

class TestToolExclusion:
    """Verify Bash/Grep/Glob are NOT excluded from compression."""

    def test_bash_not_excluded(self):
        assert "Bash" not in DEFAULT_EXCLUDE_TOOLS
        assert "bash" not in DEFAULT_EXCLUDE_TOOLS

    def test_grep_not_excluded(self):
        assert "Grep" not in DEFAULT_EXCLUDE_TOOLS
        assert "grep" not in DEFAULT_EXCLUDE_TOOLS

    def test_glob_not_excluded(self):
        assert "Glob" not in DEFAULT_EXCLUDE_TOOLS
        assert "glob" not in DEFAULT_EXCLUDE_TOOLS

    def test_read_still_excluded(self):
        assert "Read" in DEFAULT_EXCLUDE_TOOLS
        assert "read" in DEFAULT_EXCLUDE_TOOLS

    def test_write_edit_still_excluded(self):
        assert "Write" in DEFAULT_EXCLUDE_TOOLS
        assert "Edit" in DEFAULT_EXCLUDE_TOOLS


# ── Content Detection Tests ───────────────────────────────────────────────

class TestContentDetection:
    """Verify content type detection for tool outputs."""

    def test_bash_detected_as_build(self):
        result = detect_content_type(BASH_OUTPUT)
        assert result.content_type == ContentType.BUILD_OUTPUT
        assert result.confidence >= 0.5

    def test_grep_detected_as_search(self):
        result = detect_content_type(GREP_OUTPUT)
        assert result.content_type == ContentType.SEARCH_RESULTS
        assert result.confidence >= 0.6

    def test_glob_detected_as_search(self):
        result = detect_content_type(GLOB_OUTPUT)
        assert result.content_type == ContentType.SEARCH_RESULTS
        assert result.confidence >= 0.6

    def test_json_detected_as_json_array(self):
        result = detect_content_type(JSON_ARRAY)
        assert result.content_type == ContentType.JSON_ARRAY

    def test_small_text_is_plain(self):
        result = detect_content_type(SMALL_MESSAGE)
        assert result.content_type == ContentType.PLAIN_TEXT

    def test_diff_detected(self):
        diff = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,4 @@\n line\n+new"
        result = detect_content_type(diff)
        assert result.content_type == ContentType.GIT_DIFF


# ── Compression Tests ─────────────────────────────────────────────────────

class TestBashCompression:
    """Bash (build log) compression via ContentRouter."""

    def test_bash_compresses(self):
        router = ContentRouter(ContentRouterConfig())
        result = router.compress(BASH_OUTPUT)
        assert result.compression_ratio < 0.95, f"Bash should compress, got ratio {result.compression_ratio}"

    def test_bash_saves_tokens(self):
        router = ContentRouter(ContentRouterConfig())
        result = router.compress(BASH_OUTPUT)
        original = len(BASH_OUTPUT.split())
        compressed = len(result.compressed.split())
        assert compressed < original


class TestGrepCompression:
    """Grep (search results) compression via ContentRouter."""

    def test_grep_compresses(self):
        router = ContentRouter(ContentRouterConfig())
        result = router.compress(GREP_OUTPUT)
        assert result.compression_ratio < 0.95, f"Grep should compress, got ratio {result.compression_ratio}"

    def test_grep_search_compressor_direct(self):
        cfg = SearchCompressorConfig(max_matches_per_file=3, max_files=10)
        sc = SearchCompressor(cfg)
        result = sc.compress(GREP_OUTPUT, "", 1.0)
        assert result.compression_ratio < 0.5, f"SearchCompressor should compress grep, got {result.compression_ratio}"


class TestGlobCompression:
    """Glob (file listing) compression via ContentRouter."""

    def test_glob_compresses(self):
        router = ContentRouter(ContentRouterConfig())
        result = router.compress(GLOB_OUTPUT)
        assert result.compression_ratio < 0.95, f"Glob should compress, got ratio {result.compression_ratio}"

    def test_glob_deduplicates(self):
        duped = GLOB_OUTPUT + "\n" + GLOB_OUTPUT  # duplicate entries
        router = ContentRouter(ContentRouterConfig())
        result = router.compress(duped)
        original = len(duped.split())
        compressed = len(result.compressed.split())
        assert compressed < original


class TestJsonCompression:
    """JSON array compression via SmartCrusher."""

    def test_json_compresses(self):
        router = ContentRouter(ContentRouterConfig())
        result = router.compress(JSON_ARRAY)
        assert result.compression_ratio < 0.95

    def test_smart_crusher_direct(self):
        sc = SmartCrusher(SmartCrusherConfig(max_items_after_crush=10))
        result = sc.crush(JSON_ARRAY, "", 1.0)
        assert result.was_modified
        assert len(result.compressed) < len(JSON_ARRAY)


class TestSmallContentSkip:
    """Small messages should be skipped (< min_tokens)."""

    def test_small_message_skipped(self):
        router = ContentRouter(ContentRouterConfig())
        result = router.compress(SMALL_MESSAGE)
        # Small messages pass through unchanged
        assert result.compression_ratio >= 0.95 or result.strategy_used.value == "passthrough"


# ── Threshold Tests ───────────────────────────────────────────────────────

class TestThresholds:
    """Verify compression threshold configuration."""

    def test_min_ratio_relaxed(self):
        config = ContentRouterConfig()
        assert config.min_ratio_relaxed >= 0.95, "min_ratio_relaxed should accept marginal compression"

    def test_min_ratio_aggressive(self):
        config = ContentRouterConfig()
        assert config.min_ratio_aggressive <= 0.60, "min_ratio_aggressive should be permissive under pressure"

    def test_kompress_disabled(self):
        config = ContentRouterConfig()
        assert config.enable_kompress is False, "Kompress should be disabled for Windows CPU"


# ── End-to-End Compression Savings ────────────────────────────────────────

class TestEndToEndSavings:
    """End-to-end compression savings across all content types."""

    def test_total_savings(self):
        router = ContentRouter(ContentRouterConfig())
        test_cases = [
            ("bash", BASH_OUTPUT),
            ("grep", GREP_OUTPUT),
            ("glob", GLOB_OUTPUT),
            ("json", JSON_ARRAY),
        ]
        total_original = 0
        total_compressed = 0
        for name, content in test_cases:
            result = router.compress(content)
            total_original += len(content.split())
            total_compressed += len(result.compressed.split())

        overall_ratio = total_compressed / total_original
        assert overall_ratio < 0.50, (
            f"Overall compression should be <50%, got {overall_ratio:.1%}"
        )
