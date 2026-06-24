"""Comprehensive tests for the public DiffCompressor API.

Tests cover:
1. Context line reduction
2. Hunk selection and limiting
3. Compression ratios
4. Edge cases
5. Bug-fix regressions and routing-gap fixtures

Stage 3b note (2026-04-25): the Python `DiffCompressor` implementation
was retired in favor of the Rust-backed shim (`headroom._core` via PyO3).
Tests that probed Python-only internals — `_parse_diff`, `_score_hunks`,
the `DiffHunk` / `DiffFile` parser dataclasses — were removed because
the Rust crate has its own parallel coverage in
`crates/headroom-core/tests`. Public-API tests (anything calling
`compressor.compress(...)`) are preserved unchanged: they exercise the
Rust backend through the same import path and assert the same outputs.
"""

from headroom.transforms.diff_compressor import (
    DiffCompressionResult,
    DiffCompressor,
    DiffCompressorConfig,
)


class TestContextReduction:
    """Tests for context line reduction."""

    def test_reduce_context_lines(self):
        """Context lines are reduced to configured maximum."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,10 +1,11 @@
 context1
 context2
 context3
 context4
+added
 context5
 context6
 context7
 context8
"""
        # Default max_context_lines is 2
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_context_lines=2,
                min_lines_for_ccr=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should keep 2 context before and 2 after the +added line
        # Plus the added line itself
        lines = result.compressed.split("\n")
        context_count = sum(1 for line in lines if line.startswith(" "))

        # At most 4 context lines (2 before + 2 after)
        assert context_count <= 4

    def test_preserve_all_changes(self):
        """All addition and deletion lines are preserved."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,10 +1,10 @@
 ctx1
 ctx2
-removed1
+added1
 ctx3
 ctx4
-removed2
+added2
 ctx5
 ctx6
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                min_lines_for_ccr=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert "-removed1" in result.compressed
        assert "-removed2" in result.compressed
        assert "+added1" in result.compressed
        assert "+added2" in result.compressed


class TestHunkSelection:
    """Tests for hunk selection when limiting."""

    def test_max_hunks_per_file(self):
        """Hunks are limited to max_hunks_per_file."""
        # Create a diff with many hunks
        hunks = []
        for i in range(20):
            hunks.append(f"""@@ -{i * 10},3 +{i * 10},4 @@
 context
+added_{i}
 more
""")

        content = f"""diff --git a/bigfile.py b/bigfile.py
--- a/bigfile.py
+++ b/bigfile.py
{"".join(hunks)}"""

        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_hunks_per_file=5,
                min_lines_for_ccr=10,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should have at most 5 hunks
        hunk_count = result.compressed.count("@@")
        # Each hunk has one @@ header (we count full hunk headers)
        assert hunk_count <= 10  # Each hunk header appears twice @@...@@

    def test_keeps_first_and_last_hunk(self):
        """First and last hunks are preserved when limiting."""
        hunks = []
        for i in range(10):
            hunks.append(f"""@@ -{i * 10},3 +{i * 10},4 @@
 context
+added_{i}
 more
""")

        content = f"""diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
{"".join(hunks)}"""

        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_hunks_per_file=3,
                min_lines_for_ccr=10,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # First hunk (added_0) should be present
        assert "+added_0" in result.compressed
        # Last hunk (added_9) should be present
        assert "+added_9" in result.compressed


class TestFileSelection:
    """Tests for file selection when limiting."""

    def test_max_files(self):
        """Files are limited to max_files."""
        # Create diff with many files
        files = []
        for i in range(30):
            files.append(f"""diff --git a/file{i}.py b/file{i}.py
--- a/file{i}.py
+++ b/file{i}.py
@@ -1,2 +1,3 @@
 ctx
+added
 ctx2
""")

        content = "\n".join(files)

        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_files=10,
                min_lines_for_ccr=20,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Count diff --git headers
        file_count = result.compressed.count("diff --git")
        assert file_count <= 10


class TestCompressionResult:
    """Tests for DiffCompressionResult properties."""

    def test_compression_ratio_calculation(self):
        """Compression ratio is calculated correctly."""
        result = DiffCompressionResult(
            compressed="a\nb\nc",
            original_line_count=100,
            compressed_line_count=10,
            files_affected=2,
            additions=5,
            deletions=3,
            hunks_kept=2,
            hunks_removed=5,
        )

        assert result.compression_ratio == 0.1

    def test_tokens_saved_estimate(self):
        """Token savings estimation works correctly."""
        result = DiffCompressionResult(
            compressed="short",
            original_line_count=100,
            compressed_line_count=10,
            files_affected=1,
            additions=10,
            deletions=5,
            hunks_kept=1,
            hunks_removed=0,
        )

        # 90 lines saved * 40 chars/line / 4 chars/token = 900 tokens
        assert result.tokens_saved_estimate == 900


class TestSmallDiffPassthrough:
    """Tests for small diff passthrough behavior."""

    def test_small_diff_unchanged(self):
        """Diffs smaller than threshold pass through unchanged."""
        content = """diff --git a/small.py b/small.py
--- a/small.py
+++ b/small.py
@@ -1,2 +1,3 @@
 line1
+added
 line2
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                min_lines_for_ccr=100,  # High threshold
            )
        )
        result = compressor.compress(content)

        # Should be unchanged
        assert result.compressed == content
        assert result.compression_ratio == 1.0


class TestOutputFormatting:
    """Tests for output formatting."""

    def test_summary_line_added(self):
        """Summary line is added at end of compressed diff."""
        # Large diff that will be compressed
        hunks = []
        for i in range(15):
            hunks.append(f"""@@ -{i * 10},5 +{i * 10},6 @@
 ctx1
 ctx2
+added_{i}
 ctx3
 ctx4
""")

        content = f"""diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
{"".join(hunks)}"""

        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_hunks_per_file=5,
                min_lines_for_ccr=10,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should have summary at end
        assert "files changed" in result.compressed
        assert "hunks omitted" in result.compressed

    def test_preserves_diff_format(self):
        """Output preserves valid unified diff format."""
        content = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def test():
+    # new comment
     pass
     return True
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                min_lines_for_ccr=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Should have all standard diff markers
        assert "diff --git" in result.compressed
        assert "---" in result.compressed
        assert "+++" in result.compressed
        assert "@@" in result.compressed


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_input(self):
        """Empty input is handled gracefully."""
        compressor = DiffCompressor()
        result = compressor.compress("")

        assert result.compressed == ""
        assert result.compression_ratio == 1.0

    def test_non_diff_input(self):
        """Non-diff input passes through unchanged."""
        content = "This is not a diff\nJust regular text"
        compressor = DiffCompressor()
        result = compressor.compress(content)

        # Should pass through (no diff --git found)
        assert result.compressed == content

    def test_unicode_content(self):
        """Unicode characters in diff are handled."""
        content = """diff --git a/i18n.py b/i18n.py
--- a/i18n.py
+++ b/i18n.py
@@ -1,2 +1,3 @@
 msg = "hello"
+msg_ja = "こんにちは"
 return msg
"""
        compressor = DiffCompressor()
        result = compressor.compress(content)

        assert "こんにちは" in result.compressed

    def test_no_newline_at_eof(self):
        """Handles 'No newline at end of file' indicator."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,2 +1,2 @@
 line1
-line2
\\ No newline at end of file
+line2_modified
\\ No newline at end of file
"""
        compressor = DiffCompressor()
        result = compressor.compress(content)

        # Should not crash and preserve the indicator
        assert "No newline" in result.compressed or "-line2" in result.compressed

    def test_empty_hunks(self):
        """Files with no actual hunks are handled."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
"""
        compressor = DiffCompressor()
        result = compressor.compress(content)

        # Should not crash
        assert result.compressed is not None


class TestConfigOptions:
    """Tests for configuration options."""

    def test_max_context_lines_config(self):
        """max_context_lines configuration controls context reduction."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,10 +1,11 @@
 c1
 c2
 c3
 c4
 c5
+added
 c6
 c7
 c8
 c9
 c10
"""
        # With max_context_lines=1
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                max_context_lines=1,
                min_lines_for_ccr=5,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        # Count context lines (lines starting with space)
        context_count = sum(1 for line in result.compressed.split("\n") if line.startswith(" "))

        # Should have at most 2 context lines (1 before + 1 after)
        assert context_count <= 2

    def test_always_keep_additions_default(self):
        """Additions are always kept by default."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,5 @@
 ctx
+add1
+add2
 ctx
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                always_keep_additions=True,
                min_lines_for_ccr=2,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert "+add1" in result.compressed
        assert "+add2" in result.compressed

    def test_always_keep_deletions_default(self):
        """Deletions are always kept by default."""
        content = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,5 +1,3 @@
 ctx
-del1
-del2
 ctx
"""
        compressor = DiffCompressor(
            config=DiffCompressorConfig(
                always_keep_deletions=True,
                min_lines_for_ccr=2,
                enable_ccr=False,
            )
        )
        result = compressor.compress(content)

        assert "-del1" in result.compressed
        assert "-del2" in result.compressed


# ─── Bug-fix tests (2026-04-25): four silent information-loss paths ─────────
#
# Before the fix, the parser captured these patterns but the emitter dropped
# them, or the regex didn't match them at all. Each test exercises one of
# the four paths the same way the Rust unit tests do.


def _cfg_below_threshold():
    """Small config so the parser+emitter actually run on test inputs."""
    from headroom.transforms.diff_compressor import DiffCompressorConfig

    return DiffCompressorConfig(min_lines_for_ccr=5)


class TestBugfixRenamePreservation:
    """rename/similarity/dissimilarity/copy markers were captured into
    is_renamed=True and then dropped by the emitter. Output looked like a
    plain modification of the old path."""

    def test_rename_with_similarity_index_preserved(self):
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 92%\n"
            "rename from old.py\n"
            "rename to new.py\n"
            "--- a/old.py\n"
            "+++ b/new.py\n"
            "@@ -1,3 +1,3 @@\n"
            " ctx_a\n"
            "-old\n"
            "+new\n"
            " ctx_b\n"
        )
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert "similarity index 92%" in result.compressed
        assert "rename from old.py" in result.compressed
        assert "rename to new.py" in result.compressed

    def test_dissimilarity_index_preserved(self):
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = (
            "diff --git a/x.py b/y.py\n"
            "dissimilarity index 60%\n"
            "rename from x.py\n"
            "rename to y.py\n"
            "--- a/x.py\n"
            "+++ b/y.py\n"
            "@@ -1 +1 @@\n"
            "-a\n"
            "+b\n"
        )
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert "dissimilarity index 60%" in result.compressed

    def test_copy_markers_preserved(self):
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = (
            "diff --git a/orig.py b/dup.py\n"
            "similarity index 100%\n"
            "copy from orig.py\n"
            "copy to dup.py\n"
            "--- a/orig.py\n"
            "+++ b/dup.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert "copy from orig.py" in result.compressed
        assert "copy to dup.py" in result.compressed


class TestBugfixCombinedDiff:
    """Combined-diff `@@@` hunks from merge commits had ALL content silently
    dropped because the regex hardcoded `@@`."""

    def test_3way_combined_diff_content_preserved(self):
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = (
            "diff --git a/merge.py b/merge.py\n"
            "--- a/merge.py\n"
            "+++ b/merge.py\n"
            "@@@ -1,3 -1,3 +1,4 @@@\n"
            "  unchanged_a\n"
            "- old_branch_1\n"
            " -old_branch_2\n"
            "++new_in_merge\n"
            " +new_added\n"
            "  unchanged_b\n"
        )
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert "@@@ -1,3 -1,3 +1,4 @@@" in result.compressed
        assert "++new_in_merge" in result.compressed
        assert result.files_affected > 0


class TestBugfixNoNewlineMarker:
    r"""`\ No newline at end of file` got dropped by context trim whenever it
    was further than max_context_lines from a +/- change."""

    def test_no_newline_marker_survives_distance(self):
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = (
            "diff --git a/last.txt b/last.txt\n"
            "--- a/last.txt\n"
            "+++ b/last.txt\n"
            "@@ -1,8 +1,8 @@\n"
            "-old_first\n"
            "+new_first\n"
            " ctx_a\n"
            " ctx_b\n"
            " ctx_c\n"
            " ctx_d\n"
            " ctx_e\n"
            " ctx_f\n"
            "\\ No newline at end of file\n"
        )
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert "\\ No newline at end of file" in result.compressed


class TestBugfixPreDiffContent:
    """Anything before the first `diff --git` (commit headers, email-style
    metadata) was silently dropped."""

    def test_commit_header_preserved(self):
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = (
            "commit abc1234567890abcdef\n"
            "Author: Tester <t@example.com>\n"
            "Date:   Mon Apr 25 12:00:00 2026\n"
            "\n"
            "    Refactor: rename and modify\n"
            "\n"
            "diff --git a/x.py b/x.py\n"
            "--- a/x.py\n"
            "+++ b/x.py\n"
            "@@ -1 +1 @@\n"
            "-a\n"
            "+b\n"
        )
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert result.compressed.startswith("commit abc1234567890abcdef")
        assert "Author: Tester" in result.compressed
        assert "Refactor: rename and modify" in result.compressed
        assert "diff --git a/x.py b/x.py" in result.compressed
        assert "-a" in result.compressed
        assert "+b" in result.compressed

    def test_no_pre_diff_content_does_not_add_blank_line(self):
        """Edge case: when there's no pre-diff content, output must NOT
        gain a leading blank line from a stray empty-list prepend."""
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert result.compressed.startswith("diff --git a/x.py b/x.py")


class TestRoutingGapMergeDiffs:
    """Routing gap (2026-04-25 follow-up): ContentRouter detects diff inputs
    and routes them to DiffCompressor, but the parser previously only knew
    the `diff --git` shape. Merge-commit diffs from `git log -p` use
    `diff --combined <path>` or `diff --cc <path>` and were treated as
    non-diff blobs and passed through unchanged.
    """

    def test_diff_combined_header_starts_a_file_section(self):
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = (
            "diff --combined merge_target.py\n"
            "index abc..def..ghi 100644\n"
            "--- a/merge_target.py\n"
            "+++ b/merge_target.py\n"
            "@@@ -1,3 -1,3 +1,4 @@@\n"
            "  unchanged_a\n"
            "- old_p1\n"
            " -old_p2\n"
            "++new_in_merge\n"
            "  unchanged_b\n"
        )
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert result.files_affected == 1
        assert "diff --combined merge_target.py" in result.compressed
        assert "@@@ -1,3 -1,3 +1,4 @@@" in result.compressed
        assert "++new_in_merge" in result.compressed

    def test_diff_cc_header_starts_a_file_section(self):
        from headroom.transforms.diff_compressor import DiffCompressor

        diff = (
            "diff --cc cc_target.py\n"
            "index abc..def..ghi\n"
            "--- a/cc_target.py\n"
            "+++ b/cc_target.py\n"
            "@@@ -1,3 -1,3 +1,4 @@@\n"
            "  ctx\n"
            "- removed_p1\n"
            " -removed_p2\n"
            "++added_in_merge\n"
            "  more_ctx\n"
        )
        result = DiffCompressor(_cfg_below_threshold()).compress(diff)
        assert result.files_affected == 1
        assert "diff --cc cc_target.py" in result.compressed
        assert "++added_in_merge" in result.compressed


class TestRoutingGapDetectorScanWindow:
    """Routing gap (2026-04-25 follow-up): `_try_detect_diff` only scanned
    the first 50 lines, so `git log -p` outputs with long commit messages
    pushed the diff past the detection window — input was misrouted away
    from DiffCompressor entirely. Window widened to 500 lines.
    """

    def test_detect_picks_up_diff_after_long_commit_message(self):
        from headroom.transforms.content_detector import (
            ContentType,
            detect_content_type,
        )

        # 60 lines of commit message before the diff. Old 50-line cap
        # would have missed the `diff --git` header entirely.
        msg_lines = [
            "commit abc123",
            "Author: Tester <t@example.com>",
            "Date:   Mon Apr 25 12:00:00 2026",
            "",
        ] + [f"    msg line {i}" for i in range(60)]
        diff = (
            "\n".join(msg_lines)
            + "\n\n"
            + "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
        )
        result = detect_content_type(diff)
        assert result.content_type == ContentType.GIT_DIFF
        assert result.confidence >= 0.7

    def test_detect_recognizes_combined_diff_headers(self):
        """The detector also gained recognition for combined-diff hunk
        headers (`@@@`+) — useful when the only signal in a snippet is
        the merge-style hunk."""
        from headroom.transforms.content_detector import (
            ContentType,
            detect_content_type,
        )

        # Full merge diff (with `--- a/` shared with regular diffs as a
        # belt-and-suspenders signal).
        diff = (
            "diff --combined m.py\n--- a/m.py\n+++ b/m.py\n@@@ -1,2 -1,2 +1,3 @@@\n  ctx\n++added\n"
        )
        result = detect_content_type(diff)
        assert result.content_type == ContentType.GIT_DIFF
