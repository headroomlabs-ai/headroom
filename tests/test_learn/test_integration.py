"""Integration tests for headroom learn — using real session data.

These tests run against actual conversation data on the machine.
They verify the full pipeline: scan → analyze → recommend → write.
Tests are skipped if the required data directories don't exist.
The LLM-based analyzer tests require ANTHROPIC_API_KEY.

Key behaviors tested:
- Empty recommendations don't create files
- Real Codex/Claude Code sessions can be scanned
- Skip logic: no file writes when nothing meaningful is found
- Idempotency: running twice produces same output
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.learn.analyzer import SessionAnalyzer
from headroom.learn.models import (
    ProjectInfo,
    Recommendation,
    RecommendationTarget,
)
from headroom.learn.scanner import _greedy_path_decode
from headroom.learn.writer import ClaudeCodeWriter, CodexWriter

# =============================================================================
# Writer Tests (no LLM needed)
# =============================================================================


class TestSkipWriteLogic:
    """Verify files are NOT written when there's nothing to write."""

    def test_empty_recommendations_no_write(self, tmp_path):
        """Zero recommendations = zero files written."""
        proj = ProjectInfo(
            name="clean-project",
            project_path=tmp_path / "proj",
            data_path=tmp_path / "data",
        )
        (tmp_path / "proj").mkdir()
        (tmp_path / "data" / "memory").mkdir(parents=True)

        writer = ClaudeCodeWriter()
        result = writer.write([], proj, dry_run=False)

        assert result.files_written == []
        assert not (tmp_path / "proj" / "CLAUDE.md").exists()
        assert not (tmp_path / "data" / "memory" / "MEMORY.md").exists()

    def test_codex_empty_no_write(self, tmp_path):
        """Codex writer also skips on empty recommendations."""
        proj = ProjectInfo(name="clean", project_path=tmp_path, data_path=tmp_path)
        writer = CodexWriter()
        result = writer.write([], proj, dry_run=False)
        assert result.files_written == []
        assert not (tmp_path / "AGENTS.md").exists()


class TestIdempotency:
    """Running learn twice should produce the same output, not duplicate."""

    def test_double_write_replaces_not_appends(self, tmp_path):
        proj = ProjectInfo(name="test", project_path=tmp_path, data_path=tmp_path / "data")
        (tmp_path / "data" / "memory").mkdir(parents=True)

        recs = [
            Recommendation(
                target=RecommendationTarget.CONTEXT_FILE,
                section="Environment",
                content="- Use `uv run python`",
                confidence=0.9,
                evidence_count=10,
            )
        ]

        writer = ClaudeCodeWriter()
        # First write
        writer.write(recs, proj, dry_run=False)
        first_content = (tmp_path / "CLAUDE.local.md").read_text()

        # Second write (same recs)
        writer.write(recs, proj, dry_run=False)
        second_content = (tmp_path / "CLAUDE.local.md").read_text()

        # Content should be identical (replaced, not appended)
        assert first_content == second_content
        assert second_content.count("uv run python") == 1


class TestFalsePositiveFiltering:
    """Verify that common false positives don't generate recommendations."""

    def test_sed_output_not_error(self):
        """sed printing file content with 'Error:' in it isn't a real error."""
        from headroom.learn.scanner import is_error_content

        # Normal code output that happens to contain "error" in identifiers
        assert not is_error_content("def handle_error(e):\n    print('ok')")

    def test_real_error_detected(self):
        """Actual errors should be detected."""
        from headroom.learn.scanner import is_error_content

        assert is_error_content("ModuleNotFoundError: No module named 'flask'")
        assert is_error_content(
            "FileNotFoundError: [Errno 2] No such file or directory: '/bad/path'"
        )
        assert is_error_content("bash: unknown_cmd: command not found")


# =============================================================================
# Real-World Integration Tests (skipped if data not present)
# =============================================================================


CLAUDE_DIR = Path.home() / ".claude" / "projects"
CODEX_DIR = Path.home() / ".codex" / "sessions"
HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_CODEX_DATA = CODEX_DIR.exists() and (
    any(CODEX_DIR.rglob("*.json")) or any(CODEX_DIR.rglob("*.jsonl"))
)


@pytest.mark.skipif(not CLAUDE_DIR.exists(), reason="No Claude Code data")
class TestClaudeCodeIntegration:
    """Integration tests against real Claude Code session data."""

    def test_scanner_discovers_projects(self):
        from headroom.learn.scanner import ClaudeCodeScanner

        scanner = ClaudeCodeScanner()
        projects = scanner.discover_projects()
        if not projects:
            pytest.skip("No Claude Code projects found")
        assert len(projects) > 0
        for p in projects:
            assert p.name
            assert p.data_path.exists()

    def test_scanner_extracts_events(self):
        """Scanner should extract events including user messages."""
        from headroom.learn.scanner import ClaudeCodeScanner

        scanner = ClaudeCodeScanner()
        projects = scanner.discover_projects()
        if not projects:
            pytest.skip("No Claude Code projects found")
        best = max(projects, key=lambda p: len(list(p.data_path.glob("*.jsonl"))))
        sessions = scanner.scan_project(best)
        if not sessions:
            pytest.skip("No Claude Code sessions found")
        assert len(sessions) > 0

        # At least some sessions should have events
        sessions_with_events = [s for s in sessions if s.events]
        assert len(sessions_with_events) > 0

    @pytest.mark.skipif(not HAS_API_KEY, reason="No ANTHROPIC_API_KEY")
    def test_full_pipeline_produces_output(self):
        """Scan → analyze on real data produces valid output."""
        from headroom.learn.scanner import ClaudeCodeScanner

        scanner = ClaudeCodeScanner()
        projects = scanner.discover_projects()
        if not projects:
            pytest.skip("No Claude Code projects found")
        best = max(projects, key=lambda p: len(list(p.data_path.glob("*.jsonl"))))
        sessions = scanner.scan_project(best)
        if not sessions:
            pytest.skip("No Claude Code sessions found")
        assert len(sessions) > 0

        analyzer = SessionAnalyzer()
        result = analyzer.analyze(best, sessions)

        assert result.total_calls > 0
        assert result.total_sessions > 0
        assert 0 <= result.failure_rate <= 0.5

    def test_dry_run_writes_nothing(self):
        """Dry run should never create files."""
        from headroom.learn.scanner import ClaudeCodeScanner

        scanner = ClaudeCodeScanner()
        projects = scanner.discover_projects()
        if not projects:
            pytest.skip("No Claude Code projects found")
        best = max(projects, key=lambda p: len(list(p.data_path.glob("*.jsonl"))))

        # Use mock LLM to avoid needing API key
        mock_response = {
            "context_file_rules": [
                {
                    "section": "Test",
                    "content": "- test rule",
                    "estimated_tokens_saved": 100,
                    "evidence_count": 2,
                }
            ],
            "memory_file_rules": [],
        }
        with patch("headroom.learn.analyzer._call_llm", return_value=mock_response):
            sessions = scanner.scan_project(best)
            result = SessionAnalyzer(model="gpt-4o").analyze(best, sessions)
            recs = result.recommendations

            writer = ClaudeCodeWriter()
            write_result = writer.write(recs, best, dry_run=True)

            assert write_result.dry_run is True
            for fp in write_result.files_written:
                assert "CLAUDE" in fp.name or "MEMORY" in fp.name


class TestDecodeProjectPath:
    """Unit tests for _greedy_path_decode — covers dot-in-path bug (GitHub.nosync)."""

    def test_dot_in_directory_name(self, tmp_path):
        """Paths with dots (e.g. GitHub.nosync) must decode correctly.

        Claude Code encodes '/' and '.' both as '-', so 'GitHub.nosync'
        becomes 'GitHub-nosync' in the directory name.
        _greedy_path_decode must reconstruct it by trying '.' as a join.
        """
        # base = tmp_path, remaining parts = ["GitHub", "nosync", "myproject"]
        # which came from encoding "GitHub.nosync/myproject" as "GitHub-nosync-myproject"
        (tmp_path / "GitHub.nosync" / "myproject").mkdir(parents=True)
        result = _greedy_path_decode(tmp_path, ["GitHub", "nosync", "myproject"])
        assert result == tmp_path / "GitHub.nosync" / "myproject"

    def test_hyphen_in_directory_name(self, tmp_path):
        """Paths with literal hyphens decode correctly (existing behavior preserved)."""
        (tmp_path / "my-project").mkdir()
        result = _greedy_path_decode(tmp_path, ["my", "project"])
        assert result == tmp_path / "my-project"

    def test_simple_path_no_ambiguity(self, tmp_path):
        """Plain paths with no special chars still decode correctly."""
        (tmp_path / "myproject").mkdir()
        result = _greedy_path_decode(tmp_path, ["myproject"])
        assert result == tmp_path / "myproject"

    def test_dot_preferred_over_slash_when_slash_missing(self, tmp_path):
        """When GitHub/nosync doesn't exist but GitHub.nosync does, use dot."""
        # Only create the dot version, not the slash version
        (tmp_path / "GitHub.nosync").mkdir()
        result = _greedy_path_decode(tmp_path, ["GitHub", "nosync"])
        assert result == tmp_path / "GitHub.nosync"


@pytest.mark.skipif(not HAS_CODEX_DATA, reason="No Codex data")
class TestCodexIntegration:
    """Integration tests against real Codex session data."""

    def test_scanner_discovers_sessions(self):
        from headroom.learn.scanner import CodexScanner

        scanner = CodexScanner()
        projects = scanner.discover_projects()
        if not projects:
            pytest.skip("No Codex projects found")
        assert len(projects) == 1  # Codex returns one "project"

    def test_full_pipeline(self):
        """Full pipeline on real Codex data."""
        from headroom.learn.scanner import CodexScanner

        scanner = CodexScanner()
        projects = scanner.discover_projects()
        if not projects:
            pytest.skip("No Codex projects found")
        sessions = scanner.scan_project(projects[0])

        assert len(sessions) > 0

        # Codex has only Bash tool (shell)
        all_tools = {tc.name for s in sessions for tc in s.tool_calls}
        assert "Bash" in all_tools

    def test_codex_writer_targets_agents_md(self, tmp_path):
        """Codex writer should target AGENTS.md, not CLAUDE.md."""
        proj = ProjectInfo(name="codex-test", project_path=tmp_path, data_path=tmp_path)
        recs = [
            Recommendation(
                target=RecommendationTarget.CONTEXT_FILE,
                section="Commands",
                content="- Use npm run test",
                confidence=0.9,
                evidence_count=5,
            )
        ]

        writer = CodexWriter()
        result = writer.write(recs, proj, dry_run=True)
        for fp in result.files_written:
            assert fp.name in ("AGENTS.md", "instructions.md")
            assert "CLAUDE.md" not in fp.name
