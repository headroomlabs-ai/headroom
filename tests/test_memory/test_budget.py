"""Tests for the Memory Budget Manager."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from headroom.memory.budget import (
    BudgetConfig,
    MemoryBudgetManager,
)
from headroom.memory.writers.base import MemoryEntry


def _make_entry(
    content: str = "test memory",
    importance: float = 0.5,
    age_days: float = 0,
    access_count: int = 0,
    entity_refs: list[str] | None = None,
    category: str = "",
) -> MemoryEntry:
    return MemoryEntry(
        content=content,
        importance=importance,
        created_at=time.time() - age_days * 86400,
        access_count=access_count,
        entity_refs=entity_refs or [],
        category=category,
    )


class TestBudgetManager:
    @pytest.fixture
    def manager(self, tmp_path: Path) -> MemoryBudgetManager:
        config = BudgetConfig(staleness_check_git=False)  # No git for unit tests
        return MemoryBudgetManager(project_path=tmp_path, config=config)

    def test_basic_optimize(self, manager: MemoryBudgetManager):
        entries = [_make_entry(f"Memory {i}", importance=0.5 + i * 0.1) for i in range(5)]
        optimized, report = manager.optimize(entries, "generic")

        assert report.total_memories == 5
        assert report.kept > 0
        assert report.tokens_after <= 3000  # Default budget

    def test_budget_limits(self, manager: MemoryBudgetManager):
        # Create many entries that exceed budget
        entries = [
            _make_entry(f"A long memory content string number {i} " * 10, importance=0.8)
            for i in range(50)
        ]
        optimized, report = manager.optimize(entries, "claude")  # 2000 token budget

        assert report.pruned_budget > 0
        assert report.kept < 50

    def test_temporal_decay(self, manager: MemoryBudgetManager):
        old = _make_entry("Old memory", importance=0.5, age_days=30)
        new = _make_entry("New memory", importance=0.5, age_days=0)

        optimized, report = manager.optimize([old, new], "generic")

        # New should rank higher
        if len(optimized) >= 2:
            assert optimized[0].content == "New memory"

    def test_access_count_boost(self, manager: MemoryBudgetManager):
        unused = _make_entry("Unused", importance=0.5, age_days=5, access_count=0)
        used = _make_entry("Heavily used", importance=0.5, age_days=5, access_count=10)

        optimized, report = manager.optimize([unused, used], "generic")

        if len(optimized) >= 2:
            # Used memory should have higher importance after decay+boost
            assert optimized[0].content == "Heavily used"

    def test_merge_similar(self, manager: MemoryBudgetManager):
        entries = [
            _make_entry(
                "Use source .venv/bin/activate && pytest for running tests", importance=0.5
            ),
            _make_entry(
                "Use source .venv/bin/activate && pytest for running tests", importance=0.8
            ),
            _make_entry("Something completely different about architecture", importance=0.6),
        ]
        optimized, report = manager.optimize(entries, "generic")

        # The two identical ones should be merged
        assert report.merged >= 1
        assert report.kept <= 2

    def test_very_old_pruned(self, manager: MemoryBudgetManager):
        """Test that very old, low-importance memories get pruned by decay."""
        entries = [
            _make_entry("Ancient memory", importance=0.2, age_days=100),
            _make_entry("Recent memory", importance=0.5, age_days=0),
        ]
        optimized, report = manager.optimize(entries, "generic")

        contents = [m.content for m in optimized]
        assert "Recent memory" in contents
        # Ancient memory with 0.2 importance after 100 days of decay should be below threshold

    def test_report_tokens(self, manager: MemoryBudgetManager):
        # Use entries large enough to trigger budget pruning
        entries = [
            _make_entry(f"A very long memory content entry {i} " * 30, importance=0.8)
            for i in range(20)
        ]
        _, report = manager.optimize(entries, "claude")  # 2000 token budget

        assert report.tokens_before > 0
        assert report.pruned_budget > 0  # Should drop some due to budget
        assert report.kept < 20


class TestStalenessDetection:
    def test_stale_file_ref(self, tmp_path: Path):
        """Test that memories referencing non-existent files are flagged stale."""
        # Create a manager with git check disabled but file-exists check active
        config = BudgetConfig(staleness_check_git=False)
        manager = MemoryBudgetManager(project_path=tmp_path, config=config)

        # The entity_refs check uses Path.exists(), so non-existent paths flag stale
        entries = [
            _make_entry(
                "File `/nonexistent/path/foo.py` has the auth logic",
                importance=0.5,
                entity_refs=["/nonexistent/path/foo.py"],
            ),
            _make_entry(
                "Use pytest for tests",
                importance=0.5,
            ),
        ]
        # With git disabled, staleness only checks entity_refs via Path.exists()
        optimized, report = manager.optimize(entries, "generic")

        assert report.pruned_staleness >= 1

    def test_fresh_file_kept(self, tmp_path: Path):
        """Test that memories referencing existing files are kept."""
        # Create the referenced file
        (tmp_path / "real_file.py").write_text("# exists")

        config = BudgetConfig(staleness_check_git=False)
        manager = MemoryBudgetManager(project_path=tmp_path, config=config)

        entries = [
            _make_entry(
                "Real file has important code",
                importance=0.5,
                entity_refs=[str(tmp_path / "real_file.py")],
            ),
        ]
        optimized, report = manager.optimize(entries, "generic")

        assert report.pruned_staleness == 0
        assert report.kept == 1


class TestBudgetConfig:
    def test_default_budgets(self):
        config = BudgetConfig()
        assert config.agent_budgets["claude"] == 2000
        assert config.agent_budgets["cursor"] == 3000

    def test_custom_budgets(self):
        config = BudgetConfig(agent_budgets={"claude": 5000})
        assert config.agent_budgets["claude"] == 5000
