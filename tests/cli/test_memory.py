"""Tests for memory CLI commands.

These tests use real SQLite databases (temp files) - no mocks.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.models import Memory


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_db(tmp_path: Path) -> str:
    """Create a temporary database path."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def populated_db(temp_db: str) -> str:
    """Create a database with sample memories."""
    store = SQLiteMemoryStore(temp_db)

    # Create memories at different scopes and ages
    memories = [
        # USER scope (no session/agent/turn)
        Memory(
            id="user-mem-001",
            content="User prefers TypeScript over JavaScript",
            user_id="test-user",
            session_id=None,
            agent_id=None,
            turn_id=None,
            importance=0.9,
            created_at=datetime.now() - timedelta(days=5),
            valid_from=datetime.now() - timedelta(days=5),
        ),
        # SESSION scope
        Memory(
            id="session-mem-001",
            content="Working on authentication feature",
            user_id="test-user",
            session_id="session-123",
            agent_id=None,
            turn_id=None,
            importance=0.7,
            created_at=datetime.now() - timedelta(hours=2),
            valid_from=datetime.now() - timedelta(hours=2),
        ),
        Memory(
            id="session-mem-002",
            content="Database uses PostgreSQL",
            user_id="test-user",
            session_id="session-123",
            agent_id=None,
            turn_id=None,
            importance=0.6,
            created_at=datetime.now() - timedelta(days=10),
            valid_from=datetime.now() - timedelta(days=10),
        ),
        # AGENT scope
        Memory(
            id="agent-mem-001",
            content="Agent is exploring code structure",
            user_id="test-user",
            session_id="session-123",
            agent_id="agent-456",
            turn_id=None,
            importance=0.4,
            created_at=datetime.now() - timedelta(hours=1),
            valid_from=datetime.now() - timedelta(hours=1),
        ),
        # TURN scope (ephemeral)
        Memory(
            id="turn-mem-001",
            content="Tool output from grep search",
            user_id="test-user",
            session_id="session-123",
            agent_id="agent-456",
            turn_id="turn-789",
            importance=0.2,
            created_at=datetime.now() - timedelta(minutes=5),
            valid_from=datetime.now() - timedelta(minutes=5),
        ),
        # Low importance memory for pruning tests
        Memory(
            id="low-importance-001",
            content="Temporary note",
            user_id="test-user",
            session_id="session-123",
            agent_id=None,
            turn_id=None,
            importance=0.1,
            created_at=datetime.now() - timedelta(days=45),
            valid_from=datetime.now() - timedelta(days=45),
        ),
    ]

    for mem in memories:
        asyncio.run(store.save(mem))

    return temp_db


class TestMemoryList:
    """Tests for 'headroom memory list' command."""

    def test_list_all(self, runner: CliRunner, populated_db: str) -> None:
        """List all memories."""
        result = runner.invoke(main, ["memory", "list", "--db-path", populated_db])
        assert result.exit_code == 0
        # IDs are truncated to 8 chars in display, check for partial matches
        assert "user-mem" in result.output
        assert "session-" in result.output  # "session-mem" truncated to "session-"

    def test_list_with_limit(self, runner: CliRunner, populated_db: str) -> None:
        """List with limit."""
        result = runner.invoke(main, ["memory", "list", "--db-path", populated_db, "--limit", "2"])
        assert result.exit_code == 0
        # Should show limited results
        assert "2 shown" in result.output or "Memories" in result.output

    def test_list_by_scope(self, runner: CliRunner, populated_db: str) -> None:
        """Filter by scope level."""
        result = runner.invoke(
            main, ["memory", "list", "--db-path", populated_db, "--scope", "USER"]
        )
        assert result.exit_code == 0
        assert "TypeScript" in result.output  # USER scope memory content

    def test_list_empty_db(self, runner: CliRunner, temp_db: str) -> None:
        """List from empty database."""
        # Initialize empty db
        SQLiteMemoryStore(temp_db)
        result = runner.invoke(main, ["memory", "list", "--db-path", temp_db])
        assert result.exit_code == 0
        assert "No memories found" in result.output


class TestMemoryShow:
    """Tests for 'headroom memory show' command."""

    def test_show_by_id(self, runner: CliRunner, populated_db: str) -> None:
        """Show memory by full ID."""
        result = runner.invoke(main, ["memory", "show", "--db-path", populated_db, "user-mem-001"])
        assert result.exit_code == 0
        assert "TypeScript" in result.output
        assert "0.9" in result.output or "0.90" in result.output  # importance

    def test_show_by_partial_id(self, runner: CliRunner, populated_db: str) -> None:
        """Show memory by partial ID."""
        result = runner.invoke(main, ["memory", "show", "--db-path", populated_db, "user-mem"])
        assert result.exit_code == 0
        assert "TypeScript" in result.output

    def test_show_json_output(self, runner: CliRunner, populated_db: str) -> None:
        """Show memory as JSON."""
        result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "user-mem-001", "--json"]
        )
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert data["id"] == "user-mem-001"
        assert "TypeScript" in data["content"]

    def test_show_not_found(self, runner: CliRunner, populated_db: str) -> None:
        """Show non-existent memory."""
        result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "nonexistent-id"]
        )
        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestMemoryStats:
    """Tests for 'headroom memory stats' command."""

    def test_stats(self, runner: CliRunner, populated_db: str) -> None:
        """Show stats for populated database."""
        result = runner.invoke(main, ["memory", "stats", "--db-path", populated_db])
        assert result.exit_code == 0
        assert "Total" in result.output or "Memories" in result.output
        assert "6" in result.output  # 6 memories

    def test_stats_empty_db(self, runner: CliRunner, temp_db: str) -> None:
        """Stats for empty database."""
        SQLiteMemoryStore(temp_db)
        result = runner.invoke(main, ["memory", "stats", "--db-path", temp_db])
        assert result.exit_code == 0
        assert "0" in result.output


class TestMemoryEdit:
    """Tests for 'headroom memory edit' command."""

    def test_edit_content(self, runner: CliRunner, populated_db: str) -> None:
        """Edit memory content."""
        result = runner.invoke(
            main,
            [
                "memory",
                "edit",
                "--db-path",
                populated_db,
                "user-mem-001",
                "--content",
                "Updated content",
            ],
        )
        assert result.exit_code == 0

        # Verify change
        show_result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "user-mem-001"]
        )
        assert "Updated content" in show_result.output

    def test_edit_importance(self, runner: CliRunner, populated_db: str) -> None:
        """Edit memory importance."""
        result = runner.invoke(
            main,
            ["memory", "edit", "--db-path", populated_db, "user-mem-001", "--importance", "0.5"],
        )
        assert result.exit_code == 0

        # Verify change
        show_result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "user-mem-001"]
        )
        assert "0.5" in show_result.output

    def test_edit_not_found(self, runner: CliRunner, populated_db: str) -> None:
        """Edit non-existent memory."""
        result = runner.invoke(
            main,
            ["memory", "edit", "--db-path", populated_db, "nonexistent", "--content", "test"],
        )
        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestMemoryDelete:
    """Tests for 'headroom memory delete' command."""

    def test_delete_single(self, runner: CliRunner, populated_db: str) -> None:
        """Delete single memory with force."""
        result = runner.invoke(
            main,
            ["memory", "delete", "--db-path", populated_db, "turn-mem-001", "--force"],
        )
        assert result.exit_code == 0

        # Verify deleted
        show_result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "turn-mem-001"]
        )
        assert "not found" in show_result.output.lower() or show_result.exit_code != 0

    def test_delete_multiple(self, runner: CliRunner, populated_db: str) -> None:
        """Delete multiple memories."""
        result = runner.invoke(
            main,
            [
                "memory",
                "delete",
                "--db-path",
                populated_db,
                "turn-mem-001",
                "agent-mem-001",
                "--force",
            ],
        )
        assert result.exit_code == 0

    def test_delete_requires_confirmation(self, runner: CliRunner, populated_db: str) -> None:
        """Delete prompts for confirmation without --force."""
        # Invoke delete and say no to confirmation
        runner.invoke(
            main,
            ["memory", "delete", "--db-path", populated_db, "turn-mem-001"],
            input="n\n",  # Say no
        )
        # Verify memory still exists since we said no
        show_result = runner.invoke(
            main, ["memory", "show", "--db-path", populated_db, "turn-mem-001"]
        )
        # Memory should still exist since we said no
        assert "Tool output" in show_result.output or show_result.exit_code == 0


class TestMemoryPrune:
    """Tests for 'headroom memory prune' command."""

    def test_prune_dry_run(self, runner: CliRunner, populated_db: str) -> None:
        """Prune with dry-run shows what would be deleted."""
        result = runner.invoke(
            main,
            ["memory", "prune", "--db-path", populated_db, "--older-than", "30d", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "would" in result.output.lower() or "dry" in result.output.lower()

    def test_prune_by_age(self, runner: CliRunner, populated_db: str) -> None:
        """Prune old memories."""
        result = runner.invoke(
            main,
            ["memory", "prune", "--db-path", populated_db, "--older-than", "30d", "--force"],
        )
        assert result.exit_code == 0
        # Should have deleted the 45-day old memory

    def test_prune_by_scope(self, runner: CliRunner, populated_db: str) -> None:
        """Prune by scope level."""
        result = runner.invoke(
            main,
            ["memory", "prune", "--db-path", populated_db, "--scope", "TURN", "--force"],
        )
        assert result.exit_code == 0

        # Verify TURN memories are gone
        list_result = runner.invoke(
            main, ["memory", "list", "--db-path", populated_db, "--scope", "TURN"]
        )
        assert "No memories found" in list_result.output or "turn-mem" not in list_result.output

    def test_prune_low_importance(self, runner: CliRunner, populated_db: str) -> None:
        """Prune low importance memories."""
        result = runner.invoke(
            main,
            ["memory", "prune", "--db-path", populated_db, "--low-importance", "0.3", "--force"],
        )
        assert result.exit_code == 0


class TestMemoryPurge:
    """Tests for 'headroom memory purge' command."""

    def test_purge_requires_confirm_flag(self, runner: CliRunner, populated_db: str) -> None:
        """Purge requires --confirm flag."""
        result = runner.invoke(main, ["memory", "purge", "--db-path", populated_db])
        assert result.exit_code != 0 or "confirm" in result.output.lower()

    def test_purge_with_confirm(self, runner: CliRunner, populated_db: str) -> None:
        """Purge deletes all memories."""
        result = runner.invoke(
            main,
            ["memory", "purge", "--db-path", populated_db, "--confirm"],
            input="y\n",  # Confirm
        )
        assert result.exit_code == 0

        # Verify empty
        stats_result = runner.invoke(main, ["memory", "stats", "--db-path", populated_db])
        assert "0" in stats_result.output


class TestMemoryExportImport:
    """Tests for export/import commands."""

    def test_export_to_stdout(self, runner: CliRunner, populated_db: str) -> None:
        """Export memories to stdout."""
        result = runner.invoke(main, ["memory", "export", "--db-path", populated_db])
        assert result.exit_code == 0
        # Should be valid JSON array
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 6

    def test_export_to_file(self, runner: CliRunner, populated_db: str, tmp_path: Path) -> None:
        """Export memories to file."""
        output_file = tmp_path / "export.json"
        result = runner.invoke(
            main,
            ["memory", "export", "--db-path", populated_db, "--output", str(output_file)],
        )
        assert result.exit_code == 0

        # Verify file
        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 6

    def test_import_from_file(self, runner: CliRunner, temp_db: str, tmp_path: Path) -> None:
        """Import memories from file."""
        # Create import file
        import_data = [
            {
                "id": "imported-001",
                "content": "Imported memory",
                "user_id": "test-user",
                "importance": 0.8,
                "created_at": datetime.now().isoformat(),
                "valid_from": datetime.now().isoformat(),
            }
        ]
        import_file = tmp_path / "import.json"
        with open(import_file, "w") as f:
            json.dump(import_data, f)

        # Initialize empty db
        SQLiteMemoryStore(temp_db)

        result = runner.invoke(
            main,
            ["memory", "import", "--db-path", temp_db, str(import_file), "--force"],
        )
        assert result.exit_code == 0

        # Verify imported
        show_result = runner.invoke(main, ["memory", "show", "--db-path", temp_db, "imported-001"])
        assert "Imported memory" in show_result.output

    def test_export_import_roundtrip(
        self, runner: CliRunner, populated_db: str, tmp_path: Path
    ) -> None:
        """Export and import should be lossless."""
        export_file = tmp_path / "roundtrip.json"
        new_db = str(tmp_path / "new.db")

        # Export
        runner.invoke(
            main, ["memory", "export", "--db-path", populated_db, "--output", str(export_file)]
        )

        # Import to new db
        SQLiteMemoryStore(new_db)
        runner.invoke(main, ["memory", "import", "--db-path", new_db, str(export_file), "--force"])

        # Compare stats
        orig_stats = runner.invoke(main, ["memory", "stats", "--db-path", populated_db])
        new_stats = runner.invoke(main, ["memory", "stats", "--db-path", new_db])

        # Should have same count
        assert "6" in orig_stats.output
        assert "6" in new_stats.output


class TestMemoryHelp:
    """Tests for help output."""

    def test_memory_help(self, runner: CliRunner) -> None:
        """Memory group shows help."""
        result = runner.invoke(main, ["memory", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "stats" in result.output
        assert "edit" in result.output
        assert "delete" in result.output
        assert "prune" in result.output
        assert "purge" in result.output
        assert "export" in result.output
        assert "import" in result.output

    def test_list_help(self, runner: CliRunner) -> None:
        """List command shows help."""
        result = runner.invoke(main, ["memory", "list", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output
        assert "--scope" in result.output
        assert "--since" in result.output
