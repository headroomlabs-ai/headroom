"""Tests for SQLite storage implementation."""

import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from headroom.config import RequestMetrics
from headroom.storage.sqlite import SQLiteStorage


class TestSQLiteStorageInit:
    """Tests for SQLiteStorage initialization."""

    def test_creates_db_file(self, temp_sqlite_db):
        """Test that initialization creates the database file."""
        # Remove the temp file first so we can verify it gets created
        Path(temp_sqlite_db).unlink(missing_ok=True)
        assert not Path(temp_sqlite_db).exists()

        storage = SQLiteStorage(temp_sqlite_db)
        assert Path(temp_sqlite_db).exists()
        storage.close()

    def test_creates_tables(self, temp_sqlite_db):
        """Test that initialization creates the required tables."""
        storage = SQLiteStorage(temp_sqlite_db)

        conn = sqlite3.connect(temp_sqlite_db)
        cursor = conn.cursor()

        # Check that requests table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "requests"

        # Verify table schema has expected columns
        cursor.execute("PRAGMA table_info(requests)")
        columns = {row[1] for row in cursor.fetchall()}
        expected_columns = {
            "id",
            "timestamp",
            "model",
            "stream",
            "mode",
            "tokens_input_before",
            "tokens_input_after",
            "tokens_output",
            "block_breakdown",
            "waste_signals",
            "stable_prefix_hash",
            "cache_alignment_score",
            "cached_tokens",
            "transforms_applied",
            "tool_units_dropped",
            "turns_dropped",
            "messages_hash",
            "error",
        }
        assert expected_columns.issubset(columns)

        conn.close()
        storage.close()

    def test_creates_indices(self, temp_sqlite_db):
        """Test that initialization creates the required indices."""
        storage = SQLiteStorage(temp_sqlite_db)

        conn = sqlite3.connect(temp_sqlite_db)
        cursor = conn.cursor()

        # Get all indices
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in cursor.fetchall()}

        # Check expected indices exist
        assert "idx_timestamp" in indices
        assert "idx_model" in indices
        assert "idx_mode" in indices

        conn.close()
        storage.close()

    def test_parent_directory_created(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "nested" / "test.db"
            assert not db_path.parent.exists()

            storage = SQLiteStorage(str(db_path))
            assert db_path.parent.exists()
            assert db_path.exists()
            storage.close()


class TestSave:
    """Tests for SQLiteStorage.save method."""

    def test_save_request_metrics(self, temp_sqlite_db, sample_request_metrics):
        """Test saving request metrics to database."""
        storage = SQLiteStorage(temp_sqlite_db)
        storage.save(sample_request_metrics)

        # Verify data was saved
        result = storage.get(sample_request_metrics.request_id)
        assert result is not None
        assert result.request_id == sample_request_metrics.request_id
        assert result.model == sample_request_metrics.model
        assert result.tokens_input_before == sample_request_metrics.tokens_input_before
        storage.close()

    def test_save_overwrites_existing(self, temp_sqlite_db, sample_request_metrics):
        """Test that save with same request_id overwrites existing record (INSERT OR REPLACE)."""
        storage = SQLiteStorage(temp_sqlite_db)

        # Save initial metrics
        storage.save(sample_request_metrics)

        # Create modified metrics with same request_id
        modified_metrics = RequestMetrics(
            request_id=sample_request_metrics.request_id,
            timestamp=sample_request_metrics.timestamp,
            model="gpt-4o-mini",  # Changed model
            stream=True,  # Changed stream
            mode="optimize",  # Changed mode
            tokens_input_before=2000,  # Changed tokens
            tokens_input_after=1500,
            tokens_output=300,
            block_breakdown={"system": 200},
            waste_signals={"json_bloat": 100},
            stable_prefix_hash="xyz789",
            cache_alignment_score=95.0,
            cached_tokens=200,
            transforms_applied=["ContentRouter"],
            tool_units_dropped=2,
            turns_dropped=1,
            messages_hash="ghi789",
        )

        # Save modified metrics
        storage.save(modified_metrics)

        # Verify only one record exists and it has the modified values
        result = storage.get(sample_request_metrics.request_id)
        assert result is not None
        assert result.model == "gpt-4o-mini"
        assert result.stream is True
        assert result.mode == "optimize"
        assert result.tokens_input_before == 2000
        assert result.tokens_input_after == 1500

        # Verify count is still 1
        assert storage.count() == 1
        storage.close()

    def test_save_all_fields(self, temp_sqlite_db):
        """Test that all fields are correctly saved and retrieved."""
        storage = SQLiteStorage(temp_sqlite_db)

        metrics = RequestMetrics(
            request_id="full-test-123",
            timestamp=datetime(2025, 1, 6, 14, 30, 45),
            model="claude-3-opus",
            stream=True,
            mode="optimize",
            tokens_input_before=5000,
            tokens_input_after=3500,
            tokens_output=1200,
            block_breakdown={"system": 500, "user": 1000, "assistant": 2000, "tool": 1500},
            waste_signals={"json_bloat": 200, "whitespace": 100, "repetition": 50},
            stable_prefix_hash="stablehash123",
            cache_alignment_score=92.5,
            cached_tokens=750,
            transforms_applied=["CacheAligner", "SmartCrusher", "ContentRouter"],
            tool_units_dropped=3,
            turns_dropped=2,
            messages_hash="msgshash456",
            error=None,
        )

        storage.save(metrics)
        result = storage.get("full-test-123")

        assert result is not None
        assert result.request_id == "full-test-123"
        assert result.timestamp == datetime(2025, 1, 6, 14, 30, 45)
        assert result.model == "claude-3-opus"
        assert result.stream is True
        assert result.mode == "optimize"
        assert result.tokens_input_before == 5000
        assert result.tokens_input_after == 3500
        assert result.tokens_output == 1200
        assert result.block_breakdown == {
            "system": 500,
            "user": 1000,
            "assistant": 2000,
            "tool": 1500,
        }
        assert result.waste_signals == {"json_bloat": 200, "whitespace": 100, "repetition": 50}
        assert result.stable_prefix_hash == "stablehash123"
        assert result.cache_alignment_score == 92.5
        assert result.cached_tokens == 750
        assert result.transforms_applied == ["CacheAligner", "SmartCrusher", "ContentRouter"]
        assert result.tool_units_dropped == 3
        assert result.turns_dropped == 2
        assert result.messages_hash == "msgshash456"
        assert result.error is None
        storage.close()


class TestGet:
    """Tests for SQLiteStorage.get method."""

    def test_get_by_request_id(self, temp_sqlite_db, sample_request_metrics):
        """Test retrieving metrics by request ID."""
        storage = SQLiteStorage(temp_sqlite_db)
        storage.save(sample_request_metrics)

        result = storage.get(sample_request_metrics.request_id)

        assert result is not None
        assert result.request_id == sample_request_metrics.request_id
        assert result.model == sample_request_metrics.model
        assert result.mode == sample_request_metrics.mode
        storage.close()

    def test_get_nonexistent_returns_none(self, temp_sqlite_db):
        """Test that getting a non-existent record returns None."""
        storage = SQLiteStorage(temp_sqlite_db)

        result = storage.get("nonexistent-id")

        assert result is None
        storage.close()


class TestQuery:
    """Tests for SQLiteStorage.query method."""

    @pytest.fixture
    def storage_with_data(self, temp_sqlite_db):
        """Create storage with multiple test records."""
        storage = SQLiteStorage(temp_sqlite_db)

        # Create multiple records with different attributes
        base_time = datetime(2025, 1, 6, 12, 0, 0)
        records = [
            RequestMetrics(
                request_id=f"query-test-{i}",
                timestamp=base_time + timedelta(hours=i),
                model="gpt-4o" if i % 2 == 0 else "gpt-4o-mini",
                stream=i % 2 == 0,
                mode="audit" if i % 3 == 0 else "optimize",
                tokens_input_before=1000 + i * 100,
                tokens_input_after=800 + i * 50,
                tokens_output=200 + i * 10,
                block_breakdown={"system": 100},
                waste_signals={},
                stable_prefix_hash=f"hash{i}",
                cache_alignment_score=80.0 + i,
                cached_tokens=50 + i * 10,
                transforms_applied=[],
            )
            for i in range(10)
        ]

        for record in records:
            storage.save(record)

        yield storage
        storage.close()

    def test_query_by_model(self, storage_with_data):
        """Test querying by model filter."""
        results = storage_with_data.query(model="gpt-4o")

        assert len(results) == 5
        for result in results:
            assert result.model == "gpt-4o"

    def test_query_by_mode(self, storage_with_data):
        """Test querying by mode filter."""
        results = storage_with_data.query(mode="audit")

        # i % 3 == 0 for i in 0-9: 0, 3, 6, 9 = 4 records
        assert len(results) == 4
        for result in results:
            assert result.mode == "audit"

    def test_query_by_time_range(self, storage_with_data):
        """Test querying by time range."""
        base_time = datetime(2025, 1, 6, 12, 0, 0)
        start_time = base_time + timedelta(hours=3)
        end_time = base_time + timedelta(hours=7)

        results = storage_with_data.query(start_time=start_time, end_time=end_time)

        # Hours 3, 4, 5, 6, 7 = 5 records
        assert len(results) == 5
        for result in results:
            assert start_time <= result.timestamp <= end_time

    def test_query_with_limit_offset(self, storage_with_data):
        """Test querying with limit and offset."""
        # Get all to verify total
        all_results = storage_with_data.query(limit=100)
        assert len(all_results) == 10

        # Test limit
        limited_results = storage_with_data.query(limit=3)
        assert len(limited_results) == 3

        # Test offset
        offset_results = storage_with_data.query(limit=3, offset=3)
        assert len(offset_results) == 3

        # Verify offset works correctly (results are ordered by timestamp DESC)
        assert limited_results[0].request_id != offset_results[0].request_id

    def test_query_order_by_timestamp_desc(self, storage_with_data):
        """Test that query results are ordered by timestamp descending."""
        results = storage_with_data.query()

        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i].timestamp >= results[i + 1].timestamp

        # The most recent record (hour 9) should be first
        assert results[0].request_id == "query-test-9"


class TestCount:
    """Tests for SQLiteStorage.count method."""

    @pytest.fixture
    def storage_with_mixed_data(self, temp_sqlite_db):
        """Create storage with mixed test data for counting."""
        storage = SQLiteStorage(temp_sqlite_db)

        base_time = datetime(2025, 1, 6, 12, 0, 0)
        records = [
            RequestMetrics(
                request_id=f"count-test-{i}",
                timestamp=base_time + timedelta(hours=i),
                model="gpt-4o" if i < 5 else "claude-3-opus",
                stream=False,
                mode="audit" if i < 3 else "optimize",
                tokens_input_before=1000,
                tokens_input_after=800,
                block_breakdown={},
                waste_signals={},
            )
            for i in range(10)
        ]

        for record in records:
            storage.save(record)

        yield storage
        storage.close()

    def test_count_all(self, storage_with_mixed_data):
        """Test counting all records."""
        count = storage_with_mixed_data.count()
        assert count == 10

    def test_count_with_filters(self, storage_with_mixed_data):
        """Test counting with various filters."""
        # Count by model
        gpt_count = storage_with_mixed_data.count(model="gpt-4o")
        assert gpt_count == 5

        claude_count = storage_with_mixed_data.count(model="claude-3-opus")
        assert claude_count == 5

        # Count by mode
        audit_count = storage_with_mixed_data.count(mode="audit")
        assert audit_count == 3

        optimize_count = storage_with_mixed_data.count(mode="optimize")
        assert optimize_count == 7

        # Count by time range
        base_time = datetime(2025, 1, 6, 12, 0, 0)
        time_count = storage_with_mixed_data.count(
            start_time=base_time + timedelta(hours=2),
            end_time=base_time + timedelta(hours=5),
        )
        assert time_count == 4

        # Combined filters
        combined_count = storage_with_mixed_data.count(model="gpt-4o", mode="audit")
        assert combined_count == 3  # First 3 are both gpt-4o and audit


class TestIterAll:
    """Tests for SQLiteStorage.iter_all method."""

    @pytest.fixture
    def storage_with_ordered_data(self, temp_sqlite_db):
        """Create storage with data for iteration testing."""
        storage = SQLiteStorage(temp_sqlite_db)

        # Create records with specific timestamps for ordering verification
        timestamps = [
            datetime(2025, 1, 6, 10, 0, 0),
            datetime(2025, 1, 6, 14, 0, 0),
            datetime(2025, 1, 6, 8, 0, 0),
            datetime(2025, 1, 6, 16, 0, 0),
            datetime(2025, 1, 6, 12, 0, 0),
        ]

        for i, ts in enumerate(timestamps):
            storage.save(
                RequestMetrics(
                    request_id=f"iter-test-{i}",
                    timestamp=ts,
                    model="gpt-4o",
                    stream=False,
                    mode="audit",
                    tokens_input_before=1000,
                    tokens_input_after=800,
                    block_breakdown={},
                    waste_signals={},
                )
            )

        yield storage
        storage.close()

    def test_iter_all_returns_all(self, storage_with_ordered_data):
        """Test that iter_all returns all records."""
        results = list(storage_with_ordered_data.iter_all())
        assert len(results) == 5

        # Verify all request IDs are present
        request_ids = {r.request_id for r in results}
        expected_ids = {f"iter-test-{i}" for i in range(5)}
        assert request_ids == expected_ids

    def test_iter_all_ordered_by_timestamp(self, storage_with_ordered_data):
        """Test that iter_all returns results ordered by timestamp ascending."""
        results = list(storage_with_ordered_data.iter_all())

        # Verify ascending order
        for i in range(len(results) - 1):
            assert results[i].timestamp <= results[i + 1].timestamp

        # The earliest record (8:00) should be first
        assert results[0].timestamp == datetime(2025, 1, 6, 8, 0, 0)
        # The latest record (16:00) should be last
        assert results[-1].timestamp == datetime(2025, 1, 6, 16, 0, 0)


class TestGetSummaryStats:
    """Tests for SQLiteStorage.get_summary_stats method."""

    @pytest.fixture
    def storage_with_stats_data(self, temp_sqlite_db):
        """Create storage with data for statistics testing."""
        storage = SQLiteStorage(temp_sqlite_db)

        base_time = datetime(2025, 1, 6, 12, 0, 0)
        records = [
            RequestMetrics(
                request_id="stats-1",
                timestamp=base_time,
                model="gpt-4o",
                stream=False,
                mode="audit",
                tokens_input_before=1000,
                tokens_input_after=800,
                tokens_output=200,
                block_breakdown={},
                waste_signals={},
                cache_alignment_score=80.0,
            ),
            RequestMetrics(
                request_id="stats-2",
                timestamp=base_time + timedelta(hours=1),
                model="gpt-4o",
                stream=False,
                mode="optimize",
                tokens_input_before=2000,
                tokens_input_after=1500,
                tokens_output=300,
                block_breakdown={},
                waste_signals={},
                cache_alignment_score=90.0,
            ),
            RequestMetrics(
                request_id="stats-3",
                timestamp=base_time + timedelta(hours=2),
                model="gpt-4o",
                stream=False,
                mode="audit",
                tokens_input_before=1500,
                tokens_input_after=1200,
                tokens_output=250,
                block_breakdown={},
                waste_signals={},
                cache_alignment_score=85.0,
            ),
            RequestMetrics(
                request_id="stats-4",
                timestamp=base_time + timedelta(hours=3),
                model="gpt-4o",
                stream=False,
                mode="optimize",
                tokens_input_before=3000,
                tokens_input_after=2000,
                tokens_output=400,
                block_breakdown={},
                waste_signals={},
                cache_alignment_score=95.0,
            ),
        ]

        for record in records:
            storage.save(record)

        yield storage
        storage.close()

    def test_summary_stats_totals(self, storage_with_stats_data):
        """Test that summary statistics calculates correct totals."""
        stats = storage_with_stats_data.get_summary_stats()

        assert stats["total_requests"] == 4
        # Total tokens before: 1000 + 2000 + 1500 + 3000 = 7500
        assert stats["total_tokens_before"] == 7500
        # Total tokens after: 800 + 1500 + 1200 + 2000 = 5500
        assert stats["total_tokens_after"] == 5500
        # Total tokens saved: 7500 - 5500 = 2000
        assert stats["total_tokens_saved"] == 2000
        # Audit count: 2, Optimize count: 2
        assert stats["audit_count"] == 2
        assert stats["optimize_count"] == 2

    def test_summary_stats_averages(self, storage_with_stats_data):
        """Test that summary statistics calculates correct averages."""
        stats = storage_with_stats_data.get_summary_stats()

        # Average tokens saved: (200 + 500 + 300 + 1000) / 4 = 500
        assert stats["avg_tokens_saved"] == 500.0
        # Average cache alignment: (80 + 90 + 85 + 95) / 4 = 87.5
        assert stats["avg_cache_alignment"] == 87.5

    def test_summary_stats_with_time_range(self, storage_with_stats_data):
        """Test summary statistics with time range filter."""
        base_time = datetime(2025, 1, 6, 12, 0, 0)

        # Get stats for only the middle 2 records (hours 1 and 2)
        stats = storage_with_stats_data.get_summary_stats(
            start_time=base_time + timedelta(hours=1),
            end_time=base_time + timedelta(hours=2),
        )

        assert stats["total_requests"] == 2
        # Tokens before: 2000 + 1500 = 3500
        assert stats["total_tokens_before"] == 3500
        # Tokens after: 1500 + 1200 = 2700
        assert stats["total_tokens_after"] == 2700
        # Average cache alignment: (90 + 85) / 2 = 87.5
        assert stats["avg_cache_alignment"] == 87.5

    def test_summary_stats_empty_db(self, temp_sqlite_db):
        """Test summary statistics on empty database."""
        storage = SQLiteStorage(temp_sqlite_db)
        stats = storage.get_summary_stats()

        assert stats["total_requests"] == 0
        assert stats["total_tokens_before"] == 0
        assert stats["total_tokens_after"] == 0
        assert stats["total_tokens_saved"] == 0
        assert stats["avg_tokens_saved"] == 0
        assert stats["avg_cache_alignment"] == 0
        assert stats["audit_count"] == 0
        assert stats["optimize_count"] == 0
        storage.close()


class TestRowToMetrics:
    """Tests for SQLiteStorage._row_to_metrics method."""

    def test_converts_all_fields(self, temp_sqlite_db):
        """Test that _row_to_metrics correctly converts all database fields."""
        storage = SQLiteStorage(temp_sqlite_db)

        original = RequestMetrics(
            request_id="row-convert-test",
            timestamp=datetime(2025, 1, 6, 15, 30, 0),
            model="gpt-4o",
            stream=True,
            mode="optimize",
            tokens_input_before=2500,
            tokens_input_after=2000,
            tokens_output=500,
            block_breakdown={"system": 200, "user": 300, "assistant": 400, "tool": 100},
            waste_signals={"json_bloat": 150, "whitespace": 75},
            stable_prefix_hash="prefix123",
            cache_alignment_score=88.5,
            cached_tokens=450,
            transforms_applied=["Transform1", "Transform2"],
            tool_units_dropped=5,
            turns_dropped=3,
            messages_hash="msghash789",
            error="Test error message",
        )

        storage.save(original)
        retrieved = storage.get("row-convert-test")

        assert retrieved is not None
        assert retrieved.request_id == original.request_id
        assert retrieved.timestamp == original.timestamp
        assert retrieved.model == original.model
        assert retrieved.stream == original.stream
        assert retrieved.mode == original.mode
        assert retrieved.tokens_input_before == original.tokens_input_before
        assert retrieved.tokens_input_after == original.tokens_input_after
        assert retrieved.tokens_output == original.tokens_output
        assert retrieved.block_breakdown == original.block_breakdown
        assert retrieved.waste_signals == original.waste_signals
        assert retrieved.stable_prefix_hash == original.stable_prefix_hash
        assert retrieved.cache_alignment_score == original.cache_alignment_score
        assert retrieved.cached_tokens == original.cached_tokens
        assert retrieved.transforms_applied == original.transforms_applied
        assert retrieved.tool_units_dropped == original.tool_units_dropped
        assert retrieved.turns_dropped == original.turns_dropped
        assert retrieved.messages_hash == original.messages_hash
        assert retrieved.error == original.error
        storage.close()

    def test_handles_null_optional_fields(self, temp_sqlite_db):
        """Test that _row_to_metrics handles NULL values for optional fields."""
        storage = SQLiteStorage(temp_sqlite_db)

        # Create metrics with minimal/null optional fields
        minimal = RequestMetrics(
            request_id="null-fields-test",
            timestamp=datetime(2025, 1, 6, 12, 0, 0),
            model="gpt-4o",
            stream=False,
            mode="audit",
            tokens_input_before=1000,
            tokens_input_after=800,
            tokens_output=None,  # NULL
            block_breakdown={},
            waste_signals={},
            stable_prefix_hash="",  # Will be stored as NULL or empty
            cache_alignment_score=0.0,
            cached_tokens=None,  # NULL
            transforms_applied=[],
            tool_units_dropped=0,
            turns_dropped=0,
            messages_hash="",
            error=None,  # NULL
        )

        storage.save(minimal)
        retrieved = storage.get("null-fields-test")

        assert retrieved is not None
        assert retrieved.tokens_output is None
        assert retrieved.cached_tokens is None
        assert retrieved.error is None
        # Empty strings should be handled properly
        assert retrieved.stable_prefix_hash == ""
        assert retrieved.messages_hash == ""
        assert retrieved.block_breakdown == {}
        assert retrieved.waste_signals == {}
        assert retrieved.transforms_applied == []
        storage.close()


class TestConcurrency:
    """Tests for SQLiteStorage concurrency handling."""

    def test_multiple_saves(self, temp_sqlite_db):
        """Test that multiple sequential saves work correctly."""
        storage = SQLiteStorage(temp_sqlite_db)

        # Save many records sequentially
        for i in range(100):
            metrics = RequestMetrics(
                request_id=f"concurrent-{i}",
                timestamp=datetime(2025, 1, 6, 12, 0, 0),
                model="gpt-4o",
                stream=False,
                mode="audit",
                tokens_input_before=1000,
                tokens_input_after=800,
                block_breakdown={},
                waste_signals={},
            )
            storage.save(metrics)

        # Verify all records were saved
        count = storage.count()
        assert count == 100

        # Verify specific records can be retrieved
        assert storage.get("concurrent-0") is not None
        assert storage.get("concurrent-50") is not None
        assert storage.get("concurrent-99") is not None
        storage.close()

    def test_connection_management(self, temp_sqlite_db):
        """Test that connections are properly managed."""
        storage = SQLiteStorage(temp_sqlite_db)

        # Connection should be None initially
        assert storage._conn is None

        # Save should create connection
        metrics = RequestMetrics(
            request_id="conn-test",
            timestamp=datetime(2025, 1, 6, 12, 0, 0),
            model="gpt-4o",
            stream=False,
            mode="audit",
            tokens_input_before=1000,
            tokens_input_after=800,
            block_breakdown={},
            waste_signals={},
        )
        storage.save(metrics)
        assert storage._conn is not None

        # Multiple operations should reuse connection
        conn_before = storage._conn
        storage.get("conn-test")
        assert storage._conn is conn_before

        storage.query()
        assert storage._conn is conn_before

        # Close should clear connection
        storage.close()
        assert storage._conn is None

        # Operations after close should create new connection
        result = storage.get("conn-test")
        assert result is not None
        assert storage._conn is not None
        storage.close()

    def test_thread_safety_multiple_instances(self, temp_sqlite_db):
        """Test that multiple storage instances can work with same database."""
        results = []
        errors = []

        def worker(worker_id: int):
            try:
                # Each thread creates its own storage instance
                storage = SQLiteStorage(temp_sqlite_db)
                for i in range(10):
                    metrics = RequestMetrics(
                        request_id=f"thread-{worker_id}-{i}",
                        timestamp=datetime(2025, 1, 6, 12, 0, 0),
                        model="gpt-4o",
                        stream=False,
                        mode="audit",
                        tokens_input_before=1000,
                        tokens_input_after=800,
                        block_breakdown={},
                        waste_signals={},
                    )
                    storage.save(metrics)
                storage.close()
                results.append(worker_id)
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Run multiple threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # Verify all records were saved
        storage = SQLiteStorage(temp_sqlite_db)
        count = storage.count()
        assert count == 50  # 5 threads * 10 records each
        storage.close()
