"""Security validation tests for Headroom.

These tests verify security measures against common attack vectors:
- SQL injection via metadata keys
- CCR hash spoofing attacks
- JSON path injection

These tests exist as regression tests to ensure security fixes remain effective.
"""

import pytest

from headroom.memory.adapters.sqlite import _validate_metadata_key


class TestSQLiteMetadataKeyValidation:
    """Test metadata key validation to prevent JSON path injection.

    Metadata keys are interpolated into json_extract() SQL expressions.
    Without validation, malicious keys could escape the JSON path and
    inject arbitrary SQL.
    """

    def test_valid_simple_key(self):
        """Accept simple alphanumeric keys."""
        assert _validate_metadata_key("status") is True
        assert _validate_metadata_key("user_id") is True
        assert _validate_metadata_key("item_count") is True

    def test_valid_key_with_hyphens(self):
        """Accept keys with hyphens (common in APIs)."""
        assert _validate_metadata_key("content-type") is True
        assert _validate_metadata_key("x-custom-header") is True

    def test_valid_key_starting_with_underscore(self):
        """Accept keys starting with underscore."""
        assert _validate_metadata_key("_internal") is True
        assert _validate_metadata_key("_id") is True

    def test_rejects_empty_key(self):
        """Reject empty keys."""
        assert _validate_metadata_key("") is False

    def test_rejects_json_path_injection(self):
        """Reject keys that could escape JSON path expression."""
        # Attempt to close the JSON path and add SQL
        assert _validate_metadata_key("'] OR 1=1--") is False
        assert _validate_metadata_key("key') OR 1=1--") is False
        assert _validate_metadata_key('key") OR 1=1--') is False

    def test_rejects_sql_injection_patterns(self):
        """Reject keys with SQL injection patterns."""
        assert _validate_metadata_key("key; DROP TABLE memories;--") is False
        assert _validate_metadata_key("key UNION SELECT * FROM users") is False
        assert _validate_metadata_key("1=1") is False

    def test_rejects_special_characters(self):
        """Reject keys with special characters."""
        assert _validate_metadata_key("key.nested") is False  # Dots
        assert _validate_metadata_key("key[0]") is False  # Brackets
        assert _validate_metadata_key("key$") is False  # Dollar sign
        assert _validate_metadata_key("key@domain") is False  # At sign
        assert _validate_metadata_key("key/path") is False  # Slashes
        assert _validate_metadata_key("key\\path") is False  # Backslashes
        assert _validate_metadata_key("key'quote") is False  # Quotes
        assert _validate_metadata_key('key"quote') is False  # Double quotes

    def test_rejects_keys_starting_with_number(self):
        """Reject keys starting with a number."""
        assert _validate_metadata_key("123key") is False
        assert _validate_metadata_key("0_prefix") is False

    def test_rejects_very_long_keys(self):
        """Reject excessively long keys (potential DoS)."""
        long_key = "a" * 256
        assert _validate_metadata_key(long_key) is False

        # 255 chars should be acceptable
        valid_long_key = "a" * 255
        assert _validate_metadata_key(valid_long_key) is True

    def test_rejects_unicode_bypass_attempts(self):
        """Reject Unicode characters that might bypass filtering."""
        # Various Unicode quote-like characters
        assert _validate_metadata_key("key\u2019") is False  # Right single quote
        assert _validate_metadata_key("key\u201c") is False  # Left double quote
        assert _validate_metadata_key("key\u0000") is False  # Null byte


class TestSQLiteMetadataFilteringIntegration:
    """Integration tests for metadata filtering with validation."""

    @pytest.mark.asyncio
    async def test_malicious_metadata_filter_is_skipped(self):
        """Malicious metadata keys should be silently skipped, not cause errors."""
        import tempfile
        from pathlib import Path

        from headroom.memory.adapters.sqlite import SQLiteMemoryStore
        from headroom.memory.models import Memory
        from headroom.memory.ports import MemoryFilter

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            store = SQLiteMemoryStore(db_path)

            # Create a memory with safe metadata
            memory = Memory(
                id="test-1",
                content="Test content",
                user_id="alice",
                metadata={"safe_key": "value"},
            )
            await store.save(memory)

            # Attempt to query with malicious metadata key - should not raise
            # The malicious key should be silently skipped
            malicious_filter = MemoryFilter(
                user_id="alice",
                metadata_filters={"'] OR 1=1--": "malicious"},
            )
            results = await store.query(malicious_filter)

            # Query should succeed (malicious key skipped)
            # Results may or may not include our memory depending on other conditions
            assert isinstance(results, list)

            # Query with valid metadata filter should work normally
            valid_filter = MemoryFilter(
                user_id="alice",
                metadata_filters={"safe_key": "value"},
            )
            results = await store.query(valid_filter)
            assert len(results) == 1
            assert results[0].id == "test-1"

        finally:
            db_path.unlink(missing_ok=True)
