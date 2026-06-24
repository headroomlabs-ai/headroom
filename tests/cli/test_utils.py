"""Tests for CLI utilities."""

from datetime import datetime, timedelta

import click
import pytest

from headroom.cli._utils import (
    format_age,
    format_bytes,
    parse_duration,
    truncate,
)


class TestParseDuration:
    """Tests for parse_duration function."""

    def test_parse_days(self) -> None:
        """Parse days correctly."""
        result = parse_duration("7d")
        assert result == timedelta(days=7)

    def test_parse_weeks(self) -> None:
        """Parse weeks correctly."""
        result = parse_duration("2w")
        assert result == timedelta(weeks=2)

    def test_parse_months(self) -> None:
        """Parse months as 30 days."""
        result = parse_duration("1m")
        assert result == timedelta(days=30)

    def test_parse_hours(self) -> None:
        """Parse hours correctly."""
        result = parse_duration("24h")
        assert result == timedelta(hours=24)

    def test_parse_large_numbers(self) -> None:
        """Parse large duration values."""
        result = parse_duration("365d")
        assert result == timedelta(days=365)

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_duration("invalid")

    def test_empty_string_raises(self) -> None:
        """Empty string raises BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_duration("")

    def test_no_unit_raises(self) -> None:
        """Number without unit raises BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_duration("7")

    def test_invalid_unit_raises(self) -> None:
        """Invalid unit raises BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_duration("7x")

    def test_negative_not_allowed(self) -> None:
        """Negative values raise BadParameter."""
        with pytest.raises(click.BadParameter):
            parse_duration("-7d")


class TestFormatAge:
    """Tests for format_age function."""

    def test_minutes_ago(self) -> None:
        """Format recent time as minutes."""
        dt = datetime.now() - timedelta(minutes=30)
        result = format_age(dt)
        assert result == "30m"

    def test_hours_ago(self) -> None:
        """Format hours ago."""
        dt = datetime.now() - timedelta(hours=5)
        result = format_age(dt)
        assert result == "5h"

    def test_days_ago(self) -> None:
        """Format days ago."""
        dt = datetime.now() - timedelta(days=3)
        result = format_age(dt)
        assert result == "3d"

    def test_weeks_ago(self) -> None:
        """Format weeks ago for older dates."""
        dt = datetime.now() - timedelta(weeks=2)
        result = format_age(dt)
        assert result == "14d"

    def test_just_now(self) -> None:
        """Very recent time shows as 'now'."""
        dt = datetime.now() - timedelta(seconds=30)
        result = format_age(dt)
        assert result == "now"


class TestFormatBytes:
    """Tests for format_bytes function."""

    def test_bytes(self) -> None:
        """Format small values as bytes."""
        assert format_bytes(500) == "500 B"

    def test_kilobytes(self) -> None:
        """Format KB correctly."""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(2048) == "2.0 KB"

    def test_megabytes(self) -> None:
        """Format MB correctly."""
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self) -> None:
        """Format GB correctly."""
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"

    def test_zero_bytes(self) -> None:
        """Zero bytes handled correctly."""
        assert format_bytes(0) == "0 B"

    def test_fractional_kb(self) -> None:
        """Fractional KB shows decimal."""
        result = format_bytes(1536)  # 1.5 KB
        assert "1.5 KB" == result


class TestTruncate:
    """Tests for truncate function."""

    def test_short_string_unchanged(self) -> None:
        """Short strings are not modified."""
        result = truncate("hello", max_len=10)
        assert result == "hello"

    def test_exact_length_unchanged(self) -> None:
        """String at exact length is not modified."""
        result = truncate("hello", max_len=5)
        assert result == "hello"

    def test_long_string_truncated(self) -> None:
        """Long strings are truncated with ellipsis."""
        result = truncate("hello world", max_len=8)
        assert result == "hello..."
        assert len(result) == 8

    def test_empty_string(self) -> None:
        """Empty string returns empty."""
        result = truncate("", max_len=10)
        assert result == ""

    def test_default_length(self) -> None:
        """Default max length is 50."""
        long_string = "a" * 100
        result = truncate(long_string)
        assert len(result) == 50
        assert result.endswith("...")

    def test_with_newlines(self) -> None:
        """Truncate works with newlines in text."""
        result = truncate("hello\nworld", max_len=20)
        # Newlines are preserved
        assert result == "hello\nworld"
