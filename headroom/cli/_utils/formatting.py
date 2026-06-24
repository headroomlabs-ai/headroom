"""Formatting utilities for CLI output using Rich."""

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Shared console instance for consistent output
console = Console()


def print_table(
    headers: list[str],
    rows: list[list[str]],
    title: str | None = None,
) -> None:
    """Print a Rich table with headers and rows.

    Args:
        headers: Column headers for the table.
        rows: List of rows, where each row is a list of cell values.
        title: Optional title to display above the table.
    """
    table = Table(title=title)

    for header in headers:
        table.add_column(header)

    for row in rows:
        table.add_row(*row)

    console.print(table)


def print_stats(stats: dict[str, Any], title: str = "Statistics") -> None:
    """Print statistics in a nicely formatted panel.

    Args:
        stats: Dictionary of stat names to values.
        title: Title for the panel.
    """
    lines = [f"[bold]{key}:[/bold] {value}" for key, value in stats.items()]
    content = "\n".join(lines)
    console.print(Panel(content, title=title))


def print_error(msg: str) -> None:
    """Print an error message in red.

    Args:
        msg: The error message to display.
    """
    console.print(f"[bold red]Error:[/bold red] {msg}")


def print_success(msg: str) -> None:
    """Print a success message in green.

    Args:
        msg: The success message to display.
    """
    console.print(f"[bold green]Success:[/bold green] {msg}")


def print_warning(msg: str) -> None:
    """Print a warning message in yellow.

    Args:
        msg: The warning message to display.
    """
    console.print(f"[bold yellow]Warning:[/bold yellow] {msg}")


def truncate(text: str, max_len: int = 50) -> str:
    """Truncate text to a maximum length with ellipsis.

    Args:
        text: The text to truncate.
        max_len: Maximum length including ellipsis.

    Returns:
        Truncated text with ellipsis if it exceeded max_len.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_age(dt: datetime) -> str:
    """Format a datetime as a human-readable age string.

    Args:
        dt: The datetime to format.

    Returns:
        A string like "2d", "5h", "30m" representing time elapsed.
    """
    now = datetime.now(tz=dt.tzinfo)
    delta = now - dt

    total_seconds = int(delta.total_seconds())

    if total_seconds < 0:
        return "now"

    minutes = total_seconds // 60
    hours = total_seconds // 3600
    days = total_seconds // 86400

    if days > 0:
        return f"{days}d"
    if hours > 0:
        return f"{hours}h"
    if minutes > 0:
        return f"{minutes}m"
    return "now"


def format_bytes(num_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        num_bytes: Number of bytes to format.

    Returns:
        A string like "1.2 MB", "500 KB", "256 B".
    """
    if num_bytes < 0:
        return "0 B"

    units = [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]

    for unit, threshold in units:
        if num_bytes >= threshold:
            value = num_bytes / threshold
            if value >= 10:
                return f"{value:.0f} {unit}"
            return f"{value:.1f} {unit}"

    return "0 B"
