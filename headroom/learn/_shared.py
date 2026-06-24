"""Shared utilities for headroom learn plugins.

Error classification, tool name normalization, and other helpers
used across all scanner plugins.
"""

from __future__ import annotations

import re

from .models import ErrorCategory

# =============================================================================
# Error Classification
# =============================================================================

# Patterns checked in order — first match wins
_ERROR_PATTERNS: list[tuple[re.Pattern[str], ErrorCategory]] = [
    (
        re.compile(r"No such file or directory|ENOENT|FileNotFoundError|does not exist", re.I),
        ErrorCategory.FILE_NOT_FOUND,
    ),
    (
        re.compile(r"ModuleNotFoundError|ImportError|No module named", re.I),
        ErrorCategory.MODULE_NOT_FOUND,
    ),
    (re.compile(r"command not found", re.I), ErrorCategory.COMMAND_NOT_FOUND),
    (
        re.compile(r"Permission denied|EACCES|EPERM|auto-denied", re.I),
        ErrorCategory.PERMISSION_DENIED,
    ),
    (
        re.compile(r"file is too large|too many lines|exceeds.*limit", re.I),
        ErrorCategory.FILE_TOO_LARGE,
    ),
    (re.compile(r"EISDIR|Is a directory", re.I), ErrorCategory.IS_DIRECTORY),
    (re.compile(r"SyntaxError|IndentationError", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"Traceback \(most recent|Exception:|Error:", re.I), ErrorCategory.RUNTIME_ERROR),
    (re.compile(r"timed? ?out|TimeoutError|deadline exceeded", re.I), ErrorCategory.TIMEOUT),
    (re.compile(r"No (?:matches|files|results) found|0 matches", re.I), ErrorCategory.NO_MATCHES),
    (
        re.compile(r"user.*reject|user.*denied|declined|didn't want to proceed", re.I),
        ErrorCategory.USER_REJECTED,
    ),
    (re.compile(r"[Ss]ibling tool call errored", re.I), ErrorCategory.SIBLING_ERROR),
    (re.compile(r"exit code|non-zero|exited with", re.I), ErrorCategory.EXIT_CODE),
    (
        re.compile(r"ConnectionError|ConnectionRefused|ECONNREFUSED|network", re.I),
        ErrorCategory.CONNECTION_ERROR,
    ),
    (
        re.compile(r"BUILD FAILED|compilation error|compile error", re.I),
        ErrorCategory.BUILD_FAILURE,
    ),
]


def classify_error(content: str) -> ErrorCategory:
    """Classify an error message into a category."""
    for pattern, category in _ERROR_PATTERNS:
        if pattern.search(content[:2000]):  # Only check first 2KB
            return category
    return ErrorCategory.UNKNOWN


def is_error_content(content: str) -> bool:
    """Heuristic: does this tool result look like an error?"""
    if not content or len(content) < 10:
        return False
    # Check for common error indicators in first 1KB
    snippet = content[:1000]
    indicators = [
        "Error:",
        "error:",
        "ENOENT",
        "No such file",
        "command not found",
        "Permission denied",
        "ModuleNotFoundError",
        "Traceback (most recent",
        "FAILED",
        "EISDIR",
        "auto-denied",
        "Sibling tool call errored",
        "timed out",
        "exit code",
        "FileNotFoundError",
    ]
    return any(ind in snippet for ind in indicators)


# =============================================================================
# Tool Name Normalization
# =============================================================================

# Consolidated mapping from all agent-specific tool names to the cross-agent schema.
# Plugins can use normalize_tool_name() or extend this map for custom tools.
_TOOL_NAME_MAP: dict[str, str] = {
    # Shell / command execution
    "shell": "Bash",
    "run_shell_command": "Bash",
    "execute_command": "Bash",
    "exec_command": "Bash",
    "terminal": "Bash",
    "run_command": "Bash",
    "run_terminal_command": "Bash",
    # File reading
    "read_file": "Read",
    "read_many_files": "Read",
    "readfile": "Read",
    "view_file": "Read",
    "cat": "Read",
    # File writing
    "write_file": "Write",
    "write_new_file": "Write",
    "create_file": "Write",
    "writefile": "Write",
    # File editing
    "edit_file": "Edit",
    "replace_in_file": "Edit",
    "editfile": "Edit",
    "apply_diff": "Edit",
    # File search / glob
    "search_files": "Glob",
    "find_files": "Glob",
    "glob": "Glob",
    "list_directory": "Glob",
    "list_dir": "Glob",
    # Text search / grep
    "grep": "Grep",
    "search_text": "Grep",
    "search_code": "Grep",
    "codebase_search": "Grep",
    # Web
    "browser": "WebFetch",
    "web_search": "WebSearch",
}


def normalize_tool_name(name: str) -> str:
    """Map agent-specific tool names to the cross-agent schema.

    Looks up the name (case-insensitive) in the shared tool name map.
    Returns the original name if no mapping exists.
    """
    return _TOOL_NAME_MAP.get(name.lower(), _TOOL_NAME_MAP.get(name, name))
