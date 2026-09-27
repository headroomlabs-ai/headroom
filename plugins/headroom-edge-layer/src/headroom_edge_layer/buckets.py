"""Tool-name-first bucketing, shared by census.py and the pipeline dispatcher.

Grouping by *tool name* first (not detected content type) is deliberate per
the correction that content-type-first grouping would hide the actual go/kill
question: whether the tools Headroom already excludes by name (`Read`,
`Glob`, `Grep`, `WebFetch`, ... — see `headroom/config.py`) matter. A JSON
config file read through `Read` must show up under `file_read`, not `mcp_json`,
or the census can't answer that question. Content type is a secondary column,
computed separately (see census.py), never the primary grouping key.

Tool names drift across agent-client versions, so every set below is small,
documented, and meant to be extended rather than treated as exhaustive.
"""

from __future__ import annotations

BUCKET_FILE_READ = "file_read"
BUCKET_GREP = "grep"
BUCKET_WEB = "web"
BUCKET_SHELL_LOG = "shell_log"
BUCKET_MCP_JSON = "mcp_json"
BUCKET_OTHER = "other"

FILE_READ_TOOL_NAMES = frozenset({"Read", "read", "read_file", "view", "cat", "fs_read"})
GREP_TOOL_NAMES = frozenset({"Grep", "grep", "search", "ripgrep", "rg", "ag", "Glob", "glob"})
WEB_TOOL_NAMES = frozenset(
    {"WebFetch", "WebSearch", "web_fetch", "web_search", "browser", "fetch"}
)
SHELL_LOG_TOOL_NAMES = frozenset(
    {"Bash", "bash", "shell", "run_command", "execute", "terminal", "test", "run_tests"}
)


def classify_tool_name(tool_name: str | None) -> str:
    if not tool_name:
        return BUCKET_OTHER
    if tool_name in FILE_READ_TOOL_NAMES:
        return BUCKET_FILE_READ
    if tool_name in GREP_TOOL_NAMES:
        return BUCKET_GREP
    if tool_name in WEB_TOOL_NAMES:
        return BUCKET_WEB
    if tool_name in SHELL_LOG_TOOL_NAMES:
        return BUCKET_SHELL_LOG
    if "mcp" in tool_name.lower():
        return BUCKET_MCP_JSON
    return BUCKET_OTHER
