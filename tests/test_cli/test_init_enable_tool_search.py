"""`headroom init claude` must set ENABLE_TOOL_SEARCH (GH #746).

Claude Code disables on-demand tool loading when ANTHROPIC_BASE_URL is a custom
host and ENABLE_TOOL_SEARCH is unset, materializing all MCP/system tool schemas
into the context window — which breaks sub-agent spawns and forces compaction.
`headroom wrap claude` already sets it; `init` (and the persistent install) must
too, otherwise init-wired users silently get eager tools.
"""

from __future__ import annotations

import json
from pathlib import Path

from headroom.cli import init as init_cli
from headroom.providers.claude import install as claude_install


def test_ensure_claude_hooks_sets_enable_tool_search(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    init_cli._ensure_claude_hooks(settings, profile="init-user", port=8787)

    env = json.loads(settings.read_text(encoding="utf-8"))["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert env["ENABLE_TOOL_SEARCH"] == "true"


def test_ensure_claude_hooks_respects_user_tool_search_value(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"env": {"ENABLE_TOOL_SEARCH": "auto"}}) + "\n", encoding="utf-8"
    )
    init_cli._ensure_claude_hooks(settings, profile="init-user", port=8787)

    env = json.loads(settings.read_text(encoding="utf-8"))["env"]
    assert env["ENABLE_TOOL_SEARCH"] == "auto"  # setdefault preserves the user's choice


def test_build_install_env_includes_enable_tool_search() -> None:
    env = claude_install.build_install_env(port=8787, backend="anthropic")
    assert env["ENABLE_TOOL_SEARCH"] == "true"
    assert env["ANTHROPIC_BASE_URL"].endswith(":8787")
