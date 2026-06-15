# agy Parity Matrix

Claude feature parity table for `headroom wrap agy`.
Status: **WIRED** = actively wired and tested; **N/A** = not wired (concrete evidence given); **DEFERRED** = ticket filed.

| Feature | Status | Mechanism / Evidence |
|---------|--------|----------------------|
| **Context-tool: lean-ctx** | **WIRED** | `wrap.py:4857-4859` calls `_setup_lean_ctx_agent("antigravity-cli", verbose=False)` when `HEADROOM_CONTEXT_TOOL=lean-ctx`. Evidence: `lean-ctx init --help` (run 2026-06-15) lists `antigravity-cli` in supported agents; `lean-ctx init --agent antigravity-cli --dry-run` exits 0 and writes `~/.gemini/antigravity-cli/mcp_config.json`. |
| **Context-tool: rtk** | **WIRED** | `wrap.py:4862` calls `_inject_gemini_md_block(gemini_md, RTK_INSTRUCTIONS_BLOCK)` via the RTK path (default when `HEADROOM_CONTEXT_TOOL` is unset or `rtk`). The block is written to `~/.gemini/GEMINI.md` with markers `<!-- headroom:agy-instructions -->`. `unwrap_agy` removes the block via `_remove_gemini_md_block`. |
| **Context-instructions (GEMINI.md)** | **WIRED** | Same as rtk path above. Injection helpers `_inject_gemini_md_block` (`wrap.py:1355`) / `_remove_gemini_md_block` (`wrap.py:1398`). Merge-not-clobber: user content outside markers is preserved. `unwrap_agy` at `wrap.py:4931` removes only the Headroom block. |
| **Headroom MCP retrieve tool (per-run)** | **N/A-v1 → DEFERRED** | `agy_dispatch.py` binds `port=0` (ephemeral, in-process, dies on session exit). Registering `http://127.0.0.1:<port>` in persistent `~/.gemini/antigravity-cli/mcp_config.json` would leave a dead pointer the next session. Per-run registration is intentionally skipped. `AgyRegistrar` is usable for stable-proxy scenarios via `headroom mcp install`. Follow-up ticket: **headroom-2i0** (stable dispatch endpoint). |
| **Headroom MCP retrieve tool (mcp install / stable proxy)** | **WIRED (install fleet)** | `AgyRegistrar` registered in `get_all_registrars()` (`install.py:21`). `headroom mcp install` will register the spec into `~/.gemini/antigravity-cli/mcp_config.json`. Merge-not-clobber: other `mcpServers` entries preserved. `unwrap_agy` defensively unregisters the `headroom` entry at `wrap.py:4940`. |
| **Serena MCP** | **WIRED** | Serena is a generic `uvx` stdio MCP server (`build_serena_spec`, `install.py:41`) with no proxy-URL/ephemeral-port dependency, so it persists cleanly in `mcp_config.json`. Wired via `_setup_serena_mcp(AgyRegistrar(), context="ide-assistant", force=True)` at `wrap.py:4872` (Antigravity is an IDE agent → Serena's generic IDE profile). `--no-serena` flag on the agy command actively removes a prior Headroom entry via `_disable_serena_mcp` (`wrap.py:4876`). Reverted by ledger-gated `_remove_headroom_installed_serena_mcp(AgyRegistrar())` in `unwrap_agy` (`wrap.py:4947`) — preserves user-managed Serena entries. |
| **Code-graph** | **N/A → DEFERRED** | Code-graph installs `codebase-memory-mcp` via `_setup_code_graph` (`wrap.py:721`) → `_register_cbm_mcp_server`, which is hardwired to the Claude CLI (`shutil.which("claude")` + `claude mcp add` at `wrap.py:697-710`). `codebase-memory-mcp` is a generic MCP server but is NOT wired for agy in v1 (no AgyRegistrar path exists for it). Follow-up ticket: **headroom-30y.13** (wire codebase-memory-mcp via AgyRegistrar). |
| **--memory** | **N/A** | `--memory` wires `ClaudeCodeAdapter` writing to `claude_memory_dir` (`wrap.py:3338-3344`). agy has no equivalent persistent memory directory API exposed to external processes. Deferred: **headroom-2i0** notes this as follow-on. |
| **--learn** | **N/A** | `--learn` wires the RTK learning surface through `AgyDispatchServer` transport side (not MCP). The dispatch server (`agy_dispatch.py`) would need a `--learn` POST endpoint exposed to agy's tool calls. This is transport-side work outside T9 scope. Deferred: **headroom-2i0**. |
| **ENABLE_TOOL_SEARCH** | **N/A** | `ENABLE_TOOL_SEARCH` is a claude-specific env var that activates the web-search tool in Claude Code. agy uses Google Search natively; no equivalent env plumbing needed. No ticket required. |

## Evidence for lean-ctx agy support

```
$ lean-ctx init --help
...
For AI tool integration: lean-ctx init --agent <tool> [--mode <mode>]
  Supported: aider, amazonq, amp, antigravity, antigravity-cli, augment,
    claude, cline, codex, continue, copilot, ...

$ lean-ctx init --agent antigravity-cli --dry-run
Antigravity CLI MCP: lean-ctx already configured at /home/dd/.gemini/antigravity-cli/mcp_config.json
Installed Antigravity CLI plugin at /home/dd/.gemini/config/plugins/lean-ctx
  ✓ Antigravity rules up-to-date
```

## Wiring file:line reference

| Wiring point | File:line |
|---|---|
| GEMINI.md block markers | `wrap.py:863-864` (`_AGY_GEMINI_BLOCK_START/END`) |
| `_inject_gemini_md_block` definition | `wrap.py:1355` |
| `_remove_gemini_md_block` definition | `wrap.py:1398` |
| Context-tool + GEMINI.md injection in `agy()` | `wrap.py:4857-4862` |
| Serena MCP wiring in `agy()` | `wrap.py:4872` (setup) / `wrap.py:4876` (`--no-serena` disable) |
| `unwrap_agy` GEMINI.md reversion | `wrap.py:4931` |
| `unwrap_agy` Headroom MCP unregister | `wrap.py:4940` |
| `unwrap_agy` ledger-gated Serena removal | `wrap.py:4947` (`_remove_headroom_installed_serena_mcp`) |
| `AgyRegistrar` definition | `headroom/mcp_registry/agy.py` |
| `AgyRegistrar` in fleet | `headroom/mcp_registry/install.py:21` |
| `AgyRegistrar` exported | `headroom/mcp_registry/__init__.py` |

## Follow-up tickets

| Ticket | Feature | What's needed |
|--------|---------|---------------|
| **headroom-2i0** | Per-run headroom MCP retrieve wiring; `--learn`; `--memory` | Stable dispatch endpoint (named pipe or fixed port) so `AgyRegistrar.register_server(build_headroom_spec(stable_url))` can be called in `agy()` and reverted in `finally`/`unwrap_agy`. |
| **headroom-30y.13** | Code-graph (`codebase-memory-mcp`) | `_register_cbm_mcp_server` is hardwired to the Claude CLI (`claude mcp add`). Wire `codebase-memory-mcp` via `AgyRegistrar` so agy gets the code-graph MCP. |
