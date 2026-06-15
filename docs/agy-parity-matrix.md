# agy Parity Matrix

Claude feature parity table for `headroom wrap agy`.
Status: **WIRED** = actively wired and tested; **N/A** = not wired (concrete evidence given); **DEFERRED** = ticket filed.

| Feature | Status | Mechanism / Evidence |
|---------|--------|----------------------|
| **Context-tool: lean-ctx** | **WIRED (verified, interactive only)** | When `HEADROOM_CONTEXT_TOOL=lean-ctx`, interactive `wrap agy` registers an explicit `lean-ctx mcp` MCP entry via `AgyRegistrar` (`build_lean_ctx_spec`, `install.py`) and smoke-verifies the MCP `initialize` handshake (`_smoke_verify_mcp_handshake`); on handshake failure the entry is removed so a broken tool can never persist. **Caveat (live-verified 2026-06-16):** agy's `--print` / `-p` / `--prompt` single-shot mode HANGS whenever a context-tool MCP is active (lean-ctx confirmed hangs even though it handshakes fine standalone), so `wrap agy` skips context-tool wiring for **all** print-mode forms — both space-separated (`--print hi`) and `=`-joined (`--print=hi`, `--prompt=hi`, `-p=hi`) — detected by `_agy_print_mode` (`wrap.py`). (The attached short form `-pVALUE` is intentionally not matched: agy itself rejects it with exit 2 before MCP init, so it cannot hang.) Requires the `lean-ctx` binary present; absent → skipped with a notice (agy still works transport-only). |
| **Context-tool: rtk** | **WIRED (verified, presence-gated)** | Default path (when `HEADROOM_CONTEXT_TOOL` is unset or `rtk`). Interactive `wrap agy` injects `RTK_INSTRUCTIONS_BLOCK` into `~/.gemini/GEMINI.md` **only when `shutil.which("rtk")` is present** — otherwise the block would tell agy to use a missing tool, so it is skipped with a notice. The block uses markers `<!-- headroom:agy-instructions -->`; `unwrap_agy` removes it via `_remove_gemini_md_block`. Print-mode runs skip context wiring (see lean-ctx caveat). |
| **Context-instructions (GEMINI.md)** | **WIRED** | Same as rtk path above. Injection helpers `_inject_gemini_md_block` (`wrap.py:1355`) / `_remove_gemini_md_block` (`wrap.py:1398`). Merge-not-clobber: user content outside markers is preserved. `unwrap_agy` at `wrap.py:4931` removes only the Headroom block. |
| **Headroom MCP retrieve tool (per-run)** | **WIRED (interactive only; ephemeral, reverted)** | Resolves `[Retrieve more: hash=…]` markers produced by the HTTPS dispatch MITM. The marker cache is the **process-global** `get_compression_store()` singleton (`headroom/cache/compression_store.py`), so a SECOND in-process `create_app()` shares it. Because the dispatch server is HTTPS with a Cloud-Code-SNI leaf only (an `headroom mcp serve` stdio child can't reach it over loopback), interactive `wrap agy` stands up a **second loopback listener — PLAIN HTTP, no TLS — on an ephemeral port** (`AgyRetrieveServer`, `headroom/proxy/agy_retrieve.py`) for the session, started alongside the terminator+dispatch on the same background loop in `_start_agy_servers(..., start_retrieve=True)` (`wrap.py`). The headroom retrieve MCP (`build_headroom_spec(f"http://127.0.0.1:{retrieve_port}")`, `install.py`) is then registered via `AgyRegistrar` and smoke-verified (`_smoke_verify_mcp_handshake`, verify-then-remove on failure) by `_setup_headroom_retrieve_mcp_agy` (`wrap.py`). The per-run URL is **ephemeral**, so the entry is **reverted** in `agy()`'s `finally` and the SIGTERM handler via `_revert_headroom_retrieve_mcp_agy` (`wrap.py`) — never a dead pointer in `mcp_config.json`; `unwrap_agy` also removes the `headroom` entry. **Caveats:** (1) **interactive-only** — agy's `--print`/`-p`/`--prompt` mode HANGS with ANY MCP server active (`_agy_print_mode`, headroom-30y.18), so in print mode the listener is NOT started and no entry is registered; (2) the listener + retrieve HTTP endpoint are **headless-testable** (load-bearing test: a hash stored via `get_compression_store()` resolves over plain HTTP `GET /v1/retrieve/{hash}` from the second `create_app()` — `tests/test_agy_retrieve.py`), but agy actually **invoking** the tool mid-conversation is interactive-only and not headless-proven. Ref: **headroom-2i0**. |
| **Headroom MCP retrieve tool (mcp install / stable proxy)** | **WIRED (install fleet)** | `AgyRegistrar` registered in `get_all_registrars()` (`install.py:21`). `headroom mcp install` will register the spec into `~/.gemini/antigravity-cli/mcp_config.json`. Merge-not-clobber: other `mcpServers` entries preserved. `unwrap_agy` defensively unregisters the `headroom` entry at `wrap.py:4940`. |
| **Serena MCP** | **WIRED** | Serena is a generic `uvx` stdio MCP server (`build_serena_spec`, `install.py:41`) with no proxy-URL/ephemeral-port dependency, so it persists cleanly in `mcp_config.json`. Wired via `_setup_serena_mcp(AgyRegistrar(), context="ide-assistant", force=True)` at `wrap.py:4872` (Antigravity is an IDE agent → Serena's generic IDE profile). `--no-serena` flag on the agy command actively removes a prior Headroom entry via `_disable_serena_mcp` (`wrap.py:4876`). Reverted by ledger-gated `_remove_headroom_installed_serena_mcp(AgyRegistrar())` in `unwrap_agy` (`wrap.py:4947`) — preserves user-managed Serena entries. |
| **Print-mode MCP suppression (scope)** | **WIRED (honest limitation)** | For print-mode runs (`_agy_print_mode`), `wrap agy` suppresses only **Headroom-owned** MCP entries: it skips context-tool wiring and removes a Headroom-installed Serena via ledger-gated `_disable_serena_mcp`. **A user's own pre-existing MCP servers** in `~/.gemini/antigravity-cli/mcp_config.json` are deliberately **left untouched** — Headroom never silently deletes user-managed MCP entries. Consequence: a user-managed MCP that misbehaves in agy print mode (e.g. a first-run `uvx` cold-start can be slow, or a stalling server) is subject to agy's own print-mode MCP behavior and is outside Headroom's control. Users who hit this can remove or `lean-ctx`-disable the offending entry themselves. |
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
| **headroom-2i0** | Per-run headroom MCP retrieve wiring — **DONE** (interactive only, per-run ephemeral PLAIN-HTTP loopback listener `AgyRetrieveServer`, registered+smoke-verified then reverted on teardown). Remaining: `--learn`; `--memory`. | Retrieve markers resolve via a second loopback `create_app()` sharing the process-global compression cache; see the matrix row above. `--learn`/`--memory` remain transport-side / no-equivalent-API work. |
| **headroom-30y.13** | Code-graph (`codebase-memory-mcp`) | `_register_cbm_mcp_server` is hardwired to the Claude CLI (`claude mcp add`). Wire `codebase-memory-mcp` via `AgyRegistrar` so agy gets the code-graph MCP. |
