# D2 (Rust engine) — remaining work

Status snapshot for branch `realign-D1-engine-facade` (PR #606). Temporary
tracking doc — delete once D2 lands.

## Goal (plain words)
Move Headroom's compression "brain" out of the Python proxy into Rust, so it
plugs into any proxy as lightweight hooks — with **no regressions**, and
**no prompt-cache busts, no matter what**.

Phases: **D1** = split engine→facade in Python (done) → **D2** = rebuild engine
in Rust + prove parity → **D3** = retire the Python proxy.

## Done (built in Rust + parity-proven, on #606)
- **OpenAI Responses** — `compress_openai_responses_request`; harness
  `crates/headroom-proxy/tests/engine_parity_responses.rs`.
- **OpenAI Chat** — `compress_openai_chat_request`; harness
  `crates/headroom-proxy/tests/engine_parity_chat.rs`.
- **Anthropic Messages** — `compress_anthropic_request`; harness
  `crates/headroom-proxy/tests/engine_parity_anthropic.rs` (17/17). Validated
  the key hypothesis: **byte-surgery supersedes the Python cache state machine**
  (frozen prefix preserved without the Rust path consuming
  `frozen_count`/`prev_forwarded_messages`).
- **E3 `cache_control` auto-placement** made config-gated:
  `cache_control_auto_place` (CLI `--cache-control-auto-place`, env
  `HEADROOM_PROXY_CACHE_CONTROL_AUTO_PLACE`), default `Enabled`; `Disabled` ==
  exact Python-proxy parity (Python never injects). Decision: keep ON — it's
  double-gated (PAYG-only + skip-if-marker), so Claude Code is unaffected either
  way (subscription → auth gate; API-key → its own markers trip the skip gate).
- Fixed 4 pre-existing CI failures on #606 (engine adapter + a test dummy that
  hadn't tracked handler-surface drift from the main-merge / D1 flip).

Parity bar (established D2.4): NOT byte-identity to the Python golden. Assert
cache-safety + quality + correctness, because Rust reassembles via byte-surgery
(cache-safe by construction) while Python re-serializes with `json.dumps`.

## Left

### D2.8 — wire the Rust engine into the Python proxy (IN PROGRESS, blocked on disk)
The shadow/flip machinery already exists from D1: handlers
(`headroom/proxy/handlers/anthropic.py`, `.../openai.py`) call the engine facade
under `HEADROOM_ENGINE_REQUEST_PATH` = `off` | `shadow` | `on`. Today the facade
runs **Python** compression, so shadow validates Python-vs-Python. D2.8 makes
the facade call **Rust** via the `_core` pyo3 extension, so shadow becomes a live
Rust-vs-Python comparison and `on` flips to Rust.

Key findings from recon:
- pyo3 extension = `crates/headroom-py` (builds `headroom._core`), depends
  **only** on `headroom-core` (NOT `headroom-proxy`). So the seam is the
  **live-zone primitives** in `headroom-core`, not the request-level
  `compress_*_request` wrappers (those live in `headroom-proxy`). The facade
  already owns request orchestration (mode/auth/skip) from D1.
- Currently exposed: `compress_openai_responses_live_zone(body, auth_mode="payg",
  model="") -> (bytes, changed, savings, strategies, reason)` — but it has
  **zero Python callers** (exposed-but-unused; D2 prep). So the cutover is not
  live for any provider yet.
- NOT yet exposed (exist in `headroom-core`, see `transforms/mod.rs`):
  `compress_openai_chat_live_zone`, `compress_anthropic_live_zone`.

Sub-steps:
1. Expose `compress_openai_chat_live_zone` + `compress_anthropic_live_zone` via
   pyo3 in `crates/headroom-py/src/lib.rs` (mirror the responses wrapper at
   ~line 1497 + register in the `_core` pymodule at ~line 1564).
2. Make the engine facade (`headroom/engine/facade.py`) call the `_core` Rust
   primitives for compression instead of the Python pipeline (today
   `_ResponsesCompressor` etc. bind `OpenAIHandlerMixin` Python methods).
3. Build the extension: `maturin develop` (or `scripts/build_rust_extension.sh`).
4. Run `HEADROOM_ENGINE_REQUEST_PATH=shadow` → confirm Rust-vs-Python parity in
   the existing handler shadow comparisons; then validate `on`.
5. Decide the **prose** gap (below) before flipping `on` for any path that hits
   prose.
6. Per-provider routing during cutover: gemini stays on Python until D2.7 (this
   is explicit routing, not a silent fallback).

### Prose compressor — DECISION NEEDED
The Python ML "Kompress" prose compressor has **no Rust equivalent** — the one
genuine gap. Decide: keep prose in Python (facade calls Python for the
prose/structure-boundary path, Rust for everything else) vs defer prose entirely
for the cutover. Until decided, do not flip `on` for paths that compress prose.

### D2.7 — Gemini in Rust (GREENFIELD, after D2.8 per user)
Nothing exists yet:
- No Rust gemini compression surface (no `compress_gemini_request`, nothing
  gemini in `crates/headroom-proxy/src/compression/`).
- No gemini fixtures (only `engine_request_golden_openai/` + `engine_request_golden/`).
- Python path: `headroom/proxy/handlers/gemini.py` + the `GeminiHandlerMixin`
  (internal codename "ln"); converts gemini `contents[]`/`parts` ↔ messages and
  reuses the shared compression pipeline.

Work: build a Rust gemini live-zone walker (byte-surgery over `contents`/`parts`
+ `systemInstruction`), generate goldens, add a parity harness. Note gemini's
cache semantics differ from Anthropic (context caching, different API) — design
the live-zone boundary deliberately.

### D3 — retire the Python proxy
Once Rust is flipped `on` and proven in production (shadow clean, then on),
remove the Python compression path. This is Phase H.

## Tool-sort nuance (don't blindly match Python)
Python sorts `tools[]` unconditionally (all auth modes, even optimize-off); Rust
gates E1/E2 sort behind optimize-on + PAYG. Rust is **safer** for
OAuth/Subscription (leaves their bytes byte-exact, avoiding the
cache-evasion/revocation risk Python's unconditional mutation takes). The one
real gap: an optimize-off PAYG client with unsorted tools loses cache-stable
ordering under Rust. Separate decision; not folded into E3.

## Blocker
Disk is **100% full system-wide** (~888G/926G used, ~1 GB free). cargo `target/`
is ~14G of it. D2.8's pyo3/maturin rebuild won't fit. Either `cargo clean`
(frees ~14G, next build is a slow from-scratch recompile) or free system space
first. Cannot validate the cutover locally until resolved.

## Key files
- Rust request entries: `crates/headroom-proxy/src/compression/{live_zone_responses,live_zone_anthropic}.rs`,
  `compress_openai_chat_request` (compression module).
- Rust live-zone primitives (pyo3 seam): `crates/headroom-core/src/transforms/live_zone.rs`
  (`compress_*_live_zone`, exported in `transforms/mod.rs`).
- pyo3 bindings: `crates/headroom-py/src/lib.rs`.
- Python facade: `headroom/engine/facade.py`.
- Cutover flag handling: `headroom/proxy/handlers/{anthropic,openai}.py`,
  `headroom/proxy/models.py` (`HEADROOM_ENGINE_REQUEST_PATH`).
- Config flags: `crates/headroom-proxy/src/config.rs`
  (`CacheControlAutoFrozen`, `CacheControlAutoPlace`).
- Parity harnesses: `crates/headroom-proxy/tests/engine_parity_*.rs`.
- Fixtures: `tests/parity/fixtures/engine_request_golden{,_openai}/`.
