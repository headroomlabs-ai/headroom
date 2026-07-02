# Changelog

All notable changes to Headroom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Fixed
- `headroom learn` now honors `CLAUDE_CONFIG_DIR`. It resolved the Claude
  config directory as `~/.claude` and wrote global memory to
  `~/.claude/CLAUDE.md`, so users who relocate their Claude config via that
  env var had `learn` scan the wrong directory and detect no projects. The
  scanner and memory writer now read/write the configured directory
  ([#1630](https://github.com/headroomlabs-ai/headroom/issues/1630)).
- `--backend bedrock` now fails fast with an actionable error when temporary
  AWS credentials (`AWS_SESSION_TOKEN`) are used but botocore is not installed
  (e.g. the slim default Docker image). litellm's session-token auth path
  imports botocore, so the missing dependency previously surfaced only at
  request time as a misleading `authentication_error: No module named
  'botocore'`. The proxy now tells the user to install the `bedrock` extra up
  front ([#1551](https://github.com/headroomlabs-ai/headroom/issues/1551)).
- Content detection no longer crashes the proxy on text containing an
  orphaned `+++ ` target line with no preceding `--- ` source line (common in
  `set -x` xtrace output and partial diffs). The bundled `unidiff` 0.4.0 parser
  panics on that input instead of returning an error; the Rust diff detector now
  contains the panic and treats the fragment as plain text, so the request is
  compressed and forwarded normally instead of returning HTTP 500
  ([#1547](https://github.com/headroomlabs-ai/headroom/issues/1547)).
- Proactive expansion blocks injected into user turns are now wrapped in
  `<headroom_proactive_expansion>` XML tags, giving downstream consumers
  (LLMs, loggers, attribution parsers) a machine-readable provenance
  boundary and preventing misattribution in multi-agent threads.
- **cli:** the startup banner no longer advertises
  `HEADROOM_COMPRESSION_STABLE_AFTER_TURN` and
  `HEADROOM_STALE_READ_COMPRESS_AFTER_TURNS` as tuning knobs. Both were read
  only to render the `Performance Tuning` banner section and were never wired
  into the compression path, so setting them changed the banner but had no
  effect on behavior. The banner now surfaces only the embedding sidecar,
  which is a real, consumed setting.
- **memory/embedder:** cap CPU thread oversubscription in the local
  torch/sentence-transformers embedder. Concurrent encodes previously each
  fanned out to ~`os.cpu_count()` BLAS/OpenMP threads, so under load the memory
  path starved the asyncio event loop and spiked `/livez` latency to several
  seconds. CPU encodes now run on a dedicated, size-limited executor whose
  workers each pin their thread pool, bounding total embedding threads to
  `HEADROOM_EMBED_CONCURRENCY` × `HEADROOM_EMBED_NUM_THREADS` (defaults
  `min(4, cpu)` × 1). The ONNX embedder already capped its threads; this brings
  the torch path to parity
  ([#198](https://github.com/headroomlabs-ai/headroom/issues/198)).

### Changed

* **telemetry:** anonymous usage telemetry is now **opt-in** (off by default) instead of opt-out. Nothing is collected or sent unless you set `HEADROOM_TELEMETRY=on` or pass `--telemetry` to `headroom proxy` / `headroom install apply`. `is_telemetry_enabled()` is fail-closed — only explicit on-values (`on`/`true`/`1`/`yes`/`enable`/`enabled`) enable it; unset, empty, or unrecognized values stay disabled. The existing `--no-telemetry` flag and `HEADROOM_TELEMETRY=off` remain accepted for back-compat, and install manifests now write the `HEADROOM_TELEMETRY` value explicitly so generated deployments are unambiguous.
* **ccr:** `headroom_stats` now labels its formatted proxy output as a rolling/window-scoped session and adds a lifetime savings section from `/stats persistent_savings.lifetime` when present, while keeping existing summary structure and fallback JSON output behavior.

### Features

* **wrap:** `headroom wrap claude --1m` preserves the 1M context window. Behind a custom `ANTHROPIC_BASE_URL` (the proxy) Claude Code drops the `context-1m` beta header and caps the window at 200k for entitled subscription users; the opt-in flag sets `ANTHROPIC_MODEL=<opus>[1m]` on the launched process so the 1M window activates through Headroom. A model already selected via `ANTHROPIC_MODEL` is preserved (only the `[1m]` suffix is appended) ([#1158](https://github.com/chopratejas/headroom/issues/1158)).
* **learn:** weight loops in `headroom learn`. A new loop detector (`headroom/learn/loops.py`) recognizes repeated tool-call patterns — including RTK re-fetch loops, where RTK's output truncation makes the agent re-run larger-limit variants of a *successful* command — collapses output-limit variants to one signature, measures the wasted tokens, surfaces loops as a highest-priority digest section, and weights loop guardrails above one-off rules by their measured waste. Previously loops had no special weight and a no-failure re-fetch loop was skipped entirely. Adds an RTK-loop eval (`benchmarks/rtk_loop_learn_eval.py`) that reproduces a loop, runs it through Learn, and asserts the generated guardrail ranks first and prevents re-triggering.
* **learn:** write per-project learnings to the personal, gitignored `CLAUDE.local.md` by default instead of the team-shared `CLAUDE.md`, matching Claude Code's memory convention so machine-specific paths and tool-discovery byproducts no longer pollute the shared file. Adds a `--target` flag to override the destination (e.g. `--target CLAUDE.md` to opt back into the shared file, or any custom path), and auto-migrates a stale learned-patterns block out of an existing `CLAUDE.md` into `CLAUDE.local.md` with a warning ([#1072](https://github.com/chopratejas/headroom/issues/1072)).
* **proxy/transforms:** take large cold-start contexts off the synchronous kompress path — the root cause behind the `compression_first_stage` 30s-timeout + leaked-thread → executor-saturation cascade ([#1171](https://github.com/chopratejas/headroom/issues/1171)). A token size-gate inside the ML boundary routes oversized text away from ModernBERT (`HEADROOM_KOMPRESS_MAX_TOKENS`); a cooperative chunk-deadline bounds any kompress run that does proceed (`HEADROOM_COMPRESSION_DEADLINE_MS`); an opt-in off-path mode forwards uncompressed immediately and compresses in a single per-process background drain so the request never blocks on ML (`HEADROOM_BACKGROUND_COMPRESSION`); and a new native `TextCrusher` — a fast deterministic extractive prose compressor in `headroom._core` that reuses the shared BM25 relevance scorer — is the fast alternative to ModernBERT for large plain text (`HEADROOM_TEXT_CRUSHER`). All default off and fail-open. On a SQuAD answer-retention eval (requires the SQuAD dev set) TextCrusher keeps ~94% of buried answers at 30% size vs ~36% for truncate/random, and runs in one O(n) pass -- sub-second where ModernBERT takes minutes (self-contained speed benchmark in `benchmarks/text_crusher_quality_eval.py`).
* **proxy:** measure and surface rolling and current token throughput metrics (active/wall-clock input, compression, effective forward, and streamed generation) in `headroom perf` CLI and the dashboard ([#959](https://github.com/chopratejas/headroom/issues/959)).
* **vibe:** add Mistral Vibe CLI support with `headroom wrap vibe`.
* **proxy:** per-project savings breakdown on the dashboard for all wrapped agents — Claude Code, Codex, aider, Copilot, and Cursor ([#802](https://github.com/chopratejas/headroom/issues/802)). `headroom wrap claude`/`codex` tag requests with an `X-Headroom-Project` header (launch-directory name); `wrap aider`/`copilot`/`cursor` — whose clients cannot send custom headers — use a `/p/<name>` base-URL prefix the proxy strips. Savings are aggregated per project (persisted, schema v3 with transparent v2 migration), exposed as `savings.per_project` in `/stats` and `projects` in `/stats-history`, and shown in a Per-Project Savings dashboard table.
* **memory:** opt-in Apple-GPU (MPS) embedding offload via `HEADROOM_EMBEDDER_RUNTIME=pytorch_mps`. When set (and Apple MPS is available), the memory embedder runs on the torch sentence-transformers backend on the Apple GPU instead of the default ONNX CPU embedder, freeing the CPU under load. If MPS or the dependencies are unavailable, Headroom logs a warning and uses the existing default embedder selection path (ONNX when available, then the pre-existing local fallback). MPS encode calls are serialized internally (torch-MPS is not thread-safe). Adds the new `[pytorch-mps]` extra (`pip install 'headroom-ai[pytorch-mps]'`). Default behavior is unchanged.
* **proxy:** cross-region Bedrock inference-profile detection — geo-prefixed model IDs (`eu.`/`us.`/`apac.`/`global.`) are now resolved to their canonical vendor, so Anthropic cross-region profiles (e.g. `eu.anthropic.claude-haiku-4-5-20251001-v1:0`) receive live-zone compression instead of being silently skipped ([#999](https://github.com/chopratejas/headroom/pull/999)).
* **proxy:** Converse-body compression on the native Bedrock route — the live-zone dispatcher now recognizes Bedrock Converse content blocks (typeless `{"text": …}`, not only Anthropic `{"type":"text", …}`), so Converse user-message text compresses; `run_anthropic_compression` no longer bails to passthrough when the body lacks an InvokeModel `anthropic_version` envelope, and envelope re-emit stays gated on successful parse ([#999](https://github.com/chopratejas/headroom/pull/999)).
* **docker:** bundle `headroom-proxy` binary in published `runtime` and `runtime-slim` images — closes [#976](https://github.com/chopratejas/headroom/issues/976) ([#999](https://github.com/chopratejas/headroom/pull/999)).

### Bug Fixes

* **mcp/claude:** stop the file-based MCP registrar from destroying an existing but unparseable Claude config. When the `claude` CLI is unavailable, `_register_via_file` read `~/.claude/.claude.json` via a helper that returns `{}` on `JSONDecodeError`, then rewrote the whole file with only `{"mcpServers": {...}}` — wiping unrelated Claude state (`projects`, `oauthAccount`, session history) if the file was momentarily corrupt or hand-edited. The write path now refuses to overwrite a present-but-invalid config and returns a `FAILED` result with an actionable message; an absent or empty file still registers fresh, and a valid file still merges with all other keys preserved.
* **install:** stop leaking a file descriptor on every `headroom install start`. `start_detached_agent()` opened the agent log file and handed it to `subprocess.Popen` but never closed the parent's copy, so each call leaked one fd (and pinned the log file open against rotation). The parent now closes its copy in a `try/finally` once the child has inherited it — the close also runs if `Popen` raises ([#1554](https://github.com/headroomlabs-ai/headroom/issues/1554)).
* **proxy:** include the system prompt, tools, and the response-shaping request fields in the SemanticCache key. `_compute_key` hashed only `{model, messages}`, so two non-streaming requests with identical messages but a different top-level `system` prompt, tool set, sampling config, or output-shaping field collided on one key and the second caller was served the first's cached response — generated under different request semantics, in the default config (`cache_enabled` defaults on). The key now folds the request fields that shape generation — `temperature`/`top_p`/`top_k`/`max_tokens`/`stop`, plus OpenAI `tool_choice`/`response_format`/`parallel_tool_calls`/`seed`/`presence_penalty`/`frequency_penalty`/`logit_bias`/`n`/`logprobs`/`top_logprobs`/`reasoning_effort`/`verbosity`/`modalities` and Anthropic `thinking`/`tool_choice`/`output_config` — canonicalizing `system`/`tools` so a moved `cache_control` breakpoint does not fragment it, and the handlers snapshot the fields once at the cache read and reuse them at write so a body mutated by the pipeline cannot diverge the key. Non-streaming path only.
* **learn (verbosity):** `--verbosity --apply --all` now aggregates the savings baseline across every project instead of overwriting it per project (last-project-wins), which previously left the output shaper with a tiny, unrepresentative baseline. The applied verbosity level comes from the project with the most samples ([#1288](https://github.com/headroomlabs-ai/headroom/pull/1288)).
* **proxy/anthropic:** restore token-mode compression on continued Claude Code turns with a frozen prefix and deferred CCR tool injection. Token mode now runs request-side compression even when the client did not pre-register `headroom_retrieve`, relying on the existing marker-triggered injection override to keep emitted CCR markers redeemable ([#1487](https://github.com/headroomlabs-ai/headroom/issues/1487)).
* **proxy:** the dedicated OpenAI handlers (`/v1/chat/completions`, `/v1/responses`) now honor the `x-headroom-base-url` request header, matching the generic passthrough route. Previously only the catch-all passthrough honored it, so OpenAI-compatible gateways (LiteLLM, CPA, self-hosted vLLM, Azure OpenAI) routed correctly for passthrough traffic but the dedicated chat/responses handlers ignored the header and fell back to the default `OPENAI_API_URL`, sending requests (and the user's provider key) to the wrong upstream.
* **subscription:** stop zeroing the 5-hour headroom contribution counters on every poll. The rollover check compared `five_hour.resets_at` with a bare `!=`, but the usage API reports that timestamp with second-level jitter (observed flapping between `01:59:59Z` and `02:00:00Z` on consecutive polls within the same window), so a spurious "5h window rolled over" reset fired every poll interval (~5 min) and the dashboard's per-window savings stuck near 0%. Only a forward jump larger than `_ROLLOVER_MIN_ADVANCE` (1 minute) now counts as a real rollover.
* **wrap:** keep the shared proxy alive when the agent that launched it closes *ungracefully* on Windows. `_start_proxy` spawned the proxy without detaching it, so it stayed in the launcher's console and Job object; closing that terminal window (or `taskkill`/a crash) tree-killed the proxy, bypassing the marker-based reference counting in `_make_cleanup` and breaking every other `headroom wrap` instance routed through the same port. The proxy is now created with `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB` (with a graceful fallback when the launcher's Job forbids breakaway); POSIX behavior is unchanged. `CREATE_NO_WINDOW` (rather than `DETACHED_PROCESS`) gives the proxy its own *hidden* console: `DETACHED_PROCESS` leaves a console-subsystem exe (`python.exe`) consoleless, so Windows surfaces a visible console window whose close button kills the proxy.
* **transforms/content_router:** stop replacing `role="tool"` output with a lossy-unrecoverable summary on the live compression path (refs [#1307](https://github.com/chopratejas/headroom/issues/1307)). `ContentRouter.apply()` routed OpenAI-style `role="tool"` string messages — `Bash`/`grep`/`ls`/`cat` output — through the ML/word-drop summarizers; when the result carried no CCR retrieve marker (CCR off, ratio >= 0.8, or the size-gate fallback) the original was unrecoverable and the agent acted on a fabricated summary. Tool-role string content is now kept verbatim unless the compressed form is CCR-recoverable. Assistant/user text is unaffected, and structurally-lossless passes (SmartCrusher/Log/Search) still apply. The Anthropic `tool_result` block path is tracked separately.
* **rtk:** stop `rtk` hook registration from spuriously timing out during `headroom wrap`. Output is captured to a temp file instead of pipes, and `stdin` is closed, so a background process forked by `rtk init` can no longer hold the pipe open and block `subprocess.run` past its 10s timeout after the hooks were already registered.
* **ccr:** stop re-compressing `headroom_retrieve` output, which created an infinite retrieval loop, and stop emitting retrieval markers when the `headroom_retrieve` tool is not injected, which silently dropped data ([#1077](https://github.com/chopratejas/headroom/issues/1077), [#1006](https://github.com/chopratejas/headroom/issues/1006)).
* **dashboard:** include RTK stats in the Historical tab; `/stats-history` now attaches live RTK/CLI-filtering stats the same way the Session tab does, so they survive a proxy restart ([#1177](https://github.com/chopratejas/headroom/issues/1177)).
* **proxy:** stop discarding a finished compression on very large requests. After the transform pipeline completed, a telemetry-only waste-signal re-parse of the *original* messages ran on the critical path; on huge Claude Code transcripts (~400k tokens) that parse could exceed the Anthropic compression timeout, so the proxy failed open and forwarded the uncompressed request despite "Pipeline complete" logging real savings (`tokens_saved: 0`, `transforms_applied: []`, ~31s latency). Waste-signal detection is now skipped above `MAX_WASTE_SIGNAL_DETECTION_TOKENS` (100k) so the compression result stays on the critical path ([#296](https://github.com/chopratejas/headroom/issues/296)).
* **codex:** retag existing Codex threads when `headroom init` injects the `headroom` provider, so Codex Desktop history stays visible. Codex filters its sidebar/search by the active `model_provider`; the init path set `model_provider = "headroom"` without retagging, so existing native `openai` threads disappeared from the menu (data was never deleted, only hidden). `_ensure_codex_provider` now reconciles thread tags openai→headroom, matching what the install and `wrap` paths already do; `headroom unwrap codex` handles the revert direction ([#961](https://github.com/chopratejas/headroom/issues/961)).
* **install:** stop duplicating the container ENTRYPOINT in the `persistent-docker` runtime command. The published image already runs `headroom proxy` as its ENTRYPOINT, but `build_runtime_command` re-added `headroom proxy` after the image name, so the container ran `headroom proxy headroom proxy --host 0.0.0.0 …` and Click aborted with "Got unexpected extra arguments (headroom proxy)" — the deployment never became ready and rollback left nothing running. The runtime command now appends only the proxy flags ([#833](https://github.com/chopratejas/headroom/issues/833)).
* **proxy:** retry upstream `529 overloaded_error` like a 429 on both the streaming and non-streaming forwarders, honoring `Retry-After`. The streaming path previously surfaced a 529 straight to the client with no retry (interactive sessions saw "Overloaded" immediately), and `_retry_request` retried it only via the generic 5xx path — raising on exhaustion instead of returning the 529 verbatim, and ignoring `Retry-After`. A shared `RETRYABLE_OVERLOAD_STATUSES = {429, 529}` keeps the two forwarders in agreement (extends [#1221](https://github.com/headroomlabs-ai/headroom/issues/1221)).
* **gemini:** run compression off the asyncio event loop. The Gemini handlers (`generateContent`, Cloud Code stream, `countTokens`) ran the CPU-bound compression pipeline (Magika detection plus ML compression) synchronously on the loop, stalling every concurrent request for the duration of each Gemini request's compression. They now offload it via the shared compression executor, matching the existing OpenAI and Anthropic paths.
* **proxy:** queue mid-turn user messages on non-Bedrock streaming path instead of silently dropping them — closes [#902](https://github.com/headroomlabs-ai/headroom/issues/902).
* **proxy:** add `--protect-tool-results` / `HEADROOM_PROTECT_TOOL_RESULTS` to prevent lossy compression of exact-output tool results (e.g. `Bash cat`/`grep` results) — closes [#1307](https://github.com/headroomlabs-ai/headroom/issues/1307).
* **cli:** add `--rpm`/`--tpm` and `HEADROOM_RPM`/`HEADROOM_TPM` to the Click proxy command for rate-limit parity with the legacy CLI -- closes [#1350](https://github.com/headroomlabs-ai/headroom/issues/1350) (Problem 1).
* **proxy:** register `ToolResultInterceptorTransform` in explicit transforms list when `HEADROOM_INTERCEPT_ENABLED` is set — closes [#829](https://github.com/headroomlabs-ai/headroom/issues/829).
* **opencode:** write Headroom MCP config as a local stdio server instead of a remote `/mcp` URL, keep provider-only installs from adding MCP config, and allow `install apply --target opencode` ([#1380](https://github.com/headroomlabs-ai/headroom/issues/1380)).
* **code:** keep Python `from __future__` imports before executable code during AST compression and validate compressed Python with `compile(..., "exec")` so compile-time syntax rules are enforced ([#1233](https://github.com/chopratejas/headroom/issues/1233)).
* **proxy:** report real input tokens on the streaming `message_start` event for LiteLLM/Bedrock-backed requests. LiteLLM streaming never surfaces prompt tokens mid-stream, so `message_start.usage.input_tokens` was always `0`; Anthropic clients (e.g. Claude Code) read input-token metrics from that event, underreporting token usage by ~99% in OTel/CloudWatch dashboards. The Bedrock streamer now backfills `input_tokens` with the count Headroom actually sent upstream when the backend leaves it unset, preserving any non-zero value the backend genuinely reports ([#1132](https://github.com/chopratejas/headroom/issues/1132)).
* **proxy:** give buffered Anthropic request paths their own longer read timeout, so long `/v1/messages` turns and Anthropic batch or passthrough reads no longer trip the generic proxy cap while unrelated request timeouts stay unchanged.
* **proxy:** retry upstream 429 rate limits honoring `Retry-After` instead of passing them straight to the client. Both the non-streaming (`_retry_request`) and streaming (`_stream_response`) forwarders returned an upstream 429 verbatim, so a parallel agent fan-out that exceeded the per-minute limit aborted every run; 429s are now retried with backoff (honoring the upstream `Retry-After`, capped at `retry_max_delay_ms`), surfacing only the exhausted 429 to the client ([#1221](https://github.com/chopratejas/headroom/issues/1221)).
* **proxy:** force Responses API `store=true` when Headroom injects memory tools so `previous_response_id` continuations work after memory tool calls from clients that requested `store=false` ([#1103](https://github.com/chopratejas/headroom/pull/1103)).
* **proxy:** build SSL contexts for custom CA bundles so enterprise/private PKI roots work with Python/OpenSSL strict verification.
* **dashboard:** the Proxy $ Saved tile no longer shows a bare `$0.00` when cost pricing is unavailable. Pricing depends on litellm, which pyproject gates off on Python 3.14+, so `/stats` now exposes a top-level `litellm_available` flag and the tile points you to reinstall on Python 3.13 when it is false ([#1296](https://github.com/chopratejas/headroom/pull/1296)).
* **proxy:** the output-savings recorder now reloads the learned baseline before estimating and before each flush, so a baseline written by `headroom learn --verbosity --apply` while the proxy is running takes effect without a restart and the periodic flush no longer overwrites it. Fixes Output Tokens Saved staying at "—" after enabling the shaper ([#1296](https://github.com/chopratejas/headroom/pull/1296)).
* **tokenizers:** bound token-counting of oversized tool-content blobs instead of running `count_text` over the whole serialized string. `count_messages` runs on the proxy request path; serializing is cheap, but `count_text` over a multi-megabyte `tool_result` / `tool_use` string took seconds and could freeze `/health` and in-flight requests. For payloads over ~50KB serialized, `count_text` now runs on an even-spread sample of the string and scales by length; it stays model-accurate, bounded for any blob shape, and biased to under-count. Smaller payloads stay exact.
* **codex:** stop persisting a project-specific `--db` path in the global `headroom_memory` MCP config, so `headroom wrap codex --memory` falls back to the active cwd's `.headroom/memory.db` at runtime while keeping the current project's local bootstrap work scoped correctly ([#1147](https://github.com/chopratejas/headroom/issues/1147)).
* **ccr:** stop emitting Anthropic request-side retrieval markers on frozen-prefix turns when `headroom_retrieve` injection is deferred, so cache-preserving requests forward original content instead of irrecoverable marker-only payloads ([#1006](https://github.com/chopratejas/headroom/issues/1006)).
* **proxy:** route Codex OAuth image generation and edit requests through the ChatGPT Codex image backend, while preserving OpenAI API-key image passthrough ([#1215](https://github.com/chopratejas/headroom/pull/1215)).
* **wrap (codex):** keep RTK guidance in the global Codex `AGENTS.md` instead of modifying the shared project `AGENTS.md` ([#1235](https://github.com/chopratejas/headroom/issues/1235)).
* **subscription:** run the transcript token-window scan off the event loop (`asyncio.to_thread`). The subscription tracker's poll loop scanned every `~/.claude/projects/**/*.jsonl` transcript and `json.loads`'d each line inline on the proxy's single asyncio event loop; on large or long-running sessions this took seconds and froze `/health` and every in-flight proxied request — a periodic "wedge" recurring on the poll interval. The scan now runs in a worker thread so the loop stays responsive.
* **gemini:** resolve future Gemini model capabilities through the shared model registry so token counting and context lookup no longer reject new Gemini families.
* **proxy:** enable SSO credential resolution in the native Bedrock route via the `aws-config` `sso` feature flag, making the credential chain match what `docs/bedrock.md` already documented ([#999](https://github.com/chopratejas/headroom/pull/999)).
* **proxy:** route native Bedrock `/model/{id}/converse` requests to the upstream Converse endpoint instead of the hard-coded `/invoke` action — the non-streaming handler now resolves the action from the inbound path, matching the streaming handler ([#999](https://github.com/chopratejas/headroom/pull/999)).
* **proxy:** preserve byte-faithful `/v1/messages` forwarding when Anthropic tool arrays are already canonical, and only canonicalize-and-mutate tool lists when sorting changes ordering ([#1042](https://github.com/chopratejas/headroom/issues/1042)).
* **ccr:** make retrieval store TTL configurable with `HEADROOM_CCR_TTL_SECONDS`, expose the effective TTL in `/v1/retrieve/stats`, and distinguish expired retrievals from missing hashes.
* **proxy:** make `force_kompress` skip ContentRouter auto-detection during compression and pass savings-profile kwargs through Anthropic batch requests.
* **proxy:** add native Bedrock `/model/{id}/converse-stream` route and forward it through the existing streaming EventStream/SSE pipeline.
* **proxy/kompress:** make pre-upstream backpressure and kompress execution saturation fail-open, so Anthropic requests no longer return 503 during temporary saturation while healthy capacity still compresses and explicit passthrough markers preserve operator visibility ([#1025](https://github.com/headroomlabs-ai/headroom/issues/1025)).
* **wrap (codex):** fix `headroom wrap codex` producing a `config.toml` with duplicate top-level `model_provider` / `openai_base_url` keys (TOML-spec error) when the user had already configured their own provider. The injector now rewrites pre-existing top-level `model_provider` and `openai_base_url` lines in place — the previous value is kept in a `# was: …` trailing comment — instead of unconditionally prepending a duplicate, so `codex` can start against the proxy. The pre-wrap snapshot mechanism continues to byte-for-byte restore the original file on `headroom unwrap codex`.
* **install (macOS):** fix `headroom install restart` / `install start` for launchd `persistent-service` deployments. `stop` `bootout`s the job but `start` only ran `launchctl kickstart`, which cannot recover the un-bootstrapped state `stop`/`restart` leave behind (launchctl error 113), so the proxy was left stopped. `start` now tries `kickstart` (fast path for an already-bootstrapped job) and, on failure, `bootstrap`s the plist fresh — retrying for ~15s to ride out the transient `bootstrap` EIO (error 5) window while launchd releases the label after a `bootout`. `stop` tolerates only the already-absent case (`bootout` ESRCH / error 3) and still raises on any other `bootout` failure ([#1289](https://github.com/headroomlabs-ai/headroom/issues/1289)).
* **wrap:** isolate wrapped proxy subprocess stdout/stderr into `proxy-stdio.log`, so `proxy.log` remains the canonical rotating runtime log and Windows rollover failures from `RotatingFileHandler` are no longer blocked by wrapper stdio handles ([#1184](https://github.com/chopratejas/headroom/issues/1184)).
* **langchain:** fix `HeadroomChatModel.ainvoke()` crashing with `AttributeError: 'AsyncStream' object has no attribute 'model_dump'` when the wrapped model has `streaming=True`. `_agenerate()` now uses a per-call non-streaming copy of the wrapped model instead of mutating shared state across an `await` ([#1285](https://github.com/headroomlabs-ai/headroom/issues/1285)).



## [0.28.0](https://github.com/headroomlabs-ai/headroom/compare/v0.27.0...v0.28.0) (2026-06-29)


### Features

* add --disable-kompress-fallback to restore legacy PASSTHROUGH fallback ([#1185](https://github.com/headroomlabs-ai/headroom/issues/1185)) ([f309244](https://github.com/headroomlabs-ai/headroom/commit/f309244a77fc3fbb74c5db0082e7dcbebd6ffe52))
* add first-class OpenCode support (wrap, learn, mcp install) ([#559](https://github.com/headroomlabs-ai/headroom/issues/559)) ([91cd210](https://github.com/headroomlabs-ai/headroom/commit/91cd2102d7e9bc5d48a594725ecc9593096996ec))
* add HEADROOM_KEEPALIVE_EXPIRY to keep upstream connections warm ([#1124](https://github.com/headroomlabs-ai/headroom/issues/1124)) ([85786b3](https://github.com/headroomlabs-ai/headroom/commit/85786b33a3a88b8c905739aa34ccfafa01a89e5d))
* **azure-foundry:** derive upstream URL from ANTHROPIC_FOUNDRY_RESOURCE ([#1138](https://github.com/headroomlabs-ai/headroom/issues/1138)) ([e5031b0](https://github.com/headroomlabs-ai/headroom/commit/e5031b01219278620431b5560b247e65f1b08a13))
* **cache:** attribute prompt-cache misses to TTL lapse vs prefix change ([#1313](https://github.com/headroomlabs-ai/headroom/issues/1313)) ([#1343](https://github.com/headroomlabs-ai/headroom/issues/1343)) ([4658721](https://github.com/headroomlabs-ai/headroom/commit/4658721ea0bae5d0d061d377428d4031b9722d75))
* **code:** add Perl support to code-aware compressor ([#1125](https://github.com/headroomlabs-ai/headroom/issues/1125)) ([f39858c](https://github.com/headroomlabs-ai/headroom/commit/f39858c23325f9f27b47a738731e7260f7b59d9e))
* headroom wrap opencode / unwrap opencode CLI ([#1105](https://github.com/headroomlabs-ai/headroom/issues/1105)) ([b4571cc](https://github.com/headroomlabs-ai/headroom/commit/b4571cc346f6bba29e600fa82bbf5cf302e8ea27))
* **learn:** weight loops in Headroom Learn + RTK-loop eval ([#1160](https://github.com/headroomlabs-ai/headroom/issues/1160)) ([14e8dc4](https://github.com/headroomlabs-ai/headroom/commit/14e8dc4c8408b8014433ba7589bbb1dff7805134))
* **learn:** write per-project learnings to CLAUDE.local.md by default ([#1115](https://github.com/headroomlabs-ai/headroom/issues/1115)) ([ced75e4](https://github.com/headroomlabs-ai/headroom/commit/ced75e4718b5fd84d07cbd68273dcf9b9ef878a3))
* **proxy:** add request timeout config ([#738](https://github.com/headroomlabs-ai/headroom/issues/738)) ([c0745d4](https://github.com/headroomlabs-ai/headroom/commit/c0745d4161d19e21ca36506f7733f0776e19e1a8))
* **proxy:** pilot hardening — inbound auth, security headers, audit log, air-gap switch ([#1537](https://github.com/headroomlabs-ai/headroom/issues/1537)) ([546ab55](https://github.com/headroomlabs-ai/headroom/commit/546ab553dc31af91d5ef4cec0589ad6db8e76a1d))
* **proxy:** support glob patterns in exclude_tools ([#870](https://github.com/headroomlabs-ai/headroom/issues/870)) ([#1259](https://github.com/headroomlabs-ai/headroom/issues/1259)) ([a2159c0](https://github.com/headroomlabs-ai/headroom/commit/a2159c0b66a7aa1b7f64057a1c8e3e50f0a43e37))
* **read-maturation:** activity-based hold-back Read maturation (Mechanism B) ([#1068](https://github.com/headroomlabs-ai/headroom/issues/1068)) ([723b80c](https://github.com/headroomlabs-ai/headroom/commit/723b80c09123f902197b45b3676065d0e9c77af0))
* **savings:** durable savings ledger + headroom savings command ([#1127](https://github.com/headroomlabs-ai/headroom/issues/1127)) ([978ffa0](https://github.com/headroomlabs-ai/headroom/commit/978ffa0a6ab9da1a75239270e17961530c213b9d))
* **wrap:** add --1m to preserve the 1M context window on wrap claude ([#1158](https://github.com/headroomlabs-ai/headroom/issues/1158)) ([#1351](https://github.com/headroomlabs-ai/headroom/issues/1351)) ([b50d9c1](https://github.com/headroomlabs-ai/headroom/commit/b50d9c17ceca890a0fcc2469b9aff27d0026ca39))
* **wrap:** make tokensave the primary coding-task compressor, Serena the backup ([#1230](https://github.com/headroomlabs-ai/headroom/issues/1230)) ([dca9853](https://github.com/headroomlabs-ai/headroom/commit/dca9853ed9d09fe1bb6d56fcb7bb82b9e90b7dff))


### Bug Fixes

* **agent-evals:** Phase 0 — coding-agent accuracy A/B framework ([#1037](https://github.com/headroomlabs-ai/headroom/issues/1037)) ([84f9871](https://github.com/headroomlabs-ai/headroom/commit/84f9871e303d587f5b406036b97b9f5a689c1b05))
* **agno:** tolerate streaming tool-call SDK objects in parser ([#1312](https://github.com/headroomlabs-ai/headroom/issues/1312)) ([#1336](https://github.com/headroomlabs-ai/headroom/issues/1336)) ([5986c22](https://github.com/headroomlabs-ai/headroom/commit/5986c2260f07788e356e0884179d9b3f4c0df6e3))
* **bedrock:** add boto3 1.41 + CRT for aws login credentials ([#1486](https://github.com/headroomlabs-ai/headroom/issues/1486)) ([4db3bc9](https://github.com/headroomlabs-ai/headroom/commit/4db3bc91d9153ca1acccdc0cb5280da01194bf3e))
* bump codebase-memory-mcp to v0.8.1 ([#1284](https://github.com/headroomlabs-ai/headroom/issues/1284)) ([530318b](https://github.com/headroomlabs-ai/headroom/commit/530318b425cba8fb161111b135451a838d628e96))
* **ccr:** make headroom_retrieve a hash-only full-content lookup ([#1532](https://github.com/headroomlabs-ai/headroom/issues/1532)) ([c2fc4d3](https://github.com/headroomlabs-ai/headroom/commit/c2fc4d3753c193eb61f78286741431fd1303e8ee))
* **ccr:** propagate --no-ccr-marker flag to all compressors ([#1022](https://github.com/headroomlabs-ai/headroom/issues/1022)) ([#1197](https://github.com/headroomlabs-ai/headroom/issues/1197)) ([0c9b42a](https://github.com/headroomlabs-ai/headroom/commit/0c9b42a919b0c570094b7934de686b93dd89b05c))
* **ccr:** skip Anthropic marker emission when tool injection is deferred ([#1273](https://github.com/headroomlabs-ai/headroom/issues/1273)) ([2cae13d](https://github.com/headroomlabs-ai/headroom/commit/2cae13dd798b8abdd9ef94fbcf10a968e70e714e))
* **ci:** extend gitleaks allowlist to cover test fixtures + verified examples ([#1539](https://github.com/headroomlabs-ai/headroom/issues/1539)) ([d2565a6](https://github.com/headroomlabs-ai/headroom/commit/d2565a6983f99fe6733d412405ee7c9e54d99624))
* **ci:** guarantee model present in test shards to end cache-miss flakiness ([#1399](https://github.com/headroomlabs-ai/headroom/issues/1399)) ([2e29c72](https://github.com/headroomlabs-ai/headroom/commit/2e29c7223f7a7694060dfe4e1d99332ad766a70b))
* **ci:** normalize Windows CRLF line endings in PR governance script ([#1012](https://github.com/headroomlabs-ai/headroom/issues/1012)) ([5194388](https://github.com/headroomlabs-ai/headroom/commit/5194388b6652d823ad6ab1d8c17d5572b7f0ec23))
* **cli:** add explicit UTF-8 encoding to file I/O in wrap commands ([#1126](https://github.com/headroomlabs-ai/headroom/issues/1126)) ([#1164](https://github.com/headroomlabs-ai/headroom/issues/1164)) ([a0cb798](https://github.com/headroomlabs-ai/headroom/commit/a0cb7982e3cda52221719b9cceecd4d07e30c176))
* **cli:** fall back gracefully when embedding-server sidecar is absent ([#1206](https://github.com/headroomlabs-ai/headroom/issues/1206)) ([38f1404](https://github.com/headroomlabs-ai/headroom/commit/38f1404432984915924f74997d886b89c420b2a8))
* **cli:** harden all CLI surfaces + fix docs accuracy ([#1491](https://github.com/headroomlabs-ai/headroom/issues/1491)) ([bd76235](https://github.com/headroomlabs-ai/headroom/commit/bd76235f5c43bf2e3184a2c7e40a9954dc347afc))
* **cli:** wire --http2/--no-http2 (HEADROOM_HTTP2) into proxy command ([#1373](https://github.com/headroomlabs-ai/headroom/issues/1373)) ([e06b616](https://github.com/headroomlabs-ai/headroom/commit/e06b61671f5cc23832e7d67bce7944e3601a0732))
* **cli:** wire --rpm/--tpm and HEADROOM_RPM/HEADROOM_TPM to the Click proxy command ([#1375](https://github.com/headroomlabs-ai/headroom/issues/1375)) ([8aab8f2](https://github.com/headroomlabs-ai/headroom/commit/8aab8f22cbd11061484991262d3fee3268e95bfa))
* **code:** slice tree-sitter byte offsets as UTF-8 ([#1332](https://github.com/headroomlabs-ai/headroom/issues/1332)) ([8238402](https://github.com/headroomlabs-ai/headroom/commit/82384022bd38304a37e7eade4b5fc98d42f747a8))
* **code:** validate Python compressed syntax ([#1302](https://github.com/headroomlabs-ai/headroom/issues/1302)) ([cbd361d](https://github.com/headroomlabs-ai/headroom/commit/cbd361de2af266b6d72e246185f622c48ec5a6dc))
* **code:** verify a real parse in tree-sitter availability check ([#1231](https://github.com/headroomlabs-ai/headroom/issues/1231)) ([#1299](https://github.com/headroomlabs-ai/headroom/issues/1299)) ([5e0bb69](https://github.com/headroomlabs-ai/headroom/commit/5e0bb697254b7ec87e3191fa73031bde9321a79c))
* **codex:** retag threads on init so Codex Desktop history stays visible ([#961](https://github.com/headroomlabs-ai/headroom/issues/961)) ([#1349](https://github.com/headroomlabs-ai/headroom/issues/1349)) ([e6bbc40](https://github.com/headroomlabs-ai/headroom/commit/e6bbc40b115bc3b31d68da4dabe280d38e1b691c))
* **codex:** stop pinning Codex memory MCP to one project db ([#1269](https://github.com/headroomlabs-ai/headroom/issues/1269)) ([ad7993b](https://github.com/headroomlabs-ai/headroom/commit/ad7993bf15e590a7d164407264721ce1b5128b1e))
* **dashboard:** include RTK stats in the historical tab ([#1324](https://github.com/headroomlabs-ai/headroom/issues/1324)) ([35939c3](https://github.com/headroomlabs-ai/headroom/commit/35939c3536cbaf6e1df01d099943e90ddb364b06))
* **deps:** remediate dependency CVEs and publish SBOM ([#1509](https://github.com/headroomlabs-ai/headroom/issues/1509)) ([5771a80](https://github.com/headroomlabs-ai/headroom/commit/5771a8020e2666503d87f1298070b44e35aad655))
* **docker:** persist session history across container revisions ([#1118](https://github.com/headroomlabs-ai/headroom/issues/1118)) ([5912d65](https://github.com/headroomlabs-ai/headroom/commit/5912d65674c708b00cff9a8cbc3b529fd2ab69fa))
* **gemini:** offload compression to the executor ([#1382](https://github.com/headroomlabs-ai/headroom/issues/1382)) ([615848e](https://github.com/headroomlabs-ai/headroom/commit/615848eba408997c1850319028815afadc6c49ed))
* **gemini:** resolve Google model capabilities through ModelRegistry ([#1276](https://github.com/headroomlabs-ai/headroom/issues/1276)) ([17ecad9](https://github.com/headroomlabs-ai/headroom/commit/17ecad9d89b81313f131d569cfed532f9d42e82a))
* **install:** guard install_agent_ensure against duplicate runtime spawns ([#1301](https://github.com/headroomlabs-ai/headroom/issues/1301)) ([8da0b4e](https://github.com/headroomlabs-ai/headroom/commit/8da0b4e565be2d5f798741bb9b7bee70c2102c8c))
* **install:** repair macOS launchd restart/start lifecycle ([#1290](https://github.com/headroomlabs-ai/headroom/issues/1290)) ([da1a397](https://github.com/headroomlabs-ai/headroom/commit/da1a3973ed79d89617087ec315e77fb82356c03b))
* **install:** stop duplicating ENTRYPOINT in persistent-docker runtime command ([#833](https://github.com/headroomlabs-ai/headroom/issues/833)) ([#1348](https://github.com/headroomlabs-ai/headroom/issues/1348)) ([feedead](https://github.com/headroomlabs-ai/headroom/commit/feedead07772a27b872a448281a2d17e539d4702))
* **io:** use UTF-8 with locale fallback and preserve line endings on config/text I/O ([#1498](https://github.com/headroomlabs-ai/headroom/issues/1498)) ([1baa04e](https://github.com/headroomlabs-ai/headroom/commit/1baa04ef6576e08eeed685890354fca16ad4e6e3))
* **kompress:** hard override keeps must-keep tokens regardless of model score ([#1400](https://github.com/headroomlabs-ai/headroom/issues/1400)) ([42612c8](https://github.com/headroomlabs-ai/headroom/commit/42612c86dfc25a56a6ec6c1da74914e0741a51f6))
* **langchain:** disable streaming on wrapped model during ainvoke() ([#1287](https://github.com/headroomlabs-ai/headroom/issues/1287)) ([3590046](https://github.com/headroomlabs-ai/headroom/commit/359004646bb2cda2b99cf3ef154539b7fa81aa72))
* **mcp:** register managed installs with a resolvable headroom command ([#1386](https://github.com/headroomlabs-ai/headroom/issues/1386)) ([22def93](https://github.com/headroomlabs-ai/headroom/commit/22def931770e6138d16f62daec39501951e68e64))
* **mcp:** report correct savings_percent in headroom_compress ([#1106](https://github.com/headroomlabs-ai/headroom/issues/1106)) ([f216e43](https://github.com/headroomlabs-ai/headroom/commit/f216e430559759f51b53eb44e76e030e6a83c80a))
* **opencode:** write local MCP config ([#1381](https://github.com/headroomlabs-ai/headroom/issues/1381)) ([6c83790](https://github.com/headroomlabs-ai/headroom/commit/6c837906802f9c211513a182de2365071e4f7765))
* **packaging:** move hnswlib to optional [vector] extra so [all] needs no C++ toolchain ([#1499](https://github.com/headroomlabs-ai/headroom/issues/1499)) ([80fa086](https://github.com/headroomlabs-ai/headroom/commit/80fa086660b277798ba9e6c6ed8645ec029362da))
* patch rtk hook script to use absolute path after register_claude_hooks ([#571](https://github.com/headroomlabs-ai/headroom/issues/571)) ([b618d2d](https://github.com/headroomlabs-ai/headroom/commit/b618d2d11a25ffaa00729b17fb41bd41037f4090))
* **perf:** surface RTK/CLI context-tool savings in perf and the session card ([#1433](https://github.com/headroomlabs-ai/headroom/issues/1433)) ([9362747](https://github.com/headroomlabs-ai/headroom/commit/93627471b72e3200e3ca78e1fb345c174414b716))
* **proxy:** add --protect-tool-results to prevent lossy compression of exact-output Bash results ([#1374](https://github.com/headroomlabs-ai/headroom/issues/1374)) ([51d4bcf](https://github.com/headroomlabs-ai/headroom/commit/51d4bcfc113d95a9c843937fbdd3751483bc1dab))
* **proxy:** add an Anthropic buffered read-timeout override ([#1331](https://github.com/headroomlabs-ai/headroom/issues/1331)) ([3be2526](https://github.com/headroomlabs-ai/headroom/commit/3be2526b76caa8ff1050e44807386874571e079b))
* **proxy:** add versionless Vertex AI routes for Claude Code compatibility ([#1321](https://github.com/headroomlabs-ai/headroom/issues/1321)) ([bb3e040](https://github.com/headroomlabs-ai/headroom/commit/bb3e040a463b66801323c261e9547f1e4a2ccfbd))
* **proxy:** bind before eager preload so a hung compressor load can't block startup ([#1500](https://github.com/headroomlabs-ai/headroom/issues/1500)) ([d5ac07f](https://github.com/headroomlabs-ai/headroom/commit/d5ac07fc451516c3b1fe7ece2f01f8d85c126925))
* **proxy:** build SSL contexts for custom CA bundles ([#1134](https://github.com/headroomlabs-ai/headroom/issues/1134)) ([561ba17](https://github.com/headroomlabs-ai/headroom/commit/561ba17ec2e05b463682fd3ecfe7ca43b558684f))
* **proxy:** forward request-id headers on the streaming path ([#1100](https://github.com/headroomlabs-ai/headroom/issues/1100)) ([#1258](https://github.com/headroomlabs-ai/headroom/issues/1258)) ([3d59df7](https://github.com/headroomlabs-ai/headroom/commit/3d59df7be889d6d7218c5552e40a4f736d80a3af))
* **proxy:** gate CCR retrieve/compress endpoints to loopback ([#1338](https://github.com/headroomlabs-ai/headroom/issues/1338)) ([acafb2d](https://github.com/headroomlabs-ai/headroom/commit/acafb2d0f668dc5f5848fa2940545743899a30c2))
* **proxy:** honor force_kompress routing profile ([#996](https://github.com/headroomlabs-ai/headroom/issues/996)) ([b4682d6](https://github.com/headroomlabs-ai/headroom/commit/b4682d6f91c782286553875b7fd8cee6101f1b0f))
* **proxy:** keep large compression results on the critical path ([#296](https://github.com/headroomlabs-ai/headroom/issues/296)) ([#1352](https://github.com/headroomlabs-ai/headroom/issues/1352)) ([90734b6](https://github.com/headroomlabs-ai/headroom/commit/90734b691a50669eaaae7c8739243e7bfc313326))
* **proxy:** offload /v1/compress to the compression executor to stop blocking the loop ([#1501](https://github.com/headroomlabs-ai/headroom/issues/1501)) ([27e010e](https://github.com/headroomlabs-ai/headroom/commit/27e010e38f37e64767e94d144fd4353fcdbe1e47))
* **proxy:** preserve Responses memory continuations with store=false ([#1103](https://github.com/headroomlabs-ai/headroom/issues/1103)) ([cdfeeac](https://github.com/headroomlabs-ai/headroom/commit/cdfeeacc63e6cb98d34e245f2330f0e1af531d32))
* **proxy:** queue mid-turn user messages on non-Bedrock streaming path ([#1377](https://github.com/headroomlabs-ai/headroom/issues/1377)) ([b09f027](https://github.com/headroomlabs-ai/headroom/commit/b09f0270625a4dbee6fc2805f52f19492e68f1f6))
* **proxy:** register interceptor in explicit transforms list when HEADROOM_INTERCEPT_ENABLED ([#1376](https://github.com/headroomlabs-ai/headroom/issues/1376)) ([55c700c](https://github.com/headroomlabs-ai/headroom/commit/55c700c686309c63eb8d9d7d21f30d1838e1c9e7))
* **proxy:** report real input tokens on streaming message_start ([#1132](https://github.com/headroomlabs-ai/headroom/issues/1132)) ([#1305](https://github.com/headroomlabs-ai/headroom/issues/1305)) ([70cc96a](https://github.com/headroomlabs-ai/headroom/commit/70cc96a386baff345669722dc15fde694811d2d6))
* **proxy:** retry upstream 429 with Retry-After on both forwarders ([#1329](https://github.com/headroomlabs-ai/headroom/issues/1329)) ([90bee89](https://github.com/headroomlabs-ai/headroom/commit/90bee89243004846cfc86ad3bf888579acb27522))
* **proxy:** retry upstream 529 overloaded like 429 on both forwarders ([#1495](https://github.com/headroomlabs-ai/headroom/issues/1495)) ([547b15d](https://github.com/headroomlabs-ai/headroom/commit/547b15dab2c18b8d70504c366dc33e22111255e5))
* **proxy:** stop re-compressing headroom_retrieve output and emitting unredeemable markers ([#1323](https://github.com/headroomlabs-ai/headroom/issues/1323)) ([43494ff](https://github.com/headroomlabs-ai/headroom/commit/43494ff526468a63ecf028e081a357d1f619ef56))
* **proxy:** strip Codex lite header from OpenAI WebSockets ([#1543](https://github.com/headroomlabs-ai/headroom/issues/1543)) ([5d3803a](https://github.com/headroomlabs-ai/headroom/commit/5d3803a21c53907e2fea900524e48b510dd59d7a))
* **read-lifecycle:** persist STALE Read originals in the CCR store ([#1488](https://github.com/headroomlabs-ai/headroom/issues/1488)) ([9157173](https://github.com/headroomlabs-ai/headroom/commit/915717301860036005f3a51a5306762ae588ed11))
* recover persistent proxy feature checks and reject non-Copilot exchange URL ([#1465](https://github.com/headroomlabs-ai/headroom/issues/1465)) ([16c638b](https://github.com/headroomlabs-ai/headroom/commit/16c638bc211ecc6d1768bbe36e0c12971996e104))
* remove agents.md ([#1540](https://github.com/headroomlabs-ai/headroom/issues/1540)) ([a7d3360](https://github.com/headroomlabs-ai/headroom/commit/a7d3360a05d4fd139cceab5f72d7de4ef7c712b0))
* respect COPILOT_PROVIDER_TYPE env var when provider_type is auto ([#549](https://github.com/headroomlabs-ai/headroom/issues/549)) ([24cf256](https://github.com/headroomlabs-ai/headroom/commit/24cf256e50fbd0df8ac67fefa90982cd20807274))
* restore token-mode compression on frozen prefixes ([#1489](https://github.com/headroomlabs-ai/headroom/issues/1489)) ([8e0dadf](https://github.com/headroomlabs-ai/headroom/commit/8e0dadfe02da144ca0b27906a8a82bb4be2cb720))
* **router:** degrade to pure-Python detection on native panic ([#1123](https://github.com/headroomlabs-ai/headroom/issues/1123)) ([#1260](https://github.com/headroomlabs-ai/headroom/issues/1260)) ([a00fb67](https://github.com/headroomlabs-ai/headroom/commit/a00fb6761eddf59ede6767211da06f8840552f14))
* **rtk:** stop hook registration timing out on a forked daemon ([#1314](https://github.com/headroomlabs-ai/headroom/issues/1314)) ([9758817](https://github.com/headroomlabs-ai/headroom/commit/97588179790da9fa13ad6793b3cb8e485b43f9b3))
* **smart-crusher:** honor enable_ccr_marker on the opaque-blob path ([#1130](https://github.com/headroomlabs-ai/headroom/issues/1130)) ([27d6f8e](https://github.com/headroomlabs-ai/headroom/commit/27d6f8e2a767b58eb7d2f47599f68e8bdc49fb7f))
* **subscription:** only reset 5h contribution on real rollover, not API jitter ([#1255](https://github.com/headroomlabs-ai/headroom/issues/1255)) ([8d6c175](https://github.com/headroomlabs-ai/headroom/commit/8d6c175d605b88d1c5a7f5e7671778a0e54fb09e))
* **subscription:** run transcript token scan off the event loop ([#1263](https://github.com/headroomlabs-ai/headroom/issues/1263)) ([f03021f](https://github.com/headroomlabs-ai/headroom/commit/f03021f1b69ec1a099436a5f80e68d5266cad8bf))
* surface output reduction without a restart, and explain $0.00 savings on Python 3.14 ([#1296](https://github.com/headroomlabs-ai/headroom/issues/1296)) ([c30ec4c](https://github.com/headroomlabs-ai/headroom/commit/c30ec4cda8d5340dd98ba1653a7e85f684eb7c3d))
* **tests:** reset whole headroom logger subtree so caplog stays deterministic ([#1117](https://github.com/headroomlabs-ai/headroom/issues/1117)) ([fda4670](https://github.com/headroomlabs-ai/headroom/commit/fda4670ef8a8ee279f5afc38ccfecf966762ada2))
* **tls:** add HEADROOM_TLS_STRICT=0 toggle for corporate SSL inspection ([#1308](https://github.com/headroomlabs-ai/headroom/issues/1308)) ([#1341](https://github.com/headroomlabs-ai/headroom/issues/1341)) ([52068dd](https://github.com/headroomlabs-ai/headroom/commit/52068dd650d06d400db472efe6c7b47f539612aa))
* **tokenizers:** price CJK/Kana/Hangul at ~1 token per char in EstimatingTokenCounter ([#1093](https://github.com/headroomlabs-ai/headroom/issues/1093)) ([a35fe86](https://github.com/headroomlabs-ai/headroom/commit/a35fe86e87725e660779f9cbbb0825f87f59d532))
* **transforms:** gate tool string output from lossy compression ([#1307](https://github.com/headroomlabs-ai/headroom/issues/1307)) ([#1387](https://github.com/headroomlabs-ai/headroom/issues/1387)) ([c6c921a](https://github.com/headroomlabs-ai/headroom/commit/c6c921a7c135a19c68fcd85ac5bdddd4ee9c1e8d))
* **websocket:** harden responses websocket origin handling ([#1481](https://github.com/headroomlabs-ai/headroom/issues/1481)) ([c632023](https://github.com/headroomlabs-ai/headroom/commit/c632023cc1ec61d15f8f8e86efe3b54d51604a64))
* **windows:** pin UTF-8 encoding on text-mode subprocess calls ([#1311](https://github.com/headroomlabs-ai/headroom/issues/1311)) ([d633e81](https://github.com/headroomlabs-ai/headroom/commit/d633e8172ccfde4b08c302ecc4c4ef4ce27785f1))
* **wrap:** add Copilot unwrap command ([#1251](https://github.com/headroomlabs-ai/headroom/issues/1251)) ([b4fde0c](https://github.com/headroomlabs-ai/headroom/commit/b4fde0c3a4c2585d4aeda2c6987fe509a5296fe5))
* **wrap:** isolate proxy stdio from proxy.log on Windows ([#1191](https://github.com/headroomlabs-ai/headroom/issues/1191)) ([959ab0d](https://github.com/headroomlabs-ai/headroom/commit/959ab0de471293e76df1f124ed0090c62e62c308))
* **wrap:** keep agent savings opt-in ([#1294](https://github.com/headroomlabs-ai/headroom/issues/1294)) ([b829ceb](https://github.com/headroomlabs-ai/headroom/commit/b829ceba84ce058dadb4e70f6766af13806a4385))
* **wrap:** show the dashboard URL when the proxy is already running ([#1313](https://github.com/headroomlabs-ai/headroom/issues/1313)) ([b0146c4](https://github.com/headroomlabs-ai/headroom/commit/b0146c4ccd1e75dc7db21ef7f00dd4b3aa80e276))


### Performance Improvements

* **compression:** take large cold-start contexts off the synchronous kompress path ([#1171](https://github.com/headroomlabs-ai/headroom/issues/1171)) ([#1298](https://github.com/headroomlabs-ai/headroom/issues/1298)) ([6c68ff4](https://github.com/headroomlabs-ai/headroom/commit/6c68ff4e9f911af9dbd6108367acb3cab80d6f5e))

## [0.27.0](https://github.com/chopratejas/headroom/compare/v0.26.0...v0.27.0) (2026-06-22)


### Features

* **cli:** add headroom doctor setup diagnostics ([#926](https://github.com/chopratejas/headroom/issues/926)) ([e45cf4e](https://github.com/chopratejas/headroom/commit/e45cf4e0618b4de02608f68c502ac4cf1270eb84))
* **cli:** add headroom update command and release banner ([#1088](https://github.com/chopratejas/headroom/issues/1088)) ([26be2c3](https://github.com/chopratejas/headroom/commit/26be2c39cb8a3c23edc08516f01cf91fad33c117))
* compression extraction — Rust knob exposure, CCR hardening, traffic audits ([#818](https://github.com/chopratejas/headroom/issues/818)) ([b7be381](https://github.com/chopratejas/headroom/commit/b7be3814f1d38375bc27901272bbe919e6b35940))
* measure and surface token throughput (tokens/sec) through the proxy ([#983](https://github.com/chopratejas/headroom/issues/983)) ([0d89c67](https://github.com/chopratejas/headroom/commit/0d89c674cd3522c0a46e3df9b98426e59b337b10))
* output-token reduction — verbosity shaper, per-user learning, counterfactual savings ([#965](https://github.com/chopratejas/headroom/issues/965)) ([a99dc61](https://github.com/chopratejas/headroom/commit/a99dc61424df4c7b22c37986fb8dfc648f3ac3b8))
* **policy:** decay P_alive from idle time near cache TTL ([#856](https://github.com/chopratejas/headroom/issues/856) P3b) ([#1028](https://github.com/chopratejas/headroom/issues/1028)) ([fe4f9ee](https://github.com/chopratejas/headroom/commit/fe4f9ee478f50a84190a2d44de2b9fbf24272acf))
* **providers:** add Cortex Code (Snowflake CoCo) as a supported agent ([#1190](https://github.com/chopratejas/headroom/issues/1190)) ([d9d0bf4](https://github.com/chopratejas/headroom/commit/d9d0bf4b79f57ce760f4ac236afe19721727d936))
* **proxy:** cc-switch reconciler — keep Headroom in the request path alongside cc-switch ([#1030](https://github.com/chopratejas/headroom/issues/1030)) ([e8fc8a0](https://github.com/chopratejas/headroom/commit/e8fc8a0d18a551bad572ec21aa92a424748683a5))
* **proxy:** hot-reload live env knobs so a reused proxy picks them up without a restart ([#1090](https://github.com/chopratejas/headroom/issues/1090)) ([6904d47](https://github.com/chopratejas/headroom/commit/6904d47a01e7be496e21d8ebcf34739db5c3b7dd))
* **proxy:** make COMPRESSION_TIMEOUT_SECONDS configurable via env ([#946](https://github.com/chopratejas/headroom/issues/946)) ([#991](https://github.com/chopratejas/headroom/issues/991)) ([addebdb](https://github.com/chopratejas/headroom/commit/addebdb29c3b4a877ed46553d9b0c0a128d62cef))
* **transforms:** tabular + spreadsheet (.xlsx/.xls) compression ([#1128](https://github.com/chopratejas/headroom/issues/1128)) ([d789a7c](https://github.com/chopratejas/headroom/commit/d789a7c528ceee1f4ba648a1002f2e6b6f620854))
* **vertex:** turnkey Claude Code + Vertex compression (+ fixes from the Vertex review) ([#1113](https://github.com/chopratejas/headroom/issues/1113)) ([0e05915](https://github.com/chopratejas/headroom/commit/0e0591506c3f120b96cdc98054114d9ec1771f67))


### Bug Fixes

* **ccr:** accept 12-char SmartCrusher hashes in tool injection ([#1095](https://github.com/chopratejas/headroom/issues/1095)) ([#1141](https://github.com/chopratejas/headroom/issues/1141)) ([9f7f3ad](https://github.com/chopratejas/headroom/commit/9f7f3adfea03710d5e67c4c630b3c8061ff6d161))
* **ccr:** return stored content when headroom_retrieve query matches nothing ([#1213](https://github.com/chopratejas/headroom/issues/1213)) ([#1236](https://github.com/chopratejas/headroom/issues/1236)) ([08fb845](https://github.com/chopratejas/headroom/commit/08fb845fe37478af2c2f55c402df77d7a448fc86))
* **content-router:** honor target_ratio in compression cache + add proxy --target-ratio flag ([#1108](https://github.com/chopratejas/headroom/issues/1108)) ([8894ee0](https://github.com/chopratejas/headroom/commit/8894ee0c18e6dfe858cf0034ec424fd0768a1334))
* **dashboard:** light-mode backgrounds + aligned savings tables ([#1064](https://github.com/chopratejas/headroom/issues/1064)) ([5eae32b](https://github.com/chopratejas/headroom/commit/5eae32ba47fd2e6479cbc1cef1ef4f2fb992fe15))
* **deps:** make litellm optional on Python 3.14 ([#956](https://github.com/chopratejas/headroom/issues/956)) ([#993](https://github.com/chopratejas/headroom/issues/993)) ([b2f04e4](https://github.com/chopratejas/headroom/commit/b2f04e4ef714fb6f2776ed95ee9157c34333e6c3))
* **e2e:** align Codex wrap e2e with global-only RTK guidance ([#1240](https://github.com/chopratejas/headroom/issues/1240)) ([#1254](https://github.com/chopratejas/headroom/issues/1254)) ([bc12ace](https://github.com/chopratejas/headroom/commit/bc12acef5998f264f22ca6d36b17337791a62e6f))
* **init:** set ENABLE_TOOL_SEARCH=true so Claude Code keeps deferring tools ([#746](https://github.com/chopratejas/headroom/issues/746)) ([#995](https://github.com/chopratejas/headroom/issues/995)) ([500ec2b](https://github.com/chopratejas/headroom/commit/500ec2b7faebfd24c9ea404ae1dece40b3b14b84))
* **kompress:** never block the request path on the cold-cache model download ([#1161](https://github.com/chopratejas/headroom/issues/1161)) ([3fc2a78](https://github.com/chopratejas/headroom/commit/3fc2a78a5e20f159f7c5f198de6b91788dc64287))
* **memory:** use ONNX embedder for `wrap --memory` sync ([#1092](https://github.com/chopratejas/headroom/issues/1092)) ([#1262](https://github.com/chopratejas/headroom/issues/1262)) ([4f9feda](https://github.com/chopratejas/headroom/commit/4f9fedaa7a02e41114b5d5f4606f95f903e17b2a))
* **openclaw:** wrap plugin export as {register} object for OpenClaw 2026.x compatibility ([#1218](https://github.com/chopratejas/headroom/issues/1218)) ([2e6c442](https://github.com/chopratejas/headroom/commit/2e6c442dc87f0853313b18ab1a7c80e991058bf7))
* **providers:** update DeepSeek V3 context limit from 128K to 1M ([#1038](https://github.com/chopratejas/headroom/issues/1038)) ([#1137](https://github.com/chopratejas/headroom/issues/1137)) ([bcabc5c](https://github.com/chopratejas/headroom/commit/bcabc5cb11c7c411ed29dac1fcc3771833ac8524))
* **proxy:** allow disabling periodic TOIN stats logging ([#1265](https://github.com/chopratejas/headroom/issues/1265)) ([b5f63d8](https://github.com/chopratejas/headroom/commit/b5f63d8fa9f81f39eab854f29a2fdc39878566df))
* **proxy:** honor HEADROOM_EXCLUDE_TOOLS for Codex /v1/responses tool outputs ([#940](https://github.com/chopratejas/headroom/issues/940)) ([#1053](https://github.com/chopratejas/headroom/issues/1053)) ([f03e77b](https://github.com/chopratejas/headroom/commit/f03e77bec05494aebb4de188eddf2b57f99f6997))
* **proxy:** preserve byte-faithful Anthropic tool forwarding ([#1222](https://github.com/chopratejas/headroom/issues/1222)) ([1f18d59](https://github.com/chopratejas/headroom/commit/1f18d5980972fc7b2091ca0be5318d06c4edfa79))
* **proxy:** route Codex OAuth image requests ([#1215](https://github.com/chopratejas/headroom/issues/1215)) ([381d771](https://github.com/chopratejas/headroom/commit/381d771e4618585e5756e20c090354ccad09183f))
* **proxy:** scope CORS to loopback + gate operator/content endpoints ([#1226](https://github.com/chopratejas/headroom/issues/1226)) ([bd55a42](https://github.com/chopratejas/headroom/commit/bd55a426bc3ec6cd3e0ad46cd3182209afb84937))
* **proxy:** stamp X-Client: codex on Responses endpoint for unidentified callers ([#1036](https://github.com/chopratejas/headroom/issues/1036)) ([b0cd032](https://github.com/chopratejas/headroom/commit/b0cd0329c75c8556c51c1c96dc19f2ab6a23677d))
* **proxy:** treat NODE_EXTRA_CA_CERTS as additive, not replacement ([#998](https://github.com/chopratejas/headroom/issues/998)) ([#1031](https://github.com/chopratejas/headroom/issues/1031)) ([c987283](https://github.com/chopratejas/headroom/commit/c98728363a1079f39bb19da2955cc859b35900a8))
* **telemetry:** switch anonymous telemetry to opt-in (off by default) ([#1223](https://github.com/chopratejas/headroom/issues/1223)) ([b998697](https://github.com/chopratejas/headroom/commit/b99869778bb3ebe223015bdd051e3b9746c8a22c))
* **tokenizers:** bound tiktoken vocab load so a stalled download cannot hang requests ([#956](https://github.com/chopratejas/headroom/issues/956)) ([#994](https://github.com/chopratejas/headroom/issues/994)) ([7e86baf](https://github.com/chopratejas/headroom/commit/7e86bafb9004e40716a04e22398d24157928ca67))
* **unwrap:** remove ANTHROPIC_BASE_URL + ENABLE_TOOL_SEARCH and init hooks on unwrap ([#992](https://github.com/chopratejas/headroom/issues/992)) ([5b84691](https://github.com/chopratejas/headroom/commit/5b846917701e346739346c99c48d5ab6e226e17d))
* **wrap:** keep Codex RTK guidance global ([#1240](https://github.com/chopratejas/headroom/issues/1240)) ([7c26a54](https://github.com/chopratejas/headroom/commit/7c26a54d53aa06a3d75e1111b285c2593155c43e))
* **wrap:** percent-encode non-ASCII cwd names in X-Headroom-Project header ([#1071](https://github.com/chopratejas/headroom/issues/1071)) ([9f712cc](https://github.com/chopratejas/headroom/commit/9f712ccbd7ec27b74f6ac7f20b7d2a9743dba1d8))
* **wrap:** write env.ANTHROPIC_BASE_URL to settings.json so daemon-spawned conversations inherit proxy ([#951](https://github.com/chopratejas/headroom/issues/951)) ([#1078](https://github.com/chopratejas/headroom/issues/1078)) ([a554c3a](https://github.com/chopratejas/headroom/commit/a554c3a0e6c5c57a7c745d8648024362d9d502a4))

## [0.26.0](https://github.com/chopratejas/headroom/compare/v0.25.0...v0.26.0) (2026-06-16)


### Features

* add Copilot BYOK provider wrapper utilities and CLI support ([#1041](https://github.com/chopratejas/headroom/issues/1041)) ([e67ee2a](https://github.com/chopratejas/headroom/commit/e67ee2af658bce35fb4c71b45a0c5b294d7dcfdc))
* add dashboard agent usage stats ([#814](https://github.com/chopratejas/headroom/issues/814)) ([6d3f39f](https://github.com/chopratejas/headroom/commit/6d3f39f213f4eb2d1c6c814b34e1bf6fe2a5c959))
* Add support for Mistral Vibe CLI ([#935](https://github.com/chopratejas/headroom/issues/935)) ([0932b8b](https://github.com/chopratejas/headroom/commit/0932b8bef4db9109665382b6d7c079a368f08d52))
* attribute reread waste to over-compression via marker check ([#901](https://github.com/chopratejas/headroom/issues/901)) ([f928576](https://github.com/chopratejas/headroom/commit/f9285766dda77b116c7834165849264e55339720))
* **bedrock:** cross-region + Converse compression; bundle proxy binary in images ([#999](https://github.com/chopratejas/headroom/issues/999)) ([0dc2e1c](https://github.com/chopratejas/headroom/commit/0dc2e1cb3f7278332d450644831007316d6ac18c))
* **dashboard:** surface compression-vs-cache net impact in Prefix Cache panel ([#913](https://github.com/chopratejas/headroom/issues/913)) ([2a4d300](https://github.com/chopratejas/headroom/commit/2a4d300841c8cbb55435f821fc2d01c3b3b43a59))
* **evals:** adversarial-input robustness grid for compressors ([#918](https://github.com/chopratejas/headroom/issues/918)) ([5939004](https://github.com/chopratejas/headroom/commit/5939004185a1f9b4ef2e88ee3e72a10e5c8fa4a6))
* **parser:** detect re-issued identical tool calls as reread waste ([#909](https://github.com/chopratejas/headroom/issues/909)) ([7d4ae86](https://github.com/chopratejas/headroom/commit/7d4ae86ec0bb09efff765422b89db587b050cd08))
* **policy:** batch deep edits through one cache-bust ([#856](https://github.com/chopratejas/headroom/issues/856) P3a) ([#1015](https://github.com/chopratejas/headroom/issues/1015)) ([c2e52fe](https://github.com/chopratejas/headroom/commit/c2e52fe7439b464edaee83827ca7d8c8091d7e9a))
* **policy:** consume net-cost mutation gate in ContentRouter ([#856](https://github.com/chopratejas/headroom/issues/856) P2) ([#905](https://github.com/chopratejas/headroom/issues/905)) ([553ade4](https://github.com/chopratejas/headroom/commit/553ade4ec66793c1707df6a95888ca2c1506c0b1))
* **proxy:** compress AWS Bedrock InvokeModel requests via configurable upstream ([#720](https://github.com/chopratejas/headroom/issues/720)) ([7edb27a](https://github.com/chopratejas/headroom/commit/7edb27ab2496b070cbe835b31eb2f828798ddfaa))


### Bug Fixes

* **anthropic:** strip styled Claude model ids ([#651](https://github.com/chopratejas/headroom/issues/651)) ([0c5c89d](https://github.com/chopratejas/headroom/commit/0c5c89d05cefabaa833e54decfdeb677edacc0d7))
* **anyllm:** forward openai api_base/api_key to the any-llm backend ([#942](https://github.com/chopratejas/headroom/issues/942)) ([#954](https://github.com/chopratejas/headroom/issues/954)) ([a7ee8a6](https://github.com/chopratejas/headroom/commit/a7ee8a60a7ac28a8adcc7a7fa83a04a59afe41d5))
* **cache:** guard None exemplar embeddings in dynamic detector ([#950](https://github.com/chopratejas/headroom/issues/950)) ([1ec9320](https://github.com/chopratejas/headroom/commit/1ec93208883f2606cc7ec3db0b8bd8e071646984))
* **cache:** name the missing piece in semantic detector guard ([#1018](https://github.com/chopratejas/headroom/issues/1018)) ([3b0bcee](https://github.com/chopratejas/headroom/commit/3b0bceecf4281eb34112de8dd546d4a58beb3fcc))
* **ci:** check out repo in PR Governance label job ([#1021](https://github.com/chopratejas/headroom/issues/1021)) ([4558bc2](https://github.com/chopratejas/headroom/commit/4558bc2465e52d575070e5a0d6312cd400c8aee1))
* **ci:** make PR governance advisory ([#1047](https://github.com/chopratejas/headroom/issues/1047)) ([74dff94](https://github.com/chopratejas/headroom/commit/74dff94fb8580426f5713991be71df94c4f31598))
* **codex:** compute waste signals on the OpenAI Responses path ([#898](https://github.com/chopratejas/headroom/issues/898)) ([b9e2761](https://github.com/chopratejas/headroom/commit/b9e27614c613a1e5f97eb51af74d3c796fb1ab18))
* **codex:** poll /wham/usage for subscription limits (handshake no longer sends x-codex-* headers) ([#924](https://github.com/chopratejas/headroom/issues/924)) ([8c00f71](https://github.com/chopratejas/headroom/commit/8c00f7103cf0288991d703cc002ac354e6266534))
* **codex:** PR health label check state ([#986](https://github.com/chopratejas/headroom/issues/986)) ([99c874d](https://github.com/chopratejas/headroom/commit/99c874d4233ec2d35c5c12a709ba32fd2fd96f3d))
* **codex:** retag thread providers so history menu stays whole across the proxy boundary ([#1034](https://github.com/chopratejas/headroom/issues/1034)) ([74ae781](https://github.com/chopratejas/headroom/commit/74ae7816444ae972b55f3da0ff5e28c8638ab4f3))
* **codex:** write canonical hooks feature flag and migrate deprecated codex_hooks ([#743](https://github.com/chopratejas/headroom/issues/743)) ([dff6a19](https://github.com/chopratejas/headroom/commit/dff6a19946b8f96bb8b16fa945b69a1ed09709af))
* **compression:** convert tree-sitter byte offsets to char offsets ([#892](https://github.com/chopratejas/headroom/issues/892)) ([b1f700f](https://github.com/chopratejas/headroom/commit/b1f700fc275bf1d7e9461b61a9ebfdb1fba19620))
* **compression:** correct JSON array item counting and entropy gate ([#887](https://github.com/chopratejas/headroom/issues/887)) ([d6f0f0f](https://github.com/chopratejas/headroom/commit/d6f0f0f64269bfbdf36070cb304703c606c64b72))
* **compression:** keep container bodies compressible in code handler ([#890](https://github.com/chopratejas/headroom/issues/890)) ([16ed73b](https://github.com/chopratejas/headroom/commit/16ed73bca68e602a86a385480d484c3a60025b8c))
* **compression:** measure short-value threshold on payload, not token ([#889](https://github.com/chopratejas/headroom/issues/889)) ([65b0e8c](https://github.com/chopratejas/headroom/commit/65b0e8c58dbbc0b77e4b7159b279287979767c4c))
* **compression:** use thread-local tree-sitter parsers in code handler ([#893](https://github.com/chopratejas/headroom/issues/893)) ([6cdb846](https://github.com/chopratejas/headroom/commit/6cdb8462000d9610b5d15f6c7c45adb787bfec1e))
* **gemini:** surface functionResponse payloads to waste-signal detection ([#897](https://github.com/chopratejas/headroom/issues/897)) ([9b0c840](https://github.com/chopratejas/headroom/commit/9b0c840dd7c181d6266b31cd16f493393ccc5c1a))
* **learn:** decode directory names with spaces in Windows project paths ([#997](https://github.com/chopratejas/headroom/issues/997)) ([#1027](https://github.com/chopratejas/headroom/issues/1027)) ([2d3701b](https://github.com/chopratejas/headroom/commit/2d3701b59e9ff8aedc2a282c4467f27ca2355d62))
* **learn:** scan subagent and workflow transcripts ([#1045](https://github.com/chopratejas/headroom/issues/1045)) ([0ddd4ed](https://github.com/chopratejas/headroom/commit/0ddd4ed9e92fe898373036ba3be228f9afc3bc5a))
* **openclaw:** declare headroom_retrieve tool contract ([#947](https://github.com/chopratejas/headroom/issues/947)) ([7c8c909](https://github.com/chopratejas/headroom/commit/7c8c909c853a264c833c645403cbbb1894b91432))
* **policy:** correct warm-cache penalty in net_mutation_gain to (S + dT) ([#903](https://github.com/chopratejas/headroom/issues/903)) ([0632eba](https://github.com/chopratejas/headroom/commit/0632eba6c3bdf5b030d794d3dfefa3c29543d2e8))
* **proxy:** add native Bedrock converse-stream route ([#917](https://github.com/chopratejas/headroom/issues/917)) ([b08ec15](https://github.com/chopratejas/headroom/commit/b08ec15b0d392b8b8cf93dbadaee4b7e6b465f1c))
* **proxy:** keep codex image-generation WS turns alive through the relay ([#1000](https://github.com/chopratejas/headroom/issues/1000)) ([7dbbb40](https://github.com/chopratejas/headroom/commit/7dbbb4077e7bb11b3da4634573cfc1d998e139ec))
* **proxy:** make budget enforcement actually work ([#885](https://github.com/chopratejas/headroom/issues/885)) ([a14ab45](https://github.com/chopratejas/headroom/commit/a14ab45cf0e6e698c52a0efd0448ca7c8ba0b31f))
* **proxy:** read RTK gain stats globally by default ([#957](https://github.com/chopratejas/headroom/issues/957)) ([b70fccb](https://github.com/chopratejas/headroom/commit/b70fccbe174e1adff0f52ceaf9bec0dcda0c73da))
* route v1internal code assist requests to cloudcode-pa.googleapis… ([#821](https://github.com/chopratejas/headroom/issues/821)) ([e20f16b](https://github.com/chopratejas/headroom/commit/e20f16b1a65710f532aa019ef60ac7a18a4e7f46))
* **serena:** stop the Serena dashboard popup and make --no-serena actually disable Serena ([#1003](https://github.com/chopratejas/headroom/issues/1003)) ([919379a](https://github.com/chopratejas/headroom/commit/919379a8a1731a0002d813a79d880ad35f8bbbc9))
* support Copilot Business subscription auth ([#641](https://github.com/chopratejas/headroom/issues/641)) ([0b4a4bd](https://github.com/chopratejas/headroom/commit/0b4a4bd4830ecec1bca64c2f62455c4c923d91df))
* wire HEADROOM_EXCLUDE_TOOLS / HEADROOM_TOOL_PROFILES into Click proxy entrypoint ([#943](https://github.com/chopratejas/headroom/issues/943)) ([9b7b436](https://github.com/chopratejas/headroom/commit/9b7b436b04118d6ec4dcaebafc1c82e03e786f27))
* **wrap:** avoid duplicate top-level keys when injecting codex provider ([#884](https://github.com/chopratejas/headroom/issues/884)) ([dd22cfd](https://github.com/chopratejas/headroom/commit/dd22cfd72ad9265c25a95ef5536dc3d17e85dbbf))


### Code Refactoring

* DRY cache logic, add thread safety, fix Bash exclusion ([#704](https://github.com/chopratejas/headroom/issues/704)) ([e36fccd](https://github.com/chopratejas/headroom/commit/e36fccd8cfe6b963398d3d0fa1637a45bd6421af))

## [0.25.0](https://github.com/chopratejas/headroom/compare/v0.24.0...v0.25.0) (2026-06-12)


### Features

* add differential network capture harness ([#761](https://github.com/chopratejas/headroom/issues/761)) ([11ab5f8](https://github.com/chopratejas/headroom/commit/11ab5f83a1ccd617a2608349a42feff7f7e72b98))
* add light mode for dashboard ([#834](https://github.com/chopratejas/headroom/issues/834)) ([c425893](https://github.com/chopratejas/headroom/commit/c425893d123e67c62ee20ff64ae350eb4ea56477))
* add OAuth2 client-credentials upstream-auth proxy extension ([#778](https://github.com/chopratejas/headroom/issues/778)) ([#784](https://github.com/chopratejas/headroom/issues/784)) ([eb2e50f](https://github.com/chopratejas/headroom/commit/eb2e50feb26bacadf8812d6e608a458a990096b9))
* add Vertex AI proxy routing ([#793](https://github.com/chopratejas/headroom/issues/793)) ([3c77e52](https://github.com/chopratejas/headroom/commit/3c77e52ce431210e6045671cf5f7c66c79f90a32))
* **cli:** comprehensive help text, validation, and exception handling improvements ([#640](https://github.com/chopratejas/headroom/issues/640)) ([028efab](https://github.com/chopratejas/headroom/commit/028efabb4e611d77118baefb8ffdd13b0edc4fc5))
* compression safety rails — error-output protection, pipeline circuit breaker, library inflation guard ([#851](https://github.com/chopratejas/headroom/issues/851)) ([c0cadcc](https://github.com/chopratejas/headroom/commit/c0cadccff98e572f126185f371e4de9e241b12e0))
* **dashboard:** per-model savings breakdown and expected-vs-actual cost on historical charts ([#807](https://github.com/chopratejas/headroom/issues/807)) ([34dafe6](https://github.com/chopratejas/headroom/commit/34dafe69d907c9a2971abc0d801ff9bfa498b3a8))
* detect re-served tool results as over-compression waste signal ([#854](https://github.com/chopratejas/headroom/issues/854)) ([5f1d88a](https://github.com/chopratejas/headroom/commit/5f1d88ad2701ed186df93d8e2a3980f0329d9dbb))
* **evals:** add zero-cost tool schema compaction integrity eval ([#817](https://github.com/chopratejas/headroom/issues/817)) ([53a08c6](https://github.com/chopratejas/headroom/commit/53a08c63bf56a76d4fb7b649e37c8e62b0b4cebf))
* gated Markdown-KV compaction formatter (serialization-aware output) ([#859](https://github.com/chopratejas/headroom/issues/859)) ([06b2625](https://github.com/chopratejas/headroom/commit/06b2625b17b0b032f688d321c6aa30ae3f2b7d96))
* **kompress:** warn on unrecognized HEADROOM_KOMPRESS_BACKEND + document backend selection ([#204](https://github.com/chopratejas/headroom/issues/204)) ([6367d0b](https://github.com/chopratejas/headroom/commit/6367d0b7228f53b29bbd20f55c1729476ba5ea68))
* **memory:** add opt-in Apple-GPU (MPS) embedding runtime ([#766](https://github.com/chopratejas/headroom/issues/766)) ([c71592d](https://github.com/chopratejas/headroom/commit/c71592d4214adf1022e4c608518ae0c3ac4aa5e9))
* net-cost cache mutation formula on CompressionPolicy ([#856](https://github.com/chopratejas/headroom/issues/856) P1) ([#857](https://github.com/chopratejas/headroom/issues/857)) ([d5f5802](https://github.com/chopratejas/headroom/commit/d5f58026e2a882bc508acfbddfc9d472100d6e16))
* **plugins:** Hermes agent headroom_retrieve plugin ([#824](https://github.com/chopratejas/headroom/issues/824)) ([058bced](https://github.com/chopratejas/headroom/commit/058bcedab838f3b34ac8e38853e1924329efd820))
* probe-based retention scoring of recorded compression events ([#862](https://github.com/chopratejas/headroom/issues/862)) ([c2106cb](https://github.com/chopratejas/headroom/commit/c2106cbdabb905e1980c6694000c220a5042171c))
* **proxy:** add CLI opt-outs for CCR injection (compression-only mode) ([#823](https://github.com/chopratejas/headroom/issues/823)) ([693d9d2](https://github.com/chopratejas/headroom/commit/693d9d20e2b2d9bfce3a0c48314850ee77ff8af3))
* **proxy:** attribute savings history rollups per provider ([#791](https://github.com/chopratejas/headroom/issues/791)) ([0b8b8d9](https://github.com/chopratejas/headroom/commit/0b8b8d92de3bd5e0301eadedacfb4b1d20a8de7f))
* **proxy:** log compressed messages alongside original request ([#261](https://github.com/chopratejas/headroom/issues/261)) ([2269e40](https://github.com/chopratejas/headroom/commit/2269e40bde7e1b9fb0620bd2cec9e33a92834080))
* **proxy:** per-project savings breakdown on the dashboard (claude, codex, aider, copilot, cursor) ([#803](https://github.com/chopratejas/headroom/issues/803)) ([914a60a](https://github.com/chopratejas/headroom/commit/914a60a2b07caad8488c1e19a5465726b95f83d3))
* support Python 3.14+ via pyo3 abi3 stable ABI ([#516](https://github.com/chopratejas/headroom/issues/516)) ([19eac8e](https://github.com/chopratejas/headroom/commit/19eac8e00dc9e3911f3afe8e8e5dcc9e00346baa))
* switch Kompress default to kompress-v2-base with weight-only int8 ONNX ([#799](https://github.com/chopratejas/headroom/issues/799)) ([74392b2](https://github.com/chopratejas/headroom/commit/74392b238e4f76fa061e673d1415fc7fa2830011))
* **transforms:** attribute read_lifecycle + smart_crush tags ([#249](https://github.com/chopratejas/headroom/issues/249)) ([8f37426](https://github.com/chopratejas/headroom/commit/8f374263d3971c072b5c977375c873864fb05763))


### Bug Fixes

* **anthropic:** CCR exception must re-raise, not silently swallow ([#838](https://github.com/chopratejas/headroom/issues/838)) ([8db5efc](https://github.com/chopratejas/headroom/commit/8db5efc6f9f6de59e9d55cbcd63b75c37a81a26e))
* **ccr:** key Rust search/diff/log markers with explicit_hash ([#852](https://github.com/chopratejas/headroom/issues/852)) ([bfcb07d](https://github.com/chopratejas/headroom/commit/bfcb07d78ea7eba539a65b11e100ec23b336d8d1))
* **ccr:** make retrieval TTL configurable ([#715](https://github.com/chopratejas/headroom/issues/715)) ([2533f77](https://github.com/chopratejas/headroom/commit/2533f7703ee261dc35767b11e46b8eab6e0c454d))
* **ccr:** skip CCR when model calls headroom_retrieve alongside user tools ([#839](https://github.com/chopratejas/headroom/issues/839)) ([30078f8](https://github.com/chopratejas/headroom/commit/30078f8465fb6bb78a5a9c394b75e60cd3c4eeec))
* **ccr:** use shared compression store ([#875](https://github.com/chopratejas/headroom/issues/875)) ([249af6c](https://github.com/chopratejas/headroom/commit/249af6cc7b379678e60da3e98e552368632fd4f4))
* **ci:** correct comments, timeouts, and pip reliability in native e2e workflows ([#878](https://github.com/chopratejas/headroom/issues/878)) ([b716c8c](https://github.com/chopratejas/headroom/commit/b716c8c2ee7ccc68dd1b9294760db1af866843f2))
* **ci:** pin cosign-installer to v3 (v4 does not exist) ([#774](https://github.com/chopratejas/headroom/issues/774)) ([199d693](https://github.com/chopratejas/headroom/commit/199d693f98ecd72d80181c8fee8422b6b64651a2))
* **codex:** respect CODEX_HOME for wrap config ([#731](https://github.com/chopratejas/headroom/issues/731)) ([96abf38](https://github.com/chopratejas/headroom/commit/96abf38b0972adf5e5c66f9a49aa9d9f951b1aa0))
* **content_router:** guard against empty compression output causing Anthropic 400 ([#771](https://github.com/chopratejas/headroom/issues/771)) ([2f9ff07](https://github.com/chopratejas/headroom/commit/2f9ff07e6caef0fe32d00ece6266a476eecff5a3))
* **copilot:** use responses API for subscription reasoning models ([#647](https://github.com/chopratejas/headroom/issues/647)) ([84ac332](https://github.com/chopratejas/headroom/commit/84ac332d14dafacedc2f0b46f5ac6b3977b098d0))
* correct preserved-entry index mapping in Gemini content round-trip ([#836](https://github.com/chopratejas/headroom/issues/836)) ([0ffe2b6](https://github.com/chopratejas/headroom/commit/0ffe2b6ea49e5c8d3bff5fe2c90873c71a95c457))
* **dashboard:** stable 'Proxy $ Saved' hero tile under --workers &gt; 1 ([#481](https://github.com/chopratejas/headroom/issues/481)) ([fd73b88](https://github.com/chopratejas/headroom/commit/fd73b88368b22beeb586b8e1aa37fcd2afb12532))
* don't inject empty tools:[] when client omitted the tools field ([#772](https://github.com/chopratejas/headroom/issues/772)) ([574bbae](https://github.com/chopratejas/headroom/commit/574bbae2cbe2f20b3f0e12b421c25ac256712f0a))
* harden Copilot API auth token handling ([#557](https://github.com/chopratejas/headroom/issues/557)) ([6b0c09f](https://github.com/chopratejas/headroom/commit/6b0c09ffd5f2ce18c4d2cfa6233feaf37d487ead))
* **health:** readyz verifies upstream connectivity, not just process liveness ([#744](https://github.com/chopratejas/headroom/issues/744)) ([5dfb446](https://github.com/chopratejas/headroom/commit/5dfb446da1fb65002e0dea18a90210a2a026f0b3))
* **init:** guard persistent task startup ([#616](https://github.com/chopratejas/headroom/issues/616)) ([9252d85](https://github.com/chopratejas/headroom/commit/9252d852c5a4c716eb5438b8f438d50e59a55fef))
* **init:** normalize Windows hook paths to forward slashes ([#788](https://github.com/chopratejas/headroom/issues/788)) ([6ea6e31](https://github.com/chopratejas/headroom/commit/6ea6e31f09845b2ad5c8bae73bcf353f3b629188))
* **init:** suppress hook recovery output ([#760](https://github.com/chopratejas/headroom/issues/760)) ([b439599](https://github.com/chopratejas/headroom/commit/b4395993aecbb65b85a5b2479dfdb35ea243bf54))
* **learn:** claude-cli streams output with idle timeout ([#373](https://github.com/chopratejas/headroom/issues/373)) ([9bff575](https://github.com/chopratejas/headroom/commit/9bff5752bbd769902f249cdfde42bc53539afd02))
* make headroom wrap readiness probe timeout configurable for slow ML imports ([#581](https://github.com/chopratejas/headroom/issues/581)) ([163677b](https://github.com/chopratejas/headroom/commit/163677b405d7ca8a54d6d7c798bf6ead90da7880))
* **parser:** detect waste signals in Anthropic tool_result content blocks ([#815](https://github.com/chopratejas/headroom/issues/815)) ([929698a](https://github.com/chopratejas/headroom/commit/929698af1030e5926f3766d7d6ac292d6e38437b))
* **proxy:** F4 — trust X-Forwarded-* only behind allow-listed gateway ([d10bd5f](https://github.com/chopratejas/headroom/commit/d10bd5f59c5a36e14f6c5f0480b821532521b753))
* **proxy:** lazy-import server to avoid fastapi crash ([#442](https://github.com/chopratejas/headroom/issues/442)) ([93c6937](https://github.com/chopratejas/headroom/commit/93c69372e614f2b04873bed75602a88d2256a7fc))
* **proxy:** make CCR multi-worker warning conditional on backend ([#770](https://github.com/chopratejas/headroom/issues/770)) ([d76a729](https://github.com/chopratejas/headroom/commit/d76a7296df121365d74c415b8c702a3ad80abd30))
* **proxy:** make Kompress eager preload cache-only so a cold cache can't block startup ([#783](https://github.com/chopratejas/headroom/issues/783)) ([841663d](https://github.com/chopratejas/headroom/commit/841663da16971b1e0d8e204fdf18e4bafedaf9e0))
* **proxy:** restore Codex usage headers on WS and streaming SSE transports ([#577](https://github.com/chopratejas/headroom/issues/577)) ([#794](https://github.com/chopratejas/headroom/issues/794)) ([0ce68de](https://github.com/chopratejas/headroom/commit/0ce68dedd770d5411d16abe30e5ea9dd0b7d8eee))
* schema compaction must not drop property names that match DROP_KEYS ([#785](https://github.com/chopratejas/headroom/issues/785)) ([ae2122f](https://github.com/chopratejas/headroom/commit/ae2122fda8ff0efc03d609d27270453fea3a8718))
* **security:** block DNS-rebinding on /debug/* and /stats/reset via Host-header allowlist ([#605](https://github.com/chopratejas/headroom/issues/605)) ([b4b5025](https://github.com/chopratejas/headroom/commit/b4b50253f16d0a30f1d17a959753137e997efbac))
* **ssl:** upstream httpx client inherits SSL_CERT_FILE, REQUESTS_CA_BUNDLE, NODE_EXTRA_CA_CERTS ([#745](https://github.com/chopratejas/headroom/issues/745)) ([e50fbb3](https://github.com/chopratejas/headroom/commit/e50fbb3e0d61d561456d7b0ff9e0a8ee106a2f02))
* suppress LiteLLM provider banner before import ([#874](https://github.com/chopratejas/headroom/issues/874)) ([f9384ef](https://github.com/chopratejas/headroom/commit/f9384ef4b780eaa1d8ca6dcc314ad430b87f524a))
* **transforms:** use thread-local tree-sitter parsers to prevent pyo3 Unsendable panic ([#604](https://github.com/chopratejas/headroom/issues/604)) ([2ad300a](https://github.com/chopratejas/headroom/commit/2ad300aff801838efe5649b00a0396523a401a2a))
* **wrap:** track shared proxy clients with markers ([#877](https://github.com/chopratejas/headroom/issues/877)) ([05bd56b](https://github.com/chopratejas/headroom/commit/05bd56bcb6b103fab5522da2b14295cf7bd8dbc1))


### Code Refactoring

* extract litellm model resolution to shared utility ([ec7d006](https://github.com/chopratejas/headroom/commit/ec7d0065cc5055e504e79cf24f3951e404fe4cb9))

## [0.24.0](https://github.com/chopratejas/headroom/compare/v0.23.0...v0.24.0) (2026-06-08)


### Features

* **perf:** add --format {text,json,csv} to `headroom perf` ([#648](https://github.com/chopratejas/headroom/issues/648)) ([9fe4886](https://github.com/chopratejas/headroom/commit/9fe4886cf6b612452f7271d3204872f804074c1f))
* **proxy:** show resolved upstream API targets in startup banner ([#586](https://github.com/chopratejas/headroom/issues/586)) ([8dbe7ad](https://github.com/chopratejas/headroom/commit/8dbe7ad41b3a1d33c01874be5c1cbc68a5e68111)), closes [#583](https://github.com/chopratejas/headroom/issues/583)
* **relevance:** weight BM25 score_batch by corpus IDF ([#646](https://github.com/chopratejas/headroom/issues/646)) ([88177bd](https://github.com/chopratejas/headroom/commit/88177bd7a680490ac85d244c5fff90f21a3be27c))
* support CLAUDE_CODE_USE_FOUNDRY and custom upstream gateways ([#726](https://github.com/chopratejas/headroom/issues/726)) ([d90cdce](https://github.com/chopratejas/headroom/commit/d90cdce3b69bbf27e0f5feea461766a9d797cf7e))


### Bug Fixes

* **ci:** restore green lint gate on main ([fe50f9d](https://github.com/chopratejas/headroom/commit/fe50f9daed35151134f79b767733d4be8093e325))
* **codex:** auto-enable fail-open on compression timeout in headroom wrap codex ([#531](https://github.com/chopratejas/headroom/issues/531)) ([5f5f261](https://github.com/chopratejas/headroom/commit/5f5f261a035d12d069eb212eb75c472e2c9edeff))
* **copilot:** restore generic endpoint for non-subscription OAuth ([#610](https://github.com/chopratejas/headroom/issues/610)) ([#612](https://github.com/chopratejas/headroom/issues/612)) ([18925b8](https://github.com/chopratejas/headroom/commit/18925b8c6e343c9d593891cd29ac27fee1cb9836))
* **deps:** move gunicorn to [proxy-prod] extra, add Windows guard ([#537](https://github.com/chopratejas/headroom/issues/537)) ([fa558c5](https://github.com/chopratejas/headroom/commit/fa558c5647a91562f4a8fba0271d27b02c8ae01f))
* **proxy:** fail-open on corrupt golden bytes instead of RuntimeError ([#603](https://github.com/chopratejas/headroom/issues/603)) ([2170a1b](https://github.com/chopratejas/headroom/commit/2170a1b4a00e9c46e845993c9b0f6cb2ef0c0684))
* **proxy:** route Claude Code model metadata to Anthropic ([#627](https://github.com/chopratejas/headroom/issues/627)) ([30c1ac8](https://github.com/chopratejas/headroom/commit/30c1ac8656bcc3d11755daef8d1d27cd8770ebc7))
* **security:** patch loopback guard, retry None raise, async subprocess, and cache race ([06d7cb9](https://github.com/chopratejas/headroom/commit/06d7cb9e6c011711a478864a970f7c87ee853a97))
* **security:** patch loopback guard, retry None raise, blocking subprocess, and cache stats race ([78f3a4d](https://github.com/chopratejas/headroom/commit/78f3a4dd3e8e26525822a3c830d576d702dfed8b))
* **startup:** move HF/httpx log suppression before sentence_transformers init ([#622](https://github.com/chopratejas/headroom/issues/622)) ([176d4c7](https://github.com/chopratejas/headroom/commit/176d4c772a7ca8c9da58ca2403f890ba85e8bad8))
* **startup:** suppress proxy startup log noise ([#619](https://github.com/chopratejas/headroom/issues/619)) ([4555901](https://github.com/chopratejas/headroom/commit/45559011b16a2e084dda22c675c819a4789f961d))
* **wrap:** report unbindable proxy ports ([#602](https://github.com/chopratejas/headroom/issues/602)) ([6dfcaa8](https://github.com/chopratejas/headroom/commit/6dfcaa839f1175518e378963c79cc7bd3ceb7946))

## [Unreleased]

### Added

* **kompress:** warn when `HEADROOM_KOMPRESS_BACKEND` is set to an unrecognized
  value instead of silently falling back to `auto`, and document the backend
  selection env var (`auto` / `onnx` / `onnx_cpu` / `onnx_coreml` / `pytorch` /
  `pytorch_mps` plus shorthand aliases) in `wiki/configuration.md` (issue
  [#202](https://github.com/chopratejas/headroom/issues/202), PR
  [#204](https://github.com/chopratejas/headroom/pull/204)).
* **proxy:** per-provider attribution in the savings history rollups. Each `/stats-history` bucket (hourly/daily/weekly/monthly) now carries a `by_provider` map breaking down `tokens_saved`, `compression_savings_usd_delta`, `total_input_tokens_delta`, and `total_input_cost_usd_delta` per provider, so consumers can show how savings and spend are distributed across providers within a time period. Providers only appear in a bucket where they moved a counter; legacy history checkpoints with no provider collapse into `"unknown"`. Affected files: `headroom/proxy/savings_tracker.py`, `headroom/proxy/prometheus_metrics.py`.
* **cli:** startup banner now includes a `Performance Tuning` section that surfaces active `HEADROOM_COMPRESSION_STABLE_AFTER_TURN`, `HEADROOM_STALE_READ_COMPRESS_AFTER_TURNS`, and embedding-server socket values when set; shows a hint to set them when all defaults are in use.

### Changed

* **deps:** loosen over-pinned constraints and add upper bounds
  - `litellm==1.82.3` -> `>=1.86.2,<2.0` (exact pin blocked security patches; floor stays above the CVE-2026-42271 fix)
  - `transformers>=4.30.0` -> `>=4.30.0,<6.0` (add upper bound; library already crossed a major version silently)
  - `sentence-transformers>=2.2.0` -> `>=2.2.0,<6.0` (same; applied in `memory`, `evals`, and `dev` extras)
  - `neo4j>=5.20.0` -> `>=5.20.0,<7.0` (client had already crossed the 5.x/6.x boundary)
  - `mem0ai>=0.1.100` -> `>=1.0.0,<2.0` (floor was pre-1.0; locked package is already 1.0.11)
  - `langchain-core>=0.2.0` -> `>=1.3.3,<4.0` (floor stays above current high-severity advisory fixes)
  - `langchain-openai>=0.1.0` -> `>=1.1.14,<2.0` (floor stays above current advisory fixes)
  - `qdrant-client>=1.9.0` -> `>=1.9.0,<2.0`
  - `uvicorn>=0.23.0` -> `>=0.23.0,<1.0` (applied in `proxy` and `dev` extras)
  - Same `transformers` and `litellm` bounds applied consistently across `ml`, `voice`, and `dev` extras
* **docker:** bump `neo4j` image in `docker-compose.yml` from `5.15.0` to `5.26` (latest 5.x LTS)
* **docker:** bump `UV_VERSION` in `Dockerfile` from `0.11.16` to `0.11.18`

### Bug Fixes

* **codex:** respect `CODEX_HOME` when `headroom wrap codex` writes provider, MCP, memory, backup, and global `AGENTS.md` config, and warn when `unwrap codex` may be looking at the default Codex home because `CODEX_HOME` is unset.
* **proxy:** multi-worker CCR warning is now conditional on backend — when `HEADROOM_CCR_BACKEND` is unset (default `InMemoryBackend`, per-process), the startup warning includes CCR retrieval failures and suggests `HEADROOM_CCR_BACKEND=sqlite`; when a cross-worker backend is already configured, the warning covers only the remaining per-worker stores (compression cache, prefix tracker, TOIN, CostTracker). Updated `RUST_DEV.md` to accurately document Python `CompressionStore` as per-process by default.
* **deps:** move `gunicorn` to `[proxy-prod]` extra with `sys_platform != 'win32'` guard; removed from `[proxy]` to avoid forcing a Unix-only package on dev, CI, and Windows users ([#537](https://github.com/chopratejas/headroom/pull/537))
* **startup:** suppress proxy startup log noise -- litellm banner, trafilatura parse errors, HuggingFace Hub unauthenticated warnings, tiktoken fallback warning, and httpx INFO lines from sentence_transformers HEAD checks. Affected files: `headroom/providers/litellm.py`, `headroom/transforms/html_extractor.py`, `headroom/memory/adapters/embedders.py`, `headroom/providers/anthropic.py`, `headroom/providers/registry.py`, `headroom/image/onnx_router.py`, `headroom/transforms/kompress_compressor.py`.

## [0.23.0](https://github.com/chopratejas/headroom/compare/v0.22.4...v0.23.0) (2026-06-04)

### Features

* **copilot:** GitHub Copilot subscription mode through Headroom ([f4dff9b](https://github.com/chopratejas/headroom/commit/f4dff9b4885b5c62d79396bbb0847ae3e39a9bd9))


### Bug Fixes

* **ccr:** scope proactive expansion by workspace (cross-project leak) ([197601b](https://github.com/chopratejas/headroom/commit/197601bc64ee72e786bf6b94cd90efcac4269bcf))
* **ccr:** scope proactive expansion by workspace (cross-project leak) ([1bc163f](https://github.com/chopratejas/headroom/commit/1bc163f5bc1a8422f9ad659061e1fdd8cfeb077b))
* **codex:** keep init model_provider at config root ([#260](https://github.com/chopratejas/headroom/issues/260)) ([304dcc7](https://github.com/chopratejas/headroom/commit/304dcc78047bc744fc2f7656b484ec54dc271354))
* **codex:** keep init model_provider at config root ([#260](https://github.com/chopratejas/headroom/issues/260)) ([849b46d](https://github.com/chopratejas/headroom/commit/849b46de5934a88369af2fd7f7d52e9af0536a7e))
* **copilot:** deterministic subscription token handoff to the proxy ([72da461](https://github.com/chopratejas/headroom/commit/72da46121726074515e0c1eb9745498457a1a8d5))
* **copilot:** support subscription auth through Headroom ([ff4a0c6](https://github.com/chopratejas/headroom/commit/ff4a0c6bc64e5e68ab76c38047a36a3c7a6aaacf))
* correct tiktoken encoding for unknown gpt-4 model snapshots ([#552](https://github.com/chopratejas/headroom/issues/552)) ([0e551de](https://github.com/chopratejas/headroom/commit/0e551de9d81021bb7f0dde1857a2341408606969))
* decode/encode owned config, state and template assets as UTF-8 ([2f1538a](https://github.com/chopratejas/headroom/commit/2f1538a641dd0e60a7be3de85646a70c4bf7e287))
* decode/encode owned config, state and template assets as UTF-8 (fixes [#533](https://github.com/chopratejas/headroom/issues/533)) ([92075b9](https://github.com/chopratejas/headroom/commit/92075b95af799951c90a305a08ec4e958473967a))
* **docker:** upgrade base images to Python 3.13 / debian13 ([e6bf7a0](https://github.com/chopratejas/headroom/commit/e6bf7a03fef8a9f2e4802d63afdafb40627c7ad9))
* **docker:** upgrade base images to Python 3.13 / debian13, drop digest pinning ([08a2197](https://github.com/chopratejas/headroom/commit/08a219708c97dcdc678483a0e6891306624a1fad))
* **docs:** bump next.js to 16.2.6 for GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([a6a09e6](https://github.com/chopratejas/headroom/commit/a6a09e6cfbe6962a70a6fb2e4bebeee80756e304))
* **docs:** mkdocs configuration to build with correct folder ([#543](https://github.com/chopratejas/headroom/issues/543)) ([5557944](https://github.com/chopratejas/headroom/commit/55579445f84c363219f45dc5358599a04d4263ed))
* **docs:** update brace-expansion to 5.0.6 to remediate GHSA-jxxr-4gwj-5jf2 (CVE-2026-45149) ([6eb6fb5](https://github.com/chopratejas/headroom/commit/6eb6fb5941adfbd056daa1689c3fa0c3755fd298))
* **docs:** update bun.lock to next 16.2.6 for GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([91e0937](https://github.com/chopratejas/headroom/commit/91e0937243c801fa5f1021b4c47debef2444650c))
* ignore brackets inside JSON strings when splitting mixed content ([#553](https://github.com/chopratejas/headroom/issues/553)) ([bdcfc32](https://github.com/chopratejas/headroom/commit/bdcfc322da0c4cde69931d641cfa18c76ddb138b))
* **learn:** decode Unix home dirs whose username contains '.', '-' or '_' ([211daae](https://github.com/chopratejas/headroom/commit/211daae25687901d1f893714d877b25606d0ef69))
* **learn:** decode Unix home dirs whose username contains '.', '-' or '_' ([491a8b3](https://github.com/chopratejas/headroom/commit/491a8b3a1b260f42f503b3553a04c578c18e1cc0))
* **learn:** finish gemini-flash-latest default model sweep ([982d01b](https://github.com/chopratejas/headroom/commit/982d01b9c996fd5fe26154dc2f94d567192f6ff6))
* **learn:** finish gemini-flash-latest default model sweep ([#532](https://github.com/chopratejas/headroom/issues/532)) ([d797366](https://github.com/chopratejas/headroom/commit/d7973665f4e2f40f2b3acadd0ec584609fb33c6c))
* **memory:** READ-ONLY framing + fail-closed unresolved-project fallback ([a178249](https://github.com/chopratejas/headroom/commit/a178249fc0af4a1b6f212decb4f6d2793d57fae8))
* **memory:** READ-ONLY framing + fail-closed unresolved-project fallback ([482f80e](https://github.com/chopratejas/headroom/commit/482f80e735f124ee6860f6854255c77170b862e7))
* update dashboard doc link ([#544](https://github.com/chopratejas/headroom/issues/544)) ([378d77e](https://github.com/chopratejas/headroom/commit/378d77e79d0020ca7fba3de8df7aaf910056ad2a))
* Update Next.js to 16.2.4 in docs/bun.lock to address GHSA-gx5p-jg67-6x7h (CVE-2026-44580) ([0b9f11a](https://github.com/chopratejas/headroom/commit/0b9f11a223bb6e6a6c1660ff1dfc1df6d67dfa84))
* Update Next.js to 16.2.6 in docs/package.json and package-lock.json to address GHSA-h64f-5h5j-jqjh (CVE-2026-44577) ([db5d15f](https://github.com/chopratejas/headroom/commit/db5d15f99e71b69a369eb9c161e04dbffb9b5d4a))
* Upgrade litellm to 1.86.2 to remediate CVE-2026-42271 ([07581b9](https://github.com/chopratejas/headroom/commit/07581b9e8075b833a6b543149008547260fe9dc0))


### Code Refactoring

* **cli:** factor shared wrap-subcommand scaffolding ([8eeb926](https://github.com/chopratejas/headroom/commit/8eeb9261680dd071654a87204521ccd3703ef77d))
* **cli:** factor shared wrap-subcommand scaffolding ([c74ad11](https://github.com/chopratejas/headroom/commit/c74ad113a4ced9968e45cad1077e6a020dc6a401))

## [0.22.4](https://github.com/chopratejas/headroom/compare/v0.22.3...v0.22.4) (2026-05-26)


### Bug Fixes

* **cli:** G1 remediation — non-string clobber, per-model systemMessage, openhands gate ([ea1976e](https://github.com/chopratejas/headroom/commit/ea1976e37a5147ecf37dbf5ffe4af5c2f2d1be6a))
* **cli:** wrap CLI breadth — cline, continue, goose, openhands ([8625f80](https://github.com/chopratejas/headroom/commit/8625f8075ed75d2a002f6ba357697de0fa1ec434))
* **cli:** wrap subcommands for cline, continue, goose, openhands ([c375fa1](https://github.com/chopratejas/headroom/commit/c375fa156dd0434256805f274c07be4f45db9814))
* **observability:** G3 remediation — bound cardinality + wire dead metrics ([2a717a9](https://github.com/chopratejas/headroom/commit/2a717a993ee99f9401f5cdf78a23dcecd7cb1a51))
* **observability:** RTK metrics + Rust observability (Phase H blocker) ([b36ad9f](https://github.com/chopratejas/headroom/commit/b36ad9fe1c6a488eb9ffbf0e8b38d989278cf8ef))
* **observability:** wire Phase G PR-G3 RTK + proxy metrics (H-blocker) ([5f264a5](https://github.com/chopratejas/headroom/commit/5f264a53292e292c9c56b837c2750d1a415b1ea9))
* **release:** tag format vX.Y.Z (drop release-please component prefix) ([4a39ef5](https://github.com/chopratejas/headroom/commit/4a39ef54ed6cdaa24d8f9fa49bbd3daf7100658e))
* **release:** tag format vX.Y.Z (drop release-please component prefix) ([0f3e3af](https://github.com/chopratejas/headroom/commit/0f3e3af6b2a154c5ecaeda3f9770cec97e9a3ba0))
* **subscription:** address G2 review findings — phantom delta, multi-worker race, silent fallbacks ([f68090c](https://github.com/chopratejas/headroom/commit/f68090c5b4bd9670ee7fc9a0c71e57f05072c18c))
* **subscription:** wire tokens_saved_rtk data plane ([c7d1247](https://github.com/chopratejas/headroom/commit/c7d1247a2bd06738c3b6c8e73e15902a7e428467))
* **subscription:** wire tokens_saved_rtk from RTK stats endpoint ([44c605f](https://github.com/chopratejas/headroom/commit/44c605fbb0e3ae4e7a92d9693d0da8bc21115b81))
* **tests:** drive RTK subprocess failure with real exec, not monkeypatched run ([9b6d637](https://github.com/chopratejas/headroom/commit/9b6d6374f13a88842a1944688005649ad3680acd))
* **tests:** mock logger.warning directly instead of relying on caplog ([c38dac3](https://github.com/chopratejas/headroom/commit/c38dac301e6bc702979ab11357a9c27a180ae060))
* **tests:** patch headroom.rtk.get_rtk_path, not the helpers alias ([317dffe](https://github.com/chopratejas/headroom/commit/317dffe58fb0c6233210bbc9e42ebf16b9288391))
* **tests:** tomllib fallback to tomli on python 3.10 ([74843d1](https://github.com/chopratejas/headroom/commit/74843d1d626de70158a359661a540c615ef1a6c5))

## [Unreleased]

### Security
- **`/debug/memory` loopback guard.** The endpoint was missing the
  `Depends(_require_loopback)` guard that all other `/debug/*` endpoints carry.
  External callers can no longer reach it.
- **`retry_max_attempts` zero guard.** When `retry_enabled=True` and
  `retry_max_attempts=0` the retry loop exited without setting `last_error`,
  causing `raise last_error` to raise `TypeError: exceptions must derive from
  BaseException`. A `RuntimeError` with an actionable message is now raised
  instead, and `ProxyConfig.__post_init__` rejects `retry_max_attempts < 1`
  at construction time.
- **Blocking subprocess on async event loop.** `_read_rtk_lifetime_stats` and
  `_read_lean_ctx_lifetime_stats` called `subprocess.run` directly on the
  asyncio thread. The `initialize_context_tool_session_baseline` function is
  now `async` and offloads the subprocess via `asyncio.to_thread`; the stats
  endpoint uses `await asyncio.to_thread(_get_context_tool_stats)`.
- **Hardcoded Neo4j credential in `docker-compose.yml`.** `NEO4J_AUTH` now
  defaults to `${NEO4J_AUTH:-neo4j/devpassword}` and is documented in
  `.env.example` (excluded from `.gitignore` via `!.env.example`).
- **`SemanticCache.get_memory_stats()` concurrent iteration.** The method
  iterates `self._cache.values()` without holding the async lock. A snapshot
  is now taken via `list(self._cache.values())` before iterating to avoid
  `RuntimeError: dictionary changed size during iteration` under async load.
- **Default Neo4j password in `ProxyConfig`.** `memory_neo4j_password` default
  changed from `"password"` to `""`. The proxy startup path now emits a
  `logger.warning` when `memory_backend == "qdrant-neo4j"` and the password
  is empty, prompting operators to set a real credential.

### Fixed
- **PyPI install clarity and release gating.** Documented `pipx --python python3.13`
  for environments where unsupported Python wheel tags cause older-version
  resolution, made PyPI publish failures block GitHub Releases unless
  `PYPI_SKIP=true`, and added an sdist `LICENSE` invariant.

- **`headroom learn` with claude-cli no longer fails silently on slow
  networks or large digests.** The CLI backend timeout was a hard 120s
  wall-clock cap with no liveness signal: a successful long analysis and
  a hung connection looked identical, and exit 0 with "no recommendations"
  was the only user-visible signal. Two changes:
  (1) **Streaming + idle timeout for claude-cli**: the command now uses
  `--output-format stream-json --verbose` and a watchdog thread reads
  events as they arrive. The process is killed only after
  `HEADROOM_LEARN_CLI_IDLE_TIMEOUT_SECS` (default 60s) of zero output, or
  after `HEADROOM_LEARN_CLI_TIMEOUT_SECS` (default 300s, was 120s) total.
  Long-but-active analyses run to completion; genuine hangs are caught
  fast. The final `type:"result"` event carries the assistant response.
  Drains stdout/stderr via reader threads so the watchdog works on
  Windows too. (2) **Env-var overrides for all CLI backends**:
  `HEADROOM_LEARN_CLI_TIMEOUT_SECS` is honored by gemini-cli and
  codex-cli as the wall-clock timeout; idle override applies only to the
  streaming claude-cli path.
- **`Learned: error recovery` section in MEMORY.md no longer bloats with
  stale, one-shot, or contradictory entries.** The matchers paired up
  unrelated tool calls (e.g. `state.rs` and `lib.rs` in the same dir
  becoming `File state.rs does not exist. The correct path is lib.rs.`),
  the dedup key was the literal rendered bullet text so near-duplicates
  each created their own row, the shutdown flush dropped the evidence
  gate to 1 so every singleton landed at session end, and there was no
  TTL or re-validation. Fixed at every layer:
  (1) **Emission**: Read recoveries require the failed/successful
  basenames to be identical or close in edit distance; Bash recoveries
  require a shared binary (allowing `python`↔`python3` and
  `ruff`↔`.venv/bin/ruff` variants) plus low-edit-distance OR a shared
  substantive non-flag token. Unrelated pairs are rejected at the source.
  (2) **Dedup**: error-recovery rows are hashed on recovery intent —
  Read on `(basename(error_path), basename(success_path))`, Bash on the
  primary command stripped of volatile suffixes (`| tail -N`, `2>&1`,
  etc.). Near-duplicates collapse into one row.
  (3) **Evidence gating**: default `min_evidence` raised from 2 to 5;
  shutdown-relaxation removed; new `--min-evidence` flag and
  `HEADROOM_MIN_EVIDENCE` envvar so embedded clients can tighten the
  threshold further.
  (4) **Render-time refinement**: drop rows not re-observed in 21 days,
  re-validate Read success paths against the filesystem, collapse
  same-error_path-with-multiple-targets into one "use Glob/Grep first"
  bullet, rank by `evidence_count * 0.5 ** (days/5)`, cap the section
  at 15. A→B / B→A contradiction pairs are also dropped at flush time.
  Patterns now stamp `first_seen_at` / `last_seen_at` on every save;
  `_bump_persisted_evidence` updates them via `json_set`. Other
  `Learned: …` categories (environment, preference, architecture) are
  untouched.
- **`headroom unwrap codex` now actually undoes `headroom wrap codex`** —
  previously there was no `unwrap codex` subcommand at all, so the injected
  `model_provider = "headroom"` / `[model_providers.headroom]` block stayed
  in `~/.codex/config.toml` forever and Codex continued routing through the
  (potentially stopped) proxy, surfacing as `Missing environment variable:
  OPENAI_API_KEY`. `wrap codex` now snapshots the pre-wrap
  `config.toml` to `config.toml.headroom-backup` before its first injection,
  and `unwrap codex` restores that snapshot byte-for-byte (or, if the
  backup is missing, strips only the Headroom-managed block and leaves
  surrounding user content intact). Safe no-op when run without a prior
  wrap. Reported by @raenaryl in Discord.
- **Image compressors now release shared router models after use and proxy shutdown** —
  the proxy/image compression path no longer keeps global `technique-router`
  and `SigLIP` model instances pinned in memory after one-off image
  optimization work. The `get_compressor()` helper now returns a fresh,
  caller-owned compressor instead of a process-lifetime singleton.
- **`headroom learn` no longer clobbers prior recommendations on re-run** —
  the marker block in `CLAUDE.md` / `MEMORY.md` is now merged with the
  prior block instead of wholesale-replaced. Sections re-surfaced by the
  new run win; sections not re-surfaced are carried forward so learnings
  accumulate across runs instead of disappearing. To fully rebuild the
  block, delete it manually and re-run. (#231)
- **`headroom learn` no longer emits dangling cross-references when a
  section is re-surfaced** — the analyzer now includes the project's
  current `<!-- headroom:learn -->` block (from `CLAUDE.md` and
  `MEMORY.md`) in the LLM digest as a "Prior Learned Patterns" section,
  and the system prompt instructs the LLM that re-emitting a section
  replaces the prior one wholesale. Prevents bullets like "`X` is *also*
  large — same rule as `Y`, `Z`" from appearing after `Y` and `Z` got
  dropped during per-section replacement. The writer's section-level
  carry-forward from #231 remains in place as a safety net for sections
  the LLM omits entirely. New helper `extract_marker_block` added to
  `headroom.learn.writer`.

### Added
- **`turn_id` linking agent-loop API calls to a single user prompt** — a new
  `compute_turn_id(model, system, messages)` helper in
  `headroom/proxy/helpers.py` hashes the message prefix up to and including
  the last user-text message, yielding an id that is stable across every
  agent-loop iteration of one prompt but rolls over when the user sends a
  new prompt (or runs `/compact`, `/clear`). `RequestLog` gained a
  `turn_id: str | None` field, which is stamped at every log site
  (anthropic handler bedrock + direct branches, and the streaming handler)
  and surfaced as `turn_id` in `/transformations/feed`. Lets downstream
  consumers (e.g. the Headroom Desktop Activity tab) aggregate savings per
  user prompt rather than per API call.
- **Live flush of traffic-learned patterns to CLAUDE.md / MEMORY.md** — the
  `TrafficLearner` now writes to agent-native context files continuously
  during proxy operation, not just at shutdown. A new dirty-flag debounced
  `_flush_worker` (10s window, `FLUSH_DEBOUNCE_SECONDS`) calls
  `flush_to_file()` whenever `_accumulate()` marks the learner dirty, so
  patterns surface in `CLAUDE.md` / `MEMORY.md` near real-time. Flushes
  read both persisted rows (via `_load_persisted_patterns_from_sqlite`)
  and the in-memory accumulator, bucket patterns by project via the learn
  plugin registry (`plugin.discover_projects()` + longest-path anchoring
  in `_project_for_pattern`), and route by `PatternCategory` to the
  correct file (`_patterns_to_recommendations` +
  `_CATEGORY_TO_TARGET`). Live flushes require `evidence_count >= 2`;
  the shutdown flush accepts single-evidence rows.

### Fixed
- **Traffic-learner evidence count stuck at 1; duplicate DB rows across
  restarts.** `_accumulate` queued patterns with the default
  `ExtractedPattern.evidence_count = 1` regardless of how many times the
  pattern was actually seen, so every persisted row landed at `1` and
  never crossed the live-flush gate (`evidence_count >= 2`). Worse, once
  a pattern was in `_saved_hashes` it was early-returned on every
  re-sighting, and `_saved_hashes` reset on process restart — so a second
  sighting in a later session inserted a duplicate row rather than
  bumping the existing one. Now: `_accumulate` writes the real
  accumulated count at save time, `start()` hydrates `_saved_hashes` +
  a new `_persisted_ids` map from the DB, and re-sightings bump the
  persisted row's `metadata.evidence_count` via an atomic `json_set`
  `UPDATE` (`_bump_persisted_evidence`). `_load_persisted_patterns_from_sqlite`
  now filters via `json_extract(metadata, '$.source')` instead of a
  LIKE on the raw JSON string, so rows survive metadata rewrites.

### Added
- **`HEADROOM_QDRANT_*` environment variables for memory Qdrant configuration**
  (#31) — `Memory(backend="qdrant-neo4j")`, `Mem0Config`, `MemoryConfig`, and
  `ProxyConfig` now resolve their Qdrant connection from
  `HEADROOM_QDRANT_URL`, `HEADROOM_QDRANT_HOST`, `HEADROOM_QDRANT_PORT`,
  `HEADROOM_QDRANT_API_KEY`, `HEADROOM_QDRANT_HTTPS`,
  `HEADROOM_QDRANT_PREFER_GRPC`, and `HEADROOM_QDRANT_GRPC_PORT`. Explicit
  constructor arguments still win; unset env keeps the existing
  `localhost:6333` defaults. Adds matching `--memory-qdrant-{url,host,port,api-key}`
  CLI flags. Enables hosted Qdrant (Qdrant Cloud) and shared/remote Qdrant
  stacks without code changes. New helper:
  [`headroom/memory/qdrant_env.py`](headroom/memory/qdrant_env.py).
- **Telemetry stack & install-mode identity fields** — anonymous beacon now
  reports `headroom_stack` (how Headroom is invoked: `proxy`, `wrap_claude`,
  `adapter_ts_openai`, ...) and `install_mode` (`wrapped` / `persistent` /
  `on_demand`), plus `requests_by_stack` for proxies that serve multiple
  integrations. Proxy exposes a `by_stack` bucket alongside `by_provider` /
  `by_model` on `/stats`, a matching `headroom_requests_by_stack` Prometheus
  counter, and an `X-Headroom-Stack` header honored by the FastAPI middleware.
  `headroom wrap <tool>` sets `HEADROOM_STACK=wrap_<agent>`; the TS SDK and
  all four adapters (`openai`, `anthropic`, `gemini`, `vercel-ai`) tag their
  compress calls. Schema migration:
  [`sql/upgrade_telemetry_stack_context.sql`](sql/upgrade_telemetry_stack_context.sql).
- **Canonical filesystem contract** (issue #175) — new `HEADROOM_CONFIG_DIR`
  (default `~/.headroom/config`, read-mostly) and `HEADROOM_WORKSPACE_DIR`
  (default `~/.headroom`, read-write state) env vars recognized by the Python
  proxy/CLI and the npm SDK. Additive; all existing per-resource env vars
  (`HEADROOM_SAVINGS_PATH`, `HEADROOM_TOIN_PATH`,
  `HEADROOM_SUBSCRIPTION_STATE_PATH`, `HEADROOM_MODEL_LIMITS`) continue to
  work with identical semantics. Docker install scripts and
  `docker-compose.native.yml` forward the new vars into containers so
  savings, logs, and telemetry resolve to the bind-mounted `.headroom` path.
  See [`wiki/filesystem-contract.md`](wiki/filesystem-contract.md).

### Changed
- **`/stats-history` now returns compact checkpoint history by default** — the
  JSON response keeps recent checkpoints dense while evenly sampling older
  checkpoints so long-running installs do not return ever-growing payloads.
  Add `history_mode=full` to fetch the full retained checkpoint list, or
  `history_mode=none` to skip it entirely while still receiving the derived
  hourly/daily/weekly/monthly rollups. Responses now include a
  `history_summary` block describing stored versus returned points.

### Fixed
- **Streaming Anthropic requests are now visible to `/stats.recent_requests`
  and `/transformations/feed`** — `_finalize_stream_response` did not call
  `self.logger.log(...)`, so the entire streaming Anthropic code path (the
  one Claude Code uses) silently bypassed the request logger. Only the
  non-streaming Anthropic path and the Bedrock streaming path were logged.
  As a consequence, `--log-messages` had no observable effect on the live
  transformations feed for typical traffic. The streaming finalizer now
  emits the same `RequestLog` shape the other paths do, including
  `request_messages` when `log_full_messages` is enabled.

## [0.5.22] - 2026-04-11

### Added
- **Cross-agent memory** — Claude saves a fact, Codex reads it back. All agents sharing one proxy share one memory store. Project-scoped DB at `.headroom/memory.db`, auto user_id from `$USER`.
- **Agent provenance tracking** — every memory records which agent saved it (`source_agent`, `source_provider`, `created_via`), with edit history on updates.
- **LLM-mediated dedup** — on `memory_save`, enriched response hints similar existing memories to the LLM. Background async dedup auto-removes >92% cosine duplicates. Zero extra LLM calls.
- **Memory for OpenAI and Gemini handlers** — context injection + tool handling wired into all three provider handlers (Anthropic, OpenAI, Gemini).
- **Plugin architecture for `headroom learn`** — each agent (Claude, Codex, Gemini) is a self-contained plugin. External plugins register via `headroom.learn_plugin` entry points. `--agent` flag for CLI.
- **GeminiScanner** for `headroom learn` — reads `~/.gemini/tmp/*/chats/session-*.json` and `.jsonl`.
- **Code graph integration** — `headroom wrap claude --code-graph` auto-indexes the project via [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) for call-chain traversal, impact analysis, and architectural queries. Opt-in, ~200 token overhead with Claude Code's MCP Tool Search.
- **OpenAI embedder auto-detection** — memory backend uses OpenAI embeddings when `sentence-transformers` is unavailable (no torch/2GB dependency needed).
- **Live traffic learning flush** — `headroom wrap <agent> --learn` flushes learned patterns to the correct agent-native file (MEMORY.md / AGENTS.md / GEMINI.md) at proxy shutdown.

### Changed
- **CodeCompressor disabled by default** — AST-based code compression produced invalid syntax on 40% of real files. Code now passes through uncompressed. Use `--code-graph` for code intelligence instead, or re-enable with `--code-aware`.
- **Shared tool name map** — consolidated tool normalization across all learn plugins into `_shared.py`.
- **Dynamic CLI agent detection** — `headroom learn` discovers agents via plugin registry, no hardcoded choices.

### Fixed
- **CodeCompressor statement-based truncation** — body truncation now walks AST statements (not lines), never cuts mid-expression. Fixes syntax errors on multi-line dict literals and function calls.
- **Docstring FIRST_LINE mode** — uses source lines directly instead of reconstructing from byte offsets. Properly handles all quote styles.
- **Memory shutdown queue drain** — patterns in the save queue were lost on proxy shutdown. Now drained before exit.

## [Unreleased]

### Added
- **Codex-proxy resilience hardening** — reduces event-loop starvation under cold-start reconnect storms
  - **Stage-timing instrumentation** — per-stage durations for both Codex WS accept and Anthropic `/v1/messages` pre-upstream phases emitted as a single `STAGE_TIMINGS` structured log line per request plus Prometheus histograms
  - **Per-pipeline shared warmup** — Anthropic + OpenAI pipelines eagerly load compressors/parsers once at startup; status merged into `WarmupRegistry` for `/debug/warmup` and `/readyz`
  - **WS session registry** — first-class tracking of active Codex WS sessions with deterministic relay-task cancellation and termination-cause classification (`client_disconnect`, `upstream_error`, `client_timeout`, etc.)
  - **Bounded pre-upstream Anthropic concurrency** — `--anthropic-pre-upstream-concurrency` / `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY` caps simultaneous `/v1/messages` pre-upstream work (body read, deep copy, first compression stage, memory-context lookup, upstream connect) so replay storms cannot starve `/livez`, `/readyz`, and new Codex WS opens. Default: auto `max(2, min(8, cpu_count))`; `0` or negative disables (unbounded)
  - **Loopback-only debug endpoints** — `/debug/tasks`, `/debug/ws-sessions`, `/debug/warmup` return `404` (not `403`) to non-loopback callers so external scanners cannot enumerate them
  - **Reconnect-storm repro harness** — `scripts/repro_codex_replay.py` drives concurrent WS + HTTP replay traffic against a local proxy and asserts `/livez` p99 under threshold; `--json` output routes JSON to stdout and the human summary to stderr
- **Proxy liveness and readiness health checks**
  - Adds `GET /livez` for process liveness and `GET /readyz` for traffic readiness
  - Keeps `GET /health` backward compatible while expanding it with readiness details and subsystem checks
  - Eagerly initializes configured memory backends during proxy startup so readiness reflects real serving capability
  - Wires `/readyz` into the Docker image `HEALTHCHECK` and the example `docker-compose.yml`
- **Durable proxy savings history**
  - Persists proxy compression savings history locally at `~/.headroom/proxy_savings.json`
  - Supports `HEADROOM_SAVINGS_PATH` to override the storage location
  - Adds `/stats-history` with lifetime totals plus hourly/daily/weekly/monthly rollups
  - Supports JSON and CSV export from `/stats-history`
  - Extends `/stats` with a `persistent_savings` block while keeping `savings_history` backward compatible
  - Adds a historical mode to `/dashboard` backed by `/stats-history`, including export actions
- **Proxy telemetry SDK override** via `HEADROOM_SDK`
  - Downstream apps can override the anonymous telemetry `sdk` field without patching installed files
  - Blank values fall back to the default `proxy` label
- **`headroom learn`** — Offline failure learning for coding agents
  - Analyzes past conversation history (Claude Code, extensible to Cursor/Codex)
  - **Success correlation**: for each failure, finds what succeeded after and extracts the specific correction
  - 5 analyzers: Environment, Structure, Command Patterns, Retry Prevention, Cross-Session
  - Writes specific learnings to CLAUDE.md (stable project facts) and MEMORY.md (session patterns)
  - Generic architecture: tool-agnostic `ToolCall` model, pluggable Scanner/Writer adapters
  - Dry-run by default, `--apply` to write, `--all` for all projects
  - Example output: "FirstClassEntity.java is not at axion-formats/ — actually at axion-scala-common/"
- **Read Lifecycle Management** — Event-driven compression of stale/superseded Read outputs
  - Detects when a Read output becomes stale (file was edited after) or superseded (file was re-read)
  - Replaces stale/superseded content with compact CCR markers, stores originals for retrieval
  - 75% of Read output bytes are provably stale or redundant (from real-world analysis of 66K tool calls)
  - Fresh Reads (latest read, no subsequent edit) are never touched — Edit safety preserved
  - Opt-in via `ReadLifecycleConfig(enabled=True)`, disabled by default
  - Handles both OpenAI and Anthropic message formats
- **any-llm backend** - Route requests through 38+ LLM providers (OpenAI, Mistral, Groq, Ollama, etc.) via [any-llm](https://mozilla-ai.github.io/any-llm/providers/)
  - Enable with `--backend anyllm --anyllm-provider <provider>`
  - Install with: `pip install 'headroom-ai[anyllm]'`
- Production-ready proxy server with caching, rate limiting, and metrics
- CLI command `headroom proxy` to start the proxy server
- **IntelligentContextManager** (semantic-aware context management)
  - Multi-factor importance scoring: recency, semantic similarity, TOIN importance, error indicators, forward references, token density
  - No hardcoded patterns - all importance signals learned from TOIN or computed from metrics
  - TOIN integration for retrieval_rate and field_semantics-based scoring
  - Strategy selection: NONE, COMPRESS_FIRST, DROP_BY_SCORE based on budget overage
  - Atomic tool unit handling (call + response dropped together)
  - Configurable scoring weights via `ScoringWeights` dataclass
  - `IntelligentContextConfig` for full configuration control
  - Backwards compatible with `RollingWindowConfig`
- **LLMLingua-2 Integration** (opt-in ML-based compression)
  - `LLMLinguaCompressor` transform using Microsoft's LLMLingua-2 model
  - Content-aware compression rates (code: 0.4, JSON: 0.35, text: 0.3)
  - Memory management utilities: `unload_llmlingua_model()`, `is_llmlingua_model_loaded()`
  - Proxy integration via `--llmlingua` flag
  - Device selection: `--llmlingua-device` (auto/cuda/cpu/mps)
  - Custom compression rate: `--llmlingua-rate`
  - Helpful startup hints when llmlingua is available but not enabled
  - ~~Install with: `pip install headroom-ai[llmlingua]`~~ (the `[llmlingua]` extra was removed in 0.9.x)
- **Code-Aware Compression** (AST-based, syntax-preserving)
  - `CodeAwareCompressor` transform using tree-sitter for AST parsing
  - Supports Python, JavaScript, TypeScript, Go, Rust, Java, C, C++
  - Preserves imports, function signatures, type annotations, error handlers
  - Compresses function bodies while maintaining structural integrity
  - Guarantees syntactically valid output (no broken code)
  - Automatic language detection from code patterns
  - Memory management: `is_tree_sitter_available()`, `unload_tree_sitter()`
  - Uses `tree-sitter-language-pack` for broad language support
  - Install with: `pip install headroom-ai[code]`
- **ContentRouter** (intelligent compression orchestrator)
  - Auto-routes content to optimal compressor based on type detection
  - Source hint support for high-confidence routing (file paths, tool names)
  - Handles mixed content (e.g., markdown with code blocks)
  - Strategies: CODE_AWARE, SMART_CRUSHER, SEARCH, LOG, TEXT, LLMLINGUA
  - Configurable strategy preferences and fallbacks
  - Routing decision log for transparency and debugging
- **Custom Model Configuration**
  - Support for new models: Claude 4.5 (Opus), Claude 4 (Sonnet, Haiku), o3, o3-mini
  - Pattern-based inference for unknown models (opus/sonnet/haiku tiers)
  - Custom model config via `HEADROOM_MODEL_LIMITS` environment variable
  - Config file support: `~/.headroom/models.json`
  - Graceful fallback for unknown models (no crashes)
  - Updated pricing data for all current models

### Fixed
- **Event.wait task leak in subscription trackers** — `asyncio.shield` pattern prevents cancellation of the outer `wait_for` from leaking the inner `Event.wait` task
- **Python 3.10 compatibility for memory-context fail-open** — catches `asyncio.TimeoutError` (the 3.10-compatible alias) rather than `TimeoutError` to preserve behaviour on older runtimes
- **uvicorn `proxy_headers=False`** — refuses `Forwarded` / `X-Forwarded-For` rewrites so the loopback guard on `/debug/*` cannot be spoofed by a misconfigured reverse proxy
- **First-frame timeout for Codex WS accepts** — guards against a client that opens a handshake and never sends the first frame; relays cancel deterministically with `client_timeout`
- **Semaphore leak on unexpected exception in Anthropic pre-upstream path** — the finalizer now releases the pre-upstream semaphore on every exit path (early 4xx, cache hit, upstream error, streaming handoff)
- **`active_relay_tasks` gauge double-decrement** — `deregister_and_count` returns `(handle, released_task_count)` atomically so the handler decrements the Prometheus gauge by the exact number it registered, eliminating drift

### Internal
- **IPv6-mapped loopback recognition** — the loopback guard parses `::ffff:127.0.0.1` and other dual-stack literals through `ipaddress.ip_address(...).is_loopback`
- **Lock-free stage-timing accumulators** — `record_stage_timings` writes to per-path counters that do not contend with `/metrics` export or `record_request`
- **Narrow `contextlib.suppress` in relay classification** — only `CancelledError` is suppressed where we reclassify it; other exceptions propagate so termination cause stays truthful
- **`jitter_delay_ms` helper** — shared exponential-backoff + 50-150% jitter formula in `headroom/proxy/helpers.py`; used by three proxy retry sites and mirrored inline in the repro harness

## [0.2.0] - 2025-01-07

### Added
- **SmartCrusher**: Statistical compression for tool outputs
  - Keeps first/last K items, errors, anomalies, and relevance matches
  - Variance-based change point detection
  - Pattern detection (time series, logs, search results)
- **Relevance Scoring Engine**: ML-powered item relevance
  - `BM25Scorer`: Fast keyword matching (zero dependencies)
  - `EmbeddingScorer`: Semantic similarity with sentence-transformers
  - `HybridScorer`: Adaptive combination of both methods
- **CacheAligner**: Prefix stabilization for better cache hits
  - Dynamic date extraction
  - Whitespace normalization
  - Stable prefix hashing
- **RollingWindow**: Context management within token limits
  - Drops oldest tool units first
  - Never orphans tool results
  - Preserves recent turns
- **Multi-Provider Support**:
  - Anthropic with official `count_tokens` API
  - Google with official `countTokens` API
  - Cohere with official `tokenize` API
  - Mistral with official tokenizer
  - LiteLLM for unified interface
- **Integrations**:
  - LangChain callback handler (`HeadroomOptimizer`)
  - MCP (Model Context Protocol) utilities
- **Proxy Server** (`headroom.proxy`):
  - Semantic caching with LRU eviction
  - Token bucket rate limiting
  - Retry with exponential backoff
  - Cost tracking with budget enforcement
  - Prometheus metrics endpoint
  - Request logging (JSONL)
- **Pricing Registry**: Centralized model pricing with staleness tracking
- **Benchmarks**: Performance benchmarks for transforms and relevance scoring

### Changed
- Improved token counting accuracy across all providers
- Enhanced tool output compression with relevance-aware selection

### Fixed
- Mistral tokenizer API compatibility
- Google token counting for multi-turn conversations

## [0.1.0] - 2025-01-05

### Added
- Initial release
- `HeadroomClient`: OpenAI-compatible client wrapper
- `ToolCrusher`: Basic tool output compression
- Audit mode for observation without modification
- Optimize mode for applying transforms
- Simulate mode for previewing changes
- SQLite and JSONL storage backends
- HTML report generation
- Streaming support

### Safety Guarantees
- Never removes human content
- Never breaks tool ordering
- Parse failures are no-ops
- Preserves recency (last N turns)

---

## Migration Guide

### From 0.1.x to 0.2.x

The 0.2.0 release is backward compatible. New features are opt-in:

```python
# Old code still works
from headroom import HeadroomClient, OpenAIProvider

# New SmartCrusher (replaces ToolCrusher for better compression)
from headroom import SmartCrusher, SmartCrusherConfig

config = SmartCrusherConfig(
    min_tokens_to_crush=200,
    max_items_after_crush=50,
)
crusher = SmartCrusher(config)

# New relevance scoring
from headroom import create_scorer

scorer = create_scorer("hybrid")  # or "bm25" for zero deps
```

### Using the Proxy

New in 0.2.0 - run Headroom as a proxy server:

```bash
# Start the proxy
headroom proxy --port 8787

# Use with Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

[Unreleased]: https://github.com/chopratejas/headroom/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/chopratejas/headroom/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/chopratejas/headroom/releases/tag/v0.1.0
