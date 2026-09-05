# Changelog

All notable changes to Headroom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Features
- **proxy:** opt-in cost-aware model routing ([#1706](https://github.com/headroomlabs-ai/headroom/issues/1706)). Set `HEADROOM_MODEL_ROUTER_ENABLED=1` and `HEADROOM_MODEL_ROUTES` (a JSON array of ordered rules) to rewrite the upstream model based on estimated input size and tool presence, complementary to content compression, e.g. send small, tool-free requests to a cheaper model. First matching rule wins, and each decision is logged with a reason so routing stays observable. Malformed rules fail open (the rule is skipped, never silently widened). Disabled by default so behavior is unchanged, skipped under `x-headroom-bypass`/passthrough, and currently applied on the Anthropic `/v1/messages` path.
- **install:** `headroom install apply` now accepts `--code-aware/--no-code-aware`, `--intercept-tool-results`, `--protect-tool-results`, and `--bedrock-profile`, mirroring the equivalent flags already on `headroom proxy`. Previously the only way to run a persistent deployment with these settings was to hand-edit `manifest.json` after the fact, which silently reverts on the next `install apply`.
- **install:** `headroom install apply --env KEY=VALUE` (repeatable) passes environment variables into supervised runners (macOS launchd, Linux systemd/cron, Windows services/tasks). These runners previously started with a bare environment and did not inherit the interactive shell's exports — e.g. a custom `HEADROOM_WORKSPACE_DIR` never reached the supervised process, so `headroom install agent run` looked for its manifest in the wrong location and failed outright even though `install apply` itself succeeded. `--env` values are merged into `DeploymentManifest.base_env` last, so they can override auto-derived defaults, and are threaded into the generated `run-headroom.sh`/`ensure-headroom.sh` (and Windows equivalents) as `export`/`$env:` lines before the `exec`.

### Fixed
- **memory:** preserve semantically similar memories after `memory_save`. Cosine similarity now produces a consolidation hint only; it no longer schedules a background deletion, because related memories can describe distinct facts. Supersession remains available through the explicit `memory_update` path with a caller-supplied memory ID.
- **mcp:** reap orphaned `headroom mcp serve` processes when the launching client dies. An MCP stdio server relies on stdin EOF to shut down, but an abrupt client `SIGKILL` leaves the SDK's blocking stdin-reader thread wedged, so `server.run()` never returns; the process is reparented to init/launchd (`ppid == 1`) and lingers, pinning one Python interpreter + tree-sitter grammars per dead session (observed 3+ simultaneously). `run_stdio()` now runs a parent-death watchdog alongside `server.run()` that fires when the captured parent pid changes and `os._exit(0)`s from inside the stdio context manager, bypassing the same wedged teardown ([#2185](https://github.com/headroomlabs-ai/headroom/issues/2185), [#1761](https://github.com/headroomlabs-ai/headroom/issues/1761)).
- **cache/prefix-freeze:** resolve `PrefixCacheTracker`s per conversation lineage within a session id, so concurrent conversations sharing a fallback id no longer thrash one tracker's frozen-prefix state ([#2085](https://github.com/headroomlabs-ai/headroom/issues/2085)). Without an `x-headroom-session-id` header the fallback id hashes `model + system prompt` — identical across a Claude Code session and every one of its parallel subagents (and any set of sessions reusing one system prompt). On the shared tracker their interleaved histories cross-contaminate the freeze state: the forwarded prefix is byte-unstable on nearly every turn and the provider prompt cache is re-written instead of read — reported as ~4.4x cache-creation inflation and a 2.5–3x net cost increase under Claude Code. `SessionTrackerStore.resolve_tracker` now reuses the tracker whose previous request messages are a prefix of the incoming history (client histories are append-only, so a conversation's next request always extends its previous one), starts a fresh lineage when the history diverges or was rewritten (client-side compaction — that provider cache line is gone anyway), and caps lineages per session id (`PrefixFreezeConfig.max_lineages_per_session`, default 32; over-cap conversations share one overflow tracker instead of evicting established lineages, so a fan-out storm past the cap degrades only its own tail — and `0` disables lineage splitting). Matching compares the original client bytes under the same canonical cross-turn equivalence as the cache-stable delta path (`_canonicalize_for_prefix_compare`), so a moved cache breakpoint, string<->block content sugar, or per-turn transport annotations do not read as a rewrite. Separately, the fallback id now hashes only the LEADING run of `role:"system"` messages: agentic clients interleave `<system-reminder>` turns into the history as actual system-role messages (hook output, skills lists, truncation notices), and hashing those rotated the session id mid-conversation — orphaning the prefix tracker and every other session-sticky subsystem (beta headers, CCR/memory registries, the compression cache) each time a reminder landed. Both handler paths now derive the session id and the lineage from the same original client bytes, so a turn-dependent hook rewrite cannot rotate one without the other. The session id itself never changes: session-sticky state keyed on it (beta-header stickiness, CCR and memory-tool registries, the compression cache) is untouched, and a single-conversation session keeps its exact previous behavior (the first lineage lives under the bare id).
- **proxy/bedrock:** wire `PrefixCacheTracker` updates into both Bedrock backend paths (`handle_anthropic_messages`'s non-streaming branch in `anthropic.py`, and `_stream_response_bedrock` in `streaming.py`). `update_from_response()` was previously only called from the direct-Anthropic-API branch; both Bedrock branches returned before ever reaching it, so the tracker's state stayed permanently empty for the life of a session on any `--backend bedrock` deployment: `extract_cache_stable_delta()` always saw no previous turn, and `--mode cache` fell back to full unmodified passthrough on every turn instead of freezing the already-cached prefix and compressing only the new suffix.
- **install:** `install_supervisor`'s macOS branch did an unconditional `launchctl bootout` followed by a bare `bootstrap` with no retry, unlike `start_supervisor`, which already rides out the ~15s EIO (error 5) window launchd exhibits for several seconds after a bootout. This left `install apply`'s own reinstall path (and anything that re-applies a deployment, e.g. a future `headroom doctor --fix`) exposed to a race that previously required manual recovery (bootout + remove the plist + reapply). Extracted the retry loop already used by `start_supervisor` into a shared `_bootstrap_with_retry()` helper, now used by both call sites.
- **proxy/savings:** `SavingsTracker.record_request()` only appended a history point when `tokens_saved > 0` (headroom's own lossy compression). In `--mode cache`, `tokens_saved` is near-always 0 by design, since the frozen prefix is byte-replayed rather than compressed to keep the provider's prompt cache warm. That silently dropped every history point on a cache-mode deployment even when `cache_read_tokens`/`cache_savings_usd` were large, making `headroom-monthly`-style tooling read as a total savings collapse. The guard now fires on `tokens_saved` OR `cache_read_tokens`, and the appended entry carries `cache_read_tokens`/`cache_savings_usd` so downstream consumers can show them; `_normalize_history_entry` defaults both fields to 0/0.0 for legacy entries that predate this change.
- **litellm:** vendor-specific top-level fields on `/v1/chat/completions`, including vLLM's `chat_template_kwargs` for per-request Qwen3 thinking-mode toggles, now reach OpenAI-compatible backends through LiteLLM `extra_body` instead of being dropped by the standard-parameter allowlist ([#2128](https://github.com/headroomlabs-ai/headroom/issues/2128)).
- **cache aligner:** hash the actual frozen Claude Code prefix instead of only system-message text, so `stable_prefix_hash` / `prefix_changed` now surface prompt-cache churn when a cached tool-result block changes without any system-prompt edit ([#2085](https://github.com/headroomlabs-ai/headroom/issues/2085)).
- **proxy image compression:** run native image compression in a spawned subprocess so an OpenCV or KleidiCV crash now fails open to the original image payload instead of taking down the whole proxy process ([#2107](https://github.com/headroomlabs-ai/headroom/issues/2107)).
- **proxy/windows:** support Windows selector-event-loop startup on uvicorn versions older than 0.36. Newer uvicorn accepts `loop="asyncio:SelectorEventLoop"` as a custom loop-factory import path, but older versions treat it as an unknown built-in loop name and raise `KeyError`. Windows now sets `WindowsSelectorEventLoopPolicy` for those older versions instead of passing an unsupported `loop` value ([#1650](https://github.com/headroomlabs-ai/headroom/issues/1650), [#1621](https://github.com/headroomlabs-ai/headroom/issues/1621)).
- **backends/litellm:** preserve `cache_control` on `tool_result` blocks when converting Anthropic messages for the Bedrock Converse path, and complete streaming cache-stats surfacing in `stream_message`. Complements [#1390](https://github.com/headroomlabs-ai/headroom/pull/1390), which preserves `cache_control` on the system prompt and plain text blocks but explicitly leaves `tool_result` out of scope — in agent loops the moving cache breakpoint lands on the tail `tool_result` far more often than on the system prompt, so that gap left most of the caching benefit on the table. Separately, `stream_message` never requested `stream_options.include_usage`, so LiteLLM/Bedrock never returned a usage chunk over SSE and `cache_read_input_tokens`/`cache_creation_input_tokens` always reported 0 downstream even when the prompt cache was genuinely engaged; the terminal `message_delta` now carries the real cache values captured from the trailing usage chunk once the stream completes.
- **shared_context:** `SharedContext.put` no longer evicts an unrelated entry when it merely updates a key that is already cached at capacity — same defect class fixed for `SemanticCache` in [#2094](https://github.com/headroomlabs-ai/headroom/pull/2094).
- **compress:** stop mutating the caller's `CompressConfig`. `compress(config=my_cfg, protect_recent=0, target_ratio=0.2)` used to write those kwargs onto `my_cfg`, so a shared per-agent config was silently rewritten by every request that overrode a single option.
- **paths:** reject `.`, `..`, and NUL as plugin names so `plugin_config_dir` / `plugin_workspace_dir` cannot resolve outside the `plugins/` sandbox. Previously `plugin_config_dir("..")` returned the entire config root and `plugin_workspace_dir("..")` returned the workspace root (savings ledger, memory DB, license cache, logs).
- **backends/litellm:** drop tool names over 64 chars before calling Bedrock Converse (`send_message` and `stream_message`), instead of letting the whole request 401. The Bedrock Converse API hard-rejects any tool name past that length, and Claude Code includes every globally-added claude.ai MCP connector tool in every request, even ones the user hasn't enabled locally, so a single oversized connector name broke every call through this backend. Only the `bedrock` provider filters; other providers forward tool names unfiltered.
- **memory:** annotate `_EMBEDDER_CACHE` as `dict[tuple[str, str, str], Embedder]` to match the 3-element key (backend, model, ollama_base_url). The stale 2-tuple annotation made `mypy headroom` fail on `main`, which broke the `lint` CI job on every open PR.
- **install:** include `orjson` in the `[proxy]` extra so `uv tool install "headroom-ai[all]"` satisfies LiteLLM OpenRouter/provider backends that import it at runtime ([#2056](https://github.com/headroomlabs-ai/headroom/issues/2056)).
- The dashboard's per-request metadata (the `recent_requests` / `request_logs`
  tail and the `config` block with upstream URLs) is gated to loopback callers
  via `_request_is_loopback`. When Headroom runs in a bridge-network container
  (Docker/podman, or Apple Containerization / mocker), a browser on the host
  reaches the proxy through the container gateway, so `request.client.host` is
  the gateway IP rather than `127.0.0.1` — the sensitive block was stripped and
  the "Recent Requests" table rendered empty even though the operator is local.
  A peer inside an operator-configured trusted-gateway CIDR
  (`HEADROOM_PROXY_TRUSTED_GATEWAY_CIDRS`, already used to sanitize
  `X-Forwarded-*`) is now treated as loopback-equivalent, while the loopback
  `Host`-header gate is retained as the DNS-rebinding defence. Opt-in and empty
  by default, so there is no behavior change unless the gateway CIDR is
  allow-listed.
- Non-finite values (`NaN`, `Infinity`) in `proxy_savings.json` or in upstream
  cost/token metadata no longer crash the proxy or corrupt the savings
  dashboard. `SavingsTracker`'s numeric coercion caught only `TypeError` and
  `ValueError`, so `int(float('inf'))` raised an uncaught `OverflowError` while
  loading persisted state (`SavingsTracker.__init__` failed and the proxy would
  not start), and `float('nan')`/`float('inf')` passed straight through, then
  serialized to `NaN`/`Infinity` literals that the dashboard's `JSON.parse`
  rejects. `json.loads` accepts those literals, so one bad write poisoned every
  later start. Both coercion helpers now also catch `OverflowError` and reject
  non-finite floats, failing open to safe defaults.
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
* **docs/ccr:** qualify the current CCR auto-resolution claim by provider. The docs now state that transparent `headroom_retrieve` handling is wired on the Anthropic and OpenAI proxy paths, while native Gemini still lacks that server-side response-handler path and Gemini's OpenAI-compatible endpoint can fail round-2 continuations with `MALFORMED_FUNCTION_CALL` ([#2041](https://github.com/headroomlabs-ai/headroom/issues/2041)).
* **docs/claude:** document that `ENABLE_TOOL_SEARCH=true` is correct for the standalone Claude CLI through Headroom but currently breaks tool-result rendering in Anthropic's VSCode extension webview, and point persistent-install users at the manifest override to set `tool_envs.claude.ENABLE_TOOL_SEARCH` to `"false"` for that target ([#2028](https://github.com/headroomlabs-ai/headroom/issues/2028)).

### Added

* **integrations:** CrewAI tool compression — `wrap_tools_with_headroom()` wraps CrewAI `BaseTool` instances with automatic output compression via `compress_tool_result()`, with per-tool metrics tracking ([#1379](https://github.com/headroomlabs-ai/headroom/issues/1379)).
* **integrations:** AutoGen tool compression — `wrap_tools_with_headroom()` wraps AutoGen `FunctionTool` instances (sync and async) with automatic output compression, including per-tool metrics tracking ([#1379](https://github.com/headroomlabs-ai/headroom/issues/1379)).

### Features

* **grok-build:** add first-class Grok Build support — `headroom wrap grok-build` / `headroom unwrap grok-build`, reversible `~/.grok/config.toml` injection (in-place `base_url` rewrite when `[model.grok-build]` already exists), `GrokRegistrar` MCP install, and install/telemetry wiring ([#1629](https://github.com/headroomlabs-ai/headroom/pull/1629)).
* **wrap:** add `headroom wrap omp` / `headroom unwrap omp` for Oh My Pi — points omp's built-in `anthropic` provider at the local proxy via a marker-fenced `providers.anthropic.baseUrl` override in `~/.omp/agent/models.yml`, snapshotting the pre-wrap file byte-for-byte and restoring it on unwrap. omp resolves its Anthropic chat endpoint from models.yml (`ANTHROPIC_BASE_URL` only feeds its web-search helper), and a same-ID override keeps omp's bundled model catalog and stored credentials ([#1149](https://github.com/headroomlabs-ai/headroom/issues/1149))
* **compress:** expose `frozen_message_count` in library-mode `compress()` via a new `CompressConfig` field (default `0`, unchanged behavior). `read_lifecycle.apply()` already skips stale-Read replacements inside a frozen message prefix, but only the proxy handlers could pass it — `ContentRouter` reads it from transform kwargs and the public API never forwarded it. Library-mode callers that manage their own conversation loop can now stop transforms from rewriting messages already anchored in the provider's prompt cache, which would otherwise convert 0.1x cached prefix reads into full-price cache writes ([#2178](https://github.com/headroomlabs-ai/headroom/pull/2178)).

* **proxy:** report a new-content-relative input savings rate in `/stats`: `tokens.new_input_tokens` (provider-billed non-cache-read input: uncached + cache-write tokens, from response usage) and `tokens.new_input_savings_percent` (savings as a fraction of new input plus the tokens compression removed before they could be billed). The existing whole-request ratios recount the full transcript on every turn, so a 200-turn session counts its history 200x into the denominator and long-running cached sessions (especially 1M-context models, which never compact) dilute toward ~0% regardless of how well compression performs on content newly entering context. Purely additive; existing fields unchanged. Reports 0 when no cache usage data exists (e.g. providers without cache metrics) rather than dividing savings by themselves.
* **transforms:** first-class C# support in `CodeAwareCompressor` via the tree-sitter `csharp` grammar already shipped in the pinned `tree-sitter-language-pack` — no new dependencies ([#1664](https://github.com/headroomlabs-ai/headroom/issues/1664)). Parity with Java/C++/Rust: signatures preserved verbatim, method/constructor/destructor/operator/local-function bodies compressed; block-scoped and file-scoped namespaces, records, structs, interfaces, and enums handled; C#-distinctive auto-detection. Preprocessor conditionals (`#if`…`#endif`) are preserved verbatim as opaque regions (blocks wrapping only `using` directives stay with the imports), `#region` markers no longer swallow the following line during class-member extraction, and top-of-file license banners / `#region License` headers stay on top instead of being relocated below the code. Real-repo runs: 16.1% tokens saved on Newtonsoft.Json (945 files), 37.8% on Polly (797 files), output syntax-valid for 1742/1742 files.
* **proxy:** add provider-only HTTP proxy routing via `--http-proxy` and `HEADROOM_HTTP_PROXY`. Upstream LLM provider calls can now use an HTTP proxy without setting process-wide `HTTP_PROXY`/`HTTPS_PROXY` variables that are inherited by tool executions; proxied provider clients use HTTP/1.1 so HTTPS provider APIs can tunnel through CONNECT.
* **proxy:** add output shaping for OpenAI Responses traffic on `/v1/responses` HTTP requests and Codex WebSocket `response.create` frames, with stable output-savings holdout keys and counted WS token strata for the experiment.
* **stats:** per-bucket output-shaping savings in `/stats-history`. Each `series` bucket (hourly/daily/weekly/monthly) now carries `output_tokens_saved_delta` and `output_savings_usd_delta` alongside the existing compression deltas, sourced from a per-request synthetic-control estimate (`SavingsRecorder.estimate_request_savings`) threaded through `record_request` into the rollup. Lets dashboards chart output-shaping savings over time as a distinct series — previously it existed only as a single global aggregate. Additive and backward-compatible: pre-feature checkpoints default the new fields to 0 ([#1816](https://github.com/headroomlabs-ai/headroom/issues/1816)).
* **observability:** the `headroom.compression.pipeline` span now also carries the OpenTelemetry GenAI semantic-convention attribute `gen_ai.request.model` alongside the existing `headroom.*` attributes, so Headroom's traces group and filter by the standard `gen_ai.*` schema in any OTel-native backend (Grafana, Datadog, etc.). Purely additive; no existing attribute changed. `gen_ai.operation.name`, `gen_ai.provider.name`, and `gen_ai.usage.*` are deliberately deferred (they need per-caller operation threading, reliable upstream-provider resolution, and response-path usage respectively).
* **wrap:** `headroom wrap claude --1m` preserves the 1M context window. Behind a custom `ANTHROPIC_BASE_URL` (the proxy) Claude Code drops the `context-1m` beta header and caps the window at 200k for entitled subscription users; the opt-in flag sets `ANTHROPIC_MODEL=<opus>[1m]` on the launched process so the 1M window activates through Headroom. A model already selected via `ANTHROPIC_MODEL` is preserved (only the `[1m]` suffix is appended) ([#1158](https://github.com/chopratejas/headroom/issues/1158)).
* **learn:** weight loops in `headroom learn`. A new loop detector (`headroom/learn/loops.py`) recognizes repeated tool-call patterns — including RTK re-fetch loops, where RTK's output truncation makes the agent re-run larger-limit variants of a *successful* command — collapses output-limit variants to one signature, measures the wasted tokens, surfaces loops as a highest-priority digest section, and weights loop guardrails above one-off rules by their measured waste. Previously loops had no special weight and a no-failure re-fetch loop was skipped entirely. Adds an RTK-loop eval (`benchmarks/rtk_loop_learn_eval.py`) that reproduces a loop, runs it through Learn, and asserts the generated guardrail ranks first and prevents re-triggering.
* **learn:** write per-project learnings to the personal, gitignored `CLAUDE.local.md` by default instead of the team-shared `CLAUDE.md`, matching Claude Code's memory convention so machine-specific paths and tool-discovery byproducts no longer pollute the shared file. Adds a `--target` flag to override the destination (e.g. `--target CLAUDE.md` to opt back into the shared file, or any custom path), and auto-migrates a stale learned-patterns block out of an existing `CLAUDE.md` into `CLAUDE.local.md` with a warning ([#1072](https://github.com/chopratejas/headroom/issues/1072)).
* **proxy/transforms:** take large cold-start contexts off the synchronous kompress path — the root cause behind the `compression_first_stage` 30s-timeout + leaked-thread → executor-saturation cascade ([#1171](https://github.com/chopratejas/headroom/issues/1171)). A token size-gate inside the ML boundary routes oversized text away from ModernBERT (`HEADROOM_KOMPRESS_MAX_TOKENS`); a cooperative chunk-deadline bounds any kompress run that does proceed (`HEADROOM_COMPRESSION_DEADLINE_MS`); an opt-in off-path mode forwards uncompressed immediately and compresses in a single per-process background drain so the request never blocks on ML (`HEADROOM_BACKGROUND_COMPRESSION`); and a new native `TextCrusher` — a fast deterministic extractive prose compressor in `headroom._core` that reuses the shared BM25 relevance scorer — is the fast alternative to ModernBERT for large plain text (`HEADROOM_TEXT_CRUSHER`). All default off and fail-open. On a SQuAD answer-retention eval (requires the SQuAD dev set) TextCrusher keeps ~94% of buried answers at 30% size vs ~36% for truncate/random, and runs in one O(n) pass -- sub-second where ModernBERT takes minutes (self-contained speed benchmark in `benchmarks/text_crusher_quality_eval.py`).
* **proxy/transforms:** `TextCrusher` now compresses CJK (Chinese/Japanese/Korean) text ([#1171](https://github.com/chopratejas/headroom/issues/1171)). CJK has no spaces or ASCII sentence terminators, so the prior ASCII splitter/tokenizer collapsed a whole CJK paragraph into one segment/one token and passed it through near-uncompressed. CJK-bearing input now takes an ICU (`icu_segmenter`, UAX#29 + dictionary) sentence/word segmentation path with a local BM25 relevance over the ICU tokens; pure-ASCII text is byte-identical to before, and the shared BM25 scorer is untouched. On real CMRC2018 Chinese QA, answer-retention under compression rises from 34% to ~91%; end-to-end aggregate savings on real CJK content rise from 16% to 40%.
* **proxy:** measure and surface rolling and current token throughput metrics (active/wall-clock input, compression, effective forward, and streamed generation) in `headroom perf` CLI and the dashboard ([#959](https://github.com/chopratejas/headroom/issues/959)).
* **vibe:** add Mistral Vibe CLI support with `headroom wrap vibe`.
* **proxy:** per-project savings breakdown on the dashboard for all wrapped agents — Claude Code, Codex, aider, Copilot, and Cursor ([#802](https://github.com/chopratejas/headroom/issues/802)). `headroom wrap claude`/`codex` tag requests with an `X-Headroom-Project` header (launch-directory name); `wrap aider`/`copilot`/`cursor` — whose clients cannot send custom headers — use a `/p/<name>` base-URL prefix the proxy strips. Savings are aggregated per project (persisted, schema v3 with transparent v2 migration), exposed as `savings.per_project` in `/stats` and `projects` in `/stats-history`, and shown in a Per-Project Savings dashboard table.
* **memory:** opt-in Apple-GPU (MPS) embedding offload via `HEADROOM_EMBEDDER_RUNTIME=pytorch_mps`. When set (and Apple MPS is available), the memory embedder runs on the torch sentence-transformers backend on the Apple GPU instead of the default ONNX CPU embedder, freeing the CPU under load. If MPS or the dependencies are unavailable, Headroom logs a warning and uses the existing default embedder selection path (ONNX when available, then the pre-existing local fallback). MPS encode calls are serialized internally (torch-MPS is not thread-safe). Adds the new `[pytorch-mps]` extra (`pip install 'headroom-ai[pytorch-mps]'`). Default behavior is unchanged.
* **proxy:** cross-region Bedrock inference-profile detection — geo-prefixed model IDs (`eu.`/`us.`/`apac.`/`global.`) are now resolved to their canonical vendor, so Anthropic cross-region profiles (e.g. `eu.anthropic.claude-haiku-4-5-20251001-v1:0`) receive live-zone compression instead of being silently skipped ([#999](https://github.com/chopratejas/headroom/pull/999)).
* **proxy:** Converse-body compression on the native Bedrock route — the live-zone dispatcher now recognizes Bedrock Converse content blocks (typeless `{"text": …}`, not only Anthropic `{"type":"text", …}`), so Converse user-message text compresses; `run_anthropic_compression` no longer bails to passthrough when the body lacks an InvokeModel `anthropic_version` envelope, and envelope re-emit stays gated on successful parse ([#999](https://github.com/chopratejas/headroom/pull/999)).
* **docker:** bundle `headroom-proxy` binary in published `runtime` and `runtime-slim` images — closes [#976](https://github.com/chopratejas/headroom/issues/976) ([#999](https://github.com/chopratejas/headroom/pull/999)).
* **transforms:** add opt-in audit-safe mode to `SmartCrusher` — `SmartCrusherConfig(audit_safe=True, protected_patterns=[...], fail_closed_on_protected_loss=True)`. Rows matching a protected pattern are scanned before JSON-array compression and guaranteed to survive the compressed output verbatim afterward (never dropped, never replaced by an opaque `<<ccr:...>>` marker only). Applies on both the `crush_array_json` convenience API and the `_smart_crush_content` path `apply()` uses for real tool-output compression. If a protected row still can't be preserved after the splice-back pass, the crusher fails closed by returning the original uncompressed content (or ships a best-effort result with a warning when `fail_closed_on_protected_loss=False`). Default is `audit_safe=False` — no behavior change for existing callers ([#1705](https://github.com/chopratejas/headroom/issues/1705)).

### Bug Fixes

* **ccr:** normalise a retrieved hash to lowercase so an uppercase echo still hits the store. `parse_tool_call` validated the hash case-insensitively (`hash_key.lower()`) but returned it verbatim, while the compression store keys every entry by a lowercase hash (sha256 hexdigest, and `explicit_hash.lower()` on write) and `retrieve` / `get_entry_status` look the key up as-is. A model that echoed the marker hash in uppercase therefore passed validation but missed the store, failing an otherwise-valid `headroom_retrieve` and losing the original content. `parse_tool_call` now returns the canonical lowercase form.
* **proxy:** run a cold-start fast pass before background-compression deferral so byte-identical freeze doesn't lock sessions to the uncompressed transcript. Since #1850, a session's provider-cached prefix is frozen in whatever form its cold start forwarded; deferring the WHOLE pipeline (`HEADROOM_BACKGROUND_COMPRESSION=1`, frozen=0, ≥50k tokens) therefore cached the raw transcript and forfeited the session's compression savings for its lifetime — including sub-second lossless wins like `read_lifecycle` stale-read drops, observed in the field as sessions permanently stuck at 0 savings. The deferral branch now runs the pipeline synchronously with the new `skip_kompress=True` kwarg (everything except the Kompress ML stage — the only stage that can blow the request budget per #1171) under a bounded fast-pass budget (`HEADROOM_COLD_START_FAST_PASS_TIMEOUT_SECONDS`, default 10s), forwards the pruned form, and defers only Kompress to the background job (tagged `deferred:kompress_background`). Fail-open: on fast-pass timeout/error the request forwards uncompressed exactly as before. Units routed to Kompress under `skip_kompress` take the same fallback as when the model isn't ready.
* **proxy/batch:** preserve sibling tool configs on Google batch requests. When optimizing a Gemini batch item, the handler rebuilt the request's `tools` as a single `[{"functionDeclarations": ...}]` entry, discarding any other entries in the array (`googleSearch`, `codeExecution`, etc.). A batch request that combined function calling with Google Search or code execution therefore reached Google with those siblings stripped, silently disabling them. The handler now replaces only the `functionDeclarations` entry (appending one when the original had none) and keeps every sibling entry intact.
* **savings:** record the pre-compression original as the ledger `before`, not the forwarded count. In `PrometheusMetrics.record_request` the durable savings ledger was written with `tokens_before=input_tokens`, but `input_tokens` there is the optimized (post-compression) count that was forwarded (`emit_request_outcome` passes `outcome.optimized_tokens`). `headroom savings` derives the reported reduction percent as saved / before, so understating `before` by `tokens_saved` inflated it — a real 40% reduction (1000 → 600) was reported as ~67% (400 / 600). The event now records `tokens_before=input_tokens + tokens_saved` (the reconstructed original) and `tokens_after=input_tokens` (the forwarded count); the `saved` and cost figures are unchanged.
* **proxy/gemini:** forward a non-JSON upstream error body with its real status instead of a synthetic 502. In `handle_gemini_generate_content` the token-extraction `except (KeyError, TypeError, AttributeError)` guarding `response.json()` omitted `json.JSONDecodeError` / `ValueError`, so a non-JSON body (an HTML/empty error page from an overloaded Google/Vertex/Copilot frontend, common on 5xx/429) escaped to the outer `except Exception` and was returned as a generic 502 — discarding the real upstream `status_code` and body and defeating the client's retry/backoff. The except now catches the JSON/ValueError family, matching the all-non-text sibling branch, so the true status and body are forwarded verbatim.
* **init/codex:** don't overwrite the user's `hooks.json`. `_ensure_codex_hooks` wrote a fresh payload containing only Headroom's two hooks, wholesale-replacing `~/.codex/hooks.json` — so any user-managed Codex hooks (and other top-level keys) were silently destroyed on `headroom init codex`. It now read-merges: existing entries are preserved, Headroom's are deduped on the `headroom-init-codex` marker and appended, matching `_ensure_claude_hooks` / `_ensure_copilot_hooks`.
* **cli/init:** fail with an actionable error when a target's settings file contains invalid JSON. `_json_file` (used to read-merge-write Claude's `settings.json`, Codex's `hooks.json`, etc.) called `json.loads` unguarded, so a hand-edit typo (e.g. a trailing comma) crashed `headroom init` with a raw `JSONDecodeError` traceback. It now raises a `ClickException` naming the file and the parse error and telling the user to fix it or move it aside, without touching the file (returning `{}` would have made the follow-up write overwrite the user's settings).
* **ccr:** detect `read_lifecycle` stale/superseded markers in the retrieve-tool injector so they stay redeemable. Those markers (`[Read content stale: … Retrieve original: hash=<hash>]`) store the original bytes in the CCR store under a valid hash, but none of `CCRToolInjector`'s patterns matched them — every pattern required the word "compressed" or the `<<ccr:` form. So on a frozen-prefix turn (both `read_lifecycle` and prefix freezing are on by default) the injector reported no compressed content, the `headroom_retrieve` tool was not injected, and the model was handed a marker advertising `Retrieve original: hash=X` with no tool to redeem it — silent data loss for stale reads, where retrieval is the only way to recover the original-at-read-time content (the exact case the #1006 guard exists to prevent). Added a pattern matching the load-bearing `Retrieve original: hash=` phrase, aligning the injector with the sibling `read_maturation` marker that was already (incidentally) detected.
* **install:** don't crash `headroom install status` when the health payload's `config` is a non-dict. The command did `payload.get('config', {}).get('backend', manifest.backend)`, but `dict.get(..., {})` only defaults on a *missing* key — a present-but-non-dict `config` (`null`, a string, a list, e.g. when a different or older service is answering on the port) reached the chained `.get('backend', ...)` and raised `AttributeError`, crashing the command with a raw traceback. The value is now guarded with `isinstance(config, dict)` before the lookup, mirroring `wrap.py`'s `_proxy_health_config`, so it falls back to the manifest's backend.
* **telemetry:** only advance the usage-report baseline after a confirmed 200. `UsageReporter._report_usage` sends usage as a delta against the last snapshot, but it called `_snapshot_metrics()` (and advanced `_last_report_time`) unconditionally after the POST — including when the send returned non-200 or raised. So a report that failed to reach the cloud (which the module is explicitly designed to tolerate) still rebased the baseline, permanently dropping that window's requests/tokens from usage-based billing/quota; the next report started from the advanced baseline and never re-included them. The baseline now advances only on a 200, so a failed send leaves the window intact for the next report to retry.
* **savings:** stop the durable savings ledger from billing free (0-priced) models at the `$3/M` fallback. `estimate_cost_usd` guarded the litellm estimate with `if priced > 0`, so a genuinely free model — where `_estimate_compression_savings_usd` correctly returns `0.0` — was treated as "unpriced" and fell through to the blended fallback rate, writing phantom cost-avoided into the JSONL ledger and surfacing it in `headroom savings`. The ledger now trusts the estimate verbatim for known models (it already falls back internally for models litellm can't price and returns `0.0` for free ones), fixing the same defect at this call site that was already fixed inside the helper.
* **init/codex:** stop `headroom init codex` from deleting per-profile provider settings. `_ensure_codex_provider` removed the root-level `model_provider`/`openai_base_url` (which init owns) with a multiline regex that matched those keys in **every** table, so a user's `[profiles.*]` overrides (e.g. `[profiles.work] model_provider = "azure"`) were silently stripped and those profiles fell through to the injected `"headroom"` default — config corruption. The strip is now scoped to the document root (everything before the first table header), so per-profile overrides are preserved while init still replaces a root-level assignment.
* **proxy:** strip output-only content blocks from request messages before forwarding. Anthropic's server-side refusal-fallback feature (`server-side-fallback-2026-06-01`) emits a `{"type":"fallback","from":{...},"to":{...}}` block inside the assistant response to signal that a refused request was re-served by the fallback model. That block is valid on the *response* path but rejected on the *request* path, so when a client replays the assistant turn the next request 400s (`invalid_request_error: messages.N.content.0: Input tag 'fallback' ...`) and the conversation gets permanently stuck through the proxy. `read_request_json_with_bytes` (Anthropic/OpenAI/Bedrock) and `_read_request_json` (Gemini) now drop such blocks — re-encoding the raw bytes so byte-faithful passthrough cannot leak the pre-strip body, backfilling a benign text block if a turn is emptied, and leaving requests without such blocks byte-identical (no cache churn).
* **wrap/doctor:** make the Claude Remote Control gate warning accurate and stop it firing for users who never had the feature ([#1779](https://github.com/headroomlabs-ai/headroom/issues/1779)). Claude Code 2.1.196 added a client-side check that **deterministically** disables first-party Remote Control (`/remote-control` / `/rc`) whenever `ANTHROPIC_BASE_URL` points at a non-`api.anthropic.com` host — which Headroom always does. The old notice hedged ("may hide the Remote Control menu"); it now states the disable as fact, names the `/rc` command, and detects the installed Claude Code version so the wording is exact (`2.1.196` when known, `2.1.196+` when not). The gate is upstream and RC's control-plane talks to `claude.ai` (not the API host), so Headroom cannot restore it — the warning tells you to run Claude without Headroom for RC sessions. The warning is suppressed for auth modes that never had Remote Control (API-key/PAYG via `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`, and Bedrock/Vertex/Foundry cloud IAM) and on Claude Code builds older than 2.1.196 where RC is unaffected by a custom base URL. Both the `headroom wrap claude` launch banner and `headroom doctor` co-report the sibling base-URL gates Headroom *does* restore — on-demand tool loading (#746, automatic) and the 1M context window (#1158, via `--1m`) — and the wrap-side co-report is session-accurate: it says "already restored via --1m" when the flag is in effect and reports tool deferral OFF (not falsely "kept on") when the user chose `--tool-search false`/`ENABLE_TOOL_SEARCH=false`; the `ENABLE_TOOL_SEARCH=...` banner line got the same accuracy fix. `is_custom_anthropic_base_url` now recognizes scheme-less values (`myproxy.local:8080`, `127.0.0.1:8787`) as custom hosts and degrades gracefully on malformed URLs instead of crashing `doctor`. `doctor` resolves the Claude Code version lazily, so runs with no custom base URL never pay the `claude --version` subprocess. No request bytes are touched (cache-safe); this is UX/notice-only.
* **install:** default the docker image to `ghcr.io/headroomlabs-ai/headroom:latest` instead of the dead `ghcr.io/chopratejas/headroom:latest`. After the repo moved to the `headroomlabs-ai` org, GHCR did not redirect the old package, so `headroom install` / `headroom init` and the install scripts pulled a frozen `0.27.0` image while current releases publish to the new path ([#1867](https://github.com/headroomlabs-ai/headroom/issues/1867)).
* **transforms/content-router:** stop a profile-derived `read_protection_window` kwarg from weakening an explicit `--protect-tool-results` guarantee. `ContentRouter.apply()` computes `read_protection_window` from `protect_recent_reads_fraction`, where `0.0` (the sentinel `--protect-tool-results` sets) means "protect all excluded-tool output regardless of conversation depth" per #1374's documented contract — but the method then unconditionally overwrote that window with a `read_protection_window` kwarg whenever one was present. `proxy_pipeline_kwargs()` supplies that kwarg on every request from the active `AgentSavingsProfile.protect_recent` (the default `coding` profile sets `protect_recent=2`), so in practice only the last 2 messages ever kept read-protection and older excluded-tool output silently fell through to lossy compression. The runtime kwarg may now only narrow the window when `protect_recent_reads_fraction > 0`; it can no longer shrink the "protect everything" guarantee set by `--protect-tool-results`.
* **memory:** drop a superseded memory from the search indexes so it stops resurfacing. `supersede` set the old memory's `valid_until` in the store and indexed the new version, but never touched the old entry in the vector/text index. Those indexes keep a cached metadata copy (captured at index time with `valid_until=None`), and default search filters superseded rows off that cached copy — so the superseded, outdated version kept coming back from semantic/text search alongside the new one, injecting contradictory facts into recall. `supersede` now removes the old id from the vector and text indexes (mirroring `delete`); the store still keeps the row for `get_history`.
* **proxy:** don't let a stray `HEADROOM_QDRANT_PORT` crash proxy startup. `ProxyConfig.memory_qdrant_port` used `qdrant_env.qdrant_env_port` as its field `default_factory`, and that function raises `ValueError` on a non-integer or out-of-range value. Because a `default_factory` runs on **every** `ProxyConfig()` construction, an inherited or typo'd `HEADROOM_QDRANT_PORT` crashed the proxy before it served a request — even though memory (and the qdrant backend) are off by default and unrelated to core proxying. The field now resolves the port through a fail-soft wrapper that falls back to the default (6333) with a warning; the strict `qdrant_env_port()` is unchanged for explicit qdrant setup.
* **memory:** size the HNSW `index_batch` resize off the assigned-id high-water mark, not the live entry count. hnswlib never reclaims a slot on `mark_deleted` (used by remove/evict), so its usable capacity is bounded by the number of ids ever assigned (`_next_hnsw_id`). `index_batch` computed `required_capacity = len(self._memory_to_hnsw) + len(new_memories)` — the *live* count — which drops below `_next_hnsw_id` after deletion/eviction churn, so the resize was skipped and `add_items` raised `RuntimeError: number of elements exceeds the specified limit`, crashing the save path on the HNSW backend. It now resizes off `_next_hnsw_id`, matching the single-item `index()` guard.
* **memory:** apply the `turn_id` scope filter even when `agent_id` is absent. In `SQLiteMemoryStore._build_query_conditions` the `turn_id` condition was nested inside the `agent_id` block, so a query filtered by `user_id` + `session_id` + `turn_id` (no `agent_id`) dropped the `turn_id` predicate entirely and returned every memory in the session instead of the single turn — an over-broad result that leaks sibling-turn memories into recall (and makes `count()` wrong for that scope). `agent_id` and `turn_id` are now applied independently.
* **transforms:** stop the lossless `diff` fold from silently dropping lines out of non-diff content. `ContentRouter._lossless_first` tries every `compact_lossless` fold on all content, but the `diff` kind (`diff_strip_index`) is the only one with no exact-inverse check — it removes any line shaped like `index <hex>..<hex>`. Applied to arbitrary text/log/search payloads that happen to contain such a line, that line was deleted with no CCR marker, so it was unrecoverable — a violation of the lossless no-loss contract the method's own docstring promises. The `diff` fold now runs only when the strategy is `DIFF` or the content is diff-shaped (`_looks_like_diff`); genuine diffs still have their `index` bookkeeping folded.
* **proxy:** reject a 0 `rate_limit_requests_per_minute` when rate limiting is enabled, instead of 500-ing every request. The token-bucket wait computation divides by the per-minute rate (`consume_from_bucket`), so a `rate_limit_requests_per_minute` of 0 raised `ZeroDivisionError` on every request that hit the limiter. The CLI already guards this with `IntRange(min=1)`, but the `HEADROOM_PROXY_CONFIG_JSON` / programmatic config paths bypassed it. `ProxyConfig.__post_init__` now validates `rate_limit_requests_per_minute >= 1` when `rate_limit_enabled` (mirroring the existing `retry_max_attempts` check), so a bad value fails fast at construction with a clear message; it stays inert when limiting is disabled.
* **memory:** include the Ollama server URL in the embedder cache key so a second backend can't get an embedder bound to the wrong server. `_create_embedder` cached by `(backend, model)` only, but the Ollama embedder is constructed with `base_url=config.ollama_base_url`. Two configs in the same process that shared a backend and model but pointed at different Ollama servers (e.g. a per-project storage router) collided on one cache slot, so the second silently reused the first's embedder and embedded against the wrong host. The cache key now also includes `ollama_base_url`.
* **tokenizers:** use `o200k_base` for the gpt-4.1 / gpt-4.5 / o4 families in `get_encoding_for_model`. `gpt-4.1*` and `gpt-4.5*` matched the broad `gpt-4` prefix and were encoded with `cl100k_base`, and `o4*` matched no prefix and fell through to the `cl100k_base` default — all three use `o200k_base`, so their token counts were computed with the wrong vocabulary. Added explicit `gpt-4.1`/`gpt-4.5` prefixes ahead of `gpt-4` and an `o4` prefix; `gpt-4` and `gpt-3.5` snapshots still resolve to `cl100k_base`.
* **cache/ccr:** stop counting a successful eviction as a retrieval in the compression feedback learner, which inverted the learning signal. When an entry is evicted without ever being retrieved, `CompressionStore` emits a synthetic `retrieval_type="eviction_success"` event to mark that the compression was sufficient (the LLM never needed the original). `CompressionFeedback.record_retrieval` had no branch for it, so — because the type is not `"full"` — it was counted as a *search retrieval*, inflating the tool's `retrieval_rate`/`search_rate`. `get_compression_hints` reads a high retrieval rate as "compressing too aggressively" and backs off, so a compression that actually worked pushed the learner toward *less* compression (a standalone repro scores one successful eviction as a 100% retrieval rate). The event is now recognized and left out of the retrieval counters; the compression is still counted by `record_compression` at store time, so a never-retrieved entry correctly yields a low retrieval rate. Genuine retrievals are unaffected.
* **proxy/anthropic:** don't launder a non-2xx upstream into HTTP 200 when enterprise security scans the response. On the non-streaming `/v1/messages` path the response-scan branch rebuilt the reply as `httpx.Response(status_code=200)` and returned it without checking the upstream status, so a rate-limit (429), overloaded (529), or other 4xx error whose JSON body was scanned reached the client as an HTTP 200 — the client's retry/backoff never fired and an error looked like success. The branch is now gated on a 200 upstream, matching the sibling CCR/cache/buffered-stream blocks in the same handler; non-2xx responses fall through and keep their real status.
* **learn:** classify timeout and connection tool failures correctly instead of as generic runtime errors. In `classify_error` the generic `RUNTIME_ERROR` pattern (`Traceback|Exception:|Error:`) was checked before the dedicated `TIMEOUT` and `CONNECTION_ERROR` patterns. Because every Python exception repr is `XxxError: ...`, a `TimeoutError: ...` or `ConnectionError: ...` matched the catch-all first and was miscategorized as `RUNTIME_ERROR`, leaving those two categories unreachable for the common colon-repr form (they only fired for tokenless phrasings like `deadline exceeded`). The `TIMEOUT` and `CONNECTION_ERROR` patterns are now checked before the generic catch-all; tokenless generic errors still classify as `RUNTIME_ERROR`.
* **tokenizers:** resolve HuggingFace tokenizer names by the most-specific prefix. `get_tokenizer_name` scanned `MODEL_TO_TOKENIZER` in dict-insertion order and returned the first key the model merely starts with, so a short family key shadowed a more-specific one — `qwen2-7b-instruct` matched `qwen` before `qwen2`/`qwen2-7b` and resolved to the Qwen1 tokenizer (a different vocabulary, hence wrong token counts); `qwen2.5-*` and `deepseek-v2.x` were mis-resolved the same way. It now picks the longest matching prefix, mirroring the order-dependent-prefix guard the sibling tiktoken `get_encoding_for_model` already documents.
* **pricing:** map retired `claude-3-sonnet-20240229` to a Sonnet-tier price instead of Haiku. When LiteLLM's cost DB lacks the retired model, resolution falls through to `MODEL_ALIASES`, which pointed Claude 3 Sonnet (a $3/$15-per-1M model) at `claude-3-haiku-20240307` ($0.25/$1.25) — a different tier that underpriced every cost/savings figure for that model ~12x on both input and output. It now aliases to `claude-sonnet-4-20250514`, the same-price target the other retired-Sonnet aliases already use.
* **cache/semantic:** don't evict an unrelated entry when re-storing a key that is already cached. `SemanticCache.put` ran its at-capacity eviction loop before computing the entry's key, so overwriting a key that was already present (a duplicate or retried store) still evicted the LRU-oldest distinct entry even though an in-place update grows nothing. That silently dropped a live entry and turned a later lookup for it into a false cache miss. The key is now computed first and the eviction loop only runs when the key is genuinely new (mirroring `CompressionCache.store_compressed`, which deletes-then-inserts).
* **learn:** keep `verbosity._ordered_events` in lockstep with `_parse_session` on empty assistant turns. `_parse_session` creates a `_Response` only when an assistant message has content (`words > 0 or out_tok > 0`), but `_ordered_events` consumed a response slot for *every* assistant line. An empty assistant turn (e.g. a pure `tool_use` turn with no usage) therefore shifted the `responses` list out of alignment, so a later human reply was paired with a future-timestamped response, the read gap went negative, and a spurious `fast_skip` was recorded — inflating `fast_skip_rate` and skewing the recommended verbosity level. `_ordered_events` now applies the same `words > 0 or out_tok > 0` guard before consuming a response, matching the filter the user side already mirrors.
* **wrap/claude:** don't raise `UnboundLocalError` in the cleanup `finally` when the proxy fails to start. `claude()` referenced `_wrap_settings_path` in its `finally` block but only assigned it inside the `try`, after `_ensure_proxy` (which raises on port exhaustion or a failed proxy start). An early failure therefore made the `finally` raise `UnboundLocalError` — replacing the real error with a raw traceback and, because the `finally` aborted before `cleanup()`, skipping proxy cleanup and wrap-marker clearing. `_wrap_settings_path` is now bound before the `try` alongside the other cleanup holders (`proxy_holder`, `_saved_base_url`, …), so the `finally` is always safe.
* **wrap/codex:** export the detected custom upstream base URL so Codex actually routes to it. `_inject_codex_provider_config` detects an OpenAI-compatible gateway declared in `~/.codex/config.toml` and writes an `X-Headroom-Base-Url` header mapped to `HEADROOM_CODEX_UPSTREAM_BASE_URL`, and it returns that URL for the caller to export. But `_prepare_codex_wrap_state` discarded the return and `_run_codex_wrap` never set the env var, so Codex emitted no header, the proxy fell back to `api.openai.com`, and the user's gateway key was sent to OpenAI (which rejects it). This restores the wiring a later refactor dropped: `_prepare_codex_wrap_state` now returns the URL and `_run_codex_wrap` exports it into the launch env when set (a user-provided value still wins) (regression of [#1614](https://github.com/chopratejas/headroom/issues/1614)).
* **cache:** normalize embeddings before the semantic dynamic-content similarity check. `SemanticDetector` scored sentences with `np.dot` against exemplar embeddings and compared the result to `semantic_threshold` (a 0-1 cosine value), but `sentence_transformers.encode(..., convert_to_numpy=True)` does not normalize, so the dot product was an unbounded inner product (vector norms ~5-15) rather than a cosine similarity. Nearly every sentence cleared the 0.7 threshold, so the semantic tier flagged almost all text as dynamic and stripped static content, busting the cache it is meant to protect (a standalone repro scores an unrelated sentence, true cosine ~0.1, at a raw dot of ~9). Both `encode` calls now pass `normalize_embeddings=True`, matching the siblings in `prediction/feature_extractor.py` and `memory/adapters/embedders.py`, so the dot product is a true cosine in [-1, 1].
* **subscription:** cap `HeadroomContribution.efficiency_pct` at a real removal ratio so it can't exceed 100%. The numerator used `total_saved()` (which includes `tokens_saved_cache_reads`) while the denominator `raw_without_headroom()` excludes cache reads, so a contribution with large prefix-cache reads and small forwarded input reported impossible values (e.g. `tokens_submitted=100`, `tokens_saved_cache_reads=1000` → `1000.0%` on the dashboard). Cache reads are a provider-side discount on tokens that were still forwarded, not tokens Headroom removed, so the ratio now uses the sibling `compression_saved()` (compression + CLI filtering) as the numerator — bounded by its own denominator. `total_saved()` is unchanged for its other callers.
* **proxy/anthropic:** cache the response under the same messages it was looked up by. The non-streaming `/v1/messages` path snapshots the scalar cache-key fields (system, tools, etc.) once before upstream to avoid post-mutation key drift (#327), but `messages` — the primary key component — was passed live at both `cache.get` and `cache.set`. Between the two, `messages` is reassigned by the enterprise security scan, the `pre_compress` hook, and image compression, so when any of those fired the response was stored under a different key than it was read by: the response cache never hit and accumulated unreachable entries until eviction. The lookup messages are now snapshotted alongside the other key fields and reused verbatim at `cache.set`.
* **tokenizers:** stop `TiktokenCounter.count_messages` from exploding on non-text content blocks. Its multi-part branch handled only `text` and OpenAI `image_url`; every other shape (Anthropic `image`/`tool_result`/`tool_use`, Strands blocks) fell through to `count_text(str(part))`, which json-stringified the base64 payload and tokenized it as text — a 1MB image counted as ~330K phantom tokens (~218x overcount in a standalone repro), corrupting every downstream budgeting/compression decision for multimodal OpenAI-model requests. Unknown block shapes now delegate to the base `_count_content_parts`, which prices images/documents by a bounded estimate (the overcount that helper already exists to prevent).
* **install:** don't let a host env export override the manifest in persistent-docker deployments. `build_runtime_command` emitted the manifest's pinned `--env NAME=VALUE` pairs and then, for every host var matching a passthrough prefix, a bare `--env NAME`. Docker resolves duplicate `--env` last-wins, so a stale host export (e.g. `HEADROOM_BACKEND=anyllm`) that shared a passthrough prefix with a pinned manifest value (`HEADROOM_BACKEND=anthropic`) was appended after it and silently won, diverging the container from its deployment config. The bare passthrough is now skipped for any name the manifest already pins.
* **memory:** honor explicit `store=false` on OpenAI `/v1/responses` requests by skipping Headroom memory-tool injection that depends on stored-response continuations. Memory context injection stays available, and requests no longer get rewritten to `store=true` behind the client's back ([#1944](https://github.com/headroomlabs-ai/headroom/issues/1944)).
* **proxy/batch:** stop corrupting Google `batchGenerateContent` requests whose contents interleave text turns with text-less entries (functionCall/functionResponse/images). The batch handler restored preserved (non-text) entries with the raw-index loop that #836 replaced everywhere else — indexing the shorter `optimized_contents` (text-less entries produce no message) by the original `contents[]` index, which overwrites the wrong entry and drops any preserved entry whose original index is past the optimized length. A request like `[user text, model functionCall, user functionResponse, model text]` was forwarded to Google as two entries: the model's answer overwritten by the functionCall and the functionResponse dropped. The batch handler now uses the shared `_rebuild_gemini_contents` interleaving helper, so all entries survive in order.
* **proxy/gemini:** preserve Gemini code-execution parts (`executableCode` / `codeExecutionResult`) across the compression round-trip. `_has_non_text_parts` only recognized `inlineData`/`fileData`/`functionCall`/`functionResponse`, so a content entry carrying code-execution parts was not marked as preserved. A mixed `text`+`executableCode` entry lost its code payload (only the text survived), and a text-less code-execution entry was treated as a phantom that dropped the entire turn and shifted a neighboring message into the wrong role slot. Both keys are now recognized so those entries are preserved verbatim.
* **cache/ccr:** don't evict a live entry when a duplicate hash is re-stored at capacity. `CompressionStore.store` ran `_evict_if_needed()` before checking whether the key already existed, so re-storing an already-present hash while the store was full evicted the oldest *distinct* entry to "make room" and then merely overwrote the existing key in place — no room was ever needed. The store dropped below `max_entries` and a live, never-retrieved entry was destroyed, so its `<<ccr:...>>` marker (still in the conversation) resolved to a 404. The CCR mirror bridge re-stores the same `explicit_hash` every turn a marker is re-encountered, so this fired routinely. Eviction now runs only for a genuinely new key.
* **tokenizers:** recurse into a native `tool_result` whose content is a list of blocks instead of JSON-serializing it. `_count_content_parts` counted a `tool_result` with list content via `_count_serialized` (json.dumps + sample), so a base64 image nested in a tool result (computer-use / MCP screenshot tools) was priced as text — a ~50-200x overcount (a ~200KB screenshot read as ~70K tokens instead of ~1600). It now recurses into the nested blocks, matching the sibling Strands `toolResult` branch, so the image is priced structurally. The overcount made a single screenshot appear to blow past the model's context window and triggered unnecessary/over-aggressive compression.
* **tokenizers:** price dense scripts (CJK/Kana/Hangul) in the fixed-ratio estimator path. `EstimatingTokenCounter.count_text` applied the CJK correction only on the auto-detect path; its fixed-ratio early return divided by the Latin ratio with no adjustment. The registry builds every provider-calibrated counter with a fixed ratio (Anthropic 3.5, Google 4.0, Cohere 4.0, Moonshot 3.1), and the Anthropic/Gemini proxy handlers count via `get_tokenizer(model).count_messages`, so a CJK-heavy context read as ~40-55% of its true token size — it could fall under the size/backpressure gates and skip compression, and every `x-headroom-tokens-before` metric for CJK traffic was materially wrong. The fixed-ratio path now applies the same dense-script split.
* **ccr:** don't crash `parse_tool_call` on a CCR tool call whose arguments aren't an object. For the OpenAI/`openai_responses` shape the arguments are `json.loads`-decoded and only `JSONDecodeError` was caught, so a model that emitted `arguments='[]'`/`'"abc"'`/`'123'` (decoding to a list/str/number) — or a non-dict Anthropic `input` — reached `input_data.get("hash")` and raised `AttributeError`; a null `arguments` raised an uncaught `TypeError` from `json.loads(None)`. Both are now handled: the decode also catches `TypeError`, and a non-dict `input_data` returns `None` (not a valid CCR call) instead of crashing CCR response processing.
* **proxy/memory:** capture the user's prompt from Anthropic text blocks when building the memory-retrieval query. `extract_memory_query_sources` only recorded `latest_user` when a user message's `content` was a plain string; for the standard Anthropic `/v1/messages` shape (`content=[{"type":"text","text":...}]`, used by Claude Code) it routed into the tool-result extractor and never read the `text` blocks — so the actual question was discarded. On a first turn the embedding query was then empty and memory injection was silently skipped entirely; with history it keyed on stale assistant/tool context instead of the user's ask. The user turn's `text` blocks are now captured.
* **memory/sqlite:** stop `SQLiteMemoryStore.query` from emitting `OFFSET` without a `LIMIT`. The query builder appended `LIMIT` only when `filter.limit is not None` and `OFFSET` independently when `filter.offset > 0`, but SQLite accepts `OFFSET` only as part of a `LIMIT` clause — so a `MemoryFilter(offset=N)` with no limit produced `... OFFSET ?` and crashed with `sqlite3.OperationalError: near "OFFSET": syntax error`. An offset-without-limit now emits SQLite's unbounded `LIMIT -1` so pagination works.
* **mcp/codex:** don't corrupt an unparseable or non-table `config.toml` on register. `CodexRegistrar.register_server` only guarded against clobbering a user-managed entry when `get_server` returned one, but `get_server` returns `None` both for an unparseable TOML file and for an `mcp_servers`/`mcp_servers.<name>` that is present but not a table. In those cases `register_server` fell through to `_write_block`, which blindly appended a `[mcp_servers.<name>]` table — appending into an unparseable file, or creating a duplicate `[mcp_servers.headroom]` key alongside a non-table entry (e.g. `headroom = "..."`), which `tomllib`/codex then reject, destroying a previously-valid config. It now refuses (`FAILED`) and leaves the file untouched, mirroring the claude (#1660) and opencode (#1661) guards.
* **proxy/vertex:** route Vertex `publisher=google` (Gemini) requests to the region matching the request path. `vertex_generate_content`, `vertex_stream_generate_content`, and `vertex_count_tokens` discarded the path's `location` and forwarded to the single fixed host from `_api_target(proxy, "vertex")` (default `us-central1`), instead of the region-aware `_vertex_target_for_location` the sibling Anthropic `rawPredict` route already uses. So a request to `.../locations/europe-west1/publishers/google/...` was sent to a `us-central1` host, which Vertex rejects on the region/host mismatch. The three google routes now derive the host from the request's `location` (operator-pinned upstreams are still honored).
* **proxy/anthropic:** give each Anthropic conversation its own session id. `SessionTrackerStore.compute_session_id` derived its fallback id from `model` + system text harvested only from `role:"system"` entries inside `messages` — but Anthropic carries the system prompt as a top-level `body["system"]` field, so genuine Anthropic requests (which never carry `x-headroom-session-id`) collapsed to `md5(model:[])` and every conversation on the same model shared one `PrefixCacheTracker`. That let session-sticky state cross-contaminate: conversation A's sticky `headroom_retrieve`/memory tools and `anthropic-beta` headers were injected into conversation B, and frozen-prefix/compression-cache state mixed across conversations. The Anthropic handler now folds the top-level `system` into the session-id inputs (prepending a synthetic `role:"system"` message used only to derive the id), giving distinct conversations distinct ids.
* **install/codex:** persistent provider routing is now lifecycle-coupled. `headroom install apply` waits until the runtime is ready before writing managed Codex routing, and stop/remove/recovery paths revert that routing before tearing the proxy down, so Codex Desktop is not left pointed at a dead `127.0.0.1:8787` provider after a failed or stopped deployment ([#2038](https://github.com/headroomlabs-ai/headroom/issues/2038)).
* **cache/semantic:** key entries by the full-context hash, not the trailing query text. `SemanticCache.put` stored each response under `sha256(query)[:16]` where `query` is only the last user message, and the exact-match branch of `get` returned the slot without checking the stored entry's `messages_hash`. Two requests that share a trailing message ("continue", "yes", "run the tests") but differ in earlier context therefore collided on one slot — the second overwrote the first, and the first's hash then resolved to the second's cached response (wrong data served). Entries are now keyed by `messages_hash` when present, and `get` verifies `entry.messages_hash` before returning.
* **wrap/opencode:** `headroom unwrap opencode` now removes the Headroom rtk instruction block from the project and global `AGENTS.md`. `wrap opencode` injects the marker-fenced "prefix shell commands with `rtk`" guidance into both `./AGENTS.md` and `<opencode-home>/AGENTS.md`, but unwrap only restored the config and MCP state — so a plain `opencode` launch kept following the rtk guidance and failed once the managed rtk binary was off PATH. Unwrap now strips the marker-fenced block from both files, mirroring `unwrap codex` (#1421) and `unwrap copilot`.
* **proxy/savings:** stop billing the $3/M fallback rate for genuinely free models. `_estimate_compression_savings_usd` and `_estimate_input_cost_usd` read `input_cost_per_token` from litellm and used `if not input_cost_per_token: raise`, which treats a legitimate `0.0` (a free / local / vendored-at-0 model that litellm does carry) as "price unavailable" and falls back to `DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN` — fabricating dollar savings/cost for a model that costs nothing. Both now use an explicit `is None` check so a present `0.0` flows through as `$0` while a missing key still falls back.
* **proxy/cost:** value prefix-cache savings with the most-used model's price, not the first-recorded one. `build_prefix_cache_stats` scanned `cost_tracker._tokens_sent_by_model` and broke on the *first* provider-matching model with a price — despite the "most-used model" comment — so a Claude Code session (Sonnet for the main loop, Haiku for titles/subagents) priced all of a provider's cache-read savings at whichever model happened to be recorded first. If Haiku ($0.80/M) came before Sonnet ($3/M), the dashboard understated cache savings ~3.75x (and vice-versa). It now picks the provider-matching, priced model with the highest token volume.
* **proxy/openai:** stop overriding an explicit client `stream_options.include_usage` on the streaming chat path. To count tokens from the trailing usage chunk, the handler set `include_usage: True` unconditionally — including flipping an explicit client `false` to `true`. The upstream then appended a usage-only chunk (`choices: []`) the client never requested, and the common `chunk.choices[0].delta` loop raised `IndexError`. The option is now only filled in when the client left the choice open (no `stream_options`, or a dict without `include_usage`); an explicit `true`/`false` is respected.
* **proxy/openai:** stop PRE_SEND from reintroducing `tools: []` after the direct #728 fix. The OpenAI request handler now mirrors the existing `tools or _original_tools is not None` body-write guard during PRE_SEND write-back, so providers that reject empty tool arrays no longer see a tools field when the client omitted it, while explicit client `tools: []` remains preserved ([#1983](https://github.com/headroomlabs-ai/headroom/issues/1983)).
* **proxy/openai:** keep the exact Responses function name `terminal` resident during OpenAI tool-search deferral so cache-mode optimization stops forwarding `terminal.terminal` and triggering the reserved-namespace 400 on Codex Responses ([#1946](https://github.com/headroomlabs-ai/headroom/issues/1946)).
* **proxy/transforms:** startup warmup no longer calls Kompress native preload before the proxy binds its port. Enabled Kompress is deferred to first use, unavailable Kompress stays reported unavailable, and cached-model startup avoids the native ONNX load path that crashed older glibc hosts ([#1908](https://github.com/headroomlabs-ai/headroom/issues/1908)).
* **subscription/copilot:** show a fully-consumed Copilot quota as 100% used instead of unknown. `parse_copilot_quota` read `remaining = raw.get("remaining") or raw.get("quota_remaining")`, so a category reporting `remaining: 0` (quota fully spent) had that legitimate `0` treated as falsy and — with no `quota_remaining` alias in the real payload — collapsed to `None`. `CopilotQuotaCategory.used`/`used_percent` then returned `None`, so the dashboard rendered the exhausted category as `used: -` / 0% (green gauge) rather than `300/300` / 100%. Now uses an explicit `is None` check.
* **proxy/gemini:** thread the savings-profile kwargs into the native Gemini/Vertex compression paths. `handle_gemini_generate_content`, `handle_google_cloudcode_stream`, and `handle_gemini_count_tokens` called `openai_pipeline.apply()` without `proxy_pipeline_kwargs(self.config)`, so `HEADROOM_SAVINGS_PROFILE` and the ProxyConfig knobs (`target_ratio`/`min_tokens_to_compress`/`protect_recent`/`max_items_after_crush`/...) were silently dropped on the Gemini path — those requests compressed with router defaults instead of the configured profile, diverging from the Claude/Codex/Cursor paths. This is the same fix #1534 made for the OpenAI chat path; it now covers Gemini too.
* **wrap:** `headroom wrap claude` no longer installs RTK or lean-ctx by default. Claude context-tool setup is now explicit via `--context-tool`, `--no-context-tool` remains accepted, and other wrap commands keep their current defaults ([#1915](https://github.com/headroomlabs-ai/headroom/issues/1915)).
* **proxy/openai:** thread the savings-profile kwargs into the live `/v1/chat/completions` compression path. The chat handler called `openai_pipeline.apply()` without `proxy_pipeline_kwargs(config)`, so `HEADROOM_SAVINGS_PROFILE=agent-90` (and the individual `compress_user_messages`/`target_ratio`/`min_tokens_to_compress`/... knobs) were silently dropped — OpenAI-compatible clients like OpenCode kept protecting user messages and missed the configured profile. Both the token-mode and non-token chat branches now pass the profile kwargs, matching `handlers/anthropic.py` and the dedicated OpenAI compress endpoint ([#1534](https://github.com/headroomlabs-ai/headroom/issues/1534)).
* **proxy:** forward Codex Desktop `/v1/responses` posts byte-faithfully so they stop returning upstream `400 {"detail":"Bad Request"}`. `handle_openai_responses` decoded the inbound body to inspect it but always re-serialized a canonical body on the way out, and it never stripped the inbound `content-encoding` header — so a `content-encoding: zstd` Codex Desktop request was forwarded as already-decoded JSON still advertising `zstd`, and the upstream ChatGPT Codex endpoint rejected it. The handler now keeps the original decoded bytes and forwards them verbatim whenever nothing (compression or memory injection) mutated the request, and drops the stale `content-encoding` header, mirroring the byte-faithful passthrough the chat and Anthropic paths already use ([#1542](https://github.com/headroomlabs-ai/headroom/issues/1542)).
* **wrap/codex:** `headroom unwrap codex` now removes the Headroom rtk instruction block from the Codex global `AGENTS.md`. `wrap codex` injects it there, but unwrap only restored `config.toml` and MCP state, so a plain `codex` launch kept following the "prefix shell commands with `rtk`" guidance and failed once the managed rtk binary was off PATH. Unwrap now strips the marker-fenced block (preserving the rest of the file), mirroring `unwrap copilot` ([#1421](https://github.com/headroomlabs-ai/headroom/issues/1421)).
* **proxy/auth:** classify real Anthropic OAuth tokens correctly. `classify_auth_mode` matched OAuth on the `sk-ant-oat-` prefix, but real access tokens are `sk-ant-oat01-...` (a version number, no dash after `oat`), so every real subscription/OAuth token fell through to the `sk-` branch and was tagged `PAYG` — enabling aggressive lossy compression, auto `cache_control`, and `prompt_cache_key` injection on subscription-bound requests the classifier is meant to route to the passthrough-prefer path. The prefix is now the dash-less `sk-ant-oat` (still matches the legacy dashed shape). The existing parity tests only passed because they used a synthetic `sk-ant-oat-01-` fixture; a regression test now covers the real `sk-ant-oat01-` format.
* **install:** stop leaking a file descriptor on every `headroom install start`. `start_detached_agent()` opened the agent log file and handed it to `subprocess.Popen` but never closed the parent's copy, so each call leaked one fd (and pinned the log file open against rotation). The parent now closes its copy in a `try/finally` once the child has inherited it — the close also runs if `Popen` raises ([#1554](https://github.com/headroomlabs-ai/headroom/issues/1554)).
* **memory/sync:** stop the Codex AGENTS.md sync adapter from erasing previously-synced memories on every export. `sync_export` hands each adapter only the *delta* (memories the agent lacks), but `CodexAdapter.write_memories` rebuilt its whole managed section from just that delta — so each sync overwrote the section with only the new items, thrashing the file between disjoint subsets and never accumulating. It now merges the delta into the facts already present (deduped), matching the additive contract the ClaudeCode adapter already follows.
* **memory/sync:** stop the Claude Code sync adapter from clobbering distinct memories that share a first line. `write_memories` derived each file name from the first line of the content only (`headroom_{slug}.md`), so two different DB memories whose first lines slugify identically wrote to the same file and the second silently overwrote the first — and because the loser never landed on disk, the next sync re-exported it, ping-ponging the pair forever. When the slug is already taken by a *different* memory (distinct `headroom_id`) the file name is now disambiguated with a content-hash suffix; an update to the same memory still rewrites its slug file in place, so existing file names are unchanged.
* **transforms/code:** stop raising `ValueError` on common language hints and fence tags. `CodeAwareCompressor.compress()` built the language with `CodeLanguage(language.lower())`, which only accepts the exact enum values (`python`/`javascript`/`typescript`/…). A markdown ` ```js ` / ` ```ts ` / ` ```py ` fence tag (or any caller passing an alias) raised `ValueError` — crashing direct callers, and inside the content router the error was swallowed so those blocks silently skipped code-aware compression. A new `coerce_language` helper maps the common aliases to their canonical language and returns `UNKNOWN` (never raises) for unrecognized tags, falling back to content-based detection.
* **cli/proxy:** honor `HEADROOM_MIN_TOKENS=0` / `HEADROOM_MAX_ITEMS=0`. The Click `proxy` command built these with `_get_env_int_optional(name) or 500`/`or 50`, so an explicit `0` — a legitimate value (`min_tokens_to_crush=0` means "crush every item") — was treated as falsy and silently replaced with the default. The `headroom proxy` argparse path already preserved `0` via `_get_env_int`, so the two entry points disagreed. The Click path now uses the same None-checking helper.
* **proxy:** strip the inbound `Content-Encoding`/`Transfer-Encoding` request headers on the Anthropic `/v1/messages` and OpenAI `/v1/chat/completions` paths before forwarding upstream. `read_request_json_with_bytes` already decompresses the inbound body (zstd/gzip/deflate/br), so the bytes forwarded upstream are plain JSON — but these two handlers left the original `content-encoding` header in place, so a client (or an edge proxy like a Cloudflare Worker) that sent a compressed body got its request rejected with upstream HTTP 400 because the provider tried to decompress already-decoded JSON. The `/v1/responses` handler already carried this fix (#1542); it is now applied to the messages and chat paths too.
* **models:** fix the model registry's prefix fallback silently returning the wrong context window. `ModelRegistry.get` accepted any registered name as a `str.startswith` prefix and returned the *first* match, so `gpt-4-32k-0613` resolved to `gpt-4` (8192) instead of `gpt-4-32k` (32768), and unregistered ids like `gpt-4.1`/`gpt-4.5` inherited `gpt-4`'s 8192-token window — making the proxy think a nearly-empty context was almost full and compress far too aggressively. The fallback now requires the registered name to end at a version boundary in the query (so `gpt-4.1` no longer matches `gpt-4`) and picks the longest qualifying name (so `gpt-4-32k-0613` → `gpt-4-32k`).
* **mcp/claude:** stop the file-based MCP registrar from destroying an existing but unparseable Claude config. When the `claude` CLI is unavailable, `_register_via_file` read `~/.claude/.claude.json` via a helper that returns `{}` on `JSONDecodeError`, then rewrote the whole file with only `{"mcpServers": {...}}` — wiping unrelated Claude state (`projects`, `oauthAccount`, session history) if the file was momentarily corrupt or hand-edited. The write path now refuses to overwrite a present-but-invalid config and returns a `FAILED` result with an actionable message; an absent or empty file still registers fresh, and a valid file still merges with all other keys preserved.
* **install:** stop `resolve_targets` from rejecting valid `--providers all`/`auto` installs under provider scope. The provider-scope "unsupported targets" validation ran before the mode dispatch, so `headroom install apply --scope provider --providers all --target cursor` raised `ClickException` even though `all`/`auto` ignore the requested target list entirely (user scope silently ignores the same input). The check now runs only on the manual path that actually consults the requested list.
* **mcp/opencode:** stop the OpenCode MCP registrar from destroying an existing but unparseable `opencode.json`. `_write_entry` read the config via a helper that returns `{}` on `JSONDecodeError`, then rewrote the whole file with only `{"mcp": {...}}` — wiping the user's `theme`/`model`/`provider` and any other MCP servers (OpenCode configs are commonly JSONC / hand-edited). The write path now refuses to overwrite a present-but-invalid config and returns a `FAILED` result; absent/empty files still register fresh and valid files still merge with all other keys preserved. (Same class of fix as the Claude registrar.)
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
* **opencode:** write Headroom MCP config as a local stdio server instead of a remote `/mcp` URL, keep provider-only installs from adding MCP config, and allow `install apply --target opencode` ([#1380](https://github.com/headroomlabs-ai/headroom/issues/1380)).
* **proxy:** stop discarding a finished compression on very large requests. After the transform pipeline completed, a telemetry-only waste-signal re-parse of the *original* messages ran on the critical path; on huge Claude Code transcripts (~400k tokens) that parse could exceed the Anthropic compression timeout, so the proxy failed open and forwarded the uncompressed request despite "Pipeline complete" logging real savings (`tokens_saved: 0`, `transforms_applied: []`, ~31s latency). Waste-signal detection is now skipped above `MAX_WASTE_SIGNAL_DETECTION_TOKENS` (100k) so the compression result stays on the critical path ([#296](https://github.com/chopratejas/headroom/issues/296)).
* **codex:** retag existing Codex threads when `headroom init` injects the `headroom` provider, so Codex Desktop history stays visible. Codex filters its sidebar/search by the active `model_provider`; the init path set `model_provider = "headroom"` without retagging, so existing native `openai` threads disappeared from the menu (data was never deleted, only hidden). `_ensure_codex_provider` now reconciles thread tags openai→headroom, matching what the install and `wrap` paths already do; `headroom unwrap codex` handles the revert direction ([#961](https://github.com/chopratejas/headroom/issues/961)).
* **install:** stop duplicating the container ENTRYPOINT in the `persistent-docker` runtime command. The published image already runs `headroom proxy` as its ENTRYPOINT, but `build_runtime_command` re-added `headroom proxy` after the image name, so the container ran `headroom proxy headroom proxy --host 0.0.0.0 …` and Click aborted with "Got unexpected extra arguments (headroom proxy)" — the deployment never became ready and rollback left nothing running. The runtime command now appends only the proxy flags ([#833](https://github.com/chopratejas/headroom/issues/833)).
* **proxy:** retry upstream `529 overloaded_error` like a 429 on both the streaming and non-streaming forwarders, honoring `Retry-After`. The streaming path previously surfaced a 529 straight to the client with no retry (interactive sessions saw "Overloaded" immediately), and `_retry_request` retried it only via the generic 5xx path — raising on exhaustion instead of returning the 529 verbatim, and ignoring `Retry-After`. A shared `RETRYABLE_OVERLOAD_STATUSES = {429, 529}` keeps the two forwarders in agreement (extends [#1221](https://github.com/headroomlabs-ai/headroom/issues/1221)).
* **gemini:** run compression off the asyncio event loop. The Gemini handlers (`generateContent`, Cloud Code stream, `countTokens`) ran the CPU-bound compression pipeline (Magika detection plus ML compression) synchronously on the loop, stalling every concurrent request for the duration of each Gemini request's compression. They now offload it via the shared compression executor, matching the existing OpenAI and Anthropic paths.
* **proxy:** run image compression off the asyncio event loop. The Anthropic and OpenAI handlers ran the CPU-bound image compressor (ONNX technique routing plus Pillow resize and OCR) synchronously on the loop, stalling every concurrent request for the duration of each image request's compression. They now offload it via the shared compression executor with a timeout and fail open on error, matching the existing text-compression path.
* **proxy:** queue mid-turn user messages on non-Bedrock streaming path instead of silently dropping them — closes [#902](https://github.com/headroomlabs-ai/headroom/issues/902).
* **proxy:** add `--protect-tool-results` / `HEADROOM_PROTECT_TOOL_RESULTS` to prevent lossy compression of exact-output tool results (e.g. `Bash cat`/`grep` results) — closes [#1307](https://github.com/headroomlabs-ai/headroom/issues/1307).
* **cli:** add `--rpm`/`--tpm` and `HEADROOM_RPM`/`HEADROOM_TPM` to the Click proxy command for rate-limit parity with the legacy CLI -- closes [#1350](https://github.com/headroomlabs-ai/headroom/issues/1350) (Problem 1).
* **proxy:** register `ToolResultInterceptorTransform` in explicit transforms list when `HEADROOM_INTERCEPT_ENABLED` is set — closes [#829](https://github.com/headroomlabs-ai/headroom/issues/829).
* **opencode:** write Headroom MCP config as a local stdio server instead of a remote `/mcp` URL, keep provider-only installs from adding MCP config, and allow `install apply --target opencode` ([#1380](https://github.com/headroomlabs-ai/headroom/issues/1380)).
* **code:** keep Python `from __future__` imports before executable code during AST compression and validate compressed Python with `compile(..., "exec")` so compile-time syntax rules are enforced ([#1233](https://github.com/chopratejas/headroom/issues/1233)).
* **proxy/transforms:** stop the SMART_CRUSHER → Log fallback from collapsing truncated/invalid JSON tool outputs to a single CCR-retrieval marker (99.9% data loss when CCR retrieval isn't configured). The native magika detector classifies content by shape, not parseability, so a mid-stream-truncated JSON payload is tagged `json_array` and routed to SmartCrusher, which returns it unchanged; Kompress passes it through; the LogCompressor then treated the broken JSON as a multi-thousand-line "log" and reduced it to a retrieval marker. A JSON-validity guard now skips the Log fallback for content that fails `json.loads`, so invalid JSON passes through verbatim. Valid JSON arrays still reach the Log fallback (LogCompressor is a no-op on them). The guard also catches `RecursionError` from deeply nested JSON (e.g. `[[[[...]]]]` with 10k+ levels) so the router falls through to a safe strategy instead of crashing ([#1306](https://github.com/chopratejas/headroom/issues/1306)).
* **proxy/transforms:** fix MIXED false-positive on source code. `is_mixed_content` uses regex heuristics that misclassify Python code with dict/list literals (`{`, `[` at line start → `has_json_blocks`) and docstrings/comments (`has_prose`) as mixed content, routing it through `_compress_mixed` which splits it into sections and dispatches each to KOMPRESS — wasting 1–1.4s of latency with 0% compression. When the native magika detector confidently says `SOURCE_CODE` (confidence ≥ 0.8), `_determine_strategy` now trusts it over the regex heuristics. Additionally, when `prefer_code_aware_for_code=False` (the default), source code now uses `PASSTHROUGH` instead of falling back to `KOMPRESS`, which can destroy code semantics (98% compression, 11% fact recall on large blobs). This honours the config's stated intent ("let code pass through unmangled") and reduces latency on code blobs by 23–33×.
* **proxy/transforms:** catch `RecursionError` in `_try_detect_json` so deeply nested JSON arrays (10k+ nesting levels) no longer crash the content detector. The router falls through to a safe strategy (`TEXT` or `PASSTHROUGH`) instead of raising an unhandled exception.
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
* **proxy:** a transient rtk/lean-ctx stat-read failure (timeout, non-zero exit, bad JSON) no longer corrupts the dashboard's CLI-filtering session metrics. Failed reads now return "no data" instead of a synthetic zero payload, and the session baseline is only ever pinned from successful installed-tool reads — previously one hiccup re-pinned the baseline to zero and the next successful read inflated session savings by the tool's entire lifetime, at every proxy boot and `POST /stats/reset`.
* **proxy:** Concurrent large requests no longer 502 on a transient HTTP/2 stream reset. A single upstream `StreamReset` poisons the shared h2 connection and raises `RemoteProtocolError` / `LocalProtocolError` on every in-flight request; those transport errors weren't in the proxy's retry paths, so they collapsed straight to a 502 with no reconnect. The Anthropic non-streaming and streaming retry paths now treat any `httpx.TransportError` (including h2 protocol errors) as retryable before the first client byte, so the bad connection is dropped and the request re-sent on a fresh one ([#1639](https://github.com/headroomlabs-ai/headroom/issues/1639)).
* **install:** `headroom wrap claude` no longer leaves a dead `ANTHROPIC_BASE_URL` in a project's `.claude/settings.local.json` after an unclean exit (`SIGKILL`, OOM, reboot, or terminal/tmux close via `SIGHUP`, which was not caught). `_write_claude_wrap_base_url`/`_restore_claude_wrap_base_url` only removed or restored the entry from the wrap process's own `finally` block, so a crash skipped it and every later bare `claude` invocation in that project inherited the stale proxy URL and hung indefinitely retrying a dead port. A wrap session now stamps a sidecar marker (pid, port, prior value); the next `wrap`, `unwrap`, or `headroom doctor` run detects a marker whose pid is dead or reused and restores the recorded prior value automatically. `claude()` also now catches `SIGHUP` alongside the existing `SIGTERM` handler ([#1768](https://github.com/headroomlabs-ai/headroom/issues/1768)).
* **proxy:** Non-finite values (`NaN`, `Infinity`) in `proxy_savings.json` or in upstream cost/token metadata no longer crash the proxy or corrupt the savings dashboard. `SavingsTracker`'s numeric coercion caught only `TypeError` and `ValueError`, so `int(float('inf'))` raised an uncaught `OverflowError` while loading persisted state (`SavingsTracker.__init__` failed and the proxy would not start), and `float('nan')`/`float('inf')` passed straight through, then serialized to `NaN`/`Infinity` literals that the dashboard's `JSON.parse` rejects. `json.loads` accepts those literals, so one bad write poisoned every later start. Both coercion helpers now also catch `OverflowError` and reject non-finite floats, failing open to safe defaults.
* **learn:** `headroom learn` now honors `CLAUDE_CONFIG_DIR`. It resolved the Claude config directory as `~/.claude` and wrote global memory to `~/.claude/CLAUDE.md`, so users who relocate their Claude config via that env var had `learn` scan the wrong directory and detect no projects. The scanner and memory writer now read/write the configured directory ([#1630](https://github.com/headroomlabs-ai/headroom/issues/1630)).
* **cli:** `--backend bedrock` now fails fast with an actionable error when temporary AWS credentials (`AWS_SESSION_TOKEN`) are used but botocore is not installed (e.g. the slim default Docker image). litellm's session-token auth path imports botocore, so the missing dependency previously surfaced only at request time as a misleading `authentication_error: No module named 'botocore'`. The proxy now tells the user to install the `bedrock` extra up front ([#1551](https://github.com/headroomlabs-ai/headroom/issues/1551)).
* **compression:** Content detection no longer crashes the proxy on text containing an orphaned `+++ ` target line with no preceding `--- ` source line (common in `set -x` xtrace output and partial diffs). The bundled `unidiff` 0.4.0 parser panics on that input instead of returning an error; the Rust diff detector now contains the panic and treats the fragment as plain text, so the request is compressed and forwarded normally instead of returning HTTP 500 ([#1547](https://github.com/headroomlabs-ai/headroom/issues/1547)).
* **proxy:** persist lifetime cache-read savings (tokens + USD) in `proxy_savings.json` (schema v4, additive) so cache-mode savings survive proxy restarts and upgrades. Previously prefix-cache read savings lived only in process memory and every restart reset the dashboard's cache figure to zero; the "Cache Reads (lifetime)" tile now reads the persisted value and the Prefix Cache Impact card renders after a restart with zero traffic, marking session-scoped tiles "no activity since restart".
* **compression:** Proactive expansion blocks injected into user turns are now wrapped in`<headroom_proactive_expansion>` XML tags, giving downstream consumers (LLMs, loggers, attribution parsers) a machine-readable provenance boundary and preventing misattribution in multi-agent threads.
* **cli:** the startup banner no longer advertises `HEADROOM_COMPRESSION_STABLE_AFTER_TURN` and `HEADROOM_STALE_READ_COMPRESS_AFTER_TURNS` as tuning knobs. Both were read only to render the `Performance Tuning` banner section and were never wired into the compression path, so setting them changed the banner but had no effect on behavior. The banner now surfaces only the embedding sidecar, which is a real, consumed setting.
* **memory/embedder:** cap CPU thread oversubscription in the local torch/sentence-transformers embedder. Concurrent encodes previously each fanned out to ~`os.cpu_count()` BLAS/OpenMP threads, so under load the memory path starved the asyncio event loop and spiked `/livez` latency to several seconds. CPU encodes now run on a dedicated, size-limited executor whose workers each pin their thread pool, bounding total embedding threads to `HEADROOM_EMBED_CONCURRENCY` × `HEADROOM_EMBED_NUM_THREADS` (defaults `min(4, cpu)` × 1). The ONNX embedder already capped its threads; this brings the torch path to parity ([#198](https://github.com/headroomlabs-ai/headroom/issues/198)).
* **proxy:** Buffered passthrough routes (e.g. `GET /v1/models`) no longer return an opaque HTTP 502 when an OpenAI-compatible upstream closes a pooled keep-alive connection mid-response (`httpx.RemoteProtocolError` / "incomplete chunked read"). Headroom now retries the request once on a fresh connection — mirroring a direct `curl` — and only returns a clear `upstream_protocol_error` 502 if the upstream is genuinely sending an incomplete response ([#1112](https://github.com/chopratejas/headroom/issues/1112)).
* **ccr:** buffered Anthropic CCR re-streaming now preserves adaptive-thinking response shape, including empty `thinking` blocks, `signature_delta`, `redacted_thinking.data`, verbatim `stop_reason` values such as `refusal`, and `stop_details`.
* **cursor:** `headroom wrap cursor` no longer injects the `rtk` custom-instructions block into `.cursorrules` when rtk's own native Cursor hook registers successfully. rtk supports a real hook for Cursor via `rtk init --agent cursor` (the same mechanism headroom already uses for Claude Code), which rewrites shell commands transparently — the injected `.cursorrules` text duplicated that guidance for no benefit. `wrap cursor` now tries the native hook first and only falls back to injecting `.cursorrules` if hook registration fails (#756).
* **proxy:** The Headroom dashboard no longer tunnels `GET /favicon.ico` to the wrapped upstream provider. No route matched that path, so it fell through to the proxy's catch-all passthrough route and was forwarded to the configured Anthropic/OpenAI/etc. backend — burning a real upstream request (and possibly failing auth) for a browser's automatic favicon fetch on `/dashboard`. A dedicated `/favicon.ico` route now answers with `204 No Content` directly, registered ahead of the passthrough catch-all (#1787).
* **learn:** fix three Windows-specific failures in `headroom learn --verbosity` and CLI-backed analysis ([#1624](https://github.com/headroomlabs-ai/headroom/issues/1624)). `verbosity.py` read transcripts and profiles with the platform-default text codec instead of UTF-8, so non-ASCII content raised a silently-caught `UnicodeDecodeError`, producing `Sessions: 0, human turns: 0` for every project. `_greedy_path_decode` listed a directory's children with `is_dir()` inline in the same expression as `iterdir()`, so a single `PermissionError` on an inaccessible sibling (e.g. the `AppData\Local\Temporary Internet Files` junction present on most Windows profiles) aborted the whole listing and silently mis-decoded any project path that walked through it, causing `--project <path>` to report "No matching project" or resolve the wrong directory. `_call_cli_llm` launched CLI backends via `Popen`/`run`, which use `CreateProcess` on Windows and don't apply the shell's `PATHEXT` extension search, so an npm-installed `.cmd` shim (e.g. `claude`, `codex`) raised `FileNotFoundError` even though it was on `PATH`; a `shutil.which`-based retry now resolves the shim.
* **proxy:** The Anthropic Messages route (`POST /v1/messages`) now honors the `x-headroom-base-url` per-request upstream override. It previously ignored the header and always forwarded to `api.anthropic.com`, so clients that speak the Anthropic Messages wire format while authenticating against a non-Anthropic gateway (e.g. OpenCode Zen) were rejected upstream with `401 invalid x-api-key`. The route now forwards to `<x-headroom-base-url>/v1/messages`, consistent with the OpenAI-compatible and passthrough routes ([#1760](https://github.com/headroomlabs-ai/headroom/issues/1760)).
* **proxy:** the savings store now fsyncs its parent directory after the atomic rename, so the most recent `proxy_savings.json` write survives a power-loss or crash. `_save_locked` fsynced the temp file's contents but never the directory entry the rename created, leaving the rename itself non-durable on POSIX. Best-effort — a no-op on Windows and virtual filesystems where directory fsync is unsupported.
- **code:** fix two `CodeAwareCompressor` AST-reassembly bugs: an exported JS/TS function or class (`export function foo() {`) produced a duplicated `export export` keyword and invalid syntax, because line-based node slicing (used to preserve indentation) pulled in the preceding `export` sibling's text on top of the `export_statement` handler's own prefix reconstruction. Separately, in every supported language, a doc comment immediately above a top-level function, class, or type was detached from its declaration during extraction and re-emitted in a cluster at the end of the compressed output instead of staying attached to what it documents.
- * **proxy:** Buffered upstream responses containing a `server_tool_use` (or any other unrecognized Anthropic content block) no longer turn a fully-generated response into an HTTP 502. `StreamingMixin._response_to_sse` raised `ValueError` on unknown block types after the entire upstream generation had already been buffered, so a slow-but-successful response failed and the client retried the whole multi-minute request. Unknown blocks are now emitted verbatim in `content_block_start` (following the existing redacted_thinking` pattern), so `server_tool_use`, `server_tool_result`, `mcp_tool_use`, and future block types round-trip ([#1806](https://github.com/headroomlabs-ai/headroom/issues/1806)).

## [0.37.0](https://github.com/headroomlabs-ai/headroom/compare/v0.36.5...v0.37.0) (2026-08-27)


### Features

* **compress:** session-aware /v1/compress (sidecar mode) + /v1/usage relay ([#3270](https://github.com/headroomlabs-ai/headroom/issues/3270)) ([4fa8802](https://github.com/headroomlabs-ai/headroom/commit/4fa88026d9ff64092feaa750789a28be3ed0ace8))
* **proxy:** self-limiting session state for the compression-cache registry ([#3261](https://github.com/headroomlabs-ai/headroom/issues/3261)) ([826b600](https://github.com/headroomlabs-ai/headroom/commit/826b600c9b5ac80dae5930e1b7aa376a463d28d6))
* **proxy:** unify proxy and sidecar compression on one session engine ([#3271](https://github.com/headroomlabs-ai/headroom/issues/3271)) ([d12ea50](https://github.com/headroomlabs-ai/headroom/commit/d12ea501222b92f371b5427595c14ec2466a3f39))


### Bug Fixes

* **cache/semantic:** don't semantic-match an empty query across contexts ([#3226](https://github.com/headroomlabs-ai/headroom/issues/3226)) ([455f4f2](https://github.com/headroomlabs-ai/headroom/commit/455f4f263ce2638ff387c52d42601760f5cf0784))
* **copilot:** preserve native enterprise model routing ([#2998](https://github.com/headroomlabs-ai/headroom/issues/2998)) ([997a479](https://github.com/headroomlabs-ai/headroom/commit/997a47992c490e7a13db34ef39eccf3dfb006488))
* **kimi:** route managed Kimi Code through the proxy ([#3223](https://github.com/headroomlabs-ai/headroom/issues/3223)) ([701e461](https://github.com/headroomlabs-ai/headroom/commit/701e4616d92d6d4b110bf5356675b6db409802e8))
* **learn:** surface Codex analysis failures ([#3016](https://github.com/headroomlabs-ai/headroom/issues/3016)) ([632cb81](https://github.com/headroomlabs-ai/headroom/commit/632cb81dbe7400cf77a20d0d395d4cfb0ea245e7))
* **mcp:** add explicit Serena reconciliation ([#3222](https://github.com/headroomlabs-ai/headroom/issues/3222)) ([7550efb](https://github.com/headroomlabs-ai/headroom/commit/7550efb68fde6acf0674244b03eb268b976ffced))
* **proxy/anthropic:** authenticate and attribute buffered Copilot turns ([#3277](https://github.com/headroomlabs-ai/headroom/issues/3277)) ([4f2e70a](https://github.com/headroomlabs-ai/headroom/commit/4f2e70a75cd2870c23b1905dcb8c7f8d13aea6b0))
* **proxy:** enforce HEADROOM_PROXY_TOKEN on WebSocket handshakes ([#3305](https://github.com/headroomlabs-ai/headroom/issues/3305)) ([27b4e2d](https://github.com/headroomlabs-ai/headroom/commit/27b4e2d147c566a4e5509634f6ab56696abc7b97))
* **proxy:** make output-savings flush atomic and keep it off the event loop ([#3231](https://github.com/headroomlabs-ai/headroom/issues/3231)) ([b9d7dcc](https://github.com/headroomlabs-ai/headroom/commit/b9d7dcc3da3ec67a968d0ac1e35bffba9b7cf1c2))
* **proxy:** protect file reads from lossy compression on the Responses API path (Copilot view + HEADROOM_PROTECT_READS) ([#3238](https://github.com/headroomlabs-ai/headroom/issues/3238)) ([4408e88](https://github.com/headroomlabs-ai/headroom/commit/4408e881064741a5113be69d631f336ce92631f5))
* **transforms:** stop compression garbling mixed subagent output ([#3286](https://github.com/headroomlabs-ai/headroom/issues/3286)) ([8884d87](https://github.com/headroomlabs-ai/headroom/commit/8884d8737802e61b1c9d8dfe6351f0c435eb42ae))
* **transforms:** stop folding datetime-prefixed user messages as search results ([#3221](https://github.com/headroomlabs-ai/headroom/issues/3221)) ([7784bb1](https://github.com/headroomlabs-ai/headroom/commit/7784bb184614bdcd7ba1afd93a23e8d1458ecc96))
* **vertex:** validate location region to close a path-parameter SSRF ([#3304](https://github.com/headroomlabs-ai/headroom/issues/3304)) ([7c0b886](https://github.com/headroomlabs-ai/headroom/commit/7c0b886004abf232130cd0cb6127ad7143e2ca9a))
* **windows:** terminate native proxy process tree ([#3313](https://github.com/headroomlabs-ai/headroom/issues/3313)) ([91d4fc8](https://github.com/headroomlabs-ai/headroom/commit/91d4fc8bbdd637a4d53c1c667bdc2502766be686))
* **wrap:** stop concurrent wrap sessions clobbering settings.local.json ([#3232](https://github.com/headroomlabs-ai/headroom/issues/3232)) ([f27f235](https://github.com/headroomlabs-ai/headroom/commit/f27f235032d8aef522efbcb4daf548787692c7e4))

## [0.36.5](https://github.com/headroomlabs-ai/headroom/compare/v0.36.4...v0.36.5) (2026-08-22)


### Bug Fixes

* **codex:** detect ChatGPT auth from id_token claims so wrap/init emit requires_openai_auth ([#3212](https://github.com/headroomlabs-ai/headroom/issues/3212)) ([2f81fa5](https://github.com/headroomlabs-ai/headroom/commit/2f81fa5931ddf233b908103f25c614b5a6b7e33b))
* **doctor:** report project-scoped Claude routing instead of a false negative ([#3213](https://github.com/headroomlabs-ai/headroom/issues/3213)) ([8f3e33a](https://github.com/headroomlabs-ai/headroom/commit/8f3e33a00ea377403497b600841c03344d0c1cd8))

## [0.36.4](https://github.com/headroomlabs-ai/headroom/compare/v0.36.3...v0.36.4) (2026-08-22)


### Bug Fixes

* **dashboard:** pin MIME types for the vendored static assets ([#3193](https://github.com/headroomlabs-ai/headroom/issues/3193)) ([b485768](https://github.com/headroomlabs-ai/headroom/commit/b4857685ffca656f8f4f17111b88e80637511f52))
* **proxy/responses:** keep the Codex additional_tools carrier on the wire ([#3194](https://github.com/headroomlabs-ai/headroom/issues/3194)) ([1617f83](https://github.com/headroomlabs-ai/headroom/commit/1617f839a197ed1f17ca2083fbd288ffa2af7820))
* **security:** validate caller-supplied upstreams on every resolution path ([#3195](https://github.com/headroomlabs-ai/headroom/issues/3195)) ([3e3c409](https://github.com/headroomlabs-ai/headroom/commit/3e3c409436792129259cfae3d95179a94321f9ce))
* skip cross-turn dedup pointers on OpenAI chat streaming ([#3191](https://github.com/headroomlabs-ai/headroom/issues/3191)) ([9c30b62](https://github.com/headroomlabs-ai/headroom/commit/9c30b629624a42495d82f79fb7df9f21cdac7865))
* **wrap:** make the Serena pre-index stall budget configurable ([#3183](https://github.com/headroomlabs-ai/headroom/issues/3183)) ([202c189](https://github.com/headroomlabs-ai/headroom/commit/202c1895e1c2617121f3513054e4a1306d9c573f))

## [0.36.3](https://github.com/headroomlabs-ai/headroom/compare/v0.36.2...v0.36.3) (2026-08-21)


### Bug Fixes

* **proxy/responses:** lift Codex &gt;= 0.149.0 additional_tools into top-level tools ([#3186](https://github.com/headroomlabs-ai/headroom/issues/3186)) ([25ca580](https://github.com/headroomlabs-ai/headroom/commit/25ca580825b6d1eef385042fd9524e2da2b2baee))

## [0.36.2](https://github.com/headroomlabs-ai/headroom/compare/v0.36.1...v0.36.2) (2026-08-21)


### Bug Fixes

* **copilot:** bind the minted token to the integration ID we forward ([#3164](https://github.com/headroomlabs-ai/headroom/issues/3164)) ([397803a](https://github.com/headroomlabs-ai/headroom/commit/397803a9424cf184f597a2d18af82632e7b0ac70))
* **kompress:** accept ccr_original on the remote compressor ([#3162](https://github.com/headroomlabs-ai/headroom/issues/3162)) ([45cb1b9](https://github.com/headroomlabs-ai/headroom/commit/45cb1b9c4824a3a609772d8828a119bdc1c31ad0))
* **proxy:** count output tokens from the stream's text, not its wire size ([#3163](https://github.com/headroomlabs-ai/headroom/issues/3163)) ([4006964](https://github.com/headroomlabs-ai/headroom/commit/4006964a037817b22437abcaf2c55b13085b4021))


### Dependencies

* bump ai from 6.0.138 to 7.0.59 in /sdk/typescript ([#2281](https://github.com/headroomlabs-ai/headroom/issues/2281)) ([0891062](https://github.com/headroomlabs-ai/headroom/commit/08910624fbe4877da0789038957308dbe9be9451))
* bump ai from 6.0.149 to 7.0.59 in /docs ([#2277](https://github.com/headroomlabs-ai/headroom/issues/2277)) ([f7e5d37](https://github.com/headroomlabs-ai/headroom/commit/f7e5d37f526907b2b8a22d5e7332ed3fddfa071c))
* bump md-5 from 0.10.6 to 0.11.0 ([#3146](https://github.com/headroomlabs-ai/headroom/issues/3146)) ([c6dd823](https://github.com/headroomlabs-ai/headroom/commit/c6dd82338434f964cd73ed2f742ad0c733bff79f))
* bump ruff from 0.16.2 to 0.16.3 in the pip-minor-patch group ([#3143](https://github.com/headroomlabs-ai/headroom/issues/3143)) ([c8db13d](https://github.com/headroomlabs-ai/headroom/commit/c8db13d5ad78d9d1bcf5c43dcaeba69070f26e1f))
* bump the cargo-minor-patch group with 8 updates ([#3145](https://github.com/headroomlabs-ai/headroom/issues/3145)) ([9c14e3a](https://github.com/headroomlabs-ai/headroom/commit/9c14e3aa9598a01c3cc208c8f483b5dd6398ea1f))
* bump tiktoken-rs from 0.11.0 to 0.12.0 ([#3147](https://github.com/headroomlabs-ai/headroom/issues/3147)) ([a307c11](https://github.com/headroomlabs-ai/headroom/commit/a307c11109de21b0d5d9648be69b0f08a40d3a4d))
* bump tokenizers from 0.22.2 to 0.23.1 ([#3149](https://github.com/headroomlabs-ai/headroom/issues/3149)) ([6e2e10f](https://github.com/headroomlabs-ai/headroom/commit/6e2e10f67a737d3b2d844975c1384e02637dbcc3))
* bump typescript from 5.9.3 to 7.0.2 in /plugins/openclaw ([#2279](https://github.com/headroomlabs-ai/headroom/issues/2279)) ([85774fc](https://github.com/headroomlabs-ai/headroom/commit/85774fcb70b46ae44bcd3533a214d64189908622))
* bump typescript from 5.9.3 to 7.0.2 in /plugins/opencode ([#2280](https://github.com/headroomlabs-ai/headroom/issues/2280)) ([a382137](https://github.com/headroomlabs-ai/headroom/commit/a382137844428820f49d9b43100fed8e492c2db2))
* update mcp requirement from &lt;2.0.0,&gt;=1.28.1 to &gt;=1.28.1,&lt;3.0.0 ([#3144](https://github.com/headroomlabs-ai/headroom/issues/3144)) ([6928d19](https://github.com/headroomlabs-ai/headroom/commit/6928d1932c1136fd1c6bd724ee88d12815ecd2d6))

## [0.36.1](https://github.com/headroomlabs-ai/headroom/compare/v0.36.0...v0.36.1) (2026-08-20)


### Bug Fixes

* **docker:** give :latest exactly one writer ([#3154](https://github.com/headroomlabs-ai/headroom/issues/3154)) ([bf651c3](https://github.com/headroomlabs-ai/headroom/commit/bf651c3dc1b8c43cca84d085b57528fa9c7de5cd))
* **metrics:** attribute tool-schema savings per model, not just compression ([#3155](https://github.com/headroomlabs-ai/headroom/issues/3155)) ([81fe9d5](https://github.com/headroomlabs-ai/headroom/commit/81fe9d534579d4dcac197ba901f65d6f19986d32))
* **security:** address u9up assessment findings (WEB-01–07) ([#2207](https://github.com/headroomlabs-ai/headroom/issues/2207)) ([1f96dab](https://github.com/headroomlabs-ai/headroom/commit/1f96dabc19130947770353cd6e814db4fd96e6a0))

## [0.36.0](https://github.com/headroomlabs-ai/headroom/compare/v0.35.0...v0.36.0) (2026-08-20)


### Features

* add deterministic runtime rollout controls ([#1490](https://github.com/headroomlabs-ai/headroom/issues/1490)) ([3077ac8](https://github.com/headroomlabs-ai/headroom/commit/3077ac81e8ef3ddefebbe308ea37a4e9bb2100e6))
* **proxy:** let extensions report cost savings and their own latency ([#3051](https://github.com/headroomlabs-ai/headroom/issues/3051)) ([f9807fd](https://github.com/headroomlabs-ai/headroom/commit/f9807fd69e220f43068ec168515ae886dd36166f))
* **proxy:** unify savings attribution across stats, perf, metrics, and dashboard ([1b0b0b8](https://github.com/headroomlabs-ai/headroom/commit/1b0b0b89a4bf751c8bd592890aef9c3b339e8e37)), closes [#2976](https://github.com/headroomlabs-ai/headroom/issues/2976)
* **wrap/claude:** make the --1m fallback model configurable via HEADROOM_1M_MODEL ([#2983](https://github.com/headroomlabs-ai/headroom/issues/2983)) ([2a84725](https://github.com/headroomlabs-ai/headroom/commit/2a8472525d3a027c95dc38a10c4b6707b482cabc))


### Bug Fixes

* **anthropic:** honor the [1m] 1M-context tier, and price it correctly ([#3073](https://github.com/headroomlabs-ai/headroom/issues/3073)) ([6d2254d](https://github.com/headroomlabs-ai/headroom/commit/6d2254dfb5eb97f92249e0ee7aa04b2697adfa69))
* **ccr:** make --no-ccr disable server-side response handling too ([#3101](https://github.com/headroomlabs-ai/headroom/issues/3101)) ([131b119](https://github.com/headroomlabs-ai/headroom/commit/131b119c053e66fe825dabb3c242f6dc5c6049d7)), closes [#3082](https://github.com/headroomlabs-ai/headroom/issues/3082)
* **ccr:** make StreamingCCRHandler work on OpenAI streams ([#3069](https://github.com/headroomlabs-ai/headroom/issues/3069)) ([7ef736f](https://github.com/headroomlabs-ai/headroom/commit/7ef736fb1a8852a3dee52a362043c47084628a2a))
* **ccr:** only buffer a stream when a marker is actually redeemable ([#3092](https://github.com/headroomlabs-ai/headroom/issues/3092)) ([c502087](https://github.com/headroomlabs-ai/headroom/commit/c502087db702e9b6aa1d1736086cf4a69a7775e6))
* **ccr:** re-inject headroom_retrieve when history references it on the sessionless path ([942af56](https://github.com/headroomlabs-ai/headroom/commit/942af56f11cbd8466e25ae189c65ba56a9ddd602))
* **ccr:** relay a successful upstream turn when post-processing fails ([#3094](https://github.com/headroomlabs-ai/headroom/issues/3094)) ([0ec73fa](https://github.com/headroomlabs-ai/headroom/commit/0ec73faa2805502a5c13eab7e4f086f8ae2e175e))
* **ccr:** send Accept: application/json on a buffered stream:false turn ([#3102](https://github.com/headroomlabs-ai/headroom/issues/3102)) ([139c7cb](https://github.com/headroomlabs-ai/headroom/commit/139c7cbdde6a68ae3ade24341a79e5ba659c2cf3)), closes [#3078](https://github.com/headroomlabs-ai/headroom/issues/3078)
* **ccr:** verify a scanned marker's hash before advertising it ([#2908](https://github.com/headroomlabs-ai/headroom/issues/2908)) ([41dab2d](https://github.com/headroomlabs-ai/headroom/commit/41dab2d09925658b96fed492d534346ce1930f4c))
* **ci:** prevent native detector from hanging test shards ([#2996](https://github.com/headroomlabs-ai/headroom/issues/2996)) ([a708c05](https://github.com/headroomlabs-ai/headroom/commit/a708c0571eecfb53eaab6b787b7a6ace9b21c162))
* **ci:** scope the release credential and stop persisting it to disk ([#3062](https://github.com/headroomlabs-ai/headroom/issues/3062)) ([ac8646a](https://github.com/headroomlabs-ai/headroom/commit/ac8646aa3c6323c3c0b7051e09831f779859af6f))
* **ci:** unjam release and Docker publishing ([#2958](https://github.com/headroomlabs-ai/headroom/issues/2958)) ([e269afb](https://github.com/headroomlabs-ai/headroom/commit/e269afb935f298a833a189acfb8573e908b3b60b))
* **claude:** reject conflicting auth before proxy startup ([#2993](https://github.com/headroomlabs-ai/headroom/issues/2993)) ([2d88e31](https://github.com/headroomlabs-ai/headroom/commit/2d88e31a404e2be6c1c428deb2a387599eb820ba))
* **cli/install:** resolve the deployment profile instead of dead-ending on default ([#2832](https://github.com/headroomlabs-ai/headroom/issues/2832)) ([8252619](https://github.com/headroomlabs-ai/headroom/commit/82526191a103a8d0e079d170e47631b3c2bcb0d9))
* **cli:** stop the macOS malloc re-exec replacing an embedder's process ([#3064](https://github.com/headroomlabs-ai/headroom/issues/3064)) ([96c25f5](https://github.com/headroomlabs-ai/headroom/commit/96c25f518154536cf15f4e0b2d3fed80de6e67f6))
* **copilot:** route VS Code inline completions to Copilot, not OpenAI ([#3077](https://github.com/headroomlabs-ai/headroom/issues/3077)) ([204e751](https://github.com/headroomlabs-ai/headroom/commit/204e751d2f01b0e987e9c05edec21664bb2df279))
* **copilot:** send VS Code inline completions to the host that serves them ([#3112](https://github.com/headroomlabs-ai/headroom/issues/3112)) ([b77d612](https://github.com/headroomlabs-ai/headroom/commit/b77d61291399976985f12adcd6014aba2f0275cf))
* **deps:** bump datasets past PYSEC-2026-3716 ([#3136](https://github.com/headroomlabs-ai/headroom/issues/3136)) ([df6ff6b](https://github.com/headroomlabs-ai/headroom/commit/df6ff6bd5b47837c1247cf4eb8ac151ebd799aa5))
* **deps:** clear the two Rust advisories and make cargo audit blocking ([#3121](https://github.com/headroomlabs-ai/headroom/issues/3121)) ([93c474e](https://github.com/headroomlabs-ai/headroom/commit/93c474e84b2eeee147c274f3d75f48e5ea42d0d5))
* **deps:** raise the GitPython floor to 3.1.58 to clear 9 open advisories ([#3120](https://github.com/headroomlabs-ai/headroom/issues/3120)) ([8156d4d](https://github.com/headroomlabs-ai/headroom/commit/8156d4dc3a376476513ef6f78104ff81d08967ac))
* **docker:** publish compose ports on loopback only ([#3061](https://github.com/headroomlabs-ai/headroom/issues/3061)) ([481e0b8](https://github.com/headroomlabs-ai/headroom/commit/481e0b83d5393419b27b17d95767104c7c1bda26))
* **docker:** ship Bedrock auth and current registry ([#2982](https://github.com/headroomlabs-ai/headroom/issues/2982)) ([eafdf11](https://github.com/headroomlabs-ai/headroom/commit/eafdf11a2cea44aabc51ce59bbc031e0aaee9640))
* **doctor:** surface that Claude Desktop agent sessions bypass the proxy ([#2987](https://github.com/headroomlabs-ai/headroom/issues/2987)) ([be5b26d](https://github.com/headroomlabs-ai/headroom/commit/be5b26d807be81d83594c9144a8520f6f0f1b273))
* **install:** consolidate Windows fallback and cleanup safety ([#2980](https://github.com/headroomlabs-ai/headroom/issues/2980)) ([ddd2a25](https://github.com/headroomlabs-ai/headroom/commit/ddd2a259ecce4e57202a68a74a2c1adcb879679b))
* **install:** honor HEADROOM_PORT in install apply and deploy ([#3085](https://github.com/headroomlabs-ai/headroom/issues/3085)) ([58f28dc](https://github.com/headroomlabs-ai/headroom/commit/58f28dc7a6b6ce5bbf0f88524bd78cbe3f3ffa4b))
* **install:** stop the PowerShell installer leaking temp dirs into the real user PATH ([#2985](https://github.com/headroomlabs-ai/headroom/issues/2985)) ([ddd9f76](https://github.com/headroomlabs-ai/headroom/commit/ddd9f76729d5662201b84bd0a51281cd3ac64ad3))
* **learn:** include stdout in CLI failure messages, not just stderr ([#3080](https://github.com/headroomlabs-ai/headroom/issues/3080)) ([c5563d3](https://github.com/headroomlabs-ai/headroom/commit/c5563d3a7dd8b7f88767cf503f1b1696917e36ee))
* **mcp:** restore SDK v1 compatibility cap ([#2978](https://github.com/headroomlabs-ai/headroom/issues/2978)) ([6077e5a](https://github.com/headroomlabs-ai/headroom/commit/6077e5a149ee6548edaff033f2cdffffce6ea0cf))
* **memory:** sanitize entity_refs to prevent dict-shaped entries crashing search ([#2951](https://github.com/headroomlabs-ai/headroom/issues/2951)) ([2d1e96b](https://github.com/headroomlabs-ai/headroom/commit/2d1e96b85c61cc7aab821750f549f24d54cbb6f5))
* **onnx:** enforce Rust API-24 runtime compatibility ([#2979](https://github.com/headroomlabs-ai/headroom/issues/2979)) ([a3fe5cb](https://github.com/headroomlabs-ai/headroom/commit/a3fe5cb65bed625e2a6cb415821bd0798754ce08))
* **openclaw-plugin:** circuit breaker + per-request timeout for proxy resilience ([#639](https://github.com/headroomlabs-ai/headroom/issues/639)) ([6576ef6](https://github.com/headroomlabs-ai/headroom/commit/6576ef639cbb7be8bc5e6c25134956803d18f8d8))
* **opencode:** send x-headroom-project header on all proxied requests ([#2868](https://github.com/headroomlabs-ai/headroom/issues/2868)) ([eeb038b](https://github.com/headroomlabs-ai/headroom/commit/eeb038bc0c28fc8078986db0849bfcff6743c158))
* **policy:** price net-cost mutations with the 1h cache-write tier ([#2780](https://github.com/headroomlabs-ai/headroom/issues/2780)) ([ef7e07e](https://github.com/headroomlabs-ai/headroom/commit/ef7e07e0f5d6510ab96b5abb1698b1b681b5f9bf))
* **providers:** don't crash on a non-object HEADROOM_MODEL_LIMITS / models.json ([#3089](https://github.com/headroomlabs-ai/headroom/issues/3089)) ([3ed8f76](https://github.com/headroomlabs-ai/headroom/commit/3ed8f7601935cb08eebfd34007e97675903180a5))
* **proxy/anthropic:** don't buffer a CCR stream when passthrough discards the stream flip ([#2953](https://github.com/headroomlabs-ai/headroom/issues/2953)) ([f1c34d3](https://github.com/headroomlabs-ai/headroom/commit/f1c34d336cf35db341153c1c65e8c15219398340))
* **proxy/anthropic:** don't replay recorded prefix over live history ([#3026](https://github.com/headroomlabs-ai/headroom/issues/3026)) ([#3052](https://github.com/headroomlabs-ai/headroom/issues/3052)) ([c16be9b](https://github.com/headroomlabs-ai/headroom/commit/c16be9bbbec4aec6d4b35e482c166daef8afa72c))
* **proxy/anthropic:** repair headroom_retrieve history references the tools array cannot support ([#2876](https://github.com/headroomlabs-ai/headroom/issues/2876)) ([7de3573](https://github.com/headroomlabs-ai/headroom/commit/7de35739c61bed385dd078aee1b36865938c486d))
* **proxy/anthropic:** stop answering a non-streaming turn with an event stream ([#3142](https://github.com/headroomlabs-ai/headroom/issues/3142)) ([0e26fb8](https://github.com/headroomlabs-ai/headroom/commit/0e26fb80de600795e96435473486c4a7c79c6eaa))
* **proxy/cache:** strip cache_control from messages in the semantic cache key ([#3086](https://github.com/headroomlabs-ai/headroom/issues/3086)) ([2cae0f8](https://github.com/headroomlabs-ai/headroom/commit/2cae0f8eaf627f6b743deb215f7c19c499c26bcd))
* **proxy/gemini:** guard CCR continuation usage against present-null counts ([#3035](https://github.com/headroomlabs-ai/headroom/issues/3035)) ([a01897c](https://github.com/headroomlabs-ai/headroom/commit/a01897c791f4bb6471defafd560d29d491eb2df8))
* **proxy/openai:** propagate provider usage on the Responses WS-&gt;HTTP fallback ([#2988](https://github.com/headroomlabs-ai/headroom/issues/2988)) ([536c949](https://github.com/headroomlabs-ai/headroom/commit/536c949a692f4855719d71d612abc4968040286b))
* **proxy:** adapt 200 SSE upstream replies on buffered /v1/responses instead of 502 ([#2622](https://github.com/headroomlabs-ai/headroom/issues/2622)) ([d76fce0](https://github.com/headroomlabs-ai/headroom/commit/d76fce04a39b3f206e38a02e012d50b2c728f7ca))
* **proxy:** align signed-thinking wire accounting ([#3015](https://github.com/headroomlabs-ai/headroom/issues/3015)) ([b3f4436](https://github.com/headroomlabs-ai/headroom/commit/b3f443636d279d4bad845a8ef2bddb7ca50e9bc6))
* **proxy:** complete stateless Responses and buffered CCR lifecycle ([#2997](https://github.com/headroomlabs-ai/headroom/issues/2997)) ([8a1d38b](https://github.com/headroomlabs-ai/headroom/commit/8a1d38bc5da87b49a530df22090c3a156d2d0cd6))
* **proxy:** guard feedback endpoints and add CSRF checks to loopback writes ([#3060](https://github.com/headroomlabs-ai/headroom/issues/3060)) ([a6ab359](https://github.com/headroomlabs-ai/headroom/commit/a6ab359a5d8d67a85f734131b55dbcef768a821a))
* **proxy:** keep prefixed core tools resident ([#3046](https://github.com/headroomlabs-ai/headroom/issues/3046)) ([2f4d001](https://github.com/headroomlabs-ai/headroom/commit/2f4d001c9ffd7f856c8dab3e31a8240a1c676f04))
* **proxy:** preserve Codex WebSocket model attribution ([#3029](https://github.com/headroomlabs-ai/headroom/issues/3029)) ([a06a51e](https://github.com/headroomlabs-ai/headroom/commit/a06a51eca63f88271dfa77f2ee6bf3c8da6b24e4))
* **proxy:** relocate stray system-role messages to the top-level system param ([#765](https://github.com/headroomlabs-ai/headroom/issues/765)) ([#1357](https://github.com/headroomlabs-ai/headroom/issues/1357)) ([9fde127](https://github.com/headroomlabs-ai/headroom/commit/9fde12753416a6102535235b822e44afebf76e9e))
* **proxy:** restore the buffered-CCR heartbeat behind a grace window ([#3091](https://github.com/headroomlabs-ai/headroom/issues/3091)) ([a29d201](https://github.com/headroomlabs-ai/headroom/commit/a29d2015e5eaf72730a4155f0307cbfac1ea1c9b))
* **proxy:** scope the signed-thinking lock to blocks that actually changed ([#3124](https://github.com/headroomlabs-ai/headroom/issues/3124)) ([17522fb](https://github.com/headroomlabs-ai/headroom/commit/17522fb0a1013c012e8123b1e713dbb2f3e770d9))
* **proxy:** stop a lone surrogate turning a thinking body into a 500 ([#3134](https://github.com/headroomlabs-ai/headroom/issues/3134)) ([284ff31](https://github.com/headroomlabs-ai/headroom/commit/284ff31947ec9eac1de0e2dc1cf5de4933c29a50))
* **proxy:** stop cached responses replaying the producing turn's wire framing ([#3024](https://github.com/headroomlabs-ai/headroom/issues/3024)) ([9d37059](https://github.com/headroomlabs-ai/headroom/commit/9d370592b022d01e6bc44a88649a611507794776))
* **proxy:** stop operator secrets following a client-chosen upstream ([#3122](https://github.com/headroomlabs-ai/headroom/issues/3122)) ([05f5ef4](https://github.com/headroomlabs-ai/headroom/commit/05f5ef47cbc8b31a60458553d6bf240896a47e16))
* **proxy:** tune macOS libmalloc and trim allocator pages so long-lived RSS stays bounded ([#2879](https://github.com/headroomlabs-ai/headroom/issues/2879)) ([6d87825](https://github.com/headroomlabs-ai/headroom/commit/6d87825f62e47bc65eeae05fbb8a131d545fe5a2))
* **reporting:** show net vs gross savings, real skip thresholds, and the effective profile ([#3123](https://github.com/headroomlabs-ai/headroom/issues/3123)) ([250ede2](https://github.com/headroomlabs-ai/headroom/commit/250ede2f7f4752c0ab08831013fad3f753f4a578))
* tool_search_tool_regex deferred and falsely resolved on direct-Anthropic path ([#2971](https://github.com/headroomlabs-ai/headroom/issues/2971)) ([8ea87e7](https://github.com/headroomlabs-ai/headroom/commit/8ea87e7804abfbb55beaf869e50dcb66deab975a))
* **vscode:** persist compatible Claude modes and route Copilot CAPI ([#2986](https://github.com/headroomlabs-ai/headroom/issues/2986)) ([1aa701a](https://github.com/headroomlabs-ai/headroom/commit/1aa701adaa1ff792dd0e701f498d8d0326655670))
* **wrap:** set xAI upstream for grok-build proxy ([#2772](https://github.com/headroomlabs-ai/headroom/issues/2772)) ([c831081](https://github.com/headroomlabs-ai/headroom/commit/c8310819a4221b0d120436786fc499a24c8e55f1))
* **wrap:** stop the Serena pre-index stalling the launch path for 300s ([#2945](https://github.com/headroomlabs-ai/headroom/issues/2945)) ([6147883](https://github.com/headroomlabs-ai/headroom/commit/6147883d5e3a92cc7b890e6c05dce4391090c7e4))
* **wrap:** verify proxy deps before mutating Codex config ([#1628](https://github.com/headroomlabs-ai/headroom/issues/1628)) ([b7f342c](https://github.com/headroomlabs-ai/headroom/commit/b7f342c153a3e6e43a9d3df006bcd4dd69842d00))


### Performance Improvements

* **perf:** skip rotated logs outside the requested window ([#3081](https://github.com/headroomlabs-ai/headroom/issues/3081)) ([6c9f41e](https://github.com/headroomlabs-ai/headroom/commit/6c9f41e08c47f2bfc440c5a4c6ac8a357ad5ada0))


### Dependencies

* bump axum from 0.7.9 to 0.8.9 ([#2966](https://github.com/headroomlabs-ai/headroom/issues/2966)) ([5731be7](https://github.com/headroomlabs-ai/headroom/commit/5731be7e68f57292aed40d76e770657a88f78c13))
* bump criterion from 0.5.1 to 0.8.2 ([#2965](https://github.com/headroomlabs-ai/headroom/issues/2965)) ([b30f339](https://github.com/headroomlabs-ai/headroom/commit/b30f339d694abcd8dada76a34a1d69e30390bfc2))
* bump ruff from 0.15.22 to 0.16.2 in the pip-minor-patch group across 1 directory ([#2962](https://github.com/headroomlabs-ai/headroom/issues/2962)) ([ff17961](https://github.com/headroomlabs-ai/headroom/commit/ff17961cd76a7cea1cff0a9dcfb7338929f37c5a))
* bump sha2 from 0.10.9 to 0.11.0 ([#2288](https://github.com/headroomlabs-ai/headroom/issues/2288)) ([322425c](https://github.com/headroomlabs-ai/headroom/commit/322425c43bffde1ed0b64fecf3cf5951565dd82b))
* bump the cargo-minor-patch group across 1 directory with 4 updates ([#2964](https://github.com/headroomlabs-ai/headroom/issues/2964)) ([888a9f4](https://github.com/headroomlabs-ai/headroom/commit/888a9f4e147cf1f87244977fac81d5e9613352d7))
* bump tokio-tungstenite from 0.24.0 to 0.30.0 ([#2967](https://github.com/headroomlabs-ai/headroom/issues/2967)) ([bbe9013](https://github.com/headroomlabs-ai/headroom/commit/bbe901319d49a3d70caf7b37da2c29f7d7996e07))
* update mcp requirement from &lt;2.0.0,&gt;=1.28.1 to &gt;=1.28.1,&lt;3.0.0 ([#2963](https://github.com/headroomlabs-ai/headroom/issues/2963)) ([d6fb536](https://github.com/headroomlabs-ai/headroom/commit/d6fb5365f67b9b7f90c7c55caead16ca6b41c586))

## [0.35.0](https://github.com/headroomlabs-ai/headroom/compare/v0.34.0...v0.35.0) (2026-08-12)


### Features

* **beacon:** allowlist the routing summary key ([#2818](https://github.com/headroomlabs-ai/headroom/issues/2818)) ([7940c05](https://github.com/headroomlabs-ai/headroom/commit/7940c05ebf4486c6b9d00984067ae33cedf4dddb))
* **beacon:** hourly R2 compaction, per-strategy savings, and a stack that reports ([#2853](https://github.com/headroomlabs-ai/headroom/issues/2853)) ([e0870ef](https://github.com/headroomlabs-ai/headroom/commit/e0870ef931e5ea6cc6cb52551f5d80cd9e3dc715))
* **cli,pricing:** add CLI extension seam and prompt-cache TTL pricing ([#2802](https://github.com/headroomlabs-ai/headroom/issues/2802)) ([6ec3e34](https://github.com/headroomlabs-ai/headroom/commit/6ec3e3478abf058fe1460f91342bcdadf54a1ba8))


### Bug Fixes

* **anthropic:** strip first-party tool search on custom upstreams ([#2539](https://github.com/headroomlabs-ai/headroom/issues/2539)) ([7f6950b](https://github.com/headroomlabs-ai/headroom/commit/7f6950be34e29304deae0fa5138b852491b092fe))
* **backends/anyllm:** convert Anthropic tools and tool_choice to OpenAI shape ([0d6866b](https://github.com/headroomlabs-ai/headroom/commit/0d6866b91a3777475abd58cd8b63a10cd0621e7f))
* **backends/anyllm:** stream tool_use blocks and map finish_reason on the streaming path ([e4904e2](https://github.com/headroomlabs-ai/headroom/commit/e4904e23a6ba6f5cff2488332946481172446922))
* **backends/litellm:** None-guard core token counts in OpenAI usage block ([#2324](https://github.com/headroomlabs-ai/headroom/issues/2324)) ([12f9f58](https://github.com/headroomlabs-ai/headroom/commit/12f9f58cb3dcfc67af1238424d404d8dd9bad1dd))
* **beacon:** report all-layers savings, not context-compression only ([#2796](https://github.com/headroomlabs-ai/headroom/issues/2796)) ([e9a24f3](https://github.com/headroomlabs-ai/headroom/commit/e9a24f3ec1ffd278b0b3ca547a90942c40c99ec8))
* **beacon:** split session failures by status code ([#2815](https://github.com/headroomlabs-ai/headroom/issues/2815)) ([2954e37](https://github.com/headroomlabs-ai/headroom/commit/2954e37048f8dcffe16e1c37b8f71afb0094a0a2))
* **cache:** bound compression cache bookkeeping ([0ae948c](https://github.com/headroomlabs-ai/headroom/commit/0ae948c1510735df39317bf0861f8a8750cdbf9d))
* **cache:** enforce Anthropic's 1h-before-5m cache_control ordering before forwarding ([#2941](https://github.com/headroomlabs-ai/headroom/issues/2941)) ([3752458](https://github.com/headroomlabs-ai/headroom/commit/3752458022f736c779f7b5a6c2d6d2ef0bc89f72))
* **cache:** mirror client cache_control positions instead of single-marker consolidation ([def3d76](https://github.com/headroomlabs-ai/headroom/commit/def3d76e5ab4665e609b51bfba54dd6d25116925))
* **cache:** stabilize Anthropic block-growing lineages ([#2917](https://github.com/headroomlabs-ai/headroom/issues/2917)) ([1a04c95](https://github.com/headroomlabs-ai/headroom/commit/1a04c957f53ef25ab1209166f425a7876913c4d3))
* **ccr:** avoid injecting tool on chat streaming ([d0c1f5b](https://github.com/headroomlabs-ai/headroom/commit/d0c1f5b8ad68c7a44ed3aaa0fe40e3a656950123))
* **ccr:** preserve exact SQLite TTL boundary ([#2669](https://github.com/headroomlabs-ai/headroom/issues/2669)) ([d0a86d4](https://github.com/headroomlabs-ai/headroom/commit/d0a86d409fab377f9c642d1f3680b6ece7f97b8a))
* **ccr:** report embedded hashes from compress endpoint ([#717](https://github.com/headroomlabs-ai/headroom/issues/717)) ([685ebe4](https://github.com/headroomlabs-ai/headroom/commit/685ebe457d727922ba4057515556a2d2aac0f616))
* **ccr:** resolve &lt;&lt;ccr:...&gt;&gt; markers inline when no retrieve-tool path exists ([#2512](https://github.com/headroomlabs-ai/headroom/issues/2512)) ([ce8ce83](https://github.com/headroomlabs-ai/headroom/commit/ce8ce8313f8cebf060392a62f9adaab18c0df386))
* **ccr:** tolerate null/malformed OpenAI data in response handling ([#2467](https://github.com/headroomlabs-ai/headroom/issues/2467)) ([e583e08](https://github.com/headroomlabs-ai/headroom/commit/e583e082d8dee942229ac6211c742f9c9448a905))
* **ci:** publish latest from the root Docker manifest ([#2252](https://github.com/headroomlabs-ai/headroom/issues/2252)) ([5568d73](https://github.com/headroomlabs-ai/headroom/commit/5568d738afb5e080d8df56e64500026996cbf025))
* **claude:** stop forcing tool search on Foundry ([#2477](https://github.com/headroomlabs-ai/headroom/issues/2477)) ([7981396](https://github.com/headroomlabs-ai/headroom/commit/798139608c0fb5118eb3a7a183b8b2abe92341f1))
* **cli/update:** let install ownership win over bare /.dockerenv so venv installs self-update ([#2830](https://github.com/headroomlabs-ai/headroom/issues/2830)) ([7092b53](https://github.com/headroomlabs-ai/headroom/commit/7092b53c466bf5dbda8a1cda88403d1a4b16deb1))
* **codex:** route alpha search through the Codex backend ([#2538](https://github.com/headroomlabs-ai/headroom/issues/2538)) ([a540eb2](https://github.com/headroomlabs-ai/headroom/commit/a540eb2c61b1a47e5ab8b07ea4a80fee780b6514))
* **content-router:** protect custom-tag blocks before mixed-content section split ([d7bc1e2](https://github.com/headroomlabs-ai/headroom/commit/d7bc1e275f411788abffa2d007db14aa17fd31c5))
* **deps:** bump h2 to 4.4.1 for CVE-2026-71554 ([#2839](https://github.com/headroomlabs-ai/headroom/issues/2839)) ([564e0a8](https://github.com/headroomlabs-ai/headroom/commit/564e0a8d0fe440dff21a6c405c88e05698b3059f))
* **deps:** enforce audited transitive dependency floors ([#2791](https://github.com/headroomlabs-ai/headroom/issues/2791)) ([64e2039](https://github.com/headroomlabs-ai/headroom/commit/64e203931b9810e5a010f063d26d154419016f86))
* **doctor:** flag `ollama launch claude` proxy bypass instead of misdirecting ([#2566](https://github.com/headroomlabs-ai/headroom/issues/2566)) ([7f24d69](https://github.com/headroomlabs-ai/headroom/commit/7f24d695eea00b9bb3265fbaa6629acf0c2ff181))
* emit SSE ping before message_start on Bedrock streaming path (issue [#902](https://github.com/headroomlabs-ai/headroom/issues/902)) ([#1080](https://github.com/headroomlabs-ai/headroom/issues/1080)) ([4dab254](https://github.com/headroomlabs-ai/headroom/commit/4dab254d52914c39ffe13071848604e1771b1bd1))
* **gemini:** resolve native CCR retrieval calls ([#2253](https://github.com/headroomlabs-ai/headroom/issues/2253)) ([2483f57](https://github.com/headroomlabs-ai/headroom/commit/2483f570025763cd9183a93749ea8cf38f1aeb85))
* **health:** label kompress as degraded/optional when not yet loaded ([#2865](https://github.com/headroomlabs-ai/headroom/issues/2865)) ([8949371](https://github.com/headroomlabs-ai/headroom/commit/89493714d2cffdc1f81a8f417ea09891453d7009))
* **image:** decouple routing types from trained_router so importing the compressor doesn't import torch ([#2513](https://github.com/headroomlabs-ai/headroom/issues/2513)) ([#2537](https://github.com/headroomlabs-ai/headroom/issues/2537)) ([d7cf981](https://github.com/headroomlabs-ai/headroom/commit/d7cf981093cf505192a3736dadd0254a120830a1))
* **install/windows:** register persistent-task from S4U hidden XML ([#2453](https://github.com/headroomlabs-ai/headroom/issues/2453)) ([#2459](https://github.com/headroomlabs-ai/headroom/issues/2459)) ([1edaeb8](https://github.com/headroomlabs-ai/headroom/commit/1edaeb8b76f6b872a6c810d404c944caf1a594b2))
* **install:** don't crash the PowerShell installer when $PROFILE is unset ([#2469](https://github.com/headroomlabs-ai/headroom/issues/2469)) ([fc5c4e2](https://github.com/headroomlabs-ai/headroom/commit/fc5c4e239ce32f2b90a6777772a01bdf49c66cb6))
* **install:** trust Docker bridge for dashboard metadata ([e044139](https://github.com/headroomlabs-ai/headroom/commit/e044139001680fd5198147bf373df6f00db32cc7))
* **install:** use --userns=keep-id under Podman so bind-mount writes don't fail ([#2846](https://github.com/headroomlabs-ai/headroom/issues/2846)) ([3488f8d](https://github.com/headroomlabs-ai/headroom/commit/3488f8d4b5fae4eab157e0c4031ccf712bcbcc0d))
* **learn/gemini:** stop double-counting session tokens ([#2230](https://github.com/headroomlabs-ai/headroom/issues/2230)) ([29d8a5e](https://github.com/headroomlabs-ai/headroom/commit/29d8a5e563cf16dbd3a53a1571f4f352e61e1b33))
* **learn/grok:** detect a Windows absolute project path ([#2283](https://github.com/headroomlabs-ai/headroom/issues/2283)) ([e240df2](https://github.com/headroomlabs-ai/headroom/commit/e240df2b698e601324b85956bd93cb304f6030ab))
* **learn:** stop classifying a successful exit code 0 as an error ([#2289](https://github.com/headroomlabs-ai/headroom/issues/2289)) ([a24fe7d](https://github.com/headroomlabs-ai/headroom/commit/a24fe7dcbfe5ab30d0cef631c936e2245c12d123))
* **litellm:** add async_post_call_success_hook to HeadroomCallback ([#1322](https://github.com/headroomlabs-ai/headroom/issues/1322)) ([3107994](https://github.com/headroomlabs-ai/headroom/commit/3107994aed5fd42e713d3c26f3f08121a62b980e))
* **litellm:** don't forward a caller key the target cannot accept ([#2883](https://github.com/headroomlabs-ai/headroom/issues/2883)) ([2f2950a](https://github.com/headroomlabs-ai/headroom/commit/2f2950a626cebf851aac29255e7188fbb1639f5a))
* **memory:** bound the TrafficLearner pending-pattern accumulator (memory leak) ([#2579](https://github.com/headroomlabs-ai/headroom/issues/2579)) ([1f5feff](https://github.com/headroomlabs-ai/headroom/commit/1f5fefffd3e82c73bddd928cfd53334031e807bc))
* **memory:** close DirectMem0 resources ([6596182](https://github.com/headroomlabs-ai/headroom/commit/65961827cf5e90d7b4e7026feb89aac000a73ea3))
* **memory:** close MCP backend on shutdown ([4bd8ecd](https://github.com/headroomlabs-ai/headroom/commit/4bd8ecd1e31475365801791d35630f66f7393553))
* **memory:** don't crash inline memory extraction on a non-object &lt;memory&gt; block ([#2470](https://github.com/headroomlabs-ai/headroom/issues/2470)) ([e00c6ff](https://github.com/headroomlabs-ai/headroom/commit/e00c6ff81ce2003e04042b8f2d1bd6aa3c6e885c))
* **memory:** keep vector metadata in sync ([#2295](https://github.com/headroomlabs-ai/headroom/issues/2295)) ([c471800](https://github.com/headroomlabs-ai/headroom/commit/c471800e8ee22986c308464b02a85da5575f34cc))
* **memory:** make explicit-project and user store keys collision-resistant ([#2231](https://github.com/headroomlabs-ai/headroom/issues/2231)) ([f840d5f](https://github.com/headroomlabs-ai/headroom/commit/f840d5f2fe938432e542c3f71f2218eeecd06b05))
* **memory:** skip &lt;system-reminder&gt; blocks when building the retrieval query ([#2195](https://github.com/headroomlabs-ai/headroom/issues/2195)) ([#2541](https://github.com/headroomlabs-ai/headroom/issues/2541)) ([4e5a67a](https://github.com/headroomlabs-ai/headroom/commit/4e5a67a342be4be659b62c7863a9e72422605788))
* **memory:** sync FTS5 and vector indexes on CLI delete/edit/prune/purge ([fd4628d](https://github.com/headroomlabs-ai/headroom/commit/fd4628d82156c65d4fa22df9513315790a6cd2fb))
* **oauth2:** make repository lint checks pass ([c85abf7](https://github.com/headroomlabs-ai/headroom/commit/c85abf7a87920012e01f0a677f6fbd98c4b08de0))
* **observability:** aggregate tool savings in OTEL ([#2936](https://github.com/headroomlabs-ai/headroom/issues/2936)) ([941c25d](https://github.com/headroomlabs-ai/headroom/commit/941c25d31e6c6e0b436c307cbe212771ff76b45f))
* **onnx:** stop ONNX thread pools from spinning idle cores ([#2495](https://github.com/headroomlabs-ai/headroom/issues/2495)) ([#2540](https://github.com/headroomlabs-ai/headroom/issues/2540)) ([5c561bd](https://github.com/headroomlabs-ai/headroom/commit/5c561bd913ea60fad2c3c53f4b65e679e7d248d0))
* **openai:** skip Responses tool-search deferral for clients that cannot execute it ([#2696](https://github.com/headroomlabs-ai/headroom/issues/2696)) ([54ea28d](https://github.com/headroomlabs-ai/headroom/commit/54ea28d9839a0dcfa4dd0cf4210a4421f03beeff))
* **opencode:** ship the transport hook-shim so wheel installs route Node child traffic ([702dbc5](https://github.com/headroomlabs-ai/headroom/commit/702dbc5902ff184a7c20178958a811beb9c78fa3))
* **providers/anthropic:** don't crash token estimation on null tool_calls ([#2472](https://github.com/headroomlabs-ai/headroom/issues/2472)) ([08466f3](https://github.com/headroomlabs-ai/headroom/commit/08466f3cae4dbb2647dc6f249fe42c4e840600c5))
* **providers/openai:** bound tiktoken vocab loads with the guarded loader ([#2554](https://github.com/headroomlabs-ai/headroom/issues/2554)) ([0805e8e](https://github.com/headroomlabs-ai/headroom/commit/0805e8e410543d75c7ddd3b83dde5eda3bc13144))
* **proxy/anthropic:** inject headroom_retrieve whenever a CCR marker is present, not only for new markers ([#2848](https://github.com/headroomlabs-ai/headroom/issues/2848)) ([3808f60](https://github.com/headroomlabs-ai/headroom/commit/3808f60ca61e84faf3ea8f8e003a6e6c8e9af4da))
* **proxy/anthropic:** None-guard usage token counts on the direct buffered path ([#2434](https://github.com/headroomlabs-ai/headroom/issues/2434)) ([2b5ee7c](https://github.com/headroomlabs-ai/headroom/commit/2b5ee7cde809ca37f6998d9679b1eb2133ab50ca))
* **proxy/anthropic:** run tool-search history repair after turn hooks ([c6f9948](https://github.com/headroomlabs-ai/headroom/commit/c6f99482e1bea024db6014a70c8e6da419543957))
* **proxy/batch:** don't crash an OpenAI batch on a valid-JSON non-object line ([#2316](https://github.com/headroomlabs-ai/headroom/issues/2316)) ([1f2c681](https://github.com/headroomlabs-ai/headroom/commit/1f2c681c0b48150a569277d3ebd5e95709dc7c39))
* **proxy/bedrock:** report uncached input tokens from backend usage, not the live-zone count ([#2318](https://github.com/headroomlabs-ai/headroom/issues/2318)) ([c19e412](https://github.com/headroomlabs-ai/headroom/commit/c19e412b3356d80dece001887d4ff48b6fd5150b))
* **proxy/gemini:** keep streaming-parity baseline so eligible_pct can't exceed 100 ([#2824](https://github.com/headroomlabs-ai/headroom/issues/2824)) ([b97c7c6](https://github.com/headroomlabs-ai/headroom/commit/b97c7c6e99eac84df49c7a7e5f21dedb298716fe))
* **proxy/metrics:** cap client-supplied model label cardinality ([#2480](https://github.com/headroomlabs-ai/headroom/issues/2480)) ([e24a7e6](https://github.com/headroomlabs-ai/headroom/commit/e24a7e66b95fa908c4ea6fd079809ece7692e6b2))
* **proxy/metrics:** escape label values in the Prometheus export ([#2463](https://github.com/headroomlabs-ai/headroom/issues/2463)) ([6a53861](https://github.com/headroomlabs-ai/headroom/commit/6a53861063c3839e698bbec7194517bdfd851c38))
* **proxy/openai:** don't crash the Responses memory tool loops on null arguments ([#2273](https://github.com/headroomlabs-ai/headroom/issues/2273)) ([a30db2c](https://github.com/headroomlabs-ai/headroom/commit/a30db2cae49b4ef03ebbd404ec1fc6c4f5f2404d))
* **proxy/openai:** feed Codex WS traffic into the traffic learner ([#2334](https://github.com/headroomlabs-ai/headroom/issues/2334)) ([f669149](https://github.com/headroomlabs-ai/headroom/commit/f6691497692869b7067438597421ff12aace6bf4))
* **proxy/openai:** run response hooks on Responses, and bill their re-drives ([#2872](https://github.com/headroomlabs-ai/headroom/issues/2872)) ([675d13f](https://github.com/headroomlabs-ai/headroom/commit/675d13f08d42455c8fa17bda878c1a11b905cee4))
* **proxy:** allow settings routes for trusted gateway/dashboard clients ([#2491](https://github.com/headroomlabs-ai/headroom/issues/2491)) ([a5b0a8f](https://github.com/headroomlabs-ai/headroom/commit/a5b0a8f4cc54d68afcf371a422b3a4a9635b7e7f))
* **proxy:** cache litellm model resolution to stop repeated Provider List spam ([99f07e7](https://github.com/headroomlabs-ai/headroom/commit/99f07e7bbdded9dadc70e35ee6ab025279d1aa22))
* **proxy:** cancel periodic TOIN task on shutdown ([739fdef](https://github.com/headroomlabs-ai/headroom/commit/739fdef423fa8cbc82537481c875d4570b0ecad4))
* **proxy:** close the upstream stream when a streaming body is never consumed ([0951663](https://github.com/headroomlabs-ai/headroom/commit/09516635621caccf7e3db4f537eb49ea49b8a453))
* **proxy:** compress cache-mode cold starts and tag prefix-mismatch passthrough ([#2365](https://github.com/headroomlabs-ai/headroom/issues/2365)) ([aaeba0a](https://github.com/headroomlabs-ai/headroom/commit/aaeba0a319f12b98cad3bfcf1cf991b694b946bf))
* **proxy:** emit request log timestamps in UTC ([620028f](https://github.com/headroomlabs-ai/headroom/commit/620028fa18843622d3e454bd40fb91a93e607dbf))
* **proxy:** enable tool search by default and repair poisoned transcripts ([#2807](https://github.com/headroomlabs-ai/headroom/issues/2807)) ([0237cbf](https://github.com/headroomlabs-ai/headroom/commit/0237cbffbbc456ad8a7398005602d76881862d99))
* **proxy:** gate mid-turn message coalescing to Claude Code clients ([#1643](https://github.com/headroomlabs-ai/headroom/issues/1643)) ([a4bd2e6](https://github.com/headroomlabs-ai/headroom/commit/a4bd2e62a5bb73f15b3b12e979c69e2b555bee10))
* **proxy:** give each Codex /v1/responses WS turn a unique request_id ([#2164](https://github.com/headroomlabs-ai/headroom/issues/2164)) ([d02df10](https://github.com/headroomlabs-ai/headroom/commit/d02df1075894b414d60626aca2bbcadd7a3577a0))
* **proxy:** graceful shutdown and reliable Ctrl+C exit ([#621](https://github.com/headroomlabs-ai/headroom/issues/621)) ([17cdb18](https://github.com/headroomlabs-ai/headroom/commit/17cdb185bc79d8cfec104e781a7e555af3ef11e1))
* **proxy:** guard telemetry and TOIN endpoints ([cde1513](https://github.com/headroomlabs-ai/headroom/commit/cde1513c91b6c6c240869bc5660f4b8966197bbc))
* **proxy:** include tool_search_deferral savings in the savings ledger ([12149f7](https://github.com/headroomlabs-ai/headroom/commit/12149f74466c08b69be8d5fe751425be63c2fda4))
* **proxy:** pass through cross-region prefixed Bedrock model IDs directly ([#2330](https://github.com/headroomlabs-ai/headroom/issues/2330)) ([64cb46e](https://github.com/headroomlabs-ai/headroom/commit/64cb46e24bf7b223ea71b14b6f5e86e78fa7ac45))
* **proxy:** port session-sticky beta headers to the Rust proxy ([#2381](https://github.com/headroomlabs-ai/headroom/issues/2381)) ([f6398a6](https://github.com/headroomlabs-ai/headroom/commit/f6398a64768a095b722a5fb0b2445c7953dee1c6))
* **proxy:** preserve merged session and quarantine contracts ([#2943](https://github.com/headroomlabs-ai/headroom/issues/2943)) ([039cd24](https://github.com/headroomlabs-ai/headroom/commit/039cd2431aaec7d59fefaf7e97aeda1fd7ab3afa))
* **proxy:** preserve signed Anthropic thinking blocks on outbound re-serialize ([#2254](https://github.com/headroomlabs-ai/headroom/issues/2254)) ([dc163bc](https://github.com/headroomlabs-ai/headroom/commit/dc163bcd1cba4cd8898f23286eb1365fcf6e0356))
* **proxy:** stop discarding compressed Codex WS later-frame payloads ([#2823](https://github.com/headroomlabs-ai/headroom/issues/2823)) ([4ec416d](https://github.com/headroomlabs-ai/headroom/commit/4ec416df8899036544e679f561f1cf921f3da0dd))
* **proxy:** time-cap the compression timeout-debt quarantine ([#2360](https://github.com/headroomlabs-ai/headroom/issues/2360)) ([#2412](https://github.com/headroomlabs-ai/headroom/issues/2412)) ([c5a08d2](https://github.com/headroomlabs-ai/headroom/commit/c5a08d22e05a7dd2b929f3cca76ee3fb42f122db))
* **proxy:** unwrap Hermes tool_call bridge in tool name map ([#2717](https://github.com/headroomlabs-ai/headroom/issues/2717)) ([a97b824](https://github.com/headroomlabs-ai/headroom/commit/a97b82413bdc86655c064417ed4628ff4d9d7c9d))
* publish headroom-opencode in release workflow ([#2372](https://github.com/headroomlabs-ai/headroom/issues/2372)) ([7859154](https://github.com/headroomlabs-ai/headroom/commit/78591545ceb8303fdf9b93cd5ff02b626df97d2b))
* **settings:** accept documented HEADROOM_* env names as settings keys ([#2833](https://github.com/headroomlabs-ai/headroom/issues/2833)) ([de9e052](https://github.com/headroomlabs-ai/headroom/commit/de9e0523dad47b700062464adecd60f82547f332))
* **subscription:** dedup transcript usage by message id ([#2340](https://github.com/headroomlabs-ai/headroom/issues/2340) token inflation) ([#2408](https://github.com/headroomlabs-ai/headroom/issues/2408)) ([74275b7](https://github.com/headroomlabs-ai/headroom/commit/74275b7c3e2b39be5198f9efa35057a5e026e665))
* **toin:** bound private query and pattern retention ([8cd1380](https://github.com/headroomlabs-ai/headroom/commit/8cd138039edbfc295080ec474325d527fb3aedf3))
* **tokenizer:** coerce non-string tool_call fields before counting ([#2801](https://github.com/headroomlabs-ai/headroom/issues/2801)) ([b6f9877](https://github.com/headroomlabs-ai/headroom/commit/b6f9877c78b3fa3b1d705426bd27d74be77f4fa0))
* **tokenizer:** price CJK in the Rust fixed-ratio estimator (Python parity) ([#2260](https://github.com/headroomlabs-ai/headroom/issues/2260)) ([6840153](https://github.com/headroomlabs-ai/headroom/commit/6840153473caa0d61e982215e16a8cf54b0b6cc7))
* **transforms/adaptive-sizer:** honor max_k on small-input fast path ([#2319](https://github.com/headroomlabs-ai/headroom/issues/2319)) ([8a90523](https://github.com/headroomlabs-ai/headroom/commit/8a905232091d993fac9e19a59bc449f201d4cdf3))
* **transforms/smart_crusher:** don't crash on a tool call with a null function ([#2232](https://github.com/headroomlabs-ai/headroom/issues/2232)) ([3bb02f8](https://github.com/headroomlabs-ai/headroom/commit/3bb02f8f75f12cf8258a5b1c2a7fbdc190f9d074))
* Vertex model pricing shows $0.00 for versioned model names and vertex:anthropic provider ([#2517](https://github.com/headroomlabs-ai/headroom/issues/2517)) ([eb5b5e4](https://github.com/headroomlabs-ai/headroom/commit/eb5b5e41988f5c27d29ae8ae3e5fe74e56493b8c))
* **wrap/claude:** keep --1m effective when an explicit --model is passed through ([c093bf1](https://github.com/headroomlabs-ai/headroom/commit/c093bf11eb5f356f71367ebb7b56ae3c2b434a12))
* **wrap/opencode:** verify the opencode binary before mutating config ([ae38486](https://github.com/headroomlabs-ai/headroom/commit/ae384862a4950cec057103e9daf75e74107640df))
* **wrap/serena:** install Serena from the serena-agent PyPI wheel, not the git source ([d7b25ae](https://github.com/headroomlabs-ai/headroom/commit/d7b25ae3bb3364cde4931509ecb65e32085e5b09))
* **wrap:** honor Copilot OAuth wire-api override and model default ([#2387](https://github.com/headroomlabs-ai/headroom/issues/2387)) ([1db6d88](https://github.com/headroomlabs-ai/headroom/commit/1db6d88ab4ea25654b8277358902b7df700db6b4))
* **wrap:** serialize shared proxy startup ([#2946](https://github.com/headroomlabs-ai/headroom/issues/2946)) ([e540d64](https://github.com/headroomlabs-ai/headroom/commit/e540d64febf27f2e7997d3a1a1d89478cc1ef658))
* **wrap:** stop the launch cwd from shadowing the installed package in the proxy subprocess ([#2843](https://github.com/headroomlabs-ai/headroom/issues/2843)) ([c49be26](https://github.com/headroomlabs-ai/headroom/commit/c49be269a18446779cd8a048caaa7f0ba3a3b48b))


### Performance Improvements

* cut hot-path latency 27% (token-count memo, startup preloads, JSON scan memo) ([#2838](https://github.com/headroomlabs-ai/headroom/issues/2838)) ([53af90d](https://github.com/headroomlabs-ai/headroom/commit/53af90d68c723f644a5a41dd273a606117109866))
* **proxy:** bound upstream calls and hot-path costs ([#2852](https://github.com/headroomlabs-ai/headroom/issues/2852)) ([f624d3a](https://github.com/headroomlabs-ai/headroom/commit/f624d3a00ac271db7947443ddeb0c8bc2e93d3eb))
* **subscription:** skip transcripts older than the window in compute_window_tokens ([#2861](https://github.com/headroomlabs-ai/headroom/issues/2861)) ([91d6bf3](https://github.com/headroomlabs-ai/headroom/commit/91d6bf33cde777b541375fb182d4479fdd78f81b))


### Dependencies

* bump brace-expansion from 5.0.7 to 5.0.9 in /docs ([#2751](https://github.com/headroomlabs-ai/headroom/issues/2751)) ([56ee57b](https://github.com/headroomlabs-ai/headroom/commit/56ee57be98bf109f0a46de522724ef169a4bc51c))
* bump bytesize from 1.3.3 to 2.4.2 ([#2286](https://github.com/headroomlabs-ai/headroom/issues/2286)) ([6448545](https://github.com/headroomlabs-ai/headroom/commit/6448545a7f5a1dee88bce6f0830bdbfd1c99c617))
* bump hf-hub from 0.4.3 to 0.5.0 ([#2285](https://github.com/headroomlabs-ai/headroom/issues/2285)) ([4925bf6](https://github.com/headroomlabs-ai/headroom/commit/4925bf6a829735977bab5000b469c3edb19c75b1))
* bump next from 16.2.10 to 16.3.0 in /docs ([#2750](https://github.com/headroomlabs-ai/headroom/issues/2750)) ([0fd0b99](https://github.com/headroomlabs-ai/headroom/commit/0fd0b996a4b58a166491b145f4d3885c21b27cc0))
* bump postcss from 8.5.19 to 8.5.25 in /plugins/openclaw ([#2749](https://github.com/headroomlabs-ai/headroom/issues/2749)) ([cd60ee9](https://github.com/headroomlabs-ai/headroom/commit/cd60ee9ae886b32ba5da3203e35bb6b088031fd3))
* bump postcss from 8.5.19 to 8.5.25 in /plugins/opencode ([#2748](https://github.com/headroomlabs-ai/headroom/issues/2748)) ([ff4e016](https://github.com/headroomlabs-ai/headroom/commit/ff4e0167bbccbd4ae51bf23ddec144e61c94cd68))
* bump postcss from 8.5.19 to 8.5.25 in /sdk/typescript ([#2747](https://github.com/headroomlabs-ai/headroom/issues/2747)) ([267c2bd](https://github.com/headroomlabs-ai/headroom/commit/267c2bdcb56e132b2dd9c065dab3498dbf730ca3))
* bump postcss from 8.5.19 to 8.5.26 in /docs ([#2881](https://github.com/headroomlabs-ai/headroom/issues/2881)) ([e6e5826](https://github.com/headroomlabs-ai/headroom/commit/e6e5826423a0a700a8c544ce2c8cbcdef694160e))
* bump ruff from 0.15.17 to 0.15.22 in the pip-minor-patch group ([#2501](https://github.com/headroomlabs-ai/headroom/issues/2501)) ([ecf130d](https://github.com/headroomlabs-ai/headroom/commit/ecf130d3ac6fb864098cb93fafd2621ae3ac7e12))
* bump rusqlite from 0.32.1 to 0.40.1 ([#2287](https://github.com/headroomlabs-ai/headroom/issues/2287)) ([522faa1](https://github.com/headroomlabs-ai/headroom/commit/522faa1a59aa94e4adfd4a4afe0202d1126e187d))
* bump the cargo-minor-patch group across 1 directory with 22 updates ([#2916](https://github.com/headroomlabs-ai/headroom/issues/2916)) ([148d860](https://github.com/headroomlabs-ai/headroom/commit/148d8605e2087f3c8d6a3fa4b8d248ad2da5858f))

## [0.34.0](https://github.com/headroomlabs-ai/headroom/compare/v0.33.0...v0.34.0) (2026-08-05)


### Features

* **claude:** support Claude Code in VS Code ([#2752](https://github.com/headroomlabs-ai/headroom/issues/2752)) ([13a310a](https://github.com/headroomlabs-ai/headroom/commit/13a310a00de8e967ebe09502c6b715ef577c5926))
* **code:** add PHP support to CodeAwareCompressor ([#2423](https://github.com/headroomlabs-ai/headroom/issues/2423)) ([6d5516d](https://github.com/headroomlabs-ai/headroom/commit/6d5516dcb878b6ffd139a1c7b3d480a1c8c1beb9))
* **compress:** accept config.frozen_message_count on /v1/compress ([#2718](https://github.com/headroomlabs-ai/headroom/issues/2718)) ([2797099](https://github.com/headroomlabs-ai/headroom/commit/2797099becbd078e55b8a73cf904d2e3cb0d6889))
* **compress:** reach the lossless provider seam on the general path and default /v1/compress to marker-free output ([#2691](https://github.com/headroomlabs-ai/headroom/issues/2691)) ([f2c48e2](https://github.com/headroomlabs-ai/headroom/commit/f2c48e26c684a31e2802de9f49ce2075ef9cbf4b))
* **copilot:** proxy VS Code models transparently ([#2687](https://github.com/headroomlabs-ai/headroom/issues/2687)) ([007446c](https://github.com/headroomlabs-ai/headroom/commit/007446c73a26efa729bf6d6903c828adef730089))


### Bug Fixes

* **ccr:** stop persisting retrieval markers as original content ([#2694](https://github.com/headroomlabs-ai/headroom/issues/2694)) ([#2703](https://github.com/headroomlabs-ai/headroom/issues/2703)) ([3e348f3](https://github.com/headroomlabs-ai/headroom/commit/3e348f327f05921204329b72a57d3113cf5101c4))
* **ci:** restrict Codecov shard uploads ([#2745](https://github.com/headroomlabs-ai/headroom/issues/2745)) ([3f2ca99](https://github.com/headroomlabs-ai/headroom/commit/3f2ca99fe16668e3d50b8e1706182ec7b226c352))
* **compression:** honor qualified CCR names across integrations ([#2698](https://github.com/headroomlabs-ai/headroom/issues/2698)) ([dcb674b](https://github.com/headroomlabs-ai/headroom/commit/dcb674b5e4e0d29d52672118ba3cf5062b16d280))
* **compress:** resolve the /v1/compress tokenizer per model, and document the real contract ([#2743](https://github.com/headroomlabs-ai/headroom/issues/2743)) ([6422a80](https://github.com/headroomlabs-ai/headroom/commit/6422a80a58010da805d4001e83265300aa716d8a))
* **cost:** send litellm the total prompt so --budget stops seeing $0 ([#2757](https://github.com/headroomlabs-ai/headroom/issues/2757)) ([a033ac4](https://github.com/headroomlabs-ai/headroom/commit/a033ac4176b09c716905aa0f45ae317e954f0eb9))
* **deps:** bump aiohttp and cryptography to clear the CVEs blocking 0.34.0 ([#2753](https://github.com/headroomlabs-ai/headroom/issues/2753)) ([0221e7f](https://github.com/headroomlabs-ai/headroom/commit/0221e7f240cf470628650d749dc0ab5f3f0135f3))
* **kompress:** let orgs run Kompress on their own inference stack ([#2736](https://github.com/headroomlabs-ai/headroom/issues/2736)) ([3d23d76](https://github.com/headroomlabs-ai/headroom/commit/3d23d76248d2052b846a84b70be87c8c95bad9ac))
* **kompress:** load merged.pt for the v2 checkpoint instead of the unmerged PEFT safetensors ([#2716](https://github.com/headroomlabs-ai/headroom/issues/2716)) ([46da91b](https://github.com/headroomlabs-ai/headroom/commit/46da91b2f1370b6b4910ae8a4ad0613929803887))
* **kompress:** reject artifacts that fail at run, and prefetch model files at startup ([#2740](https://github.com/headroomlabs-ai/headroom/issues/2740)) ([224578e](https://github.com/headroomlabs-ai/headroom/commit/224578e80b4abbe1e16f1952efc24af5fdee106a))
* **learn:** filter ambient user-role scaffolding ([#2275](https://github.com/headroomlabs-ai/headroom/issues/2275)) ([3eb0122](https://github.com/headroomlabs-ai/headroom/commit/3eb01220683d65660544c07631b1efb4781e1d53))
* **learn:** run project discovery off the event loop ([#2731](https://github.com/headroomlabs-ai/headroom/issues/2731)) ([a70e5ff](https://github.com/headroomlabs-ai/headroom/commit/a70e5ff78dc9486e63a6563f122d392469ceef38))
* normalize /p/&lt;project&gt; prefix on WebSocket upgrades so the Responses WS route is not rejected with 403 ([#2379](https://github.com/headroomlabs-ai/headroom/issues/2379)) ([789a4f3](https://github.com/headroomlabs-ai/headroom/commit/789a4f3060aa33a5bae82680968c3e367fa2db83))
* **providers:** give every model exactly one tokenizer ([#2761](https://github.com/headroomlabs-ai/headroom/issues/2761)) ([cd92ed5](https://github.com/headroomlabs-ai/headroom/commit/cd92ed52ff80ee7932600306349fd5d6601b5404))
* **providers:** stop a shorter model family shadowing a longer one ([#2762](https://github.com/headroomlabs-ai/headroom/issues/2762)) ([0cb72f4](https://github.com/headroomlabs-ai/headroom/commit/0cb72f45b23bdf7129822b16dd1d4cb7d6e0b062))
* **providers:** stop pricing modern content blocks at zero ([#2760](https://github.com/headroomlabs-ai/headroom/issues/2760)) ([06add9e](https://github.com/headroomlabs-ai/headroom/commit/06add9e9d833783c1144316c0cb3cb142377d897))
* **proxy/cost:** mark estimated-basis budget records and add an enforcement policy ([#2713](https://github.com/headroomlabs-ai/headroom/issues/2713)) ([#2725](https://github.com/headroomlabs-ai/headroom/issues/2725)) ([01df245](https://github.com/headroomlabs-ai/headroom/commit/01df2452529a86c689cf226fecd5918cc5d19676))
* **proxy/debug:** reconcile Kompress warmup state in /debug/warmup ([#2711](https://github.com/headroomlabs-ai/headroom/issues/2711)) ([3a27c4d](https://github.com/headroomlabs-ai/headroom/commit/3a27c4dacb08a006ca5aa71e8e7728b230c7283f))
* **proxy/openai:** run tool-description compaction on chat-completions ([#2741](https://github.com/headroomlabs-ai/headroom/issues/2741)) ([f9db5b5](https://github.com/headroomlabs-ai/headroom/commit/f9db5b506030a0e8557af8a350f3806464f8ff15))
* **proxy:** route Codex Live voice through a dedicated /v1/live transport ([#2709](https://github.com/headroomlabs-ai/headroom/issues/2709)) ([232fb49](https://github.com/headroomlabs-ai/headroom/commit/232fb49c733122652528edcf3c500f365df265c4))
* **proxy:** skip OpenAI tool_search deferral for Codex client ([#2729](https://github.com/headroomlabs-ai/headroom/issues/2729)) ([56b3e4c](https://github.com/headroomlabs-ai/headroom/commit/56b3e4c1b1e3513c409242b30e7712514f2624d5))
* **proxy:** stop toggling headroom_retrieve in the Anthropic tools array ([#2672](https://github.com/headroomlabs-ai/headroom/issues/2672)) ([08fce29](https://github.com/headroomlabs-ai/headroom/commit/08fce29b4750a79fb2fbc3969847bb38f35e29b3))
* remove rtk and lean-ctx CLI context tools ([#2677](https://github.com/headroomlabs-ai/headroom/issues/2677)) ([e0ce4b1](https://github.com/headroomlabs-ai/headroom/commit/e0ce4b1d4817e1b352e68e8b316273d863260ba7))
* **router:** stop counting an image's base64 payload as suffix tokens ([#2778](https://github.com/headroomlabs-ai/headroom/issues/2778)) ([f03cc6d](https://github.com/headroomlabs-ai/headroom/commit/f03cc6d88b826c2752b20bdce944f9ad1e507e83))
* **savings:** surface request growth the tok_saved clamp swallows ([#2708](https://github.com/headroomlabs-ai/headroom/issues/2708)) ([184146b](https://github.com/headroomlabs-ai/headroom/commit/184146b6884b7b0e4c589c5ee414f96bf56d867f))
* **stats:** report one "Tokens Saved" headline across every harness ([#2737](https://github.com/headroomlabs-ai/headroom/issues/2737)) ([8262a4a](https://github.com/headroomlabs-ai/headroom/commit/8262a4a3217bf6125f293bacc1df9ae21f63264d))
* **telemetry:** anonymous compression stats — no prompts, no data ([#2728](https://github.com/headroomlabs-ai/headroom/issues/2728)) ([9cfb008](https://github.com/headroomlabs-ai/headroom/commit/9cfb00838a197159d94aa52bc042df1a754b7984))
* **telemetry:** stop mixing tokenizer scales in RequestOutcome, and fix the overhead framing ([#2756](https://github.com/headroomlabs-ai/headroom/issues/2756)) ([04e1517](https://github.com/headroomlabs-ai/headroom/commit/04e1517ede0a17ffa950a9531f210e31d236c660))
* **tokenizers:** count HuggingFace chat templates, and resolve gpt-5 / gateway-wrapped names ([#2758](https://github.com/headroomlabs-ai/headroom/issues/2758)) ([0ed306b](https://github.com/headroomlabs-ai/headroom/commit/0ed306b22bf61bfaa421991aea0bd562fc97d910))
* **tokenizers:** resolve gpt-5 and mixed-case model names to the right encoding ([#2776](https://github.com/headroomlabs-ai/headroom/issues/2776)) ([fc4680b](https://github.com/headroomlabs-ai/headroom/commit/fc4680b37af1d522fdbeba8e5d3228769dc49ba4))
* **transforms:** stop ContentRouter recompressing headroom_retrieve results ([#2654](https://github.com/headroomlabs-ai/headroom/issues/2654)) ([677e097](https://github.com/headroomlabs-ai/headroom/commit/677e09735a41f6c37dedc842ab3c214b5bddeafc))
* **wrap/serena:** stop creating serena_config.yml, unbricking Serena on fresh installs ([#2676](https://github.com/headroomlabs-ai/headroom/issues/2676)) ([759209c](https://github.com/headroomlabs-ai/headroom/commit/759209cff3daa72dd9d47e57568e731d10573d63))


### Code Refactoring

* **pricing:** make LiteLLM the source of truth, not the hardcoded table ([#2779](https://github.com/headroomlabs-ai/headroom/issues/2779)) ([0e1d6bf](https://github.com/headroomlabs-ai/headroom/commit/0e1d6bfa797d865834cc247989115a11949ce3f5))
* remove the dead headroom/prediction module ([#2692](https://github.com/headroomlabs-ai/headroom/issues/2692)) ([b7a79ac](https://github.com/headroomlabs-ai/headroom/commit/b7a79ac31a99ec67dc5fbe7bd15e7b96f8c040ec))

## [0.33.0](https://github.com/headroomlabs-ai/headroom/compare/v0.32.0...v0.33.0) (2026-07-29)


### Features

* **lossless:** factor shared directory prefix in the grep search fold ([#2547](https://github.com/headroomlabs-ai/headroom/issues/2547)) ([7dc9a97](https://github.com/headroomlabs-ai/headroom/commit/7dc9a978ca974a2ed264bb585b187dd11e0a04f2))
* **metrics:** record per-extension token savings ([#2371](https://github.com/headroomlabs-ai/headroom/issues/2371)) ([02eb90f](https://github.com/headroomlabs-ai/headroom/commit/02eb90f24318abdfb05438e873c8f2af7023ab91))
* **opencode:** ship the transport plugin in pip installs ([#2601](https://github.com/headroomlabs-ai/headroom/issues/2601)) ([f54f04f](https://github.com/headroomlabs-ai/headroom/commit/f54f04f5bfff9ff9f9ec83b452f580447c06254a))
* **opencode:** support Copilot subscription backend for headroom models ([#2441](https://github.com/headroomlabs-ai/headroom/issues/2441)) ([#2445](https://github.com/headroomlabs-ai/headroom/issues/2445)) ([9089e7f](https://github.com/headroomlabs-ai/headroom/commit/9089e7f7d394b5a474cc99503b0197c0172f4c9c))
* **proxy/hooks:** run fold-only (stream-safe) turn hooks on streaming OpenAI chat ([#2549](https://github.com/headroomlabs-ai/headroom/issues/2549)) ([a6d4921](https://github.com/headroomlabs-ai/headroom/commit/a6d4921e82c1e9fe1a5ca8b90ffd16aa84a698d4))
* **proxy/savings:** aggregate tool-schema savings into Metrics + all reporting sinks ([#2546](https://github.com/headroomlabs-ai/headroom/issues/2546)) ([9f1ffef](https://github.com/headroomlabs-ai/headroom/commit/9f1ffefe83845a3af0ecd8013daa732c3cd56b7c))
* **proxy:** label GitHub Copilot traffic as "copilot" in the outcome… ([#2377](https://github.com/headroomlabs-ai/headroom/issues/2377)) ([d7a8cdb](https://github.com/headroomlabs-ai/headroom/commit/d7a8cdbee1c500be35b87c9da8395087a37ff8b9))
* **proxy:** make /v1/compress usable as a gateway/Kong sidecar ([#2458](https://github.com/headroomlabs-ai/headroom/issues/2458)) ([1329ed7](https://github.com/headroomlabs-ai/headroom/commit/1329ed7f1a8d7a018042ecbe41804b0be971792e))
* **proxy:** model-aware cold-prefix hook — reasoning compaction (Kimi/GLM) + cold recompaction (CC) ([#2555](https://github.com/headroomlabs-ai/headroom/issues/2555)) ([cb8f4b6](https://github.com/headroomlabs-ai/headroom/commit/cb8f4b64367f8b034315db33e451bdbe87af61f2))
* **proxy:** route selected external compressors through the content router ([#2388](https://github.com/headroomlabs-ai/headroom/issues/2388)) ([e3c7964](https://github.com/headroomlabs-ai/headroom/commit/e3c7964038116a8df4675840896712e1aa967c45))
* **proxy:** select built-in compressors via --compressor + registry inventory ([#2373](https://github.com/headroomlabs-ai/headroom/issues/2373)) ([56c7d4a](https://github.com/headroomlabs-ai/headroom/commit/56c7d4a59e67655cd24040ecf729382c81cdec23))
* **rust:** add structured prose offload plumbing ([#334](https://github.com/headroomlabs-ai/headroom/issues/334)) ([#2378](https://github.com/headroomlabs-ai/headroom/issues/2378)) ([9e07785](https://github.com/headroomlabs-ai/headroom/commit/9e0778553fc505edb2c5bc949b7277f9ffdf3bda))
* **rust:** port CodeCompressor AST compressor to Rust (parity-only) ([#1154](https://github.com/headroomlabs-ai/headroom/issues/1154)) ([e530de5](https://github.com/headroomlabs-ai/headroom/commit/e530de5ad22100bcfaa12a463961dcb08d9671c8))
* **rust:** port Kompress ML prose compressor to Rust (parity-only) ([#1153](https://github.com/headroomlabs-ai/headroom/issues/1153)) ([83e27e5](https://github.com/headroomlabs-ai/headroom/commit/83e27e50360753cf472acb99f1de992574fa80ae))
* **telemetry:** record provider cache read/write/uncached tokens per request ([#2450](https://github.com/headroomlabs-ai/headroom/issues/2450)) ([bec4cce](https://github.com/headroomlabs-ai/headroom/commit/bec4cce8a9f5623e63dba0a847719a652b47d5dc))
* **transforms:** add compressed signal + dispatch code_aware/html/diff via registry ([#2400](https://github.com/headroomlabs-ai/headroom/issues/2400)) ([7ebda67](https://github.com/headroomlabs-ai/headroom/commit/7ebda67ef65fe82803c7fb729c509a1451165f26))
* **transforms:** add pluggable compressor registry + headroom.compressor entry point ([#2370](https://github.com/headroomlabs-ai/headroom/issues/2370)) ([a02073e](https://github.com/headroomlabs-ai/headroom/commit/a02073e3327365a0220ba04eeb10039f12d61684))
* **transforms:** dispatch kompress/text via the compressor registry + forward question ([#2411](https://github.com/headroomlabs-ai/headroom/issues/2411)) ([446ec26](https://github.com/headroomlabs-ai/headroom/commit/446ec26003c8f661cec175a69e0ab8be0ae9cdea))
* **transforms:** dispatch smart_crusher via the compressor registry (defer kompress/text ML boundary) ([#2404](https://github.com/headroomlabs-ai/headroom/issues/2404)) ([7c7bf43](https://github.com/headroomlabs-ai/headroom/commit/7c7bf430576541d0fffdb8fc727b76f3dd038f55))
* **transforms:** make built-in compressors real Compressor implementations (adapters) ([#2391](https://github.com/headroomlabs-ai/headroom/issues/2391)) ([981616c](https://github.com/headroomlabs-ai/headroom/commit/981616c60ef04c32b3eb5b51c4f0f4a7ef297ef1))
* **wrap:** boost Serena — symbol-first guidance, wrap-time pre-index, repo-language scoping ([#2425](https://github.com/headroomlabs-ai/headroom/issues/2425)) ([fd0e1a8](https://github.com/headroomlabs-ai/headroom/commit/fd0e1a8afeb60748f65fef8b9197ec95e23b335a))
* **wrap:** default code-memory to Serena (dashboard browser off) behind unified --code-memory ([#2413](https://github.com/headroomlabs-ai/headroom/issues/2413)) ([6e4425a](https://github.com/headroomlabs-ai/headroom/commit/6e4425a6bdb2bfc49e1633a24b9c9e96e705e1ff))
* **wrap:** reduce-at-source — SAFE quiet-CLI env defaults for the launched agent ([#2548](https://github.com/headroomlabs-ai/headroom/issues/2548)) ([c990cfb](https://github.com/headroomlabs-ai/headroom/commit/c990cfb8037e8f355c82eb1cef87f5c4297b612d))


### Bug Fixes

* **backends/litellm:** guard None completion_tokens in usage mapping ([#2322](https://github.com/headroomlabs-ai/headroom/issues/2322)) ([44a174f](https://github.com/headroomlabs-ai/headroom/commit/44a174fef4d514eceed20a767dc87d00cfde0eaa))
* **backends:** don't crash the OpenAI-&gt;Anthropic converter on empty choices ([#2484](https://github.com/headroomlabs-ai/headroom/issues/2484)) ([43a7b57](https://github.com/headroomlabs-ai/headroom/commit/43a7b578a1377ad34d8a78ba3bcef1c276db0b4d))
* **cache:** preserve cache_control ttl when re-anchoring a breakpoint ([#2651](https://github.com/headroomlabs-ai/headroom/issues/2651)) ([e0d2cd0](https://github.com/headroomlabs-ai/headroom/commit/e0d2cd0c5a1c3ee813ac225252c9fd8db7c77c12))
* **cache:** preserve client cache_control ttl when consolidating breakpoints ([#2382](https://github.com/headroomlabs-ai/headroom/issues/2382)) ([8906d3a](https://github.com/headroomlabs-ai/headroom/commit/8906d3a6761c097bbc9d92a0b41f8c982afc633b))
* **ccr:** guard empty/malformed OpenAI choices in _extract_assistant_message ([#2389](https://github.com/headroomlabs-ai/headroom/issues/2389)) ([89319fb](https://github.com/headroomlabs-ai/headroom/commit/89319fbcaddb4be2ea11e87858ed3bd0fcf9dca5))
* **ccr:** sliding idle-window TTL with max-lifetime ceiling in the Rust core backends ([#2604](https://github.com/headroomlabs-ai/headroom/issues/2604)) ([#2631](https://github.com/headroomlabs-ai/headroom/issues/2631)) ([e825588](https://github.com/headroomlabs-ai/headroom/commit/e825588bfbc59fa9e86085e23b4a078e9a0038ba))
* **ci:** align Ruff tooling versions ([#2406](https://github.com/headroomlabs-ai/headroom/issues/2406)) ([2bb14d1](https://github.com/headroomlabs-ai/headroom/commit/2bb14d1ab24617971a657b71ead567479021119d))
* **cli:** warn when Headroom proxy URL leaks into the shell after unwrap claude ([#2238](https://github.com/headroomlabs-ai/headroom/issues/2238)) ([#2571](https://github.com/headroomlabs-ai/headroom/issues/2571)) ([904bc67](https://github.com/headroomlabs-ai/headroom/commit/904bc675b35072dc61191963cbe485fa692927d1))
* **codex:** detect keyring-backed ChatGPT auth ([#2478](https://github.com/headroomlabs-ai/headroom/issues/2478)) ([46293f4](https://github.com/headroomlabs-ai/headroom/commit/46293f4daf4d217ab6f8a83f7c571571b79bae0c))
* **compression:** report source-line span in CCR compression marker ([#2597](https://github.com/headroomlabs-ai/headroom/issues/2597)) ([18e1c3c](https://github.com/headroomlabs-ai/headroom/commit/18e1c3c9badc5169466b7f76ae08e0639f4ba104))
* **copilot:** derive GHE credential host from API URL ([#800](https://github.com/headroomlabs-ai/headroom/issues/800)) ([#2511](https://github.com/headroomlabs-ai/headroom/issues/2511)) ([4a8157f](https://github.com/headroomlabs-ai/headroom/commit/4a8157fa0a3f1d07699f1071ceb653f8902f10a4))
* **copilot:** normalize subscription API routing ([#2441](https://github.com/headroomlabs-ai/headroom/issues/2441)) ([#2455](https://github.com/headroomlabs-ai/headroom/issues/2455)) ([2eca5ee](https://github.com/headroomlabs-ai/headroom/commit/2eca5ee1140c9ce0a5fee05e604d3198f7f86026))
* **copilot:** preserve /v1 for the Anthropic /v1/messages endpoint ([#2409](https://github.com/headroomlabs-ai/headroom/issues/2409)) ([#2414](https://github.com/headroomlabs-ai/headroom/issues/2414)) ([c400f90](https://github.com/headroomlabs-ai/headroom/commit/c400f9081052f633e4e64ad70b95a0230dc6fb3d))
* **deps:** bump mcp to 1.28.1 to clear 3 high-severity CVEs ([#2348](https://github.com/headroomlabs-ai/headroom/issues/2348)) ([a90be94](https://github.com/headroomlabs-ai/headroom/commit/a90be94e32c393332d37db4fb439e0c776b89f27))
* **grok:** preserve business-seat auth while routing only inference ([#2514](https://github.com/headroomlabs-ai/headroom/issues/2514)) ([e4076bb](https://github.com/headroomlabs-ai/headroom/commit/e4076bbe99d500982b51444fe37f8f467cd6abe2))
* **image:** reuse image models instead of rebuilding them per request ([#2513](https://github.com/headroomlabs-ai/headroom/issues/2513)) ([#2536](https://github.com/headroomlabs-ai/headroom/issues/2536)) ([2a63ec7](https://github.com/headroomlabs-ai/headroom/commit/2a63ec70b65605dfcff1b0afc292ab0298459f20))
* **install:** carry upstream-routing env overrides into supervised deployments ([#2429](https://github.com/headroomlabs-ai/headroom/issues/2429)) ([170b04a](https://github.com/headroomlabs-ai/headroom/commit/170b04a74d5361cdfac4a6e265f5ea0dfecbd841))
* **install:** default to cache mode, matching `headroom proxy` ([#1893](https://github.com/headroomlabs-ai/headroom/issues/1893) follow-up) ([#2563](https://github.com/headroomlabs-ai/headroom/issues/2563)) ([b121223](https://github.com/headroomlabs-ai/headroom/commit/b121223ec97e95c5a7a4c2c5e06a4655c7328e88))
* **install:** migrate deployments off the retired chopratejas image repo ([#2427](https://github.com/headroomlabs-ai/headroom/issues/2427)) ([17ff13c](https://github.com/headroomlabs-ai/headroom/commit/17ff13ccbe274e831d5d9327740cd6d506ea8c1c))
* **install:** use CREATE_NO_WINDOW instead of DETACHED_PROCESS on Windows ([#2527](https://github.com/headroomlabs-ai/headroom/issues/2527)) ([045f3df](https://github.com/headroomlabs-ai/headroom/commit/045f3dfe6fd9f4e39e4cdd8c0c529a815d925c7e))
* **kompress:** raise the default execution-slot wait ([#2456](https://github.com/headroomlabs-ai/headroom/issues/2456)) ([5bd2266](https://github.com/headroomlabs-ai/headroom/commit/5bd2266f16bb351a7a7334e1c29c598d28187b1d))
* **learn:** detect the active OpenCode database ([#2587](https://github.com/headroomlabs-ai/headroom/issues/2587)) ([f74d874](https://github.com/headroomlabs-ai/headroom/commit/f74d87477701f1f95bd4709c4727f3d3890a4e22))
* **learn:** keep traceback tail in tool-error digest preview ([#2596](https://github.com/headroomlabs-ai/headroom/issues/2596)) ([85e8699](https://github.com/headroomlabs-ai/headroom/commit/85e869945138f06471501046c5725eac119dea58))
* **learn:** treat unreadable candidate paths as absent in project decode ([#2446](https://github.com/headroomlabs-ai/headroom/issues/2446)) ([a09ba6c](https://github.com/headroomlabs-ai/headroom/commit/a09ba6c08723618dba5f282a9beac78c9406edbf))
* **mcp:** pin mcp dependency to &lt;2.0.0 to prevent server startup crash ([#2642](https://github.com/headroomlabs-ai/headroom/issues/2642)) ([b3f016b](https://github.com/headroomlabs-ai/headroom/commit/b3f016b866375cfe2ff8518055ab93844e11ec27))
* **proxy/cost:** count Gemini thinking tokens in output usage ([#2639](https://github.com/headroomlabs-ai/headroom/issues/2639)) ([22b707f](https://github.com/headroomlabs-ai/headroom/commit/22b707fd31d75914e1677290d2a8011727eb74f5))
* **proxy/cost:** record each request's savings exactly once (drop 3 double-counts) ([#2545](https://github.com/headroomlabs-ai/headroom/issues/2545)) ([0845b26](https://github.com/headroomlabs-ai/headroom/commit/0845b26ee61c507487cd8476cfabe8284f59402b))
* **proxy/cost:** warn once per model when pricing lookup fails ([#2504](https://github.com/headroomlabs-ai/headroom/issues/2504)) ([#2535](https://github.com/headroomlabs-ai/headroom/issues/2535)) ([fa47637](https://github.com/headroomlabs-ai/headroom/commit/fa4763761b5912cccde95903f4b9a681b555465b))
* **proxy/gemini:** None-guard token counts from usageMetadata ([#2347](https://github.com/headroomlabs-ai/headroom/issues/2347)) ([f64aac9](https://github.com/headroomlabs-ai/headroom/commit/f64aac9733d5e314f381644eaea62e2c28b6dc65))
* **proxy/gemini:** tolerate malformed parts on the compression path ([#2486](https://github.com/headroomlabs-ai/headroom/issues/2486)) ([07cf547](https://github.com/headroomlabs-ai/headroom/commit/07cf5476072a45bac7dd94386de126234a8049e7))
* **proxy/metrics:** move the savings-ledger append off the event loop ([#2439](https://github.com/headroomlabs-ai/headroom/issues/2439)) ([4aac068](https://github.com/headroomlabs-ai/headroom/commit/4aac068814246db3fa250c48f5c916aa2561d8c8))
* **proxy/openai:** cache under looked-up messages ([#2420](https://github.com/headroomlabs-ai/headroom/issues/2420)) ([7052d52](https://github.com/headroomlabs-ai/headroom/commit/7052d52dcbb2fd97b756c9b60a096cdfeee32c94))
* **proxy/openai:** don't record Codex WS savings without input accounting ([#2493](https://github.com/headroomlabs-ai/headroom/issues/2493)) ([2195ba7](https://github.com/headroomlabs-ai/headroom/commit/2195ba7d917649ba2ac647fdefa661cf598e3028))
* **proxy/openai:** feed chat/completions traffic into the traffic learner ([#2333](https://github.com/headroomlabs-ai/headroom/issues/2333)) ([6cdfd3f](https://github.com/headroomlabs-ai/headroom/commit/6cdfd3f64d2f64d50ed47644126df71872a21050))
* **proxy/openai:** None-guard usage token counts on the chat path ([#2431](https://github.com/headroomlabs-ai/headroom/issues/2431)) ([313c290](https://github.com/headroomlabs-ai/headroom/commit/313c290df96ca58a19ea0f79c67f5b71bb5f4d60))
* **proxy/openai:** replay incremental events in buffered Responses SSE ([#2410](https://github.com/headroomlabs-ai/headroom/issues/2410)) ([#2415](https://github.com/headroomlabs-ai/headroom/issues/2415)) ([0cbc0e8](https://github.com/headroomlabs-ai/headroom/commit/0cbc0e8e5435cd8d743ae537cdbaa70787bfc5b4))
* **proxy/output-shaping:** tolerate a non-string system block text in steering ([#2435](https://github.com/headroomlabs-ai/headroom/issues/2435)) ([3e97671](https://github.com/headroomlabs-ai/headroom/commit/3e976712e717a53ab6aea73120ae6ffacea74250))
* **proxy/perf:** count turn-hook message folds in token accounting ([#2520](https://github.com/headroomlabs-ai/headroom/issues/2520)) ([c371d5a](https://github.com/headroomlabs-ai/headroom/commit/c371d5ad602f5ab93645b2db4673ae2c5e9f0575))
* **proxy/perf:** tokenizer-consistent token accounting + surface tool-schema savings ([#2542](https://github.com/headroomlabs-ai/headroom/issues/2542)) ([1cc53c9](https://github.com/headroomlabs-ai/headroom/commit/1cc53c9c92cd4dffaf048dc806cb8c570bdb86b6))
* **proxy/streaming:** tolerate malformed content in _response_to_sse ([#2481](https://github.com/headroomlabs-ai/headroom/issues/2481)) ([77b26c0](https://github.com/headroomlabs-ai/headroom/commit/77b26c093cfb7b5c71a46d5156cb774a2ae889b1))
* **proxy:** keep buffered CCR streams alive ([#2479](https://github.com/headroomlabs-ai/headroom/issues/2479)) ([a2e42fb](https://github.com/headroomlabs-ai/headroom/commit/a2e42fb877642e7eacfcc77655183244823d969e))
* **proxy:** keep core tools and the client's ToolSearch resident for PascalCase clients ([#2647](https://github.com/headroomlabs-ai/headroom/issues/2647)) ([1d29738](https://github.com/headroomlabs-ai/headroom/commit/1d29738818bb40e00847dba46e2f9acce773d3eb))
* **proxy:** offload OpenAI and Gemini tokenizer counting off the event loop ([#2498](https://github.com/headroomlabs-ai/headroom/issues/2498)) ([806d2e4](https://github.com/headroomlabs-ai/headroom/commit/806d2e468ace012ebfa1a0907a679781b5004c72))
* **proxy:** promote Kompress health after runtime load ([#2402](https://github.com/headroomlabs-ai/headroom/issues/2402)) ([54526bc](https://github.com/headroomlabs-ai/headroom/commit/54526bc8586cdeb248d6257dc497136a21b971c0))
* **proxy:** reassemble server_tool_use.input from streamed partial_json ([#2449](https://github.com/headroomlabs-ai/headroom/issues/2449)) ([8c8fae0](https://github.com/headroomlabs-ai/headroom/commit/8c8fae0d0bca75f7f2561136910e40f716be57ab))
* **proxy:** report deferred Kompress status and promote health from cache ([#2564](https://github.com/headroomlabs-ai/headroom/issues/2564)) ([d50cfab](https://github.com/headroomlabs-ai/headroom/commit/d50cfabedca2c4b7d83751adaa8aa7b317f13c7b))
* **proxy:** skip max_tokens rename for backend-routed openai chat ([#2401](https://github.com/headroomlabs-ai/headroom/issues/2401)) ([d6a1af4](https://github.com/headroomlabs-ai/headroom/commit/d6a1af40d5a18f4440a45e342c2d05fee7a642e3))
* **release:** publish Windows wheel + sdist (disable PyPI attestations, [#112](https://github.com/headroomlabs-ai/headroom/issues/112)) ([#2405](https://github.com/headroomlabs-ai/headroom/issues/2405)) ([f9cbdd6](https://github.com/headroomlabs-ai/headroom/commit/f9cbdd6e390714e037832f78c59d00907a26b612))
* **release:** sync generated version metadata on the release branch ([#2659](https://github.com/headroomlabs-ai/headroom/issues/2659)) ([5383c6b](https://github.com/headroomlabs-ai/headroom/commit/5383c6bf2f5209ddfe33cb9bf1c36c0b2e431bcd))
* **rust:** port CJK-aware relevance-query matching to CodeCompressor ([#2634](https://github.com/headroomlabs-ai/headroom/issues/2634)) ([e86c639](https://github.com/headroomlabs-ai/headroom/commit/e86c6390cec4fc0f932b006b36d5b924511a5b0b))
* **security:** exclude compromised ast-grep-cli 0.44.1 (supply-chain trojan) ([#2342](https://github.com/headroomlabs-ai/headroom/issues/2342)) ([494fb5a](https://github.com/headroomlabs-ai/headroom/commit/494fb5a60e15ae1ce425f79f1432827b42923c73))
* **tokenizers:** price Claude against a real BPE (tiktoken o200k) not a char estimate ([#2543](https://github.com/headroomlabs-ai/headroom/issues/2543)) ([285176b](https://github.com/headroomlabs-ai/headroom/commit/285176be54e1d179676dcf205de44d5893f8efa5))
* **transforms/cross-turn-dedup:** don't renumber-fold zero-padded line prefixes ([#2369](https://github.com/headroomlabs-ai/headroom/issues/2369)) ([f4070c4](https://github.com/headroomlabs-ai/headroom/commit/f4070c44cbd65ecf49f2ae81ad26a95296ef552b))
* **transforms/kompress-remote:** keep compress fail-open on malformed 200 ([#2320](https://github.com/headroomlabs-ai/headroom/issues/2320)) ([b759990](https://github.com/headroomlabs-ai/headroom/commit/b75999017fc060a4617077ef86c21ce3249d0842))
* **wrap:** emit bare dotted keys for Codex --config overrides ([#2383](https://github.com/headroomlabs-ai/headroom/issues/2383)) ([f57e959](https://github.com/headroomlabs-ai/headroom/commit/f57e959a506f87f14143d595cae24a1fd6084f66))
* **wrap:** make RTK opt-in (off by default) across wrap subcommands ([#2344](https://github.com/headroomlabs-ai/headroom/issues/2344)) ([44136ed](https://github.com/headroomlabs-ai/headroom/commit/44136ed0427edff338c5d7979b589f8540c9b967))
* **wrap:** skip Serena project setup outside real project roots ([#2574](https://github.com/headroomlabs-ai/headroom/issues/2574)) ([0994ea0](https://github.com/headroomlabs-ai/headroom/commit/0994ea04c869939946b91cbe52ceaf46740786be))
* **wrap:** stop same-port persistent routing during claude unwrap ([#2340](https://github.com/headroomlabs-ai/headroom/issues/2340)) ([#2350](https://github.com/headroomlabs-ai/headroom/issues/2350)) ([cf5fa64](https://github.com/headroomlabs-ai/headroom/commit/cf5fa644b6e019a3ea31b4f48509a63921055253))


### Performance Improvements

* **content_router:** dedupe content detection ([#2419](https://github.com/headroomlabs-ai/headroom/issues/2419)) ([9b016f2](https://github.com/headroomlabs-ai/headroom/commit/9b016f2b64cb50cd50ab68711ab2abdf7d74c8ec))


### Dependencies

* bump the cargo-minor-patch group with 10 updates ([#2284](https://github.com/headroomlabs-ai/headroom/issues/2284)) ([3266ed7](https://github.com/headroomlabs-ai/headroom/commit/3266ed7641cc92f5cae79b1befeb6bee7c96242e))
* bump the npm-minor-patch group across 3 directories with 7 updates ([#2276](https://github.com/headroomlabs-ai/headroom/issues/2276)) ([961866b](https://github.com/headroomlabs-ai/headroom/commit/961866ba7c277b59ccdd51e784de9547a09198af))


### Code Refactoring

* **transforms:** dispatch simple built-in strategies via the compressor registry ([#2399](https://github.com/headroomlabs-ai/headroom/issues/2399)) ([fc9c63f](https://github.com/headroomlabs-ai/headroom/commit/fc9c63f18c1a8414b62ced8b2dd54ad1fe4d1c14))
* **wrap:** retire tokensave; Serena is the code-memory MCP ([#2499](https://github.com/headroomlabs-ai/headroom/issues/2499)) ([5d23a0a](https://github.com/headroomlabs-ai/headroom/commit/5d23a0aec22dacdbd7bf221dafbb17bcf9f10c63))

## [0.32.0](https://github.com/headroomlabs-ai/headroom/compare/v0.31.0...v0.32.0) (2026-07-17)


### Features

* 3-layer context compression pipeline (L1+L2+L3) ([#1405](https://github.com/headroomlabs-ai/headroom/issues/1405)) ([3dd9660](https://github.com/headroomlabs-ai/headroom/commit/3dd9660d91abe86b8ea2bb45148a23ad25cbff30))
* add CrewAI and AutoGen tool compression integrations ([#1384](https://github.com/headroomlabs-ai/headroom/issues/1384)) ([e8bff1c](https://github.com/headroomlabs-ai/headroom/commit/e8bff1cfe3a65ae2258aecba18916d331d84c1bf))
* **cli:** add `headroom inspect` to view original vs compressed content ([#1595](https://github.com/headroomlabs-ai/headroom/issues/1595)) ([942e916](https://github.com/headroomlabs-ai/headroom/commit/942e916368fea78ba821d10f91ebe7a1b8fc9b19))
* **cli:** add `wrap openclaude` for OpenClaude CLI ([#1416](https://github.com/headroomlabs-ai/headroom/issues/1416)) ([5415008](https://github.com/headroomlabs-ai/headroom/commit/541500811fa492f6c6cdca98b8645c3fa501e37e))
* **codex:** keep wrap routing session-scoped ([#1507](https://github.com/headroomlabs-ai/headroom/issues/1507)) ([ad9d086](https://github.com/headroomlabs-ai/headroom/commit/ad9d086f43a664c4c2a19060b847f2e03ce4f6ad))
* **compress:** expose frozen_message_count in library-mode compress() ([#2178](https://github.com/headroomlabs-ai/headroom/issues/2178)) ([021a762](https://github.com/headroomlabs-ai/headroom/commit/021a762bf80f67f6b86994d9891624b102031212))
* **core:** gate ONNX transforms behind a default-on `ml` feature (static/lexical builds) ([#2165](https://github.com/headroomlabs-ai/headroom/issues/2165)) ([cdba2ec](https://github.com/headroomlabs-ai/headroom/commit/cdba2eccddaaeb469abb05728e902c43954b5a51))
* **dashboard:** add settings dashboard for proxy configuration ([#2101](https://github.com/headroomlabs-ai/headroom/issues/2101)) ([96bc4cd](https://github.com/headroomlabs-ai/headroom/commit/96bc4cd1280f327c5714d5350ac57ed5590cc901))
* **dashboard:** persist lifetime proxy metrics ([#2198](https://github.com/headroomlabs-ai/headroom/issues/2198)) ([0537cbf](https://github.com/headroomlabs-ai/headroom/commit/0537cbfde4c9d0c6d7aafdb7a56662aee768efbb))
* **deploy:** Add turnkey deploy command ([#1404](https://github.com/headroomlabs-ai/headroom/issues/1404)) ([560ffae](https://github.com/headroomlabs-ai/headroom/commit/560ffae103317b3a9925faf34e79751f0710b813))
* **evals:** register multilingual multi-wiki-qa (zh/ja/ko) dataset ([#1530](https://github.com/headroomlabs-ai/headroom/issues/1530)) ([f891506](https://github.com/headroomlabs-ai/headroom/commit/f8915067f6e8f3177e3e5f99b6bc612fdeb2fb3a))
* **evals:** weekly HotpotQA answer-recall report on the prose path ([#1188](https://github.com/headroomlabs-ai/headroom/issues/1188)) ([46d4378](https://github.com/headroomlabs-ai/headroom/commit/46d4378cf7796836f99933c8f614f84d0a08bf4a))
* **grok-build:** add Grok Build wrap command and MCP integration ([#1629](https://github.com/headroomlabs-ai/headroom/issues/1629)) ([420dc90](https://github.com/headroomlabs-ai/headroom/commit/420dc9077b204c1ad75bde25f3f23ff8c11db770))
* **install:** add apply flag parity, --env passthrough, and EIO retry ([#2152](https://github.com/headroomlabs-ai/headroom/issues/2152)) ([896454e](https://github.com/headroomlabs-ai/headroom/commit/896454e978952ffdb86c24fbe4983d4c223e1c0a))
* **kompress:** optional remote compression endpoint (HEADROOM_KOMPRESS_ENDPOINT) ([#2171](https://github.com/headroomlabs-ai/headroom/issues/2171)) ([b6eb7a7](https://github.com/headroomlabs-ai/headroom/commit/b6eb7a7613f73d4a1bce790a9da217f66d2621ad))
* **mcp:** add streamable HTTP MCP transport ([#1773](https://github.com/headroomlabs-ai/headroom/issues/1773)) ([4ea96a4](https://github.com/headroomlabs-ai/headroom/commit/4ea96a417ceeaa9a96934a7b4f33f4318cee46d9))
* **mcp:** publish canonical server.json ([#1510](https://github.com/headroomlabs-ai/headroom/issues/1510)) ([e9e9cd5](https://github.com/headroomlabs-ai/headroom/commit/e9e9cd55b75e9ac98ba1d204c272e157dc358f02))
* **memory:** add explicit supersession repair ([#2217](https://github.com/headroomlabs-ai/headroom/issues/2217)) ([ce52b30](https://github.com/headroomlabs-ai/headroom/commit/ce52b30c8fe2ad16c0a2ec174fea3cb65dad19ea))
* **metrics:** export compression-failed and kompress size-gate counters ([#1569](https://github.com/headroomlabs-ai/headroom/issues/1569)) ([d728338](https://github.com/headroomlabs-ai/headroom/commit/d7283387ac232dff89411391caa355fdb5ea6a6e))
* **observability:** add gen_ai.request.model to the compression span ([#1667](https://github.com/headroomlabs-ai/headroom/issues/1667)) ([7f7af66](https://github.com/headroomlabs-ai/headroom/commit/7f7af667ed3db718adad34f34d83517ac880760d))
* **proxy:** add opt-in cost-aware model router ([#1706](https://github.com/headroomlabs-ai/headroom/issues/1706)) ([#2205](https://github.com/headroomlabs-ai/headroom/issues/2205)) ([57e8dcb](https://github.com/headroomlabs-ai/headroom/commit/57e8dcb425ca71f8e9ce121d42360c7bd53ce414))
* **proxy:** apply output shaper to OpenAI-compatible endpoints ([#1725](https://github.com/headroomlabs-ai/headroom/issues/1725)) ([e65b9b3](https://github.com/headroomlabs-ai/headroom/commit/e65b9b3f92480f092c268b617566c344e06a3512))
* **proxy:** expose retry delay configuration ([#2077](https://github.com/headroomlabs-ai/headroom/issues/2077)) ([099c664](https://github.com/headroomlabs-ai/headroom/commit/099c66432b7343a14ee67fcfc5db296145ea87ae))
* **proxy:** extend output shaping to the OpenAI Responses path (Codex HTTP + WS) ([#1943](https://github.com/headroomlabs-ai/headroom/issues/1943)) ([71cbb6a](https://github.com/headroomlabs-ai/headroom/commit/71cbb6aaada0b5f188f874e80a65fb2092c8ac13))
* **proxy:** opt-in compression for catch-all passthrough routes ([#1699](https://github.com/headroomlabs-ai/headroom/issues/1699)) ([4cbd5da](https://github.com/headroomlabs-ai/headroom/commit/4cbd5da67328d03078bd9c1fbbf10fe738cc0b2b))
* **proxy:** persist per-model savings breakdown in proxy_savings.json ([#2055](https://github.com/headroomlabs-ai/headroom/issues/2055)) ([12a38d3](https://github.com/headroomlabs-ai/headroom/commit/12a38d31808283c935a4d15f297376d2b159e567))
* **simulators:** add provider simulator service ([#2014](https://github.com/headroomlabs-ai/headroom/issues/2014)) ([2c9eb7c](https://github.com/headroomlabs-ai/headroom/commit/2c9eb7c5f154f087538bcde9561b68736c8c5584))
* **stats:** per-bucket output-shaping savings in /stats-history ([#1819](https://github.com/headroomlabs-ai/headroom/issues/1819)) ([12a9710](https://github.com/headroomlabs-ai/headroom/commit/12a9710665b2495b9fe399e38625a9fa3ad6aa08))
* **text-crusher:** CJK-aware segmentation + relevance via ICU ([#1504](https://github.com/headroomlabs-ai/headroom/issues/1504)) ([4035c04](https://github.com/headroomlabs-ai/headroom/commit/4035c041879e422c986d9cf3b352b13bdc73a01c))
* **text-crusher:** fold full-width ASCII to half-width in CJK token keys ([#2259](https://github.com/headroomlabs-ai/headroom/issues/2259)) ([844d9ca](https://github.com/headroomlabs-ai/headroom/commit/844d9caaa1c7471cc01b9c900f42c9552421bbe2))
* **wrap:** add `headroom wrap kimi` for Kimi CLI ([#1426](https://github.com/headroomlabs-ai/headroom/issues/1426)) ([eac4965](https://github.com/headroomlabs-ai/headroom/commit/eac49656a1cd2a72533432f676a693e283c63bf4))
* **wrap:** add omp target (Oh My Pi) with models.yml override and unwrap ([#1811](https://github.com/headroomlabs-ai/headroom/issues/1811)) ([fcf455a](https://github.com/headroomlabs-ai/headroom/commit/fcf455a7eb3cf7663067aa6e76655c58685bdb12))
* **wrap:** add ZCode desktop app support ([#1845](https://github.com/headroomlabs-ai/headroom/issues/1845)) ([2a954b6](https://github.com/headroomlabs-ai/headroom/commit/2a954b69b4355416fe31341d928c3b03588b660f))
* **wrap:** allow project RTK instruction opt-out ([#2078](https://github.com/headroomlabs-ai/headroom/issues/2078)) ([f53f720](https://github.com/headroomlabs-ai/headroom/commit/f53f720eb5e378c95250b55e3174a064621807db))


### Bug Fixes

* **adaptive-sizer:** char bigrams for spaceless CJK items ([#1748](https://github.com/headroomlabs-ai/headroom/issues/1748)) ([8879c50](https://github.com/headroomlabs-ai/headroom/commit/8879c50dbe929b220b17803a054a078f26b04ca9))
* add DeepSeek V4/V3.2/R1 tokenizer mappings and context limits ([#912](https://github.com/headroomlabs-ai/headroom/issues/912)) ([0c70875](https://github.com/headroomlabs-ai/headroom/commit/0c7087539d92a9dff2fd7f1923aff529f80201ef))
* add Vercel deploy config and workflow for docs site ([#1739](https://github.com/headroomlabs-ai/headroom/issues/1739)) ([b0fa84e](https://github.com/headroomlabs-ai/headroom/commit/b0fa84e84d9c8cb08ebe0320c5baff4553c329c8))
* **auth:** support GitHub Enterprise Copilot OAuth domain ([#2192](https://github.com/headroomlabs-ai/headroom/issues/2192)) ([5dbe331](https://github.com/headroomlabs-ai/headroom/commit/5dbe3314a198e6c4381a2280b83ec1629653bedf))
* **backend/bedrock:** preserve system-prompt cache_control breakpoint (list form) ([#2225](https://github.com/headroomlabs-ai/headroom/issues/2225)) ([ea0115c](https://github.com/headroomlabs-ai/headroom/commit/ea0115cbdbbf180062cc59e794371f62bf5875c3))
* **backends/litellm:** drop oversized tool names before Bedrock Converse ([#2129](https://github.com/headroomlabs-ai/headroom/issues/2129)) ([3c1a5cd](https://github.com/headroomlabs-ai/headroom/commit/3c1a5cdb1c9388d5745a4d91ec53cd4364f26f67))
* **backends/litellm:** preserve tool_result cache_control, complete streaming cache stats ([#2144](https://github.com/headroomlabs-ai/headroom/issues/2144)) ([0ad7dc7](https://github.com/headroomlabs-ai/headroom/commit/0ad7dc7c1ce3b1eac2e15dd6f5b974541a64f393))
* **bedrock:** resolve global.* inference profiles + pin per-user app-profile ARNs ([#1795](https://github.com/headroomlabs-ai/headroom/issues/1795)) ([33c7f6c](https://github.com/headroomlabs-ai/headroom/commit/33c7f6cd3ae14857effcbd110e9cd0668cf1ac1b))
* **build:** support Intel macOS (x86_64-apple-darwin) via ort-load-dynamic (fixes [#941](https://github.com/headroomlabs-ai/headroom/issues/941)) ([#1797](https://github.com/headroomlabs-ai/headroom/issues/1797)) ([1590913](https://github.com/headroomlabs-ai/headroom/commit/1590913bb785e0b352d2192b6c595152e01feb14))
* **cache-aligner:** hash the frozen conversation prefix so Claude Code cache invalidation is detected ([#2085](https://github.com/headroomlabs-ai/headroom/issues/2085)) ([#2161](https://github.com/headroomlabs-ai/headroom/issues/2161)) ([cc072f0](https://github.com/headroomlabs-ai/headroom/commit/cc072f0821b76618659a2dd6c81a1daee49dbaab))
* **cache/ccr:** don't count a successful eviction as a retrieval ([#2106](https://github.com/headroomlabs-ai/headroom/issues/2106)) ([eecb81e](https://github.com/headroomlabs-ai/headroom/commit/eecb81e8478dbe4337a48dcdc2ace6537df0784f))
* **cache/ccr:** don't evict a live entry on a duplicate store at capacity ([#2082](https://github.com/headroomlabs-ai/headroom/issues/2082)) ([1138946](https://github.com/headroomlabs-ai/headroom/commit/113894600cbc8aae219bd733ea8460a64e3e626f))
* **cache/semantic:** don't evict an unrelated entry on an update at capacity ([#2094](https://github.com/headroomlabs-ai/headroom/issues/2094)) ([cf6367a](https://github.com/headroomlabs-ai/headroom/commit/cf6367add4fce3294a790bd1a6733c97e62072d9))
* **cache/semantic:** key entries by context hash, not query text ([#2022](https://github.com/headroomlabs-ai/headroom/issues/2022)) ([d8783ab](https://github.com/headroomlabs-ai/headroom/commit/d8783ab89b49b309df4aae927d9f89ac3cbe058b))
* **cache:** extract tool_result content from list-of-blocks format ([#2092](https://github.com/headroomlabs-ai/headroom/issues/2092)) ([3bcef2b](https://github.com/headroomlabs-ai/headroom/commit/3bcef2be37d7d187ec33d408e6a6171680ec05a0))
* **cache:** normalize embeddings before the semantic similarity check ([#2122](https://github.com/headroomlabs-ai/headroom/issues/2122)) ([f8eaaeb](https://github.com/headroomlabs-ai/headroom/commit/f8eaaeb26a65b7a92981c7061cb3650cc184f72e))
* **cache:** partial cached-prefix replay + idle-aware net-cost; don't… ([#1933](https://github.com/headroomlabs-ai/headroom/issues/1933)) ([b0440f9](https://github.com/headroomlabs-ai/headroom/commit/b0440f958dd596a13556a4d9a46d3e25d8f9f173))
* **cache:** stable session identity and per-conversation prefix trackers under agentic clients ([#2193](https://github.com/headroomlabs-ai/headroom/issues/2193)) ([7bfb1d7](https://github.com/headroomlabs-ai/headroom/commit/7bfb1d7f38ac3c3cf3f123033c157a17ad0da71a))
* **cache:** stop DynamicContentDetector false positives corrupting cached prompts ([#2110](https://github.com/headroomlabs-ai/headroom/issues/2110)) ([#2119](https://github.com/headroomlabs-ai/headroom/issues/2119)) ([908a9a1](https://github.com/headroomlabs-ai/headroom/commit/908a9a1bb18b132007d8ca84ca784ec8e809186a))
* **ccr:** detect read_lifecycle stale/superseded markers in the injector ([#2148](https://github.com/headroomlabs-ai/headroom/issues/2148)) ([ec97443](https://github.com/headroomlabs-ai/headroom/commit/ec97443e66cb65e7e16bb6a6c7ba5d0c9e36e734))
* **ccr:** don't crash parse_tool_call on non-object tool arguments ([#2071](https://github.com/headroomlabs-ai/headroom/issues/2071)) ([984a2c7](https://github.com/headroomlabs-ai/headroom/commit/984a2c702c7ea42436c914d0fdf434cf2dd3f45c))
* **ccr:** don't crash tool-call detection on a null function/functionCall ([#2269](https://github.com/headroomlabs-ai/headroom/issues/2269)) ([1612f06](https://github.com/headroomlabs-ai/headroom/commit/1612f06a4c19c53b6d8b02b02cd30d14a1858d28))
* **ccr:** lowercase a retrieved hash so an uppercase echo still hits the store ([#2236](https://github.com/headroomlabs-ai/headroom/issues/2236)) ([842d7e1](https://github.com/headroomlabs-ai/headroom/commit/842d7e1ad1869b147ec94903ac87cf1414d86cdb))
* **ccr:** skip compact summaries for proactive expansion ([#2242](https://github.com/headroomlabs-ai/headroom/issues/2242)) ([3f241e4](https://github.com/headroomlabs-ai/headroom/commit/3f241e472b07c39efd8fd51e35b7070f5d514ea1))
* **ccr:** store pre-protection original, not tag placeholder, in CCR ([#1208](https://github.com/headroomlabs-ai/headroom/issues/1208)) ([a61f534](https://github.com/headroomlabs-ai/headroom/commit/a61f534426d98f4467057794fdcb6acdbf28a3bf))
* **ci/deps:** clear audit and release smoke failures ([#2190](https://github.com/headroomlabs-ai/headroom/issues/2190)) ([fce93bf](https://github.com/headroomlabs-ai/headroom/commit/fce93bf39ae6aad5a386f044c6f5dd3933a0c0d4))
* **claude:** treat non-zero claude --version exit as version-unknown … ([#2233](https://github.com/headroomlabs-ai/headroom/issues/2233)) ([f71fef1](https://github.com/headroomlabs-ai/headroom/commit/f71fef1ca6a1540a3c13c3508c5c1468bf14ea88))
* **cli/init:** fail clearly on a target settings file with invalid JSON ([#2227](https://github.com/headroomlabs-ai/headroom/issues/2227)) ([daca1dd](https://github.com/headroomlabs-ai/headroom/commit/daca1dd7561948f460889cd69d68344d2a53b5b5))
* **code-compressor:** recover valid Python rewrites after local syntax rejection ([#2202](https://github.com/headroomlabs-ai/headroom/issues/2202)) ([dbbef4b](https://github.com/headroomlabs-ai/headroom/commit/dbbef4bd41f10ec748e462717fb976256a6a8916))
* **code:** parse-probe tree-sitter availability in code_handler ([#1231](https://github.com/headroomlabs-ai/headroom/issues/1231)) ([#1300](https://github.com/headroomlabs-ai/headroom/issues/1300)) ([1de35e7](https://github.com/headroomlabs-ai/headroom/commit/1de35e775f2f19e51258fecddcabc2b88775f7d8))
* **code:** pin tree-sitter-language-pack &lt;1.0.0 in [code] extra ([#1219](https://github.com/headroomlabs-ai/headroom/issues/1219)) ([412db40](https://github.com/headroomlabs-ai/headroom/commit/412db40a0bcb35d2a631ea5e74eb910b1f89feb3))
* **code:** quarantine Perl parser from code-aware compression ([#2204](https://github.com/headroomlabs-ai/headroom/issues/2204)) ([8522fcb](https://github.com/headroomlabs-ai/headroom/commit/8522fcbc40b716a5faf8978237e7b829a6468d09))
* **codex:** preserve wrapped sessions and recover state ([#2160](https://github.com/headroomlabs-ai/headroom/issues/2160)) ([dec60de](https://github.com/headroomlabs-ai/headroom/commit/dec60de976ee0a64b33bd4c4625c9e73688ddcd3))
* **codex:** rerun memory lookup on every response.create WS frame ([#2113](https://github.com/headroomlabs-ai/headroom/issues/2113)) ([38479fc](https://github.com/headroomlabs-ai/headroom/commit/38479fcda1d4be717b049f5bc9932bdb85e69015))
* **codex:** rewrite config.toml properly so Codex will route through … ([#2102](https://github.com/headroomlabs-ai/headroom/issues/2102)) ([5d9bbbe](https://github.com/headroomlabs-ai/headroom/commit/5d9bbbeea07c2e1ee1640cc455aaeb23d9f75768))
* **codex:** skip sockets in session home overlay ([#2104](https://github.com/headroomlabs-ai/headroom/issues/2104)) ([c4ddcb9](https://github.com/headroomlabs-ai/headroom/commit/c4ddcb93a7ffe55112e41d8d45932fda1255ba1d))
* **content_router:** pin FREEZE_BLOCK_DECISION verdict to stop cache-write churn ([#1620](https://github.com/headroomlabs-ai/headroom/issues/1620)) ([a069979](https://github.com/headroomlabs-ai/headroom/commit/a0699794660132313446e8c52c588d6ead05af21))
* **content-router:** protect_tool_results must not be weakened by profile-derived read_protection_window ([#2105](https://github.com/headroomlabs-ai/headroom/issues/2105)) ([3d0e59e](https://github.com/headroomlabs-ai/headroom/commit/3d0e59e518380972b56f017cffba96522141d53d))
* **copilot-auth:** stop discarding the caller's valid Copilot auth token ([#1879](https://github.com/headroomlabs-ai/headroom/issues/1879)) ([f52ca19](https://github.com/headroomlabs-ai/headroom/commit/f52ca19db1c28a87e9c3f6ff4ad21d6a16d7aa08))
* **copilot:** refresh wrapped subscription tokens ([#2156](https://github.com/headroomlabs-ai/headroom/issues/2156)) ([#2182](https://github.com/headroomlabs-ai/headroom/issues/2182)) ([4364eb8](https://github.com/headroomlabs-ai/headroom/commit/4364eb8dc4a1b185e30da340537e88975e91d0c4))
* **core:** avoid unidiff panic on bash xtrace ([#1506](https://github.com/headroomlabs-ai/headroom/issues/1506)) ([3757a7c](https://github.com/headroomlabs-ai/headroom/commit/3757a7cef3c3f672287244f5d5c768b9b5c6f854))
* **dashboard:** serve per-request metadata to trusted-gateway peers ([#1766](https://github.com/headroomlabs-ai/headroom/issues/1766)) ([560319c](https://github.com/headroomlabs-ai/headroom/commit/560319cef4e38d6b79c3d0302e392dd736213572))
* **dedup:** shorter fold pointer + MIN_LINES=3, dedup Anthropic list-… ([#1932](https://github.com/headroomlabs-ai/headroom/issues/1932)) ([10e4829](https://github.com/headroomlabs-ai/headroom/commit/10e4829201b4a66d6e7e8f2f2e8a9d80346de850))
* **deps:** bump pillow to 12.3.0 and click to 8.4.2 ([#2097](https://github.com/headroomlabs-ai/headroom/issues/2097)) ([8870b69](https://github.com/headroomlabs-ai/headroom/commit/8870b6971f924eb1838694c3aec1949ca641e6af))
* **deps:** clear Dependabot lockfile alerts ([#2175](https://github.com/headroomlabs-ai/headroom/issues/2175)) ([ea3d5a8](https://github.com/headroomlabs-ai/headroom/commit/ea3d5a86b725d7b0a0a024980f6e1214da4c5fcc))
* **deps:** enforce transformers security floor ([#2201](https://github.com/headroomlabs-ai/headroom/issues/2201)) ([cbfa267](https://github.com/headroomlabs-ai/headroom/commit/cbfa267c5f4f6c924c377115b5bcd1cef5230451))
* **deps:** raise transformers security floor ([09be107](https://github.com/headroomlabs-ai/headroom/commit/09be107d066fe97566a2eea0cad200d20d88c263))
* **diff-compressor:** CJK-aware relevance scoring for hunk selection ([#2220](https://github.com/headroomlabs-ai/headroom/issues/2220)) ([528517c](https://github.com/headroomlabs-ai/headroom/commit/528517cff86ae1a27e8e325d96d0124260fc9fe1))
* emit unknown Anthropic content blocks verbatim in buffered-to-SSE conversion ([#1825](https://github.com/headroomlabs-ai/headroom/issues/1825)) ([d05802b](https://github.com/headroomlabs-ai/headroom/commit/d05802b6200b94f198e99319e6c778e78b53db8b))
* harden fd lifecycle and SystemError handling in runtime and proxy kill ([#1556](https://github.com/headroomlabs-ai/headroom/issues/1556)) ([f42ce4a](https://github.com/headroomlabs-ai/headroom/commit/f42ce4a2399f3285548b5a67f2bf1cede0334816))
* harden persistent install startup ([#1851](https://github.com/headroomlabs-ai/headroom/issues/1851)) ([1d2b76e](https://github.com/headroomlabs-ai/headroom/commit/1d2b76e72e16eaf532326d9e75a481e18bde1ab7))
* **health:** exclude kompress from aggregate readiness + adversarial PBT ([#2066](https://github.com/headroomlabs-ai/headroom/issues/2066)) ([f1663ea](https://github.com/headroomlabs-ai/headroom/commit/f1663ea55700cd7a1ca4400ed27c200d27317d97))
* **init/codex:** don't delete per-profile provider settings ([#2146](https://github.com/headroomlabs-ai/headroom/issues/2146)) ([8da4384](https://github.com/headroomlabs-ai/headroom/commit/8da4384bfcb21d81f9972a8523879336bdfd36d7))
* **install:** add orjson to [proxy] extra for LiteLLM provider backends ([#2074](https://github.com/headroomlabs-ai/headroom/issues/2074)) ([4f3d5ab](https://github.com/headroomlabs-ai/headroom/commit/4f3d5ab341466297cb1b37104d93b338c4947c82))
* **install:** default docker image to headroomlabs-ai GHCR registry ([#1867](https://github.com/headroomlabs-ai/headroom/issues/1867)) ([#2039](https://github.com/headroomlabs-ai/headroom/issues/2039)) ([c3db8e4](https://github.com/headroomlabs-ai/headroom/commit/c3db8e47f8dd85caf18a8748068c627cba6ecfeb))
* **install:** don't let host env override the manifest in persistent-docker ([#2090](https://github.com/headroomlabs-ai/headroom/issues/2090)) ([b097ef3](https://github.com/headroomlabs-ai/headroom/commit/b097ef3e25029899c28bf930c8c7fe46ab384d52))
* **install:** guard non-dict health config in 'install status' ([#2150](https://github.com/headroomlabs-ai/headroom/issues/2150)) ([8f867e4](https://github.com/headroomlabs-ai/headroom/commit/8f867e46222239a45136df8f5a05515897077bbf))
* **install:** write deployment manifest atomically and tolerate corrupt manifests ([#1303](https://github.com/headroomlabs-ai/headroom/issues/1303)) ([42bdf23](https://github.com/headroomlabs-ai/headroom/commit/42bdf23d241388cd1a2567f8481b51dacdd8ec83))
* **kompress:** fail-open wall-clock guard on single-cache-miss compression ([#2114](https://github.com/headroomlabs-ai/headroom/issues/2114)) ([e900086](https://github.com/headroomlabs-ai/headroom/commit/e9000863fc5ee0a0e1d56b7b49afe36cc6111ada))
* **kompress:** surface model-not-ready state via logs and health endpoint ([#2034](https://github.com/headroomlabs-ai/headroom/issues/2034)) ([12aa2cb](https://github.com/headroomlabs-ai/headroom/commit/12aa2cbf6cc888cce1cb6a47cbea02f37c26fe4d))
* **learn/claude:** don't abort the whole scan on a null message line ([#2299](https://github.com/headroomlabs-ai/headroom/issues/2299)) ([eed80dd](https://github.com/headroomlabs-ai/headroom/commit/eed80dd4baa6a9d8be0ed36a575d0169f6b5eb3c))
* **learn:** don't desync verbosity pairing on empty assistant turns ([#2123](https://github.com/headroomlabs-ai/headroom/issues/2123)) ([def2f9a](https://github.com/headroomlabs-ai/headroom/commit/def2f9a7286a271a36402bc564867b53a3c11b70))
* **learn:** don't shadow TIMEOUT/CONNECTION with the generic RUNTIME_ERROR ([#2099](https://github.com/headroomlabs-ai/headroom/issues/2099)) ([c7b5a24](https://github.com/headroomlabs-ai/headroom/commit/c7b5a24b4ffca78bacf1919ace00087b6ab6f7d0))
* **learn:** handle Windows UTF-8, drive-letter paths, and CLI shim fallback ([#1895](https://github.com/headroomlabs-ai/headroom/issues/1895)) ([e3b45e4](https://github.com/headroomlabs-ai/headroom/commit/e3b45e402bc019c4de5f15c458c59696a68a8aff))
* **learn:** ingest OpenAI Responses HTTP traffic ([#2167](https://github.com/headroomlabs-ai/headroom/issues/2167)) ([ce14130](https://github.com/headroomlabs-ai/headroom/commit/ce141301f101b9d83ed05abbfb36f37ac53baf7d))
* **learn:** parse fenced JSON even with a prose preamble ([#1988](https://github.com/headroomlabs-ai/headroom/issues/1988)) ([d2170b1](https://github.com/headroomlabs-ai/headroom/commit/d2170b1922d63e586f672e0b86bcb8b7c9de0283))
* **litellm:** forward chat_template_kwargs and other vendor top-level fields to OpenAI-compatible backends via extra_body ([#2128](https://github.com/headroomlabs-ai/headroom/issues/2128)) ([#2163](https://github.com/headroomlabs-ai/headroom/issues/2163)) ([fb683e1](https://github.com/headroomlabs-ai/headroom/commit/fb683e18d4cbb0228dcbf951762826bc0eb63a39))
* **litellm:** surface Bedrock cache token usage in non-streaming responses ([#1848](https://github.com/headroomlabs-ai/headroom/issues/1848)) ([d604e86](https://github.com/headroomlabs-ai/headroom/commit/d604e86904525d67d618e7920cd7ce0dac6e903d))
* **mcp/claude:** don't clobber an unparseable Claude config on register ([#1660](https://github.com/headroomlabs-ai/headroom/issues/1660)) ([bc24e25](https://github.com/headroomlabs-ai/headroom/commit/bc24e258b11f3a91c23562336a3d38d90d347fc1))
* **mcp/codex:** don't clobber an unparseable/non-table config.toml ([#2062](https://github.com/headroomlabs-ai/headroom/issues/2062)) ([415e03c](https://github.com/headroomlabs-ai/headroom/commit/415e03c1688a5a1a919872335257df1b6b424c42))
* **mcp/opencode:** don't clobber an unparseable opencode.json on register ([#1661](https://github.com/headroomlabs-ai/headroom/issues/1661)) ([d079614](https://github.com/headroomlabs-ai/headroom/commit/d079614b1fe828c6b8d42ba77236e6ff8df40076))
* **mcp:** correct default Claude Code config path in ClaudeRegistrar ([#1859](https://github.com/headroomlabs-ai/headroom/issues/1859)) ([c85731d](https://github.com/headroomlabs-ai/headroom/commit/c85731dc23832971993653c2fae7ae65e1db27ac))
* **mcp:** mcp status checks ~/.claude.json, not only ~/.claude/mcp.json ([#990](https://github.com/headroomlabs-ai/headroom/issues/990)) ([9e376af](https://github.com/headroomlabs-ai/headroom/commit/9e376afabe7dbed29fb9ff33ce2f8eee283faabd))
* **mcp:** reap orphaned mcp serve on client death ([#2226](https://github.com/headroomlabs-ai/headroom/issues/2226)) ([7a5d8a7](https://github.com/headroomlabs-ai/headroom/commit/7a5d8a7ace2c86f7ab606b0746907443a3565344))
* **mcp:** regenerate stale server.json (0.27.0 -&gt; 0.32.0) ([#2218](https://github.com/headroomlabs-ai/headroom/issues/2218)) ([79d8056](https://github.com/headroomlabs-ai/headroom/commit/79d8056fd76a8f94cc17045236c15f52e3da73e0))
* **memory/sqlite:** don't emit OFFSET without LIMIT in query ([#2063](https://github.com/headroomlabs-ai/headroom/issues/2063)) ([a5bdc54](https://github.com/headroomlabs-ai/headroom/commit/a5bdc5491f91e9486af9c25d2039e848b63cc98c))
* **memory/sync:** don't clobber memories sharing a first line ([#1976](https://github.com/headroomlabs-ai/headroom/issues/1976)) ([5e14b8c](https://github.com/headroomlabs-ai/headroom/commit/5e14b8c0f293df78f2576a9fc7eb90189e604cb5))
* **memory/sync:** make Codex AGENTS.md adapter additive (stop wiping memories) ([#1674](https://github.com/headroomlabs-ai/headroom/issues/1674)) ([7fd0c42](https://github.com/headroomlabs-ai/headroom/commit/7fd0c42ced9ecdf2a5411ff85d554b9e39ceb0b6))
* **memory:** annotate _EMBEDDER_CACHE key as 3-tuple (unbreak main lint) ([#2153](https://github.com/headroomlabs-ai/headroom/issues/2153)) ([22af75a](https://github.com/headroomlabs-ai/headroom/commit/22af75adaee59ab402a9c4579dfa59e222682831))
* **memory:** apply turn_id scope filter even without agent_id ([#2130](https://github.com/headroomlabs-ai/headroom/issues/2130)) ([9e38905](https://github.com/headroomlabs-ai/headroom/commit/9e38905a7d642c9cd796ab496104583bd44a226d))
* **memory:** audit passive context injection ([#2212](https://github.com/headroomlabs-ai/headroom/issues/2212)) ([2de07db](https://github.com/headroomlabs-ai/headroom/commit/2de07db28199f2da5ef27d37a38c638addf367fc))
* **memory:** filter inactive graph-expanded results ([#2210](https://github.com/headroomlabs-ai/headroom/issues/2210)) ([aa4515c](https://github.com/headroomlabs-ai/headroom/commit/aa4515cf7aaf74e1780c170ec8fe24d5fb3a7e49))
* **memory:** honor explicit store=false on Responses requests ([#2017](https://github.com/headroomlabs-ai/headroom/issues/2017)) ([31abb69](https://github.com/headroomlabs-ai/headroom/commit/31abb696dd8f7452ae5ec66e0d55eaf28e59a62c))
* **memory:** key the embedder cache on ollama_base_url ([#2109](https://github.com/headroomlabs-ai/headroom/issues/2109)) ([1725cd1](https://github.com/headroomlabs-ai/headroom/commit/1725cd1f8376828849c0e48fd9008afa25d24fff))
* **memory:** preserve semantically similar memories ([#2303](https://github.com/headroomlabs-ai/headroom/issues/2303)) ([5279c33](https://github.com/headroomlabs-ai/headroom/commit/5279c33b192402faa7c4cbb7b1f6bae36ccda9c7))
* **memory:** remove a superseded memory from the search indexes ([#2143](https://github.com/headroomlabs-ai/headroom/issues/2143)) ([fa330f3](https://github.com/headroomlabs-ai/headroom/commit/fa330f3e2bed3f51b7f4d49bcd64ec4aaa17f44f))
* **memory:** require explicit updates for supersession ([#2188](https://github.com/headroomlabs-ai/headroom/issues/2188)) ([6d897e8](https://github.com/headroomlabs-ai/headroom/commit/6d897e8eaabaddee22a7763292ab3991ffe15af0))
* **memory:** serialize MCP backend initialization ([#2309](https://github.com/headroomlabs-ai/headroom/issues/2309)) ([0924755](https://github.com/headroomlabs-ai/headroom/commit/0924755591d0fb625439facb04905e4fc052b445))
* **memory:** size HNSW index_batch resize off the id high-water mark ([#2139](https://github.com/headroomlabs-ai/headroom/issues/2139)) ([b0afee8](https://github.com/headroomlabs-ai/headroom/commit/b0afee85b3a94f829c4743fce2bbf75c494e6312))
* **memory:** track MCP retrieval access ([#2065](https://github.com/headroomlabs-ai/headroom/issues/2065)) ([d0ecc9a](https://github.com/headroomlabs-ai/headroom/commit/d0ecc9a556047fa24a43609dd02b5b1e2b5d7b03))
* **models:** version-boundary longest-prefix match in ModelRegistry.get ([#1658](https://github.com/headroomlabs-ai/headroom/issues/1658)) ([b699bed](https://github.com/headroomlabs-ai/headroom/commit/b699bedf95286138b1dda444ebc5f86bf7041f5a))
* **opencode:** Use opencode.jsonc when present ([#1590](https://github.com/headroomlabs-ai/headroom/issues/1590)) ([4e2bbfe](https://github.com/headroomlabs-ai/headroom/commit/4e2bbfee3f65e3287ab7693da70f0d7b20dada28))
* **opencode:** use type=local + environment field for MCP config ([#1380](https://github.com/headroomlabs-ai/headroom/issues/1380)) ([#1388](https://github.com/headroomlabs-ai/headroom/issues/1388)) ([a51bbfb](https://github.com/headroomlabs-ai/headroom/commit/a51bbfb6a56fa06ef18109ee0194610c400a2f43))
* **packaging:** guard torch extras on intel macos ([#2011](https://github.com/headroomlabs-ai/headroom/issues/2011)) ([fd0d29c](https://github.com/headroomlabs-ai/headroom/commit/fd0d29c92dbf3629c87bd0aaa8960f55573fae92))
* patch nltk vulnerability (CVE-2026-54293) ([#1929](https://github.com/headroomlabs-ai/headroom/issues/1929)) ([28ca61f](https://github.com/headroomlabs-ai/headroom/commit/28ca61fc9d3e36d5f967da6b3f75d6bccfeb0306))
* **paths:** reject '.', '..', and NUL as plugin names ([#2132](https://github.com/headroomlabs-ai/headroom/issues/2132)) ([af7385a](https://github.com/headroomlabs-ai/headroom/commit/af7385a2980a79e67ac595e454ef47a4d19178f6))
* **pricing:** alias retired claude-3-sonnet to Sonnet-tier price, not Haiku ([#2095](https://github.com/headroomlabs-ai/headroom/issues/2095)) ([6137967](https://github.com/headroomlabs-ai/headroom/commit/6137967083936467c570e8c7f20e94f43ccc13aa))
* **proxy/anthropic:** cache response under the looked-up messages ([#327](https://github.com/headroomlabs-ai/headroom/issues/327)) ([#2124](https://github.com/headroomlabs-ai/headroom/issues/2124)) ([dbb4e4c](https://github.com/headroomlabs-ai/headroom/commit/dbb4e4cf4809477997b19c5ba22c9905d9fd5e02))
* **proxy/anthropic:** preserve non-2xx upstream status through security scan ([#2100](https://github.com/headroomlabs-ai/headroom/issues/2100)) ([aa78816](https://github.com/headroomlabs-ai/headroom/commit/aa788164fd5ad55ed427b25fb2e87e692921d824))
* **proxy/anthropic:** scope session id by top-level system prompt ([#2070](https://github.com/headroomlabs-ai/headroom/issues/2070)) ([ec6e60e](https://github.com/headroomlabs-ai/headroom/commit/ec6e60ea3ef2e00b245a6dd92dbe34cb145f4d33))
* **proxy/batch:** preserve sibling tool configs on Google batch requests ([#2177](https://github.com/headroomlabs-ai/headroom/issues/2177)) ([a5d7e12](https://github.com/headroomlabs-ai/headroom/commit/a5d7e12c9078c008dbb8dc34a77018e5d13cb5da))
* **proxy/bedrock:** wire PrefixCacheTracker updates into Bedrock backend paths ([#2196](https://github.com/headroomlabs-ai/headroom/issues/2196)) ([a352fa0](https://github.com/headroomlabs-ai/headroom/commit/a352fa0168180844dfac68dd053dfb1b041da822))
* **proxy/cost:** price cache savings by most-used model, not first-seen ([#2023](https://github.com/headroomlabs-ai/headroom/issues/2023)) ([b4f807f](https://github.com/headroomlabs-ai/headroom/commit/b4f807f21a5be39c690b8b5e8e236116a32dd6b6))
* **proxy/gemini:** forward a non-JSON upstream body with its real status ([#2174](https://github.com/headroomlabs-ai/headroom/issues/2174)) ([f723925](https://github.com/headroomlabs-ai/headroom/commit/f723925be75ad93978b4bfeab81df4196a9dd401))
* **proxy/gemini:** preserve non-text content across the compression round-trip ([#2079](https://github.com/headroomlabs-ai/headroom/issues/2079)) ([4056117](https://github.com/headroomlabs-ai/headroom/commit/4056117d90468619dbc2684448ebd76c48c41921))
* **proxy/gemini:** thread savings-profile kwargs into apply() ([#1994](https://github.com/headroomlabs-ai/headroom/issues/1994)) ([38306a3](https://github.com/headroomlabs-ai/headroom/commit/38306a331c1e25d688db0219920915904d5e22f3))
* **proxy/memory:** capture user text blocks for the retrieval query ([#2064](https://github.com/headroomlabs-ai/headroom/issues/2064)) ([f542b70](https://github.com/headroomlabs-ai/headroom/commit/f542b70413677260a6ebd5801df64337f66525eb))
* **proxy/memory:** don't crash memory tool-call detection on a null function ([#2272](https://github.com/headroomlabs-ai/headroom/issues/2272)) ([8b7e797](https://github.com/headroomlabs-ai/headroom/commit/8b7e797ed41ea9a37ef8711e102b355f9194c571))
* **proxy/openai:** respect explicit stream_options.include_usage ([#2026](https://github.com/headroomlabs-ai/headroom/issues/2026)) ([19201e8](https://github.com/headroomlabs-ai/headroom/commit/19201e842f30af2b842458b5a8f5891d9b631b29))
* **proxy/savings:** append history point on cache-only savings too ([#2194](https://github.com/headroomlabs-ai/headroom/issues/2194)) ([d125805](https://github.com/headroomlabs-ai/headroom/commit/d1258055893a0af854c0ff7eaf37bbeb9d697706))
* **proxy/savings:** don't bill fallback rate for free (0-priced) models ([#2024](https://github.com/headroomlabs-ai/headroom/issues/2024)) ([6ecbdd6](https://github.com/headroomlabs-ai/headroom/commit/6ecbdd6b527312ce61a1e8890f0d6d1d63ae4f48))
* **proxy/streaming:** preserve non-standard content-block fields on SSE reconstruction ([#2271](https://github.com/headroomlabs-ai/headroom/issues/2271)) ([6decbd1](https://github.com/headroomlabs-ai/headroom/commit/6decbd1e6e6732dc94f7d501116282c41994ce74))
* **proxy/vertex:** route google-publisher requests to the request region ([#2069](https://github.com/headroomlabs-ai/headroom/issues/2069)) ([1843346](https://github.com/headroomlabs-ai/headroom/commit/1843346283b1d6a8d5c932fdc3e16502ced416e8))
* **proxy:** accept Codex websocket before upstream retries ([#2203](https://github.com/headroomlabs-ai/headroom/issues/2203)) ([551f473](https://github.com/headroomlabs-ai/headroom/commit/551f473e04123a11244429456ca3df28549c6adf))
* **proxy:** aggregate tool-output size floor so Codex sessions compress ([#2050](https://github.com/headroomlabs-ai/headroom/issues/2050)) ([#2116](https://github.com/headroomlabs-ai/headroom/issues/2116)) ([dbe2558](https://github.com/headroomlabs-ai/headroom/commit/dbe2558c18552b94bb4b224c003f772e6ef7b2f0))
* **proxy:** batch small Codex Responses tool outputs ([#2239](https://github.com/headroomlabs-ai/headroom/issues/2239)) ([09c66ac](https://github.com/headroomlabs-ai/headroom/commit/09c66ac2128469422ea25f8514776546659ea2f8))
* **proxy:** cache_savings_usd silently zeroes when litellm is unavailable ([#2005](https://github.com/headroomlabs-ai/headroom/issues/2005)) ([75d7861](https://github.com/headroomlabs-ai/headroom/commit/75d786117a15b55ca31daa2b2f07b123994d74d1))
* **proxy:** cold-start fast pass — defer only Kompress, not the whole pipeline ([#2073](https://github.com/headroomlabs-ai/headroom/issues/2073)) ([fd9ddaa](https://github.com/headroomlabs-ai/headroom/commit/fd9ddaa238dfc4691383d82cdeff84cc5885f195))
* **proxy:** compress Hermes scoped coding-agent passthrough ([#1815](https://github.com/headroomlabs-ai/headroom/issues/1815)) ([09d1ef4](https://github.com/headroomlabs-ai/headroom/commit/09d1ef45be4bb8d1debed5c78610eacc7e518396))
* **proxy:** compress OpenCode tool schemas and embedded JSON ([#1535](https://github.com/headroomlabs-ai/headroom/issues/1535)) ([05932d7](https://github.com/headroomlabs-ai/headroom/commit/05932d716567cc78087fc4080d45741be18e5cba))
* **proxy:** count exhausted upstream 5xx as failed across all providers ([#1571](https://github.com/headroomlabs-ai/headroom/issues/1571)) ([e365ad7](https://github.com/headroomlabs-ai/headroom/commit/e365ad71524813d348fdc45f09535da8cbe7b53a))
* **proxy:** dedupe Codex WS request logging for accurate mixed-provider dashboards ([#2189](https://github.com/headroomlabs-ai/headroom/issues/2189)) ([e5b3a63](https://github.com/headroomlabs-ai/headroom/commit/e5b3a634df81610bb4ae5a5b9d1859f6fb38b6d3))
* **proxy:** don't 502 Anthropic streaming on a legal mixed CCR + client-tool turn ([#2089](https://github.com/headroomlabs-ai/headroom/issues/2089)) ([#2117](https://github.com/headroomlabs-ai/headroom/issues/2117)) ([4951cf8](https://github.com/headroomlabs-ai/headroom/commit/4951cf80a2ddd8c08daa64d9da3227de229e0403))
* **proxy:** fail soft on a bad HEADROOM_QDRANT_PORT during config construction ([#2141](https://github.com/headroomlabs-ai/headroom/issues/2141)) ([69aea2f](https://github.com/headroomlabs-ai/headroom/commit/69aea2fc4f56ca3402ce7cfb0323e1730c81827d))
* **proxy:** handle ClientDisconnect in passthrough body reads ([#2033](https://github.com/headroomlabs-ai/headroom/issues/2033)) ([9db8a6b](https://github.com/headroomlabs-ai/headroom/commit/9db8a6bbf661026d92647cea395954a51a075e4f))
* **proxy:** handle ClientDisconnect in passthrough body reads + log sanitization ([#2067](https://github.com/headroomlabs-ai/headroom/issues/2067)) ([605e269](https://github.com/headroomlabs-ai/headroom/commit/605e269f9844099d639418ad1825fa97212b67a7))
* **proxy:** handle content-part outputs in Codex Responses compression ([#2052](https://github.com/headroomlabs-ai/headroom/issues/2052)) ([c9a7755](https://github.com/headroomlabs-ai/headroom/commit/c9a7755a281d04d4ec8e37fa62b0d66115c89e0c))
* **proxy:** hoist ccr_workspace_key default so /v1/messages survives CCR-inject off ([#1096](https://github.com/headroomlabs-ai/headroom/issues/1096)) ([1deb947](https://github.com/headroomlabs-ai/headroom/commit/1deb947ac139f8525c0da2bbe86d6caacfc48a49))
* **proxy:** honor x-headroom-base-url on /v1/messages route ([#1763](https://github.com/headroomlabs-ai/headroom/issues/1763)) ([bb2acf7](https://github.com/headroomlabs-ai/headroom/commit/bb2acf700a62ee3a76ba181052e36904af4f11be))
* **proxy:** isolate image compression in a subprocess so a native SIGSEGV can't take down the proxy ([#2107](https://github.com/headroomlabs-ai/headroom/issues/2107)) ([#2162](https://github.com/headroomlabs-ai/headroom/issues/2162)) ([09e7212](https://github.com/headroomlabs-ai/headroom/commit/09e72125b427f9adc9239ae881cc56617550444b))
* **proxy:** keep anthropic ccr compression active across deferred injection ([#2291](https://github.com/headroomlabs-ai/headroom/issues/2291)) ([#2297](https://github.com/headroomlabs-ai/headroom/issues/2297)) ([26b43f6](https://github.com/headroomlabs-ai/headroom/commit/26b43f64d6863fafebb0e2a48e059dcb59b9b397))
* **proxy:** keep Kompress warmup off the startup path ([#2001](https://github.com/headroomlabs-ai/headroom/issues/2001)) ([10ed14e](https://github.com/headroomlabs-ai/headroom/commit/10ed14e7f68e4ad4f4c720d083032cd47d25b96c))
* **proxy:** keep PRE_SEND from reintroducing empty tool arrays ([#2015](https://github.com/headroomlabs-ai/headroom/issues/2015)) ([d1db00a](https://github.com/headroomlabs-ai/headroom/commit/d1db00ab8697505b56b294c611cf93463b71810b))
* **proxy:** keep recent stats request rows ([#1922](https://github.com/headroomlabs-ai/headroom/issues/1922)) ([bd8de9f](https://github.com/headroomlabs-ai/headroom/commit/bd8de9f3829e9f66e33d36a01464a99adcc42c1d))
* **proxy:** key drift detector on conversations, not credentials; canonicalize drift hashes ([#2301](https://github.com/headroomlabs-ai/headroom/issues/2301)) ([6744833](https://github.com/headroomlabs-ai/headroom/commit/6744833afed9aba7bdadd2aa59ed3a6fa99afc15))
* **proxy:** one bad extension no longer aborts proxy startup ([#2215](https://github.com/headroomlabs-ai/headroom/issues/2215)) ([cb6c828](https://github.com/headroomlabs-ai/headroom/commit/cb6c8284575b70d754f5fc83158d8ef38777526b))
* **proxy:** only queue mid-turn messages for opt-in clients with explicit session header ([#1951](https://github.com/headroomlabs-ai/headroom/issues/1951)) ([c365c7f](https://github.com/headroomlabs-ai/headroom/commit/c365c7ff81dd79a7abc5012271f1c73dfe84fa4c))
* **proxy:** preserve chatgpt responses streaming ([#2012](https://github.com/headroomlabs-ai/headroom/issues/2012)) ([a617455](https://github.com/headroomlabs-ai/headroom/commit/a617455f0242328862b43f4f4be9cc67a9c99e0a))
* **proxy:** preserve content-part array structure in excluded-tool lossless fold write-back ([#2261](https://github.com/headroomlabs-ai/headroom/issues/2261)) ([8951a26](https://github.com/headroomlabs-ai/headroom/commit/8951a264a2dea417a733ce446463cc5ee4d74781))
* **proxy:** preserve sub-path in X-Headroom-Base-Url custom upstream ([#2037](https://github.com/headroomlabs-ai/headroom/issues/2037)) ([#2127](https://github.com/headroomlabs-ai/headroom/issues/2127)) ([2976d49](https://github.com/headroomlabs-ai/headroom/commit/2976d49f18f393e199b2a8e98465fc5701e83cb9))
* **proxy:** preserve terminal tool on Codex Responses ([#2000](https://github.com/headroomlabs-ai/headroom/issues/2000)) ([41af39d](https://github.com/headroomlabs-ai/headroom/commit/41af39d769ef132cf953ecd18803675c871fc4a6))
* **proxy:** preserve upstream 5xx status on retry exhaustion ([#1570](https://github.com/headroomlabs-ai/headroom/issues/1570)) ([7836aea](https://github.com/headroomlabs-ai/headroom/commit/7836aea2be8577b0f593c9cfcff1da5f5e0ee3c8))
* **proxy:** protect WebSearch/WebFetch tool results from lossy compression ([#2115](https://github.com/headroomlabs-ai/headroom/issues/2115)) ([d2fbd55](https://github.com/headroomlabs-ai/headroom/commit/d2fbd55b8e230ed38dbf281520c96f9bd6c53ac9))
* **proxy:** quarantine compression while timed-out workers run ([#2292](https://github.com/headroomlabs-ai/headroom/issues/2292)) ([517bf99](https://github.com/headroomlabs-ai/headroom/commit/517bf992cfb096c9f277d1f39aa24f9a979600dc))
* **proxy:** record cache metrics for non-streaming backend paths ([#1271](https://github.com/headroomlabs-ai/headroom/issues/1271)) ([8580404](https://github.com/headroomlabs-ai/headroom/commit/85804043ff1f418148dd00c42a2dcdffe61a57a6))
* **proxy:** record Prometheus metrics for POST /v1/compress ([#2247](https://github.com/headroomlabs-ai/headroom/issues/2247)) ([81d40a6](https://github.com/headroomlabs-ai/headroom/commit/81d40a6437979d6b8ebc524ff8c5a9d9b7c2d10a))
* **proxy:** reject rate_limit_requests_per_minute=0 when limiting is enabled ([#2142](https://github.com/headroomlabs-ai/headroom/issues/2142)) ([8a71947](https://github.com/headroomlabs-ai/headroom/commit/8a71947023f41efdc81887c531b134f135e8ef67))
* **proxy:** repair main lint (ruff-format drift + mypy host_header) ([#2268](https://github.com/headroomlabs-ai/headroom/issues/2268)) ([718c8dc](https://github.com/headroomlabs-ai/headroom/commit/718c8dc559c0432d234da28053cea602c2d9245a))
* **proxy:** satisfy rustfmt import ordering ([#2158](https://github.com/headroomlabs-ai/headroom/issues/2158)) ([f008336](https://github.com/headroomlabs-ai/headroom/commit/f00833654f187eade7c70a2e5be549b75399bbab))
* **proxy:** skip Responses memory tools for ChatGPT auth ([#1579](https://github.com/headroomlabs-ai/headroom/issues/1579)) ([1c50eca](https://github.com/headroomlabs-ai/headroom/commit/1c50eca8b3f44ba06fee3d1cae22f3cf40eac5f8))
* **proxy:** strip [1m] model suffix before upstream forwarding ([#2027](https://github.com/headroomlabs-ai/headroom/issues/2027)) ([52a024d](https://github.com/headroomlabs-ai/headroom/commit/52a024d28cff7808659240b3f4c5ceb4fa11e0e8))
* **proxy:** Strip Codex responses-lite marker from response.create frame body ([#1820](https://github.com/headroomlabs-ai/headroom/issues/1820)) ([5cece7b](https://github.com/headroomlabs-ai/headroom/commit/5cece7bf587c38521f01951b96847892ba59ffe4))
* **proxy:** strip duplicated upstream server headers ([#1828](https://github.com/headroomlabs-ai/headroom/issues/1828)) ([d2a86b5](https://github.com/headroomlabs-ai/headroom/commit/d2a86b590978cae32bf95a20013ad539e942be31))
* **proxy:** strip inbound Content-Encoding on messages/chat forward ([#1970](https://github.com/headroomlabs-ai/headroom/issues/1970)) ([4cb33cd](https://github.com/headroomlabs-ai/headroom/commit/4cb33cd9e3766a1255cd9fbb7ac577c1d62b1aa2))
* **proxy:** support Codex WS compatible gateways ([#1281](https://github.com/headroomlabs-ai/headroom/issues/1281)) ([ac7ee4e](https://github.com/headroomlabs-ai/headroom/commit/ac7ee4e0bf25a49f05df44a38d40ecb806cf3972))
* **proxy:** support Windows selector loop on uvicorn &lt; 0.36 ([#1655](https://github.com/headroomlabs-ai/headroom/issues/1655)) ([e0eb094](https://github.com/headroomlabs-ai/headroom/commit/e0eb0943f0b6985a3ee6cb108636655ad80ca69e))
* **release:** sync all package versions to v0.31.0 ([#1882](https://github.com/headroomlabs-ai/headroom/issues/1882)) ([662b7bc](https://github.com/headroomlabs-ai/headroom/commit/662b7bc00eb4cfb1f72449e510fe576d240db384))
* replace computer_call_output with apply_patch_call_output in output_shaper ([#2250](https://github.com/headroomlabs-ai/headroom/issues/2250)) ([63f74aa](https://github.com/headroomlabs-ai/headroom/commit/63f74aa3e6477db9ac2049007a71ec5a3522649c))
* **router:** stop protecting passing build/test output as error traces ([#1740](https://github.com/headroomlabs-ai/headroom/issues/1740)) ([7ab83c5](https://github.com/headroomlabs-ai/headroom/commit/7ab83c5107b5183a652f566adeffc7a6ef16a8bc))
* **savings:** cap ledger retention at 30 days ([#1985](https://github.com/headroomlabs-ai/headroom/issues/1985)) ([b3a559b](https://github.com/headroomlabs-ai/headroom/commit/b3a559ba56a3b35308dbff2844408762899d037c))
* **savings:** coding profile compresses the recent delta (protect_recent 2-&gt;0, min_tokens 25-&gt;10) ([#2145](https://github.com/headroomlabs-ai/headroom/issues/2145)) ([eca3db6](https://github.com/headroomlabs-ai/headroom/commit/eca3db62a33c5802453fce87c5bfd1e5ff42b100))
* **savings:** don't bill free models at the $3/M fallback in the ledger ([#2147](https://github.com/headroomlabs-ai/headroom/issues/2147)) ([fb17156](https://github.com/headroomlabs-ai/headroom/commit/fb17156bfae156bd8f62668c19a4f560e11102e0))
* **savings:** don't fabricate output savings for a free (zero-priced) model ([#2298](https://github.com/headroomlabs-ai/headroom/issues/2298)) ([ec12e18](https://github.com/headroomlabs-ai/headroom/commit/ec12e18186852118204ec9ac8fdd33e068c0dcad))
* **savings:** record pre-compression original as ledger before, not forwarded count ([#2176](https://github.com/headroomlabs-ai/headroom/issues/2176)) ([195ed90](https://github.com/headroomlabs-ai/headroom/commit/195ed90ced746408f05d6abb7d60ac3ec1a55ccc))
* **scripts:** rename .releaseetadata to .releasemetadata ([#1246](https://github.com/headroomlabs-ai/headroom/issues/1246)) ([772adc9](https://github.com/headroomlabs-ai/headroom/commit/772adc93b253d73e91a1a4888e5338f1f71a887a))
* **search-compressor:** CJK-aware relevance + harden Rust/Python parity ([#1749](https://github.com/headroomlabs-ai/headroom/issues/1749)) ([985621d](https://github.com/headroomlabs-ai/headroom/commit/985621d60e3c80d94d1205b863bb4974cd346b62))
* **stats:** tag streamed output token source ([#2214](https://github.com/headroomlabs-ai/headroom/issues/2214)) ([1c9585d](https://github.com/headroomlabs-ai/headroom/commit/1c9585d42ee4167def21cf5e823cdb7ee329c33f))
* strip output-only fallback blocks from request messages ([#1870](https://github.com/headroomlabs-ai/headroom/issues/1870)) ([1448718](https://github.com/headroomlabs-ai/headroom/commit/1448718fcadcc6cde474719105289924155dce7c))
* **subscription/copilot:** preserve remaining=0 for exhausted quota ([#1997](https://github.com/headroomlabs-ai/headroom/issues/1997)) ([cbb7750](https://github.com/headroomlabs-ai/headroom/commit/cbb775015e30c8eb783bcec44c1f46b7c12cab48))
* **subscription:** keep efficiency_pct from exceeding 100% ([#2121](https://github.com/headroomlabs-ai/headroom/issues/2121)) ([5fb449e](https://github.com/headroomlabs-ai/headroom/commit/5fb449e90be354c5846b9ba96dedbbcd8201eff7))
* **subscription:** read newest transcript tail ([#2310](https://github.com/headroomlabs-ai/headroom/issues/2310)) ([793d20f](https://github.com/headroomlabs-ai/headroom/commit/793d20fb2a1c0d3608abd23c4587db1b17dc238d))
* **telemetry:** only advance usage-report baseline after a 200 ([#2149](https://github.com/headroomlabs-ai/headroom/issues/2149)) ([0cddac6](https://github.com/headroomlabs-ai/headroom/commit/0cddac632d8ab4e0bcd752b9c7519d5099aa085e))
* **tests:** repair three main-branch test failures ([#2306](https://github.com/headroomlabs-ai/headroom/issues/2306)) ([1d79e70](https://github.com/headroomlabs-ai/headroom/commit/1d79e70f9598f866be6c11a64e81887625127a94))
* **tokenizers:** don't tokenize image blocks as text in TiktokenCounter ([#2093](https://github.com/headroomlabs-ai/headroom/issues/2093)) ([ae10d6c](https://github.com/headroomlabs-ai/headroom/commit/ae10d6c99d7b187a92a9bcef3e7aabfe8b43a97f))
* **tokenizers:** price CJK in the fixed-ratio estimator path ([#2080](https://github.com/headroomlabs-ai/headroom/issues/2080)) ([cd3d5aa](https://github.com/headroomlabs-ai/headroom/commit/cd3d5aa10c5f43b4dbf8e3741e753d014dcfad7b))
* **tokenizers:** recurse into list-content tool_result blocks ([#2081](https://github.com/headroomlabs-ai/headroom/issues/2081)) ([dfb1d37](https://github.com/headroomlabs-ai/headroom/commit/dfb1d37ed619d30c8e79b2bc637d56e524224684))
* **tokenizers:** resolve HF tokenizer names by most-specific prefix ([#2096](https://github.com/headroomlabs-ai/headroom/issues/2096)) ([e0232df](https://github.com/headroomlabs-ai/headroom/commit/e0232df9b42b326358addacb8980aa1dd207ee01))
* **tokenizers:** use o200k_base for gpt-4.1/gpt-4.5/o4 families ([#2108](https://github.com/headroomlabs-ai/headroom/issues/2108)) ([6979b52](https://github.com/headroomlabs-ai/headroom/commit/6979b5245eb36020e8b2f806a8fe2137901961af))
* **transforms/code:** coerce language aliases instead of raising ([#1975](https://github.com/headroomlabs-ai/headroom/issues/1975)) ([27ddde1](https://github.com/headroomlabs-ai/headroom/commit/27ddde1f5e3ced40ca237bc6bbfbe76cb896d97a))
* **transforms:** guard Log fallback against invalid JSON + fix MIXED false-positive on source code ([#1347](https://github.com/headroomlabs-ai/headroom/issues/1347)) ([02c7764](https://github.com/headroomlabs-ai/headroom/commit/02c77640a95e04a33361803fbc25d63d0b86e976))
* **transforms:** guard the lossless diff fold to diff-shaped content only ([#2140](https://github.com/headroomlabs-ai/headroom/issues/2140)) ([5e0f1a2](https://github.com/headroomlabs-ai/headroom/commit/5e0f1a219f4d079a7004ca4db4532cafa0746dc4))
* **update:** let Windows self-update replace headroom.exe ([#2016](https://github.com/headroomlabs-ai/headroom/issues/2016)) ([2678bb1](https://github.com/headroomlabs-ai/headroom/commit/2678bb1db6210d71256572c913183b76ff56f00c))
* **update:** prevent _core.pyd corruption on Windows when proxy is running ([#1581](https://github.com/headroomlabs-ai/headroom/issues/1581)) ([0750bbf](https://github.com/headroomlabs-ai/headroom/commit/0750bbff4df3a66f11c8b82c33224ef3264fae42))
* **version:** mark source-checkout builds as -dev ([#2072](https://github.com/headroomlabs-ai/headroom/issues/2072)) ([1cc9979](https://github.com/headroomlabs-ai/headroom/commit/1cc99792ac087fb8df916b1fd00afa58dae85e29))
* **windows:** unwedge compression on degraded ONNX runtimes (every request timing out at 30s, 0% savings) ([#822](https://github.com/headroomlabs-ai/headroom/issues/822)) ([36202f4](https://github.com/headroomlabs-ai/headroom/commit/36202f4d0bd4dd919dba09f1b4a653754f7c2b2c))
* **wrap/claude:** bind _wrap_settings_path before the try ([#2126](https://github.com/headroomlabs-ai/headroom/issues/2126)) ([faed4dc](https://github.com/headroomlabs-ai/headroom/commit/faed4dcfe72a329cc308057efd97fc99885173c4))
* **wrap/codex:** export the detected custom upstream base URL ([#2125](https://github.com/headroomlabs-ai/headroom/issues/2125)) ([d236b27](https://github.com/headroomlabs-ai/headroom/commit/d236b27c607c980a224e24a4e7c0282aea487cc4))
* **wrap/opencode:** unwrap removes the rtk block from AGENTS.md ([#2025](https://github.com/headroomlabs-ai/headroom/issues/2025)) ([20968a4](https://github.com/headroomlabs-ai/headroom/commit/20968a4fa4611abd8059d75804879faa5499dbfa))
* **wrap:** drop -p short flag from wrap claude so claude's own -p/--print passes through ([#2048](https://github.com/headroomlabs-ai/headroom/issues/2048)) ([14011b4](https://github.com/headroomlabs-ai/headroom/commit/14011b42dd752fe9a29a988e343ae84255032ab2))
* **wrap:** keep Claude context-tool setup explicit ([#1999](https://github.com/headroomlabs-ai/headroom/issues/1999)) ([f536aa0](https://github.com/headroomlabs-ai/headroom/commit/f536aa0801af554dec87cb278a0c31cc049c0c26))
* **wrap:** preserve custom Codex provider base_url during proxy injection ([#1894](https://github.com/headroomlabs-ai/headroom/issues/1894)) ([372d6c8](https://github.com/headroomlabs-ai/headroom/commit/372d6c8cd4bec2d7f7448bf23503b11570f72b47))
* **wrap:** read/write instruction files as UTF-8 on Windows ([#1245](https://github.com/headroomlabs-ai/headroom/issues/1245)) ([6413cc7](https://github.com/headroomlabs-ai/headroom/commit/6413cc75a243d441d8eb9bb420f24da7a26ef0a9))
* **wrap:** self-heal a stale ANTHROPIC_BASE_URL left by a dead proxy ([#2223](https://github.com/headroomlabs-ai/headroom/issues/2223)) ([8537e2c](https://github.com/headroomlabs-ai/headroom/commit/8537e2cf605dd0e378dd83329640c3215699919f))
* **wrap:** surface Claude Remote Control base-URL gate accurately ([#1](https://github.com/headroomlabs-ai/headroom/issues/1)… ([#1883](https://github.com/headroomlabs-ai/headroom/issues/1883)) ([daeff69](https://github.com/headroomlabs-ai/headroom/commit/daeff69a7549d40b579caf2b5d6a2840a5df68b5))
* **wrap:** use canonical headroom-openclaw npm package for wrap openclaw ([#1969](https://github.com/headroomlabs-ai/headroom/issues/1969)) ([#2120](https://github.com/headroomlabs-ai/headroom/issues/2120)) ([c5545d6](https://github.com/headroomlabs-ai/headroom/commit/c5545d6ac47a71efce6e9c3c9293c88bc8c0951a))


### Performance Improvements

* surface optimization overhead diagnostics ([#1212](https://github.com/headroomlabs-ai/headroom/issues/1212)) ([7ddcbcb](https://github.com/headroomlabs-ai/headroom/commit/7ddcbcb616729e09d8c556f2b8510f50aa0ae298))


### Dependencies

* bump @types/node from 22.19.15 to 26.1.1 in /plugins/openclaw ([#1685](https://github.com/headroomlabs-ai/headroom/issues/1685)) ([350daeb](https://github.com/headroomlabs-ai/headroom/commit/350daeba73edd0954d405fb65580e5ce46bf2be9))
* bump @types/node from 22.20.0 to 26.1.1 in /plugins/opencode ([#1688](https://github.com/headroomlabs-ai/headroom/issues/1688)) ([8715195](https://github.com/headroomlabs-ai/headroom/commit/87151952eeb85165d14034a1a0f77a70ae824848))
* bump @types/node from 25.5.2 to 26.1.1 in /docs ([#1683](https://github.com/headroomlabs-ai/headroom/issues/1683)) ([75fff43](https://github.com/headroomlabs-ai/headroom/commit/75fff43eca7901d1e4809c8e7762b974d29f5c14))
* bump fumadocs-typescript from 4.0.14 to 5.3.0 in /docs ([#1684](https://github.com/headroomlabs-ai/headroom/issues/1684)) ([e8b66a2](https://github.com/headroomlabs-ai/headroom/commit/e8b66a27e1bb7d6452172530ffc024a5f109c49b))
* bump prometheus from 0.13.4 to 0.14.0 ([#1518](https://github.com/headroomlabs-ai/headroom/issues/1518)) ([5229c98](https://github.com/headroomlabs-ai/headroom/commit/5229c98228e524fb3df902fb0b56a3eb54e1a2c5))
* bump thiserror from 1.0.69 to 2.0.18 ([#1519](https://github.com/headroomlabs-ai/headroom/issues/1519)) ([e448d7b](https://github.com/headroomlabs-ai/headroom/commit/e448d7ba4d6366a883d77a3d7f882cb1a71a0550))
* bump toml from 0.8.23 to 1.1.2+spec-1.1.0 ([#1517](https://github.com/headroomlabs-ai/headroom/issues/1517)) ([6c705b4](https://github.com/headroomlabs-ai/headroom/commit/6c705b40667fc8998cf9b72da0e44284ed8d0853))
* update tree-sitter requirement from &lt;0.26,&gt;=0.25.2 to &gt;=0.25.2,&lt;0.27 ([#1681](https://github.com/headroomlabs-ai/headroom/issues/1681)) ([ce3c959](https://github.com/headroomlabs-ai/headroom/commit/ce3c959eaed1bd53f492ae7cc612a3fd7b12daf3))


### Code Refactoring

* **cache:** isolate compression strategy outcomes ([#1938](https://github.com/headroomlabs-ai/headroom/issues/1938)) ([b5aa8a3](https://github.com/headroomlabs-ai/headroom/commit/b5aa8a358e70edf875bb768fae0ee04af0dd7921))
* **cache:** isolate semantic key policy ([#1953](https://github.com/headroomlabs-ai/headroom/issues/1953)) ([740fb9b](https://github.com/headroomlabs-ai/headroom/commit/740fb9bc16eb3d9db57b253ac22e62e52cb19860))
* **ccr:** isolate tool call classification ([#1937](https://github.com/headroomlabs-ai/headroom/issues/1937)) ([fd5b9e7](https://github.com/headroomlabs-ai/headroom/commit/fd5b9e75ad69db670bd9e0c77a816dd0fda6a199))
* **memory:** isolate injection decision policy ([#1952](https://github.com/headroomlabs-ai/headroom/issues/1952)) ([c20f3b1](https://github.com/headroomlabs-ai/headroom/commit/c20f3b1c0434b3d100a1d5edca22d8c9c0f7c9f9))
* **memory:** isolate query construction policy ([#1950](https://github.com/headroomlabs-ai/headroom/issues/1950)) ([235c986](https://github.com/headroomlabs-ai/headroom/commit/235c986c9cc7d36fffc4298bbcab77079d8f2f43))
* **output:** isolate savings policy ([#1947](https://github.com/headroomlabs-ai/headroom/issues/1947)) ([c29b4ba](https://github.com/headroomlabs-ai/headroom/commit/c29b4ba84f021c62c869a12e14ce8671e2f01141))
* **output:** isolate verbosity steering ([#1940](https://github.com/headroomlabs-ai/headroom/issues/1940)) ([0ce09fb](https://github.com/headroomlabs-ai/headroom/commit/0ce09fb63fff6d6c5180cecc086f6ae64b29975b))
* **pricing:** isolate litellm model resolution ([#1936](https://github.com/headroomlabs-ai/headroom/issues/1936)) ([4210d6e](https://github.com/headroomlabs-ai/headroom/commit/4210d6e60954ee5a091c9669a92e36e830bae7de))
* **providers:** split proxy route adapters ([#1934](https://github.com/headroomlabs-ai/headroom/issues/1934)) ([e6243f6](https://github.com/headroomlabs-ai/headroom/commit/e6243f65c9675790515eb2894ad174ab80cea23a))
* **proxy:** extract beta header merge policy ([#1993](https://github.com/headroomlabs-ai/headroom/issues/1993)) ([f359f21](https://github.com/headroomlabs-ai/headroom/commit/f359f21424a54e9d9ef34ca7de49a9d11aa50589))
* **proxy:** extract beta header policy ([#1992](https://github.com/headroomlabs-ai/headroom/issues/1992)) ([603f5bc](https://github.com/headroomlabs-ai/headroom/commit/603f5bcfd663d2fe71f11e0aabc76c8993a0456b))
* **proxy:** extract ccr golden replay policy ([#2006](https://github.com/headroomlabs-ai/headroom/issues/2006)) ([7c9a032](https://github.com/headroomlabs-ai/headroom/commit/7c9a032f5044f24fdde71126f591f3eac0a93da1))
* **proxy:** extract ccr marker policy ([#2004](https://github.com/headroomlabs-ai/headroom/issues/2004)) ([ec3c3cd](https://github.com/headroomlabs-ai/headroom/commit/ec3c3cd2345a9aed682eec719370c36bb79e2173))
* **proxy:** extract ccr session tracker ([#2003](https://github.com/headroomlabs-ai/headroom/issues/2003)) ([e92c253](https://github.com/headroomlabs-ai/headroom/commit/e92c2539779dfb04404f7618dfb9bb87e01e88f1))
* **proxy:** extract internal header policy ([#1990](https://github.com/headroomlabs-ai/headroom/issues/1990)) ([868b88b](https://github.com/headroomlabs-ai/headroom/commit/868b88bc6400c98f11134dbbe3cb03d1ecff7e1d))
* **proxy:** extract memory golden replay policy ([#2007](https://github.com/headroomlabs-ai/headroom/issues/2007)) ([8c68f48](https://github.com/headroomlabs-ai/headroom/commit/8c68f48903b354f8ff19c74df7652bbcc3693185))
* **proxy:** extract tool definition serialization ([#1998](https://github.com/headroomlabs-ai/headroom/issues/1998)) ([ad6ab48](https://github.com/headroomlabs-ai/headroom/commit/ad6ab48cbb2a9cf12f1d7cefaf4699a9905a92bb))
* **proxy:** extract tool injection config ([#2010](https://github.com/headroomlabs-ai/headroom/issues/2010)) ([0f846e5](https://github.com/headroomlabs-ai/headroom/commit/0f846e5a8fb58942431b1edc22dfda9ab7d6de70))
* **proxy:** extract tool injection logging ([#2009](https://github.com/headroomlabs-ai/headroom/issues/2009)) ([9c7b9d5](https://github.com/headroomlabs-ai/headroom/commit/9c7b9d5a9c7a5b2c55422f85976c796045f477fc))
* **proxy:** extract tool injection policy ([#1995](https://github.com/headroomlabs-ai/headroom/issues/1995)) ([d6259b2](https://github.com/headroomlabs-ai/headroom/commit/d6259b226365abba47048a49a579ead23e55684c))
* **proxy:** extract tool injection tracker ([#2002](https://github.com/headroomlabs-ai/headroom/issues/2002)) ([d1c484b](https://github.com/headroomlabs-ai/headroom/commit/d1c484b164145f5010a048d299a61d7dac903616))
* **proxy:** extract tool name policy ([#2008](https://github.com/headroomlabs-ai/headroom/issues/2008)) ([1000175](https://github.com/headroomlabs-ai/headroom/commit/10001755e85a5acaa50644ee205adaa01d8c55d8))
* **proxy:** isolate auth classification policy ([#1945](https://github.com/headroomlabs-ai/headroom/issues/1945)) ([5a7265d](https://github.com/headroomlabs-ai/headroom/commit/5a7265daa80df88e3850a6dcc834364fc3ced5a4))
* **proxy:** isolate body forwarding policy ([#1935](https://github.com/headroomlabs-ai/headroom/issues/1935)) ([1f3696a](https://github.com/headroomlabs-ai/headroom/commit/1f3696a3d0c67e3bdff90ad0c93cf7643af0a496))
* **proxy:** isolate forwarded header policy ([#1942](https://github.com/headroomlabs-ai/headroom/issues/1942)) ([cb38f79](https://github.com/headroomlabs-ai/headroom/commit/cb38f7937705ba2c4db50792a7d1feb0a19018bb))
* **proxy:** isolate image compression policy ([#1958](https://github.com/headroomlabs-ai/headroom/issues/1958)) ([2b09ece](https://github.com/headroomlabs-ai/headroom/commit/2b09ecea76346a371bc7886eeb66d88d2489a50e))
* **proxy:** isolate memory rank policy ([#1960](https://github.com/headroomlabs-ai/headroom/issues/1960)) ([b1e871d](https://github.com/headroomlabs-ai/headroom/commit/b1e871d51ccd4e013da1cece02b18d1454503d77))
* **proxy:** isolate output effort policy ([#1961](https://github.com/headroomlabs-ai/headroom/issues/1961)) ([094a53c](https://github.com/headroomlabs-ai/headroom/commit/094a53c0479087188693d707b396d8a778b9e6bc))
* **proxy:** isolate output turn policy ([#1962](https://github.com/headroomlabs-ai/headroom/issues/1962)) ([c904a70](https://github.com/headroomlabs-ai/headroom/commit/c904a70d4ef9d2955d5c96d86900cbe9ecf1e51c))
* **proxy:** isolate output verbosity policy ([#1963](https://github.com/headroomlabs-ai/headroom/issues/1963)) ([0415dc8](https://github.com/headroomlabs-ai/headroom/commit/0415dc87656f2a87b23abee1a329498398ce4b98))
* **proxy:** isolate project attribution policy ([#1957](https://github.com/headroomlabs-ai/headroom/issues/1957)) ([1c1e360](https://github.com/headroomlabs-ai/headroom/commit/1c1e3601120df95142fb292ccc2dd5a0e3ab30e6))
* **proxy:** isolate proxy mode policy ([#1965](https://github.com/headroomlabs-ai/headroom/issues/1965)) ([82af5cd](https://github.com/headroomlabs-ai/headroom/commit/82af5cdfe256177d1aac01f191e694f81786c209))
* **proxy:** isolate rate limit policy ([#1954](https://github.com/headroomlabs-ai/headroom/issues/1954)) ([ea19515](https://github.com/headroomlabs-ai/headroom/commit/ea1951508b3af6090b71852ec58fd814032b1253))
* **proxy:** isolate semantic cache key policy ([#1964](https://github.com/headroomlabs-ai/headroom/issues/1964)) ([2f53a18](https://github.com/headroomlabs-ai/headroom/commit/2f53a18a3fe2a8ce9c8f57d68ee54a834fb4ba1f))
* **transforms:** isolate mixed content parsing ([#1939](https://github.com/headroomlabs-ai/headroom/issues/1939)) ([9bacf48](https://github.com/headroomlabs-ai/headroom/commit/9bacf4810fe5a950c644b38926801d5aa0382e25))

## [0.32.0](https://github.com/JerrettDavis/headroom/compare/v0.31.0...v0.32.0) (2026-07-13)


### Features

* add --disable-kompress-fallback to restore legacy PASSTHROUGH fallback ([#1185](https://github.com/JerrettDavis/headroom/issues/1185)) ([f309244](https://github.com/JerrettDavis/headroom/commit/f309244a77fc3fbb74c5db0082e7dcbebd6ffe52))
* add first-class OpenCode support (wrap, learn, mcp install) ([#559](https://github.com/JerrettDavis/headroom/issues/559)) ([91cd210](https://github.com/JerrettDavis/headroom/commit/91cd2102d7e9bc5d48a594725ecc9593096996ec))
* add HEADROOM_KEEPALIVE_EXPIRY to keep upstream connections warm ([#1124](https://github.com/JerrettDavis/headroom/issues/1124)) ([85786b3](https://github.com/JerrettDavis/headroom/commit/85786b33a3a88b8c905739aa34ccfafa01a89e5d))
* Add support for Mistral Vibe CLI ([#935](https://github.com/JerrettDavis/headroom/issues/935)) ([0932b8b](https://github.com/JerrettDavis/headroom/commit/0932b8bef4db9109665382b6d7c079a368f08d52))
* **agent-savings:** land coding + general workload personas on main ([#1732](https://github.com/JerrettDavis/headroom/issues/1732)) ([d8db7da](https://github.com/JerrettDavis/headroom/commit/d8db7da77e07ca9af12c31f29435f6d1e6227d95))
* **anthropic:** add Claude 5 family pricing & align current rates ([#1767](https://github.com/JerrettDavis/headroom/issues/1767)) ([e84ca98](https://github.com/JerrettDavis/headroom/commit/e84ca980cff59f082a9faf38ba39a61b566aa009))
* **azure-foundry:** derive upstream URL from ANTHROPIC_FOUNDRY_RESOURCE ([#1138](https://github.com/JerrettDavis/headroom/issues/1138)) ([e5031b0](https://github.com/JerrettDavis/headroom/commit/e5031b01219278620431b5560b247e65f1b08a13))
* **cache:** attribute prompt-cache misses to TTL lapse vs prefix change ([#1313](https://github.com/JerrettDavis/headroom/issues/1313)) ([#1343](https://github.com/JerrettDavis/headroom/issues/1343)) ([4658721](https://github.com/JerrettDavis/headroom/commit/4658721ea0bae5d0d061d377428d4031b9722d75))
* **cache:** provider-agnostic cache-mode delta + cc-agnostic prefix comparison ([#1868](https://github.com/JerrettDavis/headroom/issues/1868)) ([7c2f0ea](https://github.com/JerrettDavis/headroom/commit/7c2f0ea07953beaed45b25bd0fc8c5a34d60cb3f))
* **ccr:** wire retrieve-tool interception into OpenAI Responses handler ([#1898](https://github.com/JerrettDavis/headroom/issues/1898)) ([62cd307](https://github.com/JerrettDavis/headroom/commit/62cd3072a2ea9bcd8410e400cab6f678501b5b37))
* **cli:** add headroom doctor setup diagnostics ([#926](https://github.com/JerrettDavis/headroom/issues/926)) ([e45cf4e](https://github.com/JerrettDavis/headroom/commit/e45cf4e0618b4de02608f68c502ac4cf1270eb84))
* **cli:** add headroom update command and release banner ([#1088](https://github.com/JerrettDavis/headroom/issues/1088)) ([26be2c3](https://github.com/JerrettDavis/headroom/commit/26be2c39cb8a3c23edc08516f01cf91fad33c117))
* **code:** add Perl support to code-aware compressor ([#1125](https://github.com/JerrettDavis/headroom/issues/1125)) ([f39858c](https://github.com/JerrettDavis/headroom/commit/f39858c23325f9f27b47a738731e7260f7b59d9e))
* **codex:** keep wrap routing session-scoped ([#1507](https://github.com/JerrettDavis/headroom/issues/1507)) ([ad9d086](https://github.com/JerrettDavis/headroom/commit/ad9d086f43a664c4c2a19060b847f2e03ce4f6ad))
* compression extraction — Rust knob exposure, CCR hardening, traffic audits ([#818](https://github.com/JerrettDavis/headroom/issues/818)) ([b7be381](https://github.com/JerrettDavis/headroom/commit/b7be3814f1d38375bc27901272bbe919e6b35940))
* **compression:** add audit-safe mode with protected pattern matching ([#1899](https://github.com/JerrettDavis/headroom/issues/1899)) ([bb112dd](https://github.com/JerrettDavis/headroom/commit/bb112dd1762bf744a05689d54c50aed28265ee90))
* **content-router:** accept any real compression (remove min-savings floor) ([#1771](https://github.com/JerrettDavis/headroom/issues/1771)) ([6c31db9](https://github.com/JerrettDavis/headroom/commit/6c31db97fbd68f88c39a71785335fc8917702fc3))
* **content-router:** lossless-excluded compaction (grep/log/json) + enable in coding/general personas ([#1762](https://github.com/JerrettDavis/headroom/issues/1762)) ([f067040](https://github.com/JerrettDavis/headroom/commit/f0670404ce21b93ab2ca3d1f93e2fad183562335))
* **content-router:** lossless-first dispatch, cross-turn dedup, and A7 lossy-after-fold ([#1818](https://github.com/JerrettDavis/headroom/issues/1818)) ([60af15f](https://github.com/JerrettDavis/headroom/commit/60af15f96f1792ad50bf259a765ee188db73d1aa))
* headroom wrap opencode / unwrap opencode CLI ([#1105](https://github.com/JerrettDavis/headroom/issues/1105)) ([b4571cc](https://github.com/JerrettDavis/headroom/commit/b4571cc346f6bba29e600fa82bbf5cf302e8ea27))
* **learn:** weight loops in Headroom Learn + RTK-loop eval ([#1160](https://github.com/JerrettDavis/headroom/issues/1160)) ([14e8dc4](https://github.com/JerrettDavis/headroom/commit/14e8dc4c8408b8014433ba7589bbb1dff7805134))
* **learn:** write per-project learnings to CLAUDE.local.md by default ([#1115](https://github.com/JerrettDavis/headroom/issues/1115)) ([ced75e4](https://github.com/JerrettDavis/headroom/commit/ced75e4718b5fd84d07cbd68273dcf9b9ef878a3))
* measure and surface token throughput (tokens/sec) through the proxy ([#983](https://github.com/JerrettDavis/headroom/issues/983)) ([0d89c67](https://github.com/JerrettDavis/headroom/commit/0d89c674cd3522c0a46e3df9b98426e59b337b10))
* **observability:** add gen_ai.request.model to the compression span ([#1667](https://github.com/JerrettDavis/headroom/issues/1667)) ([7f7af66](https://github.com/JerrettDavis/headroom/commit/7f7af667ed3db718adad34f34d83517ac880760d))
* output-token reduction — verbosity shaper, per-user learning, counterfactual savings ([#965](https://github.com/JerrettDavis/headroom/issues/965)) ([a99dc61](https://github.com/JerrettDavis/headroom/commit/a99dc61424df4c7b22c37986fb8dfc648f3ac3b8))
* **policy:** decay P_alive from idle time near cache TTL ([#856](https://github.com/JerrettDavis/headroom/issues/856) P3b) ([#1028](https://github.com/JerrettDavis/headroom/issues/1028)) ([fe4f9ee](https://github.com/JerrettDavis/headroom/commit/fe4f9ee478f50a84190a2d44de2b9fbf24272acf))
* **providers:** add Cortex Code (Snowflake CoCo) as a supported agent ([#1190](https://github.com/JerrettDavis/headroom/issues/1190)) ([d9d0bf4](https://github.com/JerrettDavis/headroom/commit/d9d0bf4b79f57ce760f4ac236afe19721727d936))
* **proxy:** add --lossless no-CCR mode with format-native compaction ([#1721](https://github.com/JerrettDavis/headroom/issues/1721)) ([c75ebde](https://github.com/JerrettDavis/headroom/commit/c75ebdee6df9b1689a44ef321e36e8b360406ed7))
* **proxy:** add provider-only HTTP proxy ([#1807](https://github.com/JerrettDavis/headroom/issues/1807)) ([ebe0a3b](https://github.com/JerrettDavis/headroom/commit/ebe0a3bd7bbc8bbe4ee52bdb1ed7420a405dc224))
* **proxy:** add request timeout config ([#738](https://github.com/JerrettDavis/headroom/issues/738)) ([c0745d4](https://github.com/JerrettDavis/headroom/commit/c0745d4161d19e21ca36506f7733f0776e19e1a8))
* **proxy:** add turn-hook extension point for buffered model turns ([#1891](https://github.com/JerrettDavis/headroom/issues/1891)) ([ec950f7](https://github.com/JerrettDavis/headroom/commit/ec950f7ef131fb124b60a8e75bc6af7ab733cc7f))
* **proxy:** cc-switch reconciler — keep Headroom in the request path alongside cc-switch ([#1030](https://github.com/JerrettDavis/headroom/issues/1030)) ([e8fc8a0](https://github.com/JerrettDavis/headroom/commit/e8fc8a0d18a551bad572ec21aa92a424748683a5))
* **proxy:** expose retry delay configuration ([#2077](https://github.com/JerrettDavis/headroom/issues/2077)) ([099c664](https://github.com/JerrettDavis/headroom/commit/099c66432b7343a14ee67fcfc5db296145ea87ae))
* **proxy:** extend output shaping to the OpenAI Responses path (Codex HTTP + WS) ([#1943](https://github.com/JerrettDavis/headroom/issues/1943)) ([71cbb6a](https://github.com/JerrettDavis/headroom/commit/71cbb6aaada0b5f188f874e80a65fb2092c8ac13))
* **proxy:** hot-reload live env knobs so a reused proxy picks them up without a restart ([#1090](https://github.com/JerrettDavis/headroom/issues/1090)) ([6904d47](https://github.com/JerrettDavis/headroom/commit/6904d47a01e7be496e21d8ebcf34739db5c3b7dd))
* **proxy:** make COMPRESSION_TIMEOUT_SECONDS configurable via env ([#946](https://github.com/JerrettDavis/headroom/issues/946)) ([#991](https://github.com/JerrettDavis/headroom/issues/991)) ([addebdb](https://github.com/JerrettDavis/headroom/commit/addebdb29c3b4a877ed46553d9b0c0a128d62cef))
* **proxy:** persist per-model savings breakdown in proxy_savings.json ([#2055](https://github.com/JerrettDavis/headroom/issues/2055)) ([12a38d3](https://github.com/JerrettDavis/headroom/commit/12a38d31808283c935a4d15f297376d2b159e567))
* **proxy:** pilot hardening — inbound auth, security headers, audit log, air-gap switch ([#1537](https://github.com/JerrettDavis/headroom/issues/1537)) ([546ab55](https://github.com/JerrettDavis/headroom/commit/546ab553dc31af91d5ef4cec0589ad6db8e76a1d))
* **proxy:** support glob patterns in exclude_tools ([#870](https://github.com/JerrettDavis/headroom/issues/870)) ([#1259](https://github.com/JerrettDavis/headroom/issues/1259)) ([a2159c0](https://github.com/JerrettDavis/headroom/commit/a2159c0b66a7aa1b7f64057a1c8e3e50f0a43e37))
* **read-maturation:** activity-based hold-back Read maturation (Mechanism B) ([#1068](https://github.com/JerrettDavis/headroom/issues/1068)) ([723b80c](https://github.com/JerrettDavis/headroom/commit/723b80c09123f902197b45b3676065d0e9c77af0))
* **savings:** durable savings ledger + headroom savings command ([#1127](https://github.com/JerrettDavis/headroom/issues/1127)) ([978ffa0](https://github.com/JerrettDavis/headroom/commit/978ffa0a6ab9da1a75239270e17961530c213b9d))
* ship the coding profile as Headroom's out-of-box default posture ([#1893](https://github.com/JerrettDavis/headroom/issues/1893)) ([68676da](https://github.com/JerrettDavis/headroom/commit/68676daa5076286e2992e81a5cd7d583d82a71b9))
* **simulators:** add provider simulator service ([#2014](https://github.com/JerrettDavis/headroom/issues/2014)) ([2c9eb7c](https://github.com/JerrettDavis/headroom/commit/2c9eb7c5f154f087538bcde9561b68736c8c5584))
* **stats:** surface Codex WS compression counters in /stats summary ([#1680](https://github.com/JerrettDavis/headroom/issues/1680)) ([2fe19c3](https://github.com/JerrettDavis/headroom/commit/2fe19c39e40fc350af39f72e1a3bac28f9ce9874))
* **transforms:** adaptive Otsu KEEP/DROP threshold (+ land relevance split on main) ([#1726](https://github.com/JerrettDavis/headroom/issues/1726)) ([eea667a](https://github.com/JerrettDavis/headroom/commit/eea667a72019cc98401db9211907f67ddf45e7eb))
* **transforms:** tabular + spreadsheet (.xlsx/.xls) compression ([#1128](https://github.com/JerrettDavis/headroom/issues/1128)) ([d789a7c](https://github.com/JerrettDavis/headroom/commit/d789a7c528ceee1f4ba648a1002f2e6b6f620854))
* **vertex:** turnkey Claude Code + Vertex compression (+ fixes from the Vertex review) ([#1113](https://github.com/JerrettDavis/headroom/issues/1113)) ([0e05915](https://github.com/JerrettDavis/headroom/commit/0e0591506c3f120b96cdc98054114d9ec1771f67))
* **wrap:** add --1m to preserve the 1M context window on wrap claude ([#1158](https://github.com/JerrettDavis/headroom/issues/1158)) ([#1351](https://github.com/JerrettDavis/headroom/issues/1351)) ([b50d9c1](https://github.com/JerrettDavis/headroom/commit/b50d9c17ceca890a0fcc2469b9aff27d0026ca39))
* **wrap:** allow project RTK instruction opt-out ([#2078](https://github.com/JerrettDavis/headroom/issues/2078)) ([f53f720](https://github.com/JerrettDavis/headroom/commit/f53f720eb5e378c95250b55e3174a064621807db))
* **wrap:** make tokensave the primary coding-task compressor, Serena the backup ([#1230](https://github.com/JerrettDavis/headroom/issues/1230)) ([dca9853](https://github.com/JerrettDavis/headroom/commit/dca9853ed9d09fe1bb6d56fcb7bb82b9e90b7dff))


### Bug Fixes

* **adaptive-sizer:** char bigrams for spaceless CJK items ([#1748](https://github.com/JerrettDavis/headroom/issues/1748)) ([8879c50](https://github.com/JerrettDavis/headroom/commit/8879c50dbe929b220b17803a054a078f26b04ca9))
* **agent-evals:** Phase 0 — coding-agent accuracy A/B framework ([#1037](https://github.com/JerrettDavis/headroom/issues/1037)) ([84f9871](https://github.com/JerrettDavis/headroom/commit/84f9871e303d587f5b406036b97b9f5a689c1b05))
* **agno:** tolerate streaming tool-call SDK objects in parser ([#1312](https://github.com/JerrettDavis/headroom/issues/1312)) ([#1336](https://github.com/JerrettDavis/headroom/issues/1336)) ([5986c22](https://github.com/JerrettDavis/headroom/commit/5986c2260f07788e356e0884179d9b3f4c0df6e3))
* **bedrock:** add boto3 1.41 + CRT for aws login credentials ([#1486](https://github.com/JerrettDavis/headroom/issues/1486)) ([4db3bc9](https://github.com/JerrettDavis/headroom/commit/4db3bc91d9153ca1acccdc0cb5280da01194bf3e))
* **bedrock:** fail fast when session-token auth lacks botocore ([#1553](https://github.com/JerrettDavis/headroom/issues/1553)) ([54cfa36](https://github.com/JerrettDavis/headroom/commit/54cfa361d308dec567615c346af7c77d52ebb676))
* **bedrock:** resolve global.* inference profiles + pin per-user app-profile ARNs ([#1795](https://github.com/JerrettDavis/headroom/issues/1795)) ([33c7f6c](https://github.com/JerrettDavis/headroom/commit/33c7f6cd3ae14857effcbd110e9cd0668cf1ac1b))
* **bedrock:** route ARNs via converse, named AWS profiles, and au. re… ([#1456](https://github.com/JerrettDavis/headroom/issues/1456)) ([7d87aa2](https://github.com/JerrettDavis/headroom/commit/7d87aa2f1cbd93c970a77c6dfec8df03603251b9))
* **build:** enable Intel macOS pip installs via ort-load-dynamic ([#1538](https://github.com/JerrettDavis/headroom/issues/1538)) ([32ce99e](https://github.com/JerrettDavis/headroom/commit/32ce99e4b4a7d75f31429a553f2211a83992047a))
* bump codebase-memory-mcp to v0.8.1 ([#1284](https://github.com/JerrettDavis/headroom/issues/1284)) ([530318b](https://github.com/JerrettDavis/headroom/commit/530318b425cba8fb161111b135451a838d628e96))
* **cache/ccr:** don't count a successful eviction as a retrieval ([#2106](https://github.com/JerrettDavis/headroom/issues/2106)) ([eecb81e](https://github.com/JerrettDavis/headroom/commit/eecb81e8478dbe4337a48dcdc2ace6537df0784f))
* **cache/ccr:** don't evict a live entry on a duplicate store at capacity ([#2082](https://github.com/JerrettDavis/headroom/issues/2082)) ([1138946](https://github.com/JerrettDavis/headroom/commit/113894600cbc8aae219bd733ea8460a64e3e626f))
* **cache/semantic:** don't evict an unrelated entry on an update at capacity ([#2094](https://github.com/JerrettDavis/headroom/issues/2094)) ([cf6367a](https://github.com/JerrettDavis/headroom/commit/cf6367add4fce3294a790bd1a6733c97e62072d9))
* **cache/semantic:** key entries by context hash, not query text ([#2022](https://github.com/JerrettDavis/headroom/issues/2022)) ([d8783ab](https://github.com/JerrettDavis/headroom/commit/d8783ab89b49b309df4aae927d9f89ac3cbe058b))
* **cache:** avoid fallback session collisions ([#1827](https://github.com/JerrettDavis/headroom/issues/1827)) ([0f606b6](https://github.com/JerrettDavis/headroom/commit/0f606b6281dd4c55e1c5a32cc97c418b66860df1))
* **cache:** partial cached-prefix replay + idle-aware net-cost; don't… ([#1933](https://github.com/JerrettDavis/headroom/issues/1933)) ([b0440f9](https://github.com/JerrettDavis/headroom/commit/b0440f958dd596a13556a4d9a46d3e25d8f9f173))
* **cache:** stop DynamicContentDetector false positives corrupting cached prompts ([#2110](https://github.com/JerrettDavis/headroom/issues/2110)) ([#2119](https://github.com/JerrettDavis/headroom/issues/2119)) ([908a9a1](https://github.com/JerrettDavis/headroom/commit/908a9a1bb18b132007d8ca84ca784ec8e809186a))
* **ccr:** accept 12-char SmartCrusher hashes in tool injection ([#1095](https://github.com/JerrettDavis/headroom/issues/1095)) ([#1141](https://github.com/JerrettDavis/headroom/issues/1141)) ([9f7f3ad](https://github.com/JerrettDavis/headroom/commit/9f7f3adfea03710d5e67c4c630b3c8061ff6d161))
* **ccr:** don't crash parse_tool_call on non-object tool arguments ([#2071](https://github.com/JerrettDavis/headroom/issues/2071)) ([984a2c7](https://github.com/JerrettDavis/headroom/commit/984a2c702c7ea42436c914d0fdf434cf2dd3f45c))
* **ccr:** honor workspace dir for sqlite store ([#1564](https://github.com/JerrettDavis/headroom/issues/1564)) ([96e1dfe](https://github.com/JerrettDavis/headroom/commit/96e1dfe395a440f9e2dddf4589c4f6988f4ee4cd))
* **ccr:** make expired retrieve misses terminal ([#1781](https://github.com/JerrettDavis/headroom/issues/1781)) ([9cbdba4](https://github.com/JerrettDavis/headroom/commit/9cbdba4dc1f38f73d255211eec439674f3f2f9f1))
* **ccr:** make headroom_retrieve a hash-only full-content lookup ([#1532](https://github.com/JerrettDavis/headroom/issues/1532)) ([c2fc4d3](https://github.com/JerrettDavis/headroom/commit/c2fc4d3753c193eb61f78286741431fd1303e8ee))
* **ccr:** preserve Anthropic re-stream shape ([#1854](https://github.com/JerrettDavis/headroom/issues/1854)) ([f663894](https://github.com/JerrettDavis/headroom/commit/f663894f6072dbd13f5a1caa05dfea6657f5a3b0))
* **ccr:** preserve thinking blocks in buffered stream re-synthesis ([#1897](https://github.com/JerrettDavis/headroom/issues/1897)) ([ede085c](https://github.com/JerrettDavis/headroom/commit/ede085cc11d74778e43ce0fb0828a53a0a06a14b))
* **ccr:** propagate --no-ccr-marker flag to all compressors ([#1022](https://github.com/JerrettDavis/headroom/issues/1022)) ([#1197](https://github.com/JerrettDavis/headroom/issues/1197)) ([0c9b42a](https://github.com/JerrettDavis/headroom/commit/0c9b42a919b0c570094b7934de686b93dd89b05c))
* **ccr:** return stored content when headroom_retrieve query matches nothing ([#1213](https://github.com/JerrettDavis/headroom/issues/1213)) ([#1236](https://github.com/JerrettDavis/headroom/issues/1236)) ([08fb845](https://github.com/JerrettDavis/headroom/commit/08fb845fe37478af2c2f55c402df77d7a448fc86))
* **ccr:** skip Anthropic marker emission when tool injection is deferred ([#1273](https://github.com/JerrettDavis/headroom/issues/1273)) ([2cae13d](https://github.com/JerrettDavis/headroom/commit/2cae13dd798b8abdd9ef94fbcf10a968e70e714e))
* **ci:** extend gitleaks allowlist to cover test fixtures + verified examples ([#1539](https://github.com/JerrettDavis/headroom/issues/1539)) ([d2565a6](https://github.com/JerrettDavis/headroom/commit/d2565a6983f99fe6733d412405ee7c9e54d99624))
* **ci:** guarantee model present in test shards to end cache-miss flakiness ([#1399](https://github.com/JerrettDavis/headroom/issues/1399)) ([2e29c72](https://github.com/JerrettDavis/headroom/commit/2e29c7223f7a7694060dfe4e1d99332ad766a70b))
* **ci:** normalize Windows CRLF line endings in PR governance script ([#1012](https://github.com/JerrettDavis/headroom/issues/1012)) ([5194388](https://github.com/JerrettDavis/headroom/commit/5194388b6652d823ad6ab1d8c17d5572b7f0ec23))
* **claude:** surface Remote Control proxy incompatibility ([#1610](https://github.com/JerrettDavis/headroom/issues/1610)) ([4bf7f92](https://github.com/JerrettDavis/headroom/commit/4bf7f92417a8799ab3ae5f61b7ea9e96c5605a4f))
* **cli/proxy:** preserve explicit HEADROOM_MIN_TOKENS=0 / MAX_ITEMS=0 ([#1886](https://github.com/JerrettDavis/headroom/issues/1886)) ([3a33af1](https://github.com/JerrettDavis/headroom/commit/3a33af1af3224594581d0d27ea5b4df1a1c6ba48))
* **cli:** add explicit UTF-8 encoding to file I/O in wrap commands ([#1126](https://github.com/JerrettDavis/headroom/issues/1126)) ([#1164](https://github.com/JerrettDavis/headroom/issues/1164)) ([a0cb798](https://github.com/JerrettDavis/headroom/commit/a0cb7982e3cda52221719b9cceecd4d07e30c176))
* **cli:** fall back gracefully when embedding-server sidecar is absent ([#1206](https://github.com/JerrettDavis/headroom/issues/1206)) ([38f1404](https://github.com/JerrettDavis/headroom/commit/38f1404432984915924f74997d886b89c420b2a8))
* **cli:** harden all CLI surfaces + fix docs accuracy ([#1491](https://github.com/JerrettDavis/headroom/issues/1491)) ([bd76235](https://github.com/JerrettDavis/headroom/commit/bd76235f5c43bf2e3184a2c7e40a9954dc347afc))
* **cli:** stop advertising unwired compression tuning env vars in banner ([#1634](https://github.com/JerrettDavis/headroom/issues/1634)) ([d5bf98d](https://github.com/JerrettDavis/headroom/commit/d5bf98df31528dfd6c23ec45dbd3440efcb1cb75))
* **cli:** wire --http2/--no-http2 (HEADROOM_HTTP2) into proxy command ([#1373](https://github.com/JerrettDavis/headroom/issues/1373)) ([e06b616](https://github.com/JerrettDavis/headroom/commit/e06b61671f5cc23832e7d67bce7944e3601a0732))
* **cli:** wire --rpm/--tpm and HEADROOM_RPM/HEADROOM_TPM to the Click proxy command ([#1375](https://github.com/JerrettDavis/headroom/issues/1375)) ([8aab8f2](https://github.com/JerrettDavis/headroom/commit/8aab8f22cbd11061484991262d3fee3268e95bfa))
* **code-compressor:** CJK-aware relevance-query symbol matching ([#1747](https://github.com/JerrettDavis/headroom/issues/1747)) ([b38315c](https://github.com/JerrettDavis/headroom/commit/b38315cf72e4248cc76cc0e0d10dfa24a4a332e0))
* **code:** parse-probe tree-sitter availability in code_handler ([#1231](https://github.com/JerrettDavis/headroom/issues/1231)) ([#1300](https://github.com/JerrettDavis/headroom/issues/1300)) ([1de35e7](https://github.com/JerrettDavis/headroom/commit/1de35e775f2f19e51258fecddcabc2b88775f7d8))
* **code:** slice tree-sitter byte offsets as UTF-8 ([#1332](https://github.com/JerrettDavis/headroom/issues/1332)) ([8238402](https://github.com/JerrettDavis/headroom/commit/82384022bd38304a37e7eade4b5fc98d42f747a8))
* **code:** validate Python compressed syntax ([#1302](https://github.com/JerrettDavis/headroom/issues/1302)) ([cbd361d](https://github.com/JerrettDavis/headroom/commit/cbd361de2af266b6d72e246185f622c48ec5a6dc))
* **code:** verify a real parse in tree-sitter availability check ([#1231](https://github.com/JerrettDavis/headroom/issues/1231)) ([#1299](https://github.com/JerrettDavis/headroom/issues/1299)) ([5e0bb69](https://github.com/JerrettDavis/headroom/commit/5e0bb697254b7ec87e3191fa73031bde9321a79c))
* **codex:** avoid duplicate headroom provider config ([#1431](https://github.com/JerrettDavis/headroom/issues/1431)) ([ddd4adf](https://github.com/JerrettDavis/headroom/commit/ddd4adf911ee2d7a5323657a771ea0162b5590c4))
* **codex:** discover updated Codex state stores ([#1889](https://github.com/JerrettDavis/headroom/issues/1889)) ([9d42eba](https://github.com/JerrettDavis/headroom/commit/9d42ebaa1ab6e22e7b1398a3c0618d0d35895f40))
* **codex:** OpenCode Zen telemetry attribution ([#1648](https://github.com/JerrettDavis/headroom/issues/1648)) ([f18c6bd](https://github.com/JerrettDavis/headroom/commit/f18c6bd896f7b5a153e3b29f7a27b64c65b08fc5))
* **codex:** rerun memory lookup on every response.create WS frame ([#2113](https://github.com/JerrettDavis/headroom/issues/2113)) ([38479fc](https://github.com/JerrettDavis/headroom/commit/38479fcda1d4be717b049f5bc9932bdb85e69015))
* **codex:** retag thread providers so history menu stays whole across the proxy boundary ([#1034](https://github.com/JerrettDavis/headroom/issues/1034)) ([74ae781](https://github.com/JerrettDavis/headroom/commit/74ae7816444ae972b55f3da0ff5e28c8638ab4f3))
* **codex:** retag threads on init so Codex Desktop history stays visible ([#961](https://github.com/JerrettDavis/headroom/issues/961)) ([#1349](https://github.com/JerrettDavis/headroom/issues/1349)) ([e6bbc40](https://github.com/JerrettDavis/headroom/commit/e6bbc40b115bc3b31d68da4dabe280d38e1b691c))
* **codex:** skip sockets in session home overlay ([#2104](https://github.com/JerrettDavis/headroom/issues/2104)) ([c4ddcb9](https://github.com/JerrettDavis/headroom/commit/c4ddcb93a7ffe55112e41d8d45932fda1255ba1d))
* **codex:** stop pinning Codex memory MCP to one project db ([#1269](https://github.com/JerrettDavis/headroom/issues/1269)) ([ad7993b](https://github.com/JerrettDavis/headroom/commit/ad7993bf15e590a7d164407264721ce1b5128b1e))
* **compression:** reject lossy unmarked tool output in unit router path ([#1479](https://github.com/JerrettDavis/headroom/issues/1479)) ([de24cd5](https://github.com/JerrettDavis/headroom/commit/de24cd5fc0b894037c0481b5394e6851e87b3993))
* **content-detector:** detect and compress space-separated JSON objects ([#1742](https://github.com/JerrettDavis/headroom/issues/1742)) ([5194bdc](https://github.com/JerrettDavis/headroom/commit/5194bdc5a6e53d331ce0303aba670e8814bb5fd2))
* **content-router:** honor target_ratio in compression cache + add proxy --target-ratio flag ([#1108](https://github.com/JerrettDavis/headroom/issues/1108)) ([8894ee0](https://github.com/JerrettDavis/headroom/commit/8894ee0c18e6dfe858cf0034ec424fd0768a1334))
* **content-router:** protect_tool_results must not be weakened by profile-derived read_protection_window ([#2105](https://github.com/JerrettDavis/headroom/issues/2105)) ([3d0e59e](https://github.com/JerrettDavis/headroom/commit/3d0e59e518380972b56f017cffba96522141d53d))
* **content-router:** token-measure lossless folds at the acceptance gate ([#1772](https://github.com/JerrettDavis/headroom/issues/1772)) ([c5493ea](https://github.com/JerrettDavis/headroom/commit/c5493ea93bae798d489a82167c1f7bcff79eaecb))
* **copilot-auth:** stop discarding the caller's valid Copilot auth token ([#1879](https://github.com/JerrettDavis/headroom/issues/1879)) ([f52ca19](https://github.com/JerrettDavis/headroom/commit/f52ca19db1c28a87e9c3f6ff4ad21d6a16d7aa08))
* **copilot:** normalize subscription routing host ([#1836](https://github.com/JerrettDavis/headroom/issues/1836)) ([afd9cbd](https://github.com/JerrettDavis/headroom/commit/afd9cbdfafba0d31bd376a4a43dbcd41b30ec909))
* **copilot:** route mixed-model requests per model ([#1785](https://github.com/JerrettDavis/headroom/issues/1785)) ([5af5e22](https://github.com/JerrettDavis/headroom/commit/5af5e22862a0ce0a3d934c2f1e76ea7c1fad71e7))
* **cortex-code:** migrate to current Cortex REST API endpoints + add e2e benchmarks ([#1474](https://github.com/JerrettDavis/headroom/issues/1474)) ([f00ace6](https://github.com/JerrettDavis/headroom/commit/f00ace6da57aec2f68b833f42603ba3fda0f9110))
* **dashboard:** align token savings headline denominator ([#1653](https://github.com/JerrettDavis/headroom/issues/1653)) ([646e705](https://github.com/JerrettDavis/headroom/commit/646e7055143638ac4a2bc9980649fd046cea7840))
* **dashboard:** deduplicate repeated savings metrics ([#1804](https://github.com/JerrettDavis/headroom/issues/1804)) ([88f935a](https://github.com/JerrettDavis/headroom/commit/88f935a1eb52ec81cdd60db44627279d411b74ab))
* **dashboard:** derive per-project setup URL from live origin ([#1511](https://github.com/JerrettDavis/headroom/issues/1511)) ([e035aef](https://github.com/JerrettDavis/headroom/commit/e035aefce23fd2e20afccf2659c1db613b05d8ca))
* **dashboard:** distinguish unavailable RTK from zero stats in Docker ([#1900](https://github.com/JerrettDavis/headroom/issues/1900)) ([87f6e93](https://github.com/JerrettDavis/headroom/commit/87f6e93c14a9365695142084bc6966d7de70f437))
* **dashboard:** distinguish unavailable RTK from zero stats in Docker ([#1901](https://github.com/JerrettDavis/headroom/issues/1901)) ([361adcd](https://github.com/JerrettDavis/headroom/commit/361adcd1a00bbdcb949a3efc7b685937d4e84547))
* **dashboard:** include RTK stats in the historical tab ([#1324](https://github.com/JerrettDavis/headroom/issues/1324)) ([35939c3](https://github.com/JerrettDavis/headroom/commit/35939c3536cbaf6e1df01d099943e90ddb364b06))
* **dashboard:** light-mode backgrounds + aligned savings tables ([#1064](https://github.com/JerrettDavis/headroom/issues/1064)) ([5eae32b](https://github.com/JerrettDavis/headroom/commit/5eae32ba47fd2e6479cbc1cef1ef4f2fb992fe15))
* **dashboard:** price proxy savings without litellm ([#1728](https://github.com/JerrettDavis/headroom/issues/1728)) ([188e382](https://github.com/JerrettDavis/headroom/commit/188e382b44d09d7f16717377f908869292aab4d9))
* **dashboard:** serve per-request metadata to trusted-gateway peers ([#1766](https://github.com/JerrettDavis/headroom/issues/1766)) ([560319c](https://github.com/JerrettDavis/headroom/commit/560319cef4e38d6b79c3d0302e392dd736213572))
* **dedup:** shorter fold pointer + MIN_LINES=3, dedup Anthropic list-… ([#1932](https://github.com/JerrettDavis/headroom/issues/1932)) ([10e4829](https://github.com/JerrettDavis/headroom/commit/10e4829201b4a66d6e7e8f2f2e8a9d80346de850))
* **deps:** bump pillow to 12.3.0 and click to 8.4.2 ([#2097](https://github.com/JerrettDavis/headroom/issues/2097)) ([8870b69](https://github.com/JerrettDavis/headroom/commit/8870b6971f924eb1838694c3aec1949ca641e6af))
* **deps:** make litellm optional on Python 3.14 ([#956](https://github.com/JerrettDavis/headroom/issues/956)) ([#993](https://github.com/JerrettDavis/headroom/issues/993)) ([b2f04e4](https://github.com/JerrettDavis/headroom/commit/b2f04e4ef714fb6f2776ed95ee9157c34333e6c3))
* **deps:** remediate dependency CVEs and publish SBOM ([#1509](https://github.com/JerrettDavis/headroom/issues/1509)) ([5771a80](https://github.com/JerrettDavis/headroom/commit/5771a8020e2666503d87f1298070b44e35aad655))
* detect and clear stale ANTHROPIC_BASE_URL from crashed wrap sessions ([#1768](https://github.com/JerrettDavis/headroom/issues/1768)) ([#1837](https://github.com/JerrettDavis/headroom/issues/1837)) ([84509a4](https://github.com/JerrettDavis/headroom/commit/84509a4b892cc256331106c807a3a56107f1eec2))
* **detection:** contain unidiff panic on orphaned +++ target line ([#1548](https://github.com/JerrettDavis/headroom/issues/1548)) ([e386c09](https://github.com/JerrettDavis/headroom/commit/e386c097d6d507aa311ca3a22725b226e9d7b223))
* **docker:** persist headroom workspace in compose ([#1839](https://github.com/JerrettDavis/headroom/issues/1839)) ([5e29c06](https://github.com/JerrettDavis/headroom/commit/5e29c06aaf5e3d7d9e591914dc656f24eb72cc07))
* **docker:** persist session history across container revisions ([#1118](https://github.com/JerrettDavis/headroom/issues/1118)) ([5912d65](https://github.com/JerrettDavis/headroom/commit/5912d65674c708b00cff9a8cbc3b529fd2ab69fa))
* **docker:** report source build version ([#1862](https://github.com/JerrettDavis/headroom/issues/1862)) ([3807488](https://github.com/JerrettDavis/headroom/commit/38074888ac871b8b44418066d66b6a37159978ed))
* **e2e:** align Codex wrap e2e with global-only RTK guidance ([#1240](https://github.com/JerrettDavis/headroom/issues/1240)) ([#1254](https://github.com/JerrettDavis/headroom/issues/1254)) ([bc12ace](https://github.com/JerrettDavis/headroom/commit/bc12acef5998f264f22ca6d36b17337791a62e6f))
* emit unknown Anthropic content blocks verbatim in buffered-to-SSE conversion ([#1825](https://github.com/JerrettDavis/headroom/issues/1825)) ([d05802b](https://github.com/JerrettDavis/headroom/commit/d05802b6200b94f198e99319e6c778e78b53db8b))
* **evals:** CJK-aware F1 tokenization + token estimation ([#1527](https://github.com/JerrettDavis/headroom/issues/1527)) ([99a8540](https://github.com/JerrettDavis/headroom/commit/99a8540e657445df3204f1d15e213262f4289a42))
* **evals:** default unparseable judge scores below pass threshold ([#1892](https://github.com/JerrettDavis/headroom/issues/1892)) ([42ebbc6](https://github.com/JerrettDavis/headroom/commit/42ebbc6cce02a0fd5e0a6e614348d47f4099649a))
* **gemini:** offload compression to the executor ([#1382](https://github.com/JerrettDavis/headroom/issues/1382)) ([615848e](https://github.com/JerrettDavis/headroom/commit/615848eba408997c1850319028815afadc6c49ed))
* **gemini:** resolve Google model capabilities through ModelRegistry ([#1276](https://github.com/JerrettDavis/headroom/issues/1276)) ([17ecad9](https://github.com/JerrettDavis/headroom/commit/17ecad9d89b81313f131d569cfed532f9d42e82a))
* harden persistent install startup ([#1851](https://github.com/JerrettDavis/headroom/issues/1851)) ([1d2b76e](https://github.com/JerrettDavis/headroom/commit/1d2b76e72e16eaf532326d9e75a481e18bde1ab7))
* **health:** exclude kompress from aggregate readiness + adversarial PBT ([#2066](https://github.com/JerrettDavis/headroom/issues/2066)) ([f1663ea](https://github.com/JerrettDavis/headroom/commit/f1663ea55700cd7a1ca4400ed27c200d27317d97))
* **init:** set ENABLE_TOOL_SEARCH=true so Claude Code keeps deferring tools ([#746](https://github.com/JerrettDavis/headroom/issues/746)) ([#995](https://github.com/JerrettDavis/headroom/issues/995)) ([500ec2b](https://github.com/JerrettDavis/headroom/commit/500ec2b7faebfd24c9ea404ae1dece40b3b14b84))
* **install:** add orjson to [proxy] extra for LiteLLM provider backends ([#2074](https://github.com/JerrettDavis/headroom/issues/2074)) ([4f3d5ab](https://github.com/JerrettDavis/headroom/commit/4f3d5ab341466297cb1b37104d93b338c4947c82))
* **install:** close parent log fd in start_detached_agent ([#1576](https://github.com/JerrettDavis/headroom/issues/1576)) ([816cb85](https://github.com/JerrettDavis/headroom/commit/816cb85fa8ee8d349fe673e7affd9a54acb1207d))
* **install:** default docker image to headroomlabs-ai GHCR registry ([#1867](https://github.com/JerrettDavis/headroom/issues/1867)) ([#2039](https://github.com/JerrettDavis/headroom/issues/2039)) ([c3db8e4](https://github.com/JerrettDavis/headroom/commit/c3db8e47f8dd85caf18a8748068c627cba6ecfeb))
* **install:** don't let host env override the manifest in persistent-docker ([#2090](https://github.com/JerrettDavis/headroom/issues/2090)) ([b097ef3](https://github.com/JerrettDavis/headroom/commit/b097ef3e25029899c28bf930c8c7fe46ab384d52))
* **install:** guard install_agent_ensure against duplicate runtime spawns ([#1301](https://github.com/JerrettDavis/headroom/issues/1301)) ([8da0b4e](https://github.com/JerrettDavis/headroom/commit/8da0b4e565be2d5f798741bb9b7bee70c2102c8c))
* **install:** pass sc.exe create as raw command line so binPath= quoting survives ([#1654](https://github.com/JerrettDavis/headroom/issues/1654)) ([#1702](https://github.com/JerrettDavis/headroom/issues/1702)) ([d6e0710](https://github.com/JerrettDavis/headroom/commit/d6e07102283745a44aece2222f84c1599eabf90a))
* **install:** persist --no-http2 override through install apply ([#1676](https://github.com/JerrettDavis/headroom/issues/1676)) ([6fb5f3b](https://github.com/JerrettDavis/headroom/commit/6fb5f3bc3dfa60e56744f85cf049524d43104a31))
* **install:** repair macOS launchd restart/start lifecycle ([#1290](https://github.com/JerrettDavis/headroom/issues/1290)) ([da1a397](https://github.com/JerrettDavis/headroom/commit/da1a3973ed79d89617087ec315e77fb82356c03b))
* **install:** stop duplicating ENTRYPOINT in persistent-docker runtime command ([#833](https://github.com/JerrettDavis/headroom/issues/833)) ([#1348](https://github.com/JerrettDavis/headroom/issues/1348)) ([feedead](https://github.com/JerrettDavis/headroom/commit/feedead07772a27b872a448281a2d17e539d4702))
* **install:** use Windows-safe PID liveness probe in runtime_status ([#1544](https://github.com/JerrettDavis/headroom/issues/1544)) ([#1560](https://github.com/JerrettDavis/headroom/issues/1560)) ([6b227b9](https://github.com/JerrettDavis/headroom/commit/6b227b9c906d708923f39c0d877989a49942adae))
* **install:** write deployment manifest atomically and tolerate corrupt manifests ([#1303](https://github.com/JerrettDavis/headroom/issues/1303)) ([42bdf23](https://github.com/JerrettDavis/headroom/commit/42bdf23d241388cd1a2567f8481b51dacdd8ec83))
* **io:** use UTF-8 with locale fallback and preserve line endings on config/text I/O ([#1498](https://github.com/JerrettDavis/headroom/issues/1498)) ([1baa04e](https://github.com/JerrettDavis/headroom/commit/1baa04ef6576e08eeed685890354fca16ad4e6e3))
* **kompress:** hard override keeps must-keep tokens regardless of model score ([#1400](https://github.com/JerrettDavis/headroom/issues/1400)) ([42612c8](https://github.com/JerrettDavis/headroom/commit/42612c86dfc25a56a6ec6c1da74914e0741a51f6))
* **kompress:** never block the request path on the cold-cache model download ([#1161](https://github.com/JerrettDavis/headroom/issues/1161)) ([3fc2a78](https://github.com/JerrettDavis/headroom/commit/3fc2a78a5e20f159f7c5f198de6b91788dc64287))
* **kompress:** surface model-not-ready state via logs and health endpoint ([#2034](https://github.com/JerrettDavis/headroom/issues/2034)) ([12aa2cb](https://github.com/JerrettDavis/headroom/commit/12aa2cbf6cc888cce1cb6a47cbea02f37c26fe4d))
* **langchain:** disable streaming on wrapped model during ainvoke() ([#1287](https://github.com/JerrettDavis/headroom/issues/1287)) ([3590046](https://github.com/JerrettDavis/headroom/commit/359004646bb2cda2b99cf3ef154539b7fa81aa72))
* **learn:** aggregate verbosity baselines across projects instead of overwriting ([#1288](https://github.com/JerrettDavis/headroom/issues/1288)) ([27a5468](https://github.com/JerrettDavis/headroom/commit/27a546834960b349e710a0b2e86ca3471523f34d))
* **learn:** don't shadow TIMEOUT/CONNECTION with the generic RUNTIME_ERROR ([#2099](https://github.com/JerrettDavis/headroom/issues/2099)) ([c7b5a24](https://github.com/JerrettDavis/headroom/commit/c7b5a24b4ffca78bacf1919ace00087b6ab6f7d0))
* **learn:** handle Windows UTF-8, drive-letter paths, and CLI shim fallback ([#1895](https://github.com/JerrettDavis/headroom/issues/1895)) ([e3b45e4](https://github.com/JerrettDavis/headroom/commit/e3b45e402bc019c4de5f15c458c59696a68a8aff))
* **learn:** parse fenced JSON even with a prose preamble ([#1988](https://github.com/JerrettDavis/headroom/issues/1988)) ([d2170b1](https://github.com/JerrettDavis/headroom/commit/d2170b1922d63e586f672e0b86bcb8b7c9de0283))
* **litellm:** surface Bedrock cache token usage in non-streaming responses ([#1848](https://github.com/JerrettDavis/headroom/issues/1848)) ([d604e86](https://github.com/JerrettDavis/headroom/commit/d604e86904525d67d618e7920cd7ce0dac6e903d))
* **mcp/codex:** don't clobber an unparseable/non-table config.toml ([#2062](https://github.com/JerrettDavis/headroom/issues/2062)) ([415e03c](https://github.com/JerrettDavis/headroom/commit/415e03c1688a5a1a919872335257df1b6b424c42))
* **mcp/opencode:** don't clobber an unparseable opencode.json on register ([#1661](https://github.com/JerrettDavis/headroom/issues/1661)) ([d079614](https://github.com/JerrettDavis/headroom/commit/d079614b1fe828c6b8d42ba77236e6ff8df40076))
* **mcp:** correct default Claude Code config path in ClaudeRegistrar ([#1859](https://github.com/JerrettDavis/headroom/issues/1859)) ([c85731d](https://github.com/JerrettDavis/headroom/commit/c85731dc23832971993653c2fae7ae65e1db27ac))
* **mcp:** isolate ClaudeRegistrar CLI config env ([#1888](https://github.com/JerrettDavis/headroom/issues/1888)) ([1c947b1](https://github.com/JerrettDavis/headroom/commit/1c947b1103fa66563a01ea638f1669ee053018e6))
* **mcp:** register managed installs with a resolvable headroom command ([#1386](https://github.com/JerrettDavis/headroom/issues/1386)) ([22def93](https://github.com/JerrettDavis/headroom/commit/22def931770e6138d16f62daec39501951e68e64))
* **mcp:** report correct savings_percent in headroom_compress ([#1106](https://github.com/JerrettDavis/headroom/issues/1106)) ([f216e43](https://github.com/JerrettDavis/headroom/commit/f216e430559759f51b53eb44e76e030e6a83c80a))
* **mcp:** show lifetime totals and label rolling session scope in headroom_stats ([#1428](https://github.com/JerrettDavis/headroom/issues/1428)) ([1c0e152](https://github.com/JerrettDavis/headroom/commit/1c0e15243eda8f2dc868fe9ed4a08d944893686b))
* **mcp:** surface dead proxy state ([#1786](https://github.com/JerrettDavis/headroom/issues/1786)) ([931eed8](https://github.com/JerrettDavis/headroom/commit/931eed879d26512b2dbdf3ea4246e4f7b2c97a70))
* **memory/sqlite:** don't emit OFFSET without LIMIT in query ([#2063](https://github.com/JerrettDavis/headroom/issues/2063)) ([a5bdc54](https://github.com/JerrettDavis/headroom/commit/a5bdc5491f91e9486af9c25d2039e848b63cc98c))
* **memory/sync:** don't clobber memories sharing a first line ([#1976](https://github.com/JerrettDavis/headroom/issues/1976)) ([5e14b8c](https://github.com/JerrettDavis/headroom/commit/5e14b8c0f293df78f2576a9fc7eb90189e604cb5))
* **memory/sync:** make Codex AGENTS.md adapter additive (stop wiping memories) ([#1674](https://github.com/JerrettDavis/headroom/issues/1674)) ([7fd0c42](https://github.com/JerrettDavis/headroom/commit/7fd0c42ced9ecdf2a5411ff85d554b9e39ceb0b6))
* **memory:** annotate _EMBEDDER_CACHE key as 3-tuple (unbreak main lint) ([#2153](https://github.com/JerrettDavis/headroom/issues/2153)) ([22af75a](https://github.com/JerrettDavis/headroom/commit/22af75adaee59ab402a9c4579dfa59e222682831))
* **memory:** cap local embedder CPU thread oversubscription ([#198](https://github.com/JerrettDavis/headroom/issues/198)) ([#1559](https://github.com/JerrettDavis/headroom/issues/1559)) ([b84afbf](https://github.com/JerrettDavis/headroom/commit/b84afbfb833999ddf164d324971bf6c11014a9d3))
* **memory:** honor explicit store=false on Responses requests ([#2017](https://github.com/JerrettDavis/headroom/issues/2017)) ([31abb69](https://github.com/JerrettDavis/headroom/commit/31abb696dd8f7452ae5ec66e0d55eaf28e59a62c))
* **memory:** key the embedder cache on ollama_base_url ([#2109](https://github.com/JerrettDavis/headroom/issues/2109)) ([1725cd1](https://github.com/JerrettDavis/headroom/commit/1725cd1f8376828849c0e48fd9008afa25d24fff))
* **memory:** resolve Trae cwd metadata from user reminders ([#1737](https://github.com/JerrettDavis/headroom/issues/1737)) ([#1887](https://github.com/JerrettDavis/headroom/issues/1887)) ([3e85eb1](https://github.com/JerrettDavis/headroom/commit/3e85eb1880af5663cf083492f1dc1415a354bd99))
* **memory:** singleflight LocalBackend init to stop cold-start races ([#1691](https://github.com/JerrettDavis/headroom/issues/1691)) ([bec47a1](https://github.com/JerrettDavis/headroom/commit/bec47a1898883919ad8c5ea41e3a7443a6890e7f))
* **memory:** track MCP retrieval access ([#2065](https://github.com/JerrettDavis/headroom/issues/2065)) ([d0ecc9a](https://github.com/JerrettDavis/headroom/commit/d0ecc9a556047fa24a43609dd02b5b1e2b5d7b03))
* **memory:** use ONNX embedder for `wrap --memory` sync ([#1092](https://github.com/JerrettDavis/headroom/issues/1092)) ([#1262](https://github.com/JerrettDavis/headroom/issues/1262)) ([4f9feda](https://github.com/JerrettDavis/headroom/commit/4f9fedaa7a02e41114b5d5f4606f95f903e17b2a))
* **models:** version-boundary longest-prefix match in ModelRegistry.get ([#1658](https://github.com/JerrettDavis/headroom/issues/1658)) ([b699bed](https://github.com/JerrettDavis/headroom/commit/b699bedf95286138b1dda444ebc5f86bf7041f5a))
* **openclaw:** detect uv-installed headroom binary in ~/.local/bin ([#1459](https://github.com/JerrettDavis/headroom/issues/1459)) ([adaeb88](https://github.com/JerrettDavis/headroom/commit/adaeb88a4d5512da5bd0bf58c1e3a276a5269d44))
* **openclaw:** wrap plugin export as {register} object for OpenClaw 2026.x compatibility ([#1218](https://github.com/JerrettDavis/headroom/issues/1218)) ([2e6c442](https://github.com/JerrettDavis/headroom/commit/2e6c442dc87f0853313b18ab1a7c80e991058bf7))
* **opencode:** preserve custom OpenAI gateway paths ([#1596](https://github.com/JerrettDavis/headroom/issues/1596)) ([c19347c](https://github.com/JerrettDavis/headroom/commit/c19347c31046bf25baf9b1a816c9bede5d3ee807))
* **opencode:** route native providers + load transport plugin, fix Serena context ([#1573](https://github.com/JerrettDavis/headroom/issues/1573)) ([ad0034f](https://github.com/JerrettDavis/headroom/commit/ad0034f98191501c1a60d26383bc3ed9f6d532be))
* **opencode:** use local MCP config ([#1383](https://github.com/JerrettDavis/headroom/issues/1383)) ([4bd3ddf](https://github.com/JerrettDavis/headroom/commit/4bd3ddfaa5c5655540494b96e4f5d47724460c7d))
* **opencode:** write local MCP config ([#1381](https://github.com/JerrettDavis/headroom/issues/1381)) ([6c83790](https://github.com/JerrettDavis/headroom/commit/6c837906802f9c211513a182de2365071e4f7765))
* **packaging:** guard torch extras on intel macos ([#2011](https://github.com/JerrettDavis/headroom/issues/2011)) ([fd0d29c](https://github.com/JerrettDavis/headroom/commit/fd0d29c92dbf3629c87bd0aaa8960f55573fae92))
* **packaging:** move hnswlib to optional [vector] extra so [all] needs no C++ toolchain ([#1499](https://github.com/JerrettDavis/headroom/issues/1499)) ([80fa086](https://github.com/JerrettDavis/headroom/commit/80fa086660b277798ba9e6c6ed8645ec029362da))
* patch nltk vulnerability (CVE-2026-54293) ([#1929](https://github.com/JerrettDavis/headroom/issues/1929)) ([28ca61f](https://github.com/JerrettDavis/headroom/commit/28ca61fc9d3e36d5f967da6b3f75d6bccfeb0306))
* patch rtk hook script to use absolute path after register_claude_hooks ([#571](https://github.com/JerrettDavis/headroom/issues/571)) ([b618d2d](https://github.com/JerrettDavis/headroom/commit/b618d2d11a25ffaa00729b17fb41bd41037f4090))
* **perf:** surface RTK/CLI context-tool savings in perf and the session card ([#1433](https://github.com/JerrettDavis/headroom/issues/1433)) ([9362747](https://github.com/JerrettDavis/headroom/commit/93627471b72e3200e3ca78e1fb345c174414b716))
* preserve anthropic passthrough tool order ([#1427](https://github.com/JerrettDavis/headroom/issues/1427)) ([a932247](https://github.com/JerrettDavis/headroom/commit/a9322477e33ec2c5ccd6442d3f72c17b7388c9e0))
* **pricing:** alias retired claude-3-sonnet to Sonnet-tier price, not Haiku ([#2095](https://github.com/JerrettDavis/headroom/issues/2095)) ([6137967](https://github.com/JerrettDavis/headroom/commit/6137967083936467c570e8c7f20e94f43ccc13aa))
* **providers:** update DeepSeek V3 context limit from 128K to 1M ([#1038](https://github.com/JerrettDavis/headroom/issues/1038)) ([#1137](https://github.com/JerrettDavis/headroom/issues/1137)) ([bcabc5c](https://github.com/JerrettDavis/headroom/commit/bcabc5cb11c7c411ed29dac1fcc3771833ac8524))
* **proxy/anthropic:** preserve non-2xx upstream status through security scan ([#2100](https://github.com/JerrettDavis/headroom/issues/2100)) ([aa78816](https://github.com/JerrettDavis/headroom/commit/aa788164fd5ad55ed427b25fb2e87e692921d824))
* **proxy/anthropic:** scope session id by top-level system prompt ([#2070](https://github.com/JerrettDavis/headroom/issues/2070)) ([ec6e60e](https://github.com/JerrettDavis/headroom/commit/ec6e60ea3ef2e00b245a6dd92dbe34cb145f4d33))
* **proxy/auth:** match real Anthropic OAuth token prefix (sk-ant-oat) ([#1672](https://github.com/JerrettDavis/headroom/issues/1672)) ([8cddf9b](https://github.com/JerrettDavis/headroom/commit/8cddf9b58ea9ed11a0cd3532be6e779dffe57b55))
* **proxy/cost:** price cache savings by most-used model, not first-seen ([#2023](https://github.com/JerrettDavis/headroom/issues/2023)) ([b4f807f](https://github.com/JerrettDavis/headroom/commit/b4f807f21a5be39c690b8b5e8e236116a32dd6b6))
* **proxy/gemini:** preserve non-text content across the compression round-trip ([#2079](https://github.com/JerrettDavis/headroom/issues/2079)) ([4056117](https://github.com/JerrettDavis/headroom/commit/4056117d90468619dbc2684448ebd76c48c41921))
* **proxy/gemini:** thread savings-profile kwargs into apply() ([#1994](https://github.com/JerrettDavis/headroom/issues/1994)) ([38306a3](https://github.com/JerrettDavis/headroom/commit/38306a331c1e25d688db0219920915904d5e22f3))
* **proxy/memory:** capture user text blocks for the retrieval query ([#2064](https://github.com/JerrettDavis/headroom/issues/2064)) ([f542b70](https://github.com/JerrettDavis/headroom/commit/f542b70413677260a6ebd5801df64337f66525eb))
* **proxy/openai:** respect explicit stream_options.include_usage ([#2026](https://github.com/JerrettDavis/headroom/issues/2026)) ([19201e8](https://github.com/JerrettDavis/headroom/commit/19201e842f30af2b842458b5a8f5891d9b631b29))
* **proxy/openai:** thread savings-profile kwargs into chat completions ([#1606](https://github.com/JerrettDavis/headroom/issues/1606)) ([7ff842d](https://github.com/JerrettDavis/headroom/commit/7ff842da170b5bceb5d67048473eeb8a18e09a51))
* **proxy/openai:** translate max_tokens -&gt; max_completion_tokens on chat path ([#1774](https://github.com/JerrettDavis/headroom/issues/1774)) ([285808b](https://github.com/JerrettDavis/headroom/commit/285808b90ea5532fe319c94d408699cb46b2e5f8))
* **proxy/savings:** don't bill fallback rate for free (0-priced) models ([#2024](https://github.com/JerrettDavis/headroom/issues/2024)) ([6ecbdd6](https://github.com/JerrettDavis/headroom/commit/6ecbdd6b527312ce61a1e8890f0d6d1d63ae4f48))
* **proxy/vertex:** route google-publisher requests to the request region ([#2069](https://github.com/JerrettDavis/headroom/issues/2069)) ([1843346](https://github.com/JerrettDavis/headroom/commit/1843346283b1d6a8d5c932fdc3e16502ced416e8))
* **proxy:** add --protect-tool-results to prevent lossy compression of exact-output Bash results ([#1374](https://github.com/JerrettDavis/headroom/issues/1374)) ([51d4bcf](https://github.com/JerrettDavis/headroom/commit/51d4bcfc113d95a9c843937fbdd3751483bc1dab))
* **proxy:** add an Anthropic buffered read-timeout override ([#1331](https://github.com/JerrettDavis/headroom/issues/1331)) ([3be2526](https://github.com/JerrettDavis/headroom/commit/3be2526b76caa8ff1050e44807386874571e079b))
* **proxy:** add versionless Vertex AI routes for Claude Code compatibility ([#1321](https://github.com/JerrettDavis/headroom/issues/1321)) ([bb3e040](https://github.com/JerrettDavis/headroom/commit/bb3e040a463b66801323c261e9547f1e4a2ccfbd))
* **proxy:** aggregate tool-output size floor so Codex sessions compress ([#2050](https://github.com/JerrettDavis/headroom/issues/2050)) ([#2116](https://github.com/JerrettDavis/headroom/issues/2116)) ([dbe2558](https://github.com/JerrettDavis/headroom/commit/dbe2558c18552b94bb4b224c003f772e6ef7b2f0))
* **proxy:** allow disabling periodic TOIN stats logging ([#1265](https://github.com/JerrettDavis/headroom/issues/1265)) ([b5f63d8](https://github.com/JerrettDavis/headroom/commit/b5f63d8fa9f81f39eab854f29a2fdc39878566df))
* **proxy:** bind before eager preload so a hung compressor load can't block startup ([#1500](https://github.com/JerrettDavis/headroom/issues/1500)) ([d5ac07f](https://github.com/JerrettDavis/headroom/commit/d5ac07fc451516c3b1fe7ece2f01f8d85c126925))
* **proxy:** bound Codex WS compression fallback latency ([#1802](https://github.com/JerrettDavis/headroom/issues/1802)) ([d24a3f8](https://github.com/JerrettDavis/headroom/commit/d24a3f842551d36c14dc0ec146a9302e256c5c0f))
* **proxy:** bound HF tokenizer load and offload token counting off event loop ([#1738](https://github.com/JerrettDavis/headroom/issues/1738)) ([46d5d68](https://github.com/JerrettDavis/headroom/commit/46d5d685d9bcdced1f77ffdc0f2d3a8ee8a1f319))
* **proxy:** build SSL contexts for custom CA bundles ([#1134](https://github.com/JerrettDavis/headroom/issues/1134)) ([561ba17](https://github.com/JerrettDavis/headroom/commit/561ba17ec2e05b463682fd3ecfe7ca43b558684f))
* **proxy:** cache_savings_usd silently zeroes when litellm is unavailable ([#2005](https://github.com/JerrettDavis/headroom/issues/2005)) ([75d7861](https://github.com/JerrettDavis/headroom/commit/75d786117a15b55ca31daa2b2f07b123994d74d1))
* **proxy:** cancel retry backoff on shutdown ([#1834](https://github.com/JerrettDavis/headroom/issues/1834)) ([da2d8dc](https://github.com/JerrettDavis/headroom/commit/da2d8dc9dbf3edfcd1c3f6429db32374a6bebc64))
* **proxy:** compress Anthropic user text blocks when enabled ([#1875](https://github.com/JerrettDavis/headroom/issues/1875)) ([e36439a](https://github.com/JerrettDavis/headroom/commit/e36439a9411bf7fc93b4a5dceac50aa4570a6105))
* **proxy:** compress Hermes scoped coding-agent passthrough ([#1815](https://github.com/JerrettDavis/headroom/issues/1815)) ([09d1ef4](https://github.com/JerrettDavis/headroom/commit/09d1ef45be4bb8d1debed5c78610eacc7e518396))
* **proxy:** count exhausted upstream 5xx as failed across all providers ([#1571](https://github.com/JerrettDavis/headroom/issues/1571)) ([e365ad7](https://github.com/JerrettDavis/headroom/commit/e365ad71524813d348fdc45f09535da8cbe7b53a))
* **proxy:** expose persistent savings metrics ([#1647](https://github.com/JerrettDavis/headroom/issues/1647)) ([5fe4e7b](https://github.com/JerrettDavis/headroom/commit/5fe4e7b19530da0c2d07d17f20b18d79b6fab367))
* **proxy:** fail open when kompress saturation would exhaust pre-upstream budget ([#1430](https://github.com/JerrettDavis/headroom/issues/1430)) ([15ac650](https://github.com/JerrettDavis/headroom/commit/15ac650d409ea7def9e54d9962af1cfdc1f11f5d))
* **proxy:** forward request-id headers on the streaming path ([#1100](https://github.com/JerrettDavis/headroom/issues/1100)) ([#1258](https://github.com/JerrettDavis/headroom/issues/1258)) ([3d59df7](https://github.com/JerrettDavis/headroom/commit/3d59df7be889d6d7218c5552e40a4f736d80a3af))
* **proxy:** freeze must forward cached (compressed) prefix byte-identical — stop token-mode cache busting ([#1850](https://github.com/JerrettDavis/headroom/issues/1850)) ([248ae0f](https://github.com/JerrettDavis/headroom/commit/248ae0f3e0d4d7ff2e23837e628880dcbda4411a))
* **proxy:** fsync savings dir after atomic rename ([#1764](https://github.com/JerrettDavis/headroom/issues/1764)) ([7de2c1e](https://github.com/JerrettDavis/headroom/commit/7de2c1e4c2ca8aefd73d3c419dfbcdd881a63bd2))
* **proxy:** gate CCR retrieve/compress endpoints to loopback ([#1338](https://github.com/JerrettDavis/headroom/issues/1338)) ([acafb2d](https://github.com/JerrettDavis/headroom/commit/acafb2d0f668dc5f5848fa2940545743899a30c2))
* **proxy:** handle ClientDisconnect in passthrough body reads ([#2033](https://github.com/JerrettDavis/headroom/issues/2033)) ([9db8a6b](https://github.com/JerrettDavis/headroom/commit/9db8a6bbf661026d92647cea395954a51a075e4f))
* **proxy:** handle ClientDisconnect in passthrough body reads + log sanitization ([#2067](https://github.com/JerrettDavis/headroom/issues/2067)) ([605e269](https://github.com/JerrettDavis/headroom/commit/605e269f9844099d639418ad1825fa97212b67a7))
* **proxy:** handle content-part outputs in Codex Responses compression ([#2052](https://github.com/JerrettDavis/headroom/issues/2052)) ([c9a7755](https://github.com/JerrettDavis/headroom/commit/c9a7755a281d04d4ec8e37fa62b0d66115c89e0c))
* **proxy:** handle streaming CCR retrieval ([#1451](https://github.com/JerrettDavis/headroom/issues/1451)) ([d337e3b](https://github.com/JerrettDavis/headroom/commit/d337e3b828ffc1f22cd5ca1884500b8905e9bd82))
* **proxy:** hoist ccr_workspace_key default so /v1/messages survives CCR-inject off ([#1096](https://github.com/JerrettDavis/headroom/issues/1096)) ([1deb947](https://github.com/JerrettDavis/headroom/commit/1deb947ac139f8525c0da2bbe86d6caacfc48a49))
* **proxy:** honor force_kompress routing profile ([#996](https://github.com/JerrettDavis/headroom/issues/996)) ([b4682d6](https://github.com/JerrettDavis/headroom/commit/b4682d6f91c782286553875b7fd8cee6101f1b0f))
* **proxy:** honor HEADROOM_EXCLUDE_TOOLS for Codex /v1/responses tool outputs ([#940](https://github.com/JerrettDavis/headroom/issues/940)) ([#1053](https://github.com/JerrettDavis/headroom/issues/1053)) ([f03e77b](https://github.com/JerrettDavis/headroom/commit/f03e77bec05494aebb4de188eddf2b57f99f6997))
* **proxy:** honor x-headroom-base-url on /v1/messages route ([#1763](https://github.com/JerrettDavis/headroom/issues/1763)) ([bb2acf7](https://github.com/JerrettDavis/headroom/commit/bb2acf700a62ee3a76ba181052e36904af4f11be))
* **proxy:** include system/tools/sampling in cache key ([#1473](https://github.com/JerrettDavis/headroom/issues/1473)) ([312129a](https://github.com/JerrettDavis/headroom/commit/312129a8e7465c97402ae45b9e9d51b7f4b5b0c7))
* **proxy:** keep cache_control bounded + stable so the freeze overlay stops busting ([#1852](https://github.com/JerrettDavis/headroom/issues/1852)) ([4820134](https://github.com/JerrettDavis/headroom/commit/48201345be16a8b5aad74e8c390850dce0f34ec4))
* **proxy:** keep Kompress warmup off the startup path ([#2001](https://github.com/JerrettDavis/headroom/issues/2001)) ([10ed14e](https://github.com/JerrettDavis/headroom/commit/10ed14e7f68e4ad4f4c720d083032cd47d25b96c))
* **proxy:** keep large compression results on the critical path ([#296](https://github.com/JerrettDavis/headroom/issues/296)) ([#1352](https://github.com/JerrettDavis/headroom/issues/1352)) ([90734b6](https://github.com/JerrettDavis/headroom/commit/90734b691a50669eaaae7c8739243e7bfc313326))
* **proxy:** keep OpenAI tool observations mutable in cache mode ([#1884](https://github.com/JerrettDavis/headroom/issues/1884)) ([55efb1c](https://github.com/JerrettDavis/headroom/commit/55efb1c77d5b67f7ad0620372c6256c8b0547591))
* **proxy:** keep PRE_SEND from reintroducing empty tool arrays ([#2015](https://github.com/JerrettDavis/headroom/issues/2015)) ([d1db00a](https://github.com/JerrettDavis/headroom/commit/d1db00ab8697505b56b294c611cf93463b71810b))
* **proxy:** keep recent stats request rows ([#1922](https://github.com/JerrettDavis/headroom/issues/1922)) ([bd8de9f](https://github.com/JerrettDavis/headroom/commit/bd8de9f3829e9f66e33d36a01464a99adcc42c1d))
* **proxy:** offload /v1/compress to the compression executor to stop blocking the loop ([#1501](https://github.com/JerrettDavis/headroom/issues/1501)) ([27e010e](https://github.com/JerrettDavis/headroom/commit/27e010e38f37e64767e94d144fd4353fcdbe1e47))
* **proxy:** only queue mid-turn messages for opt-in clients with explicit session header ([#1951](https://github.com/JerrettDavis/headroom/issues/1951)) ([c365c7f](https://github.com/JerrettDavis/headroom/commit/c365c7ff81dd79a7abc5012271f1c73dfe84fa4c))
* **proxy:** persist lifetime cache-read savings across restarts ([#1665](https://github.com/JerrettDavis/headroom/issues/1665)) ([908997e](https://github.com/JerrettDavis/headroom/commit/908997ef61d91a7e912637c785176719a5f1c719))
* **proxy:** preserve byte-faithful Anthropic tool forwarding ([#1222](https://github.com/JerrettDavis/headroom/issues/1222)) ([1f18d59](https://github.com/JerrettDavis/headroom/commit/1f18d5980972fc7b2091ca0be5318d06c4edfa79))
* **proxy:** preserve chatgpt responses streaming ([#2012](https://github.com/JerrettDavis/headroom/issues/2012)) ([a617455](https://github.com/JerrettDavis/headroom/commit/a617455f0242328862b43f4f4be9cc67a9c99e0a))
* **proxy:** preserve Responses memory continuations with store=false ([#1103](https://github.com/JerrettDavis/headroom/issues/1103)) ([cdfeeac](https://github.com/JerrettDavis/headroom/commit/cdfeeacc63e6cb98d34e245f2330f0e1af531d32))
* **proxy:** preserve Responses passthrough bytes ([#1598](https://github.com/JerrettDavis/headroom/issues/1598)) ([2a34a82](https://github.com/JerrettDavis/headroom/commit/2a34a822f2a39da57fbd07575752888f5515f51a))
* **proxy:** preserve streaming passthrough beta headers ([#1783](https://github.com/JerrettDavis/headroom/issues/1783)) ([0f553a8](https://github.com/JerrettDavis/headroom/commit/0f553a8ebbd6d790ca622f95f389b5e7d11a41ce))
* **proxy:** preserve terminal tool on Codex Responses ([#2000](https://github.com/JerrettDavis/headroom/issues/2000)) ([41af39d](https://github.com/JerrettDavis/headroom/commit/41af39d769ef132cf953ecd18803675c871fc4a6))
* **proxy:** preserve upstream 5xx status on retry exhaustion ([#1570](https://github.com/JerrettDavis/headroom/issues/1570)) ([7836aea](https://github.com/JerrettDavis/headroom/commit/7836aea2be8577b0f593c9cfcff1da5f5e0ee3c8))
* **proxy:** queue mid-turn user messages on non-Bedrock streaming path ([#1377](https://github.com/JerrettDavis/headroom/issues/1377)) ([b09f027](https://github.com/JerrettDavis/headroom/commit/b09f0270625a4dbee6fc2805f52f19492e68f1f6))
* **proxy:** record cache metrics for non-streaming backend paths ([#1271](https://github.com/JerrettDavis/headroom/issues/1271)) ([8580404](https://github.com/JerrettDavis/headroom/commit/85804043ff1f418148dd00c42a2dcdffe61a57a6))
* **proxy:** register interceptor in explicit transforms list when HEADROOM_INTERCEPT_ENABLED ([#1376](https://github.com/JerrettDavis/headroom/issues/1376)) ([55c700c](https://github.com/JerrettDavis/headroom/commit/55c700c686309c63eb8d9d7d21f30d1838e1c9e7))
* **proxy:** release _active_streams session lock on setup-phase errors ([#1864](https://github.com/JerrettDavis/headroom/issues/1864)) ([2ccd831](https://github.com/JerrettDavis/headroom/commit/2ccd831032e23248879bd38c5bde947d3a0a54f3))
* **proxy:** report real input tokens on streaming message_start ([#1132](https://github.com/JerrettDavis/headroom/issues/1132)) ([#1305](https://github.com/JerrettDavis/headroom/issues/1305)) ([70cc96a](https://github.com/JerrettDavis/headroom/commit/70cc96a386baff345669722dc15fde694811d2d6))
* **proxy:** retry HTTP/2 stream resets instead of 502ing ([#1645](https://github.com/JerrettDavis/headroom/issues/1645)) ([2ce19c2](https://github.com/JerrettDavis/headroom/commit/2ce19c2c55710cdc5f7a4bb88803f05e4b31feff))
* **proxy:** retry passthrough on transient upstream connection close ([#1513](https://github.com/JerrettDavis/headroom/issues/1513)) ([5d14080](https://github.com/JerrettDavis/headroom/commit/5d14080c948b04ccd997d2434b37604440701888))
* **proxy:** retry upstream 429 with Retry-After on both forwarders ([#1329](https://github.com/JerrettDavis/headroom/issues/1329)) ([90bee89](https://github.com/JerrettDavis/headroom/commit/90bee89243004846cfc86ad3bf888579acb27522))
* **proxy:** retry upstream 529 overloaded like 429 on both forwarders ([#1495](https://github.com/JerrettDavis/headroom/issues/1495)) ([547b15d](https://github.com/JerrettDavis/headroom/commit/547b15dab2c18b8d70504c366dc33e22111255e5))
* **proxy:** route Codex OAuth image requests ([#1215](https://github.com/JerrettDavis/headroom/issues/1215)) ([381d771](https://github.com/JerrettDavis/headroom/commit/381d771e4618585e5756e20c090354ccad09183f))
* **proxy:** route Foundry Anthropic messages ([#1878](https://github.com/JerrettDavis/headroom/issues/1878)) ([739f654](https://github.com/JerrettDavis/headroom/commit/739f654bbd71b3e31ade40ae9eadf812b362beec))
* **proxy:** scope CORS to loopback + gate operator/content endpoints ([#1226](https://github.com/JerrettDavis/headroom/issues/1226)) ([bd55a42](https://github.com/JerrettDavis/headroom/commit/bd55a426bc3ec6cd3e0ad46cd3182209afb84937))
* **proxy:** serve /favicon.ico locally instead of tunneling upstream ([#1787](https://github.com/JerrettDavis/headroom/issues/1787)) ([#1847](https://github.com/JerrettDavis/headroom/issues/1847)) ([3076e32](https://github.com/JerrettDavis/headroom/commit/3076e3217228cbb208849d5005a2cd5e1d69606e))
* **proxy:** stamp X-Client: codex on Responses endpoint for unidentified callers ([#1036](https://github.com/JerrettDavis/headroom/issues/1036)) ([b0cd032](https://github.com/JerrettDavis/headroom/commit/b0cd0329c75c8556c51c1c96dc19f2ab6a23677d))
* **proxy:** stop re-compressing headroom_retrieve output and emitting unredeemable markers ([#1323](https://github.com/JerrettDavis/headroom/issues/1323)) ([43494ff](https://github.com/JerrettDavis/headroom/commit/43494ff526468a63ecf028e081a357d1f619ef56))
* **proxy:** stop rtk stat failures from corrupting session baseline ([#1693](https://github.com/JerrettDavis/headroom/issues/1693)) ([681b9a8](https://github.com/JerrettDavis/headroom/commit/681b9a8c1a96af564767d221e92e0ef6620f8a37))
* **proxy:** strip 1m model suffix before upstream forwarding ([#1840](https://github.com/JerrettDavis/headroom/issues/1840)) ([e22d745](https://github.com/JerrettDavis/headroom/commit/e22d7453d4c6fcf084135ad65c21dd4feb9927ad))
* **proxy:** strip Codex lite header from OpenAI WebSockets ([#1543](https://github.com/JerrettDavis/headroom/issues/1543)) ([5d3803a](https://github.com/JerrettDavis/headroom/commit/5d3803a21c53907e2fea900524e48b510dd59d7a))
* **proxy:** strip Codex lite header on the HTTP /responses path ([#1663](https://github.com/JerrettDavis/headroom/issues/1663)) ([9fbd47b](https://github.com/JerrettDavis/headroom/commit/9fbd47ba6bdf38b618795541ee517b7e2fa2c6df))
* **proxy:** strip duplicated upstream server headers ([#1828](https://github.com/JerrettDavis/headroom/issues/1828)) ([d2a86b5](https://github.com/JerrettDavis/headroom/commit/d2a86b590978cae32bf95a20013ad539e942be31))
* **proxy:** strip inbound Content-Encoding on messages/chat forward ([#1970](https://github.com/JerrettDavis/headroom/issues/1970)) ([4cb33cd](https://github.com/JerrettDavis/headroom/commit/4cb33cd9e3766a1255cd9fbb7ac577c1d62b1aa2))
* **proxy:** subtract cache write premiums from net savings ([#1800](https://github.com/JerrettDavis/headroom/issues/1800)) ([53a465b](https://github.com/JerrettDavis/headroom/commit/53a465b121e0a7f45f862a21829639423226a5eb))
* **proxy:** treat NODE_EXTRA_CA_CERTS as additive, not replacement ([#998](https://github.com/JerrettDavis/headroom/issues/998)) ([#1031](https://github.com/JerrettDavis/headroom/issues/1031)) ([c987283](https://github.com/JerrettDavis/headroom/commit/c98728363a1079f39bb19da2955cc859b35900a8))
* **proxy:** wire --compression-max-workers / HEADROOM_COMPRESSION_MAX_WORKERS ([#1632](https://github.com/JerrettDavis/headroom/issues/1632)) ([814ffa3](https://github.com/JerrettDavis/headroom/commit/814ffa36a4d1bb40165a630f96a855452037735e))
* **read-lifecycle:** persist STALE Read originals in the CCR store ([#1488](https://github.com/JerrettDavis/headroom/issues/1488)) ([9157173](https://github.com/JerrettDavis/headroom/commit/915717301860036005f3a51a5306762ae588ed11))
* recover persistent proxy feature checks and reject non-Copilot exchange URL ([#1465](https://github.com/JerrettDavis/headroom/issues/1465)) ([16c638b](https://github.com/JerrettDavis/headroom/commit/16c638bc211ecc6d1768bbe36e0c12971996e104))
* **release:** sync all package versions to v0.31.0 ([#1882](https://github.com/JerrettDavis/headroom/issues/1882)) ([662b7bc](https://github.com/JerrettDavis/headroom/commit/662b7bc00eb4cfb1f72449e510fe576d240db384))
* **relevance:** gate ONNX embedding backend behind AVX2 to avoid SIGILL ([#1723](https://github.com/JerrettDavis/headroom/issues/1723)) ([#1765](https://github.com/JerrettDavis/headroom/issues/1765)) ([728b330](https://github.com/JerrettDavis/headroom/commit/728b33088b25c390d9b06082afab058e1dc7f028))
* remove agents.md ([#1540](https://github.com/JerrettDavis/headroom/issues/1540)) ([a7d3360](https://github.com/JerrettDavis/headroom/commit/a7d3360a05d4fd139cceab5f72d7de4ef7c712b0))
* respect COPILOT_PROVIDER_TYPE env var when provider_type is auto ([#549](https://github.com/JerrettDavis/headroom/issues/549)) ([24cf256](https://github.com/JerrettDavis/headroom/commit/24cf256e50fbd0df8ac67fefa90982cd20807274))
* restore token-mode compression on frozen prefixes ([#1489](https://github.com/JerrettDavis/headroom/issues/1489)) ([8e0dadf](https://github.com/JerrettDavis/headroom/commit/8e0dadfe02da144ca0b27906a8a82bb4be2cb720))
* route v1internal code assist requests to cloudcode-pa.googleapis… ([#821](https://github.com/JerrettDavis/headroom/issues/821)) ([e20f16b](https://github.com/JerrettDavis/headroom/commit/e20f16b1a65710f532aa019ef60ac7a18a4e7f46))
* **router:** degrade to pure-Python detection on native panic ([#1123](https://github.com/JerrettDavis/headroom/issues/1123)) ([#1260](https://github.com/JerrettDavis/headroom/issues/1260)) ([a00fb67](https://github.com/JerrettDavis/headroom/commit/a00fb6761eddf59ede6767211da06f8840552f14))
* **router:** honor MCP aliases in excluded tools ([#1822](https://github.com/JerrettDavis/headroom/issues/1822)) ([#1863](https://github.com/JerrettDavis/headroom/issues/1863)) ([140d6e4](https://github.com/JerrettDavis/headroom/commit/140d6e4f9609eefd674dc435cfcaa9d4e451f9b0))
* **rtk:** link managed rtk onto PATH instead of mutating the hook ([#1698](https://github.com/JerrettDavis/headroom/issues/1698)) ([140cb05](https://github.com/JerrettDavis/headroom/commit/140cb05fbc76e0cd1a54d2a8f98cbbd634a227cd))
* **rtk:** stop hook registration timing out on a forked daemon ([#1314](https://github.com/JerrettDavis/headroom/issues/1314)) ([9758817](https://github.com/JerrettDavis/headroom/commit/97588179790da9fa13ad6793b3cb8e485b43f9b3))
* **savings:** cap ledger retention at 30 days ([#1985](https://github.com/JerrettDavis/headroom/issues/1985)) ([b3a559b](https://github.com/JerrettDavis/headroom/commit/b3a559ba56a3b35308dbff2844408762899d037c))
* **savings:** count cache-read tokens in input cost estimate ([#1429](https://github.com/JerrettDavis/headroom/issues/1429)) ([72ade37](https://github.com/JerrettDavis/headroom/commit/72ade3711211183b9134a46d9c5d45db6a87edc2))
* **scripts:** rename .releaseetadata to .releasemetadata ([#1246](https://github.com/JerrettDavis/headroom/issues/1246)) ([772adc9](https://github.com/JerrettDavis/headroom/commit/772adc93b253d73e91a1a4888e5338f1f71a887a))
* **search-compressor:** CJK-aware relevance + harden Rust/Python parity ([#1749](https://github.com/JerrettDavis/headroom/issues/1749)) ([985621d](https://github.com/JerrettDavis/headroom/commit/985621d60e3c80d94d1205b863bb4974cd346b62))
* skip Magika backend on x86 CPUs without AVX2 ([#1162](https://github.com/JerrettDavis/headroom/issues/1162)) ([64783d8](https://github.com/JerrettDavis/headroom/commit/64783d8824e3c3afc43d9980573d9440693d0963))
* **smart-crusher:** honor enable_ccr_marker on the opaque-blob path ([#1130](https://github.com/JerrettDavis/headroom/issues/1130)) ([27d6f8e](https://github.com/JerrettDavis/headroom/commit/27d6f8e2a767b58eb7d2f47599f68e8bdc49fb7f))
* **streaming:** preserve server_tool_use sse blocks ([#1826](https://github.com/JerrettDavis/headroom/issues/1826)) ([4ac5493](https://github.com/JerrettDavis/headroom/commit/4ac54934cbebe77f72a2cd7432ea792f17a5fd65))
* strip output-only fallback blocks from request messages ([#1870](https://github.com/JerrettDavis/headroom/issues/1870)) ([1448718](https://github.com/JerrettDavis/headroom/commit/1448718fcadcc6cde474719105289924155dce7c))
* **subscription/copilot:** preserve remaining=0 for exhausted quota ([#1997](https://github.com/JerrettDavis/headroom/issues/1997)) ([cbb7750](https://github.com/JerrettDavis/headroom/commit/cbb775015e30c8eb783bcec44c1f46b7c12cab48))
* **subscription:** only reset 5h contribution on real rollover, not API jitter ([#1255](https://github.com/JerrettDavis/headroom/issues/1255)) ([8d6c175](https://github.com/JerrettDavis/headroom/commit/8d6c175d605b88d1c5a7f5e7671778a0e54fb09e))
* **subscription:** run transcript token scan off the event loop ([#1263](https://github.com/JerrettDavis/headroom/issues/1263)) ([f03021f](https://github.com/JerrettDavis/headroom/commit/f03021f1b69ec1a099436a5f80e68d5266cad8bf))
* surface output reduction without a restart, and explain $0.00 savings on Python 3.14 ([#1296](https://github.com/JerrettDavis/headroom/issues/1296)) ([c30ec4c](https://github.com/JerrettDavis/headroom/commit/c30ec4cda8d5340dd98ba1653a7e85f684eb7c3d))
* **telemetry:** switch anonymous telemetry to opt-in (off by default) ([#1223](https://github.com/JerrettDavis/headroom/issues/1223)) ([b998697](https://github.com/JerrettDavis/headroom/commit/b99869778bb3ebe223015bdd051e3b9746c8a22c))
* **tests:** reset whole headroom logger subtree so caplog stays deterministic ([#1117](https://github.com/JerrettDavis/headroom/issues/1117)) ([fda4670](https://github.com/JerrettDavis/headroom/commit/fda4670ef8a8ee279f5afc38ccfecf966762ada2))
* **tls:** add HEADROOM_TLS_STRICT=0 toggle for corporate SSL inspection ([#1308](https://github.com/JerrettDavis/headroom/issues/1308)) ([#1341](https://github.com/JerrettDavis/headroom/issues/1341)) ([52068dd](https://github.com/JerrettDavis/headroom/commit/52068dd650d06d400db472efe6c7b47f539612aa))
* **toin:** publish skip compression recommendations ([#1782](https://github.com/JerrettDavis/headroom/issues/1782)) ([be51008](https://github.com/JerrettDavis/headroom/commit/be51008c701f18e6856efc65d46401dbd2c9856f))
* **tokenizers:** bound tiktoken vocab load so a stalled download cannot hang requests ([#956](https://github.com/JerrettDavis/headroom/issues/956)) ([#994](https://github.com/JerrettDavis/headroom/issues/994)) ([7e86baf](https://github.com/JerrettDavis/headroom/commit/7e86bafb9004e40716a04e22398d24157928ca67))
* **tokenizers:** don't tokenize image blocks as text in TiktokenCounter ([#2093](https://github.com/JerrettDavis/headroom/issues/2093)) ([ae10d6c](https://github.com/JerrettDavis/headroom/commit/ae10d6c99d7b187a92a9bcef3e7aabfe8b43a97f))
* **tokenizers:** price CJK in the fixed-ratio estimator path ([#2080](https://github.com/JerrettDavis/headroom/issues/2080)) ([cd3d5aa](https://github.com/JerrettDavis/headroom/commit/cd3d5aa10c5f43b4dbf8e3741e753d014dcfad7b))
* **tokenizers:** price CJK/Kana/Hangul at ~1 token per char in EstimatingTokenCounter ([#1093](https://github.com/JerrettDavis/headroom/issues/1093)) ([a35fe86](https://github.com/JerrettDavis/headroom/commit/a35fe86e87725e660779f9cbbb0825f87f59d532))
* **tokenizers:** recurse into list-content tool_result blocks ([#2081](https://github.com/JerrettDavis/headroom/issues/2081)) ([dfb1d37](https://github.com/JerrettDavis/headroom/commit/dfb1d37ed619d30c8e79b2bc637d56e524224684))
* **tokenizers:** resolve HF tokenizer names by most-specific prefix ([#2096](https://github.com/JerrettDavis/headroom/issues/2096)) ([e0232df](https://github.com/JerrettDavis/headroom/commit/e0232df9b42b326358addacb8980aa1dd207ee01))
* **tokenizers:** use o200k_base for gpt-4.1/gpt-4.5/o4 families ([#2108](https://github.com/JerrettDavis/headroom/issues/2108)) ([6979b52](https://github.com/JerrettDavis/headroom/commit/6979b5245eb36020e8b2f806a8fe2137901961af))
* **transforms/code:** coerce language aliases instead of raising ([#1975](https://github.com/JerrettDavis/headroom/issues/1975)) ([27ddde1](https://github.com/JerrettDavis/headroom/commit/27ddde1f5e3ced40ca237bc6bbfbe76cb896d97a))
* **transforms/content-router:** route grep/log output away from HTML extractor ([#1719](https://github.com/JerrettDavis/headroom/issues/1719)) ([0d18ef2](https://github.com/JerrettDavis/headroom/commit/0d18ef26f4d126f8eec9df1d34330a7129c4c63f))
* **transforms:** bound native content detection with a Windows watchdog ([#575](https://github.com/JerrettDavis/headroom/issues/575)) ([#1563](https://github.com/JerrettDavis/headroom/issues/1563)) ([95abca3](https://github.com/JerrettDavis/headroom/commit/95abca3abd69add5f075d241284b565e0014d5a4))
* **transforms:** gate tool string output from lossy compression ([#1307](https://github.com/JerrettDavis/headroom/issues/1307)) ([#1387](https://github.com/JerrettDavis/headroom/issues/1387)) ([c6c921a](https://github.com/JerrettDavis/headroom/commit/c6c921a7c135a19c68fcd85ac5bdddd4ee9c1e8d))
* **transforms:** normalize diff compressor context ([#1801](https://github.com/JerrettDavis/headroom/issues/1801)) ([838c523](https://github.com/JerrettDavis/headroom/commit/838c5234a877d4cf96f9e914cfd39d6d6addb211))
* **transforms:** pass through ragged tables instead of misaligning columns ([#1713](https://github.com/JerrettDavis/headroom/issues/1713)) ([c7665ca](https://github.com/JerrettDavis/headroom/commit/c7665ca08863da12dc9c656bd8bdf1f55c95bda7))
* **unwrap:** remove ANTHROPIC_BASE_URL + ENABLE_TOOL_SEARCH and init hooks on unwrap ([#992](https://github.com/JerrettDavis/headroom/issues/992)) ([5b84691](https://github.com/JerrettDavis/headroom/commit/5b846917701e346739346c99c48d5ab6e226e17d))
* **update:** prevent _core.pyd corruption on Windows when proxy is running ([#1581](https://github.com/JerrettDavis/headroom/issues/1581)) ([0750bbf](https://github.com/JerrettDavis/headroom/commit/0750bbff4df3a66f11c8b82c33224ef3264fae42))
* use rtk native Cursor hook instead of injecting .cursorrules ([#756](https://github.com/JerrettDavis/headroom/issues/756)) ([#1846](https://github.com/JerrettDavis/headroom/issues/1846)) ([1573f1f](https://github.com/JerrettDavis/headroom/commit/1573f1fd0763408246f5dd0d7a92f32464f5fdbb))
* **version:** mark source-checkout builds as -dev ([#2072](https://github.com/JerrettDavis/headroom/issues/2072)) ([1cc9979](https://github.com/JerrettDavis/headroom/commit/1cc99792ac087fb8df916b1fd00afa58dae85e29))
* Vertex AI support for Claude Code with ANTHROPIC_VERTEX_BASE_URL ([#1393](https://github.com/JerrettDavis/headroom/issues/1393)) ([cff7247](https://github.com/JerrettDavis/headroom/commit/cff7247efd6fbecc1c2e66280a4a9b6381d7b7a4))
* **websocket:** harden responses websocket origin handling ([#1481](https://github.com/JerrettDavis/headroom/issues/1481)) ([c632023](https://github.com/JerrettDavis/headroom/commit/c632023cc1ec61d15f8f8e86efe3b54d51604a64))
* **windows:** pin UTF-8 encoding on text-mode subprocess calls ([#1311](https://github.com/JerrettDavis/headroom/issues/1311)) ([d633e81](https://github.com/JerrettDavis/headroom/commit/d633e8172ccfde4b08c302ecc4c4ef4ce27785f1))
* **wrap/opencode:** unwrap removes the rtk block from AGENTS.md ([#2025](https://github.com/JerrettDavis/headroom/issues/2025)) ([20968a4](https://github.com/JerrettDavis/headroom/commit/20968a4fa4611abd8059d75804879faa5499dbfa))
* **wrap:** add Copilot unwrap command ([#1251](https://github.com/JerrettDavis/headroom/issues/1251)) ([b4fde0c](https://github.com/JerrettDavis/headroom/commit/b4fde0c3a4c2585d4aeda2c6987fe509a5296fe5))
* **wrap:** detach the shared proxy on Windows so it survives an ungraceful agent close ([#1464](https://github.com/JerrettDavis/headroom/issues/1464)) ([6cba441](https://github.com/JerrettDavis/headroom/commit/6cba4419d04bea79c1b44632a9288cde5b48bbce))
* **wrap:** isolate proxy stdio from proxy.log on Windows ([#1191](https://github.com/JerrettDavis/headroom/issues/1191)) ([959ab0d](https://github.com/JerrettDavis/headroom/commit/959ab0de471293e76df1f124ed0090c62e62c308))
* **wrap:** keep agent savings opt-in ([#1294](https://github.com/JerrettDavis/headroom/issues/1294)) ([b829ceb](https://github.com/JerrettDavis/headroom/commit/b829ceba84ce058dadb4e70f6766af13806a4385))
* **wrap:** keep Claude context-tool setup explicit ([#1999](https://github.com/JerrettDavis/headroom/issues/1999)) ([f536aa0](https://github.com/JerrettDavis/headroom/commit/f536aa0801af554dec87cb278a0c31cc049c0c26))
* **wrap:** keep Codex RTK guidance global ([#1240](https://github.com/JerrettDavis/headroom/issues/1240)) ([7c26a54](https://github.com/JerrettDavis/headroom/commit/7c26a54d53aa06a3d75e1111b285c2593155c43e))
* **wrap:** percent-encode non-ASCII cwd names in X-Headroom-Project header ([#1071](https://github.com/JerrettDavis/headroom/issues/1071)) ([9f712cc](https://github.com/JerrettDavis/headroom/commit/9f712ccbd7ec27b74f6ac7f20b7d2a9743dba1d8))
* **wrap:** preserve custom Codex provider base_url during proxy injection ([#1894](https://github.com/JerrettDavis/headroom/issues/1894)) ([372d6c8](https://github.com/JerrettDavis/headroom/commit/372d6c8cd4bec2d7f7448bf23503b11570f72b47))
* **wrap:** preserve custom Vertex base URL ([#1477](https://github.com/JerrettDavis/headroom/issues/1477)) ([75427bb](https://github.com/JerrettDavis/headroom/commit/75427bbd4ad14fcb1b205f3253ec4e24ae1d2118))
* **wrap:** remove rtk instructions from Codex AGENTS.md on unwrap ([#1604](https://github.com/JerrettDavis/headroom/issues/1604)) ([c9d717c](https://github.com/JerrettDavis/headroom/commit/c9d717c13c7ae006178e49b6570f63b3f82de9a2))
* **wrap:** replace stale-proxy detection with Vite-style port fallback ([#1406](https://github.com/JerrettDavis/headroom/issues/1406)) ([b4205c6](https://github.com/JerrettDavis/headroom/commit/b4205c68e63e1e12e354508d8c3ac7d54781268b))
* **wrap:** show the dashboard URL when the proxy is already running ([#1313](https://github.com/JerrettDavis/headroom/issues/1313)) ([b0146c4](https://github.com/JerrettDavis/headroom/commit/b0146c4ccd1e75dc7db21ef7f00dd4b3aa80e276))
* **wrap:** surface Claude Remote Control base-URL gate accurately ([#1](https://github.com/JerrettDavis/headroom/issues/1)… ([#1883](https://github.com/JerrettDavis/headroom/issues/1883)) ([daeff69](https://github.com/JerrettDavis/headroom/commit/daeff69a7549d40b579caf2b5d6a2840a5df68b5))
* **wrap:** use canonical headroom-openclaw npm package for wrap openclaw ([#1969](https://github.com/JerrettDavis/headroom/issues/1969)) ([#2120](https://github.com/JerrettDavis/headroom/issues/2120)) ([c5545d6](https://github.com/JerrettDavis/headroom/commit/c5545d6ac47a71efce6e9c3c9293c88bc8c0951a))
* **wrap:** write env.ANTHROPIC_BASE_URL to settings.json so daemon-spawned conversations inherit proxy ([#951](https://github.com/JerrettDavis/headroom/issues/951)) ([#1078](https://github.com/JerrettDavis/headroom/issues/1078)) ([a554c3a](https://github.com/JerrettDavis/headroom/commit/a554c3a0e6c5c57a7c745d8648024362d9d502a4))


### Performance Improvements

* **compression:** take large cold-start contexts off the synchronous kompress path ([#1171](https://github.com/JerrettDavis/headroom/issues/1171)) ([#1298](https://github.com/JerrettDavis/headroom/issues/1298)) ([6c68ff4](https://github.com/JerrettDavis/headroom/commit/6c68ff4e9f911af9dbd6108367acb3cab80d6f5e))
* **proxy:** cap compression workers to CPU count ([#1803](https://github.com/JerrettDavis/headroom/issues/1803)) ([0a3851b](https://github.com/JerrettDavis/headroom/commit/0a3851b24004727e734b61af4e3f59ce3b0bfe10))
* **savings:** batch tracker persistence off the request hot path ([#1817](https://github.com/JerrettDavis/headroom/issues/1817)) ([451b9f0](https://github.com/JerrettDavis/headroom/commit/451b9f0867f1eb7cf3a1b479f67a4e3106f7e9be))


### Dependencies

* bump @types/node from 22.19.15 to 26.1.1 in /plugins/openclaw ([#1685](https://github.com/JerrettDavis/headroom/issues/1685)) ([350daeb](https://github.com/JerrettDavis/headroom/commit/350daeba73edd0954d405fb65580e5ce46bf2be9))
* bump @types/node from 22.20.0 to 26.1.1 in /plugins/opencode ([#1688](https://github.com/JerrettDavis/headroom/issues/1688)) ([8715195](https://github.com/JerrettDavis/headroom/commit/87151952eeb85165d14034a1a0f77a70ae824848))
* bump @types/node from 25.5.2 to 26.1.1 in /docs ([#1683](https://github.com/JerrettDavis/headroom/issues/1683)) ([75fff43](https://github.com/JerrettDavis/headroom/commit/75fff43eca7901d1e4809c8e7762b974d29f5c14))
* bump fumadocs-typescript from 4.0.14 to 5.3.0 in /docs ([#1684](https://github.com/JerrettDavis/headroom/issues/1684)) ([e8b66a2](https://github.com/JerrettDavis/headroom/commit/e8b66a27e1bb7d6452172530ffc024a5f109c49b))
* bump prometheus from 0.13.4 to 0.14.0 ([#1518](https://github.com/JerrettDavis/headroom/issues/1518)) ([5229c98](https://github.com/JerrettDavis/headroom/commit/5229c98228e524fb3df902fb0b56a3eb54e1a2c5))
* bump the cargo-minor-patch group across 1 directory with 7 updates ([#1909](https://github.com/JerrettDavis/headroom/issues/1909)) ([45601d9](https://github.com/JerrettDavis/headroom/commit/45601d93bcd92f7f66d4c3483d9f4512a10e933c))
* bump the npm-minor-patch group across 4 directories with 18 updates ([#1907](https://github.com/JerrettDavis/headroom/issues/1907)) ([8872bbc](https://github.com/JerrettDavis/headroom/commit/8872bbc6a2fa210e9f26d33d1ff8e019954bddd9))
* bump thiserror from 1.0.69 to 2.0.18 ([#1519](https://github.com/JerrettDavis/headroom/issues/1519)) ([e448d7b](https://github.com/JerrettDavis/headroom/commit/e448d7ba4d6366a883d77a3d7f882cb1a71a0550))
* bump toml from 0.8.23 to 1.1.2+spec-1.1.0 ([#1517](https://github.com/JerrettDavis/headroom/issues/1517)) ([6c705b4](https://github.com/JerrettDavis/headroom/commit/6c705b40667fc8998cf9b72da0e44284ed8d0853))
* update tree-sitter requirement from &lt;0.26,&gt;=0.25.2 to &gt;=0.25.2,&lt;0.27 ([#1681](https://github.com/JerrettDavis/headroom/issues/1681)) ([ce3c959](https://github.com/JerrettDavis/headroom/commit/ce3c959eaed1bd53f492ae7cc612a3fd7b12daf3))


### Code Refactoring

* **cache:** isolate compression strategy outcomes ([#1938](https://github.com/JerrettDavis/headroom/issues/1938)) ([b5aa8a3](https://github.com/JerrettDavis/headroom/commit/b5aa8a358e70edf875bb768fae0ee04af0dd7921))
* **cache:** isolate semantic key policy ([#1953](https://github.com/JerrettDavis/headroom/issues/1953)) ([740fb9b](https://github.com/JerrettDavis/headroom/commit/740fb9bc16eb3d9db57b253ac22e62e52cb19860))
* **ccr:** isolate tool call classification ([#1937](https://github.com/JerrettDavis/headroom/issues/1937)) ([fd5b9e7](https://github.com/JerrettDavis/headroom/commit/fd5b9e75ad69db670bd9e0c77a816dd0fda6a199))
* **memory:** isolate injection decision policy ([#1952](https://github.com/JerrettDavis/headroom/issues/1952)) ([c20f3b1](https://github.com/JerrettDavis/headroom/commit/c20f3b1c0434b3d100a1d5edca22d8c9c0f7c9f9))
* **memory:** isolate query construction policy ([#1950](https://github.com/JerrettDavis/headroom/issues/1950)) ([235c986](https://github.com/JerrettDavis/headroom/commit/235c986c9cc7d36fffc4298bbcab77079d8f2f43))
* **output:** isolate savings policy ([#1947](https://github.com/JerrettDavis/headroom/issues/1947)) ([c29b4ba](https://github.com/JerrettDavis/headroom/commit/c29b4ba84f021c62c869a12e14ce8671e2f01141))
* **output:** isolate verbosity steering ([#1940](https://github.com/JerrettDavis/headroom/issues/1940)) ([0ce09fb](https://github.com/JerrettDavis/headroom/commit/0ce09fb63fff6d6c5180cecc086f6ae64b29975b))
* **pricing:** isolate litellm model resolution ([#1936](https://github.com/JerrettDavis/headroom/issues/1936)) ([4210d6e](https://github.com/JerrettDavis/headroom/commit/4210d6e60954ee5a091c9669a92e36e830bae7de))
* **providers:** split proxy route adapters ([#1934](https://github.com/JerrettDavis/headroom/issues/1934)) ([e6243f6](https://github.com/JerrettDavis/headroom/commit/e6243f65c9675790515eb2894ad174ab80cea23a))
* **proxy:** extract beta header merge policy ([#1993](https://github.com/JerrettDavis/headroom/issues/1993)) ([f359f21](https://github.com/JerrettDavis/headroom/commit/f359f21424a54e9d9ef34ca7de49a9d11aa50589))
* **proxy:** extract beta header policy ([#1992](https://github.com/JerrettDavis/headroom/issues/1992)) ([603f5bc](https://github.com/JerrettDavis/headroom/commit/603f5bcfd663d2fe71f11e0aabc76c8993a0456b))
* **proxy:** extract ccr golden replay policy ([#2006](https://github.com/JerrettDavis/headroom/issues/2006)) ([7c9a032](https://github.com/JerrettDavis/headroom/commit/7c9a032f5044f24fdde71126f591f3eac0a93da1))
* **proxy:** extract ccr marker policy ([#2004](https://github.com/JerrettDavis/headroom/issues/2004)) ([ec3c3cd](https://github.com/JerrettDavis/headroom/commit/ec3c3cd2345a9aed682eec719370c36bb79e2173))
* **proxy:** extract ccr session tracker ([#2003](https://github.com/JerrettDavis/headroom/issues/2003)) ([e92c253](https://github.com/JerrettDavis/headroom/commit/e92c2539779dfb04404f7618dfb9bb87e01e88f1))
* **proxy:** extract internal header policy ([#1990](https://github.com/JerrettDavis/headroom/issues/1990)) ([868b88b](https://github.com/JerrettDavis/headroom/commit/868b88bc6400c98f11134dbbe3cb03d1ecff7e1d))
* **proxy:** extract memory golden replay policy ([#2007](https://github.com/JerrettDavis/headroom/issues/2007)) ([8c68f48](https://github.com/JerrettDavis/headroom/commit/8c68f48903b354f8ff19c74df7652bbcc3693185))
* **proxy:** extract tool injection config ([#2010](https://github.com/JerrettDavis/headroom/issues/2010)) ([0f846e5](https://github.com/JerrettDavis/headroom/commit/0f846e5a8fb58942431b1edc22dfda9ab7d6de70))
* **proxy:** extract tool injection logging ([#2009](https://github.com/JerrettDavis/headroom/issues/2009)) ([9c7b9d5](https://github.com/JerrettDavis/headroom/commit/9c7b9d5a9c7a5b2c55422f85976c796045f477fc))
* **proxy:** extract tool injection policy ([#1995](https://github.com/JerrettDavis/headroom/issues/1995)) ([d6259b2](https://github.com/JerrettDavis/headroom/commit/d6259b226365abba47048a49a579ead23e55684c))
* **proxy:** extract tool injection tracker ([#2002](https://github.com/JerrettDavis/headroom/issues/2002)) ([d1c484b](https://github.com/JerrettDavis/headroom/commit/d1c484b164145f5010a048d299a61d7dac903616))
* **proxy:** extract tool name policy ([#2008](https://github.com/JerrettDavis/headroom/issues/2008)) ([1000175](https://github.com/JerrettDavis/headroom/commit/10001755e85a5acaa50644ee205adaa01d8c55d8))
* **proxy:** isolate auth classification policy ([#1945](https://github.com/JerrettDavis/headroom/issues/1945)) ([5a7265d](https://github.com/JerrettDavis/headroom/commit/5a7265daa80df88e3850a6dcc834364fc3ced5a4))
* **proxy:** isolate body forwarding policy ([#1935](https://github.com/JerrettDavis/headroom/issues/1935)) ([1f3696a](https://github.com/JerrettDavis/headroom/commit/1f3696a3d0c67e3bdff90ad0c93cf7643af0a496))
* **proxy:** isolate forwarded header policy ([#1942](https://github.com/JerrettDavis/headroom/issues/1942)) ([cb38f79](https://github.com/JerrettDavis/headroom/commit/cb38f7937705ba2c4db50792a7d1feb0a19018bb))
* **proxy:** isolate image compression policy ([#1958](https://github.com/JerrettDavis/headroom/issues/1958)) ([2b09ece](https://github.com/JerrettDavis/headroom/commit/2b09ecea76346a371bc7886eeb66d88d2489a50e))
* **proxy:** isolate memory rank policy ([#1960](https://github.com/JerrettDavis/headroom/issues/1960)) ([b1e871d](https://github.com/JerrettDavis/headroom/commit/b1e871d51ccd4e013da1cece02b18d1454503d77))
* **proxy:** isolate output effort policy ([#1961](https://github.com/JerrettDavis/headroom/issues/1961)) ([094a53c](https://github.com/JerrettDavis/headroom/commit/094a53c0479087188693d707b396d8a778b9e6bc))
* **proxy:** isolate output turn policy ([#1962](https://github.com/JerrettDavis/headroom/issues/1962)) ([c904a70](https://github.com/JerrettDavis/headroom/commit/c904a70d4ef9d2955d5c96d86900cbe9ecf1e51c))
* **proxy:** isolate output verbosity policy ([#1963](https://github.com/JerrettDavis/headroom/issues/1963)) ([0415dc8](https://github.com/JerrettDavis/headroom/commit/0415dc87656f2a87b23abee1a329498398ce4b98))
* **proxy:** isolate project attribution policy ([#1957](https://github.com/JerrettDavis/headroom/issues/1957)) ([1c1e360](https://github.com/JerrettDavis/headroom/commit/1c1e3601120df95142fb292ccc2dd5a0e3ab30e6))
* **proxy:** isolate proxy mode policy ([#1965](https://github.com/JerrettDavis/headroom/issues/1965)) ([82af5cd](https://github.com/JerrettDavis/headroom/commit/82af5cdfe256177d1aac01f191e694f81786c209))
* **proxy:** isolate rate limit policy ([#1954](https://github.com/JerrettDavis/headroom/issues/1954)) ([ea19515](https://github.com/JerrettDavis/headroom/commit/ea1951508b3af6090b71852ec58fd814032b1253))
* **proxy:** isolate semantic cache key policy ([#1964](https://github.com/JerrettDavis/headroom/issues/1964)) ([2f53a18](https://github.com/JerrettDavis/headroom/commit/2f53a18a3fe2a8ce9c8f57d68ee54a834fb4ba1f))
* **transforms:** isolate mixed content parsing ([#1939](https://github.com/JerrettDavis/headroom/issues/1939)) ([9bacf48](https://github.com/JerrettDavis/headroom/commit/9bacf4810fe5a950c644b38926801d5aa0382e25))

## [0.31.0](https://github.com/headroomlabs-ai/headroom/compare/v0.30.0...v0.31.0) (2026-07-09)


### Features

* **cache:** provider-agnostic cache-mode delta + cc-agnostic prefix comparison ([#1868](https://github.com/headroomlabs-ai/headroom/issues/1868)) ([7c2f0ea](https://github.com/headroomlabs-ai/headroom/commit/7c2f0ea07953beaed45b25bd0fc8c5a34d60cb3f))
* **ccr:** wire retrieve-tool interception into OpenAI Responses handler ([#1898](https://github.com/headroomlabs-ai/headroom/issues/1898)) ([62cd307](https://github.com/headroomlabs-ai/headroom/commit/62cd3072a2ea9bcd8410e400cab6f678501b5b37))
* **compression:** add audit-safe mode with protected pattern matching ([#1899](https://github.com/headroomlabs-ai/headroom/issues/1899)) ([bb112dd](https://github.com/headroomlabs-ai/headroom/commit/bb112dd1762bf744a05689d54c50aed28265ee90))
* **content-router:** accept any real compression (remove min-savings floor) ([#1771](https://github.com/headroomlabs-ai/headroom/issues/1771)) ([6c31db9](https://github.com/headroomlabs-ai/headroom/commit/6c31db97fbd68f88c39a71785335fc8917702fc3))
* **content-router:** lossless-first dispatch, cross-turn dedup, and A7 lossy-after-fold ([#1818](https://github.com/headroomlabs-ai/headroom/issues/1818)) ([60af15f](https://github.com/headroomlabs-ai/headroom/commit/60af15f96f1792ad50bf259a765ee188db73d1aa))
* **proxy:** add provider-only HTTP proxy ([#1807](https://github.com/headroomlabs-ai/headroom/issues/1807)) ([ebe0a3b](https://github.com/headroomlabs-ai/headroom/commit/ebe0a3bd7bbc8bbe4ee52bdb1ed7420a405dc224))
* **proxy:** add turn-hook extension point for buffered model turns ([#1891](https://github.com/headroomlabs-ai/headroom/issues/1891)) ([ec950f7](https://github.com/headroomlabs-ai/headroom/commit/ec950f7ef131fb124b60a8e75bc6af7ab733cc7f))


### Bug Fixes

* **build:** enable Intel macOS pip installs via ort-load-dynamic ([#1538](https://github.com/headroomlabs-ai/headroom/issues/1538)) ([32ce99e](https://github.com/headroomlabs-ai/headroom/commit/32ce99e4b4a7d75f31429a553f2211a83992047a))
* **cache:** avoid fallback session collisions ([#1827](https://github.com/headroomlabs-ai/headroom/issues/1827)) ([0f606b6](https://github.com/headroomlabs-ai/headroom/commit/0f606b6281dd4c55e1c5a32cc97c418b66860df1))
* **ccr:** make expired retrieve misses terminal ([#1781](https://github.com/headroomlabs-ai/headroom/issues/1781)) ([9cbdba4](https://github.com/headroomlabs-ai/headroom/commit/9cbdba4dc1f38f73d255211eec439674f3f2f9f1))
* **ccr:** preserve Anthropic re-stream shape ([#1854](https://github.com/headroomlabs-ai/headroom/issues/1854)) ([f663894](https://github.com/headroomlabs-ai/headroom/commit/f663894f6072dbd13f5a1caa05dfea6657f5a3b0))
* **ccr:** preserve thinking blocks in buffered stream re-synthesis ([#1897](https://github.com/headroomlabs-ai/headroom/issues/1897)) ([ede085c](https://github.com/headroomlabs-ai/headroom/commit/ede085cc11d74778e43ce0fb0828a53a0a06a14b))
* **cli/proxy:** preserve explicit HEADROOM_MIN_TOKENS=0 / MAX_ITEMS=0 ([#1886](https://github.com/headroomlabs-ai/headroom/issues/1886)) ([3a33af1](https://github.com/headroomlabs-ai/headroom/commit/3a33af1af3224594581d0d27ea5b4df1a1c6ba48))
* **code-compressor:** CJK-aware relevance-query symbol matching ([#1747](https://github.com/headroomlabs-ai/headroom/issues/1747)) ([b38315c](https://github.com/headroomlabs-ai/headroom/commit/b38315cf72e4248cc76cc0e0d10dfa24a4a332e0))
* **codex:** discover updated Codex state stores ([#1889](https://github.com/headroomlabs-ai/headroom/issues/1889)) ([9d42eba](https://github.com/headroomlabs-ai/headroom/commit/9d42ebaa1ab6e22e7b1398a3c0618d0d35895f40))
* **codex:** OpenCode Zen telemetry attribution ([#1648](https://github.com/headroomlabs-ai/headroom/issues/1648)) ([f18c6bd](https://github.com/headroomlabs-ai/headroom/commit/f18c6bd896f7b5a153e3b29f7a27b64c65b08fc5))
* **content-detector:** detect and compress space-separated JSON objects ([#1742](https://github.com/headroomlabs-ai/headroom/issues/1742)) ([5194bdc](https://github.com/headroomlabs-ai/headroom/commit/5194bdc5a6e53d331ce0303aba670e8814bb5fd2))
* **content-router:** token-measure lossless folds at the acceptance gate ([#1772](https://github.com/headroomlabs-ai/headroom/issues/1772)) ([c5493ea](https://github.com/headroomlabs-ai/headroom/commit/c5493ea93bae798d489a82167c1f7bcff79eaecb))
* **copilot:** normalize subscription routing host ([#1836](https://github.com/headroomlabs-ai/headroom/issues/1836)) ([afd9cbd](https://github.com/headroomlabs-ai/headroom/commit/afd9cbdfafba0d31bd376a4a43dbcd41b30ec909))
* **copilot:** route mixed-model requests per model ([#1785](https://github.com/headroomlabs-ai/headroom/issues/1785)) ([5af5e22](https://github.com/headroomlabs-ai/headroom/commit/5af5e22862a0ce0a3d934c2f1e76ea7c1fad71e7))
* **dashboard:** deduplicate repeated savings metrics ([#1804](https://github.com/headroomlabs-ai/headroom/issues/1804)) ([88f935a](https://github.com/headroomlabs-ai/headroom/commit/88f935a1eb52ec81cdd60db44627279d411b74ab))
* **dashboard:** distinguish unavailable RTK from zero stats in Docker ([#1900](https://github.com/headroomlabs-ai/headroom/issues/1900)) ([87f6e93](https://github.com/headroomlabs-ai/headroom/commit/87f6e93c14a9365695142084bc6966d7de70f437))
* **dashboard:** distinguish unavailable RTK from zero stats in Docker ([#1901](https://github.com/headroomlabs-ai/headroom/issues/1901)) ([361adcd](https://github.com/headroomlabs-ai/headroom/commit/361adcd1a00bbdcb949a3efc7b685937d4e84547))
* **dashboard:** price proxy savings without litellm ([#1728](https://github.com/headroomlabs-ai/headroom/issues/1728)) ([188e382](https://github.com/headroomlabs-ai/headroom/commit/188e382b44d09d7f16717377f908869292aab4d9))
* detect and clear stale ANTHROPIC_BASE_URL from crashed wrap sessions ([#1768](https://github.com/headroomlabs-ai/headroom/issues/1768)) ([#1837](https://github.com/headroomlabs-ai/headroom/issues/1837)) ([84509a4](https://github.com/headroomlabs-ai/headroom/commit/84509a4b892cc256331106c807a3a56107f1eec2))
* **docker:** persist headroom workspace in compose ([#1839](https://github.com/headroomlabs-ai/headroom/issues/1839)) ([5e29c06](https://github.com/headroomlabs-ai/headroom/commit/5e29c06aaf5e3d7d9e591914dc656f24eb72cc07))
* **docker:** report source build version ([#1862](https://github.com/headroomlabs-ai/headroom/issues/1862)) ([3807488](https://github.com/headroomlabs-ai/headroom/commit/38074888ac871b8b44418066d66b6a37159978ed))
* **evals:** default unparseable judge scores below pass threshold ([#1892](https://github.com/headroomlabs-ai/headroom/issues/1892)) ([42ebbc6](https://github.com/headroomlabs-ai/headroom/commit/42ebbc6cce02a0fd5e0a6e614348d47f4099649a))
* **install:** pass sc.exe create as raw command line so binPath= quoting survives ([#1654](https://github.com/headroomlabs-ai/headroom/issues/1654)) ([#1702](https://github.com/headroomlabs-ai/headroom/issues/1702)) ([d6e0710](https://github.com/headroomlabs-ai/headroom/commit/d6e07102283745a44aece2222f84c1599eabf90a))
* **install:** persist --no-http2 override through install apply ([#1676](https://github.com/headroomlabs-ai/headroom/issues/1676)) ([6fb5f3b](https://github.com/headroomlabs-ai/headroom/commit/6fb5f3bc3dfa60e56744f85cf049524d43104a31))
* **mcp:** isolate ClaudeRegistrar CLI config env ([#1888](https://github.com/headroomlabs-ai/headroom/issues/1888)) ([1c947b1](https://github.com/headroomlabs-ai/headroom/commit/1c947b1103fa66563a01ea638f1669ee053018e6))
* **mcp:** surface dead proxy state ([#1786](https://github.com/headroomlabs-ai/headroom/issues/1786)) ([931eed8](https://github.com/headroomlabs-ai/headroom/commit/931eed879d26512b2dbdf3ea4246e4f7b2c97a70))
* **memory:** resolve Trae cwd metadata from user reminders ([#1737](https://github.com/headroomlabs-ai/headroom/issues/1737)) ([#1887](https://github.com/headroomlabs-ai/headroom/issues/1887)) ([3e85eb1](https://github.com/headroomlabs-ai/headroom/commit/3e85eb1880af5663cf083492f1dc1415a354bd99))
* **opencode:** use local MCP config ([#1383](https://github.com/headroomlabs-ai/headroom/issues/1383)) ([4bd3ddf](https://github.com/headroomlabs-ai/headroom/commit/4bd3ddfaa5c5655540494b96e4f5d47724460c7d))
* **proxy/openai:** thread savings-profile kwargs into chat completions ([#1606](https://github.com/headroomlabs-ai/headroom/issues/1606)) ([7ff842d](https://github.com/headroomlabs-ai/headroom/commit/7ff842da170b5bceb5d67048473eeb8a18e09a51))
* **proxy/openai:** translate max_tokens -&gt; max_completion_tokens on chat path ([#1774](https://github.com/headroomlabs-ai/headroom/issues/1774)) ([285808b](https://github.com/headroomlabs-ai/headroom/commit/285808b90ea5532fe319c94d408699cb46b2e5f8))
* **proxy:** bound Codex WS compression fallback latency ([#1802](https://github.com/headroomlabs-ai/headroom/issues/1802)) ([d24a3f8](https://github.com/headroomlabs-ai/headroom/commit/d24a3f842551d36c14dc0ec146a9302e256c5c0f))
* **proxy:** bound HF tokenizer load and offload token counting off event loop ([#1738](https://github.com/headroomlabs-ai/headroom/issues/1738)) ([46d5d68](https://github.com/headroomlabs-ai/headroom/commit/46d5d685d9bcdced1f77ffdc0f2d3a8ee8a1f319))
* **proxy:** cancel retry backoff on shutdown ([#1834](https://github.com/headroomlabs-ai/headroom/issues/1834)) ([da2d8dc](https://github.com/headroomlabs-ai/headroom/commit/da2d8dc9dbf3edfcd1c3f6429db32374a6bebc64))
* **proxy:** compress Anthropic user text blocks when enabled ([#1875](https://github.com/headroomlabs-ai/headroom/issues/1875)) ([e36439a](https://github.com/headroomlabs-ai/headroom/commit/e36439a9411bf7fc93b4a5dceac50aa4570a6105))
* **proxy:** freeze must forward cached (compressed) prefix byte-identical — stop token-mode cache busting ([#1850](https://github.com/headroomlabs-ai/headroom/issues/1850)) ([248ae0f](https://github.com/headroomlabs-ai/headroom/commit/248ae0f3e0d4d7ff2e23837e628880dcbda4411a))
* **proxy:** fsync savings dir after atomic rename ([#1764](https://github.com/headroomlabs-ai/headroom/issues/1764)) ([7de2c1e](https://github.com/headroomlabs-ai/headroom/commit/7de2c1e4c2ca8aefd73d3c419dfbcdd881a63bd2))
* **proxy:** keep cache_control bounded + stable so the freeze overlay stops busting ([#1852](https://github.com/headroomlabs-ai/headroom/issues/1852)) ([4820134](https://github.com/headroomlabs-ai/headroom/commit/48201345be16a8b5aad74e8c390850dce0f34ec4))
* **proxy:** persist lifetime cache-read savings across restarts ([#1665](https://github.com/headroomlabs-ai/headroom/issues/1665)) ([908997e](https://github.com/headroomlabs-ai/headroom/commit/908997ef61d91a7e912637c785176719a5f1c719))
* **proxy:** preserve streaming passthrough beta headers ([#1783](https://github.com/headroomlabs-ai/headroom/issues/1783)) ([0f553a8](https://github.com/headroomlabs-ai/headroom/commit/0f553a8ebbd6d790ca622f95f389b5e7d11a41ce))
* **proxy:** release _active_streams session lock on setup-phase errors ([#1864](https://github.com/headroomlabs-ai/headroom/issues/1864)) ([2ccd831](https://github.com/headroomlabs-ai/headroom/commit/2ccd831032e23248879bd38c5bde947d3a0a54f3))
* **proxy:** retry HTTP/2 stream resets instead of 502ing ([#1645](https://github.com/headroomlabs-ai/headroom/issues/1645)) ([2ce19c2](https://github.com/headroomlabs-ai/headroom/commit/2ce19c2c55710cdc5f7a4bb88803f05e4b31feff))
* **proxy:** retry passthrough on transient upstream connection close ([#1513](https://github.com/headroomlabs-ai/headroom/issues/1513)) ([5d14080](https://github.com/headroomlabs-ai/headroom/commit/5d14080c948b04ccd997d2434b37604440701888))
* **proxy:** route Foundry Anthropic messages ([#1878](https://github.com/headroomlabs-ai/headroom/issues/1878)) ([739f654](https://github.com/headroomlabs-ai/headroom/commit/739f654bbd71b3e31ade40ae9eadf812b362beec))
* **proxy:** serve /favicon.ico locally instead of tunneling upstream ([#1787](https://github.com/headroomlabs-ai/headroom/issues/1787)) ([#1847](https://github.com/headroomlabs-ai/headroom/issues/1847)) ([3076e32](https://github.com/headroomlabs-ai/headroom/commit/3076e3217228cbb208849d5005a2cd5e1d69606e))
* **proxy:** stop rtk stat failures from corrupting session baseline ([#1693](https://github.com/headroomlabs-ai/headroom/issues/1693)) ([681b9a8](https://github.com/headroomlabs-ai/headroom/commit/681b9a8c1a96af564767d221e92e0ef6620f8a37))
* **proxy:** strip 1m model suffix before upstream forwarding ([#1840](https://github.com/headroomlabs-ai/headroom/issues/1840)) ([e22d745](https://github.com/headroomlabs-ai/headroom/commit/e22d7453d4c6fcf084135ad65c21dd4feb9927ad))
* **proxy:** subtract cache write premiums from net savings ([#1800](https://github.com/headroomlabs-ai/headroom/issues/1800)) ([53a465b](https://github.com/headroomlabs-ai/headroom/commit/53a465b121e0a7f45f862a21829639423226a5eb))
* **router:** honor MCP aliases in excluded tools ([#1822](https://github.com/headroomlabs-ai/headroom/issues/1822)) ([#1863](https://github.com/headroomlabs-ai/headroom/issues/1863)) ([140d6e4](https://github.com/headroomlabs-ai/headroom/commit/140d6e4f9609eefd674dc435cfcaa9d4e451f9b0))
* **rtk:** link managed rtk onto PATH instead of mutating the hook ([#1698](https://github.com/headroomlabs-ai/headroom/issues/1698)) ([140cb05](https://github.com/headroomlabs-ai/headroom/commit/140cb05fbc76e0cd1a54d2a8f98cbbd634a227cd))
* **streaming:** preserve server_tool_use sse blocks ([#1826](https://github.com/headroomlabs-ai/headroom/issues/1826)) ([4ac5493](https://github.com/headroomlabs-ai/headroom/commit/4ac54934cbebe77f72a2cd7432ea792f17a5fd65))
* **toin:** publish skip compression recommendations ([#1782](https://github.com/headroomlabs-ai/headroom/issues/1782)) ([be51008](https://github.com/headroomlabs-ai/headroom/commit/be51008c701f18e6856efc65d46401dbd2c9856f))
* **transforms:** normalize diff compressor context ([#1801](https://github.com/headroomlabs-ai/headroom/issues/1801)) ([838c523](https://github.com/headroomlabs-ai/headroom/commit/838c5234a877d4cf96f9e914cfd39d6d6addb211))
* **transforms:** pass through ragged tables instead of misaligning columns ([#1713](https://github.com/headroomlabs-ai/headroom/issues/1713)) ([c7665ca](https://github.com/headroomlabs-ai/headroom/commit/c7665ca08863da12dc9c656bd8bdf1f55c95bda7))
* use rtk native Cursor hook instead of injecting .cursorrules ([#756](https://github.com/headroomlabs-ai/headroom/issues/756)) ([#1846](https://github.com/headroomlabs-ai/headroom/issues/1846)) ([1573f1f](https://github.com/headroomlabs-ai/headroom/commit/1573f1fd0763408246f5dd0d7a92f32464f5fdbb))
* **wrap:** replace stale-proxy detection with Vite-style port fallback ([#1406](https://github.com/headroomlabs-ai/headroom/issues/1406)) ([b4205c6](https://github.com/headroomlabs-ai/headroom/commit/b4205c68e63e1e12e354508d8c3ac7d54781268b))


### Performance Improvements

* **proxy:** cap compression workers to CPU count ([#1803](https://github.com/headroomlabs-ai/headroom/issues/1803)) ([0a3851b](https://github.com/headroomlabs-ai/headroom/commit/0a3851b24004727e734b61af4e3f59ce3b0bfe10))
* **savings:** batch tracker persistence off the request hot path ([#1817](https://github.com/headroomlabs-ai/headroom/issues/1817)) ([451b9f0](https://github.com/headroomlabs-ai/headroom/commit/451b9f0867f1eb7cf3a1b479f67a4e3106f7e9be))


### Dependencies

* bump the cargo-minor-patch group across 1 directory with 7 updates ([#1909](https://github.com/headroomlabs-ai/headroom/issues/1909)) ([45601d9](https://github.com/headroomlabs-ai/headroom/commit/45601d93bcd92f7f66d4c3483d9f4512a10e933c))
* bump the npm-minor-patch group across 4 directories with 18 updates ([#1907](https://github.com/headroomlabs-ai/headroom/issues/1907)) ([8872bbc](https://github.com/headroomlabs-ai/headroom/commit/8872bbc6a2fa210e9f26d33d1ff8e019954bddd9))

## [0.29.0](https://github.com/headroomlabs-ai/headroom/compare/v0.28.0...v0.29.0) (2026-07-03)


### Features

* **proxy:** add --lossless no-CCR mode with format-native compaction ([#1721](https://github.com/headroomlabs-ai/headroom/issues/1721)) ([c75ebde](https://github.com/headroomlabs-ai/headroom/commit/c75ebdee6df9b1689a44ef321e36e8b360406ed7))
* **stats:** surface Codex WS compression counters in /stats summary ([#1680](https://github.com/headroomlabs-ai/headroom/issues/1680)) ([2fe19c3](https://github.com/headroomlabs-ai/headroom/commit/2fe19c39e40fc350af39f72e1a3bac28f9ce9874))
* **transforms:** adaptive Otsu KEEP/DROP threshold (+ land relevance split on main) ([#1726](https://github.com/headroomlabs-ai/headroom/issues/1726)) ([eea667a](https://github.com/headroomlabs-ai/headroom/commit/eea667a72019cc98401db9211907f67ddf45e7eb))


### Bug Fixes

* **bedrock:** fail fast when session-token auth lacks botocore ([#1553](https://github.com/headroomlabs-ai/headroom/issues/1553)) ([54cfa36](https://github.com/headroomlabs-ai/headroom/commit/54cfa361d308dec567615c346af7c77d52ebb676))
* **bedrock:** route ARNs via converse, named AWS profiles, and au. re… ([#1456](https://github.com/headroomlabs-ai/headroom/issues/1456)) ([7d87aa2](https://github.com/headroomlabs-ai/headroom/commit/7d87aa2f1cbd93c970a77c6dfec8df03603251b9))
* **ccr:** honor workspace dir for sqlite store ([#1564](https://github.com/headroomlabs-ai/headroom/issues/1564)) ([96e1dfe](https://github.com/headroomlabs-ai/headroom/commit/96e1dfe395a440f9e2dddf4589c4f6988f4ee4cd))
* **claude:** surface Remote Control proxy incompatibility ([#1610](https://github.com/headroomlabs-ai/headroom/issues/1610)) ([4bf7f92](https://github.com/headroomlabs-ai/headroom/commit/4bf7f92417a8799ab3ae5f61b7ea9e96c5605a4f))
* **cli:** stop advertising unwired compression tuning env vars in banner ([#1634](https://github.com/headroomlabs-ai/headroom/issues/1634)) ([d5bf98d](https://github.com/headroomlabs-ai/headroom/commit/d5bf98df31528dfd6c23ec45dbd3440efcb1cb75))
* **codex:** avoid duplicate headroom provider config ([#1431](https://github.com/headroomlabs-ai/headroom/issues/1431)) ([ddd4adf](https://github.com/headroomlabs-ai/headroom/commit/ddd4adf911ee2d7a5323657a771ea0162b5590c4))
* **compression:** reject lossy unmarked tool output in unit router path ([#1479](https://github.com/headroomlabs-ai/headroom/issues/1479)) ([de24cd5](https://github.com/headroomlabs-ai/headroom/commit/de24cd5fc0b894037c0481b5394e6851e87b3993))
* **cortex-code:** migrate to current Cortex REST API endpoints + add e2e benchmarks ([#1474](https://github.com/headroomlabs-ai/headroom/issues/1474)) ([f00ace6](https://github.com/headroomlabs-ai/headroom/commit/f00ace6da57aec2f68b833f42603ba3fda0f9110))
* **dashboard:** align token savings headline denominator ([#1653](https://github.com/headroomlabs-ai/headroom/issues/1653)) ([646e705](https://github.com/headroomlabs-ai/headroom/commit/646e7055143638ac4a2bc9980649fd046cea7840))
* **dashboard:** derive per-project setup URL from live origin ([#1511](https://github.com/headroomlabs-ai/headroom/issues/1511)) ([e035aef](https://github.com/headroomlabs-ai/headroom/commit/e035aefce23fd2e20afccf2659c1db613b05d8ca))
* **detection:** contain unidiff panic on orphaned +++ target line ([#1548](https://github.com/headroomlabs-ai/headroom/issues/1548)) ([e386c09](https://github.com/headroomlabs-ai/headroom/commit/e386c097d6d507aa311ca3a22725b226e9d7b223))
* **evals:** CJK-aware F1 tokenization + token estimation ([#1527](https://github.com/headroomlabs-ai/headroom/issues/1527)) ([99a8540](https://github.com/headroomlabs-ai/headroom/commit/99a8540e657445df3204f1d15e213262f4289a42))
* **install:** close parent log fd in start_detached_agent ([#1576](https://github.com/headroomlabs-ai/headroom/issues/1576)) ([816cb85](https://github.com/headroomlabs-ai/headroom/commit/816cb85fa8ee8d349fe673e7affd9a54acb1207d))
* **install:** use Windows-safe PID liveness probe in runtime_status ([#1544](https://github.com/headroomlabs-ai/headroom/issues/1544)) ([#1560](https://github.com/headroomlabs-ai/headroom/issues/1560)) ([6b227b9](https://github.com/headroomlabs-ai/headroom/commit/6b227b9c906d708923f39c0d877989a49942adae))
* **learn:** aggregate verbosity baselines across projects instead of overwriting ([#1288](https://github.com/headroomlabs-ai/headroom/issues/1288)) ([27a5468](https://github.com/headroomlabs-ai/headroom/commit/27a546834960b349e710a0b2e86ca3471523f34d))
* **mcp:** show lifetime totals and label rolling session scope in headroom_stats ([#1428](https://github.com/headroomlabs-ai/headroom/issues/1428)) ([1c0e152](https://github.com/headroomlabs-ai/headroom/commit/1c0e15243eda8f2dc868fe9ed4a08d944893686b))
* **memory:** cap local embedder CPU thread oversubscription ([#198](https://github.com/headroomlabs-ai/headroom/issues/198)) ([#1559](https://github.com/headroomlabs-ai/headroom/issues/1559)) ([b84afbf](https://github.com/headroomlabs-ai/headroom/commit/b84afbfb833999ddf164d324971bf6c11014a9d3))
* **memory:** singleflight LocalBackend init to stop cold-start races ([#1691](https://github.com/headroomlabs-ai/headroom/issues/1691)) ([bec47a1](https://github.com/headroomlabs-ai/headroom/commit/bec47a1898883919ad8c5ea41e3a7443a6890e7f))
* **openclaw:** detect uv-installed headroom binary in ~/.local/bin ([#1459](https://github.com/headroomlabs-ai/headroom/issues/1459)) ([adaeb88](https://github.com/headroomlabs-ai/headroom/commit/adaeb88a4d5512da5bd0bf58c1e3a276a5269d44))
* **opencode:** preserve custom OpenAI gateway paths ([#1596](https://github.com/headroomlabs-ai/headroom/issues/1596)) ([c19347c](https://github.com/headroomlabs-ai/headroom/commit/c19347c31046bf25baf9b1a816c9bede5d3ee807))
* **opencode:** route native providers + load transport plugin, fix Serena context ([#1573](https://github.com/headroomlabs-ai/headroom/issues/1573)) ([ad0034f](https://github.com/headroomlabs-ai/headroom/commit/ad0034f98191501c1a60d26383bc3ed9f6d532be))
* preserve anthropic passthrough tool order ([#1427](https://github.com/headroomlabs-ai/headroom/issues/1427)) ([a932247](https://github.com/headroomlabs-ai/headroom/commit/a9322477e33ec2c5ccd6442d3f72c17b7388c9e0))
* **proxy/auth:** match real Anthropic OAuth token prefix (sk-ant-oat) ([#1672](https://github.com/headroomlabs-ai/headroom/issues/1672)) ([8cddf9b](https://github.com/headroomlabs-ai/headroom/commit/8cddf9b58ea9ed11a0cd3532be6e779dffe57b55))
* **proxy:** expose persistent savings metrics ([#1647](https://github.com/headroomlabs-ai/headroom/issues/1647)) ([5fe4e7b](https://github.com/headroomlabs-ai/headroom/commit/5fe4e7b19530da0c2d07d17f20b18d79b6fab367))
* **proxy:** fail open when kompress saturation would exhaust pre-upstream budget ([#1430](https://github.com/headroomlabs-ai/headroom/issues/1430)) ([15ac650](https://github.com/headroomlabs-ai/headroom/commit/15ac650d409ea7def9e54d9962af1cfdc1f11f5d))
* **proxy:** handle streaming CCR retrieval ([#1451](https://github.com/headroomlabs-ai/headroom/issues/1451)) ([d337e3b](https://github.com/headroomlabs-ai/headroom/commit/d337e3b828ffc1f22cd5ca1884500b8905e9bd82))
* **proxy:** include system/tools/sampling in cache key ([#1473](https://github.com/headroomlabs-ai/headroom/issues/1473)) ([312129a](https://github.com/headroomlabs-ai/headroom/commit/312129a8e7465c97402ae45b9e9d51b7f4b5b0c7))
* **proxy:** preserve Responses passthrough bytes ([#1598](https://github.com/headroomlabs-ai/headroom/issues/1598)) ([2a34a82](https://github.com/headroomlabs-ai/headroom/commit/2a34a822f2a39da57fbd07575752888f5515f51a))
* **proxy:** strip Codex lite header on the HTTP /responses path ([#1663](https://github.com/headroomlabs-ai/headroom/issues/1663)) ([9fbd47b](https://github.com/headroomlabs-ai/headroom/commit/9fbd47ba6bdf38b618795541ee517b7e2fa2c6df))
* **proxy:** wire --compression-max-workers / HEADROOM_COMPRESSION_MAX_WORKERS ([#1632](https://github.com/headroomlabs-ai/headroom/issues/1632)) ([814ffa3](https://github.com/headroomlabs-ai/headroom/commit/814ffa36a4d1bb40165a630f96a855452037735e))
* **savings:** count cache-read tokens in input cost estimate ([#1429](https://github.com/headroomlabs-ai/headroom/issues/1429)) ([72ade37](https://github.com/headroomlabs-ai/headroom/commit/72ade3711211183b9134a46d9c5d45db6a87edc2))
* skip Magika backend on x86 CPUs without AVX2 ([#1162](https://github.com/headroomlabs-ai/headroom/issues/1162)) ([64783d8](https://github.com/headroomlabs-ai/headroom/commit/64783d8824e3c3afc43d9980573d9440693d0963))
* **transforms/content-router:** route grep/log output away from HTML extractor ([#1719](https://github.com/headroomlabs-ai/headroom/issues/1719)) ([0d18ef2](https://github.com/headroomlabs-ai/headroom/commit/0d18ef26f4d126f8eec9df1d34330a7129c4c63f))
* **transforms:** bound native content detection with a Windows watchdog ([#575](https://github.com/headroomlabs-ai/headroom/issues/575)) ([#1563](https://github.com/headroomlabs-ai/headroom/issues/1563)) ([95abca3](https://github.com/headroomlabs-ai/headroom/commit/95abca3abd69add5f075d241284b565e0014d5a4))
* Vertex AI support for Claude Code with ANTHROPIC_VERTEX_BASE_URL ([#1393](https://github.com/headroomlabs-ai/headroom/issues/1393)) ([cff7247](https://github.com/headroomlabs-ai/headroom/commit/cff7247efd6fbecc1c2e66280a4a9b6381d7b7a4))
* **wrap:** detach the shared proxy on Windows so it survives an ungraceful agent close ([#1464](https://github.com/headroomlabs-ai/headroom/issues/1464)) ([6cba441](https://github.com/headroomlabs-ai/headroom/commit/6cba4419d04bea79c1b44632a9288cde5b48bbce))
* **wrap:** preserve custom Vertex base URL ([#1477](https://github.com/headroomlabs-ai/headroom/issues/1477)) ([75427bb](https://github.com/headroomlabs-ai/headroom/commit/75427bbd4ad14fcb1b205f3253ec4e24ae1d2118))
* **wrap:** remove rtk instructions from Codex AGENTS.md on unwrap ([#1604](https://github.com/headroomlabs-ai/headroom/issues/1604)) ([c9d717c](https://github.com/headroomlabs-ai/headroom/commit/c9d717c13c7ae006178e49b6570f63b3f82de9a2))

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

* **wrap:** add `headroom wrap zcode` / `headroom unwrap zcode` for the ZCode
  desktop app (zcode.z.ai). Follows the Pattern-B (proxy-only watcher)
  approach: starts the proxy, injects RTK guidance into `AGENTS.md` at the
  project root, and prints the ZCode settings the user should configure
  (OpenAI and Anthropic base URLs). Auto-detects the enabled provider from
  `~/.zcode/v2/config.json` and configures the proxy upstream accordingly.
  Unwrap removes the injected RTK instructions and stops the proxy.
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

* **wrap:** check feature configuration before reusing persistent deployments. A persistent proxy started for one use case (e.g. `--backend anthropic`) would be silently reused for another (e.g. `--subscription --provider-type openai`) causing 401 auth failures because `_ensure_proxy()` only checked health + version, skipping the feature configuration check (memory, openai_api_url, learn, code_graph).
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
