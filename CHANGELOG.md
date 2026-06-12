# Changelog

All notable changes to Headroom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Unreleased

### Features

* **proxy:** per-project savings breakdown on the dashboard for all wrapped agents — Claude Code, Codex, aider, Copilot, and Cursor ([#802](https://github.com/chopratejas/headroom/issues/802)). `headroom wrap claude`/`codex` tag requests with an `X-Headroom-Project` header (launch-directory name); `wrap aider`/`copilot`/`cursor` — whose clients cannot send custom headers — use a `/p/<name>` base-URL prefix the proxy strips. Savings are aggregated per project (persisted, schema v3 with transparent v2 migration), exposed as `savings.per_project` in `/stats` and `projects` in `/stats-history`, and shown in a Per-Project Savings dashboard table.

### Features

* **memory:** opt-in Apple-GPU (MPS) embedding offload via `HEADROOM_EMBEDDER_RUNTIME=pytorch_mps`. When set (and Apple MPS is available), the memory embedder runs on the torch sentence-transformers backend on the Apple GPU instead of the default ONNX CPU embedder, freeing the CPU under load. If MPS or the dependencies are unavailable, Headroom logs a warning and uses the existing default embedder selection path (ONNX when available, then the pre-existing local fallback). MPS encode calls are serialized internally (torch-MPS is not thread-safe). Adds the new `[pytorch-mps]` extra (`pip install 'headroom-ai[pytorch-mps]'`). Default behavior is unchanged.

### Bug Fixes

* **ccr:** make retrieval store TTL configurable with `HEADROOM_CCR_TTL_SECONDS`, expose the effective TTL in `/v1/retrieve/stats`, and distinguish expired retrievals from missing hashes.


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
