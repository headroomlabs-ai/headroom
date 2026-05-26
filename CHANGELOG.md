# Changelog

All notable changes to Headroom will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.23.0](https://github.com/chopratejas/headroom/compare/headroom-ai-v0.22.3...headroom-ai-v0.23.0) (2026-05-26)


### Features

* add lean-ctx context tool support ([4b06179](https://github.com/chopratejas/headroom/commit/4b061792b29f776f36831530ba6fdbfd6d71ad0b))
* **ci:** X2 — PR-time release dry-run via path-filtered pull_request trigger ([7a06355](https://github.com/chopratejas/headroom/commit/7a063554ce03c04cbacd45d548f513f99dcde95a))
* **ci:** X2 — PR-time release dry-run via path-filtered pull_request trigger ([b9f84fa](https://github.com/chopratejas/headroom/commit/b9f84fa815a2696a85ed3f430d52b6ccaf36ac82))
* **init:** add -v/--verbose flag for debug diagnostics ([bb91cfe](https://github.com/chopratejas/headroom/commit/bb91cfe68853370a3723b7d4e6fe0cf88439a79b))
* **memory:** resolve Qdrant connection from HEADROOM_QDRANT_* env vars ([#31](https://github.com/chopratejas/headroom/issues/31)) ([d3c37d7](https://github.com/chopratejas/headroom/commit/d3c37d709841e19835608c15c09dac4debdcd018))
* **proxy:** add --exclude-tools flag + HEADROOM_EXCLUDE_TOOLS env var ([1058043](https://github.com/chopratejas/headroom/commit/10580439bb4227a3f6f375439b0baae60db2f288))
* **proxy:** add --exclude-tools flag + HEADROOM_EXCLUDE_TOOLS env var ([93cf8af](https://github.com/chopratejas/headroom/commit/93cf8af0e0c6b6b7fee0bcfe5b57a514c294a898))
* **proxy:** add client (harness) identification — per-harness analytics across every handler ([96a674f](https://github.com/chopratejas/headroom/commit/96a674f1b9c7d0b251478bb53541f8fbd6859e1f))
* **proxy:** support production multi-worker proxy startup + CLI/env tuning ([a1f1683](https://github.com/chopratejas/headroom/commit/a1f168369d4d7b9eb857d0f333eb308a816e1870))
* **python:** switch relevance scorer to fastembed (BAAI/bge-small-en-v1.5) ([fb9be13](https://github.com/chopratejas/headroom/commit/fb9be139a55067491297a0e38b4312a6c68e0c63))
* **rust:** adaptive_sizer port — Kneedle, SimHash, zlib validation ([8f810a0](https://github.com/chopratejas/headroom/commit/8f810a0c28045a356d0fd35790c468171c1d01a7))
* **rust:** axum reverse proxy skeleton + http catch-all (phase-1) ([128a910](https://github.com/chopratejas/headroom/commit/128a910ebb1b3f48e7e4fef18710c252c397696f))
* **rust:** diff_compressor port — byte-equal parity + sidecar stats ([5c3c9c4](https://github.com/chopratejas/headroom/commit/5c3c9c49f2f977a0c7c991f0a0c7c72d74d03b35))
* **rust:** HfTokenizer::from_pretrained — HuggingFace Hub auto-download ([a23ee8e](https://github.com/chopratejas/headroom/commit/a23ee8e70b4a8460f317e24965a4220c1f19e7b3))
* **rust:** hop-by-hop header filtering + X-Forwarded-* (phase-1) ([0493ebf](https://github.com/chopratejas/headroom/commit/0493ebf1fe2afafa8161b7a1439b4665ffcf3746))
* **rust:** pipeline traits + orchestrator skeleton (Phase 3g PR1) ([0397104](https://github.com/chopratejas/headroom/commit/0397104358ead96d8536d5cb9cbd0f38ea7cb9a2))
* **rust:** pipeline traits + orchestrator skeleton (Phase 3g PR1) ([caf8fef](https://github.com/chopratejas/headroom/commit/caf8fefdb368af9871fb908c1c3720213ce5e869))
* **rust:** port anchor_selector — dynamic position-based anchors ([abfeaf2](https://github.com/chopratejas/headroom/commit/abfeaf27fa04d63674f26c6b1433f261457adeb3))
* **rust:** port log_compressor + bug fixes (Phase 3e.5) ([2228d1b](https://github.com/chopratejas/headroom/commit/2228d1b0192f53177fbc36150c1072e0f0636b65))
* **rust:** port log_compressor to Rust + bug fixes (Phase 3e.5) ([4572030](https://github.com/chopratejas/headroom/commit/45720301e5a2a19e041f9d450ee84b2d827029ed))
* **rust:** port relevance/{base,bm25} — keyword scorer for SmartCrusher ([64f1092](https://github.com/chopratejas/headroom/commit/64f1092a890bcaee889fa5102b279a89740d4a80))
* **rust:** port relevance/{embedding,hybrid} — hybrid scorer with BM25 fallback ([0bae4ef](https://github.com/chopratejas/headroom/commit/0bae4ef004889612b7956173e8c9226ff8593f2d))
* **rust:** port search_compressor to Rust + signals trait consumer (Phase 3e.2) ([4d799d5](https://github.com/chopratejas/headroom/commit/4d799d52647d52268b643fbc7340e0dc252a9271))
* **rust:** port tag_protector to Rust + 5 bug fixes (Phase 3e.4) ([c89182f](https://github.com/chopratejas/headroom/commit/c89182f6cb17208e59925e8d5b427d34e81d7726))
* **rust:** port tag_protector to Rust + 5 bug fixes (Phase 3e.4) ([c9aaba3](https://github.com/chopratejas/headroom/commit/c9aaba3f5bd85f36be4ebadbe372e282e8f6218a))
* **rust:** pyo3 bridge for SmartCrusher ([5328d87](https://github.com/chopratejas/headroom/commit/5328d87b1eb70876d2df316180d3f7bdf7667fd7))
* **rust:** re-land search_compressor port (Phase 3e.2 redux) ([d6b9ccd](https://github.com/chopratejas/headroom/commit/d6b9ccdb4ebe52a14e37b7e5ae956e341be3d0c8))
* **rust:** real fastembed-rs EmbeddingScorer (BAAI/bge-small-en-v1.5) ([1945e5f](https://github.com/chopratejas/headroom/commit/1945e5f55bddc06dc67011d77d04dc6596f903ea))
* **rust:** retire python diff_compressor, ship rust-only via pyo3 ([f5f4654](https://github.com/chopratejas/headroom/commit/f5f465418bbe6b2fcffc7c28e15c5ee4dc850df1))
* **rust:** retire python smart_crusher, ship rust-only via pyo3 ([c765c53](https://github.com/chopratejas/headroom/commit/c765c53bf85161ff1893ab87fdadc7eaf9424891))
* **rust:** scaffold smart_crusher module + foundational helpers ([d219bee](https://github.com/chopratejas/headroom/commit/d219beecab0124bcd6392e246907021d631e7710))
* **rust:** scaffold workspace + parity harness (phase-0) ([0414cb7](https://github.com/chopratejas/headroom/commit/0414cb70e4b6e2620329ae7367eef99505a3f92b))
* **rust:** signals trait module + KeywordDetector (Phase 3e.1) ([cf3877d](https://github.com/chopratejas/headroom/commit/cf3877de385809bfc54d351e5889c95ea870d3b0))
* **rust:** signals trait module + KeywordDetector (Phase 3e.1) ([12c2665](https://github.com/chopratejas/headroom/commit/12c2665531483f8996a3647a28cf170f3bcfbc57))
* **rust:** smart_crusher analyzer helpers — id/score/outlier detectors + bug [#3](https://github.com/chopratejas/headroom/issues/3) fix ([f029055](https://github.com/chopratejas/headroom/commit/f029055229bbdae528d4d168bc9ef7c84c3ae715))
* **rust:** smart_crusher orchestration helpers — dedup / fill / prioritize ([abfe95d](https://github.com/chopratejas/headroom/commit/abfe95dcd9865bf45f954a70a4e3ff89c805f7ad))
* **rust:** smart_crusher planning layer — 4 plan methods + dispatcher ([a6ae8f9](https://github.com/chopratejas/headroom/commit/a6ae8f99be1694da62603d833a75793ff7d8762e))
* **rust:** smart_crusher SmartAnalyzer — field stats, change-points, crushability ([c26d225](https://github.com/chopratejas/headroom/commit/c26d2251959689b9aa36457c9a252d04ce88a534))
* **rust:** smart_crusher universal crushers — string, number, object ([9d515fb](https://github.com/chopratejas/headroom/commit/9d515fb78edf4d6dbdcfc91d9c276f8826ce3a79))
* **rust:** SmartCrusher extension surface — Constraint, Observer, Builder ([5b9cd5f](https://github.com/chopratejas/headroom/commit/5b9cd5ff1339d9acabaaff22079d6c00b7471d15))
* **rust:** SmartCrusher extension surface — Constraint, Observer, Builder ([57c16b9](https://github.com/chopratejas/headroom/commit/57c16b9e2b1038f8bcd9639cbaeff97f30091538))
* **rust:** SmartCrusher PR2 — lossless-first tabular compaction ([00e25d8](https://github.com/chopratejas/headroom/commit/00e25d847526b957c38fb120e86791c60e565b76))
* **rust:** SmartCrusher PR2 — lossless-first tabular compaction ([e640e18](https://github.com/chopratejas/headroom/commit/e640e18f374e74bfdfa3ec0d34bb3392c1e68c39))
* **rust:** SmartCrusher PR4 — lossless-first default + CCR-Dropped restoration ([8e09878](https://github.com/chopratejas/headroom/commit/8e098784a559c8904933111351abf4ee7854ce13))
* **rust:** SmartCrusher PR4 — lossless-first default + CCR-Dropped restoration ([1601591](https://github.com/chopratejas/headroom/commit/1601591900e2a719fa84a373b688d38e21e307ca))
* **rust:** SmartCrusher struct — execute_plan + crush_array + crush_mixed_array ([4c44b95](https://github.com/chopratejas/headroom/commit/4c44b95d9b14b26484ea7ca9f02204065509f122))
* **rust:** SmartCrusher::crush() top-level + recursive process_value ([4bac7d5](https://github.com/chopratejas/headroom/commit/4bac7d5b7570e43502fac209918beb9e89b3bc0c))
* **rust:** tokenizer crate with tiktoken-rs + HuggingFace + estimator ([9ce1c01](https://github.com/chopratejas/headroom/commit/9ce1c01b87a6474e5f7ac4abceff5624850e3796))
* **rust:** upstream health check endpoint (phase-1) ([e3b71b9](https://github.com/chopratejas/headroom/commit/e3b71b949f14339c5e87c880de39e36ee37a8a42))
* **rust:** websocket upgrade + bidirectional pump (phase-1) ([dd6ebef](https://github.com/chopratejas/headroom/commit/dd6ebef8cfc19d65aa9779a9b306c69228f59b3e))
* **telemetry:** surface per-strategy compression counters ([d11063e](https://github.com/chopratejas/headroom/commit/d11063eb3309214a794dd24e79ca7d788d87db28))
* **telemetry:** surface per-strategy compression counters ([e355476](https://github.com/chopratejas/headroom/commit/e3554767e645554bb5f4c20f83f5a815f3830341))


### Bug Fixes

* A0 — fail-loud rust core deployment smoke test ([00ab1ea](https://github.com/chopratejas/headroom/commit/00ab1ea74dd91c08c549d6272511762f939cc785))
* A2 — system prompt immutable; memory routes to live-zone tail; cache_aligner detector-only ([704fb2f](https://github.com/chopratejas/headroom/commit/704fb2f19df07ec88b19143c2ad423b0b335a7ab))
* A3 — byte-faithful Python forwarders; serialize canonical only when mutated ([f0dcc02](https://github.com/chopratejas/headroom/commit/f0dcc027754a68d84861c584e318f3596526410d))
* A5 — strip x-headroom-* from upstream-bound headers (P5-49) ([2e874c5](https://github.com/chopratejas/headroom/commit/2e874c5e3ee0755c63b9bd385ab9db63b2a14f9a))
* A6 — anthropic-beta and openai-beta deterministic merge + session-sticky ([aec5ba3](https://github.com/chopratejas/headroom/commit/aec5ba32538641dedf4f773ee81764eda878a318))
* A7 — memory tool injection session-sticky for both Anthropic and OpenAI ([8dcd474](https://github.com/chopratejas/headroom/commit/8dcd474aca4efd666909da8a7560d5db05d5f778))
* A8 — SSE delta arms, UTF-8 buffer, phase preservation, request-id, 413 ([148ded3](https://github.com/chopratejas/headroom/commit/148ded392aef60839a9d4b9dd591601496ef5312))
* A9 — tag protector discards wrap on placeholder loss ([6aacd48](https://github.com/chopratejas/headroom/commit/6aacd4805add7b6fe06d18134a2d4e1365960742))
* add Codex wire debug and WS usage metrics ([7c94e72](https://github.com/chopratejas/headroom/commit/7c94e728410d8784ecdcddb3d7c9bd6dc079e376))
* add Codex wire debug and WS usage metrics ([2b331e2](https://github.com/chopratejas/headroom/commit/2b331e297a02b0a685b9f3fdf2432eefc3cdcccd))
* add Kompress backend and thread controls ([a2ea964](https://github.com/chopratejas/headroom/commit/a2ea9648a4380fee97a4ad2b98ed0197d8ea03f0))
* align docs and prune dead compatibility surfaces ([359e8c9](https://github.com/chopratejas/headroom/commit/359e8c9a5b6f4dfc0c4126affd3308893da4c421))
* B1 — retire ICM, RollingWindow, scoring, relevance + dependents ([967b0db](https://github.com/chopratejas/headroom/commit/967b0db4392d358a68003cb3e0ea1af88e1e54b8))
* B2 — live-zone block dispatcher skeleton ([e190544](https://github.com/chopratejas/headroom/commit/e190544c77573e4441fd50815b7e67319df8fa44))
* B3 — wire type-aware compressors into live-zone dispatcher ([2b55050](https://github.com/chopratejas/headroom/commit/2b55050a654b78d43b78609fc5b17b27e732f6b0))
* B4 — token validation gate + per-content-type byte thresholds ([b3b3fef](https://github.com/chopratejas/headroom/commit/b3b3feff6f29b9c3eeba86c66ef2a43e5591e932))
* B5 — TOIN observation-only refactor + per-tenant aggregation key ([6819b7e](https://github.com/chopratejas/headroom/commit/6819b7e5e5e820a342fce7afc448a6ac953cf82e))
* B6 — memory injection moves to live-zone user-tail ([2ee0577](https://github.com/chopratejas/headroom/commit/2ee05774b92ddd759459bfa0aab684e41f143f5a))
* B7 — CCR hardening: persistent backends + always-on tool ([00902b8](https://github.com/chopratejas/headroom/commit/00902b8feaf4de1f173e051fcbb265a9ac9b56c1))
* batch and guard Kompress inference ([b946a44](https://github.com/chopratejas/headroom/commit/b946a44c44a74ec601fd9fb4b97a4199b6c9b792))
* **bedrock:** apply rustfmt to header_value_preview tests ([42ff8af](https://github.com/chopratejas/headroom/commit/42ff8afa90c29a11079256bbe2d4a96878d708c6))
* **bedrock:** use floor_char_boundary to avoid UTF-8 slice panic in header preview ([399cb32](https://github.com/chopratejas/headroom/commit/399cb32fa475ebbc0e6790024f77675d6b72f4fd))
* **bedrock:** use floor_char_boundary to avoid UTF-8 slice panic in header preview ([82468e4](https://github.com/chopratejas/headroom/commit/82468e4e99e6013ab9a44df2e2f12c9352bbc18f)), closes [#415](https://github.com/chopratejas/headroom/issues/415)
* **bedrock:** use is_char_boundary loop instead of floor_char_boundary (MSRV 1.80) ([b784f40](https://github.com/chopratejas/headroom/commit/b784f400c109b51352825d68341a2801a29123d6))
* **build:** shrink Rust extension wheels — strip + thin-LTO + single codegen unit ([d73cbd6](https://github.com/chopratejas/headroom/commit/d73cbd6b0aea20a0e99d29e6c775f26d27f59eb5))
* **build:** shrink Rust extension wheels (strip + thin-LTO + single codegen unit) ([ec8e208](https://github.com/chopratejas/headroom/commit/ec8e2080f3231353f11589016a191dd02e100b98))
* C1 — byte-level SSE parser + state machines ([f34e0c2](https://github.com/chopratejas/headroom/commit/f34e0c2e8be92d8b7bc6bce2d072bf8280efd7f4))
* C1 — byte-level SSE parser + state machines ([ddc6f6c](https://github.com/chopratejas/headroom/commit/ddc6f6ceb0359a7a89b470dc6330543f0cd93f31))
* C2 — /v1/chat/completions Rust handler + OpenAI live-zone ([f9b4491](https://github.com/chopratejas/headroom/commit/f9b449196c61ecf1a7947585bd43b6b75daf3e37))
* C2 — /v1/chat/completions Rust handler + OpenAI live-zone ([fe00dc0](https://github.com/chopratejas/headroom/commit/fe00dc006aec32f65e1aa6bcf8b2776f37230f4e))
* C3 — /v1/responses Rust HTTP handler + per-item-type passthrough ([70cad6c](https://github.com/chopratejas/headroom/commit/70cad6c1c6cc13e6b0794a16ad694ea98979fe9a))
* C3 — /v1/responses Rust HTTP handler + per-item-type passthrough ([57c3e38](https://github.com/chopratejas/headroom/commit/57c3e38cb3ccb65744c48927a72238315d154e61))
* C4 — /v1/responses streaming + Conversations API in Rust ([1352f62](https://github.com/chopratejas/headroom/commit/1352f621fce223fbcc8cfac0b796d1577a028407))
* C4 — /v1/responses streaming + Conversations API in Rust ([866d346](https://github.com/chopratejas/headroom/commit/866d346bd041a581fe39a68e6570b74e2d538fd1))
* **ci:** A0 verify must run outside /build cwd ([6b15802](https://github.com/chopratejas/headroom/commit/6b15802c0b7f946f5b62d4505c731da3a452821f))
* **ci:** add minimal-privilege permissions blocks to release.yml jobs ([d289c0d](https://github.com/chopratejas/headroom/commit/d289c0d4334fc6cb5331d3d2e9e16799e3429193))
* **ci:** docker per-arch bake needs explicit image name in output ([c81755c](https://github.com/chopratejas/headroom/commit/c81755c965bd5d57533d75939e1017ad0ff60435))
* **ci:** docker per-arch bake needs explicit image name in output ([8f6bc58](https://github.com/chopratejas/headroom/commit/8f6bc5865c33e5bf7e9b5f59986f24d7d98096df))
* **ci:** force-link glibc shim with -Wl,-u for aarch64 ([39cb7c4](https://github.com/chopratejas/headroom/commit/39cb7c4dc4102a9e14ecaf9c35deb6bd3186830b))
* **ci:** force-link glibc shim with -Wl,-u so aarch64 wheel includes it ([820e66c](https://github.com/chopratejas/headroom/commit/820e66cae65723a7691f7b79707a6f077ac266f0))
* **ci:** glibc shim — drop alias attribute, forward-declare strtol ([af0a960](https://github.com/chopratejas/headroom/commit/af0a9604ee5503d1e7b89e8cf41224a306568fce))
* **ci:** glibc shim — drop alias attribute, forward-declare strtol ([6b15acc](https://github.com/chopratejas/headroom/commit/6b15acc3b572381260e9c3f068bc5c43bb9eda18))
* **ci:** include NOTICE in sdist + assert License-File metadata matches tarball ([fc6b0a0](https://github.com/chopratejas/headroom/commit/fc6b0a0917063df89058d4d01292f43667656c8f))
* **ci:** include NOTICE in sdist + assert License-File metadata matches tarball ([183d51c](https://github.com/chopratejas/headroom/commit/183d51c8a860a5ca8d00c091df89c398ffb0425e))
* **ci:** install patchelf for maturin wheel-link repair ([4bf559d](https://github.com/chopratejas/headroom/commit/4bf559d5b524e618f3585845bf72c58f8b5f49fd))
* **ci:** maturin manifest-path is Cargo.toml not pyproject.toml ([1bbbf96](https://github.com/chopratejas/headroom/commit/1bbbf96700366fa0728ae1703338d240b890af5c))
* **ci:** multi-stage manylinux build for e2e dockerfiles + release workflow test ([b31a34b](https://github.com/chopratejas/headroom/commit/b31a34b4ac00f905c9e319428261be9316ef1c51))
* **ci:** pin dtolnay/rust-toolchain to 1.95.0 to match rust-toolchain.toml ([5690589](https://github.com/chopratejas/headroom/commit/5690589b3ea4cd25b9ccad1b9f54c6af88d5e7fb))
* **ci:** pin dtolnay/rust-toolchain to 1.95.0 to match rust-toolchain.toml ([beec078](https://github.com/chopratejas/headroom/commit/beec0789ed9381f67267e5bf86fede9342d5f3af))
* **ci:** pin public PyPI in pyproject.toml + scrub Netflix URLs from uv.lock ([e325d1b](https://github.com/chopratejas/headroom/commit/e325d1b86618273abb4177cce672645e0b6a8bd8))
* **ci:** pre-install rustfmt+clippy components in all Dockerfiles ([2ae5772](https://github.com/chopratejas/headroom/commit/2ae57725e75ccc5eb57d95e2611bee11f3b0d515))
* **ci:** pypi publish skip-existing to unblock idempotent re-runs ([f94a9d2](https://github.com/chopratejas/headroom/commit/f94a9d22bace56af115bb48c38f4539a9bb4a5f4))
* **ci:** pypi publish skip-existing to unblock idempotent re-runs ([6a191b4](https://github.com/chopratejas/headroom/commit/6a191b405b5c9bdbcdab4eb1250bea9fa295f18f))
* **ci:** rebuild sdist on the renamed wheel-matrix host ([31bd9f7](https://github.com/chopratejas/headroom/commit/31bd9f78e23a6de20cdc318dc26cd774a62d073b))
* **ci:** rebuild sdist on the renamed wheel-matrix host ([75576da](https://github.com/chopratejas/headroom/commit/75576dae2bca18610b870b2706c6b54e007dda13))
* **ci:** regenerate uv.lock against public PyPI (was Netflix-internal) ([ce3b2f0](https://github.com/chopratejas/headroom/commit/ce3b2f0b0ba07bb43012f162e75ccbdc2944a95d))
* **ci:** reorder Dockerfile so wheel installs before headroom-ai ([f1aa12c](https://github.com/chopratejas/headroom/commit/f1aa12cebfbd5ae4657133ec11efecf7ad6f433c))
* **ci:** replace ls with find in smoke-import wheel discovery ([3e78421](https://github.com/chopratejas/headroom/commit/3e784212673a6ffc1b82bcb3f478998c6ff8433b))
* **ci:** rustls-everywhere — eliminate openssl-sys from build tree ([6f2c0a8](https://github.com/chopratejas/headroom/commit/6f2c0a84003a0d4cee0408bee7a94b8cbb6502ae))
* **ci:** rustls-everywhere — eliminate openssl-sys from build tree (kills the wheel-build cascade) ([e731f87](https://github.com/chopratejas/headroom/commit/e731f87c5d6d790b1d94298a36de7658b25f6b34))
* **ci:** skip smoke OpenAI eval cleanly when OPENAI_API_KEY secret unset ([a281de6](https://github.com/chopratejas/headroom/commit/a281de6bb0279c05a2b44cc868bde9a0428ba82f))
* **ci:** smoke-import wheels on customer-representative envs before publish (X1) ([596212b](https://github.com/chopratejas/headroom/commit/596212b428ba2019885b49381ffdf9cb309ec8e3))
* **ci:** stage smoke-import script as host file (broke main post-[#387](https://github.com/chopratejas/headroom/issues/387)) ([7fe2d1e](https://github.com/chopratejas/headroom/commit/7fe2d1e5b632c3e5424f5364baef0ef8af6510a4))
* **ci:** stage smoke-import script as host file (broke main) ([e536d9b](https://github.com/chopratejas/headroom/commit/e536d9b1801652ff6d4b9bce9683f155029b23c7))
* **ci:** switch e2e runtime to python:3.11-slim (trixie, glibc 2.41) ([73a4782](https://github.com/chopratejas/headroom/commit/73a4782917cf393bfb71e240cc66e6629ef31702))
* **ci:** tolerate null enable_ref_tags on direct release events ([ab46594](https://github.com/chopratejas/headroom/commit/ab465948f0a76e06088e5494489e3ca2a53ad963))
* **ci:** unblock A0 Docker e2e — install pkg-config + opt out wrap-e2e ([55dfc19](https://github.com/chopratejas/headroom/commit/55dfc19e1debd9a5f4bb3ec2a632d606452e2bd7))
* **ci:** unbreak release pipeline — wheel openssl + dead npm artifact downloads ([75fa4a4](https://github.com/chopratejas/headroom/commit/75fa4a42f5cec62c969a8ef242116e38474189eb))
* **ci:** unbreak release pipeline — wheel openssl + dead npm artifact downloads ([7ba47e2](https://github.com/chopratejas/headroom/commit/7ba47e257b599856e011a1dc7abcd3c0dfb69d77))
* **ci:** update sdist license-packaging invariant test to match new shape ([cd89a82](https://github.com/chopratejas/headroom/commit/cd89a8297ae55fe26343fc101b695953f07969de))
* **ci:** update sdist license-packaging invariant test to match new shape ([37c0df2](https://github.com/chopratejas/headroom/commit/37c0df23b6cc7bf3e1fbc4f7ad67499a61eb539e))
* **ci:** update tests to assert absence of requires_openai_auth (bug 3, [#406](https://github.com/chopratejas/headroom/issues/406)) ([4071d57](https://github.com/chopratejas/headroom/commit/4071d571345c5a88df56eac5ad975200831e9b7d))
* **ci:** use ghcr devcontainer rust feature instead of manual rustup install ([9ce61af](https://github.com/chopratejas/headroom/commit/9ce61afcd4d8b48a5e3dac32cd0d89ee599532db))
* **ci:** vendor OpenSSL via cargo + drop x86_64 macOS from wheel matrix ([20e482d](https://github.com/chopratejas/headroom/commit/20e482da8fc16cc6b6ae022fbdd709a98ce70e4a))
* **ci:** vendor OpenSSL via cargo + drop x86_64 macOS from wheel matrix ([1314842](https://github.com/chopratejas/headroom/commit/1314842b1901b7465bb8d2057a3123402c208fc0))
* **ci:** vendored OpenSSL must live in headroom-py, not headroom-proxy ([f34f95a](https://github.com/chopratejas/headroom/commit/f34f95afe9170adcab3701e861d740ec141fd624))
* **ci:** vendored OpenSSL must live in headroom-py, not headroom-proxy ([a5c7f6f](https://github.com/chopratejas/headroom/commit/a5c7f6fed9af0f96ba8f02dbf161d3d969169d32))
* **ci:** wheel before-script must work on Debian aarch64-cross container ([1323830](https://github.com/chopratejas/headroom/commit/1323830f70acdc754079d6c11fbb18146de40d55))
* **ci:** wheel build before-script-linux must work on Debian aarch64-cross ([cf4ea02](https://github.com/chopratejas/headroom/commit/cf4ea0243295f2703093e0b72d3f81eb75c4e81a))
* **ci:** X1 — smoke-import wheels on customer-representative envs before publish ([4745219](https://github.com/chopratejas/headroom/commit/4745219901003be6607df2219e4ce672187b8f27))
* **ci:** yarnpkg GPG + YAML colon syntax in refactor ([86177da](https://github.com/chopratejas/headroom/commit/86177da871cc9e39999353cdcee67c1a8d2ab887))
* clear CI mypy + rust test failures introduced in eaf5980 ([17ffae0](https://github.com/chopratejas/headroom/commit/17ffae0cd8e4498bf60f4fa5a36521905f9fb151))
* **cli:** G1 remediation — non-string clobber, per-model systemMessage, openhands gate ([ea1976e](https://github.com/chopratejas/headroom/commit/ea1976e37a5147ecf37dbf5ffe4af5c2f2d1be6a))
* **cli:** proxy/perf/wrap UX cleanup + perf --hours correctness ([9ff28e9](https://github.com/chopratejas/headroom/commit/9ff28e9803eaba3c1be9c20924d7b34d98f507b0))
* **cli:** proxy/perf/wrap UX cleanup + perf --hours correctness ([265554d](https://github.com/chopratejas/headroom/commit/265554d4adee7fbf03234a6de832293dec95f811))
* **cli:** resolve duplicate --code-aware flag breaking proxy import ([5dd2ac5](https://github.com/chopratejas/headroom/commit/5dd2ac5e516b9f429648ca5a48be9bb1a6bacb0b))
* **cli:** wrap CLI breadth — cline, continue, goose, openhands ([8625f80](https://github.com/chopratejas/headroom/commit/8625f8075ed75d2a002f6ba357697de0fa1ec434))
* **cli:** wrap subcommands for cline, continue, goose, openhands ([c375fa1](https://github.com/chopratejas/headroom/commit/c375fa156dd0434256805f274c07be4f45db9814))
* Codex ws cancel logging ([73b0e56](https://github.com/chopratejas/headroom/commit/73b0e56e970c6f805a1d174f50b886000372cdb2))
* **codex:** bug 3 — strip requires_openai_auth, restore consistent openai_base_url injection ([6c4ddc8](https://github.com/chopratejas/headroom/commit/6c4ddc824e074245ffd6ea8002a068cb108f27fb))
* **codex:** drop env_key from provider blocks to preserve subscription auth ([32f499c](https://github.com/chopratejas/headroom/commit/32f499cbba4c370865105a55748a365294ec5815))
* **codex:** inject openai_base_url in init and persistent-install paths ([bf1e31b](https://github.com/chopratejas/headroom/commit/bf1e31b27c64ad9400d2997ce1f70692ebb408d0))
* **codex:** restore openai_base_url top-level injection for subscription routing ([d54c5b6](https://github.com/chopratejas/headroom/commit/d54c5b6a5840fabf0d28079fe67fb0347f016fbb))
* **codex:** strip requires_openai_auth and openai_base_url injection (bug 3, [#406](https://github.com/chopratejas/headroom/issues/406)) ([09b8510](https://github.com/chopratejas/headroom/commit/09b851001bf6a2e6ccc34478e42f54c87ddaf5d3))
* complete /v1/responses compression telemetry, multi-frame WS, frozen-count cap ([89f7b6c](https://github.com/chopratejas/headroom/commit/89f7b6c2dd0f3503312490b2052f534e76c01ac3))
* Compress Codex Responses payloads ([d90d2ca](https://github.com/chopratejas/headroom/commit/d90d2caed3b1c3199dd89714a7f0a330b23db59f))
* **content-router,proxy:** cache-safe text-block compression and online streaming usage ([ecb3e00](https://github.com/chopratejas/headroom/commit/ecb3e00451f5fd62a2be9d924996c2f075a3de2d))
* **content-router,proxy:** cache-safe text-block compression and online streaming usage ([79baee0](https://github.com/chopratejas/headroom/commit/79baee082f894cf0c467468eed1ae2fed04f37e6))
* **content-router,proxy:** compress text blocks + close DeepSeek metrics gaps ([d322e6d](https://github.com/chopratejas/headroom/commit/d322e6df1fc0f949469242488c5c0080beb206a1))
* **content-router,proxy:** compress text blocks + close DeepSeek metrics gaps ([d955abb](https://github.com/chopratejas/headroom/commit/d955abb1d65a14e07b5c6a915cafc12ddb12ace7))
* **core:** expose compress_openai_responses_live_zone via PyO3 (hot-fix c1/2) ([c48735d](https://github.com/chopratejas/headroom/commit/c48735d0298b94d25d40397ea64a2d66d1e557d0))
* **core:** F2.2 c1/3 — extend CompressionPolicy with three per-mode tuning fields ([797dc63](https://github.com/chopratejas/headroom/commit/797dc63da747dd8594958709271abd658de4a5c0))
* **core:** introduce CompressionPolicy struct + auth_mode mapping (F2.1 c1/6) ([8376630](https://github.com/chopratejas/headroom/commit/837663006254cde327e9c7f71b2ea5682f9b902e))
* **crusher:** bridge SmartCrusher row-drop hash to Python compression_store ([#389](https://github.com/chopratejas/headroom/issues/389)) ([e664f11](https://github.com/chopratejas/headroom/commit/e664f11260e997c9e643feec8b2cff27f127660f))
* **crusher:** bridge SmartCrusher row-drop hash to Python compression_store ([#389](https://github.com/chopratejas/headroom/issues/389)) ([a40d4a5](https://github.com/chopratejas/headroom/commit/a40d4a5f6d3183c94941a7ffe8c28bcb5d769792))
* **crusher:** shim __libc_single_threaded for glibc &lt; 2.32 + extend audit ([19a43f0](https://github.com/chopratejas/headroom/commit/19a43f0d52a0b406b1e0cae2804fe0e204b4dcba))
* **crusher:** shim __libc_single_threaded for glibc &lt; 2.32 + extend audit ([17d6207](https://github.com/chopratejas/headroom/commit/17d6207bf544ddb670236be67d4530c8913e735f))
* **crusher:** switch compression_store cache key from MD5 to SHA-256 (CodeQL [#395](https://github.com/chopratejas/headroom/issues/395)) ([af9e628](https://github.com/chopratejas/headroom/commit/af9e6282f89ff4fb2e0052fd58314d407997b7e6))
* **diff_compressor:** four silent information-loss paths in Python AND Rust ([6d47a0c](https://github.com/chopratejas/headroom/commit/6d47a0cd0087c5d824d0b66466b09e22a1d4a0f5))
* **diff:** close ContentRouter routing gaps for merge diffs and long preambles ([48c1324](https://github.com/chopratejas/headroom/commit/48c13245c5377aad5d1814c1089ba489ea55d27b))
* **e2e:** pin marketplace source via env var in init Dockerfile ([9ae696e](https://github.com/chopratejas/headroom/commit/9ae696eb752f1fc58b3465c4cc8a3d4dd6add2a0))
* expose code-aware flag ([7d99a71](https://github.com/chopratejas/headroom/commit/7d99a71285b3b3ec5c475c0676aa1a4a0887fbe7))
* expose compression latency bottlenecks ([4e146b0](https://github.com/chopratejas/headroom/commit/4e146b0b5a12d1040bb8dddd88b3445fc6b2af87))
* expose compression latency bottlenecks ([e9cae01](https://github.com/chopratejas/headroom/commit/e9cae0131bfe0c3b6db9096b2c5604259bda673d))
* format Codex compression changes ([03d12fc](https://github.com/chopratejas/headroom/commit/03d12fc14039d64fc6b53507a28308177661ab41))
* format Kompress tests for ruff ([fc0cba7](https://github.com/chopratejas/headroom/commit/fc0cba7b48cf7cefbef1968f57527bdf5c2fe652))
* harden learn path handling across platforms ([9577ef8](https://github.com/chopratejas/headroom/commit/9577ef811b3f078959fe821b39304977877f6f09))
* harden learn path handling across platforms ([5ceca13](https://github.com/chopratejas/headroom/commit/5ceca13c656165209dacc35d1dc2cc0aebd1afe1))
* harden release gating and clarify pipx compatibility ([9492386](https://github.com/chopratejas/headroom/commit/9492386398408ae6652aab58966fb16f4f26c2c8))
* harden release gating and clarify pipx Python compatibility ([a7e1f93](https://github.com/chopratejas/headroom/commit/a7e1f931a5511294c1e3daa6992f72d195373677))
* **init:** guide users when no agents are auto-detected ([4c06231](https://github.com/chopratejas/headroom/commit/4c062319f01e406563e649e03a6e3583f5154cdf))
* **init:** guide users when no agents are auto-detected + expand e2e ([7b99b05](https://github.com/chopratejas/headroom/commit/7b99b05e0db8c109d65009af16ffb601a2e9b962))
* inject openai_base_url so Codex subscription (ChatGPT plan) routes through proxy ([f0b40f1](https://github.com/chopratejas/headroom/commit/f0b40f11d5dfdfa41debb069478f9ad283d753d6))
* inject openai_base_url so Codex subscription (ChatGPT plan) routes through proxy ([5c7a5b4](https://github.com/chopratejas/headroom/commit/5c7a5b4857e4259ff16d272b1ba0308f3ac45282))
* integrate B6+B7 — fix cross-test contamination + injector mock parity ([2fb905f](https://github.com/chopratejas/headroom/commit/2fb905fdb016d354fa4938bd9fcb8e2654865998))
* **integrations:** filter CCR-dropped sentinel in test iteration ([b8fc7ee](https://github.com/chopratejas/headroom/commit/b8fc7eee197cc14a5d7709b19f641415e54b8f84))
* **integrations:** pin MCP server + LangChain evals to lossy+CCR path ([1688003](https://github.com/chopratejas/headroom/commit/168800329b7c93df5da6fab2e41b1958d018e8a9))
* **learn:** show prior patterns block to LLM to prevent dangling cross-references ([8d4f21b](https://github.com/chopratejas/headroom/commit/8d4f21b3c66ff2c6329f4f8f34f0e71b1525189f))
* log Codex ws cancellations safely ([62a1f23](https://github.com/chopratejas/headroom/commit/62a1f23b8864f9b0a82dbab6225514f73b7ada9d))
* make proxy upgrades version-aware ([2732ae7](https://github.com/chopratejas/headroom/commit/2732ae7d820451bafdefceb76513c4eba109b29b))
* make proxy upgrades version-aware ([ea1f608](https://github.com/chopratejas/headroom/commit/ea1f608e79d8809ff4d7d2428972fdb633a01918))
* **mcp:** auto-register headroom MCP server in wrap claude/codex and init -g ([d9d8972](https://github.com/chopratejas/headroom/commit/d9d8972ac4f51c0f018c470b8dba337cb959074d))
* **memory:** batch onnx embeddings and sqlite-vec ops ([32152f5](https://github.com/chopratejas/headroom/commit/32152f520290b54475be8408a5b2a0b24bbf185a))
* **memory:** batch onnx embeddings and sqlite-vec ops ([f5cea7c](https://github.com/chopratejas/headroom/commit/f5cea7c51efe24bf9404d8039ffda6f2eb9dd325)), closes [#240](https://github.com/chopratejas/headroom/issues/240)
* **memory:** collapse and decay error_recovery patterns in MEMORY.md ([6dede0c](https://github.com/chopratejas/headroom/commit/6dede0c2b44c01c3bdf65d2f705718a6bb7083d9))
* **memory:** collapse and decay error_recovery patterns in MEMORY.md ([879064f](https://github.com/chopratejas/headroom/commit/879064fea549538aeb8617e0b369d198dc043988))
* **memory:** drop regex-based pref extraction; filter system-reminder noise (refs [#464](https://github.com/chopratejas/headroom/issues/464)) ([1139b21](https://github.com/chopratejas/headroom/commit/1139b21f2a9a39803bcc11b981c73890980508ee))
* **memory:** expose memory IDs in auto-tail + memory_list tool + ID-usage guidance ([f844f64](https://github.com/chopratejas/headroom/commit/f844f64840491d9a838be4b00a4ca6d9ff97adba))
* **memory:** expose memory IDs in auto-tail + memory_list tool + ID-usage guidance ([c62d45e](https://github.com/chopratejas/headroom/commit/c62d45eea826c661aa8ebf1f2c8aba8408ea6109))
* **memory:** make numpy import optional for proxy boot path ([276e92e](https://github.com/chopratejas/headroom/commit/276e92e05ca2ac2745e948674141e6744e172eb3))
* **memory:** make numpy import optional for proxy boot path ([d236b4b](https://github.com/chopratejas/headroom/commit/d236b4befd79ae36feee94434c4548bce3d7c557)), closes [#332](https://github.com/chopratejas/headroom/issues/332)
* **memory:** per-project storage so projects can no longer bleed memories (refs [#462](https://github.com/chopratejas/headroom/issues/462)) ([f8ffdbf](https://github.com/chopratejas/headroom/commit/f8ffdbfe7369d49c23c67ca9a3c4668238aab508))
* **memory:** resolve Qdrant connection from HEADROOM_QDRANT_* env vars ([#31](https://github.com/chopratejas/headroom/issues/31)) ([7c1cff2](https://github.com/chopratejas/headroom/commit/7c1cff226f78e533106e52bcf86f1289b7f7a496))
* **memory:** traffic_learner indexes system-reminder fragments as user preferences (refs [#464](https://github.com/chopratejas/headroom/issues/464)) ([0be0eed](https://github.com/chopratejas/headroom/commit/0be0eede9e09466e4ec0c0ee10b1d0aa317e0e73))
* narrow compressed type for mypy 1.14 in ContentRouter.compress ([05fe6c0](https://github.com/chopratejas/headroom/commit/05fe6c010222921103651ebc3f9f4715a0418780))
* **observability:** G3 remediation — bound cardinality + wire dead metrics ([2a717a9](https://github.com/chopratejas/headroom/commit/2a717a993ee99f9401f5cdf78a23dcecd7cb1a51))
* **observability:** RTK metrics + Rust observability (Phase H blocker) ([b36ad9f](https://github.com/chopratejas/headroom/commit/b36ad9fe1c6a488eb9ffbf0e8b38d989278cf8ef))
* **observability:** wire Phase G PR-G3 RTK + proxy metrics (H-blocker) ([5f264a5](https://github.com/chopratejas/headroom/commit/5f264a53292e292c9c56b837c2750d1a415b1ea9))
* per-project memory storage so projects can no longer bleed memories (GH [#462](https://github.com/chopratejas/headroom/issues/462)) ([7694f05](https://github.com/chopratejas/headroom/commit/7694f050fe3b4d80c82903f17f38a93b4b3d670e))
* Phase A+B realignment — cache safety + live-zone-only compression + production hotfixes ([9266ea7](https://github.com/chopratejas/headroom/commit/9266ea711153c2bb7487b0c73dd9f7bfe1fb22c3))
* populate Codex WS dashboard performance metrics ([4279a7f](https://github.com/chopratejas/headroom/commit/4279a7f35349bde2b0361e9fe6e9f3a75adbf5de))
* PR [#281](https://github.com/chopratejas/headroom/issues/281) — synthesize 5h subscription window after Anthropic reset ([27adb1d](https://github.com/chopratejas/headroom/commit/27adb1da57c4cd45804aaea3942362e3a9d09289))
* PR [#281](https://github.com/chopratejas/headroom/issues/281) — synthesize 5h subscription window after Anthropic reset (no extra polling) ([0b77955](https://github.com/chopratejas/headroom/commit/0b77955ecb5b64ab4f8224f003dc532cf71d5909))
* PR [#372](https://github.com/chopratejas/headroom/issues/372) — restore [image] extra on Python 3.13 via rapidocr 3.x adapter ([486372b](https://github.com/chopratejas/headroom/commit/486372b6445bd217a6306159bb7210173d0578ec))
* PR [#372](https://github.com/chopratejas/headroom/issues/372) — restore [image] extra on Python 3.13 via rapidocr 3.x adapter ([b154e17](https://github.com/chopratejas/headroom/commit/b154e178533e51cf95fb4410812ee01fbe3975df))
* PR-C5 retire responses_converter.py — Rust owns /v1/responses ([76c113d](https://github.com/chopratejas/headroom/commit/76c113dd4a2ed210cac301b5bd46961dd34f1d0d))
* PR-C5 retire responses_converter.py — Rust owns /v1/responses ([221109d](https://github.com/chopratejas/headroom/commit/221109d95e6f18041fe664144710ae32f00401ad))
* PR-D1 native Bedrock InvokeModel route + SigV4 ([dd1dadf](https://github.com/chopratejas/headroom/commit/dd1dadfe8a4b8041cd9c6588a5b075fc4d24b4f0))
* PR-D2 Bedrock streaming via binary EventStream ([32a1fd4](https://github.com/chopratejas/headroom/commit/32a1fd4fe0dc5003ee479229890b52547b455ba8))
* PR-D3 Bedrock observability + auth-mode integration (Phase D close) ([0c46c7e](https://github.com/chopratejas/headroom/commit/0c46c7e01070a3efa2fcfb420665fa0d420eeb58))
* PR-D4 native Vertex publisher path + ADC ([d3c70ac](https://github.com/chopratejas/headroom/commit/d3c70acea5d2d893e6490b293a6b2f931a57a1d3))
* PR-E1 + PR-E2 tool array sort + schema-key sort (Phase E) ([c8e0b36](https://github.com/chopratejas/headroom/commit/c8e0b367572d6c38dc4807955599e32b90248001))
* PR-E1 tool array deterministic sort (Phase E) ([4a3b76b](https://github.com/chopratejas/headroom/commit/4a3b76bcc879b3ba4a6a29096b5b50cb87d34f40))
* PR-E2 recursive JSON Schema key sort (Phase E) ([9112fed](https://github.com/chopratejas/headroom/commit/9112fed9379cfe43d31e935c9ac855116abd60f9))
* PR-E3 Anthropic cache_control auto-placement (Phase E) ([397c805](https://github.com/chopratejas/headroom/commit/397c805698d01da20256d651a92a17429327cf6a))
* PR-E3 Anthropic cache_control auto-placement (Phase E) ([8672d5c](https://github.com/chopratejas/headroom/commit/8672d5c32680c790bfb6c431c2d1a854fac6566b))
* PR-E4 OpenAI prompt_cache_key auto-injection (Phase E) ([2fc73d0](https://github.com/chopratejas/headroom/commit/2fc73d0f73241a937304a8b8d4e11c550945e6df))
* PR-E4 OpenAI prompt_cache_key auto-injection (Phase E) ([573543f](https://github.com/chopratejas/headroom/commit/573543fce7dcf461d8a65bbfde4a6f9a93cee351))
* PR-E5 volatile-content detector + customer warning (Phase E) ([d1b836d](https://github.com/chopratejas/headroom/commit/d1b836d78c5b92ec12b565e58e47aaaa3e7bc209))
* PR-E5 volatile-content detector + customer warning (Phase E) ([d8aae38](https://github.com/chopratejas/headroom/commit/d8aae382b00c0d340dd79ef89963126156f2f805))
* PR-E6 cache-bust drift detector telemetry (Phase E) ([41ca054](https://github.com/chopratejas/headroom/commit/41ca0544c793731522a26a916bc351cc18a460bf))
* PR-E6 cache-bust drift detector telemetry (Phase E) ([ce37940](https://github.com/chopratejas/headroom/commit/ce37940d17e149bf6961742472a61885de5f2e73))
* PR-F1 classify_auth_mode helper (Phase F kickoff) ([fb25a26](https://github.com/chopratejas/headroom/commit/fb25a26180a4fb508e9e090f716f340af4a1ed03))
* PR-F1 classify_auth_mode helper (Phase F kickoff) ([ca9de93](https://github.com/chopratejas/headroom/commit/ca9de93cfc4c5e8dd4e017615aedad3249ea8c1d))
* preserve Codex OAuth provider config ([7e29a60](https://github.com/chopratejas/headroom/commit/7e29a60b6feca5848ce1a07119294d45a472c04e))
* preserve Codex OAuth proxy delivery ([06428d2](https://github.com/chopratejas/headroom/commit/06428d20fd0514375024a6784b30b873a0f8d82d))
* **providers:** register claude-opus-4-7 + [1m] tier with 1M context ([2e9f52f](https://github.com/chopratejas/headroom/commit/2e9f52fda2e81f3c3bafc3e2562e2e2ef5b321f2))
* **proxy,dashboard:** correct savings reporting ([71348f4](https://github.com/chopratejas/headroom/commit/71348f407f5d3463165a44665bc48e40f7b29a69))
* **proxy,dashboard:** correct savings reporting for backend-routed traffic and drop broken Compression Quality widget ([6643266](https://github.com/chopratejas/headroom/commit/6643266ec02abbdb54cd2910b3a23845c144805f))
* **proxy:** accept IPv6-mapped loopback hosts ([dcf501d](https://github.com/chopratejas/headroom/commit/dcf501d85299e5ef0e0788e1b2d93ca55a1fa687))
* **proxy:** add auth_mode_policy_enforcement feature flag (F2.1 c3/6) ([027b203](https://github.com/chopratejas/headroom/commit/027b203c2352693b53a54d9e7763cc61c19e2f20))
* **proxy:** bound Codex Responses compression work ([160989c](https://github.com/chopratejas/headroom/commit/160989c43ecbe86d36c1c69681c578276c756285))
* **proxy:** cache concurrency lock, multi-worker docs, bounded compre… ([51eeaf6](https://github.com/chopratejas/headroom/commit/51eeaf6662816375244c8a09009b5b77dee92d7d))
* **proxy:** defer file logging install to create_app() ([6801820](https://github.com/chopratejas/headroom/commit/6801820795aa7194cf838e9115ca0909f3f93cbf))
* **proxy:** defer file logging install to create_app() ([3d2d894](https://github.com/chopratejas/headroom/commit/3d2d894a33cc788b6d7f8582ebc8d04f1967f7c2))
* **proxy:** F2.1 — per-auth-mode CompressionPolicy gates ([3b1ee2e](https://github.com/chopratejas/headroom/commit/3b1ee2e4af9eba4318e748d65be88bb04daef81c))
* **proxy:** F2.2 — per-mode CompressionPolicy tuning fields ([294df2b](https://github.com/chopratejas/headroom/commit/294df2b894a985c8501981de35865aa3245fbdc2))
* **proxy:** guard CCR tool injection against frozen prefix to preserve cache ([9d5f15b](https://github.com/chopratejas/headroom/commit/9d5f15bdeb0dd66d0e97c24b09aa7373e585caf2))
* **proxy:** guard CCR tool injection against frozen prefix to preserve cache ([429ae00](https://github.com/chopratejas/headroom/commit/429ae0095b65a37276098ee314fc9f97dc361e1e)), closes [#294](https://github.com/chopratejas/headroom/issues/294)
* **proxy:** hoist compression_policy outside is_token_mode branch (F2.1 c5 followup) ([708b87a](https://github.com/chopratejas/headroom/commit/708b87a19fbe0c71ab5945ef8fda154f0abd2eef))
* **proxy:** hot-fix Codex /v1/responses compression via PyO3 inline call ([1a3098a](https://github.com/chopratejas/headroom/commit/1a3098a48f8021e00c1fb118b1a5dab0b8add559))
* **proxy:** inline WebSocket /v1/responses compression via PyO3 ([1050e00](https://github.com/chopratejas/headroom/commit/1050e00241960e475e63badd59c175abc5b3e7cf))
* **proxy:** inline WebSocket /v1/responses compression via PyO3 ([4a50313](https://github.com/chopratejas/headroom/commit/4a503135487182523a9db559cff23cd42427ca64))
* **proxy:** make third-party extensions opt-in ([f28f697](https://github.com/chopratejas/headroom/commit/f28f697310f7cc5cd0cd6354c1c0fa9ade158c67))
* **proxy:** MemoryDecision contract + 3 bypass bugs + drop 500-char query cap ([4e1b218](https://github.com/chopratejas/headroom/commit/4e1b21854456431253952ec4d32b8464133cc667))
* **proxy:** MemoryDecision contract + 3 bypass bugs + drop 500-char query cap ([71d5a7b](https://github.com/chopratejas/headroom/commit/71d5a7b5455f92bd07c1cfc95909738687672307))
* **proxy:** plumb CompressionPolicy through proxy + dispatchers (F2.1 c2/6) ([948c8f2](https://github.com/chopratejas/headroom/commit/948c8f2069c6c88eaba40196f2235cd279e2064f))
* **proxy:** PR-D1 native Bedrock InvokeModel route + SigV4 ([f2d4fe3](https://github.com/chopratejas/headroom/commit/f2d4fe39cb9f533e5928201c2b17e50a625c1996))
* **proxy:** PR-D2 Bedrock streaming via binary EventStream ([66426e7](https://github.com/chopratejas/headroom/commit/66426e7b75a730411c29df6501f49f5d6f5b8ab5))
* **proxy:** PR-D3 Bedrock observability + auth-mode integration ([90ef662](https://github.com/chopratejas/headroom/commit/90ef66213d8566925ce0b021d84fd094fab79340))
* **proxy:** PR-D4 native Vertex publisher path + ADC bearer auth ([c10a219](https://github.com/chopratejas/headroom/commit/c10a2195afdf86e0ade17fad53a509de577224d4))
* **proxy:** record cache reads/writes on backend-routed streaming ([#327](https://github.com/chopratejas/headroom/issues/327)) ([7ddc0ce](https://github.com/chopratejas/headroom/commit/7ddc0cef1eb7b58021fcb57f6c2a7f00a6f090c7))
* **proxy:** record cache reads/writes on backend-routed streaming ([#327](https://github.com/chopratejas/headroom/issues/327)) ([3e97a0c](https://github.com/chopratejas/headroom/commit/3e97a0cb73a5801496dc1ae2dab9e81788d87b2a))
* **proxy:** recover SSE usage from truncated final event ([b166bd3](https://github.com/chopratejas/headroom/commit/b166bd3d6a9b50925bd0fee5dedce704666ee35e))
* **proxy:** recover SSE usage from truncated final event ([d91254a](https://github.com/chopratejas/headroom/commit/d91254a0f2419dbd77e6b8f3b6718a029b725082))
* **proxy:** remove content-keyed TTL walker that conflated content wi… ([9129188](https://github.com/chopratejas/headroom/commit/9129188d4af6281844bc5428506901fa54b1c18e))
* **proxy:** remove content-keyed TTL walker that conflated content with positional cache ([#327](https://github.com/chopratejas/headroom/issues/327)) ([35eaf8d](https://github.com/chopratejas/headroom/commit/35eaf8de7f8b004b352fedbdc598a089857623d0))
* **proxy:** restore Anthropic compression on token mode (issue [#327](https://github.com/chopratejas/headroom/issues/327)) ([44944fb](https://github.com/chopratejas/headroom/commit/44944fb3fe47b10b068aadc3dea45ef6b6310ffb))
* **proxy:** route Codex subscription /backend-api/* catchall to chatgpt.com ([4073dcd](https://github.com/chopratejas/headroom/commit/4073dcd23108b2ed1e37a9acbab29f34789e2bbc))
* **proxy:** route Codex subscription /backend-api/* catchall to chatgpt.com ([341dcf0](https://github.com/chopratejas/headroom/commit/341dcf03e9a28683bc762af436347b3798102d24))
* **proxy:** rustfmt drift in live_zone_anthropic imports (F2.1 c2 followup) ([0546795](https://github.com/chopratejas/headroom/commit/05467955479d9578934a956c29581997d12a1b16))
* **proxy:** Strands MCP bundle + backend path fixes + Codex fail-closed protection ([20dc1f2](https://github.com/chopratejas/headroom/commit/20dc1f28f3ccadbc2d1109d73b5bfe875eb81c47))
* **proxy:** strip cache_control before hashing turn_id ([f2c6e90](https://github.com/chopratejas/headroom/commit/f2c6e9065f0be4d5b40a5ceb54be5e5cd07788de))
* **proxy:** strip cache_control before hashing turn_id ([084678d](https://github.com/chopratejas/headroom/commit/084678df7cf09bd01a18aa4ae2a68c607916949c))
* **proxy:** support multi-worker Docker env startup ([e2d9561](https://github.com/chopratejas/headroom/commit/e2d95614c25a3629a8429840d1982b7c82b5a345))
* **proxy:** surface CompressionDecision.passthrough_reason in tags ([88983ad](https://github.com/chopratejas/headroom/commit/88983ad34efef0556be9baeb1dd57397e4e6d724))
* **proxy:** surface CompressionDecision.passthrough_reason in tags ([e4e28b6](https://github.com/chopratejas/headroom/commit/e4e28b65f459df6879b38384842faaeba6357a2a))
* **proxy:** thread tags into 13 outcome sites + synth /v1/models + free-fn _extract_tags ([3ec5492](https://github.com/chopratejas/headroom/commit/3ec549288aed46dc46067b6dc55f287a8b19d218))
* **proxy:** thread tags into 13 outcome sites; synthesize /v1/models for ChatGPT auth ([6d62985](https://github.com/chopratejas/headroom/commit/6d62985b73b4a9e50d13ee9fc3df4b62bcba1c14))
* **proxy:** unblock Codex WS compression — delete inner-pool + global semaphore ([6cf8f7f](https://github.com/chopratejas/headroom/commit/6cf8f7f44abf82b5a34572ead0b6ee0bf523a6ce))
* **proxy:** unblock Codex WS compression — delete inner-pool + global semaphore ([a167f5c](https://github.com/chopratejas/headroom/commit/a167f5cc299ae78ce94cc6ddc3694af26c5c408b))
* **proxy:** wire CompressionPolicy through handlers + flip default enabled (F2.1 c5/5) ([0fef428](https://github.com/chopratejas/headroom/commit/0fef428f1fecff6a72a355098457bacc6465ef5f))
* **proxy:** wire PyO3 compression into /v1/responses handler (hot-fix c2/2) ([c0baaf0](https://github.com/chopratejas/headroom/commit/c0baaf052f18b2561226d20de6b182bb14300416))
* re-enable Codex /v1/responses compression + fix prose-format over-freeze ([7aaa4ac](https://github.com/chopratejas/headroom/commit/7aaa4ac48fcada1dc8e44477dc8256153d75093b))
* register serena mcp during wrap ([44231f6](https://github.com/chopratejas/headroom/commit/44231f68cd8bbbd010defdba2fb919bf49bafc89))
* register serena mcp during wrap ([a7160f7](https://github.com/chopratejas/headroom/commit/a7160f7eab87813f065ab398692231398c221b87))
* release image router models after compression ([5d1cc84](https://github.com/chopratejas/headroom/commit/5d1cc843572aa6876e2065d4a6dda65d08cd54e2))
* release image router models after compression ([cf60882](https://github.com/chopratejas/headroom/commit/cf608829494b478f493a3d346034abc571afc273))
* remove env_key from injected Codex provider — fixes [#393](https://github.com/chopratejas/headroom/issues/393) ([7fb6d7f](https://github.com/chopratejas/headroom/commit/7fb6d7fefdce6145a39485b0c7d3227bdd573530))
* **rust:** 7 proxy bugs found in review (phase-1) ([56f679f](https://github.com/chopratejas/headroom/commit/56f679f2479c857262a8d680704f04b2990890ea))
* **rust:** A4 — honor cache_control markers; serde_json arbitrary_precision + raw_value ([3f99128](https://github.com/chopratejas/headroom/commit/3f991282364d715e98882749e8fa9a5b02455609))
* **rust:** audit cleanup — DiffCompressor CCR leak, CCR TOCTOU race, … ([764b7a8](https://github.com/chopratejas/headroom/commit/764b7a802144b513506697fa18cf9215567c9d1c))
* **rust:** audit cleanup — DiffCompressor CCR leak, CCR TOCTOU race, clippy debt, dep dedup ([378d8a0](https://github.com/chopratejas/headroom/commit/378d8a0f055ff84bc4cdb0674c532e054a8e3132))
* **rust:** IntelligentContextManager port (simplified, OSS) — PR-B ([21784f4](https://github.com/chopratejas/headroom/commit/21784f4288078518cf28e0d6731764809c082355))
* **rust:** IntelligentContextManager port (simplified, OSS) — PR-B ([013344f](https://github.com/chopratejas/headroom/commit/013344f6fd31e968b466833b780eb503840696e8))
* **rust:** port MessageScorer to Rust + parity harness (PR-A) ([05f91d9](https://github.com/chopratejas/headroom/commit/05f91d9adc445f1dd6aff1300469d3aa216df176))
* **rust:** port MessageScorer to Rust + parity harness (PR-A) ([21989e3](https://github.com/chopratejas/headroom/commit/21989e36400cf425ad49bb51eecfe9cdc793ec76))
* **rust:** PR-A1 — make /v1/messages compression a passthrough ([a974bb1](https://github.com/chopratejas/headroom/commit/a974bb153a25a4b32a75ff34a5f11b8b39f675a6))
* **rust:** reformat/offload pipeline + log templates + diff noise (Ph… ([a1d62d8](https://github.com/chopratejas/headroom/commit/a1d62d81b07ceca40a1c875a46c48c101676b74b))
* **rust:** reformat/offload pipeline + log templates + diff noise (Phase 3g rework) ([01a423a](https://github.com/chopratejas/headroom/commit/01a423a3164ed67840c0420a9d8786daca4167b6))
* **rust:** smart_crusher BUG [#4](https://github.com/chopratejas/headroom/issues/4) — k-split overshoot when k_total=1 ([4b80c02](https://github.com/chopratejas/headroom/commit/4b80c025fe7f87cda6b5509f42234404f64945e0))
* **rust:** smart_crusher scaffold review findings — hash truncation, int parse, python-repr matcher ([a64716d](https://github.com/chopratejas/headroom/commit/a64716d5d1545258c811642848a7f070f329f8ed))
* **rust:** smart_crusher self-review — Python parity for f64 overflow path ([ab4d294](https://github.com/chopratejas/headroom/commit/ab4d294487482ec3eb79d88cfbb7247322e16008))
* **rust:** wire ICM compressor into Rust proxy on /v1/messages ([4c52766](https://github.com/chopratejas/headroom/commit/4c52766860c6fa1b3c7d46c223050d8a0c7d7840))
* **rust:** wire ICM compressor into Rust proxy on /v1/messages ([fa5fbfa](https://github.com/chopratejas/headroom/commit/fa5fbfabf4df6bd2dfaebec2ffee6f17b39d4a1c))
* **rust:** wire SmartCrusher as JsonOffload in the pipeline (Phase 3g PR2) ([2a0582d](https://github.com/chopratejas/headroom/commit/2a0582dee1d9da3159fe985eec624019eeb99b62))
* **security:** allowlist GitGuardian-flagged test fixtures ([cf5a715](https://github.com/chopratejas/headroom/commit/cf5a71571c8566de7d315b362d9f1382ffc9d19e))
* ship glibc 2.38 compat shim + wheel symbol audit (closes [#355](https://github.com/chopratejas/headroom/issues/355)) ([6793adb](https://github.com/chopratejas/headroom/commit/6793adb003e1efb8ba6a6446a7829de2602a266f))
* ship glibc 2.38 compat shim + wheel symbol audit (closes [#355](https://github.com/chopratejas/headroom/issues/355)) ([e214672](https://github.com/chopratejas/headroom/commit/e2146724af09801c8f75dcd357d1a61c91d6bbb0))
* **smart_crusher:** drop pre-fix CCR override hack — use dataclass defaults ([ca2cb45](https://github.com/chopratejas/headroom/commit/ca2cb450c59b9f5f952f4ea4765147e7de051405))
* **smart_crusher:** re-attach TOIN learning loop + audit known regressions ([33825c3](https://github.com/chopratejas/headroom/commit/33825c354217810c3b9339980e9cdb9642ff12f0))
* **smart_crusher:** re-attach TOIN learning loop + audit known regressions ([049ca9c](https://github.com/chopratejas/headroom/commit/049ca9cab2ec3c288c1a9c39bc7b3bba760c8a4d))
* **smart_crusher:** re-land orphaned audit close-out — CCR knob + scorer fail-loud ([5dcecdc](https://github.com/chopratejas/headroom/commit/5dcecdcd357db11579ccaf74d2a1a802867c67dc))
* **smart_crusher:** re-land orphaned audit close-out — CCR knob + scorer fail-loud ([3f8de4e](https://github.com/chopratejas/headroom/commit/3f8de4e117324eff4dadf815ffea63d75849aa8a))
* speed up Kompress inference ([98f73e3](https://github.com/chopratejas/headroom/commit/98f73e3502c62dbc2c4119dcc68ef27537cc8f58))
* stabilize codex compression, stats, and proxy lifecycle ([eaf5980](https://github.com/chopratejas/headroom/commit/eaf5980b4ac48c909a1fa2ef1ace460752ec0918))
* stabilize learn write-failure worker test ([155c069](https://github.com/chopratejas/headroom/commit/155c0691767d68f7ec5b47de476f8215946901bf))
* **subscription:** address G2 review findings — phantom delta, multi-worker race, silent fallbacks ([f68090c](https://github.com/chopratejas/headroom/commit/f68090c5b4bd9670ee7fc9a0c71e57f05072c18c))
* **subscription:** wire tokens_saved_rtk data plane ([c7d1247](https://github.com/chopratejas/headroom/commit/c7d1247a2bd06738c3b6c8e73e15902a7e428467))
* **subscription:** wire tokens_saved_rtk from RTK stats endpoint ([44c605f](https://github.com/chopratejas/headroom/commit/44c605fbb0e3ae4e7a92d9693d0da8bc21115b81))
* sync Codex MCP proxy config during wrap ([ac1d11c](https://github.com/chopratejas/headroom/commit/ac1d11c9a2a946ad5362db106cf9094262b7329a))
* **telemetry:** honour HEADROOM_TELEMETRY=off in /v1/telemetry collector ([218821e](https://github.com/chopratejas/headroom/commit/218821e4897dfcb78382b95330fc69cd08d9ccc1))
* **telemetry:** honour HEADROOM_TELEMETRY=off in /v1/telemetry collector ([#390](https://github.com/chopratejas/headroom/issues/390)) ([9ddbdf2](https://github.com/chopratejas/headroom/commit/9ddbdf2313eb4a77ad6cff3a0ae5eb0264cf7c57))
* **testing:** stabilize 3.12 suite and fingerprints ([0264e03](https://github.com/chopratejas/headroom/commit/0264e03d33c3d742d8a16760c7c4bf898dded46f))
* **tests:** drive RTK subprocess failure with real exec, not monkeypatched run ([9b6d637](https://github.com/chopratejas/headroom/commit/9b6d6374f13a88842a1944688005649ad3680acd))
* **tests:** make codex scheduler stress test machine-independent ([f82d61b](https://github.com/chopratejas/headroom/commit/f82d61b1636cfa7d75aeca01f32df070cff3318d))
* **tests:** make stress test CI-robust — uniform frame sizes, tighter ratio ([c22342e](https://github.com/chopratejas/headroom/commit/c22342e1c7e909b3270d19fbaa5583b91a166952))
* **tests:** mock logger.warning directly instead of relying on caplog ([c38dac3](https://github.com/chopratejas/headroom/commit/c38dac301e6bc702979ab11357a9c27a180ae060))
* **tests:** patch headroom.rtk.get_rtk_path, not the helpers alias ([317dffe](https://github.com/chopratejas/headroom/commit/317dffe58fb0c6233210bbc9e42ebf16b9288391))
* **tests:** pin proxy_ccr + text_compressors tests to lossy+CCR path ([3ad6965](https://github.com/chopratejas/headroom/commit/3ad69650b4340c02e384b4a18b8bc5c59c4e1257))
* **tests:** ship scripts/replay_codex_ws_load.py so CI can import it ([e532763](https://github.com/chopratejas/headroom/commit/e53276302ef9585be37519b16eb94513242372e7))
* **tests:** stop module-level dotenv loaders from polluting os.environ during pytest collection ([d5ca50c](https://github.com/chopratejas/headroom/commit/d5ca50cd038c06044090db0ed65363f47e314ac2))
* **tests:** tomllib fallback to tomli on python 3.10 ([74843d1](https://github.com/chopratejas/headroom/commit/74843d1d626de70158a359661a540c615ef1a6c5))
* **test:** stub _run_compression_in_executor on _DummyOpenAIHandler ([456a6b3](https://github.com/chopratejas/headroom/commit/456a6b33af06f9dfdd07c58cb4a7820c6b890435))
* **tests:** update CacheAligner detector-only tests for F2.2 5-field CompressionPolicy ([be5d9f7](https://github.com/chopratejas/headroom/commit/be5d9f7baaf11d706afecf7c0ca0b9da984f38a2))
* **tests:** widen wrap-e2e openclaw startup timeout from 5s to 30s ([2ae88a6](https://github.com/chopratejas/headroom/commit/2ae88a6874b19ec9c4173b5a41e4f09918d2a348))
* **tests:** widen wrap-e2e openclaw startup timeout to 30s ([2febf1b](https://github.com/chopratejas/headroom/commit/2febf1bbbfd0e2d7fde438afff93b9e2e9c9d093))
* **traffic-learner:** block bogus error_recovery pairs at the source ([dd287a8](https://github.com/chopratejas/headroom/commit/dd287a8257c13f65bfb833e1f4ff9c855b72d1a7))
* **traffic-learner:** raise min-evidence default and make it configurable ([290238f](https://github.com/chopratejas/headroom/commit/290238f39854bbeffc0d8d9c93727642c3f16be1))
* **traffic-learner:** tighten matchers and drop contradictions ([6061314](https://github.com/chopratejas/headroom/commit/606131451b411336212e80af72d99537164bf54e))
* **transforms:** F2.2 c2/3 — wire toin_read_only gate + extend policy_selected log ([5b38cbf](https://github.com/chopratejas/headroom/commit/5b38cbf8a79d146adb6ca2ee25974bc35e271b05))
* **transforms:** Python parity port of CompressionPolicy + cache_aligner gate (F2.1 c4/5) ([de8e245](https://github.com/chopratejas/headroom/commit/de8e245990399442145a2aa5a8fe61eb4ddb3a76))
* **vertex,bedrock:** remove dead handle_raw_predict, honour X-Forwarded-Proto, cap Bedrock body size ([c6ecdc4](https://github.com/chopratejas/headroom/commit/c6ecdc429923872a9821e2e191da4990d41e8ad4))
* **vertex,bedrock:** remove dead handle_raw_predict, honour X-Forwarded-Proto, cap Bedrock body size ([28b2bac](https://github.com/chopratejas/headroom/commit/28b2bacdf6ffebb1c2bfa81ea80a3862f8ef0e24))
* Wave 3 — multi-turn live integration tests for A+B realignment ([dcbc921](https://github.com/chopratejas/headroom/commit/dcbc921d63e0d647457e04bd449fcae56fdd336b))
* Windows ORT builds and Docker signing retries ([5cc3fd4](https://github.com/chopratejas/headroom/commit/5cc3fd4852337b64e635b33d9a490e0931e46cef))
* **wrap:** unwrap codex restores prior config.toml ([ab0ee65](https://github.com/chopratejas/headroom/commit/ab0ee652e1f79fd260484c08b3462623ae355850))
* **wrap:** unwrap codex restores prior config.toml ([281bc17](https://github.com/chopratejas/headroom/commit/281bc171dcc5109356fdd876d85536c77fee78a3))


### Performance Improvements

* **rust:** tier-1 multi-worker wins — GIL release, sharded CCR store, single-serialize CCR write ([f81b88b](https://github.com/chopratejas/headroom/commit/f81b88b6ac788ab93a2a6ffe600b2128550aee18))
* **rust:** tier-1 multi-worker wins — GIL release, sharded CCR store, single-serialize CCR write ([29aadb1](https://github.com/chopratejas/headroom/commit/29aadb105441e05284bd032c45df77c507ca65cc))


### Code Refactoring

* **e2e:** extract reusable harness into e2e/_lib ([3ca2ce0](https://github.com/chopratejas/headroom/commit/3ca2ce08ae2bce77a6c2f5fab3ca42dafc107f45))
* **proxy:** collapse 3 stream finalizers onto RequestOutcome.from_stream ([af50390](https://github.com/chopratejas/headroom/commit/af503905b584e61eac37d14e290b3033074ffc24))
* **proxy:** collapse 3 stream finalizers onto RequestOutcome.from_stream ([694589f](https://github.com/chopratejas/headroom/commit/694589fec4997ae60118aa3609672a9ed3262a8f))
* **proxy:** CompressionDecision contract + 4 missing-gate bug fixes ([1699435](https://github.com/chopratejas/headroom/commit/1699435ddfd10f3918fcd18e1fdcfc6b232ebdad))
* **proxy:** introduce RequestOutcome funnel; collapse 3 streaming finalizers ([3762f63](https://github.com/chopratejas/headroom/commit/3762f6375ca7a93301f6c3b946f6b1cefb9ca87d))
* **proxy:** introduce RequestOutcome funnel; collapse 3 streaming finalizers ([e898f68](https://github.com/chopratejas/headroom/commit/e898f68b89fcf3dd6cd5d7ee199e9f032d4fb20a))
* **proxy:** MemoryRanker + ImageCompressionDecision + branch-aware version-sync ([c79aa22](https://github.com/chopratejas/headroom/commit/c79aa22be798a40e19c24cfc3e33427cdd84dc29))
* **proxy:** MemoryRanker + ImageCompressionDecision + branch-aware version-sync ([a7b197c](https://github.com/chopratejas/headroom/commit/a7b197c6eca08e0080cb04bc19859026233c5603))
* **proxy:** migrate Codex WS + OpenAI HTTP + batch handlers; delete Databricks ([993c907](https://github.com/chopratejas/headroom/commit/993c9076f92558402c5f925a685fde4001f96daf))
* **proxy:** migrate Gemini + Anthropic non-streaming onto outcome funnel ([7aca004](https://github.com/chopratejas/headroom/commit/7aca00445e23f421d1c8c278bf720cab046da20f))
* **rust:** diff_compressor — surface hidden cutoffs + lossy-emit stats ([255b429](https://github.com/chopratejas/headroom/commit/255b4295dc248b973366fac82c79633bbab04a79))
* single-wheel maturin build backend (fixes [#355](https://github.com/chopratejas/headroom/issues/355)) ([aa80ec2](https://github.com/chopratejas/headroom/commit/aa80ec2cc46684e27c74f8d4f306b12059f30388))
* single-wheel maturin build backend (fixes [#355](https://github.com/chopratejas/headroom/issues/355)) ([2a91cbb](https://github.com/chopratejas/headroom/commit/2a91cbb4b41bc5805b34c0445dda6eebaba28186))

## [Unreleased]

### Fixed
- **PyPI install clarity and release gating.** Documented `pipx --python python3.13`
  for environments where unsupported Python wheel tags cause older-version
  resolution, made PyPI publish failures block GitHub Releases unless
  `PYPI_SKIP=true`, and added an sdist `LICENSE` invariant.

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
  - Install with: `pip install headroom-ai[llmlingua]`
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
python -m headroom.proxy.server --port 8787

# Use with Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude
```

[Unreleased]: https://github.com/headroom-sdk/headroom/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/headroom-sdk/headroom/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/headroom-sdk/headroom/releases/tag/v0.1.0
