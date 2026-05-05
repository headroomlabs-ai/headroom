# Headroom Rust Rewrite — Developer Guide

This document covers the Rust port of Headroom. It is the only new top-level
doc created in Phase 0; longer-form design/plan writeups live elsewhere and
are not versioned in this repo.

## Workspace layout

```
Cargo.toml                       # workspace root
rust-toolchain.toml              # pins stable rustc with rustfmt+clippy
crates/
  headroom-core/                 # library: shared types + transform trait surface
  headroom-proxy/                # binary: axum /healthz (Phase 2 grows this)
  headroom-py/                   # PyO3 cdylib exposing `headroom._core`
  headroom-parity/               # lib + `parity-run` CLI for Python parity tests
tests/parity/
  fixtures/<transform>/*.json    # recorded Python outputs (Phase 1 ports match)
  recorder.py                    # Python-side fixture recorder
scripts/record_fixtures.py       # entry point for running the recorder
```

`cargo build --workspace` builds every crate. `default-members` drops
`headroom-py` from `cargo run`/bare-`cargo test` flows so that `cargo test
--workspace` does not try to execute the PyO3 cdylib standalone (it can't
find `libpython` without a Python interpreter hosting it).

## Common commands

`just` is not installed on dev boxes here; a `Makefile` at the repo root
exposes the same targets:

| Target | What it does |
| --- | --- |
| `make test` | `cargo test --workspace` |
| `make test-parity` | Builds `headroom-py` via maturin, runs `parity-run run` |
| `make bench` | `cargo bench --workspace` |
| `make build-proxy` | Release-builds `headroom-proxy`, strips, prints size |
| `make build-wheel` | `maturin build --release -m crates/headroom-py/pyproject.toml` |
| `make fmt` | `cargo fmt --all` |
| `make lint` | `cargo fmt --check` + `cargo clippy --workspace -- -D warnings` |

## Running the proxy

`headroom-proxy` is a transparent reverse proxy. Phase 1 forwards HTTP/1.1,
HTTP/2, SSE, and WebSocket traffic verbatim to a configured upstream — no
provider logic yet. The intent is that operators run the existing Python
proxy on a private port and put `headroom-proxy` on the public port pointed
at it; end users notice nothing.

```bash
# Build
make build-proxy
./target/release/headroom-proxy --help

# Run against a local upstream
./target/release/headroom-proxy \
    --listen 0.0.0.0:8787 \
    --upstream http://127.0.0.1:8788

# Health checks
curl -s http://127.0.0.1:8787/healthz            # => {"ok":true,...}
curl -s http://127.0.0.1:8787/healthz/upstream   # => 200 if upstream reachable
```

### Operator runbook (Phase 1 cutover)

```bash
# 1. Move the Python proxy to a private port (e.g. 8788)
HEADROOM_BIND=127.0.0.1:8788 python -m headroom.proxy &     # or your existing launcher

# 2. Run the Rust proxy on the previously-public port (8787) pointing at it
./target/release/headroom-proxy --listen 0.0.0.0:8787 --upstream http://127.0.0.1:8788 &

# 3. End users keep hitting :8787 unchanged.
# 4. Confirm passthrough:
curl -si http://127.0.0.1:8787/v1/models
# 5. Rollback = stop the Rust proxy and rebind Python back to 8787.
```

### Configuration flags

| Flag | Env var | Default | Notes |
| --- | --- | --- | --- |
| `--listen` | `HEADROOM_PROXY_LISTEN` | `0.0.0.0:8787` | bind address |
| `--upstream` | `HEADROOM_PROXY_UPSTREAM` | (required) | base URL the proxy forwards to |
| `--upstream-timeout` |  | `600s` | end-to-end request timeout (long for streams) |
| `--upstream-connect-timeout` |  | `10s` | TCP/TLS connect timeout |
| `--max-body-bytes` |  | `100MB` | for buffered cases; streams bypass |
| `--log-level` |  | `info` | `RUST_LOG`-style filter |
| `--rewrite-host` / `--no-rewrite-host` | | rewrite | rewrite Host to upstream (default) |
| `--graceful-shutdown-timeout` | | `30s` | wait for in-flight on SIGTERM/SIGINT |

### Picking the next port: invocation telemetry

Before porting another Python compressor to Rust, check what's actually
running. The Python proxy already exposes per-transform telemetry on
`/stats` (`headroom.proxy.prometheus_metrics`):

```bash
# Top compressors by invocation count (last process lifetime)
curl -s http://127.0.0.1:8788/stats | jq '.compressions_by_strategy'
# {
#   "intelligent_context": 12453,
#   "smart_crusher": 487,
#   "search":         312,
#   "diff":            28,
#   "code":             0,        # ← never fires; safe to defer porting
#   ...
# }

# Per-transform timing (avg/max/count by transform name)
curl -s http://127.0.0.1:8788/stats | jq '.pipeline_timing'

# Token savings attributable to each strategy
curl -s http://127.0.0.1:8788/stats | jq '.tokens_saved_by_strategy'
```

This is the data the audit-cleanup PR (2026-04-30) recommended for
prioritizing the next Python → Rust port. Strategies with zero or
near-zero invocations are deferral candidates; strategies on the hot
path are porting candidates regardless of LOC count.

### Reserved paths

`/healthz` and `/healthz/upstream` are intercepted by the Rust proxy and
**not** forwarded. Operators must not name a real upstream route either of
these. Everything else is a catch-all forward.

## Maturin + Python wiring

`headroom-py` is a PyO3 cdylib that exposes `headroom._core` in Python. The
`extension-module` feature is opt-in so plain `cargo build --workspace` does
not try to link against `libpython` on systems that don't have it.

### First-time setup (clean venv recommended)

```bash
python3.11 -m venv /tmp/hr-rust-venv
source /tmp/hr-rust-venv/bin/activate
pip install maturin
cd crates/headroom-py
maturin develop           # editable dev build, installs headroom._core
cd /tmp                   # IMPORTANT: step out of the repo root first
python -c "from headroom._core import hello; print(hello())"
# => headroom-core
```

> Why `cd /tmp`? The repo root also contains the Python `headroom/` package.
> Running the smoke import from the repo root makes Python resolve `headroom`
> to `./headroom/__init__.py` (the full SDK, which pulls in heavy deps) instead
> of the lightweight namespace package installed by maturin. Tests should
> either run outside the repo root, or ensure `headroom` is installed into
> the same venv (then the maturin-installed `_core.so` lands alongside it and
> both imports resolve).

### Release wheels

```bash
make build-wheel
# wheels land under target/wheels/
```

CI (`.github/workflows/rust.yml`) builds linux-x86_64, macos-arm64, and
macos-x86_64 wheels via `PyO3/maturin-action` and uploads them as artifacts.

## Parity harness

`crates/headroom-parity` owns the Rust-vs-Python oracle:

- JSON fixtures under `tests/parity/fixtures/<transform>/` (schema:
  `{ transform, input, config, output, recorded_at, input_sha256 }`).
- `TransformComparator` trait — one impl per transform. Phase 0 stubs return
  `Err(...)`; the harness flags those as `Skipped`, not panics.
- `parity-run` CLI: `cargo run -p headroom-parity -- run [--only TRANSFORM]`.
- Unit tests in `crates/headroom-parity/src/lib.rs` include a **negative
  test** (`harness_reports_diff_for_divergent_comparator`) proving the
  harness detects mismatched output before any real port lands.

### Recording fresh fixtures

```bash
source .venv/bin/activate           # the main Python SDK venv
python scripts/record_fixtures.py   # uses tests/parity/recorder.py
ls tests/parity/fixtures/*/ | sort | uniq -c
```

The recorder monkey-patches the in-process transform classes (see
`record_all()` in `tests/parity/recorder.py`). It does **not** modify any
file under `headroom/`.

## Known regressions in retired-Python components

The Stage 3b/3c.1b retirements deleted Python source for `DiffCompressor`
and `SmartCrusher` and replaced them with PyO3-delegating shims. The
2026-04-28 audit found that the retirements shipped with subsystems
silently disconnected. This section tracks each gap and its disposition
so they don't regress further or get forgotten.

### SmartCrusher

| Subsystem | State | Tracked by |
|---|---|---|
| TOIN learning loop | **Re-attached 2026-04-28.** Shim's `crush()` and `_smart_crush_content()` now call `toin.record_compression()` after a real compression. Filtered on `strategy != "passthrough"` to ignore JSON re-canonicalization. Best-effort: TOIN failures are logged at debug level and don't break compression. | `tests/test_smart_crusher_toin_attachment.py` |
| CCR marker emission knob | **Honored end-to-end 2026-04-29.** New `enable_ccr_marker: bool` field on Rust `SmartCrusherConfig`; `crush_array` checks it before emitting the `<<ccr:HASH>>` marker text and the CCR store write. Python shim flips it from `ccr_config.enabled and ccr_config.inject_retrieval_marker` — both flags collapse to the same Rust gate, since storing payloads under either off-switch makes no sense. Scope: gates only the row-drop sentinel path; Stage-3c.2 opaque-string CCR substitutions still emit always (no Python equivalent, no production caller asks for suppression). | `tests/test_smart_crusher_toin_attachment.py` + `crates/headroom-core/.../crusher.rs::tests::enable_ccr_marker_*` |
| Custom relevance scorer | **Closed (fail-loud) 2026-04-29.** `relevance_config` and `scorer` constructor args remain in the signature for source compat, but the shim raises `NotImplementedError` when either is non-None — silently dropping a user-supplied scorer is a textbook silent-fallback bug. Full plumbing waits on Stage-3c.2's relevance-crate Python bridge. | `tests/test_smart_crusher_toin_attachment.py::test_custom_*_arg_raises_not_implemented` |
| Per-tool TOIN learning hook | **Re-attached partially.** `_smart_crush_content` accepts `tool_name` and now threads it into the TOIN record. The hook is best-effort — it improves `query_context` aggregation but doesn't drive per-tool overrides yet. | `tests/test_smart_crusher_toin_attachment.py::test_smart_crush_content_records_to_toin` |

### DiffCompressor

| Subsystem | State |
|---|---|
| Adaptive context windows | Honored byte-for-byte (parity fixture-locked). |
| TOIN integration | Never had one — DiffCompressor records via `_record_to_toin` in ContentRouter, which already runs for non-SmartCrusher strategies. No regression. |

### Phase 3e.1 — `signals/` trait module + KeywordDetector (2026-04-29)

The Python `error_detection.py` regex registry was retired and reborn as a
trait + tier system in `crates/headroom-core/src/signals/`. See
`signals/README.md` for the full architecture; the highlights:

- **Per-granularity traits.** `LineImportanceDetector` ships today; future
  `ContentTypeDetector` and `ItemImportanceDetector<I>` will follow as their
  consumers get touched.
- **`Tiered<T>` combinator.** Composition, not inheritance. Future ML
  detectors slot in as new tiers without changes to `KeywordDetector` or
  any caller.
- **One concrete impl.** `KeywordDetector` (aho-corasick) is the only tier
  registered today. **No NoOp/stub impls** — per project no-silent-fallbacks
  rule, future tiers land with their real implementations.
- **Bug fixes baked in.** `ERROR_KEYWORDS` regex now includes
  `timeout|abort|denied|rejected` (previously drifted from the keyword set);
  `token` dropped from `SECURITY_KEYWORDS` (false-positived on every LLM
  metric reference). Both fixed in the Python regex too via the shim that
  recompiles patterns from the Rust-exposed keyword tables.
- **Companion canonical extension path.** `signals/README.md` documents
  the BGE classifier head — a 384-dim → 4-class softmax on top of the
  already-loaded `bge-small-en-v1.5` embedder — as the natural ML tier.
  Two alternatives kept open: distilled tinyBERT in ONNX, logistic
  regression on lexical features.

### Phase 3g (queued) — Compression Pipeline Formalization (issue #315)

Strategic decision 2026-04-29: after Phase 3e (compressor ports) and
Phase 3f (Rust MCP scaffold) wrap, formalize the lossless-then-lossy-
then-CCR ordering as a cross-cutting `CompressionPipeline` orchestrator
+ `LosslessTransform` / `LossyTransform` traits in
`crates/headroom-core/src/pipeline/`. Existing compressors get
refactored as compositions of pluggable transforms. The crucial design
choice — **parsers for structure, models at the prose/structure
boundary** — is captured in issue #315 and
`memory/project_lossless_first_pipeline.md`. Do NOT start coding before
3e/3f finish.

### Watch list (potential regressions, not yet audited)

- `CCRConfig.enabled=False` end-to-end — **closed 2026-04-29**. Both `enabled=False` and `inject_retrieval_marker=False` collapse to the same Rust `enable_ccr_marker=False` gate (no marker, no store write). See the SmartCrusher table above.
- `SmartCrusherConfig.use_feedback_hints=False` — config field is forwarded to Rust but its honoring inside the Rust crusher hasn't been verified against a parity fixture for the disabled path.

When any item above changes, update both this section and the test file. The shim's docstring also references this section — keep them aligned.

## Phase 0 Blockers

These are known limitations for Phase 0. They are tracked here so Phase 1
doesn't rediscover them.

- **`cache_aligner` fixtures**: `CacheAligner.apply()` takes
  `(messages, tokenizer, **kwargs)` — a `Tokenizer` is provider-specific and
  its cheapest `NoopTokenCounter` / `TiktokenTokenCounter` construction still
  requires pulling `headroom.providers.*` which imports the full observability
  stack (opentelemetry, etc). The recorder records `cache_aligner` only if a
  usable tokenizer is cheaply available; otherwise it logs a blocker and
  skips. See `recorder.py::_build_cache_aligner_tokenizer`.
- **`ccr` is not a single class**: The repo has `CCRToolInjector`,
  `CCRResponseHandler`, `CCRToolCall`, `CCRToolResult` etc. rather than a
  single `CCR` class. The recorder targets the encoder-style entry point
  most analogous to the Rust port (`CCRToolInjector.inject_tool` and
  `CCRResponseHandler.parse_response`). If Phase 1 wants a different split
  it should update `recorder.py::record_all` accordingly.
- **Pre-commit hook noise**: `scripts/sync-plugin-versions.py` mutates
  `.claude-plugin/marketplace.json`, `.github/plugin/marketplace.json`, and
  `plugins/headroom-agent-hooks/**/plugin.json` on every commit. Those
  changes are harmless but each commit in Phase 0 picks them up. Phase 1
  does not need to do anything special — just let the hook run.
- **`rust-toolchain.toml`** pins `channel = "stable"` rather than a specific
  version so CI picks up the same toolchain the local box uses. Tighten to a
  pinned version (e.g. `1.78`) once the port stabilizes.

## Multi-worker deployment — CCR fragmentation

**Status:** PR-B7 (`REALIGNMENT/04-phase-B-live-zone.md`) introduced two
persistent CCR backends. The single-`--workers` recommendation no longer
applies once you select a persistent backend.

### Backend selection

`crates/headroom-core/src/ccr/backends/` ships three implementations of
the `CcrStore` trait:

| Backend                | When to use                                 | Persistence | Multi-worker safe          |
| ---------------------- | ------------------------------------------- | ----------- | -------------------------- |
| `InMemoryCcrStore`     | Tests, single-worker prototyping            | No          | No                         |
| `SqliteCcrStore` (default) | Single-instance prod / single-host fleet | Yes (file)  | Yes (sticky session)       |
| `RedisCcrStore` (opt-in)   | Multi-host / horizontally-scaled prod     | Yes (Redis) | Yes (no stickiness needed) |

`backends::from_config` picks one at startup from the operator's
`CcrBackendConfig`. **Init failures surface to the caller**
(`feedback_no_silent_fallbacks.md`) — a misconfigured DB path or
unreachable Redis URL aborts startup rather than silently degrading to
in-memory.

### When does what work?

- **`SqliteCcrStore`** is the default for new deploys. The DB file lives
  on the local disk; multiple workers on the **same host** share it via
  SQLite's WAL-mode locking, so `--workers N` works as long as a sticky
  load balancer routes each session to the same host. Survives proxy
  restarts: a new worker that opens the same DB file recovers every
  in-flight `<<ccr:HASH>>` marker.
- **`RedisCcrStore`** (cfg-gated behind the `redis` feature) is the
  drop-in for **horizontally-scaled** deployments. Every worker on
  every host hits the same Redis instance; no sticky session is
  required at any layer of the LB. Enable with `--features redis` in
  the proxy crate's Cargo build.
- **`InMemoryCcrStore`** is fine for tests and single-worker
  development. Production deployments using it lose every
  `<<ccr:HASH>>` marker on restart and fragment across workers — keep
  it confined to local boxes.

### What goes wrong with the in-memory backend on `--workers N > 1`

(Historical context — applies only when the operator explicitly
chooses `CcrBackendConfig::InMemory`.) Each uvicorn worker is a
separate Python process. Each process holds its own copies of:

1. **`InMemoryCcrStore`** — sharded `DashMap` mapping
   `hash → original_content` for content the compressor replaced with
   `<<ccr:HASH>>` markers.
2. **`HeadroomProxy._compression_caches`** (`headroom/proxy/server.py:367`)
   — per-session `CompressionCache` dict.
3. **`HeadroomProxy.session_tracker_store`** — per-session prefix-tracker
   state derived from Anthropic's `cache_read_input_tokens` responses.
4. **TOIN learner state** — pattern statistics used to bias the compressor.

When uvicorn round-robins requests across workers, a session whose
turn-1 landed on worker A may have turn-2 land on worker B. Worker B has
zero knowledge of what worker A did, the `<<ccr:HASH>>` marker resolves
to `None`, and the model sees an opaque directive it can't act on.
Switching to `SqliteCcrStore` (default) or `RedisCcrStore` resolves the
fragmentation directly.

### Detecting it in the wild

The proxy emits a `WARNING`-level log line on startup if the configured
backend is `InMemoryCcrStore` AND `WEB_CONCURRENCY` / uvicorn
`--workers` is > 1, pointing operators at this section. The other two
backends never warn — they're the supported multi-worker paths.
