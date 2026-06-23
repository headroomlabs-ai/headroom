//! Savings statistics store — the Rust-native source of truth for the
//! dashboard `/stats` payload.
//!
//! Phase H of the realignment retires the Python proxy and re-implements
//! `savings_tracker.py` in Rust "as part of `observability/`". This module is
//! that re-implementation. The Rust proxy serves every backend (Anthropic,
//! OpenAI, Bedrock, Vertex) in a single process, so savings are attributed and
//! aggregated here for *all* of them at once — there is no cross-process
//! merging or "federation" concept, just one store fed by one dispatch path.
//!
//! # What it tracks
//!
//! - **Lifetime** cumulative counters (requests, tokens saved, USD saved).
//! - A rolling **display session** that resets after an inactivity window, so
//!   the dashboard headline reflects "this working session" rather than
//!   all-time totals.
//! - A bounded **history** of cumulative checkpoints for sparkline / rollup
//!   charts, each tagged with the provider and model that produced it.
//! - **Per-provider** and **per-model** request and savings breakdowns, so a
//!   single view shows every backend in use.
//!
//! # Design for testability
//!
//! All state transitions are pure functions of `(state, outcome, now)` with the
//! clock injected, never read from `SystemTime::now()` inside the aggregation.
//! That keeps the record/rollover/trim logic deterministic and exhaustively
//! unit-testable (the project targets 100% coverage on this module). I/O
//! (load/save) is the only impure surface and is isolated at the edges.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Persisted-state schema version. Bumped when the on-disk shape changes in a
/// way that needs migration handling in [`SavingsState::sanitize`].
pub const SCHEMA_VERSION: u32 = 1;

/// Default rolling display-session inactivity window. After this much idle
/// time the next recorded request starts a fresh session.
pub const DEFAULT_SESSION_INACTIVITY: Duration = Duration::from_secs(60 * 60);

/// Default cap on retained history checkpoints. Old points are dropped oldest
/// first so a long-running proxy never grows the file without bound.
pub const DEFAULT_MAX_HISTORY: usize = 5000;

/// Default cap on the number of history points returned to the dashboard.
pub const DEFAULT_MAX_RESPONSE_HISTORY: usize = 500;

/// Sentinel used when a provider/model label is missing, so savings are never
/// silently dropped from the per-provider / per-model breakdowns.
const UNKNOWN: &str = "unknown";

/// One recorded request outcome — the single input to the store.
///
/// The wiring layer fills this in from the compression dispatch (tokens), the
/// route (provider), the request body (model), and — when a price book is
/// available — the per-token USD prices. The store itself does no pricing
/// lookup; it multiplies the prices it is given, keeping the aggregation pure
/// and the pricing concern separately testable.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RequestOutcome {
    /// Backend/provider label, e.g. `anthropic`, `openai`, `bedrock`, `vertex`.
    pub provider: String,
    /// Model id as seen on the request.
    pub model: String,
    /// Input tokens before live-zone compression.
    pub tokens_before: u64,
    /// Input tokens after compression (what is forwarded upstream).
    pub tokens_after: u64,
    /// Output tokens, when known (`0` means "not measured").
    pub output_tokens: u64,
    /// Provider prefix-cache read tokens, when reported. NOTE: no recorder lane
    /// populates the cache-token fields yet — they (and the `cache_savings_usd`
    /// math) are forward-schema for the response-side token-capture follow-up, so
    /// `cache_savings_usd` is `0` in production today. Kept (and tested) so that
    /// follow-up only has to feed the fields, not reshape the contract.
    pub cache_read_tokens: u64,
    /// Provider prefix-cache write tokens, when reported (see `cache_read_tokens`).
    pub cache_write_tokens: u64,
    /// Whether the upstream call failed (counts toward `requests.failed`).
    pub failed: bool,
    /// Input USD per token for this model (`0.0` when unknown).
    pub input_cost_per_token: f64,
    /// Cache-read USD per token (`0.0` when unknown).
    pub cache_read_cost_per_token: f64,
    /// Cache-write USD per token (`0.0` when unknown).
    pub cache_write_cost_per_token: f64,
    /// Correlating request id for the dashboard's recent-request feed. Empty is
    /// allowed — the store synthesizes a stable id from the request counter.
    pub request_id: String,
    /// End-to-end latency in milliseconds (`0` when not measured).
    pub latency_ms: u64,
}

impl RequestOutcome {
    /// Build a request-side outcome with per-token USD prices resolved from
    /// `prices` for `model`. The `failed`, `latency_ms`, and response-side token
    /// fields default to zero/false; callers set `failed`/`latency_ms` once the
    /// upstream result is known, then `SavingsStore::record` it.
    ///
    /// This is the single place the `ModelPrice → RequestOutcome` price mapping
    /// lives, shared by every recorder lane (forward_http, Bedrock invoke +
    /// streaming, Vertex rawPredict).
    pub fn priced(
        provider: impl Into<String>,
        model: impl Into<String>,
        tokens_before: u64,
        tokens_after: u64,
        request_id: impl Into<String>,
        prices: &crate::observability::pricing::PriceBook,
    ) -> Self {
        let model = model.into();
        let price = prices.lookup(&model).unwrap_or_default();
        Self {
            provider: provider.into(),
            model,
            tokens_before,
            tokens_after,
            input_cost_per_token: price.input,
            cache_read_cost_per_token: price.cache_read,
            cache_write_cost_per_token: price.cache_write,
            request_id: request_id.into(),
            ..Default::default()
        }
    }

    /// Saved input tokens (never negative; compression can only remove).
    pub fn tokens_saved(&self) -> u64 {
        self.tokens_before.saturating_sub(self.tokens_after)
    }

    /// USD saved by compression: removed input tokens priced at list rate.
    pub fn compression_savings_usd(&self) -> f64 {
        self.tokens_saved() as f64 * self.input_cost_per_token
    }

    /// USD saved by prefix-cache reads (input price minus the cheaper cache-read
    /// price), minus any cache-write premium. Can be negative on a write-heavy
    /// request; callers that only want gross read savings clamp separately.
    pub fn cache_savings_usd(&self) -> f64 {
        let read = self.cache_read_tokens as f64
            * (self.input_cost_per_token - self.cache_read_cost_per_token).max(0.0);
        let write_premium = self.cache_write_tokens as f64
            * (self.cache_write_cost_per_token - self.input_cost_per_token).max(0.0);
        read - write_premium
    }

    fn provider_key(&self) -> &str {
        norm_label(&self.provider)
    }

    fn model_key(&self) -> &str {
        norm_label(&self.model)
    }
}

fn norm_label(value: &str) -> &str {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        UNKNOWN
    } else {
        trimmed
    }
}

/// Lock a mutex, recovering the guard even if a previous holder panicked.
/// Savings are best-effort telemetry: a poisoned lock must never propagate a
/// panic onto the request path (which would make every subsequent request fail),
/// so we keep using the still-valid inner data.
fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Round to 6 decimals, mapping non-finite input (NaN/±Inf — e.g. from a
/// garbage price) to 0.0 so corrupt values never poison aggregates or break
/// JSON serialization (`serde_json` with `arbitrary_precision` rejects
/// non-finite floats).
fn round6(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (value * 1_000_000.0).round() / 1_000_000.0
}

/// Cumulative all-time counters.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Lifetime {
    pub requests: u64,
    /// Successful requests where compression actually reduced tokens. Lets the
    /// dashboard show coverage (`requests_compressed`/`requests`) so the savings
    /// percentage — which is the reduction over *compressed* input — is read in
    /// context rather than as an all-traffic figure. `#[serde(default)]` so
    /// pre-existing persisted files (without the field) still load.
    #[serde(default)]
    pub requests_compressed: u64,
    pub tokens_saved: u64,
    pub compression_savings_usd: f64,
    pub cache_savings_usd: f64,
}

/// Rolling working-session counters that reset after inactivity.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DisplaySession {
    pub requests: u64,
    /// Successful requests in this session where compression reduced tokens.
    #[serde(default)]
    pub requests_compressed: u64,
    pub tokens_saved: u64,
    pub total_input_tokens: u64,
    pub compression_savings_usd: f64,
    pub cache_savings_usd: f64,
    pub started_at: Option<String>,
    pub last_activity_at: Option<String>,
}

impl DisplaySession {
    fn savings_percent(&self) -> f64 {
        let before = self.tokens_saved + self.total_input_tokens;
        if before == 0 {
            0.0
        } else {
            (self.tokens_saved as f64 / before as f64) * 100.0
        }
    }
}

/// Per-provider / per-model breakdown bucket.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bucket {
    pub requests: u64,
    pub tokens_before: u64,
    pub tokens_saved: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub compression_savings_usd: f64,
    pub cache_savings_usd: f64,
}

impl Bucket {
    /// Add one request to this bucket. `count_savings` is the caller's single
    /// failure decision (`!failed`) — when false (a failed request) only the
    /// request count is bumped, no savings, so the bucket can't drift from the
    /// lifetime/session totals which apply the same rule.
    fn apply(&mut self, o: &RequestOutcome, count_savings: bool) {
        self.requests += 1;
        if !count_savings {
            return;
        }
        self.tokens_before += o.tokens_before;
        self.tokens_saved += o.tokens_saved();
        self.output_tokens += o.output_tokens;
        self.cache_read_tokens += o.cache_read_tokens;
        self.cache_write_tokens += o.cache_write_tokens;
        self.compression_savings_usd =
            round6(self.compression_savings_usd + o.compression_savings_usd());
        self.cache_savings_usd = round6(self.cache_savings_usd + o.cache_savings_usd().max(0.0));
    }

    fn reduction_pct(&self) -> f64 {
        if self.tokens_before == 0 {
            0.0
        } else {
            (self.tokens_saved as f64 / self.tokens_before as f64) * 100.0
        }
    }
}

/// One cumulative history checkpoint, tagged with its originating provider/model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryPoint {
    pub timestamp: String,
    pub provider: String,
    pub model: String,
    pub total_tokens_saved: u64,
    pub compression_savings_usd: f64,
}

/// One row of the dashboard's "Recent Requests" feed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecentRequest {
    pub request_id: String,
    pub timestamp: String,
    pub provider: String,
    pub model: String,
    pub input_tokens_original: u64,
    pub input_tokens_optimized: u64,
    pub tokens_saved: u64,
    pub output_tokens: u64,
    pub savings_percent: f64,
    pub total_latency_ms: u64,
    pub failed: bool,
}

/// The full persisted aggregate state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavingsState {
    pub schema_version: u32,
    pub lifetime: Lifetime,
    pub display_session: DisplaySession,
    pub history: Vec<HistoryPoint>,
    pub by_provider: HashMap<String, Bucket>,
    pub by_model: HashMap<String, Bucket>,
    /// Bounded newest-last ring of recent requests for the dashboard feed.
    /// `#[serde(default)]` so older persisted files (without this field) load.
    #[serde(default)]
    pub recent: Vec<RecentRequest>,
    /// Runtime-only counters (input/output tokens seen, failures). Persisted so
    /// the lifetime token totals survive restarts.
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub requests_failed: u64,
}

impl Default for SavingsState {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            lifetime: Lifetime::default(),
            display_session: DisplaySession::default(),
            history: Vec::new(),
            by_provider: HashMap::new(),
            by_model: HashMap::new(),
            recent: Vec::new(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            requests_failed: 0,
        }
    }
}

/// Tuning knobs for the store (session window, history caps).
#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub session_inactivity: Duration,
    pub max_history: usize,
    pub max_response_history: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            session_inactivity: DEFAULT_SESSION_INACTIVITY,
            max_history: DEFAULT_MAX_HISTORY,
            max_response_history: DEFAULT_MAX_RESPONSE_HISTORY,
        }
    }
}

impl SavingsState {
    /// Apply one outcome at the given wall-clock instant.
    ///
    /// Pure: all time-dependent behavior (session rollover, timestamps) derives
    /// from `now`, never from the system clock, so this is fully deterministic
    /// under test.
    pub fn record(&mut self, outcome: &RequestOutcome, now: SystemTime, cfg: &StoreConfig) {
        // A failed upstream produced no successful completion: it counts toward
        // the request/failure totals but accrues NO savings — booking savings on
        // a failure would inflate the headline numbers, and a client retry would
        // double-count. Zero the savings contribution here so every aggregate
        // below excludes failures. (The recent-request feed still logs the real
        // attempted compression, tagged `failed`.)
        let failed = outcome.failed;
        let saved = if failed { 0 } else { outcome.tokens_saved() };
        let comp_usd = if failed {
            0.0
        } else {
            outcome.compression_savings_usd()
        };
        let cache_usd = if failed {
            0.0
        } else {
            outcome.cache_savings_usd().max(0.0)
        };
        let input_tokens = if failed { 0 } else { outcome.tokens_after };
        let output_tokens = if failed { 0 } else { outcome.output_tokens };
        // Coverage: a "compressed" request is a successful one that actually
        // reduced tokens (`saved > 0` implies `!failed`).
        let compressed = saved > 0;
        let now_iso = to_rfc3339(now);

        // Lifetime
        self.lifetime.requests += 1;
        if compressed {
            self.lifetime.requests_compressed += 1;
        }
        self.lifetime.tokens_saved += saved;
        self.lifetime.compression_savings_usd =
            round6(self.lifetime.compression_savings_usd + comp_usd);
        self.lifetime.cache_savings_usd = round6(self.lifetime.cache_savings_usd + cache_usd);
        self.total_input_tokens += input_tokens;
        self.total_output_tokens += output_tokens;
        if failed {
            self.requests_failed += 1;
        }

        // Display session (rollover on inactivity)
        if session_expired(&self.display_session, now, cfg) {
            self.display_session = DisplaySession {
                started_at: Some(now_iso.clone()),
                ..DisplaySession::default()
            };
        }
        let session = &mut self.display_session;
        session.requests += 1;
        if compressed {
            session.requests_compressed += 1;
        }
        session.tokens_saved += saved;
        session.total_input_tokens += input_tokens;
        session.compression_savings_usd = round6(session.compression_savings_usd + comp_usd);
        session.cache_savings_usd = round6(session.cache_savings_usd + cache_usd);
        session.last_activity_at = Some(now_iso.clone());
        if session.started_at.is_none() {
            session.started_at = Some(now_iso.clone());
        }

        // Per-provider / per-model. `!failed` is the single failure decision,
        // the same one that zeroed the savings locals above, so buckets and the
        // lifetime/session totals can never disagree about what a failure counts.
        self.by_provider
            .entry(outcome.provider_key().to_string())
            .or_default()
            .apply(outcome, !failed);
        self.by_model
            .entry(outcome.model_key().to_string())
            .or_default()
            .apply(outcome, !failed);

        // History checkpoint (only when something was actually saved → never on
        // a failed request, since `saved` is zeroed above).
        if saved > 0 {
            self.history.push(HistoryPoint {
                timestamp: now_iso,
                provider: outcome.provider_key().to_string(),
                model: outcome.model_key().to_string(),
                total_tokens_saved: self.lifetime.tokens_saved,
                compression_savings_usd: self.lifetime.compression_savings_usd,
            });
            cap_front(&mut self.history, cfg.max_history);
        }

        // Recent-request feed (every request, success or failure). A truthful
        // per-request log: it shows the real attempted compression plus the
        // `failed` flag, even though the aggregates above excluded failures. A
        // blank request id is backfilled from the lifetime counter so the
        // dashboard's `:key="req.request_id"` never collides.
        let recent_saved = outcome.tokens_saved();
        let savings_percent = if outcome.tokens_before == 0 {
            0.0
        } else {
            (recent_saved as f64 / outcome.tokens_before as f64) * 100.0
        };
        let request_id = if outcome.request_id.trim().is_empty() {
            format!("req-{}", self.lifetime.requests)
        } else {
            outcome.request_id.clone()
        };
        self.recent.push(RecentRequest {
            request_id,
            timestamp: to_rfc3339(now),
            provider: outcome.provider_key().to_string(),
            model: outcome.model_key().to_string(),
            input_tokens_original: outcome.tokens_before,
            input_tokens_optimized: outcome.tokens_after,
            tokens_saved: recent_saved,
            output_tokens: outcome.output_tokens,
            savings_percent: round2(savings_percent),
            total_latency_ms: outcome.latency_ms,
            failed: outcome.failed,
        });
        cap_front(&mut self.recent, cfg.max_response_history);
    }

    /// Normalize a freshly-deserialized state. Today it just stamps the current
    /// `schema_version`; it is the single hook where a future schema bump adds
    /// real migration (read the old version, transform, then stamp). The version
    /// field is persisted deliberately — a durable file must carry its own format
    /// version so a later binary can detect and migrate it, which can't be
    /// retrofitted onto files already on disk.
    fn sanitize(mut self) -> Self {
        self.schema_version = SCHEMA_VERSION;
        self
    }
}

/// Whether `session`'s inactivity window has elapsed by `now`. No recorded
/// activity counts as expired, so the first request opens a fresh session.
fn session_expired(session: &DisplaySession, now: SystemTime, cfg: &StoreConfig) -> bool {
    match session.last_activity_at.as_deref().and_then(parse_rfc3339) {
        Some(last) => now
            .duration_since(last)
            .map(|d| d > cfg.session_inactivity)
            .unwrap_or(false),
        None => true,
    }
}

/// Trim a newest-last buffer to its last `max` entries, dropping the oldest.
fn cap_front<T>(buf: &mut Vec<T>, max: usize) {
    if buf.len() > max {
        buf.drain(0..buf.len() - max);
    }
}

/// Compute the live display-session view, collapsing to empty when the window
/// has elapsed (so the dashboard shows a fresh session after idle time).
fn session_view(state: &SavingsState, now: SystemTime, cfg: &StoreConfig) -> DisplaySession {
    if session_expired(&state.display_session, now, cfg) {
        DisplaySession::default()
    } else {
        state.display_session.clone()
    }
}

/// The savings store: shared, mutable aggregate behind a mutex plus its on-disk
/// path and tuning. Cloneable via the inner `Arc`-free `Mutex` is wrapped by
/// callers in an `Arc` for handler sharing.
pub struct SavingsStore {
    state: Mutex<SavingsState>,
    path: Option<PathBuf>,
    cfg: StoreConfig,
    /// Set by `record` when state changes, cleared by `flush`. Lets the request
    /// path mark "needs persisting" without ever touching disk — a background
    /// flusher (see `proxy::AppState::new`) and the shutdown hook do the I/O.
    dirty: AtomicBool,
    /// Serializes `flush` so the background flusher and the shutdown flush never
    /// write the shared temp file concurrently. Distinct from `state` so it is
    /// never held across the request path — `record` does not take it.
    flush_lock: Mutex<()>,
}

impl SavingsStore {
    /// Create a store with no persistence (in-memory only).
    pub fn in_memory() -> Self {
        Self {
            state: Mutex::new(SavingsState::default()),
            path: None,
            cfg: StoreConfig::default(),
            dirty: AtomicBool::new(false),
            flush_lock: Mutex::new(()),
        }
    }

    /// Create a store backed by `path`, loading any existing state.
    pub fn with_path(path: impl Into<PathBuf>, cfg: StoreConfig) -> Self {
        let path = path.into();
        let state = load_state(&path);
        Self {
            state: Mutex::new(state),
            path: Some(path),
            cfg,
            dirty: AtomicBool::new(false),
            flush_lock: Mutex::new(()),
        }
    }

    /// Record one outcome. `now` is injected so tests stay deterministic;
    /// production passes `SystemTime::now()`.
    ///
    /// This runs on the request path, so it does **no disk I/O** — it updates
    /// in-memory state and marks the store dirty. Persistence happens off the
    /// async request worker via [`SavingsStore::flush`], driven by a background
    /// interval task and the shutdown hook (see `proxy::AppState::new`).
    pub fn record(&self, outcome: &RequestOutcome, now: SystemTime) {
        {
            let mut state = lock_recover(&self.state);
            state.record(outcome, now, &self.cfg);
        }
        self.dirty.store(true, Ordering::Release);
    }

    /// Finalize and record a request-side outcome: stamp `failed` and the
    /// end-to-end latency (measured from `started`), then [`record`] it. The
    /// single finalize-and-record path every recorder lane (forward_http,
    /// Bedrock invoke + streaming, Vertex rawPredict) uses once the request's
    /// fate is known — the symmetric counterpart to [`RequestOutcome::priced`].
    ///
    /// [`record`]: SavingsStore::record
    pub fn record_finalized(
        &self,
        mut outcome: RequestOutcome,
        failed: bool,
        started: std::time::Instant,
    ) {
        outcome.failed = failed;
        outcome.latency_ms = started.elapsed().as_millis() as u64;
        self.record(&outcome, SystemTime::now());
    }

    /// Whether this store persists to disk (`--savings-path` was set). Used to
    /// decide whether to spawn the background flusher.
    pub fn is_persistent(&self) -> bool {
        self.path.is_some()
    }

    /// Write pending state to disk when something changed since the last flush.
    ///
    /// Performs blocking filesystem I/O, so it must be called from a blocking
    /// context (the background flusher's `spawn_blocking`, or the shutdown hook)
    /// — never directly from an async request handler. A no-op when nothing is
    /// dirty or the store is in-memory.
    pub fn flush(&self) {
        // Serialize concurrent flushers (background interval vs. shutdown hook)
        // so they never write the shared temp file at the same time.
        let _guard = lock_recover(&self.flush_lock);
        // Clear dirty up front so a `record` concurrent with the write re-arms
        // it (its update lands in the next flush). If the write itself fails,
        // re-arm so a transient I/O error is retried by the next flush rather
        // than silently dropping the accumulated state until new traffic.
        if self.dirty.swap(false, Ordering::AcqRel) && !self.persist() {
            self.dirty.store(true, Ordering::Release);
        }
    }

    /// Snapshot the current state (clone under the lock).
    pub fn snapshot(&self) -> SavingsState {
        lock_recover(&self.state).clone()
    }

    /// Build the dashboard-facing `/stats` JSON for the savings portion.
    ///
    /// `now` drives display-session expiry. The returned value is the
    /// backend-agnostic core contract the dashboard consumes
    /// (`requests`, `tokens`, `cost`, `persistent_savings`, breakdowns).
    pub fn stats_json(&self, now: SystemTime) -> Value {
        let state = self.snapshot();
        build_stats_json(&state, now, &self.cfg)
    }

    /// Build the dashboard-facing `/stats-history` JSON: lifetime totals, the
    /// raw cumulative checkpoint history, and a per-day rollup. The dashboard's
    /// Historical view reads `lifetime`, `history`, and `series.daily`.
    pub fn history_json(&self) -> Value {
        let state = self.snapshot();
        build_history_json(&state)
    }

    /// Snapshot and write to disk. Returns `true` on success (or when there is
    /// no path to persist to); `false` if the write failed, so `flush` can
    /// re-arm the dirty flag and retry.
    fn persist(&self) -> bool {
        let Some(path) = self.path.as_ref() else {
            return true;
        };
        let state = self.snapshot();
        save_state(path, &state).is_ok()
    }
}

/// Load state from disk, returning a default on any error (missing file,
/// unreadable, malformed JSON) so a corrupt file never blocks startup.
pub fn load_state(path: &Path) -> SavingsState {
    let Ok(bytes) = std::fs::read(path) else {
        return SavingsState::default();
    };
    match serde_json::from_slice::<SavingsState>(&bytes) {
        Ok(state) => state.sanitize(),
        Err(_) => SavingsState::default(),
    }
}

/// Serialize the state to pretty JSON bytes — infallibly.
///
/// `SavingsState` contains only plain scalars, strings, vecs and string-keyed
/// maps, and every `f64` is sanitized to a finite value at every write (see
/// [`round6`]), so `to_vec_pretty` cannot fail in practice. `unwrap_or_default`
/// keeps this a single executed line (no closure) so line coverage is 100% on
/// stable; the unreachable error arm lives in std, and `#[coverage(off)]` (a
/// no-op unless the nightly coverage run sets `cfg(coverage_nightly)`) makes the
/// *region* metric a true 100% too.
#[cfg_attr(coverage_nightly, coverage(off))]
fn serialize_state(state: &SavingsState) -> Vec<u8> {
    serde_json::to_vec_pretty(state).unwrap_or_default()
}

/// Create the target file's parent directory when it has a non-empty, missing
/// parent. No-op for bare filenames (empty parent) or root paths (no parent).
fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Persist state atomically: write a sibling temp file, fsync it (best-effort),
/// then rename over the target so a crash mid-write never truncates the live
/// file. The fsync forces the temp contents to disk (not just the page cache)
/// before the rename; a sync failure is non-fatal and still proceeds to the
/// atomic rename.
pub fn save_state(path: &Path, state: &SavingsState) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let json = serialize_state(state);
    // Per-process temp name: two proxies pointed at the same `--savings-path`
    // must not write the same temp file (the per-store `flush_lock` only
    // serializes within one process). The rename target is still atomic.
    let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
    std::fs::write(&tmp, &json)?;
    let _ = std::fs::File::open(&tmp).and_then(|f| f.sync_all());
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Build the `/stats-history` JSON from a state snapshot.
///
/// `history` is cumulative and appended in chronological order, so a per-day
/// rollup is just the last checkpoint seen for each calendar day (the running
/// totals at end of that day). Dates are the `YYYY-MM-DD` prefix of each
/// checkpoint's RFC3339 timestamp.
fn build_history_json(state: &SavingsState) -> Value {
    let mut daily: Vec<Value> = Vec::new();
    for pt in &state.history {
        let date = pt.timestamp.get(0..10).unwrap_or("");
        let same_day = daily
            .last()
            .and_then(|d| d.get("date"))
            .and_then(Value::as_str)
            == Some(date);
        let entry = json!({
            "date": date,
            "tokens_saved": pt.total_tokens_saved,
            "compression_savings_usd": round6(pt.compression_savings_usd),
        });
        if same_day {
            *daily.last_mut().expect("same_day implies non-empty") = entry;
        } else {
            daily.push(entry);
        }
    }
    // `to_value` over a Vec of all-finite scalar/string records cannot fail in
    // practice; `unwrap_or_default` keeps this one executed line (100% line
    // coverage on stable) rather than a panicking `unwrap`.
    let history = serde_json::to_value(&state.history).unwrap_or_default();
    json!({
        "lifetime": {
            "requests": state.lifetime.requests,
            "tokens_saved": state.lifetime.tokens_saved,
            "compression_savings_usd": round6(state.lifetime.compression_savings_usd),
            "cache_savings_usd": round6(state.lifetime.cache_savings_usd),
        },
        "history": history,
        "series": { "daily": daily },
    })
}

/// Build the savings JSON contract from a state snapshot.
fn build_stats_json(state: &SavingsState, now: SystemTime, cfg: &StoreConfig) -> Value {
    let session = session_view(state, now, cfg);
    // Serialized once; surfaced at both the top level and under
    // `persistent_savings` for the dashboard contract (compute it a single time).
    let session_json = display_session_json(&session);
    let total_before = state.total_input_tokens + state.lifetime.tokens_saved;
    let savings_percent = if total_before == 0 {
        0.0
    } else {
        (state.lifetime.tokens_saved as f64 / total_before as f64) * 100.0
    };

    let by_provider_requests: HashMap<&String, u64> = state
        .by_provider
        .iter()
        .map(|(k, b)| (k, b.requests))
        .collect();
    let by_model_requests: HashMap<&String, u64> = state
        .by_model
        .iter()
        .map(|(k, b)| (k, b.requests))
        .collect();

    let per_model: Value = state
        .by_model
        .iter()
        .map(|(model, b)| {
            (
                model.clone(),
                json!({
                    "requests": b.requests,
                    "tokens_sent": b.tokens_before,
                    "tokens_saved": b.tokens_saved,
                    "output_tokens": b.output_tokens,
                    "cache_read_tokens": b.cache_read_tokens,
                    "cache_write_tokens": b.cache_write_tokens,
                    "compression_savings_usd": round6(b.compression_savings_usd),
                    "cache_savings_usd": round6(b.cache_savings_usd),
                    "reduction_pct": round2(b.reduction_pct()),
                }),
            )
        })
        .collect::<serde_json::Map<String, Value>>()
        .into();

    let total_saved_usd =
        round6(state.lifetime.compression_savings_usd + state.lifetime.cache_savings_usd);

    json!({
        "requests": {
            "total": state.lifetime.requests,
            "failed": state.requests_failed,
            // Coverage: how many requests actually got compressed. `savings_percent`
            // is the reduction over *compressed* input, so `compressed`/`total`
            // tells the reader how much of the traffic that figure applies to.
            "compressed": state.lifetime.requests_compressed,
            "by_provider": by_provider_requests,
            "by_model": by_model_requests,
        },
        "tokens": {
            "input": state.total_input_tokens,
            "output": state.total_output_tokens,
            "saved": state.lifetime.tokens_saved,
            "proxy_compression_saved": state.lifetime.tokens_saved,
            "all_layers_saved": state.lifetime.tokens_saved,
            "total_before_compression": total_before,
            "savings_percent": round2(savings_percent),
        },
        "cost": {
            "compression_savings_usd": round6(state.lifetime.compression_savings_usd),
            "cache_savings_usd": round6(state.lifetime.cache_savings_usd),
            "savings_usd": total_saved_usd,
            "per_model": per_model,
        },
        "summary": {
            "cost": {
                "total_saved_usd": total_saved_usd,
                "breakdown": {
                    "compression_savings_usd": round6(state.lifetime.compression_savings_usd),
                    "cache_savings_usd": round6(state.lifetime.cache_savings_usd),
                },
            },
        },
        "persistent_savings": {
            "lifetime": {
                "requests": state.lifetime.requests,
                "tokens_saved": state.lifetime.tokens_saved,
                "compression_savings_usd": round6(state.lifetime.compression_savings_usd),
            },
            "display_session": session_json.clone(),
        },
        "display_session": session_json,
        "history_points": state.history.len(),
        "recent_requests": recent_requests_json(state),
    })
}

/// Map the recent-request ring into the dashboard's `recent_requests` shape,
/// newest first. The dashboard reads `request_id`, `timestamp`, `model`,
/// `input_tokens_optimized`, `output_tokens`, `savings_percent`,
/// `total_latency_ms` (and the original/saved fields in the expanded row).
fn recent_requests_json(state: &SavingsState) -> Value {
    let rows: Vec<Value> = state
        .recent
        .iter()
        .rev()
        .map(|r| {
            json!({
                "request_id": r.request_id,
                "timestamp": r.timestamp,
                "provider": r.provider,
                "model": r.model,
                "input_tokens_original": r.input_tokens_original,
                "input_tokens_optimized": r.input_tokens_optimized,
                "tokens_saved": r.tokens_saved,
                "output_tokens": r.output_tokens,
                "savings_percent": round2(r.savings_percent),
                "total_latency_ms": r.total_latency_ms,
                "optimization_latency_ms": 0,
                "transforms_applied": [],
                "waste_signals": {},
                "failed": r.failed,
            })
        })
        .collect();
    Value::Array(rows)
}

fn display_session_json(s: &DisplaySession) -> Value {
    json!({
        "requests": s.requests,
        "requests_compressed": s.requests_compressed,
        "tokens_saved": s.tokens_saved,
        "total_input_tokens": s.total_input_tokens,
        "compression_savings_usd": round6(s.compression_savings_usd),
        "cache_savings_usd": round6(s.cache_savings_usd),
        "savings_percent": round2(s.savings_percent()),
        "started_at": s.started_at,
        "last_activity_at": s.last_activity_at,
    })
}

fn round2(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (value * 100.0).round() / 100.0
}

/// Format a `SystemTime` as an RFC3339 / ISO-8601 UTC string.
fn to_rfc3339(t: SystemTime) -> String {
    humantime::format_rfc3339_seconds(t).to_string()
}

/// Parse an RFC3339 timestamp back into a `SystemTime`; `None` on bad input.
fn parse_rfc3339(s: &str) -> Option<SystemTime> {
    humantime::parse_rfc3339(s).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = 1_700_000_000; // fixed epoch for deterministic timestamps

    fn at(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
    }

    fn outcome(provider: &str, model: &str, before: u64, after: u64) -> RequestOutcome {
        RequestOutcome {
            provider: provider.to_string(),
            model: model.to_string(),
            tokens_before: before,
            tokens_after: after,
            input_cost_per_token: 0.000_001,
            ..Default::default()
        }
    }

    #[test]
    fn tokens_saved_never_negative() {
        let o = outcome("anthropic", "claude", 100, 250);
        assert_eq!(o.tokens_saved(), 0);
        let o = outcome("anthropic", "claude", 300, 100);
        assert_eq!(o.tokens_saved(), 200);
    }

    #[test]
    fn compression_savings_usd_prices_saved_tokens() {
        let o = outcome("anthropic", "claude", 1000, 600);
        assert!((o.compression_savings_usd() - 0.0004).abs() < 1e-12);
    }

    #[test]
    fn cache_savings_usd_read_minus_write_premium() {
        let mut o = outcome("anthropic", "claude", 0, 0);
        o.cache_read_tokens = 1000;
        o.cache_read_cost_per_token = 0.000_000_1; // 1/10th input
        o.cache_write_tokens = 200;
        o.cache_write_cost_per_token = 0.000_001_25; // 1.25x input
                                                     // read savings = 1000 * (1e-6 - 1e-7) = 9e-4; write premium = 200 * 2.5e-7 = 5e-5
        assert!((o.cache_savings_usd() - (0.0009 - 0.00005)).abs() < 1e-12);
    }

    #[test]
    fn cache_savings_clamps_negative_components() {
        let mut o = outcome("anthropic", "claude", 0, 0);
        // cache_read priced ABOVE input → max(0) keeps read savings at 0
        o.cache_read_tokens = 100;
        o.cache_read_cost_per_token = 0.01;
        assert_eq!(o.cache_savings_usd(), 0.0);
    }

    #[test]
    fn norm_label_falls_back_to_unknown() {
        assert_eq!(norm_label("  "), UNKNOWN);
        assert_eq!(norm_label(""), UNKNOWN);
        assert_eq!(norm_label(" openai "), "openai");
    }

    #[test]
    fn record_updates_lifetime_and_breakdowns() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig::default();
        s.record(&outcome("anthropic", "claude", 1000, 600), at(T0), &cfg);
        s.record(&outcome("openai", "gpt", 500, 500), at(T0 + 1), &cfg);

        assert_eq!(s.lifetime.requests, 2);
        assert_eq!(s.lifetime.tokens_saved, 400);
        assert!((s.lifetime.compression_savings_usd - 0.0004).abs() < 1e-9);
        assert_eq!(s.by_provider["anthropic"].requests, 1);
        assert_eq!(s.by_provider["openai"].tokens_saved, 0);
        assert_eq!(s.by_model["claude"].tokens_saved, 400);
        // history only gets a point when tokens were actually saved
        assert_eq!(s.history.len(), 1);
        assert_eq!(s.history[0].provider, "anthropic");
    }

    #[test]
    fn failed_request_counts_but_accrues_no_savings() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig::default();
        let mut o = outcome("bedrock", "claude", 100, 80); // would save 20 if it succeeded
        o.failed = true;
        s.record(&o, at(T0), &cfg);
        // Counted as a request + a failure...
        assert_eq!(s.lifetime.requests, 1);
        assert_eq!(s.requests_failed, 1);
        // ...but no savings booked anywhere (lifetime, buckets, history, coverage).
        assert_eq!(s.lifetime.tokens_saved, 0);
        assert_eq!(s.lifetime.compression_savings_usd, 0.0);
        assert_eq!(s.lifetime.requests_compressed, 0);
        assert_eq!(s.total_input_tokens, 0);
        assert_eq!(s.by_provider["bedrock"].requests, 1);
        assert_eq!(s.by_provider["bedrock"].tokens_saved, 0);
        assert!(s.history.is_empty());
        // The recent feed still logs the attempt, tagged failed, with the real
        // attempted savings.
        assert_eq!(s.recent.len(), 1);
        assert!(s.recent[0].failed);
        assert_eq!(s.recent[0].tokens_saved, 20);
    }

    #[test]
    fn requests_compressed_tracks_coverage() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig::default();
        // One request that actually compressed, one passthrough (no reduction).
        s.record(&outcome("anthropic", "claude", 100, 60), at(T0), &cfg);
        s.record(&outcome("anthropic", "claude", 100, 100), at(T0), &cfg);
        assert_eq!(s.lifetime.requests, 2);
        assert_eq!(s.lifetime.requests_compressed, 1);
        assert_eq!(s.display_session.requests_compressed, 1);
    }

    #[test]
    fn missing_labels_collapse_to_unknown_bucket() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig::default();
        s.record(&outcome("", "", 100, 40), at(T0), &cfg);
        assert_eq!(s.by_provider[UNKNOWN].requests, 1);
        assert_eq!(s.by_model[UNKNOWN].tokens_saved, 60);
    }

    #[test]
    fn session_rolls_over_after_inactivity() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig {
            session_inactivity: Duration::from_secs(10),
            ..StoreConfig::default()
        };
        s.record(&outcome("anthropic", "claude", 100, 50), at(T0), &cfg);
        assert_eq!(s.display_session.requests, 1);
        // within window → same session
        s.record(&outcome("anthropic", "claude", 100, 50), at(T0 + 5), &cfg);
        assert_eq!(s.display_session.requests, 2);
        // beyond window → fresh session
        s.record(&outcome("anthropic", "claude", 100, 50), at(T0 + 100), &cfg);
        assert_eq!(s.display_session.requests, 1);
        assert_eq!(s.display_session.started_at, Some(to_rfc3339(at(T0 + 100))));
    }

    #[test]
    fn history_is_bounded() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig {
            max_history: 3,
            ..StoreConfig::default()
        };
        for i in 0..10 {
            s.record(&outcome("anthropic", "claude", 100, 50), at(T0 + i), &cfg);
        }
        assert_eq!(s.history.len(), 3);
        // newest retained
        assert_eq!(s.history.last().unwrap().timestamp, to_rfc3339(at(T0 + 9)));
    }

    fn hp(ts: &str, saved: u64, usd: f64) -> HistoryPoint {
        HistoryPoint {
            timestamp: ts.to_string(),
            provider: "bedrock".to_string(),
            model: "claude".to_string(),
            total_tokens_saved: saved,
            compression_savings_usd: usd,
        }
    }

    #[test]
    fn history_json_rolls_up_by_day() {
        let mut s = SavingsState::default();
        s.lifetime.requests = 3;
        s.lifetime.tokens_saved = 600;
        s.lifetime.compression_savings_usd = 0.012;
        s.lifetime.cache_savings_usd = 0.004;
        s.history = vec![
            hp("2026-06-18T00:10:00Z", 100, 0.002),
            hp("2026-06-18T05:30:00Z", 300, 0.007), // same day -> collapses, last wins
            hp("2026-06-19T01:00:00Z", 600, 0.012), // new day
        ];
        let v = build_history_json(&s);
        assert_eq!(v["lifetime"]["requests"], 3);
        assert_eq!(v["lifetime"]["tokens_saved"], 600);
        assert_eq!(v["lifetime"]["cache_savings_usd"], 0.004);
        // raw checkpoints preserved
        assert_eq!(v["history"].as_array().unwrap().len(), 3);
        // daily rollup: 06-18 collapses to its last point (300), then 06-19
        let daily = v["series"]["daily"].as_array().unwrap();
        assert_eq!(daily.len(), 2);
        assert_eq!(daily[0]["date"], "2026-06-18");
        assert_eq!(daily[0]["tokens_saved"], 300);
        assert_eq!(daily[1]["date"], "2026-06-19");
        assert_eq!(daily[1]["tokens_saved"], 600);
    }

    #[test]
    fn record_populates_recent_feed_with_backfilled_id() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig::default();

        let mut o = outcome("bedrock", "claude", 100, 60); // saved 40
        o.request_id = "req-abc".to_string();
        o.latency_ms = 12;
        o.output_tokens = 5;
        s.record(&o, at(T0), &cfg);

        // Second request with a blank id and zero input → backfilled id, 0%.
        s.record(&outcome("openai", "gpt", 0, 0), at(T0 + 1), &cfg);

        assert_eq!(s.recent.len(), 2);
        let first = &s.recent[0];
        assert_eq!(first.request_id, "req-abc");
        assert_eq!(first.input_tokens_original, 100);
        assert_eq!(first.input_tokens_optimized, 60);
        assert_eq!(first.tokens_saved, 40);
        assert_eq!(first.output_tokens, 5);
        assert_eq!(first.total_latency_ms, 12);
        assert!((first.savings_percent - 40.0).abs() < 0.01);
        // The blank id is backfilled from the lifetime counter (2 at this point).
        assert_eq!(s.recent[1].request_id, "req-2");
        assert_eq!(s.recent[1].savings_percent, 0.0);

        // JSON shape: newest first, with the expanded-row defaults present.
        let v = recent_requests_json(&s);
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["request_id"], "req-2");
        assert_eq!(arr[1]["request_id"], "req-abc");
        assert_eq!(arr[1]["total_latency_ms"], 12);
        assert!(arr[0]["transforms_applied"].is_array());
        assert!(arr[0]["waste_signals"].is_object());
    }

    #[test]
    fn recent_feed_is_bounded() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig {
            max_response_history: 3,
            ..StoreConfig::default()
        };
        for i in 0..10 {
            s.record(&outcome("bedrock", "claude", 10, 5), at(T0 + i), &cfg);
        }
        assert_eq!(s.recent.len(), 3);
    }

    #[test]
    fn history_json_empty_is_well_formed() {
        let v = build_history_json(&SavingsState::default());
        assert_eq!(v["lifetime"]["requests"], 0);
        assert_eq!(v["history"].as_array().unwrap().len(), 0);
        assert_eq!(v["series"]["daily"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn session_view_collapses_when_expired() {
        let mut s = SavingsState::default();
        let cfg = StoreConfig {
            session_inactivity: Duration::from_secs(10),
            ..StoreConfig::default()
        };
        s.record(&outcome("anthropic", "claude", 100, 50), at(T0), &cfg);
        let live = session_view(&s, at(T0 + 5), &cfg);
        assert_eq!(live.requests, 1);
        let expired = session_view(&s, at(T0 + 999), &cfg);
        assert_eq!(expired.requests, 0);
    }

    #[test]
    fn display_session_savings_percent() {
        let mut session = DisplaySession::default();
        assert_eq!(session.savings_percent(), 0.0);
        session.tokens_saved = 25;
        session.total_input_tokens = 75;
        assert!((session.savings_percent() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn bucket_reduction_pct() {
        let mut b = Bucket::default();
        assert_eq!(b.reduction_pct(), 0.0);
        b.tokens_before = 200;
        b.tokens_saved = 50;
        assert!((b.reduction_pct() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn round_helpers() {
        assert_eq!(round6(0.123_456_789), 0.123_457);
        assert_eq!(round2(12.345), 12.35);
        // Non-finite input is sanitized to 0.0.
        assert_eq!(round6(f64::NAN), 0.0);
        assert_eq!(round6(f64::INFINITY), 0.0);
        assert_eq!(round2(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn record_sanitizes_non_finite_savings() {
        // A garbage (infinite) price must not produce a non-finite aggregate.
        let mut s = SavingsState::default();
        let cfg = StoreConfig::default();
        let mut o = outcome("anthropic", "claude", 1000, 400);
        o.input_cost_per_token = f64::INFINITY;
        s.record(&o, at(T0), &cfg);
        assert!(s.lifetime.compression_savings_usd.is_finite());
        assert_eq!(s.lifetime.compression_savings_usd, 0.0);
    }

    #[test]
    fn save_state_errors_when_temp_path_is_blocked() {
        // The atomic write goes to `<path>.json.tmp.<pid>`. If a directory
        // already occupies that temp path, the temp-file write fails → exercises
        // the write `?` error path (distinct from the rename and mkdir edges).
        let dir = std::env::temp_dir().join(format!("hr-stats-tmpblock-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("savings.json");
        // Pre-create a directory exactly where the temp file would be written.
        std::fs::create_dir_all(path.with_extension(format!("json.tmp.{}", std::process::id())))
            .unwrap();
        assert!(save_state(&path, &SavingsState::default()).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rfc3339_round_trip() {
        let t = at(T0);
        let s = to_rfc3339(t);
        assert_eq!(parse_rfc3339(&s), Some(t));
        assert_eq!(parse_rfc3339("not-a-date"), None);
    }

    #[test]
    fn stats_json_shape_and_values() {
        let store = SavingsStore::in_memory();
        store.record(&outcome("anthropic", "claude", 1000, 600), at(T0));
        store.record(&outcome("bedrock", "claude", 500, 300), at(T0 + 1));
        let v = store.stats_json(at(T0 + 2));

        assert_eq!(v["requests"]["total"], 2);
        assert_eq!(v["tokens"]["saved"], 600); // 400 + 200
        assert_eq!(v["tokens"]["input"], 900); // 600 + 300 forwarded
        assert_eq!(v["requests"]["by_provider"]["anthropic"], 1);
        assert!(v["cost"]["compression_savings_usd"].as_f64().unwrap() > 0.0);
        assert_eq!(v["persistent_savings"]["lifetime"]["tokens_saved"], 600);
        assert_eq!(v["cost"]["per_model"]["claude"]["requests"], 2);
        assert_eq!(v["display_session"]["requests"], 2);
    }

    #[test]
    fn stats_json_empty_store_is_well_formed() {
        let store = SavingsStore::in_memory();
        let v = store.stats_json(at(T0));
        assert_eq!(v["requests"]["total"], 0);
        assert_eq!(v["tokens"]["savings_percent"], 0.0);
        assert_eq!(v["display_session"]["requests"], 0);
    }

    #[test]
    fn persistence_round_trip_atomic() {
        let dir = std::env::temp_dir().join(format!("hr-stats-{}", uuid::Uuid::new_v4()));
        let path = dir.join("proxy_savings.json");
        let cfg = StoreConfig::default();
        {
            let store = SavingsStore::with_path(&path, cfg.clone());
            store.record(&outcome("anthropic", "claude", 1000, 400), at(T0));
            // Persistence is off the request path now — flush explicitly (in
            // production a background task / the shutdown hook does this).
            store.flush();
        }
        assert!(path.exists());
        // reload picks up persisted lifetime
        let store2 = SavingsStore::with_path(&path, cfg);
        let snap = store2.snapshot();
        assert_eq!(snap.lifetime.tokens_saved, 600);
        assert_eq!(snap.lifetime.requests, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_state_tolerates_missing_and_garbage() {
        let missing = Path::new("/nonexistent/does/not/exist/savings.json");
        assert_eq!(load_state(missing), SavingsState::default());

        let dir = std::env::temp_dir().join(format!("hr-stats-bad-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, b"{ not valid json ]").unwrap();
        assert_eq!(load_state(&path), SavingsState::default());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sanitize_forces_schema_version() {
        let s = SavingsState {
            schema_version: 999,
            ..SavingsState::default()
        };
        assert_eq!(s.sanitize().schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn record_backfills_started_at_for_partial_session() {
        // A persisted session can have a recent last_activity_at but a missing
        // started_at (older/partial file). Recording within the window must NOT
        // roll over, and must backfill started_at instead of leaving it null.
        let mut s = SavingsState {
            display_session: DisplaySession {
                last_activity_at: Some(to_rfc3339(at(T0))),
                started_at: None,
                ..DisplaySession::default()
            },
            ..SavingsState::default()
        };
        let cfg = StoreConfig::default();
        s.record(&outcome("anthropic", "claude", 100, 50), at(T0 + 1), &cfg);
        assert_eq!(s.display_session.requests, 1); // same session, not reset
        assert_eq!(s.display_session.started_at, Some(to_rfc3339(at(T0 + 1))));
    }

    #[test]
    fn ensure_parent_dir_handles_all_path_shapes() {
        // Real nested parent that does not yet exist → created.
        let dir = std::env::temp_dir().join(format!("hr-parent-{}", uuid::Uuid::new_v4()));
        let nested = dir.join("a/b/file.json");
        ensure_parent_dir(&nested).unwrap();
        assert!(nested.parent().unwrap().is_dir());
        std::fs::remove_dir_all(&dir).ok();

        // Bare filename → empty parent → no-op, still Ok.
        ensure_parent_dir(Path::new("bare_no_write.json")).unwrap();
        assert!(!Path::new("bare_no_write.json").exists());

        // Root path → no parent → no-op, still Ok.
        ensure_parent_dir(Path::new("/")).unwrap();
    }

    #[test]
    fn in_memory_store_does_not_persist() {
        let store = SavingsStore::in_memory();
        assert!(!store.is_persistent());
        store.record(&outcome("anthropic", "claude", 100, 50), at(T0));
        // No path → flush is a no-op even though dirty; snapshot still reflects
        // the record.
        store.flush();
        assert_eq!(store.snapshot().lifetime.requests, 1);
    }

    #[test]
    fn lock_recovers_from_a_poisoned_mutex() {
        // A panic while holding the state lock must not wedge the request path:
        // record()/snapshot() recover the poisoned guard instead of panicking.
        let store = SavingsStore::in_memory();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = store.state.lock().unwrap();
            panic!("intentional poison");
        }));
        assert!(store.state.is_poisoned());
        store.record(&outcome("anthropic", "claude", 100, 50), at(T0));
        assert_eq!(store.snapshot().lifetime.requests, 1);
    }

    #[test]
    fn flush_rearms_dirty_when_write_fails() {
        // A transient write failure must not drop the accumulated state: flush
        // re-arms the dirty flag so the next flush retries.
        let dir = std::env::temp_dir().join(format!("hr-rearm-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.json");
        // Make the target a directory so save_state's rename-over fails.
        std::fs::create_dir(&path).unwrap();

        let store = SavingsStore::with_path(&path, StoreConfig::default());
        store.record(&outcome("anthropic", "claude", 100, 50), at(T0));
        store.flush(); // write fails (target is a dir) → dirty re-armed
        assert!(
            path.is_dir(),
            "save_state should have failed, leaving the dir"
        );

        // Clear the obstruction; the re-armed dirty flag means the next flush
        // still persists the earlier record (it was not lost).
        std::fs::remove_dir(&path).unwrap();
        store.flush();
        assert_eq!(load_state(&path).lifetime.requests, 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn record_does_no_disk_io_until_flush() {
        // The request path must never touch disk. `record` only marks the store
        // dirty; `flush` (called off the request worker) does the actual write.
        let dir = std::env::temp_dir().join(format!("hr-flush-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.json");
        let store = SavingsStore::with_path(&path, StoreConfig::default());
        assert!(store.is_persistent());
        let o = outcome("anthropic", "claude", 100, 50);

        // record() writes nothing to disk.
        store.record(&o, at(T0));
        assert!(!path.exists(), "record must not perform disk I/O");
        assert_eq!(store.snapshot().lifetime.requests, 1);

        // flush() performs the (single) write.
        store.flush();
        assert_eq!(load_state(&path).lifetime.requests, 1);

        // flush() with nothing dirty is a no-op — it does not rewrite the file.
        std::fs::remove_file(&path).unwrap();
        store.flush();
        assert!(
            !path.exists(),
            "flush with no pending changes must not write"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_state_errors_when_target_is_a_directory() {
        // Renaming the temp file over an existing directory fails → exercises
        // the rename `?` error path. SavingsStore::persist swallows this (best
        // effort), but save_state surfaces it.
        let dir = std::env::temp_dir().join(format!("hr-stats-dir-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = save_state(&dir, &SavingsState::default());
        assert!(err.is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_state_errors_when_parent_is_a_file() {
        // A parent path component that is a regular file makes create_dir_all
        // fail → exercises the ensure_parent_dir `?` error path.
        let base = std::env::temp_dir().join(format!("hr-stats-file-{}", uuid::Uuid::new_v4()));
        std::fs::write(&base, b"i am a file").unwrap();
        let under_file = base.join("nested/savings.json");
        let err = save_state(&under_file, &SavingsState::default());
        assert!(err.is_err());
        std::fs::remove_file(&base).ok();
    }

    #[test]
    fn persist_is_best_effort_and_never_panics() {
        // Store pointed at an unwritable path (parent is a file). record() must
        // not panic even though the background persist fails.
        let base = std::env::temp_dir().join(format!("hr-stats-be-{}", uuid::Uuid::new_v4()));
        std::fs::write(&base, b"file").unwrap();
        let store = SavingsStore::with_path(base.join("x/y.json"), StoreConfig::default());
        store.record(&outcome("anthropic", "claude", 100, 50), at(T0));
        assert_eq!(store.snapshot().lifetime.requests, 1);
        std::fs::remove_file(&base).ok();
    }
}
