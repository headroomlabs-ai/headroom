//! Configuration for the proxy: CLI flags + env vars.

use clap::{Parser, ValueEnum};
use std::net::SocketAddr;
use std::time::Duration;
use url::Url;

/// Compression mode policy for the `/v1/messages` endpoint.
///
/// Drives whether `compress_anthropic_request` does any work. PR-A1
/// (Phase A lockdown) wires the flag in but both modes currently
/// passthrough — `live_zone` parses-but-warns until Phase B PR-B2
/// fills in the live-zone-only block dispatcher.
///
/// We do NOT add an `icm` mode (the deleted code path) or a
/// `passthrough` alias for `off` — those names are misleading. The
/// only legal values are `off` (compression disabled) and `live_zone`
/// (compress only the live-zone blocks; not yet implemented).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "snake_case")]
pub enum CompressionMode {
    /// Compression disabled. Body forwards byte-equal to upstream.
    /// This is the default; Phase B will switch the default to
    /// `live_zone` once that mode is implemented.
    Off,
    /// Compress only live-zone blocks (latest user message,
    /// latest tool/function/shell/patch outputs). NOT YET IMPLEMENTED:
    /// in PR-A1 this falls through to passthrough behaviour with a
    /// loud warning. Phase B PR-B2 wires in the actual dispatcher.
    LiveZone,
}

/// Policy for stripping internal `x-headroom-*` headers from upstream-bound
/// requests (PR-A5, fixes P5-49).
///
/// When `enabled` (default), every header whose name starts with
/// `x-headroom-` is dropped before the upstream call. Stops fingerprinting
/// of the proxy via subscription-revocation flags (`x-headroom-bypass`,
/// `x-headroom-mode`, etc.) and prevents leakage of internal user-id /
/// stack / base-url headers.
///
/// When `disabled`, internal headers are forwarded verbatim. This is an
/// explicit operator opt-in for diagnostic shadow tracing — NOT a fallback.
/// Document the trade-off in `docs/configuration.md` before flipping this.
///
/// Source priority: CLI flag → `HEADROOM_PROXY_STRIP_INTERNAL_HEADERS`
/// env var → default (`enabled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "snake_case")]
pub enum StripInternalHeaders {
    /// Strip every `x-headroom-*` header from upstream-bound requests.
    /// Default. Operationally safe.
    Enabled,
    /// Forward `x-headroom-*` to upstream verbatim. Diagnostic-only;
    /// exposes internal flags to the upstream and reveals the proxy.
    Disabled,
}

impl StripInternalHeaders {
    /// Stable snake_case name suitable for log fields.
    pub fn as_str(self) -> &'static str {
        match self {
            StripInternalHeaders::Enabled => "enabled",
            StripInternalHeaders::Disabled => "disabled",
        }
    }

    /// Convenience: is the strip switched on?
    pub fn is_enabled(self) -> bool {
        matches!(self, StripInternalHeaders::Enabled)
    }
}

/// Policy for automatically deriving `frozen_message_count` from the
/// customer's `cache_control` markers (PR-A4).
///
/// When `enabled` (default), the live-zone dispatcher will walk
/// `messages[*].content[*].cache_control` and bump the floor below
/// which compression is forbidden. When `disabled`, the floor stays
/// at 0 regardless of markers — Phase B's dispatcher will then treat
/// every message as live-zone, which is dangerous in production but
/// useful for benchmarking the cache-control machinery.
///
/// `system` and `tools[*]` markers never bump `frozen_count` because
/// those fields are *always* part of the cache hot zone (invariant I2);
/// they're guaranteed-immutable independently of marker placement.
///
/// Source priority: CLI flag → `HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN`
/// env var → default (`enabled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "snake_case")]
pub enum CacheControlAutoFrozen {
    /// Walk customer `cache_control` markers and derive
    /// `frozen_message_count` automatically. Default.
    Enabled,
    /// Ignore customer `cache_control` markers when deriving
    /// `frozen_message_count`; the function returns 0 regardless of
    /// what the body contains. Intended for benchmarking and the
    /// "no automatic floor" testing path; not for production use.
    Disabled,
}

impl CacheControlAutoFrozen {
    /// Stable snake_case name suitable for log fields. Mirrors
    /// `CompressionMode::as_str` so the two policy fields render
    /// identically in JSON tracing output.
    pub fn as_str(self) -> &'static str {
        match self {
            CacheControlAutoFrozen::Enabled => "enabled",
            CacheControlAutoFrozen::Disabled => "disabled",
        }
    }

    /// Convenience: is the auto-frozen derivation switched on? Most
    /// callers want the boolean rather than pattern-matching on the
    /// enum.
    pub fn is_enabled(self) -> bool {
        matches!(self, CacheControlAutoFrozen::Enabled)
    }
}

impl CompressionMode {
    /// Stable snake_case name suitable for log fields. Avoids relying
    /// on `Debug` (which renders `Off`/`LiveZone`) or `Display`
    /// (which we don't implement to keep `ValueEnum` the single
    /// source of truth for stringification).
    pub fn as_str(self) -> &'static str {
        match self {
            CompressionMode::Off => "off",
            CompressionMode::LiveZone => "live_zone",
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(
    name = "headroom-proxy",
    version,
    about = "Headroom transparent reverse proxy"
)]
pub struct CliArgs {
    /// Address the proxy listens on (e.g. 0.0.0.0:8787).
    #[arg(long, env = "HEADROOM_PROXY_LISTEN", default_value = "0.0.0.0:8787")]
    pub listen: SocketAddr,

    /// Upstream base URL the proxy forwards to (e.g. http://127.0.0.1:8788).
    /// REQUIRED — there is no default; we want operators to be explicit.
    #[arg(long, env = "HEADROOM_PROXY_UPSTREAM")]
    pub upstream: Url,

    /// End-to-end timeout for a single upstream request (long, since LLM
    /// streams may run for many minutes).
    #[arg(long, default_value = "600s", value_parser = parse_duration)]
    pub upstream_timeout: Duration,

    /// TCP/TLS connect timeout for upstream.
    #[arg(long, default_value = "10s", value_parser = parse_duration)]
    pub upstream_connect_timeout: Duration,

    /// Max body size for buffered cases (does NOT bound streaming bodies).
    #[arg(long, default_value = "100MB", value_parser = parse_bytes)]
    pub max_body_bytes: u64,

    /// Log level / filter (RUST_LOG-style). Default: info.
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Rewrite the outgoing Host header to the upstream host (default).
    /// Pair with --no-rewrite-host to preserve the client-supplied Host.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub rewrite_host: bool,

    /// Convenience flag matching the spec; sets rewrite_host=false when present.
    #[arg(long = "no-rewrite-host", default_value_t = false)]
    pub no_rewrite_host: bool,

    /// Maximum time to wait for in-flight requests to finish on shutdown.
    #[arg(long, default_value = "30s", value_parser = parse_duration)]
    pub graceful_shutdown_timeout: Duration,

    /// Enable Headroom compression on LLM-shaped requests
    /// (currently: `POST /v1/messages` for Anthropic). When off,
    /// the proxy stays a pure streaming passthrough.
    ///
    /// Off by default so existing operators get unchanged behaviour
    /// and the integration-test harness doesn't need to opt out
    /// per-test. Operators wanting to demo the compressor pass
    /// `--compression` (or set `HEADROOM_PROXY_COMPRESSION=1`).
    #[arg(
        long = "compression",
        env = "HEADROOM_PROXY_COMPRESSION",
        default_value_t = false
    )]
    pub compression: bool,

    /// Maximum body size to buffer for compression. Bodies larger
    /// than this get forwarded unchanged. Defaults to `--max-body-bytes`
    /// when unset, so operators only need to tune one knob unless
    /// they have a specific reason to cap compression separately.
    #[arg(long, value_parser = parse_bytes)]
    pub compression_max_body_bytes: Option<u64>,

    /// Compression mode policy for `/v1/messages`.
    ///
    /// `off` (default): byte-faithful passthrough on every request.
    /// `live_zone`: PR-B2 wired the dispatcher; PR-B2's per-type
    /// compressors are no-ops, so the body still round-trips
    /// byte-equal until PR-B3+ (which fills the per-type table).
    /// The flag exists so the default can flip in one config
    /// change once `live_zone` is the safer choice on real traffic.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_COMPRESSION_MODE`
    /// env var → default (`off`).
    #[arg(
        long = "compression-mode",
        env = "HEADROOM_PROXY_COMPRESSION_MODE",
        value_enum,
        default_value_t = CompressionMode::Off,
    )]
    pub compression_mode: CompressionMode,

    /// Whether to derive `frozen_message_count` from customer
    /// `cache_control` markers in the request body (PR-A4).
    ///
    /// `enabled` (default): walk `messages[*].content[*].cache_control`
    /// and bump the floor for live-zone compression so any message
    /// the customer cache-pinned is left untouched. `disabled`: skip
    /// the walk; the floor stays at 0. The off switch exists for
    /// benchmark setups that want to measure compression independent
    /// of marker placement; it is NOT recommended for production.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN`
    /// env var → default (`enabled`).
    #[arg(
        long = "cache-control-auto-frozen",
        env = "HEADROOM_PROXY_CACHE_CONTROL_AUTO_FROZEN",
        value_enum,
        default_value_t = CacheControlAutoFrozen::Enabled,
    )]
    pub cache_control_auto_frozen: CacheControlAutoFrozen,

    /// Strip internal `x-headroom-*` headers from upstream-bound
    /// requests (PR-A5, fixes P5-49). Default `enabled`. The `disabled`
    /// path is operator opt-in for diagnostic shadow tracing only —
    /// NOT a fallback per realignment build constraint #4.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_STRIP_INTERNAL_HEADERS`
    /// env var → default (`enabled`).
    #[arg(
        long = "strip-internal-headers",
        env = "HEADROOM_PROXY_STRIP_INTERNAL_HEADERS",
        value_enum,
        default_value_t = StripInternalHeaders::Enabled,
    )]
    pub strip_internal_headers: StripInternalHeaders,

    /// Phase C PR-C4: enable the `/v1/responses` SSE streaming
    /// pipeline. When `true` (default), `Accept: text/event-stream`
    /// requests on `/v1/responses` flow through the byte-level SSE
    /// framer + Responses state-machine telemetry tee that PR-C1
    /// wired into `forward_http`'s response stream. When `false`,
    /// the streaming pipeline is bypassed and the SSE response is
    /// proxied as opaque bytes (no framer, no state machine,
    /// strictly fewer logs). Bypass exists ONLY for emergency
    /// rollback of the streaming pipeline without flipping the
    /// global `--compression` switch — it is NOT a fallback path.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_ENABLE_RESPONSES_STREAMING`
    /// env var → default (`true`).
    #[arg(
        long = "enable-responses-streaming",
        env = "HEADROOM_PROXY_ENABLE_RESPONSES_STREAMING",
        default_value_t = true,
        action = clap::ArgAction::Set,
    )]
    pub enable_responses_streaming: bool,

    /// Phase C PR-C4: enable the `/v1/conversations*` passthrough
    /// surface. When `true` (default), the proxy mounts explicit
    /// axum routes for OpenAI's Conversations API
    /// (`POST/GET/DELETE /v1/conversations/...` and the nested
    /// `/items` paths) and forwards every request upstream
    /// byte-equal with structured-log instrumentation
    /// (`event = "conversations_passthrough_pr_c4"`). When `false`,
    /// requests still reach upstream via the catch-all but lose
    /// the per-route logging. Compression on conversation items
    /// is NOT performed in this PR — `enable_conversations_passthrough`
    /// is strictly an instrumentation switch.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_ENABLE_CONVERSATIONS_PASSTHROUGH`
    /// env var → default (`true`).
    #[arg(
        long = "enable-conversations-passthrough",
        env = "HEADROOM_PROXY_ENABLE_CONVERSATIONS_PASSTHROUGH",
        default_value_t = true,
        action = clap::ArgAction::Set,
    )]
    pub enable_conversations_passthrough: bool,

    /// Phase D PR-D1: enable the native Bedrock InvokeModel route.
    /// When `true` (default), `POST /model/{model_id}/invoke` is
    /// handled by the Rust `bedrock::invoke` handler — Anthropic-shape
    /// bodies run through the live-zone compression path and the
    /// proxy re-signs the request with SigV4 before forwarding to
    /// the configured Bedrock endpoint. When `false`, the routes are
    /// not mounted and requests fall through to the catch-all
    /// (which forwards to `--upstream` byte-equal but does NOT
    /// re-sign — operators MUST run an unsigned upstream that
    /// happens to know what to do, otherwise this fails closed).
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_ENABLE_BEDROCK_NATIVE`
    /// env var → default (`true`).
    #[arg(
        long = "enable-bedrock-native",
        env = "HEADROOM_PROXY_ENABLE_BEDROCK_NATIVE",
        default_value_t = true,
        action = clap::ArgAction::Set,
    )]
    pub enable_bedrock_native: bool,

    /// AWS region to use when signing Bedrock requests. Default
    /// `us-east-1`. The Bedrock endpoint URL derived from this
    /// region is `https://bedrock-runtime.{region}.amazonaws.com`
    /// (override via `--bedrock-endpoint` for FIPS or VPC endpoints).
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_BEDROCK_REGION`
    /// env var → `AWS_REGION` env var → default (`us-east-1`).
    #[arg(
        long = "bedrock-region",
        env = "HEADROOM_PROXY_BEDROCK_REGION",
        default_value = "us-east-1"
    )]
    pub bedrock_region: String,

    /// Bedrock endpoint base URL. When unset (the common case), the
    /// proxy derives `https://bedrock-runtime.{bedrock_region}.amazonaws.com`
    /// from the configured region. Override for FIPS endpoints
    /// (`bedrock-runtime-fips.{region}.amazonaws.com`), VPC endpoints,
    /// or local-mock test setups.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_BEDROCK_ENDPOINT`
    /// env var → derived-from-region.
    #[arg(long = "bedrock-endpoint", env = "HEADROOM_PROXY_BEDROCK_ENDPOINT")]
    pub bedrock_endpoint: Option<Url>,

    /// AWS profile name passed to the `aws-config` default credential
    /// chain. When unset, the chain uses the default behaviour
    /// (env vars → `[default]` profile → IMDS / ECS task role).
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_AWS_PROFILE`
    /// env var → `AWS_PROFILE` env var → default chain.
    #[arg(long = "aws-profile", env = "HEADROOM_PROXY_AWS_PROFILE")]
    pub aws_profile: Option<String>,

    /// Phase D PR-D2: validate the prelude + message CRC32 on each
    /// inbound Bedrock EventStream frame. Default `true` — production
    /// MUST validate. Operators flip to `false` ONLY for debugging a
    /// suspected wire-format issue (e.g. a corrupt-but-cooperative
    /// upstream that emits invalid CRCs intentionally). When disabled,
    /// the proxy still parses message boundaries; it just doesn't
    /// reject on CRC mismatch. Per project policy, every flag flip
    /// is logged at app-build time.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_BEDROCK_VALIDATE_EVENTSTREAM_CRC`
    /// env var → default (`true`).
    #[arg(
        long = "bedrock-validate-eventstream-crc",
        env = "HEADROOM_PROXY_BEDROCK_VALIDATE_EVENTSTREAM_CRC",
        default_value_t = true,
        action = clap::ArgAction::Set,
    )]
    pub bedrock_validate_eventstream_crc: bool,

    /// Phase D PR-D4: GCP Vertex region for the publisher path
    /// (`{region}-aiplatform.googleapis.com`). Default `us-central1`
    /// (matches the GCP-published default region for Anthropic
    /// publisher models). The proxy does NOT auto-construct the
    /// regional URL — that's an `--upstream` decision the operator
    /// makes once at startup. This flag is exposed for structured
    /// logging + observability so dashboards can group Vertex traffic
    /// by region without parsing the upstream URL.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_VERTEX_REGION`
    /// env var → default (`us-central1`).
    #[arg(
        long = "vertex-region",
        env = "HEADROOM_PROXY_VERTEX_REGION",
        default_value = "us-central1"
    )]
    pub vertex_region: String,

    /// Phase D PR-D4: OAuth scope to request from GCP ADC. Defaults
    /// to `cloud-platform`, the broad scope `gcloud` itself uses for
    /// ADC. Operators with tighter IAM postures can scope down to
    /// `cloud-platform.read-only` etc., but Vertex `:rawPredict`
    /// requires write so most deployments use the default.
    ///
    /// Source priority: CLI flag → `HEADROOM_PROXY_VERTEX_ADC_SCOPE`
    /// env var → default (`cloud-platform`).
    #[arg(
        long = "vertex-adc-scope",
        env = "HEADROOM_PROXY_VERTEX_ADC_SCOPE",
        default_value = "https://www.googleapis.com/auth/cloud-platform"
    )]
    pub vertex_adc_scope: String,
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    humantime::parse_duration(s).map_err(|e| format!("invalid duration `{s}`: {e}"))
}

fn parse_bytes(s: &str) -> Result<u64, String> {
    s.parse::<bytesize::ByteSize>()
        .map(|b| b.as_u64())
        .map_err(|e| format!("invalid byte size `{s}`: {e}"))
}

/// Resolved configuration used by the running server.
#[derive(Debug, Clone)]
pub struct Config {
    pub listen: SocketAddr,
    pub upstream: Url,
    pub upstream_timeout: Duration,
    pub upstream_connect_timeout: Duration,
    pub max_body_bytes: u64,
    pub log_level: String,
    pub rewrite_host: bool,
    pub graceful_shutdown_timeout: Duration,
    /// Master switch for the LLM compression interceptor. When `false`,
    /// the proxy is pure streaming passthrough and never buffers a body.
    pub compression: bool,
    /// Effective ceiling for compression-time body buffering.
    /// Inherits `max_body_bytes` when not overridden. Bodies larger
    /// than this still forward, just unchanged.
    pub compression_max_body_bytes: u64,
    /// Policy mode for compression on `/v1/messages`. PR-A1 lockdown:
    /// both `Off` and `LiveZone` result in byte-faithful passthrough;
    /// `LiveZone` additionally emits a `tracing::warn!` per request
    /// because the dispatcher isn't implemented yet (Phase B PR-B2
    /// fills this in).
    pub compression_mode: CompressionMode,
    /// Whether the live-zone dispatcher derives `frozen_message_count`
    /// automatically from customer `cache_control` markers. PR-A4
    /// adds the derivation function (`compute_frozen_count`); Phase
    /// B's dispatcher consumes the resolved value here.
    pub cache_control_auto_frozen: CacheControlAutoFrozen,
    /// Whether to strip internal `x-headroom-*` headers from
    /// upstream-bound requests. PR-A5 default-on guard against
    /// fingerprinting / leakage of internal flags.
    pub strip_internal_headers: StripInternalHeaders,
    /// PR-C4: enable the `/v1/responses` streaming pipeline (SSE
    /// state-machine + telemetry tee). Default `true`.
    pub enable_responses_streaming: bool,
    /// PR-C4: enable the `/v1/conversations*` passthrough surface
    /// (per-route axum handlers with explicit instrumentation).
    /// Default `true`. Strictly an instrumentation switch — does
    /// NOT gate compression of conversation items (that's
    /// C5+/B-phase territory).
    pub enable_conversations_passthrough: bool,
    /// PR-D1: enable the native Bedrock InvokeModel route. Default
    /// `true`. When disabled, the explicit Rust handlers are not
    /// mounted; operators relying on the Python LiteLLM converter
    /// keep their existing path.
    pub enable_bedrock_native: bool,
    /// PR-D1: AWS region used to sign Bedrock requests + (when no
    /// explicit endpoint is set) derive the Bedrock endpoint URL.
    pub bedrock_region: String,
    /// PR-D1: Bedrock endpoint base URL. `None` means
    /// "derive from region" (`https://bedrock-runtime.{region}.amazonaws.com`).
    pub bedrock_endpoint: Option<Url>,
    /// PR-D1: optional AWS profile name. When `None`, the default
    /// credential chain (env → `[default]` profile → IMDS) is used.
    pub aws_profile: Option<String>,
    /// PR-D2: validate prelude + message CRC32 on inbound Bedrock
    /// EventStream frames. Default `true`. Off only for debugging.
    pub bedrock_validate_eventstream_crc: bool,
    /// PR-D4: GCP Vertex region tag (e.g. `us-central1`). Surfaced
    /// in structured logs only — the actual upstream URL comes from
    /// `Config::upstream`. Operators set this so observability
    /// dashboards can group Vertex traffic by region.
    pub vertex_region: String,
    /// PR-D4: GCP ADC OAuth scope used when fetching the bearer
    /// token. Default `https://www.googleapis.com/auth/cloud-platform`.
    pub vertex_adc_scope: String,
}

impl Config {
    pub fn from_cli(args: CliArgs) -> Self {
        let rewrite_host = if args.no_rewrite_host {
            false
        } else {
            args.rewrite_host
        };
        let compression_max_body_bytes = args
            .compression_max_body_bytes
            .unwrap_or(args.max_body_bytes);
        Self {
            listen: args.listen,
            upstream: args.upstream,
            upstream_timeout: args.upstream_timeout,
            upstream_connect_timeout: args.upstream_connect_timeout,
            max_body_bytes: args.max_body_bytes,
            log_level: args.log_level,
            rewrite_host,
            graceful_shutdown_timeout: args.graceful_shutdown_timeout,
            compression: args.compression,
            compression_max_body_bytes,
            compression_mode: args.compression_mode,
            cache_control_auto_frozen: args.cache_control_auto_frozen,
            strip_internal_headers: args.strip_internal_headers,
            enable_responses_streaming: args.enable_responses_streaming,
            enable_conversations_passthrough: args.enable_conversations_passthrough,
            enable_bedrock_native: args.enable_bedrock_native,
            bedrock_region: args.bedrock_region,
            bedrock_endpoint: args.bedrock_endpoint,
            aws_profile: args.aws_profile,
            bedrock_validate_eventstream_crc: args.bedrock_validate_eventstream_crc,
            vertex_region: args.vertex_region,
            vertex_adc_scope: args.vertex_adc_scope,
        }
    }

    /// Test/library helper. Compression off by default — match
    /// production-default behaviour so existing tests stay unchanged.
    pub fn for_test(upstream: Url) -> Self {
        Self {
            listen: "127.0.0.1:0".parse().unwrap(),
            upstream,
            upstream_timeout: Duration::from_secs(60),
            upstream_connect_timeout: Duration::from_secs(5),
            max_body_bytes: 100 * 1024 * 1024,
            log_level: "warn".into(),
            rewrite_host: true,
            graceful_shutdown_timeout: Duration::from_secs(5),
            compression: false,
            compression_max_body_bytes: 100 * 1024 * 1024,
            compression_mode: CompressionMode::Off,
            // Match production default so the cache-control walker is
            // exercised under test without per-test opt-in.
            cache_control_auto_frozen: CacheControlAutoFrozen::Enabled,
            // Production default: strip internal `x-headroom-*` headers
            // from upstream-bound requests. Tests opt out per-case via
            // `start_proxy_with`.
            strip_internal_headers: StripInternalHeaders::Enabled,
            // PR-C4: streaming pipeline + conversations passthrough
            // both default-on so tests exercise the same paths
            // production traffic will hit.
            enable_responses_streaming: true,
            enable_conversations_passthrough: true,
            // PR-D1: bedrock route default-on so tests exercise
            // it without per-test opt-in. Tests that set
            // `bedrock_endpoint` to a wiremock URL get the full
            // sign-and-forward path.
            enable_bedrock_native: true,
            bedrock_region: "us-east-1".to_string(),
            bedrock_endpoint: None,
            aws_profile: None,
            // PR-D2: production default — validate every CRC. Tests
            // that exercise corruption paths flip this off per-case.
            bedrock_validate_eventstream_crc: true,
            // PR-D4: default Vertex region (used for log tagging
            // only; the upstream URL is `upstream`).
            vertex_region: "us-central1".to_string(),
            vertex_adc_scope: "https://www.googleapis.com/auth/cloud-platform".to_string(),
        }
    }
}
