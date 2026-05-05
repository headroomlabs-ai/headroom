//! Integration tests for `headroom_core::auth_mode::classify`.
//!
//! Exhaustive matrix per Phase F PR-F1 acceptance criteria. Bonus
//! cases cover the cross-precedence rules (Subscription UA wins over
//! OAuth bearer; vendor API-key headers map to PAYG).
//!
//! These are mirrored byte-for-byte by `tests/test_auth_mode.py` —
//! the Python helper MUST agree on every header set we test here.

use headroom_core::auth_mode::{classify, AuthMode};
use http::{HeaderMap, HeaderValue};

/// Helper: build a `HeaderMap` from `(name, value)` pairs in one
/// expression. Keeps the test bodies focused on the data, not the
/// `HeaderMap` boilerplate.
fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (name, value) in pairs {
        h.insert(
            http::header::HeaderName::from_bytes(name.as_bytes()).expect("valid header name"),
            HeaderValue::from_str(value).expect("valid header value"),
        );
    }
    h
}

// ── Required matrix ──────────────────────────────────────────────

#[test]
fn api_key_classified_payg() {
    // Anthropic PAYG: `Authorization: Bearer sk-ant-api03-XXX`.
    let h = headers(&[("authorization", "Bearer sk-ant-api03-abc123def456")]);
    assert_eq!(classify(&h), AuthMode::Payg);
}

#[test]
fn oauth_jwt_classified_oauth() {
    // Codex / Cursor OAuth bearer: classic 3-segment JWT.
    let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0In0.signaturepart";
    let h = headers(&[("authorization", &format!("Bearer {}", jwt))]);
    assert_eq!(classify(&h), AuthMode::OAuth);
}

#[test]
fn oauth_sk_ant_oat_classified_oauth() {
    // Claude Pro / Max OAuth: `Bearer sk-ant-oat-...`.
    let h = headers(&[("authorization", "Bearer sk-ant-oat-01-abc123def456")]);
    assert_eq!(classify(&h), AuthMode::OAuth);
}

#[test]
fn claude_code_ua_classified_subscription() {
    // Claude Code CLI: `User-Agent: claude-code/1.2.3 ...`.
    let h = headers(&[("user-agent", "claude-code/1.2.3 (darwin; arm64)")]);
    assert_eq!(classify(&h), AuthMode::Subscription);
}

#[test]
fn cursor_ua_classified_subscription() {
    // Cursor CLI: `User-Agent: cursor/1.0`.
    let h = headers(&[("user-agent", "cursor/1.0")]);
    assert_eq!(classify(&h), AuthMode::Subscription);
}

#[test]
fn no_auth_no_user_agent_default_payg() {
    // Empty headers → safest default is PAYG. The OAuth/bedrock
    // branch fires only when there's a positive non-Bearer auth
    // signal (next test). Choosing PAYG by default favors the
    // OSS-default workload (per-token cost saving).
    let h = HeaderMap::new();
    assert_eq!(classify(&h), AuthMode::Payg);
}

#[test]
fn bedrock_no_auth_classified_oauth() {
    // Bedrock SigV4: `Authorization: AWS4-HMAC-SHA256 Credential=...`.
    // Not a Bearer scheme; we treat all non-Bearer Authorization as
    // OAuth (passthrough-prefer).
    let h = headers(&[(
        "authorization",
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20260501/us-east-1/bedrock/aws4_request, \
         SignedHeaders=host;x-amz-date, Signature=fe5f80f77d5fa3beca038a248ff027",
    )]);
    assert_eq!(classify(&h), AuthMode::OAuth);
}

// ── Bonus matrix ──────────────────────────────────────────────────

#[test]
fn openai_payg_sk_classified_payg() {
    // OpenAI PAYG: `Authorization: Bearer sk-proj-...`.
    let h = headers(&[("authorization", "Bearer sk-proj-abcdef0123456789")]);
    assert_eq!(classify(&h), AuthMode::Payg);
}

#[test]
fn gemini_x_goog_api_key_classified_payg() {
    // Google Gemini API key as `x-goog-api-key`.
    let h = headers(&[("x-goog-api-key", "AIzaSyDUMMYKEY1234567890")]);
    assert_eq!(classify(&h), AuthMode::Payg);
}

#[test]
fn subscription_takes_precedence_over_oauth_token() {
    // Claude Code CLI happens to send a `Bearer sk-ant-oat-...`
    // token, but it IS a subscription client (rate-limited per
    // request count, never identify Headroom). UA wins.
    let h = headers(&[
        ("user-agent", "claude-code/1.5.0 (linux; x86_64)"),
        ("authorization", "Bearer sk-ant-oat-01-abc123"),
    ]);
    assert_eq!(classify(&h), AuthMode::Subscription);
}

// ── Edge cases (defensive coverage; not in the required matrix) ──

#[test]
fn anthropic_x_api_key_classified_payg() {
    // Anthropic API key style: `x-api-key: sk-ant-...`.
    let h = headers(&[("x-api-key", "sk-ant-api03-abcdef")]);
    assert_eq!(classify(&h), AuthMode::Payg);
}

#[test]
fn copilot_ua_classified_subscription() {
    // GitHub Copilot UA — covers the `github-copilot/` prefix.
    let h = headers(&[("user-agent", "GitHub-Copilot/1.0 (vscode)")]);
    assert_eq!(classify(&h), AuthMode::Subscription);
}

#[test]
fn anthropic_cli_ua_classified_subscription() {
    let h = headers(&[("user-agent", "anthropic-cli/0.9.1")]);
    assert_eq!(classify(&h), AuthMode::Subscription);
}

#[test]
fn antigravity_ua_classified_subscription() {
    let h = headers(&[("user-agent", "Antigravity/2.0 (build 1234)")]);
    assert_eq!(classify(&h), AuthMode::Subscription);
}

// ── Performance ──────────────────────────────────────────────────

/// Smoke perf check — a strict bench lives at
/// `crates/headroom-core/benches/auth_mode.rs`. This in-test loop
/// guards against catastrophic regressions on every `cargo test`
/// run (e.g., accidental allocator hot-path change).
#[test]
fn classify_under_10us_per_call() {
    use std::time::Instant;

    // Realistic mix: a Claude Code session (the most expensive case
    // because UA must be lowercased). Subset of headers a real proxy
    // would see.
    let h = headers(&[
        (
            "user-agent",
            "claude-code/1.5.0 (linux; x86_64) anthropic/0.42.0",
        ),
        (
            "authorization",
            "Bearer sk-ant-oat-01-abcdefghijklmnopqrstuv",
        ),
        ("content-type", "application/json"),
        ("accept", "application/json"),
        ("host", "api.anthropic.com"),
    ]);

    // Warmup so the branch predictor / icache aren't on a cold path.
    for _ in 0..1_000 {
        std::hint::black_box(classify(&h));
    }

    let iters = 100_000;
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(classify(&h));
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() / iters as u128;

    // 10us = 10_000 ns. Asserting 10x headroom guards against perf
    // regressions even on a contended CI runner.
    assert!(
        per_call_ns < 10_000,
        "classify took {} ns/call (limit: 10_000 ns); regression suspected",
        per_call_ns
    );
}
