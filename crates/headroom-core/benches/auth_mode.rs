//! Criterion benchmark for the auth-mode classifier (Phase F PR-F1).
//!
//! Acceptance criterion: <10us per call. Realistic header sets from
//! the three classes the proxy actually sees in production:
//!
//! - PAYG: `Authorization: Bearer sk-ant-api03-...`
//! - OAuth: `Authorization: Bearer <jwt>` (Codex-style)
//! - Subscription: `User-Agent: claude-code/1.5.0 ...` + `Bearer
//!   sk-ant-oat-...`
//!
//! The bench measures one classifier call per iteration. The
//! `HeaderMap` is constructed once outside the timing loop.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use headroom_core::auth_mode::classify;
use http::{HeaderMap, HeaderValue};

fn build_headers(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (name, value) in pairs {
        h.insert(
            http::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_str(value).unwrap(),
        );
    }
    h
}

fn bench_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("auth_mode/classify");

    // Empty headers — the simplest path; all branches fall through.
    let empty = HeaderMap::new();
    group.bench_function("empty", |b| b.iter(|| classify(black_box(&empty))));

    // PAYG — Authorization is Bearer, prefix matches early.
    let payg = build_headers(&[(
        "authorization",
        "Bearer sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
    )]);
    group.bench_function("payg_anthropic_api_key", |b| {
        b.iter(|| classify(black_box(&payg)))
    });

    // OAuth — JWT, three segments, last branch in the bearer match.
    let oauth = build_headers(&[(
        "authorization",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4iLCJpYXQiOjE1MTYyMzkwMjJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
    )]);
    group.bench_function("oauth_jwt", |b| b.iter(|| classify(black_box(&oauth))));

    // Subscription — UA must be lowercased; the most expensive path.
    let subscription = build_headers(&[
        (
            "user-agent",
            "claude-code/1.5.0 (linux; x86_64) anthropic/0.42.0",
        ),
        (
            "authorization",
            "Bearer sk-ant-oat-01-abcdefghijklmnopqrstuvwxyz",
        ),
        ("content-type", "application/json"),
        ("accept", "application/json"),
        ("host", "api.anthropic.com"),
    ]);
    group.bench_function("subscription_claude_code", |b| {
        b.iter(|| classify(black_box(&subscription)))
    });

    group.finish();
}

criterion_group!(benches, bench_classify);
criterion_main!(benches);
