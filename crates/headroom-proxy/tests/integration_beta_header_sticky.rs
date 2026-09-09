//! End-to-end coverage for session-sticky provider beta headers
//! (`cache_stabilization::beta_sticky` — Rust port of the Python
//! proxy's PR-A6 `SessionBetaTracker`).
//!
//! The scenario every test guards: a client (Claude Code, Codex CLI)
//! sends `anthropic-beta: a,b` on turn 1 and drops `b` on turn 2 of
//! the SAME conversation. Beta headers are part of the bytes that
//! determine the upstream prefix-cache key, so the drop rotates the
//! key and the provider re-writes the whole prefix at the customer's
//! cost. The proxy must forward the per-conversation union instead.
//!
//! These tests boot a real Rust proxy in front of a wiremock upstream
//! and assert on the headers/bytes the upstream actually receives:
//!
//! - dropped tokens are re-injected on later turns (Anthropic,
//!   OpenAI Chat, OpenAI Responses — all three intercepted routes);
//! - conversation identity works both via the explicit
//!   `x-headroom-session-id` opt-in AND via the body-derived
//!   conversation discriminator (no explicit header — the realistic
//!   Claude Code shape);
//! - the union NEVER invents tokens the client didn't send: no beta
//!   header in → no beta header out, and separate conversations don't
//!   leak tokens into each other;
//! - `--beta-header-sticky disabled` forwards the client value
//!   verbatim (diagnostic opt-out, Python
//!   `HEADROOM_BETA_HEADER_STICKY=disabled` parity);
//! - the body is forwarded byte-equal (SHA-256) while the header is
//!   rewritten — the mechanism mutates request headers, never body
//!   bytes (Phase-A cache-safety contract).

mod common;

use common::start_proxy_with;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Everything the upstream saw for one request: selected header
/// values (lower-case names) + raw body bytes.
#[derive(Clone)]
struct Seen {
    beta: Option<String>,
    session_id_header: Option<String>,
    body: Vec<u8>,
}

type Captures = Arc<Mutex<Vec<Seen>>>;

/// Mount a capture-everything mock for `route` on the upstream. The
/// `beta_header` name is which provider beta header to record
/// (`anthropic-beta` / `openai-beta`).
async fn mount_capture(upstream: &MockServer, route: &str, beta_header: &'static str) -> Captures {
    let captured: Captures = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path(route))
        .respond_with(move |req: &wiremock::Request| {
            let get = |name: &str| {
                req.headers
                    .get(name)
                    .and_then(|v| v.to_str().ok())
                    .map(|s| s.to_string())
            };
            captured_clone.lock().unwrap().push(Seen {
                beta: get(beta_header),
                session_id_header: get("x-headroom-session-id"),
                body: req.body.clone(),
            });
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

fn anthropic_body(turns: &[(&str, &str)]) -> Vec<u8> {
    let messages: Vec<serde_json::Value> = turns
        .iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect();
    serde_json::to_vec(&json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 32,
        "messages": messages,
    }))
    .unwrap()
}

fn openai_chat_body(turns: &[(&str, &str)]) -> Vec<u8> {
    let messages: Vec<serde_json::Value> = turns
        .iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect();
    serde_json::to_vec(&json!({
        "model": "gpt-4o",
        "messages": messages,
    }))
    .unwrap()
}

fn openai_responses_body(text: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "gpt-4o",
        "input": [{"role": "user", "content": text}],
    }))
    .unwrap()
}

async fn post(
    client: &reqwest::Client,
    url: String,
    body: Vec<u8>,
    headers: &[(&str, &str)],
) -> reqwest::Response {
    let mut req = client
        .post(url)
        .header("content-type", "application/json")
        .body(body);
    for (name, value) in headers {
        req = req.header(*name, *value);
    }
    req.send().await.expect("proxy reachable")
}

#[tokio::test]
async fn anthropic_dropped_beta_token_reinjected_with_explicit_session_header() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/messages", "anthropic-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", proxy.url());

    // Turn 1: two beta tokens.
    let resp = post(
        &client,
        url.clone(),
        anthropic_body(&[("user", "hello")]),
        &[
            (
                "anthropic-beta",
                "context-management-2025-06-27,interleaved-thinking-2025-05-14",
            ),
            ("x-headroom-session-id", "conv-explicit-1"),
        ],
    )
    .await;
    assert_eq!(resp.status(), 200);

    // Turn 2, same conversation: the client dropped the second token.
    let resp = post(
        &client,
        url,
        anthropic_body(&[("user", "hello"), ("assistant", "hi"), ("user", "next")]),
        &[
            ("anthropic-beta", "context-management-2025-06-27"),
            ("x-headroom-session-id", "conv-explicit-1"),
        ],
    )
    .await;
    assert_eq!(resp.status(), 200);

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[0].beta.as_deref(),
        Some("context-management-2025-06-27,interleaved-thinking-2025-05-14"),
        "turn 1 forwards the client value unchanged"
    );
    assert_eq!(
        seen[1].beta.as_deref(),
        Some("context-management-2025-06-27,interleaved-thinking-2025-05-14"),
        "turn 2 must re-inject the dropped token so the upstream \
         prefix-cache key stays byte-stable"
    );
    // PR-A5 invariant intact: the internal session header never
    // crosses the upstream boundary.
    assert!(seen.iter().all(|s| s.session_id_header.is_none()));

    proxy.shutdown().await;
}

#[tokio::test]
async fn anthropic_conversation_keyed_without_explicit_session_header() {
    // The realistic Claude Code shape: no `x-headroom-session-id`;
    // conversation identity comes from the credential arm + the
    // first-message discriminator inside `derive_session_key`.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/messages", "anthropic-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", proxy.url());

    let auth = ("authorization", "Bearer oauth-workspace-token");
    post(
        &client,
        url.clone(),
        anthropic_body(&[("user", "conversation opener")]),
        &[("anthropic-beta", "a,b"), auth],
    )
    .await;
    // Same conversation (same opener, grown transcript), token "b"
    // dropped.
    post(
        &client,
        url,
        anthropic_body(&[
            ("user", "conversation opener"),
            ("assistant", "reply"),
            ("user", "follow-up"),
        ]),
        &[("anthropic-beta", "a"), auth],
    )
    .await;

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].beta.as_deref(), Some("a,b"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_chat_dropped_beta_token_reinjected() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/chat/completions", "openai-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", proxy.url());

    post(
        &client,
        url.clone(),
        openai_chat_body(&[("user", "hello")]),
        &[
            ("openai-beta", "assistants=v2,realtime=v1"),
            ("x-headroom-session-id", "conv-oai-1"),
        ],
    )
    .await;
    post(
        &client,
        url,
        openai_chat_body(&[("user", "hello"), ("assistant", "hi"), ("user", "next")]),
        &[
            ("openai-beta", "assistants=v2"),
            ("x-headroom-session-id", "conv-oai-1"),
        ],
    )
    .await;

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].beta.as_deref(), Some("assistants=v2,realtime=v1"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn openai_responses_dropped_beta_token_reinjected() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/responses", "openai-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/responses", proxy.url());

    post(
        &client,
        url.clone(),
        openai_responses_body("hello"),
        &[
            ("openai-beta", "responses=v1,tools=v2"),
            ("x-headroom-session-id", "conv-resp-1"),
        ],
    )
    .await;
    post(
        &client,
        url,
        openai_responses_body("hello again"),
        &[
            ("openai-beta", "responses=v1"),
            ("x-headroom-session-id", "conv-resp-1"),
        ],
    )
    .await;

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].beta.as_deref(), Some("responses=v1,tools=v2"));

    proxy.shutdown().await;
}

#[tokio::test]
async fn anthropic_fully_omitted_beta_header_regains_union() {
    // The headline docs claim: "sends a token in turn N and omits it
    // in turn N+1" — here the whole header disappears, not just one
    // token, and the union must be re-added through real axum/reqwest
    // plumbing.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/messages", "anthropic-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", proxy.url());

    post(
        &client,
        url.clone(),
        anthropic_body(&[("user", "hello")]),
        &[
            ("anthropic-beta", "context-management-2025-06-27"),
            ("x-headroom-session-id", "conv-omit-1"),
        ],
    )
    .await;
    // Turn 2: no anthropic-beta header at all.
    post(
        &client,
        url,
        anthropic_body(&[("user", "hello"), ("assistant", "hi"), ("user", "next")]),
        &[("x-headroom-session-id", "conv-omit-1")],
    )
    .await;

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[1].beta.as_deref(),
        Some("context-management-2025-06-27"),
        "a fully omitted beta header must be restored from session state"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn disabled_flag_forwards_client_value_verbatim() {
    use headroom_proxy::config::BetaHeaderSticky;

    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/messages", "anthropic-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.beta_header_sticky = BetaHeaderSticky::Disabled;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", proxy.url());

    post(
        &client,
        url.clone(),
        anthropic_body(&[("user", "hello")]),
        &[
            ("anthropic-beta", "a,b"),
            ("x-headroom-session-id", "conv-d1"),
        ],
    )
    .await;
    post(
        &client,
        url,
        anthropic_body(&[("user", "hello"), ("assistant", "hi"), ("user", "next")]),
        &[
            ("anthropic-beta", "a"),
            ("x-headroom-session-id", "conv-d1"),
        ],
    )
    .await;

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(
        seen[1].beta.as_deref(),
        Some("a"),
        "disabled mode must forward the dropped-token value verbatim \
         and keep no session state"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn no_client_beta_header_is_never_invented() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/messages", "anthropic-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", proxy.url());

    for body in [
        anthropic_body(&[("user", "hello")]),
        anthropic_body(&[("user", "hello"), ("assistant", "hi"), ("user", "next")]),
    ] {
        post(
            &client,
            url.clone(),
            body,
            &[("x-headroom-session-id", "conv-n1")],
        )
        .await;
    }

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert!(
        seen.iter().all(|s| s.beta.is_none()),
        "a session that never sent a beta header must never gain one"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn separate_conversations_do_not_leak_tokens() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/messages", "anthropic-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", proxy.url());

    post(
        &client,
        url.clone(),
        anthropic_body(&[("user", "conversation A")]),
        &[
            ("anthropic-beta", "token-a"),
            ("x-headroom-session-id", "conv-A"),
        ],
    )
    .await;
    post(
        &client,
        url,
        anthropic_body(&[("user", "conversation B")]),
        &[
            ("anthropic-beta", "token-b"),
            ("x-headroom-session-id", "conv-B"),
        ],
    )
    .await;

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].beta.as_deref(), Some("token-a"));
    assert_eq!(
        seen[1].beta.as_deref(),
        Some("token-b"),
        "conversation B must not inherit conversation A's tokens"
    );

    proxy.shutdown().await;
}

#[tokio::test]
async fn body_bytes_stay_byte_equal_while_header_is_rewritten() {
    // Cache-safety contract: the sticky union mutates request
    // HEADERS only. The forwarded body must remain byte-identical
    // (SHA-256) to what the client sent — same assertion idiom as the
    // model-sanitizer integration tests.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream, "/v1/messages", "anthropic-beta").await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
    })
    .await;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", proxy.url());

    let turn1 = anthropic_body(&[("user", "hello")]);
    let turn2 = anthropic_body(&[("user", "hello"), ("assistant", "hi"), ("user", "next")]);

    post(
        &client,
        url.clone(),
        turn1,
        &[
            ("anthropic-beta", "a,b"),
            ("x-headroom-session-id", "conv-bb"),
        ],
    )
    .await;
    post(
        &client,
        url,
        turn2.clone(),
        &[
            ("anthropic-beta", "a"),
            ("x-headroom-session-id", "conv-bb"),
        ],
    )
    .await;

    let seen = captured.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    // Header was rewritten to the union…
    assert_eq!(seen[1].beta.as_deref(), Some("a,b"));
    // …but the body bytes are untouched.
    assert_eq!(
        Sha256::digest(&seen[1].body),
        Sha256::digest(&turn2),
        "sticky beta union must never mutate body bytes"
    );

    proxy.shutdown().await;
}
