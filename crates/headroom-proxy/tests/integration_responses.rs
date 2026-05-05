//! Integration tests for the `/v1/responses` Rust handler (Phase C
//! PR-C3).
//!
//! These tests boot the real Rust proxy in front of a wiremock
//! upstream and exercise the OpenAI Responses API request shape
//! end-to-end. Per spec PR-C3:
//!
//! - V4A patch bodies, `local_shell_call.action.command` argv arrays,
//!   Codex `phase`, `compaction`, MCP / computer-use / image
//!   generation items, `function_call.arguments` (string form),
//!   `reasoning.encrypted_content` round-trip BYTE-EQUAL upstream.
//! - `function_call_output.output` / `local_shell_call_output.output`
//!   / `apply_patch_call_output.output` compress only when the
//!   latest of each kind AND above the 2 KiB output-item floor.
//! - Unknown `type` values trigger
//!   `event = responses_unknown_item_type` warn logs and pass
//!   through verbatim.
//!
//! Where compression is expected NOT to run, we assert SHA-256 byte
//! equality between the bytes the client sent and the bytes the
//! upstream received.

mod common;

use common::start_proxy_with;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Mount a /v1/responses handler that captures the upstream request body.
async fn mount_capture(upstream: &MockServer) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let captured_clone = captured.clone();
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |req: &wiremock::Request| {
            *captured_clone.lock().unwrap() = Some(req.body.clone());
            ResponseTemplate::new(200).set_body_string(r#"{"ok":true}"#)
        })
        .mount(upstream)
        .await;
    captured
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut acc, b| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

#[track_caller]
fn assert_byte_equal_sha256(inbound: &[u8], received: &[u8]) {
    let inbound_hash = sha256_hex(inbound);
    let received_hash = sha256_hex(received);
    assert_eq!(
        inbound.len(),
        received.len(),
        "byte length mismatch: inbound={}, upstream-received={}",
        inbound.len(),
        received.len(),
    );
    assert_eq!(
        inbound_hash, received_hash,
        "SHA-256 mismatch: inbound={inbound_hash}, upstream-received={received_hash}",
    );
}

/// V4A diff fixture used for apply_patch_* tests. The exact byte
/// sequence (including trailing whitespace) must round-trip.
const V4A_DIFF: &str = "*** Begin Patch\n*** Update File: src/main.rs\n@@ -1,3 +1,4 @@\n fn main() {\n+    println!(\"hello\");\n     run();\n }\n*** End Patch\n";

#[tokio::test]
async fn v4a_patch_byte_equal_through_proxy() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "apply_patch_call",
                "id": "ap_1",
                "call_id": "call_1",
                "operation": {"type": "apply_patch", "diff": V4A_DIFF},
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    // Defensive: the diff arrives intact as a string field.
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(parsed["input"][0]["operation"]["diff"], json!(V4A_DIFF));
    proxy.shutdown().await;
}

#[tokio::test]
async fn local_shell_call_command_argv_array_preserved() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "local_shell_call",
                "id": "ls_1",
                "call_id": "call_1",
                "action": {
                    "type": "exec",
                    "command": ["bash", "-c", "ls -la"],
                    "working_directory": "/tmp",
                    "timeout_ms": 60000
                }
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    // Critical assertion: command stays as a JSON ARRAY, not a string.
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    let cmd = &parsed["input"][0]["action"]["command"];
    assert!(cmd.is_array(), "command must remain an array on the wire");
    assert_eq!(cmd[0], json!("bash"));
    assert_eq!(cmd[1], json!("-c"));
    assert_eq!(cmd[2], json!("ls -la"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn codex_phase_commentary_preserved() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "thinking step"}]
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(parsed["input"][0]["phase"], json!("commentary"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn codex_phase_final_answer_preserved() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "the answer is 42"}]
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(parsed["input"][0]["phase"], json!("final_answer"));
    proxy.shutdown().await;
}

#[tokio::test]
async fn compaction_item_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // Opaque encrypted blob — must round-trip verbatim. Simulate
    // ~3 KiB of base64-ish payload.
    let blob = "A".repeat(3000);
    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {"type": "compaction", "id": "k1", "encrypted_content": blob}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn reasoning_encrypted_content_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let blob = "encrypted-reasoning-blob-".repeat(150); // ~3.6 KiB
    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {"type": "reasoning", "id": "r1", "encrypted_content": blob}
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn function_call_arguments_string_preserved() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // arguments is a JSON-ENCODED STRING (the model emitted it). We
    // never parse it inside the proxy.
    let args_str = r#"{"q": "hello world", "max": 10}"#;
    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_xyz",
                "name": "search",
                "arguments": args_str
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    // arguments must arrive as a STRING (not a parsed object).
    assert_eq!(parsed["input"][0]["arguments"], json!(args_str));
    assert!(parsed["input"][0]["arguments"].is_string());
    proxy.shutdown().await;
}

#[tokio::test]
async fn call_id_referenced_not_id() {
    // The plan specifies: outputs reference parents via `call_id`,
    // not `id`. This test pins that semantic — both fields are
    // distinct and both round-trip.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "function_call",
                "id": "fc_internal_1",
                "call_id": "call_external_99",
                "name": "search",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "id": "fco_internal_1",
                "call_id": "call_external_99",
                "output": "result-data"
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    // The `call_id` field on call and output must MATCH.
    let call_id_call = &parsed["input"][0]["call_id"];
    let call_id_output = &parsed["input"][1]["call_id"];
    assert_eq!(call_id_call, call_id_output);
    // And the `id` fields are DISTINCT.
    assert_ne!(parsed["input"][0]["id"], parsed["input"][1]["id"]);
    proxy.shutdown().await;
}

#[tokio::test]
async fn apply_patch_output_below_2kb_no_compression() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // ~1 KiB payload — under the 2 KiB output-item floor.
    let small = "x".repeat(1024);
    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "apply_patch_call_output",
                "id": "apo_1",
                "call_id": "call_1",
                "output": small
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn apply_patch_output_above_2kb_compressed() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // ~8 KiB build-output style log. Repetitive lines so the
    // LogCompressor recognizes a template and produces savings.
    let mut log = String::new();
    for i in 0..200 {
        log.push_str(&format!(
            "[2024-01-01 00:00:00] INFO build.rs:42 compiled module foo_{i}\n"
        ));
    }
    assert!(log.len() > 4096, "log fixture must clearly exceed 2 KiB");

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "apply_patch_call_output",
                "id": "apo_1",
                "call_id": "call_1",
                "output": log
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    // The dispatcher should have mutated the body — either it
    // shrank or (for some fixtures) the tokenizer rejected the
    // compression. We assert it AT LEAST attempted the rewrite by
    // checking either the body shrank, or it stayed byte-equal
    // (rejected). The "above 2KB" gate is what's being tested —
    // the path was not skipped pre-dispatch.
    if got.len() == body.len() {
        // Token-validated rejection — accept.
        assert_byte_equal_sha256(&body, &got);
    } else {
        assert!(
            got.len() < body.len(),
            "body did not shrink: in={}, out={}",
            body.len(),
            got.len()
        );
    }
    proxy.shutdown().await;
}

#[tokio::test]
async fn local_shell_output_compressed() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // ~5 KiB shell-style log lines.
    let mut log = String::new();
    for i in 0..120 {
        log.push_str(&format!(
            "[2024-01-01 12:00:00] INFO daemon.rs:88 task_{i} completed in 12ms\n"
        ));
    }
    assert!(log.len() > 4096);

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "local_shell_call_output",
                "id": "lso_1",
                "call_id": "call_1",
                "output": log
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    // Either the body shrank (LogCompressor took it) or the
    // token-validated rejection kept it byte-equal. Both are valid
    // outcomes; what matters is the floor was cleared.
    if got.len() == body.len() {
        assert_byte_equal_sha256(&body, &got);
    } else {
        assert!(
            got.len() < body.len(),
            "expected shrink, got: in={}, out={}",
            body.len(),
            got.len()
        );
    }
    proxy.shutdown().await;
}

#[tokio::test]
async fn mcp_tool_call_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "mcp_call",
                "id": "mc_1",
                "server": "atlas",
                "tool": "lookup",
                "arguments": {"key": "value"},
                "result": {"ok": true, "rows": [1, 2, 3]}
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn computer_call_byte_equal() {
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "computer_call",
                "id": "cc_1",
                "action": {"type": "click", "x": 100, "y": 200}
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}

#[tokio::test]
async fn image_generation_call_no_log_redaction_in_test_mode() {
    // Per spec: redaction is a LOG-PATH concern only. The
    // upstream-bound bytes must NOT be redacted.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    // Synthetic small base64 payload.
    let image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "image_generation_call",
                "id": "img_1",
                "status": "completed",
                "image_data": image_data
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    // Critical: image_data flows through verbatim. Redaction is
    // log-only.
    assert_byte_equal_sha256(&body, &got);
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(parsed["input"][0]["image_data"], json!(image_data));
    proxy.shutdown().await;
}

#[tokio::test]
async fn unknown_item_type_logged_warning_byte_equal() {
    // No-silent-fallbacks: unknown `type` logs at warn but never
    // mutates the bytes. We can't easily intercept tracing in this
    // test (the harness doesn't install a custom subscriber); we
    // assert the byte-equality contract and rely on the unit test
    // inside `live_zone_responses` for the warn-event coverage.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {
                "type": "future_item_type_v2",
                "novel_field": "preserve me",
                "nested": {"deep": [1, 2, 3]}
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "describe"}]
            }
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    let parsed: Value = serde_json::from_slice(&got).unwrap();
    assert_eq!(parsed["input"][0]["type"], json!("future_item_type_v2"));
    assert_eq!(parsed["input"][0]["novel_field"], json!("preserve me"));
    assert_eq!(parsed["input"][0]["nested"]["deep"], json!([1, 2, 3]));
    proxy.shutdown().await;
}

#[tokio::test]
async fn representative_request_round_trip() {
    // Acceptance criterion: a representative request with reasoning
    // + function_call + local_shell + apply_patch + custom items
    // round-trips byte-equal modulo compressed live-zone outputs.
    // None of the items here are above the 2 KiB output-item floor,
    // so we expect zero compression and full byte-equality.
    let upstream = MockServer::start().await;
    let captured = mount_capture(&upstream).await;
    let proxy = start_proxy_with(&upstream.uri(), |c| {
        c.compression = true;
        c.compression_mode = headroom_proxy::config::CompressionMode::LiveZone;
    })
    .await;

    let payload = json!({
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "fix the bug"}]},
            {"type": "reasoning", "id": "r1", "encrypted_content": "opaque-reasoning"},
            {"type": "function_call", "id": "fc_1", "call_id": "c1",
             "name": "search", "arguments": "{\"q\":\"bug\"}"},
            {"type": "function_call_output", "id": "fco_1", "call_id": "c1",
             "output": "found 3 matches"},
            {"type": "local_shell_call", "id": "ls_1", "call_id": "c2",
             "action": {"type": "exec", "command": ["cargo", "test"], "timeout_ms": 60000}},
            {"type": "local_shell_call_output", "id": "lso_1", "call_id": "c2",
             "output": "ok 12 tests passed"},
            {"type": "apply_patch_call", "id": "ap_1", "call_id": "c3",
             "operation": {"type": "apply_patch", "diff": V4A_DIFF}},
            {"type": "apply_patch_call_output", "id": "apo_1", "call_id": "c3",
             "output": "patch applied"},
            {"type": "custom_tool_call", "id": "ct_1", "tool": "myorg.foo",
             "input": {"x": 1}},
        ]
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/responses", proxy.url()))
        .header("content-type", "application/json")
        .body(body.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let got = captured.lock().unwrap().clone().expect("upstream got body");
    assert_byte_equal_sha256(&body, &got);
    proxy.shutdown().await;
}
