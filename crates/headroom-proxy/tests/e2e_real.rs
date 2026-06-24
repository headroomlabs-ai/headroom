//! Real end-to-end tests: Rust proxy → Python Headroom proxy → real LLM API.
//!
//! These spawn the actual Python proxy as a subprocess and route real requests
//! to Anthropic / OpenAI through the full chain. Skipped unless HEADROOM_E2E=1
//! to keep `cargo test` fast and free.
//!
//! Run with:
//!     HEADROOM_E2E=1 cargo test -p headroom-proxy --test e2e_real -- --nocapture
//!
//! Reads API keys from .env at the repo root. No keys → individual tests skip.

mod common;

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::start_proxy;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

const E2E_GUARD: &str = "HEADROOM_E2E";

fn e2e_enabled() -> bool {
    std::env::var(E2E_GUARD).ok().as_deref() == Some("1")
}

/// Locate repo root by walking up from CARGO_MANIFEST_DIR until we find .env.
fn repo_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if p.join(".env").exists() && p.join("Cargo.toml").exists() {
            return p;
        }
        if !p.pop() {
            panic!("could not locate repo root (no .env found)");
        }
    }
}

/// Best-effort .env loader. Does NOT print values. Sets vars only if absent.
fn load_dotenv() {
    let root = repo_root();
    let env_path = root.join(".env");
    let Ok(contents) = std::fs::read_to_string(&env_path) else {
        return;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').trim_matches('\'');
        if v.is_empty() {
            continue;
        }
        if std::env::var(k).is_err() {
            // SAFETY for tests: setting env vars in single-threaded test setup.
            // Tokio's #[tokio::test] runs each test in its own runtime; this is
            // before the runtime starts spawning concurrent tasks.
            std::env::set_var(k, v);
        }
    }
}

/// A guard that kills the Python proxy on drop and waits for it to exit.
struct PythonProxy {
    child: Option<Child>,
    port: u16,
}

impl PythonProxy {
    /// Spawn `headroom proxy --port <ephemeral> --no-optimize` in passthrough
    /// mode and wait until /livez returns 200. Inherits the env including
    /// API keys loaded from .env.
    async fn spawn() -> Self {
        // Pick an ephemeral port deterministically by binding+releasing.
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let p = l.local_addr().unwrap().port();
            drop(l);
            p
        };
        let root = repo_root();
        let venv_python = root.join(".venv/bin/headroom");
        assert!(
            venv_python.exists(),
            "expected venv at {} — run `make e2e-venv` or activate venv first",
            venv_python.display()
        );

        let mut cmd = Command::new(&venv_python);
        cmd.current_dir(&root)
            .arg("proxy")
            .arg("--port")
            .arg(port.to_string())
            .arg("--no-optimize")
            .arg("--host")
            .arg("127.0.0.1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn().expect("spawn python proxy");

        // Drain stdout/stderr in background so the pipe doesn't fill up.
        if let Some(out) = child.stdout.take() {
            tokio::spawn(async move {
                let mut r = BufReader::new(out).lines();
                while let Ok(Some(line)) = r.next_line().await {
                    eprintln!("[py-stdout] {line}");
                }
            });
        }
        if let Some(err) = child.stderr.take() {
            tokio::spawn(async move {
                let mut r = BufReader::new(err).lines();
                while let Ok(Some(line)) = r.next_line().await {
                    eprintln!("[py-stderr] {line}");
                }
            });
        }

        // Poll /livez until ready, up to 30s.
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        let url = format!("http://127.0.0.1:{port}/livez");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if Instant::now() > deadline {
                panic!("python proxy did not become healthy at {url} within 30s");
            }
            match client.get(&url).send().await {
                Ok(r) if r.status().is_success() => break,
                _ => tokio::time::sleep(Duration::from_millis(200)).await,
            }
        }

        Self {
            child: Some(child),
            port,
        }
    }

    fn upstream_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for PythonProxy {
    fn drop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.start_kill();
            // Best-effort: don't block drop on tokio runtime.
        }
    }
}

// =============================================================================
//                                  TESTS
// =============================================================================

#[tokio::test]
async fn e2e_health_through_full_chain() {
    if !e2e_enabled() {
        eprintln!("skipping (set {E2E_GUARD}=1 to run)");
        return;
    }
    load_dotenv();
    let py = PythonProxy::spawn().await;
    let proxy = start_proxy(&py.upstream_url()).await;

    // Rust /healthz is intercepted (never forwarded).
    let r = reqwest::get(format!("{}/healthz", proxy.url()))
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let json: Value = r.json().await.unwrap();
    assert_eq!(json["service"], "headroom-proxy");

    // Rust /healthz/upstream pings Python /healthz.
    let r = reqwest::get(format!("{}/healthz/upstream", proxy.url()))
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // /livez is forwarded to Python.
    let r = reqwest::get(format!("{}/livez", proxy.url()))
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    proxy.shutdown().await;
    drop(py);
}

#[tokio::test]
async fn e2e_anthropic_non_streaming() {
    if !e2e_enabled() {
        eprintln!("skipping (set {E2E_GUARD}=1 to run)");
        return;
    }
    load_dotenv();
    let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
        eprintln!("skipping: ANTHROPIC_API_KEY not set");
        return;
    };

    let py = PythonProxy::spawn().await;
    let proxy = start_proxy(&py.upstream_url()).await;

    let body = json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .expect("anthropic request");
    let status = resp.status();
    let text = resp.text().await.unwrap();
    assert_eq!(status, 200, "non-200 from anthropic: {text}");
    let v: Value = serde_json::from_str(&text).expect("response is JSON");
    assert_eq!(v["type"], "message");
    let content = v["content"][0]["text"].as_str().unwrap_or("");
    assert!(
        content.to_uppercase().contains("PONG"),
        "expected PONG in response, got: {content}"
    );

    proxy.shutdown().await;
    drop(py);
}

/// Full chain streaming: Rust proxy → Python proxy → Anthropic. Validates
/// that SSE flows end-to-end with the production stack.
#[tokio::test]
async fn e2e_anthropic_streaming() {
    if !e2e_enabled() {
        eprintln!("skipping (set {E2E_GUARD}=1 to run)");
        return;
    }
    load_dotenv();
    let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
        eprintln!("skipping: ANTHROPIC_API_KEY not set");
        return;
    };

    let py = PythonProxy::spawn().await;
    let proxy = start_proxy(&py.upstream_url()).await;

    let body = json!({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 32,
        "stream": true,
        "messages": [{"role": "user", "content": "Count: 1, 2, 3."}],
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/messages", proxy.url()))
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .expect("anthropic stream request");
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/event-stream"),
        "expected SSE content-type, got: {ct}"
    );

    // Collect stream and verify SSE framing.
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut chunks = 0usize;
    let mut last_err: Option<String> = None;
    let deadline = Instant::now() + Duration::from_secs(60);
    while let Some(item) = stream.next().await {
        if Instant::now() > deadline {
            panic!("stream did not complete within 60s. chunks={chunks} buf:\n{buf}");
        }
        match item {
            Ok(c) => {
                chunks += 1;
                buf.push_str(&String::from_utf8_lossy(&c));
                if buf.contains("message_stop") {
                    break;
                }
            }
            Err(e) => {
                last_err = Some(e.to_string());
                break;
            }
        }
    }
    eprintln!(
        "[debug] chunks={chunks} bytes={} last_err={last_err:?}",
        buf.len()
    );
    let has_start = buf.contains("message_start");
    let has_delta = buf.contains("content_block_delta") || buf.contains("\"delta\"");
    let has_stop = buf.contains("message_stop");
    assert!(
        has_start && has_delta && has_stop,
        "stream missing expected events (start={has_start} delta={has_delta} stop={has_stop}). buf:\n{}",
        &buf.chars().take(2000).collect::<String>()
    );
    assert!(
        chunks >= 1,
        "expected at least one SSE chunk (got {chunks})"
    );

    proxy.shutdown().await;
    drop(py);
}

#[tokio::test]
async fn e2e_openai_non_streaming() {
    if !e2e_enabled() {
        eprintln!("skipping (set {E2E_GUARD}=1 to run)");
        return;
    }
    load_dotenv();
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("skipping: OPENAI_API_KEY not set");
        return;
    };

    let py = PythonProxy::spawn().await;
    let proxy = start_proxy(&py.upstream_url()).await;

    let body = json!({
        "model": "gpt-4o-mini",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", proxy.url()))
        .bearer_auth(&api_key)
        .json(&body)
        .send()
        .await
        .expect("openai request");
    let status = resp.status();
    let text = resp.text().await.unwrap();
    assert_eq!(status, 200, "non-200 from openai: {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert!(
        content.to_uppercase().contains("PONG"),
        "expected PONG, got: {content}"
    );

    proxy.shutdown().await;
    drop(py);
}

#[tokio::test]
async fn e2e_request_id_propagates() {
    if !e2e_enabled() {
        eprintln!("skipping (set {E2E_GUARD}=1 to run)");
        return;
    }
    load_dotenv();
    let py = PythonProxy::spawn().await;
    let proxy = start_proxy(&py.upstream_url()).await;

    // The Python proxy does not necessarily echo X-Request-Id; what we verify
    // here is that the Rust proxy GENERATES one and echoes it back to the
    // client when the upstream call returns. Use /livez (always 200).
    let resp = reqwest::Client::new()
        .get(format!("{}/livez", proxy.url()))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let rid = resp.headers().get("x-request-id");
    assert!(rid.is_some(), "Rust proxy must echo X-Request-Id back");
    let rid_str = rid.unwrap().to_str().unwrap();
    assert!(
        !rid_str.is_empty() && rid_str.len() >= 16,
        "request id looks unreasonable: {rid_str}"
    );

    // Client-supplied request id must be preserved.
    let supplied = "client-supplied-12345";
    let resp = reqwest::Client::new()
        .get(format!("{}/livez", proxy.url()))
        .header("x-request-id", supplied)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.headers().get("x-request-id").unwrap(), supplied);

    proxy.shutdown().await;
    drop(py);
}
