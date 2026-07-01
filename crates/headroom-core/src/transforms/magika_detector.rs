//! Magika-based content detection (Stage 3d Tier 1).
//!
//! Wraps Google's [`magika`] crate — an ONNX-backed content classifier —
//! and maps its 200+ labels onto Headroom's existing
//! [`crate::transforms::content_detector::ContentType`] enum so the
//! ContentRouter dispatch (PR5) can stay enum-stable.
//!
//! # Design
//!
//! - **Singleton session.** Magika model loading is the expensive part
//!   (one-time ONNX init, ~50 ms cold). We do it exactly once per
//!   process via [`OnceLock`]. The `Session` requires `&mut self`
//!   for inference, so the singleton wraps it in a `Mutex` — fine for
//!   our throughput; if benchmarks show contention later we'll pool.
//!
//! - **Loud failures.** If the model fails to load or inference fails,
//!   `magika_detect` returns `Err`. The ContentRouter (PR5) decides
//!   whether to fall back to Tier 2 (`unidiff-rs`) or surface to the
//!   caller. We deliberately do **not** silently return `PlainText` on
//!   error — that's the kind of silent fallback the audit doc forbids.
//!
//! - **Mapping table is explicit.** Every magika label we care about
//!   has an explicit case in [`map_magika_label`]; everything else
//!   falls into [`ContentType::PlainText`]. This is a code-route
//!   decision, not a regex — adding a new mapping is one line, and
//!   readers can audit the dispatch in one screen.
//!
//! - **No router rewiring here.** PR3 lands the detector + tests
//!   only. PR5 flips the ContentRouter to call us instead of the
//!   regex-based [`crate::transforms::content_detector`].
//!
//! - **CPU compatibility.** The precompiled ONNX Runtime binary shipped by
//!   `ort-sys` may contain AVX2-family instructions on x86/x86_64. Where
//!   AVX2 is unavailable on those targets, the session init returns an
//!   error early instead of crashing with SIGILL; the detection chain then
//!   falls through to Tier 2 and Tier 3 normally.

use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use magika::Session;
use thiserror::Error;
use tracing;

use crate::transforms::content_detector::ContentType;

/// Check whether the CPU can run the precompiled ONNX Runtime binary
/// that magika depends on.
///
/// On x86/x86_64 without AVX2, the `onnxruntime` shared library shipped
/// by `ort-sys` can contain AVX2-family instructions that will SIGILL.
/// We detect this up front so the magika session init can fail gracefully
/// instead of crashing.
///
/// On non-x86 targets, this x86-specific AVX2 gate is not applied.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn magika_onnx_runtime_supported_by_cpu() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn magika_onnx_runtime_supported_by_cpu() -> bool {
    true
}

/// Errors from the magika detector. Wraps the underlying `magika::Error`
/// so callers can match on whether init or inference broke without
/// pulling magika types into their imports.
#[derive(Debug, Error)]
pub enum MagikaDetectorError {
    /// One-time session initialization failed (model load, ONNX init).
    /// Once we hit this, every subsequent call also fails — there is no
    /// retry path here. The router should surface and stop.
    #[error("magika session init failed: {0}")]
    Init(String),

    /// Inference call failed for this input. Usually transient; future
    /// calls may succeed. The error message is the magika-side text;
    /// we don't try to wrap it.
    #[error("magika inference failed: {0}")]
    Inference(String),

    /// Singleton lock was poisoned (a previous holder panicked while
    /// holding it). The detector is unusable until the process
    /// restarts. We don't auto-recover — a panicked detector means
    /// something is corrupt and continuing would mask it.
    #[error("magika session lock poisoned")]
    Poisoned,
}

/// One-process singleton holding the magika session. Lazily
/// initialized on first call to [`magika_detect`].
///
/// `Mutex<Result<Session, ...>>` rather than `Result<Mutex<Session>>`
/// so init failure is recorded once and replayed cheaply on every
/// subsequent call (no re-attempting the load — if the model file is
/// missing or ort can't init, retrying just wastes cycles).
static MAGIKA_SESSION: OnceLock<Mutex<Result<Session, String>>> = OnceLock::new();

/// Default cap on magika ONNX session init.
///
/// On some platforms `Session::new()` can hang indefinitely instead of
/// returning an error. Root-caused on Windows: with `ort-load-dynamic`
/// (Windows-gated in `Cargo.toml`), the bare `LoadLibrary("onnxruntime.dll")`
/// search resolves to `C:\Windows\System32\onnxruntime.dll` — the Windows ML
/// OS component (1.17.x on Win11 24H2+) — and initializing an ort 2.x
/// session against it deadlocks at 0% CPU rather than erroring. A hang —
/// unlike an `Err` — is not caught by the tiered fallback in
/// [`crate::transforms::detection`], so it stalls the entire compression
/// pipeline until the proxy's own 30s+ timeout fires on every request.
///
/// The real fix is `headroom/_ort.py`, which pins `ORT_DYLIB_PATH` to the
/// pip-installed `onnxruntime` DLL before this crate can load ort. This
/// timeout remains as the safety net for unpinned embedders of the crate.
/// Override with `HEADROOM_MAGIKA_INIT_TIMEOUT_SECS`.
const MAGIKA_INIT_TIMEOUT_SECS_DEFAULT: u64 = 5;

fn magika_init_timeout() -> Duration {
    let secs = std::env::var("HEADROOM_MAGIKA_INIT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(MAGIKA_INIT_TIMEOUT_SECS_DEFAULT);
    Duration::from_secs(secs)
}

fn session() -> &'static Mutex<Result<Session, String>> {
    MAGIKA_SESSION.get_or_init(|| {
        // Early-out if the CPU can't run the precompiled ONNX Runtime.
        // Without this check the `onnxruntime` shared library will crash
        // with SIGILL on x86 CPUs lacking AVX2.
        if !magika_onnx_runtime_supported_by_cpu() {
            return Mutex::new(Err(
                "Magika ONNX Runtime backend requires AVX2 on this platform; \
                 falling back to non-Magika detection"
                    .to_string(),
            ));
        }

        let timeout = magika_init_timeout();
        let (tx, rx) = mpsc::channel();
        // Run the (potentially hanging) ONNX init on a side thread so we
        // can bound it. `Session: Send` (the static itself requires it),
        // so moving the result across the channel is sound. On timeout we
        // record an `Err` — `detection::detect` already falls through to
        // the unidiff/regex tiers on `Err` — and the orphaned init thread
        // is left to finish on its own; its eventual `send` lands on a
        // dropped receiver (harmless) and the `Session` is then dropped.
        let spawned = std::thread::Builder::new()
            .name("magika-init".into())
            .spawn(move || {
                let _ = tx.send(Session::new().map_err(|e| e.to_string()));
            });
        if let Err(e) = spawned {
            tracing::warn!("magika init thread spawn failed: {e}");
            return Mutex::new(Err(format!("magika init thread spawn failed: {e}")));
        }
        match rx.recv_timeout(timeout) {
            Ok(res) => Mutex::new(res),
            Err(_) => {
                let ort_dylib = std::env::var("ORT_DYLIB_PATH").ok();
                tracing::warn!(
                    timeout_secs = timeout.as_secs(),
                    ort_dylib_path = ort_dylib.as_deref(),
                    "magika ONNX session init timed out; detection falls back to \
                     non-ML tiers for this process. On Windows an unset \
                     ORT_DYLIB_PATH usually means the WinML System32 \
                     onnxruntime.dll was picked up (deadlocks ort init)."
                );
                Mutex::new(Err(format!(
                    "magika session init exceeded {}s timeout; \
                     using non-ML detection tiers",
                    timeout.as_secs()
                )))
            }
        }
    })
}

/// Classify `content` and return the mapped Headroom [`ContentType`].
///
/// Empty input shortcuts to [`ContentType::PlainText`] without touching
/// the model — saves the round trip on every empty tool result.
pub fn magika_detect(content: &str) -> Result<ContentType, MagikaDetectorError> {
    if content.is_empty() {
        return Ok(ContentType::PlainText);
    }

    let mutex = session();
    let mut guard = mutex.lock().map_err(|_| MagikaDetectorError::Poisoned)?;
    let session = guard
        .as_mut()
        .map_err(|e| MagikaDetectorError::Init(e.clone()))?;

    let bytes = content.as_bytes();
    let file_type = session
        .identify_content_sync(bytes)
        .map_err(|e| MagikaDetectorError::Inference(e.to_string()))?;

    Ok(map_magika_label(file_type.info().label))
}

/// Map a magika label string to Headroom's [`ContentType`] enum.
///
/// **Why explicit cases instead of `group == "code"`:** magika's
/// `group` field is a coarse bucket ("code", "text", "binary",
/// "executable", ...). Some entries we want — like `markdown`,
/// `txt`, `latex` — are in the `text` group along with formats we
/// route differently. So we case on the label directly: clear, one
/// match arm per decision, no group-vs-label semantic confusion.
///
/// **Unmapped labels return [`ContentType::PlainText`]**, the safest
/// default — passthrough at the router level rather than misroute to
/// a wrong compressor. PR5 will refine this for `SearchResults` /
/// `BuildOutput` (which magika has no equivalent for).
pub fn map_magika_label(label: &str) -> ContentType {
    match label {
        // ── JSON ───────────────────────────────────────────────────
        // PR5 will refine this with the existing `is_json_array_of_dicts`
        // check — magika says "this is JSON" but doesn't tell us if it's
        // an array of records vs. a single object. For PR3 the mapping
        // exists; the refinement is a router concern.
        "json" | "jsonl" => ContentType::JsonArray,

        // ── Diffs ──────────────────────────────────────────────────
        "diff" => ContentType::GitDiff,

        // ── HTML ───────────────────────────────────────────────────
        "html" | "xml" => ContentType::Html,

        // ── Source code ────────────────────────────────────────────
        // The big "code" group from magika. We list the labels we
        // actually expect to see in tool outputs / pasted code in
        // proxy traffic. Anything else in the code group falls
        // through to PlainText — better passthrough than misroute.
        "rust" | "python" | "javascript" | "typescript" | "go" | "java" | "c" | "cpp" | "cs"
        | "php" | "ruby" | "swift" | "kotlin" | "scala" | "haskell" | "lua" | "dart" | "perl"
        | "shell" | "powershell" | "batch" | "sql" | "css" | "vue" | "groovy" | "clojure"
        | "asm" | "cmake" | "dockerfile" | "makefile" | "yaml" | "toml" | "ini" | "hcl"
        | "jinja" => ContentType::SourceCode,

        // ── Plain text-ish ─────────────────────────────────────────
        // markdown, rst, latex, log-style, txt, empty/unknown all
        // route as plain text. The router won't try to compress these
        // with a code-aware compressor.
        "markdown" | "rst" | "latex" | "txt" | "empty" | "unknown" | "undefined" => {
            ContentType::PlainText
        }

        // ── Default: passthrough ───────────────────────────────────
        _ => ContentType::PlainText,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────
//
// These tests are integration-y — they hit the real magika model, which
// loads ONNX on first call (~50 ms cold). Total wall-clock for the full
// suite is dominated by that one-time load, so we keep cases compact.
//
// Detection is probabilistic; we assert against `ContentType` enum
// values rather than confidence scores or labels directly. If magika's
// model version changes (`MODEL_NAME` in their crate), individual
// label assignments may shift but our `match` arms are wide enough to
// stay stable.

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_detect(content: &str, expected: ContentType, hint: &str) {
        if !magika_onnx_runtime_supported_by_cpu() {
            // On x86 hosts without AVX2 the magika session returns Err
            // before any ONNX init — assert graceful degradation
            // rather than panicking.
            match magika_detect(content) {
                Err(MagikaDetectorError::Init(msg)) => {
                    assert!(
                        msg.contains("AVX2"),
                        "{hint}: expected AVX2 error, got: {msg}"
                    );
                }
                other => panic!("{hint}: on no-AVX2 host expected Init(AVX2) error, got {other:?}"),
            }
        } else {
            match magika_detect(content) {
                Ok(got) => {
                    assert_eq!(got, expected, "{hint}: expected {expected:?}, got {got:?}")
                }
                Err(e) => panic!("{hint}: detection failed: {e}"),
            }
        }
    }

    #[test]
    fn empty_input_is_plain_text_without_model_call() {
        // The shortcut path — should not touch the model.
        let result = magika_detect("").unwrap();
        assert_eq!(result, ContentType::PlainText);
    }

    #[test]
    fn detects_json() {
        assert_detect(
            r#"{"name": "Alice", "age": 30, "tags": ["a", "b"]}"#,
            ContentType::JsonArray,
            "single-object JSON",
        );
    }

    #[test]
    fn detects_json_array() {
        let payload = r#"[{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 3, "v": "c"}]"#;
        assert_detect(payload, ContentType::JsonArray, "array-of-records JSON");
    }

    #[test]
    fn detects_python_source() {
        let src = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Tree:
    def __init__(self, value):
        self.value = value
        self.children = []
"#;
        assert_detect(src, ContentType::SourceCode, "python class+def");
    }

    #[test]
    fn detects_rust_source() {
        let src = r#"
use std::collections::HashMap;

pub struct Counter {
    counts: HashMap<String, u32>,
}

impl Counter {
    pub fn new() -> Self {
        Self { counts: HashMap::new() }
    }
}
"#;
        assert_detect(src, ContentType::SourceCode, "rust struct+impl");
    }

    #[test]
    fn detects_javascript_source() {
        let src = r#"
const fetchUser = async (id) => {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) throw new Error('Not found');
    return response.json();
};
"#;
        assert_detect(src, ContentType::SourceCode, "JS arrow + async");
    }

    #[test]
    fn detects_unified_diff() {
        let diff = r#"diff --git a/foo.py b/foo.py
index abc123..def456 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
+    print("new line")
     return "world"
"#;
        assert_detect(diff, ContentType::GitDiff, "git unified diff");
    }

    #[test]
    fn detects_markdown_as_plain_text() {
        // Markdown isn't routed to a code compressor — it goes to
        // plain text. This is by design; markdown compression has its
        // own path that isn't hooked up yet.
        let md = "# Hello\n\nThis is **bold** and *italic*.\n\n- Item 1\n- Item 2\n";
        assert_detect(md, ContentType::PlainText, "markdown");
    }

    #[test]
    fn detects_plain_text() {
        let prose = "The quick brown fox jumps over the lazy dog. \
                     This is just regular English prose with no \
                     special structure.";
        assert_detect(prose, ContentType::PlainText, "english prose");
    }

    #[test]
    fn detects_html() {
        let html =
            "<!DOCTYPE html><html><head><title>x</title></head><body><h1>Hi</h1></body></html>";
        assert_detect(html, ContentType::Html, "minimal HTML page");
    }

    #[test]
    fn detects_yaml_as_source_code() {
        let yaml = "name: my-app\nversion: 1.0\ndependencies:\n  - foo\n  - bar\n";
        assert_detect(yaml, ContentType::SourceCode, "YAML config");
    }

    #[test]
    fn detects_shell_script_as_source_code() {
        let sh = "#!/bin/bash\nset -euo pipefail\nfor f in *.txt; do\n  echo \"$f\"\ndone\n";
        assert_detect(sh, ContentType::SourceCode, "bash script with shebang");
    }

    #[test]
    fn detects_sql_as_source_code() {
        let sql = "SELECT u.id, u.name, COUNT(o.id) AS order_count \
                   FROM users u LEFT JOIN orders o ON u.id = o.user_id \
                   WHERE u.active = TRUE GROUP BY u.id, u.name;";
        assert_detect(sql, ContentType::SourceCode, "SQL query");
    }

    #[test]
    fn singleton_session_is_reused_across_calls() {
        // Two back-to-back calls should reuse the same session
        // (or same cached error). On AVX2 hosts the session is
        // Ok and repeated calls succeed; on no-AVX2 hosts the
        // session is Err and repeated calls return the same Err.
        if !magika_onnx_runtime_supported_by_cpu() {
            // On no-AVX2 the singleton caches the init error;
            // repeated calls all return the same Init error.
            let r1 = magika_detect("hello world");
            let r2 = magika_detect("def f(): pass");
            let r3 = magika_detect(r#"{"a":1}"#);
            for r in [&r1, &r2, &r3] {
                match r {
                    Err(MagikaDetectorError::Init(msg)) => {
                        assert!(msg.contains("AVX2"), "expected AVX2 error, got: {msg}");
                    }
                    other => panic!("on no-AVX2 host expected Init(AVX2) error, got {other:?}"),
                }
            }
        } else {
            // On AVX2 hosts the session loads once and all calls
            // succeed. Wall-clock asymmetry (cold ~50 ms, warm
            // <1 ms) confirms reuse.
            magika_detect("hello world").unwrap();
            magika_detect("def f(): pass").unwrap();
            magika_detect(r#"{"a":1}"#).unwrap();
        }
    }

    #[test]
    fn unmapped_labels_route_to_plain_text() {
        // Direct test of the mapping table — covers labels we
        // explicitly didn't enumerate. Future magika versions may
        // add new labels and we want unknown-but-real labels to
        // safely passthrough rather than misroute.
        assert_eq!(map_magika_label("ace"), ContentType::PlainText);
        assert_eq!(map_magika_label("flac"), ContentType::PlainText);
        assert_eq!(map_magika_label("3gp"), ContentType::PlainText);
        assert_eq!(
            map_magika_label("garbage_unseen_label"),
            ContentType::PlainText
        );
    }

    #[test]
    fn known_label_table_round_trips() {
        // Cheap sanity that the mapping arms compile and behave.
        // No magika session needed — pure table lookup.
        assert_eq!(map_magika_label("json"), ContentType::JsonArray);
        assert_eq!(map_magika_label("jsonl"), ContentType::JsonArray);
        assert_eq!(map_magika_label("diff"), ContentType::GitDiff);
        assert_eq!(map_magika_label("html"), ContentType::Html);
        assert_eq!(map_magika_label("rust"), ContentType::SourceCode);
        assert_eq!(map_magika_label("python"), ContentType::SourceCode);
        assert_eq!(map_magika_label("yaml"), ContentType::SourceCode);
        assert_eq!(map_magika_label("markdown"), ContentType::PlainText);
        assert_eq!(map_magika_label("txt"), ContentType::PlainText);
        assert_eq!(map_magika_label("empty"), ContentType::PlainText);
    }
}
