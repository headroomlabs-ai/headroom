//! CodeBurn subprocess wrapping.
//!
//! Phase 1 delegates the dashboard layer to CodeBurn (MIT, 25 agents)
//! via `npx codeburn@<PINNED_VERSION>`. We forward stdin/stdout/stderr
//! and propagate the exit code.
//!
//! Pinned version: bump deliberately. The version string here is the
//! single source of truth for the CodeBurn release Headroom xray ships
//! against.

use std::io::Write;
use std::process::Stdio;
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

pub const PINNED_CODEBURN_VERSION: &str = "0.9.11";

#[derive(Debug, Error)]
pub enum CodeBurnError {
    #[error("Failed to spawn `npx codeburn@{version}`: {source}")]
    Spawn {
        version: String,
        source: std::io::Error,
    },

    #[error("CodeBurn exited without a status code (killed by signal?)")]
    NoExitCode,
}

/// Spawn `npx codeburn@<PINNED_VERSION> [args...]`.
///
/// Returns the exit code from CodeBurn. Stdout and stderr are streamed
/// to the parent process (line-buffered, no capture). Stdin is inherited.
///
/// If `capture_stdout` is Some, stdout is written there in addition to
/// the parent's stdout (used by the footer pipeline to scrape the JSON
/// CodeBurn emits when `--format json` is passed).
pub async fn run(
    args: &[String],
    capture_stdout: Option<&mut Vec<u8>>,
) -> Result<i32, CodeBurnError> {
    let pkg = format!("codeburn@{PINNED_CODEBURN_VERSION}");
    let mut cmd = Command::new("npx");
    cmd.arg(&pkg).args(args);

    // Stdin always inherited. Stderr always passthrough.
    cmd.stderr(Stdio::inherit());
    cmd.stdin(Stdio::inherit());

    // Stdout: passthrough by default, piped if we're capturing.
    if capture_stdout.is_some() {
        cmd.stdout(Stdio::piped());
    } else {
        cmd.stdout(Stdio::inherit());
    }

    let mut child = cmd.spawn().map_err(|e| CodeBurnError::Spawn {
        version: PINNED_CODEBURN_VERSION.to_string(),
        source: e,
    })?;

    if let Some(buf) = capture_stdout {
        let stdout = child.stdout.take().expect("piped");
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        let mut parent_stdout = std::io::stdout();
        loop {
            line.clear();
            let n = reader.read_line(&mut line).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            buf.extend_from_slice(line.as_bytes());
            // Also forward to parent stdout so the user still sees output.
            parent_stdout.write_all(line.as_bytes()).ok();
        }
        parent_stdout.flush().ok();
    }

    let status = child.wait().await.map_err(|e| CodeBurnError::Spawn {
        version: PINNED_CODEBURN_VERSION.to_string(),
        source: e,
    })?;

    status.code().ok_or(CodeBurnError::NoExitCode)
}
