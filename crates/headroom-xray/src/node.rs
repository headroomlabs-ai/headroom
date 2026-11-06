//! Node + npx detection.
//!
//! Phase 1 requires Node ≥20 (CodeBurn's minimum). We fail loudly with an
//! actionable hint when the runtime is missing — per the project's
//! "no silent fallbacks" rule.

use std::process::Command;
use thiserror::Error;

const MIN_NODE_MAJOR: u32 = 20;

#[derive(Debug, Error)]
pub enum NodeError {
    #[error(
        "Node 20+ is required for `headroom xray` (which delegates to CodeBurn via npx).\n\
         No `node` binary found on PATH.\n\n\
         Install via Homebrew:  brew install node\n\
                       apt:    apt install nodejs npm\n\
                       nvm:    nvm install --lts\n\n\
         Then re-run `headroom xray --help`."
    )]
    NotFound,

    #[error(
        "Node {found} found on PATH, but Node {min}+ is required for `headroom xray`.\n\
         Upgrade via your package manager and re-run."
    )]
    TooOld { found: String, min: u32 },

    #[error(
        "`npx` was not found alongside Node on PATH.\n\
         npx is shipped with npm (Node 20+ includes both). Reinstall Node from\n\
         https://nodejs.org/ if your distribution has split it out."
    )]
    NpxMissing,

    #[error("Failed to invoke `node --version`: {0}")]
    InvocationFailed(#[from] std::io::Error),
}

/// Probe the environment and return Ok(()) if Node ≥20 + npx are usable.
pub fn check() -> Result<(), NodeError> {
    // node --version
    let node_out = match Command::new("node").arg("--version").output() {
        Ok(out) => out,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(NodeError::NotFound),
        Err(e) => return Err(NodeError::InvocationFailed(e)),
    };

    if !node_out.status.success() {
        return Err(NodeError::NotFound);
    }

    let raw: &str = std::str::from_utf8(&node_out.stdout).map_err(|e| {
        NodeError::InvocationFailed(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    })?;
    let version_str = raw.trim().trim_start_matches('v');
    let major: u32 = version_str
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| NodeError::TooOld {
            found: version_str.to_string(),
            min: MIN_NODE_MAJOR,
        })?;

    if major < MIN_NODE_MAJOR {
        return Err(NodeError::TooOld {
            found: version_str.to_string(),
            min: MIN_NODE_MAJOR,
        });
    }

    // npx --version (presence is enough; we don't need a min)
    match Command::new("npx").arg("--version").output() {
        Ok(out) if out.status.success() => Ok(()),
        Ok(_) => Err(NodeError::NpxMissing),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(NodeError::NpxMissing),
        Err(e) => Err(NodeError::InvocationFailed(e)),
    }
}

#[cfg(test)]
mod tests {
    // Pure parsing tests — no subprocess invocation.
    use super::MIN_NODE_MAJOR;

    fn parse_major(s: &str) -> Option<u32> {
        let v = s.trim().trim_start_matches('v');
        v.split('.').next().and_then(|s| s.parse().ok())
    }

    #[test]
    fn parses_v20() {
        assert_eq!(parse_major("v20.10.0"), Some(20));
    }

    #[test]
    fn parses_no_prefix() {
        assert_eq!(parse_major("18.17.1"), Some(18));
    }

    #[test]
    fn parses_below_minimum() {
        assert!(parse_major("v19.0.0").unwrap() < MIN_NODE_MAJOR);
    }
}
