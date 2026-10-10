//! Claude Code JSONL transcript parser.
//!
//! Claude Code writes one JSON object per line to
//! `~/.claude/projects/<slug>/<session-id>.jsonl`. We're interested in:
//!
//! - `type == "user"` with content containing `tool_result` blocks
//! - `type == "assistant"` with content containing `tool_use` blocks
//!
//! Each `tool_result` is attributed to the tool that produced it. Each
//! `tool_use` carries the tool name; we link the next `tool_result` back
//! via its `tool_use_id`.

use crate::transcripts::Transcript;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct Line {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    message: Option<serde_json::Value>,
}

pub fn parse(path: &Path) -> Result<Transcript> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut transcript = Transcript::default();
    // Map tool_use_id -> tool name, so we can attribute the subsequent tool_result.
    let mut tool_use_id_to_name: HashMap<String, String> = HashMap::new();

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l,
            Err(_) => continue, // skip unreadable lines
        };

        let parsed: Line = match serde_json::from_str(&line) {
            Ok(p) => p,
            Err(_) => continue, // skip malformed lines (defensive)
        };

        if parsed.kind != "user" && parsed.kind != "assistant" {
            continue;
        }

        let Some(msg) = parsed.message else { continue };
        let Some(content) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };

        for block in content {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match block_type {
                "tool_use" => {
                    let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                    let name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown");
                    if !id.is_empty() {
                        tool_use_id_to_name.insert(id.to_string(), name.to_string());
                    }
                }
                "tool_result" => {
                    let id = block
                        .get("tool_use_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("");
                    let tool_name = tool_use_id_to_name
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| "<orphan>".to_string());
                    let text = stringify_tool_result(block);
                    transcript.push(tool_name, text);
                }
                "text" => {
                    let text = block.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    if text.contains("<system-reminder>") {
                        transcript.push("<system-reminder>", text.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    Ok(transcript)
}

/// Flatten a `tool_result` block's `content` to a single string.
fn stringify_tool_result(block: &serde_json::Value) -> String {
    let content = match block.get("content") {
        Some(c) => c,
        None => return String::new(),
    };
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        let mut acc = String::new();
        for item in arr {
            if let Some(s) = item.get("text").and_then(|t| t.as_str()) {
                acc.push_str(s);
            }
        }
        return acc;
    }
    String::new()
}

/// Find the most recently modified Claude Code session JSONL for the current
/// working directory's Claude Code project, if any.
///
/// Returns None if `~/.claude/projects/` doesn't exist or no session matches.
///
/// Claude Code slug derivation: `/Users/foo/bar` → `-Users-foo-bar`. We get
/// this by replacing `/` with `-` directly (the leading `/` becomes the
/// leading `-` that Claude Code expects).
pub fn latest_session_for_cwd() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let cwd = std::env::current_dir().ok()?;
    let slug = cwd.to_str()?.replace('/', "-");
    let dir = std::path::Path::new(&home)
        .join(".claude")
        .join("projects")
        .join(&slug);
    if !dir.exists() {
        return None;
    }
    let mut newest: Option<(std::path::PathBuf, std::time::SystemTime)> = None;
    for entry in std::fs::read_dir(&dir).ok()? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(modified) = entry.metadata().ok().and_then(|m| m.modified().ok()) else {
            continue;
        };
        if newest.as_ref().map_or(true, |(_, t)| modified > *t) {
            newest = Some((path, modified));
        }
    }
    newest.map(|(p, _)| p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("claude_code_minimal.jsonl")
    }

    #[test]
    fn parses_minimal_fixture() {
        let t = parse(&fixture_path()).expect("parser should succeed");
        let tools: Vec<&str> = t.blocks.iter().map(|b| b.tool.as_str()).collect();
        assert!(tools.contains(&"Bash"), "missing Bash block in {tools:?}");
        assert!(tools.contains(&"Read"), "missing Read block in {tools:?}");
    }

    #[test]
    fn skips_malformed_lines() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, r#"{{ "type": "user", "message": {{ "content": [] }} }}"#).unwrap();
        writeln!(f, r#"this is not json"#).unwrap();
        writeln!(
            f,
            r#"{{ "type": "assistant", "message": {{ "content": [] }} }}"#
        )
        .unwrap();
        let t = parse(f.path()).expect("parser must not bail on malformed lines");
        assert_eq!(t.blocks.len(), 0);
    }
}
