//! End-to-end footer test: real fixture → real tokens → real footer.

use assert_cmd::Command;
use headroom_xray::footer::{self, FooterContext};
use headroom_xray::tokenize::count_by_tool;
use headroom_xray::transcripts::claude_code;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("claude_code_minimal.jsonl")
}

#[test]
fn footer_renders_top3_counts_only() {
    let fixture = fixture_path();
    let t = claude_code::parse(&fixture).expect("parse");
    let counts = count_by_tool(&t).expect("count");
    let ctx = FooterContext {
        session_path: Some(fixture.clone()),
        aggregate_query: false,
    };
    let rendered = footer::render(&counts, &ctx);

    assert!(rendered.contains("top tool types by token usage"));
    assert!(rendered.contains("claude_code_minimal.jsonl"));
    assert!(
        rendered.contains("Bash") || rendered.contains("Read"),
        "footer missing tool rows:\n{rendered}"
    );
    assert!(rendered.contains("Phase 2"));
    assert!(rendered.contains("────"));
    // No compression promises in Phase 1.
    assert!(!rendered.contains("opportunities"));
    assert!(!rendered.contains("CCR-compressible"));
    // Single-session view → no aggregate caveat.
    assert!(!rendered.contains("spans many sessions"));
}

#[test]
fn aggregate_query_shows_scope_caveat() {
    let fixture = fixture_path();
    let t = claude_code::parse(&fixture).expect("parse");
    let counts = count_by_tool(&t).expect("count");
    let ctx = FooterContext {
        session_path: Some(fixture),
        aggregate_query: true,
    };
    let rendered = footer::render(&counts, &ctx);
    assert!(rendered.contains("spans many sessions"));
}

fn has_node() -> bool {
    std::process::Command::new("node")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

#[test]
fn no_footer_flag_suppresses() {
    if !has_node() {
        eprintln!("[skip] node not on PATH");
        return;
    }
    Command::cargo_bin("headroom-xray")
        .unwrap()
        .args(["--no-footer", "today", "--format", "json"])
        .assert()
        .stdout(
            contains("Headroom:")
                .not()
                .and(contains("Headroom footer:").not()),
        );
}

#[test]
fn env_var_suppresses() {
    if !has_node() {
        eprintln!("[skip] node not on PATH");
        return;
    }
    Command::cargo_bin("headroom-xray")
        .unwrap()
        .env("HEADROOM_XRAY_NO_FOOTER", "1")
        .args(["today", "--format", "json"])
        .assert()
        .stdout(
            contains("Headroom:")
                .not()
                .and(contains("Headroom footer:").not()),
        );
}
