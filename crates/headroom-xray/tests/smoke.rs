//! Smoke tests for the headroom-xray binary.
//!
//! These tests require Node 20+ + npx on PATH. They are gated by an
//! env var so they don't break CI runs that don't have Node installed.

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;

fn has_node() -> bool {
    std::process::Command::new("node")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

#[test]
fn help_exits_zero_and_shows_codeburn_flags() {
    if !has_node() {
        eprintln!("[skip] node not on PATH");
        return;
    }

    Command::cargo_bin("headroom-xray")
        .unwrap()
        .arg("--help-codeburn")
        .assert()
        .success()
        // CodeBurn's own --help should mention some of its subcommands.
        .stdout(contains("report").or(contains("optimize")));
}

#[test]
fn unknown_codeburn_subcommand_propagates_nonzero_exit() {
    if !has_node() {
        eprintln!("[skip] node not on PATH");
        return;
    }

    Command::cargo_bin("headroom-xray")
        .unwrap()
        .args(["this-subcommand-does-not-exist-zzz"])
        .assert()
        .failure();
}
