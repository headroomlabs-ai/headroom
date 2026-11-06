//! Verify clean failure when Node is not on PATH.

use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn missing_node_produces_clean_error() {
    let mut cmd = Command::cargo_bin("headroom-xray").unwrap();
    // Run with an empty PATH (no node, no npx). Forward an args list so we
    // get past arg parsing into the subprocess pipeline.
    cmd.env_clear()
        .env("PATH", "/nonexistent")
        .arg("report")
        .assert()
        .failure() // non-zero exit
        .stderr(contains("Node 20+"));
}
