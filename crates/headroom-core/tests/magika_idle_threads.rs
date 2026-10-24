//! Magika must not leave an ONNX Runtime intra-op thread pool behind (#4187).
//!
//! ORT's pool threads spin-wait between inferences, so a pool sized to the
//! core count pegs idle CPUs in a long-lived proxy. The magika session is a
//! process-global singleton and the assertion counts OS threads, so this file
//! holds exactly ONE `#[test]` fn and must run in its own process.
#![cfg(all(feature = "ml", any(target_os = "linux", target_os = "macos")))]

use std::time::Duration;

use headroom_core::transforms::magika_detector::{magika_detect, MagikaDetectorError};

#[cfg(target_os = "linux")]
fn os_thread_count() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("read /proc/self/task")
        .count()
}

#[cfg(target_os = "macos")]
fn os_thread_count() -> usize {
    let out = std::process::Command::new("ps")
        .args(["-M", "-p", &std::process::id().to_string()])
        .output()
        .expect("run ps -M");
    // One header line, then one line per thread.
    String::from_utf8_lossy(&out.stdout).lines().count() - 1
}

#[test]
fn magika_session_does_not_spawn_intra_op_pool() {
    let before = os_thread_count();
    match magika_detect("fn main() { println!(\"hi\"); }") {
        Ok(_) => {}
        // With an explicit ORT_DYLIB_PATH (as in CI) the runtime is present,
        // so an init failure is a real session-build regression, not a skip.
        Err(MagikaDetectorError::Init(reason))
            if std::env::var("ORT_DYLIB_PATH").map_or(true, |v| v.trim().is_empty()) =>
        {
            eprintln!("skipping: magika runtime unavailable on this host: {reason}");
            return;
        }
        Err(e) => panic!("magika detection failed: {e}"),
    }
    // Let the short-lived `magika-init` thread finish exiting.
    std::thread::sleep(Duration::from_millis(200));
    let after = os_thread_count();
    // ORT and macOS may leave up to 2 idle background threads (ORT telemetry,
    // libdispatch); neither spins. A pool adds its workers on top (7 on a
    // 16-core Mac), but on very small hosts a 1-2 worker pool can hide inside
    // this allowance, so a pass there is not proof of no spinning.
    let extra = after.saturating_sub(before);
    assert!(
        extra <= 2,
        "magika session left {extra} extra OS thread(s) running (before={before}, after={after})"
    );
}
