//! Magika must not leave an ONNX Runtime intra-op thread pool behind (#4187).
//!
//! ORT's pool threads spin-wait between inferences, so a pool sized to the
//! core count pegs idle CPUs in a long-lived proxy. The magika session is a
//! process-global singleton and the assertion counts OS threads, so this file
//! holds exactly ONE `#[test]` fn and must run in its own process.
#![cfg(all(feature = "ml", any(target_os = "linux", target_os = "macos")))]

use std::time::Duration;

use headroom_core::transforms::magika_detector::magika_detect;

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

/// Thread count once short-lived threads (e.g. `magika-init`) have exited.
fn settled_thread_count() -> usize {
    std::thread::sleep(Duration::from_millis(200));
    os_thread_count()
}

#[test]
fn magika_session_does_not_spawn_intra_op_pool() {
    let Some(ort_dylib) = std::env::var("ORT_DYLIB_PATH")
        .ok()
        .filter(|v| !v.trim().is_empty())
    else {
        eprintln!("skipping: set ORT_DYLIB_PATH to an onnxruntime library to run this test");
        return;
    };
    ort::init_from(&ort_dylib)
        .expect("load ONNX Runtime from ORT_DYLIB_PATH")
        .commit();

    // Sessions also start whatever their execution provider needs. ort's
    // default policy (MaxEfficiency, i.e. prefer NPU) picks Core ML on the
    // Apple Neural Engine once ORT 1.31 registers Core ML as a built-in EP,
    // adding 5-7 idle ANE threads (#4191). The control goes through the same
    // EP selection, so measure the allowance there instead of guessing it.
    let start = settled_thread_count();
    let control = magika::Session::builder()
        .with_intra_threads(1)
        .build()
        .expect("build pool-free control session");
    let control_alive = settled_thread_count();
    let allowance = control_alive.saturating_sub(start);
    drop(control);

    let before = settled_thread_count();
    magika_detect("fn main() { println!(\"hi\"); }").expect("magika detection failed");
    let after = settled_thread_count();
    // The allowance also covers threads the control leaves running for good
    // (2-4 on macOS), so on a low-core host a pool that small can still hide
    // inside it; a pass there is not proof of no spinning.
    let extra = after.saturating_sub(before);
    eprintln!(
        "OS threads: start={start}, control alive={control_alive}, control dropped={before} \
         ({} kept); magika session added {extra} (allowance {allowance})",
        before.saturating_sub(start)
    );
    assert!(
        extra <= allowance,
        "magika session left {extra} extra OS thread(s) running, a pool-free session \
         adds {allowance} (before={before}, after={after})"
    );
}
