// Compile a tiny C shim that provides local definitions of the C23
// strtol family (`__isoc23_strtol`, `__isoc23_strtoll`, etc.).
//
// glibc < 2.38 doesn't ship these symbols. Static dependencies in
// `_core.so` (notably the prebuilt ONNX Runtime artifacts compiled
// with gcc 14.x) reference them, so without this shim the wheel
// fails to import on Ubuntu 22.04, Conda envs with libc 2.35, etc.
//
// See `glibc_compat.c` for the full background, including the two
// implementation traps (no `alias` attribute, no `<stdlib.h>`).
// The shim is Linux/glibc-only — macOS, Windows, and musl don't
// ship glibc and don't reference `__isoc23_*`.
//
// Issue: #355 (https://github.com/chopratejas/headroom/issues/355)

fn main() {
    println!("cargo:rerun-if-changed=glibc_compat.c");
    println!("cargo:rerun-if-changed=build.rs");

    // The shim is glibc-specific. Skip on every other target: macOS
    // uses Darwin libc, Windows has MSVCRT, musl handles strtoll
    // identically and never emits __isoc23_*.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os != "linux" || target_env != "gnu" {
        return;
    }

    cc::Build::new()
        .file("glibc_compat.c")
        // -fPIC because we link into a cdylib. -O2 for size — the
        // file is ~10 lines but every byte counts in a wheel that's
        // already 35 MiB.
        .flag_if_supported("-fPIC")
        .opt_level(2)
        .compile("headroom_glibc_compat");

    // Force the linker to pull our shim's objects into _core.so even
    // if at archive-scan time no UND `__isoc23_*` reference exists
    // yet. Without this, the ORT prebuilt static archives — which
    // are downloaded by ort-sys and link AFTER our shim's archive
    // on aarch64 (observed in PR #386's release run) — leave the
    // `__isoc23_*` references unresolved at the .so level even
    // though our archive defines them. The audit then rightly
    // rejects the wheel.
    //
    // `-u <sym>` (a.k.a. `--undefined`) tells the linker: "treat
    // this symbol as undefined at the start of linking, which forces
    // any archive defining it to be scanned and its members pulled
    // in." Once our archive's objects are in, the shim's strong
    // definitions are present in `_core.so` and ORT's later
    // references resolve to them. On x86_64 the ORT archive
    // happened to scan first; on aarch64 it did not, so this gate
    // is the load-bearing fix that makes the shim work uniformly.
    for sym in [
        "__isoc23_strtol",
        "__isoc23_strtoll",
        "__isoc23_strtoul",
        "__isoc23_strtoull",
    ] {
        println!("cargo:rustc-link-arg=-Wl,-u,{sym}");
    }
}
