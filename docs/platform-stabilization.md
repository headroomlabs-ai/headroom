# Platform Stabilization Matrix

This matrix is the hardening source of truth for install, startup, runtime, cache, and compression behavior across Linux, macOS, and Windows. The machine-readable matrix lives in `docs/platform-feature-matrix.json`; CI tests validate its shape so every feature has explicit platform status and test evidence.

## Status Values

- `covered`: unit, integration, or native e2e coverage exists for the platform.
- `partial`: coverage exists, but a named gap remains.
- `gap`: no meaningful coverage exists yet.
- `blocked`: coverage is intentionally excluded by a known external blocker.

## Current Priorities

1. Keep install/setup/run idempotent. Persistent starts must not create duplicate proxy instances unless a future explicit opt-in exists.
2. Keep compression fail-open. Cold model downloads, saturated executors, or timeout paths must pass through unchanged traffic instead of hanging agents.
3. Keep cache behavior visible. CCR persistence and TOIN skip recommendations must have restart and served-recommendation coverage.
4. Keep Windows honest. Windows-specific tests should run locally and in CI whenever they do not require the currently blocked native wheel build.

## Issue 1843 Coverage Map

- Windows service quoting: `install_windows_service`
- Duplicate startup processes: `single_instance_start`
- Slow compression and hangs: `compression_fail_open`
- CCR restart persistence: `ccr_persistence`
- TOIN recommendation wiring: `toin_skip_recommendations`
- Install/setup/run sanity: `install_apply_python`, `init_cli`, `wrap_prepare_only`

When adding or closing a hardening item, update `docs/platform-feature-matrix.json` in the same PR as the implementation or test change.
