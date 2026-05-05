#!/usr/bin/env bash
# Build + install the Rust extension (headroom._core) into the active venv.
#
# With single-wheel architecture (post-#355), `pip install -e .` invokes
# maturin (declared in pyproject.toml's `[build-system]`) which builds the
# Rust extension and installs it into site-packages alongside the Python
# source. Earlier versions of this script symlinked the .so into the
# in-tree `headroom/` directory because the dual-package layout left the
# .so in `crates/headroom-py/python/headroom/`. That dance is no longer
# needed — maturin places the .so directly in the editable install's
# overlay and Python's import system finds it.
#
# Idempotent. Safe to run repeatedly.

set -euo pipefail

cd "$(dirname "$0")/.."

log() {
    printf '[build_rust_extension] %s\n' "$*" >&2
}

fail() {
    printf '[build_rust_extension] error: %s\n' "$*" >&2
    exit 1
}

# Pre-flight: a venv should be active. The install would otherwise write
# into the system Python.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log "warning: VIRTUAL_ENV is unset; pip will install into the system Python."
    log "         If that is not what you want, abort and 'source .venv/bin/activate' first."
fi

if ! command -v cargo >/dev/null 2>&1; then
    fail "cargo not found on PATH. Install Rust toolchain (rustup) first."
fi

# Build + install in one shot. The `[build-system] build-backend = "maturin"`
# in pyproject.toml means pip drives maturin under the hood. The resulting
# wheel contains both the Python source and the compiled `headroom/_core.so`,
# and pip installs them into the editable overlay together.
log "pip install -e . (drives maturin via build-backend)"
python -m pip install -e . || fail "pip install -e . failed (see output above)"

# End-to-end verification — same shape as Phase A0's startup smoke check.
log "verifying \`from headroom._core import DiffCompressor, SmartCrusher\`"
python -c '
import sys
try:
    from headroom._core import DiffCompressor, SmartCrusher
except Exception as exc:
    print(f"verify FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
print(f"verify OK: DiffCompressor={DiffCompressor!r}, SmartCrusher={SmartCrusher!r}")
' || fail "import verification failed (see above)"

log "headroom._core build + install + verify: OK"
