#!/usr/bin/env bash
# Set up this checkout WITHOUT a Rust toolchain.
#
# The Python package hard-requires the compiled `headroom._core` extension.
# Instead of building it (maturin + cargo), this script:
#   1. creates a venv (uv, Python 3.13 by default),
#   2. installs the matching upstream PyPI wheel `headroom-ai[proxy]==<version>`
#      to pull in every runtime dependency and the prebuilt `_core` extension,
#   3. copies `_core.abi3.so` into ./headroom/ (git-ignored),
#   4. uninstalls the upstream package so THIS checkout's code is what runs,
#      and points the venv at the checkout via a `.pth` file (no PYTHONPATH
#      needed — Claude Code launches the MCP server with its own environment),
#   5. writes a `headroom` launcher into <venv>/bin.
#
# Usage:  bash scripts/install_no_rust.sh
# Env:    VENV_DIR (default .venv)   PYTHON_VERSION (default 3.13)
#
# The only network access is PyPI (dependencies + the one upstream wheel).

set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd -P)"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

log() { printf '[install_no_rust] %s\n' "$*" >&2; }
fail() { printf '[install_no_rust] error: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || fail "uv not found (brew install uv)"

VERSION="$(grep -E '^version = ' pyproject.toml | head -1 | sed -E 's/version = "([^"]+)"/\1/')"
[[ -n "$VERSION" ]] || fail "could not read version from pyproject.toml"

log "creating venv $VENV_DIR (python $PYTHON_VERSION)"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR" >/dev/null
PY="$REPO/$VENV_DIR/bin/python"

log "installing upstream headroom-ai[proxy]==$VERSION for its dependencies + prebuilt _core"
uv pip install --python "$PY" -q "headroom-ai[proxy]==$VERSION" pytest pytest-asyncio pytest-xdist respx

SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CORE_SRC="$(ls "$SITE"/headroom/_core*.so 2>/dev/null | head -1)"
[[ -n "$CORE_SRC" ]] || fail "no _core*.so found in the upstream wheel at $SITE/headroom"
cp "$CORE_SRC" "$REPO/headroom/"
log "copied $(basename "$CORE_SRC") into ./headroom/"

log "removing the upstream package (dependencies stay); pointing the venv at this checkout"
uv pip uninstall --python "$PY" -q headroom-ai
printf '%s\n' "$REPO" > "$SITE/headroom-checkout.pth"

LAUNCHER="$REPO/$VENV_DIR/bin/headroom"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
exec "$PY" -m headroom.cli "\$@"
EOF
chmod +x "$LAUNCHER"

log "verifying"
"$PY" -c 'import headroom, headroom._core, headroom.proxy.server; print("headroom from", headroom.__file__)' >&2
"$LAUNCHER" --version >&2

cat >&2 <<EOF

Done. Put the launcher on your PATH so Claude Code's MCP/hook entries can find it:
  ln -sf "$LAUNCHER" ~/.local/bin/headroom
Then:
  headroom wrap claude
EOF
