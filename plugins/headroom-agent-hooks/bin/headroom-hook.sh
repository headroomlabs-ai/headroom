#!/bin/sh
set -u

if [ -n "${HEADROOM_BIN:-}" ] && [ -x "$HEADROOM_BIN" ]; then
    exec "$HEADROOM_BIN" init hook ensure
fi

if command -v headroom >/dev/null 2>&1; then
    exec headroom init hook ensure
fi

home=${HOME:-}
for prefix in \
    "$home/.local/bin" \
    "$home/.local/share/uv/tools/headroom-ai/bin" \
    "${PIPX_HOME:-$home/.local/pipx}/venvs/headroom-ai/bin" \
    "/opt/homebrew/bin" \
    "/usr/local/bin"
do
    for executable in headroom headroom.exe
    do
        candidate="$prefix/$executable"
        if [ -x "$candidate" ]; then
            exec "$candidate" init hook ensure
        fi
    done
done

try_python() {
    candidate=$1
    if command -v "$candidate" >/dev/null 2>&1; then
        candidate=$(command -v "$candidate")
    elif [ ! -x "$candidate" ]; then
        return
    fi
    if "$candidate" -c 'import headroom' >/dev/null 2>&1; then
        exec "$candidate" -m headroom.cli init hook ensure
    fi
}

try_python "${HEADROOM_PYTHON:-}"
try_python python3
try_python python

printf '%s\n' "headroom: CLI not found; install with 'uv tool install headroom-ai' or set HEADROOM_BIN; compression hooks are inactive." >&2
exit 0
