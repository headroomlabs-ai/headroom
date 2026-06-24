#!/usr/bin/env bash
# Create a noop executable shim at $2/$1 suitable for use in PATH during
# native (non-Docker) e2e tests. Mirrors e2e/_lib/shims.py make_shim(noop).
#
# Usage: make_shim.sh <name> <dir>
#
# Exit codes:
#   0 on success
#   2 on usage error

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <name> <dir>" >&2
    exit 2
fi

name="$1"
dir="$2"

mkdir -p "$dir"
path="$dir/$name"
cat >"$path" <<'EOS'
#!/usr/bin/env bash
exit 0
EOS
chmod +x "$path"
echo "$path"
