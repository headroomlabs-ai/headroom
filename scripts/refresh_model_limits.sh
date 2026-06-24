#!/usr/bin/env bash
#
# Refresh the vendored LiteLLM model_prices_and_context_window.json
# used by `crates/headroom-proxy/src/compression/model_limits.rs`.
#
# We vendor the snapshot rather than fetching at build/runtime so the
# proxy binary ships with no network dependency at startup. Operators
# tracking new model releases run this script and commit the diff.
#
# Validation:
#   1. JSON parses
#   2. Contains a known-stable Claude model entry
#   3. Contains a known-stable GPT model entry
# These guard against accidentally vendoring an empty / malformed file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/crates/headroom-proxy/data/model_prices_and_context_window.json"
URL="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

echo "Fetching $URL"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
curl -fsSL "$URL" -o "$TMP"

# Validate the snapshot before swapping it in.
python3 -c "
import json, sys
with open('$TMP') as f:
    data = json.load(f)
if not isinstance(data, dict):
    sys.exit('top-level not an object')
if 'sample_spec' not in data:
    sys.exit('missing sample_spec entry — schema may have changed')
# Spot-check stable entries.
required = ['claude-sonnet-4-5-20250929', 'gpt-4o-mini', 'gpt-4-turbo']
missing = [k for k in required if k not in data]
if missing:
    sys.exit(f'missing required entries: {missing!r}')
print(f'OK: {len(data)} entries, including {required}')
"

mv "$TMP" "$DEST"
trap - EXIT

echo "Updated $DEST"
echo "Run 'cargo test -p headroom-proxy --lib compression::model_limits' to verify."
