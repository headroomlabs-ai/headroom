#!/usr/bin/env bash
# Install git hooks for the Headroom repo:
#   1. pre-commit  — repo pre-commit checks (ruff, mypy, sync-plugin-versions)
#   2. commit-msg  — conventional-commit enforcement via commitlint
#   3. pre-push    — full ci-precheck (cargo fmt/clippy/test + python suite)
#
# Why pre-push was added: the 2026-04-27 push hit five CI failures that could
# all have been caught locally — cargo fmt drift, an x86_64-apple-darwin wheel
# that the project doesn't actually need, missing Rust extension in two CI
# lanes, and a commitlint warning treated as an error.
#
# Why pre-commit was added: PR #772 merged with inline-comment spacing and
# import-order violations because the ruff pre-commit hook in
# .pre-commit-config.yaml was never installed for contributors.
#
# Idempotent. Re-running is safe. Skips installation if `.git/hooks/` is
# missing (e.g. running outside a git checkout).

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .git/hooks ]]; then
    echo "error: .git/hooks/ not found — run from a git checkout root" >&2
    exit 1
fi

if ! command -v npx &>/dev/null; then
    echo "error: npx not found — install Node 18+ before installing Headroom's git hooks." >&2
    exit 1
fi

HOOK_PATH=".git/hooks/pre-push"

cat > "$HOOK_PATH" <<'HOOK_EOF'
#!/usr/bin/env bash
# Headroom pre-push hook — runs `make ci-precheck` so CI never finds a
# bug a local check could have caught.
#
# Skip with: `git push --no-verify`. Use sparingly — every skip is a roll
# of the dice on a CI break.

set -euo pipefail

# Skip the hook entirely when push goes to a ref that is not on the main
# tracking branches we gate. Adjust the pattern below if more branches
# need gating.
remote="$1"
url="$2"

while IFS=' ' read -r local_ref local_sha remote_ref remote_sha; do
    # Empty local_sha means a delete; nothing to verify.
    if [[ "$local_sha" == "0000000000000000000000000000000000000000" ]]; then
        continue
    fi
    echo "── pre-push: running 'make ci-precheck' before pushing $local_ref → $remote_ref"
done

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f .venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        echo "warn: no VIRTUAL_ENV set and no .venv/ found — python checks may use the wrong interpreter" >&2
    fi
fi

if make ci-precheck; then
    exit 0
else
    echo ""
    echo "❌ pre-push: 'make ci-precheck' failed. Fix the issues above before pushing."
    echo "   To bypass (NOT recommended): git push --no-verify"
    exit 1
fi
HOOK_EOF

chmod +x "$HOOK_PATH"

echo "✅ installed: $HOOK_PATH"
echo "   Runs 'make ci-precheck' before every git push."
echo "   Bypass (use sparingly): git push --no-verify"

# Install pre-commit hooks (repo checks on every commit).
# Prefer the project venv over a global install so contributors always run the
# pinned version. Resolution order: active $VIRTUAL_ENV → .venv → global PATH.
PRE_COMMIT_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/pre-commit" ]]; then
    PRE_COMMIT_BIN="${VIRTUAL_ENV}/bin/pre-commit"
elif [[ -x .venv/bin/pre-commit ]]; then
    PRE_COMMIT_BIN=".venv/bin/pre-commit"
elif command -v pre-commit &>/dev/null; then
    PRE_COMMIT_BIN="pre-commit"
fi

if [[ -n "$PRE_COMMIT_BIN" ]]; then
    "$PRE_COMMIT_BIN" install
    "$PRE_COMMIT_BIN" install --hook-type commit-msg
    echo "✅ installed: .git/hooks/pre-commit (repo pre-commit checks via pre-commit)"
    echo "✅ installed: .git/hooks/commit-msg (conventional commit enforcement via commitlint)"
else
    echo "error: pre-commit not found — run 'pip install -e .[dev]' first, then re-run this script." >&2
    exit 1
fi
