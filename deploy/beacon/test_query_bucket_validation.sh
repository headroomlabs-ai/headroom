#!/usr/bin/env bash
# Regression test for R2_BUCKET validation in query.sh (commit 381fdb9d).
#
# Run from any directory:
#   bash deploy/beacon/test_query_bucket_validation.sh
#
# No external dependencies — pure bash, no DuckDB or Cloudflare credentials
# required.  Accepted-value cases use a temporary empty env file so the script
# reaches the credential check ("R2_ACCOUNT_ID not set"), which is the positive
# signal that bucket validation was passed.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUERY_SH="${SCRIPT_DIR}/query.sh"

FAKE_ENV="$(mktemp)"
trap 'rm -f "$FAKE_ENV"' EXIT

PASS=0
FAIL=0
ERRORS=()

# assert_rejected LABEL VALUE
# Asserts: exit code 1 AND "Invalid R2_BUCKET" in stderr.
assert_rejected() {
  local label="$1"
  local bucket_value="$2"
  local stderr_out exit_code=0

  stderr_out="$(R2_BUCKET="$bucket_value" HEADROOM_ENV_FILE="$FAKE_ENV" \
    bash "$QUERY_SH" 2>&1 >/dev/null)" || exit_code=$?

  local ok=1
  if [[ $exit_code -ne 1 ]]; then
    ok=0
    ERRORS+=("REJECT [$label]: expected exit 1, got $exit_code")
  fi
  if ! grep -qF "Invalid R2_BUCKET" <<< "$stderr_out"; then
    ok=0
    ERRORS+=("REJECT [$label]: expected 'Invalid R2_BUCKET' in stderr; got: $stderr_out")
  fi

  if [[ $ok -eq 1 ]]; then
    echo "PASS (rejected): $label"
    PASS=$((PASS + 1))
  else
    echo "FAIL (rejected): $label"
    FAIL=$((FAIL + 1))
  fi
}

# assert_accepted LABEL VALUE
# Asserts: no "Invalid R2_BUCKET" in stderr AND "R2_ACCOUNT_ID not set" appears,
# confirming the script cleared the bucket guard and reached the credential check.
assert_accepted() {
  local label="$1"
  local bucket_value="$2"
  local stderr_out exit_code=0

  stderr_out="$(R2_BUCKET="$bucket_value" HEADROOM_ENV_FILE="$FAKE_ENV" \
    bash "$QUERY_SH" 2>&1 >/dev/null)" || exit_code=$?

  local ok=1
  if grep -qF "Invalid R2_BUCKET" <<< "$stderr_out"; then
    ok=0
    ERRORS+=("ACCEPT [$label]: unexpected 'Invalid R2_BUCKET' in stderr")
  fi
  if ! grep -qF "R2_ACCOUNT_ID not set" <<< "$stderr_out"; then
    ok=0
    ERRORS+=("ACCEPT [$label]: expected script to reach credential check; stderr: $stderr_out")
  fi

  if [[ $ok -eq 1 ]]; then
    echo "PASS (accepted): $label"
    PASS=$((PASS + 1))
  else
    echo "FAIL (accepted): $label"
    FAIL=$((FAIL + 1))
  fi
}

# --- Adversarial values: must be rejected ---
assert_rejected "path traversal"           "../etc/passwd"
assert_rejected "shell injection"          "bucket; rm -rf /"
assert_rejected "command substitution"     'bucket$(whoami)'
assert_rejected "spaces"                   "bucket with spaces"
assert_rejected "special chars"            'bucket!@#'
assert_rejected "dot"                      "bucket.name"
assert_rejected "slash"                    "bucket/subpath"

# --- Representative valid values: must pass validation ---
assert_accepted "production default"       "headroom-telemetry"
assert_accepted "underscore variant"       "headroom_backup"
assert_accepted "alphanumeric+hyphens"     "my-bucket-123"
assert_accepted "mixed case"               "MyBucket"
assert_accepted "single char"              "a"

# --- Summary ---
TOTAL=$((PASS + FAIL))
if [[ $FAIL -eq 0 ]]; then
  echo "All ${TOTAL} tests passed."
  exit 0
else
  echo "FAILED: ${FAIL} of ${TOTAL} tests." >&2
  for e in "${ERRORS[@]}"; do
    echo "  - $e" >&2
  done
  exit 1
fi
