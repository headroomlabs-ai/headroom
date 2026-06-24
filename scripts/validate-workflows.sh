#!/usr/bin/env bash
set -euo pipefail

actionlint .github/workflows/*.yml

run_act() {
  local attempt=1
  local max_attempts=3
  local delay_seconds=5

  while true; do
    if "$@"; then
      return 0
    fi

    if (( attempt >= max_attempts )); then
      return 1
    fi

    echo "act dry-run failed on attempt ${attempt}/${max_attempts}; retrying in ${delay_seconds}s..." >&2
    sleep "${delay_seconds}"
    attempt=$((attempt + 1))
    delay_seconds=$((delay_seconds * 2))
  done
}

run_act act workflow_dispatch -W .github/workflows/release.yml -e .github/act/dry-run.json -n
# release.yml's main trigger is now `release: published` (release-please
# emits this event when its release PR is merged). The earlier `push`
# trigger on main was removed in PR #495 to gate PyPI uploads behind
# the bot's release-PR pattern. Simulate the new trigger here so the
# validation step exercises the same code path CI actually fires on.
run_act act release -W .github/workflows/release.yml -e .github/act/release-published.json -n
run_act act push -W .github/workflows/release-please.yml -e .github/act/push-feat.json -n
run_act act pull_request_target -W .github/workflows/pr-health.yml -e .github/act/pr-governance-invalid.json -n
run_act act pull_request_target -W .github/workflows/pr-health.yml -e .github/act/pr-governance-valid.json -n
run_act act pull_request -W .github/workflows/docs.yml -n
run_act act workflow_dispatch -W .github/workflows/docs.yml -n
run_act act workflow_dispatch -W .github/workflows/docker.yml -e .github/act/docker-version.json -n
