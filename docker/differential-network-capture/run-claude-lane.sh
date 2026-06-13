#!/usr/bin/env sh
set -eu

echo "running Claude Code lane: ${CLAUDE_LANE:-unknown}" >&2

if [ -n "${NODE_EXTRA_CA_CERTS:-}" ]; then
  i=0
  while [ ! -f "$NODE_EXTRA_CA_CERTS" ] && [ "$i" -lt 100 ]; do
    i=$((i + 1))
    sleep 0.1
  done
fi

if [ -n "${CLAUDE_COMMAND:-}" ]; then
  sh -lc "$CLAUDE_COMMAND"
  exit $?
fi

if [ -n "${CLAUDE_ARGS:-}" ]; then
  # shellcheck disable=SC2086
  claude ${CLAUDE_ARGS} -p "${CLAUDE_PROMPT:-Summarize this repository in one sentence.}"
else
  claude -p "${CLAUDE_PROMPT:-Summarize this repository in one sentence.}"
fi
