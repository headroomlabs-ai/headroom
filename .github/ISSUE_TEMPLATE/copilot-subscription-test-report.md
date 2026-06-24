---
name: Copilot Subscription Test Report
about: Report results of testing `headroom wrap copilot --subscription` on Linux/Windows/macOS
title: '[COPILOT-SUB] <OS> test report'
labels: copilot-subscription, testing
assignees: ''
---

<!--
Thanks for helping verify Copilot subscription mode across platforms!
See TESTING-copilot-subscription.md for the step-by-step flows.
Redact your actual token everywhere.
-->

## Environment

- **OS + version**: (e.g., Ubuntu 24.04, Windows 11 23H2, macOS 14.5)
- **Architecture**: (x86_64 / arm64)
- **How you installed headroom**: (pipx/pip `--pre` wheel · Docker install.sh/ps1 · built from source)
- **headroom version**: (`headroom --version`)
- **Copilot CLI version**: (`copilot --version`)
- **Was plain `copilot` logged in before the test?**: yes / no

## Result

- **Command run**:
  ```
  headroom wrap copilot --subscription -- --model gpt-4o -p "Reply with exactly: HEADROOM_OK"
  ```
- **Did it print `HEADROOM_OK`?**: yes / no
- **Worked WITHOUT `GITHUB_COPILOT_TOKEN` (auto-discovery)?**: yes / no / didn't try
- **Worked WITH `GITHUB_COPILOT_TOKEN` set?**: yes / no / didn't try

## Error output (if any)

```
paste any error here
```

## Token storage schema (only if auto-discovery failed)

Helps us fix auto-discovery. **Redact the secret value.**

- Linux: `secret-tool search --all 2>/dev/null | sed -E 's/^secret = .*/secret = <redacted>/'`
- Windows: `cmd /c "cmdkey /list"` (paste the Copilot-related `Target:` line)
- macOS (reference): service `copilot-cli`

```
paste the attribute / Target lines here (secret redacted)
```

## Anything else

(logs from `~/.headroom/logs/proxy.log`, surprises, etc.)
