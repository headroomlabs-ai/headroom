# Plan: Address PR #1356 Review Comments

**PR:** https://github.com/headroomlabs-ai/headroom/pull/1356
**Reviewer:** JerrettDavis
**Branch:** `fix/wrap-port-conflict-cleanup` (fork: lennney/headroom)
**Base:** `headroomlabs-ai/headroom/main`

## Summary

Address 3 review issues + clean up remaining PR state.

## Task 1: Remove `--no-optimize` commit from this PR

**File:** `headroom/cli/wrap.py`
**Action:** Revert the second commit, save it as a patch for future PR.

```bash
# Commit 3bef349 saved to /tmp/headroom-pr/
git checkout fix/wrap-port-conflict-cleanup
git reset --hard HEAD~1    # Remove 3bef349, keep 630ca1c only
```

**Verification:**
- `git log --oneline` shows only `630ca1c fix(wrap): detect stale proxy...`
- `git diff HEAD~1..HEAD` shows only the stale-proxy changes

## Task 2: Fix macOS `_is_headroom_proxy` — add `ps` fallback

**File:** `headroom/cli/wrap.py`
**Action:** Rewrite `_is_headroom_proxy` to use a cross-platform `_read_process_cmdline` helper.

**Current (broken on macOS):**
```python
def _is_headroom_proxy(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="replace")
        return "headroom" in cmdline and "proxy" in cmdline
    except OSError:
        return False  # macOS always ends up here
```

**Fix:**
```python
def _is_headroom_proxy(pid: int) -> bool:
    """Check whether *pid* is a headroom proxy process (by its cmdline).
    
    Uses ``/proc/<pid>/cmdline`` (Linux) or ``ps`` (macOS/BSD fallback).
    Returns ``False`` when the cmdline can't be read.
    """
    cmdline = _read_process_cmdline(pid)
    if cmdline is None:
        return False
    return "headroom" in cmdline and "proxy" in cmdline


def _read_process_cmdline(pid: int) -> str | None:
    """Read the command-line of *pid*, cross-platform.
    
    Linux: reads ``/proc/<pid>/cmdline``.
    macOS/other: ``ps -p <pid> -o command=`` (no args).
    Returns ``None`` on failure.
    """
    # Linux path
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="replace")
    except OSError:
        pass
    # macOS / BSD fallback
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None
```

**Verification:**
- Existing tests still pass (they mock `Path.read_bytes`)
- `_read_process_cmdline` returns `None` on platforms where neither works

## Task 3: Fix unused `orig_iterdir` lint error

**File:** `tests/test_cli/test_wrap_helpers.py`
**Action:** Remove dead code block (lines ~905-921) that assigns `orig_iterdir` but never uses it.

Lines to remove (lines 903-921 in the test file):
```python
        called = False

        def _mock_iterdir(proc_dir):
            nonlocal called
            ...
            return iter([])

        # For fd_dir.iterdir() we need the fake to return our socket link
        orig_iterdir = Path.iterdir
        itercount = [0]
```

The actual mock implementation starts from `_patched_iterdir` below. The `_mock_iterdir` + `called` + `orig_iterdir` block is leftover dead code.

**Verification:**
- `ruff check tests/test_cli/test_wrap_helpers.py` shows no F841
- All tests still pass

## Task 4: Verify merge conflicts resolved

**Action:** After removing the `--no-optimize` commit, check if the `status: has conflicts` label still applies.

If merge conflicts remain (likely due to the root-commit nature of 630ca1c), resolve by:
1. Create a clean branch from upstream/main
2. Cherry-pick only the relevant changes from 630ca1c
3. Force push the new branch

**Triggers:** Only if `gh pr view 1356` still shows `DIRTY` after removing the commit.

## Task 5: Update PR description

Remove the `--no-optimize` / `#1360` references from the PR body.

## Acceptance Criteria

- [ ] PR has only 1 commit (stale-proxy cleanup, no `--no-optimize`)
- [ ] macOS `_is_headroom_proxy` uses `ps` fallback when `/proc` unavailable
- [ ] `ruff check` passes with no F841 (unused variable)
- [ ] `python -m pytest tests/test_cli/test_wrap_helpers.py::TestEnsurePortFree -v` — all 11+ tests pass
- [ ] PR body no longer references `--no-optimize` / `#1360`
- [ ] PR is in `ready for review` state (no conflicts)
