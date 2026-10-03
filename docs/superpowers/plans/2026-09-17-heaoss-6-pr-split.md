# HEAOSS-6 Pull Request Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PR #3382 with six focused draft pull requests whose cumulative tip preserves its reviewed HEAOSS-6 behavior.

**Architecture:** Build a linear Git branch stack from current upstream `main`. Each branch introduces one independently useful layer: decisions, policy contracts, evidence contracts, manifest tooling, reusable release builds, then candidate orchestration.

**Tech Stack:** Git, GitHub CLI, Python 3.13, pytest, JSON Schema 2020-12, GitHub Actions, Ruff, mypy, actionlint

**Spec:** `docs/superpowers/specs/2026-09-17-heaoss-6-pr-split-design.md`

## Global Constraints

- The first branch is based on current upstream `main`; each later branch is based on its immediate predecessor.
- Every pull request is opened as a draft and targets the preceding branch.
- The original PR #3382 stays unchanged until all six drafts and the cumulative-equivalence audit exist.
- No unrelated merge-commit changes from PR #3382 may enter the new stack.
- Release publishing behavior remains intact after every executable slice.
- Every PR body states `Part of HEAOSS-6; split from #3382.` and identifies its stack position.

---

### Task 1: Release Qualification Decisions

**Files:**
- Create: `docs/adr/0001-main-release-branch-semantics.md`
- Create: `docs/adr/0002-release-artifact-identity.md`
- Create: `docs/adr/0003-a1-b-release-benchmark.md`
- Create: `docs/adr/0004-black-box-release-benchmarks.md`
- Create: `docs/adr/0005-release-overrides-and-fail-closed-policy.md`
- Existing: `docs/superpowers/specs/2026-09-17-heaoss-6-pr-split-design.md`
- Existing: `docs/superpowers/plans/2026-09-17-heaoss-6-pr-split.md`

**Interfaces:**
- Consumes: HEAOSS-6 decisions preserved on `upstream/pr-3382`.
- Produces: Stable terminology for source, artifact, rollout, policy, benchmark, and override identity used by later PRs.

- [ ] **Step 1: Restore the five reviewed ADRs**

```powershell
git checkout upstream/pr-3382 -- docs/adr/0001-main-release-branch-semantics.md docs/adr/0002-release-artifact-identity.md docs/adr/0003-a1-b-release-benchmark.md docs/adr/0004-black-box-release-benchmarks.md docs/adr/0005-release-overrides-and-fail-closed-policy.md
```

- [ ] **Step 2: Verify documentation integrity**

Run: `git diff --check; rg -n 'TB[D]|TO[D]O' docs/adr docs/superpowers`
Expected: `git diff --check` exits 0 and no unresolved placeholder is present in the new files.

- [ ] **Step 3: Commit the documentation slice**

```powershell
git add docs/adr docs/superpowers
git commit -m "docs(release): define qualification decisions"
```

- [ ] **Step 4: Push and open draft PR 1**

```powershell
git push -u origin jd/heaoss-6-01-release-qualification-decisions
gh pr create --repo headroomlabs-ai/headroom --draft --base main --head JerrettDavis:jd/heaoss-6-01-release-qualification-decisions --title "docs(release): define qualification decisions" --body-file <body-file>
```

### Task 2: Shared Policy Contracts

**Files:**
- Create: `release/contracts/common.schema.json`
- Create: `release/contracts/policy.schema.json`
- Create: `release/contracts/examples/policy.valid.json`
- Create: `release/contracts/README.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `tests/test_release_contracts.py`

**Interfaces:**
- Consumes: ADR terminology from Task 1.
- Produces: JSON Schema definitions referenced as `common.schema.json#/$defs/...` and a complete `policy.schema.json` consumer.

- [ ] **Step 1: Create the stacked branch and restore contract sources**

```powershell
git switch -c jd/heaoss-6-02-release-policy-contracts
git checkout upstream/pr-3382 -- release/contracts/common.schema.json release/contracts/policy.schema.json release/contracts/examples/policy.valid.json release/contracts/README.md pyproject.toml uv.lock
```

- [ ] **Step 2: Add focused policy contract tests**

Extract the schema-registry helpers and policy-specific cases from PR #3382's
`tests/test_release_contracts.py`. The resulting test file must load local
schemas through `referencing.Registry`, validate `policy.valid.json`, reject an
unknown schema version, and enforce policy pass/fail invariants.

- [ ] **Step 3: Validate and commit**

Run: `uv lock --check`

Run: `uv run --extra dev pytest tests/test_release_contracts.py -q`

Run: `uv run --extra dev ruff check tests/test_release_contracts.py`

Expected: all commands exit 0.

```powershell
git add release/contracts tests/test_release_contracts.py pyproject.toml uv.lock
git commit -m "feat(release): define versioned policy contracts"
```

- [ ] **Step 4: Push and open draft PR 2 targeting PR 1's branch**

Use title `feat(release): define versioned policy contracts` and base
`jd/heaoss-6-01-release-qualification-decisions`.

### Task 3: Qualification Evidence Contracts

**Files:**
- Create: `release/contracts/benchmark-result-ref.schema.json`
- Create: `release/contracts/candidate-manifest.schema.json`
- Create: `release/contracts/gate-result.schema.json`
- Create: `release/contracts/integration-result.schema.json`
- Create: `release/contracts/qualification-manifest.schema.json`
- Create: matching files under `release/contracts/examples/`
- Modify: `tests/test_release_contracts.py`

**Interfaces:**
- Consumes: `common.schema.json` definitions and the release policy contract.
- Produces: Complete evidence formats used by candidate tooling and future qualification lanes.

- [ ] **Step 1: Create the branch and restore all remaining schemas/examples**

Create `jd/heaoss-6-03-qualification-evidence-contracts` from Task 2, then
restore the declared files from `upstream/pr-3382`.

- [ ] **Step 2: Restore the complete reviewed contract test suite**

```powershell
git checkout upstream/pr-3382 -- tests/test_release_contracts.py
```

- [ ] **Step 3: Validate and commit**

Run: `uv run --extra dev pytest tests/test_release_contracts.py -q`

Run: `uv run --extra dev ruff check tests/test_release_contracts.py`

Expected: all contract examples and negative invariants pass.

Commit as `feat(release): add qualification evidence contracts`.

- [ ] **Step 4: Push and open draft PR 3 targeting PR 2's branch**

### Task 4: Candidate Manifest CLI

**Files:**
- Create: `scripts/candidate_manifest.py`
- Create: `tests/test_candidate_manifest.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `release/contracts/candidate-manifest.schema.json` and common definitions.
- Produces: `inventory`, `create`, and `verify` CLI commands used by Task 6.

- [ ] **Step 1: Create the branch and restore reviewed CLI files**

Create `jd/heaoss-6-04-candidate-manifest-cli`, then restore the three declared
paths from `upstream/pr-3382`.

- [ ] **Step 2: Validate behavior and types**

Run: `uv run --extra dev pytest tests/test_candidate_manifest.py tests/test_release_contracts.py -q`

Run: `uv run --extra dev ruff check scripts/candidate_manifest.py tests/test_candidate_manifest.py`

Run: `uv run --extra dev mypy scripts/candidate_manifest.py`

Expected: all commands exit 0.

- [ ] **Step 3: Commit, push, and open draft PR 4 targeting PR 3's branch**

Commit as `feat(release): add immutable candidate manifest verifier`.

### Task 5: Read-Only Release Build Workflow

**Files:**
- Create: `.github/workflows/release-build.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `tests/test_release_workflows.py`
- Modify: `docs/content/docs/releases.mdx`

**Interfaces:**
- Consumes: Existing release build and smoke jobs.
- Produces: Reusable `workflow_call` accepting `source_sha`, `manual_version`, and `release_tag`, with version/npm/canonical outputs and retained artifacts.

- [ ] **Step 1: Create the branch and restore the reviewed extraction**

Create `jd/heaoss-6-05-read-only-release-build`, then restore the four declared
paths from `upstream/pr-3382`.

- [ ] **Step 2: Validate workflow structure and release invariants**

Run: `uv run --extra dev pytest tests/test_release_workflows.py -q`

Run: `actionlint .github/workflows/release.yml .github/workflows/release-build.yml`

Expected: 50 or more release workflow tests pass and actionlint emits no findings.

- [ ] **Step 3: Commit, push, and open draft PR 5 targeting PR 4's branch**

Commit as `refactor(release): extract read-only build workflow`.

### Task 6: Immutable Candidate Workflow

**Files:**
- Create: `.github/workflows/candidate-artifact.yml`
- Create: `tests/test_candidate_workflow.py`
- Modify: `docs/content/docs/releases.mdx`

**Interfaces:**
- Consumes: Task 4 CLI and Task 5 reusable workflow.
- Produces: A manual exact-main-SHA candidate artifact and independently verified retained bytes.

- [ ] **Step 1: Create the branch and restore candidate orchestration**

Create `jd/heaoss-6-06-immutable-candidate-workflow`, then restore the workflow
and test from `upstream/pr-3382`. Reconcile the release documentation so only
candidate-specific additions remain in this branch.

- [ ] **Step 2: Run cumulative focused validation**

Run: `uv run --extra dev pytest tests/test_candidate_workflow.py tests/test_candidate_manifest.py tests/test_release_contracts.py tests/test_release_workflows.py -q`

Run: `uv run --extra dev ruff check scripts/candidate_manifest.py tests/test_candidate_manifest.py tests/test_candidate_workflow.py tests/test_release_contracts.py tests/test_release_workflows.py`

Run: `uv run --extra dev mypy scripts/candidate_manifest.py`

Run: `actionlint .github/workflows/candidate-artifact.yml .github/workflows/release-build.yml .github/workflows/release.yml`

Expected: 96 focused tests or the current-main-adjusted equivalent pass; lint,
types, and workflow validation exit 0.

- [ ] **Step 3: Audit cumulative equivalence**

```powershell
git diff --stat upstream/pr-3382...HEAD
git diff upstream/pr-3382 HEAD -- .github docs/adr docs/content/docs/releases.mdx release scripts tests/test_candidate_manifest.py tests/test_candidate_workflow.py tests/test_release_contracts.py tests/test_release_workflows.py pyproject.toml uv.lock .gitignore
```

Expected: only the approved split documents and intentional current-main
reconciliation differ. Investigate and document every other difference.

- [ ] **Step 4: Commit, push, and open draft PR 6 targeting PR 5's branch**

Commit as `feat(release): build immutable exact-SHA candidates`.

### Task 7: Link the Stack and Supersede PR #3382

**Files:**
- Modify: GitHub draft PR bodies only.

**Interfaces:**
- Consumes: URLs and exact head/base SHAs for all six draft PRs.
- Produces: Navigable stack metadata and a non-destructive supersession notice on #3382.

- [ ] **Step 1: Update each draft body with complete previous/next links**

Use `gh pr edit` and verify the rendered bodies with `gh pr view --json body`.

- [ ] **Step 2: Convert #3382 to draft and prepend the supersession notice**

The notice must link all six drafts and retain the prior validation history
below it. Do not close #3382.

- [ ] **Step 3: Verify authoritative GitHub state**

Run `gh pr view` for all seven PRs and verify draft status, base branch, head
branch, mergeability, and URLs. Confirm every new PR head SHA matches the pushed
local branch.

- [ ] **Step 4: Report hosted-check status without claiming unobserved success**

List terminal, pending, or failing checks for each exact head. Hosted checks may
continue after handoff; local validation and PR creation are required before
the task is reported complete.
