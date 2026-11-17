# HEAOSS-6 Pull Request Split Design

## Purpose

Replace the 3,279-line change in PR #3382 with a reviewable stack of six draft
pull requests. Every pull request must have one coherent deliverable, pass its
own focused validation, and leave the repository in a useful state if the stack
stops there.

The final stack must preserve the behavior and safety properties already
reviewed on PR #3382: versioned qualification evidence contracts, fail-closed
candidate manifest verification, exact-main-SHA builds, a read-only build call
tree, deterministic artifact inventory, and independent verification of the
retained candidate bytes.

## Constraints

- Base the first branch on the current upstream `main`.
- Stack every later branch on its immediate predecessor.
- Keep each pull request draft until the whole stack is reconstructed and its
  cumulative tip is equivalent to PR #3382, except for deliberate documentation
  and history changes described here.
- Do not modify or delete the original PR #3382 branch while constructing the
  stack.
- Do not include unrelated changes from merge commits or the dirty primary
  checkout.
- Preserve the release workflow's existing publish behavior at every executable
  step of the stack.
- Use conventional commit messages and include the HEAOSS-6 context and stack
  position in every pull request body.

## Stack Topology

| Position | Branch | Base | Deliverable |
| --- | --- | --- | --- |
| 1 | `jd/heaoss-6-01-release-qualification-decisions` | `main` | Release qualification ADRs and this split specification |
| 2 | `jd/heaoss-6-02-release-policy-contracts` | PR 1 branch | Shared evidence vocabulary and versioned release policy contract |
| 3 | `jd/heaoss-6-03-qualification-evidence-contracts` | PR 2 branch | Candidate, benchmark, integration, gate, and qualification evidence schemas |
| 4 | `jd/heaoss-6-04-candidate-manifest-cli` | PR 3 branch | Local candidate inventory, manifest creation, and fail-closed verification CLI |
| 5 | `jd/heaoss-6-05-read-only-release-build` | PR 4 branch | Reusable read-only build/smoke workflow used by the existing release pipeline |
| 6 | `jd/heaoss-6-06-immutable-candidate-workflow` | PR 5 branch | Manual exact-SHA candidate build, bundling, retention, and independent verification |

GitHub draft PRs use the preceding branch as their base. This makes each diff
show only its own deliverable while retaining an executable cumulative tip.

## PR 1: Release Qualification Decisions

### Scope

- Add ADR 0001 defining authoritative-main release semantics.
- Add ADR 0002 defining artifact and runtime-payload identity.
- Add ADR 0003 defining the A1/B benchmark comparison.
- Add ADR 0004 defining black-box release benchmarks.
- Add ADR 0005 defining overrides and fail-closed policy.
- Add this stack design document.

### Value at Merge

The repository gains an explicit, reviewable release-qualification model before
automation encodes it. No runtime or workflow behavior changes.

### Validation

- Markdown and formatting checks applicable to documentation.
- Confirm the ADRs contain no unresolved placeholders and agree on identity,
  policy, and override terminology.

## PR 2: Shared Policy Contracts

### Scope

- Add `release/contracts/common.schema.json` with shared identity, status,
  rollout, producer, environment, override, and equivalence definitions.
- Add `release/contracts/policy.schema.json` and its valid example.
- Add `release/contracts/README.md` covering schema versioning and validation.
- Add `jsonschema>=4.23.0,<5.0` to the development dependencies and refresh
  `uv.lock`.
- Introduce the reusable schema loading and validation test helpers in
  `tests/test_release_contracts.py`, with positive and negative policy tests.

### Value at Merge

Policy authors can publish and validate a complete versioned release policy.
The shared vocabulary is exercised by a real consumer in the same pull request;
it is not unused scaffolding.

### Validation

- `uv lock --check`
- `uv run --extra dev pytest tests/test_release_contracts.py -q`
- `uv run --extra dev ruff check tests/test_release_contracts.py`

## PR 3: Qualification Evidence Contracts

### Scope

- Add candidate-manifest, benchmark-result-reference, integration-result,
  gate-result, and qualification-manifest schemas.
- Add one valid example for every schema.
- Extend contract tests to validate every example, reject malformed identities,
  enforce pass/fail invariants, enforce Gate B runtime-payload equivalence, and
  reject unsupported schema versions.

### Value at Merge

Qualification producers and consumers have a complete, machine-validated
evidence interchange format. The contracts are independently useful without
the candidate workflow and can be adopted by later qualification lanes.

### Validation

- `uv run --extra dev pytest tests/test_release_contracts.py -q`
- `uv run --extra dev ruff check tests/test_release_contracts.py`
- Validate that all relative `$ref` targets resolve from the checked-in schema
  registry.

## PR 4: Candidate Manifest CLI

### Scope

- Add `scripts/candidate_manifest.py` with `inventory`, `create`, and `verify`
  commands.
- Add `.candidate/` to `.gitignore` for local command output.
- Add focused tests for deterministic ordering, hashes, sizes, package/version
  identity, source SHA, producer repository/workflow/run identity, rollout
  identity, safe qualification eligibility, schema versions, missing files,
  changed bytes, and unsupported embedded schemas.

### Value at Merge

Maintainers can create and independently verify immutable candidate manifests
locally or from any later CI workflow. The CLI is fully functional before it is
wired into GitHub Actions.

### Validation

- `uv run --extra dev pytest tests/test_candidate_manifest.py tests/test_release_contracts.py -q`
- `uv run --extra dev ruff check scripts/candidate_manifest.py tests/test_candidate_manifest.py`
- `uv run --extra dev mypy scripts/candidate_manifest.py`

## PR 5: Read-Only Release Build Workflow

### Scope

- Extract version resolution, package construction, the cross-platform wheel
  matrix, distribution collection, and smoke imports from `release.yml` into
  the reusable `release-build.yml` workflow.
- Make `release.yml` call the reusable workflow, then retain all publishing and
  release creation in `release.yml`.
- Ensure every publisher waits for the complete build/smoke group.
- Preserve the release workflow entry points and published outputs.
- Update release workflow tests and release documentation to describe the new
  boundary.

### Value at Merge

The production release uses a reusable, contents-read-only build boundary with
no publishing job in its call tree. This is immediately useful even without the
candidate workflow and removes duplicated future build logic.

### Validation

- `uv run --extra dev pytest tests/test_release_workflows.py -q`
- `actionlint .github/workflows/release.yml .github/workflows/release-build.yml`
- Hosted PR dry-run must complete all platform builds and smoke-import jobs;
  publishing jobs must remain skipped.

## PR 6: Immutable Candidate Workflow

### Scope

- Add `candidate-artifact.yml` as a manual workflow accepting only a full SHA
  reachable from authoritative `main`.
- Call `release-build.yml` directly under `contents: read` permissions.
- Produce a deterministic artifact inventory and X-001A candidate manifest.
- Verify the manifest before upload, download the retained artifact in a
  separate job, and verify the exact bytes again.
- Reject wrong source, repository, producer, workflow, run, attempt,
  package/version, filename, size, digest, rollout identity, unsafe rollout,
  qualification-ineligible rollout, or unsupported schema.
- Add focused structural workflow tests and document the candidate workflow.

### Value at Merge

Maintainers can build and retain an immutable release candidate from an exact
commit on authoritative `main` without exposing publishing permissions. The
same bytes are verified independently after artifact retention.

### Validation

- `uv run --extra dev pytest tests/test_candidate_workflow.py tests/test_candidate_manifest.py tests/test_release_contracts.py tests/test_release_workflows.py -q`
- `uv run --extra dev ruff check scripts/candidate_manifest.py tests/test_candidate_manifest.py tests/test_candidate_workflow.py tests/test_release_contracts.py tests/test_release_workflows.py`
- `uv run --extra dev mypy scripts/candidate_manifest.py`
- `actionlint .github/workflows/candidate-artifact.yml .github/workflows/release-build.yml .github/workflows/release.yml`
- Hosted fork validation must build all five platform targets, run all smoke
  jobs, emit the candidate, download it independently, and verify its bytes.
  No publishing job may run.

## Reconstruction Method

For each branch, restore only the files or coherent hunks assigned to that pull
request from PR #3382's reviewed tip. Where one original test file spans two
stack positions, split its tests by the behavior introduced at that position.
Commit and validate each branch before creating the next branch from it.

After PR 6, compare its cumulative tree to PR #3382 with `git diff`, allowing
only this specification and any necessary stack metadata. Review every
remaining difference rather than assuming equivalence from file counts.

## Draft Pull Request Format

Each draft body must include:

- `Stack: N of 6` and links to adjacent draft PRs when available.
- The branch/base relationship.
- A concise statement of the independently deliverable value.
- Exact local validation results for that branch.
- A note that hosted validation is pending or a link to its exact run.
- `Part of HEAOSS-6; split from #3382.`
- Runtime rollout safety and rollback notes appropriate to the slice.

## Original PR Transition

Keep #3382 unchanged until all six draft heads exist and the cumulative tip has
passed local equivalence and validation checks. Then convert #3382 to draft and
replace its leading status section with a supersession notice linking the six
drafts. Do not close it until the stack is confirmed visible and complete; its
existing discussion and hosted validation remain useful review evidence.

## Completion Criteria

The split is complete only when:

1. Six draft PRs exist with the topology above.
2. Every PR diff contains only its declared slice and has passing focused local
   validation.
3. PR 5 demonstrates unchanged release build/smoke behavior with publication
   skipped on PR validation.
4. PR 6 demonstrates candidate creation and independent byte verification with
   no publishing permissions or publishing jobs in its call tree.
5. The cumulative PR 6 tree is reconciled against PR #3382 and every difference
   is documented and intentional.
6. PR #3382 is a draft with a supersession notice linking the complete stack.
