# ADR 0001: Main and release branch semantics

- Status: Accepted
- Date: 2026-08-13

## Context

Headroom currently develops and releases from `main`. The target lifecycle needs
`main` to continue moving while an exact source revision is qualified. A branch
name alone is not evidence, and creating a `release` branch before qualification
is authoritative would make unqualified code appear publishable.

## Decision

`main` remains the default integration branch and the target for ordinary pull
requests. The future protected `release` branch represents qualified source. It
may receive only an ancestry-preserving promotion of an exact Gate-A-qualified
`main` SHA, a verified metadata-only Release Please change, an authorized hotfix,
or release-control maintenance.

Promotion must leave the qualified candidate SHA reachable from `release`.
Squash and rebase merges are therefore forbidden for promotion pull requests.
Normal pull requests into `main` may retain the repository's usual merge policy.

The first `release` branch must be created from an exact SHA with a passing,
unrevoked qualification manifest. Branch creation, protection, lifecycle
activation, and publication credentials remain explicit administrator actions.
They are not performed by this contract-only change.

Gate A admits source to `release`. Gate B independently authorizes publication
of final distribution bytes. Neither branch membership nor a tag or GitHub
Release substitutes for either gate.

## Consequences

Release Please will eventually watch `release`, but only after the qualification
controller and migration guard are operational. Promotion automation must use an
ancestry-preserving merge. A hotfix made on `release` requires qualification and
forward synchronization to `main`; `release` must not become a second development
branch.
