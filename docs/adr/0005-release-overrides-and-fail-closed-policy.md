# ADR 0005: Release overrides and fail-closed policy

- Status: Accepted
- Date: 2026-08-13

## Context

Headroom product transforms may fail open to preserve requests, but release
control makes a different safety decision: publication must not proceed without
complete, matching, current evidence. Operational exceptions still need an
auditable representation.

## Decision

Release evidence uses exactly four policy statuses: `pass`, `fail`,
`inconclusive`, and `skipped_by_policy`. Missing, stale, invalid, mismatched,
revoked, unauthorized, or schema-incompatible required evidence cannot produce
`pass`. `skipped_by_policy` is allowed only when the selected policy version does
not require that evidence.

Policy is versioned data. It maps risk classes to required Gate A and Gate B
evidence and deterministic rules. Missing-data behavior is `inconclusive`.
Thresholds and exceptions are not embedded as ad hoc workflow conditions.

An override records a unique ID, narrow scope, reason, human approver, approval
time, expiry, and optional ticket URI. It never contains credentials. Controllers
authenticate the approver, reject expired or unauthorized overrides, and include
accepted overrides in the gate result. An unsafe runtime rollout override makes
the rollout qualification-ineligible and cannot be overridden into a release
pass by this mechanism.

Qualification revocation is explicit, immutable evidence linked to the original
qualification. A revoked qualification cannot authorize promotion or publication.

## Consequences

The stable GitHub check will be named `release-policy` even when risk-specific
evidence varies. Workflow code remains a thin adapter around a tested policy
evaluator. Operational urgency stays visible to reviewers instead of silently
weakening the evidence chain.
