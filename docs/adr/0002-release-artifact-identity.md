# ADR 0002: Release artifact identity and provenance

- Status: Accepted
- Date: 2026-08-13

## Context

Rebuilding a moving source reference can produce different bytes. Source identity
alone cannot prove what qualification tested or what publication shipped.

## Decision

Every candidate is bound to an exact full Git commit SHA and an immutable artifact
identified by SHA-256, filename, package name, version, size, and build-run
identity. When runtime behavior comes from a payload that can differ from package
bytes, its SHA-256 is recorded separately.

Evidence documents are themselves immutable bytes with a SHA-256 and an opaque
URI. A producer identity records repository, workflow, run ID, attempt, and the
producer commit. Cross-repository consumers verify the digest before processing
the artifact or evidence. Secrets and credentials are never evidence fields.

The candidate artifact is built once. Downstream deterministic, integration, and
benchmark lanes consume those exact bytes; they do not rebuild `main` and reuse
the candidate identity. Final Gate B compares published bytes with the qualified
artifact or applies an explicitly approved, policy-versioned runtime-payload
equivalence rule.

## Consequences

Artifact stores and workflow runs require immutable retention sufficient for
audit. A digest, source SHA, schema version, or producer mismatch makes evidence
`inconclusive` or `fail` according to policy; it can never pass silently. Future
controllers must enforce equality across documents in addition to validating
each document's JSON Schema.
