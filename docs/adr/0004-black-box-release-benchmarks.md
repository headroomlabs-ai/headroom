# ADR 0004: Black-box release benchmark boundary

- Status: Accepted
- Date: 2026-08-13

## Context

Internal unit and compression tests are necessary but do not prove that the
packaged proxy preserves real agent, streaming, and tool-use behavior. Conversely,
benchmark infrastructure must not gain release credentials or execute arbitrary
release commands.

## Decision

Release benchmarks exercise Headroom through its supported public protocol and
observable provenance surfaces, using the immutable candidate artifact. They
record taskset revision and digest, model and agent identities, environment,
resource limits, rollout snapshot, artifact digest, attempts, telemetry
completeness, and raw-result identity.

The benchmark repository receives only the minimum candidate and experiment
identities needed to run. It verifies artifact digests before installation and
returns an immutable result reference with authenticated producer identity. It
does not receive provider secrets through release evidence, package-publish
credentials, or permission to mutate the Headroom repository.

Failures and cancellations preserve terminal evidence. A provider or
infrastructure failure is `inconclusive`, not a candidate pass and not necessarily
a candidate-caused failure. The qualification controller applies the registered
policy to the returned reference; callback identity alone does not establish a
passing result.

## Consequences

Hermetic PR simulators, scheduled live smoke tests, and release-candidate lanes
can share one result shape while retaining their execution depth. Cross-repository
authentication, artifact transport, callback authorization, and retention need
separate implementation and security review in later PRs.
