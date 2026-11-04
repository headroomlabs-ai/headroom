# ADR 0003: A1 versus B is the release benchmark claim

- Status: Accepted
- Date: 2026-08-13

## Context

Comparing Headroom against a direct provider path changes more than compression
and cannot isolate Headroom's effect. Release evidence needs a paired comparison
whose intended difference is only optimization state.

## Decision

The headline release comparison is:

- A1 (`a1_passthrough`): the Headroom proxy and candidate artifact with
  optimization disabled.
- B (`b_headroom`): the same Headroom proxy and candidate artifact with
  optimization enabled.

A1 and B must have equal candidate source SHA, artifact digest, rollout registry
and snapshot digests, model, agent, taskset, environment image, and resource
limits. Tasks are paired, arm order is randomized, and the versioned policy sets
repeat count and statistical decision rules. A direct-provider A0 arm may be
recorded for diagnosis but is not the headline comparator.

The benchmark producer reports an explicit identity-verification verdict. Any
identity mismatch makes the experiment and release evidence inconclusive. JSON
Schema constrains the arm shapes and intended optimization states; the benchmark
worker and qualification controller must enforce equality between arms.

## Consequences

A single noisy run is not hard release evidence. Missing telemetry, provider
outages, unequal environments, or identity drift remain visible and cannot be
coerced into `pass`. Accuracy, savings, latency, reliability, and missing-data
rules live in versioned policy rather than workflow shell code.
