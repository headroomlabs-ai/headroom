# Release evidence contracts

These Draft 2020-12 JSON Schemas are the canonical, versioned wire contracts for
Headroom release evidence. Producers must emit `schema_version: 1`, validate the
complete document before upload, and publish the bytes immutably with a SHA-256
digest. Consumers must reject unknown schema versions, unresolved references,
unknown fields, identity mismatches, and missing required evidence.

`common.schema.json` owns shared identities and status values. The six public
documents cover candidate artifacts, deterministic policy, integration results,
cross-repository benchmark references, gate decisions, and the assembled
qualification manifest. A consumer should use the `$id` values as identifiers,
but should obtain schemas from a pinned Headroom commit or immutable release
artifact rather than fetching mutable network content at validation time.

JSON Schema establishes shape and local invariants. The qualification controller
must additionally enforce cross-document equality, evidence completeness, expiry,
authorization, reachability, and digest verification. In particular, schema-valid
benchmark arms are not enough: A1 and B identities must match in every field except
the intended optimization-enabled state.

Gate B may publish the exact qualified artifact or use an explicitly approved
runtime-payload equivalence method. In both cases the gate records qualified and
publication artifact and payload digests plus the verifier identity. The
controller, not JSON Schema alone, compares the applicable digest pair.

No contract permits a missing or invalid result to become `pass`. The shared
evidence status vocabulary is:

- `pass`
- `fail`
- `inconclusive`
- `skipped_by_policy`

`skipped_by_policy` is valid only when the selected versioned policy does not
require that evidence. It is never a substitute for missing required evidence.
