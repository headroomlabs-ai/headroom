# Release evidence contracts

This evidence/tooling slice includes the policy contracts and the following
evidence schemas. Inspect the inventory at the pinned commit before selecting a
schema; the earlier policy-only slice supplies only the common and policy files.

| Schema | Contract |
| --- | --- |
| `common.schema.json` | Shared identities and status values |
| `policy.schema.json` | Versioned Gate A and Gate B requirements |
| `candidate-manifest.schema.json` | Immutable candidate bytes and provenance |
| `integration-result.schema.json` | Deterministic and integration check results |
| `benchmark-result-ref.schema.json` | Cross-repository benchmark evidence reference |
| `gate-result.schema.json` | Gate decisions and evidence references |
| `qualification-manifest.schema.json` | Assembled artifact qualification |

Producers must emit the supported schema version, validate the complete document
before upload, and publish bytes immutably with a SHA-256 digest. Consumers must
reject unknown schema versions, unresolved references, unknown fields, identity
mismatches, and missing required evidence. Use `$id` values as identifiers, but
obtain schemas from a pinned Headroom commit or immutable release artifact rather
than fetching mutable network content at validation time.

## Version 1 policy rules

Each gate requires at least one evidence identifier and at least one rule.
The v1 rule vocabulary is closed:

| Kind | Parameters | Meaning |
| --- | --- | --- |
| `required_status` | Exactly `{"accepted_status":"pass"}` | Required evidence must pass; failed, inconclusive, or policy-skipped results cannot qualify required evidence. |
| `not_revoked` | Exactly `{}` | The qualification must not have been revoked. |

Unknown rule kinds, misspelled or extra parameters, and empty requirement arrays
fail schema validation. Additional rule kinds require an explicit pinned-schema
extension, matching controller support, and validation tests.

Schema validation does not implement rule evaluation. A controller must resolve
evidence identifiers, authenticate evidence and revocations, and apply the rule
semantics before accepting a policy decision. It must also enforce cross-document
identity, evidence completeness, expiry, authorization, reachability, and digest
verification. Schema-valid benchmark arms alone do not prove equivalent inputs:
A1 and B identities must match except for the intended optimization-enabled state.
Gate B may publish the exact qualified artifact or use an explicitly approved
runtime-payload equivalence method; the controller compares the applicable digest
pair and records the verifier identity.

The shared evidence status vocabulary is `pass`, `fail`, `inconclusive`, and
`skipped_by_policy`. Policy skipping is valid only when the selected versioned
policy does not require that evidence. It never substitutes for missing required
evidence. No contract permits a missing or invalid required result to become
`pass`.
