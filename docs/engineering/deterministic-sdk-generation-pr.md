# Draft PR: Deterministic Five-Language SDK Generation Pilot

## Summary

This draft introduces an experimental, deterministic source-to-OpenAPI-to-client pipeline for two CCR retrieval operations:

- `POST /v1/retrieve`
- `GET /v1/retrieve/{hash_key}`

Reviewed wire DTOs and no-op metadata on the existing handlers are compiled without importing or starting Headroom. The canonical OpenAPI 3.1.1 contract generates Python, TypeScript, Go, Rust, and .NET clients under `sdk/generated-pilot`.

This does not replace an existing public SDK, publish packages, or claim full Headroom API coverage.

## What Is Included

- Import-free AST discovery of explicitly exported operations.
- Fail-closed type adaptation for the pilot's reviewed `TypedDict` profile.
- Deterministic OpenAPI and generated artifacts with exact ownership/hashing.
- Generated models, operation bindings, and bounded HTTP transports for five languages.
- Required/optional/nullability fidelity, unchanged JSON wire names, and unknown response-field preservation.
- Finite timeouts, no automatic retries, disabled redirects, bounded body reads, and structured raw-body HTTP errors.
- Conservative contract diff and route inventory commands.
- Shared loopback HTTP conformance across all five generated clients.
- An additive, read-only GitHub Actions workflow with pinned action commits and tool versions.

## Server Integration

The production change is metadata-only. It adds the `sdk_contracts` import and `@sdk_operation` markers to the existing CCR retrieval handlers. The decorators return the original function object and do not change:

- handler bodies or signatures;
- FastAPI route decorators or response-model behavior;
- loopback and same-origin guards;
- provider, compression, cache, or proxy runtime behavior.

## Validation

Local Windows validation completed with:

- `python -m tools.sdkgen check` — 30 generated files matched byte-for-byte.
- `python -m unittest discover -s tests/sdkgen -p 'test_*.py' -v` — 52 passed.
- `python -m tools.sdkgen.conformance` — five languages passed over 70 loopback HTTP requests; forced disconnects were observed once per language, no providers were called, and no redirect was followed.
- `ruff format --check` and `ruff check` on handwritten SDK generator/source/tests — clean.
- `mypy tools/sdkgen` — clean across 12 source files.
- Rust `cargo test --locked` and Clippy with warnings denied — clean.
- .NET Release build — zero warnings and zero errors.
- Go vet — clean.
- Focused repository tests — 39 passed, including all five retrieval-history repair tests.

Exact local and CI-target tool versions are recorded in `sdk/codegen/toolchain.lock.json`. CI re-runs generation drift, compiler tests, native checks, and five-language conformance on Ubuntu.

## Deliberate Limits

- Only the two CCR retrieval operations are generated.
- No Pydantic/dataclass extraction, query/header adapters, authentication profiles, multipart, SSE, or WebSocket generation.
- No full runtime-route-graph coverage claim.
- No existing SDK facade migration or compatibility promise.
- Rust and .NET fail closed before rendering enum, typed-map, or explicit JSON-value fields; those backends must implement the constructs before the pilot contract expands to use them.
- No package publication, release archive, or supply-chain hermeticity claim.
- Rust and .NET direct dependencies are pinned and Rust has a committed lockfile; the broader build is not fully mirrored or offline-hermetic.
- Full proxy, supported-platform, and hosted-CI qualification remain merge gates for this draft.

## Review Focus

- Required-nullable `tool_name` behavior in every language.
- Redirect, timeout/cancellation, and bounded-read behavior.
- Deterministic ownership and stale/unowned file handling.
- Metadata-only server integration.
- Whether the experimental output location and future backend boundary are appropriate before wider API migration.
