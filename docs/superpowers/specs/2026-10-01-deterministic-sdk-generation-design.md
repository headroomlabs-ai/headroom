# Deterministic Five-Language SDK Generation Design

## Intent

Headroom needs a functional proof of concept that derives HTTP SDK contracts from explicitly reviewed Python server declarations and deterministically generates usable clients for Python, TypeScript, Go, Rust, and .NET. The proof of concept demonstrates the architecture on the two CCR retrieval operations without claiming full API coverage, replacing existing public SDKs, publishing packages, or changing production handler behavior.

This design reconciles the proposal in `C:\Users\jd\Downloads\headroom-sdk-generation-design.md` with the pilot in `C:\Users\jd\Downloads\headroom-deterministic-sdk-generation.zip`. The bundle already implements Python, TypeScript, and Go. This revision qualifies Rust and .NET at the same pilot boundary and integrates the overlay with the current repository.

## Success Criteria

The proof of concept is complete when:

1. The real `POST /v1/retrieve` and `GET /v1/retrieve/{hash_key}` handlers carry no-op SDK metadata without changing their bodies, route decorators, guards, annotations, or runtime behavior.
2. An import-free AST compiler emits canonical OpenAPI 3.1.1 from explicit source configuration and rejects unsupported or ambiguous contracts.
3. A clean generation run produces deterministic Python, TypeScript, Go, Rust, and .NET source plus a complete ownership manifest.
4. Every generated language compiles or passes its native static validation in the pinned toolchain.
5. Shared loopback HTTP fixtures verify equivalent wire behavior across all five clients.
6. `check` detects missing, changed, and stale generated files without mutating the checkout.
7. CI runs generation drift, compiler tests, native compilation/static checks, and five-language wire conformance without secrets or publication permissions.
8. The repository contains an implementation plan, a resumable progress log, validation evidence, and draft-PR notes.

## Scope

### Included

- Reviewed `TypedDict` wire DTOs and `@sdk_operation` metadata for CCR retrieval.
- Required, optional, and nullable field distinctions.
- Scalar values, homogeneous literal enums, arrays, string-keyed maps, and explicit JSON values.
- JSON request/response operations with string path parameters.
- Canonical OpenAPI, deterministic language output, ownership checks, compatibility review, inventory reporting, and generation drift checking.
- Synchronous Python; `fetch`-based TypeScript with `AbortSignal`; context-aware Go; async Rust; async .NET.
- Finite timeout, no retries by default, no redirect following, bounded response reads, status/headers/raw-body errors, base-path support, path encoding, Unicode wire keys, unknown response-field preservation, and typed-shape validation.
- Experimental output under `sdk/generated-pilot` only.

### Excluded

- Full Headroom API coverage or claims that every public Python symbol is remotely callable.
- Existing SDK facade replacement, backwards-compatibility promises, package publication, or release automation.
- Pydantic/dataclass extraction, query/header parameter adapters, provider authentication, SSE, WebSockets, multipart, streaming, recursive models, arbitrary unions, or automatic typed exceptions.
- A complete JSON Schema inclusion solver or complete runtime route-graph proof.
- Java generation.

## Architecture

The server source is authoritative. Reviewed DTOs and no-op decorators identify the deliberately exported HTTP surface. The compiler parses configured Python files with `ast` and never imports the application, starts the proxy, reads credentials, or triggers background work. It builds one normalized in-memory representation and serializes it as canonical OpenAPI 3.1.1; OpenAPI remains the interchange contract rather than introducing a second checked-in schema language.

Five local emitters consume the normalized contract. Python, TypeScript, Go, Rust, and .NET generation use deterministic templates stored in the repository. Each emitter returns a mapping of relative paths to bytes; a shared ownership layer validates paths, refuses symlinks or unowned files, writes atomically, and records every managed artifact in `manifest.json`.

Transport behavior remains in narrow, reviewed runtime templates rather than being inferred from schema. The generated endpoint methods bind models and operation metadata to those runtimes. This keeps retry, redirect, cancellation, timeout, bounded-read, and error semantics visible and testable.

## Language Outputs

### Python

Generate dataclass-like wire models with explicit runtime validation and a synchronous standard-library HTTP client. Preserve arbitrary-size integers. The public pilot client exposes retrieval operations and structured HTTP errors.

### TypeScript

Generate interfaces/types, validators, and a `fetch` client. Reject integers outside JavaScript's safe integer range rather than silently rounding. Accept `AbortSignal` and preserve unknown response properties.

### Go

Generate structs, validators, and a context-aware `net/http` client. Use `int64`, pointers for required-nullable values, and an explicit optional representation where omission must be distinguished from null.

### Rust

Generate a Cargo library containing Serde models, validators, and an async `reqwest` transport. Required-nullable fields use `Option<T>` but remain required during deserialization; optional fields use a generated presence-aware wrapper when omission and null differ. The runtime disables redirects and retries, applies a finite timeout and bounded body read, accepts cancellation through future cancellation, and exposes status, headers, and raw error bytes. Dependencies and Rust edition are pinned in the toolchain/configuration records.

### .NET

Generate an SDK-style C# project with `System.Text.Json` models, validators, and an async `HttpClient` transport. Nullable reference types are enabled. Required properties use `required` members and explicit validation; optional-nullable properties use a generated presence-aware value type. The runtime uses `HttpCompletionOption.ResponseHeadersRead`, linked cancellation tokens, finite timeouts, bounded body reads, disabled automatic redirects, no retries, and structured HTTP errors. The target framework and SDK are pinned in the toolchain/configuration records.

## Contract and Wire Semantics

JSON property names never undergo recursive case conversion. Language identifiers may be idiomatic, but serializers bind the unchanged wire names. Generated decoders retain unknown response properties so additive server fields survive round trips.

The model system distinguishes:

- required and non-nullable;
- required and nullable;
- optional and non-nullable;
- optional and nullable;
- present false or zero values; and
- omitted values.

The compiler rejects unsupported syntax rather than weakening it to `Any`, `object`, `dynamic`, or untyped JSON. Language capability differences are documented and tested. In particular, TypeScript safe-integer rejection and Go/.NET bounded integers are explicit rather than presented as perfect numeric equivalence.

## Determinism and Ownership

Canonical JSON uses UTF-8, sorted object keys, stable semantic ordering, two-space indentation, a final line feed, and no non-finite numbers. Order-sensitive arrays are not blindly sorted. Source paths, line numbers, timestamps, environment values, temporary directories, and Git SHAs do not affect generated content.

`manifest.json` records the OpenAPI digest, generator/runtime input digests, and every generated file digest, excluding the manifest's own digest. Provenance and toolchain versions live outside semantic schema inputs. Generation refuses a nonempty unowned destination, unexpected files, missing ownership metadata, path traversal, and symlink content. Drift checking regenerates in memory and compares the exact expected file set and bytes.

Determinism tests vary working directory, `PYTHONHASHSEED`, timezone, source comments, and irrelevant source order. The claim is limited to the tested source-generation profile; it is not a claim of hermetic release archives across arbitrary operating systems and compilers.

## Error Handling and Safety

Compiler diagnostics identify the source declaration and unsupported construct without importing application code. Duplicate operation IDs, duplicate model names, unsafe names, unsupported route parameters, ambiguous unions, streaming responses, conflicting metadata, and invalid integer literals fail generation.

Clients accept only `http` and `https` base URLs, correctly join base paths, percent-encode path parameters, reject redirects, use finite timeouts, bound successful and error response reads, and never retry automatically. Error messages omit response payloads while error objects retain raw bytes for callers that intentionally inspect them.

Server-enforced loopback and same-origin protections remain unchanged. The `x-headroom-access` extension documents qualification requirements; it neither implements authentication nor bypasses server authorization.

## Testing and Acceptance

The test pyramid has four independent gates:

1. Compiler/unit tests cover AST discovery, fail-closed type adaptation, canonicalization, ownership, drift, compatibility reporting, and deterministic output.
2. Native language checks compile or statically validate every generated SDK using pinned Python, Node/TypeScript, Go, Rust, and .NET versions.
3. A shared loopback mock server exercises successful retrieval, required-nullable fields, null/empty/named tool values, missing fields, unknown fields, base paths, encoded paths, Unicode, non-JSON success, HTTP errors, redirects, oversized bodies, timeouts, and cancellation where supported.
4. Repository integration checks exercise the actual handler source and confirm the metadata change does not alter FastAPI routing, loopback/same-origin guards, or existing regression suites.

CI is additive, uses read-only contents permission, receives no secrets, performs no publication, and does not disable existing gates.

## Delivery and Documentation

Work lands in small commits: reviewed baseline/notes, reconciled three-language pilot, Rust generation, .NET generation, five-language conformance, and CI/documentation qualification. `docs/engineering/deterministic-sdk-generation-progress.md` records completed tasks, exact commands/results, decisions, and remaining risks after every checkpoint. Draft-PR notes clearly label the output experimental and list any unexecuted platform or repository gates.

Because this checkout is a detached linked worktree, commits are valid checkpoints but cannot be pushed directly as a named branch until the work is attached to a feature branch. No merge or package publication is part of this task.

## Selected Approach

Use native, dependency-conscious Rust and .NET emitters alongside the existing native Python, TypeScript, and Go emitters. This keeps generation offline, deterministic, and small enough to audit for the pilot. A future production system may qualify OpenAPI Generator or another backend behind the same canonical contract boundary, but introducing that toolchain is not required to prove five-language generation here.
