# Deterministic SDK Generation Progress

## Working State

- Approved design: `docs/superpowers/specs/2026-10-01-deterministic-sdk-generation-design.md`
- Approved plan: `docs/superpowers/plans/2026-10-01-deterministic-sdk-generation.md`
- Checkout: detached linked worktree; commits are checkpoints until attached to a feature branch.
- Source bundle: `C:\Users\jd\Downloads\headroom-deterministic-sdk-generation.zip`

## 2026-10-01 — Task 1: Three-Language Baseline

- Installer dry-run correctly refused the detached checkout. The safety check was retained; the checksum-reviewed new-file overlay was installed explicitly and server metadata was reconciled by hand.
- Added import-free AST compiler, deterministic emitter, CLI, conformance runner, Python/TypeScript/Go runtime templates, codegen configuration, tests, and engineering notes.
- Added no-op `sdk_operation` metadata to the real CCR POST and GET retrieval handlers. Existing route decorators, guards, signatures, handler bodies, and response behavior remain unchanged.
- RED: `SourceCompilerTests.test_real_pilot_paths_and_types` failed with `No annotated operations found` before server metadata was added.
- GREEN: the same production-source test passed after metadata integration.
- Windows qualification found the test AST loader relied on the platform default encoding. Root cause was UTF-8 source containing non-CP1252 bytes; the loader now requests UTF-8 explicitly.
- Validation: `python -m unittest discover -s tests/sdkgen -p 'test_*.py' -v` — 37 tests passed; generated fixture snapshot check reported 20 byte-for-byte files.

## Remaining

1. Harden ownership/determinism and generate the current snapshot.
2. Isolate emitter modules.
3. Add and qualify Rust generation.
4. Add and qualify .NET generation.
5. Enforce five-language behavioral conformance.
6. Add CI, repository qualification, and draft-PR notes.

## 2026-10-01 — Task 2: Ownership and Determinism

- Added explicit generated-path validation for absolute paths, parent traversal, Windows drive syntax, backslash aliases, and non-byte emitter output.
- Generation now stages every output before removing stale managed files. A forced staging failure proved an existing stale managed file remains intact.
- Added checks that drift detection is read-only, symlinked output is rejected, and separate output roots are byte-identical.
- RED: unsafe paths failed with a missing validation interface, and a forced temporary-file staging error removed `stale.txt` before the fix.
- GREEN: `OwnershipTests` and `DeterminismTests` passed all 5 cases.
- Generated the real current-source snapshot under `sdk/generated-pilot`.
- Validation: source `generate` emitted 20 files; source `check` matched all 20 byte-for-byte; the complete SDK generator suite passed 42 tests.

## 2026-10-01 — Task 3: Emitter Boundary

- Added a typed emitter registry and independent Python, TypeScript, and Go adapters accepting the canonical document and returning fully qualified output paths.
- Added collision-checked emitter composition; duplicate paths cannot silently overwrite another backend's artifact.
- RED: `EmitterCompositionTests` failed because `render_with_emitters` did not exist.
- GREEN: the collision test passed and the full suite reached 43 passing tests.
- Adding emitter modules intentionally changed only `manifest.json` because generator input provenance includes Python source files. Regeneration left all language artifact bytes unchanged, and source drift checking matched all 20 files.
- The proven legacy emission bodies remain in `emit.py` behind the new adapters for this pilot; Rust and .NET use the stable adapter interface directly.

## 2026-10-01 — Task 4: Rust Generation

- Added deterministic Rust 2021 generation with Serde wire models, manual presence-aware deserialization, unknown-field preservation, async Reqwest client bindings, pinned direct dependencies, and a generated Cargo lockfile.
- The transport rejects non-HTTP base URLs and credentials/query/fragment components, disables redirects, applies a finite timeout, bounds streamed response bodies, performs no automatic retries, percent-encodes path segments, and retains status/headers/raw error bytes without including response bodies in display text.
- Required-nullable fields remain `Option<T>` to callers while custom deserialization rejects omission. Generated unit coverage accepts explicit null and rejects a missing `tool_name`.
- RED: Rust output tests initially failed because no Rust artifacts existed. Cargo then exposed the repository workspace-boundary requirement, and Clippy exposed an oversized error enum variant.
- GREEN: generated model/runtime tests passed; Cargo compiled the async client with the committed lockfile; Clippy passed with warnings denied after boxing the API-error variant.
- Rust build output is directed outside the managed generated tree so ownership checking remains exact. Generated Rust is compiler-owned and not rewritten by rustfmt; Cargo compilation and Clippy are the native static gates.
- Added the Rust shared-wire test source for Task 6 loopback execution.

## 2026-10-01 — Task 5: .NET Generation

- Added deterministic .NET 10/C# generation with nullable reference types, required members, JSON wire-name attributes, `JsonExtensionData`, and a presence-aware `Optional<T>` converter for omitted versus present optional properties.
- Added async `HttpClient` bindings with cancellation tokens, linked finite timeouts, disabled redirects, `ResponseHeadersRead`, streamed bounded reads, path escaping, and structured status/header/raw-body exceptions whose display text omits the payload.
- RED: .NET output tests initially failed because no .NET artifacts existed.
- GREEN: generated-output tests passed; `dotnet build` completed with zero warnings/errors; the executable model test accepted explicit-null `tool_name`, rejected omission, and preserved an unknown nested field.
- `dotnet format --verify-no-changes` formatted 0 of 6 files. Its local `bin`/`obj` build artifacts were removed after an exact `git clean -ndx` preview; all later .NET commands use an external artifacts directory to keep generated ownership exact.
- Source drift checking now matches 30 generated files byte-for-byte.

## 2026-10-01 — Task 6: Five-Language Wire Conformance

- Extended the shared loopback harness to require Node/TypeScript, Go, Cargo/Rust, and .NET and to execute all five generated clients against the same HTTP fixture.
- Added a cross-platform tool launcher so Windows `.cmd` shims such as `tsc.cmd` run without a shell-dependent failure.
- Rust and .NET now exercise successful POST/GET exchange, required-nullable fields, Unicode and reserved path characters, unknown response fields, retained raw HTTP errors, malformed/non-JSON response rejection, wide integers, redirect blocking, bounded reads, invalid base URLs, dot-segment rejection, and timeout/cancellation.
- Added a delayed fixture response so cancellation/timeout behavior is exercised rather than merely compiled. Client disconnects are treated as expected fixture behavior without noisy server tracebacks.
- RED: the original harness stopped at Python because Windows could locate but not execute the `tsc.cmd` shim directly. The first timeout run also exposed an uncaught Windows `ConnectionAbortedError` in the fixture server.
- GREEN: the conformance command completed successfully for Python, TypeScript, Go, Rust, and .NET; it observed 47 loopback requests, followed no redirect, and made no provider calls.
- The exact TypeScript 5.8.3 compiler was installed in a temporary tool directory for local qualification; no repository dependency or generated artifact was changed by that installation.

## 2026-10-01 — Task 7: Delivery Qualification

- Added an additive GitHub Actions workflow with read-only contents permission, path filtering, concurrency cancellation, a 20-minute timeout, exact action commits, and pinned Python, Node/TypeScript, Go, Rust, .NET, Ruff, and mypy versions.
- Updated the architecture and codegen documentation for five generated languages and recorded both CI-target and locally tested tool versions.
- Added a draft-PR body that states the experimental two-operation scope, safety semantics, evidence, limitations, and absence of publication or facade replacement.
- Repository static validation: Ruff format check clean across 15 handwritten files; Ruff lint clean; mypy clean across 12 generator source files; `git diff --check` clean.
- Generation validation: 30 managed files matched byte-for-byte after formatting/regeneration; 47 compiler/ownership/handler tests passed.
- Native validation: five-language conformance passed over 47 loopback requests; Rust Clippy passed with warnings denied; .NET Release build produced zero warnings/errors; Go vet passed.
- Focused production integration: all 5 retrieval-history repair tests passed. The wider two-file CCR run passed 19 tests and failed only `TestCCREdgeCases.test_ccr_disabled_no_caching` because the temporary Python 3.12 environment could not load the prebuilt `headroom._core` native DLL; that test does not exercise SDK generation or retrieval metadata.
- Ruff initially found 11 inherited bundle files needing formatting and two loop-variable lambda bindings; both were corrected before regeneration. Mypy initially found 18 type issues and finished clean after explicit narrowing/annotations.

## 2026-10-02 — Final Review Fix Pass

- Closed all nine Important whole-branch review findings with focused RED→GREEN regressions.
- Ownership now trusts only the prior manifest and rejects output-root symlink/junction traversal plus unsafe Windows path components.
- Rust/.NET preserve optional presence and enforce nested non-nullable values. They reject enum, typed-map, and explicit JSON-value fields before output rather than silently weakening them.
- Reqwest retries are explicitly disabled; the default Go transport prevents reused-connection replay. Five forced disconnect probes were each observed exactly once.
- Rust protocol diagnostics are response-value independent. Rust/.NET enforce JSON success media types.
- Conformance separates timeout from caller cancellation, validates wrong media independently from JSON syntax, exercises nested presence/null states in Go/Rust/.NET, and asserts attempt counts.
- Final evidence: 52 SDK tests passed; 30 generated files matched exactly; five-language conformance passed over 70 requests; Rust tests and Clippy passed with warnings denied; .NET Release build had zero warnings/errors; Go test/vet passed; 39 focused CCR integration tests passed; full-range whitespace check clean.

## Draft PR Readiness

- Code, generated artifacts, CI, design, plan, progress log, and PR notes are present in the detached worktree.
- Remaining external steps: attach commits to a named branch, push the intended remote, open the draft PR, and observe hosted CI.
- No merge or package publication has been performed.
