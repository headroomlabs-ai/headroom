# PR #405: generation-oriented review

This is a scoped source review that motivated the generation design. It is **not** a comprehensive approval, merge-readiness assessment, or independent verification of the contributor's reported tests. No review comment was posted and no PR was changed.

Reviewed PR head: `2851cbc964ac24d51e825361d23c857f671ffb8f` in `pacifio/headroom`.
Reviewed main: `c46e74d06b5ffa9643f420f28d2da6686fb7e487`.

## Finding 1: nullable wire data is weakened by a handwritten model

The server's `CompressionEntry.tool_name` is `str | None`; both retrieval handlers always emit the `tool_name` key. The Go PR's `RetrieveResult.ToolName` is a plain `string`. That representation loses the distinction between null and an empty tool name. It may appear harmless for current display uses, but it is incorrect fidelity for a reusable wire model and round-trip clients.

**Proposed contract:** required key with `string | null`, not an optional key. The generated Go model uses `*string`, while generated validation rejects an absent key. Tests cover null, empty string, a supplied name, and missing-field failure.

Sources:
- https://github.com/headroomlabs-ai/headroom/blob/c46e74d06b5ffa9643f420f28d2da6686fb7e487/headroom/cache/compression_store.py
- https://github.com/headroomlabs-ai/headroom/blob/c46e74d06b5ffa9643f420f28d2da6686fb7e487/headroom/proxy/server.py
- https://github.com/pacifio/headroom/blob/2851cbc964ac24d51e825361d23c857f671ffb8f/sdk/golang/models.go

## Finding 2: retired retrieval variants remain in the model

`RetrieveResult` still contains a `// Search variant` section with `Query`, `Results`, and `Count`. Current retrieval is hash-only and returns full original content. Extra optional fields alone need not break a client, but retaining a retired response variant is evidence of maintenance drift and can mislead consumers about supported behavior.

**Proposed action:** preserve any public compatibility aliases deliberately, mark retired fields deprecated where necessary, and use the generated seven-field wire model internally. Do not restore obsolete server behavior to satisfy an old SDK shape.

Sources: same server and Go model permalinks above; the PR description also states it was updated to the current hash-only contract.

## Finding 3: zero retries is indistinguishable from unspecified retries

`WithRetries(0)` assigns `ClientOptions.Retries = 0`. `newClientFromOptions` then replaces zero with `defaultRetries`, which is one. A caller cannot explicitly select zero retries through that option. This is a direct static consequence of those two functions, not a claim based on running the entire Go PR.

**Proposed action for the contributor:** represent presence independently, for example a pointer or an internal `retriesSet` field, reject negative counts, and test omitted/default, zero, and positive values. The SDK-generation pilot does not modify this contributor code; its new transport kernel simply defaults to no retries.

Sources:
- https://github.com/pacifio/headroom/blob/2851cbc964ac24d51e825361d23c857f671ffb8f/sdk/golang/options.go
- https://github.com/pacifio/headroom/blob/2851cbc964ac24d51e825361d23c857f671ffb8f/sdk/golang/client.go

## Architectural interpretation

Findings 1 and 2 are the kind of drift source-owned models prevent. Finding 3 illustrates the limit of generation: retry policy and option-presence semantics still require careful runtime design and tests. Generating models alone is not equivalent to generating a correct SDK.

Keep the useful adapters, hooks, SSE helpers, and ergonomic APIs in the contribution. Introduce generated internals behind that facade gradually. Provider payloads and streams need their own qualification; the current JSON-only retrieval pilot does not claim parity with the full Go PR.
