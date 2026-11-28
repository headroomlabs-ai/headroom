# Unified API Gateway Design

**Status:** Approved for implementation planning  
**Baseline:** `origin/main` at `94206e265203acfd72a3b939e9a964e29175ad50`  
**Source proposal:** `headroom-unified-proxy-proposal-2026-09-21.zip`  
**Scope:** Proposal core completion (GW00–GW10 and GW13; R01–R19)

## Purpose

Headroom will expose one authenticated gateway listener through which ordinary API
clients can use explicitly configured provider identities. The gateway will keep
the client's Headroom credential separate from upstream credentials, select only
routes granted to that client, and preserve native protocol behavior unless a
route advertises a qualified translation.

This extends the existing Python proxy. It does not create a second proxy, expose
an agent runtime as a transparent inference API, collect arbitrary subscription
credentials, or authorize an identity merely because a token was discovered.

## Scope and completion boundary

Core completion includes:

- A gateway operating profile and offline configuration validation.
- A caller grant model, credential broker, egress restrictions, and secret-safe
  error behavior.
- Native OpenAI Chat and Responses, Anthropic Messages, Gemini, Vertex, Bedrock,
  and configured OpenAI-compatible routes where their configured identity type is
  admitted.
- A capability-aware model catalog and explicit unsupported-capability errors.
- Qualified OpenAI, Anthropic, and Gemini protocol translations.
- Native SSE and Responses WebSocket behavior, stateful resource ownership, and
  identity affinity.
- Multiple configured accounts, cooldowns, safe retry classification, atomic
  budget/concurrency reservations, and truthful usage accounting.
- Local operator readiness, redacted status, validated reload, and owned shutdown.
- API-key, Google ADC, and AWS credential sources. Native account sources remain
  unavailable unless their provider admission dossier and required live evidence
  exist.
- The R01–R19 automated acceptance inventory, including negative behavior for
  unavailable conditional capabilities.
- Source and locally built artifact evidence. Live-provider qualification and
  release approval are recorded as external gates, never simulated locally.

GW11 auxiliary media/files/batches expansion and GW12 gateway optimization are
optional extensions. They are not enabled or advertised by this implementation.
Existing non-gateway optimization behavior remains unchanged.

## User interface

The new profile is explicit:

```console
headroom proxy --gateway --gateway-config ./gateway.json --check-config
headroom proxy --gateway --gateway-config ./gateway.json
```

`--check-config` performs schema and referential validation without reading secret
values, refreshing credentials, resolving DNS, or contacting an upstream.
`--gateway` requires `--gateway-config`. Gateway configuration rejects conflicting
compression, memory, learning, content-shaping, or runtime credential-mutation
options before the server starts.

The first configuration version follows the intent of the proposal's version-1
schema. The bundled draft accidentally requires `upstream_path_prefix` and
`ingress_protocols` while omitting their property definitions; the runtime schema
will define both fields so the supplied examples validate under
`additionalProperties: false`. It contains runtime, transform, privacy, client
principal, credential, route, and transport sections. Secrets are referenced
(`env:NAME`, ADC, or AWS chain), never embedded. Unknown keys fail validation so
misspelled security settings cannot be ignored.

Clients continue to use their SDK's normal credential field, but supply a
Headroom client token. The gateway never forwards that token upstream. The public
model list is filtered to routes granted to the authenticated principal.

## Architecture

The Python proxy remains the single enforcement authority. A new
`headroom.proxy.gateway` package contains focused, independently testable units:

- `config`: immutable configuration models, offline validation, references, and
  effective configuration redaction.
- `auth`: constant-time caller authentication, scope and route grants, conflicting
  credential-header detection, and browser/host guards.
- `credentials`: source interfaces, immutable leases, refresh generations,
  invalidation, and redacted availability status.
- `egress`: exact origin/path/audience enforcement, redirect denial, DNS/address
  checks for private endpoints, and final request signing hooks.
- `models`: principal-filtered aliases and capability declarations.
- `dispatch`: the native-or-translated decision and integration with existing
  byte-faithful forwarders.
- `protocols`: typed content/event representations and independently qualified
  translation directions.
- `resources`: principal/route/account ownership records for stateful HTTP and
  WebSocket operations.
- `routing`: account selection, affinity, cooldown state, and retry eligibility.
- `admission`: atomic concurrency and budget reservations finalized against known
  usage.
- `control`: readiness, redacted status, snapshot reload, revocation, and graceful
  shutdown behavior.

The request sequence is fixed:

1. Authenticate the caller and reject conflicting credential headers.
2. Authorize the ingress protocol, scope, route, model, and requested capability.
3. Acquire concurrency and budget reservations.
4. Select a configured eligible account while honoring state affinity and cooldown.
5. Acquire an opaque credential lease.
6. Construct the native or explicitly translated request.
7. Enforce the credential's final origin, port, path, project, and audience; then
   sign or attach the upstream credential.
8. Stream the response while recording exposure and cancellation state.
9. Finalize accounting and release leases/reservations.

No secret discovery or upstream operation occurs before steps 1 and 2 succeed.

## Pure-mode fidelity

Gateway version 1 is transform-off. Startup must not initialize compression
models, CCR, memory workers, learning, semantic cache, output shaping, effort
clamps, prompt rewriting, tool injection, or provider-hosted tool execution.

Strict-native routes preserve the successful request entity bytes and response
entity/event bytes. Routed-native routes may change only the declared model,
endpoint, or envelope fields and record a non-content mutation reason. Translated
routes preserve declared semantics and event ordering, not JSON byte layout.
Unsupported roles, content forms, signed state, tools, structured output, or event
semantics fail before upstream dispatch rather than being discarded.

Security-required removal of client credentials and transport-level provider
signing are explicit exceptions to entity-byte fidelity.

## Authentication and authorization

Every inference request, model listing, resource operation, and WebSocket
generation turn requires a configured principal. Authentication uses the normal
protocol credential headers. If multiple supported credential headers are
present, their values must agree; otherwise the request is rejected.

Principals have scopes and explicit route grants. Inference credentials cannot
invoke control operations. Resource reads, continuation, cancellation, and
deletion repeat authorization and check the local ownership binding before any
provider call. An absent binding is not permission to probe upstream accounts.

Loopback binding is not treated as authentication. Browser origins default deny,
query-string credentials are forbidden, and forwarded host/protocol data is
trusted only from configured proxy addresses. Version 1 binds to `127.0.0.1`, uses
one Python worker, and keeps remote/shared mode unavailable.

## Credential broker and provider admission

A credential source exposes metadata discovery, availability, lease acquisition,
refresh delegation, invalidation, and redacted status. A lease contains an opaque
secret handle plus provider, account, audience, expiry, and generation constraints.
Leases never mutate global environment variables.

Version 1 implements explicit environment references, Google ADC, and the AWS
credential chain. SDK-managed sources retain SDK refresh ownership. Refresh is
single-flight per account and publishes by generation so a late failure cannot
overwrite a newer credential.

Provider admission is separate from technical token parsing. OpenAI, Anthropic,
Gemini API, Vertex, Bedrock, and configured compatible endpoints admit only their
documented API/workload identity forms. Claude subscription brokerage is rejected.
Copilot, Codex/ChatGPT, Gemini CLI, Antigravity, Grok native, Muse Code, and Davin
remain unavailable until a provider-specific admission dossier and required live
qualification exist. Discovery is metadata-only and never imports or activates an
account.

OAuth browser/device machinery is implemented as reusable, closed-by-default
infrastructure for an admitted adapter: PKCE S256, one-time state, issuer/account
binding, exact redirect validation, callback replay prevention, bounded polling,
cancellation cleanup, and non-impersonation checks. With no admitted interactive
adapter, it exposes no login command that can mint a credential.

## Egress security

Each credential is bound to exact HTTPS origins and path prefixes. Provider,
project, region, and audience constraints are verified on the final URL after
route construction and before a credential is attached. Managed requests reject
per-request base URL overrides and redirects.

Configured private compatible endpoints have their own credentials and may not
receive public-provider or cloud identities. Runtime address validation rejects
metadata, link-local, loopback, and other unapproved destinations after DNS
resolution and on connection. Private CAs do not disable hostname verification.

Client tokens, upstream tokens, signed headers, query credentials, provider error
bodies, and private request content are removed from exceptions, logs, metrics,
traces, readiness, status, and effective-configuration output.

## Routing, retries, and accounting

Model aliases are unique within a principal-visible catalog. Routes declare ingress
and native protocols, translation capabilities, identity candidates, retry policy,
and billing constraints. Selection considers grants, capability, account scope,
residency, affinity, cooldown, concurrency, and budget before dispatch.

Stateful response IDs and WebSocket sessions stay bound to their creating
principal, route, and account. The gateway does not migrate them when an account is
unavailable.

The default is one attempt. A retry is allowed only for an explicitly classified,
pre-commit failure on a route whose provider contract permits it. Ambiguous
acceptance, any exposed output/tool event, or an accepted WebSocket operation is
never replayed. Provider `Retry-After` and quota scope drive bounded cooldowns; an
account is never rotated to evade a provider policy.

Concurrency and cost admission reserve capacity atomically before generation,
including each WebSocket `response.create` and fallback attempt. Reservations are
released or finalized on every terminal path. Unknown price/usage cannot satisfy a
strict currency cap. Provider limits and Headroom budget denials remain distinct.

## Streaming and lifecycle

Native streaming uses existing incremental forwarders. A complete upstream event
must reach the client before upstream completion. Framing has explicit finite
limits and applies backpressure; malformed or oversized frames fail without
unbounded buffering. Cancellation closes owned upstream work and releases account,
budget, and concurrency state.

Configuration reload builds and validates a complete immutable snapshot before an
atomic publication. Existing work retains its admitted snapshot unless immediate
revocation requires cancellation. Invalid reloads leave the prior snapshot active.

`GET /readyz` reports only local readiness with `service: "headroom"` and
`profile: "gateway"`; it does not resolve credentials or call providers. Shutdown
stops admission, drains or cancels within a bound, closes owned resources, and
never kills or reuses an unrelated listener process.

## Observability and privacy

Metrics use bounded route classes, provider, adapter, source kind, failure origin,
retry reason, and terminal result. Raw model names, account emails, tokens, session
IDs, prompts, tool results, query strings, and provider exception text never become
labels. Account references in authenticated status are local pseudonyms.

Pure mode disables beaconing, payload logging, persistence of content, and learning.
Accounting distinguishes logical requests from provider attempts and preserves
unknown values rather than inventing token-dollar conversions.

## Compatibility

Without `--gateway`, CLI defaults, `--no-optimize`, wrappers, SDK integrations,
provider routes, runtime environment behavior, and the Rust front proxy remain
unchanged. The Rust proxy is not permitted to terminate managed gateway routes
until it independently passes the same enforcement suite.

Configuration and credential state are additive. No native application's cache is
migrated or rewritten, so disabling the gateway or rolling back the artifact is
non-destructive.

## Testing and evidence

The proposal's T001–T100 scenarios are retained as stable IDs and implemented at
the lowest level that proves their behavior:

- Unit/property tests cover configuration, grants, leases, selectors, translation,
  frame parsing, retry classification, ownership, and reservation invariants.
- ASGI tests use the composed application and real middleware ordering.
- Process tests launch the actual CLI with isolated environment, fake identity
  services, and fake upstreams over HTTP and WebSocket.
- SDK tests use maintained OpenAI, Anthropic, and Gemini clients against the local
  process, including incremental streaming and error parsing.
- Security tests prove zero broker/upstream calls for failed authentication,
  authorization, capability, audience, and ownership checks.
- Concurrency tests cover refresh single-flight, stale publication, reload races,
  reservations, cancellation, cooldown, and shutdown cleanup.
- Strict-native byte oracles use independent literal fixtures including unknown
  fields, Unicode, numeric representation, tools, and signed content.
- Translation tests use independently reviewed directed fixture pairs and reject
  unmappable semantics.
- Packaging tests install the locally retained wheel into a clean environment and
  rerun the gateway smoke matrix against its digest.

Every implementation change follows red-green-refactor. A scenario is not counted
as covered merely because a test name exists: the test must exercise the real
boundary and must have been observed failing for the intended missing behavior.

Live account expiry, provider entitlement, operating-system secret-store behavior,
and official release-byte qualification cannot be established with local fake
services. Those cells remain disabled and explicitly marked `not_run` until a
maintainer supplies approved credentials, platform access, and release artifacts.
They do not block local completion of unavailable-capability rejection behavior.

## Requirement-to-evidence map

| Requirement | Implementation area | Required local evidence |
|---|---|---|
| R01–R02 | profile, config, pure dispatch | T001–T010; legacy regression and no-initialization probes |
| R03–R04, R13 | auth, grants, egress | T011–T020, T061–T065; zero-side-effect process checks |
| R05, R15 | credential sources | T021–T025, T071–T075; fake expiry and SDK-chain contracts |
| R06 | model registry | T026–T030; principal-filtered catalogs and collision rejection |
| R07 | protocol adapters | T031–T035; directed semantic fixtures and rejection cases |
| R08 | streaming | T036–T040; barrier streaming, bounds, cancellation |
| R09 | resources and WebSockets | T041–T045; per-turn auth and cross-principal negatives |
| R10–R11 | routing and admission | T046–T055; ambiguity, cooldown, atomic reservations |
| R12, R14 | control and operations | T056–T060, T066–T070; sentinel scans and lifecycle tests |
| R16–R17 | regressions and artifacts | T076–T085; full suite, SDK process tests, retained-wheel smoke |
| R18–R19 | provider admission/OAuth | T086–T100; unavailable-provider and hostile-flow fixtures |

## Delivery structure

Implementation stays on one draft feature branch with independently reviewable
commits aligned to GW01–GW10 and GW13. Each commit leaves legacy behavior green and
new capability disabled until its own tests pass. The pull request remains draft,
does not merge or release, and reports separately:

- locally executed evidence,
- skipped or unavailable conditional capabilities,
- external live-provider and final-release gates,
- exact source, wheel, fixture, and configuration digests.
