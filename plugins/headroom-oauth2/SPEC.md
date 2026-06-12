# Spec — `headroom-oauth2` (generic OAuth2 client-credentials upstream auth)

Feature request: [chopratejas/headroom#778](https://github.com/chopratejas/headroom/issues/778).
Status: implementation ready; awaiting maintainer 👍 before merge.

## Summary

A vendor-neutral proxy extension (registers on Headroom's `headroom.proxy_extension` seam) that
mints an OAuth2 **client-credentials** (RFC 6749 §4.4) bearer from a configured token endpoint and
injects it as the upstream `Authorization` on every proxied request. Lets Headroom front any
gateway that requires a minted machine token (not a static API key) — with **zero core changes**
and **no vendor specifics** (the gateway is entirely config/env).

It complements `#510` (env-var auth), which assumes a long-lived static key; this covers the
"mint-then-refresh a short-lived token" case.

## API surface (config / CLI / env)

Opt-in only, via Headroom's existing flags — **no new CLI flags**:

    headroom proxy --backend litellm-openai --proxy-extension oauth2
    # or HEADROOM_PROXY_EXTENSIONS=oauth2

All configuration is env (12-factor; nothing baked in):

| Env var | Required | Meaning |
|---|---|---|
| `HEADROOM_OAUTH2_TOKEN_URL` | yes (else no-op) | OAuth2 token endpoint; must be `https` (loopback `http` allowed for tests) |
| `HEADROOM_OAUTH2_CLIENT_ID` / `_CLIENT_SECRET` | yes | client credentials |
| `HEADROOM_OAUTH2_SCOPES` | no | space/comma-separated scopes |
| `HEADROOM_OAUTH2_AUDIENCE` | no | `audience` form param |
| `HEADROOM_OAUTH2_RESOURCE` | no | RFC 8707 target `resource` form param |
| `HEADROOM_OAUTH2_GRANT_TYPE` | no | default `client_credentials` |
| `HEADROOM_OAUTH2_AUTH_STYLE` | no | `post` (form creds) or `basic` (HTTP Basic) |
| `HEADROOM_OAUTH2_HEADERS` | no | static upstream headers, `K=V,K2=V2` (control chars rejected) |
| `HEADROOM_OAUTH2_TIMEOUT` / `_SKEW` | no | token request timeout / pre-expiry refresh skew (s) |
| `HEADROOM_OAUTH2_ALLOW_INSECURE` | no | `1` to allow a non-loopback `http` token_url (discouraged) |

Public Python API: `OAuth2ClientCredentials`, `OAuth2Middleware`, `OAuth2Error`, `install`,
`provider_from_env`, `parse_headers`.

## Changes to existing behavior / defaults / compatibility

- **None unless explicitly enabled.** The entry point is dormant until `--proxy-extension oauth2`
  is passed, and even then a **no-op** unless `HEADROOM_OAUTH2_TOKEN_URL` is set.
- When active, it **overwrites the request `Authorization` header** with the minted bearer before
  the backend runs. The client's own `Authorization`/`x-api-key` is intentionally replaced (the
  proxy authenticates to the gateway on the client's behalf). *Compatibility note:* because the
  request then carries a bearer, Headroom classifies it as OAuth-mode auth — same as supplying a
  bearer yourself; no new classification path.
- No change to defaults, the request/response body, model routing, or compression.

## User stories (Given / When / Then)

- **Golden path** — *Given* a proxy started with `--proxy-extension oauth2` and valid
  `TOKEN_URL`/`CLIENT_ID`/`CLIENT_SECRET`, *When* a client sends `/v1/messages`, *Then* the
  extension mints (or reuses a cached) bearer and the upstream receives `Authorization: Bearer
  <minted>` plus any static headers; the client never sees the secret.
- **Edge: token endpoint down** — *Given* an unreachable/erroring `TOKEN_URL`, *When* a request
  arrives, *Then* the proxy returns `502 upstream_auth_error` (no upstream call, no secret/body
  leak) and stays up; the next request retries.
- **Edge: wrong backend** — *Given* `--backend bedrock` (env-auth), *When* the extension installs,
  *Then* it logs a loud warning that the injected bearer will have no effect and to use an
  OpenAI-compatible/passthrough backend.

## Failure modes & recovery

| Failure | Behavior |
|---|---|
| Missing/invalid config at startup | `install()` raises -> proxy **fails closed** (won't start mis-auth'd) |
| Token endpoint unreachable / non-2xx / non-JSON / no `access_token` | `OAuth2Error` -> `502`, per-request, proxy stays up, retried next request |
| `expires_in` = 0/negative/absent | clamped to a positive TTL (never stale, never per-request mint) |
| Concurrent first requests | single-flight lock -> exactly one mint per refresh |
| Malformed `HEADROOM_OAUTH2_HEADERS` (CR/LF) | offending pair dropped with a warning (no header injection) |

## Resilience (Docker / native / wrappers / providers / multi-process)

- **Native & Docker:** identical; pure env-driven, std-lib only. Token minted via `urllib` against
  the **system cert store**, so a corporate-injected CA is trusted with no bundled roots (works in
  SSL-inspection networks).
- **Wrappers (`headroom wrap`, agent hooks):** the extension lives at the proxy layer, so anything
  routed through the proxy inherits it transparently.
- **Providers:** effective for OpenAI-compatible / passthrough litellm backends (those that forward
  the request bearer upstream). `bedrock`/`vertex`/`sagemaker` authenticate from env and ignore the
  bearer -> the extension warns and is a no-op there.
- **Multi-process (multiple workers):** the token cache is per-process; each worker mints/refreshes
  independently. Acceptable for client-credentials (idempotent, low rate); no shared state, no
  cross-process lock needed. Documented so operators can size token-endpoint rate limits.

## Security & privacy

- Secrets are env-only; **never logged** and **never returned** to the client.
- The IdP error body is **drained, not surfaced** (may echo sensitive context).
- `token_url` is **https-enforced** (loopback exception for tests; explicit opt-out env).
- The minted bearer is sent only to the configured upstream; the client's inbound credential is
  replaced, not forwarded onward.

## Observability / logging / telemetry

- `INFO` on install (token_url + auth style, no secrets) and on each mint (`ttl`, scopes).
- `WARNING` on mint failure, env-auth-backend no-op, and dropped malformed static headers.
- No metrics/telemetry emitted; piggybacks on Headroom's existing request logging. (A future
  counter for mint/refresh/failure could be added if maintainers want it.)

## Rollback / migration

- **No migration** — additive and opt-in; existing deployments are unaffected.
- **Instant rollback:** drop `--proxy-extension oauth2` (or unset `HEADROOM_PROXY_EXTENSIONS`), or
  uninstall the package. No state to clean up, no config format changes.

## Dependencies

- **Runtime:** standard library only (no new core dependency). `litellm` is touched **only** if
  `HEADROOM_OAUTH2_HEADERS` is set, and it is already a Headroom backend dependency — declared here
  as the optional `[litellm]` extra, not a hard requirement.
