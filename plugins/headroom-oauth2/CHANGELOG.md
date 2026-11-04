# Changelog

## 0.1.1

- Inject the upstream bearer only on requests that go upstream. Local routes
  (`/health`, `/stats*`, `/metrics`, `/v1/compress*`, `/ext/*`, …; see README)
  are passed through untouched, so an unreachable IdP no longer 502s health
  probes and management calls no longer mint tokens. `/p/<project>/` prefixes
  are stripped first.
- Startup log carries `scheme://host` of the token URL and the mint log a scope
  count, instead of the full URL and scope list.
- Requires `headroom-ai` ≥ 0.40 (extension middleware inside the proxy-token gate).

## 0.1.0

Initial release — generic OAuth2 client-credentials upstream-auth extension for the Headroom proxy.

- Mints an OAuth2 client-credentials (RFC 6749 §4.4) bearer from a configurable token endpoint and
  injects it as the upstream `Authorization` on each proxied request, via Headroom's opt-in
  `headroom.proxy_extension` seam (`--proxy-extension oauth2`). No core changes; vendor-neutral.
- `post` and `basic` client-auth styles; scopes, `audience`, RFC 8707 `resource`, static upstream
  headers, configurable timeout/skew — all env-driven.
- Token caching with single-flight refresh and pre-expiry skew; `expires_in` clamped to a positive
  TTL.
- Fails closed on misconfiguration; returns `502 upstream_auth_error` on mint failure without
  leaking the IdP error body. `token_url` https-enforced (loopback exempt). Std-lib only (system
  cert store -> works behind corporate SSL inspection).
