/**
 * Per-provider static session-header injection for OpenClaw provider configs.
 *
 * Use case: some third-party OpenAI-compatible providers route requests
 * through a server-side plan/accounting layer keyed by a caller-supplied
 * session identifier. The provider returns an error such as
 * `MissingSessionID` when that header is absent, but does not advertise
 * the requirement in its OpenAI-shape surface — the docs are linked only
 * from the error body. The OpenClaw provider plugin for such a provider
 * may not generate the header itself, so the headroom plugin offers a
 * generic mechanism for the operator to declare
 *
 *     providerSessionHeaders:
 *       "<providerId>": "<Header-Name>"
 *
 * and the headroom plugin generates a per-session UUID and injects it
 * into the provider's `headers` map at in-memory rewrite time. Each entry
 * is a one-header, one-UUID mapping by design: callers that need multiple
 * headers per provider can layer their own provider plugin; the headroom
 * plugin keeps this surface minimal and predictable.
 *
 * The header value is generated once per gateway process and reused for
 * every request to that provider for the lifetime of the process. This is
 * intentional: providers treat the value as a coarse identifier that
 * maps a sequence of related requests to the same plan/accounting
 * bucket, and stable values across runs of a single agent session are
 * what providers expect.
 */

export type SessionHeaderMap = Readonly<Record<string, string>>;

/**
 * Read `providerSessionHeaders` from the headroom plugin's config.
 *
 * Returns an empty object (not undefined) when the field is absent or
 * contains no usable entries, so callers can treat the result as a plain
 * map without nullability checks.
 */
export function readProviderSessionHeaders(apiConfig: any): SessionHeaderMap {
  const entries = apiConfig?.plugins?.entries;
  if (!entries || typeof entries !== "object") return {};
  const headroom = entries.headroom;
  if (!headroom || typeof headroom !== "object") return {};
  const configured = headroom.config?.providerSessionHeaders;
  if (!configured || typeof configured !== "object") return {};
  const out: Record<string, string> = {};
  for (const [providerId, headerName] of Object.entries(configured)) {
    if (
      typeof headerName === "string" &&
      headerName.trim().length > 0 &&
      typeof providerId === "string" &&
      providerId.length > 0
    ) {
      out[providerId] = headerName.trim();
    }
  }
  return out;
}

/**
 * Lazily build a per-provider session header value. Uses
 * `crypto.randomUUID` where available (Node 19+, modern browsers) and
 * falls back to a 128-bit hex string assembled from `Math.random` for
 * older runtimes; both forms satisfy the format the third-party providers
 * we have encountered accept.
 *
 * The result is cached on first generation so every request within the
 * lifetime of the gateway process carries the same session id for a
 * given provider — see the module-level docstring for the rationale.
 */
const sessionIdCache = new Map<string, string>();

export function ensureSessionId(providerId: string): string {
  const cached = sessionIdCache.get(providerId);
  if (cached !== undefined) return cached;
  let fresh: string;
  const cryptoRef: any = (globalThis as any).crypto;
  if (cryptoRef && typeof cryptoRef.randomUUID === "function") {
    fresh = cryptoRef.randomUUID();
  } else {
    // 128-bit hex from two Math.random draws; sufficient entropy for
    // session-style routing identifiers (not a security primitive).
    const part = (): string =>
      Math.floor(Math.random() * 0xffffffff)
        .toString(16)
        .padStart(8, "0");
    fresh = `${part()}${part()}${part()}${part()}`;
  }
  sessionIdCache.set(providerId, fresh);
  return fresh;
}

/**
 * Reset the session-id cache. Intended for test isolation only.
 *
 * The production process generates one session id per provider and
 * keeps it for the lifetime of the process; the cache is intentionally
 * a module-level singleton. Tests that exercise multiple providers in
 * a single vitest run can call this between cases to keep deterministic
 * assertions about "the session id is a UUID".
 */
export function __resetSessionIdCacheForTests(): void {
  sessionIdCache.clear();
}
