/* eslint-disable @typescript-eslint/no-explicit-any */

import { resolveProxyPathPrefix } from "./proxy-routing.js";
import { ensureSessionId } from "./session-headers.js";

/**
 * Gateway-side rewriting for OpenClaw model provider base URLs.
 *
 * When a plugin routes a request to the local headroom proxy, two things
 * must happen on the in-memory provider config:
 *
 *   1. `baseUrl` must point at the proxy (host + port) so subsequent
 *      OpenClaw HTTP calls land at the proxy rather than the upstream.
 *   2. The proxy must know which upstream to forward to. The proxy's
 *      documented mechanism for per-request upstream routing is the
 *      `x-headroom-base-url` header (see
 *      https://docs.headroomlabs.ai/docs/configuration#proxy-upstream-override-x-headroom-base-url).
 *
 * Item (2) lets one proxy instance serve multiple OpenAI-compatible
 * upstreams without needing extra processes or extra config slots.
 */

export const DEFAULT_GATEWAY_PROVIDER_IDS = ["openai-codex"] as const;

const DEFAULT_PROVIDER_BASE_URLS: Readonly<Record<string, string>> = {
  "openai-codex": "https://chatgpt.com/backend-api",
};

const GATEWAY_PROVIDER_ID_ALIASES: Readonly<Record<string, string>> = {
  codex: "openai-codex",
  claude: "anthropic",
  copilot: "github-copilot",
  gemini: "google",
};

const EXPLICIT_BASE_URL_REQUIRED_PROVIDER_IDS = new Set<string>(["github-copilot"]);

export function resolveGatewayProviderIds(config: Record<string, unknown> | undefined): string[] {
  const configuredProviderIds = normalizeGatewayProviderIds(config?.gatewayProviderIds);
  if (configuredProviderIds.length > 0) {
    return configuredProviderIds;
  }

  if (config?.routeCodexViaProxy === false) {
    return [];
  }

  return [...DEFAULT_GATEWAY_PROVIDER_IDS];
}

function normalizeGatewayProviderIds(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const seen = new Set<string>();
  const normalized: string[] = [];

  for (const entry of value) {
    if (typeof entry !== "string") {
      continue;
    }

    const rawProviderId = entry.trim();
    const providerId = GATEWAY_PROVIDER_ID_ALIASES[rawProviderId.toLowerCase()] ?? rawProviderId;
    if (!providerId || seen.has(providerId)) {
      continue;
    }

    seen.add(providerId);
    normalized.push(providerId);
  }

  return normalized;
}

/**
 * Optional per-provider routing overrides.
 *
 * `providerUpstreams[providerId]` is the absolute URL of the upstream the
 * proxy should forward to when the rewritten provider is contacted. URLs
 * must not include a trailing `/v1`; the proxy appends the request path
 * (`/v1/messages`, `/v1/chat/completions`, ...) directly to whatever the
 * override URL is.
 *
 * `providerSessionHeaders[providerId]` is the name of a header the plugin
 * should add to that provider's `headers` map, populated with a stable
 * per-process session id. See `../session-headers.ts` for the rationale
 * and the id-generation policy.
 */
export interface GatewayRoutingOverrides {
  providerUpstreams?: Readonly<Record<string, string>>;
  providerSessionHeaders?: Readonly<Record<string, string>>;
}

export function applyGatewayProviderBaseUrls<T>(
  cfg: T,
  proxyUrl: string,
  providerIds: readonly string[],
  overrides?: GatewayRoutingOverrides,
): { changed: boolean; config: T } {
  const next = structuredClone((cfg ?? {}) as any);
  const changed = applyGatewayProviderBaseUrlsInPlace(
    next,
    proxyUrl,
    providerIds,
    overrides,
  );
  return { changed, config: next as T };
}

export function applyGatewayProviderBaseUrlsInPlace(
  cfg: any,
  proxyUrl: string,
  providerIds: readonly string[],
  overrides?: GatewayRoutingOverrides,
): boolean {
  if (!cfg || typeof cfg !== "object" || providerIds.length === 0) {
    return false;
  }

  const providerUpstreams = overrides?.providerUpstreams ?? {};
  const providerSessionHeaders = overrides?.providerSessionHeaders ?? {};

  const models = (cfg.models ??= {});
  const providers = (models.providers ??= {});
  let changed = false;

  for (const providerId of providerIds) {
    const currentValue = providers[providerId];
    const currentConfig =
      currentValue && typeof currentValue === "object" && !Array.isArray(currentValue)
        ? currentValue
        : {};
    const nextConfig = { ...currentConfig };
    const currentBaseUrl =
      typeof nextConfig.baseUrl === "string" && nextConfig.baseUrl.trim().length > 0
        ? nextConfig.baseUrl
        : undefined;
    const defaultBaseUrl = DEFAULT_PROVIDER_BASE_URLS[providerId];
    if (
      !currentBaseUrl &&
      !defaultBaseUrl &&
      EXPLICIT_BASE_URL_REQUIRED_PROVIDER_IDS.has(providerId)
    ) {
      continue;
    }
    const nextBaseUrl = routeBaseUrlThroughProxy({
      providerId,
      proxyUrl,
      currentBaseUrl,
    });

    if (!Array.isArray(nextConfig.models)) {
      nextConfig.models = [];
      changed = true;
    }

    let mutated = false;

    if (nextConfig.baseUrl !== nextBaseUrl) {
      nextConfig.baseUrl = nextBaseUrl;
      mutated = true;
    }

    mutated = injectHeader(
      nextConfig,
      "x-headroom-base-url",
      providerUpstreams[providerId],
    ) || mutated;

    const sessionHeaderName = providerSessionHeaders[providerId];
    if (sessionHeaderName) {
      mutated = injectHeader(
        nextConfig,
        sessionHeaderName,
        ensureSessionId(providerId),
      ) || mutated;
    }

    if (mutated) {
      providers[providerId] = nextConfig;
      changed = true;
    }
  }

  return changed;
}

/**
 * Merge `headerName: headerValue` into `target.headers` if a non-empty
 * value is supplied and the value is not already present. Returns true
 * when the header was added or changed. The merge preserves unrelated
 * existing headers so other plugins' contributions are not disturbed.
 */
function injectHeader(
  target: Record<string, any>,
  headerName: string,
  headerValue: string | undefined,
): boolean {
  if (!headerName || !headerValue || headerValue.length === 0) {
    return false;
  }
  const existingHeaders =
    target.headers && typeof target.headers === "object" && !Array.isArray(target.headers)
      ? target.headers
      : {};
  if (existingHeaders[headerName] === headerValue) {
    return false;
  }
  target.headers = { ...existingHeaders, [headerName]: headerValue };
  return true;
}

function routeBaseUrlThroughProxy(params: {
  providerId: string;
  proxyUrl: string;
  currentBaseUrl?: string;
}): string {
  const upstreamBaseUrl = params.currentBaseUrl ?? DEFAULT_PROVIDER_BASE_URLS[params.providerId];
  if (!upstreamBaseUrl) {
    // No upstream URL is configured for this provider (e.g. a built-in
    // bundled provider with a hard-coded DEFAULT entry that we didn't
    // match, or a provider id that simply doesn't have a baseUrl yet).
    // Normalize the proxy pathname to a provider-recognized route prefix.
    try {
      const proxy = new URL(params.proxyUrl);
      proxy.pathname = resolveProxyPathPrefix({
        providerId: params.providerId,
      });
      return proxy.toString().replace(/\/$/, "");
    } catch {
      return params.proxyUrl;
    }
  }

  try {
    const proxy = new URL(params.proxyUrl);
    const upstream = new URL(upstreamBaseUrl);
    // Normalize the rewritten proxy URL's pathname to a proxy-recognized
    // route prefix. OpenAI-compatible providers use `/v1/...`; Gemini native
    // traffic must keep `/v1beta/...` so requests reach
    // `handle_gemini_generate_content` instead of the generic passthrough.
    //
    // Per-provider upstream targeting is handled separately via the
    // `x-headroom-base-url` request header, so collapsing the proxy pathname
    // here loses no upstream information.
    proxy.pathname = resolveProxyPathPrefix({
      providerId: params.providerId,
      upstreamBaseUrl,
    });
    proxy.search = upstream.search;
    proxy.hash = "";
    return proxy.toString().replace(/\/$/, "");
  } catch {
    return params.proxyUrl;
  }
}
