/* eslint-disable @typescript-eslint/no-explicit-any */

export const DEFAULT_GATEWAY_PROVIDER_IDS = ["openai-codex"] as const;

const DEFAULT_PROVIDER_BASE_URLS: Readonly<Record<string, string>> = {
  "openai-codex": "https://chatgpt.com/backend-api",
  anthropic: "https://api.anthropic.com/v1",
  google: "https://generativelanguage.googleapis.com/v1beta",
  openrouter: "https://openrouter.ai/api/v1",
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

export function applyGatewayProviderBaseUrls<T>(
  cfg: T,
  proxyUrl: string,
  providerIds: readonly string[],
): { changed: boolean; config: T } {
  const next = structuredClone((cfg ?? {}) as any);
  const changed = applyGatewayProviderBaseUrlsInPlace(next, proxyUrl, providerIds);
  return { changed, config: next as T };
}

export function applyGatewayProviderBaseUrlsInPlace(
  cfg: any,
  proxyUrl: string,
  providerIds: readonly string[],
): boolean {
  if (!cfg || typeof cfg !== "object" || providerIds.length === 0) {
    return false;
  }

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
    let nextBaseUrl: string;
    const isOpenAI =
      providerId === "openai" ||
      nextConfig.api === "openai-completions" ||
      providerId === "opencode-go" ||
      providerId === "openrouter" ||
      providerId === "opencode" ||
      providerId === "huggingface";

    const upstreamBaseUrl = currentBaseUrl ?? defaultBaseUrl;

    const isAlreadyProxied =
      currentBaseUrl &&
      (currentBaseUrl.startsWith(proxyUrl) ||
       currentBaseUrl.includes("127.0.0.1") ||
       currentBaseUrl.includes("localhost")) &&
      nextConfig.headers &&
      typeof nextConfig.headers === "object" &&
      typeof nextConfig.headers["x-headroom-base-url"] === "string";

    if (isAlreadyProxied) {
      nextBaseUrl = currentBaseUrl;
    } else if (isOpenAI && upstreamBaseUrl && providerId !== "openai-codex") {
      try {
        const upstreamUrl = new URL(upstreamBaseUrl);
        const upstreamOrigin = upstreamUrl.origin;
        const upstreamPath = upstreamUrl.pathname.replace(/\/$/, "");

        nextBaseUrl = proxyUrl.replace(/\/$/, "") + "/v1";

        nextConfig.headers = {
          ...(nextConfig.headers || {}),
          "x-headroom-base-url": upstreamOrigin,
          "x-headroom-original-path": `${upstreamPath}/chat/completions`,
        };
      } catch {
        nextBaseUrl = routeBaseUrlThroughProxy({
          providerId,
          proxyUrl,
          currentBaseUrl,
        });
      }
    } else {
      nextBaseUrl = routeBaseUrlThroughProxy({
        providerId,
        proxyUrl,
        currentBaseUrl,
      });
    }

    if (!Array.isArray(nextConfig.models)) {
      nextConfig.models = [];
      changed = true;
    }

    const headersChanged = JSON.stringify(currentConfig.headers) !== JSON.stringify(nextConfig.headers);
    if (nextConfig.baseUrl === nextBaseUrl && !headersChanged) {
      providers[providerId] = nextConfig;
      continue;
    }

    nextConfig.baseUrl = nextBaseUrl;
    providers[providerId] = nextConfig;
    changed = true;
  }

  return changed;
}

function routeBaseUrlThroughProxy(params: {
  providerId: string;
  proxyUrl: string;
  currentBaseUrl?: string;
}): string {
  const upstreamBaseUrl = params.currentBaseUrl ?? DEFAULT_PROVIDER_BASE_URLS[params.providerId];
  if (!upstreamBaseUrl) {
    return params.proxyUrl;
  }

  try {
    const proxy = new URL(params.proxyUrl);
    const upstream = new URL(upstreamBaseUrl);
    proxy.pathname = upstream.pathname;
    proxy.search = upstream.search;
    proxy.hash = "";
    return proxy.toString().replace(/\/$/, "");
  } catch {
    return params.proxyUrl;
  }
}
