/**
 * Proxy route prefix resolution for gateway provider baseUrl rewriting.
 *
 * Headroom's proxy registers provider-specific handlers on fixed path prefixes.
 * OpenAI-compatible traffic uses `/v1/...`; Gemini native traffic uses
 * `/v1beta/models/...:generateContent` and related routes.
 */

export const GEMINI_PROXY_PROVIDER_IDS = new Set(["google", "gemini"]);

const GEMINI_GENERATE_CONTENT_PATH =
  /^\/v1beta\/models\/[^:]+:generateContent$/;

export function resolveProxyPathPrefix(params: {
  providerId: string;
  upstreamBaseUrl?: string;
}): string {
  const providerId = params.providerId.trim().toLowerCase();

  if (GEMINI_PROXY_PROVIDER_IDS.has(providerId)) {
    return "/v1beta";
  }

  if (params.upstreamBaseUrl) {
    try {
      const upstream = new URL(params.upstreamBaseUrl);
      const normalizedPath = upstream.pathname.replace(/\/$/, "") || "/";
      if (normalizedPath.endsWith("/v1beta")) {
        return "/v1beta";
      }
    } catch {
      // Fall through to the OpenAI-compatible default.
    }
  }

  return "/v1";
}

export function buildGeminiGenerateContentRequestUrl(
  providerBaseUrl: string,
  model = "gemini-2.5-flash",
): string {
  const base = providerBaseUrl.replace(/\/$/, "");
  return `${base}/models/${model}:generateContent`;
}

/** True when the URL matches Headroom's Gemini generateContent handler route. */
export function matchesGeminiGenerateContentRoute(requestUrl: string): boolean {
  try {
    const url = new URL(requestUrl);
    return GEMINI_GENERATE_CONTENT_PATH.test(url.pathname);
  } catch {
    return false;
  }
}
