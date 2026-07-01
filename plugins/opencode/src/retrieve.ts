import type { CompressResult } from "headroom-ai";
import { compress } from "headroom-ai";

let _proxyUrlCache: string | null = null;

export function setDefaultProxyUrl(url: string): void {
  _proxyUrlCache = url;
}

export function getDefaultProxyUrl(): string {
  return _proxyUrlCache ?? process.env.HEADROOM_BASE_URL ?? "http://localhost:8787";
}

export interface RetrieveToolConfig {
  proxyBaseUrl: string;
}

export function createHeadroomRetrieveTool(config: RetrieveToolConfig) {
  const origin = config.proxyBaseUrl.replace(/\/+$/, "");

  return {
    name: "headroom_retrieve",
    description: "Retrieve original uncompressed content that was compressed to save tokens. Trust kept rows unless you have a concrete gap. Retrieve when you need raw, original, or complete content, or when you need to inspect the original payload for a specific follow-up. The hash is provided in compression markers like [N items compressed... hash=abc123].",
    parameters: {
      type: "object" as const,
      properties: {
        hash: {
          type: "string",
          description: "The 24-character hex hash from the compression marker",
        },
        query: {
          type: "string",
          description:
            "Optional context hint for the concrete gap you are checking. The hint is recorded for feedback and stats; retrieval still returns the full original content.",
        },
      },
      required: ["hash"],
    },
    execute: async (args: { hash: string; query?: string }): Promise<string> => {
      const { hash, query } = args;

      if (!/^[a-f0-9]{24}$/i.test(hash)) {
        return JSON.stringify({
          error: "Invalid hash format. Expected 24 hex characters.",
        });
      }

      try {
        const url = `${origin}/v1/retrieve`;
        const body: { hash: string; query?: string } = { hash };
        if (query) {
          body.query = query;
        }

        const resp = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(10_000),
        });

        if (!resp.ok) {
          const body = await resp.text().catch(() => "");
          return JSON.stringify({
            error: `Retrieval failed: HTTP ${resp.status}`,
            details: body,
          });
        }

        const data = await resp.json();
        return typeof data === "string" ? data : JSON.stringify(data);
      } catch (error) {
        return JSON.stringify({
          error: `Retrieval failed: ${error}`,
          hint: "The compressed content may have expired under the current proxy TTL.",
        });
      }
    },
  };
}

export async function compressWithHeadroom(
  messages: unknown[],
  options: {
    model?: string;
    tokenBudget?: number;
    proxyUrl?: string;
  } = {},
): Promise<CompressResult> {
  return compress(messages, {
    baseUrl: options.proxyUrl ?? getDefaultProxyUrl(),
    model: options.model ?? "gpt-4o",
    tokenBudget: options.tokenBudget,
    stack: "opencode",
  });
}
