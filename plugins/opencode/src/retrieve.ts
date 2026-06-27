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
    description: "Retrieve original uncompressed content that was compressed to save tokens. Trust kept rows unless you have a concrete gap. Retrieve when you need raw, original, or complete content, or when a targeted follow-up cannot be answered from the kept summary. The hash is provided in compression markers like [N items compressed... hash=abc123].",
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
            "Optional targeted search query for a concrete gap. Use it when the kept summary cannot answer a specific follow-up. If omitted, returns all original items.",
        },
      },
      required: ["hash"],
    },
    execute: async (args: { hash: string }): Promise<string> => {
      const { hash } = args;

      if (!/^[a-f0-9]{24}$/i.test(hash)) {
        return JSON.stringify({
          error: "Invalid hash format. Expected 24 hex characters.",
        });
      }

      try {
        const url = `${origin}/v1/retrieve/${hash}`;

        const resp = await fetch(url, {
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
