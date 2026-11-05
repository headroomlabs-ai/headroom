/**
 * CCR (Compress-Cache-Retrieve) tool for OpenClaw.
 *
 * Allows the agent to retrieve original uncompressed content
 * from the Headroom proxy's compression store.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
import { normalizeAndValidateProxyUrl } from "../proxy-manager.js";

export interface RetrieveToolConfig {
  proxyUrl: string;
}

export function createHeadroomRetrieveTool(config: RetrieveToolConfig) {
  const proxyOrigin = normalizeAndValidateProxyUrl(config.proxyUrl);

  return {
    name: "headroom_retrieve",
    description:
      "Retrieve original uncompressed content from Headroom's compression store. " +
      "Use when compressed context mentions a hash and you need the full details. " +
      "Pass the hash from the compression marker (12 or 24 hex characters). " +
      "Retrieval is by hash and always returns the full original content.",
    parameters: {
      type: "object" as const,
      properties: {
        hash: {
          type: "string",
          description: "The 12- or 24-character hex hash from the compression marker",
        },
      },
      required: ["hash"],
    },
    // OpenClaw calls a registered tool as execute(toolCallId, params, signal?,
    // onUpdate?). The hash used to be read from the first argument, so the
    // tool-call id was validated as a hash and every real call failed before
    // any request was sent (#4638). A direct execute({ hash }) call still works.
    execute: async (
      toolCallIdOrArgs: string | { hash: string },
      params?: { hash: string },
    ): Promise<string> => {
      const args = typeof toolCallIdOrArgs === "string" ? params : toolCallIdOrArgs;
      const hash = typeof args?.hash === "string" ? args.hash : "";

      // Validate hash format
      if (!/^(?:[a-f0-9]{12}|[a-f0-9]{24})$/i.test(hash)) {
        return JSON.stringify({
          error: "Invalid hash format. Expected 12 or 24 hex characters.",
        });
      }

      try {
        // The store keys entries by lowercase hex, so normalize once here
        // instead of requiring every caller to pass a pre-lowercased hash.
        const normalizedHash = hash.toLowerCase();
        const url = `${proxyOrigin}/v1/retrieve/${normalizedHash}`;

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
          hint: "The compressed content may have expired (default TTL: 30 minutes)",
        });
      }
    },
  };
}
