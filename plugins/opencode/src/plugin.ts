import { mkdir, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import { z } from "zod";

import { createHeadroomRetrieveTool, getDefaultProxyUrl } from "./retrieve.js";
import { installHeadroomTransport } from "./transport.js";

type HeadroomOpenCodeMode = "native-fetch" | "transport";

export interface HeadroomOpenCodePluginOptions {
  proxyUrl?: string;
  project?: string;
  backend?: string;
  debug?: boolean;
  mode?: HeadroomOpenCodeMode;
}

function normalizeProxyUrl(url: string): string {
  return url.replace(/\/+$/, "");
}

function resolveProxyUrl(options?: HeadroomOpenCodePluginOptions): string {
  return normalizeProxyUrl(
    options?.proxyUrl ??
      process.env.HEADROOM_PROXY_URL ??
      process.env.HEADROOM_BASE_URL ??
      getDefaultProxyUrl(),
  );
}

function resolveMode(options?: HeadroomOpenCodePluginOptions): HeadroomOpenCodeMode {
  return options?.mode ?? "native-fetch";
}

function proxyBaseUrl(proxyUrl: string): string {
  return `${normalizeProxyUrl(proxyUrl)}/v1`;
}

// These providers can be routed cleanly by swapping only `baseURL` to Headroom's
// native `/v1` surface while preserving provider identity, auth, and model shape.
//
// Do not add `opencode*` or `zenmux` here yet: those providers still need the
// original upstream base preserved so the transport fallback can forward with
// `x-headroom-base-url` until Headroom's native proxy path supports them.
const NATIVE_BASE_URL_PROVIDERS = ["anthropic", "openai", "github-copilot"] as const;
const OPENAI_COMPATIBLE_HEADER = "x-headroom-base-url";
const ZENMUX_UPSTREAM_BASE = "https://zenmux.ai/api";
const OPENCODE_PROVIDER_ID = "opencode";
const OPENCODE_GO_PROVIDER_ID = "opencode-go";
const OPENCODE_GO_UPSTREAM_BASE = "https://opencode.ai/zen/go";
const HEADROOM_WORKSPACE_DIR_ENV = "HEADROOM_WORKSPACE_DIR";
const OPENCODE_CLIENT_MARKER_PREFIX = "opencode-";

function normalizeUpstreamBaseUrl(url: string): string {
  const normalized = normalizeProxyUrl(url);
  return normalized.endsWith("/v1") ? normalized.slice(0, -3) : normalized;
}

function resolveHeadroomWorkspaceDir(): string {
  const override = process.env[HEADROOM_WORKSPACE_DIR_ENV]?.trim();
  return override || path.join(os.homedir(), ".headroom");
}

function resolveProxyClientMarkerDir(proxyUrl: string): string | undefined {
  try {
    const url = new URL(proxyUrl);
    const port = url.port || (url.protocol === "https:" ? "443" : "80");
    return path.join(resolveHeadroomWorkspaceDir(), "clients", port);
  } catch {
    return undefined;
  }
}

async function registerProxyClientMarker(proxyUrl: string): Promise<string | undefined> {
  const markerDir = resolveProxyClientMarkerDir(proxyUrl);
  if (!markerDir) {
    return undefined;
  }

  const markerPath = path.join(
    markerDir,
    `${OPENCODE_CLIENT_MARKER_PREFIX}${process.pid}-${Date.now()}.json`,
  );

  try {
    await mkdir(markerDir, { recursive: true });
    await writeFile(
      markerPath,
      JSON.stringify({
        pid: process.pid,
        started_at: Date.now() / 1000,
        source: "opencode-plugin",
      }),
      "utf8",
    );
    return markerPath;
  } catch {
    return undefined;
  }
}

async function unregisterProxyClientMarker(markerPath: string | undefined): Promise<void> {
  if (!markerPath) {
    return;
  }
  try {
    await unlink(markerPath);
  } catch {
    // Best effort: stale markers are pruned by the wrapper liveness checks.
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function ensureProviderOptions(config: Record<string, unknown>, providerId: string): Record<string, unknown> {
  const providers = isRecord(config.provider) ? config.provider : {};
  config.provider = providers;

  const provider = isRecord(providers[providerId]) ? providers[providerId] : {};
  providers[providerId] = provider;

  const options = isRecord(provider.options) ? provider.options : {};
  provider.options = options;
  return options;
}

function setOpenAICompatibleUpstreamHeader(
  options: Record<string, unknown>,
  upstreamBaseUrl: string | undefined,
): void {
  if (!upstreamBaseUrl) {
    return;
  }
  const headers = isRecord(options.headers) ? options.headers : {};
  options.headers = headers;
  headers[OPENAI_COMPATIBLE_HEADER] = upstreamBaseUrl;
}

export function applyNativeProviderOverrides(config: Record<string, unknown>, proxyUrl: string): void {
  const baseURL = proxyBaseUrl(proxyUrl);
  for (const providerId of NATIVE_BASE_URL_PROVIDERS) {
    const options = ensureProviderOptions(config, providerId);
    options.baseURL = baseURL;
  }

  const opencodeOptions = ensureProviderOptions(config, OPENCODE_PROVIDER_ID);
  setOpenAICompatibleUpstreamHeader(
    opencodeOptions,
    typeof opencodeOptions.baseURL === "string" && opencodeOptions.baseURL.trim() !== ""
      ? normalizeUpstreamBaseUrl(opencodeOptions.baseURL)
      : undefined,
  );
  opencodeOptions.baseURL = baseURL;

  const opencodeGoOptions = ensureProviderOptions(config, OPENCODE_GO_PROVIDER_ID);
  setOpenAICompatibleUpstreamHeader(opencodeGoOptions, OPENCODE_GO_UPSTREAM_BASE);
  opencodeGoOptions.baseURL = baseURL;

  const zenmuxOptions = ensureProviderOptions(config, "zenmux");
  setOpenAICompatibleUpstreamHeader(zenmuxOptions, ZENMUX_UPSTREAM_BASE);
  zenmuxOptions.baseURL = baseURL;
}

export const HeadroomPlugin: Plugin = async (input, options = {}) => {
  const pluginOptions = options as HeadroomOpenCodePluginOptions;
  const proxyUrl = resolveProxyUrl(pluginOptions);
  const mode = resolveMode(pluginOptions);
  const retrieveTool = createHeadroomRetrieveTool({ proxyBaseUrl: proxyUrl });
  const uninstallTransport = installHeadroomTransport({
    proxyUrl,
    debug: pluginOptions.debug,
  });
  const markerPath = await registerProxyClientMarker(proxyUrl);

  return {
    dispose: async () => {
      await unregisterProxyClientMarker(markerPath);
      uninstallTransport();
    },
    tool: {
      headroom_retrieve: tool({
        description: retrieveTool.description,
        args: {
          hash: z
            .string()
            .regex(/^[a-f0-9]{24}$/i, "Expected 24-character hex hash"),
          query: z.string().optional(),
        },
        async execute(args) {
          return retrieveTool.execute(args);
        },
      }),
    },
    config: async (config) => {
      if (mode !== "native-fetch") {
        return;
      }
      applyNativeProviderOverrides(config as unknown as Record<string, unknown>, proxyUrl);
    },
    "shell.env": async (_input, output) => {
      output.env.HEADROOM_ACTIVE = "1";
      output.env.HEADROOM_PROXY_URL = proxyUrl;
      output.env.HEADROOM_PROJECT =
        pluginOptions.project ??
        (input.project as { id?: string }).id ??
        input.directory;
      if (pluginOptions.backend) {
        output.env.HEADROOM_BACKEND = pluginOptions.backend;
      }
    },
    "chat.headers": async (incoming, output) => {
      if (
        incoming.model.providerID !== OPENCODE_PROVIDER_ID &&
        incoming.model.providerID !== OPENCODE_GO_PROVIDER_ID
      ) {
        return;
      }
      const upstream =
        typeof incoming.model.api?.url === "string" && incoming.model.api.url.trim() !== ""
          ? normalizeUpstreamBaseUrl(incoming.model.api.url)
          : incoming.model.providerID === OPENCODE_GO_PROVIDER_ID
            ? OPENCODE_GO_UPSTREAM_BASE
            : undefined;
      if (!upstream) {
        return;
      }
      output.headers[OPENAI_COMPATIBLE_HEADER] = upstream;
    },
  };
};

export default HeadroomPlugin;
