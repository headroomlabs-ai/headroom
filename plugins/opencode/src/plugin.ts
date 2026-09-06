import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import { z } from "zod";

import { createHeadroomRetrieveTool, getDefaultProxyUrl } from "./retrieve.js";
import { installHeadroomTransport } from "./transport.js";

export interface HeadroomOpenCodePluginOptions {
  proxyUrl?: string;
  project?: string;
  backend?: string;
  debug?: boolean;
  sessionToken?: string;
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

// Read by workspace_registry.resolve_registered_cwd() on the proxy side --
// mirrors wrap.py's _apply_session_token_header_env for claude.
function resolveSessionToken(options?: HeadroomOpenCodePluginOptions): string | undefined {
  return options?.sessionToken ?? process.env.HEADROOM_OPENCODE_SESSION_TOKEN;
}

export const HeadroomPlugin: Plugin = async (input, options = {}) => {
  const pluginOptions = options as HeadroomOpenCodePluginOptions;
  const proxyUrl = resolveProxyUrl(pluginOptions);
  const project =
    pluginOptions.project ??
    (input.project as { id?: string } | undefined)?.id ??
    input.directory;
  const retrieveTool = createHeadroomRetrieveTool({ proxyBaseUrl: proxyUrl });
  const uninstallTransport = installHeadroomTransport({
    proxyUrl,
    project,
    debug: pluginOptions.debug,
    sessionToken: resolveSessionToken(pluginOptions),
  });

  return {
    dispose: async () => {
      uninstallTransport();
    },
    tool: {
      headroom_retrieve: tool({
        description: retrieveTool.description,
        args: {
          hash: z
            .string()
            .regex(/^[a-f0-9]{24}$/i, "Expected 24-character hex hash"),
        },
        async execute(args) {
          return retrieveTool.execute(args);
        },
      }),
    },
    "shell.env": async (_input, output) => {
      output.env.HEADROOM_ACTIVE = "1";
      output.env.HEADROOM_PROXY_URL = proxyUrl;
      output.env.HEADROOM_PROJECT = project;
      if (pluginOptions.backend) {
        output.env.HEADROOM_BACKEND = pluginOptions.backend;
      }
    },
  };
};

export default HeadroomPlugin;
