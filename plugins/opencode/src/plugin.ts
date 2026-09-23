import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import type { Plugin as PluginV2 } from "@opencode/plugin";
import { z } from "zod";

import { createHeadroomRetrieveTool, getDefaultProxyUrl } from "./retrieve.js";
import {
  BASE_URL_HEADER,
  ORIGINAL_PATH_HEADER,
  PROJECT_HEADER,
  installHeadroomTransport,
  routesThroughProxy,
} from "./transport.js";

export interface HeadroomOpenCodePluginOptions {
  proxyUrl?: string;
  project?: string;
  backend?: string;
  debug?: boolean;
}

export const HEADROOM_PLUGIN_ID = "headroom";

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

function headroomShellEnv(
  proxyUrl: string,
  project: string,
  options: HeadroomOpenCodePluginOptions,
): Record<string, string> {
  return {
    HEADROOM_ACTIVE: "1",
    HEADROOM_PROXY_URL: proxyUrl,
    HEADROOM_PROJECT: project,
    ...(options.backend ? { HEADROOM_BACKEND: options.backend } : {}),
  };
}

// The endpoint an OpenCode 2.x package calls under its baseURL, for packages
// on the OpenAI chat-completions or responses wire format; undefined otherwise.
// `.../messages` (Anthropic wire) and other protocols are left alone.
function openAiWireSuffix(pkg: string | undefined): string | undefined {
  if (!pkg?.startsWith("@opencode/ai/providers/")) return undefined;
  if (pkg === "@opencode/ai/providers/openai-compatible" || pkg.endsWith("/chat")) {
    return "/chat/completions";
  }
  if (pkg.endsWith("-responses") || pkg.endsWith("/responses")) return "/responses";
  return undefined;
}

// OpenCode 2.x sends model traffic through Effect's FetchHttpClient, which
// captures `globalThis.fetch` on the process's first HTTP request, before any
// plugin runs, so the transport's fetch patch never sees it. Point each
// remote OpenAI-wire model at the proxy instead, with the same routing headers
// the transport sends: the upstream origin, and the real path the proxy
// appends to it.
type ModelEditor = Parameters<Parameters<PluginV2.Context["model"]["transform"]>[0]>[0];

function routeModelsThroughProxy(
  models: ModelEditor,
  proxyUrl: string,
  project: string,
): void {
  for (const model of models.list()) {
    // DeepMutable turns the branded ID strings into object types.
    const providerID = String(model.providerID);
    const modelID = String(model.id);
    const provider = models.provider.get(providerID)?.provider;
    const suffix = openAiWireSuffix(model.package ?? provider?.package);
    const baseURL = model.settings?.baseURL ?? provider?.settings?.baseURL;
    if (!suffix || typeof baseURL !== "string") continue;
    if (!routesThroughProxy(baseURL, proxyUrl)) continue;
    const upstream = new URL(baseURL);
    models.update(providerID, modelID, (draft) => {
      draft.settings = { ...draft.settings, baseURL: `${proxyUrl}/v1` };
      draft.headers = {
        ...draft.headers,
        [BASE_URL_HEADER]: upstream.origin,
        [ORIGINAL_PATH_HEADER]: `${upstream.pathname.replace(/\/+$/, "")}${suffix}`,
        [PROJECT_HEADER]: project,
      };
    });
  }
}

// OpenCode 1.x: a factory returning a hooks object.
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
      Object.assign(output.env, headroomShellEnv(proxyUrl, project, pluginOptions));
    },
  };
};

// OpenCode 2.x: setup registers hooks imperatively and returns its cleanup.
// OpenCode reruns setup (after cleanup) whenever it hot-reloads the plugin;
// the transport is refcounted, so reinstalling is safe.
export const headroomSetup: PluginV2.Plugin["setup"] = async (ctx) => {
  const pluginOptions = ctx.options as HeadroomOpenCodePluginOptions;
  const proxyUrl = resolveProxyUrl(pluginOptions);
  const project =
    pluginOptions.project ?? ctx.location.project.id ?? ctx.location.directory;
  const retrieveTool = createHeadroomRetrieveTool({ proxyBaseUrl: proxyUrl });
  const uninstallTransport = installHeadroomTransport({
    proxyUrl,
    project,
    debug: pluginOptions.debug,
  });

  try {
    await ctx.model.transform((models) => {
      routeModelsThroughProxy(models, proxyUrl, project);
    });
    await ctx.tool.transform((editor) => {
      editor.add({
        name: retrieveTool.name,
        description: retrieveTool.description,
        input: retrieveTool.parameters,
        // Offer the tool directly, like OpenCode's built-ins, rather than only
        // through the code-mode `execute` tool.
        options: { codemode: false },
        async execute(args) {
          return { content: await retrieveTool.execute(args as { hash: string }) };
        },
      });
    });
    await ctx.shell.hook("create.before", (shell) => {
      Object.assign(shell.env, headroomShellEnv(proxyUrl, project, pluginOptions));
    });
  } catch (error) {
    uninstallTransport();
    throw error;
  }

  return () => {
    uninstallTransport();
  };
};

// One default export for both loaders: OpenCode 1.x reads `server`, 2.x reads
// `setup`, and each ignores the other's key.
export const HeadroomOpenCodePlugin = {
  id: HEADROOM_PLUGIN_ID,
  server: HeadroomPlugin,
  setup: headroomSetup,
};

export default HeadroomOpenCodePlugin;
