import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import type { Plugin as PluginV2 } from "@opencode/plugin";
import { z } from "zod";

import {
  createHeadroomRetrieveTool,
  getDefaultProxyUrl,
  trimTrailingSlashes,
} from "./retrieve.js";
import {
  BASE_URL_HEADER,
  EXCLUDE_HOSTS_ENV,
  ORIGINAL_PATH_HEADER,
  PROJECT_HEADER,
  SESSION_TOKEN_HEADER,
  installHeadroomTransport,
  projectHeaderValue,
  modelBaseRoutesThroughProxy,
  normalizeExcludeHosts,
  setHeadroomWireRoots,
} from "./transport.js";

export interface HeadroomOpenCodePluginOptions {
  proxyUrl?: string;
  project?: string;
  excludeHosts?: string[];
  backend?: string;
  debug?: boolean;
  sessionToken?: string;
  /**
   * Provider base URLs whose wire the transport must route. Normally read from
   * OpenCode's own configuration, so this is only needed when that is
   * unavailable (or to cover a base the plugin cannot see).
   */
  wireRoots?: string[];
}

export const HEADROOM_PLUGIN_ID = "headroom";

// OpenCode's own configuration is the authority on where a provider mounts its
// wire: `provider.<id>.options.baseURL` in the config, or the runtime list with
// injective defaults resolved. A base URL the user configured must route, even
// when its path has no version segment and no recognisable root name
// (/tenant/acme), because OpenCode 1.x sends those calls through the patched
// fetch and would otherwise bypass Headroom silently (#4747 review).
function collectBaseUrls(config: unknown, providers: unknown): string[] {
  const urls = new Set<string>();
  const add = (value: unknown) => {
    if (typeof value === "string" && value.trim()) {
      urls.add(value.trim());
    }
  };

  const configProviders = (config as { provider?: Record<string, { options?: { baseURL?: unknown } }> })
    ?.provider;
  for (const entry of Object.values(configProviders ?? {})) {
    add(entry?.options?.baseURL);
  }

  // /provider returns { all: [...] } in the generated API and the bare list in
  // older servers; handle both rather than depending on one version.
  const list = Array.isArray(providers) ? providers : (providers as { all?: unknown })?.all;
  for (const entry of Array.isArray(list) ? list : []) {
    add((entry as { options?: { baseURL?: unknown } })?.options?.baseURL);
  }
  return [...urls];
}

// The plugin must not block startup on a round trip to the server that is
// loading it, and a server that is not serving yet must not disable routing.
// So install first with whatever is known, then widen the rule when the answer
// arrives. Requests that land in between fall back to the path heuristics.
//
// Each endpoint is applied on its own: keeping a base back because the other
// request is still pending would leave a configured wire bypassing Headroom for
// as long as that request stalls (#4747 review). A failed or missing read only
// costs its own bases, never the other's.
function resolveConfiguredWireRoots(
  client: { config?: { get?: () => Promise<unknown> }; provider?: { list?: () => Promise<unknown> } } | undefined,
  onResolved: (urls: string[]) => void,
): void {
  if (!client?.config?.get && !client?.provider?.list) {
    return;
  }

  const apply = (read: (() => Promise<unknown>) | undefined, pick: (value: unknown) => string[]) => {
    if (!read) {
      return;
    }
    // Promise.resolve() turns a synchronous throw into a rejection, and the
    // catch keeps a failing read on the path heuristics instead of surfacing.
    void Promise.resolve()
      .then(read)
      .then((value) => {
        const urls = pick(value);
        if (urls.length > 0) {
          onResolved(urls);
        }
      })
      .catch(() => {
        // Routing stays on the path heuristics; not worth failing a session for.
      });
  };

  apply(client.config?.get?.bind(client.config), (config) =>
    collectBaseUrls((config as { data?: unknown })?.data ?? config, undefined),
  );
  apply(client.provider?.list?.bind(client.provider), (providers) =>
    collectBaseUrls(undefined, (providers as { data?: unknown })?.data ?? providers),
  );
}

function normalizeProxyUrl(url: string): string {
  return trimTrailingSlashes(url);
}

function resolveProxyUrl(options?: HeadroomOpenCodePluginOptions): string {
  return normalizeProxyUrl(
    options?.proxyUrl ||
      process.env.HEADROOM_PROXY_URL ||
      process.env.HEADROOM_BASE_URL ||
      getDefaultProxyUrl(),
  );
}

// Read by workspace_registry.resolve_registered_cwd() on the proxy side --
// mirrors wrap.py's _apply_session_token_header_env for claude.
function resolveSessionToken(options?: HeadroomOpenCodePluginOptions): string | undefined {
  return options?.sessionToken ?? process.env.HEADROOM_OPENCODE_SESSION_TOKEN;
}


// The exclusion list governs both the transport patch and the 2.x model
// rewrite, so resolve it once: option over environment variable, normalized
// the same way the transport does.
function resolveExcludeHosts(options?: HeadroomOpenCodePluginOptions): string[] {
  return normalizeExcludeHosts(options?.excludeHosts ?? process.env[EXCLUDE_HOSTS_ENV] ?? "");
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

// The endpoint an OpenCode 2.x package calls under its baseURL, for packages on
// a wire format the proxy can route; undefined otherwise. GitHub Copilot's
// Claude models use the Anthropic Messages wire (`/messages`), so they must be
// rewritten like the OpenAI wires: skipping them let those requests bypass the
// proxy entirely (#4248).
function modelWireSuffix(pkg: string | undefined): string | undefined {
  if (!pkg?.startsWith("@opencode/ai/providers/")) return undefined;
  if (pkg === "@opencode/ai/providers/openai-compatible" || pkg.endsWith("/chat")) {
    return "/chat/completions";
  }
  if (pkg.endsWith("-responses") || pkg.endsWith("/responses")) return "/responses";
  if (
    pkg === "@opencode/ai/providers/anthropic"
    || pkg === "@opencode/ai/providers/anthropic-compatible"
    || pkg.endsWith("/messages")
  ) {
    return "/messages";
  }
  return undefined;
}

// The proxy's Anthropic route rebuilds the upstream path from its incoming
// `/v1/messages` and ignores `x-headroom-original-path`, so a rewritten model
// reaches the right endpoint only when its base URL has no path prefix beyond
// `/v1`. A prefixed gateway (for example `https://api.z.ai/api/coding/paas/v4`)
// would lose that prefix, so those models stay on their own base URL.
function anthropicBaseKeepsPath(baseURL: string): boolean {
  try {
    const pathname = trimTrailingSlashes(new URL(baseURL).pathname);
    return pathname === "" || pathname === "/v1";
  } catch {
    return false;
  }
}

// OpenCode 2.x sends model traffic through Effect's FetchHttpClient, which
// captures `globalThis.fetch` on the process's first HTTP request, before any
// plugin runs, so the transport's fetch patch never sees it. Point each
// remote model on a routable wire format at the proxy instead, with the same
// the transport sends: the upstream origin, and the real path the proxy
// appends to it.
type ModelEditor = Parameters<Parameters<PluginV2.Context["model"]["transform"]>[0]>[0];

function routeModelsThroughProxy(
  models: ModelEditor,
  proxyUrl: string,
  project: string,
  excludeHosts: string[],
  sessionToken?: string,
): void {
  for (const model of models.list()) {
    // DeepMutable turns the branded ID strings into object types.
    const providerID = String(model.providerID);
    const modelID = String(model.id);
    const provider = models.provider.get(providerID)?.provider;
    const suffix = modelWireSuffix(model.package ?? provider?.package);
    const baseURL = model.settings?.baseURL ?? provider?.settings?.baseURL;
    if (!suffix || typeof baseURL !== "string") continue;
    if (!modelBaseRoutesThroughProxy(baseURL, proxyUrl, excludeHosts)) continue;
    if (suffix === "/messages" && !anthropicBaseKeepsPath(baseURL)) continue;
    const upstream = new URL(baseURL);
    models.update(providerID, modelID, (draft) => {
      draft.settings = { ...draft.settings, baseURL: `${proxyUrl}/v1` };
      draft.headers = {
        ...draft.headers,
        [BASE_URL_HEADER]: upstream.origin,
        [ORIGINAL_PATH_HEADER]: `${trimTrailingSlashes(upstream.pathname)}${suffix}`,
        [PROJECT_HEADER]: projectHeaderValue(project),
        ...(sessionToken ? { [SESSION_TOKEN_HEADER]: sessionToken } : {}),
      };
    });
  }
}

// Every base URL the 2.x transform can see. The transport only needs the ones
// it must keep routing, but passing them all is harmless: it ignores entries it
// cannot parse and bases it would not route anyway.
function modelBaseUrls(models: ModelEditor): string[] {
  const urls = new Set<string>();
  for (const model of models.list()) {
    const provider = models.provider.get(String(model.providerID))?.provider;
    const baseURL = model.settings?.baseURL ?? provider?.settings?.baseURL;
    if (typeof baseURL === "string" && baseURL.trim()) {
      urls.add(baseURL.trim());
    }
  }
  return [...urls];
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
    excludeHosts: pluginOptions.excludeHosts,
    debug: pluginOptions.debug,
    sessionToken: resolveSessionToken(pluginOptions),
    wireRoots: pluginOptions.wireRoots,
  });
  resolveConfiguredWireRoots(input.client, (urls) => {
    if (urls.length > 0) {
      setHeadroomWireRoots(urls.join(","));
    }
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
  const excludeHosts = resolveExcludeHosts(pluginOptions);
  const sessionToken = resolveSessionToken(pluginOptions);
  const uninstallTransport = installHeadroomTransport({
    proxyUrl,
    project,
    excludeHosts,
    debug: pluginOptions.debug,
    sessionToken,
    wireRoots: pluginOptions.wireRoots,
  });

  try {
    await ctx.model.transform((models) => {
      const wireRoots = modelBaseUrls(models);
      if (wireRoots.length > 0) {
        setHeadroomWireRoots(wireRoots.join(","));
      }
      routeModelsThroughProxy(models, proxyUrl, project, excludeHosts, sessionToken);
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
