import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import path from "node:path";
import { z } from "zod";

import { createHeadroomRetrieveTool, getDefaultProxyUrl } from "./retrieve.js";
import type { HeadroomToolPolicyConfig } from "./transport.js";
import {
  acknowledgeNativeToolExecution,
  enforceNativeToolExecution,
  installHeadroomTransport,
  refreshHeadroomToolPolicy,
  TOOL_POLICY_ENV,
  TOOL_POLICY_PATH_ENV,
  TOOL_POLICY_REFRESH_SECONDS_ENV,
  TOOL_POLICY_TOKEN_ENV,
  TOOL_POLICY_URL_ENV,
} from "./transport.js";

export interface HeadroomOpenCodePluginOptions {
  proxyUrl?: string;
  project?: string;
  backend?: string;
  debug?: boolean;
  toolPolicy?: HeadroomToolPolicyConfig | string;
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

export const HeadroomPlugin: Plugin = async (input, options = {}) => {
  const pluginOptions = options as HeadroomOpenCodePluginOptions;
  const proxyUrl = resolveProxyUrl(pluginOptions);
  const projectPath = input.worktree || input.directory;
  const project = pluginOptions.project ?? projectPath;
  const retrieveTool = createHeadroomRetrieveTool({ proxyBaseUrl: proxyUrl });
  const uninstallTransport = installHeadroomTransport({
    proxyUrl,
    project,
    policyProject: projectPath,
    debug: pluginOptions.debug,
    toolPolicy: pluginOptions.toolPolicy,
  });
  await refreshHeadroomToolPolicy();
  const pendingPreflights = new Map<
    string,
    Awaited<ReturnType<typeof enforceNativeToolExecution>>
  >();

  const effectiveCwd = (args: Record<string, unknown>): string => {
    const configured = typeof args.workdir === "string" ? args.workdir : projectPath;
    return path.resolve(projectPath, configured);
  };

  const freezeArguments = (value: unknown, seen = new Set<object>()): void => {
    if (!value || typeof value !== "object" || seen.has(value)) return;
    seen.add(value);
    for (const child of Object.values(value as Record<string, unknown>)) {
      freezeArguments(child, seen);
    }
    Object.freeze(value);
  };

  return {
    dispose: async () => {
      pendingPreflights.clear();
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
      delete output.env[TOOL_POLICY_TOKEN_ENV];
      delete output.env[TOOL_POLICY_URL_ENV];
      output.env.HEADROOM_ACTIVE = "1";
      output.env.HEADROOM_PROXY_URL = proxyUrl;
      output.env.HEADROOM_PROJECT = project;
      if (pluginOptions.backend) {
        output.env.HEADROOM_BACKEND = pluginOptions.backend;
      }
      if (process.env[TOOL_POLICY_ENV]) {
        output.env[TOOL_POLICY_ENV] = process.env[TOOL_POLICY_ENV];
      }
      if (process.env[TOOL_POLICY_PATH_ENV]) {
        output.env[TOOL_POLICY_PATH_ENV] = process.env[TOOL_POLICY_PATH_ENV];
      }
      for (const name of [TOOL_POLICY_REFRESH_SECONDS_ENV]) {
        if (process.env[name]) {
          output.env[name] = process.env[name];
        }
      }
    },
    "tool.execute.before": async (hookInput, output) => {
      const preflight = await enforceNativeToolExecution(
        hookInput.tool,
        output.args,
        effectiveCwd(output.args),
        {
          sessionID: hookInput.sessionID,
          callID: hookInput.callID,
        },
      );
      if (preflight) {
        freezeArguments(output.args);
        Object.freeze(output);
        pendingPreflights.set(`${hookInput.sessionID}\0${hookInput.callID}`, preflight);
      }
    },
    "tool.execute.after": async (hookInput) => {
      const key = `${hookInput.sessionID}\0${hookInput.callID}`;
      const preflight = pendingPreflights.get(key);
      pendingPreflights.delete(key);
      if (!preflight) return;
      acknowledgeNativeToolExecution(
        preflight,
        hookInput.tool,
        hookInput.args,
        effectiveCwd(hookInput.args),
        {
          sessionID: hookInput.sessionID,
          callID: hookInput.callID,
        },
      );
    },
  };
};

export default HeadroomPlugin;
