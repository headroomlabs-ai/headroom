import type {
  ExtensionAPI,
  ExtensionCommandContext,
  ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { PreparedCache } from "./cache.js";
import { HeadroomClient } from "./client.js";
import { loadConfig } from "./config.js";
import { HeadroomRuntime } from "./runtime.js";
import { formatStats, formatStatus } from "./status.js";

export type RuntimeFactory = () => Promise<HeadroomRuntime>;

export async function createDefaultRuntime(): Promise<HeadroomRuntime> {
  const loaded = await loadConfig();
  const client = new HeadroomClient({
    baseUrl: loaded.config.baseUrl,
    timeoutMs: 15_000,
  });
  return new HeadroomRuntime({
    config: loaded.config,
    client,
    cache: new PreparedCache(loaded.config.maxCacheBytes),
    warnings: loaded.warnings,
  });
}

function notify(
  ctx: ExtensionContext,
  message: string,
  type: "info" | "warning" | "error" = "info",
): void {
  if (!ctx.hasUI) return;
  try {
    ctx.ui.notify(message, type);
  } catch {
    // Commands and lifecycle callbacks remain fail-open when UI adapters fail.
  }
}

export function registerHeadroomExtension(
  pi: ExtensionAPI,
  runtimeFactory: RuntimeFactory = createDefaultRuntime,
): void {
  let runtime: HeadroomRuntime | undefined;
  let initializing: Promise<HeadroomRuntime> | undefined;
  let lifecycleGeneration = 0;
  let sessionOpen = false;

  const initialize = async (): Promise<HeadroomRuntime> => {
    if (runtime) return runtime;
    initializing ??= runtimeFactory().then((created) => {
      runtime = created;
      return created;
    });
    try {
      return await initializing;
    } finally {
      initializing = undefined;
    }
  };

  pi.on("session_start", async (_event, ctx) => {
    sessionOpen = true;
    const generation = ++lifecycleGeneration;
    try {
      const active = await initialize();
      if (generation !== lifecycleGeneration) {
        if (!sessionOpen && runtime === active) {
          runtime = undefined;
          active.stop();
        }
        return;
      }
      active.start(ctx);
    } catch {
      if (generation === lifecycleGeneration) {
        notify(
          ctx,
          "Headroom extension could not initialize; context is unchanged.",
          "warning",
        );
      }
    }
  });

  pi.on("tool_result", (event, ctx) => {
    try {
      runtime?.observeToolResult(event, ctx);
    } catch {
      return undefined;
    }
    return undefined;
  });

  pi.on("context", (event, ctx) => {
    try {
      const messages = runtime?.transform(event.messages, ctx);
      return messages ? { messages } : undefined;
    } catch {
      return undefined;
    }
  });

  pi.on("session_shutdown", (_event, ctx) => {
    sessionOpen = false;
    lifecycleGeneration += 1;
    const active = runtime;
    runtime = undefined;
    try {
      active?.stop();
    } catch {
      try {
        ctx.ui.setStatus("headroom", undefined);
      } catch {
        // Shutdown remains fail-open.
      }
    }
  });

  pi.registerTool({
    name: "headroom_retrieve",
    label: "Headroom Retrieve",
    description:
      "Retrieve exact original content for a CCR hash emitted by Headroom compression.",
    promptSnippet: "Retrieve original Headroom-compressed content by CCR hash",
    promptGuidelines: [
      "Use headroom_retrieve when a compressed tool result says to retrieve a CCR hash.",
    ],
    parameters: Type.Object({
      hash: Type.String({
        minLength: 1,
        description: "CCR hash from a Headroom compression marker",
      }),
    }),
    executionMode: "parallel",
    async execute(_toolCallId, params, signal) {
      const miss = `Headroom retrieval miss for ${params.hash}. Rerun the originating tool.`;
      let text = miss;
      try {
        if (runtime) text = await runtime.retrieve(params.hash, signal);
      } catch {
        text = miss;
      }
      return { content: [{ type: "text", text }], details: {} };
    },
  });

  pi.registerCommand("headroom", {
    description: "Show or change Headroom compression status",
    getArgumentCompletions(argumentPrefix) {
      const values = ["status", "on", "off", "health", "stats"];
      const matches = values.filter((value) => value.startsWith(argumentPrefix));
      return matches.map((value) => ({ value, label: value }));
    },
    async handler(args: string, ctx: ExtensionCommandContext) {
      try {
        const active = runtime;
        if (!active) {
          notify(ctx, "Headroom is not initialized.", "warning");
          return;
        }

        const command = args.trim().toLowerCase();
        if (command.length === 0) {
          notify(ctx, formatStatus(active.snapshot()));
          return;
        }
        if (command === "on" || command === "off") {
          active.setSessionEnabled(command === "on");
          notify(ctx, formatStatus(active.snapshot()));
          return;
        }
        if (command === "health") {
          const healthy = await active.health();
          notify(ctx, healthy ? "Headroom health: online" : "Headroom health: offline", healthy ? "info" : "warning");
          return;
        }
        if (command === "status") {
          notify(ctx, formatStatus(active.snapshot(), true));
          return;
        }
        if (command === "stats") {
          notify(ctx, formatStats(active.snapshot()));
          return;
        }

        notify(ctx, "Usage: /headroom [status|on|off|health|stats]", "warning");
      } catch {
        notify(ctx, "Headroom command failed; context is unchanged.", "warning");
      }
    },
  });
}

export default function headroomExtension(pi: ExtensionAPI): void {
  registerHeadroomExtension(pi);
}
