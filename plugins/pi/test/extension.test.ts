import type {
  ExtensionAPI,
  ExtensionCommandContext,
  ExtensionContext,
  ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { describe, expect, it, vi } from "vitest";

import {
  registerHeadroomExtension,
  type RuntimeFactory,
} from "../src/index.js";
import { HeadroomRuntime } from "../src/runtime.js";

interface RegisteredCommand {
  description?: string;
  handler: (args: string, ctx: ExtensionCommandContext) => Promise<void>;
}

type EventHandler = (
  event: { type: string; messages?: unknown[] },
  ctx: ExtensionContext,
) => unknown;

function harness() {
  const handlers = new Map<string, EventHandler>();
  const commands = new Map<string, RegisteredCommand>();
  let tool: ToolDefinition | undefined;
  const api = {
    on: (name: string, handler: EventHandler) => handlers.set(name, handler),
    registerTool: (definition: ToolDefinition) => {
      tool = definition;
    },
    registerCommand: (name: string, command: RegisteredCommand) => {
      commands.set(name, command);
    },
  } as unknown as ExtensionAPI;
  return { api, handlers, commands, get tool() { return tool; } };
}

function context(): ExtensionContext {
  return {
    mode: "tui",
    hasUI: true,
    ui: { notify: vi.fn(), setStatus: vi.fn() },
  } as unknown as ExtensionContext;
}

describe("registerHeadroomExtension", () => {
  it("registers lifecycle hooks, one retrieval tool, and one command", () => {
    const test = harness();
    const runtimeFactory = vi.fn<RuntimeFactory>();

    registerHeadroomExtension(test.api, runtimeFactory);

    expect([...test.handlers.keys()]).toEqual([
      "session_start",
      "tool_result",
      "context",
      "session_shutdown",
    ]);
    expect(test.tool?.name).toBe("headroom_retrieve");
    expect([...test.commands.keys()]).toEqual(["headroom"]);
  });

  it("does not start a runtime after its session already shut down", async () => {
    const test = harness();
    const fakeRuntime = {
      start: vi.fn(),
      stop: vi.fn(),
    } as unknown as HeadroomRuntime;
    let resolveRuntime!: (runtime: HeadroomRuntime) => void;
    const pendingRuntime = new Promise<HeadroomRuntime>((resolve) => {
      resolveRuntime = resolve;
    });
    const runtimeFactory: RuntimeFactory = () => pendingRuntime;
    registerHeadroomExtension(test.api, runtimeFactory);
    const ctx = context();

    const starting = test.handlers
      .get("session_start")
      ?.({ type: "session_start" }, ctx);
    test.handlers
      .get("session_shutdown")
      ?.({ type: "session_shutdown" }, ctx);
    resolveRuntime(fakeRuntime);
    await Promise.resolve(starting);

    expect(fakeRuntime.start).not.toHaveBeenCalled();
    expect(fakeRuntime.stop).toHaveBeenCalledTimes(1);
  });

  it("catches host callback failures and returns fail-open results", async () => {
    const test = harness();
    const fakeRuntime = {
      start: vi.fn(() => {
        throw new Error("start failure");
      }),
      observeToolResult: vi.fn(() => {
        throw new Error("observation failure");
      }),
      transform: vi.fn(() => {
        throw new Error("transform failure");
      }),
      retrieve: vi.fn(async () => {
        throw new Error("retrieval failure");
      }),
      setSessionEnabled: vi.fn(() => {
        throw new Error("toggle failure");
      }),
      health: vi.fn(async () => {
        throw new Error("health failure");
      }),
      snapshot: vi.fn(() => {
        throw new Error("snapshot failure");
      }),
      stop: vi.fn(() => {
        throw new Error("stop failure");
      }),
    } as unknown as HeadroomRuntime;
    const runtimeFactory: RuntimeFactory = async () => fakeRuntime;
    registerHeadroomExtension(test.api, runtimeFactory);
    const ctx = context();

    await expect(
      test.handlers.get("session_start")?.({ type: "session_start" }, ctx),
    ).resolves.toBeUndefined();
    expect(() =>
      test.handlers.get("tool_result")?.({ type: "tool_result" }, ctx),
    ).not.toThrow();
    expect(
      test.handlers.get("context")?.({ type: "context", messages: [] }, ctx),
    ).toBeUndefined();
    expect(() =>
      test.handlers.get("session_shutdown")?.({ type: "session_shutdown" }, ctx),
    ).not.toThrow();

    const toolResult = await test.tool?.execute(
      "call",
      { hash: "abc" },
      undefined,
      undefined,
      ctx,
    );
    expect(toolResult?.content).toEqual([
      {
        type: "text",
        text: "Headroom retrieval miss for abc. Rerun the originating tool.",
      },
    ]);

    await expect(
      test.commands.get("headroom")?.handler(
        "status",
        ctx as ExtensionCommandContext,
      ),
    ).resolves.toBeUndefined();
  });
});
