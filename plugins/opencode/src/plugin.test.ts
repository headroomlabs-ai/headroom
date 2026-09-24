import { afterEach, describe, expect, it, vi } from "vitest";

import HeadroomOpenCodePlugin, { HeadroomPlugin, headroomSetup } from "./plugin.js";

const TRANSPORT_STATE = Symbol.for("headroom.opencode.transport");

function pluginInput() {
  return {
    client: {},
    project: { id: "project-1" },
    directory: "/repo",
    worktree: "/repo",
    experimental_workspace: {
      register: vi.fn(),
    },
    $: {},
  } as never;
}

interface FakeModel {
  id: string;
  providerID: string;
  package?: string;
  settings?: Record<string, unknown>;
  headers?: Record<string, string>;
}

interface FakeProvider {
  id: string;
  package: string;
  settings?: Record<string, unknown>;
}

// Minimal OpenCode 2.x plugin context: records what setup registers.
function pluginContextV2(
  options: Record<string, unknown>,
  catalog: { providers?: FakeProvider[]; models?: FakeModel[] } = {},
) {
  const tools: Array<Record<string, any>> = [];
  const shellHooks: Array<(input: { env: Record<string, string | undefined> }) => unknown> = [];
  const providers = catalog.providers ?? [];
  const models = catalog.models ?? [];
  const context = {
    options,
    location: {
      directory: "/repo",
      project: { id: "project-1", directory: "/repo", canonical: "/repo" },
    },
    model: {
      transform: vi.fn(async (callback: (editor: Record<string, any>) => void) => {
        callback({
          list: () => models,
          update: (providerID: string, modelID: string, update: (model: FakeModel) => void) => {
            const model = models.find((m) => m.providerID === providerID && m.id === modelID);
            if (model) update(model);
          },
          provider: {
            get: (providerID: string) => {
              const provider = providers.find((p) => p.id === providerID);
              return provider ? { provider, models: new Map() } : undefined;
            },
          },
        });
        return { dispose: async () => {} };
      }),
    },
    tool: {
      transform: vi.fn(async (callback: (editor: { add: (tool: Record<string, any>) => void }) => void) => {
        callback({ add: (tool) => tools.push(tool) });
        return { dispose: async () => {} };
      }),
    },
    shell: {
      hook: vi.fn(async (name: string, callback: (input: { env: Record<string, string | undefined> }) => unknown) => {
        expect(name).toBe("create.before");
        shellHooks.push(callback);
        return { dispose: async () => {} };
      }),
    },
  };
  return { context: context as never, tools, shellHooks };
}

function transportState(): { refs: number } | undefined {
  return (globalThis as Record<symbol, { refs: number } | undefined>)[TRANSPORT_STATE];
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("HeadroomPlugin", () => {
  it("adds only Headroom metadata to shell env", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787/",
      backend: "litellm",
    });
    const output = {
      env: {
        OPENAI_BASE_URL: "https://deepseek.example/v1",
        ANTHROPIC_BASE_URL: "https://anthropic.example",
      },
    };

    await plugin["shell.env"]?.({ cwd: "/repo" }, output);
    await plugin.dispose?.();

    expect(output.env).toMatchObject({
      HEADROOM_ACTIVE: "1",
      HEADROOM_PROXY_URL: "http://127.0.0.1:8787",
      HEADROOM_PROJECT: "project-1",
      HEADROOM_BACKEND: "litellm",
      OPENAI_BASE_URL: "https://deepseek.example/v1",
      ANTHROPIC_BASE_URL: "https://anthropic.example",
    });
  });

  it("exposes a headroom_retrieve tool backed by the proxy", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => "original content",
    }));
    vi.stubGlobal("fetch", fetchMock);

    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
    });
    const result = await plugin.tool?.headroom_retrieve.execute(
      { hash: "0123456789abcdef01234567" },
      {} as never,
    );
    await plugin.dispose?.();

    expect(result).toBe("original content");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8787/v1/retrieve/0123456789abcdef01234567",
      expect.any(Object),
    );
  });
});

describe("headroomSetup (OpenCode 2.x)", () => {
  it("registers headroom_retrieve as a direct tool backed by the proxy", async () => {
    const { context, tools } = pluginContextV2({ proxyUrl: "http://127.0.0.1:8787/" });
    const cleanup = await headroomSetup(context);

    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => "original content",
    }));
    vi.stubGlobal("fetch", fetchMock);

    expect(tools).toHaveLength(1);
    const [retrieve] = tools;
    expect(retrieve).toMatchObject({
      name: "headroom_retrieve",
      options: { codemode: false },
      input: { type: "object", required: ["hash"] },
    });
    const result = await retrieve.execute(
      { hash: "0123456789abcdef01234567" },
      { signal: new AbortController().signal, progress: async () => {} },
    );
    await cleanup?.();

    expect(result).toEqual({ content: "original content" });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8787/v1/retrieve/0123456789abcdef01234567",
      expect.any(Object),
    );
  });

  it("adds only Headroom metadata to shell env", async () => {
    const { context, shellHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787/",
      backend: "litellm",
    });
    const cleanup = await headroomSetup(context);
    const shell = {
      env: {
        OPENAI_BASE_URL: "https://deepseek.example/v1",
        ANTHROPIC_BASE_URL: "https://anthropic.example",
      } as Record<string, string | undefined>,
    };

    expect(shellHooks).toHaveLength(1);
    await shellHooks[0](shell);
    await cleanup?.();

    expect(shell.env).toMatchObject({
      HEADROOM_ACTIVE: "1",
      HEADROOM_PROXY_URL: "http://127.0.0.1:8787",
      HEADROOM_PROJECT: "project-1",
      HEADROOM_BACKEND: "litellm",
      OPENAI_BASE_URL: "https://deepseek.example/v1",
      ANTHROPIC_BASE_URL: "https://anthropic.example",
    });
  });

  it("points remote OpenAI-wire models at the proxy and names the real upstream", async () => {
    const models: FakeModel[] = [
      { id: "deepseek-v4.1-flash", providerID: "opencode-go" },
      { id: "gpt-6-luna", providerID: "gateway", package: "@opencode/ai/providers/openai/responses" },
    ];
    const { context } = pluginContextV2(
      { proxyUrl: "http://127.0.0.1:8787/" },
      {
        providers: [
          {
            id: "opencode-go",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "https://opencode.ai/zen/go/v1/" },
          },
          {
            id: "gateway",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "https://gateway.example/api/v1" },
          },
        ],
        models,
      },
    );

    const cleanup = await headroomSetup(context);
    await cleanup?.();

    expect(models[0]).toMatchObject({
      settings: { baseURL: "http://127.0.0.1:8787/v1" },
      headers: {
        "x-headroom-base-url": "https://opencode.ai",
        "x-headroom-original-path": "/zen/go/v1/chat/completions",
        "x-headroom-project": "project-1",
      },
    });
    expect(models[1]).toMatchObject({
      settings: { baseURL: "http://127.0.0.1:8787/v1" },
      headers: {
        "x-headroom-base-url": "https://gateway.example",
        "x-headroom-original-path": "/api/v1/responses",
      },
    });
  });

  it("prefers a model's own baseURL over its provider's", async () => {
    const models: FakeModel[] = [
      {
        id: "custom",
        providerID: "opencode-go",
        settings: { baseURL: "https://model.example/v1", temperature: 0 },
        headers: { "x-user": "kept" },
      },
    ];
    const { context } = pluginContextV2(
      { proxyUrl: "http://127.0.0.1:8787" },
      {
        providers: [
          {
            id: "opencode-go",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "https://opencode.ai/zen/go/v1" },
          },
        ],
        models,
      },
    );

    const cleanup = await headroomSetup(context);
    await cleanup?.();

    expect(models[0]).toEqual({
      id: "custom",
      providerID: "opencode-go",
      settings: { baseURL: "http://127.0.0.1:8787/v1", temperature: 0 },
      headers: {
        "x-user": "kept",
        "x-headroom-base-url": "https://model.example",
        "x-headroom-original-path": "/v1/chat/completions",
        "x-headroom-project": "project-1",
      },
    });
  });

  it("leaves local, already-proxied, non-OpenAI-wire and URL-less models alone", async () => {
    const models: FakeModel[] = [
      { id: "llama", providerID: "ollama" },
      { id: "gpt-4o", providerID: "openai" },
      { id: "claude", providerID: "anthropic-gateway" },
      { id: "glm", providerID: "zai-coding-plan", package: "@opencode/ai/providers/zai-coding-plan/messages" },
      { id: "sonnet", providerID: "anthropic" },
      { id: "gemini", providerID: "google" },
    ];
    const snapshot = structuredClone(models);
    const { context } = pluginContextV2(
      { proxyUrl: "http://127.0.0.1:8787" },
      {
        providers: [
          {
            id: "ollama",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "http://127.0.0.1:11434/v1" },
          },
          {
            id: "openai",
            package: "@opencode/ai/providers/openai/responses",
            settings: { baseURL: "http://127.0.0.1:8787/v1" },
          },
          {
            id: "anthropic-gateway",
            package: "@opencode/ai/providers/anthropic",
            settings: { baseURL: "https://gateway.example/v1" },
          },
          {
            id: "zai-coding-plan",
            package: "@opencode/ai/providers/zai-coding-plan/chat",
            settings: { baseURL: "https://api.z.ai/api/coding/paas/v4" },
          },
          { id: "anthropic", package: "@opencode/ai/providers/anthropic" },
          {
            id: "google",
            package: "@ai-sdk/google",
            settings: { baseURL: "https://generativelanguage.googleapis.com/v1beta" },
          },
        ],
        models,
      },
    );

    const cleanup = await headroomSetup(context);
    await cleanup?.();

    expect(models).toEqual(snapshot);
  });

  it("releases the transport on cleanup so reloads do not leak it", async () => {
    const { context } = pluginContextV2({ proxyUrl: "http://127.0.0.1:8787" });

    const first = await headroomSetup(context);
    expect(transportState()?.refs).toBe(1);
    await first?.();
    expect(transportState()).toBeUndefined();

    const second = await headroomSetup(context);
    expect(transportState()?.refs).toBe(1);
    await second?.();
    expect(transportState()).toBeUndefined();
  });

  it("releases the transport when hook registration fails", async () => {
    const { context } = pluginContextV2({ proxyUrl: "http://127.0.0.1:8787" });
    (context as { shell: { hook: () => Promise<never> } }).shell.hook = async () => {
      throw new Error("registration failed");
    };

    await expect(headroomSetup(context)).rejects.toThrow("registration failed");
    expect(transportState()).toBeUndefined();
  });
});

describe("default export", () => {
  it("carries both the OpenCode 1.x and 2.x entry points", () => {
    expect(HeadroomOpenCodePlugin).toEqual({
      id: "headroom",
      server: HeadroomPlugin,
      setup: headroomSetup,
    });
  });
});
