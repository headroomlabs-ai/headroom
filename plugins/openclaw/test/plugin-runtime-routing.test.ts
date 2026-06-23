import { afterEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
  ensureProxyUrl: vi.fn(async () => "http://127.0.0.1:8787"),
  ensureProxyStarted: vi.fn(),
  getProxyUrl: vi.fn(() => null as string | null),
  createHeadroomRetrieveTool: vi.fn(({ proxyUrl }: { proxyUrl: string }) => ({ proxyUrl })),
}));

const proxyReadyListeners: Array<(proxyUrl: string) => void | Promise<void>> = [];

vi.mock("../src/engine.js", () => ({
  HeadroomContextEngine: class {
    ensureProxyUrl = mocked.ensureProxyUrl;
    ensureProxyStarted = mocked.ensureProxyStarted;
    getProxyUrl = mocked.getProxyUrl;
    onProxyReady(listener: (proxyUrl: string) => void | Promise<void>) {
      proxyReadyListeners.push(listener);
      return () => {};
    }
  },
}));

vi.mock("../src/tools/headroom-retrieve.js", () => ({
  createHeadroomRetrieveTool: mocked.createHeadroomRetrieveTool,
}));

import headroomPlugin from "../src/plugin/index.js";

afterEach(() => {
  mocked.ensureProxyUrl.mockClear();
  mocked.ensureProxyStarted.mockClear();
  mocked.getProxyUrl.mockClear();
  mocked.createHeadroomRetrieveTool.mockClear();
  proxyReadyListeners.length = 0;
});

describe("headroomPlugin runtime routing", () => {
  it("declares and registers the retrieval tool by name", async () => {
    const api: any = {
      config: {
        plugins: {
          entries: {
            headroom: {
              config: {
                routeCodexViaProxy: false,
                autoStart: false,
              },
            },
          },
        },
      },
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      },
      registerContextEngine: vi.fn(),
      registerTool: vi.fn(),
      on: vi.fn(),
    };

    headroomPlugin(api);
    await Promise.resolve();

    expect(api.registerTool).toHaveBeenCalledWith(expect.any(Function), {
      name: "headroom_retrieve",
    });
    expect(mocked.ensureProxyStarted).not.toHaveBeenCalled();
  });

  it("does not start the proxy when gateway routing is disabled", async () => {
    const gatewayHandlers = new Map<string, () => Promise<void>>();
    const api: any = {
      config: {
        plugins: {
          entries: {
            headroom: {
              config: {
                routeCodexViaProxy: false,
                autoStart: false,
              },
            },
          },
        },
      },
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      },
      registerContextEngine: vi.fn(),
      registerTool: vi.fn(),
      on: vi.fn((event: string, handler: () => Promise<void>) => {
        gatewayHandlers.set(event, handler);
      }),
    };

    headroomPlugin(api);
    await Promise.resolve();
    await gatewayHandlers.get("gateway_start")?.();

    expect(mocked.ensureProxyStarted).not.toHaveBeenCalled();
  });

  it("routes configured providers in memory once the proxy becomes available", async () => {
    const gatewayHandlers = new Map<string, () => Promise<void>>();
    const writeConfigFile = vi.fn();
    const loadConfig = vi.fn(() => ({
      models: {
        providers: {
          anthropic: {
            api: "anthropic-messages",
          },
        },
      },
    }));

    const api: any = {
      config: {
        plugins: {
          entries: {
            headroom: {
              config: {
                gatewayProviderIds: ["codex", "claude", "copilot", "gemini", "openrouter"],
              },
            },
          },
        },
        models: {
          providers: {
            anthropic: {
              api: "anthropic-messages",
              baseUrl: "https://api.anthropic.com",
            },
            "github-copilot": {
              baseUrl: "https://api.githubcopilot.com/v1",
            },
            google: {
              baseUrl: "https://generativelanguage.googleapis.com/v1beta",
            },
            openrouter: {
              baseUrl: "https://openrouter.ai/api/v1",
            },
          },
        },
      },
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      },
      registerContextEngine: vi.fn(),
      registerTool: vi.fn(),
      on: vi.fn((event: string, handler: () => Promise<void>) => {
        gatewayHandlers.set(event, handler);
      }),
      runtime: {
        config: {
          loadConfig,
          writeConfigFile,
        },
      },
    };

    headroomPlugin(api);
    await Promise.resolve();

    expect(mocked.ensureProxyUrl).not.toHaveBeenCalled();
    expect(mocked.ensureProxyStarted).not.toHaveBeenCalled();
    expect(writeConfigFile).not.toHaveBeenCalled();
    expect(loadConfig).not.toHaveBeenCalled();
    expect(api.config.models.providers["openai-codex"]).toBeUndefined();

    await proxyReadyListeners[0]?.("http://127.0.0.1:8787");

    expect(api.config.models.providers["openai-codex"]).toEqual({
      baseUrl: "http://127.0.0.1:8787/backend-api",
      models: [],
    });
    expect(api.config.models.providers.anthropic).toEqual({
      api: "anthropic-messages",
      baseUrl: "http://127.0.0.1:8787",
      models: [],
    });
    expect(api.config.models.providers["github-copilot"]).toEqual({
      baseUrl: "http://127.0.0.1:8787/v1",
      models: [],
    });
    expect(api.config.models.providers.google).toEqual({
      baseUrl: "http://127.0.0.1:8787/v1beta",
      models: [],
    });
    expect(api.config.models.providers.openrouter).toEqual({
      baseUrl: "http://127.0.0.1:8787/api/v1",
      models: [],
    });

    const gatewayStart = gatewayHandlers.get("gateway_start");
    expect(gatewayStart).toBeTypeOf("function");
    await gatewayStart?.();
    expect(mocked.ensureProxyStarted).toHaveBeenCalledTimes(1);
    expect(writeConfigFile).not.toHaveBeenCalled();
    expect(loadConfig).not.toHaveBeenCalled();
    expect(mocked.ensureProxyUrl).not.toHaveBeenCalled();
  });
});
