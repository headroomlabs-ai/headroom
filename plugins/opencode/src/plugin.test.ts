import { mkdtemp, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import pluginModule from "./index.js";
import { HeadroomPlugin, applyNativeProviderOverrides } from "./plugin.js";

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

afterEach(() => {
  vi.restoreAllMocks();
  delete process.env.HEADROOM_WORKSPACE_DIR;
});

describe("HeadroomPlugin", () => {
  it("exports the OpenCode v1 plugin contract", () => {
    expect(pluginModule).toMatchObject({
      id: "headroom-opencode",
      server: HeadroomPlugin,
    });
  });

  it("injects native provider baseURLs for anthropic, openai, and github-copilot", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      mode: "native-fetch",
    });
    const config = {
      provider: {
        openai: { options: { apiKey: "test-openai" } },
        anthropic: { options: { apiKey: "test-anthropic" } },
        "github-copilot": { options: { fetch: vi.fn() } },
      },
    };

    await plugin.config?.(config as never);

    expect(config).toMatchObject({
      provider: {
        openai: { options: { apiKey: "test-openai", baseURL: "http://127.0.0.1:8787/v1" } },
        anthropic: { options: { apiKey: "test-anthropic", baseURL: "http://127.0.0.1:8787/v1" } },
        "github-copilot": { options: { fetch: expect.any(Function), baseURL: "http://127.0.0.1:8787/v1" } },
      },
    });
    expect((config.provider as Record<string, unknown>).headroom).toBeUndefined();
  });

  it("creates provider stubs when native overrides run against empty config", () => {
    const config: Record<string, unknown> = {};

    applyNativeProviderOverrides(config, "http://127.0.0.1:8787");

    expect(config).toEqual({
      provider: {
        anthropic: { options: { baseURL: "http://127.0.0.1:8787/v1" } },
        "github-copilot": { options: { baseURL: "http://127.0.0.1:8787/v1" } },
        openai: { options: { baseURL: "http://127.0.0.1:8787/v1" } },
        opencode: { options: { baseURL: "http://127.0.0.1:8787/v1" } },
        "opencode-go": {
          options: {
            baseURL: "http://127.0.0.1:8787/v1",
            headers: {
              "x-headroom-base-url": "https://opencode.ai/zen/go",
            },
          },
        },
        zenmux: {
          options: {
            baseURL: "http://127.0.0.1:8787/v1",
            headers: {
              "x-headroom-base-url": "https://zenmux.ai/api",
            },
          },
        },
      },
    });
  });

  it("does not create native baseURL overrides for opencode family and preserves zenmux upstream", () => {
    const config = {
      provider: {
        opencode: { options: { apiKey: "test-opencode", baseURL: "https://console.opencode.ai/api/v1" } },
        "opencode-go": { options: { apiKey: "test-opencode-go", baseURL: "https://console.opencode.ai/api/v1" } },
        zenmux: {
          options: {
            apiKey: "test-zenmux",
            baseURL: "https://zenmux.ai/api/v1",
            headers: { "X-Title": "opencode" },
          },
        },
      },
    };

    applyNativeProviderOverrides(config as unknown as Record<string, unknown>, "http://127.0.0.1:8787");

    expect(config).toEqual({
      provider: {
        opencode: {
          options: {
            apiKey: "test-opencode",
            baseURL: "http://127.0.0.1:8787/v1",
            headers: {
              "x-headroom-base-url": "https://console.opencode.ai/api",
            },
          },
        },
        "opencode-go": {
          options: {
            apiKey: "test-opencode-go",
            baseURL: "http://127.0.0.1:8787/v1",
            headers: {
              "x-headroom-base-url": "https://opencode.ai/zen/go",
            },
          },
        },
        zenmux: {
          options: {
            apiKey: "test-zenmux",
            baseURL: "http://127.0.0.1:8787/v1",
            headers: {
              "X-Title": "opencode",
              "x-headroom-base-url": "https://zenmux.ai/api",
            },
          },
        },
        anthropic: { options: { baseURL: "http://127.0.0.1:8787/v1" } },
        "github-copilot": { options: { baseURL: "http://127.0.0.1:8787/v1" } },
        openai: { options: { baseURL: "http://127.0.0.1:8787/v1" } },
      },
    });
  });

  it("preserves the upstream header for opencode non-chat requests", () => {
    const config = {
      provider: {
        opencode: {
          options: {
            baseURL: "https://console.opencode.ai/proxy/connections/fixture/v1",
          },
        },
      },
    };

    applyNativeProviderOverrides(config as unknown as Record<string, unknown>, "http://127.0.0.1:8787");

    expect(config).toMatchObject({
      provider: {
        opencode: {
          options: {
            baseURL: "http://127.0.0.1:8787/v1",
            headers: {
              "x-headroom-base-url": "https://console.opencode.ai/proxy/connections/fixture",
            },
          },
        },
      },
    });
  });

  it("adds runtime upstream header for opencode models", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      mode: "native-fetch",
    });
    const output = { headers: {} as Record<string, string> };

    await plugin["chat.headers"]?.(
      {
        sessionID: "session-1",
        agent: "build",
        model: {
          providerID: "opencode",
          id: "gpt-5.2-codex",
          api: {
            id: "gpt-5.2-codex",
            npm: "@ai-sdk/openai-compatible",
            url: "https://console.opencode.ai/proxy/connections/fixture/v1",
          },
        },
        provider: { id: "opencode", source: "custom", info: { id: "opencode" }, options: {} },
        message: {},
      } as never,
      output,
    );

    expect(output.headers).toEqual({
      "x-headroom-base-url": "https://console.opencode.ai/proxy/connections/fixture",
    });
  });

  it("adds runtime upstream header for opencode-go models", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      mode: "native-fetch",
    });
    const output = { headers: {} as Record<string, string> };

    await plugin["chat.headers"]?.(
      {
        sessionID: "session-1",
        agent: "build",
        model: {
          providerID: "opencode-go",
          id: "glm-5.2",
          api: {
            id: "glm-5.2",
            npm: "@ai-sdk/openai-compatible",
            url: "https://opencode.ai/zen/go/v1",
          },
        },
        provider: { id: "opencode-go", source: "custom", info: { id: "opencode-go" }, options: {} },
        message: {},
      } as never,
      output,
    );

    expect(output.headers).toEqual({
      "x-headroom-base-url": "https://opencode.ai/zen/go",
    });
  });

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
      { hash: "0123456789abcdef01234567", query: "needle" },
      {} as never,
    );

    expect(result).toBe("original content");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8787/v1/retrieve/0123456789abcdef01234567?query=needle",
      expect.any(Object),
    );
  });

  it("registers and unregisters an OpenCode proxy client marker", async () => {
    const workspaceDir = await mkdtemp(path.join(os.tmpdir(), "headroom-opencode-"));
    process.env.HEADROOM_WORKSPACE_DIR = workspaceDir;

    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
    });

    const markerDir = path.join(workspaceDir, "clients", "8787");
    const activeMarkers = await readdir(markerDir);
    expect(activeMarkers).toHaveLength(1);
    expect(activeMarkers[0]).toMatch(/^opencode-\d+-\d+\.json$/);

    await plugin.dispose?.();

    await expect(readdir(markerDir)).resolves.toEqual([]);
  });
});
