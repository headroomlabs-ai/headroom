import { describe, expect, it } from "vitest";
import {
  applyGatewayProviderBaseUrls,
  applyGatewayProviderBaseUrlsInPlace,
  resolveGatewayProviderIds,
} from "../src/gateway-config.js";

describe("resolveGatewayProviderIds", () => {
  it("routes openai-codex by default", () => {
    expect(resolveGatewayProviderIds(undefined)).toEqual(["openai-codex"]);
  });

  it("allows an explicit provider list to override the default", () => {
    expect(
      resolveGatewayProviderIds({
        gatewayProviderIds: ["anthropic", "github-copilot", "minimax-portal"],
      }),
    ).toEqual(["anthropic", "github-copilot", "minimax-portal"]);
  });

  it("normalizes explicit provider ids and friendly aliases", () => {
    expect(
      resolveGatewayProviderIds({
        gatewayProviderIds: [" claude ", "", "copilot", "codex", "gemini", "anthropic"],
      }),
    ).toEqual(["anthropic", "github-copilot", "openai-codex", "google"]);
  });

  it("allows routing to be disabled", () => {
    expect(resolveGatewayProviderIds({ routeCodexViaProxy: false })).toEqual([]);
  });
});

describe("applyGatewayProviderBaseUrls", () => {
  it("creates an openai-codex provider config when missing", () => {
    const result = applyGatewayProviderBaseUrls({}, "http://127.0.0.1:8787", ["openai-codex"]);

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["openai-codex"]).toEqual({
      baseUrl: "http://127.0.0.1:8787/backend-api",
      models: [],
    });
  });

  it("creates provider configs for multiple configured provider ids", () => {
    const result = applyGatewayProviderBaseUrls(
      {},
      "http://127.0.0.1:8787",
      ["anthropic", "openrouter", "google", "minimax-portal"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers).toEqual({
      anthropic: {
        baseUrl: "http://127.0.0.1:8787",
        models: [],
      },
      openrouter: {
        baseUrl: "http://127.0.0.1:8787",
        models: [],
      },
      google: {
        baseUrl: "http://127.0.0.1:8787",
        models: [],
      },
      "minimax-portal": {
        baseUrl: "http://127.0.0.1:8787",
        models: [],
      },
    });
  });

  it("preserves existing provider config fields", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "openai-codex": {
              api: "openai-codex-responses",
              baseUrl: "https://chatgpt.com/backend-api",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["openai-codex"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["openai-codex"]).toEqual({
      api: "openai-codex-responses",
      baseUrl: "http://127.0.0.1:8787/backend-api",
      models: [],
    });
  });

  it("is a no-op when the provider already points at headroom", () => {
    const cfg = {
      models: {
        providers: {
          "openai-codex": {
            baseUrl: "http://127.0.0.1:8787/backend-api",
            models: [],
          },
        },
      },
    };

    const result = applyGatewayProviderBaseUrls(cfg, "http://127.0.0.1:8787", ["openai-codex"]);

    expect(result.changed).toBe(false);
    expect(result.config).toEqual(cfg);
  });

  it("preserves upstream path segments when routing through the proxy", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            anthropic: {
              baseUrl: "https://api.anthropic.com/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["anthropic"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers.anthropic).toEqual({
      baseUrl: "http://127.0.0.1:8787/v1",
      models: [],
    });
  });

  it("preserves protocol-specific GitHub Copilot OpenAI-family paths", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "github-copilot": {
              baseUrl: "https://api.githubcopilot.com/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["github-copilot"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["github-copilot"]).toEqual({
      baseUrl: "http://127.0.0.1:8787/v1",
      models: [],
    });
  });

  it("preserves protocol-specific GitHub Copilot Claude-family paths", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "github-copilot": {
              baseUrl: "https://api.githubcopilot.com/anthropic",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["github-copilot"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["github-copilot"]).toEqual({
      baseUrl: "http://127.0.0.1:8787/anthropic",
      models: [],
    });
  });

  it("preserves OpenAI-compatible /api/v1 paths", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            openrouter: {
              baseUrl: "https://openrouter.ai/api/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["openrouter"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers.openrouter).toEqual({
      baseUrl: "http://127.0.0.1:8787/api/v1",
      models: [],
    });
  });

  it("preserves Gemini /v1beta paths", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            google: {
              baseUrl: "https://generativelanguage.googleapis.com/v1beta",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["google"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers.google).toEqual({
      baseUrl: "http://127.0.0.1:8787/v1beta",
      models: [],
    });
  });

  it("does not invent a GitHub Copilot proxy baseUrl without an upstream baseUrl", () => {
    const result = applyGatewayProviderBaseUrls({}, "http://127.0.0.1:8787", ["github-copilot"]);

    expect(result.changed).toBe(false);
    expect((result.config as any).models?.providers?.["github-copilot"]).toBeUndefined();
  });

  it("documents the Gate-D risk: anthropic without an explicit baseUrl routes to the bare proxy origin", () => {
    const result = applyGatewayProviderBaseUrls({}, "http://127.0.0.1:8787", ["anthropic"]);

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers.anthropic).toEqual({
      baseUrl: "http://127.0.0.1:8787",
      models: [],
    });
  });

  it("documents the multi-provider risk: providers sharing /v1 collapse to the same proxy path", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            openai: {
              baseUrl: "https://api.openai.com/v1",
            },
            "github-copilot": {
              baseUrl: "https://api.githubcopilot.com/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["openai", "github-copilot"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers.openai.baseUrl).toBe(
      "http://127.0.0.1:8787/v1",
    );
    expect((result.config as any).models.providers["github-copilot"].baseUrl).toBe(
      "http://127.0.0.1:8787/v1",
    );
  });

  it("re-points an already routed provider to a new proxy origin without duplicating paths", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "openai-codex": {
              baseUrl: "http://127.0.0.1:8787/backend-api",
            },
          },
        },
      },
      "http://localhost:8787",
      ["openai-codex"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["openai-codex"]).toEqual({
      baseUrl: "http://localhost:8787/backend-api",
      models: [],
    });
  });
});

describe("applyGatewayProviderBaseUrlsInPlace", () => {
  it("updates the live config object in place", () => {
    const cfg: any = { models: { providers: {} } };

    const changed = applyGatewayProviderBaseUrlsInPlace(
      cfg,
      "http://127.0.0.1:8787",
      ["openai-codex"],
    );

    expect(changed).toBe(true);
    expect(cfg.models.providers["openai-codex"]).toEqual({
      baseUrl: "http://127.0.0.1:8787/backend-api",
      models: [],
    });
  });

  it("does not clobber existing provider logic when changing only the base URL", () => {
    const cfg: any = {
      models: {
        providers: {
          "openai-codex": {
            api: "openai-codex-responses",
            baseUrl: "https://chatgpt.com/backend-api",
            envKey: "OPENAI_API_KEY",
            models: ["gpt-5.3-codex"],
          },
        },
      },
    };

    const changed = applyGatewayProviderBaseUrlsInPlace(
      cfg,
      "http://127.0.0.1:8787",
      ["openai-codex"],
    );

    expect(changed).toBe(true);
    expect(cfg.models.providers["openai-codex"]).toEqual({
      api: "openai-codex-responses",
      envKey: "OPENAI_API_KEY",
      baseUrl: "http://127.0.0.1:8787/backend-api",
      models: ["gpt-5.3-codex"],
    });
  });
});
