import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  applyGatewayProviderBaseUrls,
  applyGatewayProviderBaseUrlsInPlace,
  resolveGatewayProviderIds,
} from "../src/gateway-config.js";
import {
  buildGeminiGenerateContentRequestUrl,
  matchesGeminiGenerateContentRoute,
} from "../src/proxy-routing.js";
import { __resetSessionIdCacheForTests } from "../src/session-headers.js";

// The session-id cache is a module-level singleton so the production
// gateway process keeps a stable session id per provider. Tests can run
// many cases against the same provider in one vitest run; reset the
// cache between cases so test expectations about "no header" stay clean.
afterEach(() => {
  __resetSessionIdCacheForTests();
});

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
      baseUrl: "http://127.0.0.1:8787/v1",
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
        baseUrl: "http://127.0.0.1:8787/v1",
        models: [],
      },
      openrouter: {
        baseUrl: "http://127.0.0.1:8787/v1",
        models: [],
      },
      google: {
        baseUrl: "http://127.0.0.1:8787/v1beta",
        models: [],
      },
      "minimax-portal": {
        baseUrl: "http://127.0.0.1:8787/v1",
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
      baseUrl: "http://127.0.0.1:8787/v1",
      models: [],
    });
  });

  it("is a no-op when the provider already points at headroom with normalized /v1", () => {
    const cfg = {
      models: {
        providers: {
          "openai-codex": {
            baseUrl: "http://127.0.0.1:8787/v1",
            models: [],
          },
        },
      },
    };

    const result = applyGatewayProviderBaseUrls(cfg, "http://127.0.0.1:8787", ["openai-codex"]);

    expect(result.changed).toBe(false);
    expect(result.config).toEqual(cfg);
  });

  it("normalizes an Anthropic-format upstream URL to /v1 on the proxy", () => {
    // The proxy only matches a small set of route prefixes — `/v1/messages`,
    // `/v1/chat/completions`, `/v1/responses`, `/anthropic/v1/messages`,
    // `/v1internal:streamGenerateContent`, `/v1/projects/.../publishers/...`.
    // Per-provider upstream targeting is handled separately via the
    // `x-headroom-base-url` header so collapsing the proxy pathname to
    // `/v1` loses no information.
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

  it("normalizes a GitHub Copilot OpenAI-family upstream URL to /v1 on the proxy", () => {
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

  it("normalizes a GitHub Copilot Claude-family upstream URL to /v1 on the proxy", () => {
    // The proxy routes /v1/messages to the Anthropic upstream (the
    // ANTHROPIC_TARGET_API_URL). Collapsing `/anthropic` -> `/v1` is
    // safe because the proxy handles both shapes for Anthropic.
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
      baseUrl: "http://127.0.0.1:8787/v1",
      models: [],
    });
  });

  it("normalizes a third-party OpenAI-compatible /api/v1 upstream URL to /v1 on the proxy", () => {
    // The proxy's routing table does not include `/api/v1/...`. Without
    // path normalization, providers like OpenRouter at `/api/v1` would
    // get rewritten to `http://127.0.0.1:8787/api/v1` and the proxy
    // would 404 the request. Per-provider upstream targeting is handled
    // via `x-headroom-base-url` so we lose nothing by collapsing here.
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
      baseUrl: "http://127.0.0.1:8787/v1",
      models: [],
    });
  });

  it("preserves Gemini /v1beta upstream routing on the proxy", () => {
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
    const googleBaseUrl = (result.config as any).models.providers.google.baseUrl;
    expect(googleBaseUrl).toBe("http://127.0.0.1:8787/v1beta");

    const requestUrl = buildGeminiGenerateContentRequestUrl(googleBaseUrl);
    expect(matchesGeminiGenerateContentRoute(requestUrl)).toBe(true);
  });

  it("routes google without an explicit upstream baseUrl to /v1beta", () => {
    const result = applyGatewayProviderBaseUrls({}, "http://127.0.0.1:8787", ["google"]);

    expect(result.changed).toBe(true);
    const googleBaseUrl = (result.config as any).models.providers.google.baseUrl;
    expect(googleBaseUrl).toBe("http://127.0.0.1:8787/v1beta");
    expect(
      matchesGeminiGenerateContentRoute(
        buildGeminiGenerateContentRequestUrl(googleBaseUrl),
      ),
    ).toBe(true);
  });

  it("routes OpenAI-compatible providers through /v1 chat/completions", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            openai: {
              baseUrl: "https://api.openai.com/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["openai"],
    );

    const openaiBaseUrl = (result.config as any).models.providers.openai.baseUrl;
    expect(openaiBaseUrl).toBe("http://127.0.0.1:8787/v1");
    expect(new URL(`${openaiBaseUrl}/chat/completions`).pathname).toBe(
      "/v1/chat/completions",
    );
  });

  it("does not invent a GitHub Copilot proxy baseUrl without an upstream baseUrl", () => {
    const result = applyGatewayProviderBaseUrls({}, "http://127.0.0.1:8787", ["github-copilot"]);

    expect(result.changed).toBe(false);
    expect((result.config as any).models?.providers?.["github-copilot"]).toBeUndefined();
  });

  it("routes anthropic without an explicit baseUrl to the bare proxy /v1 origin", () => {
    const result = applyGatewayProviderBaseUrls({}, "http://127.0.0.1:8787", ["anthropic"]);

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers.anthropic).toEqual({
      baseUrl: "http://127.0.0.1:8787/v1",
      models: [],
    });
  });

  it("routes multiple /v1-rooted providers to the same proxy /v1 path (per-provider upstream targeting via x-headroom-base-url)", () => {
    // After path normalization, OpenAI and GitHub Copilot both end up at
    // `/v1` on the proxy. They are distinguished by the
    // `x-headroom-base-url` header (injected separately, see Patch 2).
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

  it("re-points an already routed provider to a new proxy origin", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "openai-codex": {
              baseUrl: "http://127.0.0.1:8787/v1",
            },
          },
        },
      },
      "http://localhost:8787",
      ["openai-codex"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["openai-codex"]).toEqual({
      baseUrl: "http://localhost:8787/v1",
      models: [],
    });
  });

  it("preserves the upstream query string on the rewritten proxy URL", () => {
    // Model catalog hints and similar upstream query params should survive
    // the rewrite so the proxy sees them on the inbound request.
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            anthropic: {
              baseUrl: "https://api.anthropic.com/v1?beta=1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["anthropic"],
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers.anthropic.baseUrl).toBe(
      "http://127.0.0.1:8787/v1?beta=1",
    );
  });
});

describe("applyGatewayProviderBaseUrls with providerUpstreams", () => {
  it("injects an x-headroom-base-url header per provider from the overrides map", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "minimax-portal": {
              api: "anthropic-messages",
              baseUrl: "https://api.minimax.io/anthropic/v1",
            },
            openrouter: {
              api: "openai-completions",
              baseUrl: "https://openrouter.ai/api/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["minimax-portal", "openrouter"],
      {
        providerUpstreams: {
          "minimax-portal": "https://api.minimax.io/anthropic",
          openrouter: "https://openrouter.ai/api",
        },
      },
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers["minimax-portal"]).toEqual({
      api: "anthropic-messages",
      baseUrl: "http://127.0.0.1:8787/v1",
      headers: { "x-headroom-base-url": "https://api.minimax.io/anthropic" },
      models: [],
    });
    expect((result.config as any).models.providers.openrouter).toEqual({
      api: "openai-completions",
      baseUrl: "http://127.0.0.1:8787/v1",
      headers: { "x-headroom-base-url": "https://openrouter.ai/api" },
      models: [],
    });
  });

  it("does not overwrite unrelated existing provider headers", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            openrouter: {
              api: "openai-completions",
              baseUrl: "https://openrouter.ai/api/v1",
              headers: {
                "x-stainless-arch": "x64",
              },
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["openrouter"],
      {
        providerUpstreams: {
          openrouter: "https://openrouter.ai/api",
        },
      },
    );

    expect(result.changed).toBe(true);
    expect((result.config as any).models.providers.openrouter.headers).toEqual({
      "x-stainless-arch": "x64",
      "x-headroom-base-url": "https://openrouter.ai/api",
    });
  });

  it("does not mutate the result when providerUpstreams has no entry for a provider", () => {
    const cfg = {
      models: {
        providers: {
          anthropic: {
            api: "anthropic-messages",
            baseUrl: "https://api.anthropic.com/v1",
            headers: {
              "anthropic-version": "2023-06-01",
            },
          },
        },
      },
    };

    const result = applyGatewayProviderBaseUrls(cfg, "http://127.0.0.1:8787", ["anthropic"], {
      providerUpstreams: {
        "openrouter": "https://openrouter.ai/api",
      },
    });

    expect(result.changed).toBe(true);
    // baseUrl rewritten, no x-headroom-base-url added because the
    // operator did not list anthropic in providerUpstreams.
    expect((result.config as any).models.providers.anthropic.headers).toEqual({
      "anthropic-version": "2023-06-01",
    });
  });
});

describe("applyGatewayProviderBaseUrls with providerSessionHeaders", () => {
  it("injects a session id under the operator-chosen header name", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "opencode-go": {
              api: "openai-completions",
              baseUrl: "https://opencode.ai/zen/go/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["opencode-go"],
      {
        providerSessionHeaders: {
          "opencode-go": "x-opencode-session",
        },
      },
    );

    expect(result.changed).toBe(true);
    const headers = (result.config as any).models.providers["opencode-go"].headers;
    expect(headers["x-opencode-session"]).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });

  it("uses a stable session id across multiple rewrites in the same process", () => {
    // The session id is generated once per gateway process per provider
    // so that repeated calls land in the same upstream-side bucket.
    const result1 = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "opencode-go": {
              api: "openai-completions",
              baseUrl: "https://opencode.ai/zen/go/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["opencode-go"],
      {
        providerSessionHeaders: { "opencode-go": "x-opencode-session" },
      },
    );
    const session1 = (result1.config as any).models.providers["opencode-go"].headers[
      "x-opencode-session"
    ];

    const result2 = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "opencode-go": {
              api: "openai-completions",
              baseUrl: "https://opencode.ai/zen/go/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["opencode-go"],
      {
        providerSessionHeaders: { "opencode-go": "x-opencode-session" },
      },
    );
    const session2 = (result2.config as any).models.providers["opencode-go"].headers[
      "x-opencode-session"
    ];

    expect(session1).toBe(session2);
  });

  it("injects both x-headroom-base-url and the session header when both are configured", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            "opencode-go": {
              api: "openai-completions",
              baseUrl: "https://opencode.ai/zen/go/v1",
            },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["opencode-go"],
      {
        providerUpstreams: {
          "opencode-go": "https://opencode.ai/zen/go",
        },
        providerSessionHeaders: {
          "opencode-go": "x-opencode-session",
        },
      },
    );

    expect(result.changed).toBe(true);
    const headers = (result.config as any).models.providers["opencode-go"].headers;
    expect(headers["x-headroom-base-url"]).toBe("https://opencode.ai/zen/go");
    expect(headers["x-opencode-session"]).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
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
      baseUrl: "http://127.0.0.1:8787/v1",
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
      baseUrl: "http://127.0.0.1:8787/v1",
      models: ["gpt-5.3-codex"],
    });
  });
});
