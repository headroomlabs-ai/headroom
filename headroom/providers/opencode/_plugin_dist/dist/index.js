import {
  HeadroomPlugin,
  compressWithHeadroom,
  createHeadroomRetrieveTool,
  getDefaultProxyUrl,
  installHeadroomTransport,
  setDefaultProxyUrl,
  stripTrailingSlashes
} from "./chunk-O36TDGAF.js";

// src/provider.ts
var DEFAULT_MODELS = {
  "claude-sonnet-4-6": {
    name: "Claude Sonnet 4.6",
    limit: { context: 2e5, output: 16384 }
  },
  "claude-opus-4-6": {
    name: "Claude Opus 4.6",
    limit: { context: 2e5, output: 16384 }
  },
  "claude-haiku-4-5-20251001": {
    name: "Claude Haiku 4.5",
    limit: { context: 2e5, output: 8192 }
  },
  "gpt-4o": {
    name: "GPT-4o",
    limit: { context: 128e3, output: 16384 }
  },
  "gpt-4.1": {
    name: "GPT-4.1",
    limit: { context: 1048576, output: 32768 }
  }
};
var DEFAULT_MODEL = "claude-sonnet-4-6";
function resolveBaseUrl(options) {
  if (options.proxyBaseUrl) return stripTrailingSlashes(options.proxyBaseUrl);
  const port = options.proxyPort ?? 8787;
  return `http://127.0.0.1:${port}`;
}
function createHeadroomProvider(options = {}) {
  const baseUrl = resolveBaseUrl(options);
  const models = options.models ?? DEFAULT_MODELS;
  return {
    npm: "@ai-sdk/openai-compatible",
    name: "Headroom Proxy",
    options: { baseURL: `${baseUrl}/v1` },
    // OpenCode namespaces model ids by provider key, so entries must be bare
    // ids ("claude-sonnet-4-6"), referenced as "headroom/<id>".
    models: { ...models }
  };
}
function buildOpencodeConfigContent(options = {}) {
  const defaultModel = options.defaultModel ?? DEFAULT_MODEL;
  const provider = createHeadroomProvider(options);
  return {
    provider: { headroom: provider },
    model: `headroom/${defaultModel}`
  };
}
function buildOpencodeConfigContentJson(options = {}) {
  return JSON.stringify(buildOpencodeConfigContent(options));
}

// src/index.ts
var src_default = {
  id: "headroom-opencode",
  server: HeadroomPlugin
};
export {
  DEFAULT_MODEL,
  DEFAULT_MODELS,
  buildOpencodeConfigContent,
  buildOpencodeConfigContentJson,
  compressWithHeadroom,
  createHeadroomProvider,
  createHeadroomRetrieveTool,
  src_default as default,
  getDefaultProxyUrl,
  installHeadroomTransport,
  setDefaultProxyUrl
};
//# sourceMappingURL=index.js.map