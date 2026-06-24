/**
 * HeadroomClient — HTTP client for the Headroom compression proxy.
 *
 * Supports:
 * - compress() — direct compression via /v1/compress
 * - chat.completions.create() — OpenAI-style passthrough via /v1/chat/completions
 * - messages.create() — Anthropic-style passthrough via /v1/messages
 * - Metrics, health, CCR retrieve, telemetry, feedback, TOIN
 */

import type {
  OpenAIMessage,
  CompressResult,
  HeadroomClientOptions,
  HeadroomClientInterface,
  ProxyCompressResponse,
  ProxyErrorResponse,
} from "./types.js";
import { mapProxyError, HeadroomConnectionError, HeadroomAuthError, HeadroomCompressError } from "./errors.js";
import { deepCamelCase, deepSnakeCase } from "./utils/case.js";
import { parseSSE } from "./utils/stream.js";
import type { HeadroomConfig, HeadroomMode } from "./types/config.js";
import type {
  SimulationResult,
  RequestMetrics,
  MetricsSummary,
  SessionStats,
  ValidationResult,
  HealthStatus,
  ProxyStats,
  MemoryUsage,
  RetrieveResult,
  RetrieveSearchResult,
  CCRStats,
  TelemetryStats,
  ToolHints,
  TOINStats,
  TOINPattern,
  MetricsQuery,
  SummaryQuery,
  StatsHistoryQuery,
} from "./types/models.js";

const DEFAULT_BASE_URL = "http://localhost:8787";
const DEFAULT_TIMEOUT = 30_000;
const DEFAULT_RETRIES = 1;

function getEnv(key: string): string | undefined {
  if (typeof process !== "undefined" && process.env) {
    return process.env[key];
  }
  return undefined;
}

function makeFallbackResult(messages: OpenAIMessage[]): CompressResult {
  return {
    messages,
    tokensBefore: 0,
    tokensAfter: 0,
    tokensSaved: 0,
    compressionRatio: 1.0,
    transformsApplied: [],
    ccrHashes: [],
    compressed: false,
  };
}

// --- Headroom request params ---

export interface HeadroomParams {
  headroomMode?: HeadroomMode;
  headroomCachePrefixTokens?: number;
  headroomOutputBufferTokens?: number;
  headroomKeepTurns?: number;
  headroomToolProfiles?: Record<string, Record<string, any>>;
}

// --- Sub-clients ---

class ChatCompletions {
  constructor(private client: HeadroomClient) {}

  /**
   * Create a chat completion with automatic compression.
   * Routes through proxy's POST /v1/chat/completions.
   */
  async create(
    params: { model: string; messages: OpenAIMessage[]; stream?: boolean; [key: string]: any } & HeadroomParams,
  ): Promise<any> {
    const { headroomMode, headroomCachePrefixTokens, headroomOutputBufferTokens, headroomKeepTurns, headroomToolProfiles, ...apiParams } = params;

    const headers: Record<string, string> = {};
    if (headroomMode) headers["x-headroom-mode"] = headroomMode;

    const providerKey = this.client.providerApiKey ?? getEnv("OPENAI_API_KEY");
    if (providerKey) headers["Authorization"] = `Bearer ${providerKey}`;

    const response = await this.client.rawFetch("/v1/chat/completions", {
      method: "POST",
      headers,
      body: apiParams,
      stream: params.stream,
    });

    if (params.stream) {
      return parseSSE(response);
    }

    return response.json();
  }

  /**
   * Simulate compression without calling the LLM.
   */
  async simulate(
    params: { model: string; messages: OpenAIMessage[] } & HeadroomParams,
  ): Promise<SimulationResult> {
    const body = {
      messages: params.messages,
      model: params.model,
      config: { default_mode: "simulate", generate_diff_artifact: true },
    };
    const result = await this.client.compressRaw(body);
    return deepCamelCase<SimulationResult>(result);
  }
}

class Messages {
  constructor(private client: HeadroomClient) {}

  /**
   * Create a message with automatic compression.
   * Routes through proxy's POST /v1/messages (Anthropic).
   */
  async create(
    params: { model: string; messages: any[]; max_tokens?: number; system?: string | any[]; stream?: boolean; [key: string]: any } & HeadroomParams,
  ): Promise<any> {
    const { headroomMode, headroomCachePrefixTokens, headroomOutputBufferTokens, headroomKeepTurns, headroomToolProfiles, ...apiParams } = params;

    const headers: Record<string, string> = {
      "anthropic-version": "2023-06-01",
    };
    if (headroomMode) headers["x-headroom-mode"] = headroomMode;

    const providerKey = this.client.providerApiKey ?? getEnv("ANTHROPIC_API_KEY");
    if (providerKey) headers["x-api-key"] = providerKey;

    if (!apiParams.max_tokens) apiParams.max_tokens = 1024;

    const response = await this.client.rawFetch("/v1/messages", {
      method: "POST",
      headers,
      body: apiParams,
      stream: params.stream,
    });

    if (params.stream) {
      return parseSSE(response);
    }

    return response.json();
  }

  /**
   * Stream a message with automatic compression.
   */
  stream(
    params: { model: string; messages: any[]; max_tokens?: number; system?: string | any[]; [key: string]: any } & HeadroomParams,
  ): Promise<AsyncGenerator<any>> {
    return this.create({ ...params, stream: true }) as Promise<AsyncGenerator<any>>;
  }

  /**
   * Simulate compression without calling the LLM.
   */
  async simulate(
    params: { model: string; messages: any[] } & HeadroomParams,
  ): Promise<SimulationResult> {
    const body = {
      messages: params.messages,
      model: params.model,
      config: { default_mode: "simulate", generate_diff_artifact: true },
    };
    const result = await this.client.compressRaw(body);
    return deepCamelCase<SimulationResult>(result);
  }
}

// --- Main client ---

export interface ExtendedClientOptions extends HeadroomClientOptions {
  providerApiKey?: string;
  defaultMode?: HeadroomMode;
  config?: HeadroomConfig;
}

export class HeadroomClient implements HeadroomClientInterface {
  private baseUrl: string;
  private apiKey: string | undefined;
  private timeout: number;
  private fallback: boolean;
  private retries: number;
  private config: HeadroomConfig | undefined;
  private stack: string | undefined;

  /** @internal */ providerApiKey: string | undefined;

  /** OpenAI-style chat completions API. */
  readonly chat: { completions: ChatCompletions };
  /** Anthropic-style messages API. */
  readonly messages: Messages;

  constructor(options: ExtendedClientOptions = {}) {
    this.baseUrl = (
      options.baseUrl ??
      getEnv("HEADROOM_BASE_URL") ??
      DEFAULT_BASE_URL
    ).replace(/\/+$/, "");
    this.apiKey = options.apiKey ?? getEnv("HEADROOM_API_KEY");
    this.timeout = options.timeout ?? DEFAULT_TIMEOUT;
    this.fallback = options.fallback ?? true;
    this.retries = options.retries ?? DEFAULT_RETRIES;
    this.providerApiKey = options.providerApiKey;
    this.config = options.config;
    this.stack = options.stack;

    this.chat = { completions: new ChatCompletions(this) };
    this.messages = new Messages(this);
  }

  // ============================================================
  // Core: compress
  // ============================================================

  async compress(
    messages: OpenAIMessage[],
    options: { model?: string; tokenBudget?: number } = {},
  ): Promise<CompressResult> {
    const model = options.model ?? "gpt-4o";

    let lastError: unknown;
    const maxAttempts = 1 + this.retries;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        return await this._doCompress(messages, model, options.tokenBudget);
      } catch (error) {
        lastError = error;
        if (error instanceof HeadroomAuthError) throw error;
        if (
          error instanceof HeadroomCompressError &&
          error.statusCode < 500
        ) {
          throw error;
        }
      }
    }

    if (this.fallback) {
      return makeFallbackResult(messages);
    }
    if (lastError instanceof HeadroomConnectionError) throw lastError;
    if (lastError instanceof HeadroomCompressError) throw lastError;
    throw new HeadroomConnectionError(
      `Failed after ${maxAttempts} attempts: ${lastError}`,
    );
  }

  /**
   * Raw compress call — sends body directly to /v1/compress.
   * Used by simulate() and other advanced features.
   * @internal
   */
  async compressRaw(body: Record<string, any>): Promise<any> {
    const response = await this._fetch("/v1/compress", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return response.json();
  }

  // ============================================================
  // Health & Stats
  // ============================================================

  /** Check if the proxy is running and healthy. */
  async health(): Promise<HealthStatus> {
    const resp = await this._fetch("/health", { method: "GET" });
    return deepCamelCase<HealthStatus>(await resp.json());
  }

  /** Get comprehensive proxy statistics. */
  async proxyStats(): Promise<ProxyStats> {
    const resp = await this._fetch("/stats", { method: "GET" });
    return deepCamelCase<ProxyStats>(await resp.json());
  }

  /** Get Prometheus-format metrics. */
  async prometheusMetrics(): Promise<string> {
    const resp = await this._fetch("/metrics", { method: "GET" });
    return resp.text();
  }

  /** Get historical stats. */
  async statsHistory(query?: StatsHistoryQuery): Promise<any> {
    const params = new URLSearchParams();
    if (query?.format) params.set("format", query.format);
    if (query?.series) params.set("series", query.series);
    const qs = params.toString();
    const resp = await this._fetch(`/stats-history${qs ? `?${qs}` : ""}`, { method: "GET" });
    return resp.json();
  }

  /** Get proxy memory usage. */
  async memoryUsage(): Promise<MemoryUsage> {
    const resp = await this._fetch("/debug/memory", { method: "GET" });
    return deepCamelCase<MemoryUsage>(await resp.json());
  }

  /** Clear the response cache. */
  async clearCache(): Promise<{ status: string }> {
    const resp = await this._fetch("/cache/clear", { method: "POST" });
    return resp.json() as Promise<{ status: string }>;
  }

  // ============================================================
  // Metrics & Observability
  // ============================================================

  /** Get request metrics from the proxy. */
  async getMetrics(query?: MetricsQuery): Promise<RequestMetrics[]> {
    const resp = await this._fetch("/stats", { method: "GET" });
    const stats = (await resp.json()) as any;
    let metrics: any[] = stats.recent_requests ?? [];

    if (query?.model) {
      metrics = metrics.filter((m: any) => m.model === query.model);
    }
    if (query?.mode) {
      metrics = metrics.filter((m: any) => m.mode === query.mode);
    }
    if (query?.limit) {
      metrics = metrics.slice(0, query.limit);
    }

    return metrics.map((m: any) => deepCamelCase<RequestMetrics>(m));
  }

  /** Get aggregated metrics summary. */
  async getSummary(query?: SummaryQuery): Promise<MetricsSummary> {
    const resp = await this._fetch("/stats", { method: "GET" });
    const stats = (await resp.json()) as any;
    return deepCamelCase<MetricsSummary>({
      total_requests: stats.requests?.total ?? 0,
      total_tokens_before: stats.tokens?.total_before_compression ?? 0,
      total_tokens_after: (stats.tokens?.total_before_compression ?? 0) - (stats.tokens?.saved ?? 0),
      total_tokens_saved: stats.tokens?.saved ?? 0,
      average_compression_ratio: stats.tokens?.savings_percent ? stats.tokens.savings_percent / 100 : 0,
      models: stats.requests?.by_model ?? {},
      modes: {},
      error_count: stats.requests?.failed ?? 0,
    });
  }

  /** Get in-memory session stats. */
  async getStats(): Promise<SessionStats> {
    const resp = await this._fetch("/stats", { method: "GET" });
    const stats = (await resp.json()) as any;
    return deepCamelCase<SessionStats>({
      total_requests: stats.requests?.total ?? 0,
      total_tokens_before: stats.tokens?.total_before_compression ?? 0,
      total_tokens_after: (stats.tokens?.total_before_compression ?? 0) - (stats.tokens?.saved ?? 0),
      total_tokens_saved: stats.tokens?.saved ?? 0,
      average_compression_ratio: stats.tokens?.savings_percent ? stats.tokens.savings_percent / 100 : 0,
      cache_hits: stats.requests?.cached ?? 0,
      by_mode: {},
    });
  }

  /** Validate proxy configuration. */
  async validateSetup(): Promise<ValidationResult> {
    const resp = await this._fetch("/health", { method: "GET" });
    const health = (await resp.json()) as any;
    return {
      valid: health.status === "healthy",
      provider: "",
      errors: health.status !== "healthy" ? ["Proxy unhealthy"] : [],
      warnings: [],
      config: health.config ?? {},
    };
  }

  // ============================================================
  // CCR Retrieve
  // ============================================================

  /** Retrieve original content from the CCR compression store. */
  async retrieve(
    hash: string,
    options?: { query?: string },
  ): Promise<RetrieveResult | RetrieveSearchResult> {
    const body: Record<string, any> = { hash };
    if (options?.query) body.query = options.query;
    const resp = await this._fetch("/v1/retrieve", {
      method: "POST",
      body: JSON.stringify(body),
    });
    return deepCamelCase(await resp.json());
  }

  /** Get CCR store statistics. */
  async getCCRStats(): Promise<CCRStats> {
    const resp = await this._fetch("/v1/retrieve/stats", { method: "GET" });
    return deepCamelCase<CCRStats>(await resp.json());
  }

  /** Handle an LLM tool call for headroom_retrieve. */
  async handleToolCall(request: {
    toolCall: any;
    provider?: "anthropic" | "openai";
  }): Promise<any> {
    const resp = await this._fetch("/v1/retrieve/tool_call", {
      method: "POST",
      body: JSON.stringify(deepSnakeCase(request)),
    });
    return deepCamelCase(await resp.json());
  }

  // ============================================================
  // Telemetry & Feedback
  // ============================================================

  readonly telemetry = {
    getStats: async (): Promise<TelemetryStats> => {
      const resp = await this._fetch("/v1/telemetry", { method: "GET" });
      return deepCamelCase<TelemetryStats>(await resp.json());
    },
    export: async (): Promise<any> => {
      const resp = await this._fetch("/v1/telemetry/export", { method: "GET" });
      return resp.json();
    },
    import: async (data: any): Promise<{ status: string }> => {
      const resp = await this._fetch("/v1/telemetry/import", {
        method: "POST",
        body: JSON.stringify(data),
      });
      return resp.json() as Promise<{ status: string }>;
    },
    getTools: async (): Promise<any> => {
      const resp = await this._fetch("/v1/telemetry/tools", { method: "GET" });
      return deepCamelCase(await resp.json());
    },
    getTool: async (signatureHash: string): Promise<any> => {
      const resp = await this._fetch(`/v1/telemetry/tools/${signatureHash}`, { method: "GET" });
      return deepCamelCase(await resp.json());
    },
  };

  readonly feedback = {
    getStats: async (): Promise<any> => {
      const resp = await this._fetch("/v1/feedback", { method: "GET" });
      return deepCamelCase(await resp.json());
    },
    getHints: async (toolName: string): Promise<ToolHints> => {
      const resp = await this._fetch(`/v1/feedback/${encodeURIComponent(toolName)}`, { method: "GET" });
      return deepCamelCase<ToolHints>(await resp.json());
    },
  };

  readonly toin = {
    getStats: async (): Promise<TOINStats> => {
      const resp = await this._fetch("/v1/toin/stats", { method: "GET" });
      return deepCamelCase<TOINStats>(await resp.json());
    },
    getPatterns: async (limit?: number): Promise<TOINPattern[]> => {
      const qs = limit ? `?limit=${limit}` : "";
      const resp = await this._fetch(`/v1/toin/patterns${qs}`, { method: "GET" });
      return deepCamelCase<TOINPattern[]>(await resp.json());
    },
    getPattern: async (hashPrefix: string): Promise<any> => {
      const resp = await this._fetch(`/v1/toin/pattern/${encodeURIComponent(hashPrefix)}`, { method: "GET" });
      return deepCamelCase(await resp.json());
    },
  };

  // ============================================================
  // Lifecycle
  // ============================================================

  /** Close the client (no-op for HTTP client, included for API parity). */
  close(): void {
    // HTTP client is stateless — nothing to close
  }

  // ============================================================
  // Internal HTTP helpers
  // ============================================================

  /**
   * Raw fetch with proxy base URL, auth, and timeout.
   * @internal
   */
  async rawFetch(
    path: string,
    options: { method: string; headers?: Record<string, string>; body?: any; stream?: boolean },
  ): Promise<Response> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...options.headers,
    };
    if (this.apiKey) {
      // Don't override provider auth headers
      if (!headers["Authorization"] && !headers["x-api-key"]) {
        headers["Authorization"] = `Bearer ${this.apiKey}`;
      }
    }
    if (this.stack && !headers["X-Headroom-Stack"]) {
      headers["X-Headroom-Stack"] = this.stack;
    }

    let response: Response;
    try {
      response = await fetch(url, {
        method: options.method,
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: AbortSignal.timeout(this.timeout),
      });
    } catch (error) {
      throw new HeadroomConnectionError(
        `Failed to connect to Headroom at ${this.baseUrl}: ${error}`,
      );
    }

    if (!response.ok) {
      let errorBody: ProxyErrorResponse | undefined;
      try {
        errorBody = (await response.json()) as ProxyErrorResponse;
      } catch {
        // ignore
      }
      throw mapProxyError(
        response.status,
        errorBody?.error?.type ?? "unknown",
        errorBody?.error?.message ?? `HTTP ${response.status}`,
      );
    }

    return response;
  }

  /** @internal */
  private async _fetch(
    path: string,
    init: { method: string; body?: string; headers?: Record<string, string> },
  ): Promise<Response> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...init.headers,
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    if (this.stack && !headers["X-Headroom-Stack"]) {
      headers["X-Headroom-Stack"] = this.stack;
    }

    let response: Response;
    try {
      response = await fetch(url, {
        method: init.method,
        headers,
        body: init.body,
        signal: AbortSignal.timeout(this.timeout),
      });
    } catch (error) {
      throw new HeadroomConnectionError(
        `Failed to connect to Headroom at ${this.baseUrl}: ${error}`,
      );
    }

    if (!response.ok) {
      let errorBody: ProxyErrorResponse | undefined;
      try {
        errorBody = (await response.json()) as ProxyErrorResponse;
      } catch {
        // ignore
      }
      throw mapProxyError(
        response.status,
        errorBody?.error?.type ?? "unknown",
        errorBody?.error?.message ?? `HTTP ${response.status}`,
      );
    }

    return response;
  }

  private async _doCompress(
    messages: OpenAIMessage[],
    model: string,
    tokenBudget?: number,
  ): Promise<CompressResult> {
    const body: Record<string, unknown> = { messages, model };
    if (tokenBudget) {
      body.token_budget = tokenBudget;
    }
    if (this.config) {
      body.config = deepSnakeCase(this.config);
    }

    const response = await this._fetch("/v1/compress", {
      method: "POST",
      body: JSON.stringify(body),
    });

    const data = (await response.json()) as ProxyCompressResponse;

    return {
      messages: data.messages,
      tokensBefore: data.tokens_before,
      tokensAfter: data.tokens_after,
      tokensSaved: data.tokens_saved,
      compressionRatio: data.compression_ratio,
      transformsApplied: data.transforms_applied,
      ccrHashes: data.ccr_hashes,
      compressed: true,
    };
  }
}
