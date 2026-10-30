/**
 * Core types for the Headroom TypeScript SDK.
 * Error classes moved to errors.ts — re-exported here for backwards compatibility.
 */

import type { CompressionHooks } from "./hooks.js";
import type { CompressRequestConfig } from "./types/config.js";
import type { MessageFormat } from "./utils/format.js";

// --- Message types (OpenAI chat format) ---

/**
 * Provider-specific options as the Vercel AI SDK attaches them to messages and
 * parts (`{ anthropic: { cacheControl }, google: { thoughtSignature }, ... }`).
 * The SDK carries them through `/v1/compress` unchanged so the AI SDK provider
 * sees exactly what it produced; they are never sent to an OpenAI endpoint.
 */
export type ProviderOptions = Record<string, Record<string, unknown>>;

export interface TextContentPart {
  type: "text";
  text: string;
  providerOptions?: ProviderOptions;
}

export interface ImageContentPart {
  type: "image_url";
  image_url: { url: string; detail?: "auto" | "low" | "high" };
}

/**
 * A Vercel AI SDK `reasoning` part (the assistant's thinking, with the
 * provider's signature or redacted data in `providerOptions`). Carried verbatim
 * through `/v1/compress`, which passes unknown parts through, so the provider
 * gets its reasoning back on the next step of a tool loop.
 */
export interface ReasoningContentPart {
  type: "reasoning";
  text: string;
  providerOptions?: ProviderOptions;
}

/**
 * Other Vercel AI SDK assistant parts with no OpenAI equivalent, carried verbatim
 * the same way: `reasoning-file`, `custom`, provider-executed `tool-result`,
 * `tool-approval-request`.
 */
export interface PassthroughContentPart {
  type: "reasoning-file" | "custom" | "tool-result" | "tool-approval-request";
  [key: string]: unknown;
}

export type ContentPart =
  | TextContentPart
  | ImageContentPart
  | ReasoningContentPart
  | PassthroughContentPart;

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
  /** Vercel AI SDK: the call was executed by the provider. Carried, never sent to OpenAI. */
  providerExecuted?: boolean;
  providerOptions?: ProviderOptions;
}

export interface SystemMessage {
  role: "system";
  content: string;
  providerOptions?: ProviderOptions;
}

export interface UserMessage {
  role: "user";
  content: string | ContentPart[];
  providerOptions?: ProviderOptions;
}

export interface AssistantMessage {
  role: "assistant";
  /**
   * A string, or an ordered part array when the assistant turn carries media
   * (e.g. a Vercel AI SDK file part): text parts plus `image_url` parts.
   */
  content: string | ContentPart[] | null;
  tool_calls?: ToolCall[];
  providerOptions?: ProviderOptions;
}

export interface ToolMessage {
  role: "tool";
  content: string;
  tool_call_id: string;
  providerOptions?: ProviderOptions;
}

export type OpenAIMessage =
  | SystemMessage
  | UserMessage
  | AssistantMessage
  | ToolMessage;

// --- Compress API ---

export interface CompressOptions {
  model?: string;
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  fallback?: boolean;
  retries?: number;
  client?: HeadroomClientInterface;
  /** Token budget — compress to fit within this limit. Used for compaction. */
  tokenBudget?: number;
  /** Compression hooks for pre/post processing. */
  hooks?: CompressionHooks;
  /** Integration slug sent as X-Headroom-Stack (e.g. "adapter_ts_openai"). */
  stack?: string;
  /** Per-call /v1/compress config, merged over the client-level config
   *  (per-call keys override at the top level). */
  config?: CompressRequestConfig;
  /**
   * The format of `messages`, when the caller knows it. Skips autodetection,
   * which cannot tell a text-only Vercel AI SDK prompt from OpenAI messages.
   */
  format?: MessageFormat;
}

export interface CompressResult {
  /**
   * Compressed messages in the same format as input. Typed `any[]` because the
   * universal `compress()` returns messages in the *input* format — it calls
   * `fromOpenAI(messages, inputFormat)`, so for Anthropic/Vercel/Gemini input
   * these are not OpenAI-shaped. See the note on `HeadroomClient.compress`.
   */
  messages: any[];
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  compressionRatio: number;
  transformsApplied: string[];
  ccrHashes: string[];
  compressed: boolean;
}

// --- Client ---

export interface HeadroomClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  fallback?: boolean;
  retries?: number;
  /** Integration slug sent as X-Headroom-Stack on every request. */
  stack?: string;
}

export interface HeadroomClientInterface {
  compress(
    messages: OpenAIMessage[],
    options?: { model?: string; tokenBudget?: number; config?: CompressRequestConfig },
  ): Promise<CompressResult>;
}

// --- Re-export errors for backwards compatibility ---

export {
  HeadroomError,
  HeadroomConnectionError,
  HeadroomAuthError,
  HeadroomCompressError,
  ConfigurationError,
  ProviderError,
  StorageError,
  TokenizationError,
  CacheError,
  ValidationError,
  TransformError,
  mapProxyError,
} from "./errors.js";

// --- Proxy response (internal) ---

export interface ProxyCompressResponse {
  messages: OpenAIMessage[];
  tokens_before: number;
  tokens_after: number;
  tokens_saved: number;
  compression_ratio: number;
  transforms_applied: string[];
  ccr_hashes: string[];
}

export interface ProxyErrorResponse {
  error: {
    type: string;
    message: string;
  };
}
