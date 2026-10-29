import { compress } from "../compress.js";
import type { CompressOptions } from "../types.js";

/* eslint-disable @typescript-eslint/no-explicit-any */

interface AnthropicLike {
  messages: {
    create: (params: any) => any;
  };
  [key: string]: any;
}

/**
 * Wrap an Anthropic client to auto-compress messages before each request.
 *
 * Intercepts `client.messages.create()` only. All other methods pass through.
 * Messages reach `/v1/compress` in Anthropic shape and come back in Anthropic
 * shape: the proxy compresses Anthropic content blocks natively, so thinking
 * blocks, cache_control markers and tool_result blocks are forwarded exactly
 * as the caller built them.
 *
 * @example
 * ```typescript
 * import { withHeadroom } from 'headroom-ai/anthropic';
 * import Anthropic from '@anthropic-ai/sdk';
 *
 * const client = withHeadroom(new Anthropic());
 * const response = await client.messages.create({
 *   model: 'claude-sonnet-4-5-20250929',
 *   messages: longConversation,
 *   max_tokens: 1024,
 * });
 * ```
 */
export function withHeadroom<T extends AnthropicLike>(
  client: T,
  options: CompressOptions = {},
): T {
  const originalCreate = client.messages.create.bind(client.messages);

  const compressedCreate = async (params: any) => {
    const messages = params.messages;
    const model =
      options.model ?? params.model ?? "claude-sonnet-4-5-20250929";

    const result = await compress(messages, {
      stack: "adapter_ts_anthropic",
      ...options,
      model,
    });

    return originalCreate({
      ...params,
      messages: result.compressed ? result.messages : messages,
    });
  };

  const messagesProxy = new Proxy(client.messages, {
    get(target, prop) {
      if (prop === "create") return compressedCreate;
      return (target as any)[prop];
    },
  });

  return new Proxy(client, {
    get(target, prop) {
      if (prop === "messages") return messagesProxy;
      return (target as any)[prop];
    },
  }) as T;
}
