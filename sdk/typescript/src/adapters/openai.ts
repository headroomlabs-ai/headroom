import { compress } from "../compress.js";
import type { CompressOptions, OpenAIMessage } from "../types.js";

/* eslint-disable @typescript-eslint/no-explicit-any */

interface OpenAILike {
  chat: {
    completions: {
      create: (params: any) => any;
    };
  };
  [key: string]: any;
}

/**
 * Wrap an OpenAI client to auto-compress messages before each request.
 *
 * Intercepts `client.chat.completions.create()` only. All other methods
 * (embeddings, images, audio, etc.) pass through unchanged.
 *
 * @example
 * ```typescript
 * import { withHeadroom } from 'headroom-ai/openai';
 * import OpenAI from 'openai';
 *
 * const client = withHeadroom(new OpenAI());
 * const response = await client.chat.completions.create({
 *   model: 'gpt-4o',
 *   messages: longConversation,
 * });
 * ```
 */
export function withHeadroom<T extends OpenAILike>(
  client: T,
  options: CompressOptions = {},
): T {
  const originalCreate = client.chat.completions.create.bind(
    client.chat.completions,
  );

  const compressedCreate = async (params: any) => {
    const messages: OpenAIMessage[] = params.messages;
    const model = options.model ?? params.model ?? "gpt-4o";

    const result = await compress(messages, {
      stack: "adapter_ts_openai",
      ...options,
      model,
    });

    return originalCreate({
      ...params,
      messages: result.messages,
    });
  };

  const completionsProxy = new Proxy(client.chat.completions, {
    get(target, prop) {
      if (prop === "create") return compressedCreate;
      return (target as any)[prop];
    },
  });

  const chatProxy = new Proxy(client.chat, {
    get(target, prop) {
      if (prop === "completions") return completionsProxy;
      return (target as any)[prop];
    },
  });

  return new Proxy(client, {
    get(target, prop) {
      if (prop === "chat") return chatProxy;
      return (target as any)[prop];
    },
  }) as T;
}
