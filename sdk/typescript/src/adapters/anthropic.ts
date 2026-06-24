import { compress } from "../compress.js";
import type { CompressOptions, OpenAIMessage } from "../types.js";
import type { AssistantMessage, ToolCall } from "../types.js";

/* eslint-disable @typescript-eslint/no-explicit-any */

interface AnthropicLike {
  messages: {
    create: (params: any) => any;
  };
  [key: string]: any;
}

/**
 * Convert Anthropic messages to OpenAI format for compression.
 *
 * Anthropic format:
 *   { role: 'user' | 'assistant', content: string | ContentBlock[] }
 *   ContentBlock = { type: 'text', text } | { type: 'tool_use', ... } | { type: 'tool_result', ... }
 */
function anthropicToOpenAI(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({ role: "user", content: msg.content });
      } else if (Array.isArray(msg.content)) {
        const toolResults = msg.content.filter(
          (b: any) => b.type === "tool_result",
        );
        const textBlocks = msg.content.filter((b: any) => b.type === "text");

        if (textBlocks.length > 0) {
          result.push({
            role: "user",
            content: textBlocks.map((b: any) => b.text).join("\n"),
          });
        }
        for (const tr of toolResults) {
          result.push({
            role: "tool",
            content:
              typeof tr.content === "string"
                ? tr.content
                : JSON.stringify(tr.content),
            tool_call_id: tr.tool_use_id,
          });
        }
      }
      continue;
    }

    if (msg.role === "assistant") {
      if (typeof msg.content === "string") {
        result.push({ role: "assistant", content: msg.content });
      } else if (Array.isArray(msg.content)) {
        const textBlocks = msg.content.filter((b: any) => b.type === "text");
        const toolUseBlocks = msg.content.filter(
          (b: any) => b.type === "tool_use",
        );

        const content =
          textBlocks.length > 0
            ? textBlocks.map((b: any) => b.text).join("\n")
            : null;

        const openaiMsg: AssistantMessage = { role: "assistant", content };
        if (toolUseBlocks.length > 0) {
          openaiMsg.tool_calls = toolUseBlocks.map(
            (b: any): ToolCall => ({
              id: b.id,
              type: "function",
              function: {
                name: b.name,
                arguments: JSON.stringify(b.input),
              },
            }),
          );
        }
        result.push(openaiMsg);
      }
      continue;
    }
  }

  return result;
}

/**
 * Convert compressed OpenAI messages back to Anthropic format.
 */
function openAIToAnthropic(messages: OpenAIMessage[]): any[] {
  const result: any[] = [];

  for (const msg of messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({ role: "user", content: msg.content });
      } else if (Array.isArray(msg.content)) {
        result.push({
          role: "user",
          content: msg.content.map((p) => {
            if (p.type === "text") return { type: "text", text: p.text };
            return { type: "text", text: "" };
          }),
        });
      }
      continue;
    }

    if (msg.role === "assistant") {
      const blocks: any[] = [];
      if (msg.content) blocks.push({ type: "text", text: msg.content });
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          blocks.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: JSON.parse(tc.function.arguments),
          });
        }
      }
      result.push({
        role: "assistant",
        content:
          blocks.length === 1 && blocks[0].type === "text"
            ? blocks[0].text
            : blocks,
      });
      continue;
    }

    if (msg.role === "tool") {
      result.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: msg.tool_call_id,
            content: msg.content,
          },
        ],
      });
      continue;
    }

    if (msg.role === "system") {
      // System messages are handled separately in Anthropic (as `system` param)
      // Pass through as user message if it appears in the messages array
      result.push({ role: "user", content: msg.content });
    }
  }

  return result;
}

/**
 * Wrap an Anthropic client to auto-compress messages before each request.
 *
 * Intercepts `client.messages.create()` only. All other methods pass through.
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

    const openaiMessages = anthropicToOpenAI(messages);
    const result = await compress(openaiMessages, {
      stack: "adapter_ts_anthropic",
      ...options,
      model,
    });

    const anthropicMessages = result.compressed
      ? openAIToAnthropic(result.messages)
      : messages;

    return originalCreate({
      ...params,
      messages: anthropicMessages,
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
