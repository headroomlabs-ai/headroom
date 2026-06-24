/**
 * Convert between OpenClaw's AgentMessage format and OpenAI message format.
 *
 * AgentMessage uses:
 *   role: "user" | "assistant" | "toolResult"
 *   content: string | ContentBlock[]
 *
 * OpenAI uses:
 *   role: "user" | "assistant" | "system" | "tool"
 *   content: string
 *   tool_calls?: ToolCall[]
 *   tool_call_id?: string
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface OpenAIMessage {
  role: string;
  content: string | null;
  tool_calls?: any[];
  tool_call_id?: string;
  name?: string;
  _headroomMeta?: Record<string, unknown>;
}

/**
 * Convert AgentMessage[] to OpenAI message format for compression.
 */
export function agentToOpenAI(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    const normalized = normalizeAgentMessage(msg);
    const role = normalized.role;

    const buildMeta = (): Record<string, unknown> => {
      const meta = { ...normalized } as Record<string, unknown>;
      delete meta.role;
      delete meta.content;
      return meta;
    };

    if (role === "system") {
      result.push({
        role: "system",
        content:
          typeof normalized.content === "string"
            ? normalized.content
            : extractText(normalized.content),
        _headroomMeta: buildMeta(),
      });
      continue;
    }

    if (role === "user") {
      result.push({
        role: "user",
        content:
          typeof normalized.content === "string"
            ? normalized.content
            : extractText(normalized.content),
        _headroomMeta: buildMeta(),
      });
      continue;
    }

    if (role === "assistant") {
      const content = normalized.content;
      if (typeof content === "string") {
        result.push({ role: "assistant", content, _headroomMeta: buildMeta() });
        continue;
      }

      // Content blocks: extract text and tool call blocks.
      // OpenClaw uses `toolCall`; some adapters still emit legacy `tool_use`.
      if (Array.isArray(content)) {
        const textParts: string[] = [];
        const toolCalls: any[] = [];

        for (const block of content) {
          if (typeof block === "string") {
            textParts.push(block);
          } else if (block.type === "text") {
            textParts.push(block.text);
          } else if (block.type === "tool_use" || block.type === "toolCall") {
            const args =
              block.type === "toolCall"
                ? block.arguments
                : block.input;
            toolCalls.push({
              id: block.id,
              type: "function",
              function: {
                name: block.name,
                arguments:
                  typeof args === "string"
                    ? args
                    : JSON.stringify(args ?? {}),
              },
            });
          }
        }

        const openaiMsg: OpenAIMessage = {
          role: "assistant",
          content: textParts.length > 0 ? textParts.join("") : null,
          _headroomMeta: buildMeta(),
        };
        if (toolCalls.length > 0) {
          openaiMsg.tool_calls = toolCalls;
        }
        result.push(openaiMsg);
      }
      continue;
    }

    if (role === "toolResult" || role === "tool_result") {
      const content =
        typeof normalized.content === "string"
          ? normalized.content
          : Array.isArray(normalized.content)
            ? extractText(normalized.content)
            : JSON.stringify(normalized.content);

      result.push({
        role: "tool",
        content,
        tool_call_id:
          normalized.toolCallId ??
          normalized.tool_use_id ??
          normalized.id ??
          "unknown",
        _headroomMeta: buildMeta(),
      });
      continue;
    }

    // Fallback: pass through as user message
    result.push({
      role: "user",
      content:
        typeof normalized.content === "string"
          ? normalized.content
          : JSON.stringify(normalized.content),
      _headroomMeta: buildMeta(),
    });
  }

  return result;
}

/**
 * Convert compressed OpenAI messages back to AgentMessage format.
 */
export function openAIToAgent(messages: OpenAIMessage[]): any[] {
  const result: any[] = [];

  for (const msg of messages) {
    const meta = (msg._headroomMeta ?? {}) as Record<string, unknown>;
    const timestamp =
      typeof meta.timestamp === "number" ? meta.timestamp : Date.now();

    if (msg.role === "system") {
      result.push({
        role: "system",
        content: msg.content ?? "",
        timestamp,
      });
      continue;
    }

    if (msg.role === "user") {
      result.push({
        role: "user",
        content: msg.content ?? "",
        timestamp,
      });
      continue;
    }

    if (msg.role === "assistant") {
      const blocks: any[] = [];
      if (msg.content) {
        blocks.push({ type: "text", text: msg.content });
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          let input: any;
          try {
            input = JSON.parse(tc.function.arguments);
          } catch {
            input = tc.function.arguments ?? {};
          }
          // Emit OpenClaw-native block shape so downstream transports keep call linkage.
          blocks.push({
            type: "toolCall",
            id: tc.id,
            name: tc.function.name,
            arguments: input,
          });
        }
      }
      // OpenClaw's Pi agent expects content to always be an array for assistant messages
      // (it calls .flatMap() on it). Never flatten to a string.
      result.push({
        ...(meta as object),
        role: "assistant",
        content: blocks,
        api: typeof meta.api === "string" ? meta.api : "headroom",
        provider: typeof meta.provider === "string" ? meta.provider : "headroom",
        model: typeof meta.model === "string" ? meta.model : "headroom",
        usage:
          isRecord(meta.usage)
            ? meta.usage
            : {
                input: 0,
                output: 0,
                cacheRead: 0,
                cacheWrite: 0,
                totalTokens: 0,
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
              },
        stopReason:
          typeof meta.stopReason === "string" ? meta.stopReason : "stop",
        timestamp,
      });
      continue;
    }

    if (msg.role === "tool") {
      const textContent =
        typeof msg.content === "string"
          ? msg.content
          : msg.content == null
            ? ""
            : JSON.stringify(msg.content);
      const toolCallId = msg.tool_call_id ?? "unknown";
      result.push({
        ...(meta as object),
        role: "toolResult",
        // OpenClaw transport layers expect toolResult content blocks, not a raw string.
        content: [{ type: "text", text: textContent }],
        toolCallId:
          typeof meta.toolCallId === "string" ? meta.toolCallId : toolCallId,
        tool_use_id:
          typeof meta.tool_use_id === "string" ? meta.tool_use_id : toolCallId,
        toolName:
          typeof meta.toolName === "string" ? meta.toolName : "headroom",
        isError: typeof meta.isError === "boolean" ? meta.isError : false,
        timestamp,
      });
      continue;
    }
  }

  return result;
}

export function normalizeAgentMessages(messages: any[]): any[] {
  return messages.map((message) => normalizeAgentMessage(message));
}

/**
 * Extract text from content blocks.
 */
function extractText(content: any): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return JSON.stringify(content);

  return content
    .map((block: any) => {
      if (typeof block === "string") return block;
      if (block.type === "text") return block.text;
      if (block.type === "tool_result") {
        return typeof block.content === "string" ? block.content : JSON.stringify(block.content);
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

function normalizeAgentMessage(message: any): any {
  if (!isRecord(message)) return message;

  if (message.role === "assistant") {
    return normalizeAssistantMessage(message);
  }

  if (message.role === "toolResult" || message.role === "tool_result") {
    return normalizeToolResultMessage(message);
  }

  return message;
}

function normalizeAssistantMessage(message: Record<string, any>): Record<string, any> {
  const normalizedContent = normalizeAssistantContent(message.content);

  return {
    ...message,
    content: normalizedContent,
    api: typeof message.api === "string" ? message.api : "headroom",
    provider: typeof message.provider === "string" ? message.provider : "headroom",
    model: typeof message.model === "string" ? message.model : "headroom",
    usage: isRecord(message.usage)
      ? message.usage
      : {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        },
    stopReason: typeof message.stopReason === "string" ? message.stopReason : "stop",
    timestamp: typeof message.timestamp === "number" ? message.timestamp : Date.now(),
  };
}

function normalizeToolResultMessage(message: Record<string, any>): Record<string, any> {
  const normalizedContent = normalizeToolResultContent(message.content);
  const toolCallId =
    typeof message.toolCallId === "string"
      ? message.toolCallId
      : typeof message.tool_use_id === "string"
        ? message.tool_use_id
        : typeof message.id === "string"
          ? message.id
          : "unknown";

  return {
    ...message,
    role: "toolResult",
    content: normalizedContent,
    toolCallId,
    tool_use_id:
      typeof message.tool_use_id === "string" ? message.tool_use_id : toolCallId,
    toolName: typeof message.toolName === "string" ? message.toolName : "headroom",
    isError: typeof message.isError === "boolean" ? message.isError : false,
    timestamp: typeof message.timestamp === "number" ? message.timestamp : Date.now(),
  };
}

function normalizeAssistantContent(content: unknown): any[] {
  if (Array.isArray(content)) {
    return content.flatMap((block) => {
      if (typeof block === "string") return [{ type: "text", text: block }];
      if (!isRecord(block) || typeof block.type !== "string") return [];
      if (block.type === "text" && typeof block.text === "string") return [block];
      if (block.type === "thinking" && typeof block.thinking === "string") return [block];
      if (
        (block.type === "toolCall" || block.type === "tool_use") &&
        typeof block.name === "string"
      ) {
        return [
          {
            type: "toolCall",
            id: typeof block.id === "string" ? block.id : "unknown",
            name: block.name,
            arguments:
              "arguments" in block
                ? block.arguments
                : "input" in block
                  ? block.input
                  : {},
          },
        ];
      }
      return [];
    });
  }

  if (typeof content === "string" && content.length > 0) {
    return [{ type: "text", text: content }];
  }

  if (content == null) {
    return [];
  }

  return [{ type: "text", text: JSON.stringify(content) }];
}

function normalizeToolResultContent(content: unknown): any[] {
  if (Array.isArray(content)) {
    return content.flatMap((block) => {
      if (typeof block === "string") return [{ type: "text", text: block }];
      if (!isRecord(block) || typeof block.type !== "string") return [];
      if (block.type === "text" && typeof block.text === "string") return [block];
      if (
        block.type === "image" &&
        typeof block.data === "string" &&
        typeof block.mimeType === "string"
      ) {
        return [block];
      }
      if (block.type === "tool_result" && "content" in block) {
        return normalizeToolResultContent(block.content);
      }
      return [];
    });
  }

  if (typeof content === "string" && content.length > 0) {
    return [{ type: "text", text: content }];
  }

  if (content == null) {
    return [];
  }

  return [{ type: "text", text: JSON.stringify(content) }];
}

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
