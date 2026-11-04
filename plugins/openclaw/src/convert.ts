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

/** `_headroomMeta` key recording which AgentMessage a converted message came from. */
const SOURCE_INDEX_KEY = "headroomSourceIndex";

/**
 * Like `agentToOpenAI`, but stamps every converted message with the index of the AgentMessage it came from.
 * The proxy returns `_headroomMeta` unchanged, so `restoreAgentMessages` can map its result back onto the
 * originals even when it drops messages to fit a token budget.
 */
export function agentToOpenAIIndexed(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];
  messages.forEach((message, index) => {
    for (const converted of agentToOpenAI([message])) {
      result.push({
        ...converted,
        _headroomMeta: { ...(converted._headroomMeta ?? {}), [SOURCE_INDEX_KEY]: index },
      });
    }
  });
  return result;
}

/**
 * Map the proxy's result back onto the AgentMessages OpenClaw passed in.
 *
 * `openAIToAgent(agentToOpenAI(m))` is lossy: it drops `thinking` blocks and flattens user and tool content
 * blocks into one string. Applying it to every message whenever anything is compressed rewrites the whole
 * history, which invalidates the provider's prompt cache from the first message on every compressing turn.
 * So a message the proxy returned unchanged is passed through exactly as it came in, and a message it did
 * change keeps every original field and block except the text that compression replaced.
 *
 * `sent` must be the `agentToOpenAIIndexed(originals)` output that was given to the proxy.
 */
export function restoreAgentMessages(
  originals: any[],
  sent: OpenAIMessage[],
  returned: OpenAIMessage[],
): any[] {
  const sentByIndex = new Map<number, OpenAIMessage>();
  for (const message of sent) {
    const index = sourceIndexOf(message);
    if (index !== undefined) sentByIndex.set(index, message);
  }

  const result: any[] = [];
  for (const message of returned) {
    const index = sourceIndexOf(message);
    const original = index === undefined ? undefined : originals[index];
    const before = index === undefined ? undefined : sentByIndex.get(index);
    if (original === undefined || before === undefined) {
      // Not one of ours (e.g. a message the proxy added): convert it the way it always was.
      result.push(...openAIToAgent([withoutSourceIndex(message)]));
    } else if (sameWireMessage(before, message)) {
      result.push(normalizeAgentMessage(original));
    } else {
      result.push(applyCompressedMessage(original, message));
    }
  }
  return result;
}

function sourceIndexOf(message: OpenAIMessage): number | undefined {
  const index = message._headroomMeta?.[SOURCE_INDEX_KEY];
  return typeof index === "number" ? index : undefined;
}

function withoutSourceIndex(message: OpenAIMessage): OpenAIMessage {
  if (!message._headroomMeta || !(SOURCE_INDEX_KEY in message._headroomMeta)) return message;
  const { [SOURCE_INDEX_KEY]: _index, ...meta } = message._headroomMeta;
  return { ...message, _headroomMeta: meta };
}

function sameWireMessage(a: OpenAIMessage, b: OpenAIMessage): boolean {
  return (
    a.role === b.role &&
    a.content === b.content &&
    a.tool_call_id === b.tool_call_id &&
    a.name === b.name &&
    JSON.stringify(a.tool_calls ?? null) === JSON.stringify(b.tool_calls ?? null)
  );
}

function applyCompressedMessage(original: any, compressed: OpenAIMessage): any {
  const base = normalizeAgentMessage(original);
  const text = typeof compressed.content === "string" ? compressed.content : "";

  if (base.role === "assistant") {
    let content = replaceTextBlocks(base.content, text);
    if (Array.isArray(compressed.tool_calls)) {
      content = content.map((block: any) => withCompressedArguments(block, compressed.tool_calls!));
    }
    return { ...base, content };
  }

  if (typeof base.content === "string") return { ...base, content: text };
  return { ...base, content: replaceTextBlocks(base.content, text) };
}

/** Replace the text blocks with one block holding `text`, where the first one was; keep every other block. */
function replaceTextBlocks(content: unknown, text: string): any[] {
  const blocks = Array.isArray(content) ? content : [{ type: "text", text: String(content ?? "") }];
  const result: any[] = [];
  let placed = false;
  for (const block of blocks) {
    const isText = typeof block === "string" || (isRecord(block) && block.type === "text");
    if (!isText) {
      result.push(block);
      continue;
    }
    if (!placed) {
      if (text) result.push(isRecord(block) ? { ...block, text } : { type: "text", text });
      placed = true;
    }
  }
  if (!placed && text) result.unshift({ type: "text", text });
  return result;
}

function withCompressedArguments(block: any, toolCalls: any[]): any {
  if (!isRecord(block) || (block.type !== "toolCall" && block.type !== "tool_use")) return block;
  const call = toolCalls.find((tc) => tc?.id === block.id);
  const raw = call?.function?.arguments;
  if (typeof raw !== "string") return block;
  const current = block.type === "toolCall" ? block.arguments : block.input;
  if ((typeof current === "string" ? current : JSON.stringify(current ?? {})) === raw) return block;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    parsed = raw;
  }
  return block.type === "toolCall" ? { ...block, arguments: parsed } : { ...block, input: parsed };
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
        // Legacy input becomes arguments; do not retain an alias that can go stale after compression.
        const { input: _legacyInput, ...metadata } = block;
        return [
          {
            ...(block.type === "tool_use" ? metadata : block),
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
