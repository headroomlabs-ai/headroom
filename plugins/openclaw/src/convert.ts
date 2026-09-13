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
 *
 * Non-text blocks (images, tool envelopes) are never sent to the proxy. They are
 * replaced by placeholders on the wire and restored from the original messages
 * by `openAIToAgent(..., { originals })`. `_headroomMeta` is still attached as a
 * best-effort hint, but restoration does not depend on the proxy echoing it.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import {
  type ContentBlock,
  IMAGE_TOKEN_ESTIMATE,
  isImageBlock,
  isRecord,
  isTextBlock,
  mergeCompressedTextIntoBlocks,
  serializeAssistantTextForWire,
  serializeBlocksForWire,
} from "./content-blocks.js";
import { buildOriginalLookup, findOriginal } from "./original-lookup.js";
import { buildToolCallNameMap, isProtectedToolName, resolveToolName } from "./tool-names.js";

/** Joiner for text blocks inside tool results and user messages on the wire. */
const BLOCK_JOINER = "\n";

/** Keys that exist only for the wire round-trip and must not leak into AgentMessages. */
const INTERNAL_META_KEYS = new Set(["hrIndex"]);

const DEFAULT_USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

/** Rough token estimate (~4 chars/token, fixed cost per image) for assemble budget short-circuit. */
export function estimateRoughTokens(messages: any[]): number {
  let chars = 0;
  for (const msg of messages) {
    const content = msg?.content;
    if (typeof content === "string") {
      chars += content.length;
      continue;
    }
    if (Array.isArray(content)) {
      for (const block of content) {
        if (isTextBlock(block)) {
          chars += block.text.length;
        } else if (isImageBlock(block)) {
          chars += IMAGE_TOKEN_ESTIMATE * 4;
        } else {
          chars += JSON.stringify(block).length;
        }
      }
      continue;
    }
    if (content != null) {
      chars += JSON.stringify(content).length;
    }
  }
  return Math.max(1, Math.ceil(chars / 4));
}

export interface OpenAIMessage {
  role: string;
  content: string | null;
  tool_calls?: any[];
  tool_call_id?: string;
  name?: string;
  _headroomMeta?: Record<string, unknown>;
}

export interface OpenAIToAgentOptions {
  /**
   * The AgentMessages that were passed to `agentToOpenAI`. When provided, image
   * blocks, tool call blocks, thinking blocks, `isError`, `toolName` and other
   * metadata are restored from these instead of from proxy-echoed metadata.
   */
  originals?: any[];
  /**
   * Lowercase tool names (or `*` globs) whose results are restored verbatim from
   * `originals`, whatever the proxy returned for them. Matched against the
   * resolved tool name and its aliases (see `tool-names.ts`). Requires `originals`.
   */
  protectedToolNames?: ReadonlySet<string>;
}

/**
 * Convert AgentMessage[] to OpenAI message format for compression.
 *
 * Deferred `tool_call` wrappers are sent under their resolved tool name so the
 * proxy's per-tool protection/exclusion lists can match them; the wrapper name
 * itself is restored from the originals by `openAIToAgent`.
 */
export function agentToOpenAI(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];
  const normalizedMessages = normalizeAgentMessages(messages);
  const toolCallNames = buildToolCallNameMap(normalizedMessages);

  normalizedMessages.forEach((normalized, index) => {
    const role = normalized.role;

    const buildMeta = (): Record<string, unknown> => {
      const meta = { ...normalized } as Record<string, unknown>;
      delete meta.role;
      delete meta.content;
      meta.hrIndex = index;
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
      return;
    }

    if (role === "user") {
      const content = normalized.content;
      if (typeof content === "string") {
        result.push({ role: "user", content, _headroomMeta: buildMeta() });
        return;
      }

      if (Array.isArray(content)) {
        result.push({
          role: "user",
          content: serializeBlocksForWire(normalizeUserContent(content), BLOCK_JOINER),
          _headroomMeta: buildMeta(),
        });
        return;
      }

      result.push({
        role: "user",
        content: JSON.stringify(content),
        _headroomMeta: buildMeta(),
      });
      return;
    }

    if (role === "assistant") {
      const content = normalized.content;
      if (typeof content === "string") {
        result.push({ role: "assistant", content, _headroomMeta: buildMeta() });
        return;
      }

      // Content blocks: text goes on the wire, tool calls become `tool_calls`,
      // thinking and unknown blocks stay local and are restored from originals.
      if (Array.isArray(content)) {
        const blocks = normalizeAssistantContent(content);
        const wireTextValue = serializeAssistantTextForWire(blocks);
        const toolCalls: any[] = [];

        for (const block of blocks) {
          if (block.type === "toolCall") {
            const args = block.arguments;
            toolCalls.push({
              id: block.id,
              type: "function",
              function: {
                name: resolveToolName(typeof block.name === "string" ? block.name : "", args),
                arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}),
              },
            });
          }
        }

        const openaiMsg: OpenAIMessage = {
          role: "assistant",
          content: wireTextValue.length > 0 ? wireTextValue : null,
          _headroomMeta: buildMeta(),
        };
        if (toolCalls.length > 0) {
          openaiMsg.tool_calls = toolCalls;
        }
        result.push(openaiMsg);
      }
      return;
    }

    if (role === "toolResult" || role === "tool_result") {
      const toolBlocks = normalizeToolResultContent(normalized.content);
      const toolCallId = typeof normalized.toolCallId === "string" ? normalized.toolCallId : "unknown";
      const rawToolName =
        typeof normalized.toolName === "string" ? normalized.toolName : undefined;
      const toolName = toolCallNames.get(toolCallId) ?? rawToolName;

      result.push({
        role: "tool",
        content: serializeBlocksForWire(toolBlocks, BLOCK_JOINER),
        tool_call_id: toolCallId,
        ...(toolName ? { name: toolName } : {}),
        _headroomMeta: buildMeta(),
      });
      return;
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
  });

  return result;
}

/**
 * Convert compressed OpenAI messages back to AgentMessage format.
 *
 * Pass `options.originals` (the input to `agentToOpenAI`) to restore images, tool
 * call blocks, thinking blocks and metadata regardless of what the proxy echoed.
 */
export function openAIToAgent(
  messages: OpenAIMessage[],
  options: OpenAIToAgentOptions = {},
): any[] {
  const normalizedOriginals = options.originals ? normalizeAgentMessages(options.originals) : null;
  const lookup = normalizedOriginals ? buildOriginalLookup(normalizedOriginals) : null;
  const protectedToolNames = options.protectedToolNames ?? new Set<string>();
  const originalToolCallNames =
    normalizedOriginals && protectedToolNames.size > 0
      ? buildToolCallNameMap(normalizedOriginals)
      : new Map<string, string>();
  const result: any[] = [];

  messages.forEach((msg, index) => {
    const meta = stripInternalMeta(msg._headroomMeta);
    const original = lookup
      ? findOriginal({ compressed: msg, index, compressedCount: messages.length, lookup })
      : null;
    // Original fields win over echoed metadata; both are already normalized.
    const base: Record<string, unknown> = original ? { ...original } : { ...meta };
    delete base.role;
    delete base.content;
    const timestamp = typeof base.timestamp === "number" ? base.timestamp : Date.now();

    if (msg.role === "system") {
      result.push({ role: "system", content: msg.content ?? "", timestamp });
      return;
    }

    if (msg.role === "user") {
      result.push({
        ...base,
        role: "user",
        content: restoreUserContent(msg, original),
        timestamp,
      });
      return;
    }

    if (msg.role === "assistant") {
      // OpenClaw's Pi agent expects content to always be an array for assistant messages
      // (it calls .flatMap() on it). Never flatten to a string.
      result.push({
        ...base,
        role: "assistant",
        content: restoreAssistantContent(msg, original),
        api: typeof base.api === "string" ? base.api : "headroom",
        provider: typeof base.provider === "string" ? base.provider : "headroom",
        model: typeof base.model === "string" ? base.model : "headroom",
        usage: isRecord(base.usage) ? base.usage : DEFAULT_USAGE,
        stopReason: typeof base.stopReason === "string" ? base.stopReason : "stop",
        timestamp,
      });
      return;
    }

    if (msg.role === "tool") {
      const wireCallId = msg.tool_call_id ?? "unknown";
      const toolCallId = typeof base.toolCallId === "string" ? base.toolCallId : wireCallId;
      const content = isProtectedToolResult({
        original,
        toolCallId,
        originalToolCallNames,
        protectedToolNames,
      })
        ? normalizeToolResultContent(original?.content)
        : restoreToolResultContent(msg, original);
      result.push({
        ...base,
        role: "toolResult",
        content,
        toolCallId,
        tool_use_id: typeof base.tool_use_id === "string" ? base.tool_use_id : toolCallId,
        toolName:
          typeof base.toolName === "string"
            ? base.toolName
            : typeof msg.name === "string"
              ? msg.name
              : "headroom",
        isError: inferToolResultIsError(content, base),
        timestamp,
      });
    }
  });

  return result;
}

export function normalizeAgentMessages(messages: any[]): any[] {
  return messages.map((message) => normalizeAgentMessage(message));
}

function stripInternalMeta(meta: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!isRecord(meta)) return {};
  const cleaned: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(meta)) {
    if (!INTERNAL_META_KEYS.has(key)) cleaned[key] = value;
  }
  return cleaned;
}

function wireText(msg: OpenAIMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (msg.content == null) return "";
  return JSON.stringify(msg.content);
}

function restoreUserContent(
  msg: OpenAIMessage,
  original: Record<string, unknown> | null,
): string | ContentBlock[] {
  const text = wireText(msg);
  if (!original || !Array.isArray(original.content)) return text;
  const originalBlocks = normalizeUserContent(original.content);
  return mergeCompressedTextIntoBlocks({
    originalBlocks,
    originalWireText: serializeBlocksForWire(originalBlocks, BLOCK_JOINER),
    compressedText: text,
  });
}

function restoreAssistantContent(
  msg: OpenAIMessage,
  original: Record<string, unknown> | null,
): ContentBlock[] {
  if (original && Array.isArray(original.content)) {
    const originalBlocks = normalizeAssistantContent(original.content);
    return mergeCompressedTextIntoBlocks({
      originalBlocks,
      originalWireText: serializeAssistantTextForWire(originalBlocks),
      compressedText: wireText(msg),
    });
  }

  const blocks: ContentBlock[] = [];
  if (msg.content) {
    blocks.push({ type: "text", text: msg.content });
  }
  for (const tc of msg.tool_calls ?? []) {
    let input: unknown;
    try {
      input = JSON.parse(tc.function.arguments);
    } catch {
      input = tc.function.arguments ?? {};
    }
    blocks.push({ type: "toolCall", id: tc.id, name: tc.function.name, arguments: input });
  }
  return blocks;
}

/**
 * True when the tool result belongs to a protected tool. The tool name comes from
 * the original assistant `toolCall` block (wrapper-resolved), falling back to the
 * original result's `toolName`.
 */
function isProtectedToolResult(params: {
  original: Record<string, unknown> | null;
  toolCallId: string;
  originalToolCallNames: ReadonlyMap<string, string>;
  protectedToolNames: ReadonlySet<string>;
}): boolean {
  const { original, toolCallId, originalToolCallNames, protectedToolNames } = params;
  if (!original || protectedToolNames.size === 0) return false;
  const fromCall = originalToolCallNames.get(toolCallId);
  const fromResult = typeof original.toolName === "string" ? original.toolName : undefined;
  const name = fromCall ?? fromResult;
  return typeof name === "string" && isProtectedToolName(name, protectedToolNames);
}

function restoreToolResultContent(
  msg: OpenAIMessage,
  original: Record<string, unknown> | null,
): ContentBlock[] {
  const text = wireText(msg);
  if (original) {
    const originalBlocks = normalizeToolResultContent(original.content);
    return mergeCompressedTextIntoBlocks({
      originalBlocks,
      originalWireText: serializeBlocksForWire(originalBlocks, BLOCK_JOINER),
      compressedText: text,
    });
  }
  return [{ type: "text", text }];
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
  return {
    ...message,
    content: normalizeAssistantContent(message.content),
    api: typeof message.api === "string" ? message.api : "headroom",
    provider: typeof message.provider === "string" ? message.provider : "headroom",
    model: typeof message.model === "string" ? message.model : "headroom",
    usage: isRecord(message.usage) ? message.usage : DEFAULT_USAGE,
    stopReason: typeof message.stopReason === "string" ? message.stopReason : "stop",
    timestamp: typeof message.timestamp === "number" ? message.timestamp : Date.now(),
  };
}

function normalizeToolResultMessage(message: Record<string, any>): Record<string, any> {
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
    content: normalizeToolResultContent(message.content),
    toolCallId,
    tool_use_id:
      typeof message.tool_use_id === "string" ? message.tool_use_id : toolCallId,
    toolName: typeof message.toolName === "string" ? message.toolName : "headroom",
    isError: typeof message.isError === "boolean" ? message.isError : false,
    timestamp: typeof message.timestamp === "number" ? message.timestamp : Date.now(),
  };
}

function normalizeAssistantContent(content: unknown): ContentBlock[] {
  if (Array.isArray(content)) {
    return content.flatMap((block): ContentBlock[] => {
      if (typeof block === "string") return [{ type: "text", text: block }];
      if (!isRecord(block) || typeof block.type !== "string") return [];
      if (isTextBlock(block)) return [block];
      if (block.type === "thinking" && typeof block.thinking === "string") {
        return [block as ContentBlock];
      }
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
              "arguments" in block ? block.arguments : "input" in block ? block.input : {},
          },
        ];
      }
      return [block as ContentBlock];
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

function normalizeUserContent(content: unknown): ContentBlock[] {
  if (Array.isArray(content)) {
    return content.flatMap((block): ContentBlock[] => {
      if (typeof block === "string") return [{ type: "text", text: block }];
      if (!isRecord(block) || typeof block.type !== "string") return [];
      if (isTextBlock(block)) return [block];
      if (block.type === "tool_result" && "content" in block) {
        return [
          {
            type: "tool_result",
            tool_use_id:
              typeof block.tool_use_id === "string"
                ? block.tool_use_id
                : typeof block.id === "string"
                  ? block.id
                  : "unknown",
            content: normalizeToolResultContent(block.content),
          },
        ];
      }
      return [block as ContentBlock];
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

function normalizeToolResultContent(content: unknown): ContentBlock[] {
  if (Array.isArray(content)) {
    return content.flatMap((block): ContentBlock[] => {
      if (typeof block === "string") return [{ type: "text", text: block }];
      if (!isRecord(block) || typeof block.type !== "string") return [];
      if (isTextBlock(block)) return [block];
      if (isImageBlock(block)) return [block];
      if (block.type === "tool_result" && "content" in block) {
        return normalizeToolResultContent(block.content);
      }
      return [block as ContentBlock];
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

/**
 * `isError` comes from the original message when available. Without it we only
 * trust the OpenClaw error envelope shape (`{"status":"error","tool":"..."}`) so
 * that ordinary tool output containing a `status` field is not misclassified.
 */
function inferToolResultIsError(blocks: ContentBlock[], base: Record<string, unknown>): boolean {
  if (typeof base.isError === "boolean") return base.isError;
  for (const block of blocks) {
    if (!isTextBlock(block)) continue;
    const trimmed = block.text.trim();
    if (!trimmed.startsWith("{")) continue;
    try {
      const parsed = JSON.parse(trimmed) as { status?: unknown; tool?: unknown };
      if (parsed.status === "error" && typeof parsed.tool === "string") return true;
    } catch {
      // not JSON — ignore
    }
  }
  return false;
}
