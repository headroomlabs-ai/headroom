/**
 * Compression hooks matching Python headroom.hooks.
 * Allows customizing compression behavior with pre/post hooks.
 */

export interface CompressContext {
  model: string;
  userQuery: string;
  turnNumber: number;
  toolCalls: string[];
  provider: string;
}

export interface CompressEvent {
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  compressionRatio: number;
  transformsApplied: string[];
  ccrHashes: string[];
  model: string;
  userQuery: string;
  provider: string;
}

/**
 * Base class for compression hooks. Override methods to customize behavior.
 *
 * @example
 * ```typescript
 * class LoggingHooks extends CompressionHooks {
 *   postCompress(event: CompressEvent) {
 *     console.log(`Saved ${event.tokensSaved} tokens (${event.compressionRatio})`);
 *   }
 * }
 * ```
 */
export class CompressionHooks {
  /**
   * Called before compression. Modify messages before they're sent to the proxy.
   */
  preCompress(
    messages: any[],
    _ctx: CompressContext,
  ): any[] | Promise<any[]> {
    return messages;
  }

  /**
   * Compute per-message compression biases.
   * Return a map of message index -> bias (>1 = preserve more, <1 = compress more).
   */
  computeBiases(
    _messages: any[],
    _ctx: CompressContext,
  ): Record<number, number> | Promise<Record<number, number>> {
    return {};
  }

  /**
   * Called after compression. Observe-only — cannot modify the result.
   */
  postCompress(_event: CompressEvent): void | Promise<void> {
    // no-op default
  }
}

/**
 * Extract the last user message text from a messages array (any format).
 */
export function extractUserQuery(messages: any[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "user") {
      if (typeof msg.content === "string") return msg.content;
      if (Array.isArray(msg.content)) {
        const textPart = msg.content.find(
          (p: any) => p.type === "text" || p.text,
        );
        if (textPart) return textPart.text ?? textPart.content ?? "";
      }
    }
  }
  return "";
}

/**
 * Count conversation turns (user+assistant pairs).
 */
export function countTurns(messages: any[]): number {
  return messages.filter((m) => m.role === "user").length;
}

/**
 * Extract tool call names from messages.
 */
export function extractToolCalls(messages: any[]): string[] {
  const names: string[] = [];
  for (const msg of messages) {
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        names.push(tc.function?.name ?? tc.name ?? "unknown");
      }
    }
    if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type === "tool_use") names.push(part.name ?? "unknown");
        if (part.type === "tool-call") names.push(part.toolName ?? "unknown");
      }
    }
  }
  return names;
}
