/**
 * Locate the original (pre-compression) AgentMessage for a compressed OpenAI
 * message returned by the Headroom proxy.
 *
 * The proxy is not guaranteed to echo `_headroomMeta` (some transforms rebuild
 * message objects), so restoration relies on local state:
 *   1. tool messages   -> unique `tool_call_id`
 *   2. other messages  -> echoed `_headroomMeta.hrIndex` when present and role-consistent
 *   3. fallback        -> positional alignment when the proxy kept message count
 */

import type { OpenAIMessage } from "./convert.js";
import { isRecord } from "./content-blocks.js";

export interface OriginalLookup {
  /** Normalized originals in request order. */
  byIndex: readonly Record<string, unknown>[];
  /** Tool results keyed by tool call id; ids appearing more than once are excluded. */
  toolByCallId: ReadonlyMap<string, Record<string, unknown>>;
}

function toolCallIdOf(message: Record<string, unknown>): string | null {
  const id = message.toolCallId ?? message.tool_use_id ?? message.id;
  return typeof id === "string" ? id : null;
}

/** Map an AgentMessage role onto the wire role produced by `agentToOpenAI`. */
export function wireRoleOf(message: Record<string, unknown>): string {
  switch (message.role) {
    case "system":
      return "system";
    case "assistant":
      return "assistant";
    case "toolResult":
    case "tool_result":
      return "tool";
    default:
      return "user";
  }
}

export function buildOriginalLookup(
  normalizedOriginals: readonly Record<string, unknown>[],
): OriginalLookup {
  const seen = new Map<string, number>();
  for (const message of normalizedOriginals) {
    if (wireRoleOf(message) !== "tool") continue;
    const id = toolCallIdOf(message);
    if (id) seen.set(id, (seen.get(id) ?? 0) + 1);
  }

  const toolByCallId = new Map<string, Record<string, unknown>>();
  for (const message of normalizedOriginals) {
    if (wireRoleOf(message) !== "tool") continue;
    const id = toolCallIdOf(message);
    if (id && seen.get(id) === 1) toolByCallId.set(id, message);
  }

  return { byIndex: normalizedOriginals, toolByCallId };
}

export function findOriginal(params: {
  compressed: OpenAIMessage;
  index: number;
  compressedCount: number;
  lookup: OriginalLookup;
}): Record<string, unknown> | null {
  const { compressed, index, compressedCount, lookup } = params;

  if (compressed.role === "tool" && typeof compressed.tool_call_id === "string") {
    const byId = lookup.toolByCallId.get(compressed.tool_call_id);
    if (byId) return byId;
  }

  const meta = isRecord(compressed._headroomMeta) ? compressed._headroomMeta : {};
  const hinted = meta.hrIndex;
  if (typeof hinted === "number" && Number.isInteger(hinted)) {
    const candidate = lookup.byIndex[hinted];
    if (candidate && wireRoleOf(candidate) === compressed.role) return candidate;
  }

  if (compressedCount === lookup.byIndex.length) {
    const candidate = lookup.byIndex[index];
    if (candidate && wireRoleOf(candidate) === compressed.role) return candidate;
  }

  return null;
}
