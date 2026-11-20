/**
 * Turn-boundary selection for durable transcript truncation.
 *
 * Mirrors OpenClaw's native compaction cut rules (`isTurnStartMessage` in the
 * runtime's compaction module): a kept tail must begin at a message that starts
 * a turn, never at an assistant continuation or a `toolResult`, so that no
 * tool result is left without the assistant tool call that produced it.
 */

import { isRecord } from "./content-blocks.js";

/** Roles that start a logical turn in an OpenClaw transcript. */
const TURN_START_ROLES = new Set([
  "user",
  "bashExecution",
  "branchSummary",
  "compactionSummary",
]);

const TOOL_CALL_BLOCK_TYPES = new Set(["toolCall", "tool_use"]);

export interface TruncateSelection {
  /** Index into the original message list where the kept tail begins. */
  startIndex: number;
  /** How far the cut moved from the requested index to reach a safe boundary. */
  shiftedBy: number;
}

/**
 * Runtime-context carriers are host-managed `custom` messages that OpenClaw
 * re-injects every turn; they are not user-visible turn starts.
 */
function isRuntimeContextCarrier(message: Record<string, unknown>): boolean {
  return (
    message.role === "custom" &&
    isRecord(message.details) &&
    message.details.runtimeContextCarrier === true
  );
}

export function isTurnStartMessage(message: unknown): boolean {
  if (!isRecord(message) || typeof message.role !== "string") return false;
  if (message.role === "custom") return !isRuntimeContextCarrier(message);
  return TURN_START_ROLES.has(message.role);
}

/**
 * Pick the index the kept tail should start at, given the caller's desired cut.
 *
 * Preference order:
 *   1. the first turn start at or after `desiredStart` (keeps slightly less,
 *      never splits a turn),
 *   2. otherwise the nearest turn start before it (keeps slightly more),
 *   3. otherwise `null` — the transcript has no safe boundary and must not be
 *      truncated.
 *
 * A start of `0` is returned as-is; callers treat it as "nothing to drop".
 */
export function selectTruncateStart(
  messages: readonly unknown[],
  desiredStart: number,
): TruncateSelection | null {
  if (messages.length === 0) return null;
  const desired = Math.min(Math.max(0, Math.floor(desiredStart)), messages.length - 1);

  for (let index = desired; index < messages.length; index += 1) {
    if (isTurnStartMessage(messages[index])) {
      return { startIndex: index, shiftedBy: index - desired };
    }
  }
  for (let index = desired - 1; index >= 0; index -= 1) {
    if (isTurnStartMessage(messages[index])) {
      return { startIndex: index, shiftedBy: index - desired };
    }
  }
  return null;
}

function toolCallIdsOf(message: unknown): string[] {
  if (!isRecord(message) || message.role !== "assistant" || !Array.isArray(message.content)) {
    return [];
  }
  return message.content.flatMap((block) =>
    isRecord(block) &&
    typeof block.type === "string" &&
    TOOL_CALL_BLOCK_TYPES.has(block.type) &&
    typeof block.id === "string"
      ? [block.id]
      : [],
  );
}

function toolResultCallId(message: unknown): string | null {
  if (!isRecord(message) || (message.role !== "toolResult" && message.role !== "tool_result")) {
    return null;
  }
  const id = message.toolCallId ?? message.tool_use_id ?? message.id;
  return typeof id === "string" ? id : "";
}

/**
 * Remove tool results whose originating assistant tool call is not part of
 * the same list. Defensive post-pass after boundary selection: a kept tail
 * that starts at a turn boundary should already be pair-complete.
 */
export function dropOrphanToolResults<T>(messages: readonly T[]): T[] {
  const seenToolCalls = new Set<string>();
  const kept: T[] = [];
  for (const message of messages) {
    for (const id of toolCallIdsOf(message)) seenToolCalls.add(id);
    const resultId = toolResultCallId(message);
    if (resultId !== null && !seenToolCalls.has(resultId)) continue;
    kept.push(message);
  }
  return kept;
}
