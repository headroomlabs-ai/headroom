/**
 * Tool-name resolution for deferred-tool wrappers and per-tool result protection.
 *
 * OpenClaw's Tool Search bridge emits every deferred (MCP) call as a function
 * named `tool_call`; the real target lives in the arguments:
 *
 *   OpenClaw: { "id": "mcp:<server>:<server>__<tool>", "args": {...} }
 *   Hermes:   { "name": "<tool>", "arguments": {...} }
 *
 * The Headroom proxy keys tool-result protection (`--protect-tool-results`,
 * `exclude_tools`) on the tool name it sees on the wire, so `agentToOpenAI` sends
 * the resolved name instead of the wrapper. `openAIToAgent` additionally restores
 * results of tools listed in `protectToolResults` verbatim, regardless of what the
 * proxy did to them.
 */

import { isRecord } from "./content-blocks.js";

export const TOOL_CALL_WRAPPER = "tool_call";
const OPENCLAW_MCP_ID_PREFIX = "mcp:";

/** Parse wrapper arguments that may arrive as a JSON string or an object. */
function parseArguments(args: unknown): Record<string, unknown> | null {
  if (isRecord(args)) return args;
  if (typeof args !== "string") return null;
  try {
    const parsed: unknown = JSON.parse(args);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * Map an OpenClaw Tool Search id to a canonical `mcp__<server>__<tool>` name.
 * Non-MCP ids (`<source>:<tool>`) resolve to their last `:` segment.
 * Returns null when the id carries no usable tool name.
 */
export function resolveOpenClawToolId(toolId: string): string | null {
  const trimmed = toolId.trim();
  if (trimmed.length === 0) return null;
  if (!trimmed.startsWith(OPENCLAW_MCP_ID_PREFIX)) {
    const tail = trimmed.slice(trimmed.lastIndexOf(":") + 1).trim();
    return tail.length > 0 ? tail : null;
  }
  const firstColon = trimmed.indexOf(":");
  const secondColon = trimmed.indexOf(":", firstColon + 1);
  if (secondColon < 0) return null;
  const server = trimmed.slice(firstColon + 1, secondColon);
  const fullName = trimmed.slice(secondColon + 1);
  if (server.length === 0 || fullName.length === 0) return null;
  const prefix = `${server}__`;
  const tool = fullName.startsWith(prefix) ? fullName.slice(prefix.length) : fullName;
  if (tool.length === 0) return null;
  return `mcp__${server}__${tool}`;
}

/**
 * Resolve the real tool name behind a deferred `tool_call` wrapper.
 * Non-wrapper names pass through unchanged; malformed wrappers fail open to the
 * wrapper name so callers never lose the tool_call_id pairing.
 */
export function resolveToolName(name: string, args: unknown): string {
  if (name !== TOOL_CALL_WRAPPER) return name;
  const parsed = parseArguments(args);
  if (!parsed) return name;
  const inner = parsed.name;
  if (typeof inner === "string" && inner.trim().length > 0) return inner.trim();
  const toolId = parsed.id;
  if (typeof toolId === "string") {
    const resolved = resolveOpenClawToolId(toolId);
    if (resolved) return resolved;
  }
  return name;
}

/**
 * Equivalent spellings of a tool name for protection matching:
 * `mcp__server__tool` → also `mcp_server_tool`, `server__tool`, `tool`.
 */
export function toolNameAliases(name: string): string[] {
  const aliases = [name];
  if (name.startsWith("mcp__")) {
    const rest = name.slice("mcp__".length);
    const split = rest.indexOf("__");
    if (split > 0 && split + 2 < rest.length) {
      const server = rest.slice(0, split);
      const tool = rest.slice(split + 2);
      aliases.push(`mcp_${server}_${tool}`, `${server}__${tool}`, tool);
    }
  }
  return [...new Set(aliases)];
}

/** Normalize a `protectToolResults` config value into a lowercase name set. */
export function normalizeProtectedToolNames(value: unknown): ReadonlySet<string> {
  if (!Array.isArray(value)) return new Set();
  const names = value
    .filter((entry): entry is string => typeof entry === "string")
    .map((entry) => entry.trim().toLowerCase())
    .filter((entry) => entry.length > 0);
  return new Set(names);
}

/** Case-insensitive glob (`*` only) match. */
function globMatches(pattern: string, value: string): boolean {
  if (!pattern.includes("*")) return pattern === value;
  const escaped = pattern.split("*").map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return new RegExp(`^${escaped.join(".*")}$`).test(value);
}

/** True when a resolved tool name (or any alias of it) is in the protected set. */
export function isProtectedToolName(name: string, protectedNames: ReadonlySet<string>): boolean {
  if (protectedNames.size === 0) return false;
  const aliases = toolNameAliases(name).map((alias) => alias.toLowerCase());
  for (const pattern of protectedNames) {
    if (aliases.some((alias) => globMatches(pattern, alias))) return true;
  }
  return false;
}

/**
 * Build `toolCallId → resolved tool name` from normalized AgentMessages by
 * scanning assistant `toolCall` blocks.
 */
export function buildToolCallNameMap(messages: readonly unknown[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const message of messages) {
    if (!isRecord(message) || message.role !== "assistant" || !Array.isArray(message.content)) {
      continue;
    }
    for (const block of message.content) {
      if (!isRecord(block) || block.type !== "toolCall") continue;
      const id = block.id;
      const name = block.name;
      if (typeof id !== "string" || typeof name !== "string") continue;
      map.set(id, resolveToolName(name, block.arguments));
    }
  }
  return map;
}
